#!/usr/bin/env python3
"""Fork SFT Analysis — Main Entry Point

Uses backward state matching to compare success and failure trajectories
and identify critical fork points.  Original data is never modified;
analysis results are saved to a separate log directory including per-step
system prompts, user prompts, actions, and screenshots for manual review.

Algorithm overview:
    Backward state matching walks the success trajectory from the end
    toward the beginning, searching for steps in the failure trajectory
    that share the same screenshot but take a different action.  This
    finds the decision point closest to the goal rather than the earliest
    divergence.  Transition comparison (s, a, s') filters out matches
    where the different actions lead to the same next state.

Directory structure (auto-detected):
    The tool looks for task data under {data_dir}/results/ and supports
    two layouts:

    3-level (multiple models):
        {data_dir}/results/{model_name}/{timestamp}/seed_{N}/{TaskName}/
    2-level (single model, produced by run_group_sample.sh):
        {data_dir}/results/{timestamp}/seed_{N}/{TaskName}/

    Inside each {TaskName}/ directory, the expected files are:
        repeat_{N}_succ.jsonl   — successful trajectory
        repeat_{N}_fail.jsonl   — failed trajectory

    An optional sft_data/ sub-level is also recognised:
        .../seed_{N}/sft_data/{TaskName}/

Usage:
    # Analyze a single model's data
    python cores/fork_main.py \\
        --data_dir eval_results/Qwen3-VL-4B-Instruct \\
        --ssim_threshold 0.95 \\
        --mode shortest_base

    # Analyze all models at once
    python cores/fork_main.py \\
        --data_dir eval_results \\
        --mode all_pairs
"""

import argparse
import copy
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from fork_algorithm import analyze_single_task
from fork_utils import discover_all_sft_dirs, discover_seed_tasks


# ==================== Statistics Aggregation ====================

def _aggregate_seed_stats(
    task_results: List[Tuple[str, dict, List[str]]],
    total_tasks: int,
) -> dict:
    """Aggregate per-task analysis results into seed-level statistics."""
    stats = {
        "total_tasks": total_tasks,
        "mixed_tasks": 0,
        "total_pairs_compared": 0,
        "total_pairs_with_fork": 0,
        "total_fork_points": 0,
        "skipped_all_succ": 0,
        "skipped_all_fail": 0,
        "per_task": {},
    }

    for task_name, task_stat, logs in task_results:
        status = task_stat.get("_status", "unknown")
        if status == "all_fail":
            stats["skipped_all_fail"] += 1
        elif status == "all_succ":
            stats["skipped_all_succ"] += 1
        else:
            stats["mixed_tasks"] += 1
            stats["total_pairs_compared"] += task_stat["n_pairs_compared"]
            stats["total_pairs_with_fork"] += task_stat["n_pairs_with_fork"]
            stats["total_fork_points"] += task_stat["n_total_fork_points"]

        stats["per_task"][task_name] = task_stat

    return stats


# ==================== Report Printing ====================

def print_analysis_report(
    all_stats: Dict[Tuple[str, str, str], dict],
    data_dir: str,
    ssim_threshold: float,
) -> None:
    """Print a formatted analysis report to stdout."""
    print("\n" + "=" * 60)
    print("Fork SFT Analysis Report (Backward State Matching)")
    print("=" * 60)
    print(f"Data dir: {data_dir}")
    print(f"SSIM threshold: {ssim_threshold}")

    total_fork_points = 0
    total_pairs_compared = 0

    for (model_name, timestamp, seed_name), stats in sorted(all_stats.items()):
        label = f"{model_name}/{timestamp}/{seed_name}"
        print(f"\n{label}:")
        print(f"  Total tasks: {stats['total_tasks']}")
        print(f"  Tasks with mixed results (succ+fail): {stats['mixed_tasks']}")
        print(f"  Tasks all success (skipped): {stats['skipped_all_succ']}")
        print(f"  Tasks all fail (skipped): {stats['skipped_all_fail']}")
        print(f"  Total pairs compared: {stats['total_pairs_compared']}")
        print(f"  Pairs with fork points: {stats['total_pairs_with_fork']}")
        print(f"  Total fork points found: {stats['total_fork_points']}")

        total_fork_points += stats["total_fork_points"]
        total_pairs_compared += stats["total_pairs_compared"]

        # Per-task summary
        if stats["per_task"]:
            print(f"\n  Per-task summary:")
            for task_name, ts in sorted(stats["per_task"].items()):
                if ts["n_succ"] == 0 and ts["n_fail"] == 0:
                    continue
                if ts["n_succ"] == 0:
                    status = "-> skipped (all fail)"
                elif ts["n_fail"] == 0:
                    status = "-> skipped (all success)"
                else:
                    n_forks = ts["n_total_fork_points"]
                    status = (
                        f"-> {ts['n_pairs_compared']} pairs, "
                        f"{ts['n_pairs_with_fork']} with fork, "
                        f"{n_forks} fork point(s)"
                    )
                    if ts["fork_details"]:
                        detail_parts = []
                        for succ_id, fail_id, fps in ts["fork_details"]:
                            fork_indices = [f"f{fp[0]}<->b{fp[1]}" for fp in fps]
                            detail_parts.append(f"s{succ_id}vs.f{fail_id}:{','.join(fork_indices)}")
                        status += f" [{'; '.join(detail_parts)}]"
                print(f"    {task_name:45s} {ts['n_succ']} succ / {ts['n_fail']} fail {status}")

    print(f"\n{'=' * 60}")
    print(f"Grand total: {total_pairs_compared} pairs compared, {total_fork_points} fork points found")
    print("=" * 60)


# ==================== Main Entry Point ====================

def main():
    parser = argparse.ArgumentParser(
        description="Fork Step Analysis: identify fork points via backward state matching and save analysis logs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Root data directory (e.g. eval_results/Qwen3-VL-4B-Instruct). "
             "Supports results/{model}/{timestamp}/seed_{N}/ and "
             "results/{timestamp}/seed_{N}/ layouts.",
    )
    parser.add_argument(
        "--ssim_threshold", type=float, default=0.95,
        help="SSIM screenshot similarity threshold (default: 0.95).",
    )
    parser.add_argument(
        "--mode", type=str, default="shortest_base",
        choices=["all_pairs", "shortest_base"],
        help="Pairing strategy: 'shortest_base' = use shortest success "
             "trajectory as base (default), 'all_pairs' = full Cartesian product.",
    )
    parser.add_argument(
        "--num_workers", type=int, default=32,
        help="Number of parallel workers (default: 32).",
    )
    parser.add_argument(
        "--log_dir", type=str, default=None,
        help="Fork analysis log output directory. "
             "Default: {data_dir}/fork_analysis_logs/.",
    )
    parser.add_argument(
        "--max_log_seeds", type=int, default=5,
        help="Only write disk logs for the first N seeds (default: 5). "
             "Later seeds are still analyzed but their trajectory logs are "
             "not saved to disk.",
    )
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    if not os.path.isdir(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)

    # Determine log output directory
    log_base_dir = os.path.abspath(args.log_dir) if args.log_dir else os.path.join(data_dir, "fork_analysis_logs")
    os.makedirs(log_base_dir, exist_ok=True)
    print(f"Log directory created: {log_base_dir}")

    # Discover all sft_data directories
    sft_dirs = discover_all_sft_dirs(data_dir)

    if not sft_dirs:
        print(f"Error: No task data directories found under {data_dir}/results/")
        sys.exit(1)

    # Group by model/timestamp for display
    groups = defaultdict(list)
    for model_name, timestamp, seed_name, sft_data_dir in sft_dirs:
        groups[(model_name, timestamp)].append((seed_name, sft_data_dir))

    total_seeds = len(sft_dirs)
    total_groups = len(groups)
    print(f"Discovered {total_seeds} seed(s) across {total_groups} model/timestamp group(s)")
    for (model, ts), seeds in sorted(groups.items()):
        print(f"  {model}/{ts}: {len(seeds)} seed(s)")
    print(f"Algorithm: Backward State Matching")
    print(f"Pairing mode: {args.mode}")
    print(f"SSIM threshold: {args.ssim_threshold}")
    print(f"Num workers: {args.num_workers}")
    print(f"Log output dir: {log_base_dir}")

    # ===== Flatten all seeds and tasks for global parallel processing =====
    max_log_seeds = args.max_log_seeds

    SeedKey = Tuple[str, str, str]
    # task_item: (seed_key, task_name, task_dir, ssim_threshold, mode, log_dir)
    all_task_items: List[Tuple[SeedKey, str, str, float, str, Optional[str]]] = []
    seed_total_tasks: Dict[SeedKey, int] = {}

    for seed_idx, (model_name, timestamp, seed_name, sft_data_dir) in enumerate(sft_dirs):
        seed_key: SeedKey = (model_name, timestamp, seed_name)
        log_enabled = seed_idx < max_log_seeds
        seed_log_dir = os.path.join(log_base_dir, model_name, timestamp, seed_name) if log_enabled else None

        label = f"{model_name}/{timestamp}/{seed_name}"
        disk_tag = "" if log_enabled else " [disk log disabled]"
        print(f"  Discovered {label} ({sft_data_dir}){disk_tag}")

        task_names = discover_seed_tasks(sft_data_dir)
        seed_total_tasks[seed_key] = len(task_names)

        for task_name in task_names:
            task_dir = os.path.join(sft_data_dir, task_name)
            all_task_items.append((seed_key, task_name, task_dir, args.ssim_threshold, args.mode, seed_log_dir))

    total_tasks = len(all_task_items)
    print(f"\nTotal tasks to analyze: {total_tasks} across {total_seeds} seed(s)")

    if total_tasks == 0:
        print("No tasks found, exiting.")
        sys.exit(0)

    # Parallel processing with a global ProcessPoolExecutor
    seed_results: Dict[SeedKey, List[Tuple[str, dict, List[str]]]] = defaultdict(list)
    all_fork_sft_entries: List[dict] = []
    effective_workers = min(args.num_workers, total_tasks)

    futures = {}
    with ProcessPoolExecutor(max_workers=effective_workers) as executor:
        for seed_key, task_name, task_dir, ssim_th, mode, log_dir in all_task_items:
            future = executor.submit(
                analyze_single_task,
                task_name, task_dir, ssim_th, mode, log_dir,
            )
            futures[future] = seed_key

        with tqdm(total=total_tasks, desc="Analyzing all tasks", unit="task") as pbar:
            for future in as_completed(futures):
                seed_key = futures[future]
                task_name, task_stat, logs, fork_sft_entries = future.result()
                seed_results[seed_key].append((task_name, task_stat, logs))
                all_fork_sft_entries.extend(fork_sft_entries)
                # Print log messages in real time
                for msg in logs:
                    tqdm.write(msg)
                pbar.update(1)

    # Aggregate statistics by seed
    all_stats = {}
    for seed_key in seed_total_tasks:
        results = seed_results.get(seed_key, [])
        all_stats[seed_key] = _aggregate_seed_stats(results, seed_total_tasks[seed_key])

    # Print summary report
    print_analysis_report(all_stats, data_dir, args.ssim_threshold)

    # Save report as JSON
    report_path = os.path.join(log_base_dir, "fork_analysis_report.json")
    serializable_stats = {}
    for (model_name, timestamp, seed_name), stats in sorted(all_stats.items()):
        key = f"{model_name}/{timestamp}/{seed_name}"
        s = copy.deepcopy(stats)
        # Convert tuples to lists for JSON serialization
        for task_name, ts in s.get("per_task", {}).items():
            if "fork_details" in ts:
                ts["fork_details"] = [
                    [succ_id, fail_id, [[fp[0], fp[1], fp[2], fp[3]] for fp in fps]]
                    for succ_id, fail_id, fps in ts["fork_details"]
                ]
        serializable_stats[key] = s

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(serializable_stats, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to: {report_path}")

    # Save fork SFT data
    if all_fork_sft_entries:
        fork_sft_path = os.path.join(log_base_dir, "fork_steps.jsonl")
        with open(fork_sft_path, "w", encoding="utf-8") as f:
            for entry in all_fork_sft_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"Fork SFT data saved to: {fork_sft_path} ({len(all_fork_sft_entries)} entries)")
    else:
        print("No fork SFT data generated.")

    print(f"Fork analysis logs saved to: {log_base_dir}")


if __name__ == "__main__":
    main()
