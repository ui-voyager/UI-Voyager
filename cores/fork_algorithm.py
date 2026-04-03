#!/usr/bin/env python3
"""Fork SFT Core Matching Algorithm Module

Implements the backward state matching algorithm to compare success and
failure trajectories and identify critical fork points.  Contains
deduplication, matching, pairing strategies, and the single-task analysis
entry point.
"""

import copy
import itertools
from typing import Dict, List, Optional, Tuple

from fork_utils import (
    clear_image_caches,
    extract_tool_call,
    get_step_image_path,
    load_task_repeats,
    obs_similarity,
    save_fork_log,
)


# ==================== Backward State Matching Core Algorithm ====================


def _backward_state_match(
    base_steps: List[dict],
    failed_steps: List[dict],
    ssim_threshold: float,
) -> Tuple[List[Tuple[int, int]], dict, dict]:
    """Backward state matching: for each failed step, find the best
    teacher step in the success trajectory.

    Algorithm:
    1. For every failed step, search all base (success) steps for a pair
       where the screenshots are similar but the tool_call actions differ
       (same state, different decision).
       - Additionally check the transition (s, a, s'):
         * Last step: record as candidate directly.
         * Non-last step: also verify that the *next* screenshots differ.
           If they are similar the actions have the same effect — filter out.
    2. Among candidates, select the base step with the highest SSIM.
       Ties are broken by preferring the earlier base step (smallest index).
    3. Monotonic non-decreasing constraint: as failed steps are processed
       in order, each matched base step index must be >= the previous one.
    4. Each failed step matches at most one base step, but a single base
       step may serve as teacher for multiple failed steps.
    5. Transition advance mechanism: before searching candidates for a
       failed step, check whether its transition matches a base step J
       (current screenshot SSIM >= threshold AND next screenshot SSIM >=
       threshold).  If so, the failure trajectory has correctly passed
       base step J — advance min_base_idx to J+1 and skip the current
       failed step.
       *Important*: only match the first qualifying base step (break),
       to avoid over-advancing min_base_idx in loop scenarios.

    Args:
        base_steps: Success (base) trajectory step list.
        failed_steps: Failure trajectory step list.
        ssim_threshold: SSIM similarity threshold.

    Returns:
        tuple:
            - List of match pairs [(failed_step_idx, base_step_idx), ...].
            - Filter diagnosis log
              {failed_step_idx: [(base_step_idx, ssim_score, reason), ...]}
              reason in: "passed" / "ssim_low" / "same_tool_call" /
              "same_transition" / "same_transition_advance" /
              "below_advanced_min" / "skipped_by_transition_advance"
            - Debug info dict.
    """
    if not base_steps or not failed_steps:
        return [], {}, {}

    # Pre-extract image paths and tool_calls for all base steps.
    base_info: List[Tuple[int, Optional[str], Optional[dict]]] = []
    for base_step_idx in range(len(base_steps)):
        img_path = get_step_image_path(base_steps[base_step_idx])
        tool_call = extract_tool_call(base_steps[base_step_idx])
        base_info.append((base_step_idx, img_path, tool_call))

    match_pairs: List[Tuple[int, int]] = []
    # Filter diagnosis log: records why each candidate was filtered or passed.
    filter_log: Dict[int, List[Tuple[int, float, str]]] = {}
    # Debug info: min_base_idx, image path, etc. for each failed step.
    debug_info: Dict[int, dict] = {}
    # Monotonic non-decreasing constraint lower bound.
    min_base_idx: int = 0

    for failed_step_idx in range(len(failed_steps)):
        failed_img_path = get_step_image_path(failed_steps[failed_step_idx])
        if failed_img_path is None:
            continue

        failed_tool_call = extract_tool_call(failed_steps[failed_step_idx])
        is_failed_last_step = (failed_step_idx == len(failed_steps) - 1)

        debug_info[failed_step_idx] = {
            "min_base_idx": min_base_idx,
            "image_path": failed_img_path,
        }

        # ------ Transition advance check ------
        # Before searching candidates, check whether the current failed step's
        # transition matches a base step's transition.  If so, the failure
        # trajectory correctly passed that base step — advance min_base_idx
        # and skip the current failed step.
        transition_advanced = False
        if not is_failed_last_step:
            failed_next_img = get_step_image_path(failed_steps[failed_step_idx + 1])
            if failed_img_path is not None and failed_next_img is not None:
                for base_step_idx_check, base_img_path_check, _ in base_info:
                    if base_img_path_check is None:
                        continue
                    if base_step_idx_check < min_base_idx:
                        continue
                    if base_step_idx_check >= len(base_steps) - 1:
                        continue
                    sim_current = obs_similarity(base_img_path_check, failed_img_path)
                    if sim_current < ssim_threshold - 1e-4:
                        continue
                    base_next_img_check = get_step_image_path(base_steps[base_step_idx_check + 1])
                    if base_next_img_check is None:
                        continue
                    sim_next = obs_similarity(base_next_img_check, failed_next_img)
                    if sim_next >= ssim_threshold - 1e-4:
                        transition_advanced = True
                        new_min = base_step_idx_check + 1
                        if failed_step_idx not in filter_log:
                            filter_log[failed_step_idx] = []
                        filter_log[failed_step_idx].append(
                            (base_step_idx_check, sim_current, "same_transition_advance")
                        )
                        debug_info[failed_step_idx].setdefault("transition_advances", []).append({
                            "from_base_idx": base_step_idx_check,
                            "new_min_base_idx": max(new_min, min_base_idx),
                            "sim_current": round(sim_current, 4),
                            "sim_next": round(sim_next, 4),
                        })
                        if new_min > min_base_idx:
                            min_base_idx = new_min
                        # Only match the first qualifying base step to avoid
                        # over-advancing in loop scenarios.
                        break

        if transition_advanced:
            if failed_step_idx not in filter_log:
                filter_log[failed_step_idx] = []
            filter_log[failed_step_idx].append(
                (-1, -1.0, "skipped_by_transition_advance")
            )
            debug_info[failed_step_idx]["skipped"] = True
            debug_info[failed_step_idx]["skip_reason"] = "transition_advance"
            continue

        # Collect candidate base steps: (base_step_idx, ssim_score)
        candidates: List[Tuple[int, float]] = []
        step_filter_log: List[Tuple[int, float, str]] = []

        for base_step_idx, base_img_path, base_tool_call in base_info:
            if base_img_path is None:
                continue

            # Monotonic constraint: skip base steps below the lower bound.
            if base_step_idx < min_base_idx:
                continue

            sim_score = obs_similarity(base_img_path, failed_img_path)
            if sim_score < ssim_threshold - 1e-4:
                continue

            # Screenshots are similar — check whether actions differ.
            if (base_tool_call is not None
                    and failed_tool_call is not None
                    and base_tool_call == failed_tool_call):
                step_filter_log.append((base_step_idx, sim_score, "same_tool_call"))
                continue

            # For non-last steps, check the transition s' (next screenshot).
            is_base_last_step = (base_step_idx == len(base_steps) - 1)
            if not is_base_last_step and not is_failed_last_step:
                base_next_img = get_step_image_path(base_steps[base_step_idx + 1])
                failed_next_img = get_step_image_path(failed_steps[failed_step_idx + 1])
                if base_next_img is not None and failed_next_img is not None:
                    next_sim = obs_similarity(base_next_img, failed_next_img)
                    if next_sim >= ssim_threshold - 1e-4:
                        step_filter_log.append((base_step_idx, sim_score, "same_transition"))
                        continue

            # Passed all filters — add to candidates.
            step_filter_log.append((base_step_idx, sim_score, "passed"))
            candidates.append((base_step_idx, sim_score))

        if step_filter_log:
            if failed_step_idx in filter_log:
                filter_log[failed_step_idx].extend(step_filter_log)
            else:
                filter_log[failed_step_idx] = step_filter_log

        if not candidates:
            continue

        # Select the best teacher: highest SSIM, tie-break by earliest index.
        best_base_idx, best_score = max(candidates, key=lambda x: (x[1], -x[0]))
        match_pairs.append((failed_step_idx, best_base_idx))
        # Update the monotonic lower bound.
        min_base_idx = best_base_idx

    return match_pairs, filter_log, debug_info


# ==================== Pairing Strategies ====================

def _select_pairs_all(
    succ_repeats: Dict[int, dict],
    fail_repeats: Dict[int, dict],
) -> List[Tuple[int, int]]:
    """All-pairs mode: generate the full Cartesian product of success and failure trajectories."""
    return list(itertools.product(succ_repeats.keys(), fail_repeats.keys()))


def _select_pairs_shortest_base(
    succ_repeats: Dict[int, dict],
    fail_repeats: Dict[int, dict],
) -> List[Tuple[int, int]]:
    """Shortest-base mode: use the shortest success trajectory as the sole
    base and pair it with every failure trajectory."""
    shortest_id = min(succ_repeats.keys(), key=lambda sid: len(succ_repeats[sid]["steps"]))
    return [(shortest_id, fid) for fid in fail_repeats.keys()]


_PAIR_SELECTORS = {
    "all_pairs": _select_pairs_all,
    "shortest_base": _select_pairs_shortest_base,
}


# ==================== Fork SFT Data Construction ====================

def _get_assistant_content(step: dict) -> Optional[str]:
    """Extract the assistant's content from a step's conversations."""
    for conv in step.get("conversations", []):
        if conv.get("role") == "assistant":
            return conv.get("content", "")
    return None


def _build_fork_sft_entry(
    fail_step: dict,
    base_step: dict,
    task_name: str,
    succ_id: int,
    fail_id: int,
    failed_step_idx: int,
    base_step_idx: int,
    sim_current: float,
    sim_next: float,
) -> Optional[dict]:
    """Build one fork SFT entry: replace the failed step's assistant
    response with the success base step's response.

    The failed step's context (system prompt, user prompt, images) is
    preserved; only the assistant content is swapped.

    Args:
        fail_step: Failure trajectory fork step data.
        base_step: Success trajectory base step data.
        task_name: Task name.
        succ_id: Success trajectory repeat_id.
        fail_id: Failure trajectory repeat_id.
        failed_step_idx: Fork step index in the failure trajectory.
        base_step_idx: Base step index in the success trajectory.
        sim_current: Current-step screenshot SSIM similarity.
        sim_next: Next-step screenshot SSIM similarity.

    Returns:
        Constructed SFT data dict, or None if the assistant content
        cannot be extracted.
    """
    base_response = _get_assistant_content(base_step)
    if base_response is None:
        return None

    # Deep-copy the failed step to avoid mutating the original data.
    entry = copy.deepcopy(fail_step)

    # Replace assistant content with the success trajectory's response.
    for conv in entry.get("conversations", []):
        if conv.get("role") == "assistant":
            conv["content"] = base_response
            break

    # Attach fork metadata for traceability.
    entry["fork_info"] = {
        "task_name": task_name,
        "succ_id": succ_id,
        "fail_id": fail_id,
        "failed_step_idx": failed_step_idx,
        "base_step_idx": base_step_idx,
        "sim_current": round(sim_current, 4),
        "sim_next": round(sim_next, 4),
    }

    return entry


# ==================== Single-Task Analysis Entry Point ====================

def analyze_single_task(
    task_name: str,
    task_dir: str,
    ssim_threshold: float,
    mode: str,
    log_dir: Optional[str],
) -> Tuple[str, dict, List[str], List[dict]]:
    """Analyze a single task (designed for parallel execution).

    Uses the backward state matching algorithm to identify fork points and
    saves analysis logs to the log directory.  For each fork point, builds
    an SFT entry by replacing the failed step's response with the success
    base step's response.

    Args:
        task_name: Task name.
        task_dir: Path to the task data directory.
        ssim_threshold: SSIM similarity threshold.
        mode: Pairing strategy name (key in _PAIR_SELECTORS).
        log_dir: Log output directory (None to skip disk logging).

    Returns:
        (task_name, task_stat, log_messages, fork_sft_entries)
    """
    pair_selector = _PAIR_SELECTORS[mode]
    logs: List[str] = []
    fork_sft_entries: List[dict] = []

    task_data = load_task_repeats(task_dir)
    succ_repeats = task_data["succ"]
    fail_repeats = task_data["fail"]

    task_stat = {
        "n_succ": len(succ_repeats),
        "n_fail": len(fail_repeats),
        "n_pairs_compared": 0,
        "n_pairs_with_fork": 0,
        "n_total_fork_points": 0,
        "fork_details": [],  # [(succ_id, fail_id, [(fail_idx, base_idx, sim_c, sim_n), ...])]
        "_status": "mixed",
    }

    if not succ_repeats or not fail_repeats:
        task_stat["_status"] = "all_fail" if not succ_repeats else "all_succ"
        return task_name, task_stat, logs, fork_sft_entries

    pairs = pair_selector(succ_repeats, fail_repeats)

    for succ_id, fail_id in pairs:
        succ_steps = succ_repeats[succ_id]["steps"]
        fail_steps = fail_repeats[fail_id]["steps"]
        task_stat["n_pairs_compared"] += 1

        # Check for screenshots
        has_succ_img = any(get_step_image_path(s) is not None for s in succ_steps)
        has_fail_img = any(get_step_image_path(s) is not None for s in fail_steps)
        if not has_succ_img or not has_fail_img:
            logs.append(f"  [SKIP] {task_name}: succ_r{succ_id}/fail_r{fail_id} - no screenshots")
            continue

        # Run backward state matching
        match_pairs, filter_log, debug_info = _backward_state_match(succ_steps, fail_steps, ssim_threshold)

        if not match_pairs:
            continue

        # Compute similarity scores for each match pair
        enriched_pairs = []
        for failed_step_idx, base_step_idx in match_pairs:
            base_img = get_step_image_path(succ_steps[base_step_idx])
            fail_img = get_step_image_path(fail_steps[failed_step_idx])
            sim_current = obs_similarity(base_img, fail_img) if base_img and fail_img else -1.0

            sim_next = -1.0
            if base_step_idx + 1 < len(succ_steps) and failed_step_idx + 1 < len(fail_steps):
                base_next_img = get_step_image_path(succ_steps[base_step_idx + 1])
                fail_next_img = get_step_image_path(fail_steps[failed_step_idx + 1])
                if base_next_img and fail_next_img:
                    sim_next = obs_similarity(base_next_img, fail_next_img)

            enriched_pairs.append((failed_step_idx, base_step_idx, sim_current, sim_next))

        task_stat["n_pairs_with_fork"] += 1
        task_stat["n_total_fork_points"] += len(enriched_pairs)
        task_stat["fork_details"].append((succ_id, fail_id, enriched_pairs))

        # Build SFT entries for each fork point
        for failed_step_idx, base_step_idx, sim_c, sim_n in enriched_pairs:
            sft_entry = _build_fork_sft_entry(
                fail_step=fail_steps[failed_step_idx],
                base_step=succ_steps[base_step_idx],
                task_name=task_name,
                succ_id=succ_id,
                fail_id=fail_id,
                failed_step_idx=failed_step_idx,
                base_step_idx=base_step_idx,
                sim_current=sim_c,
                sim_next=sim_n,
            )
            if sft_entry is not None:
                fork_sft_entries.append(sft_entry)

        fork_desc = ", ".join(
            f"fail[{fp[0]}]<->base[{fp[1]}](sim={fp[2]:.3f})"
            for fp in enriched_pairs
        )
        logs.append(
            f"  [FORK] {task_name}: succ_r{succ_id} vs fail_r{fail_id} "
            f"-> {len(enriched_pairs)} fork(s): {fork_desc}"
        )

        # Save fork analysis log to disk
        if log_dir:
            save_fork_log(
                log_dir=log_dir,
                task_name=task_name,
                succ_id=succ_id,
                fail_id=fail_id,
                succ_steps=succ_steps,
                fail_steps=fail_steps,
                match_pairs=enriched_pairs,
                filter_log=filter_log,
                debug_info=debug_info,
            )

    if task_stat["n_pairs_with_fork"] == 0:
        task_stat["_status"] = "mixed_no_fork"

    # Clear process-level caches to prevent memory leaks.
    clear_image_caches()

    return task_name, task_stat, logs, fork_sft_entries
