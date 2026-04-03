#!/usr/bin/env python3
"""Fork SFT Utility Module

Provides screenshot comparison, action extraction, data loading, directory
discovery, and fork-analysis logging utilities.

This is the base module imported by fork_algorithm.py and fork_main.py.
"""

import json
import os
import re
import shutil
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    raise ImportError(
        "scikit-image is required but not installed. "
        "Please install it with: pip install scikit-image"
    )

# ==================== Pre-compiled Regexes ====================

# Regex for extracting <tool_call>...</tool_call> content
_TOOL_CALL_RE = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)

# ==================== Screenshot Comparison Constants ====================

_SSIM_THRESHOLD = 0.95

# Resize dimensions (width, height) used for SSIM computation.
# Down-scaling from 1080p to ~270p provides sufficient accuracy for
# screenshot matching while giving a 5-10x speed-up.
_SSIM_RESIZE_DIM = (480, 270)

# Crop ratio: remove the top 25/878 pixels of each screenshot to eliminate
# the device status bar, which changes frequently and adds noise.
# 878 corresponds to a common mobile screen height; adjust if needed.
_CROP_TOP_RATIO = 25 / 878

# Mean-hash grid size and fast pre-filter threshold.
# Hash comparison is O(1) and quickly excludes obviously different image
# pairs, avoiding the more expensive SSIM computation.
_HASH_SIZE = 16
_HASH_FAST_REJECT_THRESHOLD = 0.80  # skip pair if hash similarity < this

# ==================== Process-level Image Caches ====================
# Independent per worker in a ProcessPoolExecutor.

_IMAGE_CACHE: Dict[str, Optional[np.ndarray]] = {}
# Gray thumbnail cache: avoid repeated resize + grayscale conversion.
_GRAY_THUMB_CACHE: Dict[str, Optional[np.ndarray]] = {}
# Mean-hash cache.
_HASH_CACHE: Dict[str, Optional[np.ndarray]] = {}


def clear_image_caches() -> None:
    """Clear process-level image caches to prevent memory leaks."""
    _IMAGE_CACHE.clear()
    _GRAY_THUMB_CACHE.clear()
    _HASH_CACHE.clear()


# ==================== Screenshot Comparison Functions ====================

def crop_top(image: np.ndarray, ratio: float = _CROP_TOP_RATIO) -> np.ndarray:
    """Crop the top portion of an image by the given ratio.

    Args:
        image: Input image array (H, W, C) or (H, W).
        ratio: Fraction of height to remove from the top.

    Returns:
        Cropped image array.
    """
    h = image.shape[0]
    crop_h = int(h * ratio)
    return image[crop_h:]


def _read_image(image_path: str) -> Optional[np.ndarray]:
    """Read an image file, crop its top, and return a numpy array (cached)."""
    if image_path in _IMAGE_CACHE:
        return _IMAGE_CACHE[image_path]
    try:
        arr = cv2.imread(image_path)
        if arr is None:
            print(f"  [WARN] Cannot read image: {image_path}")
            _IMAGE_CACHE[image_path] = None
            return None
        # Crop top pixels (remove status bar noise)
        arr = crop_top(arr)
        _IMAGE_CACHE[image_path] = arr
        return arr
    except Exception as e:
        print(f"  [WARN] Error reading image {image_path}: {e}")
        _IMAGE_CACHE[image_path] = None
        return None


def _get_gray_thumbnail(image_path: str) -> Optional[np.ndarray]:
    """Get a grayscale thumbnail of the image (cached)."""
    if image_path in _GRAY_THUMB_CACHE:
        return _GRAY_THUMB_CACHE[image_path]
    arr = _read_image(image_path)
    if arr is None:
        _GRAY_THUMB_CACHE[image_path] = None
        return None
    try:
        thumb = cv2.resize(arr, _SSIM_RESIZE_DIM, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY).astype(np.float64)
        _GRAY_THUMB_CACHE[image_path] = gray
        return gray
    except Exception as e:
        print(f"  [WARN] _get_gray_thumbnail exception: {e}")
        _GRAY_THUMB_CACHE[image_path] = None
        return None


def _get_image_hash(image_path: str) -> Optional[np.ndarray]:
    """Compute the mean hash of an image (cached). Used for fast pre-filtering."""
    if image_path in _HASH_CACHE:
        return _HASH_CACHE[image_path]
    arr = _read_image(image_path)
    if arr is None:
        _HASH_CACHE[image_path] = None
        return None
    try:
        small = cv2.resize(arr, (_HASH_SIZE, _HASH_SIZE), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        mean_val = gray.mean()
        hash_bits = (gray >= mean_val).astype(np.uint8).flatten()
        _HASH_CACHE[image_path] = hash_bits
        return hash_bits
    except Exception:
        _HASH_CACHE[image_path] = None
        return None


def _hash_similarity(hash_a: np.ndarray, hash_b: np.ndarray) -> float:
    """Compute similarity between two mean hashes (normalized Hamming distance)."""
    total_bits = len(hash_a)
    matching_bits = np.sum(hash_a == hash_b)
    return float(matching_bits) / total_bits


def get_step_image_path(step: dict) -> Optional[str]:
    """Return the screenshot path for a step (last image in the images list)."""
    images = step.get("images", [])
    if not images:
        return None
    return images[-1]


def obs_similarity(
    image_path_a: str,
    image_path_b: str,
) -> float:
    """Compute SSIM similarity between two screenshots.

    Optimization pipeline:
    1. Mean-hash pre-filter: reject obviously different pairs (~0.01 ms).
    2. Pixel-exact match: return 1.0 immediately if thumbnails are identical.
    3. SSIM on grayscale thumbnails: 5-10x faster than on full-size images.

    Returns a similarity score in [0, 1], or -1.0 on error.
    """
    try:
        # Hash pre-filter
        hash_a = _get_image_hash(image_path_a)
        hash_b = _get_image_hash(image_path_b)
        if hash_a is not None and hash_b is not None:
            h_sim = _hash_similarity(hash_a, hash_b)
            if h_sim < _HASH_FAST_REJECT_THRESHOLD:
                return h_sim  # certainly below any reasonable ssim_threshold

        gray_a = _get_gray_thumbnail(image_path_a)
        gray_b = _get_gray_thumbnail(image_path_b)

        if gray_a is None or gray_b is None:
            return -1.0
        if gray_a.shape != gray_b.shape:
            return -1.0

        # Pixel-exact match
        if np.array_equal(gray_a, gray_b):
            return 1.0

        if ssim is None:
            return -1.0

        score = ssim(gray_a, gray_b, data_range=255.0)
        return float(score)
    except Exception as e:
        print(f"  [WARN] obs_similarity exception: {e}")
        return -1.0


def same_observation(
    image_path_a: str,
    image_path_b: str,
    ssim_threshold: float = _SSIM_THRESHOLD,
) -> bool:
    """Check whether two screenshots represent the same observation state.

    Uses obs_similarity internally.  A small epsilon (1e-4) is subtracted
    from the threshold to avoid floating-point precision issues.
    """
    score = obs_similarity(image_path_a, image_path_b)
    return score >= ssim_threshold - 1e-4


# ==================== Action Extraction ====================

def extract_tool_call(step: dict) -> Optional[dict]:
    """Extract and parse the <tool_call>...</tool_call> content from a step.

    Parsing pipeline:
    1. Find the assistant response in conversations.
    2. Extract the <tool_call> tag content via regex.
    3. Attempt json.loads to parse into a dict.
    4. On failure, return the raw string (enables string-level comparison).
    5. Return None if no tag is found.
    """
    response = ""
    conversations = step.get("conversations", [])
    for conv in conversations:
        if conv.get("role") == "assistant":
            response = conv.get("content", "")
            break
    if not response:
        return None

    match = _TOOL_CALL_RE.search(response)
    if not match:
        return None
    raw_content = match.group(1).strip()
    if not raw_content:
        return None

    try:
        return json.loads(raw_content)
    except (json.JSONDecodeError, ValueError):
        pass
    # Attempt to fix common formatting issues (e.g. property_name= style)
    try:
        fixed_str = re.sub(r'(\w+)=', r'"\1":', raw_content)
        return json.loads(fixed_str)
    except (json.JSONDecodeError, ValueError):
        pass
    # Fall back to raw string
    return raw_content


# ==================== Data Loading ====================

def load_task_repeats(task_dir: str) -> Dict[str, Dict[int, dict]]:
    """Load all repeat JSONL files under a task directory.

    Expected file naming: repeat_{N}_{succ|fail}.jsonl

    Returns:
        {
            "succ": {repeat_id: {"jsonl_path": path, "steps": [step_data, ...]}},
            "fail": {repeat_id: {"jsonl_path": path, "steps": [step_data, ...]}}
        }
    """
    result: Dict[str, Dict[int, dict]] = {"succ": {}, "fail": {}}

    pattern = re.compile(r"repeat_(\d+)_(succ|fail)\.jsonl$")

    for filename in sorted(os.listdir(task_dir)):
        match = pattern.match(filename)
        if not match:
            continue

        repeat_id = int(match.group(1))
        status = match.group(2)
        jsonl_path = os.path.join(task_dir, filename)

        steps: List[dict] = []
        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        steps.append(json.loads(line))
        except Exception as e:
            print(f"  [WARN] Failed to load {jsonl_path}: {e}")
            continue

        if steps:
            result[status][repeat_id] = {
                "jsonl_path": jsonl_path,
                "steps": steps,
            }

    return result


# ==================== Directory Discovery ====================

def discover_seed_tasks(
    seed_sft_dir: str,
) -> List[str]:
    """Discover all task names under a single seed directory.

    Returns:
        Sorted list of task directory names.
    """
    if not os.path.isdir(seed_sft_dir):
        print(f"  [WARN] Directory not found: {seed_sft_dir}")
        return []

    return sorted([
        d for d in os.listdir(seed_sft_dir)
        if os.path.isdir(os.path.join(seed_sft_dir, d)) and d != "images"
    ])


def _collect_seeds_under_ts_dir(
    ts_dir: str,
    model_name: str,
    timestamp: str,
) -> List[Tuple[str, str, str, str]]:
    """Collect (model_name, timestamp, seed_name, task_data_path) entries under a timestamp directory."""
    entries: List[Tuple[str, str, str, str]] = []
    for seed_name in sorted(os.listdir(ts_dir)):
        seed_dir = os.path.join(ts_dir, seed_name)
        if not os.path.isdir(seed_dir) or not seed_name.startswith("seed_"):
            continue

        sft_data_dir = os.path.join(seed_dir, "sft_data")
        if os.path.isdir(sft_data_dir):
            entries.append((model_name, timestamp, seed_name, sft_data_dir))
        else:
            has_task_subdir = any(
                os.path.isdir(os.path.join(seed_dir, d))
                for d in os.listdir(seed_dir)
                if d != "images"
            )
            if has_task_subdir:
                entries.append((model_name, timestamp, seed_name, seed_dir))
    return entries


def discover_all_sft_dirs(
    data_dir: str,
) -> List[Tuple[str, str, str, str]]:
    """Discover all task data directories under data_dir/results/.

    Supports two directory layouts:
        3-level: results/{model_name}/{timestamp}/seed_{N}/...
                 Used when data_dir contains data for multiple models
                 (e.g. data_dir = eval_results/)
        2-level: results/{timestamp}/seed_{N}/...
                 Used when data_dir is a single model directory
                 (e.g. data_dir = eval_results/Qwen3-VL-4B-Instruct/)
                 model_name is inferred from the data_dir basename

    Within each seed directory, tasks may live directly or under sft_data/:
        seed_{N}/{TaskName}/           (direct layout)
        seed_{N}/sft_data/{TaskName}/  (sft_data sub-level)

    Auto-detection: if a first-level subdirectory under results/ contains
    seed_* children, it is treated as a timestamp directory (2-level layout);
    otherwise it is treated as a model_name directory (3-level layout).

    Returns:
        [(model_name, timestamp, seed_name, task_data_path), ...] sorted
    """
    results_dir = os.path.join(data_dir, "results")
    if not os.path.isdir(results_dir):
        print(f"Error: results/ directory not found under {data_dir}")
        return []

    discovered: List[Tuple[str, str, str, str]] = []

    for first_level_name in sorted(os.listdir(results_dir)):
        first_level_dir = os.path.join(results_dir, first_level_name)
        if not os.path.isdir(first_level_dir):
            continue

        # Auto-detect layout: check if first_level_dir directly contains seed_* dirs
        has_seed_children = any(
            d.startswith("seed_") and os.path.isdir(os.path.join(first_level_dir, d))
            for d in os.listdir(first_level_dir)
        )

        if has_seed_children:
            # 2-level layout: results/{timestamp}/seed_{N}/...
            timestamp = first_level_name
            model_name = os.path.basename(data_dir.rstrip("/"))
            discovered.extend(
                _collect_seeds_under_ts_dir(first_level_dir, model_name, timestamp)
            )
        else:
            # 3-level layout: results/{model_name}/{timestamp}/seed_{N}/...
            model_name = first_level_name
            for timestamp in sorted(os.listdir(first_level_dir)):
                ts_dir = os.path.join(first_level_dir, timestamp)
                if not os.path.isdir(ts_dir):
                    continue
                discovered.extend(
                    _collect_seeds_under_ts_dir(ts_dir, model_name, timestamp)
                )

    return discovered


# ==================== Fork Analysis Logging ====================
# (Absorbed from the former fork_log_utils.py module)
#
# Log directory structure:
#   {log_dir}/{task_name}/succ_{succ_id}_vs_fail_{fail_id}/
#   +-- summary.txt               # Fork point summary
#   +-- succ_trajectory/           # Full success trajectory info
#   |   +-- step_0.txt
#   |   +-- step_0.png
#   |   +-- ...
#   +-- fail_trajectory/           # Full failure trajectory info
#       +-- step_0.txt
#       +-- step_0.png
#       +-- step_1_fork_base3_0.982_0.451.txt  # Fork point annotation
#       +-- step_1_fork_base3_0.982_0.451.png
#       +-- ...

# Truncation limit for system prompts in log text output.
_SYSTEM_PROMPT_TRUNCATE_LEN = 500


def _get_step_info_text(
    step: dict,
    step_idx: int,
    is_fork: bool = False,
    base_step_idx: int = -1,
    sim_current: float = -1.0,
    sim_next: float = -1.0,
) -> str:
    """Format key information of a step into human-readable text.

    Args:
        step: Step data (contains conversations, images, index, etc.).
        step_idx: Step index.
        is_fork: Whether this step is a fork point.
        base_step_idx: Corresponding success trajectory step index (fork only).
        sim_current: Current-step screenshot similarity (fork only).
        sim_next: Next-step screenshot similarity (fork only).

    Returns:
        Formatted text string.
    """
    lines: List[str] = []
    lines.append(f"[Step {step_idx}]")
    lines.append("=" * 60)

    if is_fork:
        lines.append(f"[FORK POINT] base_step={base_step_idx}, "
                      f"sim_current={sim_current:.4f}, sim_next={sim_next:.4f}")
        lines.append("=" * 60)

    conversations = step.get("conversations", [])
    for conv in conversations:
        role = conv.get("role", "unknown")
        content = conv.get("content", "")
        if role == "system":
            lines.append("[System Prompt]")
            if len(content) > _SYSTEM_PROMPT_TRUNCATE_LEN:
                lines.append(content[:_SYSTEM_PROMPT_TRUNCATE_LEN] + "\n... (truncated)")
            else:
                lines.append(content)
        elif role == "user":
            lines.append("[User Prompt]")
            lines.append(content)
        elif role == "assistant":
            lines.append("[Action (Assistant Response)]")
            lines.append(content)
        lines.append("")

    lines.append(f"[Images]: {step.get('images', [])}")
    lines.append(f"[Index]: {step.get('index', 'N/A')}")
    lines.append("")
    return "\n".join(lines)


def save_fork_log(
    log_dir: str,
    task_name: str,
    succ_id: int,
    fail_id: int,
    succ_steps: List[dict],
    fail_steps: List[dict],
    match_pairs: List[Tuple[int, int, float, float]],
    filter_log: Optional[dict] = None,
    debug_info: Optional[dict] = None,
) -> None:
    """Save fork-point analysis logs without modifying the original data.

    Args:
        log_dir: Root log directory.
        task_name: Task name.
        succ_id: Success trajectory repeat_id.
        fail_id: Failure trajectory repeat_id.
        succ_steps: Success trajectory step list.
        fail_steps: Failure trajectory step list.
        match_pairs: Match pair list [(failed_step_idx, base_step_idx, sim_current, sim_next), ...].
        filter_log: Filter diagnosis log
            {failed_step_idx: [(base_step_idx, ssim_score, reason), ...]}.
        debug_info: Debug info {failed_step_idx: {"min_base_idx": int, "image_path": str}}.
    """
    pair_dir = os.path.join(log_dir, task_name, f"succ_{succ_id:02d}_vs_fail_{fail_id:02d}")
    succ_dir = os.path.join(pair_dir, "succ_trajectory")
    fail_dir = os.path.join(pair_dir, "fail_trajectory")
    os.makedirs(succ_dir, exist_ok=True)
    os.makedirs(fail_dir, exist_ok=True)

    # Build fork lookup table: {failed_step_idx: (base_step_idx, sim_current, sim_next)}
    fork_lookup: Dict[int, Tuple[int, float, float]] = {}
    for fp in match_pairs:
        failed_step_idx, base_step_idx = fp[0], fp[1]
        sim_current = fp[2] if len(fp) > 2 else -1.0
        sim_next = fp[3] if len(fp) > 3 else -1.0
        fork_lookup[failed_step_idx] = (base_step_idx, sim_current, sim_next)

    # Write summary
    _write_summary(pair_dir, task_name, succ_id, fail_id,
                   succ_steps, fail_steps, match_pairs, filter_log, debug_info)

    # Write success trajectory (no fork annotations)
    _write_trajectory(succ_dir, succ_steps, fork_lookup=None)

    # Write failure trajectory (with fork annotations)
    _write_trajectory(fail_dir, fail_steps, fork_lookup=fork_lookup)


def _write_summary(
    pair_dir: str,
    task_name: str,
    succ_id: int,
    fail_id: int,
    succ_steps: List[dict],
    fail_steps: List[dict],
    match_pairs: List[Tuple[int, int, float, float]],
    filter_log: Optional[dict] = None,
    debug_info: Optional[dict] = None,
) -> None:
    """Write a fork-point summary file."""
    with open(os.path.join(pair_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"Task: {task_name}\n")
        f.write(f"Success trajectory: repeat_{succ_id:02d} ({len(succ_steps)} steps)\n")
        f.write(f"Failed trajectory: repeat_{fail_id:02d} ({len(fail_steps)} steps)\n")
        f.write(f"Fork points found: {len(match_pairs)}\n")
        f.write("=" * 60 + "\n\n")

        for i, fp in enumerate(match_pairs):
            failed_idx, base_idx = fp[0], fp[1]
            sim_c = fp[2] if len(fp) > 2 else -1.0
            sim_n = fp[3] if len(fp) > 3 else -1.0
            f.write(f"Fork #{i+1}: fail_step={failed_idx} <-> base_step={base_idx}, "
                    f"sim_current={sim_c:.4f}, sim_next={sim_n:.4f}\n")

            # Write the action from the success trajectory at this step
            if base_idx < len(succ_steps):
                base_tool_call = extract_tool_call(succ_steps[base_idx])
                f.write(f"  Base action (correct): {base_tool_call}\n")
            # Write the action from the failure trajectory at this step
            if failed_idx < len(fail_steps):
                fail_tool_call = extract_tool_call(fail_steps[failed_idx])
                f.write(f"  Fail action (wrong):   {fail_tool_call}\n")
            f.write("\n")

        # SSIM matrix: each failed step vs each succ step
        f.write("\n" + "=" * 60 + "\n")
        f.write("SSIM Matrix: each failed step vs each succ step\n")
        f.write("=" * 60 + "\n\n")

        succ_img_indices: List[int] = []
        succ_img_paths: List[str] = []
        for s_idx, step in enumerate(succ_steps):
            img_path = get_step_image_path(step)
            if img_path:
                succ_img_indices.append(s_idx)
                succ_img_paths.append(img_path)

        if not succ_img_indices:
            f.write("(No succ steps with screenshots)\n")
        else:
            col_label = "fail\\succ"
            header = f"{col_label:>12s}"
            for s_idx in succ_img_indices:
                header += f"  s{s_idx:>3d}"
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")

            for f_idx, f_step in enumerate(fail_steps):
                f_img_path = get_step_image_path(f_step)
                if not f_img_path:
                    continue
                row = f"f{f_idx:>3d}        "
                for s_img_path in succ_img_paths:
                    sim = obs_similarity(f_img_path, s_img_path)
                    row += f"  {sim:>.3f}"
                f.write(row + "\n")

        # Adjacent failed-step SSIM (for debugging duplicate screenshots)
        f.write("\n" + "=" * 60 + "\n")
        f.write("Adjacent Failed Steps SSIM (fail_step[i] vs fail_step[i+1])\n")
        f.write("  Useful for verifying whether adjacent failed steps share the same screenshot.\n")
        f.write("=" * 60 + "\n\n")
        for f_idx in range(len(fail_steps) - 1):
            img_a = get_step_image_path(fail_steps[f_idx])
            img_b = get_step_image_path(fail_steps[f_idx + 1])
            if img_a and img_b:
                sim = obs_similarity(img_a, img_b)
                marker = "=" if sim >= 0.999 else ("~" if sim >= 0.95 else " ")
                f.write(f"  fail_step {f_idx:>3d} vs fail_step {f_idx+1:>3d}:  SSIM={sim:.6f}  {marker}\n")
        f.write("\n")

        # Per-failed-step debug info (min_base_idx, image path)
        if debug_info:
            f.write("\n" + "=" * 60 + "\n")
            f.write("Debug Info: min_base_idx and image path for each failed step\n")
            f.write("  min_base_idx is the monotonic constraint lower bound;\n")
            f.write("  only base steps with index >= min_base_idx are searched.\n")
            f.write("=" * 60 + "\n\n")
            for f_idx in sorted(debug_info.keys()):
                info = debug_info[f_idx]
                min_b = info.get("min_base_idx", "?")
                img_path = info.get("image_path", "?")
                img_name = os.path.basename(img_path) if isinstance(img_path, str) else str(img_path)
                f.write(f"  fail_step {f_idx:>3d}:  min_base_idx={min_b:>3}  image={img_name}\n")
            f.write("\n")

        # Filter diagnosis log
        if filter_log:
            f.write("\n" + "=" * 60 + "\n")
            f.write("Filter Diagnosis: why each candidate was filtered or passed\n")
            f.write("  Reasons: passed = all filters passed, "
                    "same_tool_call = identical tool_call filtered, "
                    "same_transition = identical next-step screenshot filtered\n")
            f.write("=" * 60 + "\n\n")

            for f_idx in sorted(filter_log.keys()):
                entries = filter_log[f_idx]
                f.write(f"fail_step {f_idx}:\n")
                for base_idx, ssim_score, reason in entries:
                    marker = "+" if reason == "passed" else "x"
                    f.write(f"  {marker} succ_step {base_idx:>3d}  SSIM={ssim_score:.4f}  reason={reason}\n")
                f.write("\n")


def _write_trajectory(
    traj_dir: str,
    steps: List[dict],
    fork_lookup: Optional[Dict[int, Tuple[int, float, float]]] = None,
) -> None:
    """Write per-step info and screenshots for a single trajectory.

    Args:
        traj_dir: Trajectory log directory.
        steps: Trajectory step list.
        fork_lookup: Fork lookup table {step_idx: (base_idx, sim_c, sim_n)}.
                     None means no fork annotations (used for success trajectories).
    """
    for s_idx, step in enumerate(steps):
        is_fork = fork_lookup is not None and s_idx in fork_lookup

        if is_fork:
            base_idx, sim_c, sim_n = fork_lookup[s_idx]
            suffix = f"_fork_base{base_idx}_{sim_c:.3f}_{sim_n:.3f}"
            txt_path = os.path.join(traj_dir, f"step_{s_idx}{suffix}.txt")
            img_dst_name = f"step_{s_idx}{suffix}.png"
        else:
            txt_path = os.path.join(traj_dir, f"step_{s_idx}.txt")
            img_dst_name = f"step_{s_idx}.png"

        # Write step text info
        with open(txt_path, "w", encoding="utf-8") as f:
            if is_fork:
                base_idx, sim_c, sim_n = fork_lookup[s_idx]
                f.write(_get_step_info_text(step, s_idx, is_fork=True,
                                            base_step_idx=base_idx,
                                            sim_current=sim_c, sim_next=sim_n))
            else:
                f.write(_get_step_info_text(step, s_idx))

        # Save cropped screenshot (matching the crop used during comparison)
        img_path = get_step_image_path(step)
        if img_path and os.path.isfile(img_path):
            dst_img = os.path.join(traj_dir, img_dst_name)
            arr = cv2.imread(img_path)
            if arr is not None:
                cropped = crop_top(arr)
                cv2.imwrite(dst_img, cropped)
            else:
                shutil.copy2(img_path, dst_img)
