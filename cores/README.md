# Fork Algorithm — Fork Step SFT Data Generation

Identify critical **fork points** between success and failure trajectories using a **backward state matching** algorithm, then generate SFT training data that teaches a model the correct action at each fork.

## Overview

Given a set of task trajectories (some successful, some failed), the algorithm:

1. **Pairs** each failure trajectory with one or more success trajectories.
2. **Walks backward** through the success trajectory and, for each failure step, finds the state (screenshot) that matches but where the action differs — the *fork point*.
3. **Builds SFT entries** by taking the failure step's context (system prompt, user prompt, screenshot) and replacing the assistant response with the success trajectory's correct action.

The result is a JSONL file (`fork_steps.jsonl`) of training examples that focus on the exact moments where the model made the wrong decision.

## Prerequisites

- Python 3.8+
- [opencv-python](https://pypi.org/project/opencv-python/)
- [scikit-image](https://pypi.org/project/scikit-image/)
- [numpy](https://pypi.org/project/numpy/)
- [tqdm](https://pypi.org/project/tqdm/)

```bash
pip install opencv-python scikit-image numpy tqdm
```

## Data Format

The data directory is produced by `run_group_sample.sh` (in the project root) and follows this layout:

```
eval_results/{MODEL_NAME}/results/{TIMESTAMP}/
├── config.yaml
├── seed_0/
│   ├── {TaskName}/
│   │   ├── repeat_0_succ.jsonl
│   │   ├── repeat_1_fail.jsonl
│   │   └── ...
│   └── ...
├── seed_1/
│   └── ...
└── ...
```

Two directory layouts are auto-detected under `<data_dir>/results/`:

- **3-level** (multiple models): `results/{model_name}/{timestamp}/seed_{N}/{TaskName}/`
- **2-level** (single model): `results/{timestamp}/seed_{N}/{TaskName}/`

An optional `sft_data/` sub-level (`seed_{N}/sft_data/{TaskName}/`) is also recognised.

Each JSONL line is one step:

```json
{
  "conversations": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "images": ["path/to/screenshot.png"],
  "index": 0
}
```

## Usage

### Step 1 — Collect Trajectories (`run_group_sample.sh`)

`run_group_sample.sh` in the project root launches parallel Android emulators and runs the evaluation agent repeatedly to collect diverse success/failure trajectories.

```bash
# Basic usage (defaults: 3 seeds, 8 repeats per seed, 4 workers)
./run_group_sample.sh

# Custom configuration via environment variables
SEEDS="42,43,44" N_REPEATS_PER_SEED=5 NUM_WORKERS=16 ./run_group_sample.sh

# Monitor / stop
tail -f eval_results/<model>/logs/<timestamp>/eval_*.log
kill $(cat eval_results/<model>/logs/<timestamp>/eval.pid)
```

Key environment variables:

| Variable | Default | Description |
|---|---|---|
| `SEEDS` | `0,1,2` | Comma-separated seed list |
| `N_REPEATS_PER_SEED` | `8` | Repeats per seed |
| `CONFIG_NAME` | `Qwen3-VL-4B-Instruct` | Config / model name |
| `NUM_WORKERS` | `4` | Parallel emulator count |

> **Note:** Before running, update the Android SDK paths (`EMULATOR_PATH`, `ADB_PATH`, `ANDROID_SDK_ROOT`, `ANDROID_AVD_HOME`) to match your local environment.

### Step 2 — Fork Analysis (`cores/run_fork_analysis.sh`)

```bash
# Analyze a single model
DATA_DIR=eval_results/Qwen3-VL-4B-Instruct bash cores/run_fork_analysis.sh

# Analyze all models
DATA_DIR=eval_results bash cores/run_fork_analysis.sh

# With overrides
DATA_DIR=eval_results SSIM_THRESHOLD=0.99 MODE=all_pairs bash cores/run_fork_analysis.sh
```

Or call `fork_main.py` directly:

```bash
python cores/fork_main.py \
    --data_dir eval_results/Qwen3-VL-4B-Instruct \
    --ssim_threshold 0.95 \
    --mode all_pairs
```

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | *(required)* | Root data directory |
| `--ssim_threshold` | `0.95` | SSIM screenshot similarity threshold |
| `--mode` | `shortest_base` | Pairing strategy: `shortest_base` or `all_pairs` |
| `--num_workers` | `32` | Number of parallel workers |
| `--log_dir` | `{data_dir}/fork_analysis_logs/` | Log output directory |
| `--max_log_seeds` | `5` | Only write detailed disk logs for the first N seeds |

## Algorithm

### Backward State Matching

1. **For each failed step**, search all success (base) steps for a candidate where the screenshots are similar (SSIM >= threshold) but the actions differ.
2. **Transition filtering** — for non-terminal steps, verify that the *next* screenshots differ; otherwise the different actions had the same effect (false positive).
3. **Best teacher selection** — pick the candidate with the highest SSIM; ties broken by earliest index.
4. **Monotonic constraint** — matched base step indices must be non-decreasing to preserve trajectory ordering.
5. **Transition advance** — if a failed step's transition `(s, a, s')` matches a base step, the failure trajectory correctly passed that point; advance the lower bound and skip.

### Pairing Strategies

- **`shortest_base`** (default): Pair the shortest success trajectory with every failure trajectory. Faster, higher-quality fork points.
- **`all_pairs`**: Full Cartesian product of success x failure trajectories.

### Screenshot Comparison Pipeline

1. **Mean-hash pre-filter** — O(1) rejection of obviously different pairs.
2. **Pixel-exact match** — return 1.0 if thumbnails are identical.
3. **SSIM on thumbnails** (480x270) — 5-10x faster than full-resolution.

All images are cropped at the top (`_CROP_TOP_RATIO = 25/878`) to remove the device status bar. Adjust this constant in `fork_utils.py` if your device has a different status bar size.

## Output

### `fork_steps.jsonl`

Each line is a fork SFT training entry — the failure step's context with the assistant response replaced by the correct action:

```json
{
  "conversations": [...],
  "images": [...],
  "fork_info": {
    "task_name": "TaskName",
    "succ_id": 0, "fail_id": 1,
    "failed_step_idx": 3, "base_step_idx": 2,
    "sim_current": 0.9812, "sim_next": 0.4510
  }
}
```

### `fork_analysis_report.json`

Aggregated statistics per seed, including per-task breakdowns.

### Log Directory

For each analyzed pair (up to `--max_log_seeds` seeds):

```
{log_dir}/{model}/{timestamp}/{seed}/{TaskName}/succ_00_vs_fail_01/
├── summary.txt           # Fork summary with SSIM matrix
├── succ_trajectory/      # Success trajectory steps + screenshots
└── fail_trajectory/      # Failure trajectory with fork annotations
```

## Tunable Constants

Defined in `fork_utils.py`:

| Constant | Value | Description |
|---|---|---|
| `_SSIM_RESIZE_DIM` | `(480, 270)` | Thumbnail size for SSIM computation |
| `_CROP_TOP_RATIO` | `25/878` | Top crop ratio (status bar removal) |
| `_HASH_SIZE` | `16` | Mean-hash grid size |
| `_HASH_FAST_REJECT_THRESHOLD` | `0.80` | Hash similarity below this skips SSIM |

## License
