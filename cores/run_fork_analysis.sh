#!/bin/bash

# =============================================================================
# Fork Step Analysis Runner
# =============================================================================
# Runs fork_main.py using the backward state matching algorithm to analyze
# fork points between success and failure trajectories.
#
# Expected data directory structure (produced by run_group_sample.sh):
#   eval_results/{MODEL_NAME}/results/{TIMESTAMP}/seed_{N}/{TaskName}/
#       repeat_{N}_succ.jsonl
#       repeat_{N}_fail.jsonl
#
# Output:
#   - fork_analysis_report.json    Fork point statistics report
#   - fork_steps.jsonl             Fork step SFT data (success response
#                                  replaces failure response at fork points)
#   - {model}/{timestamp}/{seed}/  Detailed fork logs per seed
#
# Usage:
#   # Analyze a specific model's data
#   DATA_DIR=eval_results/Qwen3-VL-4B-Instruct bash cores/run_fork_analysis.sh
#
#   # Analyze all models
#   DATA_DIR=eval_results bash cores/run_fork_analysis.sh
#
# Environment variables (override defaults):
#   DATA_DIR           - Root data directory (REQUIRED)
#   SSIM_THRESHOLD     - SSIM screenshot similarity threshold (default: 0.95)
#   MODE               - Pairing strategy: shortest_base / all_pairs (default: shortest_base)
#   NUM_WORKERS        - Number of parallel workers (default: 32)
#   LOG_DIR            - Log output directory (default: {DATA_DIR}/fork_analysis_logs/)
#   MAX_LOG_SEEDS      - Only write disk logs for the first N seeds (default: 5)
# =============================================================================

set -euo pipefail

# ==================== Configuration ====================

DATA_DIR="${DATA_DIR:-}"
SSIM_THRESHOLD="${SSIM_THRESHOLD:-0.95}"
MODE="${MODE:-shortest_base}"
NUM_WORKERS="${NUM_WORKERS:-32}"
LOG_DIR="${LOG_DIR:-}"
MAX_LOG_SEEDS="${MAX_LOG_SEEDS:-5}"

if [[ -z "${DATA_DIR}" ]]; then
    echo "Error: DATA_DIR is required. Set it via environment variable."
    echo ""
    echo "  Example (single model):"
    echo "    DATA_DIR=eval_results/Qwen3-VL-4B-Instruct bash $0"
    echo ""
    echo "  Example (all models):"
    echo "    DATA_DIR=eval_results bash $0"
    exit 1
fi

# ==================== Locate script and project directories ====================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Resolve DATA_DIR relative to PROJECT_ROOT if not absolute
if [[ "${DATA_DIR}" != /* ]]; then
    DATA_DIR="${PROJECT_ROOT}/${DATA_DIR}"
fi

# ==================== Build command ====================

CMD=(
    python "${SCRIPT_DIR}/fork_main.py"
    --data_dir "${DATA_DIR}"
    --ssim_threshold "${SSIM_THRESHOLD}"
    --mode "${MODE}"
    --num_workers "${NUM_WORKERS}"
    --max_log_seeds "${MAX_LOG_SEEDS}"
)

[[ -n "${LOG_DIR}" ]] && CMD+=(--log_dir "${LOG_DIR}")

# ==================== Print and execute ====================

echo "=========================================="
echo "Fork Step Analysis"
echo "=========================================="
echo "PROJECT_ROOT:   ${PROJECT_ROOT}"
echo "DATA_DIR:       ${DATA_DIR}"
echo "SSIM_THRESHOLD: ${SSIM_THRESHOLD}"
echo "MODE:           ${MODE}"
echo "NUM_WORKERS:    ${NUM_WORKERS}"
echo "MAX_LOG_SEEDS:  ${MAX_LOG_SEEDS}"
echo "LOG_DIR:        ${LOG_DIR:-(default: DATA_DIR/fork_analysis_logs/)}"
echo "=========================================="
echo ""
echo "Running: ${CMD[*]}"
echo ""

"${CMD[@]}"
