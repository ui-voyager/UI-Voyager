#!/bin/bash
#
# Android World Evaluation Script (Background)
#
# Usage: ./run_android_eval.sh [workers] [config] [start_port] [avd_name] [repeats]
# Example: ./run_android_eval.sh 8 Qwen3-VL-4B-Instruct 5556 AndroidWorldAvd 1
#
# Management:
#   tail -f logs/eval_*.log      # View logs
#   kill $(cat logs/eval.pid)    # Stop evaluation
#

set -e

# ==================== Configuration ====================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Project root and android_env path
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ANDROID_ENV_PATH="${PROJECT_ROOT}/android_env"

# Set PYTHONPATH to include android_env
export PYTHONPATH="${SCRIPT_DIR}:${ANDROID_ENV_PATH}:${PYTHONPATH}"

# Arguments
NUM_WORKERS=${1:-8}
CONFIG_NAME=${2:-Qwen3-VL-4B-Instruct}
START_PORT=${3:-5556}
AVD_NAME=${4:-AndroidWorldAvd}
N_REPEATS=${5:-1}

# Paths
CONFIG_FILE="eval/configs/${CONFIG_NAME}.yaml"
LOG_DIR="logs"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
MAIN_LOG="${LOG_DIR}/eval_${TIMESTAMP}_${CONFIG_NAME}_${NUM_WORKERS}workers.log"
PID_FILE="${LOG_DIR}/eval.pid"
EMU_PID_FILE="${LOG_DIR}/emulator_pids.txt"
AVD_COPIES_FILE="${LOG_DIR}/avd_copies.txt"

# Environment
EMULATOR_PATH="${EMULATOR_PATH:-/root/android/emulator/emulator}"
ADB_PATH="${ADB_PATH:-$HOME/android/platform-tools/adb}"
export ANDROID_SDK_ROOT="${ANDROID_SDK_ROOT:-/root/android}"
export ANDROID_AVD_HOME="${ANDROID_AVD_HOME:-/root/android/avd}"

# Emulator parameters
EMU_MEMORY=3072
EMU_CORES=2
EMU_WAIT_TIME=90

# ==================== Helper Functions ====================
log() { echo "[$(date '+%H:%M:%S')] $*"; }

die() { log "Error: $*"; exit 1; }

# ==================== AVD Management ====================
# Create worker-specific AVD copy for parallel isolation
create_avd_copy() {
    local worker_id=$1
    local target_avd="${AVD_NAME}_worker_${worker_id}"
    local source_dir="$ANDROID_AVD_HOME/${AVD_NAME}.avd"
    local target_dir="$ANDROID_AVD_HOME/${target_avd}.avd"
    local source_ini="$ANDROID_AVD_HOME/${AVD_NAME}.ini"
    local target_ini="$ANDROID_AVD_HOME/${target_avd}.ini"

    # Skip if already exists
    if [[ -d "$target_dir" ]]; then
        log "  [SKIP] AVD copy already exists: $target_avd"
        echo "$target_avd" >> "$AVD_COPIES_FILE"
        return 0
    fi

    log "  [CREATE] AVD copy: $target_avd"
    cp -r "$source_dir" "$target_dir"
    cp "$source_ini" "$target_ini"
    
    # Update INI paths
    sed -i "s|path=.*|path=${target_dir}|g" "$target_ini"
    sed -i "s|path.rel=.*|path.rel=avd/${target_avd}.avd|g" "$target_ini"
    
    # Update hardware-qemu.ini
    [[ -f "$target_dir/hardware-qemu.ini" ]] && \
        sed -i "s|${AVD_NAME}|${target_avd}|g" "$target_dir/hardware-qemu.ini"

    echo "$target_avd" >> "$AVD_COPIES_FILE"
}

# Cleanup all AVD copies
cleanup_avd_copies() {
    [[ ! -f "$AVD_COPIES_FILE" ]] && return
    
    log "Cleaning up AVD copies..."
    while read -r avd_name; do
        [[ "$avd_name" == *"_worker_"* ]] || continue
        rm -rf "$ANDROID_AVD_HOME/${avd_name}.avd" "$ANDROID_AVD_HOME/${avd_name}.ini"
        log "  [DELETE] $avd_name"
    done < "$AVD_COPIES_FILE"
    rm -f "$AVD_COPIES_FILE"
}

# ==================== Emulator Management ====================
# Kill processes (SIGTERM first, then SIGKILL)
kill_pids() {
    local pid_file=$1
    [[ ! -f "$pid_file" ]] && return
    
    while read -r pid; do
        [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
    done < "$pid_file"
    sleep 3
    while read -r pid; do
        [[ -n "$pid" ]] && kill -9 "$pid" 2>/dev/null || true
    done < "$pid_file"
}

# Cleanup all resources
cleanup() {
    log "Cleaning up resources..."
    kill_pids "$EMU_PID_FILE"
    pkill -f "emulator.*${AVD_NAME}" 2>/dev/null || true
    cleanup_avd_copies
    rm -f "$EMU_PID_FILE" "$PID_FILE"
    log "Cleanup completed"
}

# Start all emulators
start_emulators() {
    log "Starting $NUM_WORKERS emulators..."
    local failed=0

    # Create AVD copies
    log "Creating AVD copies..."
    rm -f "$AVD_COPIES_FILE"
    for i in $(seq 0 $((NUM_WORKERS - 1))); do
        create_avd_copy "$i" || ((failed++))
    done
    [[ $failed -gt 0 ]] && { log "Failed to create AVD copies: $failed"; return $failed; }

    # Start emulator instances
    failed=0
    for i in $(seq 0 $((NUM_WORKERS - 1))); do
        local console_port=$((START_PORT + i * 2))
        local adb_port=$((console_port + 1))
        local grpc_port=$((8554 + i))
        local worker_avd="${AVD_NAME}_worker_${i}"
        local emu_log="${LOG_DIR}/emulators/emu_${i}_port_${console_port}.log"

        "$EMULATOR_PATH" \
            -adb-path "$ADB_PATH" \
            -gpu swiftshader_indirect \
            -no-audio -no-skin -no-window \
            -show-kernel -verbose \
            -avd "$worker_avd" \
            -memory $EMU_MEMORY \
            -cores $EMU_CORES \
            -grpc "$grpc_port" \
            -ports "${console_port},${adb_port}" \
            -snapshot default_boot \
            -feature AllowSnapshotMigration,MigratableSnapshotSave \
            > "$emu_log" 2>&1 &
        
        local emu_pid=$!
        echo "$emu_pid" >> "$EMU_PID_FILE"
        sleep 5

        if kill -0 "$emu_pid" 2>/dev/null; then
            log "  [OK] Worker $i: avd=$worker_avd, ports=($console_port,$adb_port,$grpc_port), pid=$emu_pid"
        else
            log "  [FAIL] Worker $i failed to start"
            [[ -f "$emu_log" ]] && tail -3 "$emu_log" | sed 's/^/    /'
            ((failed++))
        fi
    done

    return $failed
}

# Wait for emulators to be ready
wait_emulators() {
    log "Waiting for emulators to start (${EMU_WAIT_TIME}s)..."
    sleep "$EMU_WAIT_TIME"

    # Count alive emulators
    ALIVE_WORKERS=0
    [[ -f "$EMU_PID_FILE" ]] && while read -r pid; do
        [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null && ((ALIVE_WORKERS++))
    done < "$EMU_PID_FILE"

    log "Alive emulators: $ALIVE_WORKERS"
    [[ $ALIVE_WORKERS -eq 0 ]] && { log "Error: All emulators stopped"; return 1; }

    # Wait for ADB devices
    log "Checking ADB devices..."
    adb devices
    
    local devices=$(adb devices | grep -c "emulator-" || echo 0)
    if [[ $devices -eq 0 ]]; then
        log "Waiting for ADB devices (30s)..."
        sleep 30
        devices=$(adb devices | grep -c "emulator-" || echo 0)
    fi

    [[ $devices -eq 0 ]] && { log "Error: No ADB devices detected"; return 1; }
    log "Detected $devices ADB devices"
    return 0
}

# ==================== Evaluation Execution ====================
run_eval() {
    local workers=$1 repeat_id=$2
    log "Starting evaluation (workers=$workers, repeat=$repeat_id)..."

    python run_eval_parallel.py \
        --config "$CONFIG_FILE" \
        --num_workers "$workers" \
        --start_port "$START_PORT" \
        --log_dir "$LOG_DIR" \
        --repeat_id "$repeat_id"

    local ret=$?
    log "Evaluation completed (exit code: $ret)"
    return $ret
}

# ==================== Pre-checks ====================
print_header() {
    cat << EOF
==============================================
Android World Evaluation (Background)
==============================================
Time: $(date)
Config: $CONFIG_FILE
Workers: $NUM_WORKERS
Start port: $START_PORT
Repeats: $N_REPEATS
==============================================
EOF
}

print_header

[[ ! -f "$CONFIG_FILE" ]] && {
    echo "Error: Config file not found: $CONFIG_FILE"
    echo "Available configs:"
    ls -1 eval/configs/*.yaml 2>/dev/null | xargs -n1 basename | sed 's/.yaml$//'
    exit 1
}
[[ ! -f "$EMULATOR_PATH" ]] && die "Emulator not found: $EMULATOR_PATH"
[[ ! -d "$ANDROID_AVD_HOME/$AVD_NAME.avd" ]] && die "AVD not found: $AVD_NAME"

mkdir -p "$LOG_DIR/emulators"
rm -f "$EMU_PID_FILE"

# ==================== Main Process (Background) ====================
(
    trap cleanup EXIT INT TERM

    for repeat_id in $(seq 0 $((N_REPEATS - 1))); do
        log "========================================"
        log "Starting round $((repeat_id + 1))/$N_REPEATS"
        log "========================================"

        # Cleanup previous round (except first)
        if [[ $repeat_id -gt 0 ]]; then
            log "Cleaning up previous round..."
            cleanup
            rm -f "$EMU_PID_FILE"
            sleep 5
        fi

        # Start emulators
        start_emulators
        failed=$?
        [[ $((NUM_WORKERS - failed)) -eq 0 ]] && { log "Error: All emulators failed to start"; continue; }
        [[ $failed -gt 0 ]] && log "Warning: $failed emulators failed to start"

        # Wait for ready
        wait_emulators || { log "Error: Emulators not ready, skipping round"; continue; }

        # Run evaluation
        run_eval "$ALIVE_WORKERS" "$repeat_id" || log "Round $((repeat_id + 1)) evaluation error"
        log "Round $((repeat_id + 1))/$N_REPEATS completed"
    done

    log "All $N_REPEATS rounds completed!"
) > "$MAIN_LOG" 2>&1 &

MAIN_PID=$!
echo "$MAIN_PID" > "$PID_FILE"
disown "$MAIN_PID"

cat << EOF

==============================================
Evaluation started in background
==============================================
Main PID: $MAIN_PID
Log file: $MAIN_LOG
==============================================

Management:
  View logs: tail -f $MAIN_LOG
  Stop: kill \$(cat $PID_FILE)

You can now safely disconnect from terminal.
EOF
