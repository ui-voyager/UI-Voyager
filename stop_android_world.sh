#!/bin/bash

# Android World Evaluation Stop Script
# Used to interrupt evaluation, clean up evaluation processes and resources
#
# Usage: ./stop_android_world.sh
# Or specify a log directory:
#   ./stop_android_world.sh /path/to/log_dir
#
# Functions:
#   1. Stop the main evaluation process
#   2. Stop all emulator processes
#   3. Stop all ADB servers
#   4. Clean up AVD copies
#   5. Clean up residual Python and QEMU processes

# Note: Not using set -e because cleanup commands (e.g. pgrep, pkill) return non-zero when no processes are found
# set -e

# ==================== Configuration ====================

# AVD name (must match the start script)
AVD_NAME="${AVD_NAME:-AndroidWorldAvd}"

# Number of parallel workers (used for cleaning up ADB servers)
NUM_WORKERS="${NUM_WORKERS:-16}"

# ADB Server starting port
ADB_SERVER_START_PORT="${ADB_SERVER_START_PORT:-5037}"

# ==================== Path Configuration ====================

# Get script directory and switch to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
cd "${PROJECT_ROOT}" || exit 1

# Log directory (can be specified via argument)
if [[ -n "$1" ]]; then
    LOG_DIR="$1"
else
    # Find the latest log directory
    LOG_BASE="${PROJECT_ROOT}/eval_results"
    if [[ -d "$LOG_BASE" ]]; then
        LOG_DIR=$(ls -dt "$LOG_BASE"/*/ 2>/dev/null | head -1)
    fi
fi

# ADB path
ADB_PATH="${ADB_PATH:-$HOME/android/platform-tools/adb}"

# AVD directory
export ANDROID_AVD_HOME="${ANDROID_AVD_HOME:-/root/android/avd}"

# ==================== Helper Functions ====================

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log_section() {
    echo ""
    log "=========================================="
    log "$*"
    log "=========================================="
}

# Safely terminate a process (SIGTERM first, then SIGKILL)
safe_kill() {
    local pid=$1
    local name=$2
    
    if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
        return 0
    fi
    
    log "  [TERM] PID $pid ($name)"
    kill "$pid" 2>/dev/null || true
    sleep 1
    
    if kill -0 "$pid" 2>/dev/null; then
        log "  [KILL] PID $pid ($name) - force kill"
        kill -9 "$pid" 2>/dev/null || true
    fi
}

# Terminate processes from PID file
kill_from_pid_file() {
    local pid_file=$1
    local name=$2
    
    if [[ ! -f "$pid_file" ]]; then
        log "  [SKIP] PID file not found: $pid_file"
        return 0
    fi
    
    log "  Reading PID file: $pid_file"
    while read -r pid; do
        [[ -n "$pid" ]] && safe_kill "$pid" "$name"
    done < "$pid_file"
    
    rm -f "$pid_file"
}

# ==================== Cleanup Functions ====================

# 1. Stop main evaluation process
stop_main_process() {
    log_section "1. Stop main evaluation process"
    
    if [[ -n "$LOG_DIR" && -f "${LOG_DIR}/eval.pid" ]]; then
        kill_from_pid_file "${LOG_DIR}/eval.pid" "main eval process"
    else
        log "  [INFO] eval.pid file not found"
    fi
    
    # Find and terminate all test_android_world.py processes
    local pids=$(pgrep -f "python.*test_android_world.py" 2>/dev/null || true)
    if [[ -n "$pids" ]]; then
        log "  Found test_android_world.py processes:"
        for pid in $pids; do
            safe_kill "$pid" "test_android_world.py"
        done
    else
        log "  [OK] No running test_android_world.py processes"
    fi
}

# 2. Stop emulator processes
stop_emulators() {
    log_section "2. Stop emulator processes"
    
    # Terminate from PID file
    if [[ -n "$LOG_DIR" && -f "${LOG_DIR}/emulator_pids.txt" ]]; then
        kill_from_pid_file "${LOG_DIR}/emulator_pids.txt" "emulator"
    fi
    
    # Find and terminate all related emulators by process name
    local emu_pids=$(pgrep -f "emulator.*${AVD_NAME}" 2>/dev/null || true)
    if [[ -n "$emu_pids" ]]; then
        log "  Found emulator processes:"
        for pid in $emu_pids; do
            safe_kill "$pid" "emulator"
        done
    fi
    
    # Terminate residual QEMU processes
    local qemu_pids=$(pgrep -f "qemu-system-x86" 2>/dev/null || true)
    if [[ -n "$qemu_pids" ]]; then
        log "  Found QEMU processes:"
        for pid in $qemu_pids; do
            safe_kill "$pid" "qemu-system-x86"
        done
    fi
    
    # Force cleanup
    sleep 2
    pkill -9 -f "emulator.*${AVD_NAME}" 2>/dev/null || true
    pkill -9 -f "qemu-system-x86" 2>/dev/null || true
    
    log "  [OK] Emulator process cleanup completed"
}

# 3. Stop all ADB Servers
stop_adb_servers() {
    log_section "3. Stop ADB Servers"
    
    # Stop each worker's independent ADB server
    for i in $(seq 0 $((NUM_WORKERS - 1))); do
        local adb_server_port=$((ADB_SERVER_START_PORT + i))
        if "$ADB_PATH" -P "$adb_server_port" kill-server 2>/dev/null; then
            log "  [OK] Stopped ADB server (port: $adb_server_port)"
        fi
    done
    
    # Stop default ADB server
    if "$ADB_PATH" kill-server 2>/dev/null; then
        log "  [OK] Stopped default ADB server"
    fi
    
    log "  [OK] ADB servers cleanup completed"
}

# 4. Clean up AVD copies
cleanup_avd_copies() {
    log_section "4. Clean up AVD copies"
    
    local avd_copies_file=""
    if [[ -n "$LOG_DIR" && -f "${LOG_DIR}/avd_copies.txt" ]]; then
        avd_copies_file="${LOG_DIR}/avd_copies.txt"
    fi
    
    local cleaned=0
    
    # Clean up from record file
    if [[ -f "$avd_copies_file" ]]; then
        log "  Cleaning up AVD copies from record file..."
        while read -r avd_name; do
            [[ "$avd_name" == *"_worker_"* ]] || continue
            if [[ -d "$ANDROID_AVD_HOME/${avd_name}.avd" ]]; then
                rm -rf "$ANDROID_AVD_HOME/${avd_name}.avd" "$ANDROID_AVD_HOME/${avd_name}.ini"
                log "  [DELETE] $avd_name"
                cleaned=$((cleaned + 1))
            fi
        done < "$avd_copies_file"
        rm -f "$avd_copies_file"
    fi
    
    # Scan and clean up all matching AVD copies
    log "  Scanning for residual AVD copies..."
    for avd_dir in "$ANDROID_AVD_HOME"/${AVD_NAME}_worker_*.avd; do
        if [[ -d "$avd_dir" ]]; then
            local avd_name=$(basename "$avd_dir" .avd)
            rm -rf "$avd_dir" "$ANDROID_AVD_HOME/${avd_name}.ini"
            log "  [DELETE] $avd_name"
            cleaned=$((cleaned + 1))
        fi
    done
    
    if [[ $cleaned -eq 0 ]]; then
        log "  [OK] No AVD copies to clean up"
    else
        log "  [OK] Cleaned up $cleaned AVD copies"
    fi
}

# 5. Clean up residual processes
cleanup_residual_processes() {
    log_section "5. Clean up residual processes"
    
    # Clean up residual Python evaluation processes
    local python_pids=$(pgrep -f "python.*android_world" 2>/dev/null || true)
    if [[ -n "$python_pids" ]]; then
        log "  Found residual Python processes:"
        for pid in $python_pids; do
            safe_kill "$pid" "python (android_world)"
        done
    else
        log "  [OK] No residual Python processes"
    fi
    
    # Clean up all ADB-related processes (logcat, shell, etc.)
    log "  Cleaning up ADB-related processes..."
    local adb_pids=$(pgrep -f "adb.*logcat|adb.*shell|adb.*exec" 2>/dev/null || true)
    if [[ -n "$adb_pids" ]]; then
        log "  Found ADB subprocesses:"
        for pid in $adb_pids; do
            safe_kill "$pid" "adb subprocess"
        done
    fi
    
    # Force cleanup all ADB processes related to the current user
    pkill -f "adb -P 50" 2>/dev/null || true
    sleep 1
    pkill -9 -f "adb -P 50" 2>/dev/null || true

    # Clean up adb fork-server processes (adb -L tcp:PORT fork-server server ...)
    local fork_server_pids=$(pgrep -f "adb.*fork-server" 2>/dev/null || true)
    if [[ -n "$fork_server_pids" ]]; then
        log "  Found adb fork-server processes:"
        for pid in $fork_server_pids; do
            safe_kill "$pid" "adb fork-server"
        done
        sleep 1
        pkill -9 -f "adb.*fork-server" 2>/dev/null || true
    fi

    # Clean up all residual adb server processes (adb -L tcp:PORT ...)
    local adb_server_pids=$(pgrep -f "adb -L tcp:" 2>/dev/null || true)
    if [[ -n "$adb_server_pids" ]]; then
        log "  Found adb server (-L) processes:"
        for pid in $adb_server_pids; do
            safe_kill "$pid" "adb server"
        done
        sleep 1
        pkill -9 -f "adb -L tcp:" 2>/dev/null || true
    fi
    
    # Clean up crash_service processes (emulator-related)
    pkill -f "crash_service" 2>/dev/null || true
    
    log "  [OK] Residual process cleanup completed"
}

# 5.1 Clean up zombie processes
cleanup_zombie_processes() {
    log_section "5.1 Clean up zombie processes"
    
    # Count zombie processes
    local zombie_count=$(ps aux 2>/dev/null | awk '$8 ~ /Z/ {print}' | wc -l)
    log "  Current zombie process count: $zombie_count"
    
    if [[ $zombie_count -gt 0 ]]; then
        log "  Zombie process list (top 20):"
        ps aux 2>/dev/null | awk '$8 ~ /Z/ {print "    PID:", $2, "PPID:", $3, "CMD:", $11}' | head -20
        
        # Find parent processes of zombies and try to notify them
        local zombie_ppids=$(ps aux 2>/dev/null | awk '$8 ~ /Z/ {print $3}' | sort -u)
        if [[ -n "$zombie_ppids" ]]; then
            log "  Attempting to notify parent processes of zombies..."
            for ppid in $zombie_ppids; do
            # Send SIGCHLD to let parent processes reap children
                kill -SIGCHLD "$ppid" 2>/dev/null || true
            done
            sleep 2
            
            # Recount
            local new_zombie_count=$(ps aux 2>/dev/null | awk '$8 ~ /Z/ {print}' | wc -l)
            log "  Zombie process count after cleanup: $new_zombie_count"
        fi
    else
        log "  [OK] No zombie processes"
    fi
}

# 6. Clean up temporary files
cleanup_temp_files() {
    log_section "6. Clean up temporary files"
    
    # Clean up lock files
    local lock_files=$(find /tmp -name "*.lock" -user "$(whoami)" 2>/dev/null | grep -i "android\|emulator\|avd" || true)
    if [[ -n "$lock_files" ]]; then
        log "  Cleaning up lock files..."
        echo "$lock_files" | while read -r f; do
            rm -f "$f" && log "  [DELETE] $f"
        done
    fi
    
    # Clean up emulator temp files in /tmp
    rm -rf /tmp/android-*/emulator-* 2>/dev/null || true
    
    log "  [OK] Temporary file cleanup completed"
}

# ==================== Main Process ====================

main() {
    echo ""
    echo "=========================================="
    echo "Android World Evaluation Stop Script"
    echo "=========================================="
    echo "Time: $(date)"
    echo "Log dir: ${LOG_DIR:-not specified}"
    echo "AVD name: ${AVD_NAME}"
    echo "Num workers: ${NUM_WORKERS}"
    echo "ADB Server start port: ${ADB_SERVER_START_PORT}"
    echo "=========================================="
    
    # Execute cleanup steps
    stop_main_process
    stop_emulators
    stop_adb_servers
    cleanup_avd_copies
    cleanup_residual_processes
    cleanup_zombie_processes
    cleanup_temp_files
    
    log_section "Cleanup completed"
    log "All evaluation processes and resources have been cleaned up"
    
    # Show current status
    echo ""
    log "Current process status check:"
    local remaining_emu=$(pgrep -f "emulator" 2>/dev/null | wc -l || echo 0)
    local remaining_qemu=$(pgrep -f "qemu-system" 2>/dev/null | wc -l || echo 0)
    local remaining_python=$(pgrep -f "test_android_world" 2>/dev/null | wc -l || echo 0)
    local remaining_adb=$(pgrep -f "adb" 2>/dev/null | wc -l || echo 0)
    local remaining_zombie=$(ps aux 2>/dev/null | awk '$8 ~ /Z/' | wc -l || echo 0)
    
    log "  - Emulator processes: $remaining_emu"
    log "  - QEMU processes: $remaining_qemu"
    log "  - Eval Python processes: $remaining_python"
    local remaining_adb_fork=$(pgrep -f "adb.*fork-server" 2>/dev/null | wc -l || echo 0)
    log "  - ADB processes: $remaining_adb"
    log "  - ADB fork-server processes: $remaining_adb_fork"
    log "  - Zombie processes: $remaining_zombie"
    
    if [[ $remaining_emu -eq 0 && $remaining_qemu -eq 0 && $remaining_python -eq 0 && $remaining_adb_fork -eq 0 ]]; then
        log "  [OK] All main processes have been cleaned up"
    else
        log "  [WARN] There may be residual processes, please check manually"
    fi
    
    if [[ $remaining_zombie -gt 0 ]]; then
        log "  [WARN] Zombie processes exist, may need to restart the system or kill their parent processes to fully clean up"
    fi
    
    echo ""
}

# Run main process
main "$@"
