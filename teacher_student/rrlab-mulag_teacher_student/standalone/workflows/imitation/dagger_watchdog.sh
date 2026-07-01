#!/usr/bin/env bash
set -euo pipefail

############################################
# DAgger Watchdog: RAM threshold + restart
############################################
# Usage:
#   CDIR="/path/to/rrlab" ./dagger_watchdog.sh
#   CDIR="/path/to/rrlab" ./dagger_watchdog.sh <existing_pid>
#
# What it does:
# - Monitors system RAM usage (used/total + %)
# - If RAM >= threshold: stop training PID, find latest checkpoint, restart training
# - Stops permanently after deadline (e.g., 2 hours)

#
# chmod +x dagger_watchdog.sh
# CDIR="/home/qili/Software/IsaacSim_Exts/rrlab/standalone/workflows/imitation/dagger_watchdog.sh" ./dagger_watchdog.sh
# # or monitor an existing PID:
# CDIR="/home/qili/Software/IsaacSim_Exts/rrlab/standalone/workflows/imitation/dagger_watchdog.sh" ./dagger_watchdog.sh 12345
############################################

############################
# User-configurable settings
############################

# Root folder that contains changing timestamp subfolders
LOG_ROOT="/home/qili/Software/IsaacSim_Exts/rrlab/logs/dagger/RRLAB-Obstacle-Avoidance-Mulag-v0"

# Memory threshold (percent)
MEM_THRESHOLD_PCT=95

# Check interval (seconds)
CHECK_EVERY_SEC=5

# Deadline (seconds): 2 hours default
DEADLINE_SEC=$((4 * 60 * 60))

# Conda env name
CONDA_ENV="isaaclab5.0"

# Working directory (project root). You can override by exporting CDIR=...
CDIR="${CDIR:-/home/qili/Software/IsaacSim_Exts/rrlab}"

# Training command base (checkpoint will be appended on restart if found)
TRAIN_CMD_BASE=(python standalone/workflows/imitation/train.py
  --task RRLAB-Obstacle-Avoidance-Mulag-v0
  --num_envs 15
  --enable_camera
)

#################################
# Helper functions
#################################

timestamp() { date +"%Y-%m-%d %H:%M:%S"; }

# Print memory as "55.0G / 62.0G (88%)"
print_mem_usage() {
  local mem_total_kb mem_avail_kb used_kb used_pct total_gb used_gb
  mem_total_kb=$(awk '/MemTotal:/ {print $2}' /proc/meminfo)
  mem_avail_kb=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)

  used_kb=$((mem_total_kb - mem_avail_kb))
  used_pct=$((used_kb * 100 / mem_total_kb))

  total_gb=$(awk "BEGIN {printf \"%.1f\", ${mem_total_kb}/1024/1024}")
  used_gb=$(awk "BEGIN {printf \"%.1f\", ${used_kb}/1024/1024}")

  echo "${used_gb}G / ${total_gb}G (${used_pct}%)"
}

mem_used_pct() {
  local mem_total_kb mem_avail_kb used_kb pct
  mem_total_kb=$(awk '/MemTotal:/ {print $2}' /proc/meminfo)
  mem_avail_kb=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
  used_kb=$((mem_total_kb - mem_avail_kb))
  pct=$((used_kb * 100 / mem_total_kb))
  echo "${pct}"
}

kill_gracefully() {
  local pid="$1"
  if kill -0 "$pid" 2>/dev/null; then
    echo "[$(timestamp)] Stopping PID $pid (SIGTERM)..."
    kill -TERM "$pid" 2>/dev/null || true

    # wait up to 30s, then SIGKILL
    for _ in {1..30}; do
      if ! kill -0 "$pid" 2>/dev/null; then
        echo "[$(timestamp)] PID $pid exited."
        return 0
      fi
      sleep 1
    done

    echo "[$(timestamp)] PID $pid still alive, sending SIGKILL..."
    kill -KILL "$pid" 2>/dev/null || true
  fi
}

activate_conda() {
  # Works if `conda` is on PATH. If not, source your conda.sh manually.
  local conda_base
  conda_base="$(conda info --base 2>/dev/null || true)"
  if [[ -z "${conda_base}" ]]; then
    echo "[$(timestamp)] ERROR: conda not found in PATH."
    exit 1
  fi
  # shellcheck source=/dev/null
  source "${conda_base}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
}

find_latest_checkpoint() {
  # Newest timestamp folder under LOG_ROOT
  local latest_run ckpt
  latest_run=$(ls -1dt "${LOG_ROOT}"/*/ 2>/dev/null | head -n 1 || true)
  if [[ -z "${latest_run}" ]]; then
    echo ""
    return 0
  fi

  # Newest dagger_roundXXXX_policy.pt inside that run folder (any depth)
  ckpt=$(find "${latest_run}" -type f -name "dagger_round*_policy.pt" -printf "%T@ %p\n" 2>/dev/null \
      | sort -nr | head -n 1 | awk '{print $2}' || true)

  echo "${ckpt}"
}

start_training() {
  local ckpt="$1"

  # Make restarts robust: always ensure env + cwd right here
  activate_conda
  cd "${CDIR}"

  echo "[$(timestamp)] Starting training in: ${CDIR}"
  if [[ -n "${ckpt}" ]]; then
    echo "[$(timestamp)] Using checkpoint: ${ckpt}"
    "${TRAIN_CMD_BASE[@]}" --student_checkpoint "${ckpt}" &
  else
    echo "[$(timestamp)] No checkpoint found; starting fresh."
    "${TRAIN_CMD_BASE[@]}" &
  fi

  local new_pid=$!
  echo "[$(timestamp)] Started PID: ${new_pid}"
  echo "${new_pid}"
}

#################################
# Main
#################################

INITIAL_PID="${1:-}"
START_TIME_EPOCH=$(date +%s)

echo "[$(timestamp)] DAgger watchdog starting."
echo "[$(timestamp)] LOG_ROOT=${LOG_ROOT}"
echo "[$(timestamp)] MEM_THRESHOLD_PCT=${MEM_THRESHOLD_PCT}%"
echo "[$(timestamp)] CHECK_EVERY_SEC=${CHECK_EVERY_SEC}s"
echo "[$(timestamp)] DEADLINE_SEC=${DEADLINE_SEC}s"
echo "[$(timestamp)] CDIR=${CDIR}"
echo "[$(timestamp)] CONDA_ENV=${CONDA_ENV}"

PID=""
if [[ -n "${INITIAL_PID}" ]]; then
  PID="${INITIAL_PID}"
  echo "[$(timestamp)] Monitoring existing PID: ${PID}"
else
  CKPT="$(find_latest_checkpoint)"
  PID="$(start_training "${CKPT}")"
fi

while true; do
  now=$(date +%s)
  elapsed=$((now - START_TIME_EPOCH))

  # Stop permanently at deadline
  if (( elapsed >= DEADLINE_SEC )); then
    echo "[$(timestamp)] Deadline reached (${elapsed}s). Stopping and not restarting."
    kill_gracefully "${PID}"
    exit 0
  fi

  # If process died, restart (unless deadline)
  if ! kill -0 "${PID}" 2>/dev/null; then
    echo "[$(timestamp)] PID ${PID} is not running. Restarting..."
    CKPT="$(find_latest_checkpoint)"
    PID="$(start_training "${CKPT}")"
    sleep "${CHECK_EVERY_SEC}"
    continue
  fi

  # Print memory usage line each check
  mem_line="$(print_mem_usage)"
  used_pct="$(mem_used_pct)"
  echo "[$(timestamp)] Memory: ${mem_line} | PID: ${PID}"

  # If over threshold, restart
  if (( used_pct >= MEM_THRESHOLD_PCT )); then
    echo "[$(timestamp)] Memory >= ${MEM_THRESHOLD_PCT}%, restarting training..."
    kill_gracefully "${PID}"

    CKPT="$(find_latest_checkpoint)"
    if [[ -z "${CKPT}" ]]; then
      echo "[$(timestamp)] WARNING: No checkpoint found. Restarting fresh."
    fi
    PID="$(start_training "${CKPT}")"
  fi

  sleep "${CHECK_EVERY_SEC}"
done