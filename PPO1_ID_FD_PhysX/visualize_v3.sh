#!/usr/bin/env bash
set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CHECKPOINT="logs/rsl_rl/id_fd_residual_v3/2026-07-21_20-56-31/model_575.pt"
if [[ $# -gt 0 && "$1" != --* ]]; then
  CHECKPOINT="$1"
  shift
fi

echo "[PPO1_A V3] checkpoint=$CHECKPOINT"

/home/rrlab/miniconda3/envs/env_isaaclab/bin/python -u \
  PPO1_ID_FD_PhysX/policy/scripts/stop_and_work.py \
  --device cuda:0 \
  --checkpoint "$CHECKPOINT" \
  --cylinder4_residual_scale_m 0.00025 \
  --trajectory_library_path \
    PPO2_ROM_PhysX/models/plant_reachable_trajectory_library_v3_large.json \
  --path_speed_mps 0.18 \
  --stations 5 \
  --station_spacing_m 2.0 \
  --station_shapes line,arc,sine,circle,square \
  --drive_speed_mps 0.6 \
  --plot_update_steps 15 \
  --realtime \
  "$@"
