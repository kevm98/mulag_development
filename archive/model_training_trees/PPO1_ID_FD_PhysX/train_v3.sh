#!/usr/bin/env bash
set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "[PPO1_A V3] Balanced large-shape training with per-iteration metrics."
echo "[PPO1_A V3] Finite safety events remain logged/penalized; resets occur at timeout."

/home/rrlab/miniconda3/envs/env_isaaclab/bin/python -u \
  PPO1_ID_FD_PhysX/policy/scripts/train_v3.py \
  --headless --device cuda:0 --num_envs 128 --seed 17 \
  --balanced_shape_rounds 40 --iterations_per_shape 2 \
  --trajectory_shapes line,arc,sine,circle,square,sine,circle,square \
  --trajectory_library_path \
    PPO2_ROM_PhysX/models/plant_reachable_trajectory_library_v3_large.json \
  --path_speed_mps 0.18 \
  --residual_scale_m 0.002 \
  --ee_correction_gain 0.05 \
  --idfd_target_filter_beta 0.0 \
  --policy_std 0.002 \
  --cylinder4_std 0.03 \
  --refinement_cylinder4_std 0.015 \
  --cylinder4_residual_scale_m 0.00025 \
  --learning_rate 5e-6 \
  --early_stop_probe_interval_rounds 2 \
  --early_stop_probe_steps 120 \
  --preflight_min_pass_fraction 0.94 \
  --early_stop_min_completion_fraction 0.95 \
  --early_stop_min_mean_improvement 0.02 \
  --early_stop_min_p95_improvement 0.02 \
  --early_stop_max_acceleration_ratio 1.05 \
  --early_stop_max_shape_acceleration_ratio 1.08 \
  --early_stop_max_shape_regression_m 0.001 \
  --early_stop_consecutive_passes 2 \
  --save_interval 10 \
  --checkpoint \
    logs/rsl_rl/id_fd_residual/2026-07-15_16-24-23/model_1500.pt \
  "$@"
