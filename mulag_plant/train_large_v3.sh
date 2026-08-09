#!/usr/bin/env bash
set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "[PPO2_A V3] Each shape is a short PPO phase; ETA is phase-local."
echo "[PPO2_A V3] Finite safety events remain logged/penalized; resets occur at episode timeout."

/home/rrlab/miniconda3/envs/env_isaaclab/bin/python -u \
  mulag_plant/policy_4cyl/scripts/train.py \
  --headless --device cuda:0 --num_envs 128 --seed 17 \
  --joint_finetune \
  --balanced_shape_rounds 40 --iterations_per_shape 2 \
  --trajectory_shapes line,arc,sine,circle,square,sine,circle,square \
  --trajectory_library_path \
    mulag_plant/models/plant_reachable_trajectory_library_v3_large.json \
  --command_rom_checkpoint \
    ROM-cylinders_pipeline/checkpoints/plant_reachable_v1_finetune_blend25/best_rom.pt \
  --command_rom_direction_guard \
  --path_speed_mps 0.18 \
  --cylinder4_low_scale_m 0.00025 \
  --cylinder123_std 0.002 \
  --cylinder4_std 0.03 \
  --refinement_cylinder4_std 0.015 \
  --learning_rate 5e-6 \
  --early_stop_probe_interval_rounds 2 \
  --early_stop_probe_steps 120 \
  --early_stop_min_completion_fraction 0.95 \
  --early_stop_min_mean_improvement 0.02 \
  --early_stop_min_p95_improvement 0.02 \
  --early_stop_max_acceleration_ratio 1.05 \
  --early_stop_max_shape_acceleration_ratio 1.08 \
  --early_stop_max_shape_regression_m 0.001 \
  --early_stop_consecutive_passes 2 \
  --save_interval 10 \
  --checkpoint3 \
    logs/rsl_rl/ppo2_rom_residual/2026-07-19_18-55-23/model_249.pt \
  "$@"
