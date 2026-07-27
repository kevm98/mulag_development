#!/usr/bin/env bash
set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

/home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  PPO2_ROM_PhysX/policy/scripts/stop_and_work.py \
  --device cuda:0 \
  --checkpoint \
    logs/rsl_rl/ppo2_rom_residual_4cyl_joint/2026-07-21_02-09-15/model_607.pt \
  --cylinder4_residual_scale_m 0.00025 \
  --command_rom_checkpoint \
    ROM-cylinders_pipeline/checkpoints/plant_reachable_v1_finetune_blend25/best_rom.pt \
  --trajectory_library_path \
    PPO2_ROM_PhysX/models/plant_reachable_trajectory_library_v3_large.json \
  --command_rom_direction_guard \
  --action_filter 0.80 \
  --cylinder4_action_filter 0.95 \
  --dls_velocity_filter_beta 0.0 \
  --dls_action_smoothing 0.20 \
  --hydraulic_target_filter_beta 0.50 \
  --ee_correction_gain 0.25 \
  --stations 5 \
  --station_spacing_m 2.0 \
  --station_shapes line,arc,sine,circle,square \
  --drive_speed_mps 0.6 \
  --path_speed_mps 0.18 \
  --plot_arm_start_tolerance_m 0.04 \
  --plot_update_steps 15 \
  --viewport_line_sample_steps 3 \
  --metrics_out \
    outputs/ppo2_rom_residual_4cyl_joint/model_607_v3_smooth_stop_and_work_metrics.json \
  --realtime \
  --quiet \
  "$@" 2>&1 | awk '
    /^\[ARM\]/ || /^\[DRIVE\]/ || /^\[METRICS\]/ ||
    /Traceback/ || /Error:/ {
      print
      fflush()
    }
  '
