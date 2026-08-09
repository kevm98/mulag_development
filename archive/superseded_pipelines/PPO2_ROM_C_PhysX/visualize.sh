#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 /path/to/ppo2_c_checkpoint.pt [extra options]" >&2
  exit 2
fi
CHECKPOINT="$1"
shift

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

exec /home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  PPO2_ROM_PhysX/policy/scripts/stop_and_work.py \
  --task_id RRLAB-Mulag-PPO2-ROM-C-AAM-Policy-PLAY-v0 \
  --task_module PPO2_ROM_C_PhysX.tasks \
  --runner_cfg PPO2_ROM_C_PhysX.env_cfg:PPO2CAAMPPORunnerCfg \
  --checkpoint "$CHECKPOINT" \
  --aam_checkpoint AAM_SystemFactor/checkpoints/aam_system_factor_runtime.pt \
  --command_rom_checkpoint \
    ROM-cylinders_pipeline/checkpoints/plant_reachable_v1_finetune_blend25/best_rom.pt \
  --trajectory_library_path \
    PPO2_ROM_PhysX/models/plant_reachable_trajectory_library_v3_circle_c3.json \
  --cylinder4_residual_scale_m 0.00025 \
  --command_rom_direction_guard \
  --action_filter 0.80 \
  --cylinder4_action_filter 0.95 \
  --dls_velocity_filter_beta 0.0 \
  --dls_action_smoothing 0.20 \
  --hydraulic_target_filter_beta 0.50 \
  --ee_correction_gain 0.25 \
  --station_spacing_m 2.0 \
  --station_shapes line,arc,sine,circle,square \
  --drive_speed_mps 0.6 \
  --path_speed_mps 0.18 \
  --plot_arm_start_tolerance_m 0.04 \
  --plot_update_steps 15 \
  --viewport_line_sample_steps 3 \
  --metrics_out outputs/ppo2_rom_c/stop_and_work_metrics.json \
  --realtime \
  --quiet \
  "$@"
