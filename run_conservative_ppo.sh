#!/usr/bin/env bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd /home/rrlab/rrlab_mulag
python kevin_integration/scripts/train_stage1_yz_tracking_ppo.py \
  --num_envs 128 --max_iterations 50 --headless --device cuda:0 \
  --trajectory_mode sequence --target_hold_steps 50 \
  --target_y_range 0.03 --target_z_range 0.01 \
  --backend explicit_effort_pid_bias --kp_scale 0.1 --kd_scale 0.5 \
  --post_reset_settle_steps 5 --bias_calibration_time_s 5.0 \
  --force_rate_limit_fraction 0.01 --velocity_filter_beta 0.8 \
  --max_cylinder_delta_m 0.005 --reward_mode distance_only \
  --action_penalty_weight 0.005 --ee_jump_penalty_weight 0.05 \
  --action_smoothing_beta 0.8 --action_rate_limit 0.1 \
  --transfer_output_scale 0.25 --ppo_lr_scale 0.1 \
  --ppo_clip_param 0.05 --ppo_entropy_coef 0.0 \
  --ppo_num_learning_epochs 2 \
  --experiment_name stage1_sequence_bias_conservative \
  --stage0_checkpoint logs/rsl_rl/stage0_yz_nopin_bc/2026-07-03_17-15-08/model_final.pt
