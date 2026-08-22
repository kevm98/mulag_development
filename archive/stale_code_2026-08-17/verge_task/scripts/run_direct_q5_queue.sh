#!/usr/bin/env bash
# Warm-start PPO3 and PPO4 with direct Messerkopf q5 control. The source
# checkpoints are immutable; each was expanded 98x4 -> 106x5 with neutral new
# weights and requires an optimizer reset on its first block.
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1
PY=/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
PPO3_SEED=logs/rsl_rl/ppo3_sole_rom_four_action_recovery_v1/model_direct_q5_106x5.pt
PPO4_SEED=experiments/preflight_four_action_idfd/model_direct_q5_106x5.pt

for checkpoint in "$PPO3_SEED" "$PPO4_SEED"; do
  "$PY" verge_task/scripts/preflight.py \
    --checkpoint "$checkpoint" --expected_width 106 --expected_actions 5
done

train_one() {
  local actuator=$1 checkpoint=$2 run_dir=$3
  "$PY" verge_task/scripts/train_until_converged.py \
    --python "$PY" --actuator "$actuator" --device cuda:0 \
    --resume "$checkpoint" --run_dir "$run_dir" --restart \
    --stage 6 --num_envs 256 --eval_envs 32 --eval_steps 960 \
    --block_iterations 100 --min_blocks 3 --max_blocks 3 \
    --learning_rate 5.0e-7 --entropy_coef 0.001 \
    --head_joint_action --q5_exploration_std 0.04 \
    --reset_optimizer_first_block \
    --contact_observation true --contact_safety_reward true \
    --head_tilt_arm_frame true --head_frame_observation true \
    --contact_latch true --contact_latch_release_steps 60 \
    --model_error_history 3 --action_scale 0.0035 \
    --arm_stiffness_scale 0.35 --cylinder_target_max_lead_m 0.018 \
    --cylinder_target_leak 0.0 --tracking_amplitude 2.2 \
    --height_schedule 0.50,0.70,0.85,1.00 --strict_contact
}

train_one rom "$PPO3_SEED" logs/rsl_rl/ppo3_sole_rom_direct_q5_v1
train_one idfd "$PPO4_SEED" logs/rsl_rl/ppo4_sole_idfd_direct_q5_v1

echo "[DIRECT-Q5 QUEUE COMPLETE]"
