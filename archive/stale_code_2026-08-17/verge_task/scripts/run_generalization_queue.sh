#!/usr/bin/env bash
# Continue PPO3 and PPO4 at the CURRENT Stage-6 difficulty, then measure each
# result on six meshes x three families. The harder composed trajectory remains
# disabled: this run asks whether the policies can generalize before difficulty
# is raised.
#
# Differences from the old 40-iteration headfix driver are deliberate:
#   * 200-iteration blocks;
#   * critic and optimizer are preserved;
#   * exploration std is not re-forced at block boundaries;
#   * entropy_coef=0.001 instead of zero;
#   * training continues from the latest block instead of rolling back after
#     every regression. model_best is still retained for final evaluation.
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1
PY=/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
REPORT_ROOT=verge_task/reports/stage6/generalization_v1
mkdir -p "$REPORT_ROOT"

train_one() {
  local actuator=$1 seed_checkpoint=$2 experiment=$3
  local run_dir="logs/rsl_rl/$experiment"
  mkdir -p "$run_dir"
  echo "########## TRAIN $actuator -> $experiment ##########"
  set +e
  "$PY" verge_task/scripts/train_until_converged.py \
    --python "$PY" --actuator "$actuator" --device cuda:0 \
    --num_envs 256 --eval_envs 32 \
    --block_iterations 200 --eval_steps 960 \
    --eval_timeout_s 3600 --train_timeout_s 14400 \
    --stage 6 --learning_rate 4.0e-6 --entropy_coef 0.001 \
    --head_joint_action --contact_observation true \
    --contact_safety_reward true --head_tilt_arm_frame true \
    --head_frame_observation true --contact_latch true \
    --contact_latch_release_steps 60 --model_error_history 3 \
    --action_scale 0.0035 --arm_stiffness_scale 0.35 \
    --cylinder_target_max_lead_m 0.018 --cylinder_target_leak 0.0 \
    --tracking_amplitude 2.2 --height_schedule 0.50,0.70,0.85,1.00 \
    --strict_contact --keep_regressed_checkpoint \
    --min_blocks 3 --max_blocks 3 --restart \
    --resume "$seed_checkpoint" --run_dir "$run_dir"
  local rc=$?
  set -e
  local accepted=0
  if [ -f "$run_dir/block_provenance.jsonl" ]; then
    accepted=$(grep -c '"accepted": true' "$run_dir/block_provenance.jsonl" || true)
  fi
  echo "[$actuator TRAIN END] exit=$rc accepted=$accepted/3"
  if [ "$accepted" -ne 3 ] || [ ! -f "$run_dir/model_best.pt" ]; then
    echo "[$actuator TRAIN FAILED] refusing to evaluate incomplete artifacts"
    return 4
  fi
}

evaluate_matrix() {
  local actuator=$1 checkpoint=$2 tag=$3
  local out="$REPORT_ROOT/$tag"
  mkdir -p "$out"
  for mesh in 1337 909 4242 17 2024 555; do
    for family in mixed boom_sweep s_curve; do
      local report="$out/${family}_m${mesh}.json"
      echo "[EVAL] $tag $family mesh=$mesh"
      "$PY" verge_task/scripts/evaluate.py "$checkpoint" \
        --actuator "$actuator" --headless --device cuda:0 \
        --num_envs 32 --seed 17 --steps 960 --stage 6 --eval_terrain \
        --eval_terrain_seed "$mesh" --path_family "$family" \
        --tracking_amplitude 2.2 --contact_observation true \
        --contact_safety_reward true --head_frame_observation true \
        --head_tilt_arm_frame true --contact_latch true \
        --contact_latch_release_steps 60 --model_error_history 3 \
        --action_scale 0.0035 --cylinder_target_max_lead_m 0.018 \
        --arm_stiffness_scale 0.35 --strict_contact --report "$report" \
        > "$out/${family}_m${mesh}.log" 2>&1
    done
  done
}

PPO3_EXPERIMENT=ppo3_sole_rom_generalization_v1
PPO4_EXPERIMENT=ppo4_sole_idfd_generalization_v1

train_one rom logs/rsl_rl/ppo3_sole_rom_headfix_v1/model_best.pt "$PPO3_EXPERIMENT"
evaluate_matrix rom "logs/rsl_rl/$PPO3_EXPERIMENT/model_best.pt" ppo3

train_one idfd logs/rsl_rl/ppo4_sole_idfd_headfix_v1/model_best.pt "$PPO4_EXPERIMENT"
evaluate_matrix idfd "logs/rsl_rl/$PPO4_EXPERIMENT/model_best.pt" ppo4

echo "[GENERALIZATION QUEUE DONE] $REPORT_ROOT"
