#!/usr/bin/env bash
# Stage 6: flat ground with random raised chunks, wide tracking path.
#
#   bash verge_task/scripts/run_stage6_training.sh rom
#   bash verge_task/scripts/run_stage6_training.sh idfd
#
# WHY A RAMP. The stage-5 checkpoints do NOT transfer. Measured on stage 6,
# 32 envs x 960 steps, three random meshes:
#
#   PPO3   17-gate  contact  lat err  lifts  airborne  max gap  head align
#   m1337    FAIL   0.8812   0.1797    272     317      1.273     0.5911
#   m4242    FAIL   0.9384   0.1583    164      62      1.186     0.7476
#   m909     FAIL   0.9487   0.1572    157      71      0.670     0.9491
#   PPO4     FAIL   0.908-0.935  0.154-0.188  162-221  35-43  0.54-0.70  0.55-0.59
#
# PPO3 went from 32/32 passing on stage 5 to 0/3 here, and the dominant new
# failure is HEAD ALIGNMENT. That is not a surprise and it is not new: the
# repository already records "a wider path demands more slew and slew tilts the
# head, so this widening AMPLIFIES the outstanding head coupling", measured at
# 1.8x giving alignment 0.3379 on an untrained checkpoint. Stage 6 is
# off-distribution twice over -- new ground AND a 2.2x path.
#
# So the amplitude is ramped rather than jumped. Each sub-stage keeps the same
# ground (flat_patches, randomised per environment and per block) and widens
# only the tracking line:
#
#   S6a  amplitude 1.4   relief 0.50/0.70
#   S6b  amplitude 1.8   relief 0.70/0.85/1.00
#   S6c  amplitude 2.2   relief 0.50/0.70/0.85/1.00   <- the stage-6 contract
#
# TERRAIN RANDOMISATION. train.py seeds the mesh from --seed, and the
# supervisor varies that per block (seed + 10000 + block_index), so every block
# draws a DIFFERENT set of chunks -- different count, positions, heights and
# widths -- in every parallel environment. The policy cannot memorise a layout;
# that is the adaptivity this stage is for.
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1
PY=/home/rrlab/miniconda3/envs/env_isaaclab/bin/python

ACTUATOR=${1:-rom}
case "$ACTUATOR" in
  rom)  EXPERIMENT=ppo3_sole_rom_stage6_v1;  BASE=logs/rsl_rl/ppo3_sole_rom_contact_safe_v1 ;;
  idfd) EXPERIMENT=ppo4_sole_idfd_stage6_v1; BASE=logs/rsl_rl/ppo4_sole_idfd_contact_safe_v1 ;;
  *) echo "usage: $0 {rom|idfd}"; exit 2 ;;
esac

ROOT_DIR=logs/rsl_rl/$EXPERIMENT
mkdir -p "$ROOT_DIR"
SEED_CKPT=${SEED_CKPT:-$BASE/model_best.pt}
[ -f "$SEED_CKPT" ] || { echo "missing seed checkpoint $SEED_CKPT"; exit 3; }

# Plant frozen from the V2 Phase 1 audit. Unchanged by this stage.
ACTION_SCALE=${ACTION_SCALE:-0.0035}
ARM_STIFFNESS=${ARM_STIFFNESS:-0.35}
TARGET_LEAD=${TARGET_LEAD:-0.018}

NUM_ENVS=${NUM_ENVS:-256}
EVAL_ENVS=${EVAL_ENVS:-32}
BLOCK_ITERS=${BLOCK_ITERS:-40}
LR=${LR:-1.0e-5}

common=(
  --python "$PY" --actuator "$ACTUATOR" --device cuda:0
  --num_envs "$NUM_ENVS" --eval_envs "$EVAL_ENVS"
  --block_iterations "$BLOCK_ITERS" --eval_steps 960
  --eval_timeout_s 3600 --train_timeout_s 7200
  --stage 6 --learning_rate "$LR" --head_joint_action
  --contact_observation true --contact_safety_reward true
  --contact_latch true --contact_latch_release_steps 60
  --model_error_history 3
  --action_scale "$ACTION_SCALE" --arm_stiffness_scale "$ARM_STIFFNESS"
  --cylinder_target_max_lead_m "$TARGET_LEAD" --cylinder_target_leak 0.0
  --strict_contact --restart
)

stage() {
  local label=$1 amp=$2 heights=$3 blocks=$4 resume=$5; shift 5
  local dir="$ROOT_DIR/stage_${label}"
  mkdir -p "$dir"
  echo "=== stage6 $label ($ACTUATOR) amplitude=$amp heights=$heights ==="
  local rc=0
  $PY verge_task/scripts/train_until_converged.py \
    "${common[@]}" "$@" \
    --run_dir "$dir" --tracking_amplitude "$amp" \
    --height_schedule "$heights" \
    --min_blocks "$blocks" --max_blocks "$blocks" \
    --resume "$resume" 2>&1 | tee "$dir/stage.log" || rc=$?
  local accepted=0
  [ -f "$dir/block_provenance.jsonl" ] &&
    accepted=$(grep -c '"accepted": true' "$dir/block_provenance.jsonl" || true)
  echo "[STAGE $label] exit=$rc accepted_blocks=$accepted/$blocks"
  [ "$accepted" -lt "$blocks" ] && {
    echo "[STAGE $label] FAILED: see $dir/block_provenance.jsonl"; exit 4; }
  cp "$dir/model_best.pt" "$ROOT_DIR/model_best.pt"
  [ -f "$dir/convergence_best.json" ] &&
    cp "$dir/convergence_best.json" "$ROOT_DIR/convergence_best.json"
  echo "[STAGE $label] published -> $ROOT_DIR/model_best.pt"
}

stage S6a 1.4 "0.50,0.70"           "${BLOCKS_A:-3}" "$SEED_CKPT" \
  --reset_optimizer_first_block --reset_critic_head_first_block
stage S6b 1.8 "0.70,0.85,1.00"      "${BLOCKS_B:-3}" "$ROOT_DIR/stage_S6a/model_best.pt"
stage S6c 2.2 "0.50,0.70,0.85,1.00" "${BLOCKS_C:-3}" "$ROOT_DIR/stage_S6b/model_best.pt"

echo "[STAGE6 DONE] $EXPERIMENT -> $ROOT_DIR/model_best.pt"
