#!/usr/bin/env bash
# Contact-safe V2: Stage B recovery, then Stage C strict.
#
#   bash PPO3_SOLE_PhysX/scripts/run_contact_safe_v2.sh rom
#   bash PPO3_SOLE_PhysX/scripts/run_contact_safe_v2.sh idfd
#
# THE FROZEN PLANT (Phase 1, reports/contact_safe_v2/plant_freeze/). Do not
# tune these during B or C unless a measured failure proves one is responsible.
#
#   cylinder_target_max_lead_m = 0.018
#
#     Chosen from the clamp instrumented WHERE IT IS APPLIED, plus a
#     zero-action control that separates plant windup from policy demand.
#     boom_sweep @ relief 1.0, held-out, 480 steps x 16 envs:
#
#       lead    bind% (policy)  bind% (zero action)  airborne  max gap
#       0.010        43.4              21.6            130     0.627 m
#       0.018        32.3              13.1             20     0.244 m
#       0.025        22.3               6.6             17     0.251 m
#
#     The plant's OWN command-vs-plant lead is ~4.5-5.0 mm p50 at every bound,
#     so 0.010 is only ~2x it and clips a normal operating signal -- which is
#     what produced the 130-step airborne excursion. 0.018 is ~3.5x and
#     recovers the airborne behaviour to the 0.025 level while keeping
#     saturation well below it. V1's claim that 0.010 "removed integrator
#     saturation" was withdrawn: it doubled it.
#
#     Peak force is NOT monotone across these (92 / 120 / 113 kN) and is a
#     single max over 7680 samples, so it is not used to choose here.
#
#   action_scale               = 0.0035   (V1 sweep: PPO3's best row on every axis)
#   terrain_arm_stiffness_scale= 0.35     (unchanged; non-monotone, do not chase)
#   sole_max_action_rate       = curriculum 0.25 (V1 sweep pointed opposite ways
#                                on the two actuators, so it is not set)
#   cylinder_target_leak       = 0.0
#   terrain_head_joint_stiffness_scale = 1.0
#
# STAGE B -- recover contact on the corrected plant, on the MEASURED worst
# region. Half the environments are boom_sweep at late path phases; the V1
# gate matrix found boom_sweep @ 1.0 produces 251 consecutive airborne steps
# and a 1.18 m gap WHILE PASSING all 17 historical gates. Two grace
# sub-stages, 4 steps then 2.
#
# STAGE C -- strict, teacher-free, grace 1, full 960-step episodes, balanced
# families and heights. The horizon is never shortened: the policy has to learn
# the corrected actuator, not rely on termination arriving before the hard part.
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1
PY=/home/rrlab/miniconda3/envs/env_isaaclab/bin/python

ACTUATOR=${1:-rom}
case "$ACTUATOR" in
  rom)  EXPERIMENT=ppo3_sole_rom_contact_safe_v1 ;;
  idfd) EXPERIMENT=ppo4_sole_idfd_contact_safe_v1 ;;
  *) echo "usage: $0 {rom|idfd}"; exit 2 ;;
esac

ROOT_DIR=logs/rsl_rl/$EXPERIMENT
SEED_CKPT=${SEED_CKPT:-$ROOT_DIR/stage_A/model_best.pt}
[ -f "$SEED_CKPT" ] || SEED_CKPT=$ROOT_DIR/model_seed_96.pt
[ -f "$SEED_CKPT" ] || { echo "no seed checkpoint under $ROOT_DIR"; exit 3; }

ACTION_SCALE=${ACTION_SCALE:-0.0035}
ARM_STIFFNESS=${ARM_STIFFNESS:-0.35}
TARGET_LEAD=${TARGET_LEAD:-0.018}
TARGET_LEAK=${TARGET_LEAK:-0.0}

NUM_ENVS=${NUM_ENVS:-256}
EVAL_ENVS=${EVAL_ENVS:-32}
BLOCK_ITERS=${BLOCK_ITERS:-40}
EVAL_STEPS=${EVAL_STEPS:-960}
LR=${LR:-1.0e-5}

common=(
  --python "$PY"
  --actuator "$ACTUATOR"
  --device cuda:0
  --num_envs "$NUM_ENVS"
  --eval_envs "$EVAL_ENVS"
  --block_iterations "$BLOCK_ITERS"
  --eval_steps "$EVAL_STEPS"
  --eval_timeout_s 3600
  --train_timeout_s 7200
  --stage 5
  --learning_rate "$LR"
  --head_joint_action
  --contact_observation true
  --contact_safety_reward true
  --model_error_history 3
  --action_scale "$ACTION_SCALE"
  --arm_stiffness_scale "$ARM_STIFFNESS"
  --cylinder_target_max_lead_m "$TARGET_LEAD"
  --cylinder_target_leak "$TARGET_LEAK"
  --strict_contact
  --contact_latch_release_steps 60
  --restart
)

stage() {
  local label=$1 grace=$2 heights=$3 blocks=$4 resume=$5; shift 5
  local dir="$ROOT_DIR/stage_${label}"
  mkdir -p "$dir"
  echo "=== V2 stage $label ($ACTUATOR) grace=$grace heights=$heights resume=$resume ==="
  local extra=("$@")
  # The supervisor exits NON-ZERO for "stopped at --max_blocks without
  # convergence", which for a fixed-block curriculum stage is the NORMAL
  # outcome, not an error. Under `set -euo pipefail` that aborted the whole
  # driver: PPO4 completed all three accepted blocks of stage B1 and then
  # never ran B2 or C, so its strict matrix was scored on a B1 checkpoint
  # while PPO3's was scored after B1+B2+C -- an unfair comparison produced by
  # a shell exit code.
  #
  # A stage is judged by what it produced, below, not by the exit code.
  local rc=0
  $PY PPO3_SOLE_PhysX/scripts/train_until_converged.py \
    "${common[@]}" "${extra[@]}" \
    --run_dir "$dir" \
    --contact_stage "$grace" \
    --height_schedule "$heights" \
    --min_blocks "$blocks" --max_blocks "$blocks" \
    --resume "$resume" 2>&1 | tee "$dir/stage.log" || rc=$?
  local accepted=0
  [ -f "$dir/block_provenance.jsonl" ] &&
    accepted=$(grep -c '"accepted": true' "$dir/block_provenance.jsonl" || true)
  echo "[STAGE $label] exit=$rc accepted_blocks=$accepted/$blocks"
  if [ "$accepted" -lt "$blocks" ]; then
    echo "[STAGE $label] FAILED: only $accepted of $blocks blocks were accepted."
    echo "                 See $dir/block_provenance.jsonl for the reasons."
    exit 4
  fi
  if [ -f "$dir/model_best.pt" ]; then
    cp "$dir/model_best.pt" "$ROOT_DIR/model_best.pt"
    [ -f "$dir/convergence_best.json" ] &&
      cp "$dir/convergence_best.json" "$ROOT_DIR/convergence_best.json"
    echo "[STAGE $label] published -> $ROOT_DIR/model_best.pt"
  fi
}

# Stage B1: grace 4, full relief, boom_sweep-heavy. A visible lift LATCHES for
# the rest of the episode rather than terminating -- termination was measured
# to collapse the mean episode length to 10 of 960 steps at 256 environments,
# because resets are all-or-nothing. The latch gives the same incentive (no
# task reward after a lift) at zero reset cost, releasing after 60 consecutive
# recovered contact steps so a policy that recovers is not punished forever.
stage B1 A "0.70,0.85,1.00" "${BLOCKS_B1:-3}" "$SEED_CKPT" \
  --boom_focus --contact_latch true \
  --reset_optimizer_first_block --reset_critic_head_first_block

# Stage B2: grace 2, same distribution.
stage B2 B "0.70,0.85,1.00" "${BLOCKS_B2:-3}" "$ROOT_DIR/stage_B1/model_best.pt" \
  --boom_focus --contact_latch true

# Stage C: grace 1, teacher-free, balanced families AND heights, full horizon.
stage C C "0.50,0.70,0.85,1.00" "${BLOCKS_C:-3}" "$ROOT_DIR/stage_B2/model_best.pt" \
  --boom_focus --contact_latch true

echo "[V2 DONE] $EXPERIMENT -> $ROOT_DIR/model_best.pt"
