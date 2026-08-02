#!/usr/bin/env bash
# Phase 7: train the contact-safe policy on the MEASURED failure region.
#
#   bash PPO3_SOLE_PhysX/scripts/run_contact_safe_training.sh rom
#   bash PPO3_SOLE_PhysX/scripts/run_contact_safe_training.sh idfd
#
# PPO3 and PPO4 are trained SEPARATELY, each from its own migrated seed. The
# two actuators ARE the experiment; assuming one checkpoint serves both would
# destroy the only claim PPO4 exists to support.
#
# Three curriculum stages, each its own supervisor run directory so each has
# its own state.json and its own block reports, chained by copying the
# previous stage's model_best.pt forward:
#
#   A  grace 4 steps, relief 0.50/0.70           -- learn to stay planted
#   B  grace 2 steps, relief 0.70/0.85/1.00      -- at full relief
#   C  grace 1 step,  relief 0.50/0.70/0.85/1.00 -- balanced, near-strict
#
# Strict evaluation tolerates NO post-acquisition contact loss. Stage C stops
# one step short of that on purpose: a training signal that terminates on a
# single missed physics substep trains against the contact sensor rather than
# against the tool leaving the ground.
#
# The optimizer and critic head are reset in stage A ONLY. The seed is a
# widened migration whose new input columns have never had a gradient (Adam's
# second moment is indexed by shape), and the reward gained seven terms plus a
# finite terminal penalty, so the resumed value head predicts returns from a
# different objective. Stages B and C must NOT repeat that -- they would throw
# away the run's own learning at every curriculum step.
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
SEED_CKPT=$ROOT_DIR/model_seed_96.pt
if [ ! -f "$SEED_CKPT" ]; then
  echo "missing seed checkpoint $SEED_CKPT"
  echo "create it with: $PY PPO3_SOLE_PhysX/scripts/migrate_shared_policy.py <66-D ckpt> $SEED_CKPT --mode widen_observation --target_observations 96"
  exit 3
fi

# --- plant/command settings, chosen from the Phase 6 sweep -----------------
# Measured, 960 steps x 16 envs, held-out, s_curve, relief 1.0, shipped policy
# held fixed (reports/contact_safe_v1/sweep/). Peak force is the quantity to
# read: it is the stored command being released into the ground.
#
#   action_scale 0.0035  PPO3 best row on EVERY axis vs the 0.004 baseline
#                        (contact 0.9585 vs 0.9575, events 152 vs 188, lifts
#                        114 vs 146, peak force 125 vs 146 kN, ejection
#                        2.40 vs 2.65 m/s). PPO4 0.0035 and 0.0030 are within
#                        noise of each other and both cut peak force
#                        186 -> ~147 kN, so 0.0035 is used for BOTH and the
#                        two actuators keep an identical action contract.
#
#   lead 0.010           the only knob that reliably moves the mechanism:
#                        peak force 146 -> 107 kN (PPO3), 186 -> 127 kN (PPO4),
#                        and measured max lead 0.030 -> 0.0166, i.e. the
#                        integrator stops running saturated. It costs the FIXED
#                        policy contact (0.9575 -> 0.9205) because that policy
#                        was trained with 25 mm of authority and is co-adapted
#                        to it; recovering that is what the retraining is for.
#
#   stiffness 0.35       UNCHANGED. 0.25 gives PPO3 a lower peak force (96 kN)
#                        but costs contact, 0.20 is worse on both actuators,
#                        and the existing notes record this knob as
#                        non-monotonic. Not a knob to chase.
#
#   rate                 NOT overridden -- the curriculum's 0.25 stands. 0.20
#                        was catastrophic for PPO4 (386 kN peak force, 1.09 m
#                        max gap, 7.03 m/s ejection) while 0.15 was its best
#                        row and PPO3's second worst. A knob that points in
#                        opposite directions on the two actuators cannot be set
#                        from this sweep, and setting it differently per
#                        actuator would break the attribution PPO4 exists for.
ACTION_SCALE=${ACTION_SCALE:-0.0035}
MAX_ACTION_RATE=${MAX_ACTION_RATE:-}
ARM_STIFFNESS=${ARM_STIFFNESS:-0.35}
TARGET_LEAD=${TARGET_LEAD:-0.010}
TARGET_LEAK=${TARGET_LEAK:-0.0}

NUM_ENVS=${NUM_ENVS:-256}
EVAL_ENVS=${EVAL_ENVS:-32}
BLOCK_ITERS=${BLOCK_ITERS:-40}
EVAL_STEPS=${EVAL_STEPS:-960}
LR=${LR:-1.0e-5}
TRAIN_TIMEOUT=${TRAIN_TIMEOUT:-7200}
EVAL_TIMEOUT=${EVAL_TIMEOUT:-3600}

common=(
  --python "$PY"
  --actuator "$ACTUATOR"
  --device cuda:0
  --num_envs "$NUM_ENVS"
  --eval_envs "$EVAL_ENVS"
  --block_iterations "$BLOCK_ITERS"
  --eval_steps "$EVAL_STEPS"
  --eval_timeout_s "$EVAL_TIMEOUT"
  --train_timeout_s "$TRAIN_TIMEOUT"
  --stage 5
  --learning_rate "$LR"
  --head_joint_action
  --contact_focus
  --contact_observation true
  --contact_safety_reward true
  --model_error_history 3
  --action_scale "$ACTION_SCALE"
  --arm_stiffness_scale "$ARM_STIFFNESS"
  --cylinder_target_max_lead_m "$TARGET_LEAD"
  --cylinder_target_leak "$TARGET_LEAK"
  --strict_contact
  --restart
)
# Opt-in only: the sweep points in opposite directions on the two actuators.
if [ -n "$MAX_ACTION_RATE" ]; then
  common+=(--max_action_rate "$MAX_ACTION_RATE")
fi

# Termination is OFF by default in every stage, and that is a measured
# decision, not caution. Resets here are all-or-nothing --
# `_run_reset_preposition` raises when handed fewer than num_envs
# environments -- so a contact termination synchronizes the whole scene, and
# the pre-fix policy loses support ~430 times per 32-env/960-step rollout. At
# 256 training environments that is continuous reset work rather than
# training, which is exactly the trap `defer_safety_terminations` exists to
# escape. The seven-term contact reward carries the signal instead; it is
# constructed so floating is strictly worse than staying planted without the
# episode having to end.
#
# Set STAGE_C_TERMINATION=true to add termination in the final stage once the
# policy is already close, and read stage C's block reports to check the run
# is training rather than resetting.
STAGE_C_TERMINATION=${STAGE_C_TERMINATION:-false}

stage() {
  local label=$1 grace=$2 heights=$3 blocks=$4 resume=$5
  local dir="$ROOT_DIR/stage_${label}"
  mkdir -p "$dir"
  echo "=== contact-safe stage $label ($ACTUATOR) grace=$grace heights=$heights resume=$resume ==="
  local extra=()
  if [ "$label" = "A" ]; then
    extra+=(--reset_optimizer_first_block --reset_critic_head_first_block)
  fi
  if [ "$label" = "C" ] && [ "$STAGE_C_TERMINATION" = "true" ]; then
    extra+=(--contact_termination true)
  fi
  $PY PPO3_SOLE_PhysX/scripts/train_until_converged.py \
    "${common[@]}" "${extra[@]}" \
    --run_dir "$dir" \
    --contact_stage "$grace" \
    --height_schedule "$heights" \
    --min_blocks "$blocks" --max_blocks "$blocks" \
    --resume "$resume" 2>&1 | tee "$dir/stage.log"
  # Publish after EVERY stage, not only after C. A run stopped early -- which
  # is the normal case while the curriculum is being tuned -- otherwise leaves
  # no `model_best.pt` at the experiment root, and anything downstream that
  # expects one (the gate matrix driver, visualize.sh) silently finds nothing.
  # Measured: a stage-A-only run left the gate matrices unable to start.
  if [ -f "$dir/model_best.pt" ]; then
    cp "$dir/model_best.pt" "$ROOT_DIR/model_best.pt"
    [ -f "$dir/convergence_best.json" ] &&
      cp "$dir/convergence_best.json" "$ROOT_DIR/convergence_best.json"
    echo "[STAGE $label] published $dir/model_best.pt -> $ROOT_DIR/model_best.pt"
  fi
}

stage A A "0.50,0.70"           "${BLOCKS_A:-3}" "$SEED_CKPT"
stage B B "0.70,0.85,1.00"      "${BLOCKS_B:-3}" "$ROOT_DIR/stage_A/model_best.pt"
stage C C "0.50,0.70,0.85,1.00" "${BLOCKS_C:-3}" "$ROOT_DIR/stage_B/model_best.pt"

# stage() already published each stage; nothing left to copy here.

echo "[CONTACT-SAFE TRAINING DONE] $EXPERIMENT -> $ROOT_DIR/model_best.pt"
