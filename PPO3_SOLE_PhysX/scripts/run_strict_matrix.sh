#!/usr/bin/env bash
# Phase 5 strict promotion matrix.
#
#   bash PPO3_SOLE_PhysX/scripts/run_strict_matrix.sh rom  <ckpt> [outdir]
#
# 4 families x 2 height conditions x 4 meshes = 32 cells per actuator, every
# one held-out, stage 5, the FULL 960-step horizon, teacher off, contact demo
# off, and the evaluator never terminating on contact loss.
#
# Meshes, not --seed. A held-out rollout is deterministic -- pinned mesh, fixed
# start phase, actor at its mean -- so four --seed values return four identical
# reports. Measured; see reports/contact_safe_v1/prefix_identical_seed_artifact/.
# --eval_terrain_seed is the only thing that varies a held-out evaluation, and
# 1337 reproduces the pinned gate mesh exactly, which is the control.
set -u
cd "$(dirname "$0")/../.." || exit 1
PY=/home/rrlab/miniconda3/envs/env_isaaclab/bin/python

ACTUATOR=${1:-rom}; CKPT=${2:-}; OUT=${3:-PPO3_SOLE_PhysX/reports/contact_safe_v2/strict}
[ -f "$CKPT" ] || { echo "usage: $0 {rom|idfd} <checkpoint> [outdir]"; exit 2; }
case "$ACTUATOR" in rom) TAG=ppo3 ;; idfd) TAG=ppo4 ;; *) exit 2 ;; esac
mkdir -p "$OUT"
ENVS=${ENVS:-32}; STEPS=${STEPS:-960}

for family in boom_sweep s_curve lane_change_hold mixed; do
  for height in 1.0 balanced; do
    for mesh in 1337 909 4242 17; do
      hflag=""; [ "$height" = "1.0" ] && hflag="--terrain_height_scale 1.0"
      label="${family}_${height}_m${mesh}"
      r="$OUT/${TAG}_${label}.json"
      [ -f "$r" ] && { echo "[SKIP] $label"; continue; }
      echo "[RUN] $TAG $label"
      $PY PPO3_SOLE_PhysX/scripts/evaluate.py "$CKPT" \
        --actuator "$ACTUATOR" --headless --device cuda:0 \
        --num_envs "$ENVS" --seed 17 --steps "$STEPS" --stage 5 --eval_terrain \
        --eval_terrain_seed "$mesh" --path_family "$family" $hflag \
        --contact_observation true --contact_safety_reward true \
        --contact_latch true --contact_latch_release_steps 60 \
        --model_error_history 3 --action_scale 0.0035 \
        --cylinder_target_max_lead_m 0.018 --arm_stiffness_scale 0.35 \
        --strict_contact --report "$r" > "$OUT/${TAG}_${label}.log" 2>&1
      echo "[DONE] exit=$?"
    done
  done
done
echo "[STRICT MATRIX DONE] $TAG"
