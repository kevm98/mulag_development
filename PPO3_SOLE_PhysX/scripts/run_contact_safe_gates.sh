#!/usr/bin/env bash
# Final strict-gate matrix for a contact-safe checkpoint.
#
#   bash PPO3_SOLE_PhysX/scripts/run_contact_safe_gates.sh rom  <checkpoint>
#   bash PPO3_SOLE_PhysX/scripts/run_contact_safe_gates.sh idfd <checkpoint>
#
# Every run is held-out, stage 5, 960 steps, teacher off, contact demo off, and
# the evaluator never terminates on contact loss -- a gate that truncates its
# own evidence cannot size an excursion.
#
# The matrix is deliberately wider than the single s_curve/1.0 cell the failure
# was found in: an aggregate over easy families can hide one failing path, and
# the strict gates are absolute counts, so a family that lifts once fails.
set -u

cd "$(dirname "$0")/../.." || exit 1
PY=/home/rrlab/miniconda3/envs/env_isaaclab/bin/python

ACTUATOR=${1:-rom}
CKPT=${2:-}
if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
  echo "usage: $0 {rom|idfd} <checkpoint>"; exit 2
fi
case "$ACTUATOR" in
  rom)  TAG=ppo3 ;;
  idfd) TAG=ppo4 ;;
  *) echo "usage: $0 {rom|idfd} <checkpoint>"; exit 2 ;;
esac

OUT=${OUT:-PPO3_SOLE_PhysX/reports/contact_safe_v1/gates}
mkdir -p "$OUT"
ENVS=${ENVS:-32}
STEPS=${STEPS:-960}

run() {
  local label=$1; shift
  local report="$OUT/${TAG}_${label}.json"
  if [ -f "$report" ]; then echo "[SKIP] $report"; return; fi
  echo "[RUN] $TAG $label"
  $PY PPO3_SOLE_PhysX/scripts/evaluate.py "$CKPT" \
    --actuator "$ACTUATOR" --headless --device cuda:0 \
    --num_envs "$ENVS" --seed 17 --steps "$STEPS" --stage 5 --eval_terrain \
    --contact_observation true --contact_safety_reward true \
    --model_error_history 3 --action_scale 0.0035 \
    --cylinder_target_max_lead_m 0.010 \
    --strict_contact "$@" --report "$report" > "$OUT/${TAG}_${label}.log" 2>&1
  echo "[DONE] exit=$? -> $report"
}

# --- the failure cell, and the two control families -------------------------
run s_curve_h1.0        --path_family s_curve          --terrain_height_scale 1.0
run boom_sweep_h1.0     --path_family boom_sweep       --terrain_height_scale 1.0
run lane_change_h1.0    --path_family lane_change_hold --terrain_height_scale 1.0

# --- mixed families, full relief and the balanced evaluation distribution ----
run mixed_h1.0          --path_family mixed            --terrain_height_scale 1.0
run mixed_balanced      --path_family mixed

# --- multi-mesh. --seed cannot do this; the held-out mesh is pinned. --------
run mixed_mesh909       --path_family mixed --eval_terrain_seed 909
run mixed_mesh4242      --path_family mixed --eval_terrain_seed 4242
run s_curve_mesh909     --path_family s_curve --terrain_height_scale 1.0 --eval_terrain_seed 909

echo "[GATES DONE] $TAG -> $OUT"
