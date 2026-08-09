#!/usr/bin/env bash
# Phase 6: how much stored command error can be released into the ground?
#
# One knob at a time from the shipped baseline, not a cross product: the
# repository's own notes record `terrain_arm_stiffness_scale` as NON-monotonic
# (0.5 leaves a 19.7 mm hop, 0.2/0.1 lose contact, 0.05 is worse than 0.2), so
# a coordinate sweep is the only reading these knobs support. A full grid would
# also be 81 runs on a 6 GB laptop card.
#
# 960 steps, NOT 480. The measured contact-loss events concentrate late in the
# path, so a half-horizon sweep would compare configurations on the easy part.
#
# This sweeps the PLANT, with the shipped policy held fixed. It answers "how
# much impact energy does this configuration admit", not "can a policy trained
# for it do better" -- that is Phase 7.
set -u

cd "$(dirname "$0")/../.." || exit 1
PY=/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
OUT=${1:-verge_task/reports/contact_safe_v1/sweep}
mkdir -p "$OUT"

PPO3_CKPT=logs/rsl_rl/ppo3_sole_terrain/shared_v2_rom/model_migrated.pt
PPO4_CKPT=logs/rsl_rl/ppo4_sole_idfd/shared_v2_idfd/model_validated.pt

STEPS=${STEPS:-960}
ENVS=${ENVS:-16}
SEED=${SEED:-17}

run() {
  local tag=$1 act=$2 ckpt=$3; shift 3
  local report="$OUT/${tag}.json"
  if [ -f "$report" ]; then echo "[SKIP] $report"; return; fi
  echo "[RUN] $tag  $*"
  $PY verge_task/scripts/probe_contact_loss.py \
    --checkpoint "$ckpt" --actuator "$act" \
    --headless --device cuda:0 \
    --num_envs "$ENVS" --seed "$SEED" --steps "$STEPS" --stage 5 \
    --eval_terrain --path_family s_curve --terrain_height_scale 1.0 --max_event_windows 0 \
    "$@" --report "$report" > "$OUT/${tag}.log" 2>&1
  echo "[DONE] exit=$? -> $report"
}

for pair in "rom:$PPO3_CKPT:ppo3" "idfd:$PPO4_CKPT:ppo4"; do
  act=${pair%%:*}; rest=${pair#*:}; ckpt=${rest%%:*}; tag=${rest##*:}

  # Baseline at this width/seed, so every sweep row has a like-for-like control.
  run "${tag}_baseline"            "$act" "$ckpt"

  # action_scale: metres of cylinder command per unit action.
  run "${tag}_action_scale_0.0035" "$act" "$ckpt" --action_scale 0.0035
  run "${tag}_action_scale_0.0030" "$act" "$ckpt" --action_scale 0.0030

  # sole_max_action_rate: per-step slew limit on the applied action.
  run "${tag}_rate_0.20"           "$act" "$ckpt" --max_action_rate 0.20
  run "${tag}_rate_0.15"           "$act" "$ckpt" --max_action_rate 0.15

  # terrain_arm_stiffness_scale: compliance in the vertical load path.
  # Head joint stays at 1.0 -- softening it was measured to cost head
  # alignment while buying none of the bounce reduction.
  run "${tag}_stiffness_0.25"      "$act" "$ckpt" --arm_stiffness_scale 0.25
  run "${tag}_stiffness_0.20"      "$act" "$ckpt" --arm_stiffness_scale 0.20

  # cylinder_target_max_lead_m: the anti-windup bound on how far the commanded
  # PhysX target may lead the plant. Measured SATURATED at the inherited 25 mm
  # on BOTH actuators, and at ~5e6 N/m of cylinder stiffness 25 mm of lead is
  # ~125 kN of commanded force -- which is the order of the 119/170 kN peaks.
  run "${tag}_lead_0.015"          "$act" "$ckpt" --cylinder_target_max_lead_m 0.015
  run "${tag}_lead_0.010"          "$act" "$ckpt" --cylinder_target_max_lead_m 0.010
done
echo "[ALL DONE]"
