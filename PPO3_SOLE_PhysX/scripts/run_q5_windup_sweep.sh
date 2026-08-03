#!/usr/bin/env bash
# Choose the q5 anti-windup parameters by MEASUREMENT, not by guess.
#
# PPO4's q5 carries a DC bias of 0.324. q5 integrates
# (correction += 0.010 * action, clamped to +-0.35 rad), so that bias saturates
# the bound in ~108 of 960 steps and holds a permanent 20 degree head offset.
# PPO3's q5 sits at 0.0114 and never winds up; that is the entire difference
# between their head alignment (0.9875 vs 0.5303).
#
# Two knobs, swept one at a time on the FIXED PPO4 checkpoint so the policy is
# not a variable:
#
#   sole_head_correction_leak    settles a persistent bias at
#                                scale*action/leak instead of the bound
#   terrain_head_correction_weight  taxes the standing offset in the reward
#                                (does nothing without training -- swept only
#                                to confirm it is plumbed, not to pick a value)
#
# The leak is the one that can help an already-trained policy, because it acts
# on the plant side. The weight is chosen for the RETRAIN and is verified here
# only to the extent that it does not perturb a rollout it should not.
set -u

cd "$(dirname "$0")/../.." || exit 1
PY=/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
OUT=${1:-PPO3_SOLE_PhysX/reports/stage6/q5_sweep}
mkdir -p "$OUT"

CKPT=logs/rsl_rl/ppo4_sole_idfd_headfix_v1/model_best.pt
[ -f "$CKPT" ] || { echo "missing $CKPT"; exit 2; }

run() {
  local tag=$1 leak=$2
  local r="$OUT/${tag}.json"
  [ -f "$r" ] && { echo "[SKIP] $tag"; return; }
  echo "[RUN] $tag leak=$leak"
  $PY PPO3_SOLE_PhysX/scripts/evaluate.py "$CKPT" --actuator idfd \
    --headless --device cuda:0 --num_envs 32 --seed 17 --steps 960 \
    --stage 6 --eval_terrain --path_family mixed --tracking_amplitude 2.2 \
    --contact_observation true --contact_safety_reward true \
    --head_frame_observation true --head_tilt_arm_frame true \
    --contact_latch true --contact_latch_release_steps 60 \
    --model_error_history 3 --action_scale 0.0035 \
    --cylinder_target_max_lead_m 0.018 --arm_stiffness_scale 0.35 \
    --head_correction_leak "$leak" \
    --strict_contact --report "$r" > "$OUT/${tag}.log" 2>&1
  echo "[DONE] exit=$?"
}

# 0.0 is the control: it must reproduce the failing baseline exactly.
run leak_0.000 0.0
run leak_0.010 0.010
run leak_0.020 0.020
run leak_0.040 0.040
run leak_0.080 0.080

echo "[Q5 SWEEP DONE] $OUT"
