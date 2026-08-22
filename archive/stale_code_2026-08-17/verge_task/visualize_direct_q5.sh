#!/usr/bin/env bash
# Watch the direct-q5 PPO3 plant in the Isaac viewer.
#
# The older visualize.sh targets the 96-wide contact-safe contract at stage 5
# with head_action_to_c4_gain 0.4. That is wrong for a direct-q5 checkpoint:
# these are 106 wide, 5 actions, stage 6, and q5 is a position command that
# must NOT be mixed back into c4. Passing the old flags makes the evaluator
# refuse the width outright, so this is a separate entry point rather than
# another branch in that script.
#
# Omitting --headless is what opens the viewer; evaluate.py then turns on the
# road scenery and grass automatically.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/home/rrlab/miniconda3/envs/env_isaaclab/bin/python

# The 16/17 checkpoint: best standard-gate result to date. Fails only
# worst_family_lateral.
CHECKPOINT="$ROOT/logs/rsl_rl/ppo3_sole_rom_direct_q5_v1/model_best.pt"
NUM_ENVS=6

if [[ $# -gt 0 && "$1" != --* ]]; then
  CHECKPOINT="$1"
  shift
fi

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "PPO3 checkpoint not found: $CHECKPOINT" >&2
  echo "usage: $0 [/path/to/model.pt] [extra evaluator arguments]" >&2
  exit 2
fi

cd "$ROOT"
exec "$PY" verge_task/scripts/evaluate.py "$CHECKPOINT" \
  --actuator rom \
  --device cuda:0 \
  --num_envs "$NUM_ENVS" \
  --seed 17 \
  --steps 960 \
  --stage 6 \
  --eval_terrain \
  --path_family mixed \
  --head_joint_action \
  --contact_observation true \
  --contact_safety_reward true \
  --contact_latch true \
  --contact_latch_release_steps 60 \
  --model_error_history 3 \
  --action_scale 0.0035 \
  --arm_stiffness_scale 0.35 \
  --cylinder_target_max_lead_m 0.018 \
  --cylinder_target_leak 0.0 \
  --tracking_amplitude 2.2 \
  --head_frame_observation true \
  --head_tilt_arm_frame true \
  "$@"
