#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
PRODUCTION="$ROOT/logs/rsl_rl/ppo3_sole_terrain/shared_v2_rom/production/model_converged.pt"
MIGRATED="$ROOT/logs/rsl_rl/ppo3_sole_terrain/shared_v2_rom/model_migrated.pt"
CHECKPOINT="$PRODUCTION"
[[ -f "$CHECKPOINT" ]] || CHECKPOINT="$MIGRATED"

if [[ $# -gt 0 && "$1" != --* ]]; then
  CHECKPOINT="$1"
  shift
fi

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "PPO3 checkpoint not found: $CHECKPOINT" >&2
  echo "usage: $0 [/path/to/model.pt] [additional evaluator arguments]" >&2
  exit 2
fi

cd "$ROOT"
exec "$PY" PPO3_SOLE_PhysX/scripts/evaluate.py "$CHECKPOINT" \
  --actuator rom \
  --device cuda:0 \
  --num_envs 1 \
  --seed 17 \
  --steps 960 \
  --stage 5 \
  --eval_terrain \
  --head_joint_action \
  --head_action_to_c4_gain 0.4 \
  --drive_substep_interpolation true \
  --path_family s_curve \
  --plot_tracking \
  "$@"
