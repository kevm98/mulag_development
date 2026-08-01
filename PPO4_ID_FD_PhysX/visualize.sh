#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
CHECKPOINT="$ROOT/logs/rsl_rl/ppo4_sole_idfd/q5_hydraulic_gain040_stable_v6/convergence/model_best.pt"

if [[ $# -gt 0 && "$1" != --* ]]; then
  CHECKPOINT="$1"
  shift
fi

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "PPO4 checkpoint not found: $CHECKPOINT" >&2
  echo "usage: $0 [/path/to/model.pt] [additional evaluator arguments]" >&2
  exit 2
fi

cd "$ROOT"
echo "[PPO4] checkpoint=$CHECKPOINT"

exec "$PY" PPO3_SOLE_PhysX/scripts/evaluate.py "$CHECKPOINT" \
  --actuator idfd \
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
