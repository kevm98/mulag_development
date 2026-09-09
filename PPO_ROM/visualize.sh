#!/usr/bin/env bash
set -euo pipefail

ROOT=$HOME/rrlab_mulag
PY=$HOME/miniconda3/envs/env_isaaclab/bin/python
DEFAULT_CHECKPOINT="$ROOT/logs/rsl_rl/ppo3_1_sole/speed_range_2_7_pathfix_v2_rom/model_promoted_2_7kmh.pt"

SUBJECT=$DEFAULT_CHECKPOINT
if [ "$#" -gt 0 ]; then
  case "$1" in
    --teacher|--null)
      SUBJECT=$1
      shift
      ;;
    -*) ;;
    *)
      SUBJECT=$1
      shift
      ;;
  esac
fi

if [ "$SUBJECT" != "--teacher" ] && [ "$SUBJECT" != "--null" ] && [ ! -f "$SUBJECT" ]; then
  echo "checkpoint not found: $SUBJECT" >&2
  exit 1
fi

cd "$ROOT"
exec "$PY" PPO_ROM/scripts/evaluate_gates.py "$SUBJECT" \
  --mesh "${MESH:-1337}" \
  --steps "${STEPS:-2808}" \
  --num-envs "${ENVS:-1}" \
  --stage "${STAGE:-6}" \
  --natural-visuals \
  "$@"
