#!/usr/bin/env bash
# Visualize the standalone PPO_ID_FD task in Isaac Sim.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$HOME/miniconda3/envs/env_isaaclab/bin/python"
DEFAULT_CHECKPOINT="$ROOT/logs/rsl_rl/ppo4_1_sole/speed_range_2_7_pathfix_v2_idfd/model_promoted_2_7kmh.pt"
SUBJECT="$DEFAULT_CHECKPOINT"
MESH=${MESH:-1337}
ENVS=${ENVS:-1}
STEPS=${STEPS:-2808}

if [ "$#" -gt 0 ]; then
  case "$1" in
    --teacher|--null)
      SUBJECT=$1
      shift
      ;;
    -*)
      ;;
    *)
      SUBJECT=$1
      shift
      ;;
  esac
fi

cd "$ROOT"

if [ "$SUBJECT" != "--teacher" ] && [ "$SUBJECT" != "--null" ]; then
  if [ ! -f "$SUBJECT" ]; then
    echo "checkpoint not found: $SUBJECT" >&2
    echo "run the pipeline first or pass a valid checkpoint path" >&2
    exit 1
  fi
fi
echo "[VISUALIZE] PPO_ID_FD policy -> ID -> FD -> PhysX"
echo "[VISUALIZE] subject=$SUBJECT mesh=$MESH envs=$ENVS steps=$STEPS"
exec "$PY" PPO_ROM/scripts/evaluate_gates.py "$SUBJECT" \
  --meshes "$MESH" --steps "$STEPS" --num_envs "$ENVS" --stage 6 \
  --task RRLAB-Mulag-Verge-IDFD-v0 --natural-visuals "$@"
