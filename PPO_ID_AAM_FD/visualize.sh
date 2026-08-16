#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/rrlab/rrlab_mulag
PY=/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
SUBJECT=${1:---teacher}
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
MESH=${MESH:-1337}
ENVS=${ENVS:-1}
STEPS=${STEPS:-960}

cd "$ROOT"

if [ "$SUBJECT" != "--teacher" ] && [ "$SUBJECT" != "--null" ]; then
  if [ ! -f "$SUBJECT" ]; then
    # Training iterations are zero-based: the completed run ends at 599.
    REQUESTED_ALIAS="logs/rsl_rl/ppo_id_aam_fd/train/model_600.pt"
    REQUESTED_ALIAS_ABS="$ROOT/$REQUESTED_ALIAS"
    CANONICAL="$ROOT/logs/rsl_rl/ppo_id_aam_fd/train_final/model_599.pt"
    if { [ "$SUBJECT" = "$REQUESTED_ALIAS" ] || [ "$SUBJECT" = "$REQUESTED_ALIAS_ABS" ]; } && [ -f "$CANONICAL" ]; then
      echo "[VISUALIZE] requested $SUBJECT is an old alias; using $CANONICAL"
      SUBJECT="$CANONICAL"
    else
      echo "checkpoint not found: $SUBJECT" >&2
      echo "available completed checkpoint: $CANONICAL" >&2
      exit 1
    fi
  fi
fi
echo "[VISUALIZE] Stage 31 policy -> causal AAM -> ID adapter -> frozen FD -> PhysX"
echo "[VISUALIZE] subject=$SUBJECT mesh=$MESH envs=$ENVS steps=$STEPS"
echo "[VISUALIZE] close the Isaac viewer window to stop."
exec "$PY" PPO_ROM/scripts/evaluate_gates.py "$SUBJECT" \
  --meshes "$MESH" --steps "$STEPS" --num_envs "$ENVS" --stage 6 \
  --natural-visuals \
  --task RRLAB-Mulag-Verge-ID-AAM-FD-v0 "$@"
