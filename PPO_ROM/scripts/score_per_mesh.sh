#!/usr/bin/env bash
# Score a checkpoint (or --teacher / --null) on the 8-gate contract, running
# ONE MESH PER PROCESS and merging the results.
#
# WHY: evaluate_gates.py loops meshes inside a single process, calling
# env.close() then gym.make() again. In this Isaac Sim build that deadlocks --
# measured four times on 2026-08-06 (null eval, teacher eval, student eval,
# each hanging with exactly two PhysX env constructions logged, process alive at
# ~109% CPU, no output for 15-40 minutes). A single mesh completes in ~50 s, so
# the rollout itself is fine; it is the second env build in the same process
# that wedges.
#
# One process per mesh sidesteps it entirely and is also more robust: a hang on
# one mesh costs that mesh, not the whole evaluation. merge_gates.py then does
# the worst-mesh aggregation exactly once, which is the only place it is
# allowed to happen.
#
# Usage:
#   score_per_mesh.sh <name> <ckpt-path|--teacher|--null> [steps] [num_envs]
set -uo pipefail

ROOT=/home/rrlab/rrlab_mulag
PY=/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
OUT=$ROOT/PPO_ROM/reports/frontier
NAME=${1:?usage: score_per_mesh.sh <name> <ckpt|--teacher|--null> [steps] [envs]}
SUBJECT=${2:?}
STEPS=${3:-480}
ENVS=${4:-32}
MESHES=${MESHES:-"1337 17 909"}
STALL=${STALL:-240}
# TASK selects the pipeline: empty = PPO3_1 (ROM-A),
# RRLAB-Mulag-Verge-IDFD-v0 = PPO4_1 (ID/FD). Same scorer, same gates,
# same terrain -- which is what makes the two numbers comparable.
TASK=${TASK:-}
TASK_ARG=""
[ -n "$TASK" ] && TASK_ARG="--task $TASK"
# AMP overrides the path amplitude, collapsing the randomization band onto it.
# Used to measure the generalization envelope outside the trained 0.30-0.75.
AMP=${AMP:-}
[ -n "$AMP" ] && TASK_ARG="$TASK_ARG --tracking_amplitude $AMP"

mkdir -p "$OUT"
cd "$ROOT"

for MESH in $MESHES; do
  REPORT="$OUT/${NAME}_m${MESH}.json"
  LOG="$OUT/${NAME}_m${MESH}.log"
  rm -f "$REPORT"
  echo "[SCORE] mesh $MESH -> $REPORT"

  if [ "$SUBJECT" = "--teacher" ] || [ "$SUBJECT" = "--null" ]; then
    "$PY" PPO_ROM/scripts/evaluate_gates.py "$SUBJECT" \
        --meshes "$MESH" --steps "$STEPS" --num_envs "$ENVS" --stage 5 \
        $TASK_ARG --report "$REPORT" --headless > "$LOG" 2>&1 &
  else
    "$PY" PPO_ROM/scripts/evaluate_gates.py "$SUBJECT" \
        --meshes "$MESH" --steps "$STEPS" --num_envs "$ENVS" --stage 5 \
        $TASK_ARG --report "$REPORT" --headless > "$LOG" 2>&1 &
  fi
  PID=$!

  # Stall guard: the rollout prints progress every 10%, so a log that stops
  # growing while the process lives is the hang signature, not slow work.
  while kill -0 "$PID" 2>/dev/null; do
    sleep 15
    LAST=$(stat -c %Y "$LOG" 2>/dev/null || echo 0)
    AGE=$(( $(date +%s) - LAST ))
    if [ "$AGE" -gt "$STALL" ]; then
      echo "[SCORE] mesh $MESH STALLED (${AGE}s silent) -- killing"
      kill -9 "$PID" 2>/dev/null
      break
    fi
    [ -f "$REPORT" ] && break
  done
  wait "$PID" 2>/dev/null
  # Isaac Sim often refuses to exit after writing; the report is the contract.
  pkill -9 -f "evaluate_gates.py .*--meshes $MESH " 2>/dev/null
  sleep 4
  [ -f "$REPORT" ] && echo "[SCORE] mesh $MESH ok" || echo "[SCORE] mesh $MESH FAILED"
done

echo "[SCORE] merging"
"$PY" PPO_ROM/scripts/merge_gates.py "$OUT/${NAME}_m"*.json \
    --report "$OUT/${NAME}_gates.json"
