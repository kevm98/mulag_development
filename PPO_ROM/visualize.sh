#!/usr/bin/env bash
# Watch a PPO3_1 policy drive the verge task in the Isaac Sim viewer.
#
# Defaults to the shipped student: student_distilled_v2.pt, which passes all
# eight gates on three held-out meshes with no thin margins.
#
# WHY THIS REUSES evaluate_gates.py RATHER THAN THE "PLAY" TASK
# ------------------------------------------------------------
# `RRLAB-Mulag-Verge-ROM-PLAY-v0` builds TerrainC4AuthorityEnvCfg,
# which does NOT carry the feasibility-derived terrain -- that lives in
# Ppo31SoleEnvCfg (amplitude 0.7, sweep gain 0.5, heights 0.50-1.00, plus the
# randomization band). Playing the PLAY task would render the OLD amplitude-2.2
# terrain, on which 9 of 11 path families demand more lateral cutter speed than
# the cylinders can deliver -- i.e. it would show the policy failing at a task
# it was never trained for, which is exactly the mismatch that cost a day.
#
# evaluate_gates.py builds the same TERRAIN_TASK_ID config the scoring runs
# used, so what you see is what was measured. Dropping --headless renders it.
#
# Usage:
#   PPO_ROM/visualize.sh                      # shipped student, mesh 1337
#   PPO_ROM/visualize.sh <ckpt.pt>            # a different checkpoint
#   MESH=17 ENVS=4 STEPS=960 PPO_ROM/visualize.sh
#   PPO_ROM/visualize.sh --teacher            # the privileged DLS teacher
set -euo pipefail

ROOT=/home/rrlab/rrlab_mulag
PY=/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
SUBJECT=${1:-$ROOT/logs/rsl_rl/ppo3_1_sole/student_distilled_v2.pt}
MESH=${MESH:-1337}
ENVS=${ENVS:-1}
STEPS=${STEPS:-960}
# TASK selects the pipeline; empty = PPO3_1 (ROM-A).
TASK=${TASK:-}
TASK_ARG=""
[ -n "$TASK" ] && TASK_ARG="--task $TASK"

cd "$ROOT"

if [ "$SUBJECT" != "--teacher" ] && [ "$SUBJECT" != "--null" ] && [ ! -f "$SUBJECT" ]; then
  echo "checkpoint not found: $SUBJECT" >&2
  exit 1
fi

echo "[VISUALIZE] subject=$SUBJECT mesh=$MESH envs=$ENVS steps=$STEPS"
echo "[VISUALIZE] terrain: amplitude 0.7, sweep gain 0.5, heights 0.50-1.00"
echo "[VISUALIZE] close the viewer window to stop."

# No --headless: this is the whole point. Everything else matches the scoring
# runs so the render and the measured numbers describe the same thing.
exec "$PY" PPO_ROM/scripts/evaluate_gates.py "$SUBJECT" \
    --meshes "$MESH" \
    --steps "$STEPS" \
    --num_envs "$ENVS" \
    --stage 5 $TASK_ARG
