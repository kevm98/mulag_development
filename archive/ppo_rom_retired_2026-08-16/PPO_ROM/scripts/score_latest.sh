#!/usr/bin/env bash
# Score the newest checkpoint of a run on the real acceptance contract:
# 3 held-out meshes, 8 gates, worst-mesh aggregation.
#
# This exists because the training log CANNOT tell you whether the policy is
# acceptable. TensorBoard metrics are means on TRAINING terrain; the contract in
# gates.py is a WORST-FAMILY, WORST-MESH number on held-out meshes. A run whose
# training lateral error looks fine can still fail the gate, and has before.
#
# Settings match the measured baselines in reports/frontier/ exactly
# (stage 5, 480 steps, 32 envs, meshes 1337/17/909) so the numbers are directly
# comparable with student_gates.json and teacher_gates.json. Changing any of
# them silently makes the comparison meaningless.
set -euo pipefail

ROOT=/home/rrlab/rrlab_mulag
PY=/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
RUN_DIR=${RUN_DIR:-$ROOT/logs/rsl_rl/ppo3_1_sole/warmup_prod}

cd "$ROOT"

CKPT=$(ls -1 "$RUN_DIR"/model_*.pt 2>/dev/null \
  | sed -E 's/.*model_([0-9]+)\.pt/\1 &/' | sort -n -k1 | tail -1 | cut -d' ' -f2-)
if [ -z "${CKPT:-}" ]; then
  echo "no checkpoints in $RUN_DIR" >&2
  exit 1
fi
NAME=$(basename "$CKPT" .pt)
OUT="$ROOT/PPO_ROM/reports/frontier/${NAME}_gates.json"

echo "[SCORE] $CKPT -> $OUT"
exec "$PY" PPO_ROM/scripts/evaluate_gates.py "$CKPT" \
    --meshes 1337,17,909 \
    --steps 480 \
    --num_envs 32 \
    --stage 5 \
    --report "$OUT" \
    --headless
