#!/usr/bin/env bash
# Stage 2 for run 1: the payload estimated ONLINE, never read.
#
# Skipped by ctxonly.sh because the estimator directory alias was an ABSOLUTE
# symlink, which is broken inside the container: the host path
# /home/kevinmurera/projects/rrlab_mulag does not exist there, the mount is at
# /workspace/projects/rrlab_mulag. Remade relative and the fit then ran.
#
# The fit: ridge on the 66 task columns, trained on the mixed-payload run and
# held out on the fixed ones. R2 0.726, and per payload it estimates 1.41x /
# 3.11x / 4.10x against a true 1 / 3 / 5. Accurate in the middle, shrunk at the
# extremes, so this arm should recover part of the oracle's gain and not all of
# it. Written down before the arm is scored.
set -u
export IMAGE=isaac-lab:2.3.2-rl052
MESH=2718; ENVS_N=176; STEPS=900
PAYLOADS=${PAYLOADS:-"5.0 3.0 1.0"}
cd /home/kevinmurera
R=/home/kevinmurera/projects/rrlab_mulag
run() { ~/bin/isaaclab-run bash -lc "$1" 2>/dev/null || true; }
exec >>/home/kevinmurera/stage2_run.log 2>&1
echo "=== stage2 armed $(date -u +%FT%TZ) ==="
C=$(ls $R/logs/rsl_rl/ppo_id_aam_fd/ctxonly17/model_*.pt | sed 's/.*model_//;s/\.pt//' | sort -n | tail -1)
CKPT=logs/rsl_rl/ppo_id_aam_fd/ctxonly17/model_${C}.pt
[ -f "$R/outputs/estimator/ctxonly_fit.npz" ] || { echo "STAGE2_ABORT no fit"; exit 1; }

for M in $PAYLOADS; do
  # One evaluator at a time: each peaks near 17 GB and two OOM this 31 GB box.
  while ps -eo args | grep -qE "[e]valuate_gates.py"; do sleep 45; done
  echo "=== SCORE ctxest p${M} ==="; date -u +%FT%TZ
  run "
cd /workspace/projects/rrlab_mulag
export MULAG_DIRECT_VALVE=1
export MULAG_RMA_CONTEXT=1
export MULAG_PAYLOAD_RANGE=$M,$M
export MULAG_CONTEXT_ABLATE=estimated
export MULAG_ESTIMATOR_PATH=/workspace/projects/rrlab_mulag/outputs/estimator/ctxonly_fit.npz
export MULAG_ESTIMATOR_BETA=0.02
/isaac-sim/python.sh PPO_AAM_FD/scripts/evaluate_gates.py $CKPT \
  --num_envs $ENVS_N --steps $STEPS --meshes $MESH --seed 17 \
  --report outputs/ctxonly/ctxest_p${M}_m${MESH}.json 2>&1 | grep -E 'wrote|rror|estimator loaded' | tail -3
"
  echo "SCORED_ctxest_p${M}"
done
echo "STAGE2_COMPLETE $(date -u +%FT%TZ)"
