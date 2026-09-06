#!/usr/bin/env bash
# Re-record the trace figure with PER-ENVIRONMENT series.
#
# The first attempt averaged across 176 environments. The periodic plant
# phase-offsets each one, so at any instant about half are light and half heavy
# and the fleet mean is a flat 2.0: the square wave the figure exists to show is
# destroyed by the averaging. Individual environments keep it.
set -u
export IMAGE=isaac-lab:2.3.2-rl052
cd /home/kevinmurera
R=/home/kevinmurera/projects/rrlab_mulag
FIT=/workspace/projects/rrlab_mulag/outputs/estimator/ctxonly_fit.npz
run() { ~/bin/isaaclab-run bash -lc "$1" 2>/dev/null || true; }
waitgpu() { while ps -eo args | grep -qE "[e]valuate_gates.py"; do sleep 45; done; }
exec >>/home/kevinmurera/trace2.log 2>&1
echo "=== trace2 armed $(date -u +%FT%TZ) ==="
C=$(ls $R/logs/rsl_rl/ppo_id_aam_fd/ctxonly17/model_*.pt | sed 's/.*model_//;s/\.pt//' | sort -n | tail -1)
CKPT=logs/rsl_rl/ppo_id_aam_fd/ctxonly17/model_${C}.pt
mkdir -p $R/outputs/trace2
for SPEC in "ctxest_live:estimated:0.0" "ctxest_frozen4s:estimated:4.0" "ctxzero:zero:0.0"; do
  NAME=${SPEC%%:*}; REST=${SPEC#*:}; ABL=${REST%%:*}; FR=${REST#*:}
  EX=""; [ "$ABL" = estimated ] && EX="export MULAG_ESTIMATOR_PATH=$FIT
export MULAG_ESTIMATOR_BETA=0.02
export MULAG_ESTIMATOR_FREEZE_S=$FR"
  waitgpu
  echo "=== trace2 $NAME ==="; date -u +%FT%TZ
  run "
cd /workspace/projects/rrlab_mulag
export MULAG_DIRECT_VALVE=1
export MULAG_RMA_CONTEXT=1
export MULAG_MASS_PERIOD_S=6.0
export MULAG_MASS_STEP_SCALE=3.0
export MULAG_CONTEXT_ABLATE=$ABL
$EX
/isaac-sim/python.sh PPO_AAM_FD/scripts/evaluate_gates.py $CKPT \
  --num_envs 176 --steps 900 --meshes 2718 --seed 17 \
  --adapt_trace outputs/trace2/${NAME}.json \
  --report outputs/trace2/${NAME}_report.json 2>&1 | grep -E 'wrote|rror' | tail -2
"
  echo "SCORED_trace2_${NAME}"
done
echo "TRACE2_COMPLETE $(date -u +%FT%TZ)"
