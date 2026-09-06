#!/usr/bin/env bash
# The figure the adaptive-control literature actually asks for.
#
# Everything so far is episode aggregates. The canonical demonstration is two
# time series: the estimated parameter against the true parameter, and the
# tracking error through the transient after a change. On a PERIODIC payload
# (~5 changes per episode) both are readable at a glance and neither can be
# faked by a single early identification.
#
# Three traces, all on the same plant:
#   ctxzero      the frozen policy: what the errors do with no adaptation
#   ctxest_live  the online estimate: does it FOLLOW the square wave, and do the
#                errors recover after each change
#   ctxest_frozen4s  the estimate latched at 4 s: the same policy that cannot
#                follow, so the two traces differ only in whether the estimate
#                is allowed to update
set -u
export IMAGE=isaac-lab:2.3.2-rl052
cd /home/kevinmurera
R=/home/kevinmurera/projects/rrlab_mulag
FIT=/workspace/projects/rrlab_mulag/outputs/estimator/ctxonly_fit.npz
run() { ~/bin/isaaclab-run bash -lc "$1" 2>/dev/null || true; }
waitgpu() { while ps -eo args | grep -qE "[e]valuate_gates.py"; do sleep 45; done; }
exec >>/home/kevinmurera/trace.log 2>&1
echo "=== trace armed $(date -u +%FT%TZ) ==="
# sequencing is handled by queue.sh
C=$(ls $R/logs/rsl_rl/ppo_id_aam_fd/ctxonly17/model_*.pt | sed 's/.*model_//;s/\.pt//' | sort -n | tail -1)
CKPT=logs/rsl_rl/ppo_id_aam_fd/ctxonly17/model_${C}.pt
mkdir -p $R/outputs/trace
for SPEC in "ctxzero:zero:0.0" "ctxest_live:estimated:0.0" "ctxest_frozen4s:estimated:4.0"; do
  NAME=${SPEC%%:*}; REST=${SPEC#*:}; ABL=${REST%%:*}; FR=${REST#*:}
  EX=""; [ "$ABL" = estimated ] && EX="export MULAG_ESTIMATOR_PATH=$FIT
export MULAG_ESTIMATOR_BETA=0.02
export MULAG_ESTIMATOR_FREEZE_S=$FR"
  waitgpu
  echo "=== trace $NAME ==="; date -u +%FT%TZ
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
  --adapt_trace outputs/trace/${NAME}.json \
  --report outputs/trace/${NAME}_report.json 2>&1 | grep -E 'wrote|rror' | tail -3
"
  echo "SCORED_trace_${NAME}"
done
echo "TRACE_COMPLETE $(date -u +%FT%TZ)"
