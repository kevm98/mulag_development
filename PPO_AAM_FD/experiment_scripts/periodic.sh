#!/usr/bin/env bash
# The experiment a single step could not decide.
#
# The freeze ablation found an estimate latched BEFORE a one-shot step performs
# as well as one that follows it (cut +19.1% ns, lateral -3.2% ns), and the beta
# ladder agreed. But a single change in a 15 s episode is covered by ONE early
# identification, so that experiment cannot separate "tracks the payload" from
# "identified it once" no matter how it comes out. That was a design limit, not
# a finding about the arm.
#
# A payload that alternates every 3 s cannot be covered by one identification:
# the estimate must be revised repeatedly or it is wrong most of the time.
# MULAG_MASS_PERIOD_S=6.0 toggles base <-> base*scale each half period, with a
# per-environment phase offset so the fleet does not switch in lockstep and a
# timer cannot substitute for sensing.
#
# The decisive comparison is ctxest LIVE against ctxest FROZEN-at-4s on this
# plant. On a periodic plant, if tracking does any work at all, live must win.
set -u
export IMAGE=isaac-lab:2.3.2-rl052
cd /home/kevinmurera
R=/home/kevinmurera/projects/rrlab_mulag
FIT=/workspace/projects/rrlab_mulag/outputs/estimator/ctxonly_fit.npz
run() { ~/bin/isaaclab-run bash -lc "$1" 2>/dev/null || true; }
waitgpu() { while ps -eo args | grep -qE "[e]valuate_gates.py"; do sleep 45; done; }
exec >>/home/kevinmurera/periodic.log 2>&1
echo "=== periodic armed $(date -u +%FT%TZ) ==="
C=$(ls $R/logs/rsl_rl/ppo_id_aam_fd/ctxonly17/model_*.pt | sed 's/.*model_//;s/\.pt//' | sort -n | tail -1)
CKPT=logs/rsl_rl/ppo_id_aam_fd/ctxonly17/model_${C}.pt
mkdir -p $R/outputs/periodic

# The periodic path is NEW CODE. Smoke it before paying for full runs: if the
# mass never toggles, every arm below would be a constant-payload run wearing a
# different name.
echo "=== SMOKE periodic plant ==="; date -u +%FT%TZ
waitgpu
run "
cd /workspace/projects/rrlab_mulag
export MULAG_DIRECT_VALVE=1
export MULAG_RMA_CONTEXT=1
export MULAG_MASS_PERIOD_S=6.0
export MULAG_MASS_STEP_SCALE=3.0
export MULAG_CONTEXT_ABLATE=zero
/isaac-sim/python.sh PPO_AAM_FD/scripts/evaluate_gates.py $CKPT \
  --num_envs 16 --steps 60 --meshes 2718 --seed 17 \
  --report outputs/periodic/smoke.json 2>&1 | grep -E 'wrote|rror|Traceback' | tail -4
"
if [ ! -f "$R/outputs/periodic/smoke.json" ]; then
  echo "PERIODIC_SMOKE_FAILED -- new code is broken, nothing else run"; exit 1
fi
echo "PERIODIC_SMOKE_OK $(date -u +%FT%TZ)"

for SC in 3.0 5.0; do
  for SPEC in "ctxzero:zero:0.0" "ctxest_live:estimated:0.0" "ctxest_frozen4s:estimated:4.0"; do
    NAME=${SPEC%%:*}; REST=${SPEC#*:}; ABL=${REST%%:*}; FR=${REST#*:}
    EX=""; [ "$ABL" = estimated ] && EX="export MULAG_ESTIMATOR_PATH=$FIT
export MULAG_ESTIMATOR_BETA=0.02
export MULAG_ESTIMATOR_FREEZE_S=$FR"
    waitgpu
    echo "=== periodic $NAME scale $SC ==="; date -u +%FT%TZ
    run "
cd /workspace/projects/rrlab_mulag
export MULAG_DIRECT_VALVE=1
export MULAG_RMA_CONTEXT=1
export MULAG_MASS_PERIOD_S=6.0
export MULAG_MASS_STEP_SCALE=$SC
export MULAG_CONTEXT_ABLATE=$ABL
$EX
/isaac-sim/python.sh PPO_AAM_FD/scripts/evaluate_gates.py $CKPT \
  --num_envs 176 --steps 900 --meshes 2718 --seed 17 \
  --report outputs/periodic/${NAME}_s${SC}_m2718.json 2>&1 | grep -E 'wrote|rror' | tail -2
"
    echo "SCORED_periodic_${NAME}_s${SC}"
  done
done
echo "PERIODIC_COMPLETE $(date -u +%FT%TZ)"
echo "=== resuming finish.sh (held-out step, then seeds) ==="
cd /home/kevinmurera && ./finish.sh
