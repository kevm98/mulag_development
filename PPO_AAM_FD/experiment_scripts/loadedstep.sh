#!/usr/bin/env bash
# A cleaner adaptation test: a step between two LOADED payloads.
#
# The 1.0 -> 3.0x step is contaminated by design. Its first six seconds sit at
# nominal, which is precisely where this arm is known to lose lateral (-43.9% at
# constant 1.0x), so the stepping result mixes a regime where the arm helps with
# one where it hurts. Measured: cut +175.4%*, contact +0.058*, safety 8.6x
# better, lateral -13.8%*.
#
# Stepping 2.0 -> 5.0x keeps BOTH phases loaded, so the adaptation claim is
# tested without the nominal defect confounding it. MULAG_MASS_STEP_SCALE
# multiplies the base draw, so PAYLOAD_RANGE=2.0,2.0 with STEP_SCALE=2.5 gives
# 2.0 -> 5.0.
#
# This does not excuse the nominal defect, which is reported separately and is
# what A-RMA phase 3 is trying to fix. It isolates one question from the other.
set -u
export IMAGE=isaac-lab:2.3.2-rl052
cd /home/kevinmurera
R=/home/kevinmurera/projects/rrlab_mulag
FIT=/workspace/projects/rrlab_mulag/outputs/estimator/ctxonly_fit.npz
run() { ~/bin/isaaclab-run bash -lc "$1" 2>/dev/null || true; }
exec >>/home/kevinmurera/loadedstep.log 2>&1
echo "=== loadedstep armed $(date -u +%FT%TZ) ==="
while ps -eo args | grep -qE "[a]dapt2.sh"; do sleep 60; done
C=$(ls $R/logs/rsl_rl/ppo_id_aam_fd/ctxonly17/model_*.pt | sed 's/.*model_//;s/\.pt//' | sort -n | tail -1)
CKPT=logs/rsl_rl/ppo_id_aam_fd/ctxonly17/model_${C}.pt
mkdir -p $R/outputs/loadedstep
# ctxzero at constant 2.0x and 5.0x bracket the stepping plant, so the same
# verification gate applies: the frozen policy must land between them.
for SPEC in "const2.0:zero::export MULAG_PAYLOAD_RANGE=2.0,2.0" \
            "const5.0:zero::export MULAG_PAYLOAD_RANGE=5.0,5.0" \
            "step_ctxzero:zero::export MULAG_PAYLOAD_RANGE=2.0,2.0
export MULAG_MASS_STEP_TIME_S=6.0
export MULAG_MASS_STEP_JITTER_S=2.0
export MULAG_MASS_STEP_SCALE=2.5" \
            "step_ctxest:estimated:$FIT:export MULAG_PAYLOAD_RANGE=2.0,2.0
export MULAG_MASS_STEP_TIME_S=6.0
export MULAG_MASS_STEP_JITTER_S=2.0
export MULAG_MASS_STEP_SCALE=2.5"; do
  NAME=${SPEC%%:*}; REST=${SPEC#*:}; ABL=${REST%%:*}; REST=${REST#*:}; FITP=${REST%%:*}; PLANT=${REST#*:}
  EX=""; [ -n "$FITP" ] && EX="export MULAG_ESTIMATOR_PATH=$FITP
export MULAG_ESTIMATOR_BETA=0.02"
  while ps -eo args | grep -qE "[e]valuate_gates.py"; do sleep 45; done
  echo "=== $NAME ==="; date -u +%FT%TZ
  run "
cd /workspace/projects/rrlab_mulag
export MULAG_DIRECT_VALVE=1
export MULAG_RMA_CONTEXT=1
$PLANT
export MULAG_CONTEXT_ABLATE=$ABL
$EX
/isaac-sim/python.sh PPO_AAM_FD/scripts/evaluate_gates.py $CKPT \
  --num_envs 176 --steps 900 --meshes 2718 --seed 17 \
  --report outputs/loadedstep/${NAME}_m2718.json 2>&1 | grep -E 'wrote|rror' | tail -2
"
  echo "SCORED_${NAME}"
done
echo "LOADEDSTEP_COMPLETE $(date -u +%FT%TZ)"
