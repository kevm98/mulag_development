#!/usr/bin/env bash
# The falsification for the adaptation claim itself.
#
# On the 1.0 -> 3.0x stepping plant the online arm improves cut, contact and
# safety. But beta = 0.005, whose 3.3 s time constant should badly lag a step
# landing at 4-8 s of a 15 s episode, performs the SAME as beta = 0.02
# (+172.7% vs +175.4% cut). If the benefit came from rapidly tracking the
# change, a four-times slower estimator would have shown it. That is evidence
# the benefit may be conditioning averaged over the episode rather than
# tracking.
#
# This settles it. MULAG_ESTIMATOR_FREEZE_S=4.0 latches the estimate at 4 s,
# just BEFORE the step window opens at 4-8 s, so the frozen arm carries a
# pre-step estimate for the rest of the episode and cannot follow the change.
# Everything else is identical.
#
#   live beats frozen   -> the arm genuinely tracks the payload change
#   live equals frozen  -> the benefit is conditioning, not tracking, and the
#                          adaptation claim must be worded accordingly
set -u
export IMAGE=isaac-lab:2.3.2-rl052
cd /home/kevinmurera
R=/home/kevinmurera/projects/rrlab_mulag
FIT=/workspace/projects/rrlab_mulag/outputs/estimator/ctxonly_fit.npz
run() { ~/bin/isaaclab-run bash -lc "$1" 2>/dev/null || true; }
exec >>/home/kevinmurera/freeze.log 2>&1
echo "=== freeze ablation armed $(date -u +%FT%TZ) ==="
while ps -eo args | grep -qE "[a]dapt2.sh|[l]oadedstep.sh"; do sleep 60; done
C=$(ls $R/logs/rsl_rl/ppo_id_aam_fd/ctxonly17/model_*.pt | sed 's/.*model_//;s/\.pt//' | sort -n | tail -1)
CKPT=logs/rsl_rl/ppo_id_aam_fd/ctxonly17/model_${C}.pt
mkdir -p $R/outputs/freeze
for FR in 4.0 0.0; do
  NAME=$([ "$FR" = "4.0" ] && echo frozen_at4s || echo live)
  while ps -eo args | grep -qE "[e]valuate_gates.py"; do sleep 45; done
  echo "=== $NAME (freeze_s=$FR) ==="; date -u +%FT%TZ
  run "
cd /workspace/projects/rrlab_mulag
export MULAG_DIRECT_VALVE=1
export MULAG_RMA_CONTEXT=1
export MULAG_MASS_STEP_TIME_S=6.0
export MULAG_MASS_STEP_JITTER_S=2.0
export MULAG_MASS_STEP_SCALE=3.0
export MULAG_CONTEXT_ABLATE=estimated
export MULAG_ESTIMATOR_PATH=$FIT
export MULAG_ESTIMATOR_BETA=0.02
export MULAG_ESTIMATOR_FREEZE_S=$FR
/isaac-sim/python.sh PPO_AAM_FD/scripts/evaluate_gates.py $CKPT \
  --num_envs 176 --steps 900 --meshes 2718 --seed 17 \
  --report outputs/freeze/${NAME}_m2718.json 2>&1 | grep -E 'wrote|rror' | tail -2
"
  echo "SCORED_${NAME}"
done
echo "FREEZE_COMPLETE $(date -u +%FT%TZ)"
