#!/usr/bin/env bash
# Re-score the online arm with the degree-2 estimator.
#
# The linear estimator reads the nominal plant as 1.41x, telling the policy it
# carries a heavy cutter when it does not. Deployed, that cost 45.99 -> 81.50 mm
# of lateral error at payload 1.0x, a significant fail, while 3.0x and 5.0x both
# passed. Degree 2 reads nominal as 1.14x and is better at every payload
# (bias +0.142 / +0.059 / -0.737 against +0.410 / +0.109 / -0.898).
#
# The deployment path for degree 2 is NEW CODE, so it is smoke-gated: if the
# smoke does not load the estimator and produce a report, the re-score does not
# run. A silent fallback to linear would look like a null result for degree 2.
set -u
export IMAGE=isaac-lab:2.3.2-rl052
cd /home/kevinmurera
R=/home/kevinmurera/projects/rrlab_mulag
FIT=/workspace/projects/rrlab_mulag/outputs/estimator/ctxonly_fit_d2.npz
run() { ~/bin/isaaclab-run bash -lc "$1" 2>/dev/null || true; }
exec >>/home/kevinmurera/d2rescore.log 2>&1
echo "=== d2 rescore armed $(date -u +%FT%TZ) ==="
while ps -eo args | grep -qE "[h]eldout.sh"; do sleep 60; done
while ps -eo args | grep -qE "[e]valuate_gates.py"; do sleep 45; done
echo "GPU free $(date -u +%FT%TZ)"

C=$(ls $R/logs/rsl_rl/ppo_id_aam_fd/ctxonly17/model_*.pt | sed 's/.*model_//;s/\.pt//' | sort -n | tail -1)
CKPT=logs/rsl_rl/ppo_id_aam_fd/ctxonly17/model_${C}.pt
mkdir -p $R/outputs/ctxonly_d2

echo "=== SMOKE degree-2 deployment ==="
run "
cd /workspace/projects/rrlab_mulag
export MULAG_DIRECT_VALVE=1
export MULAG_RMA_CONTEXT=1
export MULAG_PAYLOAD_RANGE=3.0,3.0
export MULAG_CONTEXT_ABLATE=estimated
export MULAG_ESTIMATOR_PATH=$FIT
/isaac-sim/python.sh PPO_AAM_FD/scripts/evaluate_gates.py $CKPT \
  --num_envs 16 --steps 30 --meshes 2718 --seed 17 \
  --report outputs/ctxonly_d2/smoke.json 2>&1 | grep -E 'estimator loaded|Traceback|rror|wrote' | tail -5
"
if [ ! -f "$R/outputs/ctxonly_d2/smoke.json" ]; then
  echo "D2_SMOKE_FAILED -- rescore not run, degree-2 deployment is broken"; exit 1
fi
echo "D2_SMOKE_OK $(date -u +%FT%TZ)"

for M in 1.0 3.0 5.0; do
  while ps -eo args | grep -qE "[e]valuate_gates.py"; do sleep 45; done
  echo "=== ctxest_d2 payload $M ==="; date -u +%FT%TZ
  run "
cd /workspace/projects/rrlab_mulag
export MULAG_DIRECT_VALVE=1
export MULAG_RMA_CONTEXT=1
export MULAG_PAYLOAD_RANGE=$M,$M
export MULAG_CONTEXT_ABLATE=estimated
export MULAG_ESTIMATOR_PATH=$FIT
export MULAG_ESTIMATOR_BETA=0.02
/isaac-sim/python.sh PPO_AAM_FD/scripts/evaluate_gates.py $CKPT \
  --num_envs 176 --steps 900 --meshes 2718 --seed 17 \
  --report outputs/ctxonly_d2/ctxest_p${M}_m2718.json 2>&1 | grep -E 'wrote|rror' | tail -2
"
  echo "SCORED_ctxest_d2_p${M}"
done
echo "D2_RESCORE_COMPLETE $(date -u +%FT%TZ)"
