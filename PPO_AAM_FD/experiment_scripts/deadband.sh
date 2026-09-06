#!/usr/bin/env bash
# Does a deadband remove the nominal-payload harm without costing the loaded gain?
#
# At payload 1.0x the online arm loses lateral, 45.99 -> 81.50 mm. Halving the
# estimator's bias (1.41x -> 1.14x via degree-2 features) moved that to 79.65 mm
# and it stayed significant, so the harm is NOT proportional to how wrong the
# estimate is. It tracks the estimate being nonzero and time-varying at all.
#
# A deadband forces the estimate to exactly nominal below a threshold, so the
# policy is bit-for-bit frozen on a plant that does not need it. In normalised
# units the payload maps as (m - 1) / 0.5, so 1.0 corresponds to a 1.5x cutter.
# The linear estimator reads nominal as 1.41x, i.e. 0.82 normalised, which a
# threshold of 1.0 suppresses while leaving 3.0x (4.22) and 5.0x (6.2) untouched.
#
# Scored at 1.0x to see whether the harm goes, and at 3.0x and 5.0x to confirm
# the loaded gains survive. A fix that rescues nominal by killing the result is
# not a fix.
set -u
export IMAGE=isaac-lab:2.3.2-rl052
cd /home/kevinmurera
R=/home/kevinmurera/projects/rrlab_mulag
FIT=/workspace/projects/rrlab_mulag/outputs/estimator/ctxonly_fit.npz
run() { ~/bin/isaaclab-run bash -lc "$1" 2>/dev/null || true; }
exec >>/home/kevinmurera/deadband.log 2>&1
echo "=== deadband armed $(date -u +%FT%TZ) ==="
while ps -eo args | grep -qE "[a]blate.sh|[d]2rescore.sh|[c]txonly3.sh"; do sleep 60; done
while ps -eo args | grep -qE "[e]valuate_gates.py"; do sleep 45; done
echo "GPU free $(date -u +%FT%TZ)"
C=$(ls $R/logs/rsl_rl/ppo_id_aam_fd/ctxonly17/model_*.pt | sed 's/.*model_//;s/\.pt//' | sort -n | tail -1)
CKPT=logs/rsl_rl/ppo_id_aam_fd/ctxonly17/model_${C}.pt
mkdir -p $R/outputs/deadband
for M in 1.0 3.0 5.0; do
  while ps -eo args | grep -qE "[e]valuate_gates.py"; do sleep 45; done
  echo "=== deadband 1.0 payload $M ==="; date -u +%FT%TZ
  run "
cd /workspace/projects/rrlab_mulag
export MULAG_DIRECT_VALVE=1
export MULAG_RMA_CONTEXT=1
export MULAG_PAYLOAD_RANGE=$M,$M
export MULAG_CONTEXT_ABLATE=estimated
export MULAG_ESTIMATOR_PATH=$FIT
export MULAG_ESTIMATOR_BETA=0.02
export MULAG_ESTIMATOR_DEADBAND=1.0
/isaac-sim/python.sh PPO_AAM_FD/scripts/evaluate_gates.py $CKPT \
  --num_envs 176 --steps 900 --meshes 2718 --seed 17 \
  --report outputs/deadband/ctxest_db_p${M}_m2718.json 2>&1 | grep -E 'wrote|rror' | tail -2
"
  echo "SCORED_deadband_p${M}"
done
echo "DEADBAND_COMPLETE $(date -u +%FT%TZ)"
