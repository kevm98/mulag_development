#!/usr/bin/env bash
# Try the mechanism's own prescription on the nominal-payload harm.
#
# Three fixes for the 1.0x lateral loss have now failed:
#   estimator bias   degree-2 cut the nominal over-read 1.41x -> 1.14x and
#                    lateral moved 81.50 -> 79.65 mm, still significant
#   deadband         zeroing the estimate below a threshold gave 89.92 mm,
#                    WORSE, because the estimate wobbles across the threshold
#                    and the context then jumps discontinuously between 0 and
#                    0.82 -- precisely the abrupt arrival the beta ladder had
#                    just shown to be harmful
#
# The ladder's own prescription was never tested at nominal: it ran only at
# 3.0x, where slower smoothing was monotonically better on both channels
# (beta 1.0 -> 0.02 -> 0.005 giving cut -39.3% -> +74.6%* -> +143.6%*). A slower
# ramp holds the context near zero longer and varies it less, which is what a
# plant needing no adaptation should see.
#
# Scored at 1.0x for the harm and at 5.0x to confirm the loaded gain survives.
set -u
export IMAGE=isaac-lab:2.3.2-rl052
cd /home/kevinmurera
R=/home/kevinmurera/projects/rrlab_mulag
FIT=/workspace/projects/rrlab_mulag/outputs/estimator/ctxonly_fit.npz
run() { ~/bin/isaaclab-run bash -lc "$1" 2>/dev/null || true; }
exec >>/home/kevinmurera/nominal_beta.log 2>&1
echo "=== nominal_beta armed $(date -u +%FT%TZ) ==="
while ps -eo args | grep -qE "[d]eadband.sh"; do sleep 45; done
C=$(ls $R/logs/rsl_rl/ppo_id_aam_fd/ctxonly17/model_*.pt | sed 's/.*model_//;s/\.pt//' | sort -n | tail -1)
CKPT=logs/rsl_rl/ppo_id_aam_fd/ctxonly17/model_${C}.pt
mkdir -p $R/outputs/nominal_beta
for M in 1.0 5.0; do
  while ps -eo args | grep -qE "[e]valuate_gates.py"; do sleep 45; done
  echo "=== beta 0.005 payload $M ==="; date -u +%FT%TZ
  run "
cd /workspace/projects/rrlab_mulag
export MULAG_DIRECT_VALVE=1
export MULAG_RMA_CONTEXT=1
export MULAG_PAYLOAD_RANGE=$M,$M
export MULAG_CONTEXT_ABLATE=estimated
export MULAG_ESTIMATOR_PATH=$FIT
export MULAG_ESTIMATOR_BETA=0.005
/isaac-sim/python.sh PPO_AAM_FD/scripts/evaluate_gates.py $CKPT \
  --num_envs 176 --steps 900 --meshes 2718 --seed 17 \
  --report outputs/nominal_beta/ctxest_b0005_p${M}_m2718.json 2>&1 | grep -E 'wrote|rror' | tail -2
"
  echo "SCORED_b0005_p${M}"
done
echo "NOMINAL_BETA_COMPLETE $(date -u +%FT%TZ)"
