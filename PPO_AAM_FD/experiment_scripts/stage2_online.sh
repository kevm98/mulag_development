#!/usr/bin/env bash
# Stage 2: the payload estimated ONLINE, never read.
#
# Runs only after Stage A, and only if its dumps exist. The policy is the SAME
# adaptive checkpoint; the only change is where its payload number comes from:
#
#   oracle      the true payload            (upper bound, privileged)
#   estimated   computed at runtime         (the deployable arm)
#   wrong       told it carries nominal     (falsification)
#   blind       the capacity-matched twin   (the arm to beat)
#
# The estimate starts at nominal each episode and converges while driving, so
# this is adaptation rather than a lookup.
set -u
export IMAGE=isaac-lab:2.3.2-rl052
cd /home/kevinmurera
R=/home/kevinmurera/projects/rrlab_mulag
MESH=2718; ENVS_N=176; STEPS=900
PAYLOADS=${PAYLOADS:-"3.0 5.0 1.0"}
run() { ~/bin/isaaclab-run bash -lc "$1" 2>/dev/null || true; }

last_ckpt() { ls $R/logs/rsl_rl/ppo_id_aam_fd/$1/model_*.pt 2>/dev/null | sed 's/.*model_//;s/\.pt//' | sort -n | tail -1; }
A=$(last_ckpt payload17_adaptive)
ADAPT=logs/rsl_rl/ppo_id_aam_fd/payload17_adaptive/model_${A}.pt
FIT=$R/outputs/estimator/payload_fit.npz

echo "=== fitting the estimator ==="; date -u +%FT%TZ
run "
cd /workspace/projects/rrlab_mulag
/isaac-sim/python.sh PPO_AAM_FD/experiment_scripts/fit_estimator.py \
  --dir outputs/estimator --out outputs/estimator/payload_fit.npz 2>&1 | tail -25
"
if [ ! -f "$FIT" ]; then echo "STAGE2_ABORT estimator fit missing"; exit 1; fi

for M in $PAYLOADS; do
  echo "=== SCORE estimated payload ${M}x ==="; date -u +%FT%TZ
  run "
cd /workspace/projects/rrlab_mulag
export MULAG_DIRECT_VALVE=1
export MULAG_RMA_CONTEXT=1
export MULAG_PAYLOAD_RANGE=$M,$M
export MULAG_CONTEXT_ABLATE=estimated
export MULAG_ESTIMATOR_PATH=/workspace/projects/rrlab_mulag/outputs/estimator/payload_fit.npz
export MULAG_ESTIMATOR_BETA=0.02
/isaac-sim/python.sh PPO_AAM_FD/scripts/evaluate_gates.py $ADAPT \
  --num_envs $ENVS_N --steps $STEPS --meshes $MESH --seed 17 \
  --report outputs/payload_cond/estimated_p${M}_m${MESH}.json 2>&1 | grep -E 'wrote|rror|estimator loaded' | tail -3
"
  echo "SCORED_estimated_p${M}"
done
echo "STAGE2_COMPLETE"
