#!/usr/bin/env bash
# Make the estimate actually track, by fixing three separate causes.
#
# The trace shows the estimate hovering 1.04-2.42 against a true square wave of
# 1.0 <-> 3.0, correlation 0.07-0.46 across environments. Three causes, each
# addressed:
#
# 1 OUT OF DISTRIBUTION. The estimator was fit on CONSTANT-payload episodes and
#   is deployed on a changing one, where the mass moves mid-episode and creates
#   transients it has never seen. Same class of error A-RMA identified for the
#   policy, one level down. Fix: refit on data collected from the PERIODIC
#   plant.
# 2 SHRINKAGE. Ridge pulls toward the mean by construction, which is exactly the
#   compression the trace shows. Fix: degree-2 features, measured offline at
#   RMSE 0.748 against the linear 0.853, and the degree travels in the fit file.
# 3 LAG. beta = 0.02 is a 50-step window against a 3 s half-period. Fix: sweep
#   0.02 / 0.05 / 0.10 and report the trace correlation for each rather than
#   assuming faster is better -- faster also admits more noise.
set -u
export IMAGE=isaac-lab:2.3.2-rl052
cd /home/kevinmurera
R=/home/kevinmurera/projects/rrlab_mulag
run() { ~/bin/isaaclab-run bash -lc "$1" 2>/dev/null || true; }
waitgpu() { while ps -eo args | grep -qE "[e]valuate_gates.py"; do sleep 45; done; }
exec >>/home/kevinmurera/estfix.log 2>&1
echo "=== estfix armed $(date -u +%FT%TZ) ==="
while ps -eo args | grep -qE "[f]ilm.sh"; do sleep 60; done
C=$(ls $R/logs/rsl_rl/ppo_id_aam_fd/ctxonly17/model_*.pt | sed 's/.*model_//;s/\.pt//' | sort -n | tail -1)
CKPT=logs/rsl_rl/ppo_id_aam_fd/ctxonly17/model_${C}.pt
mkdir -p $R/outputs/estfix $R/outputs/estimator_per

# 1. Collect on the PERIODIC plant, which is where it will be deployed.
waitgpu
echo "=== collect on the periodic plant ==="; date -u +%FT%TZ
run "
cd /workspace/projects/rrlab_mulag
export MULAG_DIRECT_VALVE=1
export MULAG_RMA_CONTEXT=1
export MULAG_MASS_PERIOD_S=6.0
export MULAG_MASS_STEP_SCALE=3.0
export MULAG_CONTEXT_ABLATE=live
/isaac-sim/python.sh PPO_AAM_FD/scripts/evaluate_gates.py $CKPT \
  --num_envs 176 --steps 900 --meshes 2718 --seed 17 \
  --obs_dump outputs/estimator_per/per_mixed.npz --obs_dump_every 4 \
  --report outputs/estfix/collect.json 2>&1 | grep -E 'wrote|rror' | tail -2
"
# A second dump at a different scale so the fit sees more of the range.
waitgpu
run "
cd /workspace/projects/rrlab_mulag
export MULAG_DIRECT_VALVE=1
export MULAG_RMA_CONTEXT=1
export MULAG_MASS_PERIOD_S=6.0
export MULAG_MASS_STEP_SCALE=5.0
export MULAG_CONTEXT_ABLATE=live
/isaac-sim/python.sh PPO_AAM_FD/scripts/evaluate_gates.py $CKPT \
  --num_envs 176 --steps 900 --meshes 2718 --seed 17 \
  --obs_dump outputs/estimator_per/per_p5.npz --obs_dump_every 4 \
  --report outputs/estfix/collect5.json 2>&1 | grep -E 'wrote|rror' | tail -2
"
# 2. Refit, degree 2, on the periodic data.
run "
cd /workspace/projects/rrlab_mulag
/isaac-sim/python.sh PPO_AAM_FD/experiment_scripts/fit_estimator.py \
  --dir outputs/estimator_per --prefix per \
  --out outputs/estimator_per/per_fit.npz 2>&1 | tail -8
"
PFIT=/workspace/projects/rrlab_mulag/outputs/estimator_per/per_fit.npz
[ -f "$R/outputs/estimator_per/per_fit.npz" ] || { echo "ESTFIX_ABORT no fit"; exit 1; }

# 3. Beta sweep, traced, so the correlation can be read rather than assumed.
for B in 0.02 0.05 0.10; do
  waitgpu
  echo "=== estfix trace beta=$B ==="; date -u +%FT%TZ
  run "
cd /workspace/projects/rrlab_mulag
export MULAG_DIRECT_VALVE=1
export MULAG_RMA_CONTEXT=1
export MULAG_MASS_PERIOD_S=6.0
export MULAG_MASS_STEP_SCALE=3.0
export MULAG_CONTEXT_ABLATE=estimated
export MULAG_ESTIMATOR_PATH=$PFIT
export MULAG_ESTIMATOR_BETA=$B
/isaac-sim/python.sh PPO_AAM_FD/scripts/evaluate_gates.py $CKPT \
  --num_envs 176 --steps 900 --meshes 2718 --seed 17 \
  --adapt_trace outputs/estfix/trace_b${B}.json \
  --report outputs/estfix/report_b${B}.json 2>&1 | grep -E 'wrote|rror' | tail -2
"
  echo "SCORED_estfix_b${B}"
done
echo "ESTFIX_COMPLETE $(date -u +%FT%TZ)"
