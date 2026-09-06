#!/usr/bin/env bash
# 1D CNN payload estimator over a history window.
#
# The per-frame regressor reaches RMSE 0.85x and needs heavy EMA smoothing,
# which is what compresses the estimate in the trace (true 1.0<->3.0, estimated
# 1.04-2.42, correlation 0.07-0.46). Payload enters the dynamics as force
# against acceleration, a relationship OVER TIME that no single frame can
# express, so the architecture was the limit rather than the tuning. This is
# also what RMA's adaptation module actually is.
#
# Needs STRIDE-1 dumps: at any other stride the rows are not contiguous in time
# and a convolution over them is meaningless. The fitter refuses a strided dump.
#
# Collected on the PERIODIC plant, so the estimator is trained on the
# distribution it is deployed on, which is the same correction A-RMA makes for
# the policy.
set -u
export IMAGE=isaac-lab:2.3.2-rl052
cd /home/kevinmurera
R=/home/kevinmurera/projects/rrlab_mulag
run() { ~/bin/isaaclab-run bash -lc "$1" 2>/dev/null || true; }
waitgpu() { while ps -eo args | grep -qE "[e]valuate_gates.py"; do sleep 45; done; }
exec >>/home/kevinmurera/cnn.log 2>&1
echo "=== cnn armed $(date -u +%FT%TZ) ==="
while ps -eo args | grep -qE "[e]stfix.sh|[f]ilm.sh"; do sleep 60; done
C=$(ls $R/logs/rsl_rl/ppo_id_aam_fd/ctxonly17/model_*.pt | sed 's/.*model_//;s/\.pt//' | sort -n | tail -1)
CKPT=logs/rsl_rl/ppo_id_aam_fd/ctxonly17/model_${C}.pt
mkdir -p $R/outputs/cnn $R/outputs/estimator_cnn

for SC in 3.0 5.0; do
  waitgpu
  echo "=== collect stride-1, periodic scale $SC ==="; date -u +%FT%TZ
  run "
cd /workspace/projects/rrlab_mulag
export MULAG_DIRECT_VALVE=1
export MULAG_RMA_CONTEXT=1
export MULAG_MASS_PERIOD_S=6.0
export MULAG_MASS_STEP_SCALE=$SC
export MULAG_CONTEXT_ABLATE=live
/isaac-sim/python.sh PPO_AAM_FD/scripts/evaluate_gates.py $CKPT \
  --num_envs 176 --steps 900 --meshes 2718 --seed 17 \
  --obs_dump outputs/estimator_cnn/cnn_s${SC}.npz --obs_dump_every 1 \
  --report outputs/cnn/collect_s${SC}.json 2>&1 | grep -E 'wrote obs dump|rror' | tail -2
"
done
echo "=== train the CNN ==="; date -u +%FT%TZ
run "
cd /workspace/projects/rrlab_mulag
/isaac-sim/python.sh PPO_AAM_FD/experiment_scripts/fit_cnn_estimator.py \
  --dir outputs/estimator_cnn --prefix cnn \
  --out outputs/estimator_cnn/cnn.pt 2>&1 | tail -14
"
CNNPT=/workspace/projects/rrlab_mulag/outputs/estimator_cnn/cnn.pt
[ -f "$R/outputs/estimator_cnn/cnn.pt" ] || { echo "CNN_ABORT no model"; exit 1; }

# Trace first: the whole point is whether the estimate now FOLLOWS the payload.
waitgpu
echo "=== CNN trace on the periodic plant ==="; date -u +%FT%TZ
run "
cd /workspace/projects/rrlab_mulag
export MULAG_DIRECT_VALVE=1
export MULAG_RMA_CONTEXT=1
export MULAG_MASS_PERIOD_S=6.0
export MULAG_MASS_STEP_SCALE=3.0
export MULAG_CONTEXT_ABLATE=estimated
export MULAG_ESTIMATOR_PATH=$CNNPT
/isaac-sim/python.sh PPO_AAM_FD/scripts/evaluate_gates.py $CKPT \
  --num_envs 176 --steps 900 --meshes 2718 --seed 17 \
  --adapt_trace outputs/cnn/trace_periodic.json \
  --report outputs/cnn/report_periodic.json 2>&1 | grep -E 'wrote|CNN payload|rror' | tail -3
"
echo "SCORED_cnn_periodic"
# Then the configurations the linear estimator failed on lateral.
for SPEC in "p1.0:export MULAG_PAYLOAD_RANGE=1.0,1.0" \
            "p5.0:export MULAG_PAYLOAD_RANGE=5.0,5.0" \
            "cyc3:export MULAG_MASS_PERIOD_S=6.0
export MULAG_MASS_STEP_SCALE=3.0"; do
  NAME=${SPEC%%:*}; PLANT=${SPEC#*:}
  waitgpu
  echo "=== cnn ctxest $NAME ==="; date -u +%FT%TZ
  run "
cd /workspace/projects/rrlab_mulag
export MULAG_DIRECT_VALVE=1
export MULAG_RMA_CONTEXT=1
$PLANT
export MULAG_CONTEXT_ABLATE=estimated
export MULAG_ESTIMATOR_PATH=$CNNPT
/isaac-sim/python.sh PPO_AAM_FD/scripts/evaluate_gates.py $CKPT \
  --num_envs 176 --steps 900 --meshes 2718 --seed 17 \
  --report outputs/cnn/ctxest_${NAME}.json 2>&1 | grep -E 'wrote|rror' | tail -2
"
  echo "SCORED_cnn_${NAME}"
done
echo "CNN_COMPLETE $(date -u +%FT%TZ)"
