#!/usr/bin/env bash
# Experiment A: the adaptation test proper -- payload STEPS mid-episode.
#
# TIMING CORRECTED. The first attempt scheduled the step at 20 s +- 8 s, but a
# 900-step rollout at control_dt = 1/60 s lasts exactly 15.0 s, so the step could
# only fire in the last 0-3 s of roughly a fifth of the environments. The plant
# sanity check caught it: the frozen policy on the "stepping" plant read cut
# 4.17 mm against 3.23 mm on a constant nominal plant, where a real step to 3.0x
# should have pushed it toward the 16.91 mm it scores at constant 3.0x. Those
# reports were deleted rather than kept.
#
# 6 s +- 2 s fires at 4-8 s, leaving 7-11 s of post-step episode to measure.
#
# Everything before this held payload constant per episode, which measures
# conditioning, not adaptation. Here every environment starts at its base mass
# and steps at 6 s +- 2 s of jitter. The jitter is the point: a fixed instant
# could be compensated on a timer without sensing anything.
#
# MULAG_MASS_BIMODAL is deliberately OMITTED. Invariant 21: setting it assigns
# half the fleet to each mode from t=0, making the plant statically heavier AND
# the step larger together, which confounds them. Omitted, the step applies
# uniformly and both arms share a common pre-step baseline.
#
# Runs first, before the A-RMA retrain, because it needs no training and is the
# strongest claim available from the checkpoint that already exists.
set -u
export IMAGE=isaac-lab:2.3.2-rl052
cd /home/kevinmurera
R=/home/kevinmurera/projects/rrlab_mulag
FIT=/workspace/projects/rrlab_mulag/outputs/estimator/ctxonly_fit.npz
run() { ~/bin/isaaclab-run bash -lc "$1" 2>/dev/null || true; }
exec >>/home/kevinmurera/adapt.log 2>&1
echo "=== adapt armed $(date -u +%FT%TZ) ==="
while ps -eo args | grep -qE "[d]eadband.sh|[n]ominal_beta.sh"; do sleep 45; done
C=$(ls $R/logs/rsl_rl/ppo_id_aam_fd/ctxonly17/model_*.pt | sed 's/.*model_//;s/\.pt//' | sort -n | tail -1)
CKPT=logs/rsl_rl/ppo_id_aam_fd/ctxonly17/model_${C}.pt
mkdir -p $R/outputs/massstep

for STEP in 3.0 5.0; do
  for ARM in ctxzero ctxest ctxest_b0005; do
    case $ARM in
      ctxzero)       ABL=zero;      EX="" ;;
      ctxest)        ABL=estimated; EX="export MULAG_ESTIMATOR_PATH=$FIT
export MULAG_ESTIMATOR_BETA=0.02" ;;
      ctxest_b0005)  ABL=estimated; EX="export MULAG_ESTIMATOR_PATH=$FIT
export MULAG_ESTIMATOR_BETA=0.005" ;;
    esac
    while ps -eo args | grep -qE "[e]valuate_gates.py"; do sleep 45; done
    echo "=== $ARM step x$STEP ==="; date -u +%FT%TZ
    run "
cd /workspace/projects/rrlab_mulag
export MULAG_DIRECT_VALVE=1
export MULAG_RMA_CONTEXT=1
export MULAG_MASS_STEP_TIME_S=6.0
export MULAG_MASS_STEP_JITTER_S=2.0
export MULAG_MASS_STEP_SCALE=$STEP
export MULAG_CONTEXT_ABLATE=$ABL
$EX
/isaac-sim/python.sh PPO_AAM_FD/scripts/evaluate_gates.py $CKPT \
  --num_envs 176 --steps 900 --meshes 2718 --seed 17 \
  --report outputs/massstep/${ARM}_step${STEP}_m2718.json 2>&1 | grep -E 'wrote|rror|mass_scale' | tail -3
"
    echo "SCORED_${ARM}_step${STEP}"
  done
done
echo "MASSSTEP_COMPLETE $(date -u +%FT%TZ)"

# --- Experiment B: A-RMA phase 3, policy fine-tuned WITH the estimator in loop
echo "=== A-RMA phase 3 training ==="; date -u +%FT%TZ
run "
set -u
cd /workspace/projects/rrlab_mulag
export MULAG_DIRECT_VALVE=1
export MULAG_RMA_CONTEXT=1
export MULAG_PAYLOAD_RANGE=1.0,5.0
export MULAG_PAYLOAD_RESAMPLE=1
export MULAG_CONTEXT_ABLATE=estimated
export MULAG_ESTIMATOR_PATH=$FIT
export MULAG_ESTIMATOR_BETA=0.02
/isaac-sim/python.sh PPO_AAM_FD/scripts/train.py \
  --resume logs/rsl_rl/ppo_id_aam_fd/distilled_v1/phase1_init_70.pt \
  --train_context_only \
  --num_envs 256 --stage_iterations 400 --seed 17 \
  --lateral_tracking_std_m 0.09 --lateral_tracking_weight 4.0 \
  --log_dir logs/rsl_rl/ppo_id_aam_fd/arma3 \
  --headless 2>&1 | tail -3
"
echo "TRAINED_arma3"
A=$(ls $R/logs/rsl_rl/ppo_id_aam_fd/arma3/model_*.pt 2>/dev/null | sed 's/.*model_//;s/\.pt//' | sort -n | tail -1)
[ -z "$A" ] && { echo "ARMA3_ABORT no checkpoint"; exit 1; }
ACK=logs/rsl_rl/ppo_id_aam_fd/arma3/model_${A}.pt
mkdir -p $R/outputs/arma3
for M in 1.0 3.0 5.0; do
  for ARM in ctxzero ctxest; do
    if [ "$ARM" = ctxest ]; then ABL=estimated; EX="export MULAG_ESTIMATOR_PATH=$FIT
export MULAG_ESTIMATOR_BETA=0.02"; else ABL=zero; EX=""; fi
    while ps -eo args | grep -qE "[e]valuate_gates.py"; do sleep 45; done
    echo "=== arma3 $ARM payload $M ==="; date -u +%FT%TZ
    run "
cd /workspace/projects/rrlab_mulag
export MULAG_DIRECT_VALVE=1
export MULAG_RMA_CONTEXT=1
export MULAG_PAYLOAD_RANGE=$M,$M
export MULAG_CONTEXT_ABLATE=$ABL
$EX
/isaac-sim/python.sh PPO_AAM_FD/scripts/evaluate_gates.py $ACK \
  --num_envs 176 --steps 900 --meshes 2718 --seed 17 \
  --report outputs/arma3/${ARM}_p${M}_m2718.json 2>&1 | grep -E 'wrote|rror' | tail -2
"
    echo "SCORED_arma3_${ARM}_p${M}"
  done
done
echo "ARMA3_COMPLETE $(date -u +%FT%TZ)"
