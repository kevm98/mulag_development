#!/usr/bin/env bash
# Close the two remaining gaps, in order of value per GPU-hour.
#
# GAP 1: the time-varying result is selection-mesh only. The constant-payload
#        claim is confirmed on 3141 and 8191; the mass-step claim is not.
# GAP 2: n = 1. This project's bar is a median over five seeds and the starved
#        campaign was held to it.
#
# Configuration replicated is run 1 (ctxonly17, trained on the true context).
# A-RMA phase 3 is NOT replicated: it was rejected for destroying cut and
# contact at every payload.
set -u
export IMAGE=isaac-lab:2.3.2-rl052
cd /home/kevinmurera
R=/home/kevinmurera/projects/rrlab_mulag
BASE=logs/rsl_rl/ppo_id_aam_fd/distilled_v1/phase1_init_70.pt
FIT=/workspace/projects/rrlab_mulag/outputs/estimator/ctxonly_fit.npz
run() { ~/bin/isaaclab-run bash -lc "$1" 2>/dev/null || true; }
waitgpu() { while ps -eo args | grep -qE "[e]valuate_gates.py"; do sleep 45; done; }
exec >>/home/kevinmurera/finish.log 2>&1
echo "=== finish armed $(date -u +%FT%TZ) ==="
while ps -eo args | grep -qE "[l]oadedstep.sh|[f]reeze.sh"; do sleep 60; done
echo "prior work clear $(date -u +%FT%TZ)"

C=$(ls $R/logs/rsl_rl/ppo_id_aam_fd/ctxonly17/model_*.pt | sed 's/.*model_//;s/\.pt//' | sort -n | tail -1)
CKPT=logs/rsl_rl/ppo_id_aam_fd/ctxonly17/model_${C}.pt
mkdir -p $R/outputs/massstep_heldout $R/outputs/seeds $R/outputs/estimator_seeds

# ---- GAP 1: mass step on the held-out meshes ------------------------------
# Includes the constant-1.0x arm on each mesh so the plant gate can be applied
# there too: the frozen policy on the stepping plant must land between the two
# static plants, and the constant-5.0x arms already exist from the held-out run.
for MESH in 3141 8191; do
  for SPEC in "gate1p0:zero:export MULAG_PAYLOAD_RANGE=1.0,1.0" \
              "step_ctxzero:zero:export MULAG_MASS_STEP_TIME_S=6.0
export MULAG_MASS_STEP_JITTER_S=2.0
export MULAG_MASS_STEP_SCALE=5.0" \
              "step_ctxest:estimated:export MULAG_MASS_STEP_TIME_S=6.0
export MULAG_MASS_STEP_JITTER_S=2.0
export MULAG_MASS_STEP_SCALE=5.0"; do
    NAME=${SPEC%%:*}; REST=${SPEC#*:}; ABL=${REST%%:*}; PLANT=${REST#*:}
    EX=""; [ "$ABL" = estimated ] && EX="export MULAG_ESTIMATOR_PATH=$FIT
export MULAG_ESTIMATOR_BETA=0.02"
    waitgpu
    echo "=== heldout massstep $NAME mesh $MESH ==="; date -u +%FT%TZ
    run "
cd /workspace/projects/rrlab_mulag
export MULAG_DIRECT_VALVE=1
export MULAG_RMA_CONTEXT=1
$PLANT
export MULAG_CONTEXT_ABLATE=$ABL
$EX
/isaac-sim/python.sh PPO_AAM_FD/scripts/evaluate_gates.py $CKPT \
  --num_envs 176 --steps 900 --meshes $MESH --seed 17 \
  --report outputs/massstep_heldout/${NAME}_m${MESH}.json 2>&1 | grep -E 'wrote|rror' | tail -2
"
    echo "SCORED_heldoutstep_${NAME}_m${MESH}"
  done
done
echo "MASSSTEP_HELDOUT_COMPLETE $(date -u +%FT%TZ)"

# ---- GAP 2: seeds, on BOTH the constant and the time-varying claim --------
# Each seed gets its OWN estimator, refit from its own policy's rollout: the
# regression is on the observation distribution that policy induces, so reusing
# seed 17's fit would leak that seed's behaviour into the others.
for S in 23 101 42 7; do
  TAG=ctxonly${S}
  echo "=== TRAIN seed $S ==="; date -u +%FT%TZ
  run "
set -u
cd /workspace/projects/rrlab_mulag
export MULAG_DIRECT_VALVE=1
export MULAG_RMA_CONTEXT=1
export MULAG_PAYLOAD_RANGE=1.0,5.0
export MULAG_PAYLOAD_RESAMPLE=1
export MULAG_CONTEXT_ABLATE=live
/isaac-sim/python.sh PPO_AAM_FD/scripts/train.py \
  --resume $BASE --train_context_only \
  --num_envs 256 --stage_iterations 400 --seed $S \
  --lateral_tracking_std_m 0.09 --lateral_tracking_weight 4.0 \
  --log_dir logs/rsl_rl/ppo_id_aam_fd/${TAG} --headless 2>&1 | tail -2
"
  SC=$(ls $R/logs/rsl_rl/ppo_id_aam_fd/${TAG}/model_*.pt 2>/dev/null | sed 's/.*model_//;s/\.pt//' | sort -n | tail -1)
  [ -z "$SC" ] && { echo "SEED_${S}_ABORT no checkpoint"; continue; }
  SCK=logs/rsl_rl/ppo_id_aam_fd/${TAG}/model_${SC}.pt
  echo "TRAINED_seed_${S}"
  waitgpu
  run "
cd /workspace/projects/rrlab_mulag
export MULAG_DIRECT_VALVE=1
export MULAG_RMA_CONTEXT=1
export MULAG_PAYLOAD_RANGE=1.0,5.0
export MULAG_PAYLOAD_RESAMPLE=1
export MULAG_CONTEXT_ABLATE=live
/isaac-sim/python.sh PPO_AAM_FD/scripts/evaluate_gates.py $SCK \
  --num_envs 176 --steps 900 --meshes 2718 --seed 17 \
  --obs_dump outputs/estimator_seeds/s${S}_mixed.npz --obs_dump_every 10 \
  --report outputs/seeds/mix_s${S}.json 2>&1 | grep -E 'wrote|rror' | tail -2
"
  run "
cd /workspace/projects/rrlab_mulag
/isaac-sim/python.sh PPO_AAM_FD/experiment_scripts/fit_estimator.py \
  --dir outputs/estimator_seeds --prefix s${S} \
  --out outputs/estimator_seeds/s${S}_fit.npz 2>&1 | tail -3
"
  [ -f "$R/outputs/estimator_seeds/s${S}_fit.npz" ] || { echo "SEED_${S}_ABORT no fit"; continue; }
  SFIT=/workspace/projects/rrlab_mulag/outputs/estimator_seeds/s${S}_fit.npz
  # constant 5.0x AND the 5.0x mid-episode step: the two claims that survive.
  for PLANTNAME in const5 step5; do
    if [ "$PLANTNAME" = const5 ]; then P="export MULAG_PAYLOAD_RANGE=5.0,5.0"
    else P="export MULAG_MASS_STEP_TIME_S=6.0
export MULAG_MASS_STEP_JITTER_S=2.0
export MULAG_MASS_STEP_SCALE=5.0"; fi
    for ARM in ctxzero ctxest; do
      if [ "$ARM" = ctxest ]; then ABL=estimated; EX="export MULAG_ESTIMATOR_PATH=$SFIT
export MULAG_ESTIMATOR_BETA=0.02"; else ABL=zero; EX=""; fi
      waitgpu
      echo "=== seed $S $ARM $PLANTNAME ==="; date -u +%FT%TZ
      run "
cd /workspace/projects/rrlab_mulag
export MULAG_DIRECT_VALVE=1
export MULAG_RMA_CONTEXT=1
$P
export MULAG_CONTEXT_ABLATE=$ABL
$EX
/isaac-sim/python.sh PPO_AAM_FD/scripts/evaluate_gates.py $SCK \
  --num_envs 176 --steps 900 --meshes 2718 --seed 17 \
  --report outputs/seeds/${ARM}_${PLANTNAME}_s${S}.json 2>&1 | grep -E 'wrote|rror' | tail -2
"
      echo "SCORED_s${S}_${ARM}_${PLANTNAME}"
    done
  done
  echo "SEED_DONE_${S} $(date -u +%FT%TZ)"
done
echo "FINISH_COMPLETE $(date -u +%FT%TZ)"
