#!/usr/bin/env bash
# Stage A of reports/PREREG_PAYLOAD_CONDITIONING_2026-09-06.md.
#
# Train the context columns the frozen base multiplies by exactly zero, on the
# payload range where the payload actually costs something (1-5x, 95x on cut).
#
# Two trained arms from the SAME init. phase1_init_70.pt has context weights of
# exactly 0.000000, so at iteration 0 the two arms are literally the same policy
# for any context value; only one of them can see the payload it is carrying.
#
# The fine-tune is deliberately UNANCHORED. A trajectory-regularised anchor is
# what memory.md measures as unable to acquire a new input dependence, and a new
# input dependence is the entire point here. The risk is that PPO walks away
# from the distilled behaviour; that risk lands on BOTH arms equally, which is
# why the primary comparison is blind -> adaptive and not base -> adaptive.
set -u
export IMAGE=isaac-lab:2.3.2-rl052
BASE=logs/rsl_rl/ppo_id_aam_fd/distilled_v1/phase1_init_70.pt
SEED=${SEED:-17}
ITERS=${ITERS:-400}
TRAIN_ENVS=${TRAIN_ENVS:-256}
MESH=${MESH:-2718}
SCORE_ENVS=${SCORE_ENVS:-176}
STEPS=${STEPS:-900}
PAYLOADS=${PAYLOADS:-"3.0 5.0 1.0"}
LSTD=0.09
LW=4.0

PLANT="
export MULAG_DIRECT_VALVE=1
export MULAG_RMA_CONTEXT=1
export MULAG_PAYLOAD_RANGE=1.0,5.0
export MULAG_PAYLOAD_RESAMPLE=1
"

cd /home/kevinmurera
run() { ~/bin/isaaclab-run bash -lc "$1" 2>/dev/null || true; }
mkdir -p /home/kevinmurera/projects/rrlab_mulag/outputs/payload_cond

# ---- Train both arms -------------------------------------------------------
# Serial, not parallel: two trainings fit in RAM but a failure in one would be
# ambiguous, and the whole point of this design is that the arms differ in
# exactly one thing.
for ARM in adaptive blind; do
  if [ "$ARM" = blind ]; then ABL="export MULAG_CONTEXT_ABLATE=zero"; else ABL="export MULAG_CONTEXT_ABLATE=live"; fi
  TAG=payload${SEED}_${ARM}
  echo "=== TRAIN ${TAG} ==="; date -u +%FT%TZ
  run "
set -u
cd /workspace/projects/rrlab_mulag
$PLANT
$ABL
/isaac-sim/python.sh PPO_AAM_FD/scripts/train.py \
  --resume $BASE \
  --num_envs $TRAIN_ENVS --stage_iterations $ITERS --seed $SEED \
  --lateral_tracking_std_m $LSTD --lateral_tracking_weight $LW \
  --log_dir logs/rsl_rl/ppo_id_aam_fd/${TAG} \
  --headless 2>&1 | tee outputs/payload_cond/${TAG}.log | tail -3
"
  echo "TRAINED_${TAG}"
done

# ---- Score ----------------------------------------------------------------
# Separate processes with common random numbers, paired per environment index.
# That is this project's established convention (memory.md invariant 3); the
# arms are different checkpoints so they cannot share one process.
#
# Fixed payload per run, so the result can be read per payload rather than
# pooled across a range the arms may handle very differently.
for M in $PAYLOADS; do
  for ARM in base blind adaptive adaptive_shuffle; do
    case $ARM in
      base)             CKPT=$BASE;                                          ABL=live    ;;
      blind)            CKPT=logs/rsl_rl/ppo_id_aam_fd/payload${SEED}_blind/model_$((ITERS-1)).pt;    ABL=zero    ;;
      adaptive)         CKPT=logs/rsl_rl/ppo_id_aam_fd/payload${SEED}_adaptive/model_$((ITERS-1)).pt; ABL=live    ;;
      adaptive_shuffle) CKPT=logs/rsl_rl/ppo_id_aam_fd/payload${SEED}_adaptive/model_$((ITERS-1)).pt; ABL=shuffle ;;
    esac
    OUT="outputs/payload_cond/${ARM}_p${M}_m${MESH}.json"
    echo "=== SCORE ${ARM} payload ${M}x ==="; date -u +%FT%TZ
    run "
cd /workspace/projects/rrlab_mulag
export MULAG_DIRECT_VALVE=1
export MULAG_RMA_CONTEXT=1
export MULAG_PAYLOAD_RANGE=$M,$M
export MULAG_CONTEXT_ABLATE=$ABL
/isaac-sim/python.sh PPO_AAM_FD/scripts/evaluate_gates.py $CKPT \
  --num_envs $SCORE_ENVS --steps $STEPS --meshes $MESH --seed 17 \
  --report $OUT 2>&1 | grep -E 'wrote|rror' | tail -2
"
    echo "SCORED_${ARM}_p${M}"
  done
done
echo "PAYLOAD_CONDITIONING_STAGE_A_COMPLETE"
