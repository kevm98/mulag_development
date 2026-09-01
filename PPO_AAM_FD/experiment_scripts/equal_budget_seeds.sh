#!/usr/bin/env bash
# EQUAL-BUDGET SEED REPLICATION at 1200 iterations.
#
# Why: at 400 iterations every seed was still RISING in reward, and the paired
# scores against the frozen base spanned -59.3% to +160.8% with training metrics
# carrying no predictive information (corr +0.02). Seed 7 resumed to 1200 moved
# -59.3% -> -15.3%, so undertraining explains part of the spread but did not
# close it.
#
# The comparison is only valid at EQUAL BUDGET: a median that mixes a
# 1200-iteration seed 7 with 400-iteration seeds measures budget, not seed.
# Every seed therefore gets 400 + 800 = 1200.
#
# Pre-registered read, unchanged from longtrain.sh: if the spread NARROWS and
# seeds 7/101 move up, the spread was undertraining. If the spread stays wide,
# the method is genuinely unstable at this scale -- and that is the answer to
# report, not a failure to hide.
#
# PHASE A (necessary): history arms -> the headline claim, adaptive vs base.
# PHASE B (attribution): control arms for the seeds where the history-zeroed
# control BEAT the history arm at 400 (101: -33.7%, 7: -79.6%) plus the headline
# seed 17. Decides whether that inversion is an undertraining artefact.
set -u
export IMAGE=isaac-lab:2.3.2-rl052
cd /home/kevinmurera

ENVS="
export MULAG_DIRECT_VALVE=1
export MULAG_RMA_CONTEXT=1
export MULAG_PAYLOAD_RANGE=1.0,1.0
export MULAG_MASS_STEP_TIME_S=20.0
export MULAG_MASS_STEP_SCALE=3.0
export MULAG_MASS_STEP_JITTER_S=8.0
export MULAG_MASS_BIMODAL=1.0,3.0
"

# resume_dir  out_tag  extra_flags
run_arm () {
  local SRC=$1 TAG=$2 SEED=$3 EXTRA=${4:-}
  ~/bin/isaaclab-run bash -lc "
set -u
cd /workspace/projects/rrlab_mulag
$ENVS
/isaac-sim/python.sh PPO_AAM_FD/scripts/train_residual_adapter.py \
  logs/rsl_rl/ppo_id_aam_fd/distilled_v1/phase1_init_70.pt \
  --resume_residual logs/rsl_rl/ppo_id_aam_fd/$SRC/model_399.pt \
  --num_envs 256 --iterations 800 --history 20 --latent_dim 8 --alpha 0.5 \
  --lr 3e-4 --seed $SEED $EXTRA \
  --log_dir logs/rsl_rl/ppo_id_aam_fd/$TAG \
  --headless 2>&1 | tee outputs/$TAG.log | tail -2
" || true
  echo "TRAINED_$TAG"
  for M in 1337 17 4242; do
    ~/bin/isaaclab-run bash -lc "
cd /workspace/projects/rrlab_mulag
$ENVS
/isaac-sim/python.sh PPO_AAM_FD/scripts/evaluate_gates.py \
  logs/rsl_rl/ppo_id_aam_fd/distilled_v1/phase1_init_70.pt \
  --residual_checkpoint logs/rsl_rl/ppo_id_aam_fd/$TAG/model_799.pt \
  --residual_alpha 0.5 --residual_history 20 ${EXTRA/--no_history/--residual_no_history} \
  --num_envs 176 --steps 900 --meshes $M --seed 17 \
  --report outputs/rif_delay/${TAG}_m${M}.json 2>&1 | grep -E 'wrote|rror' | tail -2
" 2>/dev/null || true
    echo "SCORED_${TAG}_m${M}"
  done
  echo "ARM_DONE_$TAG"
}

echo "=== PHASE A: history arms to 1200 (seed 7 already done) ==="
run_arm seed101_history   long101_history 101
run_arm seed42_history    long42_history  42
run_arm seed23_history    long23_history  23
run_arm residual_history  long17_history  17
echo PHASE_A_DONE

echo "=== PHASE B: control arms to 1200, seeds where control beat history ==="
run_arm residual_nohistory long17_nohistory 17  --no_history
run_arm seed101_nohistory  long101_nohistory 101 --no_history
run_arm seed7_nohistory    long7_nohistory   7   --no_history
echo PHASE_B_DONE
echo EQUAL_BUDGET_DONE
