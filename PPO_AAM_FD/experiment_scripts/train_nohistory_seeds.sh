#!/usr/bin/env bash
# CAPACITY-MATCHED CONTROLS for the multi-seed triplet experiment.
#
# Trains the zero-history residual for seeds 23, 42, 101, 7. Seed 17 already
# exists (rebal17_nohistory). Together with rebal<seed>_history and the frozen
# base these complete five {base, zero-history, live-history} triplets.
#
# Identical to seeds_rebalanced.sh in EVERY argument except --no_history, which
# zeroes the history block while keeping the input width -- so parameter count,
# architecture, reward, budget, plant and seed are matched within each triplet
# and the only difference is access to the live history.
#
# Four concurrent trainings at 256 envs hold ~13 GB of the 31 GB box. Do NOT add
# a fifth, and do NOT run evaluations alongside (one eval alone peaks at ~17 GB).
set -u
export IMAGE=isaac-lab:2.3.2-rl052
SEEDS=${SEEDS:-"23 42 101 7"}
LSTD=0.09
LW=4.0
cd /home/kevinmurera/projects/rrlab_mulag
ENVS="
export MULAG_DIRECT_VALVE=1
export MULAG_RMA_CONTEXT=1
export MULAG_PAYLOAD_RANGE=1.0,1.0
export MULAG_MASS_STEP_TIME_S=20.0
export MULAG_MASS_STEP_SCALE=3.0
export MULAG_MASS_STEP_JITTER_S=8.0
export MULAG_MASS_BIMODAL=1.0,3.0
"
for SEED in $SEEDS; do
  ~/bin/isaaclab-run bash -lc "
set -u
cd /workspace/projects/rrlab_mulag
$ENVS
/isaac-sim/python.sh PPO_AAM_FD/scripts/train_residual_adapter.py \
  logs/rsl_rl/ppo_id_aam_fd/distilled_v1/phase1_init_70.pt \
  --num_envs 256 --iterations 400 --history 20 --latent_dim 8 --alpha 0.5 \
  --lr 3e-4 --seed $SEED --no_history \
  --lateral_tracking_std_m $LSTD --lateral_tracking_weight $LW \
  --log_dir logs/rsl_rl/ppo_id_aam_fd/rebal${SEED}_nohistory \
  --headless
" < /dev/null > outputs/rebal${SEED}_nohistory.log 2>&1 &
  echo "LAUNCHED rebal${SEED}_nohistory (pid $!)  $(date -u +%H:%M)"
  sleep 20
done
wait
for SEED in $SEEDS; do
  if [ -s logs/rsl_rl/ppo_id_aam_fd/rebal${SEED}_nohistory/model_399.pt ]; then
    echo "TRAINED_rebal${SEED}_nohistory"
  else
    echo "FAILED_rebal${SEED}_nohistory -- see outputs/rebal${SEED}_nohistory.log"
  fi
done
echo NOHISTORY_SEEDS_DONE
