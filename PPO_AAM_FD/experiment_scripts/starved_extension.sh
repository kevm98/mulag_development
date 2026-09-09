#!/usr/bin/env bash
# Extension seeds for the starved-encoder campaign, per
# reports/PREREG_STARVED_EXTENSION_2026-09-05.md. Seeds fixed there in advance.
#
# Phase 0 (gate) and the base arm are NOT repeated: they are identical to the
# first campaign and already scored. Training config is byte-identical to the
# original driver, with the control-arm scoring flag now paired correctly.
set -u
export IMAGE=isaac-lab:2.3.2-rl052
LSTD=0.09
LW=4.0
SEEDS=${SEEDS:-"1 2 3 4 5"}
MESHES=${MESHES:-"2718 3141 8191"}
BASE=logs/rsl_rl/ppo_id_aam_fd/distilled_v1/phase1_init_70.pt
cd /home/kevinmurera

ENVS="
export MULAG_DIRECT_VALVE=1
export MULAG_RMA_CONTEXT=1
export MULAG_PAYLOAD_RANGE=1.0,5.0
export MULAG_PAYLOAD_RESAMPLE=1
"
run() { ~/bin/isaaclab-run bash -lc "$1" 2>/dev/null || true; }
mkdir -p /home/kevinmurera/projects/rrlab_mulag/outputs/starved

for SEED in $SEEDS; do
  for ARM in adaptive control; do
    # training uses --no_history, scoring uses --residual_no_history.
    if [ "$ARM" = control ]; then
      EXTRA="--no_history"; SCORE_EXTRA="--residual_no_history"
    else
      EXTRA=""; SCORE_EXTRA=""
    fi
    TAG=starved${SEED}_${ARM}
    ~/bin/isaaclab-run bash -lc "
set -u
cd /workspace/projects/rrlab_mulag
$ENVS
/isaac-sim/python.sh PPO_AAM_FD/scripts/train_residual_adapter.py $BASE \
  --num_envs 256 --iterations 400 --history 20 --latent_dim 8 --alpha 0.5 \
  --lr 3e-4 --seed $SEED --history_cols plant $EXTRA \
  --lateral_tracking_std_m $LSTD --lateral_tracking_weight $LW \
  --log_dir logs/rsl_rl/ppo_id_aam_fd/${TAG} \
  --headless 2>&1 | tee outputs/starved/${TAG}.log | tail -2
" || true
    echo "TRAINED_${TAG}"
    for MESH in $MESHES; do
      run "
cd /workspace/projects/rrlab_mulag
$ENVS
/isaac-sim/python.sh PPO_AAM_FD/scripts/evaluate_gates.py $BASE \
  --residual_checkpoint logs/rsl_rl/ppo_id_aam_fd/${TAG}/model_399.pt \
  --residual_alpha 0.5 --residual_history 20 $SCORE_EXTRA \
  --num_envs 176 --steps 900 --meshes $MESH --seed 17 \
  --report outputs/rif_delay/${TAG}_m${MESH}.json 2>&1 | grep -E 'wrote|rror' | tail -2
"
      echo "SCORED_${TAG}_m${MESH}"
    done
  done
  echo "SEED_DONE_${SEED}"
done
echo STARVED_EXTENSION_DONE
