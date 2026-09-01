#!/usr/bin/env bash
# EXPERIMENT 2: the history-zeroed control.
#
# Same frozen base, same architecture, same capacity, same budget, same seed,
# same rebalanced reward. The ONLY difference is that the history window is
# zeroed, so the residual has capacity but no online estimate of the plant.
# This is the ablation RMA-style papers report, and it is what separates
# "online adaptation helps" from "a bigger network helps".
#
# NOTE the flag asymmetry, which has bitten this repo before:
#   training  uses --no_history            (train_residual_adapter.py)
#   scoring   uses --residual_no_history   (evaluate_gates.py)
set -u
export IMAGE=isaac-lab:2.3.2-rl052
SEED=${SEED:-17}
LSTD=${LSTD:-0.09}
LW=${LW:-4.0}
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
  --headless 2>&1 | tee outputs/rebal${SEED}_nohistory.log | tail -2
" || true
echo "TRAINED_rebal${SEED}_nohistory"
for M in 1337 17 4242; do
  ~/bin/isaaclab-run bash -lc "
cd /workspace/projects/rrlab_mulag
$ENVS
/isaac-sim/python.sh PPO_AAM_FD/scripts/evaluate_gates.py \
  logs/rsl_rl/ppo_id_aam_fd/distilled_v1/phase1_init_70.pt \
  --residual_checkpoint logs/rsl_rl/ppo_id_aam_fd/rebal${SEED}_nohistory/model_399.pt \
  --residual_alpha 0.5 --residual_history 20 --residual_no_history \
  --num_envs 176 --steps 900 --meshes $M --seed 17 \
  --report outputs/rif_delay/rebal${SEED}_nohistory_m${M}.json 2>&1 | grep -E 'wrote|rror' | tail -2
" 2>/dev/null || true
  echo "SCORED_rebal${SEED}_nohistory_m${M}"
done
echo "ARM_DONE_rebal${SEED}_nohistory"
