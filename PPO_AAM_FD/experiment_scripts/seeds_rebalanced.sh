#!/usr/bin/env bash
# SEED REPLICATION under the rebalanced reward.
#
# Seed 17 passes both metrics; seed 101 still trades. The same five seeds used
# for the original replication (17, 23, 42, 101, 7) are run here so the two
# sweeps are directly comparable -- 17 and 101 are already done, this adds the
# remaining three.
#
# Pre-registered bar (reports/PREREG_both_metrics_2026-08-30.md): an arm passes
# only if BOTH cut-height and lateral error improve against the frozen base with
# the paired CI excluding zero; the claim holds if the MEDIAN seed passes.
# Reporting the best seed is not a result.
#
# Everything except the training seed is held fixed: same frozen base, same
# hyperparameters, same rebalanced kernel, same 400-iteration protocol budget,
# same evaluation plant, meshes and env count.
set -u
export IMAGE=isaac-lab:2.3.2-rl052
LSTD=0.09
LW=4.0
SEEDS=${SEEDS:-"23 42 7"}
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
for SEED in $SEEDS; do
  ~/bin/isaaclab-run bash -lc "
set -u
cd /workspace/projects/rrlab_mulag
$ENVS
/isaac-sim/python.sh PPO_AAM_FD/scripts/train_residual_adapter.py \
  logs/rsl_rl/ppo_id_aam_fd/distilled_v1/phase1_init_70.pt \
  --num_envs 256 --iterations 400 --history 20 --latent_dim 8 --alpha 0.5 \
  --lr 3e-4 --seed $SEED \
  --lateral_tracking_std_m $LSTD --lateral_tracking_weight $LW \
  --log_dir logs/rsl_rl/ppo_id_aam_fd/rebal${SEED}_history \
  --headless 2>&1 | tee outputs/rebal${SEED}.log | tail -2
" || true
  echo "TRAINED_rebal${SEED}"
  for M in 1337 17 4242; do
    ~/bin/isaaclab-run bash -lc "
cd /workspace/projects/rrlab_mulag
$ENVS
/isaac-sim/python.sh PPO_AAM_FD/scripts/evaluate_gates.py \
  logs/rsl_rl/ppo_id_aam_fd/distilled_v1/phase1_init_70.pt \
  --residual_checkpoint logs/rsl_rl/ppo_id_aam_fd/rebal${SEED}_history/model_399.pt \
  --residual_alpha 0.5 --residual_history 20 \
  --num_envs 176 --steps 900 --meshes $M --seed 17 \
  --report outputs/rif_delay/rebal${SEED}_history_m${M}.json 2>&1 | grep -E 'wrote|rror' | tail -2
" 2>/dev/null || true
    echo "SCORED_rebal${SEED}_m${M}"
  done
  echo "ARM_DONE_rebal${SEED}"
done
echo SEEDS_REBALANCED_DONE
