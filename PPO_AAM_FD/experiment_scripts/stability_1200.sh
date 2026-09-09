#!/usr/bin/env bash
# STABILITY TEST: does "both metrics improve" survive a longer budget?
#
# Under the OLD reward (34x gradient imbalance toward cut height) longer training
# DESTROYED the result: seed 23 passed both metrics at 400 iterations and then
# collapsed to -60.7% lateral at 1200, because more optimisation meant more
# exploitation of the cheap axis. Seed 42 swapped which metric it won. That is
# the signature of a mis-specified objective.
#
# If the rebalance (lateral sigma 0.30 -> 0.09, weight 3.0 -> 4.0, ratio 34x ->
# 3.5x) actually fixed the specification, the rebalanced arm should STILL pass
# both metrics at 1200 -- exactly where the old objective fell apart.
#
# Secondary motive: the cycle-aligned curve was still rising at 400
# (final equal-window transition +1.357 peak / +0.769 mean, lateral falling
# 218 -> 163 -> 112 mm), so 400 was not converged.
#
# Read: PASS at 1200 => the fix removed the pathology, not just moved it.
#       FAIL at 1200 => the rebalance only delayed the collapse, and 400 was luck.
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
~/bin/isaaclab-run bash -lc "
set -u
cd /workspace/projects/rrlab_mulag
$ENVS
/isaac-sim/python.sh PPO_AAM_FD/scripts/train_residual_adapter.py \
  logs/rsl_rl/ppo_id_aam_fd/distilled_v1/phase1_init_70.pt \
  --resume_residual logs/rsl_rl/ppo_id_aam_fd/rebal17_history/model_399.pt \
  --num_envs 256 --iterations 800 --history 20 --latent_dim 8 --alpha 0.5 \
  --lr 3e-4 --seed 17 \
  --lateral_tracking_std_m 0.09 --lateral_tracking_weight 4.0 \
  --log_dir logs/rsl_rl/ppo_id_aam_fd/rebal17_long_history \
  --headless 2>&1 | tee outputs/rebal17_long.log | tail -2
" || true
echo "TRAINED_rebal17_long"
for M in 1337 17 4242; do
  ~/bin/isaaclab-run bash -lc "
cd /workspace/projects/rrlab_mulag
$ENVS
/isaac-sim/python.sh PPO_AAM_FD/scripts/evaluate_gates.py \
  logs/rsl_rl/ppo_id_aam_fd/distilled_v1/phase1_init_70.pt \
  --residual_checkpoint logs/rsl_rl/ppo_id_aam_fd/rebal17_long_history/model_799.pt \
  --residual_alpha 0.5 --residual_history 20 \
  --num_envs 176 --steps 900 --meshes $M --seed 17 \
  --report outputs/rif_delay/rebal17_long_history_m${M}.json 2>&1 | grep -E 'wrote|rror' | tail -2
" 2>/dev/null || true
  echo "SCORED_rebal17_long_m${M}"
done
echo "ARM_DONE_rebal17_long"
