#!/usr/bin/env bash
# FIX THE TRADE-OFF: rebalance the lateral reward kernel.
#
# Diagnosis (2026-08-30). At the frozen base's operating point the two task
# rewards are Gaussian kernels with wildly different gradients:
#
#   cut     w=8.0  sigma=0.030 m   error  7.96 mm  ->  dr/de = 132
#   lateral w=3.0  sigma=0.300 m   error 61.16 mm  ->  dr/de =   3.9     (34x weaker)
#
# The lateral kernel is ~10x wider than the error it is meant to correct, so it
# is nearly flat there and the residual has almost no incentive to protect
# lateral. It optimises cut height and lets lateral drift -- which is why the
# trade-off direction is SEED DEPENDENT, and why seed 42 swapped which metric it
# won when trained from 400 to 1200 iterations. Under-specified objective, not
# instability.
#
# Fix: tighten the lateral kernel to the scale of the actual error. A Gaussian's
# gradient is maximal at e = sigma/sqrt(2), i.e. sigma = 0.086 m here. Using
# sigma=0.09, w=4.0 brings the ratio from 34x to ~3.5x.
#
# ONLY the adapter's training signal changes. The evaluation metrics (mm of cut
# and lateral error), the meshes, and the frozen base are all untouched, so
# these arms are directly comparable to the existing 400-iteration arms.
#
# Arms: seed 101 (clearest trade-off: +23.1% cut, -61.2% lateral) and seed 17
# (currently passes both) -- does the fix convert a failing seed without
# breaking a passing one.
set -u
export IMAGE=isaac-lab:2.3.2-rl052
LSTD=0.09
LW=4.0
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

# ---- SMOKE TEST: prove the override actually reaches the reward -----------
echo "=== smoke: verifying the override reaches the reward kernel ==="
~/bin/isaaclab-run bash -lc "
set -u
cd /workspace/projects/rrlab_mulag
$ENVS
/isaac-sim/python.sh PPO_AAM_FD/scripts/train_residual_adapter.py \
  logs/rsl_rl/ppo_id_aam_fd/distilled_v1/phase1_init_70.pt \
  --num_envs 16 --iterations 2 --history 20 --latent_dim 8 --alpha 0.5 \
  --lr 3e-4 --seed 17 \
  --lateral_tracking_std_m $LSTD --lateral_tracking_weight $LW \
  --log_dir logs/rsl_rl/ppo_id_aam_fd/rebal_smoke \
  --headless 2>&1 | grep -E 'reward kernel|Error|error' | head -5
" 2>&1 | tee /home/kevinmurera/isaaclab-cloud/projects/rrlab_mulag/outputs/rebal_smoke.log

if ! grep -q "sigma=$LSTD" /home/kevinmurera/isaaclab-cloud/projects/rrlab_mulag/outputs/rebal_smoke.log; then
  echo "SMOKE_FAILED: override did not reach the reward kernel -- ABORTING"
  echo REBAL_ABORTED
  exit 1
fi
echo "SMOKE_OK: reward kernel carries the override"

# ---- ARMS -----------------------------------------------------------------
for SEED in 101 17; do
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
echo REBAL_DONE
