#!/usr/bin/env bash
# alpha = 1.0, to establish whether performance is MONOTONE in residual authority.
#
# Measured so far on mesh 1337 (cut-height error against a 7.246 mm base):
#   alpha 0.15 -> 14.138 mm   WORSE than doing nothing
#   alpha 0.50 ->  3.427 mm   base is +88.7% worse
#
# A 3x mass change needs a correction commensurate with it: 0.15 caps the
# correction below what the disturbance demands while still being large enough to
# perturb. If 1.0 continues the trend, the claim is "the residual needs authority
# proportional to the disturbance" -- a physical statement -- rather than a lucky
# hyperparameter. If 1.0 is worse than 0.5, there is an optimum and that is worth
# reporting too.
set -u
cd /tmp/rrlab_scratch
TAG=a10_history   ALPHA=1.0 bash residual_run.sh
TAG=a10_nohistory ALPHA=1.0 ARG=--no_history bash residual_run.sh
export IMAGE=isaac-lab:2.3.2-rl052
cd /home/kevinmurera
for ARM in history nohistory; do
  EXTRA=""; [ "$ARM" = nohistory ] && EXTRA="--residual_no_history"
  for M in 1337 17 4242; do
    ~/bin/isaaclab-run bash -lc "
cd /workspace/projects/rrlab_mulag
export MULAG_DIRECT_VALVE=1
export MULAG_RMA_CONTEXT=1
export MULAG_PAYLOAD_RANGE=1.0,1.0
export MULAG_MASS_STEP_TIME_S=20.0
export MULAG_MASS_STEP_SCALE=3.0
export MULAG_MASS_STEP_JITTER_S=8.0
export MULAG_MASS_BIMODAL=1.0,3.0
/isaac-sim/python.sh PPO_AAM_FD/scripts/evaluate_gates.py \
  logs/rsl_rl/ppo_id_aam_fd/distilled_v1/phase1_init_70.pt \
  --residual_checkpoint logs/rsl_rl/ppo_id_aam_fd/residual_a10_${ARM}/model_399.pt \
  --residual_alpha 1.0 --residual_history 20 $EXTRA \
  --num_envs 176 --steps 900 --meshes $M --seed 17 \
  --report outputs/rif_delay/a10_${ARM}_m${M}.json 2>&1 | grep -E 'wrote|rror' | tail -2
" 2>/dev/null || true
    echo "A10_${ARM}_M${M}_DONE"
  done
done
echo ALPHA10_DONE
