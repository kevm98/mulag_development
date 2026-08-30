#!/usr/bin/env bash
# Score every arm through the VALIDATED gate path.
#
# Validation: --residual_alpha 0.0 reproduced the frozen base at cut 8.873 mm
# against the 7.572 mm reference (1.17x, i.e. plant-draw noise). The standalone
# evaluator this replaces was 29x off.
#
# Three arms, one fresh process each per mesh:
#   base       alpha=0            -- frozen policy, residual contributes nothing
#   adaptive   alpha=0.5, history -- can sense the mid-episode flip
#   control    alpha=0.5, zeroed  -- same weights and capacity, cannot sense
#
# The claim (RMA's, and the one that matters) is adaptive < control AND
# adaptive < base on cut-height error. A result where adaptive ~ base would only
# show the latent keeps the residual safe, which is not the claim.
set -u
export IMAGE=isaac-lab:2.3.2-rl052
cd /home/kevinmurera
H=logs/rsl_rl/ppo_id_aam_fd/residual_history/model_399.pt
N=logs/rsl_rl/ppo_id_aam_fd/residual_nohistory/model_399.pt

score () {  # tag ckpt alpha extra
  local TAG=$1 CK=$2 AL=$3 EXTRA=${4:-}
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
  --residual_checkpoint $CK --residual_alpha $AL --residual_history 20 $EXTRA \
  --num_envs 176 --steps 900 --meshes $M --seed 17 \
  --report outputs/rif_delay/g_${TAG}_m${M}.json 2>&1 \
  | grep -E 'wrote|rror' | tail -2
"
    echo "G_${TAG}_M${M}_DONE"
  done
}
score base     "$H" 0.0
score adaptive "$H" 0.5
score control  "$N" 0.5 --residual_no_history
echo SCORE_ARMS_DONE
