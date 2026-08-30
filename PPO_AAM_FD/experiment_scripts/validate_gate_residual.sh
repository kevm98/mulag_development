#!/usr/bin/env bash
# Validate the residual path INSIDE evaluate_gates against the known reference.
#
# --residual_alpha 0.0 makes the residual contribute exactly nothing, so this run
# IS the frozen base and must reproduce cut 7.572 mm / lateral 61.130 mm on this
# plant. Anything else means the splice is wrong and no arm number may be read.
#
# This check is run BEFORE any arm, because a standalone evaluator was trusted
# without it and produced a frozen base 29x off the reference -- twice.
set -u
export IMAGE=isaac-lab:2.3.2-rl052
cd /home/kevinmurera
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
  --residual_checkpoint logs/rsl_rl/ppo_id_aam_fd/residual_history/model_399.pt \
  --residual_alpha 0.0 --residual_history 20 \
  --num_envs 176 --steps 900 --meshes 1337 --seed 17 \
  --report outputs/rif_delay/gateval_alpha0.json 2>&1 \
  | grep -E 'residual policy|wrote|rror|Traceback' | tail -5
"
echo GATEVAL_DONE
