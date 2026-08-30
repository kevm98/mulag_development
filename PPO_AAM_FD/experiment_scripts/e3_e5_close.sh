#!/usr/bin/env bash
# Close the two outstanding items E3 and E5 leave open.
#
# ---------------------------------------------------------------- SHAM (E3)
# E3: "sham is what makes tracking falsifiable". Its schedule fires but the
# plant does NOT change. Without it, E3 would have reported position-along-path
# artefacts as adaptation evidence -- error ramps 39 -> 128 mm along the path in
# the perturbed run and 39 -> 138 mm in the sham.
#
# Here the sham sets both bimodal modes to 1.0, so the flip schedule fires on
# the same jittered instants and writes the same mass. If the adaptive arm still
# beats the base by a similar margin with NO real plant change, the advantage is
# not adaptation to the flip and the headline claim fails.
#
# Expected if the claim is sound: the adaptive advantage SHRINKS markedly under
# sham, because there is nothing to adapt to.
#
# ----------------------------------------------------------- BREAKDOWN (E5)
# E5 concluded a breakdown point "cannot be found by scaling factors" -- true for
# the hydraulic axes, which are inert on FD, and for authority, capped at 62%
# removable. Mass is neither: it reaches PhysX natively via set_masses.
#
# E5's criterion is FUNCTIONAL BREAKDOWN: ground contact < 0.700 or safety
# termination rising. Measured so far on the distilled base: 1.0x -> 0.875,
# 1.5x -> 0.840, 2.0x -> 0.798. This sweep extends past it to locate the knee.
set -u
export IMAGE=isaac-lab:2.3.2-rl052
cd /home/kevinmurera
BASE=logs/rsl_rl/ppo_id_aam_fd/distilled_v1/phase1_init_70.pt
HIST=logs/rsl_rl/ppo_id_aam_fd/residual_history/model_399.pt

# ---- SHAM: schedule fires, mass unchanged --------------------------------
for ARM in base adaptive; do
  AL=0.5; [ "$ARM" = base ] && AL=0.0
  for M in 1337 17 4242; do
    ~/bin/isaaclab-run bash -lc "
cd /workspace/projects/rrlab_mulag
export MULAG_DIRECT_VALVE=1
export MULAG_RMA_CONTEXT=1
export MULAG_PAYLOAD_RANGE=1.0,1.0
export MULAG_MASS_STEP_TIME_S=20.0
export MULAG_MASS_STEP_SCALE=1.0
export MULAG_MASS_STEP_JITTER_S=8.0
export MULAG_MASS_BIMODAL=1.0,1.0
/isaac-sim/python.sh PPO_AAM_FD/scripts/evaluate_gates.py $BASE \
  --residual_checkpoint $HIST --residual_alpha $AL --residual_history 20 \
  --num_envs 176 --steps 900 --meshes $M --seed 17 \
  --report outputs/rif_delay/sham_${ARM}_m${M}.json 2>&1 | grep -E 'wrote|rror' | tail -2
" 2>/dev/null || true
    echo "SHAM_${ARM}_M${M}_DONE"
  done
done
echo SHAM_DONE

# ---- BREAKDOWN: push mass until ground contact < 0.700 -------------------
for LV in 2.5 3.0 4.0 5.0; do
  TAG=$(echo $LV | tr . _)
  ~/bin/isaaclab-run bash -lc "
cd /workspace/projects/rrlab_mulag
export MULAG_DIRECT_VALVE=1
export MULAG_RMA_CONTEXT=1
export MULAG_PAYLOAD_RANGE=$LV,$LV
/isaac-sim/python.sh PPO_AAM_FD/scripts/evaluate_gates.py $BASE \
  --num_envs 176 --steps 900 --meshes 1337 --seed 17 \
  --report outputs/rif_delay/breakdown_${TAG}.json 2>&1 | grep -E 'wrote|rror' | tail -2
" 2>/dev/null || true
  echo "BREAKDOWN_${TAG}_DONE"
done
echo E3_E5_DONE
