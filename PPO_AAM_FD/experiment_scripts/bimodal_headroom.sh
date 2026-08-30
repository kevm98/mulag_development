#!/usr/bin/env bash
# PRECONDITION for the adaptation experiment: does the bimodal mid-episode flip
# actually degrade the FROZEN base policy?
#
# G1: adaptation can only recover headroom a degradation creates. If the base
# policy copes with the flip, there is nothing for an adaptive residual to
# recover and the experiment is dead before it starts. This is the check that
# was missing when earlier ablations were run on a nominal plant and returned
# nulls that meant nothing.
#
# Two fresh processes, same seed / mesh / env count, so env k is the same terrain
# in both and the comparison is paired env-for-env (common random numbers).
set -u
export IMAGE=isaac-lab:2.3.2-rl052
CK=logs/rsl_rl/ppo_id_aam_fd/distilled_v1/phase1_init_70.pt
cd /home/kevinmurera

echo "=== A: nominal (no flip) ==="
~/bin/isaaclab-run bash -lc "
cd /workspace/projects/rrlab_mulag
export MULAG_DIRECT_VALVE=1
export MULAG_RMA_CONTEXT=1
/isaac-sim/python.sh PPO_AAM_FD/scripts/evaluate_gates.py $CK \
  --num_envs 176 --steps 900 --meshes 1337 --seed 17 \
  --report outputs/rif_delay/bimodal_nominal.json 2>&1 | grep -E 'wrote|rror' | tail -2
"
echo BIMODAL_NOMINAL_DONE

echo "=== B: bimodal flip 1.0 <-> 3.0 at 20s +/- 8s ==="
~/bin/isaaclab-run bash -lc "
cd /workspace/projects/rrlab_mulag
export MULAG_DIRECT_VALVE=1
export MULAG_RMA_CONTEXT=1
export MULAG_PAYLOAD_RANGE=1.0,1.0
export MULAG_MASS_STEP_TIME_S=20.0
export MULAG_MASS_STEP_SCALE=3.0
export MULAG_MASS_STEP_JITTER_S=8.0
export MULAG_MASS_BIMODAL=1.0,3.0
/isaac-sim/python.sh PPO_AAM_FD/scripts/evaluate_gates.py $CK \
  --num_envs 176 --steps 900 --meshes 1337 --seed 17 \
  --report outputs/rif_delay/bimodal_flip.json 2>&1 | grep -E 'wrote|rror' | tail -2
"
echo BIMODAL_FLIP_DONE
echo BIMODAL_ALL_DONE
