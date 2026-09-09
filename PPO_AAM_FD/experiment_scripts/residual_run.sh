#!/usr/bin/env bash
# Frozen base + history-conditioned residual, on a TIME-VARYING plant.
#
# The plant changes MID-EPISODE (cutter mass steps at t=20 s). That is the whole
# point: E4 measured a per-episode-CONSTANT latent at +13.6% error versus the
# online one, while knowing which plant you are on was worth 0.43% -- 32x
# smaller. A static context is measurably close to worthless here, so a static
# ablation would be testing the wrong axis. With a mid-episode step, a context
# drawn at reset is STALE by construction and only an estimate from recent
# history can track the change. That is what makes this a test of ONLINE
# adaptation rather than of plant identification.
#
# Two arms, identical except for one flag:
#   history live   -- the residual can adapt online
#   --no_history   -- the history is zeroed, so the residual has the same
#                     capacity and the same training but CANNOT adapt
#
# That pair separates "a residual head helps" from "ONLINE ADAPTATION helps",
# which is the claim. Both start from a residual that is exactly zero, so both
# start as the identical frozen policy.
set -u
export IMAGE=isaac-lab:2.3.2-rl052
CK=${CK:-logs/rsl_rl/ppo_id_aam_fd/distilled_v1/phase1_init_70.pt}
ARG=${ARG:-}
TAG=${TAG:-history}
cd /home/kevinmurera
~/bin/isaaclab-run bash -lc "
set -u
cd /workspace/projects/rrlab_mulag
export MULAG_DIRECT_VALVE=1
export MULAG_RMA_CONTEXT=1
export MULAG_PAYLOAD_RANGE=1.0,1.0
export MULAG_MASS_STEP_TIME_S=20.0
export MULAG_MASS_STEP_SCALE=3.0
export MULAG_MASS_STEP_JITTER_S=8.0
export MULAG_MASS_BIMODAL=1.0,3.0
/isaac-sim/python.sh PPO_AAM_FD/scripts/train_residual_adapter.py $CK \
  --num_envs 256 --iterations 400 --history 20 --latent_dim 8 --alpha ${ALPHA:-0.5} \
  --lr 3e-4 --seed 17 $ARG \
  --log_dir logs/rsl_rl/ppo_id_aam_fd/residual_${TAG} \
  --headless 2>&1 | tee /workspace/projects/rrlab_mulag/outputs/residual_${TAG}.log | tail -2
"
echo RESIDUAL_${TAG}_DONE
