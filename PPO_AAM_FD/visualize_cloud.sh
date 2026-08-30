#!/usr/bin/env bash
# Record an mp4 of a policy driving the machine, on the headless cloud box.
#
# `visualize.sh` opens a live Isaac window and assumes the Legion machine's conda
# python. This box has NO DISPLAY and runs everything through docker, so the live
# viewer cannot open here. Recording is the headless equivalent and produces a
# file you can scrub through.
#
# Usage:
#   PPO_AAM_FD/visualize_cloud.sh adaptive     # frozen base + online residual
#   PPO_AAM_FD/visualize_cloud.sh base         # frozen base alone
#   PPO_AAM_FD/visualize_cloud.sh control      # residual with history zeroed
#   PPO_AAM_FD/visualize_cloud.sh nominal      # adaptive, no plant change
#
# Env overrides: MESH (default 1337), ENVS (1), STEPS (900), FRAMES (600).
#
# The three policy arms run on the SAME plant -- cutter mass flips 1.0x <-> 3.0x
# mid-episode at a jittered instant -- so playing them side by side shows what
# the online latent is actually buying.
set -euo pipefail

ARM=${1:-adaptive}
MESH=${MESH:-1337}
ENVS=${ENVS:-1}
STEPS=${STEPS:-900}
FRAMES=${FRAMES:-600}
export IMAGE=${IMAGE:-isaac-lab:2.3.2-rl052}

BASE=logs/rsl_rl/ppo_id_aam_fd/distilled_v1/phase1_init_70.pt
HIST=logs/rsl_rl/ppo_id_aam_fd/residual_history/model_399.pt
NOHI=logs/rsl_rl/ppo_id_aam_fd/residual_nohistory/model_399.pt

case "$ARM" in
  adaptive) CK=$HIST; ALPHA=0.5; EXTRA=""; PLANT=flip ;;
  base)     CK=$HIST; ALPHA=0.0; EXTRA=""; PLANT=flip ;;
  control)  CK=$NOHI; ALPHA=0.5; EXTRA="--residual_no_history"; PLANT=flip ;;
  nominal)  CK=$HIST; ALPHA=0.5; EXTRA=""; PLANT=none ;;
  *) echo "unknown arm '$ARM' (adaptive|base|control|nominal)" >&2; exit 1 ;;
esac

if [ "$PLANT" = "flip" ]; then
  PLANT_ENV='
export MULAG_PAYLOAD_RANGE=1.0,1.0
export MULAG_MASS_STEP_TIME_S=20.0
export MULAG_MASS_STEP_SCALE=3.0
export MULAG_MASS_STEP_JITTER_S=8.0
export MULAG_MASS_BIMODAL=1.0,3.0'
else
  PLANT_ENV=''
fi

echo "[VISUALIZE] arm=$ARM  plant=$PLANT  mesh=$MESH  envs=$ENVS  frames=$FRAMES"
echo "[VISUALIZE] PPO -> valve -> FD -> PhysX (ID bypassed), residual alpha=$ALPHA"

cd /home/kevinmurera
~/bin/isaaclab-run bash -lc "
set -u
cd /workspace/projects/rrlab_mulag
export MULAG_DIRECT_VALVE=1
export MULAG_RMA_CONTEXT=1${PLANT_ENV}
/isaac-sim/python.sh PPO_AAM_FD/scripts/evaluate_gates.py $BASE \
  --residual_checkpoint $CK --residual_alpha $ALPHA --residual_history 20 $EXTRA \
  --num_envs $ENVS --steps $STEPS --meshes $MESH --seed 17 \
  --video --video_length $FRAMES --video_dir outputs/videos/${ARM} \
  --enable_cameras --headless \
  --natural-visuals \
  --report outputs/rif_delay/viz_${ARM}_m${MESH}.json 2>&1 \
  | grep -viE 'warning|deprecat' | tail -25
"
echo "[VISUALIZE] mp4 under outputs/videos/${ARM}/mesh${MESH}/"
