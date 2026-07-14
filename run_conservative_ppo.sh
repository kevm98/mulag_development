#!/usr/bin/env bash
set -euo pipefail

source /home/rrlab/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd /home/rrlab/rrlab_mulag

/home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  ID_FD_pipline/residual_policy/scripts/train.py \
  --headless --num_envs 64 --max_iterations 1000 \
  --curriculum_stage 1 --seed 17
