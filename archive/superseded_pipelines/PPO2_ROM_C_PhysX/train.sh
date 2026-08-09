#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

exec /home/rrlab/miniconda3/envs/env_isaaclab/bin/python \
  PPO2_ROM_C_PhysX/scripts/train.py \
  --headless \
  --device cuda:0 \
  --num_envs 128 \
  --seed 17 \
  --balanced_shape_rounds 50 \
  --iterations_per_shape 2 \
  "$@"
