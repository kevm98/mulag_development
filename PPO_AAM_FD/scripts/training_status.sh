#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/rrlab/rrlab_mulag
PY=/home/rrlab/miniconda3/envs/env_isaaclab/bin/python

cd "$ROOT"
exec "$PY" PPO_ROM/scripts/training_status.py "$@"
