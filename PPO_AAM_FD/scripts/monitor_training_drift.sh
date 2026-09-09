#!/usr/bin/env bash
set -euo pipefail

ROOT=$HOME/rrlab_mulag
PY=$HOME/miniconda3/envs/env_isaaclab/bin/python

cd "$ROOT"
exec "$PY" PPO_ROM/scripts/monitor_training_drift.py "$@"
