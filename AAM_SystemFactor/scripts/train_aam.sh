#!/bin/bash
# Train AAM SystemFactor LSTM
set -e
cd "$(dirname "$0")/.."
PYTHON=/home/rrlab/miniconda3/envs/env_isaaclab/bin/python

T=${1:-10}
EPOCHS=${2:-100}

echo "=== AAM SystemFactor: Training (T=$T, epochs=$EPOCHS) ==="
$PYTHON training/train_aam.py --T $T --epochs $EPOCHS
echo "Training complete. See outputs/"
