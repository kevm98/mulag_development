#!/bin/bash
# Full AAM SystemFactor pipeline
set -e
cd "$(dirname "$0")/.."
PYTHON=/home/rrlab/miniconda3/envs/env_isaaclab/bin/python

echo "=== AAM SystemFactor: Full Pipeline ==="

# Phase A: Audit
echo -e "\n--- Phase A: Data Audit ---"
bash scripts/run_data_audit.sh

# Phase B: Smoke test
echo -e "\n--- Phase B: Smoke Test (5 epochs) ---"
$PYTHON training/train_aam.py --T 10 --epochs 5 --warmup

echo -e "\n--- Phase C: Full Training ---"
echo "Command: $PYTHON training/train_aam.py --T 10 --epochs 100"
echo "Estimated from smoke: ~X minutes"
echo "Run the above command to start full training."
