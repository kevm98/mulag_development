#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/rrlab/rrlab_mulag"
PYTHON="/home/rrlab/miniconda3/envs/env_isaaclab/bin/python"
WAIT_FOR_PID="${WAIT_FOR_PID:-68216}"
DATASET="$ROOT/ROM-cylinders_pipeline/data/base_rom_v1"
RUN_DIR="$ROOT/ROM-cylinders_pipeline/checkpoints/base_run_001"
REPORT="$ROOT/ROM-cylinders_pipeline/reports/base_offline_validation.json"

while kill -0 "$WAIT_FOR_PID" 2>/dev/null; do
  sleep 30
done

cd "$ROOT"
"$PYTHON" ROM-cylinders_pipeline/scripts/train_base_rom.py \
  "$DATASET" "$RUN_DIR" \
  --config ROM-cylinders_pipeline/configs/base_training.json

"$PYTHON" ROM-cylinders_pipeline/scripts/validate_base_rom.py \
  "$RUN_DIR/best_base_rom.pt" "$DATASET" \
  --device cpu --output "$REPORT" --require-pass
