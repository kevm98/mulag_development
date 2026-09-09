#!/usr/bin/env bash
# Rebuild the AAM SD pipeline after the Stage 22 coverage expansion.
#
# Assumes:
#   - existing collected runs 0-27 are in  AAM_module/datasets/master_v1/runs
#   - new runs 28+ were collected into      /tmp/master_v2/runs   (see below)
#   - the combined manifest is             /tmp/expanded_manifest.csv
#
# Steps: merge -> freeze -> splits -> timebase -> SD dataset -> train
# Stage 19 baselines -> train Stage 21 SD_z (encoder + direct) -> Stage 22 gate.
#
# Usage:
#   bash AAM_module/tools/rebuild_after_collection.sh [--skip-train] [--gate-only]
set -euo pipefail

AAM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=$HOME/miniconda3/envs/env_isaaclab/bin/python

SRC_COLLECT=/tmp/master_v2
SRC_EXISTING="$AAM_ROOT/datasets/master_v1"
MANIFEST=/tmp/expanded_manifest.csv
MERGED=/tmp/master_v2_merged
FROZEN="$AAM_ROOT/datasets/master_v2_frozen"

SKIP_TRAIN=0
for arg in "$@"; do
  case "$arg" in
    --skip-train) SKIP_TRAIN=1 ;;
    --gate-only) SKIP_TRAIN=1 ;;
  esac
done

echo "== merge runs 0-27 (existing) + 28+ (new) into $MERGED =="
rm -rf "$MERGED"
mkdir -p "$MERGED/runs"
cp -r "$SRC_EXISTING/runs/." "$MERGED/runs/"
cp -r "$SRC_COLLECT/runs/." "$MERGED/runs/"
cp "$MANIFEST" "$MERGED/manifest.csv"
n_runs=$(ls "$MERGED/runs" | wc -l)
echo "merged runs: $n_runs (expect 89)"

echo "== freeze -> $FROZEN =="
rm -rf "$FROZEN"
$PY "$AAM_ROOT/tools/freeze_dataset.py" --src "$MERGED" --out "$FROZEN" --readonly

echo "== splits =="
$PY "$AAM_ROOT/tools/make_splits.py" --root "$FROZEN"

echo "== timebase samples =="
$PY "$AAM_ROOT/tools/build_timebase_samples.py" --root "$FROZEN" --force

echo "== SD dataset =="
$PY "$AAM_ROOT/../sd_model/dataset_tools/build_sd_dataset.py" --root "$FROZEN" --force

if [ "$SKIP_TRAIN" = "1" ]; then
  echo "skip-train set; pipeline ready at $FROZEN"
  exit 0
fi

echo "== Stage 19 no-latent baselines (full-set retrain) =="
for mode in su su_delta squ hist3 hist6; do
  echo "  -- $mode"
  $PY "$AAM_ROOT/../sd_model/training/train_sd_baseline.py" --root "$FROZEN" \
      --input "$mode" --epochs 150 --out "$AAM_ROOT/outputs/sd_baseline_v2/$mode"
done

echo "== Stage 21 SD_z (MLP-encoder latent, dz=2, full curriculum) =="
$PY "$AAM_ROOT/../sd_model/training/train_sd_z.py" --root "$FROZEN" --latent-dim 2 \
    --unroll 10 --l1-weight 1.0 --rollout-weight 1.0 --rollout-ramp-epochs 60 \
    --epochs 300 --patience 30 --seed 20260808 \
    --out "$AAM_ROOT/outputs/sd_z_stage21_v2/dz2_ramp60"

echo "== Stage 21 SD_z (direct factors, dz=6) =="
$PY "$AAM_ROOT/../sd_model/training/train_sd_z.py" --root "$FROZEN" --latent-dim 2 \
    --latent-source direct --unroll 10 --l1-weight 1.0 --rollout-weight 1.0 \
    --rollout-ramp-epochs 60 --epochs 300 --patience 30 --seed 20260808 \
    --out "$AAM_ROOT/outputs/sd_z_stage21_v2/direct_dz6"

echo "== Stage 22 gate ablation (z_true vs z_zero vs z_shuffled) =="
for model in dz2_ramp60 direct_dz6; do
  for mode in true zero shuffled; do
    $PY "$AAM_ROOT/../sd_model/evaluation/evaluate_sd_z_rollout.py" --root "$FROZEN" \
        --model-dir "$AAM_ROOT/outputs/sd_z_stage21_v2/$model" \
        --latent-mode "$mode" --horizons 1 5 10 20 --device cuda \
        --out "$AAM_ROOT/outputs/sd_z_stage21_v2/$model/rollout_latent_$mode.json"
  done
done

echo "== done. Gate files under outputs/sd_z_stage21_v2/ =="
