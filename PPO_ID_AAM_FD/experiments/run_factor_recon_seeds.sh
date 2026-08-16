#!/usr/bin/env bash
# Step 1: pin the factor-recon weight across seeds.
#
# One seed cannot select a weight -- the held-out mass R2 moved
# 0.68 / 0.35 / 0.44 / 0.61 across weights on 16 test runs, which is sampling
# noise, not a trend. This repeats each weight over 3 seeds.
#
# CPU only, ~2 min per run, 12 runs.
set -euo pipefail

ROOT_DIR=/home/rrlab/rrlab_mulag
cd "$ROOT_DIR"
DATA=AAM_module/datasets/rom_deployment_stage31_v1_frozen
OUT=outputs/factor_recon_seeds
mkdir -p "$OUT"

PY=/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
[ -x "$PY" ] || PY=python3

for W in 0.0 0.5 1.0 2.0; do
  for S in 20260807 20260808 20260809; do
    D="$OUT/w${W}_s${S}"
    if [ -f "$D/metrics.json" ]; then
      echo "SKIP  w=$W seed=$S (already done)"
      continue
    fi
    echo "TRAIN w=$W seed=$S -> $D"
    "$PY" AAM_module/tools/train_sd_z.py \
      --root "$DATA" \
      --latent-dim 2 \
      --latent-source encoder \
      --unroll 5 \
      --val-unroll 5 \
      --rollout-q-mode hold-start \
      --rollout-ramp-epochs 40 \
      --factor-recon-weight "$W" \
      --seed "$S" \
      --out "$D" \
      --log "$D.log" \
      > "$D.stdout" 2>&1 || { echo "FAILED w=$W seed=$S"; tail -5 "$D.stdout"; exit 1; }
  done
done
echo "ALL DONE"
