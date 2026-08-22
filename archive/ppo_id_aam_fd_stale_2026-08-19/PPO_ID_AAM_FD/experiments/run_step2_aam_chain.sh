#!/usr/bin/env bash
# Step 2: does the AAM actually estimate the NEW latent from history?
#
# Turning on --factor-recon-weight changed the TARGET only. The AAM still has
# to infer that latent from [q, qdot, dP, previous_valve]. This chain answers
# the gate question:
#
#     does z_hat track the mass component, or only seal?
#
# CRITICAL: z_target is baked into the AAM dataset by build_stage31_aam_dataset
# (it runs the frozen SD encoder on each run's true factors and repeats the
# result across every window). Retraining the AAM WITHOUT regenerating those
# labels silently trains against the OLD mass-blind targets and proves nothing.
#
# Safety: the builder does `shutil.rmtree(out)` and its --out DEFAULTS to a path
# inside the frozen dataset. This script always passes an explicit NEW --out and
# never --force over an existing directory.
#
# CPU only. Usage:  run_step2_aam_chain.sh <weight> <seed>
set -euo pipefail

W=${1:?usage: run_step2_aam_chain.sh <factor_recon_weight> [seed]}
S=${2:-20260807}

ROOT_DIR=/home/rrlab/rrlab_mulag
cd "$ROOT_DIR"
DATA=AAM_module/datasets/rom_deployment_stage31_v1_frozen
SRC=outputs/factor_recon_seeds/w${W}_s${S}
TAG=recon_w${W}
FROZEN=sd_model/stage31_frozen/z2_rom_${TAG}
AAMSET=$DATA/aam/dataset_multilag_${TAG}
AAMOUT=outputs/stage31_rom/aam_cnn_${TAG}_L20_d64

PY=/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
[ -x "$PY" ] || PY=python3

test -f "$SRC/encoder.pt" || { echo "missing trained SD: $SRC" >&2; exit 2; }
for p in "$FROZEN" "$AAMSET" "$AAMOUT"; do
  if [ -e "$p" ]; then
    echo "REFUSING: $p already exists. Remove it deliberately or pick another weight." >&2
    exit 3
  fi
done

echo "=== 1/3 freeze SD (w=$W seed=$S) -> $FROZEN"
"$PY" AAM_module/tools/freeze_stage31_sd.py \
  --src "$SRC" --out "$FROZEN" --dataset-root "$DATA"

echo "=== 2/3 regenerate z_target labels -> $AAMSET"
# explicit --out to a NEW path; no --force, so an existing dir aborts
"$PY" AAM_module/tools/build_stage31_aam_dataset.py \
  --root "$DATA" \
  --sd-model "$FROZEN" \
  --out "$AAMSET" \
  --lags 20

echo "=== 3/3 retrain AAM on the new labels -> $AAMOUT"
# hyperparameters copied from the deployed AAM (outputs/stage31_rom/aam_cnn/L20_d64/metrics.json)
"$PY" AAM_module/tools/train_stage31_aam.py \
  --root "$DATA" \
  --aam-root "$AAMSET" \
  --lag 20 \
  --architecture cnn \
  --hidden 64 \
  --depth 4 \
  --dropout 0.05 \
  --temporal-consistency-weight 0.1 \
  --out "$AAMOUT"

echo "=== DONE ==="
echo "frozen SD : $FROZEN"
echo "labels    : $AAMSET"
echo "AAM       : $AAMOUT"
echo
echo "Next: decode the true factors from the AAM's predicted z_hat on held-out"
echo "runs, carrying seal as the positive control, to answer whether z_hat"
echo "tracks mass or only seal."
