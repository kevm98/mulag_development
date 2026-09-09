#!/usr/bin/env bash
set -euo pipefail

# Open-loop excitation -> joint AAM+SD -> matched comparison against a no-latent
# baseline trained on the SAME data.
#
# The question this answers
# -------------------------
# On the DLS+PRBS collection, neither the privileged latent nor the end-to-end
# inferred one beats a plain no-latent SD at free-running rollout:
#
#     SD_0 (no latent)      0.17278      <- best
#     SD+AAM e2e (z_hat)    0.17596      +1.8 %
#     SD_z (privileged)     0.18515      +7.2 %
#
# One explanation is that the ARCHITECTURE has nothing to add. The other is that
# the DATA does not separate plants: a task controller computes its command from
# the state, so command and response are nearly redundant and there is little
# for a plant latent to explain. The open-loop collection removes the controller
# entirely and is the direct test of the second explanation.
#
# Why SD_0 is retrained here
# --------------------------
# Comparing an e2e model trained on the open-loop data against the SD_0 number
# above would confound the data change with the model change. The baseline is
# therefore retrained on the identical dataset, split and objective, and only
# the latent differs.
#
# Usage:  experiments/E4/code/run_openloop_e2e.sh [raw_dataset_dir]

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
py="$HOME/miniconda3/envs/env_isaaclab/bin/python"
aam="$repo/AAM_module"

raw="${1:-$aam/datasets/rom_openloop_v1}"
frozen="${raw}_frozen"
frozen_sd="$repo/sd_model/stage31_frozen/z2_rom"
out="$repo/outputs/stage31_openloop"

die() { echo "openloop: $*" >&2; exit 2; }
[[ -d "$raw" ]] || die "missing collection: $raw"

echo "== 1/6 validate the collection =="
"$py" "$aam/tools/validate_verge_classical.py" \
    --dataset "$raw" --out "$raw/collection_gate.json"
"$py" - "$raw/collection_gate.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
if d.get("blocking_issues"):
    raise SystemExit(f"collection gate FAILED: {d['blocking_issues']}")
if not d.get("passed_with_sample_mask"):
    raise SystemExit("collection gate FAILED and is not maskable")
n = len(d.get("maskable_issues", []))
print(f"  gate: 0 blocking issues over {len(d['runs'])} runs; "
      f"{n} saturation/limit breaches handled by the training sample mask")
PY

echo "== 2/6 freeze and split =="
[[ -d "$frozen" ]] || "$py" "$aam/tools/freeze_dataset.py" --src "$raw" --out "$frozen" --readonly
"$py" "$aam/tools/make_splits.py" --root "$frozen"

echo "== 3/6 build SD samples and AAM rows =="
"$py" "$aam/tools/build_stage31_rom_datasets.py" \
    --root "$frozen" --sd-out "$frozen/sd/dataset" \
    --aam-out "$frozen/aam/rom_rows" --lags 5 10 20 40 --force

echo "== 4/6 build the 18-feature AAM windows =="
# The e2e trainer reads aam/dataset_multilag, which build_stage31_rom_datasets
# does NOT produce: its sd/dataset histories carry 20 features per row, not the
# 18 of the deployed AAM contract. --sd-model is required by the tool because it
# also writes z_target; that label is unused by the e2e route but stays valid
# here, since z = mu(e) is a function of the per-run factors and this collection
# reuses the same manifest.
"$py" "$aam/tools/build_stage31_aam_dataset.py" \
    --root "$frozen" --sd-model "$frozen_sd" \
    --rows "$frozen/aam/rom_rows" --out "$frozen/aam/dataset_multilag" \
    --lags 20 --force

mkdir -p "$out"

echo "== 5/6 no-latent baseline on THIS data =="
"$py" "$repo/sd_model/training/train_sd_z.py" \
    --root "$frozen" --history-lag 6 --belief-dim 16 \
    --unroll 5 --l1-weight 1.0 --rollout-weight 1.0 \
    --state-loss-weight 1.0 --rollout-q-mode hold-start \
    --rollout-ramp-epochs 40 --val-unroll 5 --epochs 200 --patience 25 \
    --seed 20260820 --latent-source none --sample-mask \
    --out "$out/sd_none"

echo "== 6/6 joint AAM+SD, no privileged encoder =="
"$py" "$repo/sd_model/training/train_sd_aam_e2e.py" \
    --root "$frozen" --lag 20 --latent-dim 2 --epochs 60 --device cpu \
    --unroll 5 --rollout-weight 1.0 --l1-weight 1.0 --tf-floor 0.2 \
    --rollout-ramp-epochs 20 --out "$out/sd_aam_e2e_L20_d2_unroll5"

echo
echo "== matched comparison (free-running accumulated state error, H=5) =="
"$py" - "$out" <<'PY'
import json, sys
from pathlib import Path
o = Path(sys.argv[1])
rows = [("SD_0  (no latent, open loop)", o / "sd_none/metrics.json", "best_val_loss"),
        ("SD+AAM e2e (open loop)", o / "sd_aam_e2e_L20_d2_unroll5/metrics.json",
         "val_rollout_state_mse")]
base = None
for name, p, k in rows:
    if not p.is_file():
        print(f"  {name}: MISSING {p}"); continue
    v = json.load(open(p))[k]
    base = v if base is None else base
    print(f"  {name:32s} {v:.5f}   {100 * (v / base - 1):+6.1f}% vs baseline")
print("\n  DLS+PRBS reference: SD_0 0.17278, e2e 0.17596 (+1.8%).")
print("  The latent earns its place only if e2e now beats its own baseline.")
PY
