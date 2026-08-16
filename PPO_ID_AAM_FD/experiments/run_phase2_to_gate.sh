#!/usr/bin/env bash
# Wait for the rom_phase2_v2 collection to finish, then run the whole chain
# through the Phase 3 gate unattended.
#
#   validate -> freeze -> splits -> rom rows -> AAM dataset
#     -> step-observability check
#     -> latent gate (existing AAM)
#     -> retrain AAM on per-window labels
#     -> latent gate (retrained AAM)
#
# Every stage writes to $LOG. The gate is the decision point: it exits non-zero
# on failure, and the script records that rather than pressing on, because a
# latent that has not passed the gate must not reach Phase 4 ablations.
set -uo pipefail

ROOT=/home/rrlab/rrlab_mulag
cd "$ROOT"
PY=/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
[ -x "$PY" ] || PY=python3

RAW=AAM_module/datasets/rom_phase2_v2
FROZ=AAM_module/datasets/rom_phase2_v2_frozen
OUT=outputs/stage31_rom
LOG=$ROOT/PPO_ID_AAM_FD/experiments/results/phase2_to_gate.log
mkdir -p "$(dirname "$LOG")" "$OUT"

say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
run() { say "--- $1"; shift; "$@" >>"$LOG" 2>&1; local rc=$?; say "    rc=$rc"; return $rc; }

: > "$LOG"
say "waiting for the collector to finish"
while pgrep -f collect_verge_classical >/dev/null; do sleep 20; done
N=$(ls "$RAW/runs" 2>/dev/null | wc -l)
say "collector finished with $N/64 runs"
if [ "$N" -lt 64 ]; then
    say "INCOMPLETE COLLECTION ($N/64). Rerun with --resume (see HANDOVER_PHASES.md step 0). Stopping."
    exit 2
fi

rm -rf "$FROZ"
run "validate"      "$PY" AAM_module/tools/validate_verge_classical.py --dataset "$RAW" --out "$RAW/collection_gate.json" || exit 3
run "freeze"        "$PY" AAM_module/tools/freeze_dataset.py --src "$RAW" --out "$FROZ" --readonly || exit 3
run "splits"        "$PY" AAM_module/tools/make_splits.py --root "$FROZ" || exit 3
run "rom rows"      "$PY" AAM_module/tools/build_stage31_rom_datasets.py --root "$FROZ" \
                        --sd-out "$FROZ/sd/dataset" --aam-out "$FROZ/aam/rom_rows" --lags 5 10 20 40 --force || exit 3
run "aam dataset"   "$PY" AAM_module/tools/build_stage31_aam_dataset.py --root "$FROZ" \
                        --sd-model sd_model/stage31_frozen/z2_rom \
                        --out "$FROZ/aam/dataset_multilag" --lags 20 || exit 3

# Data sanity BEFORE the gate: if the step is not observable in the recorded
# response, a tracking failure would be a data result, not a latent result.
run "step observability" "$PY" AAM_module/tools/verify_stepped_collection.py \
                        --dataset "$RAW" --out "$RAW/stepped_collection_check.json"
STEP_RC=$?

run "GATE: existing AAM" "$PY" AAM_module/tools/gate_latent_quality.py --root "$FROZ" \
                        --model-dir "$OUT/aam_cnn/L20_d64" --out "$OUT/gate_phase2_v2_existing.json"
GATE_EXISTING=$?

# Retrain on the per-window labels regardless: it is the comparison Phase 2 is
# for, and it is the fix if the existing AAM falls short.
run "retrain AAM (per-window labels)" "$PY" AAM_module/tools/train_stage31_aam.py \
                        --root "$FROZ" --aam-root "$FROZ/aam/dataset_multilag" \
                        --lag 20 --architecture cnn --hidden 64 --depth 4 --dropout 0.05 \
                        --temporal-consistency-weight 0.1 \
                        --out "$OUT/aam_cnn_perwindow_L20_d64"
TRAIN_RC=$?

GATE_NEW=99
if [ "$TRAIN_RC" -eq 0 ]; then
    run "GATE: retrained AAM" "$PY" AAM_module/tools/gate_latent_quality.py --root "$FROZ" \
                        --model-dir "$OUT/aam_cnn_perwindow_L20_d64" \
                        --out "$OUT/gate_phase2_v2_perwindow.json"
    GATE_NEW=$?
fi

say ""
say "================ SUMMARY ================"
say "step observability : rc=$STEP_RC   (0 = step is observable in control effort)"
say "gate, existing AAM : rc=$GATE_EXISTING   (0 = PASSED)"
say "retrain            : rc=$TRAIN_RC"
say "gate, retrained    : rc=$GATE_NEW   (0 = PASSED, 99 = not run)"
say ""
for f in "$OUT/gate_phase2_v2_existing.json" "$OUT/gate_phase2_v2_perwindow.json"; do
    [ -f "$f" ] || continue
    say "--- $f"
    "$PY" - "$f" <<'PYEOF' | tee -a "$LOG"
import json, sys
d = json.load(open(sys.argv[1]))
sep = d.get("separation", {}).get("r2_per_factor", {})
tr = d.get("tracking", {})
print("  passed:", d.get("passed"))
if sep:
    print("  separation seal (control): %+.3f" % sep.get("seal", float("nan")))
if tr.get("n_stepped_scored"):
    print("  tracking shift/noise: %.2f (need 3.0), %.0f%% of runs, lag %s windows"
          % (tr["median_shift_over_noise"],
             100 * tr["frac_runs_above_threshold"],
             tr["median_lag_windows"]))
for p in d.get("problems", []):
    print("  PROBLEM:", p)
PYEOF
done
say ""
if [ "$GATE_EXISTING" -eq 0 ] || [ "$GATE_NEW" -eq 0 ]; then
    say "PHASE 3 PASSED -> Phase 4 ablations are unblocked."
    say "Use whichever model passed as --model-dir; see HANDOVER_PHASES.md step 4."
else
    say "PHASE 3 NOT PASSED. Do NOT run Phase 4 ablations on this latent."
    say "Triage in order: realised deltas -> label flags -> then treat as a"
    say "learning problem. See HANDOVER_PHASES.md step 2."
fi
say "done"
