#!/usr/bin/env bash
set -uo pipefail

ROOT=${ROOT:-$HOME/rrlab_mulag}
PY=${PY:-$HOME/miniconda3/envs/env_isaaclab/bin/python}
RUN=${1:?usage: select_checkpoint.sh <run-dir> <name> [iters...]}
NAME=${2:?}
shift 2
ITERS=${*:-}
OUT=${REPORT_ROOT:-$ROOT/PPO_ROM/reports/final/select_$NAME}
MESHES=${MESHES:-"1337 17 909"}
STEPS=${STEPS:-480}
ENVS=${ENVS:-32}
STAGE=${STAGE:-6}
TASK=${TASK:-}
TASK_ARG=""
[ -n "$TASK" ] && TASK_ARG="--task $TASK"
STALL=${STALL:-300}

if [ -z "$ITERS" ]; then
  ITERS=$(ls -1 "$RUN"/model_*.pt 2>/dev/null \
    | sed 's/.*model_//;s/\.pt//' | sort -n | tr '\n' ' ')
fi

mkdir -p "$OUT"
cd "$ROOT"

for IT in $ITERS; do
  CKPT="$RUN/model_${IT}.pt"
  [ -f "$CKPT" ] || { echo "[SELECT] missing $CKPT"; continue; }
  for MESH in $MESHES; do
    REPORT="$OUT/it${IT}_m${MESH}.json"
    LOG="$OUT/it${IT}_m${MESH}.log"
    if [ -f "$REPORT" ]; then
      if "$PY" - "$REPORT" "$CKPT" <<'PYEOF'
import json, sys
from pathlib import Path
try:
    recorded = json.loads(Path(sys.argv[1]).read_text()).get("checkpoint", "")
    sys.exit(0 if Path(recorded) == Path(sys.argv[2]).resolve() else 1)
except Exception:
    sys.exit(1)
PYEOF
      then
        continue
      fi
      echo "[SELECT] it$IT m$MESH report is from a DIFFERENT checkpoint -- rescoring"
      rm -f "$REPORT"
    fi
    "$PY" PPO_ROM/scripts/evaluate_gates.py "$CKPT" \
        --meshes "$MESH" --steps "$STEPS" --num_envs "$ENVS" --stage "$STAGE" \
        $TASK_ARG --report "$REPORT" --headless > "$LOG" 2>&1 &
    PID=$!
    while kill -0 "$PID" 2>/dev/null; do
      sleep 10
      LAST=$(stat -c %Y "$LOG" 2>/dev/null || echo 0)
      if [ $(( $(date +%s) - LAST )) -gt "$STALL" ]; then
        echo "[SELECT] it$IT m$MESH STALLED -- killing"; kill -9 "$PID" 2>/dev/null; break
      fi
      [ -f "$REPORT" ] && break
    done
    if [ -f "$REPORT" ] && kill -0 "$PID" 2>/dev/null; then
      kill -TERM "$PID" 2>/dev/null
      sleep 1
      kill -9 "$PID" 2>/dev/null
    fi
    wait "$PID" 2>/dev/null
    pkill -9 -f "evaluate_gates.py $CKPT --meshes $MESH " 2>/dev/null
    sleep 3
  done
  echo "[SELECT] iteration $IT done"
done

"$PY" - "$OUT" "$RUN" <<'PYEOF'
import json, re, sys
import os
from collections import defaultdict
from pathlib import Path

from PPO_ROM.gates import GATES, REQUIRED_MESHES, check_gates

required_meshes = int(os.environ.get("REQUIRED_MESHES", REQUIRED_MESHES))
if required_meshes < 1:
    raise SystemExit("REQUIRED_MESHES must be positive")

out = Path(sys.argv[1])
run = Path(sys.argv[2]).resolve()
rows = defaultdict(dict)
for path in sorted(out.glob("it*_m*.json")):
    m = re.match(r"it(\d+)_m(\d+)\.json", path.name)
    if not m:
        continue
    payload = json.loads(path.read_text())
    metrics = next(iter(payload["per_mesh"].values()))
    if "safety_violation_rate" not in metrics:
        metrics["safety_violation_rate"] = metrics["safety_termination_rate"]
    rows[int(m.group(1))][m.group(2)] = metrics

print(f"\n{'iter':>6}{'lateral':>10}{'contact':>10}{'cut_h':>9}{'head':>9}"
      f"{'impacts':>9}{'safety':>9}{'pass':>7}{'meshes':>8}")
best, best_key = None, None
selection_rows = {}
for it in sorted(rows):
    per = rows[it]
    lat = max(v["worst_family_mean_abs_lateral_error_m"] for v in per.values())
    con = min(v["worst_family_ground_contact_fraction"] for v in per.values())
    cut = max(v["mean_abs_cut_height_error_m"] for v in per.values())
    hea = min(v["p01_head_alignment"] for v in per.values())
    imp = max(v["hard_impact_step_fraction"] for v in per.values())
    ter = max(v["safety_violation_rate"] for v in per.values())
    if len(per) < required_meshes:
        print(f"{it:>6}{lat:>10.4f}{con:>10.4f}{cut:>9.4f}{hea:>9.4f}"
              f"{imp:>9.4f}{ter:>9.4f}{'n/a':>7}{len(per):>8}")
        continue
    gate_result = check_gates(per)
    normalized_margins = {
        name: gate_result["margins"][name] / max(abs(gate.threshold), 1.0e-9)
        for name, gate in GATES.items()
    }
    violations = [max(0.0, -value) for value in normalized_margins.values()]
    if gate_result["passed"]:
        key = (
            0,
            -min(normalized_margins.values()),
            round(lat, 6),
            -round(con, 6),
        )
    else:
        key = (
            1,
            len(gate_result["failing"]),
            max(violations),
            sum(violations),
            round(lat, 6),
            -round(con, 6),
        )
    selection_rows[str(it)] = {
        "checkpoint": str(run / f"model_{it}.pt"),
        "gates": gate_result,
        "ranking_key": list(key),
    }
    print(f"{it:>6}{lat:>10.4f}{con:>10.4f}{cut:>9.4f}{hea:>9.4f}"
          f"{imp:>9.4f}{ter:>9.4f}{str(gate_result['passed']):>7}{len(per):>8}")
    if best is None or key < best_key:
        best, best_key = it, key
if best is None:
    raise SystemExit("no checkpoint has enough meshes for selection")
best_result = selection_rows[str(best)]["gates"]
selection = {
    "selected_iteration": best,
    "promotable": bool(best_result["passed"]),
    "required_meshes": required_meshes,
    "failing_gates": best_result["failing"],
    "ranking": (
        "passing checkpoints first by weakest normalized gate margin; otherwise "
        "fewest failing gates, smallest worst/summed normalized violation"
    ),
    "checkpoints": selection_rows,
}
(out / "selection.json").write_text(json.dumps(selection, indent=2))
if selection["promotable"]:
    print(f"\nPROMOTABLE: model_{best}.pt passes every held-out gate")
else:
    print(
        f"\nNOT PROMOTABLE: model_{best}.pt is the least-bad diagnostic candidate; "
        f"failing gates: {', '.join(best_result['failing'])}"
    )
print(f"selection report: {out / 'selection.json'}")
PYEOF
