#!/usr/bin/env bash
# Rebuild both pipelines on the WIDE tracking task: big lateral shapes, slower
# truck, longer episodes. Scripted speed -- the policy does NOT choose it.
#
# WHY THE TRUCK SLOWS DOWN TO MAKE THE PATH WIDER
# -----------------------------------------------
# The binding constraint is lateral cutter speed, not lateral reach:
#
#     demand = path_slope x truck_speed <= cylinder rate ceiling (0.240 m/s)
#
# and path_slope goes as amplitude x cycles / active_length. Measured with
# probe_path_feasibility.py at 4 km/h over 3 meshes x 11 families, worst family
# double_s_curve:
#
#     amplitude   0.75   1.10   1.40   2.20   3.30   4.40   6.60
#     peak offset 0.152  0.222  0.281  0.432  0.621  0.783  1.022  m
#     demand      0.122  0.178  0.226  0.353  0.523  0.686  0.985  m/s
#     max speed    7.89   5.39   4.24   2.72   1.83   1.40   0.97  km/h
#
# So the wide shapes ARE trackable -- just not at 4 km/h. This configuration
# takes the 2 km/h row: amplitude up to 2.2, peak offset 0.432 m (0.86 m
# peak-to-peak), at 74% of the ceiling.
#
# THE TRAP THIS AVOIDS
# --------------------
# Slowing the truck only helps if the path keeps the SAME GROUND LENGTH. If the
# authored path shrinks with the speed, slope rises by exactly the factor the
# speed fell and demand is unchanged -- substituting active_length = v * T into
# the demand gives `2*pi*A*f/T`, with v cancelled entirely. So the episode has to
# lengthen instead: 26 m of path at 2 km/h takes 46.8 s, not 16 s. That is where
# the longer training cost comes from, and it is not optional.
#
# Amplitude is randomised per environment across the ladder. That is the safe
# axis: it is applied AFTER the endpoint window, so every family still starts and
# ends at zero offset no matter how it is scaled. Frequency is NOT safe to
# randomise -- it moves cycle counts off multiples of 0.5, the family stops
# landing on zero at q=1, and the endpoint window then shuts it discontinuously
# (measured at 11x the rate demand on offset_return).
set -uo pipefail

ROOT=$HOME/rrlab_mulag
PY=$HOME/miniconda3/envs/env_isaaclab/bin/python
OUT=$ROOT/PPO_ROM/reports/wide
SP=${SP:-/tmp/rrlab_scratch}
MESHES_ALL=${MESHES_ALL:-"1337 17 2024 4242 555 909"}
SELECT_MESHES=${SELECT_MESHES:-"1337 17 909"}
SELECT_ITERS=${SELECT_ITERS:-"200 300 400 500 598"}
# At 60 Hz, 2808 control steps are the complete 46.8 s / 26 m traverse.
# Anything shorter is a screening prefix, not acceptance evidence.
STEPS=${STEPS:-2808}
ENVS=${ENVS:-32}
TRAIN_ITERS=${TRAIN_ITERS:-600}
FREEZE=${FREEZE:-150}
TRAIN_ENVS=${TRAIN_ENVS:-256}
DIVERGENCE_GUARD=${DIVERGENCE_GUARD:-1}
DIVERGENCE_FACTOR=${DIVERGENCE_FACTOR:-3.0}
DIVERGENCE_STRIKES=${DIVERGENCE_STRIKES:-3}
DISTILL_ITERS=${DISTILL_ITERS:-120}
DISTILL_ENVS=${DISTILL_ENVS:-128}
IDFD_TASK=RRLAB-Mulag-Verge-IDFD-v0
ROM_TASK=RRLAB-Mulag-Verge-ROM-v0

ROM_STUDENT=$ROOT/logs/rsl_rl/ppo3_1_sole/student_distilled_wide.pt
IDFD_STUDENT=$ROOT/logs/rsl_rl/ppo4_1_sole/student_distilled_wide.pt
ROM_RUN=$ROOT/logs/rsl_rl/ppo3_1_sole/rom_wide
IDFD_RUN=$ROOT/logs/rsl_rl/ppo4_1_sole/idfd_wide

mkdir -p "$OUT" "$SP"
cd "$ROOT"
say() { echo "[WIDE $(date '+%m-%d %H:%M:%S')] $*"; }
have() { [ -e "$1" ]; }

newest_model() {
  ls -1 "$1"/model_*.pt 2>/dev/null \
    | sed -E 's/.*model_([0-9]+)\.pt/\1 &/' | sort -n -k1 | tail -1 | cut -d' ' -f2-
}

best_from_selection() {
  "$PY" - "$1" <<'PYEOF'
import json, re, sys
from collections import defaultdict
from pathlib import Path
rows = defaultdict(dict)
for path in Path(sys.argv[1]).glob("it*_m*.json"):
    m = re.match(r"it(\d+)_m(\d+)\.json", path.name)
    if m:
        rows[int(m.group(1))][m.group(2)] = next(
            iter(json.loads(path.read_text())["per_mesh"].values())
        )
expected = max((len(v) for v in rows.values()), default=0)
best, key = None, None
for it, per in rows.items():
    if len(per) < expected:      # never rank on a partial mesh set
        continue
    lat = max(v["worst_family_mean_abs_lateral_error_m"] for v in per.values())
    con = min(v["worst_family_ground_contact_fraction"] for v in per.values())
    k = (round(lat, 4), -round(con, 4))
    if key is None or k < key:
        best, key = it, k
print(best if best is not None else "")
PYEOF
}

# --- Phase 0: the wide task must still be inside the plant ------------------
say "=== PHASE 0: preflight on the wide task ==="
"$PY" PPO_ROM/scripts/preflight.py --num_envs 32 --steps 240 --stage 6 \
    --report "$OUT/preflight_wide.json" > "$SP/preflight_wide.log" 2>&1
if grep -q '"passed": true' "$OUT/preflight_wide.json" 2>/dev/null; then
  say "preflight passed"
  grep -m1 "P4 every path family" "$SP/preflight_wide.log"
else
  say "PREFLIGHT FAILED -- refusing to spend GPU-hours on it"
  grep -E "^\[FAIL" "$SP/preflight_wide.log" | head
  exit 1
fi

# --- Phase 1: re-distil, because the task changed ---------------------------
distill() {  # distill <label> <actuator> <task> <out> <report>
  local label=$1 actuator=$2 task=$3 out=$4 report=$5
  if have "$out"; then say "$label student exists -- skipping"; return 0; fi
  say "distilling $label student"
  "$PY" verge_task/scripts/distill_terrain.py \
      --actuator "$actuator" --task "$task" \
      --stage 6 --num_envs "$DISTILL_ENVS" --seed 17 \
      --iterations "$DISTILL_ITERS" --out "$out" --report "$report" --headless \
      > "$SP/distill_${label}_wide.log" 2>&1
  [ -f "$out" ] && say "$label student written" || say "$label DISTILL FAILED"
}
say "=== PHASE 1: re-distillation on the wide task ==="
distill rom  rom  "$ROM_TASK"  "$ROM_STUDENT"  "$ROOT/PPO_ROM/reports/distillation_wide.json"
distill idfd idfd "$IDFD_TASK" "$IDFD_STUDENT" "$ROOT/PPO_ID_FD/reports/distillation_wide.json"

# --- Phase 2: references on the wide task -----------------------------------
score() {  # score <name> <subject> [task]
  local name=$1 subject=$2 task=${3:-}
  if have "$OUT/${name}_gates.json"; then say "$name scored already"; return 0; fi
  say "scoring $name"
  MESHES="$MESHES_ALL" STAGE=6 REPORT_ROOT="$OUT" TASK="$task" \
    bash PPO_ROM/scripts/score_per_mesh.sh "$name" "$subject" "$STEPS" "$ENVS" \
    > "$SP/score_${name}_wide.log" 2>&1
  [ -f "$OUT/${name}_gates.json" ] && say "$name scored" || say "$name SCORE FAILED"
}
say "=== PHASE 2: references ==="
score teacher_rom  --teacher
score teacher_idfd --teacher      "$IDFD_TASK"
score null_rom     --null
score romstudent   "$ROM_STUDENT"
score idfdstudent  "$IDFD_STUDENT" "$IDFD_TASK"

# --- Phases 3-4: PPO on the wide task ---------------------------------------
train_and_select() {  # train_and_select <label> <student> <run> <task>
  local label=$1 student=$2 run=$3 task=${4:-}
  [ -f "$student" ] || { say "$label has no student -- skipping"; return 1; }
  say "=== training $label on the wide task ==="
  mkdir -p "$run"
  NUM_ENVS=$TRAIN_ENVS STAGE_ITERATIONS=$TRAIN_ITERS FREEZE_ITERATIONS=$FREEZE \
    STUDENT="$student" LOG_DIR="$run" TASK="$task" \
    DIVERGENCE_GUARD="$DIVERGENCE_GUARD" \
    DIVERGENCE_FACTOR="$DIVERGENCE_FACTOR" \
    DIVERGENCE_STRIKES="$DIVERGENCE_STRIKES" \
    bash PPO_ROM/scripts/supervise_training.sh > "$SP/${label}_wide.log" 2>&1
  say "$label training finished"

  say "=== selecting $label on held-out meshes ==="
  MESHES="$SELECT_MESHES" STAGE=6 TASK="$task" REPORT_ROOT="$OUT/select_${label}" \
    bash PPO_ROM/scripts/select_checkpoint.sh "$run" "$label" $SELECT_ITERS \
    > "$SP/select_${label}_wide.log" 2>&1
  local best
  best=$(best_from_selection "$OUT/select_${label}")
  [ -n "$best" ] || best=$(newest_model "$run" | sed 's/.*model_//;s/\.pt//')
  say "$label best: model_${best}.pt"
  echo "$best" > "$OUT/${label}_best_iteration.txt"
  score "${label}_best" "$run/model_${best}.pt" "$task"
}

train_and_select romppo  "$ROM_STUDENT"  "$ROM_RUN"
train_and_select idfdppo "$IDFD_STUDENT" "$IDFD_RUN" "$IDFD_TASK"

say "=== summary ==="
"$PY" archive/ppo_rom_stage6_tools/summarize_stage6.py "$OUT" \
    --report "$ROOT/reports/WIDE_TASK.md" > "$SP/summary_wide.log" 2>&1 \
    && say "summary at $ROOT/reports/WIDE_TASK.md" || say "summary FAILED"
say "done"
