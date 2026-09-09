#!/usr/bin/env bash
# Retrain both pipelines against the REWARD-ALIGNED contract, and select the
# shipped checkpoint on held-out terrain rather than by iteration number.
#
# WHY A SECOND TRAINING PASS
# --------------------------
# The first correct stage-6 run (rom_s6_feasible) established that PPO was
# maximising its reward faithfully and that the reward was wrong. `hard_impacts`
# is an acceptance gate; the reward had no term for it. Measured over that run:
#
#     hard-impact step fraction   0.193 (distilled student) -> 0.536 (iter 250)
#     Episode_Reward/total        20.22                     -> 20.63
#     contact_force contribution  -0.026 against +9.96 from ground_contact+cut_height
#
# Every checkpoint scored from that run confirms the trade: worst-mesh
# worst-family lateral 0.179-0.269 against the student's 0.0909, with impacts
# 0.37-0.55 against 0.193. Not one of them is shippable, and the cause is an
# exchange rate of roughly 400:1 in favour of driving the cutter into the ground.
#
# `PPO_ROM/rewards.py` now carries a tenth term, `hard_impact`, at -4.0, fed by
# the same per-step level the gate counts. See the weight's note for the
# derivation and `PPO_ROM/tests/test_rewards.py` for the two properties it pins
# -- that the penalty overturns the measured mis-ranking, and that it is NOT so
# large that leaving the ground becomes rational.
#
# The students are NOT re-distilled: DAgger trains on a behaviour-cloning loss
# and never reads the reward, so `student_distilled_s6feasible.pt` is unaffected
# by the change and remains the correct warm start.
#
# WHAT ELSE CHANGED, AND WHY IT IS THE SAME FOR BOTH PIPELINES
# ------------------------------------------------------------
# The observation normalizer is frozen for the whole fine-tune
# (`--obs_normalizer freeze`, now the default). This is a contract choice, not a
# fix for the regression above -- a surgery test (PPO weights + the student's
# normalizer) made tracking WORSE, so the weights and statistics co-adapted and
# the drift is not separable from the learning. It is frozen because the running
# mean/std are part of a resumed policy, and because this repo has already
# measured ID/FD drifting to 1.13 m lateral with a FROZEN ACTOR while the same
# checkpoint scored 0.0628 in evaluation.
#
# Whatever ROM gets, ID/FD gets identically. A PPO_ROM vs PPO_ID_FD comparison
# is only about the actuator if everything else matches.
set -uo pipefail

ROOT=$HOME/rrlab_mulag
PY=$HOME/miniconda3/envs/env_isaaclab/bin/python
OUT=$ROOT/PPO_ROM/reports/stage6
SP=${SP:-/tmp/rrlab_scratch}
MESHES_ALL=${MESHES_ALL:-"1337 17 2024 4242 555 909"}
SELECT_MESHES=${SELECT_MESHES:-"1337 17 909"}
SELECT_ITERS=${SELECT_ITERS:-"150 200 250 300 350 400 450 500 550 598"}
STEPS=${STEPS:-480}
ENVS=${ENVS:-32}
TRAIN_ITERS=${TRAIN_ITERS:-600}
FREEZE=${FREEZE:-150}
TRAIN_ENVS=${TRAIN_ENVS:-256}
IDFD_TASK=RRLAB-Mulag-Verge-IDFD-v0

ROM_STUDENT=$ROOT/logs/rsl_rl/ppo3_1_sole/student_distilled_s6feasible.pt
IDFD_STUDENT=$ROOT/logs/rsl_rl/ppo4_1_sole/student_distilled_s6feasible.pt
ROM_RUN=$ROOT/logs/rsl_rl/ppo3_1_sole/rom_s6_aligned
IDFD_RUN=$ROOT/logs/rsl_rl/ppo4_1_sole/idfd_s6_aligned

mkdir -p "$OUT" "$SP"
cd "$ROOT"
say() { echo "[ALIGNED $(date '+%m-%d %H:%M:%S')] $*"; }

newest_model() {
  ls -1 "$1"/model_*.pt 2>/dev/null \
    | sed -E 's/.*model_([0-9]+)\.pt/\1 &/' | sort -n -k1 | tail -1 | cut -d' ' -f2-
}

best_from_selection() {  # best_from_selection <select-dir>
  "$PY" - "$1" <<'PYEOF'
import json, re, sys
from collections import defaultdict
from pathlib import Path
rows = defaultdict(dict)
for path in Path(sys.argv[1]).glob("it*_m*.json"):
    m = re.match(r"it(\d+)_m(\d+)\.json", path.name)
    if m:
        payload = json.loads(path.read_text())
        rows[int(m.group(1))][m.group(2)] = next(iter(payload["per_mesh"].values()))
best, key = None, None
for it, per in rows.items():
    lat = max(v["worst_family_mean_abs_lateral_error_m"] for v in per.values())
    con = min(v["worst_family_ground_contact_fraction"] for v in per.values())
    k = (round(lat, 4), -round(con, 4))
    if key is None or k < key:
        best, key = it, k
print(best if best is not None else "")
PYEOF
}

# --- Phase A: the new reward must still satisfy the reward contracts --------
say "=== PHASE A: preflight with the aligned reward ==="
"$PY" PPO_ROM/scripts/preflight.py --num_envs 32 --steps 240 --stage 6 \
    --report "$OUT/preflight_aligned.json" > "$SP/preflight_aligned.log" 2>&1
if grep -q '"passed": true' "$OUT/preflight_aligned.json" 2>/dev/null; then
  say "preflight passed"
else
  say "PREFLIGHT FAILED -- refusing to spend 4 GPU-hours on it"
  grep -E "^\[FAIL" "$SP/preflight_aligned.log" | head
  exit 1
fi

# --- Phases B-E: train, then select on held-out terrain ---------------------
train_and_select() {  # train_and_select <label> <student> <run> <task>
  local label=$1 student=$2 run=$3 task=${4:-}
  say "=== training $label ($TRAIN_ITERS iters, freeze $FREEZE) ==="
  mkdir -p "$run"
  NUM_ENVS=$TRAIN_ENVS STAGE_ITERATIONS=$TRAIN_ITERS FREEZE_ITERATIONS=$FREEZE \
    STUDENT="$student" LOG_DIR="$run" TASK="$task" \
    bash PPO_ROM/scripts/supervise_training.sh > "$SP/${label}_aligned.log" 2>&1
  say "$label training finished"

  say "=== selecting $label checkpoint on held-out meshes ==="
  MESHES="$SELECT_MESHES" STAGE=6 TASK="$task" \
    REPORT_ROOT="$OUT/select_${label}" \
    bash PPO_ROM/scripts/select_checkpoint.sh "$run" "$label" $SELECT_ITERS \
    > "$SP/select_${label}.log" 2>&1
  local best
  best=$(best_from_selection "$OUT/select_${label}")
  if [ -z "$best" ]; then
    say "$label selection produced nothing -- falling back to the newest"
    best=$(newest_model "$run" | sed 's/.*model_//;s/\.pt//')
  fi
  say "$label best checkpoint: model_${best}.pt"
  echo "$best" > "$OUT/${label}_best_iteration.txt"

  say "=== scoring $label best on all $(echo $MESHES_ALL | wc -w) meshes ==="
  MESHES="$MESHES_ALL" STAGE=6 REPORT_ROOT="$OUT" TASK="$task" \
    bash PPO_ROM/scripts/score_per_mesh.sh "${label}_best" \
    "$run/model_${best}.pt" "$STEPS" "$ENVS" > "$SP/score_${label}_best.log" 2>&1
  [ -f "$OUT/${label}_best_gates.json" ] && say "$label best scored" \
      || say "$label best SCORE FAILED"
}

train_and_select romppo  "$ROM_STUDENT"  "$ROM_RUN"
train_and_select idfdppo "$IDFD_STUDENT" "$IDFD_RUN" "$IDFD_TASK"

say "=== summary ==="
"$PY" archive/ppo_rom_stage6_tools/summarize_stage6.py "$OUT" \
    --report "$ROOT/reports/STAGE6_REBUILD.md" > "$SP/summary.log" 2>&1 \
    && say "summary at $ROOT/reports/STAGE6_REBUILD.md" || say "summary FAILED"
say "done"
