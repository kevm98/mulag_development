#!/usr/bin/env bash
# Rebuild both legacy pipelines on the CORRECTED stage-6 task, end to end.
#
# WHY THIS EXISTS
# ---------------
# Three defects were found on 2026-08-11, all measured, all fixed in the code
# this script drives. Nothing produced before them is comparable.
#
#   1. PPO TRAINED WITH A PARKED TRUCK. `apply_sole_curriculum_stage` leaves
#      `randomize_trajectory_shape = True`; the base env picks a library entry at
#      CONSTRUCTION time and the parked entries (sine, circle) execute
#      `vehicle_speed_min_kmh = vehicle_speed_max_kmh = 0.0`. Both trainers
#      pinned the deployment shape only AFTER gym.make, so the speed was never
#      restored. Probed directly: 0.0000 m of truck travel over 180 control steps
#      with the old ordering, 3.3333 m (exactly 4 km/h) with the fixed one.
#      Distillation and evaluate_gates.py always pinned it before construction,
#      so only PPO trained on the wrong task -- which is the whole of the "PPO
#      regresses" history.
#
#   2. ACCEPTANCE SCORED STAGE 5 while both pipelines trained at stage 6.
#
#   3. THE STAGE-6 PATH WAS KINEMATICALLY OUT OF REACH. The shipped amplitude
#      ladder (2.2, 3.3, 4.4, 5.5, 6.6) has EVERY rung demanding more lateral
#      cutter speed than the cylinder rate ceiling delivers at 4 km/h -- 24 of 33
#      family-mesh cells at 4.4, 26 at 6.6, the worst on 56% of steps. Confirmed
#      on the simulator: the privileged DLS teacher failed 5 of 8 gates there,
#      sitting at exactly 0.0000 m cylinder headroom. The ladder is now
#      (0.8, 1.0, 1.2, 1.4, 1.6), every rung inside the measured ceiling.
#
# Because the terrain changed, THE STUDENTS MUST BE RE-DISTILLED. A checkpoint
# distilled on the 2.2-6.6 ladder is not a warm start for this task; this repo
# has already measured what re-using a stale student costs.
#
# RESUMABILITY
# ------------
# Every phase skips itself if its artifact already exists, so an interrupted run
# is restarted by re-invoking it. Set FORCE=1 to redo everything.
set -uo pipefail

ROOT=$HOME/rrlab_mulag
PY=$HOME/miniconda3/envs/env_isaaclab/bin/python
OUT=$ROOT/PPO_ROM/reports/stage6
SP=${SP:-/tmp/rrlab_scratch}
MESHES_ALL=${MESHES_ALL:-"1337 17 2024 4242 555 909"}
STEPS=${STEPS:-480}
ENVS=${ENVS:-32}
TRAIN_ITERS=${TRAIN_ITERS:-600}
FREEZE=${FREEZE:-150}
TRAIN_ENVS=${TRAIN_ENVS:-256}
DISTILL_ITERS=${DISTILL_ITERS:-120}
DISTILL_ENVS=${DISTILL_ENVS:-128}
FORCE=${FORCE:-0}
IDFD_TASK=RRLAB-Mulag-Verge-IDFD-v0
ROM_TASK=RRLAB-Mulag-Verge-ROM-v0

ROM_STUDENT=$ROOT/logs/rsl_rl/ppo3_1_sole/student_distilled_s6feasible.pt
IDFD_STUDENT=$ROOT/logs/rsl_rl/ppo4_1_sole/student_distilled_s6feasible.pt
ROM_RUN=$ROOT/logs/rsl_rl/ppo3_1_sole/rom_s6_feasible
IDFD_RUN=$ROOT/logs/rsl_rl/ppo4_1_sole/idfd_s6_feasible

mkdir -p "$OUT" "$SP" "$(dirname "$ROM_STUDENT")" "$(dirname "$IDFD_STUDENT")"
cd "$ROOT"
say() { echo "[REBUILD $(date '+%m-%d %H:%M:%S')] $*"; }
have() { [ "$FORCE" = "0" ] && [ -e "$1" ]; }

distill() {  # distill <label> <actuator> <task> <out.pt> <report.json>
  local label=$1 actuator=$2 task=$3 out=$4 report=$5
  if have "$out"; then say "$label student already distilled -- skipping"; return 0; fi
  say "distilling $label student ($DISTILL_ITERS iters, $DISTILL_ENVS envs)"
  "$PY" verge_task/scripts/distill_terrain.py \
      --actuator "$actuator" --task "$task" \
      --stage 6 --num_envs "$DISTILL_ENVS" --seed 17 \
      --iterations "$DISTILL_ITERS" \
      --out "$out" --report "$report" --headless \
      > "$SP/distill_${label}.log" 2>&1
  [ -f "$out" ] && say "$label student written" \
                || say "$label DISTILLATION FAILED (see $SP/distill_${label}.log)"
}

score() {  # score <name> <subject> [task]
  local name=$1 subject=$2 task=${3:-}
  if have "$OUT/${name}_gates.json"; then say "$name already scored -- skipping"; return 0; fi
  say "scoring $name"
  MESHES="$MESHES_ALL" STAGE=6 REPORT_ROOT="$OUT" TASK="$task" \
    bash PPO_ROM/scripts/score_per_mesh.sh "$name" "$subject" "$STEPS" "$ENVS" \
    > "$SP/score_${name}.log" 2>&1
  [ -f "$OUT/${name}_gates.json" ] && say "$name scored" \
      || say "$name SCORE FAILED (see $SP/score_${name}.log)"
}

newest_model() {
  ls -1 "$1"/model_*.pt 2>/dev/null \
    | sed -E 's/.*model_([0-9]+)\.pt/\1 &/' | sort -n -k1 | tail -1 | cut -d' ' -f2-
}

train() {  # train <label> <student> <run-dir> [task]
  local label=$1 student=$2 run=$3 task=${4:-}
  if [ ! -f "$student" ]; then say "$label has no student -- skipping training"; return 1; fi
  if have "$run/model_$((TRAIN_ITERS - 2)).pt"; then
    say "$label already trained -- skipping"; return 0
  fi
  say "training $label PPO ($TRAIN_ITERS iters, freeze $FREEZE)"
  mkdir -p "$run"
  NUM_ENVS=$TRAIN_ENVS STAGE_ITERATIONS=$TRAIN_ITERS FREEZE_ITERATIONS=$FREEZE \
    STUDENT="$student" LOG_DIR="$run" TASK="$task" \
    bash PPO_ROM/scripts/supervise_training.sh > "$SP/${label}_ppo.log" 2>&1
  say "$label PPO finished"
}

# --- Phase 0: the contracts still hold on the corrected terrain -------------
say "=== PHASE 0: preflight on the corrected stage-6 terrain ==="
if ! have "$OUT/preflight_rom.json"; then
  "$PY" PPO_ROM/scripts/preflight.py --num_envs 32 --steps 120 \
      --report "$OUT/preflight_rom.json" > "$SP/preflight_rom.log" 2>&1 \
      && say "ROM preflight ok" || say "ROM preflight FAILED (see $SP/preflight_rom.log)"
fi

# --- Phase 1: re-distil both students on the feasible terrain ---------------
say "=== PHASE 1: re-distillation ==="
distill rom  rom  "$ROM_TASK"  "$ROM_STUDENT"  "$ROOT/PPO_ROM/reports/distillation_s6feasible.json"
distill idfd idfd "$IDFD_TASK" "$IDFD_STUDENT" "$ROOT/PPO_ID_FD/reports/distillation_s6feasible.json"

# --- Phase 2: what the plant can do, and what doing nothing gets ------------
# The teacher is scored through BOTH tasks: the actuator is part of the plant,
# so the ROM teacher anchors PPO_ROM and the ID/FD teacher anchors PPO_ID_FD.
# The null policy is scored because a gate a do-nothing policy passes is
# measuring the terrain rather than the task.
say "=== PHASE 2: stage-6 reference measurements ==="
score teacher_rom  --teacher
score teacher_idfd --teacher      "$IDFD_TASK"
score null_rom     --null
score romstudent   "$ROM_STUDENT"
score idfdstudent  "$IDFD_STUDENT" "$IDFD_TASK"

say "=== PHASE 3: threshold re-derivation (report only, never applied) ==="
"$PY" archive/ppo_rom_stage6_tools/derive_thresholds.py "$OUT" \
    --report "$OUT/threshold_proposal.json" > "$SP/derive_thresholds.log" 2>&1 \
    && say "threshold proposal written" \
    || say "threshold derivation FAILED (see $SP/derive_thresholds.log)"

# --- Phase 4: PPO, the point of the exercise --------------------------------
say "=== PHASE 4: ROM PPO ==="
train rom "$ROM_STUDENT" "$ROM_RUN"
ROM_CKPT=$(newest_model "$ROM_RUN")
[ -n "${ROM_CKPT:-}" ] && { say "ROM final: $ROM_CKPT"; score romppo "$ROM_CKPT"; } \
    || say "ROM produced no checkpoint"

say "=== PHASE 5: ID/FD PPO ==="
train idfd "$IDFD_STUDENT" "$IDFD_RUN" "$IDFD_TASK"
IDFD_CKPT=$(newest_model "$IDFD_RUN")
[ -n "${IDFD_CKPT:-}" ] && { say "ID/FD final: $IDFD_CKPT"; score idfdppo "$IDFD_CKPT" "$IDFD_TASK"; } \
    || say "ID/FD produced no checkpoint"

# --- Phase 6: one summary ---------------------------------------------------
say "=== PHASE 6: summary ==="
"$PY" archive/ppo_rom_stage6_tools/summarize_stage6.py "$OUT" \
    --report "$ROOT/reports/STAGE6_REBUILD.md" > "$SP/summary.log" 2>&1 \
    && say "summary at $ROOT/reports/STAGE6_REBUILD.md" \
    || say "summary FAILED (see $SP/summary.log)"

say "done"
