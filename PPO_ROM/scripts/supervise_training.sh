#!/usr/bin/env bash
# Run PPO3_1 training under a stall watchdog, resuming from the newest checkpoint.
#
# WHY THIS EXISTS
# ---------------
# On 2026-08-06 a run hung at iteration 4 and sat there for FOUR HOURS: the
# process stayed alive (state R, 108% CPU, GPU at 40%), so nothing detected a
# failure, and the log simply stopped growing. Every watcher we had was looking
# for a crash or a completion, and a hang is neither. The whole night produced
# four iterations.
#
# The watchdog treats "train.log has not grown in STALL_SECONDS" as the failure
# signal, because that is the one symptom a hang actually presents. On a stall
# it kills the process and restarts from the newest checkpoint, so progress is
# preserved rather than restarted from zero.
#
# Usage:  [NUM_ENVS=256] [STAGE_ITERATIONS=600] [FREEZE_ITERATIONS=150] \
#         [STALL_SECONDS=420] [MAX_RESTARTS=6] supervise_training.sh
set -uo pipefail

ROOT=/home/rrlab/rrlab_mulag
PY=/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
RUN_DIR=${LOG_DIR:-$ROOT/logs/rsl_rl/ppo3_1_sole/warmup_prod}
STALL_SECONDS=${STALL_SECONDS:-420}
MAX_RESTARTS=${MAX_RESTARTS:-6}
NUM_ENVS=${NUM_ENVS:-256}
STAGE_ITERATIONS=${STAGE_ITERATIONS:-600}
FREEZE_ITERATIONS=${FREEZE_ITERATIONS:-150}
# Which warm start to fine-tune. Override with STUDENT=... -- do NOT leave this
# pinned to a stale checkpoint: student_distilled.pt was distilled on the OLD
# infeasible terrain (amplitude 2.2) and tracks at 0.303 m on the current
# terrain, which permits 0.058 m. Fine-tuning from it is what produced four
# consecutive runs that degraded tracking.
STUDENT=${STUDENT:-$ROOT/logs/rsl_rl/ppo3_1_sole/student_distilled_v2.pt}
# TASK selects the pipeline. train.py builds PPO3_1's ROM task when this is
# empty, REGARDLESS of which checkpoint is resumed, so an ID/FD run without it
# silently trains on the wrong plant.
TASK=${TASK:-}
TASK_ARG=""
[ -n "$TASK" ] && TASK_ARG="--task $TASK"
if [ ! -f "$STUDENT" ]; then
  echo "[SUPERVISOR] warm start not found: $STUDENT" >&2
  exit 1
fi

mkdir -p "$RUN_DIR"
cd "$ROOT"

newest_checkpoint() {
  ls -1 "$RUN_DIR"/model_*.pt 2>/dev/null \
    | sed -E 's/.*model_([0-9]+)\.pt/\1 &/' | sort -n -k1 | tail -1 | cut -d' ' -f2-
}

for attempt in $(seq 0 "$MAX_RESTARTS"); do
  CKPT=$(newest_checkpoint)
  DONE_ITERS=0
  if [ -n "${CKPT:-}" ]; then
    DONE_ITERS=$(basename "$CKPT" .pt | sed 's/model_//')
  fi

  # Restarting mid-run must NOT redo the frozen window: the critic warm-start
  # and the frozen iterations already happened, and repeating them would reset
  # the critic's progress and re-freeze an actor that was already released.
  if [ "$DONE_ITERS" -gt 0 ] && [ -n "${CKPT:-}" ]; then
    RESUME=$CKPT
    REMAINING=$((STAGE_ITERATIONS - DONE_ITERS))
    FREEZE_NOW=$((FREEZE_ITERATIONS - DONE_ITERS))
    [ "$FREEZE_NOW" -lt 0 ] && FREEZE_NOW=0
    echo "[SUPERVISOR] attempt $attempt: resuming $CKPT ($DONE_ITERS done, $REMAINING left, freeze $FREEZE_NOW)"
  else
    RESUME=$STUDENT
    REMAINING=$STAGE_ITERATIONS
    FREEZE_NOW=$FREEZE_ITERATIONS
    echo "[SUPERVISOR] attempt $attempt: fresh start from the distilled student"
  fi

  if [ "$REMAINING" -le 0 ]; then
    echo "[SUPERVISOR] target of $STAGE_ITERATIONS iterations reached"
    exit 0
  fi

  "$PY" PPO_ROM/scripts/train.py \
      --resume "$RESUME" $TASK_ARG \
      --stages 6 \
      --num_envs "$NUM_ENVS" \
      --seed $((17 + attempt)) \
      --stage_iterations "$REMAINING" \
      --freeze_actor_iterations "$FREEZE_NOW" \
      --learning_rate 3e-4 \
      --desired_kl 0.008 \
      --entropy_coef 0.001 \
      --std_clamp 0.03,0.15 \
      --save_interval 25 \
      --log_dir "$RUN_DIR" \
      --headless >> "$RUN_DIR/train.log" 2>&1 &
  TRAIN_PID=$!
  echo "[SUPERVISOR] pid $TRAIN_PID (stall limit ${STALL_SECONDS}s)"

  # Watch for a stall: the log stops growing while the process stays alive.
  while kill -0 "$TRAIN_PID" 2>/dev/null; do
    sleep 60
    LAST_WRITE=$(stat -c %Y "$RUN_DIR/train.log" 2>/dev/null || echo 0)
    AGE=$(( $(date +%s) - LAST_WRITE ))
    if [ "$AGE" -gt "$STALL_SECONDS" ]; then
      echo "[SUPERVISOR] STALLED: no log output for ${AGE}s -- killing $TRAIN_PID"
      kill -9 "$TRAIN_PID" 2>/dev/null
      sleep 15
      break
    fi
  done

  wait "$TRAIN_PID" 2>/dev/null
  RC=$?
  # A clean exit means training finished its remaining iterations.
  if [ "$RC" -eq 0 ]; then
    echo "[SUPERVISOR] training exited cleanly"
    exit 0
  fi
  echo "[SUPERVISOR] exit code $RC -- restarting from the newest checkpoint"
done

echo "[SUPERVISOR] gave up after $MAX_RESTARTS restarts" >&2
exit 1
