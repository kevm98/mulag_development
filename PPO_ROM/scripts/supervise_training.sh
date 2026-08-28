#!/usr/bin/env bash
set -uo pipefail

ROOT=${ROOT:-/home/rrlab/rrlab_mulag}
PY=${PY:-/home/rrlab/miniconda3/envs/env_isaaclab/bin/python}
RUN_DIR=${LOG_DIR:-$ROOT/logs/rsl_rl/ppo3_1_sole/warmup_prod}
STALL_SECONDS=${STALL_SECONDS:-420}
MAX_RESTARTS=${MAX_RESTARTS:-6}
NUM_ENVS=${NUM_ENVS:-256}
STAGE_ITERATIONS=${STAGE_ITERATIONS:-600}
FREEZE_ITERATIONS=${FREEZE_ITERATIONS:-150}
SEED=${SEED:-17}
DIVERGENCE_GUARD=${DIVERGENCE_GUARD:-0}
DIVERGENCE_FACTOR=${DIVERGENCE_FACTOR:-3.0}
DIVERGENCE_STRIKES=${DIVERGENCE_STRIKES:-3}
DIVERGENCE_POLL=${DIVERGENCE_POLL:-60}
DIVERGENCE_BASELINE_ITERATIONS=${DIVERGENCE_BASELINE_ITERATIONS:-50}
FIXED_STD=${FIXED_STD:-0.05}
GAMMA=${GAMMA:-}
LAM=${LAM:-}
DEMO_REGULARIZED=${DEMO_REGULARIZED:-0}
DEMO_LAMBDA=${DEMO_LAMBDA:-0.25}
DEMO_KL_BETA=${DEMO_KL_BETA:-0.02}
DEMO_MAX_KL=${DEMO_MAX_KL:-0.25}
DEMO_LATERAL_THRESHOLD=${DEMO_LATERAL_THRESHOLD:-0.10}
TRAJECTORY_REGULARIZED=${TRAJECTORY_REGULARIZED:-0}
TRAJECTORY_LOCAL_COEF=${TRAJECTORY_LOCAL_COEF:-0.05}
TRAJECTORY_FUTURE_COEF=${TRAJECTORY_FUTURE_COEF:-0.0}
TRAJECTORY_MAX_UPDATE_KL=${TRAJECTORY_MAX_UPDATE_KL:-0.01}
TRAJECTORY_MAX_REFERENCE_KL=${TRAJECTORY_MAX_REFERENCE_KL:-0.05}
RECOVERY_GAMMA=${RECOVERY_GAMMA:-0.97}
STUDENT=${STUDENT:-$ROOT/logs/rsl_rl/ppo3_1_sole/student_distilled_v2.pt}
TASK=${TASK:-}
TASK_ARG=""
[ -n "$TASK" ] && TASK_ARG="--task $TASK"
build_experiment_args() {
  EXPERIMENT_ARGS=()
  [ -n "$GAMMA" ] && EXPERIMENT_ARGS+=(--gamma "$GAMMA")
  [ -n "$LAM" ] && EXPERIMENT_ARGS+=(--lam "$LAM")
  if [ "$DEMO_REGULARIZED" = "1" ]; then
    EXPERIMENT_ARGS+=(
      --demo_regularized
      --demo_lambda "$DEMO_LAMBDA"
      --demo_kl_beta "$DEMO_KL_BETA"
      --demo_max_kl "$DEMO_MAX_KL"
      --demo_lateral_threshold "$DEMO_LATERAL_THRESHOLD"
    )
  fi
  if [ "$TRAJECTORY_REGULARIZED" = "1" ]; then
    EXPERIMENT_ARGS+=(
      --trajectory_regularized
      --trajectory_local_coef "$TRAJECTORY_LOCAL_COEF"
      --trajectory_future_coef "$TRAJECTORY_FUTURE_COEF"
      --trajectory_max_update_kl "$TRAJECTORY_MAX_UPDATE_KL"
      --trajectory_max_reference_kl "$TRAJECTORY_MAX_REFERENCE_KL"
      --trajectory_clip_param 0.10
      --trajectory_learning_epochs 3
      --trajectory_actor_learning_rate 1e-5
      --trajectory_critic_learning_rate 1e-4
    )
    STABILITY_ARGS=(
      --entropy_coef 0.001
      --fixed_std "$FIXED_STD"
    )
  else
    STABILITY_ARGS=(
      --learning_rate 3e-4
      --desired_kl 0.008
      --entropy_coef 0.001
      --std_clamp 0.03,0.15
    )
  fi
}
build_experiment_args
if [ ! -f "$STUDENT" ]; then
  echo "[SUPERVISOR] warm start not found: $STUDENT" >&2
  exit 1
fi

mkdir -p "$RUN_DIR"
cd "$ROOT"

TRAIN_PID=""
GUARD_PID=""

cleanup_children() {
  if [ -n "$GUARD_PID" ]; then
    kill "$GUARD_PID" 2>/dev/null || true
    wait "$GUARD_PID" 2>/dev/null || true
    GUARD_PID=""
  fi
  if [ -n "$TRAIN_PID" ]; then
    kill -TERM "$TRAIN_PID" 2>/dev/null || true
    TRAIN_PID=""
  fi
}

on_signal() {
  cleanup_children
  exit 143
}

trap cleanup_children EXIT
trap on_signal INT TERM

newest_checkpoint() {
  ls -1 "$RUN_DIR"/model_*.pt 2>/dev/null \
    | sed -E 's/.*model_([0-9]+)\.pt/\1 &/' | sort -n -k1 | tail -1 | cut -d' ' -f2-
}

save_warmup_checkpoint() {
  local boundary="$RUN_DIR/model_${FREEZE_ITERATIONS}.pt"
  if [ ! -f "$WARMUP_CHECKPOINT" ] && [ -f "$boundary" ]; then
    cp "$boundary" "$WARMUP_CHECKPOINT"
    echo "[SUPERVISOR] saved warm-up boundary $WARMUP_CHECKPOINT"
  fi
}

WARMUP_CHECKPOINT="$RUN_DIR/warmup_checkpoint.pt"
FORCE_RESUME=""
REGRESSION_RECOVERY_USED=0

for attempt in $(seq 0 "$MAX_RESTARTS"); do
  VERDICT="$RUN_DIR/divergence_verdict.txt"
  if [ -f "$VERDICT" ]; then
    mv "$VERDICT" "$RUN_DIR/divergence_verdict_stale_before_attempt_${attempt}.txt"
  fi
  if [ -n "$FORCE_RESUME" ] && [ -f "$FORCE_RESUME" ]; then
    CKPT="$FORCE_RESUME"
    DONE_ITERS="$FREEZE_ITERATIONS"
    FORCE_RESUME=""
    echo "[SUPERVISOR] recovery restart from preserved warm-up boundary $CKPT"
  else
    CKPT=$(newest_checkpoint)
    DONE_ITERS=0
    if [ -n "${CKPT:-}" ]; then
      DONE_ITERS=$(basename "$CKPT" .pt | sed 's/model_//')
    fi
  fi

  if [ -f "$WARMUP_CHECKPOINT" ] && [ "$DONE_ITERS" -lt "$FREEZE_ITERATIONS" ]; then
    CKPT="$WARMUP_CHECKPOINT"
    DONE_ITERS="$FREEZE_ITERATIONS"
    echo "[SUPERVISOR] warm-up boundary found: resuming $CKPT ($DONE_ITERS done, actor already unfrozen)"
  fi

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

  LOG_BYTES_BEFORE=$(stat -c %s "$RUN_DIR/train.log" 2>/dev/null || echo 0)
  "$PY" PPO_ROM/scripts/train.py \
      --resume "$RESUME" $TASK_ARG \
      --stages 6 \
      --num_envs "$NUM_ENVS" \
      --seed "$SEED" \
      --stage_iterations "$REMAINING" \
      --freeze_actor_iterations "$FREEZE_NOW" \
      "${STABILITY_ARGS[@]}" \
      --obs_normalizer freeze \
      --save_interval 25 \
      --log_dir "$RUN_DIR" \
      --headless "${EXPERIMENT_ARGS[@]}" >> "$RUN_DIR/train.log" 2>&1 &
  TRAIN_PID=$!
  echo "[SUPERVISOR] pid $TRAIN_PID (stall limit ${STALL_SECONDS}s)"

  if [ "$DIVERGENCE_GUARD" = "1" ]; then
    POLL="$DIVERGENCE_POLL" bash "$ROOT/PPO_ROM/scripts/divergence_guard.sh" \
        "$RUN_DIR/train.log" "$TRAIN_PID" "$DIVERGENCE_BASELINE_ITERATIONS" \
        "$DIVERGENCE_FACTOR" "$DIVERGENCE_STRIKES" "$LOG_BYTES_BEFORE" \
        > "$RUN_DIR/divergence_guard_attempt_${attempt}.log" 2>&1 &
    GUARD_PID=$!
    echo "[SUPERVISOR] divergence guard pid $GUARD_PID " \
         "(factor ${DIVERGENCE_FACTOR}x, strikes ${DIVERGENCE_STRIKES})"
  fi

  while kill -0 "$TRAIN_PID" 2>/dev/null; do
    sleep 60
    save_warmup_checkpoint
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
  save_warmup_checkpoint
  RC=$?
  TRAIN_PID=""
  if [ -n "$GUARD_PID" ]; then
    kill "$GUARD_PID" 2>/dev/null || true
    wait "$GUARD_PID" 2>/dev/null || true
    GUARD_PID=""
  fi
  if [ "$RC" -eq 0 ]; then
    if tail -c +$((LOG_BYTES_BEFORE + 1)) "$RUN_DIR/train.log" 2>/dev/null \
        | grep -Eq 'Traceback \(most recent call last\)|RuntimeError:|CUDA out of memory|CUBLAS_STATUS|CUSOLVER_STATUS'; then
      echo "[SUPERVISOR] traceback/error detected in this attempt despite rc=0"
      RC=1
    fi
  fi
  if [ -f "$VERDICT" ]; then
    ATTEMPT_VERDICT="$RUN_DIR/divergence_verdict_attempt_${attempt}.txt"
    mv "$VERDICT" "$ATTEMPT_VERDICT"
    if [ "$REGRESSION_RECOVERY_USED" -eq 1 ]; then
      echo "[SUPERVISOR] recovery configuration regressed again; verdict=$ATTEMPT_VERDICT"
      echo "[SUPERVISOR] stopping instead of repeating a scientifically identical run"
      exit 2
    fi
    DEMO_REGULARIZED=0
    TRAJECTORY_REGULARIZED=1
    REGRESSION_RECOVERY_USED=1
    GAMMA="$RECOVERY_GAMMA"
    FORCE_RESUME="$WARMUP_CHECKPOINT"
    build_experiment_args
    echo "[SUPERVISOR] health regression detected; escalating to " \
         "reference-trajectory recovery gamma=$GAMMA from $FORCE_RESUME"
  fi
  if [ -f "$RUN_DIR/run_manifest.json" ]; then
    cp "$RUN_DIR/run_manifest.json" "$RUN_DIR/run_manifest_attempt_${attempt}.json"
  fi
  if [ "$RC" -eq 0 ]; then
    echo "[SUPERVISOR] training exited cleanly"
    exit 0
  fi
  echo "[SUPERVISOR] exit code $RC -- restarting from the newest checkpoint"
done

echo "[SUPERVISOR] gave up after $MAX_RESTARTS restarts" >&2
exit 1
