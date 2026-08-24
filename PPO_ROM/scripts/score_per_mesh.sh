#!/usr/bin/env bash
set -uo pipefail

ROOT=/home/rrlab/rrlab_mulag
PY=/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
OUT=${REPORT_ROOT:-$ROOT/PPO_ROM/reports/final}
NAME=${1:?usage: score_per_mesh.sh <name> <ckpt|--teacher|--null> [steps] [envs]}
SUBJECT=${2:?}
STEPS=${3:-2808}
ENVS=${4:-32}
MESHES=${MESHES:-"1337 17 909"}
STALL=${STALL:-240}
STAGE=${STAGE:-6}
TASK=${TASK:-}
TASK_ARG=""
[ -n "$TASK" ] && TASK_ARG="--task $TASK"
AMP=${AMP:-}
[ -n "$AMP" ] && TASK_ARG="$TASK_ARG --tracking_amplitude $AMP"
SPEED_ARGS=()
[ -n "${SPEED_KMH:-}" ] && SPEED_ARGS+=(--vehicle-speed-kmh "$SPEED_KMH")
[ -n "${SPEED_MIN_KMH:-}" ] && SPEED_ARGS+=(--vehicle-speed-min-kmh "$SPEED_MIN_KMH")
[ -n "${SPEED_MAX_KMH:-}" ] && SPEED_ARGS+=(--vehicle-speed-max-kmh "$SPEED_MAX_KMH")
SHAPE_ARGS=()
[ -n "${TRAJECTORY_SHAPE:-}" ] && SHAPE_ARGS+=(--trajectory-shape "$TRAJECTORY_SHAPE")
[ -n "${TRAJECTORY_SHAPES:-}" ] && SHAPE_ARGS+=(--trajectory-shapes "$TRAJECTORY_SHAPES")
[ "${NATURAL_VISUALS:-0}" = "1" ] && SHAPE_ARGS+=(--natural-visuals)
SEED_ARGS=()
[ -n "${SEED:-}" ] && SEED_ARGS+=(--seed "$SEED")
# Per-cylinder anti-windup lead caps, e.g. LEAD_PER_AXIS=0.030,0.030,0.030,0.02.
# Scoring an authority change has to go through THIS script rather than calling
# the evaluator directly: the evaluator hangs when handed several meshes in one
# process, and the one-mesh-per-process loop plus the STALL watchdog below is
# what contains that.
LEAD_ARGS=()
[ -n "${LEAD_PER_AXIS:-}" ] && LEAD_ARGS+=(--cylinder_target_max_lead_per_axis_m "$LEAD_PER_AXIS")
# PPO-ID(AAM)-FD latent residual actuation scale; 0 isolates residual damage
# without changing the 68-D policy contract. Empty leaves the task default.
[ -n "${RESIDUAL_SCALE:-}" ] && LEAD_ARGS+=(--online_aam_residual_scale "$RESIDUAL_SCALE")
# Deployment factor domain (nominal/hydraulic/mechanical/combined_train/
# combined_test). Empty leaves the task nominal. The AAM latent is constant
# under nominal, so any z_hat ablation must set this or it measures nothing.
# Leaky integration of the cylinder reference target. Only bites on the ID/FD
# tasks, which integrate from the previous target; ROM integrates from measured
# and is unaffected. Empty leaves the task default (0.0 = unleaked).
[ -n "${TARGET_LEAK:-}" ] && LEAD_ARGS+=(--cylinder_target_leak "$TARGET_LEAK")
# Stroke-limit reference governor: metres of learned extension held back from
# each stop when clamping the commanded target. Empty leaves the task default.
# Null-space stroke centering gain for the redundant 3x4 deployment IK solve.
# ROM integrates the cylinder target from MEASURED; ID-FD/AAM from the PREVIOUS
# target. Values: true|false. Empty leaves the task default.
[ -n "${INTEGRATE_PREV:-}" ] && LEAD_ARGS+=(--cylinder_target_integrates_from_previous_target "$INTEGRATE_PREV")
# Physical extension reserve on cylinder 3, metres of learned extension.
# AAM latent ablation: estimated|shuffled|mean|zero. Empty = deployed system.
# Speed-scheduled lead cap: LEAD_PER_AXIS applies at/below LEAD_SPEED, LEAD_HIGH above.
[ -n "${LEAD_HIGH:-}" ] && LEAD_ARGS+=(--cylinder_target_lead_high_speed_per_axis_m "$LEAD_HIGH")
[ -n "${LEAD_SPEED:-}" ] && LEAD_ARGS+=(--cylinder_target_lead_schedule_speed_kmh "$LEAD_SPEED")
[ -n "${LATENT_MODE:-}" ] && LEAD_ARGS+=(--online_aam_latent_mode "$LATENT_MODE")
[ -n "${CONTROLLER_LATENT_MODE:-}" ] && LEAD_ARGS+=(--online_aam_controller_latent_mode "$CONTROLLER_LATENT_MODE")
[ -n "${C3_RESERVE:-}" ] && LEAD_ARGS+=(--c3_physical_extension_reserve_m "$C3_RESERVE")
[ -n "${NULL_GAIN:-}" ] && LEAD_ARGS+=(--dls_null_space_centering_gain "$NULL_GAIN")
[ -n "${LIMIT_INSET:-}" ] && LEAD_ARGS+=(--cylinder_target_limit_inset_m "$LIMIT_INSET")
[ -n "${FACTOR_DOMAIN:-}" ] && LEAD_ARGS+=(--factor_domain "$FACTOR_DOMAIN")
[ -n "${FACTOR_KEY:-}" ] && LEAD_ARGS+=(--factor_key "$FACTOR_KEY")

# Phase 4 (online adaptation). The plant perturbation changes the PLANT and is
# never observed by the controller; FREEZE_TICK is required by latent_mode
# 'frozen', which latches a pre-change estimate so the ablation isolates online
# updating from static conditioning. ROM_FACTOR_* only bite on the command-ROM
# actuator; the ID-AAM-FD task runs the FD plant and uses PLANT_ALPHA_RANGE.
[ -n "${FREEZE_TICK:-}" ] && LEAD_ARGS+=(--online_aam_latent_freeze_tick "$FREEZE_TICK")
# Policy-latent low-pass cutoff in Hz. Only read by LATENT_MODE=lpf; the
# evaluator rejects it on any other arm rather than recording a cutoff that was
# never applied.
[ -n "${LPF_HZ:-}" ] && LEAD_ARGS+=(--online_aam_policy_latent_lpf_hz "$LPF_HZ")
[ -n "${PLANT_ALPHA_RANGE:-}" ] && LEAD_ARGS+=(--plant_alpha_range "$PLANT_ALPHA_RANGE")
[ -n "${PLANT_DELAY_RANGE:-}" ] && LEAD_ARGS+=(--plant_delay_steps_range "$PLANT_DELAY_RANGE")
[ -n "${PLANT_STEP_TIME_S:-}" ] && LEAD_ARGS+=(--plant_step_time_s "$PLANT_STEP_TIME_S")
[ -n "${PLANT_STEP_ALPHA_SCALE:-}" ] && LEAD_ARGS+=(--plant_step_alpha_scale "$PLANT_STEP_ALPHA_SCALE")
[ -n "${PLANT_STEP_DELAY_STEPS:-}" ] && LEAD_ARGS+=(--plant_step_delay_steps "$PLANT_STEP_DELAY_STEPS")
# Actuator-authority degradation. The ONE stress axis that reaches every stack:
# the deployment factors are inert on the ID/FD plant and clamped on the ROM.
[ -n "${AUTHORITY_ALPHA:-}" ] && LEAD_ARGS+=(--actuator_authority_alpha "$AUTHORITY_ALPHA")
[ -n "${AUTHORITY_ALPHA_AFTER:-}" ] && LEAD_ARGS+=(--actuator_authority_alpha_after "$AUTHORITY_ALPHA_AFTER")
[ -n "${AUTHORITY_STEP_TIME_S:-}" ] && LEAD_ARGS+=(--actuator_authority_step_time_s "$AUTHORITY_STEP_TIME_S")
# Native mid-episode plant change: mass reaches every stack unclamped.
[ -n "${MASS_STEP_TIME_S:-}" ] && LEAD_ARGS+=(--deployment_mass_step_time_s "$MASS_STEP_TIME_S")
[ -n "${MASS_STEP_SCALE:-}" ] && LEAD_ARGS+=(--deployment_mass_step_scale "$MASS_STEP_SCALE")
[ -n "${ROM_FACTOR_PERTURBATION:-}" ] && LEAD_ARGS+=(--rom_factor_perturbation "$ROM_FACTOR_PERTURBATION")
[ -n "${ROM_FACTOR_STEP:-}" ] && LEAD_ARGS+=(--rom_factor_step "$ROM_FACTOR_STEP")
# Commanded-line step: changes the REFERENCE rather than the plant, so the
# latency it exposes belongs to the control stack and not to the hydraulics.
# The adapter must match the latent it is fed; see --online_aam_adapter_checkpoint.
[ -n "${ADAPTER_CKPT:-}" ] && LEAD_ARGS+=(--online_aam_adapter_checkpoint "$ADAPTER_CKPT")
[ -n "${FACTOR_SCALE:-}" ] && LEAD_ARGS+=(--deployment_factor_scale "$FACTOR_SCALE")
[ -n "${REF_STEP_M:-}" ] && LEAD_ARGS+=(--tracking_reference_step_m "$REF_STEP_M")
[ -n "${REF_STEP_TIME_S:-}" ] && LEAD_ARGS+=(--tracking_reference_step_time_s "$REF_STEP_TIME_S")

mkdir -p "$OUT"
cd "$ROOT"

# Meshes are independent scoring runs, so they can overlap on one GPU. Measured
# 2026-08-14: one 32-env evaluator holds 1436 MiB, against 6144 MiB total with
# ~650 MiB of desktop -- three fit with room to spare and the wall clock drops
# by ~3x. Kept at 1 by default because the STALL watchdog below reads one log
# at a time; PARALLEL_MESHES>1 uses the per-mesh watchdog loop further down.
PARALLEL_MESHES=${PARALLEL_MESHES:-1}
if [ "$PARALLEL_MESHES" -gt 1 ]; then
  declare -A MESH_PID
  for MESH in $MESHES; do
    REPORT="$OUT/${NAME}_m${MESH}.json"
    LOG="$OUT/${NAME}_m${MESH}.log"
    rm -f "$REPORT"
    # Per-mesh trace file: one shared path would have each mesh overwrite the
    # last, leaving a trace that silently describes only the final mesh.
    TRACE_ARGS=()
    [ -n "${TRACE_DIR:-}" ] && TRACE_ARGS=(--adaptation_trace_path "$TRACE_DIR/${NAME}_m${MESH}.npz")
    echo "[SCORE] mesh $MESH -> $REPORT (parallel)"
    "$PY" PPO_ROM/scripts/evaluate_gates.py "$SUBJECT" \
        --meshes "$MESH" --steps "$STEPS" --num_envs "$ENVS" --stage "$STAGE" \
        $TASK_ARG --report "$REPORT" --headless "${SEED_ARGS[@]}" "${SPEED_ARGS[@]}" "${SHAPE_ARGS[@]}" "${LEAD_ARGS[@]}" "${TRACE_ARGS[@]}" > "$LOG" 2>&1 &
    MESH_PID[$MESH]=$!
    # Stagger: three simultaneous Isaac startups thrash the GPU and can push
    # peak allocation above what steady-state measurement predicts.
    sleep 20
  done
  for MESH in $MESHES; do
    PID=${MESH_PID[$MESH]}
    LOG="$OUT/${NAME}_m${MESH}.log"
    REPORT="$OUT/${NAME}_m${MESH}.json"
    while kill -0 "$PID" 2>/dev/null; do
      sleep 15
      LAST=$(stat -c %Y "$LOG" 2>/dev/null || echo 0)
      AGE=$(( $(date +%s) - LAST ))
      if [ "$AGE" -gt "$STALL" ]; then
        echo "[SCORE] mesh $MESH STALLED (${AGE}s silent) -- killing"
        kill -9 "$PID" 2>/dev/null
        break
      fi
      [ -f "$REPORT" ] && break
    done
    [ -f "$REPORT" ] && kill -0 "$PID" 2>/dev/null && { kill -TERM "$PID" 2>/dev/null; sleep 1; kill -9 "$PID" 2>/dev/null; }
    wait "$PID" 2>/dev/null
    [ -f "$REPORT" ] && echo "[SCORE] mesh $MESH ok" || echo "[SCORE] mesh $MESH FAILED"
  done
  sleep 4
  MESHES=""
fi

for MESH in $MESHES; do
  REPORT="$OUT/${NAME}_m${MESH}.json"
  LOG="$OUT/${NAME}_m${MESH}.log"
  rm -f "$REPORT"
  TRACE_ARGS=()
  [ -n "${TRACE_DIR:-}" ] && TRACE_ARGS=(--adaptation_trace_path "$TRACE_DIR/${NAME}_m${MESH}.npz")
  echo "[SCORE] mesh $MESH -> $REPORT"

  if [ "$SUBJECT" = "--teacher" ] || [ "$SUBJECT" = "--null" ]; then
    "$PY" PPO_ROM/scripts/evaluate_gates.py "$SUBJECT" \
        --meshes "$MESH" --steps "$STEPS" --num_envs "$ENVS" --stage "$STAGE" \
        $TASK_ARG --report "$REPORT" --headless "${SEED_ARGS[@]}" "${SPEED_ARGS[@]}" "${SHAPE_ARGS[@]}" "${LEAD_ARGS[@]}" "${TRACE_ARGS[@]}" > "$LOG" 2>&1 &
  else
    "$PY" PPO_ROM/scripts/evaluate_gates.py "$SUBJECT" \
        --meshes "$MESH" --steps "$STEPS" --num_envs "$ENVS" --stage "$STAGE" \
        $TASK_ARG --report "$REPORT" --headless "${SEED_ARGS[@]}" "${SPEED_ARGS[@]}" "${SHAPE_ARGS[@]}" "${LEAD_ARGS[@]}" "${TRACE_ARGS[@]}" > "$LOG" 2>&1 &
  fi
  PID=$!

  while kill -0 "$PID" 2>/dev/null; do
    sleep 15
    LAST=$(stat -c %Y "$LOG" 2>/dev/null || echo 0)
    AGE=$(( $(date +%s) - LAST ))
    if [ "$AGE" -gt "$STALL" ]; then
      echo "[SCORE] mesh $MESH STALLED (${AGE}s silent) -- killing"
      kill -9 "$PID" 2>/dev/null
      break
    fi
    [ -f "$REPORT" ] && break
  done
  if [ -f "$REPORT" ] && kill -0 "$PID" 2>/dev/null; then
    kill -TERM "$PID" 2>/dev/null
    sleep 1
    kill -9 "$PID" 2>/dev/null
  fi
  wait "$PID" 2>/dev/null
  sleep 4
  [ -f "$REPORT" ] && echo "[SCORE] mesh $MESH ok" || echo "[SCORE] mesh $MESH FAILED"
done

echo "[SCORE] merging"
REPORTS=("$OUT/${NAME}_m"*.json)
if [ ! -f "${REPORTS[0]}" ]; then
  echo "[SCORE] no per-mesh reports were produced" >&2
  exit 1
fi
"$PY" PPO_ROM/scripts/merge_gates.py "${REPORTS[@]}" \
    --report "$OUT/${NAME}_gates.json"
