#!/usr/bin/env bash
# Phase 4b -- the online-adaptation grid, anchored at the OPTIMUM.
#
# Phase 4a could not test the claim, and the reason is worth stating plainly.
# It anchored at nominal (alpha=1.0, delay=0) and perturbed away from it. But
# nominal is not a good operating point: the measured dose-response on mesh
# 909 (outputs/stage31_rom/phase4_dose) is U-shaped on BOTH axes, and nominal
# sits on the shoulder of each.
#
#   alpha  1.00 -> 0.657    delay 0 -> 0.657
#          0.60 -> 0.530          2 -> 0.565
#          0.30 -> 0.546          4 -> 0.644
#          0.15 -> 0.592          8 -> 0.690
#
# So every perturbation in 4a slid the plant TOWARD its optimum. A 40% cut in
# valve effectiveness cost 0.2% of lateral error; both sides of the step sat
# in the flat bottom of the U. There was no deficit for adaptation to recover,
# and "frozen ~ estimated" (0.1%) measured nothing.
#
# This run inverts it:
#   * alpha held at 0.60 -- the measured optimum, so alpha is not a confound
#   * delay steps 2 -> 8 at 9 s, a measured +22% lateral degradation
#
# That is a change the controller demonstrably cannot absorb, so if the latent
# supports online adaptation there is finally something for it to recover.
#
# Four arms, the ones the claim turns on:
#   A4_full      estimated/estimated    the deployed system
#   C_frozen     frozen/frozen          latched BEFORE the change: isolates
#                                       online updating from static conditioning
#   C_mean       mean/mean              in-distribution no-information control
#                                       (`zero` is 2-3 sigma out; see 4a)
#   C_sham_step  no delay change        how much of the outcome is the step
#
# NOT promotion evidence; see run_phase4_ablations.sh.
set -uo pipefail

ROOT=/home/rrlab/rrlab_mulag
cd "$ROOT"
PY=/home/rrlab/miniconda3/envs/env_isaaclab/bin/python

CKPT=${CKPT:-$ROOT/logs/rsl_rl/ppo_id_aam_fd/estimated_after_true_l20_s5_combined_train_20260816/model_4200.pt}
OUT=${OUT:-$ROOT/outputs/stage31_rom/phase4b_optimum}
TRACES=$OUT/traces
LOG=$OUT/phase4.log

MESHES=${MESHES:-"1337 17 909"}
SPEED=${SPEED:-5}
STEPS=${STEPS:-2808}
ENVS=${ENVS:-32}
SEED=${SEED:-17}
FACTOR_DOMAIN=${FACTOR_DOMAIN:-combined_test}
RESIDUAL_SCALE=${RESIDUAL_SCALE:-0.25}
C3_RESERVE=${C3_RESERVE:-0.008}

# Plant change. alpha is valve effectiveness at the FD boundary; the range
# gives per-environment heterogeneity to identify, and the step is the
# mid-episode change the latent has to track.
#
# The step must land inside the SCORED window, which is NOT the 47.3 s episode
# length: evaluate_gates' roll_mesh breaks as soon as every replica reaches the
# 26 m endpoint, so at 5 km/h the rollout ends after 26/(5/3.6) = 18.7 s no
# matter how many --steps are requested. A step scheduled at 20 s never fires;
# every arm then runs on an unperturbed plant and the whole grid returns nulls
# with nothing in the report to say why. Measured: the first attempt recorded
# 1124 steps with theta_elapsed_s topping out at 18.73 s and theta_stepped
# identically zero. The guard below refuses that configuration.
ALPHA_RANGE=${ALPHA_RANGE:-0.6:0.6}
STEP_TIME_S=${STEP_TIME_S:-9.0}
STEP_ALPHA_SCALE=${STEP_ALPHA_SCALE:-1.0}
PATH_LENGTH_M=${PATH_LENGTH_M:-26.0}
# The perturbation that actually bites: transport delay, in control steps.
DELAY_BASE=${DELAY_BASE:-2:2}
STEP_DELAY=${STEP_DELAY:-8}
# Latch two seconds BEFORE the plant changes: the frozen arm must hold a
# pre-change estimate, and it must be latched after the AAM is valid (the L20
# window is 1.0 s = 20 ticks, so anything past ~20 ticks is warm).
#
# UNITS: this counter advances once per 50 ms MODEL tick, not per 60 Hz control
# step -- on the ID-FD path `observe` is only entered on the model clock. So
# 18 s = 360, not 1080. Getting this wrong by 3x would latch AFTER the plant
# change and the "frozen" arm would quietly hold a POST-change estimate,
# turning the decisive control into a near-copy of the full arm.
FREEZE_TICK=${FREEZE_TICK:-140}

test -f "$CKPT" || { echo "missing PPO checkpoint: $CKPT" >&2; exit 2; }
# the sham arm holds the baseline delay, so the schedule fires and nothing changes
DELAY_BASE_N=${DELAY_BASE%%:*}

# --- the step and the latch must both land inside the scored window --------
SCORED_S=$(awk "BEGIN{printf \"%.2f\", $PATH_LENGTH_M/($SPEED/3.6)}")
FREEZE_S=$(awk "BEGIN{printf \"%.2f\", $FREEZE_TICK/20.0}")
awk "BEGIN{exit !($STEP_TIME_S < 0.75*$SCORED_S)}" || {
    echo "plant step at ${STEP_TIME_S}s leaves too little of the ${SCORED_S}s" \
         "scored window to measure recovery; the rollout ends when the truck" \
         "reaches ${PATH_LENGTH_M}m at ${SPEED} km/h" >&2; exit 2; }
awk "BEGIN{exit !($STEP_TIME_S > 0.25*$SCORED_S)}" || {
    echo "plant step at ${STEP_TIME_S}s leaves too little PRE-step window" >&2
    exit 2; }
awk "BEGIN{exit !($FREEZE_S < $STEP_TIME_S)}" || {
    echo "freeze latch at ${FREEZE_S}s is not BEFORE the plant step at" \
         "${STEP_TIME_S}s; the frozen arm would hold a POST-change estimate" \
         "and stop being a control" >&2; exit 2; }
awk "BEGIN{exit !($FREEZE_S > 1.0)}" || {
    echo "freeze latch at ${FREEZE_S}s is inside the 1.0s AAM warmup" >&2
    exit 2; }
mkdir -p "$OUT" "$TRACES"
: > "$LOG"
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "checkpoint      : $CKPT"
say "meshes          : $MESHES   speed ${SPEED} km/h   envs $ENVS   steps $STEPS"
say "plant           : alpha $ALPHA_RANGE (at the measured optimum), delay $DELAY_BASE -> $STEP_DELAY steps at ${STEP_TIME_S}s (+22% lateral)"
say "scored window   : ${SCORED_S}s (truck reaches ${PATH_LENGTH_M}m at ${SPEED} km/h; the rollout ends there, NOT at the 47.3s episode length)"
say "freeze tick     : $FREEZE_TICK model ticks = $(awk "BEGIN{printf \"%.1f\", $FREEZE_TICK/20}")s (plant steps at ${STEP_TIME_S}s)"
say ""

# name             policy      controller  freeze          step_delay
ARMS=(
  "A4_full         estimated   estimated   0               $STEP_DELAY"
  "C_frozen        frozen      frozen      $FREEZE_TICK    $STEP_DELAY"
  "C_mean          mean        mean        0               $STEP_DELAY"
  "C_sham_step     estimated   estimated   0               $DELAY_BASE_N"
)

for row in "${ARMS[@]}"; do
    read -r NAME POLICY CONTROLLER FRZ SDELAY <<<"$row"
    if [ -s "$OUT/${NAME}_gates.json" ]; then
        say "reusing $NAME"
        continue
    fi
    say "--- $NAME  (policy=$POLICY controller=$CONTROLLER freeze=$FRZ step_delay=$SDELAY)"
    FREEZE_ARG=""
    [ "$FRZ" != "0" ] && FREEZE_ARG="$FRZ"
    TASK=RRLAB-Mulag-Verge-ID-AAM-FD-v0 \
    MESHES="$MESHES" SPEED_KMH="$SPEED" STAGE=6 SEED="$SEED" \
    REPORT_ROOT="$OUT" TRACE_DIR="$TRACES" \
    LATENT_MODE="$POLICY" CONTROLLER_LATENT_MODE="$CONTROLLER" \
    FREEZE_TICK="$FREEZE_ARG" \
    PLANT_ALPHA_RANGE="$ALPHA_RANGE" PLANT_STEP_TIME_S="$STEP_TIME_S" \
    PLANT_STEP_ALPHA_SCALE="$STEP_ALPHA_SCALE" \
    PLANT_DELAY_RANGE="$DELAY_BASE" PLANT_STEP_DELAY_STEPS="$SDELAY" \
    RESIDUAL_SCALE="$RESIDUAL_SCALE" C3_RESERVE="$C3_RESERVE" \
    FACTOR_DOMAIN="$FACTOR_DOMAIN" PARALLEL_MESHES="${PARALLEL_MESHES:-1}" \
        PPO_ROM/scripts/score_per_mesh.sh "$NAME" "$CKPT" "$STEPS" "$ENVS" \
        >>"$LOG" 2>&1
    rc=$?
    # score_per_mesh's exit status is the legacy binary safety contract. These
    # arms are DELIBERATELY degraded, so most will fail it; the ablation reads
    # the continuous metrics. A missing report is the only real failure.
    if [ -s "$OUT/${NAME}_gates.json" ]; then
        say "    $NAME ok (scorer rc=$rc)"
    else
        say "    $NAME PRODUCED NO REPORT (rc=$rc) -- see $LOG"
    fi
done

say ""
say "================ SUMMARY ================"
"$PY" "$ROOT/PPO_ID_AAM_FD/experiments/summarise_phase4.py" \
    --dir "$OUT" --out "$OUT/phase4b_summary.json" 2>&1 | tee -a "$LOG"

say ""
say "traces in $TRACES; adaptation figures:"
say "  \$PY PPO_ID_AAM_FD/experiments/analyze_adaptation_trace.py \\"
say "      --trace estimated=$TRACES/A4_full_m909.npz \\"
say "      --trace frozen=$TRACES/C_frozen_m909.npz \\"
say "      --trace zero=$TRACES/A1_baseline_m909.npz"
say "done"
