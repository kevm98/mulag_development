#!/usr/bin/env bash
# Phase 4 -- the online-adaptation ablation grid.
#
# Claim under test:
#   the AAM estimates a latent of the CURRENT plant from interaction history,
#   that latent conditions BOTH the PPO policy and the ID model, and this
#   enables ONLINE adaptation to a plant that changes mid-episode.
#
# Every arm below removes exactly one part of that sentence.
#
#   ROUTING (2x2: where the latent goes)
#     A1 baseline    policy zero       controller zero
#     A2 ID only     policy zero       controller estimated
#     A3 PPO only    policy estimated  controller zero
#     A4 full        policy estimated  controller estimated   <- deployed
#
#   CONTENT (at full routing: is it the latent's content and updating?)
#     C_shuffled     both shuffled     same marginal distribution, wrong plant
#     C_frozen       both frozen       right content, stops updating
#
# `shuffled` rules out "extra input dimensions help"; it has the deployed
# latent's distribution but every environment receives another environment's
# context. `frozen` rules out "static conditioning is enough": it latches the
# live estimate BEFORE the plant changes, so it is identical to A4 up to the
# latch and differs only in that it stops tracking. A4 beating C_frozen after
# the step is the online-adaptation result; A4 beating only C_shuffled would
# not be.
#
# The plant changes mid-episode (valve effectiveness steps at PLANT_STEP_TIME_S)
# and the controller never sees the perturbation parameters -- it must infer
# their effect from the commanded/response mismatch.
#
# NOT promotion evidence. promote_system.py rejects controller modes other than
# `estimated` on purpose, and half this grid needs them; the ablation is scored
# here, separately, and the promotion contract is left untouched.
set -uo pipefail

ROOT=/home/rrlab/rrlab_mulag
cd "$ROOT"
PY=/home/rrlab/miniconda3/envs/env_isaaclab/bin/python

CKPT=${CKPT:-$ROOT/logs/rsl_rl/ppo_id_aam_fd/estimated_after_true_l20_s5_combined_train_20260816/model_4200.pt}
OUT=${OUT:-$ROOT/outputs/stage31_rom/phase4_ablation}
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
ALPHA_RANGE=${ALPHA_RANGE:-0.7:1.0}
STEP_TIME_S=${STEP_TIME_S:-9.0}
STEP_ALPHA_SCALE=${STEP_ALPHA_SCALE:-0.6}
PATH_LENGTH_M=${PATH_LENGTH_M:-26.0}
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
say "plant           : alpha $ALPHA_RANGE, x${STEP_ALPHA_SCALE} at ${STEP_TIME_S}s"
say "scored window   : ${SCORED_S}s (truck reaches ${PATH_LENGTH_M}m at ${SPEED} km/h; the rollout ends there, NOT at the 47.3s episode length)"
say "freeze tick     : $FREEZE_TICK model ticks = $(awk "BEGIN{printf \"%.1f\", $FREEZE_TICK/20}")s (plant steps at ${STEP_TIME_S}s)"
say ""

# `zero` is NOT a neutral latent value. Measured on A4's trace: the two
# channels have mean -1.193/-2.344 and every sample is <= 0, so zero sits 2.0
# and 3.4 standard deviations out and at the extreme edge of the range. The
# zero arms are therefore OUT-OF-DISTRIBUTION arms, not "no latent" arms, and
# A4-vs-A1 cannot be read as "does the latent help". C_mean supplies the
# in-distribution no-plant-information control that comparison needs.
#
# C_sham_step repeats A4 with step_alpha_scale=1.0: the schedule fires and the
# plant does not change. Without it, "z_hat moved 0.31 across the step" is not
# evidence of tracking -- the latent drifts as the trajectory evolves anyway,
# and the sham arm is what separates the two.
#
# name             policy      controller  freeze          alpha_scale
ARMS=(
  "A1_baseline     zero        zero        0               $STEP_ALPHA_SCALE"
  "A2_id_only      zero        estimated   0               $STEP_ALPHA_SCALE"
  "A3_ppo_only     estimated   zero        0               $STEP_ALPHA_SCALE"
  "A4_full         estimated   estimated   0               $STEP_ALPHA_SCALE"
  "C_shuffled      shuffled    shuffled    0               $STEP_ALPHA_SCALE"
  "C_frozen        frozen      frozen      $FREEZE_TICK    $STEP_ALPHA_SCALE"
  "C_mean          mean        mean        0               $STEP_ALPHA_SCALE"
  "C_sham_step     estimated   estimated   0               1.0"
)

for row in "${ARMS[@]}"; do
    read -r NAME POLICY CONTROLLER FRZ ASCALE <<<"$row"
    if [ -s "$OUT/${NAME}_gates.json" ]; then
        say "reusing $NAME"
        continue
    fi
    say "--- $NAME  (policy=$POLICY controller=$CONTROLLER freeze=$FRZ alpha_scale=$ASCALE)"
    FREEZE_ARG=""
    [ "$FRZ" != "0" ] && FREEZE_ARG="$FRZ"
    TASK=RRLAB-Mulag-Verge-ID-AAM-FD-v0 \
    MESHES="$MESHES" SPEED_KMH="$SPEED" STAGE=6 SEED="$SEED" \
    REPORT_ROOT="$OUT" TRACE_DIR="$TRACES" \
    LATENT_MODE="$POLICY" CONTROLLER_LATENT_MODE="$CONTROLLER" \
    FREEZE_TICK="$FREEZE_ARG" \
    PLANT_ALPHA_RANGE="$ALPHA_RANGE" PLANT_STEP_TIME_S="$STEP_TIME_S" \
    PLANT_STEP_ALPHA_SCALE="$ASCALE" \
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
    --dir "$OUT" --out "$OUT/phase4_summary.json" 2>&1 | tee -a "$LOG"

say ""
say "traces in $TRACES; adaptation figures:"
say "  \$PY PPO_ID_AAM_FD/experiments/analyze_adaptation_trace.py \\"
say "      --trace estimated=$TRACES/A4_full_m909.npz \\"
say "      --trace frozen=$TRACES/C_frozen_m909.npz \\"
say "      --trace zero=$TRACES/A1_baseline_m909.npz"
say "done"
