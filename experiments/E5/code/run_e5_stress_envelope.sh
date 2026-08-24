#!/usr/bin/env bash
set -uo pipefail
# E5 (replacement) -- PLANT ROBUSTNESS ENVELOPE by ACTUATOR-AUTHORITY DEGRADATION.
#
# The superseded sweep (run_e5_robustness.sh) scaled the six-factor draw and
# found a flat curve. Two audits explain that flatness without reference to
# controller robustness, and both are reproducible without a GPU:
#
#   1. check_factor_reachability.sh / the runtime itself: the ID->FD plant takes
#      no factor argument, so on IF and AIF the four hydraulic factors are INERT
#      and the old sweep varied tool mass and COM height alone.
#   2. audit_factor_clamping.py: on R the hydraulic factors do act but are
#      clamped back to the identified range above scale 1 (13/16 held-out rows
#      at 1.5x, 16/16 at 2x), and the ROM's own response gain is bounded to
#      [0.35, 1.50].
#
# The factor draw is therefore not a common stress axis and cannot reach a
# breakdown. This experiment replaces it with one that can:
#
#   ACTUATOR-AUTHORITY DEGRADATION.  alpha scales the realised cylinder motion
#   the actuator produced, immediately before it updates the PhysX target:
#
#       R      : delta_x from the command ROM -> alpha * delta_x -> target
#       IF/AIF : delta_x from ID->FD          -> alpha * delta_x -> target
#
#   Same location, same meaning, all three stacks. It is NOT named after a
#   physical factor: it is not seal, not flow, not a hydraulic parameter. It is
#   loss of realised actuator authority at the deployment interface, and it has
#   an obvious endpoint -- at alpha -> 0 the moving reference must become
#   untrackable, or the harness is wrong.
#
# STAGES
#   STAGE_E5=0  authority knockout positive control   R, IF, AIF   ~6 episodes
#   STAGE_E5=C  clamp sentinel (audit figure)         R only       2 episodes
#   STAGE_E5=A  degradation discovery                 IF ONLY      ~11 episodes
#   STAGE_E5=B  frozen-level evaluation               R, IF, AIF   11 families
#
# SELECTION DISCIPLINE. Stage A runs IF alone and writes the frozen stress
# levels to stress_levels.json. Stage B refuses to run without that file and
# does not re-select. AIF is never inspected while the levels are chosen, so
# the stress condition cannot have been picked to flatter the adaptive stack.
#
# THRESHOLDS, declared here and not after the curve:
#   degradation onset   : median lateral error >= 1.5x its own alpha=1 value,
#                         or ground contact >= 5 percentage points below it
#   functional breakdown: ground contact < 0.700, or safety termination > 0.080

ROOT=${ROOT:-/home/rrlab/rrlab_mulag}
cd "$ROOT"
STAGE_E5=${STAGE_E5:-0}
OUT=${OUT:-$ROOT/outputs/e5_authority_stage${STAGE_E5}}
LEVELS=${LEVELS:-$ROOT/experiments/E5/results/stress_levels.json}
TRACES=${TRACES:-$OUT/traces}
export MESHES=${MESHES:-1337}
export PARALLEL_MESHES=${PARALLEL_MESHES:-1}
export SPEED_KMH=${SPEED_KMH:-4.0}
export STALL=${STALL:-420}
export STAGE=6                      # pipeline stage, not this script's stage
export REPORT_ROOT="$OUT"
export FACTOR_DOMAIN=${FACTOR_DOMAIN:-combined_test}
ENVS=${ENVS:-16}
STEPS=${STEPS:-2808}
mkdir -p "$OUT" "$TRACES" "$(dirname "$LEVELS")"
LOG=$OUT/e5_authority.log
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

IF_CKPT=logs/rsl_rl/ppo4_1_sole/idfd_s6_aligned/model_598.pt
IF_TASK=RRLAB-Mulag-Verge-IDFD-v0
R_CKPT=logs/rsl_rl/ppo3_1_sole/rom_s6_aligned/model_598.pt
R_TASK=""
AIF_CKPT=logs/rsl_rl/ppo_id_aam_fd/headroom_priced_v1/model_299.pt
AIF_TASK=RRLAB-Mulag-Verge-ID-AAM-FD-v0

score() {
  local tag=$1 ckpt=$2 task=$3
  [ -s "$OUT/${tag}_m${MESHES}.json" ] && { say "reusing $tag"; return 0; }
  local t0; t0=$(date +%s)
  say "start $tag"
  env TASK="$task" bash PPO_ROM/scripts/score_per_mesh.sh \
      "$tag" "$ckpt" "$STEPS" "$ENVS" >>"$LOG" 2>&1 || say "nonzero exit for $tag"
  [ -s "$OUT/${tag}_m${MESHES}.json" ] \
    && say "done  $tag in $(( $(date +%s) - t0 ))s" || say "FAILED $tag -- no report"
}

case "$STAGE_E5" in

0)
  # ---- Stage 0: authority knockout, the true positive control -------------
  # Not a robustness condition and not part of any operating envelope. Two
  # questions, in order: does the factor reach the realised motion (checked
  # numerically from the trace, not inferred from the path), and does the
  # cutter then lose the reference?
  say "STAGE 0 -- authority knockout on all three stacks"
  export TRACE_DIR="$TRACES"
  export TRAJECTORY_SHAPES=${TRAJECTORY_SHAPES:-"straight,double_s_curve"}
  for row in "ppo_rom|$R_CKPT|$R_TASK" "ppo_id_fd|$IF_CKPT|$IF_TASK" \
             "ppo_aam_id_fd|$AIF_CKPT|$AIF_TASK"; do
    IFS='|' read -r name ckpt task <<<"$row"
    for alpha in 1.00 0.01; do
      AUTHORITY_ALPHA="$alpha" score "${name}_a${alpha}" "$ckpt" "$task"
    done
  done
  say ""
  say "instrumentation check -- |delta_applied| / |delta_raw| must equal alpha:"
  python3 experiments/E5/code/check_authority_applied.py \
      --trace-dir "$TRACES" --mesh "$MESHES" --report-dir "$OUT" | tee -a "$LOG"
  ;;

C)
  # ---- Stage C: clamp sentinel -------------------------------------------
  # Two REQUESTS that map to the same internal hydraulic value, with every
  # other parameter identical -- mass and COM stay nominal because the inline
  # factor spec leaves unnamed factors at nominal. flow is clamped to
  # [-0.25, 0.25], so -0.40 and -0.80 both arrive as -0.25 and the two episodes
  # must agree to numerical tolerance. If they differ, the clamp is not where
  # the audit says it is.
  say "STAGE C -- clamp sentinel on R (audit evidence, 2 episodes)"
  export TRAJECTORY_SHAPES=${TRAJECTORY_SHAPES:-"double_s_curve"}
  for req in -0.40 -0.80; do
    FACTOR_KEY="flow=${req}" score "r_clampsentinel_${req}" "$R_CKPT" "$R_TASK"
  done
  # And the control for the control: a request INSIDE the range must differ.
  FACTOR_KEY="flow=-0.10" score "r_clampsentinel_-0.10" "$R_CKPT" "$R_TASK"
  say ""
  say "expected: -0.40 and -0.80 identical (both clamped to -0.25);"
  say "          -0.10 different from both (inside the range, delivered intact)."
  ;;

A)
  # ---- Stage A: degradation discovery, IF ONLY ----------------------------
  # Coarse scan first. Bisect around the knee only if the coarse grid brackets
  # it too loosely; each point is one episode per family, so the grid is cheap.
  say "STAGE A -- authority degradation curve, IF only, 3 families"
  export TRAJECTORY_SHAPES=${TRAJECTORY_SHAPES:-"straight,double_s_curve,lane_change_hold"}
  for alpha in ${ALPHAS:-1.00 0.90 0.80 0.70 0.60 0.50 0.40 0.30 0.20 0.10 0.01}; do
    AUTHORITY_ALPHA="$alpha" score "if_alpha_${alpha}" "$IF_CKPT" "$IF_TASK"
  done
  # Native mechanical stress, the one deployment factor that does reach every
  # stack unclamped. Kept as a second axis, not folded into the first.
  for m in ${MASSES:-1.00 1.25 1.50 1.75 2.00 2.50}; do
    FACTOR_KEY="mass_scale=$m" score "if_mass_${m}" "$IF_CKPT" "$IF_TASK"
  done
  say ""
  say "locating the knee and freezing the levels:"
  python3 experiments/E5/code/summarise_authority.py \
      --report-dir "$OUT" --mesh "$MESHES" --stack if --write-levels "$LEVELS" \
      | tee -a "$LOG"
  say ""
  say "Levels are now frozen in $LEVELS. Do not re-run Stage A after looking"
  say "at Stage B: the selection is only blind while it happens first."
  ;;

B)
  # ---- Stage B: frozen-level evaluation, all three stacks -----------------
  [ -s "$LEVELS" ] || { say "no frozen levels at $LEVELS -- run STAGE_E5=A first"; exit 3; }
  say "STAGE B -- evaluating R, IF and AIF at levels frozen by Stage A (IF only)"
  export TRAJECTORY_SHAPES=""      # all eleven families
  ALPHA_LIST=$(python3 -c "
import json,sys
levels = json.load(open('$LEVELS'))
print(' '.join(f'{a:.3f}' for a in levels['alpha_levels']))")
  say "frozen alphas: $ALPHA_LIST"
  for alpha in $ALPHA_LIST; do
    for row in "ppo_rom|$R_CKPT|$R_TASK" "ppo_id_fd|$IF_CKPT|$IF_TASK" \
               "ppo_aam_id_fd|$AIF_CKPT|$AIF_TASK"; do
      IFS='|' read -r name ckpt task <<<"$row"
      AUTHORITY_ALPHA="$alpha" score "${name}_a${alpha}" "$ckpt" "$task"
    done
  done
  say ""
  python3 experiments/E5/code/summarise_authority.py \
      --report-dir "$OUT" --mesh "$MESHES" --stack all | tee -a "$LOG"
  ;;

*)
  say "unknown STAGE_E5=$STAGE_E5 (expected 0, C, A or B)"; exit 2 ;;
esac

say "reports in $OUT"
