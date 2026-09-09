#!/usr/bin/env bash
set -uo pipefail
# E3 -- mid-episode plant change, scored on all three control stacks.
#
# Chapter 7, H4. The causal demonstration the title rests on: the plant changes
# at a known time inside the scored window, and the question is whether the
# adaptive stack recovers where the non-adaptive ones cannot.
#
# WHAT THE ARMS ARE
#   The latent is SUBSTITUTED only in PPO-(AAM)-ID-FD, the only stack that has
#   one. PPO-ROM and PPO-ID-FD are scored under the same schedule as fixed
#   reference rows, which is what makes an ablation readable: a shuffled latent
#   that drops the adaptive stack to PPO-ID-FD's level says something a
#   within-stack comparison cannot.
#
#   estimated   the deployed system
#   frozen      latches a PRE-change estimate; identical to estimated up to the
#               latch, and differs only in that it stops tracking. Beating THIS
#               is the online-adaptation result.
#   shuffled    same latent distribution, wrong plant -- rules out "extra input
#               dimensions help"
#   mean        in-distribution, no plant information. `zero` is NOT used as the
#               no-information control: measured on a real trace both channels
#               are <= 0 with means -1.19/-2.34, so zero sits 2-3 s.d. outside
#               the distribution and is an OUT-OF-DISTRIBUTION arm.
#   sham        estimated, with the schedule firing but the plant NOT changing.
#               Without it, "z_hat moved across the step" is not evidence of
#               tracking: the latent drifts as the trajectory evolves anyway.
#
# TWO THINGS THAT MAKE THIS MEASURE ANYTHING
#   factor_domain=combined_test. Under `nominal` the latent is constant and
#   every arm collapses to the same thing.
#   The step must land inside the SCORED window, which ends when the truck
#   reaches 26 m -- NOT at the 47.3 s episode length. The guards below refuse a
#   configuration where the step never fires.
#
# NOT MATCHED ACROSS STACKS: the perturbation mechanism differs by actuator.
# PPO-ID-FD and PPO-(AAM)-ID-FD share the FD plant and take a valve-effectiveness
# step; PPO-ROM's actuator is the command ROM and takes a ROM factor step. The
# rows are matched on task and schedule, not on the physical parameter changed.

ROOT=$HOME/rrlab_mulag
cd "$ROOT"
PY=$HOME/miniconda3/envs/env_isaaclab/bin/python

OUT=${OUT:-$ROOT/outputs/e3_plant_change}
export MESHES=${MESHES:-"1337 17 909"}
export SPEED_KMH=${SPEED_KMH:-4.0}
export PARALLEL_MESHES=${PARALLEL_MESHES:-3}
export STALL=${STALL:-420}
export STAGE=6
export REPORT_ROOT="$OUT"
export FACTOR_DOMAIN=${FACTOR_DOMAIN:-combined_test}
ENVS=${ENVS:-32}
STEPS=${STEPS:-2808}
AAM_CKPT=${AAM_CKPT:-$ROOT/logs/rsl_rl/ppo_id_aam_fd/headroom_priced_v1/model_299.pt}
ROM_CKPT=$ROOT/logs/rsl_rl/ppo3_1_sole/rom_s6_aligned/model_598.pt
IDFD_CKPT=$ROOT/logs/rsl_rl/ppo4_1_sole/idfd_s6_aligned/model_598.pt

ALPHA_RANGE=${ALPHA_RANGE:-0.7:1.0}
STEP_TIME_S=${STEP_TIME_S:-9.0}
STEP_ALPHA_SCALE=${STEP_ALPHA_SCALE:-0.6}
FREEZE_TICK=${FREEZE_TICK:-140}     # 20 Hz MODEL ticks, not 60 Hz control steps
PATH_LENGTH_M=${PATH_LENGTH_M:-26.0}
mkdir -p "$OUT"
LOG=$OUT/e3.log
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

# --- the step and the latch must land inside the scored window -------------
SCORED_S=$(awk "BEGIN{printf \"%.2f\", $PATH_LENGTH_M/($SPEED_KMH/3.6)}")
FREEZE_S=$(awk "BEGIN{printf \"%.2f\", $FREEZE_TICK/20.0}")
awk "BEGIN{exit !($STEP_TIME_S < 0.75*$SCORED_S)}" || { echo "step at ${STEP_TIME_S}s leaves too little of the ${SCORED_S}s scored window" >&2; exit 2; }
awk "BEGIN{exit !($STEP_TIME_S > 0.25*$SCORED_S)}" || { echo "step at ${STEP_TIME_S}s leaves too little PRE-step window" >&2; exit 2; }
awk "BEGIN{exit !($FREEZE_S < $STEP_TIME_S)}" || { echo "freeze latch ${FREEZE_S}s is not BEFORE the step at ${STEP_TIME_S}s" >&2; exit 2; }
awk "BEGIN{exit !($FREEZE_S > 1.0)}" || { echo "freeze latch ${FREEZE_S}s is inside the 1.0s AAM warmup" >&2; exit 2; }

say "E3: domain=$FACTOR_DOMAIN speed=${SPEED_KMH}km/h scored=${SCORED_S}s"
say "    plant alpha $ALPHA_RANGE, x${STEP_ALPHA_SCALE} at ${STEP_TIME_S}s; latch ${FREEZE_S}s"
say "    aam ckpt: $(basename "$(dirname "$AAM_CKPT")")/$(basename "$AAM_CKPT")"

run() { # name ckpt task policy_mode ctrl_mode freeze alpha_scale
  local name=$1 ckpt=$2 task=$3 pol=$4 ctl=$5 frz=$6 asc=$7
  local have=1
  for m in $MESHES; do [ -s "$OUT/${name}_m${m}.json" ] || have=0; done
  [ "$have" = 1 ] && { say "reusing $name"; return; }
  local t0=$(date +%s)
  say "start $name"
  env TASK="$task" \
      ${pol:+LATENT_MODE="$pol"} ${ctl:+CONTROLLER_LATENT_MODE="$ctl"} \
      ${frz:+FREEZE_TICK="$frz"} \
      PLANT_ALPHA_RANGE="$ALPHA_RANGE" PLANT_STEP_TIME_S="$STEP_TIME_S" \
      PLANT_STEP_ALPHA_SCALE="$asc" \
      bash PPO_ROM/scripts/score_per_mesh.sh "$name" "$ckpt" "$STEPS" "$ENVS" \
      >>"$LOG" 2>&1 || say "nonzero exit for $name"
  say "done  $name in $(( $(date +%s) - t0 ))s"
}

# adaptive stack: the substituted arms
run aam_estimated "$AAM_CKPT" RRLAB-Mulag-Verge-ID-AAM-FD-v0 estimated estimated ""            "$STEP_ALPHA_SCALE"
run aam_frozen    "$AAM_CKPT" RRLAB-Mulag-Verge-ID-AAM-FD-v0 frozen    frozen    "$FREEZE_TICK" "$STEP_ALPHA_SCALE"
run aam_shuffled  "$AAM_CKPT" RRLAB-Mulag-Verge-ID-AAM-FD-v0 shuffled  shuffled  ""            "$STEP_ALPHA_SCALE"
run aam_mean      "$AAM_CKPT" RRLAB-Mulag-Verge-ID-AAM-FD-v0 mean      mean      ""            "$STEP_ALPHA_SCALE"
run aam_sham      "$AAM_CKPT" RRLAB-Mulag-Verge-ID-AAM-FD-v0 estimated estimated ""            1.0

# reference rows: same task, same schedule, no latent to substitute
run idfd_ref "$IDFD_CKPT" RRLAB-Mulag-Verge-IDFD-v0 "" "" "" "$STEP_ALPHA_SCALE"

say "all arms finished"
