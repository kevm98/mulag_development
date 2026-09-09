#!/usr/bin/env bash
set -uo pipefail
# E4 -- latent ablation in closed loop, with an oracle ceiling.
#
# E3 asked whether the stack recovers from a plant change and could not answer,
# because the change it applied did not degrade the task. E4 asks the prior and
# simpler question: with the plant varying BETWEEN episodes, does knowing which
# plant you are on help at all, and how much could it possibly help?
#
# No mid-episode perturbation is applied. The plant is drawn per episode from
# the held-out factor set, so a latent has genuine identity to convey, and the
# arms differ only in the quality of that identity.
#
#   estimated   the deployed system: the AAM infers the latent from history
#   privileged  the PRIVILEGED encoder, which reads the true factors directly.
#               NOTE the names, which are NOT the same on both switches:
#                 --online_aam_latent_mode            wants `privileged`
#                 --online_aam_controller_latent_mode wants `true`
#               One concept, two spellings. Passing either name to both switches
#               fails argument validation, the arm exits in about a minute, and
#               without the coverage check below the driver reports it as done.
#               This is the ORACLE and the point of the experiment: it bounds
#               what any estimator could achieve. If the oracle does not beat
#               `mean`, then no latent of any quality helps in closed loop and
#               the estimator's accuracy is not the limiting factor.
#   mean        in-distribution, no plant information -- the floor
#   shuffled    right distribution, wrong plant -- rules out "extra input
#               dimensions help"
#
# `zero` is deliberately excluded: measured on a real trace both latent channels
# are <= 0 with means -1.19/-2.34, so zero lies 2-3 s.d. outside the
# distribution and is an out-of-distribution arm rather than an ablation.
#
# PPO-ROM and PPO-ID-FD are scored under the identical factor domain so each arm
# can be read against what the stack degrades TO, not only against itself.
#
# FACTOR_DOMAIN=combined_test is mandatory. Under `nominal` the plant does not
# vary, the latent is constant, and every arm collapses to the same thing.

ROOT=$HOME/rrlab_mulag
cd "$ROOT"
PY=$HOME/miniconda3/envs/env_isaaclab/bin/python

OUT=${OUT:-$ROOT/outputs/e4_latent_ablation}
export MESHES=${MESHES:-"1337 17 909"}
export SPEED_KMH=${SPEED_KMH:-4.0}
# Serial. PARALLEL_MESHES in score_per_mesh.sh is effectively a BOOLEAN and not
# a count: any value above 1 launches EVERY mesh at once. Three concurrent
# 32-env evaluators no longer fit, because the AAM stack loads an AAM, a bounded
# adapter and a privileged encoder on top of the policy; the third stream dies
# with "Failed to allocate ... on device cuda:0" and its mesh goes missing.
# Setting 2 does not halve anything -- it only reads as though it does.
export PARALLEL_MESHES=${PARALLEL_MESHES:-1}
export STALL=${STALL:-420}
export STAGE=6
export REPORT_ROOT="$OUT"
export FACTOR_DOMAIN=${FACTOR_DOMAIN:-combined_test}
ENVS=${ENVS:-32}
STEPS=${STEPS:-2808}
AAM_CKPT=${AAM_CKPT:-$ROOT/logs/rsl_rl/ppo_id_aam_fd/headroom_priced_v1/model_299.pt}
ROM_CKPT=$ROOT/logs/rsl_rl/ppo3_1_sole/rom_s6_aligned/model_598.pt
IDFD_CKPT=$ROOT/logs/rsl_rl/ppo4_1_sole/idfd_s6_aligned/model_598.pt
mkdir -p "$OUT"
LOG=$OUT/e4.log
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "E4: domain=$FACTOR_DOMAIN speed=${SPEED_KMH}km/h envs=$ENVS"
say "    aam ckpt: $(basename "$(dirname "$AAM_CKPT")")/$(basename "$AAM_CKPT")"

run() {  # name ckpt task policy_mode controller_mode  [ADAPTER_CKPT in env]
  local ADAPTER_CKPT=${ADAPTER_CKPT:-}
  local name=$1 ckpt=$2 task=$3 pol=$4 ctl=$5
  local have=1
  for m in $MESHES; do [ -s "$OUT/${name}_m${m}.json" ] || have=0; done
  [ "$have" = 1 ] && { say "reusing $name"; return; }
  [ -f "$ckpt" ] || { say "MISSING $ckpt"; return; }
  local t0=$(date +%s)
  say "start $name"
  env TASK="$task" ${pol:+LATENT_MODE="$pol"} ${ctl:+CONTROLLER_LATENT_MODE="$ctl"} \
      ${ADAPTER_CKPT:+ADAPTER_CKPT="$ADAPTER_CKPT"} \
      bash PPO_ROM/scripts/score_per_mesh.sh "$name" "$ckpt" "$STEPS" "$ENVS" \
      >>"$LOG" 2>&1 || say "nonzero exit for $name"

  # Verify, then retry what is missing ONE MESH AT A TIME. A partially scored
  # arm is worse than a failed one: the summariser drops it for missing cells
  # and the gap reads as an arm that was never requested.
  local missing
  missing=$(for m in $MESHES; do [ -s "$OUT/${name}_m${m}.json" ] || echo "$m"; done)
  if [ -n "$missing" ]; then
    say "  $name incomplete, retrying serially: $(echo $missing | tr '\n' ' ')"
    for m in $missing; do
      env TASK="$task" ${pol:+LATENT_MODE="$pol"} ${ctl:+CONTROLLER_LATENT_MODE="$ctl"} \
          ${ADAPTER_CKPT:+ADAPTER_CKPT="$ADAPTER_CKPT"} \
          MESHES="$m" PARALLEL_MESHES=1 \
          bash PPO_ROM/scripts/score_per_mesh.sh "$name" "$ckpt" "$STEPS" "$ENVS" \
          >>"$LOG" 2>&1 || true
    done
  fi
  local got
  got=$(ls "$OUT/${name}"_m*.json 2>/dev/null | wc -l)
  say "done  $name in $(( $(date +%s) - t0 ))s  [$got/$(echo $MESHES | wc -w) meshes]"
}

run aam_estimated "$AAM_CKPT" RRLAB-Mulag-Verge-ID-AAM-FD-v0 estimated estimated
# The oracle arm needs its OWN adapter. The deployed bounded adapter is trained
# on estimated latents and refuses a privileged one -- "requires an adapter
# trained with latent_source='true'" -- because it would otherwise consume a
# distribution it never saw. This is unavoidable and means the oracle differs
# from the other arms in two respects, the latent source and the adapter
# weights; the chapter states that rather than presenting it as a pure swap.
ADAPTER_TRUE=${ADAPTER_TRUE:-$ROOT/outputs/stage31_rom/id_adapter_cnn_l20_s5_true/true_adapter.pt}
ADAPTER_CKPT="$ADAPTER_TRUE" \
  run aam_true      "$AAM_CKPT" RRLAB-Mulag-Verge-ID-AAM-FD-v0 privileged true
run aam_mean      "$AAM_CKPT" RRLAB-Mulag-Verge-ID-AAM-FD-v0 mean      mean
run aam_shuffled  "$AAM_CKPT" RRLAB-Mulag-Verge-ID-AAM-FD-v0 shuffled  shuffled
run rom_ref       "$ROM_CKPT" ""                              ""        ""
run idfd_ref      "$IDFD_CKPT" RRLAB-Mulag-Verge-IDFD-v0      ""        ""

say ""
say "arm coverage:"
incomplete=0
for name in aam_estimated aam_true aam_mean aam_shuffled rom_ref idfd_ref; do
  got=$(ls "$OUT/${name}"_m*.json 2>/dev/null | wc -l)
  want=$(echo $MESHES | wc -w)
  if [ "$got" -lt "$want" ]; then
    say "  $name  $got/$want  INCOMPLETE"
    incomplete=$((incomplete + 1))
  else
    say "  $name  $got/$want"
  fi
done
if [ "$incomplete" -gt 0 ]; then
  say "$incomplete arm(s) INCOMPLETE -- do not summarise this grid"
  exit 3
fi
say "all arms complete"
