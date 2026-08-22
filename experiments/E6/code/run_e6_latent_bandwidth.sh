#!/usr/bin/env bash
set -uo pipefail
# E6 -- latent BANDWIDTH sweep on one frozen checkpoint.
#
# WHY THIS EXISTS
# E4 found that the deployed estimate is the WORST of four latents in closed
# loop: it loses to a constant carrying no plant information and to a constant
# that is actively wrong about the plant. Chapter 7 reads that as the latent's
# time-variation disturbing the loop. That reading is not established by E4,
# because `mean` and `shuffled` differ from `estimated` in TWO respects at once
# -- how much plant information they carry, and how much they move. Either could
# be doing the work.
#
# E6 separates them by holding content and sweeping only bandwidth. Every arm
# below carries the SAME estimate; a one-pole low-pass on the POLICY path alone
# limits how fast it may move. The ID adapter keeps the full-rate estimate in
# every arm, which is also the split the thesis proposes as the fix: band-limit
# the latent where a feedback loop consumes it, leave it responsive where a
# one-step model consumes it.
#
#   bw_estimated   policy=estimated          unlimited bandwidth (deployed)
#   bw_lpf_5p0     policy=lpf   5.00 Hz      near-passthrough
#   bw_lpf_1p0     policy=lpf   1.00 Hz
#   bw_lpf_0p2     policy=lpf   0.20 Hz
#   bw_lpf_0p04    policy=lpf   0.04 Hz      tau = 4 s, nearly constant
#   bw_frozen      policy=frozen@tick 40     zero bandwidth, CORRECT content
#   bw_mean        policy=mean               zero bandwidth, NO plant content
#
# READING IT
# The last two are the point of the design. They sit at the same end of the
# bandwidth axis and differ only in content, so:
#
#   monotone trend across the lpf series      -> bandwidth is doing the work
#   frozen ~ mean, both beating estimated     -> bandwidth, not content
#   frozen clearly better than mean           -> content matters after all
#   flat everywhere                           -> neither; the E4 ranking was noise
#
# A flat result is a real outcome and must be reported as one. It would say the
# thesis's "responsiveness" mechanism is unsupported, and Chapter 8's design
# recommendation would have to be withdrawn rather than softened.
#
# TRAPS
# * FACTOR_DOMAIN=combined_test is mandatory. Under `nominal` the plant does not
#   vary, the latent is near-constant, and every arm collapses to the same thing.
# * The controller arm stays `estimated` everywhere. Degrading it too would
#   reintroduce the confound E6 exists to remove.
# * FREEZE_TICK is in 50 ms MODEL ticks, not 60 Hz control steps. 40 = 2.0 s,
#   which is after the AAM's 20-row (1.0 s) warm-up, so the latch holds a real
#   estimate rather than a warm-up artefact.
# * No privileged arm here. It needs its own adapter and would change two things
#   at once, which is exactly what this experiment is built to avoid. E4 already
#   reports it.

ROOT=/home/rrlab/rrlab_mulag
cd "$ROOT"

OUT=${OUT:-$ROOT/outputs/e6_latent_bandwidth}
export MESHES=${MESHES:-"1337 17 909"}
export SPEED_KMH=${SPEED_KMH:-4.0}
# Serial. PARALLEL_MESHES is effectively a BOOLEAN, not a count: any value above
# 1 launches EVERY mesh at once, and three concurrent 32-env evaluators do not
# fit once the AAM and adapter are loaded on top of the policy.
export PARALLEL_MESHES=${PARALLEL_MESHES:-1}
export STALL=${STALL:-420}
export STAGE=6
export REPORT_ROOT="$OUT"
export FACTOR_DOMAIN=${FACTOR_DOMAIN:-combined_test}
ENVS=${ENVS:-32}
STEPS=${STEPS:-2808}
FREEZE_AT=${FREEZE_AT:-40}
AAM_CKPT=${AAM_CKPT:-$ROOT/logs/rsl_rl/ppo_id_aam_fd/headroom_priced_v1/model_299.pt}
TASK=RRLAB-Mulag-Verge-ID-AAM-FD-v0
mkdir -p "$OUT"
LOG=$OUT/e6.log
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "E6: domain=$FACTOR_DOMAIN speed=${SPEED_KMH}km/h envs=$ENVS steps=$STEPS"
say "    ckpt: $(basename "$(dirname "$AAM_CKPT")")/$(basename "$AAM_CKPT")"
say "    every arm shares this checkpoint; only the policy-path latent changes"

run() {  # name policy_mode [lpf_hz] [freeze_tick]
  local name=$1 pol=$2 hz=${3:-} tick=${4:-}
  local have=1
  for m in $MESHES; do [ -s "$OUT/${name}_m${m}.json" ] || have=0; done
  [ "$have" = 1 ] && { say "reusing $name"; return; }
  [ -f "$AAM_CKPT" ] || { say "MISSING $AAM_CKPT"; return; }
  local t0=$(date +%s)
  say "start $name  (policy=$pol${hz:+ @${hz}Hz}${tick:+ @tick$tick})"
  env TASK="$TASK" LATENT_MODE="$pol" CONTROLLER_LATENT_MODE=estimated \
      ${hz:+LPF_HZ="$hz"} ${tick:+FREEZE_TICK="$tick"} \
      bash PPO_ROM/scripts/score_per_mesh.sh "$name" "$AAM_CKPT" "$STEPS" "$ENVS" \
      >>"$LOG" 2>&1 || say "nonzero exit for $name"

  # Verify, then retry what is missing ONE MESH AT A TIME. A partially scored
  # arm is worse than a failed one: the summariser drops it for missing cells
  # and the gap reads as an arm that was never requested.
  local missing
  missing=$(for m in $MESHES; do [ -s "$OUT/${name}_m${m}.json" ] || echo "$m"; done)
  if [ -n "$missing" ]; then
    say "  $name incomplete, retrying serially: $(echo $missing | tr '\n' ' ')"
    for m in $missing; do
      env TASK="$TASK" LATENT_MODE="$pol" CONTROLLER_LATENT_MODE=estimated \
          ${hz:+LPF_HZ="$hz"} ${tick:+FREEZE_TICK="$tick"} \
          MESHES="$m" PARALLEL_MESHES=1 \
          bash PPO_ROM/scripts/score_per_mesh.sh "$name" "$AAM_CKPT" "$STEPS" "$ENVS" \
          >>"$LOG" 2>&1 || true
    done
  fi
  local got
  got=$(ls "$OUT/${name}"_m*.json 2>/dev/null | wc -l)
  say "done  $name in $(( $(date +%s) - t0 ))s  [$got/$(echo $MESHES | wc -w) meshes]"
}

# Cutoffs are stated as f_c; the sweep axis is the time constant
# tau = 1/(2*pi*f_c). The first grid sampled tau = 0.032, 0.159, 0.796, 3.98 s
# and jumped straight over 0.25-0.50 s, which is where an optimum was predicted
# and where the first grid's best arm (tau 0.796) sits closest. The two arms
# marked BAND below fill that hole and are the reason this grid exists.
run bw_estimated estimated
run bw_lpf_5p0   lpf 5.0       # tau 0.032
run bw_lpf_1p0   lpf 1.0       # tau 0.159
run bw_lpf_0p64  lpf 0.6366    # tau 0.25   BAND
run bw_lpf_0p32  lpf 0.3183    # tau 0.50   BAND
run bw_lpf_0p2   lpf 0.2       # tau 0.796
run bw_lpf_0p04  lpf 0.04      # tau 3.98
run bw_frozen    frozen "" "$FREEZE_AT"
run bw_mean      mean

say ""
say "arm coverage:"
incomplete=0
for name in bw_estimated bw_lpf_5p0 bw_lpf_1p0 bw_lpf_0p64 bw_lpf_0p32 \
            bw_lpf_0p2 bw_lpf_0p04 bw_frozen bw_mean; do
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
