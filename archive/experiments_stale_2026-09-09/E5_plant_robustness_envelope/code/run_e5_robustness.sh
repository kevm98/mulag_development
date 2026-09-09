#!/usr/bin/env bash
set -uo pipefail
# E5 -- robustness degradation curve across the three stacks.
#
# Reports a BREAKDOWN POINT rather than a pass/fail verdict: a gate that goes
# green tells you nothing about where a stack stops working.
#
# The sweep varies only the MAGNITUDE of the deployment-factor randomisation.
# Each environment keeps the draw it would have had, stretched away from the
# nominal plant along the same line, so the curve is not confounded with a
# change in which plants are visited. Scale 0 is the nominal plant, 1 is the
# manifest as drawn, 2 doubles every deviation.
#
# One mesh. The unit is the trajectory family, giving 11 cells per point, which
# resolves a trend across five points without the 3x cost of the full mesh set;
# the absolute values are therefore not comparable with the three-mesh tables
# elsewhere in the chapter.

ROOT=$HOME/rrlab_mulag
cd "$ROOT"
OUT=${OUT:-$ROOT/outputs/e5_robustness}
export MESHES=${MESHES:-1337}
export PARALLEL_MESHES=1
export SPEED_KMH=${SPEED_KMH:-4.0}
export STALL=${STALL:-420}
export STAGE=6
export REPORT_ROOT="$OUT"
export FACTOR_DOMAIN=${FACTOR_DOMAIN:-combined_test}
ENVS=${ENVS:-16}
STEPS=${STEPS:-2808}
SCALES=${SCALES:-"0.0 0.5 1.0 1.5 2.0"}
mkdir -p "$OUT"
LOG=$OUT/e5.log
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

STACKS=(
  "ppo_rom|logs/rsl_rl/ppo3_1_sole/rom_s6_aligned/model_598.pt|"
  "ppo_id_fd|logs/rsl_rl/ppo4_1_sole/idfd_s6_aligned/model_598.pt|RRLAB-Mulag-Verge-IDFD-v0"
  "ppo_aam_id_fd|logs/rsl_rl/ppo_id_aam_fd/headroom_priced_v1/model_299.pt|RRLAB-Mulag-Verge-ID-AAM-FD-v0"
)

say "E5: scales=($SCALES) domain=$FACTOR_DOMAIN mesh=$MESHES envs=$ENVS"
for sc in $SCALES; do
  for row in "${STACKS[@]}"; do
    IFS='|' read -r name ckpt task <<<"$row"
    tag="${name}_s${sc}"
    [ -s "$OUT/${tag}_m${MESHES}.json" ] && { say "reusing $tag"; continue; }
    t0=$(date +%s)
    say "start $tag"
    env TASK="$task" FACTOR_SCALE="$sc" \
        bash PPO_ROM/scripts/score_per_mesh.sh "$tag" "$ckpt" "$STEPS" "$ENVS" \
        >>"$LOG" 2>&1 || say "nonzero exit for $tag"
    [ -s "$OUT/${tag}_m${MESHES}.json" ] \
      && say "done  $tag in $(( $(date +%s) - t0 ))s" \
      || say "FAILED $tag -- no report"
  done
done

say ""
say "coverage:"
miss=0
for sc in $SCALES; do for row in "${STACKS[@]}"; do
  IFS='|' read -r name _ _ <<<"$row"
  [ -s "$OUT/${name}_s${sc}_m${MESHES}.json" ] || { say "  MISSING ${name} @ $sc"; miss=$((miss+1)); }
done; done
[ "$miss" -gt 0 ] && { say "$miss point(s) missing -- curve is incomplete"; exit 3; }
say "all points present"
