#!/usr/bin/env bash
# G1 degradation envelope for the RIF adaptation axis (RLA protocol, GPU item 5).
#
# Sweeps ONE actuator-boundary stress axis at a time on a frozen checkpoint and
# records the primary task error together with the saturation diagnostics, so a
# degradation can be separated from a command-authority collapse. The RLA
# analytic screen (PPO_RLA_ID_FD/memory.md) ranks the axes
#   delay -> deadband -> lag,  authority demoted,  rate_limit dropped
# because only delay buys action separation at no command-authority cost.
#
# Usage: rif_stress_sweep.sh <checkpoint> <axis> <values...>
#   axis: delay | alpha
set -Eeuo pipefail
ROOT=/workspace/projects/rrlab_mulag
CKPT="${1:?checkpoint}"; AXIS="${2:?axis}"; shift 2
for v in "$@"; do
  case "$AXIS" in
    delay) FLAG=(--plant_delay_steps_range "${v}:${v}") ;;
    alpha) FLAG=(--plant_alpha_range "${v}:${v}") ;;
    *) echo "unknown axis $AXIS" >&2; exit 2 ;;
  esac
  out="$ROOT/outputs/rif_delay/${AXIS}_${v}.json"
  echo "=== $AXIS=$v -> $out"
  /isaac-sim/python.sh "$ROOT/PPO_AAM_FD/scripts/evaluate_gates.py" "$CKPT" \
    --num_envs 32 --steps 600 --meshes 1337 --seed 17 \
    "${FLAG[@]}" --report "$out" >/dev/null 2>&1 || echo "  FAILED $AXIS=$v"
done
