#!/usr/bin/env bash
set -euo pipefail

ROOT=$HOME/rrlab_mulag
PY=$HOME/miniconda3/envs/env_isaaclab/bin/python
TASK=RRLAB-Mulag-Verge-IDFD-v0

cd "$ROOT"
ARGS=("$@")
has_speed_domain=0
has_speed_sampling=0
for value in "${ARGS[@]}"; do
  case "$value" in
    --vehicle_speed_kmh|--vehicle_speed_kmh=*) has_speed_domain=1 ;;
    --vehicle_speed_min_kmh|--vehicle_speed_min_kmh=*|--vehicle_speed_max_kmh|--vehicle_speed_max_kmh=*) has_speed_domain=1 ;;
    --vehicle_speed_sampling|--vehicle_speed_sampling=*) has_speed_sampling=1 ;;
  esac
done
# Direct ID-FD launches must use the same teacher-student-style domain as the
# queued campaign. Explicit scalar/range arguments remain available for probes.
if [ "$has_speed_domain" -eq 0 ]; then
  ARGS+=(--vehicle_speed_min_kmh 2.0 --vehicle_speed_max_kmh 7.0)
fi
if [ "$has_speed_sampling" -eq 0 ]; then
  ARGS+=(--vehicle_speed_sampling uniform)
fi
exec "$PY" PPO_ROM/scripts/train.py --task "$TASK" "${ARGS[@]}"
