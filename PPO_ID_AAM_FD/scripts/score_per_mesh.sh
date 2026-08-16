#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/rrlab/rrlab_mulag
export TASK="${TASK:-RRLAB-Mulag-Verge-ID-AAM-FD-v0}"
export MESHES="${MESHES:-1337 17 2024 4242 555 909}"
export STAGE="${STAGE:-6}"
export STEPS="${STEPS:-2808}"
export ENVS="${ENVS:-32}"
export REPORT_ROOT="${REPORT_ROOT:-$ROOT/PPO_ID_AAM_FD/reports/score}"

exec bash "$ROOT/PPO_ROM/scripts/score_per_mesh.sh" "$@"
