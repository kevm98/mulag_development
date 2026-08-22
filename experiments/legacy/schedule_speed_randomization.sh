#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/rrlab/rrlab_mulag
exec bash "$ROOT/experiments/schedule_speed_curriculum_aam_first.sh" "$@"
