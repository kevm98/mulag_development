#!/usr/bin/env bash
set -uo pipefail
# Re-select the PPO-(AAM)-ID-FD checkpoint on the FULL 26 m path.
#
# Why. select_checkpoint.sh defaults to STEPS=480. At 4 km/h that is 8 s, about
# 8.9 m of the 26 m scored path, so the deployed checkpoint (model_375) was
# promoted on the first third of the course. Scored over the whole path in E1 it
# holds ground contact of only 0.53 and 7 cm of cut-height error, against the
# 0.916 and 0.0069 recorded in its own selection report. The promotion gate does
# not see the part of the episode where the stack fails.
#
# This re-scores every checkpoint at STEPS=2808 and re-applies the same gate
# contract, so the selection is made on the horizon the thesis actually reports.
#
# Parallelism without touching the shared script: select_checkpoint.sh names its
# reports it{ITER}_m{MESH}.json, so three instances pinned to one mesh each write
# into the same directory without colliding. A fourth pass then sees all three
# meshes present, skips every report and only runs the ranking.

ROOT=$HOME/rrlab_mulag
cd "$ROOT"
RUN=${RUN:-$ROOT/logs/rsl_rl/ppo_id_aam_fd/stage31_promotable_v2}
OUT=${OUT:-$ROOT/outputs/aam_reselect_fullpath}
export REPORT_ROOT="$OUT"
export TASK=${TASK:-RRLAB-Mulag-Verge-ID-AAM-FD-v0}
export STEPS=${STEPS:-2808}
export ENVS=${ENVS:-32}
export STAGE=6
export STALL=${STALL:-420}
export REQUIRED_MESHES=${REQUIRED_MESHES:-3}
# Coarse-to-fine. Scoring all 24 checkpoints over the full path costs ~2 h of
# GPU; this pass spans the run at 8 points, including the incumbent 375 so the
# comparison against its own selection report is direct. Refine locally around
# the winner afterwards if the curve is not flat.
ITERS=${ITERS:-"100 175 250 325 375 450 525 599"}
mkdir -p "$OUT"
LOG=$OUT/reselect.log
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "re-selecting $(basename "$RUN") over the full path (STEPS=$STEPS, iters: $ITERS)"
pids=()
for MESH in 1337 17 909; do
  MESHES="$MESH" bash PPO_ROM/scripts/select_checkpoint.sh "$RUN" aam_fullpath \
      $ITERS > "$OUT/stream_m${MESH}.log" 2>&1 &
  pids+=($!)
  sleep 20   # stagger: three simultaneous Isaac startups thrash the GPU
done
for p in "${pids[@]}"; do wait "$p"; done
say "all mesh streams finished; ranking"

MESHES="1337 17 909" bash PPO_ROM/scripts/select_checkpoint.sh "$RUN" aam_fullpath \
    $ITERS 2>&1 | tee -a "$LOG"
