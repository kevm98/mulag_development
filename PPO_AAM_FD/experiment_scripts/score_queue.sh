#!/usr/bin/env bash
# Serial scoring queue for the residual arms.
#
# WHY SERIAL (learned the hard way, 2026-08-31): one evaluate_gates.py at
# --num_envs 176 peaks at ~17 GB of HOST RAM. Two concurrent evals exceed the
# 31 GB box and the kernel OOM-kills one -- surfacing only as the generic
# "There was an error running python", with the GPU idle and plenty of VRAM
# free. Three parallel TRAININGS at 256 envs fit fine (~10 GB total), so
# training throughput says nothing about evaluation throughput. Do not
# parallelise this loop.
#
# Each job writes a full log to outputs/scorelogs/<tag>_m<mesh>.log so a failure
# can be diagnosed instead of grepped away. Existing reports are skipped, so the
# script is safe to re-run to fill gaps.
#
# Usage:  score_queue.sh <jobsfile>
#   jobsfile lines:  <tag> <residual_ckpt> <alpha> <mesh> [extra flags...]
#   lines beginning # are comments; PLANT=<name> selects the plant block.
set -u
export IMAGE=isaac-lab:2.3.2-rl052
cd /home/kevinmurera/projects/rrlab_mulag
mkdir -p outputs/scorelogs
BASE=logs/rsl_rl/ppo_id_aam_fd/distilled_v1/phase1_init_70.pt

plant_real="
export MULAG_DIRECT_VALVE=1
export MULAG_RMA_CONTEXT=1
export MULAG_PAYLOAD_RANGE=1.0,1.0
export MULAG_MASS_STEP_TIME_S=20.0
export MULAG_MASS_STEP_SCALE=3.0
export MULAG_MASS_STEP_JITTER_S=8.0
export MULAG_MASS_BIMODAL=1.0,3.0
"
# Sham: the flip schedule fires on the same jittered instants and writes the
# same mass, but both modes are 1.0 so the mass never actually moves.
plant_sham="
export MULAG_DIRECT_VALVE=1
export MULAG_RMA_CONTEXT=1
export MULAG_PAYLOAD_RANGE=1.0,1.0
export MULAG_MASS_STEP_TIME_S=20.0
export MULAG_MASS_STEP_SCALE=1.0
export MULAG_MASS_STEP_JITTER_S=8.0
export MULAG_MASS_BIMODAL=1.0,1.0
"
PLANT="$plant_real"

JOBS=${1:?usage: score_queue.sh <jobsfile>}
# Read the queue on fd 3, NOT stdin: isaaclab-run calls `docker run -i`, which
# attaches and drains stdin. With the loop reading stdin, docker eats the rest
# of the job list and the queue silently stops after the first job.
while read -r TAG CKPT ALPHA M EXTRA <&3; do
  [ -z "${TAG:-}" ] && continue
  case "$TAG" in \#*) continue;; esac
  if [ "$TAG" = "PLANT=sham" ]; then PLANT="$plant_sham"; echo "PLANT -> sham"; continue; fi
  if [ "$TAG" = "PLANT=real" ]; then PLANT="$plant_real"; echo "PLANT -> real"; continue; fi
  OUT=outputs/rif_delay/${TAG}_m${M}.json
  LOG=outputs/scorelogs/${TAG}_m${M}.log
  if [ -s "$OUT" ]; then echo "SKIP  $TAG m$M"; continue; fi
  for try in 1 2; do
    ~/bin/isaaclab-run bash -lc "
cd /workspace/projects/rrlab_mulag
$PLANT
/isaac-sim/python.sh PPO_AAM_FD/scripts/evaluate_gates.py $BASE \
  --residual_checkpoint $CKPT --residual_alpha $ALPHA --residual_history 20 ${EXTRA:-} \
  --num_envs 176 --steps 900 --meshes $M --seed 17 \
  --report $OUT
" < /dev/null > "$LOG" 2>&1 || true
    if [ -s "$OUT" ]; then echo "OK    $TAG m$M (try $try)  $(date -u +%H:%M)"; break; fi
    echo "RETRY $TAG m$M (try $try) -- see $LOG"
    tail -5 "$LOG" | sed 's/^/      | /'
    sleep 20
  done
  [ -s "$OUT" ] || echo "FAIL  $TAG m$M"
done 3< "$JOBS"
echo "QUEUE_DONE $JOBS"
