#!/usr/bin/env bash
set -uo pipefail
# E1 -- closed-loop tracking error across the three control stacks.
#
# Chapter 7, H3. Everything in the thesis up to this point is model-level
# prediction error; this is the first measurement of CONTROL.
#
# Scoring goes through score_per_mesh.sh and NOT through evaluate_gates.py
# directly. The evaluator hangs when handed several meshes in one process --
# it completes the first mesh, re-creates the environment for the second and
# then spins on one core indefinitely. score_per_mesh.sh contains that with a
# one-mesh-per-process loop and a stall watchdog, and PARALLEL_MESHES=3 runs
# the three meshes concurrently (three 32-env evaluators hold ~1.4 GiB each
# against 6 GiB total).
#
# Fairness is by construction. Every arm is scored on the same meshes, the same
# trajectory families, the same pinned truck speed, the same scored window and
# the same evaluation seed. The seed fixes terrain and initial conditions, so a
# given seed presents all three stacks with an identical task; the comparison is
# paired at the seed and the per-seed sign count is reported with the mean.
#
# Checkpoints are declared, not defaulted:
#   PPO-ROM    rom_s6_aligned/model_598    seed 17, stage 6, terrain seed 17,
#   PPO-ID-FD  idfd_s6_aligned/model_598   legacy_arc, 4 km/h, freeze 150
#                                          -- verified an exactly matched pair
#   PPO-(AAM)-ID-FD  stage31_promotable_v2/model_375  the declared promotable
#                                          checkpoint (selection.json, no failing gates)
# The AAM run's manifest uses a different schema and does not record truck speed
# or trajectory shape, so its TRAINING conditions are not verifiably identical
# to the other two. Evaluation conditions are identical for all three; the
# training asymmetry is reported as a limitation, not hidden.

ROOT=/home/rrlab/rrlab_mulag
cd "$ROOT"
PY=/home/rrlab/miniconda3/envs/env_isaaclab/bin/python

OUT=${OUT:-$ROOT/outputs/e1_tracking}
export MESHES=${MESHES:-"1337 17 909"}
export SPEED_KMH=${SPEED_KMH:-4.0}
export PARALLEL_MESHES=${PARALLEL_MESHES:-3}
export STALL=${STALL:-300}
export STAGE=6
export REPORT_ROOT="$OUT"
ENVS=${ENVS:-32}
STEPS=${STEPS:-2808}
SEEDS=${SEEDS:-"17 101 202 303 404"}
mkdir -p "$OUT"
LOG=$OUT/e1.log

STACKS=(
  "ppo_rom|logs/rsl_rl/ppo3_1_sole/rom_s6_aligned/model_598.pt|"
  "ppo_id_fd|logs/rsl_rl/ppo4_1_sole/idfd_s6_aligned/model_598.pt|RRLAB-Mulag-Verge-IDFD-v0"
  "ppo_aam_id_fd|logs/rsl_rl/ppo_id_aam_fd/stage31_promotable_v2/model_375.pt|RRLAB-Mulag-Verge-ID-AAM-FD-v0"
)

say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
say "E1: meshes='$MESHES' speed=${SPEED_KMH}km/h envs=$ENVS parallel=$PARALLEL_MESHES seeds=($SEEDS)"

for seed in $SEEDS; do
  for row in "${STACKS[@]}"; do
    IFS='|' read -r name ckpt task <<<"$row"
    export SEED="$seed"
    export TASK="$task"
    done_all=1
    for m in $MESHES; do [ -s "$OUT/${name}_s${seed}_m${m}.json" ] || done_all=0; done
    [ "$done_all" = 1 ] && { say "reusing $name seed $seed"; continue; }
    [ -f "$ckpt" ] || { say "MISSING checkpoint $ckpt"; continue; }
    t0=$(date +%s)
    say "start $name seed $seed"
    bash PPO_ROM/scripts/score_per_mesh.sh "${name}_s${seed}" "$ckpt" "$STEPS" "$ENVS" \
      >>"$LOG" 2>&1 || say "score_per_mesh returned nonzero for $name seed $seed"
    say "done  $name seed $seed in $(( $(date +%s) - t0 ))s"
  done
done

say "scoring"
"$PY" "$ROOT/experiments/E1/code/summarise_e1.py" --root "$OUT" \
  --meshes "$MESHES" --out "$OUT/e1_summary.json" 2>&1 | tee -a "$LOG"
