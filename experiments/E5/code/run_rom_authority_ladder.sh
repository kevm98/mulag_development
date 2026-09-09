#!/usr/bin/env bash
# ROM authority-degradation ladder -- the robustness curve and its breaking point.
#
# Sweeps AUTHORITY_ALPHA, which scales the realised cylinder increment right
# before it updates the PhysX target. It is not a hydraulic parameter; it is
# loss of realised actuator authority at the deployment interface, and it has a
# guaranteed endpoint: at alpha -> 0 the moving reference must become
# untrackable, or the harness is wrong. That endpoint is what makes a genuine
# breaking point measurable rather than a flat curve.
#
# REPEATS ARE NOT OPTIONAL. Measured 2026-08-28 on the L4: two runs of an
# identical config (same seed, same mesh) differ by ~10.8% in lateral error and
# ~3.7 pp in ground contact, because tiny floating-point differences in the GPU
# solver amplify chaotically over a long rollout -- divergence grows from 0.95%
# in the 0-3 m band to 13.7% by 23-26 m. Ground-contact noise alone is 74% of
# the 5 pp onset criterion, so a single run per level can manufacture a
# breaking point that is not there. Each level is therefore run REPEATS times
# and the summariser reports the spread; an effect only counts when it clears
# the nominal noise band.
#
#   bash experiments/E5/code/run_rom_authority_ladder.sh
#
#   ALPHAS    levels to sweep      (default: the ladder below)
#   REPEATS   runs per level       (default 2)
#   ENVS      envs per run         (default 32; more averages the chaos down)
#   STEPS     steps per run        (default 2808, the protocol value)
#   MESHES    terrain mesh         (default 1337)
#   FACTOR_DOMAIN  nominal|combined_test  (default nominal)
set -uo pipefail

ROOT=${ROOT:-$HOME/rrlab_mulag}
cd "$ROOT"
PY=${PY:-python3}

OUT=${OUT:-$ROOT/outputs/e5_rom_authority_ladder}
export MESHES=${MESHES:-1337}
export SPEED_KMH=${SPEED_KMH:-4.0}     # the trained speed; not the 2-7 default
export STAGE=6
export REPORT_ROOT="$OUT"
# Empty by default, and '-' not ':-' so an explicit empty value survives.
# Passing --factor_domain at all sets deployment_factor_enabled=True, which
# needs AAM_module/datasets/.../manifest.csv (not on the VM). Empty runs the
# nominal plant, which is also the cleaner baseline for this axis: vary
# authority alone and hold the plant fixed.
export FACTOR_DOMAIN=${FACTOR_DOMAIN-}
export PARALLEL_MESHES=1
export STALL=${STALL:-420}
export TRAJECTORY_SHAPES=${TRAJECTORY_SHAPES:-""}   # all families
ENVS=${ENVS:-32}
STEPS=${STEPS:-2808}
REPEATS=${REPEATS:-2}
# Dense near 1.0 to catch onset, dense near 0 to bracket breakdown.
ALPHAS=${ALPHAS:-"1.00 0.95 0.90 0.85 0.80 0.70 0.60 0.50 0.40 0.30 0.20 0.10 0.05 0.01 0.00"}
CKPT=${CKPT:-logs/rsl_rl/ppo3_1_sole/rom_s6_aligned/model_598.pt}

mkdir -p "$OUT"
LOG=$OUT/ladder.log
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "ROM authority ladder: alphas=[$ALPHAS] repeats=$REPEATS envs=$ENVS steps=$STEPS"
say "domain=$FACTOR_DOMAIN speed=${SPEED_KMH}km/h mesh=$MESHES"

total=0; ok=0
for alpha in $ALPHAS; do
  for r in $(seq 1 "$REPEATS"); do
    tag="rom_a${alpha}_r${r}"
    total=$((total+1))
    # Reuse only when EVERY requested mesh is already present: MESHES may hold
    # several, and "$OUT/${tag}_m${MESHES}.json" would otherwise expand to one
    # nonsense filename and silently rerun the lot.
    have_all=1
    for m in $MESHES; do
      [ -s "$OUT/${tag}_m${m}.json" ] || have_all=0
    done
    if [ "$have_all" = 1 ]; then
      say "reusing $tag"; ok=$((ok+1)); continue
    fi
    t0=$(date +%s)
    say "start $tag"
    # Each repeat gets its own seed so the spread reflects both the solver
    # nondeterminism and the episode draw, not one of them alone.
    AUTHORITY_ALPHA="$alpha" SEED=$((17 + r - 1)) \
      bash PPO_ROM/scripts/score_per_mesh.sh "$tag" "$CKPT" "$STEPS" "$ENVS" \
      >>"$LOG" 2>&1 || say "nonzero exit for $tag"
    if [ -s "$OUT/${tag}_m${MESHES}.json" ]; then
      ok=$((ok+1)); say "done  $tag in $(( $(date +%s) - t0 ))s"
    else
      say "FAILED $tag -- no report"
    fi
  done
done

say "completed $ok/$total runs"
say ""
"$PY" experiments/E5/code/summarise_rom_ladder.py \
    --report-dir "$OUT" --mesh "$MESHES" \
    --out "$ROOT/experiments/E5/results/rom_authority_ladder.json" | tee -a "$LOG"
