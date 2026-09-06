#!/usr/bin/env bash
# Resume FiLM scoring. The TRAINING succeeded (model_1596, ~95 min of GPU); only
# the scoring failed, because a FiLM checkpoint carries mlp.film / mlp.head /
# mlp.tail keys and the evaluator built a plain actor to load them into. The
# evaluator now detects those keys in the checkpoint and wraps the actor before
# loading, so no retraining is needed.
set -u
export IMAGE=isaac-lab:2.3.2-rl052
cd /home/kevinmurera
R=/home/kevinmurera/projects/rrlab_mulag
run() { ~/bin/isaaclab-run bash -lc "$1" 2>/dev/null || true; }
waitgpu() { while ps -eo args | grep -qE "[e]valuate_gates.py"; do sleep 45; done; }
exec >>/home/kevinmurera/film2.log 2>&1
echo "=== film2 armed $(date -u +%FT%TZ) ==="
F=$(ls $R/logs/rsl_rl/ppo_id_aam_fd/film17/model_*.pt 2>/dev/null | sed 's/.*model_//;s/\.pt//' | sort -n | tail -1)
[ -z "$F" ] && { echo "FILM2_ABORT no checkpoint"; exit 1; }
FCK=logs/rsl_rl/ppo_id_aam_fd/film17/model_${F}.pt
echo "scoring $FCK"
mkdir -p $R/outputs/film $R/outputs/estimator_film

# Smoke the LOAD path first: a plain-actor load error is exactly what killed the
# first attempt, and it surfaces in three minutes here instead of after an hour.
waitgpu
echo "=== SMOKE film checkpoint load ==="; date -u +%FT%TZ
run "
cd /workspace/projects/rrlab_mulag
export MULAG_DIRECT_VALVE=1
export MULAG_RMA_CONTEXT=1
export MULAG_PAYLOAD_RANGE=3.0,3.0
export MULAG_CONTEXT_ABLATE=zero
/isaac-sim/python.sh PPO_AAM_FD/scripts/evaluate_gates.py $FCK \
  --num_envs 16 --steps 30 --meshes 2718 --seed 17 \
  --report outputs/film/smoke_load.json 2>&1 | grep -E 'FiLM checkpoint|wrote|rror|Traceback' | tail -4
"
[ -f "$R/outputs/film/smoke_load.json" ] || { echo "FILM2_ABORT load still broken"; exit 1; }
echo "FILM2_LOAD_OK $(date -u +%FT%TZ)"

waitgpu
run "
cd /workspace/projects/rrlab_mulag
export MULAG_DIRECT_VALVE=1
export MULAG_RMA_CONTEXT=1
export MULAG_PAYLOAD_RANGE=1.0,5.0
export MULAG_PAYLOAD_RESAMPLE=1
export MULAG_CONTEXT_ABLATE=live
/isaac-sim/python.sh PPO_AAM_FD/scripts/evaluate_gates.py $FCK \
  --num_envs 176 --steps 900 --meshes 2718 --seed 17 \
  --obs_dump outputs/estimator_film/film_mixed.npz --obs_dump_every 10 \
  --report outputs/film/mix.json 2>&1 | grep -E 'wrote|rror' | tail -2
"
run "
cd /workspace/projects/rrlab_mulag
/isaac-sim/python.sh PPO_AAM_FD/experiment_scripts/fit_estimator.py \
  --dir outputs/estimator_film --prefix film \
  --out outputs/estimator_film/film_fit.npz 2>&1 | tail -3
"
FFIT=/workspace/projects/rrlab_mulag/outputs/estimator_film/film_fit.npz
[ -f "$R/outputs/estimator_film/film_fit.npz" ] || { echo "FILM2_ABORT no fit"; exit 1; }
for SPEC in "p1.0:export MULAG_PAYLOAD_RANGE=1.0,1.0" \
            "p5.0:export MULAG_PAYLOAD_RANGE=5.0,5.0" \
            "cyc3:export MULAG_MASS_PERIOD_S=6.0
export MULAG_MASS_STEP_SCALE=3.0"; do
  NAME=${SPEC%%:*}; PLANT=${SPEC#*:}
  for ARM in ctxzero ctxest; do
    if [ "$ARM" = ctxest ]; then ABL=estimated; EX="export MULAG_ESTIMATOR_PATH=$FFIT
export MULAG_ESTIMATOR_BETA=0.02"; else ABL=zero; EX=""; fi
    waitgpu
    echo "=== film $ARM $NAME ==="; date -u +%FT%TZ
    run "
cd /workspace/projects/rrlab_mulag
export MULAG_DIRECT_VALVE=1
export MULAG_RMA_CONTEXT=1
$PLANT
export MULAG_CONTEXT_ABLATE=$ABL
$EX
/isaac-sim/python.sh PPO_AAM_FD/scripts/evaluate_gates.py $FCK \
  --num_envs 176 --steps 900 --meshes 2718 --seed 17 \
  --report outputs/film/${ARM}_${NAME}.json 2>&1 | grep -E 'wrote|rror' | tail -2
"
    echo "SCORED_film_${ARM}_${NAME}"
  done
done
echo "FILM_COMPLETE $(date -u +%FT%TZ)"
