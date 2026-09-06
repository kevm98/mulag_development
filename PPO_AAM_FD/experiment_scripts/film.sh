#!/usr/bin/env bash
# Give the adaptation a second degree of freedom.
#
# Context columns pass cut in 12/12 configurations and lateral in 8/12. The four
# lateral failures are all where the frozen policy still holds high ground
# contact, and the cause is structural: four input columns can only slide the
# policy along one cut/contact-versus-lateral curve, so where there is no
# contact left to buy lateral back with, there is nothing to trade. An eightfold
# lateral price did not bend that curve and A-RMA only moved along it.
#
# FiLM sets a per-unit scale and shift on the first hidden representation, so
# the payload can act differently on each hidden unit rather than through one
# shared direction. 2560 trainable parameters against 59,588, zero-initialised
# so the wrapped policy is an exact identity at construction -- verified in the
# trainer, which aborts if it is not.
set -u
export IMAGE=isaac-lab:2.3.2-rl052
cd /home/kevinmurera
R=/home/kevinmurera/projects/rrlab_mulag
BASE=logs/rsl_rl/ppo_id_aam_fd/distilled_v1/phase1_init_70.pt
FIT=/workspace/projects/rrlab_mulag/outputs/estimator/ctxonly_fit.npz
run() { ~/bin/isaaclab-run bash -lc "$1" 2>/dev/null || true; }
waitgpu() { while ps -eo args | grep -qE "[e]valuate_gates.py"; do sleep 45; done; }
exec >>/home/kevinmurera/film.log 2>&1
echo "=== film armed $(date -u +%FT%TZ) ==="

echo "=== SMOKE: does --train_film construct, verify identity, and step? ==="
run "
cd /workspace/projects/rrlab_mulag
export MULAG_DIRECT_VALVE=1
export MULAG_RMA_CONTEXT=1
export MULAG_PAYLOAD_RANGE=1.0,5.0
export MULAG_PAYLOAD_RESAMPLE=1
export MULAG_CONTEXT_ABLATE=live
/isaac-sim/python.sh PPO_AAM_FD/scripts/train.py \
  --resume $BASE --train_film \
  --num_envs 64 --stage_iterations 2 --seed 17 \
  --lateral_tracking_std_m 0.09 --lateral_tracking_weight 4.0 \
  --log_dir logs/rsl_rl/ppo_id_aam_fd/film_smoke --headless 2>&1 | grep -E 'train_film|identity|Traceback|rror' | tail -5
"
if ! ls $R/logs/rsl_rl/ppo_id_aam_fd/film_smoke/model_*.pt >/dev/null 2>&1; then
  echo "FILM_SMOKE_FAILED -- nothing else run"; exit 1
fi
echo "FILM_SMOKE_OK $(date -u +%FT%TZ)"

echo "=== TRAIN film ==="; date -u +%FT%TZ
run "
set -u
cd /workspace/projects/rrlab_mulag
export MULAG_DIRECT_VALVE=1
export MULAG_RMA_CONTEXT=1
export MULAG_PAYLOAD_RANGE=1.0,5.0
export MULAG_PAYLOAD_RESAMPLE=1
export MULAG_CONTEXT_ABLATE=live
/isaac-sim/python.sh PPO_AAM_FD/scripts/train.py \
  --resume $BASE --train_film \
  --num_envs 256 --stage_iterations 400 --seed 17 \
  --lateral_tracking_std_m 0.09 --lateral_tracking_weight 4.0 \
  --log_dir logs/rsl_rl/ppo_id_aam_fd/film17 --headless 2>&1 | tail -3
"
F=$(ls $R/logs/rsl_rl/ppo_id_aam_fd/film17/model_*.pt 2>/dev/null | sed 's/.*model_//;s/\.pt//' | sort -n | tail -1)
[ -z "$F" ] && { echo "FILM_ABORT no checkpoint"; exit 1; }
FCK=logs/rsl_rl/ppo_id_aam_fd/film17/model_${F}.pt
echo "TRAINED_film at model_${F}"
mkdir -p $R/outputs/film $R/outputs/estimator_film

# This policy induces its own observation distribution, so it needs its own
# estimator; reusing the context-column run's fit would be a train/deploy gap.
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
[ -f "$R/outputs/estimator_film/film_fit.npz" ] || { echo "FILM_ABORT no fit"; exit 1; }

# The four configurations where context columns FAILED lateral, plus one that
# passed, so a fix cannot be claimed without showing it kept what worked.
for SPEC in "p1.0:export MULAG_PAYLOAD_RANGE=1.0,1.0" \
            "p5.0:export MULAG_PAYLOAD_RANGE=5.0,5.0" \
            "step3:export MULAG_MASS_STEP_TIME_S=6.0
export MULAG_MASS_STEP_JITTER_S=2.0
export MULAG_MASS_STEP_SCALE=3.0" \
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
