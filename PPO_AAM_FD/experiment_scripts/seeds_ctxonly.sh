#!/usr/bin/env bash
# Seed replication of the online result.
#
# Everything so far is n = 1 (seed 17). This project's own bar is a median over
# five seeds, and the starved campaign was held to it, so the online result is
# not comparable to previous campaigns until it is too.
#
# Per seed: train the 1024 context parameters, collect estimator data from a
# mixed-payload rollout of THAT seed's policy, fit its own estimator, then score
# ctxzero and ctxest at 3.0x and 5.0x. The estimator is refitted per seed on
# purpose: it regresses on the observation distribution the policy induces, and
# a different policy induces a different one, so reusing seed 17's fit would
# leak that seed's behaviour into the others.
#
# Roughly 2.3 h per seed. Runs last, behind everything else.
set -u
export IMAGE=isaac-lab:2.3.2-rl052
BASE=logs/rsl_rl/ppo_id_aam_fd/distilled_v1/phase1_init_70.pt
SEEDS=${SEEDS:-"23 101 42 7"}
MESH=2718; ENVS_N=176; STEPS=900
cd /home/kevinmurera
R=/home/kevinmurera/projects/rrlab_mulag
run() { ~/bin/isaaclab-run bash -lc "$1" 2>/dev/null || true; }
exec >>/home/kevinmurera/seeds.log 2>&1
echo "=== seeds armed $(date -u +%FT%TZ) ==="
while ps -eo args | grep -qE "[a]blate.sh|[d]eadband.sh|[c]txonly3.sh|[d]2rescore.sh"; do sleep 60; done
echo "queue clear $(date -u +%FT%TZ)"
mkdir -p $R/outputs/seeds $R/outputs/estimator_seeds

waitgpu() { while ps -eo args | grep -qE "[e]valuate_gates.py"; do sleep 45; done; }

for S in $SEEDS; do
  TAG=ctxonly${S}
  echo "=== TRAIN seed $S ==="; date -u +%FT%TZ
  run "
set -u
cd /workspace/projects/rrlab_mulag
export MULAG_DIRECT_VALVE=1
export MULAG_RMA_CONTEXT=1
export MULAG_PAYLOAD_RANGE=1.0,5.0
export MULAG_PAYLOAD_RESAMPLE=1
export MULAG_CONTEXT_ABLATE=live
/isaac-sim/python.sh PPO_AAM_FD/scripts/train.py \
  --resume $BASE --train_context_only \
  --num_envs 256 --stage_iterations 400 --seed $S \
  --lateral_tracking_std_m 0.09 --lateral_tracking_weight 4.0 \
  --log_dir logs/rsl_rl/ppo_id_aam_fd/${TAG} \
  --headless 2>&1 | tail -2
"
  C=$(ls $R/logs/rsl_rl/ppo_id_aam_fd/${TAG}/model_*.pt 2>/dev/null | sed 's/.*model_//;s/\.pt//' | sort -n | tail -1)
  if [ -z "$C" ]; then echo "SEED_${S}_ABORT no checkpoint"; continue; fi
  CKPT=logs/rsl_rl/ppo_id_aam_fd/${TAG}/model_${C}.pt
  echo "TRAINED_seed_${S} at model_${C}"

  # This seed's own estimator, from this seed's own policy.
  waitgpu
  run "
cd /workspace/projects/rrlab_mulag
export MULAG_DIRECT_VALVE=1
export MULAG_RMA_CONTEXT=1
export MULAG_PAYLOAD_RANGE=1.0,5.0
export MULAG_PAYLOAD_RESAMPLE=1
export MULAG_CONTEXT_ABLATE=live
/isaac-sim/python.sh PPO_AAM_FD/scripts/evaluate_gates.py $CKPT \
  --num_envs $ENVS_N --steps $STEPS --meshes $MESH --seed 17 \
  --obs_dump outputs/estimator_seeds/s${S}_mixed.npz --obs_dump_every 10 \
  --report outputs/seeds/ctxlive_mix_s${S}.json 2>&1 | grep -E 'wrote|rror' | tail -2
"
  run "
cd /workspace/projects/rrlab_mulag
/isaac-sim/python.sh PPO_AAM_FD/experiment_scripts/fit_estimator.py \
  --dir outputs/estimator_seeds --prefix s${S} \
  --out outputs/estimator_seeds/s${S}_fit.npz 2>&1 | tail -4
"
  if [ ! -f "$R/outputs/estimator_seeds/s${S}_fit.npz" ]; then
    echo "SEED_${S}_ABORT no fit"; continue
  fi

  for M in 5.0 3.0; do
    for ARM in ctxzero ctxest; do
      if [ "$ARM" = ctxest ]; then
        ABL=estimated
        EXTRA="export MULAG_ESTIMATOR_PATH=/workspace/projects/rrlab_mulag/outputs/estimator_seeds/s${S}_fit.npz
export MULAG_ESTIMATOR_BETA=0.02"
      else ABL=zero; EXTRA=""; fi
      waitgpu
      echo "=== seed $S $ARM payload $M ==="; date -u +%FT%TZ
      run "
cd /workspace/projects/rrlab_mulag
export MULAG_DIRECT_VALVE=1
export MULAG_RMA_CONTEXT=1
export MULAG_PAYLOAD_RANGE=$M,$M
export MULAG_CONTEXT_ABLATE=$ABL
$EXTRA
/isaac-sim/python.sh PPO_AAM_FD/scripts/evaluate_gates.py $CKPT \
  --num_envs $ENVS_N --steps $STEPS --meshes $MESH --seed 17 \
  --report outputs/seeds/${ARM}_p${M}_s${S}.json 2>&1 | grep -E 'wrote|rror' | tail -2
"
      echo "SCORED_s${S}_${ARM}_p${M}"
    done
  done
  echo "SEED_DONE_${S} $(date -u +%FT%TZ)"
done
echo "SEEDS_COMPLETE $(date -u +%FT%TZ)"
