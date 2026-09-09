#!/usr/bin/env bash
# SEED REPLICATION of the headline result.
#
# A single-seed claim is the first thing an examiner attacks, and rightly: RL
# runs vary a lot between seeds. E4 in this repo used 5 seeds for its headline
# and quoted the MEDIAN across them, which is the standard to match.
#
# Design: vary the TRAINING seed only. The evaluation plant, meshes and env
# count are held fixed, so what varies is training stochasticity -- the thing
# under question -- and not the test conditions.
#
# Each seed trains BOTH arms (history live / history zeroed) so the paired
# comparison is available within every seed, then all arms are scored through
# the validated gate path.
set -u
cd /tmp/rrlab_scratch
export IMAGE=isaac-lab:2.3.2-rl052

train_seed () {
  local SEED=$1 ARM=$2 EXTRA=${3:-}
  cd /home/kevinmurera
  ~/bin/isaaclab-run bash -lc "
set -u
cd /workspace/projects/rrlab_mulag
export MULAG_DIRECT_VALVE=1
export MULAG_RMA_CONTEXT=1
export MULAG_PAYLOAD_RANGE=1.0,1.0
export MULAG_MASS_STEP_TIME_S=20.0
export MULAG_MASS_STEP_SCALE=3.0
export MULAG_MASS_STEP_JITTER_S=8.0
export MULAG_MASS_BIMODAL=1.0,3.0
/isaac-sim/python.sh PPO_AAM_FD/scripts/train_residual_adapter.py \
  logs/rsl_rl/ppo_id_aam_fd/distilled_v1/phase1_init_70.pt \
  --num_envs 256 --iterations 400 --history 20 --latent_dim 8 --alpha 0.5 \
  --lr 3e-4 --seed $SEED $EXTRA \
  --log_dir logs/rsl_rl/ppo_id_aam_fd/seed${SEED}_${ARM} \
  --headless 2>&1 | tail -2
"
  echo "SEED_${SEED}_${ARM}_TRAINED"
}

score_seed () {
  local SEED=$1 ARM=$2 EXTRA=${3:-}
  local CK=logs/rsl_rl/ppo_id_aam_fd/seed${SEED}_${ARM}/model_399.pt
  cd /home/kevinmurera
  for M in 1337 17 4242; do
    ~/bin/isaaclab-run bash -lc "
cd /workspace/projects/rrlab_mulag
export MULAG_DIRECT_VALVE=1
export MULAG_RMA_CONTEXT=1
export MULAG_PAYLOAD_RANGE=1.0,1.0
export MULAG_MASS_STEP_TIME_S=20.0
export MULAG_MASS_STEP_SCALE=3.0
export MULAG_MASS_STEP_JITTER_S=8.0
export MULAG_MASS_BIMODAL=1.0,3.0
/isaac-sim/python.sh PPO_AAM_FD/scripts/evaluate_gates.py \
  logs/rsl_rl/ppo_id_aam_fd/distilled_v1/phase1_init_70.pt \
  --residual_checkpoint $CK --residual_alpha 0.5 --residual_history 20 $EXTRA \
  --num_envs 176 --steps 900 --meshes $M --seed 17 \
  --report outputs/rif_delay/s${SEED}_${ARM}_m${M}.json 2>&1 \
  | grep -E 'wrote|rror' | tail -2
" 2>/dev/null || true
    echo "SEED_${SEED}_${ARM}_M${M}_SCORED"
  done
}

# TRAINING ALREADY COMPLETE for all four seeds -- only the scoring failed, on a
# shell-escaping bug (\$M inside the container string became a literal, so
# --meshes got an empty argument). Re-running scoring only.
for SEED in 23 42 101 7; do
  score_seed "$SEED" history
  score_seed "$SEED" nohistory --residual_no_history
  echo "SEED_${SEED}_DONE"
done
echo SEEDS_ALL_DONE
