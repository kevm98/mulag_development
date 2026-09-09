#!/usr/bin/env bash
# Starved-encoder adaptation campaign.
# Executes reports/PREREG_STARVED_2026-09-03.md exactly as written.
#
# Two changes from the 2026-09-02 campaign and only two:
#   1. --history_cols plant   (30 features/step instead of 70)
#   2. constant per-episode cutter mass, no mid-episode flip
# Both residual arms are scored at the SAME final checkpoint (model_399).
#
# One mesh per evaluator call. Three meshes in one call peaked at 29 of 31 GB
# and did not finish; the per-mesh layout is also what
# PPO_AAM_FD/experiment_scripts/triplet.py already reads.
set -u
export IMAGE=isaac-lab:2.3.2-rl052
LSTD=0.09
LW=4.0
SEEDS=${SEEDS:-"17 23 101 42 7"}
MESHES=${MESHES:-"2718 3141 8191"}
BASE=logs/rsl_rl/ppo_id_aam_fd/distilled_v1/phase1_init_70.pt
cd /home/kevinmurera

# Constant per-episode mass over the range the base is MEASURABLY sensitive to
# (1-5x; the manifest's own 0.8-1.5 costs a fixed policy nothing). No MASS_STEP.
ENVS="
export MULAG_DIRECT_VALVE=1
export MULAG_RMA_CONTEXT=1
export MULAG_PAYLOAD_RANGE=1.0,5.0
export MULAG_PAYLOAD_RESAMPLE=1
"

run() { ~/bin/isaaclab-run bash -lc "$1" 2>/dev/null || true; }

mkdir -p /home/kevinmurera/projects/rrlab_mulag/outputs/starved

# ---- Phase 0: the pre-registered gate --------------------------------------
# The frozen base must actually degrade with cutter mass. One mesh is enough to
# read a slope; this is a sensitivity check, not a gated comparison.
echo "=== PHASE 0 gate: base sensitivity to constant cutter mass ==="
for M in 1.0 3.0 5.0; do
  run "
cd /workspace/projects/rrlab_mulag
export MULAG_DIRECT_VALVE=1
export MULAG_RMA_CONTEXT=1
export MULAG_PAYLOAD_RANGE=$M,$M
/isaac-sim/python.sh PPO_AAM_FD/scripts/evaluate_gates.py $BASE \
  --num_envs 176 --steps 900 --meshes 2718 --seed 17 \
  --report outputs/starved/gate_base_payload${M}.json 2>&1 | grep -E 'wrote|rror' | tail -2
"
  echo "GATE_DONE_payload${M}"
done
echo "=== PHASE 0 COMPLETE ==="

# ---- Base arm under the campaign plant -------------------------------------
for MESH in $MESHES; do
  run "
cd /workspace/projects/rrlab_mulag
$ENVS
/isaac-sim/python.sh PPO_AAM_FD/scripts/evaluate_gates.py $BASE \
  --num_envs 176 --steps 900 --meshes $MESH --seed 17 \
  --report outputs/rif_delay/starvedbase_m${MESH}.json 2>&1 | grep -E 'wrote|rror' | tail -2
"
  echo "SCORED_starvedbase_m${MESH}"
done
echo "=== BASE ARM COMPLETE ==="

# ---- Phase 1+2: five seeds, two trained arms each --------------------------
for SEED in $SEEDS; do
  for ARM in adaptive control; do
    # NOTE the flag asymmetry, which control_arm.sh warns about and which this
    # driver originally got wrong: training uses --no_history, scoring uses
    # --residual_no_history. Both must be set together for the control arm.
    if [ "$ARM" = control ]; then
      EXTRA="--no_history"; SCORE_EXTRA="--residual_no_history"
    else
      EXTRA=""; SCORE_EXTRA=""
    fi
    TAG=starved${SEED}_${ARM}
    ~/bin/isaaclab-run bash -lc "
set -u
cd /workspace/projects/rrlab_mulag
$ENVS
/isaac-sim/python.sh PPO_AAM_FD/scripts/train_residual_adapter.py $BASE \
  --num_envs 256 --iterations 400 --history 20 --latent_dim 8 --alpha 0.5 \
  --lr 3e-4 --seed $SEED --history_cols plant $EXTRA \
  --lateral_tracking_std_m $LSTD --lateral_tracking_weight $LW \
  --log_dir logs/rsl_rl/ppo_id_aam_fd/${TAG} \
  --headless 2>&1 | tee outputs/starved/${TAG}.log | tail -2
" || true
    echo "TRAINED_${TAG}"

    for MESH in $MESHES; do
      run "
cd /workspace/projects/rrlab_mulag
$ENVS
/isaac-sim/python.sh PPO_AAM_FD/scripts/evaluate_gates.py $BASE \
  --residual_checkpoint logs/rsl_rl/ppo_id_aam_fd/${TAG}/model_399.pt \
  --residual_alpha 0.5 --residual_history 20 $SCORE_EXTRA \
  --num_envs 176 --steps 900 --meshes $MESH --seed 17 \
  --report outputs/rif_delay/${TAG}_m${MESH}.json 2>&1 | grep -E 'wrote|rror' | tail -2
"
      echo "SCORED_${TAG}_m${MESH}"
    done
  done
  echo "SEED_DONE_${SEED}"
done
echo STARVED_CAMPAIGN_DONE
