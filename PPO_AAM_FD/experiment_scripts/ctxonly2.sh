#!/usr/bin/env bash
# Context-only conditioning, with a lateral kernel that actually has gradient.
#
# Run 1 (ctxonly17) gave, at payload 3.0x against its own zeroed-context twin:
#   cut     16.91 -> 3.66 mm    (+408.6%)
#   contact  0.803 -> 0.936
#   safety   0.098 -> 0.007     (14x fewer violations)
#   lateral 93.06 -> 509.40 mm  (-63.7%)
#
# The lateral collapse is an objective bug, not a property of the mechanism.
# Training-time lateral error ran 0.18-0.55 m against a Gaussian kernel of
# sigma = 0.09 m, so exp(-(e/sigma)^2) is 1e-7 to 1e-11 and its gradient with
# it: Episode_Reward/lateral came in around 5-10 against a possible ~3600, i.e.
# 0.2% of maximum. Lateral supplied no learning signal while cut and contact
# did, and the context columns optimised the axes that had gradient.
#
# memory.md records the same error inverted: sigma 0.300 was once "ten times
# wider than the error it was meant to correct" at 61 mm, and the fix to 0.09
# was calibrated there. On the 1-5x payload range the errors are 200-550 mm, so
# 0.09 is now far too narrow. A Gaussian's gradient is maximal at e = sigma/
# sqrt(2), so sigma = 0.25 puts the peak at 0.18 m, inside the range the policy
# actually occupies, and still gives useful gradient out to 0.6 m.
#
# One flag changes. Everything else is identical to run 1.
set -u
export IMAGE=isaac-lab:2.3.2-rl052
BASE=logs/rsl_rl/ppo_id_aam_fd/distilled_v1/phase1_init_70.pt
TAG=ctxonly17w
MESH=2718; ENVS_N=176; STEPS=900
PAYLOADS=${PAYLOADS:-"3.0 5.0 1.0"}
cd /home/kevinmurera
R=/home/kevinmurera/projects/rrlab_mulag
run() { ~/bin/isaaclab-run bash -lc "$1" 2>/dev/null || true; }
exec >>/home/kevinmurera/ctxonly2.log 2>&1
echo "=== ctxonly2 armed $(date -u +%FT%TZ) ==="
while ps -eo args | grep -qE "[c]txonly.sh"; do sleep 60; done
echo "run 1 finished $(date -u +%FT%TZ)"
mkdir -p $R/outputs/ctxonly2 $R/outputs/estimator2

echo "=== TRAIN ${TAG} (sigma 0.25) ==="; date -u +%FT%TZ
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
  --num_envs 256 --stage_iterations 400 --seed 17 \
  --lateral_tracking_std_m 0.25 --lateral_tracking_weight 4.0 \
  --log_dir logs/rsl_rl/ppo_id_aam_fd/${TAG} \
  --headless 2>&1 | tee outputs/ctxonly2/${TAG}.log | tail -3
"
echo "TRAINED_${TAG}"
C=$(ls $R/logs/rsl_rl/ppo_id_aam_fd/${TAG}/model_*.pt 2>/dev/null | sed 's/.*model_//;s/\.pt//' | sort -n | tail -1)
[ -z "$C" ] && { echo "CTXONLY2_ABORT no checkpoint"; exit 1; }
CKPT=logs/rsl_rl/ppo_id_aam_fd/${TAG}/model_${C}.pt
echo "scoring $CKPT"

score() { # name ablate plantenv out [dump]
  echo "=== SCORE $1 ==="; date -u +%FT%TZ
  DUMP=""; [ -n "${5:-}" ] && DUMP="--obs_dump $5 --obs_dump_every 10"
  # Wait out any other evaluator: one peaks near 17 GB and two OOM this box.
  while ps -eo args | grep -qE "[e]valuate_gates.py"; do sleep 45; done
  run "
cd /workspace/projects/rrlab_mulag
export MULAG_DIRECT_VALVE=1
export MULAG_RMA_CONTEXT=1
$3
export MULAG_CONTEXT_ABLATE=$2
/isaac-sim/python.sh PPO_AAM_FD/scripts/evaluate_gates.py $CKPT \
  --num_envs $ENVS_N --steps $STEPS --meshes $MESH --seed 17 $DUMP \
  --report $4 2>&1 | grep -E 'wrote|rror' | tail -2
"
  echo "SCORED_$1"
}
for M in $PAYLOADS; do
  P="export MULAG_PAYLOAD_RANGE=$M,$M"
  score "ctxzero_p${M}" zero "$P" "outputs/ctxonly2/ctxzero_p${M}_m${MESH}.json"
  score "ctxlive_p${M}" live "$P" "outputs/ctxonly2/ctxlive_p${M}_m${MESH}.json" "outputs/estimator2/ctxonly2_p${M}.npz"
done
MIX="export MULAG_PAYLOAD_RANGE=1.0,5.0
export MULAG_PAYLOAD_RESAMPLE=1"
score "ctxlive_mixed" live "$MIX" "outputs/ctxonly2/ctxlive_pmix_m${MESH}.json" "outputs/estimator2/ctxonly2_mixed.npz"
echo "CTXONLY2_SCORING_COMPLETE"
