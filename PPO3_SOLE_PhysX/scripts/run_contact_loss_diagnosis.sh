#!/usr/bin/env bash
# Phase 1 diagnostic matrix: reproduce and measure the cutter contact loss on
# BOTH actuators, on the held-out verge, at full relief, over the full horizon.
#
# Deliberately s_curve at height scale 1.0 -- that is where the failure is
# visible, and a mixed-family aggregate would average it away.
set -u

cd "$(dirname "$0")/../.." || exit 1
PY=/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
OUT=${1:-PPO3_SOLE_PhysX/reports/contact_safe_v1}
mkdir -p "$OUT"

PPO3_CKPT=logs/rsl_rl/ppo3_sole_terrain/shared_v2_rom/model_migrated.pt
PPO4_CKPT=logs/rsl_rl/ppo4_sole_idfd/shared_v2_idfd/model_validated.pt

STEPS=${STEPS:-960}
ENVS=${ENVS:-32}
FAMILY=${FAMILY:-s_curve}
HEIGHT=${HEIGHT:-1.0}

# Seed 17 with NO mesh override is the pinned gate mesh. The remaining seeds
# move the held-out MESH via --eval_terrain_seed, because --seed alone cannot:
# with the mesh pinned at 1337, the start phase fixed and the actor evaluated
# at its mean, a held-out rollout is deterministic and every --seed returns
# bit-identical numbers. That was measured, not assumed -- the first pass of
# this script ran seeds 17/909/1337/4242 and produced four identical reports.
for spec in "17:0" "909:909" "1337:1337" "4242:4242"; do
  seed=${spec%%:*}
  mesh=${spec##*:}
  for pair in "rom:$PPO3_CKPT:ppo3" "idfd:$PPO4_CKPT:ppo4"; do
    act=${pair%%:*}
    rest=${pair#*:}
    ckpt=${rest%%:*}
    tag=${rest##*:}
    report="$OUT/prefix_${tag}_${FAMILY}_seed${seed}.json"
    if [ -f "$report" ]; then
      echo "[SKIP] $report"
      continue
    fi
    echo "[RUN] $tag seed=$seed mesh=$mesh"
    $PY PPO3_SOLE_PhysX/scripts/probe_contact_loss.py \
      --checkpoint "$ckpt" --actuator "$act" \
      --headless --device cuda:0 \
      --num_envs "$ENVS" --seed "$seed" --steps "$STEPS" --stage 5 \
      --eval_terrain --eval_terrain_seed "$mesh" --path_family "$FAMILY" \
      --terrain_height_scale "$HEIGHT" \
      --report "$report" > "$OUT/prefix_${tag}_${FAMILY}_seed${seed}.log" 2>&1
    echo "[DONE] exit=$? -> $report"
  done
done
echo "[ALL DONE]"
