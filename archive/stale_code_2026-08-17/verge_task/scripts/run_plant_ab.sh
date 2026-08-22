#!/usr/bin/env bash
# Controlled A/B on the two plant fields the fix changes.
#
# Same checkpoint, same path family, same relief, same mesh, same seed, same
# horizon. ONLY `action_scale` and `cylinder_target_max_lead_m` differ. This is
# the run that lets "the plant change alone did X" be stated as a comparison
# rather than as two numbers that happen to sit next to each other.
#
# Both cells use the 96-D contact-safe observation, so the observation is NOT
# a variable either -- the seed checkpoint is a behaviour-preserving migration,
# so its extra columns carry zero weight and contribute nothing until trained.
set -u

cd "$(dirname "$0")/../.." || exit 1
PY=/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
OUT=${OUT:-verge_task/reports/contact_safe_v1/plant_ab}
mkdir -p "$OUT"

ENVS=${ENVS:-32}
STEPS=${STEPS:-960}
FAMILY=${FAMILY:-s_curve}
HEIGHT=${HEIGHT:-1.0}

cell() {
  local tag=$1 act=$2 ckpt=$3 scale=$4 lead=$5
  local report="$OUT/${tag}.json"
  if [ -f "$report" ]; then echo "[SKIP] $report"; return; fi
  echo "[RUN] $tag  action_scale=$scale lead=$lead"
  $PY verge_task/scripts/evaluate.py "$ckpt" \
    --actuator "$act" --headless --device cuda:0 \
    --num_envs "$ENVS" --seed 17 --steps "$STEPS" --stage 5 --eval_terrain \
    --path_family "$FAMILY" --terrain_height_scale "$HEIGHT" \
    --contact_observation true --contact_safety_reward true \
    --model_error_history 3 --strict_contact \
    --action_scale "$scale" --cylinder_target_max_lead_m "$lead" \
    --report "$report" > "$OUT/${tag}.log" 2>&1
  echo "[DONE] exit=$? -> $report"
}

PPO3_SEED=logs/rsl_rl/ppo3_sole_rom_contact_safe_v1/model_seed_96.pt
PPO4_SEED=logs/rsl_rl/ppo4_sole_idfd_contact_safe_v1/model_seed_96.pt

# A = the inherited plant, B = the chosen one. Nothing else differs.
cell ppo3_A_old_plant rom  "$PPO3_SEED" 0.0040 0.025
cell ppo3_B_new_plant rom  "$PPO3_SEED" 0.0035 0.010
cell ppo4_A_old_plant idfd "$PPO4_SEED" 0.0040 0.025
cell ppo4_B_new_plant idfd "$PPO4_SEED" 0.0035 0.010

echo "[PLANT AB DONE] $OUT"
