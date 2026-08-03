#!/usr/bin/env bash
# The queue, in order, with NOTHING edited while it runs.
#   1. PPO3 adaptivity: 6 unseen chunk layouts x 3 families
#   2. PPO4 q5 anti-windup sweep (measured, on the fixed checkpoint)
#   3. PPO4 retrain through the amplitude ramp with the windup fixed
#   4. PPO4 adaptivity on the result
set -u
cd "$(dirname "$0")/../.." || exit 1
PY=/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
OUT=PPO3_SOLE_PhysX/reports/stage6/headfix
mkdir -p "$OUT"

echo "########## 1. PPO3 adaptivity ##########"
CK=logs/rsl_rl/ppo3_sole_rom_headfix_v1/model_best.pt
for mesh in 1337 909 4242 17 2024 555; do
  for fam in mixed boom_sweep s_curve; do
    r="$OUT/ppo3_${fam}_m${mesh}.json"; [ -f "$r" ] && { echo "[SKIP] $fam m$mesh"; continue; }
    echo "[RUN] ppo3 $fam m$mesh"
    $PY PPO3_SOLE_PhysX/scripts/evaluate.py "$CK" --actuator rom \
      --headless --device cuda:0 --num_envs 32 --seed 17 --steps 960 \
      --stage 6 --eval_terrain --eval_terrain_seed "$mesh" --path_family "$fam" \
      --tracking_amplitude 2.2 --contact_observation true --contact_safety_reward true \
      --head_frame_observation true --head_tilt_arm_frame true \
      --contact_latch true --contact_latch_release_steps 60 \
      --model_error_history 3 --action_scale 0.0035 \
      --cylinder_target_max_lead_m 0.018 --arm_stiffness_scale 0.35 \
      --strict_contact --report "$r" > "$OUT/ppo3_${fam}_m${mesh}.log" 2>&1
  done
done
echo "[PPO3 ADAPTIVITY DONE]"

echo "########## 2. PPO4 q5 windup sweep ##########"
bash PPO3_SOLE_PhysX/scripts/run_q5_windup_sweep.sh || echo "[SWEEP STOPPED]"
echo "[Q5 SWEEP STAGE DONE]"
