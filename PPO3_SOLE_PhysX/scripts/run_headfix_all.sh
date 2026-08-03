#!/usr/bin/env bash
# End-to-end head-fix run: train both actuators through the amplitude ramp,
# then measure adaptivity across SIX independently generated chunk layouts and
# three path families, and re-run the head-frame probe on the result.
#
# Preflight is enforced inside run_headfix_training.sh; it refuses to start if
# any entry point is missing a flag, a width-changing flag is applied after the
# width is derived, or a seed checkpoint disagrees with its recorded width.
set -u

cd "$(dirname "$0")/../.." || exit 1
PY=/home/rrlab/miniconda3/envs/env_isaaclab/bin/python

for act in rom idfd; do
  bash PPO3_SOLE_PhysX/scripts/run_headfix_training.sh "$act" \
    || echo "[$act HEADFIX STOPPED]"
  echo "[$act HEADFIX DONE]"
done

OUT=PPO3_SOLE_PhysX/reports/stage6/headfix
mkdir -p "$OUT"

for pair in "rom:logs/rsl_rl/ppo3_sole_rom_headfix_v1/model_best.pt:ppo3" \
            "idfd:logs/rsl_rl/ppo4_sole_idfd_headfix_v1/model_best.pt:ppo4"; do
  act=${pair%%:*}; rest=${pair#*:}; ckpt=${rest%%:*}; tag=${rest##*:}
  [ -f "$ckpt" ] || { echo "[skip] no $ckpt"; continue; }
  # Adaptivity: the SAME checkpoint across layouts it has never seen. Mesh
  # 1337 is the pinned gate mesh; the other five are fresh draws.
  for mesh in 1337 909 4242 17 2024 555; do
    for fam in mixed boom_sweep s_curve; do
      r="$OUT/${tag}_${fam}_m${mesh}.json"
      [ -f "$r" ] && { echo "[SKIP] $tag $fam m$mesh"; continue; }
      echo "[RUN] $tag $fam m$mesh"
      $PY PPO3_SOLE_PhysX/scripts/evaluate.py "$ckpt" --actuator "$act" \
        --headless --device cuda:0 --num_envs 32 --seed 17 --steps 960 \
        --stage 6 --eval_terrain --eval_terrain_seed "$mesh" --path_family "$fam" \
        --tracking_amplitude 2.2 \
        --contact_observation true --contact_safety_reward true \
        --head_frame_observation true --head_tilt_arm_frame true \
        --contact_latch true --contact_latch_release_steps 60 \
        --model_error_history 3 --action_scale 0.0035 \
        --cylinder_target_max_lead_m 0.018 --arm_stiffness_scale 0.35 \
        --strict_contact --report "$r" > "$OUT/${tag}_${fam}_m${mesh}.log" 2>&1
    done
  done
  # Did the frame fix actually change the mechanism, or only the score?
  $PY PPO3_SOLE_PhysX/scripts/probe_head_frame.py \
    --checkpoint "$ckpt" --actuator "$act" --headless --device cuda:0 \
    --num_envs 32 --steps 960 --stage 6 --eval_terrain \
    --path_family mixed --tracking_amplitude 2.2 \
    --report "$OUT/${tag}_headframe_after.json" \
    > "$OUT/${tag}_headframe_after.log" 2>&1
done
echo "[HEADFIX ALL DONE]"
