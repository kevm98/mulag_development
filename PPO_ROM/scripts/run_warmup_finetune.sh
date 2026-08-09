#!/usr/bin/env bash
# PPO3_1: critic warm-up with a frozen actor, then a conservative fine-tune.
#
# WHY THESE SETTINGS
# ------------------
# Every prior PPO run descended its own reward (model850: +13.44 -> +3.24 over
# 850 iterations). The measured cause is not the reward and not entropy: with a
# cold critic, rsl_rl's advantage normalization leaves the surrogate gradient
# noise-dominated, value loss sits at 6k-26k, and the policy random-walks away
# from the distilled student. So the critic is trained FIRST, against the
# frozen student's own rollouts, before the actor is allowed to move.
#
#   --freeze_actor_iterations 150  critic-only window. The actor (mean net AND
#                                  exploration std) is held exactly at the
#                                  student's weights. The adaptive KL schedule
#                                  is suspended inside the window and the LR
#                                  reset at unfreeze -- otherwise a frozen
#                                  actor's tiny-but-nonzero KL ratchets the
#                                  shared LR x1.5 per minibatch to its 1e-2 cap.
#   --learning_rate 3e-4           conservative vs the 1e-3 that model850 used.
#   --desired_kl 0.008             tighter than the inherited 0.02, so the
#                                  adaptive schedule pulls the LR down sooner.
#   --entropy_coef 0.001           model850 descended at 0.005; 0.001 is also
#                                  what the teacher_student repo's rsl_rl cfg
#                                  uses. Probe-validated: std holds near 0.10.
#   --std_clamp 0.03,0.15          per-axis std ceiling (this rsl_rl has no
#                                  built-in clip_log_std). The student ships at
#                                  0.10 on all four axes; the entropy probe saw
#                                  lateral error grow 0.06 -> 0.59 m when std
#                                  was allowed to inflate to 0.31.
#
# NOT set, deliberately:
#   --reset_critic_head_on_resume  the checkpoint's source already triggers the
#                                  teacher-handoff path (actor+normalizer only,
#                                  fresh critic, zeroed head), which is what
#                                  the warm-up then trains. Passing the flag
#                                  would be redundant, not additive.
#   --until_converged              refused by train.py: it scores the legacy
#                                  26-gate contract, whose thresholds the DLS
#                                  teacher itself fails. Fixed budget instead.
#   --vehicle_speed_min/max_kmh    truck-speed randomization exists but is OFF.
#                                  Lateral rate demand scales linearly with
#                                  truck speed, so raising it makes the lateral
#                                  gate harder by construction, and varying it
#                                  shifts the task away from the fixed 4 km/h
#                                  the student was distilled at -- a confound
#                                  for the warm-up experiment this run IS.
#
# Acceptance is NOT this script's job. Score checkpoints with:
#   scripts/evaluate_gates.py <ckpt> --meshes 1337,17,909 --steps 480 \
#       --num_envs 32 --stage 5 --report <out.json> --headless
# against reports/frontier/{student,teacher,null}_gates.json.
set -euo pipefail

ROOT=/home/rrlab/rrlab_mulag
PY=/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
STAMP=$(date +%Y-%m-%d_%H-%M-%S)
LOG_DIR=${LOG_DIR:-$ROOT/logs/rsl_rl/ppo3_1_sole/warmup_$STAMP}

mkdir -p "$LOG_DIR"
cd "$ROOT"

exec "$PY" PPO_ROM/scripts/train.py \
    --resume "$ROOT/logs/rsl_rl/ppo3_1_sole/student_distilled.pt" \
    --stages 6 \
    --num_envs "${NUM_ENVS:-256}" \
    --seed 17 \
    --stage_iterations "${STAGE_ITERATIONS:-600}" \
    --freeze_actor_iterations "${FREEZE_ITERATIONS:-150}" \
    --learning_rate 3e-4 \
    --desired_kl 0.008 \
    --entropy_coef 0.001 \
    --std_clamp 0.03,0.15 \
    --save_interval 25 \
    --log_dir "$LOG_DIR" \
    --headless 2>&1 | tee "$LOG_DIR/train.log"
