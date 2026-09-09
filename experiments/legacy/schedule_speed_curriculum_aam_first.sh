#!/usr/bin/env bash
set -euo pipefail

ROOT=$HOME/rrlab_mulag
PY=$HOME/miniconda3/envs/env_isaaclab/bin/python
CAMPAIGN=${CAMPAIGN:-speed_curriculum_aam_first_v1}
OUT="$ROOT/experiments/$CAMPAIGN"
MESHES=${MESHES:-"1337 17 909"}
EVAL_ENVS=${EVAL_ENVS:-32}
EVAL_STEPS=${EVAL_STEPS:-2808}
LOCK="$ROOT/experiments/.speed_randomization_campaign.lock"

AAM_BASE="$ROOT/logs/rsl_rl/ppo_id_aam_fd/speed_range_2_7_contact_safe_v2/model_700.pt"
ROM_BASE="$ROOT/logs/rsl_rl/ppo3_1_sole/rom_wide_reference_kl_block2_v1/model_selected.pt"
IDFD_BASE="$ROOT/logs/rsl_rl/ppo4_1_sole/idfd_promotable_v1/promoted_policy.pt"

mkdir -p "$OUT"

acquire_lock() {
  local owner="" owner_cmd=""
  if ! mkdir "$LOCK" 2>/dev/null; then
    owner=$(cat "$LOCK/pid" 2>/dev/null || true)
    if [ -n "$owner" ] && kill -0 "$owner" 2>/dev/null; then
      owner_cmd=$(tr '\0' ' ' <"/proc/$owner/cmdline" 2>/dev/null || true)
      echo "speed campaign already running: pid=$owner command=$owner_cmd" >&2
      exit 2
    fi
    rm -f "$LOCK/pid"
    rmdir "$LOCK" 2>/dev/null || {
      echo "cannot reclaim stale speed campaign lock: $LOCK" >&2
      exit 2
    }
    mkdir "$LOCK"
  fi
  printf '%s\n' "$$" >"$LOCK/pid"
}

release_lock() {
  [ "$(cat "$LOCK/pid" 2>/dev/null || true)" = "$$" ] || return 0
  rm -f "$LOCK/pid"
  rmdir "$LOCK" 2>/dev/null || true
}

acquire_lock
trap release_lock EXIT INT TERM

for checkpoint in "$AAM_BASE" "$ROM_BASE" "$IDFD_BASE"; do
  [ -f "$checkpoint" ] || { echo "missing checkpoint: $checkpoint" >&2; exit 1; }
done

latest_checkpoint() {
  find "$1" -maxdepth 1 -type f -name 'model_[0-9]*.pt' -printf '%f\n' \
    | sed -E 's/model_([0-9]+)\.pt/\1 &/' | sort -n | tail -1 \
    | cut -d' ' -f2- | sed "s|^|$1/|"
}

score_fixed() {
  local name=$1 task=$2 checkpoint=$3 speed=$4 report_root=$5
  local report="$report_root/${name}_gates.json"
  if [ ! -f "$report" ]; then
    MESHES="$MESHES" TASK="$task" SPEED_KMH="$speed" \
      REPORT_ROOT="$report_root" bash PPO_ROM/scripts/score_per_mesh.sh \
      "$name" "$checkpoint" "$EVAL_STEPS" "$EVAL_ENVS"
  fi
  jq -e '.gates.passed == true' "$report" >/dev/null
}

cd "$ROOT"
promoted="$AAM_BASE"
for maximum in 3.0 4.0 5.0 6.0 7.0; do
  tag=${maximum//./_}
  run="$ROOT/logs/rsl_rl/ppo_id_aam_fd/${CAMPAIGN}_to_${tag}"
  reports="$OUT/aam_to_${tag}"
  mkdir -p "$run" "$reports"
  if [ ! -f "$run/.training_complete" ]; then
    iterations=50
    [ "$maximum" = 3.0 ] && iterations=25
    "$PY" PPO_AAM_FD/scripts/train.py \
      --resume "$promoted" --stages 6 --stage_iterations "$iterations" \
      --num_envs 256 --seed 71 \
      --vehicle_speed_min_kmh 2.0 --vehicle_speed_max_kmh "$maximum" \
      --vehicle_speed_sampling uniform --speed_segment_duration_s 13.4 \
      --learning_rate 5e-6 --fixed_learning_rate \
      --gamma 0.97 --lam 0.95 --entropy_coef 0.0 --resume_std 0.02 \
      --reset_optimizer_on_resume --reset_critic_head_on_resume \
      --contact_safety_reward true --contact_latch true \
      --contact_latch_release_steps 60 --online_aam_residual_scale 0.25 \
      --obs_normalizer freeze --save_interval 5 \
      --log_dir "$run" --headless --device cuda:0 \
      > "$OUT/aam_to_${tag}_train.log" 2>&1
    touch "$run/.training_complete"
  fi
  candidate=$(latest_checkpoint "$run")
  score_fixed "aam_to_${tag}_low" RRLAB-Mulag-Verge-ID-AAM-FD-v0 \
    "$candidate" 2.0 "$reports"
  score_fixed "aam_to_${tag}_high" RRLAB-Mulag-Verge-ID-AAM-FD-v0 \
    "$candidate" "$maximum" "$reports"
  promoted="$candidate"
  printf '%s\n' "$promoted" > "$run/.best_checkpoint"
done
printf '%s\n' "$promoted" > "$OUT/aam_promoted_checkpoint"

rom_run="$ROOT/logs/rsl_rl/ppo3_1_sole/${CAMPAIGN}_rom"
mkdir -p "$rom_run"
"$PY" PPO_ROM/scripts/train.py \
  --resume "$ROM_BASE" --stages 6 --stage-iterations 300 --num-envs 128 \
  --trajectory-shapes all --vehicle_speed_min_kmh 2.0 \
  --vehicle_speed_max_kmh 7.0 --vehicle_speed_sampling uniform \
  --speed_segment_duration_s 13.4 --trajectory_regularized \
  --trajectory_local_coef 0.10 --trajectory_max_update_kl 0.01 \
  --trajectory_max_reference_kl 0.05 --trajectory_actor_learning_rate 1e-5 \
  --trajectory_critic_learning_rate 1e-4 --reset_optimizer_on_resume \
  --reset_critic_head_on_resume --freeze_actor_iterations 100 \
  --reset_actor_optimizer_on_resume --gamma 0.97 --lam 0.95 \
  --fixed_std 0.15 --entropy_coef 0.001 --obs_normalizer freeze \
  --save-interval 25 --log_dir "$rom_run" --headless --device cuda:0 \
  > "$OUT/rom_train.log" 2>&1
rom_checkpoint=$(latest_checkpoint "$rom_run")
score_fixed rom_low RRLAB-Mulag-Verge-ROM-v0 "$rom_checkpoint" 2.0 "$OUT/rom_eval"
score_fixed rom_high RRLAB-Mulag-Verge-ROM-v0 "$rom_checkpoint" 7.0 "$OUT/rom_eval"

idfd_run="$ROOT/logs/rsl_rl/ppo4_1_sole/${CAMPAIGN}_idfd"
mkdir -p "$idfd_run"
bash PPO_ID_FD/scripts/train.sh \
  --resume "$IDFD_BASE" --stages 6 --stage-iterations 600 --num-envs 128 \
  --trajectory-shapes all --vehicle_speed_min_kmh 2.0 \
  --vehicle_speed_max_kmh 7.0 --vehicle_speed_sampling uniform \
  --speed_segment_duration_s 13.4 --trajectory_regularized \
  --trajectory_local_coef 0.10 --trajectory_future_coef 0.0 \
  --trajectory_max_update_kl 0.01 --trajectory_max_reference_kl 0.05 \
  --reset_optimizer_on_resume --reset_critic_head_on_resume \
  --gamma 0.97 --lam 0.95 --fixed_std 0.05 --entropy_coef 0.001 \
  --obs_normalizer freeze --save-interval 25 --log_dir "$idfd_run" \
  --headless --device cuda:0 > "$OUT/idfd_train.log" 2>&1
idfd_checkpoint=$(latest_checkpoint "$idfd_run")
score_fixed idfd_low RRLAB-Mulag-Verge-IDFD-v0 "$idfd_checkpoint" 2.0 "$OUT/idfd_eval"
score_fixed idfd_high RRLAB-Mulag-Verge-IDFD-v0 "$idfd_checkpoint" 7.0 "$OUT/idfd_eval"
