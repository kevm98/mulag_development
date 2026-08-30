#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/rrlab/rrlab_mulag
PY=/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
CAMPAIGN=${CAMPAIGN:-speed_range_2_7_pathfix_v2}
OUT="$ROOT/experiments/$CAMPAIGN"
MESHES=${MESHES:-"1337 17 909"}
EVAL_ENVS=${EVAL_ENVS:-32}
EVAL_STEPS=${EVAL_STEPS:-2808}
GPU_QUIET_SECONDS=${GPU_QUIET_SECONDS:-0}

ROM_BASE="$ROOT/logs/rsl_rl/ppo3_1_sole/rom_wide_reference_kl_block2_v1/model_selected.pt"
IDFD_BASE="$ROOT/logs/rsl_rl/ppo4_1_sole/idfd_promotable_v1/promoted_policy.pt"
AAM_BASE="$ROOT/logs/rsl_rl/ppo_id_aam_fd/speed_range_2_7_contact_safe_v2/model_700.pt"
ROM_RUN="$ROOT/logs/rsl_rl/ppo3_1_sole/${CAMPAIGN}_rom"
IDFD_RUN="$ROOT/logs/rsl_rl/ppo4_1_sole/${CAMPAIGN}_idfd"
AAM_RUN="$ROOT/logs/rsl_rl/ppo_id_aam_fd/${CAMPAIGN}_id_aam_fd"
LOCK="$OUT/ordered_queue.lock"
STATUS="$OUT/ordered_status.log"

mkdir -p "$OUT" "$ROM_RUN" "$IDFD_RUN" "$AAM_RUN"
if ! mkdir "$LOCK" 2>/dev/null; then
  owner=$(cat "$LOCK/pid" 2>/dev/null || true)
  owner_cmd=$(tr '\0' ' ' <"/proc/$owner/cmdline" 2>/dev/null || true)
  if [ -n "$owner" ] && kill -0 "$owner" 2>/dev/null && [[ "$owner_cmd" == *run_ordered_speed_campaign.sh* ]]; then
    exit 2
  fi
  rm -f "$LOCK/pid"
  rmdir "$LOCK" 2>/dev/null || exit 2
  mkdir "$LOCK"
fi
printf '%s\n' "$$" >"$LOCK/pid"
trap 'rm -f "$LOCK/pid"; rmdir "$LOCK" 2>/dev/null || true' EXIT

log() { echo "[$(date -Is)] $*" | tee -a "$STATUS"; }
latest() {
  find "$1" -maxdepth 1 -type f -name 'model_[0-9]*.pt' -printf '%f\n' \
    | sed -E 's/model_([0-9]+)\.pt/\1 &/' | sort -n | tail -1 | cut -d' ' -f2- \
    | sed "s|^|$1/|"
}
resume_from() {
  local run=$1 fallback=$2 candidate
  if [ -s "$run/.best_checkpoint" ]; then
    candidate=$(head -n1 "$run/.best_checkpoint")
    [ -f "$candidate" ] && { printf '%s\n' "$candidate"; return; }
  fi
  [ -f "$run/model_selected.pt" ] && { printf '%s\n' "$run/model_selected.pt"; return; }
  printf '%s\n' "$fallback"
}
wait_gpu() {
  local quiet=0 active
  while true; do
    active=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^$/d' | wc -l || true)
    if [ "${active:-0}" -eq 0 ]; then
      quiet=$((quiet + 1))
      [ "$quiet" -ge "$GPU_QUIET_SECONDS" ] && return
    else
      quiet=0
    fi
    sleep 1
  done
}
score_domain() {
  local label=$1 task=$2 checkpoint=$3 root=$4 speed name report
  for speed in 2.0 4.5 7.0; do
    name="${label}_fixed_${speed//./_}"
    report="$root/fixed_${speed//./_}"
    [ -f "$report/${name}_gates.json" ] && jq -e '.gates.passed == true' "$report/${name}_gates.json" >/dev/null 2>&1 && continue
    wait_gpu
    MESHES="$MESHES" TASK="$task" SPEED_KMH="$speed" STEPS="$EVAL_STEPS" ENVS="$EVAL_ENVS" REPORT_ROOT="$report" \
      bash "$ROOT/PPO_ROM/scripts/score_per_mesh.sh" "$name" "$checkpoint" "$EVAL_STEPS" "$EVAL_ENVS"
  done
  name="${label}_random_2_7"
  report="$root/random_2_7"
  if [ ! -f "$report/${name}_gates.json" ] || ! jq -e '.gates.passed == true' "$report/${name}_gates.json" >/dev/null 2>&1; then
    wait_gpu
    MESHES="$MESHES" TASK="$task" SPEED_MIN_KMH=2.0 SPEED_MAX_KMH=7.0 STEPS="$EVAL_STEPS" ENVS="$EVAL_ENVS" REPORT_ROOT="$report" \
      bash "$ROOT/PPO_ROM/scripts/score_per_mesh.sh" "$name" "$checkpoint" "$EVAL_STEPS" "$EVAL_ENVS"
  fi
}
train_rom() {
  local resume
  [ -f "$ROM_RUN/.training_complete" ] && { latest "$ROM_RUN"; return; }
  resume=$(resume_from "$ROM_RUN" "$ROM_BASE")
  wait_gpu
  "$PY" PPO_ROM/scripts/train.py --resume "$resume" --stages 6 --stage-iterations 300 --num-envs 128 --trajectory-shapes all \
    --vehicle_speed_min_kmh 2.0 --vehicle_speed_max_kmh 7.0 --vehicle_speed_sampling uniform \
    --trajectory_regularized --trajectory_local_coef 0.10 --trajectory_max_update_kl 0.01 --trajectory_max_reference_kl 0.05 \
    --trajectory_actor_learning_rate 1e-5 --trajectory_critic_learning_rate 1e-4 --reset_optimizer_on_resume --reset_critic_head_on_resume \
    --freeze_actor_iterations 100 --reset_actor_optimizer_on_resume --gamma 0.97 --lam 0.95 --fixed_std 0.15 --entropy_coef 0.001 \
    --obs_normalizer freeze --save-interval 25 --log_dir "$ROM_RUN" --headless --device cuda:0 >"$OUT/rom_ordered_train.log" 2>&1
  latest "$ROM_RUN" >"$ROM_RUN/.training_complete"
  latest "$ROM_RUN"
}
train_idfd() {
  local resume
  [ -f "$IDFD_RUN/.training_complete" ] && { latest "$IDFD_RUN"; return; }
  resume=$(resume_from "$IDFD_RUN" "$IDFD_BASE")
  wait_gpu
  bash PPO_ID_FD/scripts/train.sh --resume "$resume" --stages 6 --stage-iterations 600 --freeze_actor_iterations 150 --num-envs 128 \
    --trajectory-shapes all --vehicle_speed_min_kmh 2.0 --vehicle_speed_max_kmh 7.0 --vehicle_speed_sampling uniform \
    --reset_optimizer_on_resume --reset_critic_head_on_resume --gamma 0.97 --lam 0.95 --trajectory_regularized \
    --trajectory_local_coef 0.10 --trajectory_future_coef 0.0 --trajectory_max_update_kl 0.01 --trajectory_max_reference_kl 0.05 \
    --fixed_std 0.05 --entropy_coef 0.001 --obs_normalizer freeze --save-interval 25 --log_dir "$IDFD_RUN" --headless --device cuda:0 >"$OUT/idfd_ordered_train.log" 2>&1
  latest "$IDFD_RUN" >"$IDFD_RUN/.training_complete"
  latest "$IDFD_RUN"
}
train_aam() {
  local resume
  [ -f "$AAM_RUN/.training_complete" ] && { latest "$AAM_RUN"; return; }
  resume=$(resume_from "$AAM_RUN" "$AAM_BASE")
  wait_gpu
  "$PY" PPO_AAM_FD/scripts/train.py --resume "$resume" --stages 6 --stage_iterations 300 --num_envs 256 --seed 41 \
    --vehicle_speed_min_kmh 2.0 --vehicle_speed_max_kmh 7.0 --vehicle_speed_sampling uniform --learning_rate 1e-5 --fixed_learning_rate \
    --gamma 0.97 --lam 0.95 --entropy_coef 0.001 --contact_safety_reward true --contact_focus --contact_stage B \
    --contact_latch true --contact_latch_release_steps 60 --online_aam_residual_scale 0.25 --obs_normalizer freeze \
    --save_interval 25 --log_dir "$AAM_RUN" --headless --device cuda:0 >"$OUT/aam_ordered_train.log" 2>&1
  latest "$AAM_RUN" >"$AAM_RUN/.training_complete"
  latest "$AAM_RUN"
}

for checkpoint in "$ROM_BASE" "$IDFD_BASE" "$AAM_BASE"; do [ -f "$checkpoint" ] || { log "missing checkpoint $checkpoint"; exit 1; }; done
log "stage 1/3: PPO-ROM, constant per-episode truck speed uniformly sampled in 2--7 km/h"
ROM_CKPT=$(train_rom)
log "PPO-ROM checkpoint $ROM_CKPT; scoring fixed 2/4.5/7 and randomized 2--7 km/h"
score_domain rom RRLAB-Mulag-Verge-ROM-v0 "$ROM_CKPT" "$OUT/rom_ordered_eval"
printf '%s\n' "$ROM_CKPT" >"$ROM_RUN/.best_checkpoint"
log "stage 2/3: PPO-ID-FD"
IDFD_CKPT=$(train_idfd)
score_domain idfd RRLAB-Mulag-Verge-IDFD-v0 "$IDFD_CKPT" "$OUT/idfd_ordered_eval"
printf '%s\n' "$IDFD_CKPT" >"$IDFD_RUN/.best_checkpoint"
log "stage 3/3: PPO-ID(AAM)-FD"
AAM_CKPT=$(train_aam)
score_domain id_aam_fd RRLAB-Mulag-Verge-ID-AAM-FD-v0 "$AAM_CKPT" "$OUT/aam_ordered_eval"
printf '%s\n' "$AAM_CKPT" >"$AAM_RUN/.best_checkpoint"
log "ordered speed campaign complete"
