#!/usr/bin/env bash
# PPO-ID(AAM)-FD factor curriculum: mechanical -> hydraulic -> combined.
#
# Adaptation contract for every stage (both consumers of the latent are ON):
#   PPO gets z_hat : online_aam_policy_observation=True, observation_space=68
#   ID  gets z_hat : online_aam_residual_scale=0.25 over the Stage-31 adapter
# Neither is ablated here. The 2x2 ablation ladder comes after this works.
#
# Factor manifest (AAM_module/datasets/rom_deployment_stage31_v1_frozen):
#   mechanical      12 runs   train only
#   hydraulic       40 runs   train only
#   combined_train  48 runs
#   combined_test   16 runs   HELD OUT -- gating only, never trained on
#
# Speeds 2-7 km/h every stage, sampled once per episode reset, with
# active_time_balanced sampling so the faster episodes are not under-weighted
# by their shorter duration.
#
# Warm start: the migrated ID-FD policy at the 68-D contract. Measured on
# nominal, mesh 1337: AAM 0.01085 p01 headroom at 2 km/h against ID-FD's
# 0.00515, so the AAM path starts ahead of its own base, not behind it.
set -uo pipefail

ROOT=/home/rrlab/rrlab_mulag
PY=/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
CAMPAIGN=${CAMPAIGN:-aam_factor_curriculum_v1}
ITERS=${ITERS:-300}
ENVS=${ENVS:-256}
BASE=$ROOT/logs/rsl_rl/ppo_id_aam_fd/idfd_init_speed_2_7/model_init.pt
OUT=$ROOT/experiments/$CAMPAIGN
mkdir -p "$OUT"
cd "$ROOT"

latest() { ls -t "$1"/model_*.pt 2>/dev/null | head -n1; }

train_stage() {  # name domain resume_checkpoint
  local name=$1 domain=$2 resume=$3
  local dir=$ROOT/logs/rsl_rl/ppo_id_aam_fd/${CAMPAIGN}_${name}
  mkdir -p "$dir"
  if [ -f "$dir/.done" ]; then
    echo "[curr] stage $name already complete"; return 0
  fi
  echo "[curr] stage $name  domain=$domain  resume=$(basename "$resume")"
  "$PY" PPO_AAM_FD/scripts/train.py \
    --resume "$resume" --reset_optimizer_on_resume \
    --stages 6 --stage_iterations "$ITERS" --num_envs "$ENVS" --seed 41 \
    --vehicle_speed_min_kmh 2.0 --vehicle_speed_max_kmh 7.0 \
    --vehicle_speed_sampling active_time_balanced \
    --factor_domain "$domain" \
    --learning_rate 3e-6 --fixed_learning_rate --gamma 0.97 --lam 0.95 \
    --entropy_coef 0.001 \
    --contact_safety_reward true --contact_focus --contact_stage B \
    --contact_latch true --contact_latch_release_steps 60 \
    --online_aam_residual_scale 0.25 \
    --obs_normalizer freeze --save_interval 25 \
    --log_dir "$dir" --headless --device cuda:0 \
    > "$OUT/${name}_train.log" 2>&1
  local ckpt; ckpt=$(latest "$dir")
  [ -n "$ckpt" ] || { echo "[curr] stage $name produced no checkpoint"; return 1; }
  printf '%s\n' "$ckpt" > "$dir/.done"
  echo "[curr] stage $name -> $ckpt"
}

# Cheap between-stage check: one mesh, three speeds. Catches a collapse before
# the next stage builds on it. The full contract is scored only at the end.
probe() {  # name checkpoint
  local name=$1 ckpt=$2
  for SPEED in 2 4 7; do
    REPORT_ROOT=$OUT/probe_$name MESHES=1337 STALL=300 STAGE=6 \
    TASK=RRLAB-Mulag-Verge-ID-AAM-FD-v0 \
    SPEED_KMH="${SPEED}.0" FACTOR_DOMAIN=combined_test \
      bash PPO_ROM/scripts/score_per_mesh.sh "${name}_s${SPEED}" "$ckpt" 2808 32 \
      > "$OUT/probe_${name}_s${SPEED}.log" 2>&1
  done
}

# Full promotion contract: three meshes, every speed, held-out factors.
gate() {  # name checkpoint
  local name=$1 ckpt=$2
  for SPEED in 2 3 4 5 6 7; do
    REPORT_ROOT=$OUT/gate_$name MESHES="1337 17 909" PARALLEL_MESHES=3 \
    STALL=300 STAGE=6 TASK=RRLAB-Mulag-Verge-ID-AAM-FD-v0 \
    SPEED_KMH="${SPEED}.0" FACTOR_DOMAIN=combined_test \
      bash PPO_ROM/scripts/score_per_mesh.sh "${name}_s${SPEED}" "$ckpt" 2808 32 \
      > "$OUT/gate_${name}_s${SPEED}.log" 2>&1
  done
}

D=$ROOT/logs/rsl_rl/ppo_id_aam_fd

train_stage mechanical mechanical     "$BASE"                             || exit 1
CK_A=$(cat "$D/${CAMPAIGN}_mechanical/.done"); probe mechanical "$CK_A"

train_stage hydraulic  hydraulic      "$CK_A"                             || exit 1
CK_B=$(cat "$D/${CAMPAIGN}_hydraulic/.done"); probe hydraulic "$CK_B"

train_stage combined   combined_train "$CK_B"                             || exit 1
CK_C=$(cat "$D/${CAMPAIGN}_combined/.done")

gate combined "$CK_C"
printf '%s\n' "$CK_C" > "$OUT/final_checkpoint.txt"
touch "$OUT/DONE"
echo "[curr] finished -> $CK_C"
