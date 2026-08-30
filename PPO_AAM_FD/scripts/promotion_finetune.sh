#!/usr/bin/env bash
set -euo pipefail

# Long PPO continuation for the promoted 1 s causal CNN estimator.  This is
# deliberately a fixed, reproducible fine-tune: the AAM/ID/FD artifacts are
# frozen and only the PPO policy is updated.
ROOT=/home/rrlab/rrlab_mulag
PY=/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
CKPT=${1:-$ROOT/logs/rsl_rl/ppo_id_aam_fd/stage31_promotable_v2/model_375.pt}
OUT=${2:-$ROOT/logs/rsl_rl/ppo_id_aam_fd/promotion_finetune_l20_cnn}
ITERATIONS=${PPO_FINETUNE_ITERATIONS:-1500}
ENVS=${PPO_FINETUNE_ENVS:-256}
OBS_NORMALIZER=${PPO_FINETUNE_OBS_NORMALIZER:-freeze}
LATENT_MODE=${PPO_FINETUNE_LATENT_MODE:-estimated}
case "$LATENT_MODE" in
  true) DEFAULT_ADAPTER="$ROOT/outputs/stage31_rom/id_adapter_cnn_l20_s5_true/true_adapter.pt" ;;
  estimated|zero|mean|shuffled) DEFAULT_ADAPTER="$ROOT/outputs/stage31_rom/id_adapter_cnn_l20_s5/zhat_adapter.pt" ;;
  *) echo "unsupported PPO_FINETUNE_LATENT_MODE: $LATENT_MODE" >&2; exit 2 ;;
esac
ADAPTER_MODEL=${MULAG_AAM_ADAPTER_MODEL:-$DEFAULT_ADAPTER}
FACTOR_DOMAIN=${PPO_FINETUNE_FACTOR_DOMAIN:-combined_train}
FREEZE_ACTOR_ITERATIONS=${PPO_FINETUNE_FREEZE_ACTOR_ITERATIONS:-100}
LEARNING_RATE=${PPO_FINETUNE_LEARNING_RATE:-1e-5}
SAVE_INTERVAL=${PPO_FINETUNE_SAVE_INTERVAL:-50}
RESET_OPTIMIZER=${PPO_FINETUNE_RESET_OPTIMIZER:-1}
RESET_CRITIC_HEAD=${PPO_FINETUNE_RESET_CRITIC_HEAD:-1}
LATENT_TEACHER=${PPO_FINETUNE_LATENT_TEACHER:-0}
LATENT_TEACHER_LAMBDA=${PPO_FINETUNE_LATENT_TEACHER_LAMBDA:-0.5}
LATENT_TEACHER_KL_BETA=${PPO_FINETUNE_LATENT_TEACHER_KL_BETA:-0.05}
LATENT_TEACHER_MAX_KL=${PPO_FINETUNE_LATENT_TEACHER_MAX_KL:-0.05}
SPEED_MIN_KMH=${PPO_FINETUNE_SPEED_MIN_KMH:-2.0}
SPEED_MAX_KMH=${PPO_FINETUNE_SPEED_MAX_KMH:-7.0}

test -f "$CKPT" || { echo "missing PPO checkpoint: $CKPT" >&2; exit 2; }
"$PY" - "$ROOT/outputs/stage31_rom/aam_cnn/L20_d64/metrics.json" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.is_file():
    raise SystemExit(f"missing promoted AAM metrics: {p}")
m = json.loads(p.read_text())
if int(m.get("lag_ticks", -1)) != 20 or m.get("architecture") != "cnn":
    raise SystemExit(f"promotion fine-tune requires CNN L20 AAM, got {m.get('architecture')} L{m.get('lag_ticks')}")
PY
test -f "$ADAPTER_MODEL" || {
  echo "missing L20 CNN-matched adapter; set MULAG_AAM_ADAPTER_MODEL for a declared ablation" >&2
  exit 2
}
# Ensure a non-exported shell assignment is still seen by the Python child;
# otherwise the preflight checks one adapter while the environment silently
# loads another from its default path.
export MULAG_AAM_ADAPTER_MODEL="$ADAPTER_MODEL"

mkdir -p "$OUT"
cmd=("$PY" "$ROOT/PPO_AAM_FD/scripts/train.py" \
  --resume "$CKPT" \
  --stages 6 \
  --stage_iterations "$ITERATIONS" \
  --num-envs "$ENVS" \
  --seed 17 \
  --vehicle_speed_min_kmh "$SPEED_MIN_KMH" \
  --vehicle_speed_max_kmh "$SPEED_MAX_KMH" \
  --vehicle_speed_sampling uniform \
  --factor_domain "$FACTOR_DOMAIN" \
  --learning_rate "$LEARNING_RATE" \
  --fixed_learning_rate \
  --entropy_coef 0.0 \
  --freeze_actor_iterations "$FREEZE_ACTOR_ITERATIONS" \
  --obs_normalizer "$OBS_NORMALIZER" \
  --online_aam_residual_scale 0.25 \
  --online_aam_latent_mode "$LATENT_MODE" \
  --save_interval "$SAVE_INTERVAL" \
  --log_dir "$OUT" \
  --headless)
if [[ "$RESET_OPTIMIZER" == "1" ]]; then
  cmd+=(--reset_optimizer_on_resume)
fi
if [[ "$RESET_CRITIC_HEAD" == "1" ]]; then
  cmd+=(--reset_critic_head_on_resume)
fi
if [[ "$LATENT_TEACHER" == "1" ]]; then
  if [[ "$LATENT_MODE" != "estimated" ]]; then
    echo "latent-teacher handoff requires PPO_FINETUNE_LATENT_MODE=estimated" >&2
    exit 2
  fi
  cmd+=(
    --latent_teacher_regularized
    --latent_teacher_lambda "$LATENT_TEACHER_LAMBDA"
    --latent_teacher_kl_beta "$LATENT_TEACHER_KL_BETA"
    --latent_teacher_max_kl "$LATENT_TEACHER_MAX_KL"
  )
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi
exec "${cmd[@]}"
