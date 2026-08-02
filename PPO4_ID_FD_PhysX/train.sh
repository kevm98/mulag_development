#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
CHECKPOINT="$ROOT/logs/rsl_rl/ppo4_sole_idfd/shared_v2_idfd/model_validated.pt"
RUN_DIR="$ROOT/logs/rsl_rl/ppo4_sole_idfd/production"

if [[ $# -gt 0 && "$1" != --* ]]; then
  CHECKPOINT="$1"
  shift
fi

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "PPO4 checkpoint not found: $CHECKPOINT" >&2
  echo "usage: $0 [/path/to/model.pt] [additional supervisor arguments]" >&2
  exit 2
fi

cd "$ROOT"
echo "[PPO4] resume=$CHECKPOINT"
echo "[PPO4] run_dir=$RUN_DIR"

exec "$PY" PPO3_SOLE_PhysX/scripts/train_until_converged.py \
  --resume "$CHECKPOINT" \
  --run_dir "$RUN_DIR" \
  --actuator idfd \
  --device cuda:0 \
  --stage 5 \
  --num_envs 128 \
  --eval_envs 64 \
  --eval_steps 960 \
  --block_iterations 30 \
  --learning_rate 5e-7 \
  --reset_optimizer_first_block \
  --c4_exploration_std 0.04 \
  --q5_exploration_std 0.04 \
  --head_joint_action \
  --head_action_to_c4_gain 0.4 \
  --drive_substep_interpolation true \
  "$@"
