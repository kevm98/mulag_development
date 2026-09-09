#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=$HOME/miniconda3/envs/env_isaaclab/bin/python
CHECKPOINT="$ROOT/logs/rsl_rl/ppo3_sole_terrain/shared_v2_rom/model_migrated.pt"
RUN_DIR="$ROOT/logs/rsl_rl/ppo3_sole_terrain/shared_v2_rom/production"

if [[ $# -gt 0 && "$1" != --* ]]; then
  CHECKPOINT="$1"
  shift
fi

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "PPO3 checkpoint not found: $CHECKPOINT" >&2
  echo "usage: $0 [/path/to/model.pt] [additional supervisor arguments]" >&2
  exit 2
fi

cd "$ROOT"
exec "$PY" verge_task/scripts/train_until_converged.py \
  --resume "$CHECKPOINT" \
  --run_dir "$RUN_DIR" \
  --actuator rom \
  --device cuda:0 \
  --stage 5 \
  --num_envs 128 \
  --eval_envs 32 \
  --eval_steps 960 \
  --block_iterations 30 \
  --min_blocks 1 \
  --learning_rate 5e-7 \
  --reset_optimizer_first_block \
  --c4_exploration_std 0.04 \
  --q5_exploration_std 0.04 \
  --head_joint_action \
  --head_action_to_c4_gain 0.4 \
  --drive_substep_interpolation true \
  "$@"
