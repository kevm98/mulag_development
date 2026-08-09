#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
PRODUCTION="$ROOT/logs/rsl_rl/ppo3_sole_terrain/shared_v2_rom/production/model_converged.pt"
MIGRATED="$ROOT/logs/rsl_rl/ppo3_sole_terrain/shared_v2_rom/model_migrated.pt"
CHECKPOINT="$PRODUCTION"
[[ -f "$CHECKPOINT" ]] || CHECKPOINT="$MIGRATED"

if [[ $# -gt 0 && "$1" != --* ]]; then
  CHECKPOINT="$1"
  shift
fi

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "PPO3 checkpoint not found: $CHECKPOINT" >&2
  echo "usage: $0 [/path/to/model.pt] [additional evaluator arguments]" >&2
  exit 2
fi

cd "$ROOT"
# The contact-safe experiment is 96 wide (6 contact channels + model-error
# history depth 3) while the previous production contract is 66. Read the width
# the CHECKPOINT was trained at and configure the environment to match, rather
# than making the caller remember three flags: the evaluator refuses a width
# mismatch outright, so getting this wrong is a confusing failure at startup
# rather than a wrong number, but it is still a failure.
WIDTH="$("$PY" - "$CHECKPOINT" <<'PYWIDTH'
import sys, torch
payload = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int((payload.get("infos") or {}).get("observation_width", -1)))
PYWIDTH
)"
EXTRA=()
if [[ "$WIDTH" == "96" ]]; then
  EXTRA+=(--contact_observation true --contact_safety_reward true
          --model_error_history 3 --strict_contact
          --action_scale 0.0035 --cylinder_target_max_lead_m 0.018)
  echo "[VISUALIZE] 96-wide contact-safe checkpoint: contact channels ON"
else
  echo "[VISUALIZE] ${WIDTH}-wide checkpoint: previous contract"
fi

exec "$PY" verge_task/scripts/evaluate.py "$CHECKPOINT" \
  --actuator rom \
  --device cuda:0 \
  --num_envs 1 \
  --seed 17 \
  --steps 960 \
  --stage 5 \
  --eval_terrain \
  --head_joint_action \
  --head_action_to_c4_gain 0.4 \
  --drive_substep_interpolation true \
  --path_family s_curve \
  --plot_tracking \
  "${EXTRA[@]}" \
  "$@"
