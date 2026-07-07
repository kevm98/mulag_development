#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."
SCRIPT_DIR="$(pwd)"

# ---------------------------------------------------------------------------
# Python compatibility filter: scan all stage0_yz checkpoints by mtime,
# prefer nopin_bc > bc > vanilla, keep only obs_dim=4 act_dim=2.
# Outputs the selected checkpoint path to stdout on success.
# ---------------------------------------------------------------------------
select_checkpoint() {
    /home/rrlab/miniconda3/envs/env_isaaclab/bin/python3 - "$@" << 'PYEOF'
import sys, os, glob
import torch

EXPECTED_OBS_DIM = 4
EXPECTED_ACT_DIM = 2

search_pattern = sys.argv[1] if len(sys.argv) > 1 else "logs/rsl_rl/**/model_*.pt"

all_ckpts = sorted(glob.glob(search_pattern, recursive=True), key=os.path.getmtime)
if not all_ckpts:
    sys.exit(2)

compatible = []
for p in all_ckpts:
    ckpt = torch.load(p, map_location="cpu", weights_only=False)
    sd = ckpt.get("actor_state_dict", {})
    in_dim = sd.get("mlp.0.weight", None)
    out_dim = sd.get("mlp.4.weight", None)
    if in_dim is None or out_dim is None:
        print(f"[SKIP] {p}  (missing mlp weights)", file=sys.stderr)
        continue
    in_dim = in_dim.shape[1]
    out_dim = out_dim.shape[0]
    if in_dim == EXPECTED_OBS_DIM and out_dim == EXPECTED_ACT_DIM:
        compatible.append(p)
    else:
        print(f"[SKIP] {p}  obs_dim={in_dim} act_dim={out_dim}", file=sys.stderr)

if not compatible:
    print("", file=sys.stderr)
    print("[ERROR] No compatible 4D Stage-0 Y-Z checkpoint found.", file=sys.stderr)
    print("        Current env expects obs_dim=4.", file=sys.stderr)
    sys.exit(1)

# Prefer nopin_bc > bc > vanilla by sorting with priority
def priority(p):
    if "stage0_yz_nopin_bc" in p:
        return 0
    elif "stage0_yz_bc" in p:
        return 1
    else:
        return 2

compatible.sort(key=lambda p: (priority(p), -os.path.getmtime(p)))

chosen = compatible[0]
print(f"[INFO] Selected compatible checkpoint: {chosen}", file=sys.stderr)
print("[INFO] obs_dim=4 act_dim=2", file=sys.stderr)
print(chosen, file=sys.stdout)
PYEOF
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
CKPT=$(select_checkpoint "logs/rsl_rl/**/model_*.pt") || {
    exit 1
}

echo "[INFO] num_envs=${NUM_ENVS:-1}"
echo "[INFO] num_episodes=${NUM_EPISODES:-100}"
echo "[INFO] settle_steps=${SETTLE_STEPS:-30}"
echo "[INFO] kp_scale=${KP_SCALE:-0.1}"
echo "[INFO] kd_scale=${KD_SCALE:-0.5}"
echo "[INFO] no-pin hold config"

./rrlab.sh -p kevin_integration/new_observations_pipeline/scripts/evaluate_stage0_yz_policy.py \
  --checkpoint "$CKPT" \
  --num_envs "${NUM_ENVS:-1}" \
  --num_episodes "${NUM_EPISODES:-100}" \
  --settle_steps "${SETTLE_STEPS:-30}" \
  --kp_scale "${KP_SCALE:-0.1}" \
  --kd_scale "${KD_SCALE:-0.5}" \
  --headless \
  --device "${DEVICE:-cuda:0}"
