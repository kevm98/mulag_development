#!/usr/bin/env bash
# Online gradient adaptation, per reports/PREREG_ONLINE_GRAD_2026-09-05.md.
#
# Stage A selects the learning rate on mesh 2718 alone, from the fixed ladder,
# on the mean absolute prediction residual only. Scored gate metrics are not
# consulted for selection.
# Stage B scores three arms on all three reserved meshes.
set -u
export IMAGE=isaac-lab:2.3.2-rl052
BASE=logs/rsl_rl/ppo_id_aam_fd/distilled_v1/phase1_init_70.pt
MESHES="2718 3141 8191"
LADDER="1e-5 1e-4 1e-3"
OUT=outputs/online_grad
cd /home/kevinmurera

ENVS="
export MULAG_DIRECT_VALVE=1
export MULAG_RMA_CONTEXT=1
export MULAG_PAYLOAD_RANGE=1.0,5.0
export MULAG_PAYLOAD_RESAMPLE=1
"
run() { ~/bin/isaaclab-run bash -lc "$1" 2>/dev/null || true; }
mkdir -p /home/kevinmurera/projects/rrlab_mulag/$OUT

# Wait for the extension campaign so the two never share the GPU.
while pgrep -f '[s]tarved_ext' >/dev/null; do sleep 60; done
while pgrep -f '[t]rain_residual_adapter|[e]valuate_gates' >/dev/null; do sleep 30; done
echo "GPU free, starting"

echo "=== STAGE A: learning-rate ladder on mesh 2718 ==="
for LR in $LADDER; do
  run "
cd /workspace/projects/rrlab_mulag
$ENVS
/isaac-sim/python.sh PPO_ROM/scripts/evaluate_gates.py $BASE \
  --task RRLAB-Mulag-Verge-ID-AAM-FD-v0 \
  --online_grad_lr $LR --online_grad_trust 0.05 \
  --num_envs 176 --steps 900 --meshes 2718 --seed 17 \
  --report $OUT/ladder_lr${LR}.json 2>&1 | grep -aE 'online gradient|wrote|rror' | tail -3
"
  echo "LADDER_DONE_${LR}"
done
echo "=== STAGE A COMPLETE ==="

# Pick the rung with the lowest mean absolute prediction residual.
LR=$(cd /home/kevinmurera/projects/rrlab_mulag && python3 - <<'PY'
import json, glob, os
best, blr = None, None
for f in sorted(glob.glob("outputs/online_grad/ladder_lr*.json")):
    try:
        d = json.load(open(f))
        v = d.get("online_grad_diag", {}).get("mean_abs_prediction_residual")
        lr = d.get("online_grad_lr")
    except Exception:
        continue
    if v is None or lr is None:
        continue
    print(f"  lr={lr:g}  mean|d|={v:.6f}", flush=True)
    if best is None or v < best:
        best, blr = v, lr
print(f"SELECTED {blr:g}" if blr else "SELECTED none", flush=True)
open("outputs/online_grad/selected_lr.txt", "w").write("" if blr is None else f"{blr:g}")
PY
)
echo "$LR"
LR=$(cat /home/kevinmurera/projects/rrlab_mulag/outputs/online_grad/selected_lr.txt 2>/dev/null)
if [ -z "$LR" ]; then echo "NO LR SELECTED, stopping"; exit 1; fi
echo "=== selected learning rate: $LR ==="

echo "=== STAGE B: three arms on three meshes ==="
for MESH in $MESHES; do
  # off: the frozen base, no adaptation
  run "
cd /workspace/projects/rrlab_mulag
$ENVS
/isaac-sim/python.sh PPO_ROM/scripts/evaluate_gates.py $BASE \
  --task RRLAB-Mulag-Verge-ID-AAM-FD-v0 \
  --num_envs 176 --steps 900 --meshes $MESH --seed 17 \
  --report outputs/rif_delay/ogradoff_m${MESH}.json 2>&1 | grep -aE 'wrote|rror' | tail -2
"
  echo "SCORED_ogradoff_m${MESH}"

  # on: live gradient
  run "
cd /workspace/projects/rrlab_mulag
$ENVS
/isaac-sim/python.sh PPO_ROM/scripts/evaluate_gates.py $BASE \
  --task RRLAB-Mulag-Verge-ID-AAM-FD-v0 \
  --online_grad_lr $LR --online_grad_trust 0.05 \
  --num_envs 176 --steps 900 --meshes $MESH --seed 17 \
  --report outputs/rif_delay/ogradon_m${MESH}.json 2>&1 | grep -aE 'wrote|rror' | tail -2
"
  echo "SCORED_ogradon_m${MESH}"

  # sham: same motion, randomised direction
  run "
cd /workspace/projects/rrlab_mulag
$ENVS
/isaac-sim/python.sh PPO_ROM/scripts/evaluate_gates.py $BASE \
  --task RRLAB-Mulag-Verge-ID-AAM-FD-v0 \
  --online_grad_lr $LR --online_grad_trust 0.05 --online_grad_sham \
  --num_envs 176 --steps 900 --meshes $MESH --seed 17 \
  --report outputs/rif_delay/ogradsham_m${MESH}.json 2>&1 | grep -aE 'wrote|rror' | tail -2
"
  echo "SCORED_ogradsham_m${MESH}"
done
echo ONLINE_GRAD_DONE
