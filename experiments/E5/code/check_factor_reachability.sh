#!/usr/bin/env bash
set -uo pipefail
# E5 Stage 0a -- FACTOR REACHABILITY.
#
# Static inspection of the runtime says the six deployment factors do not all
# reach the plant, and that what reaches it differs by stack:
#
#   R   (command ROM) : hydraulic factors enter as `system_factors`, but
#                       HydraulicROM.forward CLAMPS them back to the ranges it
#                       was identified on and bounds the response gain to
#                       [0.35, 1.50].
#   IF / AIF (ID->FD) : `CylinderFD.step(valve_cmd, extension, velocity, dP,
#                       force)` takes NO factor argument. The four hydraulic
#                       factors are inert on this plant; they reach only the
#                       privileged latent encoder and the logs.
#   all stacks        : mass_scale and com_z are applied through
#                       set_masses/set_coms with no clamp.
#
# This script tests that empirically, because a claim about what the plant
# ignores must not rest on reading the source. It scores one stack twice under
# factor draws that differ ONLY in the hydraulic channels. If the two reports
# are identical, the hydraulic factors are confirmed inert for that stack.
#
# Expected: IF and AIF identical, R different. Anything else means the static
# reading is wrong and the sweep below must not be interpreted until it is.

ROOT=${ROOT:-/home/rrlab/rrlab_mulag}
cd "$ROOT"
OUT=${OUT:-$ROOT/outputs/e5_reachability}
export MESHES=${MESHES:-1337}
export PARALLEL_MESHES=1
export SPEED_KMH=${SPEED_KMH:-4.0}
export STAGE=6
export REPORT_ROOT="$OUT"
export TRAJECTORY_SHAPES=${TRAJECTORY_SHAPES:-"straight,double_s_curve,lane_change_hold"}
ENVS=${ENVS:-8}
STEPS=${STEPS:-2808}
mkdir -p "$OUT"
LOG=$OUT/reachability.log
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

# seal_min and seal_max differ ONLY in the seal channel; both leave mass and
# com_z at nominal, so any difference in the report is hydraulic.
PAIRS=("seal_min" "seal_max" "flow_min" "flow_max")

STACKS=(
  "ppo_rom|logs/rsl_rl/ppo3_1_sole/rom_s6_aligned/model_598.pt|"
  "ppo_id_fd|logs/rsl_rl/ppo4_1_sole/idfd_s6_aligned/model_598.pt|RRLAB-Mulag-Verge-IDFD-v0"
  "ppo_aam_id_fd|logs/rsl_rl/ppo_id_aam_fd/headroom_priced_v1/model_299.pt|RRLAB-Mulag-Verge-ID-AAM-FD-v0"
)

for row in "${STACKS[@]}"; do
  IFS='|' read -r name ckpt task <<<"$row"
  for key in "${PAIRS[@]}"; do
    tag="${name}_${key}"
    [ -s "$OUT/${tag}_m${MESHES}.json" ] && { say "reusing $tag"; continue; }
    say "start $tag"
    env TASK="$task" FACTOR_KEY="$key" FACTOR_DOMAIN=hydraulic \
        bash PPO_ROM/scripts/score_per_mesh.sh "$tag" "$ckpt" "$STEPS" "$ENVS" \
        >>"$LOG" 2>&1 || say "nonzero exit for $tag"
  done
done

say ""
say "verdict (lateral error must MOVE if the factor reaches the plant):"
python3 - "$OUT" "$MESHES" <<'PY' | tee -a "$LOG"
import json, sys
from pathlib import Path
out, mesh = Path(sys.argv[1]), sys.argv[2]
stacks = ("ppo_rom", "ppo_id_fd", "ppo_aam_id_fd")
for stack in stacks:
    vals = {}
    for key in ("seal_min", "seal_max", "flow_min", "flow_max"):
        path = out / f"{stack}_{key}_m{mesh}.json"
        if not path.is_file():
            vals[key] = None
            continue
        report = json.loads(path.read_text())
        vals[key] = report.get("lateral_error_mean_m", report.get("lateral_error_mean"))
    def delta(a, b):
        if vals.get(a) is None or vals.get(b) is None:
            return "missing"
        d = abs(vals[a] - vals[b])
        return f"{d:.9f}" + ("   <-- INERT" if d < 1e-9 else "")
    print(f"  {stack:>14}: seal {delta('seal_min','seal_max')} | "
          f"flow {delta('flow_min','flow_max')}")
print("\nINERT on both channels means the hydraulic factors do not reach that")
print("stack's plant, and a robustness sweep over them measures nothing.")
PY
