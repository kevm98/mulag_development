#!/usr/bin/env bash
# Can the DLS teacher track the NEW stage-6 terrain?
#
# Every teacher number on record was measured at STAGE 4 on the old lawn_rough
# relief, on the RIGID arm. Two things have changed since, and both bear
# directly on the question:
#
#   * the arm is now compliant (terrain_arm_stiffness_scale = 0.35), and the
#     teacher degrades badly on it -- re-derived 2026-08-01 at stage 4, ROM-A
#     went from ground_contact 0.972 / p95 cut height 0.0 m to 0.888 / 0.114 m;
#   * stage 6 is a different task: flat road with discrete raised chunks, and a
#     tracking line swung to 2.2x amplitude.
#
# So the honest answer to "can the teacher track this" is: unmeasured. This
# measures it, on the same terrain the student is graded on.
#
# --probe_only drives the teacher with the execute mask fully on and the student
# action pinned to zero, then reports tracking, contact and head metrics without
# training anything. It writes NO checkpoint and touches NO policy.
#
# Reference to compare against -- the current PPO3 seed checkpoint on this exact
# terrain (stage 6 held-out, 2.2x, 32 envs, 960 steps), all 17 gates passing:
#
#     ground_contact_fraction        0.9475
#     mean_abs_cut_height_error_m    0.0038
#     p95_abs_cut_height_error_m     0.0
#     mean_abs_lateral_error_m       0.0740
#     min_head_alignment             0.9744
#     p95_tool_acceleration_mps2     44.55
#     visible_lift_event_count       152
#
# If the teacher lands worse than that, seeding from it is a step backwards and
# the distillation route is closed for this problem -- which is the decision
# this probe exists to make.
set -u

cd "$(dirname "$0")/../.." || exit 1
PY=/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
OUT=${1:-PPO3_SOLE_PhysX/reports/stage6/teacher_probe}
NUM_ENVS=${NUM_ENVS:-32}
STEPS=${STEPS:-960}
mkdir -p "$OUT"

for act in rom idfd; do
  r="$OUT/teacher_stage6_${act}.json"
  [ -f "$r" ] && { echo "[SKIP] $act"; continue; }
  echo "[RUN] teacher probe $act stage 6 (flat_patches, 2.2x)"
  # NOTE: --probe_only raises if the teacher preflight fails. That is the
  # desired behaviour -- a broken teacher must not be reported as a slow one.
  $PY PPO3_SOLE_PhysX/scripts/distill_terrain.py \
    --headless --device cuda:0 --actuator "$act" \
    --num_envs "$NUM_ENVS" --seed 17 --stage 6 \
    --probe_steps "$STEPS" --probe_only \
    --head_joint_action \
    --report "$r" > "$OUT/teacher_stage6_${act}.log" 2>&1
  echo "[DONE] $act exit=$?"
done

echo "[TEACHER PROBE DONE] $OUT"
$PY - "$OUT" <<'PYEOF'
import json, sys
from pathlib import Path

# Only the fields that answer the question, side by side with the student.
STUDENT = {
    "ground_contact_fraction": 0.9475,
    "mean_abs_cut_height_error_m": 0.0038,
    "p95_abs_cut_height_error_m": 0.0,
    "mean_abs_lateral_error_m": 0.0740,
    "min_head_alignment": 0.9744,
    "p01_head_alignment": 0.9984,
    "p95_tool_acceleration_mps2": 44.55,
}
out = Path(sys.argv[1])
print(f"\n{'metric':38s} {'student':>10s} " + " ".join(f"{a:>10s}" for a in ("rom", "idfd")))
rows = {}
for act in ("rom", "idfd"):
    path = out / f"teacher_stage6_{act}.json"
    if path.is_file():
        rows[act] = json.load(path.open()).get("preflight", {})
for key, ref in STUDENT.items():
    line = f"{key:38s} {ref:10.4f} "
    for act in ("rom", "idfd"):
        value = rows.get(act, {}).get(key)
        line += f" {value:10.4f}" if isinstance(value, (int, float)) else f" {'-':>10s}"
    print(line)
for act in ("rom", "idfd"):
    if act in rows:
        print(f"{act}: teacher preflight passed = {rows[act].get('passed')}")
PYEOF
