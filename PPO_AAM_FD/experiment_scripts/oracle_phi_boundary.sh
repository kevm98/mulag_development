#!/usr/bin/env bash
# Boundary extension for reports/PREREG_ORACLE_PHI_2026-09-06.md.
#
# DECLARED SECOND LOOK. The Stage 1 grid put its most positive b_Y at +0.02 and
# the pre-registration commits to quoting a boundary rather than silently
# extending it. This is the extension, run and reported as a second look with
# the multiplicity on the record.
#
# It settles an UNRESOLVED branch, it does not chase a trend. At 3.0x the
# +0.01 -> +0.02 step is not resolved on either channel (cut +3.18 mm,
# CI [-2.46, +8.88]; lateral +11.69 mm, CI [-2.35, +22.46]), so the apparent
# march toward the positive edge is roughness. What remains true is that larger
# positive b_Y was never measured, and every point that WAS measured is
# significantly worse than the baseline on both channels.
#
# Run at 3.0x only: the payload the Stage 1 result makes decisive, where the
# base has 15.7 mm of cut headroom and every offset still lost.
set -u
export IMAGE=isaac-lab:2.3.2-rl052
BASE=logs/rsl_rl/ppo_id_aam_fd/distilled_v1/phase1_init_70.pt
MESH=${MESH:-2718}
ENVS_N=${ENVS_N:-176}
STEPS=${STEPS:-900}
M=${M:-3.0}

# Baseline first, then the unmeasured positive branch. +0.02 is repeated so the
# extension is anchored to a Stage 1 point measured in the SAME process.
GRID="0:0,0.02:0,0.03:0,0.04:0,0.06:0,0.08:0"

cd /home/kevinmurera
run() { ~/bin/isaaclab-run bash -lc "$1" 2>/dev/null || true; }
mkdir -p /home/kevinmurera/projects/rrlab_mulag/outputs/oracle_phi

OUT="outputs/oracle_phi/boundary_m${MESH}_p${M}.json"
echo "=== boundary extension, payload ${M}x, mesh ${MESH} ==="
date -u +%FT%TZ
run "
cd /workspace/projects/rrlab_mulag
export MULAG_DIRECT_VALVE=1
export MULAG_RMA_CONTEXT=1
export MULAG_PAYLOAD_RANGE=$M,$M
/isaac-sim/python.sh PPO_AAM_FD/scripts/evaluate_gates.py $BASE \
  --num_envs $ENVS_N --steps $STEPS --meshes $MESH --seed 17 \
  --paired_task_bias $GRID \
  --report $OUT 2>&1 | grep -E 'wrote|rror|paired task bias' | tail -10
"
echo "ORACLE_PHI_BOUNDARY_COMPLETE"
