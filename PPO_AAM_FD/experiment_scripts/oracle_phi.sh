#!/usr/bin/env bash
# Stage 1 of reports/PREREG_ORACLE_PHI_2026-09-06.md: does a fixed two-parameter
# task-space reference offset improve both scored channels at a given payload?
#
# One process per payload. Every offset inside a process is rolled against the
# SAME environments, terrain and plant draw, so the comparison is paired;
# invariant 19 makes across-process marginal medians unreadable.
#
# The b_Y grid is tighter than the pre-registration's first draft because the
# 2026-09-06 plumbing smoke measured a 20 mm line displacement moving cut-height
# error 15.8 -> 154 mm. The grid was retightened BEFORE any scored arm ran, and
# the pre-registration records the change.
set -u
export IMAGE=isaac-lab:2.3.2-rl052
BASE=logs/rsl_rl/ppo_id_aam_fd/distilled_v1/phase1_init_70.pt
MESH=${MESH:-2718}
ENVS_N=${ENVS_N:-176}
STEPS=${STEPS:-900}
# Informative payloads first: 1.0, 3.0 and 5.0 are the three the Phase 0 gate
# already characterised, so the shape is readable before 2.0 and 4.0 finish.
MASSES=${MASSES:-"1.0 3.0 5.0 2.0 4.0"}

# Baseline FIRST; the evaluator refuses a sweep that does not start at 0:0.
GRID="0:0,-0.02:0,-0.01:0,0.01:0,0.02:0,0:-0.15,0:-0.08,0:-0.04,0:-0.02,0:0.02"

cd /home/kevinmurera
run() { ~/bin/isaaclab-run bash -lc "$1" 2>/dev/null || true; }
mkdir -p /home/kevinmurera/projects/rrlab_mulag/outputs/oracle_phi

for M in $MASSES; do
  OUT="outputs/oracle_phi/stage1_m${MESH}_p${M}.json"
  echo "=== payload ${M}x, mesh ${MESH} ==="
  date -u +%FT%TZ
  run "
cd /workspace/projects/rrlab_mulag
export MULAG_DIRECT_VALVE=1
export MULAG_RMA_CONTEXT=1
export MULAG_PAYLOAD_RANGE=$M,$M
/isaac-sim/python.sh PPO_AAM_FD/scripts/evaluate_gates.py $BASE \
  --num_envs $ENVS_N --steps $STEPS --meshes $MESH --seed 17 \
  --paired_task_bias $GRID \
  --report $OUT 2>&1 | grep -E 'wrote|rror|paired task bias' | tail -15
"
  echo "ORACLE_PHI_DONE_p${M}"
done
echo "ORACLE_PHI_STAGE1_COMPLETE"
