#!/usr/bin/env bash
# Oracle probe on the command-smoothing knobs, at the payloads where the plant
# actually fails.
#
# Motivated by RESULT_rootcause_contact_2026-09-06.md. The heavy-payload failure
# is impact and rebound (hard impacts 0.164 -> 0.445, p95 tool acceleration
# 50.7 -> 90.4 m/s2 from payload 1.0x to 3.0x) with command authority to spare
# (saturation under 1%). Task-space reference offsets could not fix it because
# they act on position. These two knobs act on impact:
#
#   sole_action_filter    beta in filtered = beta*previous + (1-beta)*raw
#                         HIGHER = gentler
#   sole_max_action_rate  per-step clamp, LOWER = gentler
#
# If a gentler setting Pareto-improves at heavy payload and the nominal setting
# is still best at 1.0x, then a payload-scheduled scalar is the whole adaptation
# and it is about as simple as an online mechanism can be.
#
# Same discipline as the task-space probe: all arms paired inside one process
# against the same environments, baseline first, clearance guard reported.
set -u
export IMAGE=isaac-lab:2.3.2-rl052
BASE=logs/rsl_rl/ppo_id_aam_fd/distilled_v1/phase1_init_70.pt
MESH=${MESH:-2718}
ENVS_N=${ENVS_N:-176}
STEPS=${STEPS:-900}
PAYLOADS=${PAYLOADS:-"3.0 5.0 1.0"}

# base first (the evaluator refuses otherwise). Then a beta ladder at the
# nominal rate, then a rate ladder at a fixed beta with its own anchor.
GRID="base,0.35:0.25,0.70:0.25,0.85:0.25,0.60:0.25,0.60:0.12,0.60:0.06"

cd /home/kevinmurera
run() { ~/bin/isaaclab-run bash -lc "$1" 2>/dev/null || true; }
mkdir -p /home/kevinmurera/projects/rrlab_mulag/outputs/smoothing

for M in $PAYLOADS; do
  OUT="outputs/smoothing/smooth_m${MESH}_p${M}.json"
  echo "=== smoothing oracle, payload ${M}x ==="; date -u +%FT%TZ
  run "
cd /workspace/projects/rrlab_mulag
export MULAG_DIRECT_VALVE=1
export MULAG_RMA_CONTEXT=1
export MULAG_PAYLOAD_RANGE=$M,$M
/isaac-sim/python.sh PPO_AAM_FD/scripts/evaluate_gates.py $BASE \
  --num_envs $ENVS_N --steps $STEPS --meshes $MESH --seed 17 \
  --paired_smoothing $GRID \
  --report $OUT 2>&1 | grep -E 'wrote|rror|smoothing' | tail -12
"
  echo "SMOOTH_DONE_p${M}"
done
echo "SMOOTHING_ORACLE_COMPLETE"
