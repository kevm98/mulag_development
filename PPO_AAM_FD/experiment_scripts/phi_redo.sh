#!/usr/bin/env bash
# The task-space oracle, re-run under a layout whose A/A passes.
#
# The original ran all offsets inside ONE process, baseline first. That layout
# failed its A/A on 2026-09-06: a repeat of the baseline setting scored -61.8%
# cut and -43.1% lateral against itself and was the worst arm in the table. Its
# negative is withdrawn, so this is not a confirmation run; the question is open
# again.
#
# One arm per process, paired by environment index under common random numbers.
# That is invariant 3's other permitted layout and the one the context-only
# campaign uses. Six minutes per arm instead of two and a half.
#
# Payload 3.0x only: the decisive payload, where the base has 15-23 mm of cut
# headroom and real degradation. If an offset Pareto-improves here, 1.0x and
# 5.0x follow; if none does under a valid instrument, the family is closed for
# real this time.
set -u
export IMAGE=isaac-lab:2.3.2-rl052
BASE=logs/rsl_rl/ppo_id_aam_fd/distilled_v1/phase1_init_70.pt
MESH=2718; ENVS_N=176; STEPS=900; M=${M:-3.0}
cd /home/kevinmurera
run() { ~/bin/isaaclab-run bash -lc "$1" 2>/dev/null || true; }
mkdir -p /home/kevinmurera/projects/rrlab_mulag/outputs/phi_redo

# Wait for the context-only campaign so two evaluations never overlap: one eval
# peaks near 17 GB of host RAM and two OOM this 31 GB box.
exec >>/home/kevinmurera/phi_redo.log 2>&1
echo "=== phi_redo armed $(date -u +%FT%TZ) ==="
while ps -eo args | grep -qE "[c]txonly.sh"; do sleep 60; done
while ps -eo args | grep -qE "[e]valuate_gates.py"; do sleep 60; done
echo "GPU free $(date -u +%FT%TZ)"

for PHI in "0 0" "-0.02 0" "-0.01 0" "0.01 0" "0.02 0" "0 -0.15" "0 -0.08" "0 -0.04" "0 -0.02" "0 0.02"; do
  set -- $PHI; BY=$1; BZ=$2
  TAG=$(echo "y${BY}_z${BZ}" | tr -d ' ' | tr '.-' 'pm')
  echo "=== phi b_Y=$BY b_Z=$BZ ==="; date -u +%FT%TZ
  run "
cd /workspace/projects/rrlab_mulag
export MULAG_DIRECT_VALVE=1
export MULAG_RMA_CONTEXT=1
export MULAG_PAYLOAD_RANGE=$M,$M
/isaac-sim/python.sh PPO_AAM_FD/scripts/evaluate_gates.py $BASE \
  --num_envs $ENVS_N --steps $STEPS --meshes $MESH --seed 17 \
  --task_bias_y $BY --task_bias_z $BZ \
  --report outputs/phi_redo/${TAG}_p${M}_m${MESH}.json 2>&1 | grep -E 'wrote|rror' | tail -2
"
  echo "PHI_DONE_${TAG}"
done
echo "PHI_REDO_COMPLETE $(date -u +%FT%TZ)"
