#!/usr/bin/env bash
# Second held-out mesh for the payload result.
#
# The main table is scored on mesh 3141 only. Mesh 8191 is reserved and has not
# been used to design, diagnose or read this campaign, so it is an independent
# repetition of the headline claim. Held-out repetitions rank above extra
# training seeds in this project's evidence hierarchy, which is why this runs
# rather than a fourth seed.
#
# Arms are frozen (context zeroed, provably the base policy) and adaptive
# (context estimated), at the two payloads where the frozen policy has a
# measured deficit. Oracle is omitted: it is a reference point, not part of the
# claim, and it is already characterised on 3141.
set -u
export IMAGE=isaac-lab:2.3.2-rl052
cd /home/kevinmurera
R=$HOME/rrlab_mulag
MESH=8191; ENVS=176; STEPS=900
run() { ~/bin/isaaclab-run bash -lc "$1" 2>/dev/null || true; }
exec >>/home/kevinmurera/heldout8191.log 2>&1
echo "=== heldout8191 start $(TZ=Europe/Berlin date +%H:%M) CEST ==="
mkdir -p $R/outputs/heldout8191

C17=1596
for M in 4.0 5.0; do
  OUT=outputs/heldout8191/frozen_p${M}.json
  if [ -s "$R/$OUT" ]; then echo "SKIP frozen p${M}"; else
  echo "--- frozen p${M} $(TZ=Europe/Berlin date +%H:%M) ---"
  run "
cd /workspace/projects/rrlab_mulag
export MULAG_DIRECT_VALVE=1
export MULAG_RMA_CONTEXT=1
export MULAG_PAYLOAD_RANGE=$M,$M
export MULAG_CONTEXT_ABLATE=zero
/isaac-sim/python.sh PPO_AAM_FD/scripts/evaluate_gates.py logs/rsl_rl/ppo_id_aam_fd/ctxonly17/model_${C17}.pt \
  --num_envs $ENVS --steps $STEPS --meshes $MESH --seed 17 \
  --report $OUT 2>&1 | grep -E 'wrote|rror' | tail -1
"
  fi
  echo "SCORED_frozen_p${M}"
done

for S in 17 23 101; do
  FIT=/workspace/projects/rrlab_mulag/outputs/estimator_final/s${S}_fit.npz
  for M in 4.0 5.0; do
    OUT=outputs/heldout8191/adaptive_s${S}_p${M}.json
    if [ -s "$R/$OUT" ]; then echo "SKIP adaptive s${S} p${M}"; continue; fi
    echo "--- adaptive s${S} p${M} $(TZ=Europe/Berlin date +%H:%M) ---"
    run "
cd /workspace/projects/rrlab_mulag
export MULAG_DIRECT_VALVE=1
export MULAG_RMA_CONTEXT=1
export MULAG_PAYLOAD_RANGE=$M,$M
export MULAG_CONTEXT_ABLATE=estimated
export MULAG_ESTIMATOR_PATH=$FIT
export MULAG_ESTIMATOR_BETA=0.02
/isaac-sim/python.sh PPO_AAM_FD/scripts/evaluate_gates.py logs/rsl_rl/ppo_id_aam_fd/ctxonly${S}/model_1596.pt \
  --num_envs $ENVS --steps $STEPS --meshes $MESH --seed 17 \
  --report $OUT 2>&1 | grep -E 'wrote|rror' | tail -1
"
    echo "SCORED_adaptive_s${S}_p${M}"
  done
done
echo "HELDOUT8191_DONE $(TZ=Europe/Berlin date +%H:%M) CEST"
