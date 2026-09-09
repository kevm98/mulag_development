#!/usr/bin/env bash
# A/A instrument floor at 4.0x payload.
#
# Measures what this instrument reports as a difference when there is none, in
# the exact configuration of the headline claim. The existing floors were
# measured at 3.0x and under a percentage convention; the claim is now a paired
# median difference in mm at 4.0x, so the floor is remeasured there.
#
# Both arms are the frozen policy. They differ only in the fifth significant
# figure of the payload, 4.0 against 4.0001, a 0.0025 % mass change that is
# physically negligible but forces a different random draw. This follows the
# precedent of the 2026-09-06 A/A, which perturbed the fourth decimal of an
# unrelated parameter. Comparing a run against a bit-identical repeat would test
# determinism, not the floor.
#
# Any paired difference this reports is noise. It bounds what the campaign may
# call an effect.
set -u
export IMAGE=isaac-lab:2.3.2-rl052
cd /home/kevinmurera
R=$HOME/rrlab_mulag
MESH=3141; ENVS=176; STEPS=900
run() { ~/bin/isaaclab-run bash -lc "$1" 2>/dev/null || true; }
exec >>/home/kevinmurera/aa_floor.log 2>&1
echo "=== aa_floor start $(TZ=Europe/Berlin date +%H:%M) CEST ==="
mkdir -p $R/outputs/aa_floor

i=0
for M in 4.0 4.0001; do
  i=$((i+1))
  OUT=outputs/aa_floor/aa${i}_p${M}.json
  [ -s "$R/$OUT" ] && { echo "SKIP aa${i}"; continue; }
  echo "--- A/A arm ${i}, payload ${M} $(TZ=Europe/Berlin date +%H:%M) ---"
  run "
cd /workspace/projects/rrlab_mulag
export MULAG_DIRECT_VALVE=1
export MULAG_RMA_CONTEXT=1
export MULAG_PAYLOAD_RANGE=$M,$M
export MULAG_CONTEXT_ABLATE=zero
/isaac-sim/python.sh PPO_AAM_FD/scripts/evaluate_gates.py logs/rsl_rl/ppo_id_aam_fd/ctxonly17/model_1596.pt \
  --num_envs $ENVS --steps $STEPS --meshes $MESH --seed 17 \
  --report $OUT 2>&1 | grep -E 'wrote|rror' | tail -1
"
  echo "SCORED_aa${i}"
done
echo "AA_FLOOR_DONE $(TZ=Europe/Berlin date +%H:%M) CEST"
