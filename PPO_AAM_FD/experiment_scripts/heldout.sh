#!/usr/bin/env bash
# Confirm the online result on the HELD-OUT meshes.
#
# 2718 is the selection mesh: the context-only campaign was designed, diagnosed
# and read on it, so a result there is not confirmatory. 3141 and 8191 are the
# reserved meshes this project has used for every previous campaign, and the
# online-gradient investigation is the cautionary case -- a favourable rung on
# 2718 reversed completely on 3141.
#
# Arms: ctxzero (which IS the frozen base by construction) and ctxest (the
# online estimate). The oracle is skipped: the deployable claim is the online
# one, and GPU is better spent widening the evidence for it than repeating a
# ceiling.
set -u
export IMAGE=isaac-lab:2.3.2-rl052
ENVS_N=176; STEPS=900
MESHES=${MESHES:-"3141 8191"}
PAYLOADS=${PAYLOADS:-"5.0 3.0"}
cd /home/kevinmurera
R=/home/kevinmurera/projects/rrlab_mulag
run() { ~/bin/isaaclab-run bash -lc "$1" 2>/dev/null || true; }
exec >>/home/kevinmurera/heldout.log 2>&1
echo "=== heldout armed $(date -u +%FT%TZ) ==="
C=$(ls $R/logs/rsl_rl/ppo_id_aam_fd/ctxonly17/model_*.pt | sed 's/.*model_//;s/\.pt//' | sort -n | tail -1)
CKPT=logs/rsl_rl/ppo_id_aam_fd/ctxonly17/model_${C}.pt
mkdir -p $R/outputs/heldout

for MESH in $MESHES; do
  for M in $PAYLOADS; do
    for ARM in ctxzero ctxest; do
      if [ "$ARM" = ctxest ]; then
        ABL=estimated
        EXTRA="export MULAG_ESTIMATOR_PATH=/workspace/projects/rrlab_mulag/outputs/estimator/ctxonly_fit.npz
export MULAG_ESTIMATOR_BETA=0.02"
      else
        ABL=zero; EXTRA=""
      fi
      # One evaluator at a time: each peaks near 17 GB and two OOM this box.
      while ps -eo args | grep -qE "[e]valuate_gates.py"; do sleep 45; done
      echo "=== $ARM mesh $MESH payload $M ==="; date -u +%FT%TZ
      run "
cd /workspace/projects/rrlab_mulag
export MULAG_DIRECT_VALVE=1
export MULAG_RMA_CONTEXT=1
export MULAG_PAYLOAD_RANGE=$M,$M
export MULAG_CONTEXT_ABLATE=$ABL
$EXTRA
/isaac-sim/python.sh PPO_AAM_FD/scripts/evaluate_gates.py $CKPT \
  --num_envs $ENVS_N --steps $STEPS --meshes $MESH --seed 17 \
  --report outputs/heldout/${ARM}_p${M}_m${MESH}.json 2>&1 | grep -E 'wrote|rror' | tail -2
"
      echo "SCORED_${ARM}_p${M}_m${MESH}"
    done
  done
done
echo "HELDOUT_COMPLETE $(date -u +%FT%TZ)"
