#!/usr/bin/env bash
# Authority sweep under LOAD, for the payload-conditioned controller.
#
#   Delta_x_applied = kappa * Delta_x_actuator,   0 < kappa <= 1
#
# The existing nominal-payload sweep (kappa_sweep.sh, outputs/kappa) stays valid
# for R, IF and the frozen AF base; those stacks have not changed. Its AFa arm
# was the old history residual and is withdrawn.
#
# Re-running AFa at NOMINAL payload would be a tautology: the deadband holds the
# estimate at nominal there, so by Equation (6.11) the adaptive arm IS the frozen
# policy, bit for bit. The question worth asking is whether payload conditioning
# widens the authority margin where it is actually active, so this sweep runs at
# 4.0x payload, the lightest load at which the frozen controller has a measured
# deficit.
#
# Arms, same checkpoint, only the context source differs:
#   AFfL  context zeroed    -> provably the frozen policy
#   AFaL  context estimated -> the deployable arm
#
# Mesh, speed and stage match kappa_sweep.sh so the loaded curves are comparable
# to the nominal ones.
set -u
export IMAGE=isaac-lab:2.3.2-rl052
cd /home/kevinmurera/projects/rrlab_mulag
mkdir -p outputs/kappa_loaded outputs/scorelogs
MESH=${MESH:-1337}
PAYLOAD=${PAYLOAD:-4.0}
LADDER=${LADDER:-"1.0 0.8 0.6 0.5 0.4 0.3 0.2 0.1 0.01"}

CK=logs/rsl_rl/ppo_id_aam_fd/ctxonly17/model_1596.pt
FIT=/workspace/projects/rrlab_mulag/outputs/estimator_final/s17_fit.npz
TASK=RRLAB-Mulag-Verge-ID-AAM-FD-v0

exec >>/home/kevinmurera/kappa_loaded.log 2>&1
echo "=== kappa_loaded start $(TZ=Europe/Berlin date +%H:%M) CEST  payload=${PAYLOAD}x ==="

run_one() {   # $1 arm  $2 kappa
  local S=$1 K=$2
  local OUT=outputs/kappa_loaded/${S}_k${K}_m${MESH}.json
  local LOG=outputs/scorelogs/kappaL_${S}_k${K}_m${MESH}.log
  [ -s "$OUT" ] && { echo "SKIP  $S k=$K"; return; }
  local ABL EXTRA
  if [ "$S" = AFaL ]; then
    ABL=estimated
    EXTRA="export MULAG_ESTIMATOR_PATH=$FIT; export MULAG_ESTIMATOR_BETA=0.02"
  else
    ABL=zero; EXTRA=""
  fi
  ~/bin/isaaclab-run bash -lc "
cd /workspace/projects/rrlab_mulag
export MULAG_DIRECT_VALVE=1
export MULAG_RMA_CONTEXT=1
export MULAG_PAYLOAD_RANGE=${PAYLOAD},${PAYLOAD}
export MULAG_CONTEXT_ABLATE=${ABL}
${EXTRA}
/isaac-sim/python.sh PPO_AAM_FD/scripts/evaluate_gates.py $CK --task $TASK \
  --actuator_authority_alpha $K \
  --num_envs 176 --steps 900 --meshes $MESH --seed 17 \
  --vehicle-speed-kmh 4.0 --stage 6 \
  --report $OUT
" < /dev/null > "$LOG" 2>&1 || true
  if [ -s "$OUT" ]; then echo "OK    $S k=$K  $(TZ=Europe/Berlin date +%H:%M)"
  else echo "FAIL  $S k=$K -- $LOG"; tail -4 "$LOG" | tr -d '\000' | sed 's/^/      | /'; fi
}

for S in AFfL AFaL; do
  for K in $LADDER; do run_one "$S" "$K"; done
done
echo "KAPPA_LOADED_DONE $(TZ=Europe/Berlin date +%H:%M) CEST"
