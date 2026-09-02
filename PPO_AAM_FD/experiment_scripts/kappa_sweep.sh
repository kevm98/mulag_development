#!/usr/bin/env bash
# E5 common robustness-boundary sweep: actuator authority kappa.
#
#   Delta_x_applied = kappa * Delta_x_actuator,   0 < kappa <= 1
#
# Applied at the SAME boundary on every stack (after the actuator model, before
# the PhysX target), so all four receive an identical physical stress. This is
# the axis the deployment factors could not provide: they are inert on the FD
# plant and clamped on the ROM (experiments/E5/README.md).
#
# Stacks:  R   command ROM
#          IF  ID->FD
#          AFf AAM-FD frozen base, no residual
#          AFa AAM-FD base + live-history residual (seed 17, model_300, alpha 0.5)
#
# STAGE 0 runs first and is a GATE: kappa=0.01 must be catastrophically worse
# than kappa=1.0 on every stack. A flat curve has two readings -- the controller
# absorbs the stress, or the stress never arrived -- and E5's superseded sweep
# died on exactly that ambiguity.
#
# EVERY stack is scored at 4.0 km/h, the trained speed (E5's ROM ladder: "the
# trained speed; not the 2-7 default"). Over the 2-7 range R and IF sit at 910
# and 1014 mm lateral -- broken before kappa is applied -- while at 4.0 they are
# at 41 and 24 mm. A kappa_crit is undefined for a stack that fails at kappa=1.
#
# Plant is NOMINAL for every stack: kappa is the only stress. AF stacks still
# need their own observation-layout vars (DIRECT_VALVE / RMA_CONTEXT), which are
# architecture settings, not plant perturbations.
set -u
export IMAGE=isaac-lab:2.3.2-rl052
cd /home/kevinmurera/projects/rrlab_mulag
mkdir -p outputs/kappa outputs/scorelogs
MESH=${MESH:-1337}
LADDER=${LADDER:-"1.0 0.8 0.6 0.5 0.4 0.3 0.2 0.1"}
STAGE0=${STAGE0:-"1.0 0.01"}

ROM_CK=logs/rsl_rl/ppo3_1_sole/rom_s6_aligned/model_598.pt
IDFD_CK=logs/rsl_rl/ppo4_1_sole/idfd_s6_aligned/model_598.pt
AF_BASE=logs/rsl_rl/ppo_id_aam_fd/distilled_v1/phase1_init_70.pt
AF_RES=logs/rsl_rl/ppo_id_aam_fd/rebal17_history/model_300.pt

run_one() {   # $1 stack  $2 kappa
  local S=$1 K=$2
  local OUT=outputs/kappa/${S}_k${K}_m${MESH}.json
  local LOG=outputs/scorelogs/kappa_${S}_k${K}_m${MESH}.log
  [ -s "$OUT" ] && { echo "SKIP  $S k=$K"; return; }
  local TASK CK EXTRA ENVS
  case "$S" in
    R)   TASK=RRLAB-Mulag-Verge-ROM-v0;        CK=$ROM_CK;  EXTRA=""; ENVS="" ;;
    IF)  TASK=RRLAB-Mulag-Verge-IDFD-v0;       CK=$IDFD_CK; EXTRA=""; ENVS="" ;;
    AFf) TASK=RRLAB-Mulag-Verge-ID-AAM-FD-v0;  CK=$AF_BASE; EXTRA=""
         ENVS="export MULAG_DIRECT_VALVE=1; export MULAG_RMA_CONTEXT=1" ;;
    AFa) TASK=RRLAB-Mulag-Verge-ID-AAM-FD-v0;  CK=$AF_BASE
         EXTRA="--residual_checkpoint $AF_RES --residual_alpha 0.5 --residual_history 20"
         ENVS="export MULAG_DIRECT_VALVE=1; export MULAG_RMA_CONTEXT=1" ;;
  esac
  ~/bin/isaaclab-run bash -lc "
cd /workspace/projects/rrlab_mulag
${ENVS}
/isaac-sim/python.sh PPO_ROM/scripts/evaluate_gates.py $CK --task $TASK \
  --actuator_authority_alpha $K \
  --num_envs 176 --steps 900 --meshes $MESH --seed 17 \
  --vehicle-speed-kmh 4.0 --stage 6 \
  $EXTRA --report $OUT
" < /dev/null > "$LOG" 2>&1 || true
  if [ -s "$OUT" ]; then echo "OK    $S k=$K  $(date -u +%H:%M)"
  else echo "FAIL  $S k=$K -- $LOG"; tail -4 "$LOG" | tr -d '\000' | sed 's/^/      | /'; fi
}

echo "=== STAGE 0 (gate): kappa 1.0 vs 0.01 on every stack ==="
for S in AFf AFa R IF; do for K in $STAGE0; do run_one "$S" "$K"; done; done
echo "STAGE0_DONE $(date -u +%H:%M)"
python3 PPO_AAM_FD/experiment_scripts/kappa_report.py --stage0 --mesh "$MESH" || true

if [ "${GATE_ONLY:-0}" = "1" ]; then echo "GATE_ONLY set -- stopping before the ladder"; exit 0; fi

echo "=== LADDER ==="
for S in AFf AFa R IF; do for K in $LADDER; do run_one "$S" "$K"; done; done
echo "KAPPA_SWEEP_DONE $(date -u +%H:%M)"
