#!/usr/bin/env bash
set -uo pipefail
# E3 v2 -- MID-EPISODE PLANT CHANGE WITH RECOVERY HEADROOM.
#
# Replaces run_e3_plant_change.sh, which is kept unchanged because it produced
# the results in experiments/E3/results and thesis 7.6.5.
#
# WHY THE FIRST RUN COULD NOT ANSWER H4
#   The x0.6 valve-effectiveness step left the task within 2.9% of unperturbed,
#   in the wrong direction, so there was nothing to recover from. It also
#   perturbed the wrong factor: the AAM decodes seal at R^2 = 0.956 and flow at
#   0.114, so an estimator that encodes seal was asked to detect a change in
#   flow.
#
# WHY THE OBVIOUS FIX DOES NOT WORK EITHER
#   Stepping seal directly is impossible on the deployed stack.
#   RomFactorPerturbation drives _command_rom_system_factors, read only in the
#   command_rom branch; AIF inherits cylinder_dynamics_model = "fd", and
#   CylinderFD.step(valve_cmd, extension, velocity, dP, force) takes no factor
#   argument. On the plant AIF runs, seal is inert.
#
# SO THIS SCRIPT RUNS TWO EXPERIMENTS, NOT ONE. They answer different
# questions and neither is asked to carry the other's claim.
#
#   EXP=A  NATIVE plant change.  Tool mass steps mid-episode. Mass reaches
#          every stack unclamped through PhysX, so this is a property that
#          genuinely exists in the deployed simulation, with no emulator
#          anywhere. The AAM decodes mass at R^2 = -0.008, so it may well not
#          help. That is a legitimate outcome and is the point: it answers
#          "what happens when a real deployed plant property changes?"
#
#   EXP=B  AAM-ALIGNED positive control. A seal-associated GAIN PROXY: the
#          amplitude half of the ROM's seal law, applied to realised actuator
#          motion, alpha_eq(w) = 1/(1+0.625w)^2. It is NOT seal degradation --
#          it omits the leakage term (1+0.25w)^3 and the longer valve time
#          constant, and is named a proxy everywhere it is reported. It answers
#          "if a change occurs along the direction the AAM demonstrably
#          learned, can the adaptation path exploit it?"
#
# MAGNITUDE IS NOT CHOSEN HERE. It comes from the E5 authority curve, which was
# measured on IF alone. AIF is never inspected while the magnitude is selected,
# so the stress cannot have been picked to flatter the adaptive stack.
#
# SUCCESS IS NOT "AAM WINS". The experiment succeeds if it separates:
#   frozen degrades + online recovers + oracle recovers -> online adaptation works
#   frozen degrades + oracle recovers, online does not  -> estimator is limiting
#   frozen degrades + neither recovers                  -> path lacks authority
#   nobody degrades                                     -> too weak, raise it
#   everyone terminates                                 -> too strong, lower it

ROOT=${ROOT:-$HOME/rrlab_mulag}
cd "$ROOT"
EXP=${EXP:-A}
OUT=${OUT:-$ROOT/outputs/e3_plant_change_v2_${EXP}}
TRACES=${TRACES:-$OUT/traces}
LEVELS=${LEVELS:-$ROOT/experiments/E5/results/stress_levels.json}
export MESHES=${MESHES:-1337}
export PARALLEL_MESHES=1
export SPEED_KMH=${SPEED_KMH:-4.0}
export STALL=${STALL:-420}
export STAGE=6
export REPORT_ROOT="$OUT"
export FACTOR_DOMAIN=${FACTOR_DOMAIN:-combined_test}
export TRACE_DIR="$TRACES"
ENVS=${ENVS:-16}
STEPS=${STEPS:-2808}
CHANGE_S=${CHANGE_S:-9.0}
mkdir -p "$OUT" "$TRACES" "$ROOT/experiments/E3/results"
LOG=$OUT/e3_v2.log
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

AIF_CKPT=logs/rsl_rl/ppo_id_aam_fd/headroom_priced_v1/model_299.pt
AIF_TASK=RRLAB-Mulag-Verge-ID-AAM-FD-v0
IF_CKPT=logs/rsl_rl/ppo4_1_sole/idfd_s6_aligned/model_598.pt
IF_TASK=RRLAB-Mulag-Verge-IDFD-v0

run_arm() {
  local tag=$1 ckpt=$2 task=$3
  [ -s "$OUT/${tag}_m${MESHES}.json" ] && { say "reusing $tag"; return 0; }
  local t0; t0=$(date +%s)
  say "start $tag"
  env TASK="$task" bash PPO_ROM/scripts/score_per_mesh.sh \
      "$tag" "$ckpt" "$STEPS" "$ENVS" >>"$LOG" 2>&1 || say "nonzero exit for $tag"
  [ -s "$OUT/${tag}_m${MESHES}.json" ] \
    && say "done  $tag in $(( $(date +%s) - t0 ))s" || say "FAILED $tag"
}

# The AAM ticks at 20 Hz, so the freeze tick is model ticks and not control steps.
FREEZE_TICK_VAL=$(python3 -c "print(int($CHANGE_S*20))")

case "$EXP" in
A)
  # ---- Native: mid-episode tool-mass step ---------------------------------
  # MASS_AFTER is the post-step mass as a multiple of baseline. Take it from
  # the E5 Stage A mass scan: the mildest scale whose lateral error is 1.5-2.5x
  # nominal while contact and termination stay inside the gates.
  MASS_AFTER=${MASS_AFTER:-1.75}
  say "EXP A -- NATIVE mass step 1.00 -> $MASS_AFTER at t=${CHANGE_S}s"
  say "  (mass reaches every stack unclamped; no emulator involved)"
  export MASS_STEP_TIME_S="$CHANGE_S"
  PERTURB=(MASS_STEP_SCALE="$MASS_AFTER")
  SHAM=(MASS_STEP_SCALE="1.0")
  ;;
B)
  # ---- AAM-aligned: seal-associated gain proxy ----------------------------
  # w is the seal severity the proxy stands in for. alpha_eq reproduces the
  # AMPLITUDE effect of the ROM's seal law and nothing else.
  SEAL_W=${SEAL_W:-1.0}
  ALPHA_EQ=$(python3 -c "w=$SEAL_W; print(f'{1.0/(1.0+0.625*w)**2:.4f}')")
  say "EXP B -- seal-associated GAIN PROXY, w=$SEAL_W -> alpha_eq=$ALPHA_EQ at t=${CHANGE_S}s"
  say "  (amplitude half of the seal law only: no leakage term, no added lag)"
  export AUTHORITY_ALPHA="1.0"
  export AUTHORITY_STEP_TIME_S="$CHANGE_S"
  PERTURB=(AUTHORITY_ALPHA_AFTER="$ALPHA_EQ")
  SHAM=(AUTHORITY_ALPHA_AFTER="1.0")
  ;;
*)
  say "unknown EXP=$EXP (expected A or B)"; exit 2 ;;
esac

if [ -s "$LEVELS" ]; then
  say "E5 frozen levels present: $(python3 -c "
import json; d=json.load(open('$LEVELS'))
print('headroom alpha', d.get('e3_headroom_alpha'), '| onset', d.get('onset_alpha'),
      '| breakdown', d.get('breakdown_alpha'))")"
else
  say "WARNING: no frozen E5 levels at $LEVELS."
  say "         The magnitude below was not selected from a measured curve."
fi

# Six arms on ONE checkpoint plus the external reference. online vs frozen is
# the causal contrast: everything is held fixed except whether the estimate may
# keep responding after the change.
env "${PERTURB[@]}" LATENT_MODE=estimated  run_arm "aif_online"   "$AIF_CKPT" "$AIF_TASK"
env "${PERTURB[@]}" LATENT_MODE=frozen FREEZE_TICK="$FREEZE_TICK_VAL" \
                                           run_arm "aif_frozen"   "$AIF_CKPT" "$AIF_TASK"
# 'privileged' routes true z to PPO while the adapter stays on z_hat and needs
# an adapter trained for the latent it consumes, so this is a two-change
# contrast, as in thesis 7.6.6. Set ADAPTER_CKPT accordingly.
env "${PERTURB[@]}" LATENT_MODE=privileged run_arm "aif_oracle"   "$AIF_CKPT" "$AIF_TASK"
env "${PERTURB[@]}" LATENT_MODE=mean       run_arm "aif_mean"     "$AIF_CKPT" "$AIF_TASK"
env "${PERTURB[@]}" LATENT_MODE=shuffled   run_arm "aif_shuffled" "$AIF_CKPT" "$AIF_TASK"
# Sham: schedule fires, plant unchanged. Without it a rise in error after the
# change time reads as a plant effect when it is position along the path.
env "${SHAM[@]}"    LATENT_MODE=estimated  run_arm "aif_sham"     "$AIF_CKPT" "$AIF_TASK"
env "${PERTURB[@]}"                        run_arm "if_reference" "$IF_CKPT"  "$IF_TASK"

say ""
say "recovery metrics:"
python3 experiments/E3/code/analyze_recovery.py \
  --trace online="$TRACES/aif_online_m${MESHES}.npz" \
  --trace frozen="$TRACES/aif_frozen_m${MESHES}.npz" \
  --trace oracle="$TRACES/aif_oracle_m${MESHES}.npz" \
  --trace mean="$TRACES/aif_mean_m${MESHES}.npz" \
  --trace shuffled="$TRACES/aif_shuffled_m${MESHES}.npz" \
  --trace sham="$TRACES/aif_sham_m${MESHES}.npz" \
  --trace if_ref="$TRACES/if_reference_m${MESHES}.npz" \
  --change-time "$CHANGE_S" --reference-arm frozen \
  --json "experiments/E3/results/recovery_exp${EXP}.json" 2>&1 | tee -a "$LOG"

python3 experiments/E3/code/plot_recovery.py \
  --trace-dir "$TRACES" --mesh "$MESHES" --change-time "$CHANGE_S" \
  --out "experiments/E3/figures/ch7_e3_recovery_exp${EXP}.pdf" 2>&1 | tee -a "$LOG"

say ""
say "READ THE SHAM ROW FIRST. If sham post/pre is not ~1.0 the schedule itself"
say "moved the task. If every arm's post/pre is ~1.0 the change is too weak:"
say "raise it using the E5 curve, not by trial and error against AIF."
