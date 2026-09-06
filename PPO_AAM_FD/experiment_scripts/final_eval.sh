#!/usr/bin/env bash
# FINAL EVALUATION -- locked architecture (--train_context_only, context columns).
# No wait loops, no chaining: both caused stalls today. Sequential by construction.
#
# Arms, all the SAME checkpoint per seed; only the payload number's source changes:
#   Frozen    context zeroed -> base(x) + W_ctx.0 = base(x), provably the frozen policy.
#             Identical across seeds, so scored ONCE per mass rather than per seed.
#   Oracle    true payload, the privileged ceiling.
#   Adaptive  payload estimated online from the task observation. The deployable arm.
#   Wrong     true payload SHUFFLED across environments on a mixed-payload plant:
#             same distribution and magnitude, pairing destroyed. At fixed payload a
#             "wrong" context would be the zero vector, which IS the frozen policy,
#             so that test would be vacuous; the shuffle is the version with content.
set -u
export IMAGE=isaac-lab:2.3.2-rl052
cd /home/kevinmurera
R=/home/kevinmurera/projects/rrlab_mulag
MESH=3141          # held out: never used to design, diagnose or read this campaign
ENVS=176; STEPS=900
run() { ~/bin/isaaclab-run bash -lc "$1" 2>/dev/null || true; }
exec >>/home/kevinmurera/final_eval.log 2>&1
echo "=== final_eval start $(TZ=Europe/Berlin date +%H:%M) CEST ==="
mkdir -p $R/outputs/final $R/outputs/estimator_final

ck() { ls $R/logs/rsl_rl/ppo_id_aam_fd/ctxonly$1/model_*.pt 2>/dev/null | sed 's/.*model_//;s/\.pt//' | sort -n | tail -1; }
for S in 17 23 101; do
  C=$(ck $S); [ -z "$C" ] && { echo "FINAL_ABORT seed $S has no checkpoint"; exit 1; }
  echo "  seed $S -> model_${C}.pt"
done

# --- per-seed estimators: each fit from ITS OWN policy's rollout -------------
# The regression is on the observation distribution that policy induces, so
# reusing another seed's fit would leak that seed's behaviour into this one.
# Seed 17 is fitted here too, from ITS OWN rollout with THIS script. An earlier
# version reused outputs/estimator/ctxonly_fit.npz for seed 17: that file is a
# 66-feature linear fit from a pre-degree-2 script, while seeds 23 and 101 get
# 132-feature quadratic fits. Mixing estimator families across seeds makes any
# seed-17 difference unattributable, so it is refitted like the others.
for S in 17 23 101; do
  C=$(ck $S); CKPT=logs/rsl_rl/ppo_id_aam_fd/ctxonly${S}/model_${C}.pt
  if [ -f "$R/outputs/estimator_final/s${S}_fit.npz" ]; then
    echo "FIT_REUSE_s${S} (already fitted by this script)"; continue
  fi
  echo "--- estimator collect seed $S $(TZ=Europe/Berlin date +%H:%M) ---"
  run "
cd /workspace/projects/rrlab_mulag
export MULAG_DIRECT_VALVE=1
export MULAG_RMA_CONTEXT=1
export MULAG_PAYLOAD_RANGE=1.0,5.0
export MULAG_PAYLOAD_RESAMPLE=1
export MULAG_CONTEXT_ABLATE=live
/isaac-sim/python.sh PPO_AAM_FD/scripts/evaluate_gates.py $CKPT \
  --num_envs $ENVS --steps $STEPS --meshes 2718 --seed 17 \
  --obs_dump outputs/estimator_final/s${S}_mixed.npz --obs_dump_every 10 \
  --report outputs/final/collect_s${S}.json 2>&1 | grep -E 'wrote obs dump|rror' | tail -1
"
  run "
cd /workspace/projects/rrlab_mulag
/isaac-sim/python.sh PPO_AAM_FD/experiment_scripts/fit_estimator.py \
  --dir outputs/estimator_final --prefix s${S} \
  --out outputs/estimator_final/s${S}_fit.npz 2>&1 | tail -3
"
  [ -f "$R/outputs/estimator_final/s${S}_fit.npz" ] || { echo "FINAL_ABORT no fit seed $S"; exit 1; }
  echo "FIT_OK_s${S}"
done

# --- Frozen: once per mass, shared across seeds -----------------------------
C17=$(ck 17)
for M in 3.0 4.0 5.0; do
  echo "--- Frozen p${M} $(TZ=Europe/Berlin date +%H:%M) ---"
  run "
cd /workspace/projects/rrlab_mulag
export MULAG_DIRECT_VALVE=1
export MULAG_RMA_CONTEXT=1
export MULAG_PAYLOAD_RANGE=$M,$M
export MULAG_CONTEXT_ABLATE=zero
/isaac-sim/python.sh PPO_AAM_FD/scripts/evaluate_gates.py logs/rsl_rl/ppo_id_aam_fd/ctxonly17/model_${C17}.pt \
  --num_envs $ENVS --steps $STEPS --meshes $MESH --seed 17 \
  --report outputs/final/frozen_p${M}.json 2>&1 | grep -E 'wrote|rror' | tail -1
"
  echo "SCORED_frozen_p${M}"
done

# --- Oracle and Adaptive: 3 seeds x 3 masses --------------------------------
for S in 17 23 101; do
  C=$(ck $S); CKPT=logs/rsl_rl/ppo_id_aam_fd/ctxonly${S}/model_${C}.pt
  SFIT=/workspace/projects/rrlab_mulag/outputs/estimator_final/s${S}_fit.npz
  for M in 3.0 4.0 5.0; do
    for ARM in oracle adaptive; do
      if [ "$ARM" = oracle ]; then ABL=live; EX=""
      else ABL=estimated; EX="export MULAG_ESTIMATOR_PATH=$SFIT
export MULAG_ESTIMATOR_BETA=0.02"; fi
      echo "--- s${S} ${ARM} p${M} $(TZ=Europe/Berlin date +%H:%M) ---"
      run "
cd /workspace/projects/rrlab_mulag
export MULAG_DIRECT_VALVE=1
export MULAG_RMA_CONTEXT=1
export MULAG_PAYLOAD_RANGE=$M,$M
export MULAG_CONTEXT_ABLATE=$ABL
$EX
/isaac-sim/python.sh PPO_AAM_FD/scripts/evaluate_gates.py $CKPT \
  --num_envs $ENVS --steps $STEPS --meshes $MESH --seed 17 \
  --report outputs/final/${ARM}_s${S}_p${M}.json 2>&1 | grep -E 'wrote|rror' | tail -1
"
      echo "SCORED_${ARM}_s${S}_p${M}"
    done
  done
done

# --- Wrong-information control, mixed payload, seed 17 ----------------------
for ARM in zero live shuffle; do
  echo "--- wrong-info ${ARM} $(TZ=Europe/Berlin date +%H:%M) ---"
  run "
cd /workspace/projects/rrlab_mulag
export MULAG_DIRECT_VALVE=1
export MULAG_RMA_CONTEXT=1
export MULAG_PAYLOAD_RANGE=1.0,5.0
export MULAG_PAYLOAD_RESAMPLE=1
export MULAG_CONTEXT_ABLATE=${ARM}
/isaac-sim/python.sh PPO_AAM_FD/scripts/evaluate_gates.py logs/rsl_rl/ppo_id_aam_fd/ctxonly17/model_${C17}.pt \
  --num_envs $ENVS --steps $STEPS --meshes $MESH --seed 17 \
  --report outputs/final/wronginfo_${ARM}.json 2>&1 | grep -E 'wrote|rror' | tail -1
"
  echo "SCORED_wronginfo_${ARM}"
done
echo "FINAL_EVAL_COMPLETE $(TZ=Europe/Berlin date +%H:%M) CEST"
