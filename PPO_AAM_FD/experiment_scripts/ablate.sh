#!/usr/bin/env bash
# Two gaps in the online result, closed cheaply.
#
# (1) WHY does the online arm beat the oracle? At payload 3.0x the estimator
#     reads 3.11x against a true 3.00x -- essentially exact -- yet ctxest holds
#     lateral at 82 mm where ctxlive collapses to 509 mm. Accuracy cannot
#     explain that. The remaining difference is the EMA: the estimate starts
#     each episode at nominal and converges over ~1/beta steps, so the policy
#     ramps its conditioning instead of applying it from step 0.
#
#     Test it as a dose-response on beta. beta = 1.0 disables smoothing
#     entirely (the raw per-step estimate, no ramp), so if the soft start is the
#     mechanism, beta = 1.0 should behave like the oracle and lose lateral.
#     beta = 0.005 ramps four times slower and should be at least as good as
#     0.02. A monotone trend across three rungs is evidence; one rung is not.
#
# (2) The online arm has no A/A. Cut height on this instrument has a ~40 % floor
#     between runs identical by construction, so the headline +1508 % at 5.0x
#     needs its own repeat to bound it. Same config, second process.
set -u
export IMAGE=isaac-lab:2.3.2-rl052
cd /home/kevinmurera
R=/home/kevinmurera/projects/rrlab_mulag
FIT=/workspace/projects/rrlab_mulag/outputs/estimator/ctxonly_fit.npz
run() { ~/bin/isaaclab-run bash -lc "$1" 2>/dev/null || true; }
exec >>/home/kevinmurera/ablate.log 2>&1
echo "=== ablate armed $(date -u +%FT%TZ) ==="
while ps -eo args | grep -qE "[h]eldout.sh|[d]2rescore.sh"; do sleep 60; done
while ps -eo args | grep -qE "[e]valuate_gates.py"; do sleep 45; done
echo "GPU free $(date -u +%FT%TZ)"
C=$(ls $R/logs/rsl_rl/ppo_id_aam_fd/ctxonly17/model_*.pt | sed 's/.*model_//;s/\.pt//' | sort -n | tail -1)
CKPT=logs/rsl_rl/ppo_id_aam_fd/ctxonly17/model_${C}.pt
mkdir -p $R/outputs/ablate

score() { # tag payload beta report
  while ps -eo args | grep -qE "[e]valuate_gates.py"; do sleep 45; done
  echo "=== $1 ==="; date -u +%FT%TZ
  run "
cd /workspace/projects/rrlab_mulag
export MULAG_DIRECT_VALVE=1
export MULAG_RMA_CONTEXT=1
export MULAG_PAYLOAD_RANGE=$2,$2
export MULAG_CONTEXT_ABLATE=estimated
export MULAG_ESTIMATOR_PATH=$FIT
export MULAG_ESTIMATOR_BETA=$3
/isaac-sim/python.sh PPO_AAM_FD/scripts/evaluate_gates.py $CKPT \
  --num_envs 176 --steps 900 --meshes 2718 --seed 17 \
  --report $4 2>&1 | grep -E 'wrote|rror' | tail -2
"
  echo "SCORED_$1"
}

# (1) beta ladder at payload 3.0x, where oracle and online diverge most.
score "beta1.0_p3.0"   3.0 1.0   "outputs/ablate/beta1p0_p3.0_m2718.json"
score "beta0.005_p3.0" 3.0 0.005 "outputs/ablate/beta0p005_p3.0_m2718.json"
# (2) A/A on the headline arm: same config as the scored ctxest, second process.
score "aa_p5.0"        5.0 0.02  "outputs/ablate/aarepeat_p5.0_m2718.json"
echo "ABLATE_COMPLETE $(date -u +%FT%TZ)"
