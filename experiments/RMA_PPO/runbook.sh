#!/usr/bin/env bash
# GPU-only execution runbook. Preparation in this repository never invokes it.
set -euo pipefail

ROOT=${ROOT:-/home/rrlab/rrlab_mulag}
PY=${PY:-/home/rrlab/miniconda3/envs/env_isaaclab/bin/python}
SOURCE=${SOURCE:-$ROOT/logs/rsl_rl/ppo4_1_sole/speed_range_2_7_pathfix_v2_idfd/model_promoted_2_7kmh.pt}
PHASE1=${PHASE1:-$ROOT/experiments/RMA_PPO/phase1_privileged_init.pt}
PHASE2=${PHASE2:-$ROOT/experiments/RMA_PPO/on_policy_history}
PHASE3=${PHASE3:-$ROOT/experiments/RMA_PPO/phase3_estimated_finetuned}

# 0. Environment and artifact checks.
cd "$ROOT"
test -f "$SOURCE"
test -f "$PHASE1"
$PY -m pytest PPO_AAM_FD/tests/test_rma_adaptation_cpu.py -q

# 1. Nominal identity smoke in the real environment (must remain zero-change).
$PY PPO_AAM_FD/scripts/evaluate_gates.py "$PHASE1" --context_mode BASELINE --headless

# 2. Phase 1 privileged PPO: conservative staged randomization, one stage at a time.
$PY PPO_AAM_FD/scripts/train.py --resume "$PHASE1" --rma_phase 1 --rma_context_mode PRIVILEGED --rma_stage nominal --headless
$PY PPO_AAM_FD/scripts/train.py --resume "$PHASE1" --rma_phase 1 --rma_context_mode PRIVILEGED --rma_stage authority --headless
$PY PPO_AAM_FD/scripts/train.py --resume "$PHASE1" --rma_phase 1 --rma_context_mode PRIVILEGED --rma_stage lag --headless
$PY PPO_AAM_FD/scripts/train.py --resume "$PHASE1" --rma_phase 1 --rma_context_mode PRIVILEGED --rma_stage delay --headless
$PY PPO_AAM_FD/scripts/train.py --resume "$PHASE1" --rma_phase 1 --rma_context_mode PRIVILEGED --rma_stage mass --headless
$PY PPO_AAM_FD/scripts/train.py --resume "$PHASE1" --rma_phase 1 --rma_context_mode PRIVILEGED --rma_stage combined --headless

# 3. Mandatory observability gate before any AAM work.
$PY PPO_AAM_FD/scripts/evaluate_gates.py "$PHASE1" --context_mode PRIVILEGED --headless
$PY PPO_AAM_FD/scripts/evaluate_gates.py "$PHASE1" --context_mode MEAN_CONTEXT --headless
$PY PPO_AAM_FD/scripts/evaluate_gates.py "$PHASE1" --context_mode SHUFFLED_CONTEXT --headless

# 4. Record causal PPO on-policy histories only.
$PY PPO_AAM_FD/scripts/record_rma_history.py --checkpoint "$PHASE1" --output "$PHASE2" --headless

# 5. Train/evaluate the AAM on those histories, then freeze its deployment interface.
$PY PPO_AAM_FD/scripts/train_rma_aam.py --history "$PHASE2" --output "$PHASE2/aam.pt" --device cuda
$PY PPO_AAM_FD/scripts/evaluate_gates.py "$PHASE1" --context_mode AAM_ESTIMATED --aam "$PHASE2/aam.pt" --headless
$PY PPO_AAM_FD/scripts/evaluate_gates.py "$PHASE1" --context_mode AAM_FROZEN --aam "$PHASE2/aam.pt" --headless

# 6. Phase 3 PPO fine-tune with estimated context; PPO remains the sole controller.
$PY PPO_AAM_FD/scripts/train.py --resume "$PHASE1" --rma_phase 3 --rma_context_mode AAM_ESTIMATED_FINETUNED --aam "$PHASE2/aam.pt" --headless --log_dir "$PHASE3"

# 7. Mid-episode switch and recovery matrix, all declared runtime modes.
$PY PPO_AAM_FD/scripts/evaluate_rma_recovery.py --checkpoint "$PHASE3" --modes BASELINE PRIVILEGED MEAN_CONTEXT SHUFFLED_CONTEXT AAM_ESTIMATED AAM_FROZEN AAM_ESTIMATED_FINETUNED --headless

# 8. Summarize delta_adapt, recovery_fraction, stress thresholds and plots.
$PY PPO_AAM_FD/scripts/analyze_rma.py --input "$PHASE3" --output "$PHASE3/summary.json"

# 9. Promote only after privileged-vs-mean/shuffled and leakage gates pass.
$PY PPO_AAM_FD/scripts/evaluate_gates.py "$PHASE3" --context_mode AAM_ESTIMATED_FINETUNED --headless

# 10. Archive immutable source, phase-1 init, AAM, config and summary together.
sha256sum "$SOURCE" "$PHASE1" "$PHASE2/aam.pt" "$PHASE3/summary.json"
