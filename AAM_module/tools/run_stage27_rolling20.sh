#!/usr/bin/env bash
set -euo pipefail

# Inference-only Stage 27 follow-up using the Stage 25 rolling-20 smoother.
repo_root=$HOME/rrlab_mulag
aam_root="$repo_root/AAM_module"
python_bin=$HOME/miniconda3/envs/env_isaaclab/bin/python
frozen="$aam_root/datasets/master_v3_stage22_wide_frozen"
frozen_model="$repo_root/learned_models/sd_model/stage23_frozen/dz2_nodp_holdq"
out_root="$repo_root/outputs/aam_stage27_rolling20"

mkdir -p "$out_root"
"$python_bin" "$repo_root/learned_models/sd_model/evaluation/evaluate_sd_z_rollout.py" \
    --root "$frozen" --model-dir "$frozen_model" \
    --latent-mode true zhat zero shuffled mean \
    --zhat-json "$repo_root/outputs/aam_stage25_rolling20/zhat_L5_rolling20.json" \
    --horizons 1 5 10 20 --q-rollout hold-start \
    --out "$out_root/rollout_gate.json"

"$python_bin" "$repo_root/learned_models/sd_model/evaluation/compare_sd_z_rollouts.py" \
    --candidate "$out_root/rollout_gate.json" --candidate-mode zhat \
    --control "$repo_root/outputs/aam_stage26_causal/rollout_none.json" \
    --control-mode true --horizon 20 \
    --out "$out_root/zhat_vs_none_h20.json"

"$python_bin" "$aam_root/tools/validate_stage27_chain.py" \
    --gate-rollout "$out_root/rollout_gate.json" \
    --none-rollout "$repo_root/outputs/aam_stage26_causal/rollout_none.json" \
    --out "$aam_root/results/stage27_chain_verify_rolling20/verdict.json"
