# Stage 23 — Freeze the privileged representation

Status: **complete** on 2026-08-09. Stage 22 passed, the selected dz2 model
is now frozen, checksummed and provenance-recorded, and the frozen copy
reproduces the Stage 22 rollout metrics exactly.

## What was frozen

The selected Stage 22 model (`dz2_nodp_holdq`, the pipeline's source of
truth) was copied, byte for byte, to a self-contained frozen directory:

- model weights: `encoder.pt`, `sd.pt`, `belief.pt`;
- training/normalization report: `metrics.json` (train-only factor, state,
  command, joint and delta normalization statistics);
- explicit protocol contract: `protocol.json` (latent dim 2, encoder latent
  source, dP mask `[8,9,10,11]`, `state_keep` 0-7, `use_q`, belief GRU
  `belief_dim=16`, `history_lag=6`, `rollout_q_mode=hold-start`,
  `unroll=10`, loss weights, ramp, seed, factor keys);
- Stage 22 gate evidence: `evidence/rollout_gate.json`,
  `evidence/gate_verdict.json`, `evidence/true_vs_none_h20.json` and the
  protocol-matched no-latent control rollout;
- SHA-256 manifest: `checksums.sha256` covering every frozen file including
  the provenance record;
- provenance: `metadata/provenance.json` (source path, source hashes,
  dataset root and its provenance, git, environment, command,
  `do_not_finetune=true`).

The frozen tree is chmod'ed read-only and must not be fine-tuned while the
first AAM is trained.

## Location

- Frozen model: `../sd_model/stage23_frozen/dz2_nodp_holdq/`
- Source model: `../outputs/sd_z_stage22_v6_wide/dz2_nodp_holdq/`
- Dataset: `datasets/master_v3_stage22_wide_frozen/`

## Gate: frozen copy reproduces Stage 22 metrics

`tools/verify_frozen_sd_model.py` (run on 2026-08-09):

- checksums: **OK** (every frozen file, recomputed SHA-256 matches);
- re-ran the full Stage 22 gate rollout on the frozen copy (modes
  true/zero/shuffled/random/mean/nominal, horizons 1/5/10/20 ticks,
  `hold-start` q) against the same frozen dataset;
- re-ran the no-latent comparison and the strict item-55 checks;
- result: `strict_pass=True`, max absolute difference vs stored evidence
  **0.0** on aggregate rollout RMSE, per-split means, paired diffs +
  bootstrap CIs and the zero baseline. Verdict:
  `results/stage23_frozen_verify/dz2_nodp_holdq/verdict.json`.

The frozen representation is therefore safe to consume as the stopped-
gradient target for the AAM dataset (Stage 24).

## Reproduce

```text
python tools/freeze_sd_model.py --readonly --force
python tools/verify_frozen_sd_model.py \
    --model-dir ../sd_model/stage23_frozen/dz2_nodp_holdq \
    --root datasets/master_v3_stage22_wide_frozen \
    --out results/stage23_frozen_verify/dz2_nodp_holdq/verdict.json
```

Next: Stage 24 builds causal observable histories from the frozen 118-run
dataset, each window labeled with the stopped-gradient `z = mu(e)` from the
frozen encoder.
