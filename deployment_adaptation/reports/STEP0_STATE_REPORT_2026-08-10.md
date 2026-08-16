# Step 0 state report — 2026-08-10

## Decision

The prior AAM training is finished. The latest Stage-29 retry passes Gate 2 on
the validation split, so the audit is allowed to continue to Phase 1. The
earlier interim stop was based on an older adapter artifact and is superseded
by `outputs/stage29_latent_adapter_retry/evaluation/summary.json`.

The long PPO run also finished. It is a valid completed training artifact, but
it is not by itself a completed four-arm thesis comparison: the matched factor
domains, integrator arm, paired episodes, and final frozen comparison still
have to be run.

## Completed artifacts

| Artifact | Result |
|---|---|
| Pressure-inclusive causal AAM | Best epoch 26; validation RMSE 1.183013; test RMSE 1.252442; temporal-consistency weight 1.0 |
| AAM checkpoint | `outputs/aam_stage30_pressure_temporal/L5/aam.pt` — SHA256 `f9d8ee4eb398f24130e17e13fc27999dd2fb89e0469d0189e272e8e480eb2602` |
| Latest Stage-29 true-latent adapter | `outputs/stage29_latent_adapter_retry/true_adapter.pt` |
| Latest Stage-29 AAM-latent adapter | `outputs/stage29_latent_adapter_retry/zhat_adapter.pt` |
| Selected pressure AAM adapter | `outputs/stage30_pressure_adapter_aligned_gpu/zhat_adapter.pt` — SHA256 `bbbeca05e26fb2cac57a9659d53c41060868604dbaaf8a6e97386e4041e25b18` |
| Final PPO checkpoint | `logs/rsl_rl/ppo_id_aam_fd/train_final/model_599.pt` — SHA256 `f878d2168e4a105d1280b41e162785c909fe63fdd0001e1a2502aea37607bee9` |
| Final PPO manifest | `logs/rsl_rl/ppo_id_aam_fd/train_final/run_manifest.json` — SHA256 `af61a6ad57f62f2db31642aec4b5b5cad0f97bd8b551c60ded6599112135d92c` |

The PPO checkpoint reports a 68-wide observation, four actions, actuator
`id_aam_fd`, seed 17, 256 environments, six curriculum stages, fixed learning
rate 0.0003, entropy coefficient 0.001, and zero hard-failure counters at the
final printed iteration 599/600.

## Gate 2 evidence from the latest retry

The validation comparison uses 20 runs. Lower is better except for divergence
time, where higher is better.

| Metric | Frozen ID | True-latent adapter | Change |
|---|---:|---:|---:|
| Teacher-forced next extension RMSE (m) | 0.002715653 | 0.002322164 | -14.5% |
| Teacher-forced next velocity RMSE (m/s) | 0.103642446 | 0.100809401 | -2.7% |
| Teacher-forced valve RMSE | 0.308100386 | 0.307197225 | -0.3% |
| Closed-loop cylinder tracking RMSE (m) | 20.770320 | 19.587751 | -5.7% |
| Closed-loop tool-path proxy RMSE (m) | 56.986442 | 52.914567 | -7.1% |
| Pre-clip saturation fraction | 0.889885 | 0.598700 | -32.7% |
| First divergence time (s) | 30.8475 | 31.4650 | +2.0% |

Therefore the latest true-latent retry improves both required parts of Gate 2:
teacher-forced prediction and autoregressive rollout. On the 16 held-out runs,
true-latent tracking is slightly worse than frozen ID (18.989414 m versus
18.651043 m), so the adapter must not be called universally better until the
full paired-domain evaluation is complete.

## Stage status audit

| Stage | Status | Evidence / limitation |
|---|---|---|
| 23 | Complete | Frozen causal representation and reproduced verdict artifacts exist. |
| 24 | Complete | Dataset and factor schema exist; pressure-inclusive retraining is an additional later artifact. |
| 25 | Complete | Causal L5 and past-only rolling-20 rescoring artifacts exist. |
| 26 | Complete/pass | Formal causal-temporal chain artifacts exist. The pressure-inclusive Stage-30 retrain did not overwrite this verdict. |
| 27 | Pass | `AAM_module/results/stage27_chain_verify_temporal/verdict.json` reports `pass: true` over 36 runs. |
| 28 | Complete | Frozen ID transfer shows a meaningful held-out gap; report and per-run summary exist. |
| 29 | Gate passed on latest retry | True-latent validation teacher-forced and rollout improvements are present. Held-out retention still requires the matched final comparison. |
| 30 | Implementation complete, formal experiment incomplete | Causal ring buffer, pressure-inclusive AAM, 20 Hz update, warm-up, bound, reset, latency, and fallback preflight passed. The attachment's formal bit-for-bit/truncated-buffer Gate 3 record is not yet present. |
| 31 | Not complete | No frozen four-arm manifest with paired episodes across all five domains and at least six held-out meshes has been run. |

## Protocol discrepancy to resolve explicitly

The current deployment implementation uses the measured four-channel cylinder
pressure difference in the 30-D AAM row. This follows the later pressure
inclusive request and its retraining artifacts. The attached authoritative audit
brief says `dP` must not enter a deployable policy/AAM. That is a protocol
conflict, not a result to hide: the pressure-inclusive path is the current
implementation, while the no-pressure rule remains the audit brief's fixed
fact. The final thesis claim must choose and freeze one protocol before Stage 31;
no silent feature change is allowed.

## Phase-1 gate state

No new factor plumbing or new PPO training was started during Step 0. The
existing 16 reserved combined-test configurations were only read, not changed.
Phase 1 is now authorized because Gate 2 passed on the latest retry; its own
factor-sweep Gate 1 is still pending and remains a hard stop if it fails.

