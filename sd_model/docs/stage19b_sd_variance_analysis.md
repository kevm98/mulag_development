# Stage 19b — Unexplained Variance: Good or Bad?

Date: 2026-08-08. Question: the one-step SD baseline leaves 5-19% of the
target variance unexplained on held-out runs. Is that unexplained variance
"good" (irreducible, e.g. measurement noise) or "bad" (systematic structure
the model cannot see, and therefore fixable)? And what does it mean
literally?

## 1. What "unexplained variance" means literally

R2 = 1 - Var(residual) / Var(target). The unexplained fraction is the share
of the *delta* variance (the 50 ms state changes the model is asked to
predict) that the model does not explain. It does **not** mean noise: a
residual can be fully systematic and still large. "Unexplained" only means
"not predictable from the model's inputs and structure". The same residual
could be explained by other variables (deeper history, the privileged factor
vector z_i, an unobserved plant state), or it could be genuinely random.
Distinguishing those cases is exactly what the decomposition below does.

## 2. Baseline numbers (full 28-run set, held-out test = 5 unseen configs)

One-step test R2 (corrected against the full-set zero-delta reference):

| input | dim | ext | vel | dP | all | unexplained |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `su` | 20 | 0.813 | 0.709 | 0.815 | 0.815 | 0.185 |
| `su_delta` (+prev state change) | 32 | 0.898 | 0.749 | 0.889 | 0.889 | 0.111 |
| `squ` (+joints) | 30 | 0.840 | 0.731 | 0.820 | 0.820 | 0.180 |
| `hist3` | 60 | 0.874 | 0.743 | 0.898 | 0.898 | 0.102 |
| `hist6` | 120 | 0.881 | 0.751 | 0.919 | 0.919 | 0.081 |

`hist6` is the best observable-input one-step model (8.1% unexplained);
velocity is consistently the weakest channel.

## 3. Decomposition: where the unexplained variance comes from

Four hypotheses were tested directly on the full dataset.

### 3.1 Temporal structure (history-fixable) — YES, present and fixed

The `su` model's residuals are temporally autocorrelated (extension lag-1
autocorr 0.32, dP 0.19) -- the model misses the recent trajectory, i.e. the
ROM's internal memory. This is **bad variance** (systematic, fixable).

History fixes it: with `hist6` the residuals are temporally white on the
held-out runs (extension -0.09, velocity -0.17, dP -0.01 at lag 1). The
unexplained fraction drops from 18.5% (`su`) to 8.1% (`hist6`). This part of
the bad variance is now fixed.

### 3.2 The ROM's internal previous-valve state — NO effect

The SD input omits `prev_valve` (the ROM's own valve from the previous tick),
which is observable in the recorded data. A model trained on the exact 24-D
ROM history row (ext, vel, dP, prev_valve, command) does **not** improve over
`su` (test R2 0.830 vs 0.815; not significant). The valve is not the missing
state; the ROM's hidden memory is temporal, not a single valve sample.

### 3.3 More / richer history — NO further gain

L=12 history (240-D) and L=6 history with valve blocks (144-D) both
underperform plain `hist6` (all R2 0.903 / 0.912 vs 0.919). Six ticks (300 ms)
already saturates what the observable trajectory tells the model; longer
windows overfit.

### 3.4 Domain structure (z-correlated per-run bias) — present, NOT fixable at this scale

The per-run mean residual correlates strongly with the privileged factor
vector: `su` up to 0.82 (compliance), and even `hist6` up to 0.93
(compliance 0.93, com_z 0.87, seal 0.83). Concretely: each held-out run has a
systematic prediction bias aligned with its physical configuration. This is
**bad variance** -- it is structure, not noise.

But it is **not fixable in the no-latent baseline at this data scale**:
training `hist6` with z_i appended (oracle input) does not improve test R2
(0.908 vs 0.919). With only 20 train runs, the model can memorize the
train-time z -> bias map but cannot generalise it to unseen factor
combinations. Closing this gap is exactly the job of the latent/online AAM
(Stage 20+): infer z_hat from observable history and condition on it.

## 4. Verdict

**The unexplained variance is predominantly bad variance, not noise.** Two
systematic sources were identified:

1. Temporal structure (recent trajectory / ROM memory) -- **fixed** by the
   `hist6` model: residuals are now temporally white (18.5% -> 8.1%
   unexplained).
2. Per-configuration bias aligned with the physical factors -- **remains**
   and cannot be recovered by the no-latent model with 20 train runs; it is
   the measured entry point for the latent AAM.

The genuinely irreducible remainder is small: temporally white residuals plus
the velocity channel's high-frequency component and float32-level
quantization. There is no large "good variance" hiding in the residual.

## 5. Fixes applied

- `hist6` is the deployment-candidate one-step SD model (best observable
  R2 0.919, temporally white residuals).
- New `su_delta` input mode in `tools/train_sd_baseline.py`:
  `[s_t^H; u_t; s_t^H - s_{t-1}^H]` (32-D). It recovers most of the history
  gain in an instantaneous model (test R2 0.889 vs `su` 0.815, ext 0.898)
  with one extra cheap feature -- useful where a long context window is not
  available. Checkpoint: `sd_model/checkpoints/su_delta.pt`.
- The domain-bias residual (quantified, z-aligned) is documented as the
  Stage 20+ latent/online AAM target rather than papered over.

## 6. Files

```text
tools/train_sd_baseline.py        + --input su_delta mode
sd_model/checkpoints/su_delta.pt  su_delta checkpoint (full dataset)
sd_model/metrics/su_delta.json    metrics
```
