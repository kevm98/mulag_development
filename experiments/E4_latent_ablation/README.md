# E4 — Latent ablation

**Defends:** *Online Adaptation* · **Covers:** PPO-AAM-FD, one checkpoint
**Status:** model level complete · thesis Appendix E only. The arms ran on the
retired architecture, so they are a recorded measurement rather than a result
about the deployed stack.

## Model-level result (settled, n=5 seeds)

One trained model scored three ways — same weights, same data, only the latent
slot changes:

| latent | mean val error | vs online | seeds |
|---|---|---|---|
| online z_hat(t) | 0.018842 | — | — |
| per-episode constant, correct plant | 0.021402 | +13.6% | 5/5 |
| per-episode constant, wrong plant | 0.021492 | +14.1% | 5/5 |

**Time-varying component 13.6%; static identity component 0.43%** — thirty-two
times smaller, though consistently signed in all five seeds, so it is real but
nearly negligible. Say "0.43%, ~32x smaller", **not** "zero".

Companion result: the estimated latent beats a matched no-latent baseline in
**5/5 seeds, mean -24.7%**, bootstrap 95% CI [-34.2%, -18.2%]. Seed 20260823's
baseline is a high outlier — quote the **median -22.3%**.

## Interpretation

The module re-estimates from a fixed 1.0 s window every 50 ms and carries no
state between ticks: **recursive estimation with a forgetting factor**, the
standard construction for time-varying parameters. Claim that it **tracks a
time-varying plant**; do **not** claim it **identifies which plant it is on** —
at 0.43% that is measurably nearly false.

## Closed loop (outstanding)

Repeat the substitution on the deployed stack, adding the **privileged true
latent** as an oracle ceiling — without it the measured gain has no scale.
Use `headroom_priced_v1/model_299` and `FACTOR_DOMAIN=combined_test`.
