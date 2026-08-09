# ROM-B e_flow practical identifiability audit

Date: 2026-07-25

## Decision question

Does `e_flow in [-0.20, 0.20]` produce enough observable change in the fitted
command ROM to support a robustness or online-adaptation claim?

## Reproduction audit

The standard recurrent factor audit was run with 1,024 environments, 240 control
steps, and seed 101. At the factor extremes, mean final extension divergence was:

| Factor | Extreme | Mean divergence |
|---|---:|---:|
| `e_comp` | -0.125 | 7.03 mm |
| `e_comp` | +0.020 | 0.97 mm |
| `e_flow` | -0.200 | 0.052 mm |
| `e_flow` | +0.200 | 0.052 mm |
| `d_seal_1` | 1.0 | 16.59 mm |
| `d_seal_2` | 1.0 | 14.95 mm |
| `d_seal_3` | 1.0 | 21.89 mm |
| `d_seal_4` | 1.0 | 17.40 mm |

Artifact: `factor_sensitivity_2026-07-25_00-31-54.json`.

## Multi-excitation audit

The conclusion was challenged with five command families over six seconds:
slow and fast sinusoids, mixed per-cylinder sinusoids, alternating steps, and a
frequency chirp. Commands used matched desired-extension velocity feedforward.
Every pattern was run from zero pressure and from the checkpoint's training-mean
pressure, with identical initial extensions and commands across factors.

Across the ten pattern/initialization conditions:

| Quantity | `e_flow = +/-0.20` result |
|---|---:|
| Extension trace RMS | 0.010-0.060 mm |
| Endpoint extension divergence | 0.017-0.086 mm |
| Pressure trace RMS | 0.019-0.081 MPa |
| Maximum flow/compliance extension-RMS ratio | 1.46% |
| Median flow/compliance extension-RMS ratio | 0.73% |

For comparison, compliance produced 1.39-8.26 mm extension trace RMS and maximum
seal degradation produced 6.28-39.00 mm. The checkpoint's pressure target
standard deviations are approximately 18-27 MPa, far above the largest
`e_flow` pressure response.

The trained flow gains explain the result. Three cylinders are effectively at
the model's `1e-9` lower clamp; the fourth is `5.22e-9`. Learned pressure,
velocity, and position residual paths can reproduce command response while
bypassing the explicit flow channel.

## Interpretation

This is a forward-sensitivity and practical-influence audit, not a proof of
structural non-identifiability. It does establish that `e_flow` is practically
unsupported in the deployed fitted ROM at the declared range and under varied
excitation. Randomizing it cannot materially teach PPO or support a claim of
robustness to valve-flow gain.

Do not make the factor visible by scaling actuation, velocity, extension, or ROM
outputs. A defensible recovery would require re-identifying the ROM with a
constrained physical flow path and demonstrating separation from compliance,
leakage, and learned residuals. Without that re-identification, set the retained
compatibility slot to zero and remove `e_flow` from the declared uncertainty set.
