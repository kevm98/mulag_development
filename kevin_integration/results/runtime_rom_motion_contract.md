# Runtime ROM Motion Contract

Status: runtime logging implemented; final classification requires user-run Isaac telemetry.

## Static Training Reference

Causal ROM training 50 ms DeltaExtension target distribution:

| Cylinder | Median Abs (m) | P75 Abs (m) | P95 Abs (m) | Max Abs (m) |
| --- | ---: | ---: | ---: | ---: |
| Drehzapfen_joint | 0.006164 | 0.008360 | 0.010595 | 0.013583 |
| Ausleger_I_joint | 0.007772 | 0.009036 | 0.010761 | 0.012972 |
| Ausleger_II_joint | 0.025494 | 0.029049 | 0.032221 | 0.039572 |
| Messerkopf_Schwenk_joint | 0.014305 | 0.021852 | 0.030166 | 0.038033 |

The accepted final ROM equation is:

`DeltaExtension = ExtensionVelocity * 0.05 + learned_linear_correction`

The learned correction is bounded by `correction_scale = [0.010233, 0.007867, 0.040620, 0.037131]` m.

The deployed correction feature set is:

`dP, Extension, ExtensionVelocity, JointPos, JointVel`

It does not include ValveCmd coefficients. A pure-Python check produced identical ROM output for all-valve `-1` and all-valve `+1` at the same state.

## Static Bottleneck Check

If the mapped cylinder velocity starts near zero, the kinematic term is near zero. Meaningful motion must then come from the learned correction. Runtime telemetry now separates:

- `final_rom_delta_extension_kinematic_50ms`;
- `final_rom_delta_extension_correction_50ms`;
- `final_rom_delta_extension_50ms`;
- `final_rom_delta_isaac_50ms`;
- `final_rom_delta_isaac_10ms`.

This will show whether the correction initiates motion from rest or cancels the kinematic term.

## Implemented Runtime Evidence

The play script prints ROM raw correction features, normalized min/max, kinematic term, correction term, final 50 ms delta, mapped 50 ms Isaac delta, applied 10 ms Isaac delta, target accumulation, and measured cylinder movement.

The env asserts the multirate sum contract before clamps:

`applied_delta_10ms * 5 == predicted_delta_50ms`

## Classification Rule

Per cylinder:

- `TOO_SMALL`: runtime 50 ms abs delta is far below training central percentiles and target accumulation remains tiny.
- `TRAINING_COMPATIBLE`: runtime delta percentiles overlap training central support and accumulate to millimetres of stroke.
- `TOO_LARGE`: runtime delta persistently exceeds training support or clamps.
- `CLIPPED`: ROM or target clamp fraction is significant.
- `WRONG_SIGN`: mapped target motion consistently opposes intended reference/cylinder response.

## Current Static Classification

All cylinders are `REQUIRES_RUNTIME`.

Static inspection shows the training target is large enough for visible motion. If runtime ROM output is micrometre-scale, the bottleneck is live inputs, normalization, or weak learned correction from rest, not the target dataset magnitude.

Current static root-cause classification: `BLOCKED_ROM_OUTPUT_TOO_SMALL` unless runtime evidence shows significant motion from state/velocity propagation. The selected ROM may be an accurate transition predictor but is not a strong command generator from rest in this integration because ValveCmd is not coupled into the deployed correction and measured initial ExtensionVelocity is near zero.
