# Runtime Valve Command Contract

Status: runtime logging implemented; final classification requires user-run Isaac telemetry.

## Static Training Reference

Causal ROM training ValveCmd distribution at the last history sample:

| Channel | Mean | Std | Median Abs | P75 Abs | Max Abs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Drehzapfen_joint | 0.259 | 0.710 | 0.864 | 0.950 | 0.950 |
| Ausleger_I_joint | -0.118 | 0.755 | 0.907 | 0.950 | 0.950 |
| Ausleger_II_joint | -0.139 | 0.359 | 0.288 | 0.479 | 0.800 |
| Messerkopf_Schwenk_joint | -0.055 | 0.537 | 0.473 | 0.727 | 0.800 |

Runtime ValveCmd should not be nearly constant. Every active channel should show signed variation over trajectory reversals.

## Implemented Runtime Evidence

At one-second intervals, `play_mulag_adaptive_rl.py` prints:

- raw `ValveCmd[4]`;
- ValveCmd mean/std/min/max;
- saturation fraction at `abs(command) >= 0.99`;
- ID normalized min/max and z_arm norm;
- desired qddot feedback, feedforward, raw, action, and clipped command.

At shutdown, it prints max/RMS for ValveCmd and upstream qddot/action stages.

## Classification Rule

Per channel:

- `CONSTANT_OR_DEAD`: std near zero or no signed change during reversals.
- `TOO_SMALL`: runtime median absolute command is below about 10% of the training median absolute value and ROM output is also too small.
- `TRAINING_COMPATIBLE`: runtime percentiles overlap the central training support without constant output or sustained saturation.
- `TOO_LARGE`: runtime command is outside training support for sustained periods.
- `SATURATED`: saturation fraction is persistently high.

## Current Static Classification

All channels are `REQUIRES_RUNTIME`.

Static inspection proves the ID feature order and qddot scaler slots, but not live ValveCmd sufficiency. If runtime ValveCmd is too small, the likely root causes to inspect first are qddot command magnitude, ID normalization, z_arm warm-up, dP source, and target/reference scale.

Important static caveat: even training-compatible runtime ValveCmd will not be enough by itself with the deployed final causal linear ROM, because that ROM's correction does not use ValveCmd as an input coefficient. ValveCmd sufficiency must therefore be interpreted together with `runtime_rom_motion_contract.md`.
