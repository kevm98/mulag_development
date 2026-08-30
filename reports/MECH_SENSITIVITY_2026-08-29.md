# Which plant axes actually move the response

First execution of `AAM_module/tools/mech_sensitivity_gate.py`. It had never been
run -- no result existed in `reports/`, `experiments/` or `outputs/`.

Method: one fixed cylinder position-target trajectory, 4 envs, 300 steps, run
under baseline and three perturbations; peak joint and end-effector deviation
from the baseline response. Cutter body `Messerkopf`, baseline mass 218.4 kg.

| axis | joint dev | rel | EE dev | rel |
|---|---|---|---|---|
| cylinder joint damping x3 | 0.0037 m | **15.0%** | 0.0472 m | 11.3% |
| cutter mass x1.5 | 0.0026 m | **10.6%** | 0.0470 m | 11.2% |
| cutter COM z -0.10 m | 0.0003 m | 1.1% | 0.0124 m | 3.0% |

## Reading

`DIM_ORDER = (flow, compliance, seal, pressure, mass_scale, com_z)`.

- **flow / compliance / seal / pressure** -- hydraulic, and **inert on the FD
  stack**: `CylinderFD.step()` takes no factor argument (E5 documented this; it
  was independently rediscovered on 2026-08-29).
- **mass_scale** -- acts, 11.2% EE at the manifest's own maximum (1.5). This is
  the strongest axis the privileged context actually observes.
- **com_z** -- acts weakly, 3.0% EE at the manifest's minimum (-0.10). Real but
  close to the level where it will not carry an ablation on its own.
- **cylinder joint damping** -- acts as strongly as mass (15.0% joint), and is
  **not a column of the factor vector**, so nothing currently randomizes it and
  no context observes it.

So the privileged 6-D context is **not** entirely inert, contrary to how the
inert-hydraulics finding was first generalized: 2 of its 6 columns act through
PhysX. The manifest ranges are mass_scale 0.8-1.5 and com_z -0.10..+0.05.

## Consequence for the adaptation experiment

A plant can be made genuinely non-nominal through mass_scale (strong), com_z
(weak) and, if a column is added, joint damping (strong). The four hydraulic
columns cannot do it on this stack.

Note the gate measures an OPEN-LOOP response deviation, not closed-loop task
error. A fixed policy may absorb an 11% response deviation; the paired
dose-response on lateral / cut-height error is the number that sets the G1
headroom, and it must be measured with the block-44 instrument
([INSTRUMENT_ALIASING_2026-08-29.md](INSTRUMENT_ALIASING_2026-08-29.md)) --
the earlier "+24% to +59% at payload 2-5x" figure came from the sequential
instrument and is withdrawn.
