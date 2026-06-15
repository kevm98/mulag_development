# Kevin Hydraulic ROM Runtime

This directory contains only the ROM files used by the current Kevin Mulag
fixed final-pose workflow and focused offline tests.

## Active Runtime

- `valve_conditioned_causal_rom.py`
- `multirate_rom_wrapper.py`
- `cylinder_coordinate_adapter.py`
- `cylinder_coordinate_mapping.json`
- `rom_data_contract.py`
- `exported_valve_conditioned_causal_rom/`

The active export package contains:

```text
best_valve_conditioned_causal_rom.pth
best_valve_conditioned_causal_rom_scaler.pth
best_valve_conditioned_causal_rom_metadata.json
best_valve_conditioned_causal_rom_validation.md
```

The IsaacLab task feeds `ValveCmd[4]`, training-mean pressure state, cylinder
extension state, and joint state into the valve-conditioned causal ROM. The
multirate wrapper distributes each 0.05 s prediction across five 0.01 s control
steps before the coordinate adapter converts Simscape extension deltas to Isaac
prismatic joint target deltas.

## Focused Tests

```bash
python kevin_integration/rom_cylinders/test_valve_conditioned_runtime.py
python kevin_integration/rom_cylinders/test_multirate_rom_wrapper.py
python kevin_integration/rom_cylinders/test_cylinder_coordinate_adapter.py
```

The superseded linear ROM, model-fitting scripts, old datasets, smoke exports,
audit scripts, generated plots, and result dumps are not part of this runtime
tree. Do not retrain or replace the active ROM unless a separate
model-development task explicitly requests it.
