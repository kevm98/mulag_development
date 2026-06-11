# Kevin Hydraulic ROM Runtime

This directory is intentionally trimmed to the ROM files used by the current
Kevin Mulag control policy plus a few compatibility/runtime contract helpers.

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

## Compatibility Helpers

- `final_causal_linear_rom.py`
- `final_rom_runtime_contract.py`

These remain because current Kevin diagnostic scripts import them. They are not
the command-authoritative ROM used by the active policy.

## Focused Tests

```bash
python kevin_integration/rom_cylinders/test_valve_conditioned_runtime.py
python kevin_integration/rom_cylinders/test_multirate_rom_wrapper.py
python kevin_integration/rom_cylinders/test_cylinder_coordinate_adapter.py
```

Training scripts, old datasets, smoke exports, checkpoints, audit scripts,
generated plots, and result dumps were removed from this tree. Do not retrain or
replace the active ROM as part of control-policy integration work unless a
separate model-development task explicitly requests it.
