# ROM Runtime Feature Contract

The active Kevin hydraulic control policy uses the valve-conditioned causal ROM
export under `exported_valve_conditioned_causal_rom/`.

Runtime input history shape:

```text
[num_envs, history_length=10, input_dim=26]
```

Feature order:

```text
ValveCmd_1..4
dP_1..4
Extension_1..4
ExtensionVelocity_1..4
JointPos_1..5
JointVel_1..5
```

Output shape:

```text
[num_envs, 4]
```

Output meaning:

```text
DeltaExtension_simscape_1..4 over 0.05 s
```

IsaacLab control runs at 0.01 s. `multirate_rom_wrapper.py` distributes each
0.05 s prediction over five 0.01 s control steps. `cylinder_coordinate_adapter.py`
then maps Simscape extension deltas to Isaac prismatic joint deltas using
`cylinder_coordinate_mapping.json`.

The active runtime does not depend on the removed training datasets,
checkpoints, audit scripts, or generated result files.
