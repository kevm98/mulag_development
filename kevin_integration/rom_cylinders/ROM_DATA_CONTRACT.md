# ROM Data Contract

This contract is derived from `pytorch_hydraulic_rom_dataset.mat` by
`inspect_rom_mat_dataset.py`. Do not change model, loader, or inference feature
ordering without updating this document and `rom_data_contract.py`.

## Dataset Identity

- MATLAB file: `kevin_integration/rom_cylinders/pytorch_hydraulic_rom_dataset.mat`
- Format: MATLAB v7.3 / HDF5
- Metadata version: `ROM_Dataset_V1.0`
- Description: reduced hydraulic actuator transition dataset with histories of
  valve, pressure, cylinder, and mechanical state to next cylinder-extension and
  pressure increments.

## Storage and Tensor Dimensions

The MATLAB file stores core inputs as:

`X_*_rom: [input_dim, history_length, sample]`

The MATLAB file stores targets as:

`Y_*_rom: [target_dim, sample]`

The PyTorch loader returns:

`X: [batch, history_length, input_dim]`

`Y: [batch, target_dim]`

The loader performs this dimension conversion explicitly and validates all sizes.
No dataset dimensions are transposed during inspection.

## Timing

- `HISTORY_LENGTH = 10`
- `TARGET_HORIZON = 1`
- Median sample time: `0.05 s`
- Approximate history duration: `0.5 s`
- Target timing: the input window ends at timestep `k`; the target describes the
  transition from `k` to `k+1`.

## Core Input Features

The default deployable model uses the 26-dimensional core feature vector:

1. `ValveCmd_1`
2. `ValveCmd_2`
3. `ValveCmd_3`
4. `ValveCmd_4`
5. `dP_1`
6. `dP_2`
7. `dP_3`
8. `dP_4`
9. `Extension_1`
10. `Extension_2`
11. `Extension_3`
12. `Extension_4`
13. `ExtensionVelocity_1`
14. `ExtensionVelocity_2`
15. `ExtensionVelocity_3`
16. `ExtensionVelocity_4`
17. `JointPos_1`
18. `JointPos_2`
19. `JointPos_3`
20. `JointPos_4`
21. `JointPos_5`
22. `JointVel_1`
23. `JointVel_2`
24. `JointVel_3`
25. `JointVel_4`
26. `JointVel_5`

`CORE_INPUT_DIM = 26`

## Optional Force Inputs

Force is stored separately and is not part of the core input. If enabled, append:

1. `Force_1`
2. `Force_2`
3. `Force_3`
4. `Force_4`

`AUGMENTED_INPUT_DIM = 30`

Force-augmented models must be evaluated separately. They are not automatically
preferred because online Isaac deployment would require an equivalent force/Fnet
signal, likely from SD or another observer.

## Targets

Target order is fixed:

1. `DeltaExtension_next_1`
2. `DeltaExtension_next_2`
3. `DeltaExtension_next_3`
4. `DeltaExtension_next_4`
5. `DeltaDP_next_1`
6. `DeltaDP_next_2`
7. `DeltaDP_next_3`
8. `DeltaDP_next_4`

`TARGET_DIM = 8`

Indices `0:4` are cylinder stroke increments. These are the deployable ROM
outputs that are eventually accumulated into Isaac cylinder position targets:

`cylinder_target[k+1] = cylinder_target[k] + predicted_DeltaExtension[k]`

Indices `4:8` are `DeltaDP`, an auxiliary hydraulic-state prediction used for
pressure consistency, SD comparison, observer development, or optional recurrent
state update. Predicted `DeltaDP` must not be applied directly to Isaac
mechanical states.

## Normalization

The file provides training-set normalization statistics:

- `x_mean_rom`, `x_std_rom` for core inputs
- `y_mean_rom`, `y_std_rom` for all targets
- `force_mean`, `force_std` for optional Force inputs
- `delta_extension_mean/std` and `delta_dp_mean/std` mirror the target blocks

Training and evaluation default to z-score normalization from the training split.
Reports and deployment metadata must preserve the raw physical-unit target order.

## Splits

The dataset contains:

- Train: 936 run IDs, 403416 samples
- Validation: 104 run IDs, 44824 samples
- No explicit test split

Run IDs restart within each split (`TrainRunID` ranges `1..936`, `ValRunID`
ranges `1..104`). Treat run identity as split-qualified, for example
`train:17` and `val:17`, unless a future dataset adds globally unique run IDs.

Complete-run holdout validation is available. Final claims such as "validated" or
"ready for Isaac integration" still require a true held-out test split or an
explicit decision to reserve part of the validation runs as test.

## Actuator Scope

The four hydraulic actuator-side prismatic outputs are:

- `Drehzapfen_joint`
- `Ausleger_I_joint`
- `Ausleger_II_joint`
- `Messerkopf_Schwenk_joint`

Joint 5 remains passive/coupled and is represented in mechanical state features,
not as a fifth hydraulic actuator output.
