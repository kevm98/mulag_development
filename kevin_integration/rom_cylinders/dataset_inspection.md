# ROM MATLAB Dataset Inspection

- Dataset: `/home/rrlab/rrlab_mulag/kevin_integration/rom_cylinders/pytorch_hydraulic_rom_dataset.mat`
- Loader: `h5py`
- File size: `80119913` bytes
- Top-level variables: `57`

## Top-Level Variables

- `ForceAvailable_train`: shape `[1, 403416]`, dtype `uint8`, categories: splits
- `ForceAvailable_val`: shape `[1, 44824]`, dtype `uint8`, categories: splits
- `Force_train_hist`: shape `[4, 10, 403416]`, dtype `float32`, categories: input_histories, possible_state_channels, splits
- `Force_val_hist`: shape `[4, 10, 44824]`, dtype `float32`, categories: input_histories, possible_state_channels, splits
- `ROMMetadata/all_requested_files`: shape `[1, 4]`, dtype `object`, categories: possible_state_channels
- `ROMMetadata/approximate_history_duration`: shape `[1, 1]`, dtype `float64`, categories: input_histories, targets
- `ROMMetadata/description`: shape `[163, 1]`, dtype `uint16`, categories: uncategorized
- `ROMMetadata/force_feature_names`: shape `[1, 4]`, dtype `object`, categories: feature_names, input_histories, possible_state_channels
- `ROMMetadata/force_usage`: shape `[79, 1]`, dtype `uint16`, categories: uncategorized
- `ROMMetadata/global_median_sample_time`: shape `[1, 1]`, dtype `float64`, categories: timestamps
- `ROMMetadata/history_length`: shape `[1, 1]`, dtype `float64`, categories: input_histories, targets
- `ROMMetadata/input_dimension`: shape `[1, 1]`, dtype `float64`, categories: input_histories
- `ROMMetadata/input_feature_names`: shape `[1, 26]`, dtype `object`, categories: feature_names, input_histories, possible_inputs_by_dimension
- `ROMMetadata/input_layout`: shape `[94, 1]`, dtype `uint16`, categories: input_histories, targets
- `ROMMetadata/num_total_valid_runs`: shape `[1, 1]`, dtype `float64`, categories: run_indices, splits
- `ROMMetadata/num_training_runs`: shape `[1, 1]`, dtype `float64`, categories: run_indices, splits
- `ROMMetadata/num_training_samples`: shape `[1, 1]`, dtype `float64`, categories: splits, timestamps
- `ROMMetadata/num_validation_runs`: shape `[1, 1]`, dtype `float64`, categories: run_indices, splits
- `ROMMetadata/num_validation_samples`: shape `[1, 1]`, dtype `float64`, categories: splits, timestamps
- `ROMMetadata/output_dimension`: shape `[1, 1]`, dtype `float64`, categories: targets
- `ROMMetadata/output_feature_names`: shape `[1, 8]`, dtype `object`, categories: feature_names, input_histories, possible_targets_by_dimension, targets
- `ROMMetadata/output_layout`: shape `[60, 1]`, dtype `uint16`, categories: targets
- `ROMMetadata/random_seed`: shape `[1, 1]`, dtype `float64`, categories: uncategorized
- `ROMMetadata/regime_definition`: shape `[1, 3]`, dtype `object`, categories: uncategorized
- `ROMMetadata/train_fraction`: shape `[1, 1]`, dtype `float64`, categories: splits
- `ROMMetadata/training_source_files`: shape `[1, 936]`, dtype `object`, categories: splits
- `ROMMetadata/transition_definition`: shape `[83, 1]`, dtype `uint16`, categories: uncategorized
- `ROMMetadata/validation_source_files`: shape `[1, 104]`, dtype `object`, categories: splits
- `ROMMetadata/version`: shape `[16, 1]`, dtype `uint16`, categories: uncategorized
- `T`: shape `[1, 1]`, dtype `float64`, categories: uncategorized
- `TrainClass`: shape `[1, 403416]`, dtype `uint8`, categories: splits
- `TrainRunID`: shape `[1, 403416]`, dtype `uint32`, categories: run_indices, splits
- `TrainSampleTime`: shape `[1, 403416]`, dtype `float32`, categories: splits, timestamps
- `TrainTimeIndex`: shape `[1, 403416]`, dtype `uint32`, categories: input_histories, splits, timestamps
- `ValClass`: shape `[1, 44824]`, dtype `uint8`, categories: splits
- `ValRunID`: shape `[1, 44824]`, dtype `uint32`, categories: run_indices, splits
- `ValSampleTime`: shape `[1, 44824]`, dtype `float32`, categories: splits, timestamps
- `ValTimeIndex`: shape `[1, 44824]`, dtype `uint32`, categories: input_histories, splits, timestamps
- `X_train_rom`: shape `[26, 10, 403416]`, dtype `float32`, categories: input_histories, possible_inputs_by_dimension, splits
- `X_val_rom`: shape `[26, 10, 44824]`, dtype `float32`, categories: input_histories, possible_inputs_by_dimension, splits
- `Y_train_rom`: shape `[8, 403416]`, dtype `float32`, categories: possible_targets_by_dimension, splits, targets
- `Y_val_rom`: shape `[8, 44824]`, dtype `float32`, categories: possible_targets_by_dimension, splits, targets
- `delta_dp_mean`: shape `[4, 1]`, dtype `float32`, categories: normalization, possible_state_channels, targets
- `delta_dp_std`: shape `[4, 1]`, dtype `float32`, categories: normalization, possible_state_channels, targets
- `delta_extension_mean`: shape `[4, 1]`, dtype `float32`, categories: input_histories, normalization, possible_state_channels, targets
- `delta_extension_std`: shape `[4, 1]`, dtype `float32`, categories: input_histories, normalization, possible_state_channels, targets
- `force_feature_names`: shape `[1, 4]`, dtype `object`, categories: feature_names, input_histories, possible_state_channels
- `force_mean`: shape `[4, 1]`, dtype `float32`, categories: normalization, possible_state_channels
- `force_std`: shape `[4, 1]`, dtype `float32`, categories: normalization, possible_state_channels
- `input_feature_names`: shape `[1, 26]`, dtype `object`, categories: feature_names, input_histories, possible_inputs_by_dimension
- `num_input_features`: shape `[1, 1]`, dtype `float64`, categories: feature_names, input_histories
- `num_output_features`: shape `[1, 1]`, dtype `float64`, categories: feature_names, input_histories, targets
- `output_feature_names`: shape `[1, 8]`, dtype `object`, categories: feature_names, input_histories, possible_targets_by_dimension, targets
- `x_mean_rom`: shape `[26, 1]`, dtype `float32`, categories: input_histories, normalization, possible_inputs_by_dimension
- `x_std_rom`: shape `[26, 1]`, dtype `float32`, categories: input_histories, normalization, possible_inputs_by_dimension
- `y_mean_rom`: shape `[8, 1]`, dtype `float32`, categories: normalization, possible_targets_by_dimension, targets
- `y_std_rom`: shape `[8, 1]`, dtype `float32`, categories: normalization, possible_targets_by_dimension, targets

## Array Statistics

| Name | Shape | Dtype | Min | Max | Mean | Std | NaN | Inf |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ForceAvailable_train` | `[1, 403416]` | `uint8` | 1.0 | 1.0 | 1.0 | 0.0 | 0 | 0 |
| `ForceAvailable_val` | `[1, 44824]` | `uint8` | 1.0 | 1.0 | 1.0 | 0.0 | 0 | 0 |
| `Force_train_hist` | `[4, 10, 403416]` | `float32` | -32447868.0 | 29331046.0 | -958468.7775152092 | 8362659.864975832 | 0 | 0 |
| `Force_val_hist` | `[4, 10, 44824]` | `float32` | -31009300.0 | 25713522.0 | -942875.8543273291 | 8265085.733444767 | 0 | 0 |
| `ROMMetadata/all_requested_files` | `[1, 4]` | `object` |  |  |  |  |  |  |
| `ROMMetadata/approximate_history_duration` | `[1, 1]` | `float64` | 0.5000000000000027 | 0.5000000000000027 | 0.5000000000000027 | 0.0 | 0 | 0 |
| `ROMMetadata/description` | `[163, 1]` | `uint16` |  |  |  |  |  |  |
| `ROMMetadata/force_feature_names` | `[1, 4]` | `object` |  |  |  |  |  |  |
| `ROMMetadata/force_usage` | `[79, 1]` | `uint16` |  |  |  |  |  |  |
| `ROMMetadata/global_median_sample_time` | `[1, 1]` | `float64` | 0.050000000000000266 | 0.050000000000000266 | 0.050000000000000266 | 0.0 | 0 | 0 |
| `ROMMetadata/history_length` | `[1, 1]` | `float64` | 10.0 | 10.0 | 10.0 | 0.0 | 0 | 0 |
| `ROMMetadata/input_dimension` | `[1, 1]` | `float64` | 26.0 | 26.0 | 26.0 | 0.0 | 0 | 0 |
| `ROMMetadata/input_feature_names` | `[1, 26]` | `object` |  |  |  |  |  |  |
| `ROMMetadata/input_layout` | `[94, 1]` | `uint16` |  |  |  |  |  |  |
| `ROMMetadata/num_total_valid_runs` | `[1, 1]` | `float64` | 1040.0 | 1040.0 | 1040.0 | 0.0 | 0 | 0 |
| `ROMMetadata/num_training_runs` | `[1, 1]` | `float64` | 936.0 | 936.0 | 936.0 | 0.0 | 0 | 0 |
| `ROMMetadata/num_training_samples` | `[1, 1]` | `float64` | 403416.0 | 403416.0 | 403416.0 | 0.0 | 0 | 0 |
| `ROMMetadata/num_validation_runs` | `[1, 1]` | `float64` | 104.0 | 104.0 | 104.0 | 0.0 | 0 | 0 |
| `ROMMetadata/num_validation_samples` | `[1, 1]` | `float64` | 44824.0 | 44824.0 | 44824.0 | 0.0 | 0 | 0 |
| `ROMMetadata/output_dimension` | `[1, 1]` | `float64` | 8.0 | 8.0 | 8.0 | 0.0 | 0 | 0 |
| `ROMMetadata/output_feature_names` | `[1, 8]` | `object` |  |  |  |  |  |  |
| `ROMMetadata/output_layout` | `[60, 1]` | `uint16` |  |  |  |  |  |  |
| `ROMMetadata/random_seed` | `[1, 1]` | `float64` | 42.0 | 42.0 | 42.0 | 0.0 | 0 | 0 |
| `ROMMetadata/regime_definition` | `[1, 3]` | `object` |  |  |  |  |  |  |
| `ROMMetadata/train_fraction` | `[1, 1]` | `float64` | 0.9 | 0.9 | 0.9 | 0.0 | 0 | 0 |
| `ROMMetadata/training_source_files` | `[1, 936]` | `object` |  |  |  |  |  |  |
| `ROMMetadata/transition_definition` | `[83, 1]` | `uint16` |  |  |  |  |  |  |
| `ROMMetadata/validation_source_files` | `[1, 104]` | `object` |  |  |  |  |  |  |
| `ROMMetadata/version` | `[16, 1]` | `uint16` |  |  |  |  |  |  |
| `T` | `[1, 1]` | `float64` | 10.0 | 10.0 | 10.0 | 0.0 | 0 | 0 |
| `TrainClass` | `[1, 403416]` | `uint8` | 1.0 | 3.0 | 2.7610878101017313 | 0.5290985521529554 | 0 | 0 |
| `TrainRunID` | `[1, 403416]` | `uint32` | 1.0 | 936.0 | 468.5 | 270.1997717738982 | 0 | 0 |
| `TrainSampleTime` | `[1, 403416]` | `float32` | 0.05000000074505806 | 0.05000000074505806 | 0.05000000074505806 | 0.0 | 0 | 0 |
| `TrainTimeIndex` | `[1, 403416]` | `uint32` | 10.0 | 440.0 | 225.0 | 124.41864811996632 | 0 | 0 |
| `ValClass` | `[1, 44824]` | `uint8` | 1.0 | 3.0 | 2.7638764947349634 | 0.5245377527901608 | 0 | 0 |
| `ValRunID` | `[1, 44824]` | `uint32` | 1.0 | 104.0 | 52.5 | 30.02082610455615 | 0 | 0 |
| `ValSampleTime` | `[1, 44824]` | `float32` | 0.05000000074505806 | 0.05000000074505806 | 0.05000000074505806 | 0.0 | 0 | 0 |
| `ValTimeIndex` | `[1, 44824]` | `uint32` | 10.0 | 440.0 | 225.0 | 124.41864811996632 | 0 | 0 |
| `X_train_rom` | `[26, 10, 403416]` | `float32` | -123096920.0 | 122235480.0 | -518976.9267751736 | 26010175.0558245 | 0 | 0 |
| `X_val_rom` | `[26, 10, 44824]` | `float32` | -119999080.0 | 120886672.0 | -513565.87976136815 | 25977346.481231254 | 0 | 0 |
| `Y_train_rom` | `[8, 403416]` | `float32` | -107367752.0 | 114542560.0 | -35685.021681039594 | 15823575.293502875 | 0 | 0 |
| `Y_val_rom` | `[8, 44824]` | `float32` | -105812720.0 | 113266168.0 | -26948.980083808525 | 15917209.416411774 | 0 | 0 |
| `delta_dp_mean` | `[4, 1]` | `float32` | -180294.140625 | -457.11956787109375 | -71370.04747009277 | 66363.26287623988 | 0 | 0 |
| `delta_dp_std` | `[4, 1]` | `float32` | 18534198.0 | 26624212.0 | 22104101.5 | 3489136.5011238456 | 0 | 0 |
| `delta_extension_mean` | `[4, 1]` | `float32` | -0.00017572008073329926 | -1.2438244993973058e-05 | -0.00011608757699832495 | 6.330388715540895e-05 | 0 | 0 |
| `delta_extension_std` | `[4, 1]` | `float32` | 0.006648748181760311 | 0.024602295830845833 | 0.0140340969664976 | 0.007353154204063874 | 0 | 0 |
| `force_feature_names` | `[1, 4]` | `object` |  |  |  |  |  |  |
| `force_mean` | `[4, 1]` | `float32` | -7993854.5 | 3292269.5 | -958469.0390625 | 4232819.437659023 | 0 | 0 |
| `force_std` | `[4, 1]` | `float32` | 3854827.25 | 10350216.0 | 6721773.9375 | 2614401.4839225058 | 0 | 0 |
| `input_feature_names` | `[1, 26]` | `object` |  |  |  |  |  |  |
| `num_input_features` | `[1, 1]` | `float64` | 26.0 | 26.0 | 26.0 | 0.0 | 0 | 0 |
| `num_output_features` | `[1, 1]` | `float64` | 8.0 | 8.0 | 8.0 | 0.0 | 0 | 0 |
| `output_feature_names` | `[1, 8]` | `object` |  |  |  |  |  |  |
| `x_mean_rom` | `[26, 1]` | `float32` | -33502008.0 | 13069827.0 | -518977.3817957751 | 7138337.117319735 | 0 | 0 |
| `x_std_rom` | `[26, 1]` | `float32` | 0.01352985855191946 | 69767520.0 | 9738832.560387695 | 23037546.797888085 | 0 | 0 |
| `y_mean_rom` | `[8, 1]` | `float32` | -180294.140625 | -1.2438244993973058e-05 | -35685.02379309018 | 58953.051190067294 | 0 | 0 |
| `y_std_rom` | `[8, 1]` | `float32` | 0.006648748181760311 | 26624212.0 | 11052050.757017048 | 11324083.291251766 | 0 | 0 |

## Feature Name Arrays

- `ROMMetadata/all_requested_files` (4 values): `Hydraulic_Physics_Curriculum_10_2.mat`, `Hydraulic_Physics_Curriculum_10.mat`, `Hydraulic_Physics_Curriculum_20.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`
- `ROMMetadata/description` (1 values): `Reduced hydraulic actuator transition model dataset: history of valve, pressure, cylinder and mechanical states to next cylinder-extension and pressure increments.`
- `ROMMetadata/force_feature_names` (4 values): `Force_1`, `Force_2`, `Force_3`, `Force_4`
- `ROMMetadata/force_usage` (1 values): `Force is saved separately and is not part of the core 26-dimensional ROM input.`
- `ROMMetadata/input_feature_names` (26 values): `ValveCmd_1`, `ValveCmd_2`, `ValveCmd_3`, `ValveCmd_4`, `dP_1`, `dP_2`, `dP_3`, `dP_4`, `Extension_1`, `Extension_2`, `Extension_3`, `Extension_4`, `ExtensionVelocity_1`, `ExtensionVelocity_2`, `ExtensionVelocity_3`, `ExtensionVelocity_4`, `JointPos_1`, `JointPos_2`, `JointPos_3`, `JointPos_4`, `JointPos_5`, `JointVel_1`, `JointVel_2`, `JointVel_3`, `JointVel_4`, `JointVel_5`
- `ROMMetadata/input_layout` (1 values): `[ValveCmd(1:4), dP(1:4), Extension(1:4), ExtensionVelocity(1:4), JointPos(1:5), JointVel(1:5)]`
- `ROMMetadata/output_feature_names` (8 values): `DeltaExtension_next_1`, `DeltaExtension_next_2`, `DeltaExtension_next_3`, `DeltaExtension_next_4`, `DeltaDP_next_1`, `DeltaDP_next_2`, `DeltaDP_next_3`, `DeltaDP_next_4`
- `ROMMetadata/output_layout` (1 values): `[Extension(k+1,1:4)-Extension(k,1:4), dP(k+1,1:4)-dP(k,1:4)]`
- `ROMMetadata/regime_definition` (3 values): `Class 1: max(abs(final ValveCmd)) < 0.3`, `Class 2: 0.3 <= max(abs(final ValveCmd)) < 0.8`, `Class 3: max(abs(final ValveCmd)) >= 0.8`
- `ROMMetadata/training_source_files` (936 values): `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Physics_Curriculum_10.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Physics_Curriculum_20.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Physics_Curriculum_10_2.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Physics_Curriculum_20.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat` ...
- `ROMMetadata/transition_definition` (1 values): `Input window ends at timestep k. The target describes the transition from k to k+1.`
- `ROMMetadata/validation_source_files` (104 values): `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Physics_Curriculum_20.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Physics_Curriculum_10_2.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Physics_Curriculum_20.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat`, `Hydraulic_Physics_Curriculum_10.mat`, `Hydraulic_Mechanic_Dataset_Curriculum_1000.mat` ...
- `ROMMetadata/version` (1 values): `ROM_Dataset_V1.0`
- `force_feature_names` (4 values): `Force_1`, `Force_2`, `Force_3`, `Force_4`
- `input_feature_names` (26 values): `ValveCmd_1`, `ValveCmd_2`, `ValveCmd_3`, `ValveCmd_4`, `dP_1`, `dP_2`, `dP_3`, `dP_4`, `Extension_1`, `Extension_2`, `Extension_3`, `Extension_4`, `ExtensionVelocity_1`, `ExtensionVelocity_2`, `ExtensionVelocity_3`, `ExtensionVelocity_4`, `JointPos_1`, `JointPos_2`, `JointPos_3`, `JointPos_4`, `JointPos_5`, `JointVel_1`, `JointVel_2`, `JointVel_3`, `JointVel_4`, `JointVel_5`
- `output_feature_names` (8 values): `DeltaExtension_next_1`, `DeltaExtension_next_2`, `DeltaExtension_next_3`, `DeltaExtension_next_4`, `DeltaDP_next_1`, `DeltaDP_next_2`, `DeltaDP_next_3`, `DeltaDP_next_4`

## Split / Run Overlap Evidence

- `TrainRunID`: shape `[403416]`, unique values `936`, range `1..936`
- `ValRunID`: shape `[44824]`, unique values `104`, range `1..104`

Source-file metadata:
- `ROMMetadata/training_source_files`: count `936`, unique source files `4`, preview `['Hydraulic_Mechanic_Dataset_Curriculum_1000.mat', 'Hydraulic_Physics_Curriculum_10.mat', 'Hydraulic_Physics_Curriculum_10_2.mat', 'Hydraulic_Physics_Curriculum_20.mat']`
- `ROMMetadata/validation_source_files`: count `104`, unique source files `4`, preview `['Hydraulic_Mechanic_Dataset_Curriculum_1000.mat', 'Hydraulic_Physics_Curriculum_10.mat', 'Hydraulic_Physics_Curriculum_10_2.mat', 'Hydraulic_Physics_Curriculum_20.mat']`

Numeric run ID overlap evidence:
- `TrainRunID` vs `ValRunID`: 104 shared numeric IDs, preview `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]`. Numeric run IDs may be local to each split; inspect source run identity before treating this as leakage.

Source-file name overlap evidence:
- `ROMMetadata/training_source_files` vs `ROMMetadata/validation_source_files`: 4 shared source-file labels, preview `['Hydraulic_Mechanic_Dataset_Curriculum_1000.mat', 'Hydraulic_Physics_Curriculum_10.mat', 'Hydraulic_Physics_Curriculum_10_2.mat', 'Hydraulic_Physics_Curriculum_20.mat']`. Source file names identify source datasets, not necessarily individual simulation runs.
- The dataset contains train and validation run IDs, but validation IDs restart at 1.
- The safest loader convention is to namespace run identity by split unless a global run identifier is added.
- No dedicated test split was detected.

## Dimension Evidence

- `ROMMetadata/input_feature_names` shape `[1, 26]`: contains expected core input dimension 26
- `ROMMetadata/output_feature_names` shape `[1, 8]`: contains expected target dimension 8
- `X_train_rom` shape `[26, 10, 403416]`: contains expected core input dimension 26
- `X_val_rom` shape `[26, 10, 44824]`: contains expected core input dimension 26
- `Y_train_rom` shape `[8, 403416]`: contains expected target dimension 8
- `Y_val_rom` shape `[8, 44824]`: contains expected target dimension 8
- `input_feature_names` shape `[1, 26]`: contains expected core input dimension 26
- `output_feature_names` shape `[1, 8]`: contains expected target dimension 8
- `x_mean_rom` shape `[26, 1]`: contains expected core input dimension 26
- `x_std_rom` shape `[26, 1]`: contains expected core input dimension 26
- `y_mean_rom` shape `[8, 1]`: contains expected target dimension 8
- `y_std_rom` shape `[8, 1]`: contains expected target dimension 8

## Inspection Notes

- No dimensions were transposed or reshaped during inspection.
- Dimension-order ambiguity must be resolved in ROM_DATA_CONTRACT.md before training.
- Future-target timing is reported from available variable names and shapes only; loader tests must verify numeric timing once the contract is explicit.
