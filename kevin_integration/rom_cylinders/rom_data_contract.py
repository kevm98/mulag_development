from __future__ import annotations

from dataclasses import dataclass


CORE_INPUT_FEATURE_NAMES: tuple[str, ...] = (
    "ValveCmd_1",
    "ValveCmd_2",
    "ValveCmd_3",
    "ValveCmd_4",
    "dP_1",
    "dP_2",
    "dP_3",
    "dP_4",
    "Extension_1",
    "Extension_2",
    "Extension_3",
    "Extension_4",
    "ExtensionVelocity_1",
    "ExtensionVelocity_2",
    "ExtensionVelocity_3",
    "ExtensionVelocity_4",
    "JointPos_1",
    "JointPos_2",
    "JointPos_3",
    "JointPos_4",
    "JointPos_5",
    "JointVel_1",
    "JointVel_2",
    "JointVel_3",
    "JointVel_4",
    "JointVel_5",
)

FORCE_FEATURE_NAMES: tuple[str, ...] = ("Force_1", "Force_2", "Force_3", "Force_4")
AUGMENTED_INPUT_FEATURE_NAMES: tuple[str, ...] = CORE_INPUT_FEATURE_NAMES + FORCE_FEATURE_NAMES

TARGET_FEATURE_NAMES: tuple[str, ...] = (
    "DeltaExtension_next_1",
    "DeltaExtension_next_2",
    "DeltaExtension_next_3",
    "DeltaExtension_next_4",
    "DeltaDP_next_1",
    "DeltaDP_next_2",
    "DeltaDP_next_3",
    "DeltaDP_next_4",
)

CORE_INPUT_DIM = len(CORE_INPUT_FEATURE_NAMES)
FORCE_INPUT_DIM = len(FORCE_FEATURE_NAMES)
AUGMENTED_INPUT_DIM = len(AUGMENTED_INPUT_FEATURE_NAMES)
TARGET_DIM = len(TARGET_FEATURE_NAMES)

HISTORY_LENGTH = 10
TARGET_HORIZON = 1
DT_SECONDS = 0.05000000074505806
APPROXIMATE_HISTORY_DURATION_SECONDS = 0.5000000000000027

MATLAB_X_DIMS = ("input_dim", "history_length", "sample")
MATLAB_Y_DIMS = ("target_dim", "sample")
PYTORCH_X_DIMS = ("batch", "history_length", "input_dim")
PYTORCH_Y_DIMS = ("batch", "target_dim")

VALVE_SLICE = slice(0, 4)
DP_SLICE = slice(4, 8)
EXTENSION_SLICE = slice(8, 12)
EXTENSION_VELOCITY_SLICE = slice(12, 16)
JOINT_POS_SLICE = slice(16, 21)
JOINT_VEL_SLICE = slice(21, 26)
TARGET_DELTA_EXTENSION_SLICE = slice(0, 4)
TARGET_DELTA_DP_SLICE = slice(4, 8)

TRAIN_SPLIT = "train"
VALIDATION_SPLIT = "val"
SUPPORTED_SPLITS: tuple[str, ...] = (TRAIN_SPLIT, VALIDATION_SPLIT)

NORMALIZATION_CONVENTION = "zscore_from_training_split"
SPLIT_CONVENTION = (
    "The MATLAB file contains complete train and validation run splits. "
    "Run IDs are split-local, so use split-qualified run identity for leakage checks."
)
TARGET_TIMING_CONVENTION = "Input history ends at timestep k; target is transition k -> k+1."


@dataclass(frozen=True)
class ROMDataContract:
    core_input_feature_names: tuple[str, ...] = CORE_INPUT_FEATURE_NAMES
    force_feature_names: tuple[str, ...] = FORCE_FEATURE_NAMES
    augmented_input_feature_names: tuple[str, ...] = AUGMENTED_INPUT_FEATURE_NAMES
    target_feature_names: tuple[str, ...] = TARGET_FEATURE_NAMES
    core_input_dim: int = CORE_INPUT_DIM
    augmented_input_dim: int = AUGMENTED_INPUT_DIM
    target_dim: int = TARGET_DIM
    history_length: int = HISTORY_LENGTH
    target_horizon: int = TARGET_HORIZON
    dt_seconds: float = DT_SECONDS
    normalization_convention: str = NORMALIZATION_CONVENTION
    split_convention: str = SPLIT_CONVENTION
    target_timing_convention: str = TARGET_TIMING_CONVENTION


def feature_names(use_force_input: bool = False) -> tuple[str, ...]:
    return AUGMENTED_INPUT_FEATURE_NAMES if use_force_input else CORE_INPUT_FEATURE_NAMES


def input_dim(use_force_input: bool = False) -> int:
    return AUGMENTED_INPUT_DIM if use_force_input else CORE_INPUT_DIM


def validate_contract(
    *,
    input_feature_names: tuple[str, ...],
    target_feature_names: tuple[str, ...],
    history_length: int,
    input_dimension: int,
    output_dimension: int,
    use_force_input: bool = False,
) -> None:
    expected_inputs = feature_names(use_force_input)
    if input_feature_names != expected_inputs:
        raise ValueError(f"Input feature order mismatch. Expected {expected_inputs}, got {input_feature_names}.")
    if target_feature_names != TARGET_FEATURE_NAMES:
        raise ValueError(f"Target feature order mismatch. Expected {TARGET_FEATURE_NAMES}, got {target_feature_names}.")
    if int(history_length) != HISTORY_LENGTH:
        raise ValueError(f"History length mismatch. Expected {HISTORY_LENGTH}, got {history_length}.")
    if int(input_dimension) != input_dim(use_force_input):
        raise ValueError(f"Input dimension mismatch. Expected {input_dim(use_force_input)}, got {input_dimension}.")
    if int(output_dimension) != TARGET_DIM:
        raise ValueError(f"Output dimension mismatch. Expected {TARGET_DIM}, got {output_dimension}.")
