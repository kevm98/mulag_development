from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from types import MethodType
from typing import Any

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ACTUATOR_JOINT_NAMES = [
    "Drehzapfen_joint",
    "Ausleger_I_joint",
    "Ausleger_II_joint",
    "Messerkopf_Schwenk_joint",
]
STATE_JOINT_NAMES = [
    "Drehzapfen_joint",
    "Ausleger_I_joint",
    "Ausleger_II_joint",
    "Messerkopf_Schwenk_joint",
    "Messerkopf_joint",
]
MECHANICAL_FAMILY_PREFIXES = (
    "Drehzapfen_constraint_joint_",
    "Ausleger_I_constraint_joint_",
    "Ausleger_II_constraint_joint_",
    "Messerkopf_Schwenk_constraint_joint_",
)
ROM_FEATURE_ORDER = (
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

parser = argparse.ArgumentParser(description="Manual final causal ROM Isaac diagnostic. Does not validate physics.")
parser.add_argument("--task", type=str, default="Kevin-Mulag-AdaptiveRL-JointReach-v0")
parser.add_argument("--model_mode", choices=["kinematic", "linear_correction"], default="kinematic")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--valve_commands", type=str, default=None)
parser.add_argument("--steps_per_test", type=int, default=200)
parser.add_argument("--settle_steps", type=int, default=200)
parser.add_argument("--single_actuator", type=str, default=None)
parser.add_argument("--output_dir", type=Path, default=ROOT / "logs" / "final_causal_rom_integration")
parser.add_argument("--model_dir", type=Path, default=ROOT / "kevin_integration" / "rom_cylinders" / "exported_final_causal_rom")
parser.add_argument("--training_dataset", type=Path, default=ROOT / "kevin_integration" / "rom_cylinders" / "pytorch_hydraulic_rom_dataset_causal.mat")
parser.add_argument("--diagnostic_dp_source", choices=["zero", "constant", "training_mean", "environment"], default="zero")
parser.add_argument("--diagnostic_dp_values", type=str, default="0,0,0,0")
parser.add_argument("--extension_unit_scale", type=float, default=1.0)
parser.add_argument(
    "--coordinate_mapping_mode",
    choices=["none", "nominal_sign_offset", "affine_validated", "validated", "provisional"],
    default="none",
)
parser.add_argument(
    "--coordinate_mapping",
    type=Path,
    default=ROOT / "kevin_integration" / "rom_cylinders" / "cylinder_coordinate_mapping.json",
)
parser.add_argument("--allow_provisional_range_mapping", action="store_true")
parser.add_argument("--allow_provisional_nominal_reference", action="store_true")
parser.add_argument("--max_mapping_range_excess", type=float, default=0.05)
parser.add_argument("--max_abs_target_increment", type=float, default=None)
parser.add_argument("--max_abs_rom_delta_50ms", type=float, default=0.04)
parser.add_argument("--max_abs_applied_delta_10ms", type=float, default=0.008)
parser.add_argument("--max_abs_cumulative_target_change", type=float, default=0.10)
parser.add_argument("--physical_limit_tolerance", type=float, default=1.0e-5)
parser.add_argument("--passive_hold_mode", choices=["fixed", "measured_follow"], default="measured_follow")
parser.add_argument("--passive_target_tolerance", type=float, default=1.0e-6)
parser.add_argument("--dump_first_feature_row", action="store_true")
parser.add_argument("--stop_on_limit_violation", action="store_true")
parser.add_argument("--print_every", type=int, default=25)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import h5py
import numpy as np
import torch

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import kevin_integration.tasks  # noqa: F401
from kevin_integration.rom_cylinders.cylinder_coordinate_adapter import (
    CYLINDER_ORDER,
    CylinderCoordinateMapping,
    load_coordinate_mapping,
    mapping_metadata_for_logging,
)
from kevin_integration.rom_cylinders.final_rom_runtime_contract import (
    active_actuator_mask,
    apply_active_mask,
    assert_initialized_from_measured,
    assert_passive_targets_held,
    compute_limit_audit,
    merge_active_and_passive_targets,
    passive_hold_target,
)
from kevin_integration.rom_cylinders.final_causal_linear_rom import FinalCausalROMInference
from kevin_integration.utils.sim_memory import apply_kevin_sim_memory_optimizations


def parse_float_csv(text: str, *, expected: int | None = None) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("CSV argument must contain at least one value.")
    if expected is not None and len(values) != expected:
        raise ValueError(f"Expected {expected} values, got {len(values)}.")
    return values


def safe_name(value: str) -> str:
    return value.replace("-", "neg_").replace(".", "p").replace("+", "pos_")


def robot_joint_names(robot) -> list[str]:
    names = getattr(robot, "joint_names", None) or getattr(robot.data, "joint_names", None)
    return list(names) if names is not None else []


def resolve_required_joint_ids(robot, joint_names: Sequence[str]) -> list[int]:
    ids = []
    for name in joint_names:
        joint_ids, resolved = robot.find_joints(name)
        if len(joint_ids) == 0:
            raise RuntimeError(f"FAIL: could not resolve actuator joint {name!r}.")
        actual = resolved[0] if resolved else name
        print(f"PASS: resolved actuator {name} -> id={int(joint_ids[0])} actual={actual}", flush=True)
        ids.append(int(joint_ids[0]))
    return ids


def resolve_mechanical_families(robot) -> tuple[list[str], list[int]]:
    names = robot_joint_names(robot)
    out_names: list[str] = []
    out_ids: list[int] = []
    for idx, name in enumerate(names):
        if any(name.startswith(prefix) for prefix in MECHANICAL_FAMILY_PREFIXES):
            out_names.append(name)
            out_ids.append(idx)
    print(f"PASS: resolved {len(out_ids)} expected mechanical-family joints.", flush=True)
    return out_names, out_ids


def joint_limits(robot, joint_ids: list[int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    soft = getattr(robot.data, "soft_joint_pos_limits", None)
    if isinstance(soft, torch.Tensor):
        values = soft[0, joint_ids] if soft.ndim == 3 else soft[joint_ids]
        return values[:, 0].to(device), values[:, 1].to(device)
    print("WARN: soft joint limits unavailable; physical-limit checks use +/-inf.", flush=True)
    return torch.full((len(joint_ids),), -math.inf, device=device), torch.full((len(joint_ids),), math.inf, device=device)


def load_metadata(model_dir: Path) -> dict[str, Any]:
    metadata_path = model_dir / "best_causal_linear_rom_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("selected_model_type") != "causal_linear_correction":
        raise RuntimeError("FAIL: wrong checkpoint type; expected selected causal_linear_correction export.")
    if not metadata.get("causality_flag", False):
        raise RuntimeError("FAIL: exported model is not marked causal.")
    if tuple(metadata.get("feature_order", ())) != ROM_FEATURE_ORDER:
        raise RuntimeError("FAIL: feature-order mismatch.")
    if int(len(metadata.get("target_order", []))) != 4:
        raise RuntimeError("FAIL: output dimension must be 4.")
    if abs(float(metadata.get("dt", 0.0)) - 0.05) > 1.0e-6:
        raise RuntimeError("FAIL: metadata dt is not 0.05 s.")
    return metadata


def load_training_ranges(dataset_path: Path) -> dict[str, np.ndarray] | None:
    if not dataset_path.exists():
        print(f"WARN: training dataset not found for range audit: {dataset_path}", flush=True)
        return None
    minima = np.full((26,), np.inf, dtype=np.float64)
    maxima = np.full((26,), -np.inf, dtype=np.float64)
    with h5py.File(dataset_path, "r") as handle:
        for key in ("X_train_rom",):
            x = handle[key]
            for start in range(0, x.shape[2], 32768):
                arr = np.asarray(x[:, :, start : start + 32768], dtype=np.float64).reshape(26, -1)
                minima = np.minimum(minima, np.min(arr, axis=1))
                maxima = np.maximum(maxima, np.max(arr, axis=1))
    return {"min": minima, "max": maxima}


def unit_check(metadata: dict[str, Any], ranges: dict[str, np.ndarray] | None, current_pos: torch.Tensor, scale: float) -> None:
    print(f"ROM metadata extension unit: metres", flush=True)
    print(f"Isaac joint position unit assumption: metres", flush=True)
    print(f"extension_unit_scale override: {scale}", flush=True)
    if ranges is None:
        print("WARN: dataset extension min/max unavailable; unit range check skipped.", flush=True)
        return
    dataset_min = ranges["min"][8:12]
    dataset_max = ranges["max"][8:12]
    measured = current_pos.detach().cpu().numpy().reshape(-1)
    print(f"Dataset Extension min/max: {list(zip(dataset_min.tolist(), dataset_max.tolist(), strict=False))}", flush=True)
    print(f"Initial Isaac prismatic values: {measured.tolist()}", flush=True)
    span = np.maximum(dataset_max - dataset_min, 1.0e-12)
    outside = (measured * float(scale) < dataset_min - 2.0 * span) | (measured * float(scale) > dataset_max + 2.0 * span)
    if np.any(outside):
        raise RuntimeError("FAIL: probable severe extension-unit mismatch; use --extension_unit_scale only with evidence.")


def disable_control_paths(env) -> None:
    unwrapped = env.unwrapped
    cfg = getattr(unwrapped, "cfg", None)
    if cfg is not None:
        cfg.use_final_causal_cylinder_rom = False
        cfg.use_reduced_hydraulic_rom = False
        cfg.use_fd_effort = False
        cfg.use_sd_feedback = False
        cfg.use_sd_observer = False
        cfg.position_delta_source = "direct_action"
        cfg.control_mode = "position_delta"
    if hasattr(unwrapped, "_current_torque"):
        unwrapped._current_torque.zero_()

    def noop_pre_physics(self, actions):
        if hasattr(self, "_current_action"):
            self._current_action.zero_()
        if hasattr(self, "_current_torque"):
            self._current_torque.zero_()

    def noop_apply_action(self):
        return None

    unwrapped._pre_physics_step = MethodType(noop_pre_physics, unwrapped)
    unwrapped._apply_action = MethodType(noop_apply_action, unwrapped)
    print("PASS: PPO/ID/FD/SD/env action paths bypassed for diagnostic.", flush=True)


def tensor_values(tensor: torch.Tensor) -> list[float]:
    return [float(value) for value in tensor.detach().cpu().reshape(-1).tolist()]


def feature_row(
    *,
    valve_cmd: torch.Tensor,
    dP: torch.Tensor,
    extension: torch.Tensor,
    velocity: torch.Tensor,
    joint_pos5: torch.Tensor,
    joint_vel5: torch.Tensor,
) -> torch.Tensor:
    return torch.cat([valve_cmd, dP, extension, velocity, joint_pos5, joint_vel5], dim=-1).reshape(1, 1, 26)


def dp_for_step(
    env,
    explicit: torch.Tensor,
    source: str,
    device: torch.device,
    training_mean_dp: torch.Tensor | None = None,
) -> torch.Tensor:
    if source == "constant":
        return explicit.clone()
    if source == "training_mean":
        if training_mean_dp is None:
            raise RuntimeError("FAIL: training_mean dP requested, but exported scaler statistics are unavailable.")
        return training_mean_dp.to(device=device, dtype=torch.float32).clone()
    if source == "environment":
        value = getattr(env.unwrapped, "_dP", None)
        if isinstance(value, torch.Tensor):
            return value[:, :4].detach().clone().to(device)
        raise RuntimeError("FAIL: --diagnostic_dp_source environment requested, but env._dP is unavailable.")
    return torch.zeros((1, 4), dtype=torch.float32, device=device)


def load_runtime_mapping(device: torch.device) -> CylinderCoordinateMapping | None:
    del device
    if args_cli.coordinate_mapping_mode == "none":
        print("WARN: coordinate_mapping_mode=none uses direct Isaac prismatic coordinates as ROM Extension.", flush=True)
        print("WARN: this mode is plumbing-only and must not be claimed as physical ROM validation.", flush=True)
        return None
    mode = "affine_validated" if args_cli.coordinate_mapping_mode == "validated" else args_cli.coordinate_mapping_mode
    mapping = load_coordinate_mapping(
        args_cli.coordinate_mapping,
        mode=mode,
        allow_provisional_range_mapping=args_cli.allow_provisional_range_mapping,
        allow_provisional_nominal_reference=args_cli.allow_provisional_nominal_reference,
    )
    if args_cli.coordinate_mapping_mode == "provisional":
        print("WARN: using provisional_range_mapping for diagnostics only; not valid for deployment claims.", flush=True)
    if args_cli.coordinate_mapping_mode == "nominal_sign_offset":
        if args_cli.allow_provisional_nominal_reference:
            print("", flush=True)
            print("PROVISIONAL MAPPING:", flush=True)
            print("The nominal cylinder reference was extracted from one causal dataset sample", flush=True)
            print("with a nonzero pose-matching error. Results are valid only for plumbing,", flush=True)
            print("sign, scale-sanity, and safety diagnostics. They are not final physical", flush=True)
            print("validation.", flush=True)
            print("", flush=True)
        print("PASS: using nominal_sign_offset coordinate mapping.", flush=True)
    return mapping


def nominal_consistency_check(mapping: CylinderCoordinateMapping | None, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if mapping is None:
        nan = torch.full((1, 4), float("nan"), dtype=torch.float32, device=device)
        return nan, nan
    q_nominal = mapping.isaac_nominal_tensor().to(device=device, dtype=torch.float64).reshape(1, 4)
    x_expected = mapping.simscape_nominal_tensor().to(device=device, dtype=torch.float64).reshape(1, 4)
    x_reconstructed = mapping.isaac_position_to_simscape_extension(q_nominal)
    q_reconstructed = mapping.simscape_extension_to_isaac_position(x_reconstructed)
    nominal_error = torch.abs(x_reconstructed - x_expected)
    round_trip_error = torch.abs(q_reconstructed - q_nominal)
    for i, name in enumerate(ACTUATOR_JOINT_NAMES):
        print(
            f"Nominal check {name}: reconstructed nominal Extension={float(x_reconstructed[0, i]):.12g} "
            f"expected nominal Extension={float(x_expected[0, i]):.12g} "
            f"nominal reconstruction error={float(nominal_error[0, i]):.12g} "
            f"round-trip error={float(round_trip_error[0, i]):.12g}",
            flush=True,
        )
    if torch.max(nominal_error).item() > 1.0e-6:
        raise RuntimeError("FAIL: nominal reconstruction error exceeds 1e-6 m.")
    if torch.max(round_trip_error).item() > 1.0e-6:
        raise RuntimeError("FAIL: nominal mapping round-trip error exceeds 1e-6 m.")
    return nominal_error.to(dtype=torch.float32), round_trip_error.to(dtype=torch.float32)


def coordinate_state(
    mapping: CylinderCoordinateMapping | None,
    q_isaac: torch.Tensor,
    qdot_isaac: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if mapping is None:
        scale = float(args_cli.extension_unit_scale)
        extension = q_isaac * scale
        velocity = qdot_isaac * scale
        inverse_error = torch.full_like(q_isaac, float("nan"))
        range_violation_flags = torch.zeros_like(q_isaac, dtype=torch.bool)
        return extension, velocity, inverse_error, range_violation_flags
    extension = mapping.isaac_position_to_simscape_extension(q_isaac)
    velocity = mapping.isaac_velocity_to_simscape_velocity(qdot_isaac)
    inverse_error = mapping.inverse_mapping_error(q_isaac)
    range_violation_flags = mapping.isaac_limit_violation(q_isaac)
    return extension, velocity, inverse_error, range_violation_flags


def delta_simscape_to_isaac(mapping: CylinderCoordinateMapping | None, delta_extension: torch.Tensor) -> torch.Tensor:
    if mapping is None:
        return delta_extension / float(args_cli.extension_unit_scale)
    return mapping.simscape_delta_to_isaac_delta(delta_extension)


def range_violation(x: torch.Tensor, ranges: dict[str, np.ndarray] | None) -> tuple[bool, str, float]:
    if ranges is None:
        return False, "", 0.0
    values = x.detach().cpu().numpy().reshape(26)
    below = ranges["min"] - values
    above = values - ranges["max"]
    amount = np.maximum(below, above)
    idx = int(np.argmax(amount))
    if amount[idx] > 0.0:
        return True, ROM_FEATURE_ORDER[idx], float(amount[idx])
    return False, "", 0.0


def extension_range_audit(
    mapped_extension: torch.Tensor,
    ranges: dict[str, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = mapped_extension.detach().cpu().numpy().reshape(4)
    if ranges is None:
        nan = np.full((4,), np.nan, dtype=np.float64)
        return nan, nan, nan, np.zeros((4,), dtype=bool), np.zeros((4,), dtype=np.float64)
    training_min = ranges["min"][8:12]
    training_max = ranges["max"][8:12]
    below = training_min - values
    above = values - training_max
    excess = np.maximum(np.maximum(below, above), 0.0)
    margin = np.minimum(values - training_min, training_max - values)
    outside = excess > 0.0
    return training_min, training_max, margin, outside, excess


def assert_target_matches_measured(target: torch.Tensor, measured: torch.Tensor, *, label: str) -> None:
    error = torch.max(torch.abs(target - measured)).item()
    if error > 1.0e-6:
        raise RuntimeError(f"FAIL: {label} target initialization mismatch {error} m.")


def joint_position_target_buffer(robot, joint_ids: list[int], fallback: torch.Tensor) -> tuple[torch.Tensor, str]:
    value = getattr(robot.data, "joint_pos_target", None)
    if isinstance(value, torch.Tensor):
        selected = value[:, joint_ids] if value.ndim == 2 else value[joint_ids]
        return selected.detach().clone().to(device=fallback.device, dtype=fallback.dtype), "robot.data.joint_pos_target"
    return fallback.detach().clone(), "unavailable"


def main() -> None:
    valve_command_text = args_cli.valve_commands
    if valve_command_text is None:
        valve_command_text = "0.1" if args_cli.allow_provisional_nominal_reference else "0.1,0.3,0.5"
    valve_magnitudes = parse_float_csv(valve_command_text)
    max_abs_rom_delta_50ms = float(
        args_cli.max_abs_target_increment
        if args_cli.max_abs_target_increment is not None
        else args_cli.max_abs_rom_delta_50ms
    )
    explicit_dp = torch.tensor(parse_float_csv(args_cli.diagnostic_dp_values, expected=4), dtype=torch.float32).reshape(1, 4)
    metadata = load_metadata(args_cli.model_dir)
    ranges = load_training_ranges(args_cli.training_dataset)
    log_dir = args_cli.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / "final_causal_rom_integration.csv"

    @hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
    def run(env_cfg, agent_cfg):
        del agent_cfg
        if int(args_cli.num_envs) != 1:
            raise RuntimeError("FAIL: final causal ROM diagnostic currently supports --num_envs 1 only.")
        apply_kevin_sim_memory_optimizations(env_cfg.sim)
        env_cfg.scene.num_envs = int(args_cli.num_envs)
        env_cfg.use_final_causal_cylinder_rom = False
        env_cfg.use_reduced_hydraulic_rom = False
        env_cfg.use_fd_effort = False
        env_cfg.use_sd_feedback = False
        env_cfg.use_sd_observer = False
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)
        env = RslRlVecEnvWrapper(env)
        disable_control_paths(env)
        robot = env.unwrapped.robot
        device = env.unwrapped.device
        actuator_names = ACTUATOR_JOINT_NAMES
        if args_cli.single_actuator is not None:
            if args_cli.single_actuator not in ACTUATOR_JOINT_NAMES:
                raise RuntimeError(f"FAIL: --single_actuator must be one of {ACTUATOR_JOINT_NAMES}.")
            actuator_names = [args_cli.single_actuator]
        if tuple(ACTUATOR_JOINT_NAMES) != tuple(CYLINDER_ORDER):
            raise RuntimeError("FAIL: actuator order and coordinate-mapping order diverged.")
        state_joint_names = list(getattr(env_cfg, "state_joint_names", STATE_JOINT_NAMES))
        if tuple(state_joint_names) != tuple(STATE_JOINT_NAMES):
            raise RuntimeError(f"FAIL: unexpected state joint order {state_joint_names}.")
        actuator_ids_all = resolve_required_joint_ids(robot, ACTUATOR_JOINT_NAMES)
        state_joint_ids = resolve_required_joint_ids(robot, state_joint_names)
        active_mask = active_actuator_mask(args_cli.single_actuator, like=torch.zeros((1, 4), device=device))
        active_mask_bool = active_mask.to(dtype=torch.bool)
        mechanical_names, mechanical_ids = resolve_mechanical_families(robot)
        lower, upper = joint_limits(robot, actuator_ids_all, device)
        print(f"Loaded model path: {args_cli.model_dir / 'best_causal_linear_rom.pth'}", flush=True)
        print(f"Loaded scaler path: {args_cli.model_dir / 'best_causal_linear_rom_scaler.pth'}", flush=True)
        print(f"Metadata version: final_causal_linear_v1", flush=True)
        print(f"Dataset identity: {metadata.get('causal_dataset_identity')}", flush=True)
        print(f"Causality flag: {metadata.get('causality_flag')}", flush=True)
        print(f"Feature order: {metadata.get('feature_order')}", flush=True)
        print(f"Target names: {metadata.get('target_order')}", flush=True)
        print(f"dt: {metadata.get('dt')}", flush=True)
        print(f"Correction bounds: {metadata.get('correction_bounds')}", flush=True)
        print(f"Actuator joint IDs: {dict(zip(ACTUATOR_JOINT_NAMES, actuator_ids_all, strict=False))}", flush=True)
        print(f"State joint IDs: {dict(zip(state_joint_names, state_joint_ids, strict=False))}", flush=True)
        print(f"Mechanical joint IDs: {dict(zip(mechanical_names, mechanical_ids, strict=False))}", flush=True)
        print(f"Prismatic joint limits: {list(zip(tensor_values(lower), tensor_values(upper), strict=False))}", flush=True)
        print("Runtime control timestep: 0.01", flush=True)
        print("Multirate ratio: 5", flush=True)
        print(f"Active model mode: {args_cli.model_mode}", flush=True)
        print(f"Coordinate mapping mode: {args_cli.coordinate_mapping_mode}", flush=True)
        print(f"Active actuator mask: {tensor_values(active_mask)}", flush=True)
        print(f"Passive hold mode: {args_cli.passive_hold_mode}", flush=True)
        mapping = load_runtime_mapping(device)
        nominal_error, nominal_round_trip_error = nominal_consistency_check(mapping, device)
        linear = FinalCausalROMInference(args_cli.model_dir, device=device) if args_cli.model_mode == "linear_correction" else None
        scaler_payload = torch.load(args_cli.model_dir / "best_causal_linear_rom_scaler.pth", map_location=device)
        dp_feature_mean = scaler_payload["feature_mean"].to(device=device, dtype=torch.float32)[:, 0].reshape(1, 4)
        dp_feature_std = scaler_payload["feature_std"].to(device=device, dtype=torch.float32)[:, 0].reshape(1, 4)
        training_mean_dp = dp_feature_mean * 1.0e6
        zero_dp_normalized = (torch.zeros_like(dp_feature_mean) - dp_feature_mean) / torch.clamp(dp_feature_std, min=1.0e-12)
        dp_linear_weights = (
            linear.model.weights[:, 0].reshape(1, 4).to(device=device, dtype=torch.float32)
            if linear is not None
            else torch.zeros((1, 4), dtype=torch.float32, device=device)
        )
        zero_dp_contribution = zero_dp_normalized * dp_linear_weights
        print(f"Diagnostic dP source: {args_cli.diagnostic_dp_source}", flush=True)
        print(f"Training-mean dP reference raw Pa: {tensor_values(training_mean_dp)}", flush=True)
        print(f"Zero dP normalized values: {tensor_values(zero_dp_normalized)}", flush=True)
        print(f"Zero dP linear raw-score contribution: {tensor_values(zero_dp_contribution)}", flush=True)
        print("Feature contract: ValveCmd[4], dP[4], mapped Extension[4], mapped ExtensionVelocity[4], JointPos[5], JointVel[5]", flush=True)
        rows: list[dict[str, Any]] = []
        test_commands = [(0.0, "zero")]
        first_feature_dumped = False
        for magnitude in valve_magnitudes:
            test_commands.append((abs(magnitude), f"pos_{str(abs(magnitude)).replace('.', 'p')}"))
            test_commands.append((-abs(magnitude), f"neg_{str(abs(magnitude)).replace('.', 'p')}"))

        for actuator_name in actuator_names:
            actuator_index = ACTUATOR_JOINT_NAMES.index(actuator_name)
            for command_value, command_label in test_commands:
                obs, _ = env.reset()
                del obs
                disable_control_paths(env)
                current_pos = robot.data.joint_pos[:, actuator_ids_all].detach().clone()
                unit_check(metadata, ranges, current_pos, args_cli.extension_unit_scale)
                target = current_pos.clone()
                assert_initialized_from_measured(target, current_pos, tolerance=1.0e-6, label="reset")
                passive_hold = current_pos.clone()
                pending_increment = torch.zeros((1, 4), dtype=torch.float32, device=device)
                pending_delta_isaac_50ms = torch.zeros_like(pending_increment)
                pending_delta_isaac_50ms_raw = torch.zeros_like(pending_increment)
                pending_delta_simscape_50ms = torch.zeros_like(pending_increment)
                pending_kinematic = torch.zeros_like(pending_increment)
                pending_correction = torch.zeros_like(pending_increment)
                five_sum = torch.zeros_like(pending_increment)
                rom_update_index = -1
                initial_pos = current_pos.clone()
                previous_pos = current_pos.clone()
                for _ in range(args_cli.settle_steps):
                    robot.set_joint_position_target(target, joint_ids=actuator_ids_all)
                    env.step(torch.zeros((1, 4), dtype=torch.float32, device=device))
                    target = robot.data.joint_pos[:, actuator_ids_all].detach().clone()
                    qdot_zero = torch.zeros_like(target)
                    previous_pos, _, _, _ = coordinate_state(mapping, target, qdot_zero)
                target = robot.data.joint_pos[:, actuator_ids_all].detach().clone()
                passive_hold = target.clone()
                initial_pos = target.clone()
                pending_increment.zero_()
                pending_delta_isaac_50ms.zero_()
                pending_delta_isaac_50ms_raw.zero_()
                pending_delta_simscape_50ms.zero_()
                pending_kinematic.zero_()
                pending_correction.zero_()
                five_sum.zero_()
                rom_update_index = -1
                assert_initialized_from_measured(
                    target,
                    robot.data.joint_pos[:, actuator_ids_all].detach().clone(),
                    tolerance=1.0e-6,
                    label="post-settle",
                )
                reset_extension, _, _, _ = coordinate_state(mapping, target, torch.zeros_like(target))
                reset_min, reset_max, reset_margin, reset_outside, reset_excess = extension_range_audit(reset_extension, ranges)
                for i, name in enumerate(ACTUATOR_JOINT_NAMES):
                    print(
                        f"Reset range audit {name}: mapped_extension={float(reset_extension[0, i]):.12g} "
                        f"training_min={float(reset_min[i]):.12g} training_max={float(reset_max[i]):.12g} "
                        f"range_margin={float(reset_margin[i]):.12g} outside_training_range={bool(reset_outside[i])}",
                        flush=True,
                    )
                if np.nanmax(reset_excess) > float(args_cli.max_mapping_range_excess):
                    raise RuntimeError("FAIL: reset mapped Extension is severely outside the causal training range.")
                for control_step in range(args_cli.steps_per_test):
                    q_isaac = robot.data.joint_pos[:, actuator_ids_all].detach().clone()
                    qdot_isaac = robot.data.joint_vel[:, actuator_ids_all].detach().clone()
                    passive_target = passive_hold_target(
                        args_cli.passive_hold_mode,
                        fixed_hold=passive_hold,
                        measured_q=q_isaac,
                        active_mask=active_mask,
                    )
                    target = merge_active_and_passive_targets(target, passive_target, active_mask)
                    target_before = target.clone()
                    current_pos, current_vel, inverse_error, mapping_range_flags = coordinate_state(mapping, q_isaac, qdot_isaac)
                    valve_cmd = torch.zeros((1, 4), dtype=torch.float32, device=device)
                    valve_cmd[:, actuator_index] = float(command_value)
                    dP = dp_for_step(
                        env,
                        explicit_dp.to(device),
                        args_cli.diagnostic_dp_source,
                        device,
                        training_mean_dp=training_mean_dp,
                    )
                    dP_normalized = (dP * 1.0e-6 - dp_feature_mean) / torch.clamp(dp_feature_std, min=1.0e-12)
                    dP_linear_contribution = dP_normalized * dp_linear_weights
                    joint_pos5 = robot.data.joint_pos[:, state_joint_ids].detach().clone().to(device=device, dtype=torch.float32)
                    joint_vel5 = robot.data.joint_vel[:, state_joint_ids].detach().clone().to(device=device, dtype=torch.float32)
                    x = feature_row(
                        valve_cmd=valve_cmd,
                        dP=dP,
                        extension=current_pos,
                        velocity=current_vel,
                        joint_pos5=joint_pos5,
                        joint_vel5=joint_vel5,
                    )
                    if args_cli.dump_first_feature_row and not first_feature_dumped:
                        feature_values = [float(value) for value in x[0, 0].detach().cpu().tolist()]
                        print("First ROM feature row:", flush=True)
                        for feature_name, feature_value in zip(ROM_FEATURE_ORDER, feature_values, strict=True):
                            print(f"  {feature_name}: {feature_value:.12g}", flush=True)
                        first_feature_dumped = True
                    substep = control_step % 5
                    rom_update_flag = substep == 0
                    if rom_update_flag:
                        if control_step > 0:
                            error = torch.max(torch.abs(five_sum - pending_delta_isaac_50ms)).item()
                            if error > 1.0e-6:
                                raise RuntimeError(f"FAIL: previous five-substep sum error {error}")
                        rom_update_index += 1
                        five_sum.zero_()
                        kin_delta_50ms = current_vel * 0.05
                        if linear is not None:
                            out = linear.predict(x)
                            correction_50ms = out["delta_extension_correction"].to(device)
                            final_50ms = out["delta_extension_final"].to(device)
                        else:
                            correction_50ms = torch.zeros_like(kin_delta_50ms)
                            final_50ms = kin_delta_50ms
                        if torch.max(torch.abs(final_50ms)).item() > max_abs_rom_delta_50ms:
                            print("WARN: ROM DeltaExtension 50 ms exceeds --max_abs_rom_delta_50ms.", flush=True)
                        pending_kinematic = kin_delta_50ms
                        pending_correction = correction_50ms
                        pending_delta_simscape_50ms = final_50ms
                        pending_delta_isaac_50ms_raw = delta_simscape_to_isaac(mapping, final_50ms)
                        pending_delta_isaac_50ms = apply_active_mask(pending_delta_isaac_50ms_raw, active_mask)
                        pending_increment = pending_delta_isaac_50ms / 5.0
                        if torch.max(torch.abs(pending_increment)).item() > float(args_cli.max_abs_applied_delta_10ms):
                            print("WARN: applied Isaac delta 10 ms exceeds --max_abs_applied_delta_10ms.", flush=True)
                    five_sum = five_sum + pending_increment
                    unclamped_target = target + pending_increment
                    lower_row = lower.reshape(1, 4)
                    upper_row = upper.reshape(1, 4)
                    tolerance = float(args_cli.physical_limit_tolerance)
                    target = torch.clamp(unclamped_target, lower_row, upper_row)
                    limit_audit = compute_limit_audit(
                        measured_q=q_isaac,
                        target_preclamp=unclamped_target,
                        target_postclamp=target,
                        lower=lower,
                        upper=upper,
                        tolerance=tolerance,
                        active_mask=active_mask,
                    )
                    passive_drift = assert_passive_targets_held(
                        target,
                        passive_target,
                        active_mask,
                        tolerance=float(args_cli.passive_target_tolerance),
                    )
                    physical_low = limit_audit.measured_low_amount > tolerance
                    physical_high = limit_audit.measured_high_amount > tolerance
                    physical_violation = limit_audit.tested_critical or limit_audit.passive_true_violation
                    target_clamp_warning = limit_audit.target_clamp_warning
                    cumulative_target_change = torch.max(torch.abs(target - initial_pos)).item()
                    if cumulative_target_change > float(args_cli.max_abs_cumulative_target_change):
                        print("WARN: cumulative target movement exceeds --max_abs_cumulative_target_change.", flush=True)
                    if physical_violation and args_cli.stop_on_limit_violation:
                        raise RuntimeError("FAIL: physical prismatic limit violation or target clamp.")
                    finite = bool(torch.isfinite(target).all() and torch.isfinite(pending_delta_isaac_50ms).all() and torch.isfinite(x).all())
                    if not finite:
                        raise RuntimeError("FAIL: NaN/Inf in ROM runtime path.")
                    training_bad, training_feature, training_amount = range_violation(x, ranges)
                    training_min, training_max, range_margin, outside_range, range_excess = extension_range_audit(current_pos, ranges)
                    if np.nanmax(range_excess) > float(args_cli.max_mapping_range_excess):
                        raise RuntimeError("FAIL: mapped Extension is severely outside the causal training range before target application.")
                    before_write = target_before.clone()
                    robot.set_joint_position_target(target, joint_ids=actuator_ids_all)
                    target_written = target.clone()
                    target_buffer_after_write, target_buffer_source = joint_position_target_buffer(robot, actuator_ids_all, target)
                    overwrite_error = torch.max(torch.abs(target_buffer_after_write - target)).item()
                    if overwrite_error > 1.0e-9:
                        raise RuntimeError("FAIL: ROM target write/readback mismatch before environment step.")
                    env.step(torch.zeros((1, 4), dtype=torch.float32, device=device))
                    after_apply_pos = robot.data.joint_pos[:, actuator_ids_all].detach().clone()
                    measured_error = target - after_apply_pos
                    mech_pos = robot.data.joint_pos[:, mechanical_ids].detach().clone() if mechanical_ids else torch.empty((1, 0), device=device)
                    mech_vel = robot.data.joint_vel[:, mechanical_ids].detach().clone() if mechanical_ids else torch.empty((1, 0), device=device)
                    if args_cli.print_every > 0 and control_step % args_cli.print_every == 0:
                        print(f"{args_cli.model_mode}:{actuator_name}:{command_label}: step {control_step}", flush=True)
                    row: dict[str, Any] = {
                        "test_name": f"{actuator_name}_{command_label}",
                        "model_mode": args_cli.model_mode,
                        "actuator_name": actuator_name,
                        "actuator_id": actuator_ids_all[actuator_index],
                        "command_sign": 0 if command_value == 0 else (1 if command_value > 0 else -1),
                        "command_magnitude": abs(float(command_value)),
                        "control_step": control_step,
                        "phase": "active",
                        "simulation_time": control_step * 0.01,
                        "rom_update_index": rom_update_index,
                        "rom_update_flag": int(rom_update_flag),
                        "substep_index": substep,
                        "five_substep_sum_error": float(torch.max(torch.abs(five_sum - pending_increment * (substep + 1))).cpu()),
                        "substep_sum_error": float(torch.max(torch.abs(five_sum - pending_delta_isaac_50ms)).cpu()) if substep == 4 else 0.0,
                        "finite_guard_flag": int(finite),
                        "training_range_violation": int(training_bad),
                        "training_range_violation_feature": training_feature,
                        "training_range_violation_amount": training_amount,
                        "physical_limit_violation": int(physical_violation),
                        "physical_limit_joint": actuator_name if limit_audit.tested_critical else ("passive_actuator" if limit_audit.passive_true_violation else ""),
                        "physical_limit_amount": float(
                            torch.max(
                                torch.maximum(
                                    torch.maximum(limit_audit.measured_low_amount, limit_audit.measured_high_amount),
                                    torch.maximum(limit_audit.target_post_low_amount, limit_audit.target_post_high_amount),
                                )
                            ).cpu()
                        ),
                        "physical_limit_tolerance": tolerance,
                        "measured_physical_limit_violation": int(torch.any(limit_audit.measured_violation).item()),
                        "target_preclamp_limit_violation": int(torch.any(limit_audit.target_pre_violation).item()),
                        "target_postclamp_limit_violation": int(torch.any(limit_audit.target_post_violation).item()),
                        "unrelated_joint_limit_violation": int(limit_audit.passive_true_violation),
                        "limit_contact_within_tolerance": int(
                            torch.any(
                                limit_audit.measured_contact
                                | limit_audit.target_pre_contact
                                | limit_audit.target_post_contact
                            ).item()
                        ),
                        "target_clamp_warning": int(target_clamp_warning),
                        "target_clamp": int(torch.any(limit_audit.clamped).item()),
                        "max_abs_rom_delta_50ms": max_abs_rom_delta_50ms,
                        "max_abs_applied_delta_10ms": float(args_cli.max_abs_applied_delta_10ms),
                        "max_abs_cumulative_target_change": float(args_cli.max_abs_cumulative_target_change),
                        "cumulative_target_change": cumulative_target_change,
                        "passive_hold_mode": args_cli.passive_hold_mode,
                        "diagnostic_dp_source": args_cli.diagnostic_dp_source,
                        "done": 0,
                        "reset": 0,
                        "nan_inf": int(not finite),
                        "target_overwrite_error": overwrite_error,
                        "target_buffer_source": target_buffer_source,
                        "coordinate_mapping_mode": args_cli.coordinate_mapping_mode,
                        "mapping_mode": args_cli.coordinate_mapping_mode,
                        "mapping_validated": int(bool(mapping.mapping_validated) if mapping is not None else False),
                        "provisional_mapping_allowed": int(bool(args_cli.allow_provisional_nominal_reference)),
                    }
                    mapping_log = mapping_metadata_for_logging(mapping, q_isaac)
                    row["reference_confidence"] = mapping_log["nominal_reference_confidence"]
                    row["reference_sample_count"] = mapping_log["reference_sample_count"]
                    row["reference_pose_match_error"] = mapping_log["reference_pose_match_error"]
                    for i, name in enumerate(ACTUATOR_JOINT_NAMES):
                        row[f"ValveCmd_{i+1}"] = float(valve_cmd[0, i].cpu())
                        row[f"dP_{i+1}"] = float(dP[0, i].cpu())
                        row[f"dP_normalized_{i+1}"] = float(dP_normalized[0, i].cpu())
                        row[f"dP_linear_raw_contribution_{i+1}"] = float(dP_linear_contribution[0, i].cpu())
                        row[f"Extension_{i+1}"] = float(current_pos[0, i].cpu())
                        row[f"ExtensionVelocity_{i+1}"] = float(current_vel[0, i].cpu())
                        row[f"kinematic_delta_50ms_{i+1}"] = float(pending_kinematic[0, i].cpu())
                        row[f"linear_correction_50ms_{i+1}"] = float(pending_correction[0, i].cpu())
                        row[f"final_delta_50ms_{i+1}"] = float(pending_delta_simscape_50ms[0, i].cpu())
                        row[f"predicted_delta_extension_50ms_{i+1}"] = float(pending_delta_simscape_50ms[0, i].cpu())
                        row[f"applied_delta_10ms_{i+1}"] = float(pending_increment[0, i].cpu())
                        row[f"five_substep_sum_{i+1}"] = float(five_sum[0, i].cpu())
                        row[f"cylinder_target_before_{i+1}"] = float(before_write[0, i].cpu())
                        row[f"cylinder_target_after_{i+1}"] = float(target[0, i].cpu())
                        row[f"target_written_by_rom_{i+1}"] = float(target_written[0, i].cpu())
                        row[f"target_after_environment_apply_{i+1}"] = float(target[0, i].cpu())
                        row[f"target_buffer_after_write_{i+1}"] = float(target_buffer_after_write[0, i].cpu())
                        row[f"target_clamped_{i+1}"] = int(limit_audit.clamped[0, i].item())
                        row[f"lower_limit_{i+1}"] = float(lower[i].cpu())
                        row[f"upper_limit_{i+1}"] = float(upper[i].cpu())
                        row[f"target_tracking_error_{i+1}"] = float(measured_error[0, i].cpu())
                        row[f"cylinder_position_{i+1}"] = float(current_pos[0, i].cpu())
                        row[f"cylinder_velocity_{i+1}"] = float(current_vel[0, i].cpu())
                        row[f"cylinder_position_delta_{i+1}"] = float((current_pos - previous_pos)[0, i].cpu())
                        row[f"actual_prismatic_lower_limit_violation_{i+1}"] = int(physical_low[0, i].item())
                        row[f"actual_prismatic_upper_limit_violation_{i+1}"] = int(physical_high[0, i].item())
                        row[f"measured_physical_limit_violation_{i+1}"] = int(limit_audit.measured_violation[0, i].item())
                        row[f"target_preclamp_limit_violation_{i+1}"] = int(limit_audit.target_pre_violation[0, i].item())
                        row[f"target_postclamp_limit_violation_{i+1}"] = int(limit_audit.target_post_violation[0, i].item())
                        row[f"limit_contact_within_tolerance_{i+1}"] = int(
                            (
                                limit_audit.measured_contact
                                | limit_audit.target_pre_contact
                                | limit_audit.target_post_contact
                            )[0, i].item()
                        )
                        row[f"measured_below_lower_amount_{i+1}"] = float(limit_audit.measured_low_amount[0, i].cpu())
                        row[f"measured_above_upper_amount_{i+1}"] = float(limit_audit.measured_high_amount[0, i].cpu())
                        row[f"target_preclamp_below_lower_amount_{i+1}"] = float(limit_audit.target_pre_low_amount[0, i].cpu())
                        row[f"target_preclamp_above_upper_amount_{i+1}"] = float(limit_audit.target_pre_high_amount[0, i].cpu())
                        row[f"target_postclamp_below_lower_amount_{i+1}"] = float(limit_audit.target_post_low_amount[0, i].cpu())
                        row[f"target_postclamp_above_upper_amount_{i+1}"] = float(limit_audit.target_post_high_amount[0, i].cpu())
                        row[f"isaac_position_{i+1}"] = float(q_isaac[0, i].cpu())
                        row[f"isaac_velocity_{i+1}"] = float(qdot_isaac[0, i].cpu())
                        row[f"isaac_target_before_{i+1}"] = float(before_write[0, i].cpu())
                        row[f"isaac_target_after_{i+1}"] = float(target[0, i].cpu())
                        row[f"isaac_delta_50ms_{i+1}"] = float(pending_delta_isaac_50ms[0, i].cpu())
                        row[f"isaac_delta_10ms_{i+1}"] = float(pending_increment[0, i].cpu())
                        row[f"predicted_isaac_delta_50ms_{i+1}"] = float(pending_delta_isaac_50ms_raw[0, i].cpu())
                        row[f"active_actuator_mask_{i+1}"] = float(active_mask[0, i].cpu())
                        row[f"rom_prediction_applied_{i+1}"] = int(bool(active_mask_bool[0, i].item()))
                        row[f"passive_hold_target_{i+1}"] = float(passive_target[0, i].cpu())
                        row[f"passive_target_drift_{i+1}"] = float(passive_drift[0, i].cpu())
                        row[f"passive_target_overwrite_{i+1}"] = float(
                            abs(target_buffer_after_write[0, i] - target[0, i]).cpu()
                            if not bool(active_mask_bool[0, i].item())
                            else 0.0
                        )
                        row[f"simscape_extension_{i+1}"] = float(current_pos[0, i].cpu())
                        row[f"simscape_extension_velocity_{i+1}"] = float(current_vel[0, i].cpu())
                        row[f"rom_delta_extension_50ms_{i+1}"] = float(pending_delta_simscape_50ms[0, i].cpu())
                        row[f"mapped_simscape_extension_{i+1}"] = float(current_pos[0, i].cpu())
                        row[f"mapped_simscape_velocity_{i+1}"] = float(current_vel[0, i].cpu())
                        row[f"rom_delta_simscape_50ms_{i+1}"] = float(pending_delta_simscape_50ms[0, i].cpu())
                        row[f"mapped_isaac_delta_50ms_{i+1}"] = float(pending_delta_isaac_50ms_raw[0, i].cpu())
                        row[f"mapping_scale_{i+1}"] = float(mapping_log["scale"][i])
                        row[f"mapping_offset_{i+1}"] = float(mapping_log["offset"][i])
                        row[f"mapping_sign_{i+1}"] = float(mapping_log["sign"][i])
                        row[f"mapping_x_nominal_{i+1}"] = float(mapping_log["simscape_nominal_extension"][i])
                        row[f"mapping_q_nominal_{i+1}"] = float(mapping_log["isaac_nominal_position"][i])
                        row[f"simscape_nominal_extension_{i+1}"] = float(mapping_log["simscape_nominal_extension"][i])
                        row[f"isaac_nominal_position_{i+1}"] = float(mapping_log["isaac_nominal_position"][i])
                        row[f"mapping_sign_verified_{i+1}"] = int(bool(mapping_log["sign_verified"][i]))
                        row[f"mapping_direction_verified_{i+1}"] = int(bool(mapping_log["sign_verified"][i]))
                        row[f"inverse_mapping_error_{i+1}"] = float(inverse_error[0, i].cpu())
                        row[f"round_trip_error_{i+1}"] = float(inverse_error[0, i].cpu())
                        row[f"nominal_reconstruction_error_{i+1}"] = float(nominal_error[0, i].cpu())
                        row[f"mapping_round_trip_error_{i+1}"] = float(nominal_round_trip_error[0, i].cpu())
                        row[f"training_min_{i+1}"] = float(training_min[i])
                        row[f"training_max_{i+1}"] = float(training_max[i])
                        row[f"range_margin_{i+1}"] = float(range_margin[i])
                        row[f"outside_training_range_{i+1}"] = int(bool(outside_range[i]))
                        row[f"mapping_range_excess_{i+1}"] = float(range_excess[i])
                        row[f"mapping_range_violation_{i+1}"] = int(bool(outside_range[i]) or mapping_range_flags[0, i].item())
                    for j in range(5):
                        row[f"JointPos_{j+1}"] = float(joint_pos5[0, j].cpu())
                        row[f"JointVel_{j+1}"] = float(joint_vel5[0, j].cpu())
                    for mech_i, mech_name in enumerate(mechanical_names):
                        row[f"mech_{mech_name}_position"] = float(mech_pos[0, mech_i].cpu())
                        row[f"mech_{mech_name}_velocity"] = float(mech_vel[0, mech_i].cpu())
                        row[f"mech_{mech_name}_delta"] = 0.0
                    rows.append(row)
                    previous_pos = current_pos.clone()
                if args_cli.steps_per_test % 5 == 0:
                    error = torch.max(torch.abs(five_sum - pending_delta_isaac_50ms)).item()
                    if error > 1.0e-6:
                        raise RuntimeError(f"FAIL: final five-substep sum error {error}")
        if not rows:
            raise RuntimeError("FAIL: no diagnostic rows were collected.")
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
            writer.writeheader()
            writer.writerows(rows)
        print(f"PASS: wrote diagnostic CSV {csv_path}", flush=True)
        print("WARN: this manual run is controlled runtime diagnostics only, not full physical validation.", flush=True)
        env.close()

    run()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
