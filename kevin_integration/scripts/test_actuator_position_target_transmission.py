from __future__ import annotations

import argparse
import csv
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

DEFAULT_ACTUATOR_JOINT_NAMES = [
    "Drehzapfen_joint",
    "Ausleger_I_joint",
    "Ausleger_II_joint",
    "Messerkopf_Schwenk_joint",
]
DEFAULT_MECHANICAL_JOINT_NAMES = [
    "Drehzapfen_constraint_joint_1",
    "Drehzapfen_constraint_joint_3",
    "Ausleger_I_constraint_joint_1",
    "Ausleger_I_constraint_joint_3",
    "Ausleger_II_constraint_joint_1",
    "Ausleger_II_constraint_joint_3",
    "Messerkopf_Schwenk_constraint_joint_1",
    "Messerkopf_Schwenk_constraint_joint_3",
    "Messerkopf_Schwenk_constraint_joint_5",
    "Messerkopf_Schwenk_constraint_joint_6",
]
DEFAULT_BODY_NAMES = [
    "Drehzapfen",
    "Frontrahmen",
    "Ausleger_I",
    "Ausleger_II",
    "Messerkopf",
    "Messerkopf_Schwenk_base",
    "Ausleger_Schwenkzylinder_Rohr",
    "Ausleger_Schwenkzylinder_Kolben",
    "Hebezylinder_Rohr",
    "Hebezylinder_Kolben",
    "Knickzylinder_Rohr",
    "Knickzylinder_Kolben",
    "Messerkopf_Schwenkzylinder_Rohr",
    "Messerkopf_Schwenkzylinder_Kolben",
]
BODY_AUTO_INCLUDE_KEYWORDS = (
    "Drehzapfen",
    "Ausleger",
    "Messerkopf",
    "Zylinder",
    "Kolben",
    "Rohr",
)

parser = argparse.ArgumentParser(description="Position-target actuator-to-arm transmission diagnostic for Kevin Mulag.")
parser.add_argument("--task", type=str, default="Kevin-Mulag-AdaptiveRL-JointReach-v0", help="Gym task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of Isaac Lab environments.")
parser.add_argument("--steps_per_test", type=int, default=200, help="Steps per position-target test.")
parser.add_argument("--settle_steps", type=int, default=500, help="Settle steps using hold/current actuator targets.")
parser.add_argument(
    "--actuator_joint_names",
    type=str,
    default=",".join(DEFAULT_ACTUATOR_JOINT_NAMES),
    help="Comma-separated actuator-side prismatic joint names to test.",
)
parser.add_argument("--single_actuator_joint", type=str, default=None, help="Optional single actuator joint to test.")
parser.add_argument(
    "--position_offsets",
    type=str,
    default="0.005,0.01,0.02",
    help='Comma-separated actuator target offsets, e.g. "0.005,0.01,0.02".',
)
parser.add_argument("--reference_body", type=str, default="Frontrahmen", help="Reference body for summary interpretation.")
parser.add_argument(
    "--hold_other_actuator_targets",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Hold non-tested actuator targets at their settled positions.",
)
parser.add_argument(
    "--mechanical_joint_names",
    type=str,
    default=None,
    help="Optional comma-separated mechanical/revolute candidate joint names to log.",
)
parser.add_argument(
    "--body_names",
    type=str,
    default=None,
    help="Optional comma-separated body/link names to log. Keyword-matched bodies are still auto-included.",
)
parser.add_argument(
    "--log_root",
    type=str,
    default="logs/actuator_position_target_transmission",
    help="Directory under which timestamped position-target logs are written.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import kevin_integration.tasks  # noqa: F401
from kevin_integration.utils.sim_memory import apply_kevin_sim_memory_optimizations


def _parse_csv(text: str | None, default: Sequence[str] | None = None) -> list[str]:
    if text is None:
        return list(default or [])
    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("Comma-separated argument must contain at least one value.")
    return values


def _parse_float_csv(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("Comma-separated numeric argument must contain at least one value.")
    return values


def _safe_name(name: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in name).strip("_")


def _robot_joint_names(robot) -> list[str]:
    names = getattr(robot, "joint_names", None)
    if names is None:
        names = getattr(robot.data, "joint_names", None)
    return list(names) if names is not None else []


def _robot_body_names(robot) -> list[str]:
    names = getattr(robot, "body_names", None)
    if names is None:
        names = getattr(robot.data, "body_names", None)
    return list(names) if names is not None else []


def _name_from_index(names: Sequence[str], index: int, fallback_prefix: str) -> str:
    if 0 <= index < len(names):
        return names[index]
    return f"{fallback_prefix}_{index}"


def _resolve_required_joint_ids(robot, joint_names: Sequence[str], *, label: str) -> list[int]:
    ids: list[int] = []
    all_joint_names = _robot_joint_names(robot)
    for joint_name in joint_names:
        joint_ids, resolved_names = robot.find_joints(joint_name)
        if len(joint_ids) == 0:
            raise RuntimeError(f"Could not resolve {label} joint name {joint_name!r}.")
        joint_id = int(joint_ids[0])
        actual = resolved_names[0] if resolved_names else _name_from_index(all_joint_names, joint_id, "joint")
        ids.append(joint_id)
        print(f"[INFO] Resolved {label}: requested={joint_name} id={joint_id} actual={actual}", flush=True)
    return ids


def _resolve_optional_joint_ids(robot, joint_names: Sequence[str], *, label: str) -> tuple[list[str], list[int]]:
    names_out: list[str] = []
    ids_out: list[int] = []
    all_joint_names = _robot_joint_names(robot)
    for joint_name in joint_names:
        try:
            joint_ids, resolved_names = robot.find_joints(joint_name)
        except Exception as exc:
            print(f"[WARN] Could not query {label} joint {joint_name!r}: {exc}; skipping it.", flush=True)
            continue
        if len(joint_ids) == 0:
            print(f"[WARN] {label.capitalize()} joint not found: {joint_name}", flush=True)
            continue
        joint_id = int(joint_ids[0])
        actual = resolved_names[0] if resolved_names else _name_from_index(all_joint_names, joint_id, "joint")
        names_out.append(joint_name)
        ids_out.append(joint_id)
        print(f"[INFO] Resolved {label}: requested={joint_name} id={joint_id} actual={actual}", flush=True)
    return names_out, ids_out


def _resolve_body_ids(robot, requested_body_names: Sequence[str]) -> tuple[list[str], list[int]]:
    all_body_names = _robot_body_names(robot)
    if all_body_names:
        print("[INFO] Available robot body names:", flush=True)
        for index, body_name in enumerate(all_body_names):
            print(f"  [{index}] {body_name}", flush=True)
    else:
        print("[WARN] robot body names are unavailable; body pose logging may be disabled.", flush=True)

    resolved: dict[str, int] = {}

    def _add_body(name: str, body_id: int) -> None:
        if name not in resolved:
            resolved[name] = int(body_id)

    for body_name in requested_body_names:
        if body_name in all_body_names:
            _add_body(body_name, all_body_names.index(body_name))
            continue
        try:
            body_ids, resolved_names = robot.find_bodies(body_name)
        except Exception as exc:
            print(f"[WARN] Could not query body {body_name!r}: {exc}; keyword auto-include may still catch it.", flush=True)
            continue
        if len(body_ids) == 0:
            print(f"[WARN] Body not found by exact/resolver name: {body_name}", flush=True)
            continue
        for body_id, resolved_name in zip(body_ids, resolved_names, strict=False):
            actual_name = resolved_name if resolved_name else _name_from_index(all_body_names, int(body_id), "body")
            _add_body(actual_name, int(body_id))

    for body_id, body_name in enumerate(all_body_names):
        if any(keyword.lower() in body_name.lower() for keyword in BODY_AUTO_INCLUDE_KEYWORDS):
            _add_body(body_name, body_id)

    if resolved:
        print("[INFO] Resolved body/link IDs for pose logging:", flush=True)
        for body_name, body_id in resolved.items():
            print(f"  body={body_name} id={body_id}", flush=True)
    else:
        print("[WARN] No body/link IDs resolved; CSV will contain joint data only.", flush=True)
    return list(resolved.keys()), list(resolved.values())


def _field(prefix: str, name: str, suffix: str) -> str:
    return f"{prefix}_{_safe_name(name)}_{suffix}"


def _joint_values(robot, joint_ids: Sequence[int], field_name: str) -> list[float]:
    value = getattr(robot.data, field_name)
    tensor = value.detach()
    if tensor.ndim >= 2:
        tensor = tensor[0]
    return [float(tensor[joint_id].cpu()) for joint_id in joint_ids]


def _soft_limits(robot, joint_ids: Sequence[int]) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    limits = getattr(robot.data, "soft_joint_pos_limits", None)
    if not isinstance(limits, torch.Tensor):
        return None, None
    lower = limits[:, joint_ids, 0]
    upper = limits[:, joint_ids, 1]
    return lower, upper


def _body_position_tensor(robot) -> torch.Tensor | None:
    for field_name in ("body_pos_w", "body_link_pos_w"):
        value = getattr(robot.data, field_name, None)
        if isinstance(value, torch.Tensor):
            return value
    value = getattr(robot.data, "body_state_w", None)
    if isinstance(value, torch.Tensor) and value.shape[-1] >= 3:
        return value[..., 0:3]
    return None


def _body_quat_tensor(robot) -> torch.Tensor | None:
    for field_name in ("body_quat_w", "body_link_quat_w"):
        value = getattr(robot.data, field_name, None)
        if isinstance(value, torch.Tensor):
            return value
    value = getattr(robot.data, "body_state_w", None)
    if isinstance(value, torch.Tensor) and value.shape[-1] >= 7:
        return value[..., 3:7]
    return None


def _body_vectors(robot, body_ids: Sequence[int]) -> list[tuple[float, float, float]]:
    tensor = _body_position_tensor(robot)
    if tensor is None:
        return [(math.nan, math.nan, math.nan) for _ in body_ids]
    tensor = tensor.detach()
    if tensor.ndim >= 3:
        tensor = tensor[0]
    vectors = []
    for body_id in body_ids:
        if 0 <= body_id < tensor.shape[0]:
            values = tensor[body_id].detach().cpu().flatten().tolist()
            vectors.append((float(values[0]), float(values[1]), float(values[2])))
        else:
            vectors.append((math.nan, math.nan, math.nan))
    return vectors


def _body_quats(robot, body_ids: Sequence[int]) -> list[tuple[float, float, float, float]]:
    tensor = _body_quat_tensor(robot)
    if tensor is None:
        return [(math.nan, math.nan, math.nan, math.nan) for _ in body_ids]
    tensor = tensor.detach()
    if tensor.ndim >= 3:
        tensor = tensor[0]
    quats = []
    for body_id in body_ids:
        if 0 <= body_id < tensor.shape[0]:
            values = tensor[body_id].detach().cpu().flatten().tolist()
            quats.append((float(values[0]), float(values[1]), float(values[2]), float(values[3])))
        else:
            quats.append((math.nan, math.nan, math.nan, math.nan))
    return quats


def _delta_norm(delta: tuple[float, float, float]) -> float:
    if not all(math.isfinite(value) for value in delta):
        return math.nan
    return math.sqrt(sum(value * value for value in delta))


def _mean_tensor(value: Any) -> float:
    if not isinstance(value, torch.Tensor) or value.numel() == 0:
        return math.nan
    return float(value.detach().float().mean().cpu())


def _runtime_flags(unwrapped, dones: torch.Tensor) -> dict[str, float]:
    return {
        "done_frac": _mean_tensor(dones.float()),
        "reset_buf_mean": _mean_tensor(getattr(unwrapped, "reset_buf", None)),
        "time_out_frac": _mean_tensor(getattr(unwrapped, "time_out_buf", None)),
        "terminated_frac": _mean_tensor(getattr(unwrapped, "terminated_buf", None)),
    }


def _step_is_safe(unwrapped, dones: torch.Tensor) -> tuple[bool, str]:
    if isinstance(dones, torch.Tensor) and torch.any(dones):
        return False, "done/reset flag became true"
    for field_name in ("joint_pos", "joint_vel"):
        value = getattr(unwrapped.robot.data, field_name, None)
        if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
            return False, f"non-finite robot.data.{field_name}"
    body_pos = _body_position_tensor(unwrapped.robot)
    if isinstance(body_pos, torch.Tensor) and not torch.isfinite(body_pos).all():
        return False, "non-finite body position"
    return True, ""


def _expected_mechanical_keywords(actuator_name: str) -> tuple[str, ...]:
    if actuator_name == "Drehzapfen_joint":
        return ("Drehzapfen",)
    return (actuator_name.replace("_joint", ""),)


def _is_expected_family(name: str, keywords: tuple[str, ...]) -> bool:
    lowered = name.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def _max_relative_delta(
    test_deltas: dict[str, float],
    zero_deltas: dict[str, float],
    keywords: tuple[str, ...],
    *,
    invert: bool = False,
) -> tuple[str, float, float]:
    best_name = "none"
    best_relative = 0.0
    best_raw = 0.0
    for name, value in test_deltas.items():
        matches = _is_expected_family(name, keywords)
        if invert:
            matches = not matches
        if not matches or not math.isfinite(value):
            continue
        relative = value - zero_deltas.get(name, 0.0)
        if abs(relative) > abs(best_relative):
            best_name = name
            best_relative = relative
            best_raw = value
    return best_name, best_relative, best_raw


def _separated(value: float, zero_value: float, relative_value: float) -> bool:
    motion_threshold = 1.0e-5
    min_delta_from_zero = 1.0e-4
    drift_margin_ratio = 3.0
    if not all(math.isfinite(item) for item in (value, zero_value, relative_value)):
        return False
    if abs(relative_value) < max(motion_threshold, min_delta_from_zero):
        return False
    return abs(value) >= drift_margin_ratio * max(abs(zero_value), motion_threshold)


def _install_position_target_override(unwrapped, actuator_joint_ids: Sequence[int]) -> None:
    if not hasattr(unwrapped.robot, "set_joint_position_target"):
        raise RuntimeError("This IsaacLab Articulation does not expose set_joint_position_target.")
    unwrapped._direct_position_target_joint_ids = list(actuator_joint_ids)  # noqa: SLF001
    unwrapped._direct_position_targets = torch.zeros(  # noqa: SLF001
        (unwrapped.num_envs, len(actuator_joint_ids)), dtype=torch.float32, device=unwrapped.device
    )
    unwrapped._direct_position_pre_physics_calls = 0  # noqa: SLF001
    unwrapped._direct_position_apply_action_calls = 0  # noqa: SLF001

    def _direct_pre_physics_step(self, actions: torch.Tensor) -> None:
        self._direct_position_pre_physics_calls += 1
        if hasattr(self, "_update_q_ref_commanded"):
            self._update_q_ref_commanded()
        if hasattr(self, "_prev_action") and hasattr(self, "_current_action"):
            previous_action = self._prev_action.detach().clone()
            self._current_action = torch.zeros_like(previous_action)
            self._action_delta = self._current_action - previous_action
        for attr_name in (
            "_current_qddot_cmd",
            "_prev_valve_cmd",
            "_current_valve_cmd",
            "_current_fd_simscape_output",
            "_current_tau_fd_adapted",
            "_current_tau_applied_raw",
            "_current_tau_isaac_filtered",
            "_current_tau_isaac_clamped",
            "_torque_clamp_fraction",
            "_torque_adapter_clamp_fraction",
            "_torque_rate_limited_fraction",
            "_dP",
            "_fnet",
        ):
            value = getattr(self, attr_name, None)
            if isinstance(value, torch.Tensor):
                value.zero_()
        if hasattr(self, "_zero_sd_observer_state"):
            self._zero_sd_observer_state()
        if hasattr(self, "_zero_sd_feedback_state"):
            self._zero_sd_feedback_state()
        if hasattr(self, "_current_torque") and isinstance(self._current_torque, torch.Tensor):
            self._current_torque.zero_()
        _ = actions

    def _direct_apply_action(self) -> None:
        self._direct_position_apply_action_calls += 1
        self.robot.set_joint_position_target(self._direct_position_targets, joint_ids=self._direct_position_target_joint_ids)

    unwrapped._pre_physics_step = MethodType(_direct_pre_physics_step, unwrapped)  # noqa: SLF001
    unwrapped._apply_action = MethodType(_direct_apply_action, unwrapped)  # noqa: SLF001
    print(
        "[INFO] Kevin control path bypass active: _pre_physics_step/_apply_action overridden",
        flush=True,
    )


def _reset_env(env: RslRlVecEnvWrapper):
    reset_out = env.reset()
    if isinstance(reset_out, tuple):
        return reset_out[0]
    return reset_out


def _make_zero_action(env: RslRlVecEnvWrapper) -> torch.Tensor:
    return torch.zeros((env.num_envs, env.num_actions), dtype=torch.float32, device=env.device)


def _csv_fields(actuator_names: Sequence[str], mechanical_names: Sequence[str], body_names: Sequence[str]) -> list[str]:
    fields = [
        "test_name",
        "position_offset",
        "applied_actuator_name",
        "applied_actuator_id",
        "step",
        "target_position",
        "clamped_target_position",
        "done_frac",
    ]
    for name in actuator_names:
        fields.extend(
            [
                _field("actuator", name, "pos"),
                _field("actuator", name, "vel"),
                _field("actuator", name, "delta_from_start"),
                _field("actuator", name, "target"),
            ]
        )
    for name in mechanical_names:
        fields.extend(
            [
                _field("mech", name, "pos"),
                _field("mech", name, "vel"),
                _field("mech", name, "delta_from_start"),
            ]
        )
    for name in body_names:
        fields.extend(
            [
                _field("body", name, "x"),
                _field("body", name, "y"),
                _field("body", name, "z"),
                _field("body", name, "dx_from_start"),
                _field("body", name, "dy_from_start"),
                _field("body", name, "dz_from_start"),
                _field("body", name, "translation_delta_norm"),
                _field("body", name, "qw"),
                _field("body", name, "qx"),
                _field("body", name, "qy"),
                _field("body", name, "qz"),
            ]
        )
    fields.extend(["reset_buf_mean", "time_out_frac", "terminated_frac"])
    return fields


def _active_actuator(
    actuator_names: Sequence[str],
    actuator_ids: Sequence[int],
    targets: torch.Tensor,
    hold_targets: Sequence[float],
) -> tuple[str, int]:
    delta = targets[0].detach().cpu() - torch.tensor(hold_targets, dtype=torch.float32)
    active = [index for index, value in enumerate(delta.tolist()) if abs(float(value)) > 0.0]
    if not active:
        return "none", -1
    index = active[0]
    return actuator_names[index], actuator_ids[index]


def _collect_row(
    *,
    unwrapped,
    dones: torch.Tensor,
    step: int,
    test_name: str,
    position_offset: float,
    position_targets: torch.Tensor,
    requested_target: float,
    clamped_target: float,
    hold_targets: Sequence[float],
    actuator_names: Sequence[str],
    actuator_ids: Sequence[int],
    actuator_start: Sequence[float],
    mechanical_names: Sequence[str],
    mechanical_ids: Sequence[int],
    mechanical_start: Sequence[float],
    body_names: Sequence[str],
    body_ids: Sequence[int],
    body_start: Sequence[tuple[float, float, float]],
) -> dict[str, float | str]:
    actuator_pos = _joint_values(unwrapped.robot, actuator_ids, "joint_pos")
    actuator_vel = _joint_values(unwrapped.robot, actuator_ids, "joint_vel")
    mechanical_pos = _joint_values(unwrapped.robot, mechanical_ids, "joint_pos")
    mechanical_vel = _joint_values(unwrapped.robot, mechanical_ids, "joint_vel")
    body_pos = _body_vectors(unwrapped.robot, body_ids)
    body_quat = _body_quats(unwrapped.robot, body_ids)
    applied_name, applied_id = _active_actuator(actuator_names, actuator_ids, position_targets, hold_targets)
    row: dict[str, float | str] = {
        "test_name": test_name,
        "position_offset": float(position_offset),
        "applied_actuator_name": applied_name,
        "applied_actuator_id": float(applied_id),
        "step": float(step),
        "target_position": float(requested_target),
        "clamped_target_position": float(clamped_target),
    }
    for name, pos, vel, start, target in zip(
        actuator_names, actuator_pos, actuator_vel, actuator_start, position_targets[0].detach().cpu().tolist(), strict=True
    ):
        row[_field("actuator", name, "pos")] = pos
        row[_field("actuator", name, "vel")] = vel
        row[_field("actuator", name, "delta_from_start")] = pos - start
        row[_field("actuator", name, "target")] = float(target)
    for name, pos, vel, start in zip(mechanical_names, mechanical_pos, mechanical_vel, mechanical_start, strict=True):
        row[_field("mech", name, "pos")] = pos
        row[_field("mech", name, "vel")] = vel
        row[_field("mech", name, "delta_from_start")] = pos - start
    for name, pos, quat, start in zip(body_names, body_pos, body_quat, body_start, strict=True):
        dx = pos[0] - start[0]
        dy = pos[1] - start[1]
        dz = pos[2] - start[2]
        row[_field("body", name, "x")] = pos[0]
        row[_field("body", name, "y")] = pos[1]
        row[_field("body", name, "z")] = pos[2]
        row[_field("body", name, "dx_from_start")] = dx
        row[_field("body", name, "dy_from_start")] = dy
        row[_field("body", name, "dz_from_start")] = dz
        row[_field("body", name, "translation_delta_norm")] = _delta_norm((dx, dy, dz))
        row[_field("body", name, "qw")] = quat[0]
        row[_field("body", name, "qx")] = quat[1]
        row[_field("body", name, "qy")] = quat[2]
        row[_field("body", name, "qz")] = quat[3]
    row.update(_runtime_flags(unwrapped, dones))
    return row


def _max_abs_delta_name(names: Sequence[str], deltas: dict[str, float]) -> tuple[str, float]:
    if not names:
        return "none", 0.0
    name = max(names, key=lambda item: abs(deltas.get(item, 0.0)))
    return name, deltas.get(name, 0.0)


def _max_body_delta_name(names: Sequence[str], deltas: dict[str, float]) -> tuple[str, float]:
    finite_names = [name for name in names if math.isfinite(deltas.get(name, math.nan))]
    if not finite_names:
        return "none", math.nan
    name = max(finite_names, key=lambda item: abs(deltas.get(item, 0.0)))
    return name, deltas.get(name, math.nan)


def _body_delta_vectors_from_row(row: dict[str, float | str], body_names: Sequence[str]) -> dict[str, tuple[float, float, float]]:
    vectors = {}
    for name in body_names:
        vectors[name] = (
            float(row[_field("body", name, "dx_from_start")]),
            float(row[_field("body", name, "dy_from_start")]),
            float(row[_field("body", name, "dz_from_start")]),
        )
    return vectors


def _body_reference_name(body_names: Sequence[str], reference_body: str) -> str | None:
    if not reference_body:
        return None
    reference_safe = _safe_name(reference_body).lower()
    for name in body_names:
        if name == reference_body or _safe_name(name).lower() == reference_safe:
            return name
    return None


def _max_reference_body_delta_minus_zero(
    row: dict[str, Any],
    zero: dict[str, Any],
    body_names: Sequence[str],
    reference_body: str,
) -> tuple[str, float]:
    reference_name = _body_reference_name(body_names, reference_body)
    if reference_name is None:
        return "none", math.nan
    vectors = row.get("body_delta_vectors", {})
    zero_vectors = zero.get("body_delta_vectors", {})
    reference_vector = vectors.get(reference_name)
    zero_reference_vector = zero_vectors.get(reference_name)
    if reference_vector is None or zero_reference_vector is None:
        return "none", math.nan

    best_name = "none"
    best_value = math.nan
    for name in body_names:
        vector = vectors.get(name)
        zero_vector = zero_vectors.get(name)
        if vector is None or zero_vector is None:
            continue
        relative = (
            vector[0] - reference_vector[0],
            vector[1] - reference_vector[1],
            vector[2] - reference_vector[2],
        )
        zero_relative = (
            zero_vector[0] - zero_reference_vector[0],
            zero_vector[1] - zero_reference_vector[1],
            zero_vector[2] - zero_reference_vector[2],
        )
        delta_minus_zero = (
            relative[0] - zero_relative[0],
            relative[1] - zero_relative[1],
            relative[2] - zero_relative[2],
        )
        value = _delta_norm(delta_minus_zero)
        if math.isfinite(value) and (not math.isfinite(best_value) or value > best_value):
            best_name = name
            best_value = value
    return best_name, best_value


def _summarize(
    *,
    position_offset: float,
    test_name: str,
    rows: list[dict[str, float | str]],
    actuator_names: Sequence[str],
    actuator_ids: Sequence[int],
    mechanical_names: Sequence[str],
    body_names: Sequence[str],
) -> dict[str, Any]:
    if not rows:
        return {}
    last = rows[-1]
    applied_name = str(last.get("applied_actuator_name", "none"))
    applied_id = int(float(last.get("applied_actuator_id", -1.0)))
    actuator_deltas = {name: float(last[_field("actuator", name, "delta_from_start")]) for name in actuator_names}
    mechanical_deltas = {name: float(last[_field("mech", name, "delta_from_start")]) for name in mechanical_names}
    body_deltas = {name: float(last[_field("body", name, "translation_delta_norm")]) for name in body_names}
    body_delta_vectors = _body_delta_vectors_from_row(last, body_names)
    largest_mech_name, largest_mech_delta = _max_abs_delta_name(mechanical_names, mechanical_deltas)
    largest_body_name, largest_body_delta = _max_body_delta_name(body_names, body_deltas)
    if applied_name == "none":
        active_delta_name, actuator_delta = _max_abs_delta_name(actuator_names, actuator_deltas)
        applied_name = active_delta_name
        applied_id = actuator_ids[actuator_names.index(active_delta_name)] if active_delta_name in actuator_names else -1
    else:
        actuator_delta = actuator_deltas[applied_name]
    actuator_start = float(rows[0][_field("actuator", applied_name, "pos")]) - float(
        rows[0][_field("actuator", applied_name, "delta_from_start")]
    )
    return {
        "position_offset": position_offset,
        "test_name": test_name,
        "applied_actuator_name": applied_name,
        "applied_actuator_id": applied_id,
        "actuator_start": actuator_start,
        "actuator_target": float(last[_field("actuator", applied_name, "target")]),
        "actuator_end": float(last[_field("actuator", applied_name, "pos")]),
        "actuator_delta": actuator_delta,
        "actuator_deltas": actuator_deltas,
        "mechanical_deltas": mechanical_deltas,
        "body_deltas": body_deltas,
        "body_delta_vectors": body_delta_vectors,
        "expected_family_mech_delta": math.nan,
        "expected_family_mech_delta_minus_zero": math.nan,
        "actuator_delta_minus_zero": math.nan,
        "largest_mechanical_delta_joint": largest_mech_name,
        "largest_mechanical_delta": largest_mech_delta,
        "largest_moving_body": largest_body_name,
        "largest_body_translation_delta": largest_body_delta,
        "done_frac": max(float(row["done_frac"]) for row in rows),
    }


def _apply_zero_relative(summary_rows: Sequence[dict[str, Any]]) -> None:
    zero_by_offset = {float(row["position_offset"]): row for row in summary_rows if row["test_name"] == "zero"}
    for row in summary_rows:
        zero = zero_by_offset.get(float(row["position_offset"]))
        if zero is None:
            continue
        actuator = str(row["applied_actuator_name"])
        row["actuator_delta_minus_zero"] = float(row["actuator_delta"]) - float(
            zero.get("actuator_deltas", {}).get(actuator, 0.0)
        )
        expected_keywords = _expected_mechanical_keywords(actuator)
        expected_name, expected_relative, expected_raw = _max_relative_delta(
            row["mechanical_deltas"], zero["mechanical_deltas"], expected_keywords
        )
        row["expected_family_mech_joint"] = expected_name
        row["expected_family_mech_delta"] = expected_raw
        row["expected_family_mech_delta_minus_zero"] = expected_relative


def _print_summary(summary_rows: Sequence[dict[str, Any]]) -> None:
    fields = [
        "position_offset",
        "test_name",
        "applied_actuator_name",
        "actuator_start",
        "actuator_target",
        "actuator_end",
        "actuator_delta",
        "actuator_delta_minus_zero",
        "expected_family_mech_delta",
        "expected_family_mech_delta_minus_zero",
        "largest_mechanical_delta_joint",
        "largest_mechanical_delta",
        "largest_moving_body",
        "largest_body_translation_delta",
        "done_frac",
    ]
    widths = {field: max(len(field), 12) for field in fields}
    for row in summary_rows:
        for field in fields:
            value = row.get(field, "")
            text = str(value) if isinstance(value, str) else f"{float(value):.6g}"
            widths[field] = max(widths[field], len(text))
    print(" | ".join(field.ljust(widths[field]) for field in fields), flush=True)
    print("-+-".join("-" * widths[field] for field in fields), flush=True)
    for row in summary_rows:
        parts = []
        for field in fields:
            value = row.get(field, "")
            text = str(value) if isinstance(value, str) else f"{float(value):.6g}"
            parts.append(text.ljust(widths[field]))
        print(" | ".join(parts), flush=True)


def _print_diagnosis(summary_rows: Sequence[dict[str, Any]], actuator_names: Sequence[str], reference_body: str) -> None:
    rows_by_offset: dict[float, dict[str, dict[str, Any]]] = {}
    body_names = list(next((row.get("body_delta_vectors", {}).keys() for row in summary_rows if row.get("body_delta_vectors")), []))
    if reference_body and body_names and _body_reference_name(body_names, reference_body) is None:
        print(f"[WARN] reference_body={reference_body} was not resolved; body-relative diagnosis is unavailable.", flush=True)
    for row in summary_rows:
        rows_by_offset.setdefault(float(row["position_offset"]), {})[str(row["test_name"])] = row

    saw_pass = False
    saw_warn = False
    saw_fail = False
    for offset, rows in rows_by_offset.items():
        zero = rows.get("zero")
        if zero is None:
            print(f"[WARN] offset={offset:g}: missing zero row.", flush=True)
            saw_warn = True
            continue
        for actuator_name in actuator_names:
            pos = rows.get(f"{actuator_name}_pos_target")
            neg = rows.get(f"{actuator_name}_neg_target")
            if pos is None or neg is None:
                continue
            pos_act = float(pos["actuator_delta_minus_zero"])
            neg_act = float(neg["actuator_delta_minus_zero"])
            pos_mech = float(pos["expected_family_mech_delta_minus_zero"])
            neg_mech = float(neg["expected_family_mech_delta_minus_zero"])
            expected = _expected_mechanical_keywords(actuator_name)
            pos_unexpected_name, pos_unexpected, _ = _max_relative_delta(
                pos["mechanical_deltas"], zero["mechanical_deltas"], expected, invert=True
            )
            neg_unexpected_name, neg_unexpected, _ = _max_relative_delta(
                neg["mechanical_deltas"], zero["mechanical_deltas"], expected, invert=True
            )
            pos_body_relative_name, pos_body_relative = _max_reference_body_delta_minus_zero(
                pos, zero, body_names, reference_body
            )
            neg_body_relative_name, neg_body_relative = _max_reference_body_delta_minus_zero(
                neg, zero, body_names, reference_body
            )
            actuator_ok = _separated(float(pos["actuator_delta"]), 0.0, pos_act) and _separated(
                float(neg["actuator_delta"]), 0.0, neg_act
            )
            actuator_sign_ok = pos_act * neg_act < 0.0
            mech_ok = abs(pos_mech) >= 1.0e-4 and abs(neg_mech) >= 1.0e-4 and pos_mech * neg_mech < 0.0
            expected_dominates = max(abs(pos_mech), abs(neg_mech)) > max(abs(pos_unexpected), abs(neg_unexpected), 1.0e-4)
            print(
                f"[INFO] offset={offset:g} {actuator_name}: "
                f"actuator_pos_minus_zero={pos_act:.6g} actuator_neg_minus_zero={neg_act:.6g} "
                f"expected_mech_pos_minus_zero={pos_mech:.6g} expected_mech_neg_minus_zero={neg_mech:.6g} "
                f"unexpected_pos={pos_unexpected_name}:{pos_unexpected:.6g} "
                f"unexpected_neg={neg_unexpected_name}:{neg_unexpected:.6g} "
                f"body_pos_rel_to_{reference_body}_minus_zero={pos_body_relative_name}:{pos_body_relative:.6g} "
                f"body_neg_rel_to_{reference_body}_minus_zero={neg_body_relative_name}:{neg_body_relative:.6g}",
                flush=True,
            )
            if actuator_ok and actuator_sign_ok and mech_ok and expected_dominates and pos["done_frac"] == 0.0 and neg["done_frac"] == 0.0:
                print("[PASS] position target actuator-to-expected-arm-family transmission verified", flush=True)
                saw_pass = True
            elif not actuator_ok or not mech_ok:
                print("[FAIL] position target did not produce clear expected transmission beyond zero drift", flush=True)
                saw_fail = True
            else:
                print("[WARN] position target moved something, but expected family did not dominate cleanly", flush=True)
                saw_warn = True
            if not expected_dominates:
                print("[WARN] unexpected joint family moves more than expected family.", flush=True)
            for row in (pos, neg):
                largest_body = str(row.get("largest_moving_body", "none"))
                if reference_body and _safe_name(reference_body).lower() in largest_body.lower():
                    print("[WARN] largest body motion may be global/base settling.", flush=True)

    if saw_pass:
        print("[PASS] position target actuator-to-expected-arm-family transmission verified", flush=True)
    elif saw_warn:
        print("[WARN] position target motion observed but transmission is not cleanly verified", flush=True)
    elif saw_fail:
        print("[FAIL] position target did not produce clear expected transmission beyond zero drift", flush=True)
    else:
        print("[FAIL] no position-target transmission tests were classified.", flush=True)


def _apply_env_overrides(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg) -> None:
    env_cfg.control_mode = "position_delta"
    env_cfg.use_fd_effort = False
    env_cfg.use_sd_observer = False
    env_cfg.use_sd_feedback = False


def _make_target(
    *,
    unwrapped,
    actuator_ids: Sequence[int],
    start_positions: Sequence[float],
    actuator_index: int | None,
    offset: float,
    sign: float,
    actuator_names: Sequence[str],
) -> tuple[torch.Tensor, float, float]:
    targets = torch.tensor(start_positions, dtype=torch.float32, device=unwrapped.device).reshape(1, -1).repeat(
        unwrapped.num_envs, 1
    )
    requested = 0.0
    clamped = 0.0
    if actuator_index is not None and sign != 0.0:
        requested = float(start_positions[actuator_index] + sign * offset)
        targets[:, actuator_index] = requested
        lower, upper = _soft_limits(unwrapped.robot, actuator_ids)
        if lower is not None and upper is not None:
            before = targets[:, actuator_index].detach().clone()
            targets[:, actuator_index] = torch.clamp(targets[:, actuator_index], lower[:, actuator_index], upper[:, actuator_index])
            clamped = float(targets[0, actuator_index].detach().cpu())
            if abs(clamped - float(before[0].detach().cpu())) > 1.0e-9:
                print(
                    f"[WARN] target clamped for {actuator_names[actuator_index]}: "
                    f"requested={requested:.6g} clamped={clamped:.6g}",
                    flush=True,
                )
        else:
            clamped = requested
    return targets, requested, clamped


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    print("[INFO] Starting actuator position-target transmission diagnostic", flush=True)
    offsets = _parse_float_csv(args_cli.position_offsets)
    actuator_names = (
        [args_cli.single_actuator_joint]
        if args_cli.single_actuator_joint
        else _parse_csv(args_cli.actuator_joint_names, DEFAULT_ACTUATOR_JOINT_NAMES)
    )
    mechanical_requested = _parse_csv(args_cli.mechanical_joint_names, DEFAULT_MECHANICAL_JOINT_NAMES)
    body_requested = _parse_csv(args_cli.body_names, DEFAULT_BODY_NAMES)
    patterns: list[tuple[str, int | None, float]] = [("zero", None, 0.0)]
    for index, actuator_name in enumerate(actuator_names):
        patterns.append((f"{actuator_name}_pos_target", index, 1.0))
        patterns.append((f"{actuator_name}_neg_target", index, -1.0))
    print(f"[INFO] position_offsets={offsets}", flush=True)
    print(f"[INFO] tests_per_offset={len(patterns)} tests={[item[0] for item in patterns]}", flush=True)

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.debug_reward_terms = True
    env_cfg.log_full_pipeline_debug = True
    _apply_env_overrides(env_cfg)
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device
    apply_kevin_sim_memory_optimizations(env_cfg.sim, verbose=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args_cli.log_root) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "actuator_position_target_transmission.csv"

    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)
        env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))
        unwrapped = env.unwrapped
        actuator_ids = _resolve_required_joint_ids(unwrapped.robot, actuator_names, label="actuator")
        mechanical_names, mechanical_ids = _resolve_optional_joint_ids(unwrapped.robot, mechanical_requested, label="mechanical")
        body_names, body_ids = _resolve_body_ids(unwrapped.robot, body_requested)
        _install_position_target_override(unwrapped, actuator_ids)

        zero_action = _make_zero_action(env)
        csv_fields = _csv_fields(actuator_names, mechanical_names, body_names)
        summary_rows: list[dict[str, Any]] = []
        print(f"[INFO] CSV output: {csv_path}", flush=True)
        with csv_path.open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
            writer.writeheader()
            for offset in offsets:
                for test_name, actuator_index, sign in patterns:
                    print(f"[INFO] Running offset={offset:g} test={test_name}", flush=True)
                    _reset_env(env)
                    settled_positions = _joint_values(unwrapped.robot, actuator_ids, "joint_pos")
                    hold_targets = settled_positions if args_cli.hold_other_actuator_targets else _joint_values(
                        unwrapped.robot, actuator_ids, "default_joint_pos"
                    )
                    hold_tensor = torch.tensor(hold_targets, dtype=torch.float32, device=unwrapped.device).reshape(1, -1)
                    unwrapped._direct_position_targets = hold_tensor.repeat(unwrapped.num_envs, 1).detach().clone()  # noqa: SLF001
                    for _ in range(args_cli.settle_steps):
                        _, _, _, _ = env.step(zero_action)
                    actuator_start = _joint_values(unwrapped.robot, actuator_ids, "joint_pos")
                    mechanical_start = _joint_values(unwrapped.robot, mechanical_ids, "joint_pos")
                    body_start = _body_vectors(unwrapped.robot, body_ids)
                    targets, requested, clamped = _make_target(
                        unwrapped=unwrapped,
                        actuator_ids=actuator_ids,
                        start_positions=actuator_start,
                        actuator_index=actuator_index,
                        offset=offset,
                        sign=sign,
                        actuator_names=actuator_names,
                    )
                    unwrapped._direct_position_targets = targets.detach().clone()  # noqa: SLF001
                    test_rows: list[dict[str, float | str]] = []
                    for step in range(1, args_cli.steps_per_test + 1):
                        _, _, dones, _ = env.step(zero_action)
                        row = _collect_row(
                            unwrapped=unwrapped,
                            dones=dones,
                            step=step,
                            test_name=test_name,
                            position_offset=offset,
                            position_targets=targets,
                            requested_target=requested,
                            clamped_target=clamped,
                            hold_targets=actuator_start,
                            actuator_names=actuator_names,
                            actuator_ids=actuator_ids,
                            actuator_start=actuator_start,
                            mechanical_names=mechanical_names,
                            mechanical_ids=mechanical_ids,
                            mechanical_start=mechanical_start,
                            body_names=body_names,
                            body_ids=body_ids,
                            body_start=body_start,
                        )
                        writer.writerow(row)
                        test_rows.append(row)
                        is_safe, reason = _step_is_safe(unwrapped, dones)
                        if not is_safe:
                            print(
                                f"[WARN] Stopping test early: offset={offset:g} test={test_name} "
                                f"step={step} reason={reason}",
                                flush=True,
                            )
                            break
                    csv_file.flush()
                    summary = _summarize(
                        position_offset=offset,
                        test_name=test_name,
                        rows=test_rows,
                        actuator_names=actuator_names,
                        actuator_ids=actuator_ids,
                        mechanical_names=mechanical_names,
                        body_names=body_names,
                    )
                    summary_rows.append(summary)
                    _apply_zero_relative(summary_rows)
                    print(
                        f"[RESULT] offset={offset:g} test={test_name} "
                        f"actuator_delta={float(summary['actuator_delta']):.6g} "
                        f"actuator_delta_minus_zero={float(summary['actuator_delta_minus_zero']):.6g} "
                        f"largest_mechanical_delta_joint={summary['largest_mechanical_delta_joint']} "
                        f"largest_mechanical_delta={float(summary['largest_mechanical_delta']):.6g}",
                        flush=True,
                    )

        _apply_zero_relative(summary_rows)
        print("[INFO] Summary:", flush=True)
        _print_summary(summary_rows)
        print(
            f"[INFO] Direct override call counts: _pre_physics_step={unwrapped._direct_position_pre_physics_calls} "
            f"_apply_action={unwrapped._direct_position_apply_action_calls}",
            flush=True,
        )
        _print_diagnosis(summary_rows, actuator_names, args_cli.reference_body)
        print(f"[INFO] Finished. Logged actuator position-target transmission diagnostic to {csv_path}", flush=True)
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
