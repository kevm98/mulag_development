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
    "Drehzapfen_constraint_joint_2",
    "Drehzapfen_constraint_joint_3",
    "Ausleger_I_constraint_joint_1",
    "Ausleger_I_constraint_joint_2",
    "Ausleger_I_constraint_joint_3",
    "Ausleger_II_constraint_joint_1",
    "Ausleger_II_constraint_joint_2",
    "Ausleger_II_constraint_joint_3",
    "Messerkopf_Schwenk_constraint_joint_1",
    "Messerkopf_Schwenk_constraint_joint_2",
    "Messerkopf_Schwenk_constraint_joint_3",
    "Messerkopf_Schwenk_constraint_joint_4",
    "Messerkopf_Schwenk_constraint_joint_5",
    "Messerkopf_Schwenk_constraint_joint_6",
    "Messerkopf_joint",
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

parser = argparse.ArgumentParser(description="Hydraulic actuator-to-arm transmission diagnostic for Kevin Mulag.")
parser.add_argument("--task", type=str, default="Kevin-Mulag-AdaptiveRL-JointReach-v0", help="Gym task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of Isaac Lab environments.")
parser.add_argument("--steps_per_test", type=int, default=200, help="Direct-effort steps per test pattern.")
parser.add_argument("--settle_steps", type=int, default=50, help="Zero-effort settle steps before each test.")
parser.add_argument(
    "--effort_values",
    type=str,
    default="5,10,20,50",
    help='Comma-separated direct actuator effort magnitudes, e.g. "5,10,20,50".',
)
parser.add_argument(
    "--actuator_joint_names",
    type=str,
    default=",".join(DEFAULT_ACTUATOR_JOINT_NAMES),
    help="Comma-separated actuator-side prismatic joint names to test.",
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
    "--single_actuator_joint",
    type=str,
    default=None,
    help="Optional single actuator-side joint to test, e.g. Drehzapfen_joint.",
)
parser.add_argument(
    "--diagnostic_disable_actuator_drives",
    action="store_true",
    help="Temporarily set tested actuator joint stiffness/damping for force-transmission diagnosis only.",
)
parser.add_argument(
    "--diagnostic_drive_stiffness",
    type=float,
    default=0.0,
    help="Temporary stiffness used with --diagnostic_disable_actuator_drives.",
)
parser.add_argument(
    "--diagnostic_drive_damping",
    type=float,
    default=100.0,
    help="Temporary damping used with --diagnostic_disable_actuator_drives.",
)
parser.add_argument(
    "--diagnostic_max_abs_effort",
    type=float,
    default=None,
    help="Optional diagnostic effort clamp. This does not change training config.",
)
parser.add_argument(
    "--use_force_actuated_cylinders",
    action="store_true",
    help="Request a future Kevin force-actuated cylinder config variant if the env cfg supports it.",
)
parser.add_argument(
    "--log_root",
    type=str,
    default="logs/actuator_to_arm_transmission",
    help="Directory under which timestamped actuator-to-arm logs are written.",
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


def _parse_effort_values(text: str) -> list[float]:
    values: list[float] = []
    for item in text.split(","):
        stripped = item.strip()
        if stripped:
            values.append(float(stripped))
    if not values:
        raise ValueError("--effort_values must contain at least one number.")
    return values


def _safe_name(name: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in name).strip("_")


def _robot_joint_names(robot) -> list[str]:
    names = getattr(robot, "joint_names", None)
    if names is None:
        names = getattr(robot.data, "joint_names", None)
    if names is None:
        return []
    return list(names)


def _robot_body_names(robot) -> list[str]:
    names = getattr(robot, "body_names", None)
    if names is None:
        names = getattr(robot.data, "body_names", None)
    if names is None:
        return []
    return list(names)


def _resolve_required_joint_ids(robot, joint_names: Sequence[str], *, label: str) -> list[int]:
    joint_ids_out: list[int] = []
    all_joint_names = _robot_joint_names(robot)
    for joint_name in joint_names:
        joint_ids, resolved_names = robot.find_joints(joint_name)
        if len(joint_ids) == 0:
            raise RuntimeError(f"Could not resolve {label} joint name {joint_name!r}.")
        joint_id = int(joint_ids[0])
        resolved_name = resolved_names[0] if resolved_names else all_joint_names[joint_id]
        joint_ids_out.append(joint_id)
        print(f"[INFO] Resolved {label}: requested={joint_name} id={joint_id} actual={resolved_name}", flush=True)
    return joint_ids_out


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
            print(f"[WARN] Missing {label} joint {joint_name!r}; skipping it.", flush=True)
            continue
        joint_id = int(joint_ids[0])
        if resolved_names:
            resolved_name = resolved_names[0]
        elif 0 <= joint_id < len(all_joint_names):
            resolved_name = all_joint_names[joint_id]
        else:
            resolved_name = joint_name
        names_out.append(joint_name)
        ids_out.append(joint_id)
        print(f"[INFO] Resolved {label}: requested={joint_name} id={joint_id} actual={resolved_name}", flush=True)
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
            if resolved_name:
                actual_name = resolved_name
            elif 0 <= int(body_id) < len(all_body_names):
                actual_name = all_body_names[int(body_id)]
            else:
                actual_name = f"body_{int(body_id)}"
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


def _format_tensor_slice(value: Any, joint_ids: Sequence[int]) -> str:
    if not isinstance(value, torch.Tensor):
        return "unavailable"
    tensor = value.detach().cpu()
    if tensor.ndim >= 2:
        tensor = tensor[0]
    flat = tensor.flatten()
    selected = []
    for joint_id in joint_ids:
        selected.append(float(flat[joint_id]) if 0 <= joint_id < flat.numel() else float("nan"))
    return "[" + ", ".join(f"{item:.6g}" for item in selected) + "]"


def _print_joint_drive_debug(robot, actuator_joint_names: Sequence[str], actuator_joint_ids: Sequence[int]) -> None:
    print("[INFO] Actuator drive/limit fields for tested actuator joints:", flush=True)
    for field_name in (
        "joint_stiffness",
        "default_joint_stiffness",
        "stiffness",
        "joint_damping",
        "default_joint_damping",
        "damping",
        "joint_effort_limits",
        "soft_joint_effort_limits",
        "effort_limits",
        "joint_pos_target",
        "joint_position_target",
        "joint_position_targets",
        "default_joint_pos",
    ):
        value = getattr(robot.data, field_name, None)
        if isinstance(value, torch.Tensor):
            print(
                f"  robot.data.{field_name}: {_format_tensor_slice(value, actuator_joint_ids)}",
                flush=True,
            )
    print(
        "[INFO] If actuator effort does not move the actuator stroke, the high actuator drive may be holding the "
        "prismatic joint at its target. This script only reports that possibility; it does not change drive values.",
        flush=True,
    )
    print(f"[INFO] Tested actuator names: {list(actuator_joint_names)}", flush=True)
    print(f"[INFO] Tested actuator IDs: {list(actuator_joint_ids)}", flush=True)


def _read_joint_param_values(robot, field_names: tuple[str, ...], joint_ids: Sequence[int]) -> torch.Tensor | None:
    for field_name in field_names:
        value = getattr(robot.data, field_name, None)
        if isinstance(value, torch.Tensor):
            tensor = value.detach().clone()
            if tensor.ndim == 1:
                return tensor[joint_ids].reshape(1, len(joint_ids)).clone()
            if tensor.ndim >= 2:
                return tensor[:, joint_ids].clone()
    return None


def _write_joint_param_values(robot, method_name: str, values: torch.Tensor, joint_ids: Sequence[int]) -> None:
    method = getattr(robot, method_name, None)
    if method is None:
        raise RuntimeError(f"Articulation does not expose {method_name}().")
    try:
        method(values, joint_ids=list(joint_ids))
    except TypeError:
        method(values, joint_ids)


def _apply_diagnostic_drive_override(unwrapped, actuator_joint_ids: Sequence[int], stiffness: float, damping: float):
    print("[WARN] Diagnostic drive override active. This is not a training config.", flush=True)
    original_stiffness = _read_joint_param_values(
        unwrapped.robot, ("joint_stiffness", "default_joint_stiffness", "stiffness"), actuator_joint_ids
    )
    original_damping = _read_joint_param_values(
        unwrapped.robot, ("joint_damping", "default_joint_damping", "damping"), actuator_joint_ids
    )
    print(f"[INFO] Original stiffness saved: {original_stiffness is not None}", flush=True)
    print(f"[INFO] Original damping saved: {original_damping is not None}", flush=True)
    stiffness_values = torch.full(
        (unwrapped.num_envs, len(actuator_joint_ids)), float(stiffness), dtype=torch.float32, device=unwrapped.device
    )
    damping_values = torch.full(
        (unwrapped.num_envs, len(actuator_joint_ids)), float(damping), dtype=torch.float32, device=unwrapped.device
    )
    _write_joint_param_values(unwrapped.robot, "write_joint_stiffness_to_sim", stiffness_values, actuator_joint_ids)
    _write_joint_param_values(unwrapped.robot, "write_joint_damping_to_sim", damping_values, actuator_joint_ids)
    print(
        f"[INFO] Temporary stiffness/damping applied for actuator force transmission test: "
        f"stiffness={stiffness:g} damping={damping:g}",
        flush=True,
    )
    return {"stiffness": original_stiffness, "damping": original_damping}


def _restore_diagnostic_drive_override(unwrapped, actuator_joint_ids: Sequence[int], saved_state) -> None:
    if not saved_state:
        return
    try:
        if isinstance(saved_state.get("stiffness"), torch.Tensor):
            values = saved_state["stiffness"].to(device=unwrapped.device)
            _write_joint_param_values(unwrapped.robot, "write_joint_stiffness_to_sim", values, actuator_joint_ids)
        if isinstance(saved_state.get("damping"), torch.Tensor):
            values = saved_state["damping"].to(device=unwrapped.device)
            _write_joint_param_values(unwrapped.robot, "write_joint_damping_to_sim", values, actuator_joint_ids)
        print("[INFO] Restored original diagnostic stiffness/damping values.", flush=True)
    except Exception as exc:
        print(f"[WARN] Could not restore diagnostic stiffness/damping before shutdown: {exc}", flush=True)


def _build_patterns(actuator_joint_names: Sequence[str]) -> dict[str, list[float]]:
    patterns = {"zero": [0.0 for _ in actuator_joint_names]}
    for index, actuator_name in enumerate(actuator_joint_names):
        pos = [0.0 for _ in actuator_joint_names]
        pos[index] = 1.0
        neg = [0.0 for _ in actuator_joint_names]
        neg[index] = -1.0
        patterns[f"{actuator_name}_pos"] = pos
        patterns[f"{actuator_name}_neg"] = neg
    return patterns


def _make_effort(env: RslRlVecEnvWrapper, signs: Sequence[float], effort_value: float) -> torch.Tensor:
    effort = torch.tensor(signs, dtype=torch.float32, device=env.device).reshape(1, len(signs))
    return (float(effort_value) * effort).repeat(env.num_envs, 1)


def _make_zero_action(env: RslRlVecEnvWrapper) -> torch.Tensor:
    return torch.zeros((env.num_envs, env.num_actions), dtype=torch.float32, device=env.device)


def _install_direct_effort_override(unwrapped, actuator_joint_ids: Sequence[int]) -> None:
    if not hasattr(unwrapped.robot, "set_joint_effort_target"):
        raise RuntimeError("This IsaacLab Articulation does not expose set_joint_effort_target.")

    unwrapped._direct_actuator_joint_ids = list(actuator_joint_ids)  # noqa: SLF001
    unwrapped._direct_actuator_effort_command = torch.zeros(  # noqa: SLF001
        (unwrapped.num_envs, len(actuator_joint_ids)), dtype=torch.float32, device=unwrapped.device
    )
    unwrapped._direct_actuator_pre_physics_calls = 0  # noqa: SLF001
    unwrapped._direct_actuator_apply_action_calls = 0  # noqa: SLF001

    def _direct_pre_physics_step(self, actions: torch.Tensor) -> None:
        self._direct_actuator_pre_physics_calls += 1
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
        self._current_torque = self._direct_actuator_effort_command.detach().clone()
        _ = actions

    def _direct_apply_action(self) -> None:
        self._direct_actuator_apply_action_calls += 1
        self.robot.set_joint_effort_target(
            self._direct_actuator_effort_command,
            joint_ids=self._direct_actuator_joint_ids,
        )

    unwrapped._pre_physics_step = MethodType(_direct_pre_physics_step, unwrapped)  # noqa: SLF001
    unwrapped._apply_action = MethodType(_direct_apply_action, unwrapped)  # noqa: SLF001
    print(
        "[INFO] Installed actuator-effort env.step override: Kevin _pre_physics_step/_apply_action are replaced "
        "for this diagnostic, so PPO/ID/FD/SD/torque adapter cannot overwrite direct actuator effort.",
        flush=True,
    )


def _reset_env(env: RslRlVecEnvWrapper):
    reset_out = env.reset()
    if isinstance(reset_out, tuple):
        return reset_out[0]
    return reset_out


def _joint_values(robot, joint_ids: Sequence[int], field_name: str) -> list[float]:
    value = getattr(robot.data, field_name)
    tensor = value.detach()
    if tensor.ndim >= 2:
        tensor = tensor[0]
    return [float(tensor[joint_id].cpu()) for joint_id in joint_ids]


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
    vectors: list[tuple[float, float, float]] = []
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
    quats: list[tuple[float, float, float, float]] = []
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
    if isinstance(body_pos, torch.Tensor):
        if not torch.isfinite(body_pos).all():
            return False, "non-finite body position"
        if torch.max(torch.abs(body_pos.detach())).item() > 1.0e5:
            return False, "unstable body position magnitude"
    return True, ""


def _field(prefix: str, joint_name: str, suffix: str) -> str:
    return f"{prefix}_{_safe_name(joint_name)}_{suffix}"


def _csv_fields(actuator_names: Sequence[str], mechanical_names: Sequence[str], body_names: Sequence[str]) -> list[str]:
    fields = [
        "test_name",
        "effort_value",
        "applied_actuator_name",
        "applied_actuator_id",
        "step",
        "effort_command",
    ]
    for name in actuator_names:
        fields.extend(
            [
                _field("actuator", name, "pos"),
                _field("actuator", name, "vel"),
                _field("actuator", name, "delta_from_start"),
                _field("effort_command", name, "value"),
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
    fields.extend(["done_frac", "reset_buf_mean", "time_out_frac", "terminated_frac"])
    return fields


def _active_actuator(
    actuator_names: Sequence[str],
    actuator_ids: Sequence[int],
    command_values: Sequence[float],
) -> tuple[str, int]:
    active = [index for index, value in enumerate(command_values) if abs(value) > 0.0]
    if not active:
        return "none", -1
    index = active[0]
    return actuator_names[index], actuator_ids[index]


def _collect_row(
    *,
    unwrapped,
    dones: torch.Tensor,
    step: int,
    effort_value: float,
    test_name: str,
    effort_command: torch.Tensor,
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
    command_values = [float(item) for item in effort_command[0].detach().cpu().tolist()]
    applied_name, applied_id = _active_actuator(actuator_names, actuator_ids, command_values)
    row: dict[str, float | str] = {
        "test_name": test_name,
        "effort_value": float(effort_value),
        "applied_actuator_name": applied_name,
        "applied_actuator_id": float(applied_id),
        "step": float(step),
        "effort_command": float(max(command_values, key=abs)) if command_values else 0.0,
    }
    for name, command, pos, vel, start in zip(
        actuator_names, command_values, actuator_pos, actuator_vel, actuator_start, strict=True
    ):
        row[_field("actuator", name, "pos")] = pos
        row[_field("actuator", name, "vel")] = vel
        row[_field("actuator", name, "delta_from_start")] = pos - start
        row[_field("effort_command", name, "value")] = command
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


def _max_body_delta_matching(body_deltas: dict[str, float], keywords: Sequence[str], *, exclude_cylinder: bool = False) -> float:
    values = []
    for body_name, delta in body_deltas.items():
        if not math.isfinite(delta):
            continue
        lowered = body_name.lower()
        if exclude_cylinder and any(token in lowered for token in ("zylinder", "kolben", "rohr")):
            continue
        if any(keyword.lower() in lowered for keyword in keywords):
            values.append(abs(delta))
    return max(values) if values else math.nan


def _body_delta_by_name_or_contains(body_deltas: dict[str, float], names_or_keywords: Sequence[str]) -> float:
    for name in names_or_keywords:
        if name in body_deltas and math.isfinite(body_deltas[name]):
            return body_deltas[name]
    for body_name, delta in body_deltas.items():
        lowered = body_name.lower()
        if any(name.lower() in lowered for name in names_or_keywords) and math.isfinite(delta):
            return delta
    return math.nan


def _summarize(
    *,
    effort_value: float,
    test_name: str,
    rows: list[dict[str, float | str]],
    actuator_names: Sequence[str],
    actuator_ids: Sequence[int],
    mechanical_names: Sequence[str],
    body_names: Sequence[str],
    body_pose_available: bool,
) -> dict[str, Any]:
    if not rows:
        return {}
    last = rows[-1]
    command_values = [float(last[_field("effort_command", name, "value")]) for name in actuator_names]
    applied_name, applied_id = _active_actuator(actuator_names, actuator_ids, command_values)
    actuator_deltas = {
        name: float(last[_field("actuator", name, "delta_from_start")]) for name in actuator_names
    }
    mechanical_deltas = {
        name: float(last[_field("mech", name, "delta_from_start")]) for name in mechanical_names
    }
    body_deltas = {
        name: float(last[_field("body", name, "translation_delta_norm")]) for name in body_names
    }
    largest_mech_name, largest_mech_delta = _max_abs_delta_name(mechanical_names, mechanical_deltas)
    largest_body_name, largest_body_delta = _max_body_delta_name(body_names, body_deltas)
    if applied_name == "none":
        actuator_delta_name, actuator_delta = _max_abs_delta_name(actuator_names, actuator_deltas)
        actuator_start = float(rows[0][_field("actuator", actuator_delta_name, "pos")]) - float(
            rows[0][_field("actuator", actuator_delta_name, "delta_from_start")]
        )
        actuator_end = float(last[_field("actuator", actuator_delta_name, "pos")])
    else:
        actuator_delta = actuator_deltas[applied_name]
        actuator_start = float(rows[0][_field("actuator", applied_name, "pos")]) - float(
            rows[0][_field("actuator", applied_name, "delta_from_start")]
        )
        actuator_end = float(last[_field("actuator", applied_name, "pos")])
    return {
        "effort_value": effort_value,
        "test_name": test_name,
        "applied_actuator_name": applied_name,
        "applied_actuator_id": applied_id,
        "actuator_start": actuator_start,
        "actuator_end": actuator_end,
        "actuator_delta": actuator_delta,
        "actuator_deltas": actuator_deltas,
        "mechanical_deltas": mechanical_deltas,
        "body_deltas": body_deltas,
        "largest_mechanical_delta_joint": largest_mech_name,
        "largest_mechanical_delta": largest_mech_delta,
        "largest_moving_body": largest_body_name,
        "largest_body_translation_delta": largest_body_delta,
        "cylinder_rod_body_delta": _max_body_delta_matching(body_deltas, ("Kolben",)),
        "cylinder_tube_body_delta": _max_body_delta_matching(body_deltas, ("Rohr",)),
        "arm_link_body_delta": _max_body_delta_matching(
            body_deltas,
            ("Drehzapfen", "Frontrahmen", "Ausleger_I", "Ausleger_II", "Messerkopf", "Messerkopf_Schwenk_base"),
            exclude_cylinder=True,
        ),
        "Drehzapfen_body_delta": _body_delta_by_name_or_contains(body_deltas, ("Drehzapfen",)),
        "Ausleger_Schwenkzylinder_Kolben_body_delta": _body_delta_by_name_or_contains(
            body_deltas, ("Ausleger_Schwenkzylinder_Kolben",)
        ),
        "Ausleger_Schwenkzylinder_Rohr_body_delta": _body_delta_by_name_or_contains(
            body_deltas, ("Ausleger_Schwenkzylinder_Rohr",)
        ),
        "body_pose_available": body_pose_available,
        "actuator_delta_minus_zero": math.nan,
        "largest_mechanical_delta_minus_zero": math.nan,
        "Drehzapfen_constraint_joint_1_delta": mechanical_deltas.get("Drehzapfen_constraint_joint_1", math.nan),
        "Drehzapfen_constraint_joint_2_delta": mechanical_deltas.get("Drehzapfen_constraint_joint_2", math.nan),
        "Drehzapfen_constraint_joint_3_delta": mechanical_deltas.get("Drehzapfen_constraint_joint_3", math.nan),
        "done_frac": max(float(row["done_frac"]) for row in rows),
    }


def _apply_zero_relative_deltas(summary_rows: Sequence[dict[str, Any]]) -> None:
    rows_by_effort: dict[float, dict[str, Any]] = {}
    for row in summary_rows:
        if str(row.get("test_name")) == "zero":
            rows_by_effort[float(row["effort_value"])] = row
    for row in summary_rows:
        zero = rows_by_effort.get(float(row["effort_value"]))
        if zero is None:
            continue
        actuator_name = str(row.get("applied_actuator_name", "none"))
        if actuator_name == "none":
            row["actuator_delta_minus_zero"] = 0.0
        else:
            zero_delta = float(zero.get("actuator_deltas", {}).get(actuator_name, 0.0))
            row["actuator_delta_minus_zero"] = float(row["actuator_delta"]) - zero_delta
        largest_joint = str(row.get("largest_mechanical_delta_joint", "none"))
        if largest_joint == "none":
            row["largest_mechanical_delta_minus_zero"] = math.nan
        else:
            zero_mech_delta = float(zero.get("mechanical_deltas", {}).get(largest_joint, 0.0))
            row["largest_mechanical_delta_minus_zero"] = float(row["largest_mechanical_delta"]) - zero_mech_delta


def _print_summary(summary_rows: Sequence[dict[str, Any]]) -> None:
    fields = [
        "effort_value",
        "test_name",
        "applied_actuator_name",
        "actuator_start",
        "actuator_end",
        "actuator_delta",
        "actuator_delta_minus_zero",
        "largest_mechanical_delta_joint",
        "largest_mechanical_delta",
        "largest_mechanical_delta_minus_zero",
        "largest_moving_body",
        "largest_body_translation_delta",
        "cylinder_rod_body_delta",
        "cylinder_tube_body_delta",
        "arm_link_body_delta",
        "Drehzapfen_constraint_joint_1_delta",
        "Drehzapfen_constraint_joint_2_delta",
        "Drehzapfen_constraint_joint_3_delta",
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


def _print_result(row: dict[str, Any]) -> None:
    print(
        "[RESULT] "
        f"effort_value={float(row['effort_value']):.6g} "
        f"test_name={row['test_name']} "
        f"actuator_start={float(row['actuator_start']):.6g} "
        f"actuator_end={float(row['actuator_end']):.6g} "
        f"actuator_delta={float(row['actuator_delta']):.6g} "
        f"actuator_delta_minus_zero={float(row['actuator_delta_minus_zero']):.6g} "
        f"largest_mechanical_delta_joint={row['largest_mechanical_delta_joint']} "
        f"largest_mechanical_delta={float(row['largest_mechanical_delta']):.6g} "
        f"largest_mechanical_delta_minus_zero={float(row['largest_mechanical_delta_minus_zero']):.6g} "
        f"largest_moving_body={row['largest_moving_body']} "
        f"largest_body_translation_delta={float(row['largest_body_translation_delta']):.6g} "
        f"cylinder_rod_body_delta={float(row['cylinder_rod_body_delta']):.6g} "
        f"cylinder_tube_body_delta={float(row['cylinder_tube_body_delta']):.6g} "
        f"arm_link_body_delta={float(row['arm_link_body_delta']):.6g} "
        f"Drehzapfen_constraint_joint_1_delta={float(row['Drehzapfen_constraint_joint_1_delta']):.6g} "
        f"Drehzapfen_constraint_joint_2_delta={float(row['Drehzapfen_constraint_joint_2_delta']):.6g} "
        f"Drehzapfen_constraint_joint_3_delta={float(row['Drehzapfen_constraint_joint_3_delta']):.6g} "
        f"Drehzapfen_body_delta={float(row['Drehzapfen_body_delta']):.6g} "
        f"Ausleger_Schwenkzylinder_Kolben_body_delta="
        f"{float(row['Ausleger_Schwenkzylinder_Kolben_body_delta']):.6g} "
        f"Ausleger_Schwenkzylinder_Rohr_body_delta="
        f"{float(row['Ausleger_Schwenkzylinder_Rohr_body_delta']):.6g}",
        flush=True,
    )


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
        relative = value - float(zero_deltas.get(name, 0.0))
        if abs(relative) > abs(best_relative):
            best_name = name
            best_relative = relative
            best_raw = value
    return best_name, best_relative, best_raw


def _runtime_separated(value: float, zero_value: float, relative_value: float) -> bool:
    motion_threshold = 1.0e-5
    min_delta_from_zero = 1.0e-4
    drift_margin_ratio = 3.0
    if not all(math.isfinite(item) for item in (value, zero_value, relative_value)):
        return False
    if abs(relative_value) < max(motion_threshold, min_delta_from_zero):
        return False
    return abs(value) >= drift_margin_ratio * max(abs(zero_value), motion_threshold)


def _print_diagnosis(summary_rows: Sequence[dict[str, Any]], actuator_names: Sequence[str]) -> None:
    rows_by_effort: dict[float, dict[str, dict[str, Any]]] = {}
    for row in summary_rows:
        rows_by_effort.setdefault(float(row["effort_value"]), {})[str(row["test_name"])] = row

    saw_pass = False
    saw_unexpected = False
    saw_warn = False
    saw_fail = False
    for effort_value, rows in rows_by_effort.items():
        zero = rows.get("zero")
        if zero is None:
            print(f"[WARN] effort={effort_value:g}: missing zero-effort row; cannot classify drift-relative motion.")
            saw_warn = True
            continue
        for actuator_name in actuator_names:
            pos = rows.get(f"{actuator_name}_pos")
            neg = rows.get(f"{actuator_name}_neg")
            if pos is None or neg is None:
                print(f"[WARN] effort={effort_value:g} {actuator_name}: missing pos/neg rows.")
                saw_warn = True
                continue

            zero_actuator_delta = float(zero["actuator_deltas"].get(actuator_name, 0.0))
            pos_actuator_delta = float(pos["actuator_deltas"].get(actuator_name, math.nan))
            neg_actuator_delta = float(neg["actuator_deltas"].get(actuator_name, math.nan))
            pos_actuator_minus_zero = pos_actuator_delta - zero_actuator_delta
            neg_actuator_minus_zero = neg_actuator_delta - zero_actuator_delta
            actuator_separated = _runtime_separated(
                pos_actuator_delta, zero_actuator_delta, pos_actuator_minus_zero
            ) and _runtime_separated(neg_actuator_delta, zero_actuator_delta, neg_actuator_minus_zero)
            actuator_sign_ok = pos_actuator_minus_zero * neg_actuator_minus_zero < 0.0

            expected_keywords = _expected_mechanical_keywords(actuator_name)
            pos_expected_name, pos_expected_relative, pos_expected_raw = _max_relative_delta(
                pos["mechanical_deltas"], zero["mechanical_deltas"], expected_keywords
            )
            neg_expected_name, neg_expected_relative, neg_expected_raw = _max_relative_delta(
                neg["mechanical_deltas"], zero["mechanical_deltas"], expected_keywords
            )
            pos_unexpected_name, pos_unexpected_relative, _ = _max_relative_delta(
                pos["mechanical_deltas"], zero["mechanical_deltas"], expected_keywords, invert=True
            )
            neg_unexpected_name, neg_unexpected_relative, _ = _max_relative_delta(
                neg["mechanical_deltas"], zero["mechanical_deltas"], expected_keywords, invert=True
            )
            expected_separated = _runtime_separated(
                pos_expected_raw,
                float(zero["mechanical_deltas"].get(pos_expected_name, 0.0)),
                pos_expected_relative,
            ) and _runtime_separated(
                neg_expected_raw,
                float(zero["mechanical_deltas"].get(neg_expected_name, 0.0)),
                neg_expected_relative,
            )
            expected_sign_ok = pos_expected_relative * neg_expected_relative < 0.0
            expected_abs = max(abs(pos_expected_relative), abs(neg_expected_relative))
            unexpected_abs = max(abs(pos_unexpected_relative), abs(neg_unexpected_relative))
            expected_dominates = expected_abs > max(unexpected_abs, 1.0e-4)

            print(
                f"[INFO] effort={effort_value:g} {actuator_name}: "
                f"actuator_pos_minus_zero={pos_actuator_minus_zero:.6g} "
                f"actuator_neg_minus_zero={neg_actuator_minus_zero:.6g} "
                f"expected_mech_pos={pos_expected_name}:{pos_expected_relative:.6g} "
                f"expected_mech_neg={neg_expected_name}:{neg_expected_relative:.6g} "
                f"unexpected_mech_pos={pos_unexpected_name}:{pos_unexpected_relative:.6g} "
                f"unexpected_mech_neg={neg_unexpected_name}:{neg_unexpected_relative:.6g}",
                flush=True,
            )

            if unexpected_abs > expected_abs:
                print("[WARN] largest mechanical motion is from unexpected joint family", flush=True)
                saw_unexpected = True

            if expected_separated and expected_sign_ok and expected_dominates:
                print("[PASS] actuator-to-expected-arm-family transmission verified", flush=True)
                saw_pass = True
            elif not actuator_separated and not expected_separated:
                print("[FAIL] actuator effort did not produce clear transmission beyond zero drift", flush=True)
                saw_fail = True
            elif not actuator_sign_ok or not expected_sign_ok:
                print("[WARN] actuator-to-arm motion observed but pos/neg signs do not separate cleanly", flush=True)
                saw_warn = True
            else:
                print("[WARN] actuator-to-arm motion observed but not separable from zero drift", flush=True)
                saw_warn = True

            for row in (pos, neg):
                if not bool(row.get("body_pose_available", False)):
                    print(f"[WARN] {row['test_name']}: body pose data was unavailable.", flush=True)
                    continue
                largest_body = str(row.get("largest_moving_body", "none"))
                if "frontrahmen" in largest_body.lower() or "base" in largest_body.lower():
                    print(f"[WARN] {row['test_name']}: largest body motion may be global settling/base motion.", flush=True)
                body_delta = float(row.get("largest_body_translation_delta", 0.0))
                zero_body_delta = float(zero.get("body_deltas", {}).get(largest_body, 0.0))
                if body_delta > 1.0e-5 and abs(body_delta - zero_body_delta) < 1.0e-4:
                    print(f"[WARN] {row['test_name']}: body motion exists but is also present in zero-effort test.", flush=True)
                if not expected_separated and abs(float(row.get("largest_body_translation_delta", 0.0))) > 1.0e-5:
                    print(
                        f"[WARN] {row['test_name']}: body motion alone is insufficient to verify actuator-to-arm transmission.",
                        flush=True,
                    )

    if saw_pass:
        print("[PASS] actuator-to-expected-arm-family transmission verified", flush=True)
    elif saw_unexpected:
        print("[WARN] largest mechanical motion is from unexpected joint family", flush=True)
    elif saw_warn:
        print("[WARN] actuator-to-arm motion observed but not separable from zero drift", flush=True)
    elif saw_fail:
        print("[FAIL] actuator effort did not produce clear transmission beyond zero drift", flush=True)
    else:
        print("[FAIL] actuator effort did not produce clear transmission beyond zero drift", flush=True)


def _apply_env_overrides(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg) -> None:
    env_cfg.control_mode = "effort"
    env_cfg.use_fd_effort = False
    env_cfg.use_sd_observer = False
    env_cfg.use_sd_feedback = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    print("[INFO] Starting actuator-to-arm transmission diagnostic", flush=True)
    if args_cli.steps_per_test <= 0:
        raise ValueError("--steps_per_test must be positive.")
    if args_cli.settle_steps < 0:
        raise ValueError("--settle_steps must be non-negative.")
    effort_values = _parse_effort_values(args_cli.effort_values)
    if args_cli.diagnostic_max_abs_effort is not None:
        max_effort = abs(float(args_cli.diagnostic_max_abs_effort))
        effort_values = [max(-max_effort, min(max_effort, value)) for value in effort_values]
        print(f"[INFO] Diagnostic effort clamp active: max_abs_effort={max_effort:g}", flush=True)
    actuator_names = (
        [args_cli.single_actuator_joint]
        if args_cli.single_actuator_joint
        else _parse_csv(args_cli.actuator_joint_names, DEFAULT_ACTUATOR_JOINT_NAMES)
    )
    mechanical_names_requested = _parse_csv(args_cli.mechanical_joint_names, DEFAULT_MECHANICAL_JOINT_NAMES)
    body_names_requested = _parse_csv(args_cli.body_names, DEFAULT_BODY_NAMES)
    patterns = _build_patterns(actuator_names)
    print(f"[INFO] Number of effort values: {len(effort_values)} values={effort_values}", flush=True)
    print(f"[INFO] Number of tests generated per effort value: {len(patterns)} tests={list(patterns.keys())}", flush=True)
    print(
        f"[INFO] Total effort tests planned: {len(effort_values) * len(patterns)}",
        flush=True,
    )
    if args_cli.single_actuator_joint or any(abs(value) > 5.0 for value in effort_values):
        print(
            "[WARN] High/single-actuator efforts are diagnostic only. Do not use these settings for training.",
            flush=True,
        )
    if args_cli.diagnostic_disable_actuator_drives and args_cli.steps_per_test > 100:
        print(
            "[WARN] Diagnostic drive override requested with steps_per_test > 100. "
            "Keep this short for first force-transmission tests.",
            flush=True,
        )

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.debug_reward_terms = True
    env_cfg.log_full_pipeline_debug = True
    if args_cli.use_force_actuated_cylinders:
        if hasattr(env_cfg, "use_force_actuated_cylinders"):
            env_cfg.use_force_actuated_cylinders = True
            print("[INFO] Requested Kevin force-actuated cylinder config variant.", flush=True)
        else:
            print(
                "[WARN] --use_force_actuated_cylinders requested, but this env cfg has no "
                "use_force_actuated_cylinders field yet. No permanent config variant is active.",
                flush=True,
            )
    _apply_env_overrides(env_cfg)
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device

    apply_kevin_sim_memory_optimizations(env_cfg.sim, verbose=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args_cli.log_root) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "actuator_to_arm_transmission.csv"

    env = gym.make(args_cli.task, cfg=env_cfg)
    saved_drive_state = None
    try:
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)
        env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))
        unwrapped = env.unwrapped

        actuator_ids = _resolve_required_joint_ids(unwrapped.robot, actuator_names, label="actuator")
        mechanical_names, mechanical_ids = _resolve_optional_joint_ids(
            unwrapped.robot, mechanical_names_requested, label="mechanical"
        )
        body_names, body_ids = _resolve_body_ids(unwrapped.robot, body_names_requested)
        body_pose_available = _body_position_tensor(unwrapped.robot) is not None and bool(body_ids)
        print(f"[INFO] Number of mechanical joints requested: {len(mechanical_names_requested)}", flush=True)
        print(f"[INFO] Number of mechanical joints resolved: {len(mechanical_names)}", flush=True)
        print(f"[INFO] Number of body/link names requested: {len(body_names_requested)}", flush=True)
        print(f"[INFO] Number of body/link IDs resolved: {len(body_names)}", flush=True)
        print(f"[INFO] Body pose data available: {body_pose_available}", flush=True)
        _print_joint_drive_debug(unwrapped.robot, actuator_names, actuator_ids)
        if args_cli.diagnostic_disable_actuator_drives:
            saved_drive_state = _apply_diagnostic_drive_override(
                unwrapped,
                actuator_ids,
                args_cli.diagnostic_drive_stiffness,
                args_cli.diagnostic_drive_damping,
            )
            _print_joint_drive_debug(unwrapped.robot, actuator_names, actuator_ids)
        _install_direct_effort_override(unwrapped, actuator_ids)
        print("[INFO] Kevin control path bypass active: _pre_physics_step/_apply_action overridden", flush=True)

        zero_action = _make_zero_action(env)
        zero_effort = _make_effort(env, patterns["zero"], 0.0)
        csv_fields = _csv_fields(actuator_names, mechanical_names, body_names)
        summary_rows: list[dict[str, Any]] = []

        print(f"[INFO] CSV output: {csv_path}", flush=True)
        print("[INFO] Entering test loop.", flush=True)
        with csv_path.open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
            writer.writeheader()
            for effort_value in effort_values:
                for test_name, pattern_signs in patterns.items():
                    print(
                        f"[INFO] Running effort={effort_value:g} test={test_name}",
                        flush=True,
                    )
                    _reset_env(env)
                    unwrapped._direct_actuator_effort_command = zero_effort.detach().clone()  # noqa: SLF001
                    for _ in range(args_cli.settle_steps):
                        _, _, _, _ = env.step(zero_action)
                    actuator_start = _joint_values(unwrapped.robot, actuator_ids, "joint_pos")
                    mechanical_start = _joint_values(unwrapped.robot, mechanical_ids, "joint_pos")
                    body_start = _body_vectors(unwrapped.robot, body_ids)
                    effort_command = _make_effort(env, pattern_signs, effort_value)
                    if test_name == "zero":
                        effort_command.zero_()
                    unwrapped._direct_actuator_effort_command = effort_command.detach().clone()  # noqa: SLF001
                    test_rows: list[dict[str, float | str]] = []
                    for step in range(1, args_cli.steps_per_test + 1):
                        _, _, dones, _ = env.step(zero_action)
                        row = _collect_row(
                            unwrapped=unwrapped,
                            dones=dones,
                            step=step,
                            effort_value=effort_value,
                            test_name=test_name,
                            effort_command=effort_command,
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
                                f"[WARN] Stopping test early: effort={effort_value:g} "
                                f"test={test_name} step={step} reason={reason}",
                                flush=True,
                            )
                            break
                    csv_file.flush()
                    summary_row = _summarize(
                        effort_value=effort_value,
                        test_name=test_name,
                        rows=test_rows,
                        actuator_names=actuator_names,
                        actuator_ids=actuator_ids,
                        mechanical_names=mechanical_names,
                        body_names=body_names,
                        body_pose_available=body_pose_available,
                    )
                    summary_rows.append(summary_row)
                    _apply_zero_relative_deltas(summary_rows)
                    _print_result(summary_row)
                    print(f"[INFO] Finished effort={effort_value:g} test={test_name}", flush=True)

        print("[INFO] Finished all tests", flush=True)
        print(f"[INFO] CSV saved to path: {csv_path}", flush=True)
        _apply_zero_relative_deltas(summary_rows)
        print("[INFO] Summary:", flush=True)
        _print_summary(summary_rows)
        print(
            f"[INFO] Direct override call counts: _pre_physics_step={unwrapped._direct_actuator_pre_physics_calls} "
            f"_apply_action={unwrapped._direct_actuator_apply_action_calls}",
            flush=True,
        )
        _print_diagnosis(summary_rows, actuator_names)
        print(f"[INFO] Finished. Logged actuator-to-arm transmission diagnostic to {csv_path}", flush=True)
    finally:
        try:
            if "unwrapped" in locals():
                _restore_diagnostic_drive_override(unwrapped, actuator_ids if "actuator_ids" in locals() else [], saved_drive_state)
        except Exception as exc:
            print(f"[WARN] Diagnostic drive restore failed during cleanup: {exc}", flush=True)
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
