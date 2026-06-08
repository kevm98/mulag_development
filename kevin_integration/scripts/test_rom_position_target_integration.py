from __future__ import annotations

import argparse
import csv
import math
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from types import MethodType

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
]
BODY_KEYWORDS = ("Drehzapfen", "Ausleger", "Messerkopf", "Zylinder", "Kolben", "Rohr")

parser = argparse.ArgumentParser(description="ROM-to-prismatic-position-target integration diagnostic.")
parser.add_argument("--task", type=str, default="Kevin-Mulag-AdaptiveRL-JointReach-v0", help="Gym task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of Isaac Lab environments.")
parser.add_argument("--steps_per_test", type=int, default=200, help="Steps per ROM test.")
parser.add_argument("--settle_steps", type=int, default=500, help="Zero-valve settle steps before each test.")
parser.add_argument("--valve_commands", type=str, default="0.1,0.3,0.5", help="Comma-separated valve command magnitudes.")
parser.add_argument(
    "--actuator_joint_names",
    type=str,
    default=",".join(DEFAULT_ACTUATOR_JOINT_NAMES),
    help="Comma-separated actuator-side prismatic joint names.",
)
parser.add_argument(
    "--mechanical_joint_names",
    type=str,
    default=",".join(DEFAULT_MECHANICAL_JOINT_NAMES),
    help="Comma-separated mechanical candidate joints to log.",
)
parser.add_argument(
    "--body_names",
    type=str,
    default=",".join(DEFAULT_BODY_NAMES),
    help="Comma-separated body/link names to log; keyword-matched bodies are auto-included.",
)
parser.add_argument("--reference_body", type=str, default="Frontrahmen", help="Reference body for summary interpretation.")
parser.add_argument("--motion_threshold", type=float, default=1.0e-5, help="Absolute motion floor for classification.")
parser.add_argument("--min_dx_from_zero", type=float, default=1.0e-5, help="Minimum ROM dx response beyond zero drift.")
parser.add_argument(
    "--min_actuator_delta_from_zero",
    type=float,
    default=1.0e-4,
    help="Minimum actuator motion beyond zero drift.",
)
parser.add_argument(
    "--min_expected_mech_from_zero",
    type=float,
    default=1.0e-4,
    help="Minimum expected mechanical-family motion beyond zero drift.",
)
parser.add_argument(
    "--max_target_tracking_error",
    type=float,
    default=0.02,
    help="Maximum allowed actuator target tracking error.",
)
parser.add_argument(
    "--opposite_sign_required",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Require positive/negative responses to have opposite signs.",
)
parser.add_argument(
    "--rom_synthetic_dx_scale",
    type=float,
    default=1.0,
    help="Scale ROM dx before integrating targets. Diagnostic only.",
)
parser.add_argument(
    "--rom_override_dx_per_step",
    type=float,
    default=None,
    help="Override ROM dx with sign(valve_cmd) * this value. Validates target plumbing only.",
)
parser.add_argument(
    "--log_root",
    type=str,
    default="logs/rom_position_target_integration",
    help="Directory under which timestamped ROM logs are written.",
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
from kevin_integration.controllers.reduced_hydraulic_actuator import (
    ReducedHydraulicActuatorROM,
    ReducedHydraulicActuatorROMCfg,
)
from kevin_integration.scripts.analyze_rom_position_target_integration_csv import (
    analyze_rows,
    print_global_conclusion,
    print_table,
)
from kevin_integration.utils.sim_memory import apply_kevin_sim_memory_optimizations


def _parse_csv(text: str) -> list[str]:
    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("CSV argument must contain at least one value.")
    return values


def _parse_float_csv(text: str) -> list[float]:
    return [float(item) for item in _parse_csv(text)]


def _safe_name(name: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in name).strip("_")


def _field(prefix: str, name: str, suffix: str) -> str:
    return f"{prefix}_{_safe_name(name)}_{suffix}"


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


def _resolve_required_joint_ids(robot, joint_names: Sequence[str], label: str) -> list[int]:
    all_names = _robot_joint_names(robot)
    ids = []
    for name in joint_names:
        joint_ids, resolved_names = robot.find_joints(name)
        if len(joint_ids) == 0:
            raise RuntimeError(f"Could not resolve {label} joint {name!r}.")
        joint_id = int(joint_ids[0])
        actual = resolved_names[0] if resolved_names else all_names[joint_id] if joint_id < len(all_names) else name
        ids.append(joint_id)
        print(f"[INFO] Resolved {label}: requested={name} id={joint_id} actual={actual}", flush=True)
    return ids


def _resolve_optional_joint_ids(robot, joint_names: Sequence[str], label: str) -> tuple[list[str], list[int]]:
    all_names = _robot_joint_names(robot)
    names_out: list[str] = []
    ids_out: list[int] = []
    for name in joint_names:
        joint_ids, resolved_names = robot.find_joints(name)
        if len(joint_ids) == 0:
            print(f"[WARN] {label} joint not found: {name}; skipping.", flush=True)
            continue
        joint_id = int(joint_ids[0])
        actual = resolved_names[0] if resolved_names else all_names[joint_id] if joint_id < len(all_names) else name
        names_out.append(name)
        ids_out.append(joint_id)
        print(f"[INFO] Resolved {label}: requested={name} id={joint_id} actual={actual}", flush=True)
    return names_out, ids_out


def _resolve_body_ids(robot, requested_names: Sequence[str]) -> tuple[list[str], list[int]]:
    all_names = _robot_body_names(robot)
    resolved: dict[str, int] = {}
    for name in requested_names:
        if name in all_names:
            resolved[name] = all_names.index(name)
    for body_id, name in enumerate(all_names):
        if any(keyword.lower() in name.lower() for keyword in BODY_KEYWORDS):
            resolved.setdefault(name, body_id)
    for name, body_id in resolved.items():
        print(f"[INFO] Resolved body: {name} id={body_id}", flush=True)
    return list(resolved.keys()), list(resolved.values())


def _joint_values(robot, joint_ids: Sequence[int], field_name: str) -> list[float]:
    tensor = getattr(robot.data, field_name).detach()
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


def _delta_norm(delta: tuple[float, float, float]) -> float:
    if not all(math.isfinite(value) for value in delta):
        return math.nan
    return math.sqrt(sum(value * value for value in delta))


def _mean_tensor(value: torch.Tensor) -> float:
    return float(value.detach().float().mean().cpu()) if isinstance(value, torch.Tensor) and value.numel() else math.nan


def _make_rom_cfg(env_cfg) -> ReducedHydraulicActuatorROMCfg:
    return ReducedHydraulicActuatorROMCfg(
        output_mode=getattr(env_cfg, "rom_output_mode", "velocity_to_position_target"),
        Abar=tuple(float(value) for value in getattr(env_cfg, "rom_Abar", [1.0e-3] * 4)),
        Vt=tuple(float(value) for value in getattr(env_cfg, "rom_Vt", [1.0e-3] * 4)),
        E=tuple(float(value) for value in getattr(env_cfg, "rom_E", [1.0e8] * 4)),
        Kq=tuple(float(value) for value in getattr(env_cfg, "rom_Kq", [1.0e-8] * 4)),
        CL=tuple(float(value) for value in getattr(env_cfg, "rom_CL", [0.0] * 4)),
        mu=tuple(float(value) for value in getattr(env_cfg, "rom_mu", [10.0] * 4)),
        Fs=tuple(float(value) for value in getattr(env_cfg, "rom_Fs", [10.0] * 4)),
        kf=tuple(float(value) for value in getattr(env_cfg, "rom_kf", [100.0] * 4)),
        Pmax=tuple(float(value) for value in getattr(env_cfg, "rom_Pmax", [1.0e7] * 4)),
        xdot_max=tuple(float(value) for value in getattr(env_cfg, "rom_xdot_max", [0.02] * 4)),
        dx_max=tuple(float(value) for value in getattr(env_cfg, "rom_dx_max", [5.0e-4] * 4)),
        delay_steps=tuple(int(value) for value in getattr(env_cfg, "rom_delay_steps", [0] * 4)),
        udz_pos=tuple(float(value) for value in getattr(env_cfg, "rom_udz_pos", [0.0] * 4)),
        udz_neg=tuple(float(value) for value in getattr(env_cfg, "rom_udz_neg", [0.0] * 4)),
        velocity_gain=tuple(float(value) for value in getattr(env_cfg, "rom_velocity_gain", [0.02] * 4)),
        force_scale=tuple(float(value) for value in getattr(env_cfg, "rom_force_scale", [1000.0] * 4)),
    )


def _install_rom_override(
    unwrapped,
    rom: ReducedHydraulicActuatorROM,
    actuator_ids: Sequence[int],
    dt: float,
    *,
    dx_scale: float,
    override_dx_per_step: float | None,
) -> None:
    unwrapped._rom_diag_actuator_joint_ids = list(actuator_ids)  # noqa: SLF001
    unwrapped._rom_diag_valve_cmd = torch.zeros((unwrapped.num_envs, 4), dtype=torch.float32, device=unwrapped.device)  # noqa: SLF001
    unwrapped._rom_diag_pressure = torch.zeros_like(unwrapped._rom_diag_valve_cmd)  # noqa: SLF001
    unwrapped._rom_diag_target = unwrapped.robot.data.joint_pos[:, actuator_ids].detach().clone()  # noqa: SLF001
    unwrapped._rom_diag_target_start = unwrapped._rom_diag_target.detach().clone()  # noqa: SLF001
    unwrapped._rom_diag_xdot = torch.zeros_like(unwrapped._rom_diag_valve_cmd)  # noqa: SLF001
    unwrapped._rom_diag_dx = torch.zeros_like(unwrapped._rom_diag_valve_cmd)  # noqa: SLF001
    unwrapped._rom_diag_pressure_clamp_fraction = torch.zeros((unwrapped.num_envs,), dtype=torch.float32, device=unwrapped.device)  # noqa: SLF001
    unwrapped._rom_diag_target_clamp_fraction = torch.zeros((unwrapped.num_envs,), dtype=torch.float32, device=unwrapped.device)  # noqa: SLF001
    unwrapped._rom_diag_nan_guard = torch.zeros_like(unwrapped._rom_diag_target_clamp_fraction)  # noqa: SLF001
    unwrapped._rom_diag_synthetic_dx_active = override_dx_per_step is not None  # noqa: SLF001
    if override_dx_per_step is not None:
        print(
            "[WARN] Synthetic dx override active. This validates target plumbing only, not hydraulic ROM physics.",
            flush=True,
        )

    def _direct_pre_physics_step(self, actions: torch.Tensor) -> None:
        x = self.robot.data.joint_pos[:, self._rom_diag_actuator_joint_ids]
        xdot = self.robot.data.joint_vel[:, self._rom_diag_actuator_joint_ids]
        out = rom.step(
            valve_cmd=self._rom_diag_valve_cmd,
            x_cyl=x,
            xdot_cyl=xdot,
            P_L=self._rom_diag_pressure,
            dt=dt,
        )
        self._rom_diag_pressure = out.P_L_next.detach().clone()
        self._rom_diag_xdot = out.xdot_cyl_pred.detach().clone()
        dx = out.dx_cyl_pred.detach().clone() * float(dx_scale)
        if override_dx_per_step is not None:
            dx = torch.sign(self._rom_diag_valve_cmd) * abs(float(override_dx_per_step))
        self._rom_diag_dx = dx
        candidate = self._rom_diag_target + self._rom_diag_dx
        candidate = torch.nan_to_num(candidate, nan=0.0, posinf=0.0, neginf=0.0)
        lower = self.robot.data.soft_joint_pos_limits[:, self._rom_diag_actuator_joint_ids, 0]
        upper = self.robot.data.soft_joint_pos_limits[:, self._rom_diag_actuator_joint_ids, 1]
        clamp_mask = (candidate < lower) | (candidate > upper)
        self._rom_diag_target = torch.clamp(candidate, lower, upper).detach().clone()
        self._rom_diag_pressure_clamp_fraction = out.debug["rom_pressure_clamp_fraction"].detach().clone()
        self._rom_diag_target_clamp_fraction = torch.mean(clamp_mask.float(), dim=-1)
        self._rom_diag_nan_guard = out.debug["rom_nan_guard_triggered"].detach().clone()
        _ = actions

    def _direct_apply_action(self) -> None:
        self.robot.set_joint_position_target(self._rom_diag_target, joint_ids=self._rom_diag_actuator_joint_ids)

    unwrapped._pre_physics_step = MethodType(_direct_pre_physics_step, unwrapped)  # noqa: SLF001
    unwrapped._apply_action = MethodType(_direct_apply_action, unwrapped)  # noqa: SLF001
    print("[INFO] Kevin control path bypass active: ROM position-target override installed", flush=True)


def _csv_fields(actuator_names: Sequence[str], mechanical_names: Sequence[str], body_names: Sequence[str]) -> list[str]:
    fields = [
        "test_name",
        "valve_magnitude",
        "valve_command",
        "applied_actuator_name",
        "step",
        "done_frac",
        "rom_pressure_clamp_fraction",
        "rom_target_clamp_fraction",
        "rom_nan_guard_triggered",
        "rom_synthetic_dx_active",
    ]
    for name in actuator_names:
        safe = _safe_name(name)
        fields.extend(
            [
                f"valve_{safe}",
                f"rom_P_L_{safe}",
                f"rom_xdot_pred_{safe}",
                f"rom_dx_pred_{safe}",
                f"actuator_{safe}_target",
                f"actuator_{safe}_pos",
                f"actuator_{safe}_vel",
                f"actuator_{safe}_target_error",
                f"actuator_{safe}_delta_from_start",
                f"actuator_{safe}_target_delta_from_start",
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
                _field("body", name, "translation_delta_norm"),
                _field("body", name, "relative_to_reference_delta_norm"),
            ]
        )
    return fields


def _collect_row(
    *,
    unwrapped,
    dones: torch.Tensor,
    test_name: str,
    valve_magnitude: float,
    valve_command: float,
    step: int,
    actuator_names: Sequence[str],
    actuator_ids: Sequence[int],
    actuator_start: Sequence[float],
    mechanical_names: Sequence[str],
    mechanical_ids: Sequence[int],
    mechanical_start: Sequence[float],
    body_names: Sequence[str],
    body_ids: Sequence[int],
    body_start: Sequence[tuple[float, float, float]],
    reference_body: str,
) -> dict[str, float | str]:
    actuator_pos = _joint_values(unwrapped.robot, actuator_ids, "joint_pos")
    actuator_vel = _joint_values(unwrapped.robot, actuator_ids, "joint_vel")
    mechanical_pos = _joint_values(unwrapped.robot, mechanical_ids, "joint_pos")
    mechanical_vel = _joint_values(unwrapped.robot, mechanical_ids, "joint_vel")
    body_pos = _body_vectors(unwrapped.robot, body_ids)
    applied_actuator_name = "none"
    active = torch.nonzero(torch.abs(unwrapped._rom_diag_valve_cmd[0]) > 0.0, as_tuple=False).flatten()
    if active.numel() == 1:
        applied_actuator_name = actuator_names[int(active[0])]
    elif active.numel() > 1:
        applied_actuator_name = "all"
    reference_delta = (0.0, 0.0, 0.0)
    if reference_body in body_names:
        reference_index = body_names.index(reference_body)
        reference_delta = (
            body_pos[reference_index][0] - body_start[reference_index][0],
            body_pos[reference_index][1] - body_start[reference_index][1],
            body_pos[reference_index][2] - body_start[reference_index][2],
        )
    row: dict[str, float | str] = {
        "test_name": test_name,
        "valve_magnitude": float(valve_magnitude),
        "valve_command": float(valve_command),
        "applied_actuator_name": applied_actuator_name,
        "step": float(step),
        "done_frac": _mean_tensor(dones.float()),
        "rom_pressure_clamp_fraction": _mean_tensor(unwrapped._rom_diag_pressure_clamp_fraction),
        "rom_target_clamp_fraction": _mean_tensor(unwrapped._rom_diag_target_clamp_fraction),
        "rom_nan_guard_triggered": _mean_tensor(unwrapped._rom_diag_nan_guard),
        "rom_synthetic_dx_active": float(bool(unwrapped._rom_diag_synthetic_dx_active)),
    }
    for index, name in enumerate(actuator_names):
        safe = _safe_name(name)
        row[f"valve_{safe}"] = float(unwrapped._rom_diag_valve_cmd[0, index].detach().cpu())
        row[f"rom_P_L_{safe}"] = float(unwrapped._rom_diag_pressure[0, index].detach().cpu())
        row[f"rom_xdot_pred_{safe}"] = float(unwrapped._rom_diag_xdot[0, index].detach().cpu())
        row[f"rom_dx_pred_{safe}"] = float(unwrapped._rom_diag_dx[0, index].detach().cpu())
        target = float(unwrapped._rom_diag_target[0, index].detach().cpu())
        target_start = float(unwrapped._rom_diag_target_start[0, index].detach().cpu())
        row[f"actuator_{safe}_target"] = target
        row[f"actuator_{safe}_pos"] = actuator_pos[index]
        row[f"actuator_{safe}_vel"] = actuator_vel[index]
        row[f"actuator_{safe}_target_error"] = target - actuator_pos[index]
        row[f"actuator_{safe}_delta_from_start"] = actuator_pos[index] - actuator_start[index]
        row[f"actuator_{safe}_target_delta_from_start"] = target - target_start
    for name, pos, vel, start in zip(mechanical_names, mechanical_pos, mechanical_vel, mechanical_start, strict=True):
        row[_field("mech", name, "pos")] = pos
        row[_field("mech", name, "vel")] = vel
        row[_field("mech", name, "delta_from_start")] = pos - start
    for name, pos, start in zip(body_names, body_pos, body_start, strict=True):
        delta = (pos[0] - start[0], pos[1] - start[1], pos[2] - start[2])
        relative_delta = (
            delta[0] - reference_delta[0],
            delta[1] - reference_delta[1],
            delta[2] - reference_delta[2],
        )
        row[_field("body", name, "x")] = pos[0]
        row[_field("body", name, "y")] = pos[1]
        row[_field("body", name, "z")] = pos[2]
        row[_field("body", name, "translation_delta_norm")] = _delta_norm(
            delta
        )
        row[_field("body", name, "relative_to_reference_delta_norm")] = _delta_norm(relative_delta)
    return row


def _expected_keywords(actuator_name: str) -> tuple[str, ...]:
    if actuator_name == "Drehzapfen_joint":
        return ("Drehzapfen",)
    return (actuator_name.replace("_joint", ""),)


def _max_expected_mech(row: dict[str, float | str], mechanical_names: Sequence[str], actuator_name: str) -> tuple[str, float]:
    keywords = _expected_keywords(actuator_name)
    best_name = "none"
    best_value = 0.0
    for name in mechanical_names:
        if not any(keyword.lower() in name.lower() for keyword in keywords):
            continue
        value = float(row.get(_field("mech", name, "delta_from_start"), 0.0))
        if abs(value) > abs(best_value):
            best_name = name
            best_value = value
    return best_name, best_value


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.use_reduced_hydraulic_rom = True
    env_cfg.use_fd_effort = False
    env_cfg.use_sd_feedback = False
    env_cfg.control_mode = "position_delta"
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device
    apply_kevin_sim_memory_optimizations(env_cfg.sim, verbose=True)

    actuator_names = _parse_csv(args_cli.actuator_joint_names)
    mechanical_requested = _parse_csv(args_cli.mechanical_joint_names)
    body_requested = _parse_csv(args_cli.body_names)
    valve_commands = _parse_float_csv(args_cli.valve_commands)
    patterns: list[tuple[float, str, torch.Tensor, float]] = []
    for command in valve_commands:
        patterns.append((abs(command), "zero", torch.zeros((1, 4), dtype=torch.float32), 0.0))
        for index, name in enumerate(actuator_names):
            pos = torch.zeros((1, 4), dtype=torch.float32)
            neg = torch.zeros((1, 4), dtype=torch.float32)
            pos[:, index] = command
            neg[:, index] = -command
            patterns.append((abs(command), f"{name}_positive_valve", pos, command))
            patterns.append((abs(command), f"{name}_negative_valve", neg, -command))
        all_pos = command * torch.ones((1, 4), dtype=torch.float32)
        all_neg = -command * torch.ones((1, 4), dtype=torch.float32)
        patterns.append((abs(command), "all_positive_valve", all_pos, command))
        patterns.append((abs(command), "all_negative_valve", all_neg, -command))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args_cli.log_root) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "rom_position_target_integration.csv"

    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)
        env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))
        unwrapped = env.unwrapped
        actuator_ids = _resolve_required_joint_ids(unwrapped.robot, actuator_names, "actuator")
        mechanical_names, mechanical_ids = _resolve_optional_joint_ids(unwrapped.robot, mechanical_requested, "mechanical")
        body_names, body_ids = _resolve_body_ids(unwrapped.robot, body_requested)
        rom = ReducedHydraulicActuatorROM(
            num_envs=env.num_envs,
            cfg=_make_rom_cfg(env_cfg),
            device=env.device,
        )
        dt = float(getattr(env_cfg, "rom_dt", 0.01)) or float(getattr(unwrapped, "step_dt", 0.01))
        _install_rom_override(unwrapped, rom, actuator_ids, dt)

        zero_action = torch.zeros((env.num_envs, env.num_actions), dtype=torch.float32, device=env.device)
        csv_fields = _csv_fields(actuator_names, mechanical_names, body_names)
        summary_rows: list[dict[str, float | str]] = []
        print(f"[INFO] CSV output: {csv_path}", flush=True)
        with csv_path.open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
            writer.writeheader()
            for valve_magnitude, test_name, valve_template, valve_command in patterns:
                print(
                    f"[INFO] Running valve_magnitude={valve_magnitude:g} test={test_name} "
                    f"valve_command={valve_command:g}",
                    flush=True,
                )
                env.reset()
                rom.reset()
                unwrapped._rom_diag_pressure.zero_()
                unwrapped._rom_diag_valve_cmd.zero_()
                unwrapped._rom_diag_pressure_clamp_fraction.zero_()
                unwrapped._rom_diag_target_clamp_fraction.zero_()
                unwrapped._rom_diag_nan_guard.zero_()
                unwrapped._rom_diag_target = unwrapped.robot.data.joint_pos[:, actuator_ids].detach().clone()
                for _ in range(args_cli.settle_steps):
                    env.step(zero_action)
                actuator_start = _joint_values(unwrapped.robot, actuator_ids, "joint_pos")
                mechanical_start = _joint_values(unwrapped.robot, mechanical_ids, "joint_pos")
                body_start = _body_vectors(unwrapped.robot, body_ids)
                unwrapped._rom_diag_target = unwrapped.robot.data.joint_pos[:, actuator_ids].detach().clone()
                unwrapped._rom_diag_target_start = unwrapped._rom_diag_target.detach().clone()
                unwrapped._rom_diag_valve_cmd = valve_template.to(env.device).repeat(env.num_envs, 1)
                last_row: dict[str, float | str] | None = None
                for step in range(1, args_cli.steps_per_test + 1):
                    _, _, dones, _ = env.step(zero_action)
                    row = _collect_row(
                        unwrapped=unwrapped,
                        dones=dones,
                        test_name=test_name,
                        valve_magnitude=valve_magnitude,
                        valve_command=valve_command,
                        step=step,
                        actuator_names=actuator_names,
                        actuator_ids=actuator_ids,
                        actuator_start=actuator_start,
                        mechanical_names=mechanical_names,
                        mechanical_ids=mechanical_ids,
                        mechanical_start=mechanical_start,
                        body_names=body_names,
                        body_ids=body_ids,
                        body_start=body_start,
                        reference_body=args_cli.reference_body,
                    )
                    writer.writerow(row)
                    last_row = row
                    if torch.any(dones):
                        print(f"[WARN] Done flag during test={test_name} step={step}; stopping this test.", flush=True)
                        break
                if last_row is not None:
                    summary_rows.append(last_row)
            csv_file.flush()

        print("[INFO] Strict zero/positive/negative validation summary:", flush=True)
        analyzed = analyze_rows(summary_rows, args_cli)
        print_table(analyzed)
        print_global_conclusion(analyzed)
        print(f"[INFO] CSV saved to: {csv_path}", flush=True)
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
