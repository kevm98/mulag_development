from __future__ import annotations

import argparse
import csv
import sys
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
CYLINDER_BODY_PAIRS = {
    "Drehzapfen_joint": ("Ausleger_Schwenkzylinder_Rohr", "Ausleger_Schwenkzylinder_Kolben"),
    "Ausleger_I_joint": ("Hebezylinder_Rohr", "Hebezylinder_Kolben"),
    "Ausleger_II_joint": ("Knickzylinder_Rohr", "Knickzylinder_Kolben"),
    "Messerkopf_Schwenk_joint": ("Messerkopf_Schwenkzylinder_Rohr", "Messerkopf_Schwenkzylinder_Kolben"),
}
MECHANICAL_PREFIXES = (
    "Drehzapfen_constraint_joint_",
    "Ausleger_I_constraint_joint_",
    "Ausleger_II_constraint_joint_",
    "Messerkopf_Schwenk_constraint_joint_",
)

parser = argparse.ArgumentParser(description="Manual Isaac-side cylinder coordinate calibration evidence collector.")
parser.add_argument("--task", type=str, default="Kevin-Mulag-AdaptiveRL-JointReach-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--single_actuator", type=str, default=None)
parser.add_argument("--position_offsets", type=str, default="-0.01,0.01")
parser.add_argument("--settle_steps", type=int, default=200)
parser.add_argument("--steps_per_target", type=int, default=100)
parser.add_argument("--output_dir", type=Path, default=ROOT / "logs" / "cylinder_coordinate_calibration")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import kevin_integration.tasks  # noqa: F401
from kevin_integration.utils.sim_memory import apply_kevin_sim_memory_optimizations


def _parse_float_csv(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one position offset.")
    return values


def _names(robot, field: str) -> list[str]:
    values = getattr(robot, field, None) or getattr(robot.data, field, None)
    return list(values) if values is not None else []


def _resolve_joint(robot, name: str) -> int:
    ids, _ = robot.find_joints(name)
    if len(ids) == 0:
        raise RuntimeError(f"Could not resolve joint {name!r}.")
    return int(ids[0])


def _resolve_body(robot, name: str) -> int | None:
    names = _names(robot, "body_names")
    if name in names:
        return names.index(name)
    try:
        ids, _ = robot.find_bodies(name)
        if len(ids) > 0:
            return int(ids[0])
    except Exception:
        return None
    return None


def _body_pos(robot, body_id: int | None) -> torch.Tensor | None:
    if body_id is None:
        return None
    value = getattr(robot.data, "body_pos_w", None)
    if isinstance(value, torch.Tensor) and value.ndim == 3 and body_id < value.shape[1]:
        return value[:, body_id, :3].detach().clone()
    return None


def _distance(a: torch.Tensor | None, b: torch.Tensor | None) -> float | None:
    if a is None or b is None:
        return None
    return float(torch.linalg.norm(a - b, dim=-1)[0].detach().cpu())


def _joint_limits(robot, joint_ids: list[int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    soft = getattr(robot.data, "soft_joint_pos_limits", None)
    if isinstance(soft, torch.Tensor):
        values = soft[0, joint_ids] if soft.ndim == 3 else soft[joint_ids]
        return values[:, 0].to(device), values[:, 1].to(device)
    return torch.full((len(joint_ids),), -torch.inf, device=device), torch.full((len(joint_ids),), torch.inf, device=device)


def _mechanical(robot) -> tuple[list[str], list[int]]:
    names = _names(robot, "joint_names")
    out_names: list[str] = []
    out_ids: list[int] = []
    for idx, name in enumerate(names):
        if any(name.startswith(prefix) for prefix in MECHANICAL_PREFIXES):
            out_names.append(name)
            out_ids.append(idx)
    return out_names, out_ids


def _disable_control_paths(env) -> None:
    cfg = getattr(env.unwrapped, "cfg", None)
    if cfg is not None:
        cfg.use_final_causal_cylinder_rom = False
        cfg.use_reduced_hydraulic_rom = False
        cfg.use_fd_effort = False
        cfg.use_sd_feedback = False
        cfg.use_sd_observer = False

    def noop_pre_physics(self, actions):
        if hasattr(self, "_current_action"):
            self._current_action.zero_()
        if hasattr(self, "_current_torque"):
            self._current_torque.zero_()

    def noop_apply_action(self):
        return None

    env.unwrapped._pre_physics_step = MethodType(noop_pre_physics, env.unwrapped)
    env.unwrapped._apply_action = MethodType(noop_apply_action, env.unwrapped)


def main() -> None:
    offsets = _parse_float_csv(args_cli.position_offsets)
    actuator_names = [args_cli.single_actuator] if args_cli.single_actuator else list(ACTUATOR_JOINT_NAMES)
    unknown = set(actuator_names) - set(ACTUATOR_JOINT_NAMES)
    if unknown:
        raise RuntimeError(f"Unknown actuator(s): {sorted(unknown)}")
    log_dir = args_cli.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / "cylinder_coordinate_calibration.csv"

    @hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
    def run(env_cfg, agent_cfg):
        del agent_cfg
        if int(args_cli.num_envs) != 1:
            raise RuntimeError("This calibration diagnostic supports --num_envs 1.")
        apply_kevin_sim_memory_optimizations(env_cfg.sim)
        env_cfg.scene.num_envs = 1
        env_cfg.use_final_causal_cylinder_rom = False
        env_cfg.use_reduced_hydraulic_rom = False
        env_cfg.use_fd_effort = False
        env_cfg.use_sd_feedback = False
        env_cfg.use_sd_observer = False
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)
        env = RslRlVecEnvWrapper(env)
        robot = env.unwrapped.robot
        device = env.unwrapped.device
        joint_ids = [_resolve_joint(robot, name) for name in ACTUATOR_JOINT_NAMES]
        lower, upper = _joint_limits(robot, joint_ids, device)
        body_pairs = {name: tuple(_resolve_body(robot, b) for b in CYLINDER_BODY_PAIRS[name]) for name in ACTUATOR_JOINT_NAMES}
        mechanical_names, mechanical_ids = _mechanical(robot)
        rows: list[dict[str, Any]] = []
        for actuator in actuator_names:
            actuator_index = ACTUATOR_JOINT_NAMES.index(actuator)
            for offset in offsets:
                obs, _ = env.reset()
                del obs
                _disable_control_paths(env)
                target = robot.data.joint_pos[:, joint_ids].detach().clone()
                for _ in range(args_cli.settle_steps):
                    robot.set_joint_position_target(target, joint_ids=joint_ids)
                    env.step(torch.zeros((1, 4), dtype=torch.float32, device=device))
                    target = robot.data.joint_pos[:, joint_ids].detach().clone()
                baseline_q = robot.data.joint_pos[:, joint_ids].detach().clone()
                barrel0 = _body_pos(robot, body_pairs[actuator][0])
                piston0 = _body_pos(robot, body_pairs[actuator][1])
                distance0 = _distance(barrel0, piston0)
                safe_offset = min(max(float(offset), float(lower[actuator_index] - baseline_q[0, actuator_index])), float(upper[actuator_index] - baseline_q[0, actuator_index]))
                target[:, actuator_index] = baseline_q[:, actuator_index] + safe_offset
                for _ in range(args_cli.steps_per_target):
                    robot.set_joint_position_target(target, joint_ids=joint_ids)
                    env.step(torch.zeros((1, 4), dtype=torch.float32, device=device))
                final_q = robot.data.joint_pos[:, joint_ids].detach().clone()
                barrel1 = _body_pos(robot, body_pairs[actuator][0])
                piston1 = _body_pos(robot, body_pairs[actuator][1])
                distance1 = _distance(barrel1, piston1)
                row: dict[str, Any] = {
                    "actuator_name": actuator,
                    "actuator_index": actuator_index,
                    "requested_offset": float(offset),
                    "applied_safe_offset": safe_offset,
                    "q_initial": float(baseline_q[0, actuator_index].detach().cpu()),
                    "q_final": float(final_q[0, actuator_index].detach().cpu()),
                    "delta_q": float((final_q - baseline_q)[0, actuator_index].detach().cpu()),
                    "length_proxy_initial": distance0,
                    "length_proxy_final": distance1,
                    "delta_length_proxy": None if distance0 is None or distance1 is None else distance1 - distance0,
                    "positive_q_increases_length_proxy": ""
                    if distance0 is None or distance1 is None or abs(float((final_q - baseline_q)[0, actuator_index])) <= 1.0e-12
                    else int(((distance1 - distance0) / float((final_q - baseline_q)[0, actuator_index])) > 0),
                }
                mech_pos = robot.data.joint_pos[:, mechanical_ids].detach().cpu()[0].tolist() if mechanical_ids else []
                for idx, name in enumerate(mechanical_names):
                    row[f"mechanical_{name}_position"] = float(mech_pos[idx])
                rows.append(row)
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
            writer.writeheader()
            writer.writerows(rows)
        print(f"PASS: wrote {csv_path}", flush=True)
        print("WARN: Isaac-side calibration does not determine Simscape scale/offset without matched Simscape reference data.", flush=True)
        env.close()

    run()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
