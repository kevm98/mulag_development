from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser(description="Inspect Mulag articulation joint order and Kevin env effort mapping.")
parser.add_argument("--task", type=str, default="Kevin-Mulag-AdaptiveRL-JointReach-v0", help="Gym task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of Isaac Lab environments.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg
from isaaclab_tasks.utils.hydra import hydra_task_config

import kevin_integration.tasks  # noqa: F401
from kevin_integration.utils.sim_memory import apply_kevin_sim_memory_optimizations


def _shape(value: Any) -> str:
    return str(tuple(value.shape)) if hasattr(value, "shape") else "unavailable"


def _tensor_row(value: Any, *, precision: int = 6) -> str:
    if not isinstance(value, torch.Tensor):
        return "unavailable"
    tensor = value.detach().cpu()
    if tensor.ndim >= 2:
        tensor = tensor[0]
    flat = tensor.flatten().tolist()
    return "[" + ", ".join(f"{float(item):.{precision}g}" for item in flat) + "]"


def _tensor_rows(value: Any, *, precision: int = 6) -> str:
    if not isinstance(value, torch.Tensor):
        return "unavailable"
    tensor = value.detach().cpu()
    if tensor.ndim >= 3:
        tensor = tensor[0]
    if tensor.ndim == 1:
        return _tensor_row(tensor, precision=precision)
    rows = []
    for row in tensor:
        rows.append("[" + ", ".join(f"{float(item):.{precision}g}" for item in row.flatten().tolist()) + "]")
    return "[" + ", ".join(rows) + "]"


def _print_tensor_field(label: str, value: Any) -> None:
    print(f"[INFO] {label} shape: {_shape(value)}", flush=True)
    print(f"[INFO] {label} env0: {_tensor_rows(value)}", flush=True)


def _print_optional_tensor_fields(data: Any, field_names: tuple[str, ...], label: str) -> None:
    found = False
    for field_name in field_names:
        value = getattr(data, field_name, None)
        if isinstance(value, torch.Tensor):
            found = True
            _print_tensor_field(f"robot.data.{field_name} ({label})", value)
    if not found:
        print(f"[INFO] {label}: no tensor field found among {field_names}", flush=True)


def _get_joint_names(robot) -> list[str]:
    names = getattr(robot, "joint_names", None)
    if names is None:
        names = getattr(robot.data, "joint_names", None)
    if names is None:
        return []
    return list(names)


def _resolve_joint(robot, joint_name: str) -> tuple[int | None, str | None]:
    joint_ids, resolved_names = robot.find_joints(joint_name)
    if len(joint_ids) == 0:
        return None, None
    joint_id = int(joint_ids[0])
    resolved_name = resolved_names[0] if resolved_names else None
    if resolved_name is None:
        joint_names = _get_joint_names(robot)
        if joint_id < len(joint_names):
            resolved_name = joint_names[joint_id]
    return joint_id, resolved_name


def _print_actuator_info(robot) -> None:
    actuator_names = getattr(robot, "actuator_names", None)
    if actuator_names is not None:
        print(f"[INFO] robot.actuator_names: {list(actuator_names)}", flush=True)
    actuators = getattr(robot, "actuators", None)
    if isinstance(actuators, dict):
        print(f"[INFO] robot.actuators keys: {list(actuators.keys())}", flush=True)
        for name, actuator in actuators.items():
            print(f"  actuator={name} type={type(actuator).__name__}", flush=True)
            for attr in ("joint_names", "joint_indices", "joint_ids", "effort_limit", "velocity_limit", "stiffness", "damping"):
                if hasattr(actuator, attr):
                    print(f"    {attr}={getattr(actuator, attr)}", flush=True)
    cfg_actuators = getattr(getattr(robot, "cfg", None), "actuators", None)
    if cfg_actuators is not None:
        print(f"[INFO] robot.cfg.actuators: {cfg_actuators}", flush=True)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device
    apply_kevin_sim_memory_optimizations(env_cfg.sim, verbose=True)

    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)
        unwrapped = env.unwrapped
        robot = unwrapped.robot
        joint_names = _get_joint_names(robot)

        print("[INFO] Full robot articulation joint names in Isaac order:", flush=True)
        for index, joint_name in enumerate(joint_names):
            print(f"  [{index}] {joint_name}", flush=True)

        _print_tensor_field("robot.data.joint_pos", getattr(robot.data, "joint_pos", None))
        _print_tensor_field("robot.data.joint_vel", getattr(robot.data, "joint_vel", None))
        _print_tensor_field("robot.data.default_joint_pos", getattr(robot.data, "default_joint_pos", None))
        _print_tensor_field("robot.data.soft_joint_pos_limits", getattr(robot.data, "soft_joint_pos_limits", None))

        _print_optional_tensor_fields(
            robot.data,
            ("joint_effort_limits", "soft_joint_effort_limits", "effort_limits"),
            "effort limits",
        )
        _print_optional_tensor_fields(
            robot.data,
            ("joint_velocity_limits", "soft_joint_velocity_limits", "joint_vel_limits", "soft_joint_vel_limits"),
            "velocity limits",
        )
        _print_optional_tensor_fields(
            robot.data,
            ("joint_stiffness", "default_joint_stiffness", "stiffness"),
            "drive stiffness",
        )
        _print_optional_tensor_fields(
            robot.data,
            ("joint_damping", "default_joint_damping", "damping"),
            "drive damping",
        )
        _print_actuator_info(robot)

        configured_state_names = list(getattr(unwrapped.cfg, "state_joint_names", []))
        configured_command_names = list(getattr(unwrapped.cfg, "command_joint_names", []))
        state_ids = list(getattr(unwrapped, "_state_joint_ids", []))
        command_ids = list(getattr(unwrapped, "_command_joint_ids", []))
        print(f"[INFO] Kevin configured state_joint_names: {configured_state_names}", flush=True)
        print(f"[INFO] Kevin configured state_joint_ids: {state_ids}", flush=True)
        print(f"[INFO] Kevin configured actuated command_joint_names: {configured_command_names}", flush=True)
        print(f"[INFO] Kevin configured actuated command_joint_ids: {command_ids}", flush=True)

        print("[INFO] Explicit runtime arm joint name resolution:", flush=True)
        for joint_name in configured_state_names:
            resolved_id, resolved_name = _resolve_joint(robot, joint_name)
            print(
                f"  requested={joint_name} isaac_joint_id={resolved_id} actual={resolved_name}",
                flush=True,
            )

        print("[INFO] Action-index to Isaac joint mapping used by Kevin env:", flush=True)
        for action_index, expected_name in enumerate(configured_command_names):
            actual_id = command_ids[action_index] if action_index < len(command_ids) else None
            actual_name = joint_names[actual_id] if actual_id is not None and actual_id < len(joint_names) else None
            print(
                f"  action_index={action_index} expected={expected_name} "
                f"isaac_joint_id={actual_id} actual={actual_name}",
                flush=True,
            )

        drehzapfen_id, drehzapfen_name = _resolve_joint(robot, "Drehzapfen_joint")
        print(
            f"[INFO] Drehzapfen_joint exists={drehzapfen_id is not None} "
            f"isaac_joint_id={drehzapfen_id} actual_name={drehzapfen_name}",
            flush=True,
        )
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
