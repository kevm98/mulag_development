from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


parser = argparse.ArgumentParser(description="Run the learned Kevin integration controller on Unimog Mulag.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument(
    "--state_joints",
    nargs="+",
    default=[
        "Drehzapfen_joint",
        "Ausleger_I_joint",
        "Ausleger_II_joint",
        "Messerkopf_Schwenk_joint",
        "Messerkopf_joint",
    ],
    help="Five joints used for q, qdot, and qddot features.",
)
parser.add_argument(
    "--command_joints",
    nargs="+",
    default=[
        "Drehzapfen_joint",
        "Ausleger_I_joint",
        "Ausleger_II_joint",
        "Messerkopf_Schwenk_joint",
    ],
    help="Four joints receiving valve-derived commands.",
)
parser.add_argument(
    "--control_mode",
    choices=["position_delta", "velocity", "effort"],
    default="position_delta",
    help="How to apply the learned command to IsaacLab.",
)
parser.add_argument("--duration", type=float, default=0.0, help="Seconds to run. 0 means run until window closes.")
parser.add_argument("--position_scale", type=float, default=0.02, help="Position delta scale for valve command.")
parser.add_argument("--velocity_scale", type=float, default=0.2, help="Velocity target scale for valve command.")
parser.add_argument("--effort_scale", type=float, default=1.0, help="Multiplier for FD-predicted effort.")
parser.add_argument("--max_abs_effort", type=float, default=5000.0, help="Clamp for effort commands.")
parser.add_argument("--max_abs_valve", type=float, default=1.0, help="Clamp for valve command before applying.")
parser.add_argument("--print_interval", type=int, default=100, help="Print debug status every N sim steps.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

from kevin_integration.control_policy import ControlPolicy
from kevin_integration.utils.sim_memory import apply_kevin_sim_memory_optimizations
from rrlab_assets import MULAG_CFG


@configclass
class PlaneScene(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    robot = MULAG_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )


def find_joint_ids(robot, joint_names: list[str]) -> list[int]:
    ids = []
    for joint_name in joint_names:
        joint_ids, _ = robot.find_joints(joint_name)
        if len(joint_ids) == 0:
            raise RuntimeError(f"Could not find joint: {joint_name}")
        ids.append(joint_ids[0])
    return ids


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(np.float32, copy=False)


def build_aam_features(q: np.ndarray, qd: np.ndarray, dp: np.ndarray, valve_cmd: np.ndarray) -> np.ndarray:
    return np.concatenate([q, qd, dp, valve_cmd]).astype(np.float32)


def build_id_features(q: np.ndarray, qd: np.ndarray, qdd: np.ndarray, dp: np.ndarray) -> np.ndarray:
    return np.concatenate([q, qd, qdd, dp]).astype(np.float32)


def build_fd_features(q: np.ndarray, qd: np.ndarray, dp: np.ndarray, valve_cmd: np.ndarray) -> np.ndarray:
    return np.concatenate([q, qd, dp, valve_cmd]).astype(np.float32)


def clamp_to_soft_limits(robot, joint_ids: list[int], q_target: torch.Tensor) -> torch.Tensor:
    lower = robot.data.soft_joint_pos_limits[:, joint_ids, 0]
    upper = robot.data.soft_joint_pos_limits[:, joint_ids, 1]
    return torch.clamp(q_target, lower, upper)


def expand_actuator_torque_to_joint_effort(
    actuator_torque: np.ndarray,
    *,
    num_envs: int,
    device: torch.device,
    max_abs_effort: float,
    effort_scale: float,
) -> torch.Tensor:
    actuator_torque = effort_scale * np.asarray(actuator_torque, dtype=np.float32)
    actuator_torque = np.clip(actuator_torque, -max_abs_effort, max_abs_effort)
    actuator_torque_t = torch.as_tensor(actuator_torque, dtype=torch.float32, device=device)
    actuator_torque_t = actuator_torque_t.reshape(1, 4).repeat(num_envs, 1)

    joint_effort = torch.zeros((num_envs, 5), dtype=torch.float32, device=device)
    joint_effort[:, 0:4] = actuator_torque_t[:, 0:4]
    return joint_effort


def apply_command(
    robot,
    command_joint_ids: list[int],
    state_joint_ids: list[int],
    valve_cmd: np.ndarray,
    torque: np.ndarray | None,
    args: argparse.Namespace,
) -> None:
    valve_cmd = np.clip(valve_cmd, -args.max_abs_valve, args.max_abs_valve).astype(np.float32)
    device = robot.data.joint_pos.device
    num_envs = robot.data.joint_pos.shape[0]
    valve_t = torch.as_tensor(valve_cmd, dtype=torch.float32, device=device).unsqueeze(0).repeat(num_envs, 1)

    if args.control_mode == "position_delta":
        q_now = robot.data.joint_pos[:, command_joint_ids]
        q_target = clamp_to_soft_limits(robot, command_joint_ids, q_now + args.position_scale * valve_t)
        robot.set_joint_position_target(q_target, joint_ids=command_joint_ids)
        return

    if args.control_mode == "velocity":
        robot.set_joint_velocity_target(args.velocity_scale * valve_t, joint_ids=command_joint_ids)
        return

    if torque is None:
        raise RuntimeError("control_mode='effort' requires FD torque prediction.")
    if not hasattr(robot, "set_joint_effort_target"):
        raise RuntimeError("This IsaacLab Articulation does not expose set_joint_effort_target.")

    joint_effort = expand_actuator_torque_to_joint_effort(
        torque,
        num_envs=num_envs,
        device=device,
        max_abs_effort=args.max_abs_effort,
        effort_scale=args.effort_scale,
    )
    robot.set_joint_effort_target(joint_effort, joint_ids=state_joint_ids)


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, args: argparse.Namespace) -> None:
    robot = scene["robot"]
    print(f"[INFO] Actual robot joint names: {robot.joint_names}")

    state_joint_ids = find_joint_ids(robot, args.state_joints)
    command_joint_ids = find_joint_ids(robot, args.command_joints)
    if len(state_joint_ids) != 5:
        raise RuntimeError(f"Expected exactly 5 state joints for q/qdot/qddot features, got {len(state_joint_ids)}")
    if len(command_joint_ids) != 4:
        raise RuntimeError(f"Expected exactly 4 command joints for learned valve/torque output, got {len(command_joint_ids)}")

    print("[INFO] State joints:")
    for joint_name, joint_id in zip(args.state_joints, state_joint_ids):
        print(f"  {joint_name}: {joint_id}")

    print("[INFO] Command joints:")
    for joint_name, joint_id in zip(args.command_joints, command_joint_ids):
        print(f"  {joint_name}: {joint_id}")

    sim_dt = sim.get_physics_dt()
    max_steps = math.inf if args.duration <= 0.0 else int(args.duration / sim_dt)
    count = 0

    policy_device = args.device if args.device is not None else str(sim.device)
    policy = ControlPolicy(
        config={
            "pipeline": {"device": policy_device},
            "return_debug": True,
        }
    )

    dp = np.zeros(4, dtype=np.float32)
    prev_valve_cmd = np.zeros(4, dtype=np.float32)
    prev_qd = to_numpy(robot.data.joint_vel[0, state_joint_ids])

    while simulation_app.is_running() and count < max_steps:
        t = count * sim_dt
        q = to_numpy(robot.data.joint_pos[0, state_joint_ids])
        qd = to_numpy(robot.data.joint_vel[0, state_joint_ids])
        qdd = (qd - prev_qd) / max(sim_dt, 1e-8)

        policy_out = policy.compute_action(
            {
                "aam_x_t": build_aam_features(q, qd, dp, prev_valve_cmd),
                "id_x_t": build_id_features(q, qd, qdd, dp),
            }
        )
        valve_cmd = np.asarray(policy_out["valve_cmd"], dtype=np.float32)
        valve_cmd = np.nan_to_num(valve_cmd, nan=0.0, posinf=args.max_abs_valve, neginf=-args.max_abs_valve)

        torque = None
        if args.control_mode == "effort":
            fd_x_t = build_fd_features(q, qd, dp, valve_cmd)
            torque = policy.predict_torque(fd_x_t)
            torque = np.nan_to_num(torque, nan=0.0, posinf=args.max_abs_effort, neginf=-args.max_abs_effort)

        apply_command(robot, command_joint_ids, state_joint_ids, valve_cmd, torque, args)
        prev_valve_cmd = valve_cmd
        prev_qd = qd

        if args.print_interval > 0 and count % args.print_interval == 0:
            print(
                f"[{count:06d}] t={t:.2f}s "
                f"q={np.round(q, 3)} qdd={np.round(qdd, 3)} valve={np.round(valve_cmd, 3)}"
            )
            if torque is not None:
                print(f"           torque={np.round(torque, 3)}")

        scene.write_data_to_sim()
        sim.step()
        count += 1
        scene.update(sim_dt)


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device, render_interval=2)
    apply_kevin_sim_memory_optimizations(sim_cfg, verbose=True)

    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([11, 8, 1.3], [0.0, 0.0, 0.0])

    scene_cfg = PlaneScene(num_envs=args_cli.num_envs, env_spacing=20.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    print("[INFO] Setup complete.")
    run_simulator(sim, scene, args_cli)


if __name__ == "__main__":
    main()
    simulation_app.close()
