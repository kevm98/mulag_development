from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_HYDRAULIC_ROM_TASK = "Kevin-Mulag-HydraulicROM-TaskSpace-v0"

parser = argparse.ArgumentParser(description="Play the Kevin/Mulag hydraulic ROM joint-tracking task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of Isaac Lab environments.")
parser.add_argument("--task", type=str, default=DEFAULT_HYDRAULIC_ROM_TASK, help="Gym task name.")
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Optional RSL-RL checkpoint path. Omit it for the scripted visible reference demo.",
)
parser.add_argument("--steps", type=int, default=-1, help="Number of policy steps to run. Use -1 to run until closed.")
parser.add_argument(
    "--play_mode",
    choices=["joint_space_validation", "rl_task_space_tracking"],
    default=None,
    help="Optional override. Task-space tracking requires --checkpoint.",
)
parser.add_argument(
    "--enable_vehicle_motion",
    action="store_true",
    default=False,
    help="Enable RRLAB-style scripted vehicle motion for task-space play.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.checkpoint is not None and not Path(args_cli.checkpoint).is_file():
    raise FileNotFoundError(f"Checkpoint does not exist: {args_cli.checkpoint}")

sys.argv = [sys.argv[0], *hydra_args]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

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


def _as_list(value, precision: int = 4) -> list[float]:
    if isinstance(value, torch.Tensor):
        return [round(float(x), precision) for x in value.detach().flatten().cpu().tolist()]
    return []


def _scalar(value, default: float = float("nan")) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().flatten()[0].cpu())
    try:
        return float(value)
    except Exception:
        return default


def _roll_pitch_from_wxyz(quat: torch.Tensor | None) -> tuple[float, float]:
    if not isinstance(quat, torch.Tensor) or quat.numel() != 4:
        return float("nan"), float("nan")
    w, x, y, z = [float(v) for v in quat.detach().flatten().cpu().tolist()]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    return roll, pitch


def _vector_stats(value) -> tuple[float, float, float, float, float]:
    if not isinstance(value, torch.Tensor) or value.numel() == 0:
        nan = float("nan")
        return nan, nan, nan, nan, nan
    data = value.detach().float().flatten().cpu()
    saturation = torch.mean((torch.abs(data) >= 0.99).float())
    return (
        float(torch.mean(data)),
        float(torch.std(data, unbiased=False)),
        float(torch.min(data)),
        float(torch.max(data)),
        float(saturation),
    )


class RuntimeSignalStats:
    VECTOR_KEYS = (
        "q_ref_commanded",
        "q_actual",
        "q_error",
        "qdot_ref",
        "qddot_cmd",
        "id_input_normalized",
        "valve_cmd",
        "final_rom_input_features_normalized",
        "final_rom_delta_extension_kinematic_50ms",
        "final_rom_delta_extension_correction_50ms",
        "final_rom_delta_extension_50ms",
        "final_rom_delta_isaac_50ms",
        "final_rom_delta_isaac_10ms",
        "rom_cylinder_pos_target",
        "rom_cylinder_cumulative_target_motion",
        "rom_cylinder_cumulative_measured_motion",
        "rom_cylinder_tracking_ratio",
        "final_rom_extension_velocity",
        "dP",
    )
    SCALAR_KEYS = (
        "id_input_normalized_min",
        "id_input_normalized_max",
        "id_input_normalized_abs_max",
        "final_rom_input_normalized_min",
        "final_rom_input_normalized_max",
        "z_arm_norm",
        "z_arm_runtime_std",
        "rom_target_clamp_fraction",
        "reset_count",
        "timeout",
        "unsafe_termination",
    )

    def __init__(self) -> None:
        self.samples: dict[str, list[torch.Tensor]] = {key: [] for key in self.VECTOR_KEYS}
        self.scalars: dict[str, list[float]] = {key: [] for key in self.SCALAR_KEYS}

    def update(self, debug: dict) -> None:
        for key in self.VECTOR_KEYS:
            value = debug.get(key)
            if isinstance(value, torch.Tensor):
                self.samples[key].append(torch.nan_to_num(value.detach().float().flatten().cpu()))
        for key in self.SCALAR_KEYS:
            value = debug.get(key)
            if isinstance(value, torch.Tensor):
                self.scalars[key].append(_scalar(value))

    def print_summary(self) -> None:
        print("[SUMMARY] Runtime signal max/RMS by stage:", flush=True)
        for key, values in self.samples.items():
            if not values:
                continue
            max_len = max(v.numel() for v in values)
            padded = []
            for value in values:
                if value.numel() == max_len:
                    padded.append(value)
                else:
                    pad = torch.zeros((max_len,), dtype=value.dtype)
                    pad[: value.numel()] = value
                    padded.append(pad)
            data = torch.stack(padded)
            abs_max = torch.max(torch.abs(data), dim=0).values
            rms = torch.sqrt(torch.mean(torch.square(data), dim=0))
            print(f"[SUMMARY] {key}: max={_as_list(abs_max, 6)} rms={_as_list(rms, 6)}", flush=True)
        for key, values in self.scalars.items():
            if not values:
                continue
            data = torch.tensor(values, dtype=torch.float32)
            print(
                f"[SUMMARY] {key}: max={float(torch.max(torch.abs(data))):.6g} "
                f"rms={float(torch.sqrt(torch.mean(torch.square(data)))):.6g}",
                flush=True,
            )


def _print_live_telemetry(raw_env, step: int, *, root0: torch.Tensor | None) -> None:
    if not hasattr(raw_env, "get_debug_state"):
        return
    debug = raw_env.get_debug_state(0)
    root = debug.get("root_pos_w")
    if isinstance(root, torch.Tensor) and root0 is not None:
        base_move = float(torch.linalg.norm(root - root0).detach().cpu())
    else:
        base_move = float("nan")
    root_z = _scalar(root[2] if isinstance(root, torch.Tensor) and root.numel() >= 3 else None)
    root_vel = debug.get("root_lin_vel_w")
    root_vel_z = _scalar(root_vel[2] if isinstance(root_vel, torch.Tensor) and root_vel.numel() >= 3 else None)
    roll, pitch = _roll_pitch_from_wxyz(debug.get("root_quat_w"))
    valve_mean, valve_std, valve_min, valve_max, valve_sat = _vector_stats(debug.get("valve_cmd"))
    print(
        "[LIVE] "
        f"step={step} "
        f"episode_step={int(_scalar(debug.get('episode_step'), -1))} "
        f"reset_count={int(_scalar(debug.get('reset_count'), 0))} "
        f"reset_reason={debug.get('reset_reason', 'unknown')} "
        f"q_des={_as_list(debug.get('q_ref_commanded'))} "
        f"q={_as_list(debug.get('q_actual'))} "
        f"err={_as_list(debug.get('q_error'))} "
        f"qdot_des={_as_list(debug.get('qdot_ref'))} "
        f"qddot_des={_as_list(debug.get('qddot_cmd'))} "
        f"ee_des={_as_list(debug.get('ee_pose_command'))} "
        f"ee={_as_list(debug.get('current_ee_pose'))} "
        f"ee_err={_as_list(debug.get('ee_position_error'))} "
        f"ee_rot_err={_scalar(debug.get('ee_orientation_error')):.4f} "
        f"qddot_fb={_as_list(debug.get('scripted_qddot_feedback'))} "
        f"qddot_ff={_as_list(debug.get('scripted_qddot_feedforward'))} "
        f"action={_as_list(debug.get('action'))} "
        f"id_norm_minmax=({_scalar(debug.get('id_input_normalized_min')):.3f},{_scalar(debug.get('id_input_normalized_max')):.3f}) "
        f"z_arm_norm={_scalar(debug.get('z_arm_norm')):.3f} "
        f"ValveCmd={_as_list(debug.get('valve_cmd'))} "
        f"FD_tau={_as_list(debug.get('fd_tau_hat'))} "
        f"FD_norm={_as_list(debug.get('fd_tau_hat_normalized'))} "
        f"FD_load=({_scalar(debug.get('fd_soft_limit_fraction')):.3f},{_scalar(debug.get('fd_hard_limit_fraction')):.3f},{_scalar(debug.get('fd_normalized_abs_max')):.3f}) "
        f"ValveStats=({valve_mean:.3f},{valve_std:.3f},{valve_min:.3f},{valve_max:.3f},sat={valve_sat:.2f}) "
        f"ROM_norm_minmax=({_scalar(debug.get('final_rom_input_normalized_min')):.3f},{_scalar(debug.get('final_rom_input_normalized_max')):.3f}) "
        f"ROM_kin={_as_list(debug.get('final_rom_delta_extension_kinematic_50ms'), 6)} "
        f"ROM_corr={_as_list(debug.get('final_rom_delta_extension_correction_50ms'), 6)} "
        f"ROM_dExt50={_as_list(debug.get('final_rom_delta_extension_50ms'), 6)} "
        f"dIsaac50={_as_list(debug.get('final_rom_delta_isaac_50ms'), 6)} "
        f"dIsaac10={_as_list(debug.get('final_rom_delta_isaac_10ms'), 6)} "
        f"cyl_target={_as_list(debug.get('rom_cylinder_pos_target'))} "
        f"cyl_q={_as_list(debug.get('q_actual'))[:4]} "
        f"target_motion={_as_list(debug.get('rom_cylinder_cumulative_target_motion'), 5)} "
        f"measured_motion={_as_list(debug.get('rom_cylinder_cumulative_measured_motion'), 5)} "
        f"tracking_ratio={_as_list(debug.get('rom_cylinder_tracking_ratio'), 3)} "
        f"root_z={root_z:.4f} root_vz={root_vel_z:.4f} roll={roll:.4f} pitch={pitch:.4f} "
        f"vehicle_kmh={_scalar(debug.get('vehicle_speed_kmh')):.3f} "
        f"base_move={base_move:.5f} "
        f"clamp={float(debug.get('rom_target_clamp_fraction', torch.tensor(float('nan')))):.4f}",
        flush=True,
    )


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.task == DEFAULT_HYDRAULIC_ROM_TASK:
        play_mode = args_cli.play_mode
        if play_mode is None:
            play_mode = "rl_task_space_tracking" if args_cli.checkpoint is not None else "joint_space_validation"
        if play_mode == "rl_task_space_tracking" and args_cli.checkpoint is None:
            raise RuntimeError(
                "Task-space tracking requires a trained PPO checkpoint because the reference RRLAB task uses "
                "the policy, not IK, to map ee_pose commands to arm actions."
            )

        env_cfg.reference_mode = play_mode
        env_cfg.control_mode = play_mode
        if play_mode == "rl_task_space_tracking":
            env_cfg.observation_space = env_cfg.task_space_observation_space
        env_cfg.enable_scripted_vehicle_motion = (
            bool(args_cli.enable_vehicle_motion) if play_mode == "joint_space_validation" else True
        )
        print(
            "[INFO] Kevin play mode: "
            f"{play_mode} reference_mode={env_cfg.reference_mode} "
            f"enable_scripted_vehicle_motion={env_cfg.enable_scripted_vehicle_motion}",
            flush=True,
        )
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device

    apply_kevin_sim_memory_optimizations(env_cfg.sim, verbose=True)

    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)
        raw_env = env.unwrapped
        if getattr(raw_env.cfg, "control_mode", "") == "rl_task_space_tracking" and hasattr(raw_env, "run_task_space_bridge_preflight"):
            task_space_result = raw_env.run_task_space_bridge_preflight()
            print(f"[PREFLIGHT] task_space_bridge={task_space_result}", flush=True)
            if not bool(task_space_result.get("passed", False)):
                raise RuntimeError(
                    "Task-space play mode is blocked before playback: "
                    f"{task_space_result.get('preflight_decision', 'BLOCKED_TASKSPACE_REFERENCE')}"
                )

        env = RslRlVecEnvWrapper(env)
        policy = None
        if args_cli.checkpoint is not None:
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            print(f"[INFO] Loading checkpoint: {args_cli.checkpoint}")
            runner.load(args_cli.checkpoint)
            policy = runner.get_inference_policy(device=agent_cfg.device)
        else:
            print("[INFO] Running joint_space_validation with zero external actions.")

        obs = env.get_observations()
        if isinstance(obs, tuple):
            obs = obs[0]
        root0 = None
        if hasattr(raw_env, "get_debug_state"):
            root_candidate = raw_env.get_debug_state(0).get("root_pos_w")
            if isinstance(root_candidate, torch.Tensor):
                root0 = root_candidate.detach().clone()
        step_dt = float(getattr(raw_env, "step_dt", getattr(env_cfg.sim, "dt", 0.01) * getattr(env_cfg, "decimation", 1)))
        telemetry_interval = max(1, int(round(1.0 / max(step_dt, 1.0e-6))))
        stats = RuntimeSignalStats()
        step = 0
        while args_cli.steps < 0 or step < args_cli.steps:
            if args_cli.steps < 0 and hasattr(simulation_app, "is_running") and not simulation_app.is_running():
                break
            with torch.inference_mode():
                if policy is None:
                    actions = torch.zeros((env.num_envs, env.num_actions), dtype=torch.float32, device=env.device)
                else:
                    actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            if isinstance(obs, tuple):
                obs = obs[0]
            step += 1
            if hasattr(raw_env, "get_debug_state"):
                stats.update(raw_env.get_debug_state(0))
            if step % telemetry_interval == 0:
                _print_live_telemetry(raw_env, step, root0=root0)
        stats.print_summary()
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()  # type: ignore
    finally:
        simulation_app.close()
