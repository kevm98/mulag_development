from __future__ import annotations

import argparse
import csv
import math
import sys
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser(description="Compare Mulag adaptive RL checkpoints on joint tracking diagnostics.")
parser.add_argument("--task", type=str, default="Kevin-Mulag-AdaptiveRL-JointReach-v0", help="Gym task name.")
parser.add_argument("--run_dir", type=str, required=True, help="Directory containing RSL-RL checkpoints.")
parser.add_argument("--checkpoints", nargs="+", required=True, help="Checkpoint filenames or absolute paths.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of Isaac Lab environments.")
parser.add_argument("--steps", type=int, default=1000, help="Number of policy steps per checkpoint.")
parser.add_argument("--control_mode", choices=["position_delta", "effort"], default=None, help="Override env control mode.")
parser.add_argument("--use_fd_effort", action="store_true", default=None, help="Run FD and apply effort targets.")
parser.add_argument("--max_abs_torque", type=float, default=None, help="Override FD torque clamp.")
parser.add_argument("--fd_residual_alpha", type=float, default=None, help="Override FD residual gain.")
parser.add_argument("--use_sd_observer", action="store_true", default=None, help="Run SD as a passive observer.")
parser.add_argument("--use_sd_feedback", action="store_true", default=None, help="Enable learned SD feedback.")
parser.add_argument("--sd_feedback_beta", type=float, default=None, help="Blend factor for SD feedback.")
parser.add_argument("--q_ref_ramp_time_s", type=float, default=None, help="Override q_ref commanded ramp time.")
parser.add_argument("--q_ref_target_scale", type=float, default=None, help="Use a fixed q_ref target scale.")
parser.add_argument("--disable_q_ref_ramp", action="store_true", default=False, help="Disable q_ref command ramping.")
parser.add_argument(
    "--disable_q_ref_curriculum",
    action="store_true",
    default=False,
    help="Disable q_ref target-scale curriculum.",
)
parser.add_argument("--episode_length_s", type=float, default=None, help="Override episode duration.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
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


SUMMARY_FIELDS = [
    "checkpoint",
    "mean_q_error_weighted",
    "final_q_error_weighted",
    "max_q_error_weighted",
    "mean_q_error_unweighted",
    "mean_joint1_error",
    "max_joint1_error",
    "mean_joint5_error",
    "mean_joint5_coupling_error",
    "max_joint5_coupling_error",
    "mean_joint5_actual",
    "mean_joint5_qdot",
    "mean_q_ref_delta_abs_max",
    "max_q_ref_delta_abs_max",
    "mean_q_ref_target_scale",
    "mean_q_ref_ramp_progress",
    "target_change_count",
    "timeout_count",
    "mean_episode_step_at_large_error",
    "joint1_large_error_fraction",
    "mean_action_abs",
    "mean_action_saturation_fraction",
    "mean_qdot",
    "max_qdot",
    "mean_torque",
    "max_torque",
    "mean_torque_clamp_fraction",
    "mean_sd_feedback_clamped_fraction",
    "done_frac",
]


def _tensor_to_float(value: torch.Tensor | float | int | None, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return default
        return float(value.detach().float().mean().cpu())
    return float(value)


def _debug_float(debug: dict[str, torch.Tensor | str], key: str, default: float = 0.0) -> float:
    value = debug.get(key)
    if isinstance(value, str):
        return default
    return _tensor_to_float(value, default)


def _debug_vector(debug: dict[str, torch.Tensor | str], key: str) -> list[float]:
    value = debug.get(key)
    if not isinstance(value, torch.Tensor) or value.numel() == 0:
        return []
    return [float(item) for item in value.detach().flatten().cpu().tolist()]


def _mean(values: list[float]) -> float:
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return float("nan")
    return sum(finite_values) / len(finite_values)


def _max(values: list[float]) -> float:
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return float("nan")
    return max(finite_values)


def _checkpoint_path(run_dir: Path, checkpoint: str) -> Path:
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.is_absolute():
        return checkpoint_path
    return run_dir / checkpoint_path


def _apply_env_overrides(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg) -> None:
    if args_cli.control_mode is not None:
        env_cfg.control_mode = args_cli.control_mode
    if args_cli.use_fd_effort is not None:
        env_cfg.use_fd_effort = args_cli.use_fd_effort
    if args_cli.max_abs_torque is not None:
        env_cfg.max_abs_torque = args_cli.max_abs_torque
    if args_cli.fd_residual_alpha is not None:
        env_cfg.fd_residual_alpha = args_cli.fd_residual_alpha
        env_cfg.use_fd_residual_alpha = True
    if args_cli.use_sd_observer is not None:
        env_cfg.use_sd_observer = args_cli.use_sd_observer
    if args_cli.use_sd_feedback is not None:
        env_cfg.use_sd_observer = True
        env_cfg.use_sd_feedback = args_cli.use_sd_feedback
    if args_cli.sd_feedback_beta is not None:
        env_cfg.sd_feedback_beta = args_cli.sd_feedback_beta
    if args_cli.q_ref_ramp_time_s is not None:
        env_cfg.q_ref_ramp_time_s = args_cli.q_ref_ramp_time_s
    if args_cli.q_ref_target_scale is not None:
        env_cfg.q_ref_target_scale_initial = args_cli.q_ref_target_scale
        env_cfg.q_ref_target_scale_final = args_cli.q_ref_target_scale
        env_cfg.use_q_ref_curriculum = False
    if args_cli.disable_q_ref_ramp:
        env_cfg.use_q_ref_ramp = False
    if args_cli.disable_q_ref_curriculum:
        env_cfg.use_q_ref_curriculum = False
    if args_cli.episode_length_s is not None:
        env_cfg.episode_length_s = args_cli.episode_length_s


def _evaluate_checkpoint(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlOnPolicyRunnerCfg,
    checkpoint_path: Path,
) -> dict[str, float | str]:
    metrics: dict[str, list[float]] = {
        "q_error_weighted": [],
        "q_error_unweighted": [],
        "joint1_error": [],
        "joint5_error": [],
        "joint5_coupling_error": [],
        "joint5_actual": [],
        "joint5_qdot": [],
        "q_ref_delta_abs_max": [],
        "q_ref_target_scale": [],
        "q_ref_ramp_progress": [],
        "episode_step_at_large_error": [],
        "joint1_large_error": [],
        "action_abs": [],
        "action_saturation_fraction": [],
        "qdot": [],
        "qdot_max": [],
        "torque": [],
        "torque_max": [],
        "torque_clamp_fraction": [],
        "sd_feedback_clamped_fraction": [],
        "done_frac": [],
    }
    target_change_steps: set[int] = set()
    timeout_steps: set[int] = set()
    joint1_large_error_threshold = 0.15

    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)

        env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        print(f"[INFO] Loading checkpoint: {checkpoint_path}", flush=True)
        runner.load(str(checkpoint_path))
        policy = runner.get_inference_policy(device=agent_cfg.device)

        obs = env.get_observations().to(agent_cfg.device)
        for _ in range(args_cli.steps):
            with torch.inference_mode():
                actions = policy(obs).to(env.device)
                obs, _, dones, _ = env.step(actions)
                obs = obs.to(agent_cfg.device)

            debug = getattr(env.unwrapped, "_reward_debug", {})
            joint_errors = _debug_vector(debug, "q_error_abs_per_joint")
            metrics["q_error_weighted"].append(_debug_float(debug, "q_error_mean_weighted"))
            metrics["q_error_unweighted"].append(_debug_float(debug, "q_error_mean_unweighted"))
            joint1_error = joint_errors[0] if len(joint_errors) >= 1 else float("nan")
            metrics["joint1_error"].append(joint1_error)
            metrics["joint5_error"].append(joint_errors[4] if len(joint_errors) >= 5 else float("nan"))
            metrics["joint5_coupling_error"].append(
                abs(_debug_float(debug, "joint5_coupling_error", float("nan")))
            )
            metrics["joint5_actual"].append(_debug_float(debug, "joint5_actual", float("nan")))
            metrics["joint5_qdot"].append(_debug_float(debug, "joint5_qdot", float("nan")))
            metrics["q_ref_delta_abs_max"].append(_debug_float(debug, "q_ref_delta_abs_max"))
            metrics["q_ref_target_scale"].append(_debug_float(debug, "q_ref_target_scale"))
            metrics["q_ref_ramp_progress"].append(_debug_float(debug, "q_ref_ramp_progress"))
            large_joint1_error = math.isfinite(joint1_error) and joint1_error > joint1_large_error_threshold
            metrics["joint1_large_error"].append(1.0 if large_joint1_error else 0.0)
            if large_joint1_error:
                metrics["episode_step_at_large_error"].append(_debug_float(debug, "episode_step", float("nan")))
            if _debug_float(debug, "target_changed_flag") > 0.5:
                target_change_step = _debug_float(debug, "target_change_step", float("nan"))
                if math.isfinite(target_change_step):
                    target_change_steps.add(int(round(target_change_step)))
            if _debug_float(debug, "reset_due_to_timeout") > 0.5:
                timeout_step = _debug_float(debug, "last_reset_step_global", float("nan"))
                if math.isfinite(timeout_step):
                    timeout_steps.add(int(round(timeout_step)))
            metrics["action_abs"].append(_debug_float(debug, "action_abs_mean"))
            metrics["action_saturation_fraction"].append(_debug_float(debug, "action_saturation_fraction"))
            metrics["qdot"].append(_debug_float(debug, "qdot_mean"))
            metrics["qdot_max"].append(_debug_float(debug, "qdot_abs_max", _debug_float(debug, "qdot_mean")))
            metrics["torque"].append(_debug_float(debug, "torque_mean"))
            metrics["torque_max"].append(_debug_float(debug, "torque_max"))
            metrics["torque_clamp_fraction"].append(_debug_float(debug, "torque_clamp_fraction"))
            metrics["sd_feedback_clamped_fraction"].append(_debug_float(debug, "sd_feedback_clamped_fraction"))
            metrics["done_frac"].append(_tensor_to_float(dones.float()))

    finally:
        env.close()

    q_error_weighted = metrics["q_error_weighted"]
    return {
        "checkpoint": checkpoint_path.name,
        "mean_q_error_weighted": _mean(q_error_weighted),
        "final_q_error_weighted": q_error_weighted[-1] if q_error_weighted else float("nan"),
        "max_q_error_weighted": _max(q_error_weighted),
        "mean_q_error_unweighted": _mean(metrics["q_error_unweighted"]),
        "mean_joint1_error": _mean(metrics["joint1_error"]),
        "max_joint1_error": _max(metrics["joint1_error"]),
        "mean_joint5_error": _mean(metrics["joint5_error"]),
        "mean_joint5_coupling_error": _mean(metrics["joint5_coupling_error"]),
        "max_joint5_coupling_error": _max(metrics["joint5_coupling_error"]),
        "mean_joint5_actual": _mean(metrics["joint5_actual"]),
        "mean_joint5_qdot": _mean(metrics["joint5_qdot"]),
        "mean_q_ref_delta_abs_max": _mean(metrics["q_ref_delta_abs_max"]),
        "max_q_ref_delta_abs_max": _max(metrics["q_ref_delta_abs_max"]),
        "mean_q_ref_target_scale": _mean(metrics["q_ref_target_scale"]),
        "mean_q_ref_ramp_progress": _mean(metrics["q_ref_ramp_progress"]),
        "target_change_count": float(len(target_change_steps)),
        "timeout_count": float(len(timeout_steps)),
        "mean_episode_step_at_large_error": _mean(metrics["episode_step_at_large_error"]),
        "joint1_large_error_fraction": _mean(metrics["joint1_large_error"]),
        "mean_action_abs": _mean(metrics["action_abs"]),
        "mean_action_saturation_fraction": _mean(metrics["action_saturation_fraction"]),
        "mean_qdot": _mean(metrics["qdot"]),
        "max_qdot": _max(metrics["qdot_max"]),
        "mean_torque": _mean(metrics["torque"]),
        "max_torque": _max(metrics["torque_max"]),
        "mean_torque_clamp_fraction": _mean(metrics["torque_clamp_fraction"]),
        "mean_sd_feedback_clamped_fraction": _mean(metrics["sd_feedback_clamped_fraction"]),
        "done_frac": _mean(metrics["done_frac"]),
    }


def _format_value(value: float | str) -> str:
    if isinstance(value, str):
        return value
    if not math.isfinite(value):
        return "nan"
    return f"{value:.5f}"


def _print_table(rows: list[dict[str, float | str]]) -> None:
    widths = {
        field: max(len(field), *(len(_format_value(row[field])) for row in rows))
        for field in SUMMARY_FIELDS
    }
    header = " | ".join(field.ljust(widths[field]) for field in SUMMARY_FIELDS)
    separator = "-+-".join("-" * widths[field] for field in SUMMARY_FIELDS)
    print(header, flush=True)
    print(separator, flush=True)
    for row in rows:
        print(" | ".join(_format_value(row[field]).ljust(widths[field]) for field in SUMMARY_FIELDS), flush=True)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    if args_cli.steps <= 0:
        raise ValueError("--steps must be positive.")

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.debug_reward_terms = True
    _apply_env_overrides(env_cfg)
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device

    apply_kevin_sim_memory_optimizations(env_cfg.sim, verbose=True)

    run_dir = Path(args_cli.run_dir)
    rows = []
    for checkpoint in args_cli.checkpoints:
        checkpoint_path = _checkpoint_path(run_dir, checkpoint)
        rows.append(_evaluate_checkpoint(env_cfg, agent_cfg, checkpoint_path))

    output_dir = Path("logs") / "checkpoint_compare" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "checkpoint_comparison.csv"
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    _print_table(rows)
    print(f"[INFO] Wrote checkpoint comparison CSV: {csv_path}", flush=True)


if __name__ == "__main__":
    try:
        main()  # type: ignore
    finally:
        simulation_app.close()
