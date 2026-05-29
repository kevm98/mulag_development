from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kevin_integration.rl.torque_adapter import (  # noqa: E402
    DEFAULT_TORQUE_ADAPTER_PRESET,
    TORQUE_ADAPTER_MODES,
    TORQUE_ADAPTER_PRESETS,
    resolve_torque_adapter_scale,
)

parser = argparse.ArgumentParser(description="Evaluate Mulag adaptive RL reward/debug diagnostics.")
parser.add_argument("--task", type=str, default="Kevin-Mulag-AdaptiveRL-JointReach-v0", help="Gym task name.")
parser.add_argument("--checkpoint", type=str, default=None, help="Optional RSL-RL checkpoint path.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of Isaac Lab environments.")
parser.add_argument("--steps", type=int, default=1000, help="Number of policy steps to run.")
parser.add_argument("--print_interval", type=int, default=50, help="Step interval for terminal diagnostics.")
parser.add_argument("--control_mode", choices=["position_delta", "effort"], default=None, help="Override env control mode.")
parser.add_argument(
    "--position_delta_source",
    choices=["direct_action", "id_valve"],
    default=None,
    help="Override position-delta command source.",
)
parser.add_argument("--use_fd_effort", action="store_true", default=None, help="Run FD and apply effort targets.")
parser.add_argument("--use_sd_feedback", action="store_true", default=None, help="Enable learned SD feedback.")
parser.add_argument("--max_abs_torque", type=float, default=None, help="Override FD torque clamp.")
parser.add_argument("--torque_ramp_steps", type=int, default=None, help="Override effort torque ramp length.")
parser.add_argument("--torque_safety_scale", type=float, default=None, help="Override effort torque safety multiplier.")
parser.add_argument(
    "--torque_adapter_mode",
    choices=TORQUE_ADAPTER_MODES,
    default=None,
    help="Override Simscape-to-Isaac torque adapter mode.",
)
parser.add_argument(
    "--torque_adapter_preset",
    choices=TORQUE_ADAPTER_PRESETS,
    default=None,
    help="Override named torque adapter scale preset.",
)
parser.add_argument(
    "--torque_adapter_scale",
    type=str,
    default=None,
    help='Override per-joint scale as comma-separated values, e.g. "0.0001,0.001,0.0005,0.0001".',
)
parser.add_argument("--fd_torque_scale", type=float, default=None, help="Override FD scale for tanh_squash mode.")
parser.add_argument(
    "--use_fd_residual_alpha",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Enable/disable deterministic residual authority scaling after the torque adapter.",
)
parser.add_argument("--fd_residual_alpha", type=float, default=None, help="Override FD residual gain.")
parser.add_argument("--torque_rate_limit", type=float, default=None, help="Override per-step torque rate limit.")
parser.add_argument("--torque_lowpass_alpha", type=float, default=None, help="Override torque low-pass alpha.")
parser.add_argument("--action_to_qddot_scale", type=float, default=None, help="Override action-to-qddot scale.")
parser.add_argument("--qddot_cmd_clip", type=float, default=None, help="Override qddot command clamp.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.torque_adapter_preset is not None or args_cli.torque_adapter_scale is not None:
    args_cli.torque_adapter_scale = resolve_torque_adapter_scale(
        args_cli.torque_adapter_preset or DEFAULT_TORQUE_ADAPTER_PRESET,
        args_cli.torque_adapter_scale,
    )
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


def _tensor_to_float(value: torch.Tensor | float | int) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().mean().cpu())
    return float(value)


def _format_joint_errors(value: torch.Tensor | None) -> str:
    if value is None:
        return "[]"
    values = value.detach().flatten().cpu().tolist()
    return "[" + ", ".join(f"{item:.4f}" for item in values) + "]"


def _format_debug_line(
    *,
    step: int,
    reward: torch.Tensor,
    dones: torch.Tensor,
    debug: dict[str, torch.Tensor | str],
    control_mode: str,
    position_delta_source: str,
    use_sd_feedback: bool,
    use_fd_effort: bool,
) -> str:
    values = {
        "step_reward": _tensor_to_float(reward),
        "done_frac": _tensor_to_float(dones.float()),
    }
    for key in (
        "q_error_mean",
        "q_error_mean_weighted",
        "q_error_mean_unweighted",
        "qdot_mean",
        "action_mean",
        "action_delta_mean",
        "qddot_cmd_mean",
        "z_arm_mean",
        "z_arm_std",
        "valve_mean",
        "valve_max",
        "fd_simscape_mean",
        "fd_simscape_max",
        "tau_fd_adapted_mean",
        "tau_fd_adapted_max",
        "tau_applied_raw_mean",
        "tau_applied_raw_max",
        "tau_isaac_filtered_mean",
        "tau_isaac_filtered_max",
        "torque_raw_mean",
        "torque_raw_max",
        "torque_mean",
        "torque_max",
        "torque_adapter_clamp_fraction",
        "torque_clamp_fraction",
        "torque_rate_limited_fraction",
        "torque_rate_limit",
        "torque_lowpass_alpha",
        "fd_residual_alpha",
        "max_abs_torque",
        "dP_mean",
        "fnet_mean",
        "reward_total",
        "q_tracking",
        "qdot_penalty",
        "progress_reward",
        "action_smoothness_penalty",
        "valve_effort_penalty",
        "torque_effort_penalty",
        "alive",
        "termination_penalty",
    ):
        if key in debug:
            values[key] = _tensor_to_float(debug[key])

    ordered_keys = (
        "step_reward",
        "reward_total",
        "q_tracking",
        "qdot_penalty",
        "progress_reward",
        "action_smoothness_penalty",
        "valve_effort_penalty",
        "torque_effort_penalty",
        "alive",
        "termination_penalty",
        "q_error_mean",
        "q_error_mean_weighted",
        "q_error_mean_unweighted",
        "qdot_mean",
        "action_mean",
        "action_delta_mean",
        "qddot_cmd_mean",
        "z_arm_mean",
        "z_arm_std",
        "valve_mean",
        "valve_max",
        "fd_simscape_mean",
        "fd_simscape_max",
        "tau_fd_adapted_mean",
        "tau_fd_adapted_max",
        "tau_applied_raw_mean",
        "tau_applied_raw_max",
        "tau_isaac_filtered_mean",
        "tau_isaac_filtered_max",
        "torque_raw_mean",
        "torque_raw_max",
        "torque_mean",
        "torque_max",
        "torque_adapter_clamp_fraction",
        "torque_clamp_fraction",
        "torque_rate_limited_fraction",
        "torque_rate_limit",
        "torque_lowpass_alpha",
        "fd_residual_alpha",
        "max_abs_torque",
        "dP_mean",
        "fnet_mean",
        "done_frac",
    )
    mode = debug.get("control_mode", control_mode)
    delta_source = debug.get("position_delta_source", position_delta_source)
    sd_feedback = debug.get("use_sd_feedback", str(use_sd_feedback))
    fd_effort = debug.get("use_fd_effort", str(use_fd_effort))
    adapter_mode = debug.get("torque_adapter_mode", "")
    adapter_preset = debug.get("torque_adapter_preset", "")
    adapter_scale = debug.get("torque_adapter_scale", "")
    residual_enabled = debug.get("use_fd_residual_alpha", "")
    effective_scale = debug.get("effective_torque_adapter_scale", "")
    terms = [
        f"step={step}",
        f"control_mode={mode}",
        f"position_delta_source={delta_source}",
        f"use_sd_feedback={sd_feedback}",
        f"use_fd_effort={fd_effort}",
    ]
    if adapter_mode:
        terms.append(f"torque_adapter_mode={adapter_mode}")
    if adapter_preset:
        terms.append(f"torque_adapter_preset={adapter_preset}")
    if adapter_scale:
        terms.append(f"torque_adapter_scale={adapter_scale}")
    if residual_enabled:
        terms.append(f"use_fd_residual_alpha={residual_enabled}")
    if effective_scale:
        terms.append(f"effective_torque_adapter_scale={effective_scale}")
    terms.extend(f"{key}={values[key]:.4f}" for key in ordered_keys if key in values)
    if values.get("torque_clamp_fraction", 0.0) > 0.5:
        terms.append("[WARN] final torque still too clamped")
    if values.get("torque_rate_limited_fraction", 0.0) > 0.8:
        terms.append("[WARN] torque still rate-limited")
    tau_applied_raw_max = values.get("tau_applied_raw_max", 0.0)
    max_abs_torque = max(values.get("max_abs_torque", 0.0), 0.0)
    if max_abs_torque > 0.0 and tau_applied_raw_max > 2.0 * max_abs_torque:
        terms.append("[WARN] residual alpha still too high")
    if tau_applied_raw_max < 0.1:
        terms.append("[WARN] residual alpha may be too low")
    terms.append(f"q_error_abs_per_joint={_format_joint_errors(debug.get('q_error_abs_per_joint'))}")
    terms.append(f"qddot_cmd_abs_per_joint={_format_joint_errors(debug.get('qddot_cmd_abs_per_joint'))}")
    return " | ".join(terms)


def _apply_env_overrides(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg) -> None:
    if args_cli.control_mode is not None:
        env_cfg.control_mode = args_cli.control_mode
    if args_cli.position_delta_source is not None:
        env_cfg.position_delta_source = args_cli.position_delta_source
    if args_cli.use_fd_effort is not None:
        env_cfg.use_fd_effort = args_cli.use_fd_effort
    if args_cli.use_sd_feedback is not None:
        env_cfg.use_sd_feedback = args_cli.use_sd_feedback
    if args_cli.max_abs_torque is not None:
        env_cfg.max_abs_torque = args_cli.max_abs_torque
    if args_cli.torque_ramp_steps is not None:
        env_cfg.torque_ramp_steps = args_cli.torque_ramp_steps
    if args_cli.torque_safety_scale is not None:
        env_cfg.torque_safety_scale = args_cli.torque_safety_scale
    if args_cli.torque_adapter_mode is not None:
        env_cfg.torque_adapter_mode = args_cli.torque_adapter_mode
    if args_cli.torque_adapter_preset is not None:
        env_cfg.torque_adapter_preset = args_cli.torque_adapter_preset
        env_cfg.torque_adapter_scale = resolve_torque_adapter_scale(args_cli.torque_adapter_preset)
    if args_cli.torque_adapter_scale is not None:
        env_cfg.torque_adapter_scale = args_cli.torque_adapter_scale
    if args_cli.fd_torque_scale is not None:
        env_cfg.fd_torque_scale = args_cli.fd_torque_scale
    if args_cli.use_fd_residual_alpha is not None:
        env_cfg.use_fd_residual_alpha = args_cli.use_fd_residual_alpha
    if args_cli.fd_residual_alpha is not None:
        env_cfg.fd_residual_alpha = args_cli.fd_residual_alpha
    if args_cli.torque_rate_limit is not None:
        env_cfg.torque_rate_limit = args_cli.torque_rate_limit
    if args_cli.torque_lowpass_alpha is not None:
        env_cfg.torque_lowpass_alpha = args_cli.torque_lowpass_alpha
    if args_cli.action_to_qddot_scale is not None:
        env_cfg.action_to_qddot_scale = args_cli.action_to_qddot_scale
    if args_cli.qddot_cmd_clip is not None:
        env_cfg.qddot_cmd_clip = args_cli.qddot_cmd_clip


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    if args_cli.steps <= 0:
        raise ValueError("--steps must be positive.")
    if args_cli.print_interval <= 0:
        raise ValueError("--print_interval must be positive.")

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.debug_reward_terms = True
    _apply_env_overrides(env_cfg)
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device

    apply_kevin_sim_memory_optimizations(env_cfg.sim, verbose=True)

    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)

        env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))
        policy = None
        policy_device = agent_cfg.device
        if args_cli.checkpoint is not None:
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            print(f"[INFO] Loading checkpoint: {args_cli.checkpoint}")
            runner.load(args_cli.checkpoint)
            policy = runner.get_inference_policy(device=policy_device)
        else:
            print("[INFO] No checkpoint provided. Running zero-action baseline.")

        obs = env.get_observations()
        if policy is not None:
            obs = obs.to(policy_device)

        for step in range(1, args_cli.steps + 1):
            with torch.inference_mode():
                if policy is None:
                    actions = torch.zeros((env.num_envs, env.num_actions), dtype=torch.float32, device=env.device)
                else:
                    actions = policy(obs).to(env.device)
                obs, reward, dones, _ = env.step(actions)
                if policy is not None:
                    obs = obs.to(policy_device)

            if step == 1 or step % args_cli.print_interval == 0:
                debug = getattr(env.unwrapped, "_reward_debug", {})
                print(
                    _format_debug_line(
                        step=step,
                        reward=reward,
                        dones=dones,
                        debug=debug,
                        control_mode=env.unwrapped.cfg.control_mode,
                        position_delta_source=env.unwrapped.cfg.position_delta_source,
                        use_sd_feedback=env.unwrapped.cfg.use_sd_feedback,
                        use_fd_effort=env.unwrapped.cfg.use_fd_effort,
                    ),
                    flush=True,
                )
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()  # type: ignore
    finally:
        simulation_app.close()
