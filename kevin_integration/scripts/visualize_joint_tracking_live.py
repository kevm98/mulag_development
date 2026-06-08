from __future__ import annotations

import argparse
import csv
import math
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

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

JOINT_NAMES = [
    "Drehzapfen_joint",
    "Ausleger_I_joint",
    "Ausleger_II_joint",
    "Messerkopf_Schwenk_joint",
    "Messerkopf_joint",
]
COMMAND_JOINT_NAMES = JOINT_NAMES[:4]
SCALAR_DEBUG_KEYS = [
    "fd_simscape_mean",
    "fd_simscape_max",
    "tau_fd_adapted_mean",
    "tau_fd_adapted_max",
    "tau_applied_raw_mean",
    "tau_applied_raw_max",
    "torque_mean",
    "torque_max",
    "torque_clamp_fraction",
    "torque_rate_limited_fraction",
    "sd_dP_mean",
    "sd_dP_abs_max",
    "sd_fnet_mean",
    "sd_fnet_abs_max",
    "sd_output_nan_or_inf",
    "sd_output_clamped_fraction",
    "use_sd_observer",
    "use_sd_feedback",
    "episode_step",
    "target_changed_flag",
    "reset_due_to_timeout",
    "q_ref_delta_abs_max",
    "q_ref_ramp_progress",
    "q_ref_target_scale",
    "q_ref_jump_fraction",
    "q_error_mean_weighted",
    "q_error_mean_unweighted",
    "qdot_mean",
    "reward_total",
    "q_tracking",
    "progress_reward",
    "action_smoothness_penalty",
    "valve_effort_penalty",
    "torque_effort_penalty",
    "done_frac",
]
PLOT_DEBUG_KEYS = [
    "reward_total",
    "q_error_mean_weighted",
    "qdot_mean",
    "torque_mean",
    "torque_clamp_fraction",
    "torque_rate_limited_fraction",
]

parser = argparse.ArgumentParser(description="Live joint-space tracking plots for Kevin adaptive RL.")
parser.add_argument("--task", type=str, default="Kevin-Mulag-AdaptiveRL-JointReach-v0", help="Gym task name.")
parser.add_argument("--checkpoint", type=str, default=None, help="Optional RSL-RL checkpoint path.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of Isaac Lab environments.")
parser.add_argument("--steps", type=int, default=1000, help="Number of policy steps to run.")
parser.add_argument("--window_steps", type=int, default=500, help="Rolling plot window length in policy steps.")
parser.add_argument("--plot_interval", type=int, default=10, help="Policy-step interval between live plot refreshes.")
parser.add_argument("--csv_only", action="store_true", help="Disable live matplotlib windows and only log CSV data.")
parser.add_argument("--save_plots", action="store_true", help="Save final PNG plots beside the CSV log.")
parser.add_argument(
    "--log_root",
    type=str,
    default="logs/joint_tracking_live",
    help="Directory under which timestamped joint tracking logs are written.",
)
parser.add_argument("--control_mode", choices=["position_delta", "effort"], default=None, help="Override env control mode.")
parser.add_argument(
    "--position_delta_source",
    choices=["direct_action", "id_valve"],
    default=None,
    help="Override position-delta command source.",
)
parser.add_argument("--use_fd_effort", action="store_true", default=None, help="Run FD and apply effort targets.")
parser.add_argument("--use_sd_observer", action="store_true", default=None, help="Run SD as a passive observer.")
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
if args_cli.csv_only or getattr(args_cli, "headless", False):
    import matplotlib

    matplotlib.use("Agg")
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


def _field(prefix: str, name: str) -> str:
    return f"{prefix}_{name}"


def _make_csv_fields(joint_names: Sequence[str], command_joint_names: Sequence[str]) -> list[str]:
    fields = ["step", "time"]
    for prefix in (
        "q_actual",
        "q_ref",
        "q_ref_commanded",
        "q_ref_target",
        "q_error",
        "q_error_signed",
        "q_error_abs",
        "q_ref_delta",
        "q_ref_delta_abs",
        "qdot",
    ):
        fields.extend(_field(prefix, joint_name) for joint_name in joint_names)
    fields.extend(_field("action", joint_name) for joint_name in command_joint_names)
    fields.extend(_field("qddot_cmd", joint_name) for joint_name in joint_names)
    fields.extend(_field("valve_cmd", joint_name) for joint_name in command_joint_names)
    fields.extend(SCALAR_DEBUG_KEYS)
    return fields


def _tensor_to_float(value: Any, default: float = math.nan) -> float:
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return default
        return float(value.detach().float().mean().cpu())
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _flag_to_float(value: Any, default: float = math.nan) -> float:
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        return 1.0 if _tensor_to_float(value, 0.0) > 0.5 else 0.0
    if isinstance(value, str):
        return 1.0 if value.lower() in ("1", "true", "yes", "on") else 0.0
    return 1.0 if bool(value) else 0.0


def _tensor_to_list(value: Any, *, expected_len: int, name: str) -> list[float]:
    if isinstance(value, torch.Tensor):
        values = value.detach().flatten().cpu().tolist()
    elif isinstance(value, Sequence) and not isinstance(value, str):
        values = list(value)
    else:
        raise RuntimeError(f"Debug state field {name!r} is not a tensor or sequence.")
    if len(values) != expected_len:
        raise RuntimeError(f"Debug state field {name!r} has length {len(values)}, expected {expected_len}.")
    return [float(item) for item in values]


def _append_vector(row: dict[str, float], prefix: str, names: Sequence[str], values: Sequence[float]) -> None:
    for name, value in zip(names, values, strict=True):
        row[_field(prefix, name)] = float(value)


def _build_row(
    *,
    step: int,
    step_dt: float,
    debug_state: Mapping[str, Any],
    reward: torch.Tensor,
    dones: torch.Tensor,
    joint_names: Sequence[str],
    command_joint_names: Sequence[str],
) -> dict[str, float]:
    q_actual = _tensor_to_list(debug_state.get("q_actual"), expected_len=5, name="q_actual")
    q_ref = _tensor_to_list(debug_state.get("q_ref"), expected_len=5, name="q_ref")
    q_ref_commanded = _tensor_to_list(
        debug_state.get("q_ref_commanded", q_ref),
        expected_len=5,
        name="q_ref_commanded",
    )
    q_ref_target = _tensor_to_list(
        debug_state.get("q_ref_target", q_ref),
        expected_len=5,
        name="q_ref_target",
    )
    q_error = _tensor_to_list(debug_state.get("q_error"), expected_len=5, name="q_error")
    q_error_abs = _tensor_to_list(
        debug_state.get("q_error_abs", [abs(value) for value in q_error]),
        expected_len=5,
        name="q_error_abs",
    )
    q_ref_delta = _tensor_to_list(
        debug_state.get("q_ref_delta", [math.nan] * 5),
        expected_len=5,
        name="q_ref_delta",
    )
    q_ref_delta_abs = _tensor_to_list(
        debug_state.get("q_ref_delta_abs", [abs(value) for value in q_ref_delta]),
        expected_len=5,
        name="q_ref_delta_abs",
    )
    qdot = _tensor_to_list(debug_state.get("qdot"), expected_len=5, name="qdot")
    action = _tensor_to_list(debug_state.get("action"), expected_len=4, name="action")
    qddot_cmd = _tensor_to_list(debug_state.get("qddot_cmd"), expected_len=5, name="qddot_cmd")
    valve_cmd = _tensor_to_list(debug_state.get("valve_cmd"), expected_len=4, name="valve_cmd")

    reward_debug = debug_state.get("reward_debug", {})
    if not isinstance(reward_debug, Mapping):
        reward_debug = {}

    row: dict[str, float] = {"step": float(step), "time": float(step) * step_dt}
    _append_vector(row, "q_actual", joint_names, q_actual)
    _append_vector(row, "q_ref", joint_names, q_ref)
    _append_vector(row, "q_ref_commanded", joint_names, q_ref_commanded)
    _append_vector(row, "q_ref_target", joint_names, q_ref_target)
    _append_vector(row, "q_error", joint_names, q_error)
    _append_vector(row, "q_error_signed", joint_names, q_error)
    _append_vector(row, "q_error_abs", joint_names, q_error_abs)
    _append_vector(row, "q_ref_delta", joint_names, q_ref_delta)
    _append_vector(row, "q_ref_delta_abs", joint_names, q_ref_delta_abs)
    _append_vector(row, "qdot", joint_names, qdot)
    _append_vector(row, "action", command_joint_names, action)
    _append_vector(row, "qddot_cmd", joint_names, qddot_cmd)
    _append_vector(row, "valve_cmd", command_joint_names, valve_cmd)

    for key in SCALAR_DEBUG_KEYS:
        if key == "done_frac":
            row[key] = _tensor_to_float(dones.float())
        elif key == "episode_step":
            row[key] = _tensor_to_float(debug_state.get(key), _tensor_to_float(reward_debug.get(key)))
        elif key in ("use_sd_observer", "use_sd_feedback", "target_changed_flag", "reset_due_to_timeout"):
            value = debug_state.get(key) if key == "target_changed_flag" else reward_debug.get(key)
            row[key] = _flag_to_float(value)
        elif key == "reward_total":
            row[key] = _tensor_to_float(reward_debug.get(key), _tensor_to_float(reward))
        else:
            row[key] = _tensor_to_float(reward_debug.get(key))
    return row


def _append_history(history: dict[str, list[float]], row: Mapping[str, float]) -> None:
    for field in history:
        history[field].append(float(row[field]))


class JointTrackingPlotter:
    def __init__(
        self,
        *,
        joint_names: Sequence[str],
        window_steps: int,
        interactive: bool,
    ):
        import matplotlib.pyplot as plt

        self.plt = plt
        self.joint_names = list(joint_names)
        self.window_steps = window_steps
        self.interactive = interactive
        if interactive:
            self.plt.ion()

        self.fig_q, q_axes = self.plt.subplots(5, 1, sharex=True, figsize=(11, 9))
        self.q_axes = list(q_axes)
        self.q_actual_lines = []
        self.q_ref_lines = []
        self.q_ref_target_lines = []
        for axis, joint_name in zip(self.q_axes, self.joint_names, strict=True):
            actual_line = axis.plot([], [], label="q_actual", color="tab:blue", linewidth=1.4)[0]
            ref_line = axis.plot([], [], label="q_ref_commanded", color="tab:orange", linewidth=1.2)[0]
            target_line = axis.plot(
                [], [], label="q_ref_target", color="tab:gray", linewidth=1.0, linestyle="--"
            )[0]
            axis.set_ylabel(joint_name.replace("_joint", ""))
            axis.grid(True, alpha=0.3)
            self.q_actual_lines.append(actual_line)
            self.q_ref_lines.append(ref_line)
            self.q_ref_target_lines.append(target_line)
        self.q_axes[0].legend(loc="upper right")
        self.q_axes[-1].set_xlabel("step")
        self.fig_q.suptitle("Joint Position Tracking")
        self.fig_q.tight_layout()

        self.fig_error, error_axes = self.plt.subplots(5, 1, sharex=True, figsize=(11, 9))
        self.error_axes = list(error_axes)
        self.error_lines = []
        for axis, joint_name in zip(self.error_axes, self.joint_names, strict=True):
            error_line = axis.plot([], [], label="q_ref - q_actual", color="tab:red", linewidth=1.3)[0]
            axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.45)
            axis.set_ylabel(joint_name.replace("_joint", ""))
            axis.grid(True, alpha=0.3)
            self.error_lines.append(error_line)
        self.error_axes[0].legend(loc="upper right")
        self.error_axes[-1].set_xlabel("step")
        self.fig_error.suptitle("Joint Tracking Error")
        self.fig_error.tight_layout()

        self.fig_debug, debug_axes = self.plt.subplots(3, 1, sharex=True, figsize=(11, 7))
        self.debug_axes = list(debug_axes)
        self.debug_lines: dict[str, Any] = {}
        self.debug_lines["reward_total"] = self.debug_axes[0].plot(
            [], [], label="reward_total", color="tab:green", linewidth=1.3
        )[0]
        for key, color in (
            ("q_error_mean_weighted", "tab:red"),
            ("qdot_mean", "tab:purple"),
            ("torque_mean", "tab:blue"),
        ):
            self.debug_lines[key] = self.debug_axes[1].plot([], [], label=key, color=color, linewidth=1.3)[0]
        for key, color in (
            ("torque_clamp_fraction", "tab:orange"),
            ("torque_rate_limited_fraction", "tab:brown"),
        ):
            self.debug_lines[key] = self.debug_axes[2].plot([], [], label=key, color=color, linewidth=1.3)[0]
        for axis in self.debug_axes:
            axis.grid(True, alpha=0.3)
            axis.legend(loc="upper right")
        self.debug_axes[-1].set_xlabel("step")
        self.fig_debug.suptitle("RL Debug Values")
        self.fig_debug.tight_layout()

    def update(self, history: Mapping[str, Sequence[float]]) -> None:
        steps = list(history["step"])
        if not steps:
            return
        start = max(0, len(steps) - self.window_steps)
        xs = steps[start:]

        for index, joint_name in enumerate(self.joint_names):
            self.q_actual_lines[index].set_data(xs, list(history[_field("q_actual", joint_name)])[start:])
            self.q_ref_lines[index].set_data(xs, list(history[_field("q_ref_commanded", joint_name)])[start:])
            self.q_ref_target_lines[index].set_data(xs, list(history[_field("q_ref_target", joint_name)])[start:])
            self.error_lines[index].set_data(xs, list(history[_field("q_error", joint_name)])[start:])

        for key in PLOT_DEBUG_KEYS:
            self.debug_lines[key].set_data(xs, list(history[key])[start:])

        self._rescale(self.q_axes, xs)
        self._rescale(self.error_axes, xs)
        self._rescale(self.debug_axes, xs)

        for fig in (self.fig_q, self.fig_error, self.fig_debug):
            fig.canvas.draw_idle()
        if self.interactive:
            self.plt.pause(0.001)

    def save(self, output_dir: Path) -> None:
        self.fig_q.savefig(output_dir / "joint_q_tracking.png", dpi=150)
        self.fig_error.savefig(output_dir / "joint_q_error.png", dpi=150)
        self.fig_debug.savefig(output_dir / "rl_debug_values.png", dpi=150)

    def close(self) -> None:
        for fig in (self.fig_q, self.fig_error, self.fig_debug):
            self.plt.close(fig)

    @staticmethod
    def _rescale(axes: Sequence[Any], xs: Sequence[float]) -> None:
        if not xs:
            return
        x_min = xs[0]
        x_max = xs[-1]
        if x_min == x_max:
            x_min -= 1.0
            x_max += 1.0
        for axis in axes:
            axis.set_xlim(x_min, x_max)
            axis.relim()
            axis.autoscale_view(scalex=False, scaley=True)


def _apply_env_overrides(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg) -> None:
    if args_cli.control_mode is not None:
        env_cfg.control_mode = args_cli.control_mode
    if args_cli.position_delta_source is not None:
        env_cfg.position_delta_source = args_cli.position_delta_source
    if args_cli.use_fd_effort is not None:
        env_cfg.use_fd_effort = args_cli.use_fd_effort
    if args_cli.use_sd_observer is not None:
        env_cfg.use_sd_observer = args_cli.use_sd_observer
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


def _make_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(args_cli.log_root) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _step_dt(env: Any, env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg) -> float:
    value = getattr(env.unwrapped, "step_dt", None)
    if value is not None:
        return float(value)
    return float(env_cfg.sim.dt) * float(env_cfg.decimation)


def _save_static_plots(
    *,
    history: Mapping[str, Sequence[float]],
    output_dir: Path,
    joint_names: Sequence[str],
    window_steps: int,
) -> None:
    plotter = JointTrackingPlotter(joint_names=joint_names, window_steps=window_steps, interactive=False)
    try:
        plotter.update(history)
        plotter.save(output_dir)
    finally:
        plotter.close()


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    if args_cli.steps <= 0:
        raise ValueError("--steps must be positive.")
    if args_cli.window_steps <= 0:
        raise ValueError("--window_steps must be positive.")
    if args_cli.plot_interval <= 0:
        raise ValueError("--plot_interval must be positive.")

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.debug_reward_terms = True
    env_cfg.log_full_pipeline_debug = True
    _apply_env_overrides(env_cfg)
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device

    apply_kevin_sim_memory_optimizations(env_cfg.sim, verbose=True)

    output_dir = _make_output_dir()
    csv_path = output_dir / "joint_tracking.csv"
    live_plots = not args_cli.csv_only and not getattr(args_cli, "headless", False)
    if getattr(args_cli, "headless", False) and not args_cli.csv_only:
        print("[WARN] --headless disables live matplotlib windows; CSV logging will still run.", flush=True)

    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)

        env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))
        policy = None
        policy_device = agent_cfg.device
        if args_cli.checkpoint is not None:
            checkpoint = Path(args_cli.checkpoint)
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            print(f"[INFO] Loading checkpoint: {checkpoint}", flush=True)
            runner.load(str(checkpoint))
            policy = runner.get_inference_policy(device=policy_device)
        else:
            print("[INFO] No checkpoint provided. Running zero-action baseline.", flush=True)

        unwrapped = env.unwrapped
        if not hasattr(unwrapped, "get_debug_state"):
            raise RuntimeError(f"Task {args_cli.task!r} does not expose get_debug_state().")
        joint_names = list(getattr(unwrapped.cfg, "state_joint_names", JOINT_NAMES))
        command_joint_names = list(getattr(unwrapped.cfg, "command_joint_names", COMMAND_JOINT_NAMES))
        if len(joint_names) != 5:
            raise RuntimeError(f"Expected 5 state joints, got {len(joint_names)}: {joint_names}")
        if len(command_joint_names) != 4:
            raise RuntimeError(f"Expected 4 command joints, got {len(command_joint_names)}: {command_joint_names}")

        csv_fields = _make_csv_fields(joint_names, command_joint_names)
        history = {field: [] for field in csv_fields}
        plotter = None
        if live_plots:
            plotter = JointTrackingPlotter(
                joint_names=joint_names,
                window_steps=args_cli.window_steps,
                interactive=True,
            )

        obs = env.get_observations()
        if policy is not None:
            obs = obs.to(policy_device)
        dt = _step_dt(env, env_cfg)
        print(f"[INFO] Writing joint tracking CSV: {csv_path}", flush=True)

        with csv_path.open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
            writer.writeheader()
            for step in range(1, args_cli.steps + 1):
                with torch.inference_mode():
                    if policy is None:
                        actions = torch.zeros((env.num_envs, env.num_actions), dtype=torch.float32, device=env.device)
                    else:
                        actions = policy(obs).to(env.device)
                    obs, reward, dones, _ = env.step(actions)
                    if policy is not None:
                        obs = obs.to(policy_device)

                debug_state = unwrapped.get_debug_state(env_id=0)
                row = _build_row(
                    step=step,
                    step_dt=dt,
                    debug_state=debug_state,
                    reward=reward,
                    dones=dones,
                    joint_names=joint_names,
                    command_joint_names=command_joint_names,
                )
                writer.writerow(row)
                _append_history(history, row)
                if step == 1 or step % 50 == 0:
                    csv_file.flush()

                if plotter is not None and (step == 1 or step % args_cli.plot_interval == 0):
                    plotter.update(history)

        if plotter is not None:
            plotter.update(history)
        if args_cli.save_plots:
            if plotter is not None:
                plotter.save(output_dir)
            else:
                _save_static_plots(
                    history=history,
                    output_dir=output_dir,
                    joint_names=joint_names,
                    window_steps=args_cli.window_steps,
                )
            print(f"[INFO] Saved plots to: {output_dir}", flush=True)
        print(f"[INFO] Finished. Logged {args_cli.steps} steps to {csv_path}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()  # type: ignore
    finally:
        simulation_app.close()
