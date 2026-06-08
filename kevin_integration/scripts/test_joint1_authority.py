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

parser = argparse.ArgumentParser(description="Deterministic joint-1 authority test for Kevin adaptive RL.")
parser.add_argument("--task", type=str, default="Kevin-Mulag-AdaptiveRL-JointReach-v0", help="Gym task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of Isaac Lab environments.")
parser.add_argument("--steps_per_test", type=int, default=200, help="Action steps per test pattern.")
parser.add_argument("--settle_steps", type=int, default=50, help="Zero-action settle steps before each test.")
parser.add_argument("--control_mode", choices=["position_delta", "effort"], default="effort", help="Env control mode.")
parser.add_argument("--use_fd_effort", action="store_true", default=False, help="Run FD and apply effort targets.")
parser.add_argument("--max_abs_torque", type=float, default=5.0, help="FD torque clamp.")
parser.add_argument("--fd_residual_alpha", type=float, default=0.002, help="FD residual gain.")
parser.add_argument("--use_sd_observer", action="store_true", default=False, help="Run SD as a passive observer.")
parser.add_argument("--use_sd_feedback", action="store_true", default=False, help="Enable learned SD feedback.")
parser.add_argument("--sd_feedback_beta", type=float, default=0.01, help="Blend factor for SD feedback.")
parser.add_argument("--q_ref_ramp_time_s", type=float, default=2.0, help="q_ref commanded ramp time.")
parser.add_argument("--q_ref_target_scale", type=float, default=0.5, help="Fixed q_ref target scale.")
parser.add_argument("--episode_length_s", type=float, default=10.0, help="Episode duration.")
parser.add_argument(
    "--log_root",
    type=str,
    default="logs/joint1_authority",
    help="Directory under which timestamped authority logs are written.",
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


JOINT_NAMES = [
    "Drehzapfen_joint",
    "Ausleger_I_joint",
    "Ausleger_II_joint",
    "Messerkopf_Schwenk_joint",
    "Messerkopf_joint",
]
COMMAND_JOINT_NAMES = JOINT_NAMES[:4]
ACTION_PATTERNS = {
    "zero": [0.0, 0.0, 0.0, 0.0],
    "j1_pos": [1.0, 0.0, 0.0, 0.0],
    "j1_neg": [-1.0, 0.0, 0.0, 0.0],
    "j2_pos": [0.0, 1.0, 0.0, 0.0],
    "j3_pos": [0.0, 0.0, 1.0, 0.0],
    "j4_pos": [0.0, 0.0, 0.0, 1.0],
}


def _field(prefix: str, name: str) -> str:
    return f"{prefix}_{name.replace('_joint', '')}"


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


def _tensor_to_list(value: Any, *, expected_len: int, default: float = math.nan) -> list[float]:
    if isinstance(value, torch.Tensor):
        values = value.detach().flatten().cpu().tolist()
    elif isinstance(value, Sequence) and not isinstance(value, str):
        values = list(value)
    else:
        values = []
    if len(values) < expected_len:
        values = [*values, *([default] * (expected_len - len(values)))]
    return [float(item) for item in values[:expected_len]]


def _debug_scalar(debug_state: Mapping[str, Any], reward_debug: Mapping[str, Any], key: str) -> float:
    return _tensor_to_float(debug_state.get(key), _tensor_to_float(reward_debug.get(key)))


def _append_vector(row: dict[str, float | str], prefix: str, names: Sequence[str], values: Sequence[float]) -> None:
    for name, value in zip(names, values, strict=True):
        row[_field(prefix, name)] = float(value)


def _csv_fields() -> list[str]:
    fields = ["step", "test_name"]
    fields.extend(_field("action", name) for name in COMMAND_JOINT_NAMES)
    for prefix in ("q_actual", "qdot"):
        fields.extend(_field(prefix, name) for name in JOINT_NAMES)
    fields.extend(["q1", "q1_delta_from_start", "q1_velocity"])
    fields.extend(_field("qddot_cmd", name) for name in COMMAND_JOINT_NAMES)
    fields.extend(_field("valve_cmd", name) for name in COMMAND_JOINT_NAMES)
    fields.extend(_field("fd_tau_raw", name) for name in COMMAND_JOINT_NAMES)
    fields.extend(_field("tau_fd_adapted", name) for name in COMMAND_JOINT_NAMES)
    fields.extend(_field("tau_applied_raw", name) for name in COMMAND_JOINT_NAMES)
    fields.extend(_field("tau_isaac_filtered", name) for name in COMMAND_JOINT_NAMES)
    fields.extend(_field("torque", name) for name in COMMAND_JOINT_NAMES)
    fields.extend(
        [
            "dP_mean",
            "dP_abs_max",
            "fnet_mean",
            "fnet_abs_max",
            "done_frac",
            "torque_clamp_fraction",
            "sd_feedback_clamped_fraction",
        ]
    )
    return fields


def _make_action(env: RslRlVecEnvWrapper, pattern: Sequence[float]) -> torch.Tensor:
    action = torch.tensor(pattern, dtype=torch.float32, device=env.device).reshape(1, 4)
    return action.repeat(env.num_envs, 1)


def _reset_env(env: RslRlVecEnvWrapper):
    reset_out = env.reset()
    if isinstance(reset_out, tuple):
        return reset_out[0]
    return reset_out


def _step_env(env: RslRlVecEnvWrapper, action: torch.Tensor):
    return env.step(action)


def _collect_debug(unwrapped, reward: torch.Tensor, dones: torch.Tensor, test_name: str, step: int, q1_start: float):
    debug_state = unwrapped.get_debug_state(env_id=0)
    reward_debug = debug_state.get("reward_debug", {})
    if not isinstance(reward_debug, Mapping):
        reward_debug = {}

    q_actual = _tensor_to_list(debug_state.get("q_actual"), expected_len=5)
    qdot = _tensor_to_list(debug_state.get("qdot"), expected_len=5)
    action = _tensor_to_list(debug_state.get("action"), expected_len=4)
    qddot_cmd = _tensor_to_list(debug_state.get("qddot_cmd"), expected_len=5)[:4]
    valve_cmd = _tensor_to_list(debug_state.get("valve_cmd"), expected_len=4)
    fd_tau_raw = _tensor_to_list(debug_state.get("fd_tau_raw"), expected_len=4)
    tau_fd_adapted = _tensor_to_list(debug_state.get("tau_fd_adapted"), expected_len=4)
    tau_applied_raw = _tensor_to_list(debug_state.get("tau_applied_raw"), expected_len=4)
    tau_isaac_filtered = _tensor_to_list(debug_state.get("tau_isaac_filtered"), expected_len=4)
    torque = _tensor_to_list(debug_state.get("torque"), expected_len=4)
    dP = _tensor_to_list(debug_state.get("dP"), expected_len=4)
    fnet = _tensor_to_list(debug_state.get("fnet"), expected_len=4)

    row: dict[str, float | str] = {"step": float(step), "test_name": test_name}
    _append_vector(row, "action", COMMAND_JOINT_NAMES, action)
    _append_vector(row, "q_actual", JOINT_NAMES, q_actual)
    _append_vector(row, "qdot", JOINT_NAMES, qdot)
    row["q1"] = q_actual[0]
    row["q1_delta_from_start"] = q_actual[0] - q1_start
    row["q1_velocity"] = qdot[0]
    _append_vector(row, "qddot_cmd", COMMAND_JOINT_NAMES, qddot_cmd)
    _append_vector(row, "valve_cmd", COMMAND_JOINT_NAMES, valve_cmd)
    _append_vector(row, "fd_tau_raw", COMMAND_JOINT_NAMES, fd_tau_raw)
    _append_vector(row, "tau_fd_adapted", COMMAND_JOINT_NAMES, tau_fd_adapted)
    _append_vector(row, "tau_applied_raw", COMMAND_JOINT_NAMES, tau_applied_raw)
    _append_vector(row, "tau_isaac_filtered", COMMAND_JOINT_NAMES, tau_isaac_filtered)
    _append_vector(row, "torque", COMMAND_JOINT_NAMES, torque)
    row["dP_mean"] = sum(abs(value) for value in dP) / max(len(dP), 1)
    row["dP_abs_max"] = max(abs(value) for value in dP)
    row["fnet_mean"] = sum(abs(value) for value in fnet) / max(len(fnet), 1)
    row["fnet_abs_max"] = max(abs(value) for value in fnet)
    row["done_frac"] = _tensor_to_float(dones.float())
    row["torque_clamp_fraction"] = _debug_scalar(debug_state, reward_debug, "torque_clamp_fraction")
    row["sd_feedback_clamped_fraction"] = _debug_scalar(debug_state, reward_debug, "sd_feedback_clamped_fraction")
    _ = reward
    return row


def _mean(values: Sequence[float]) -> float:
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return float("nan")
    return sum(finite_values) / len(finite_values)


def _max(values: Sequence[float]) -> float:
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return float("nan")
    return max(finite_values)


def _summarize(test_name: str, rows: list[dict[str, float | str]], q1_start: float) -> dict[str, float | str]:
    q1_end = float(rows[-1]["q1"]) if rows else q1_start
    q1_delta = q1_end - q1_start
    sign = 0.0
    if q1_delta > 0.0:
        sign = 1.0
    elif q1_delta < 0.0:
        sign = -1.0
    return {
        "test_name": test_name,
        "q1_start": q1_start,
        "q1_end": q1_end,
        "q1_delta": q1_delta,
        "q1_delta_sign": sign,
        "mean_q1_velocity": _mean([float(row["q1_velocity"]) for row in rows]),
        "mean_abs_valve1": _mean([abs(float(row[_field("valve_cmd", COMMAND_JOINT_NAMES[0])])) for row in rows]),
        "mean_abs_fd_tau1": _mean([abs(float(row[_field("fd_tau_raw", COMMAND_JOINT_NAMES[0])])) for row in rows]),
        "mean_abs_tau_applied1": _mean(
            [abs(float(row[_field("tau_applied_raw", COMMAND_JOINT_NAMES[0])])) for row in rows]
        ),
        "mean_abs_torque1": _mean([abs(float(row[_field("torque", COMMAND_JOINT_NAMES[0])])) for row in rows]),
        "done_frac": _max([float(row["done_frac"]) for row in rows]),
        "torque_clamp_fraction": _mean([float(row["torque_clamp_fraction"]) for row in rows]),
    }


def _add_zero_relative_diagnostics(rows: list[dict[str, float | str]]) -> None:
    zero_row = next((row for row in rows if row["test_name"] == "zero"), None)
    if zero_row is None:
        return
    zero_q1_delta = float(zero_row["q1_delta"])
    zero_valve1 = float(zero_row["mean_abs_valve1"])
    zero_tau1 = float(zero_row["mean_abs_tau_applied1"])
    zero_torque1 = float(zero_row["mean_abs_torque1"])
    for row in rows:
        row["q1_delta_minus_zero"] = float(row["q1_delta"]) - zero_q1_delta
        row["valve1_delta_minus_zero"] = float(row["mean_abs_valve1"]) - zero_valve1
        row["tau1_delta_minus_zero"] = float(row["mean_abs_tau_applied1"]) - zero_tau1
        row["torque1_delta_minus_zero"] = float(row["mean_abs_torque1"]) - zero_torque1


def _print_summary(rows: list[dict[str, float | str]]) -> None:
    fields = [
        "test_name",
        "q1_start",
        "q1_end",
        "q1_delta",
        "q1_delta_minus_zero",
        "q1_delta_sign",
        "mean_q1_velocity",
        "mean_abs_valve1",
        "valve1_delta_minus_zero",
        "mean_abs_fd_tau1",
        "mean_abs_tau_applied1",
        "tau1_delta_minus_zero",
        "mean_abs_torque1",
        "torque1_delta_minus_zero",
        "done_frac",
        "torque_clamp_fraction",
    ]
    widths = {field: max(len(field), 12) for field in fields}
    for row in rows:
        for field in fields:
            value = row[field]
            text = str(value) if isinstance(value, str) else f"{value:.6g}"
            widths[field] = max(widths[field], len(text))
    print(" | ".join(field.ljust(widths[field]) for field in fields), flush=True)
    print("-+-".join("-" * widths[field] for field in fields), flush=True)
    for row in rows:
        parts = []
        for field in fields:
            value = row[field]
            text = str(value) if isinstance(value, str) else f"{value:.6g}"
            parts.append(text.ljust(widths[field]))
        print(" | ".join(parts), flush=True)


def _print_diagnosis(summary_rows: list[dict[str, float | str]]) -> None:
    by_name = {str(row["test_name"]): row for row in summary_rows}
    j1_pos = float(by_name.get("j1_pos", {}).get("q1_delta_minus_zero", float("nan")))
    j1_neg = float(by_name.get("j1_neg", {}).get("q1_delta_minus_zero", float("nan")))
    motion_threshold = 1.0e-4
    opposite = math.isfinite(j1_pos) and math.isfinite(j1_neg) and (j1_pos * j1_neg < 0.0)
    separated = math.isfinite(j1_pos) and math.isfinite(j1_neg) and abs(j1_pos - j1_neg) > motion_threshold
    if opposite and separated:
        print("[PASS] j1_pos and j1_neg move joint 1 in opposite directions.", flush=True)
    else:
        print("[WARN] j1_pos and j1_neg did not produce clear opposite-signed joint-1 motion.", flush=True)
    if abs(j1_pos) < motion_threshold and abs(j1_neg) < motion_threshold:
        print("[WARN] joint 1 barely moves for both j1_pos and j1_neg.", flush=True)
    max_j1_action_delta = max(abs(j1_pos), abs(j1_neg))
    cross_deltas = [
        abs(float(by_name.get(name, {}).get("q1_delta_minus_zero", 0.0))) for name in ("j2_pos", "j3_pos", "j4_pos")
    ]
    if cross_deltas and max(cross_deltas) > max(max_j1_action_delta, motion_threshold):
        print("[WARN] j2/j3/j4 commands move joint 1 more than direct joint-1 actions.", flush=True)
    for name in ("j1_pos", "j1_neg"):
        row = by_name.get(name)
        if row is None:
            continue
        if float(row["mean_abs_valve1"]) < 1.0e-4:
            print(f"[WARN] valve1 is near zero during {name}.", flush=True)
        if float(row["mean_abs_torque1"]) < 1.0e-4:
            print(f"[WARN] final torque1 is near zero during {name}.", flush=True)


def _apply_env_overrides(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg) -> None:
    env_cfg.control_mode = args_cli.control_mode
    if args_cli.use_fd_effort:
        env_cfg.use_fd_effort = True
    env_cfg.max_abs_torque = args_cli.max_abs_torque
    env_cfg.fd_residual_alpha = args_cli.fd_residual_alpha
    env_cfg.use_fd_residual_alpha = True
    if args_cli.use_sd_observer:
        env_cfg.use_sd_observer = True
    if args_cli.use_sd_feedback:
        env_cfg.use_sd_observer = True
        env_cfg.use_sd_feedback = True
    env_cfg.sd_feedback_beta = args_cli.sd_feedback_beta
    env_cfg.q_ref_ramp_time_s = args_cli.q_ref_ramp_time_s
    env_cfg.q_ref_target_scale_initial = args_cli.q_ref_target_scale
    env_cfg.q_ref_target_scale_final = args_cli.q_ref_target_scale
    env_cfg.use_q_ref_curriculum = False
    env_cfg.episode_length_s = args_cli.episode_length_s


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    if args_cli.steps_per_test <= 0:
        raise ValueError("--steps_per_test must be positive.")
    if args_cli.settle_steps < 0:
        raise ValueError("--settle_steps must be non-negative.")
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
    csv_path = output_dir / "joint1_authority.csv"

    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)

        env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))
        unwrapped = env.unwrapped
        if not hasattr(unwrapped, "get_debug_state"):
            raise RuntimeError(f"Task {args_cli.task!r} does not expose get_debug_state().")

        csv_fields = _csv_fields()
        summary_rows: list[dict[str, float | str]] = []
        print(f"[INFO] Writing joint-1 authority CSV: {csv_path}", flush=True)
        with csv_path.open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
            writer.writeheader()
            zero_action = _make_action(env, ACTION_PATTERNS["zero"])
            for test_name, pattern in ACTION_PATTERNS.items():
                _reset_env(env)
                for _ in range(args_cli.settle_steps):
                    _, _, _, _ = _step_env(env, zero_action)
                debug_state = unwrapped.get_debug_state(env_id=0)
                q_start = _tensor_to_list(debug_state.get("q_actual"), expected_len=5)
                q1_start = q_start[0]
                action = _make_action(env, pattern)
                test_rows: list[dict[str, float | str]] = []
                for step in range(1, args_cli.steps_per_test + 1):
                    _, reward, dones, _ = _step_env(env, action)
                    row = _collect_debug(unwrapped, reward, dones, test_name, step, q1_start)
                    writer.writerow(row)
                    test_rows.append(row)
                csv_file.flush()
                summary_rows.append(_summarize(test_name, test_rows, q1_start))

        _add_zero_relative_diagnostics(summary_rows)
        _print_summary(summary_rows)
        _print_diagnosis(summary_rows)
        print(f"[INFO] Finished. Logged authority test to {csv_path}", flush=True)
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
