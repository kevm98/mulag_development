from __future__ import annotations

import argparse
import csv
import math
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from types import MethodType
from typing import Any

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser(description="Direct Isaac joint-effort authority test for Kevin adaptive RL.")
parser.add_argument("--task", type=str, default="Kevin-Mulag-AdaptiveRL-JointReach-v0", help="Gym task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of Isaac Lab environments.")
parser.add_argument("--steps_per_test", type=int, default=200, help="Direct-effort steps per test pattern.")
parser.add_argument("--settle_steps", type=int, default=50, help="Zero-effort settle steps before each test.")
parser.add_argument(
    "--effort_values",
    type=str,
    default="0.25,0.5,1.0,2.0,5.0",
    help='Comma-separated direct Isaac effort magnitudes, e.g. "0.25,0.5,1.0,2.0,5.0".',
)
parser.add_argument(
    "--joint_names",
    type=str,
    default="Drehzapfen_joint,Ausleger_I_joint,Ausleger_II_joint,Messerkopf_Schwenk_joint",
    help="Comma-separated actuated joint names to resolve for direct effort.",
)
parser.add_argument(
    "--single_joint_name",
    type=str,
    default=None,
    help="Optional single-joint diagnostic override. Example: Drehzapfen_joint.",
)
parser.add_argument(
    "--log_root",
    type=str,
    default="logs/joint_effort_authority",
    help="Directory under which timestamped direct-effort logs are written.",
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


def _parse_effort_values(text: str) -> list[float]:
    values: list[float] = []
    for item in text.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        values.append(float(stripped))
    if not values:
        raise ValueError("--effort_values must contain at least one number.")
    return values


def _parse_joint_names(text: str) -> list[str]:
    names = [item.strip() for item in text.split(",") if item.strip()]
    if not names:
        raise ValueError("--joint_names must contain at least one joint name.")
    return names


def _resolve_joint_ids(robot, joint_names: Sequence[str]) -> list[int]:
    ids: list[int] = []
    for joint_name in joint_names:
        joint_ids, resolved_names = robot.find_joints(joint_name)
        if len(joint_ids) == 0:
            raise RuntimeError(f"Could not resolve joint name {joint_name!r}.")
        joint_id = int(joint_ids[0])
        resolved_name = resolved_names[0] if resolved_names else getattr(robot, "joint_names", [joint_name])[joint_id]
        ids.append(joint_id)
        print(f"[INFO] Resolved direct effort joint: requested={joint_name} id={joint_id} actual={resolved_name}", flush=True)
    return ids


def _robot_joint_names(robot) -> list[str]:
    names = getattr(robot, "joint_names", None)
    if names is None:
        names = getattr(robot.data, "joint_names", None)
    if names is None:
        return []
    return list(names)


def _format_tensor_slice(value: Any, joint_ids: Sequence[int]) -> str:
    if not isinstance(value, torch.Tensor):
        return "unavailable"
    tensor = value.detach().cpu()
    if tensor.ndim >= 2:
        tensor = tensor[0]
    selected = []
    for joint_id in joint_ids:
        if 0 <= joint_id < tensor.numel():
            selected.append(float(tensor.flatten()[joint_id]))
        else:
            selected.append(float("nan"))
    return "[" + ", ".join(f"{item:.6g}" for item in selected) + "]"


def _print_direct_effort_diagnostics(
    *,
    robot,
    command_joint_names: Sequence[str],
    direct_joint_ids: Sequence[int],
    effort_values: Sequence[float],
    single_joint_name: str | None,
    num_envs: int,
) -> None:
    joint_names = _robot_joint_names(robot)
    resolved_names = [
        joint_names[joint_id] if 0 <= joint_id < len(joint_names) else "unavailable" for joint_id in direct_joint_ids
    ]
    print(f"[INFO] Direct effort requested joint names: {list(command_joint_names)}", flush=True)
    print(f"[INFO] Direct effort resolved joint IDs: {list(direct_joint_ids)}", flush=True)
    print(f"[INFO] Direct effort resolved joint names: {resolved_names}", flush=True)
    if single_joint_name:
        print(
            f"[INFO] Single-joint diagnostic target: name={single_joint_name} "
            f"id={direct_joint_ids[0] if direct_joint_ids else 'unavailable'}",
            flush=True,
        )
    print(
        f"[INFO] Direct effort command tensor shape: ({num_envs}, {len(direct_joint_ids)})",
        flush=True,
    )
    print(f"[INFO] Diagnostic effort magnitudes: {list(effort_values)}", flush=True)
    print(f"[INFO] Drehzapfen_joint in robot joint names: {'Drehzapfen_joint' in joint_names}", flush=True)
    print(
        f"[INFO] Direct effort includes Drehzapfen_joint: {'Drehzapfen_joint' in resolved_names}",
        flush=True,
    )
    for field_name in ("joint_effort_limits", "soft_joint_effort_limits", "effort_limits"):
        value = getattr(robot.data, field_name, None)
        if isinstance(value, torch.Tensor):
            print(
                f"[INFO] robot.data.{field_name} for direct effort joints: "
                f"{_format_tensor_slice(value, direct_joint_ids)}",
                flush=True,
            )
    for field_name in ("joint_velocity_limits", "soft_joint_velocity_limits", "joint_vel_limits", "soft_joint_vel_limits"):
        value = getattr(robot.data, field_name, None)
        if isinstance(value, torch.Tensor):
            print(
                f"[INFO] robot.data.{field_name} for direct effort joints: "
                f"{_format_tensor_slice(value, direct_joint_ids)}",
                flush=True,
            )


def _build_patterns(command_joint_names: Sequence[str]) -> dict[str, list[float]]:
    patterns = {"zero": [0.0 for _ in command_joint_names]}
    if len(command_joint_names) == 1:
        patterns[f"{command_joint_names[0].replace('_joint', '')}_pos"] = [1.0]
        patterns[f"{command_joint_names[0].replace('_joint', '')}_neg"] = [-1.0]
        return patterns
    for index, _ in enumerate(command_joint_names):
        sign_vector = [0.0 for _ in command_joint_names]
        sign_vector[index] = 1.0
        patterns[f"joint{index + 1}_pos"] = sign_vector
        if index == 0:
            neg_vector = [0.0 for _ in command_joint_names]
            neg_vector[index] = -1.0
            patterns[f"joint{index + 1}_neg"] = neg_vector
    return patterns


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


def _append_vector(row: dict[str, float | str], prefix: str, names: Sequence[str], values: Sequence[float]) -> None:
    for name, value in zip(names, values, strict=True):
        row[_field(prefix, name)] = float(value)


def _csv_fields(command_joint_names: Sequence[str]) -> list[str]:
    fields = ["step", "effort_value", "test_name", "target_joint_name", "target_joint_id"]
    fields.extend(_field("effort_command", name) for name in command_joint_names)
    for prefix in ("q_actual", "qdot"):
        fields.extend(_field(prefix, name) for name in JOINT_NAMES)
    fields.extend(["q1", "q1_delta_from_start", "q1_velocity"])
    fields.extend(_field("applied_effort", name) for name in command_joint_names)
    fields.extend(["done_frac", "torque_clamp_fraction"])
    return fields


def _make_effort(env: RslRlVecEnvWrapper, pattern_signs: Sequence[float], effort_value: float) -> torch.Tensor:
    effort = torch.tensor(pattern_signs, dtype=torch.float32, device=env.device).reshape(1, len(pattern_signs))
    return (float(effort_value) * effort).repeat(env.num_envs, 1)


def _make_zero_action(env: RslRlVecEnvWrapper) -> torch.Tensor:
    return torch.zeros((env.num_envs, env.num_actions), dtype=torch.float32, device=env.device)


def _install_direct_effort_override(unwrapped, direct_joint_ids: Sequence[int]) -> None:
    if not hasattr(unwrapped.robot, "set_joint_effort_target"):
        raise RuntimeError("This IsaacLab Articulation does not expose set_joint_effort_target.")

    unwrapped._direct_effort_joint_ids = list(direct_joint_ids)  # noqa: SLF001
    unwrapped._direct_effort_command = torch.zeros(  # noqa: SLF001
        (unwrapped.num_envs, len(direct_joint_ids)), dtype=torch.float32, device=unwrapped.device
    )
    unwrapped._direct_effort_pre_physics_calls = 0  # noqa: SLF001
    unwrapped._direct_effort_apply_action_calls = 0  # noqa: SLF001

    def _direct_pre_physics_step(self, actions: torch.Tensor) -> None:
        self._direct_effort_pre_physics_calls += 1
        self._update_q_ref_commanded()
        previous_action = self._prev_action.detach().clone()
        self._current_action = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self._action_delta = self._current_action - previous_action
        self._current_qddot_cmd.zero_()
        self._prev_valve_cmd.zero_()
        self._current_valve_cmd.zero_()
        self._current_fd_simscape_output.zero_()
        self._current_tau_fd_adapted.zero_()
        self._current_tau_applied_raw.zero_()
        self._current_tau_isaac_filtered.zero_()
        self._current_tau_isaac_clamped.zero_()
        self._current_torque = self._direct_effort_command.detach().clone()
        self._torque_clamp_fraction.zero_()
        self._torque_adapter_clamp_fraction.zero_()
        self._torque_rate_limited_fraction.zero_()
        self._zero_sd_observer_state()
        self._zero_sd_feedback_state()
        self._dP.zero_()
        self._fnet.zero_()
        _ = actions

    def _direct_apply_action(self) -> None:
        self._direct_effort_apply_action_calls += 1
        self.robot.set_joint_effort_target(self._direct_effort_command, joint_ids=self._direct_effort_joint_ids)

    unwrapped._pre_physics_step = MethodType(_direct_pre_physics_step, unwrapped)  # noqa: SLF001
    unwrapped._apply_action = MethodType(_direct_apply_action, unwrapped)  # noqa: SLF001
    print(
        "[INFO] Installed direct-effort env.step override: Kevin _pre_physics_step/_apply_action are replaced "
        "for this diagnostic, so PPO/ID/FD/SD/torque adapter cannot overwrite the direct effort command.",
        flush=True,
    )


def _reset_env(env: RslRlVecEnvWrapper):
    reset_out = env.reset()
    if isinstance(reset_out, tuple):
        return reset_out[0]
    return reset_out


def _reward_debug(debug_state: Mapping[str, Any]) -> Mapping[str, Any]:
    value = debug_state.get("reward_debug", {})
    if isinstance(value, Mapping):
        return value
    return {}


def _collect_row(
    *,
    unwrapped,
    dones: torch.Tensor,
    step: int,
    effort_value: float,
    test_name: str,
    effort_command: torch.Tensor,
    command_joint_names: Sequence[str],
    command_joint_ids: Sequence[int],
    q1_start: float,
) -> dict[str, float | str]:
    debug_state = unwrapped.get_debug_state(env_id=0)
    reward_debug = _reward_debug(debug_state)
    q_actual = _tensor_to_list(debug_state.get("q_actual"), expected_len=5)
    qdot = _tensor_to_list(debug_state.get("qdot"), expected_len=5)
    command = _tensor_to_list(effort_command[0], expected_len=len(command_joint_names))
    applied_effort = _tensor_to_list(debug_state.get("torque", command), expected_len=len(command_joint_names))
    active_indices = [index for index, value in enumerate(command) if abs(value) > 0.0]
    target_index = active_indices[0] if active_indices else -1
    row: dict[str, float | str] = {
        "step": float(step),
        "effort_value": float(effort_value),
        "test_name": test_name,
        "target_joint_name": command_joint_names[target_index] if target_index >= 0 else "none",
        "target_joint_id": float(command_joint_ids[target_index]) if target_index >= 0 else -1.0,
    }
    _append_vector(row, "effort_command", command_joint_names, command)
    _append_vector(row, "q_actual", JOINT_NAMES, q_actual)
    _append_vector(row, "qdot", JOINT_NAMES, qdot)
    row["q1"] = q_actual[0]
    row["q1_delta_from_start"] = q_actual[0] - q1_start
    row["q1_velocity"] = qdot[0]
    _append_vector(row, "applied_effort", command_joint_names, applied_effort)
    row["done_frac"] = _tensor_to_float(dones.float())
    row["torque_clamp_fraction"] = _tensor_to_float(
        debug_state.get("torque_clamp_fraction"), _tensor_to_float(reward_debug.get("torque_clamp_fraction"), 0.0)
    )
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


def _summarize(effort_value: float, test_name: str, rows: list[dict[str, float | str]], q1_start: float):
    q1_end = float(rows[-1]["q1"]) if rows else q1_start
    q1_delta = q1_end - q1_start
    sign = 0.0
    if q1_delta > 0.0:
        sign = 1.0
    elif q1_delta < 0.0:
        sign = -1.0
    velocities = [float(row["q1_velocity"]) for row in rows]
    return {
        "effort_value": effort_value,
        "test_name": test_name,
        "q1_start": q1_start,
        "q1_end": q1_end,
        "q1_delta": q1_delta,
        "q1_delta_sign": sign,
        "mean_q1_velocity": _mean(velocities),
        "max_abs_q1_velocity": _max([abs(value) for value in velocities]),
        "done_frac": _max([float(row["done_frac"]) for row in rows]),
    }


def _print_summary(rows: list[dict[str, float | str]]) -> None:
    fields = [
        "effort_value",
        "test_name",
        "q1_start",
        "q1_end",
        "q1_delta",
        "q1_delta_sign",
        "mean_q1_velocity",
        "max_abs_q1_velocity",
        "done_frac",
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


def _print_diagnosis(summary_rows: list[dict[str, float | str]], command_joint_names: Sequence[str]) -> None:
    rows_by_effort: dict[float, dict[str, dict[str, float | str]]] = {}
    for row in summary_rows:
        rows_by_effort.setdefault(float(row["effort_value"]), {})[str(row["test_name"])] = row

    any_direct_pass = False
    joint1_deltas_by_effort: list[tuple[float, float]] = []
    motion_threshold = 1.0e-4
    for effort_value, rows in rows_by_effort.items():
        if len(command_joint_names) == 1:
            base = command_joint_names[0].replace("_joint", "")
            pos_name = f"{base}_pos"
            neg_name = f"{base}_neg"
            cross_names: tuple[str, ...] = ()
        else:
            pos_name = "joint1_pos"
            neg_name = "joint1_neg"
            cross_names = tuple(f"joint{index + 1}_pos" for index in range(1, len(command_joint_names)))
        j1_pos = float(rows.get(pos_name, {}).get("q1_delta", float("nan")))
        j1_neg = float(rows.get(neg_name, {}).get("q1_delta", float("nan")))
        opposite = math.isfinite(j1_pos) and math.isfinite(j1_neg) and (j1_pos * j1_neg < 0.0)
        separated = math.isfinite(j1_pos) and math.isfinite(j1_neg) and abs(j1_pos - j1_neg) > motion_threshold
        if opposite and separated:
            any_direct_pass = True
            joint1_deltas_by_effort.append((effort_value, max(abs(j1_pos), abs(j1_neg))))
            print(f"[PASS] effort={effort_value:g}: {pos_name}/{neg_name} move q1 in opposite directions.", flush=True)
        else:
            print(
                f"[WARN] effort={effort_value:g}: {pos_name}/{neg_name} did not produce clear opposite q1 motion.",
                flush=True,
            )
        direct_motion = max(abs(j1_pos), abs(j1_neg))
        if cross_names:
            cross_motion = max(abs(float(rows.get(name, {}).get("q1_delta", 0.0))) for name in cross_names)
            if cross_motion > max(direct_motion, motion_threshold):
                print(f"[WARN] effort={effort_value:g}: non-joint1 commands move q1 more than direct joint1 effort.", flush=True)

    if len(joint1_deltas_by_effort) >= 2:
        sorted_deltas = sorted(joint1_deltas_by_effort)
        if sorted_deltas[-1][1] <= sorted_deltas[0][1]:
            print("[WARN] increasing direct effort did not increase joint-1 motion magnitude.", flush=True)

    if any_direct_pass:
        print(
            "[PASS] Direct Isaac effort can move joint 1; investigate the learned ID/FD/torque-adapter path.",
            flush=True,
        )
    else:
        print(
            "[WARN] Direct Isaac effort did not clearly move joint 1; inspect joint order, effort application, "
            "asset constraints, and drive settings.",
            flush=True,
        )


def _apply_env_overrides(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg) -> None:
    env_cfg.control_mode = "effort"
    env_cfg.use_fd_effort = False
    env_cfg.use_sd_observer = False
    env_cfg.use_sd_feedback = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    if args_cli.steps_per_test <= 0:
        raise ValueError("--steps_per_test must be positive.")
    if args_cli.settle_steps < 0:
        raise ValueError("--settle_steps must be non-negative.")
    effort_values = _parse_effort_values(args_cli.effort_values)
    command_joint_names = [args_cli.single_joint_name] if args_cli.single_joint_name else _parse_joint_names(args_cli.joint_names)
    patterns = _build_patterns(command_joint_names)
    if args_cli.single_joint_name or any(abs(value) > 5.0 for value in effort_values):
        print(
            "[WARN] High/single-joint direct efforts are diagnostic only. Do not use these settings for training.",
            flush=True,
        )
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
    csv_path = output_dir / "direct_joint_effort_authority.csv"

    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)

        env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))
        unwrapped = env.unwrapped
        if not hasattr(unwrapped, "get_debug_state"):
            raise RuntimeError(f"Task {args_cli.task!r} does not expose get_debug_state().")
        direct_joint_ids = _resolve_joint_ids(unwrapped.robot, command_joint_names)
        _print_direct_effort_diagnostics(
            robot=unwrapped.robot,
            command_joint_names=command_joint_names,
            direct_joint_ids=direct_joint_ids,
            effort_values=effort_values,
            single_joint_name=args_cli.single_joint_name,
            num_envs=env.num_envs,
        )
        print("[INFO] Direct effort action-index mapping:", flush=True)
        for index, (joint_name, joint_id) in enumerate(zip(command_joint_names, direct_joint_ids, strict=True)):
            actual_name = unwrapped.robot.joint_names[joint_id] if hasattr(unwrapped.robot, "joint_names") else joint_name
            print(
                f"  action_index={index} expected={joint_name} isaac_joint_id={joint_id} actual={actual_name}",
                flush=True,
            )
        _install_direct_effort_override(unwrapped, direct_joint_ids)

        csv_fields = _csv_fields(command_joint_names)
        summary_rows: list[dict[str, float | str]] = []
        zero_action = _make_zero_action(env)
        zero_effort = _make_effort(env, patterns["zero"], 0.0)

        print(f"[INFO] Writing direct joint effort authority CSV: {csv_path}", flush=True)
        with csv_path.open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
            writer.writeheader()
            for effort_value in effort_values:
                for test_name, pattern_signs in patterns.items():
                    _reset_env(env)
                    unwrapped._direct_effort_command = zero_effort.detach().clone()  # noqa: SLF001
                    for _ in range(args_cli.settle_steps):
                        _, _, _, _ = env.step(zero_action)
                    debug_state = unwrapped.get_debug_state(env_id=0)
                    q_start = _tensor_to_list(debug_state.get("q_actual"), expected_len=5)
                    q1_start = q_start[0]
                    effort_command = _make_effort(env, pattern_signs, effort_value)
                    if test_name == "zero":
                        effort_command.zero_()
                    unwrapped._direct_effort_command = effort_command.detach().clone()  # noqa: SLF001
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
                            command_joint_names=command_joint_names,
                            command_joint_ids=direct_joint_ids,
                            q1_start=q1_start,
                        )
                        writer.writerow(row)
                        test_rows.append(row)
                    csv_file.flush()
                    summary_rows.append(_summarize(effort_value, test_name, test_rows, q1_start))

        _print_summary(summary_rows)
        print(
            f"[INFO] Direct override call counts: _pre_physics_step={unwrapped._direct_effort_pre_physics_calls} "
            f"_apply_action={unwrapped._direct_effort_apply_action_calls}",
            flush=True,
        )
        _print_diagnosis(summary_rows, command_joint_names)
        print(f"[INFO] Finished. Logged direct effort authority test to {csv_path}", flush=True)
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
