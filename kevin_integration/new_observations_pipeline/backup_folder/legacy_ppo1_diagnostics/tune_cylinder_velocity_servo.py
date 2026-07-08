#!/usr/bin/env python3
from __future__ import annotations
"""Tune the lower-level cylinder velocity servo without PPO.

Runs direct cylinder velocity references through:
desired velocity -> ID feedforward + PID residual -> FD -> adapter -> PhysX.
"""

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isaaclab.app import AppLauncher


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tune cylinder velocity servo without PPO.")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--duration_s", type=float, default=1.5)
    parser.add_argument("--settle_s", type=float, default=0.25)
    parser.add_argument("--servo_mode", type=str, default="id_ff_plus_cylinder_velocity_pid",
                        choices=("cylinder_velocity_pid", "id_ff_plus_cylinder_velocity_pid", "position_ramp_pi", "fd_tracking_servo", "id_pid_fd_tracking_servo"))
    parser.add_argument("--fd_usage", type=str, default="prediction_only",
                        choices=("prediction_only", "prediction_plus_feedback_delta", "bypass_for_diagnostic_only"))
    parser.add_argument("--velocities", type=str, default="-0.05,-0.02,-0.01,-0.005,0.005,0.01,0.02,0.05")
    parser.add_argument("--kp", type=float, default=6.0)
    parser.add_argument("--ki", type=float, default=0.0)
    parser.add_argument("--kd", type=float, default=0.0)
    parser.add_argument("--valve_cmd_limit", type=float, default=0.1)
    parser.add_argument("--cyl_vel_cmd_limit", type=float, default=0.05)
    parser.add_argument("--servo_tracking_delta_gain", type=float, default=0.0)
    parser.add_argument("--servo_fd_correction_gain", type=float, default=0.0)
    parser.add_argument("--cylinders", type=str, default="0,1,2,3",
                        help="Comma-separated cylinder indices to test, e.g. 0,2")
    parser.add_argument("--servo_rate_limit", type=float, default=0.0)
    parser.add_argument("--fd_output_gain", type=float, default=1.0)
    parser.add_argument("--sequence_fd_delta_gain", type=float, default=0.5)
    parser.add_argument("--output_dir", type=str,
                        default=str(ROOT / "outputs/cylinder_diagnostics/standard_control_reset"))
    AppLauncher.add_app_launcher_args(parser)
    return parser


parser = _build_parser()
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]
if getattr(args_cli, "device", None) in (None, ""):
    args_cli.device = "cuda:0"

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def _tensor_stats(tensor: torch.Tensor) -> dict[str, float]:
    data = tensor.detach().abs()
    return {
        "mean_abs": float(data.mean().cpu()),
        "max_abs": float(data.max().cpu()),
    }


def _write_physx_audit(env, path: Path) -> None:
    uw = env.unwrapped if hasattr(env, "unwrapped") else env
    joint_ids = uw._cylinder_joint_ids
    names = list(uw.cfg.cylinder_joint_names)
    data = uw.robot.data

    def _read_attr(name: str) -> torch.Tensor | None:
        value = getattr(data, name, None)
        if value is None:
            return None
        try:
            return value[:, joint_ids].detach().cpu()
        except Exception:
            return None

    stiffness = _read_attr("joint_stiffness")
    damping = _read_attr("joint_damping")
    effort = _read_attr("joint_effort_limits")
    velocity = _read_attr("joint_vel_limits")
    soft_limits = data.soft_joint_pos_limits[:, joint_ids].detach().cpu()

    actuator_lines = []
    for group_name, actuator_cfg in getattr(uw.cfg.robot_cfg, "actuators", {}).items():
        cls_name = getattr(getattr(actuator_cfg, "class_type", None), "__name__", str(getattr(actuator_cfg, "class_type", "")))
        joints_expr = getattr(actuator_cfg, "joint_names_expr", None)
        actuator_lines.append(f"- `{group_name}`: `{cls_name}`, joint_names_expr={joints_expr}")

    path.parent.mkdir(parents=True, exist_ok=True)
    joint_id_list = [int(x) for x in joint_ids.detach().cpu().tolist()]
    with path.open("w", encoding="utf-8") as stream:
        stream.write("# PhysX Actuator Tracking Audit\n\n")
        stream.write("Official IsaacLab actuator docs used for this checklist:\n")
        stream.write("- https://isaac-sim.github.io/IsaacLab/main/source/overview/core-concepts/actuators.html\n")
        stream.write("- https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.actuators.html\n\n")
        stream.write("IsaacLab states that implicit actuators are physics-engine actuators: desired position or velocity targets are tracked by the physics engine, and stiffness/damping are written into PhysX. Velocity limits can also constrain achieved velocity if effort is insufficient.\n\n")
        stream.write("## Runtime Configuration\n\n")
        stream.write(f"- low_level_servo_mode: `{uw.cfg.low_level_servo_mode}`\n")
        stream.write(f"- cylinder_dynamics_model: `{uw.cfg.cylinder_dynamics_model}`\n")
        stream.write(f"- use_sequence_cylinder_models: `{uw.cfg.use_sequence_cylinder_models}`\n")
        stream.write(f"- sim.dt: `{float(uw.cfg.sim.dt):.9g}`\n")
        stream.write(f"- decimation: `{int(uw.cfg.decimation)}`\n")
        stream.write(f"- env step_dt: `{float(uw.step_dt):.9g}`\n")
        stream.write(f"- sequence_model_dt: `{float(getattr(uw, '_sequence_model_dt', 0.0)):.9g}`\n")
        stream.write(f"- sequence_decimation: `{int(getattr(uw, '_sequence_decimation', 1))}`\n")
        stream.write("- command write path: `robot.set_joint_position_target(self._cylinder_pos_target, joint_ids=self._cylinder_joint_ids)` every `_apply_action()` call.\n\n")
        stream.write("## Actuator Groups\n\n")
        if actuator_lines:
            stream.write("\n".join(actuator_lines) + "\n\n")
        else:
            stream.write("- No actuator config groups found on `cfg.robot_cfg.actuators`.\n\n")
        stream.write("## Cylinder Joint Runtime Data\n\n")
        stream.write("| cylinder | joint_id | pos_limit | stiffness | damping | effort_limit | velocity_limit |\n")
        stream.write("|---|---:|---|---:|---:|---:|---:|\n")
        for local_i, name in enumerate(names):
            def _cell(t: torch.Tensor | None) -> str:
                if t is None:
                    return "nan"
                return f"{float(t[0, local_i]):.6g}"
            lo = float(soft_limits[0, local_i, 0])
            hi = float(soft_limits[0, local_i, 1])
            stream.write(
                f"| {name} | {joint_id_list[local_i]} | [{lo:.6g}, {hi:.6g}] | "
                f"{_cell(stiffness)} | {_cell(damping)} | {_cell(effort)} | {_cell(velocity)} |\n"
            )
        stream.write("\n## Audit Findings\n\n")
        stream.write("- Actuator command mode used by this env: position targets for cylinder joints.\n")
        stream.write("- Target commands are written every env step in `_apply_action()`.\n")
        stream.write("- Servo FD/adapter increments are logged in `cylinder_velocity_servo_tuning.csv` as `fd_delta_abs_mean`, `actual_delta_abs_mean`, and `target_clip_frac`.\n")
        stream.write("- If `valve_total_abs_mean` is high but `fd_delta_abs_mean` is tiny, FD is the bottleneck.\n")
        stream.write("- If FD delta is visible but `actual_delta_abs_mean` is tiny or `target_clip_frac` is high, inspect PhysX drive stiffness/damping, velocity limits, effort limits, and position-target clipping.\n")


def _stage_failure(row: dict[str, Any]) -> str:
    eps = 1.0e-9
    desired = abs(float(row["v_desired_mps"]))
    measured = abs(float(row["measured_mean_mps"]))
    if abs(float(row["valve_ff_abs_mean"])) < 1.0e-4 and "id_ff" in str(row.get("servo_mode", row.get("mode", ""))):
        return "A: valve_ff near zero; ID feedforward is weak or out of distribution"
    if abs(float(row["valve_pid_abs_mean"])) > 1.0e-3 and abs(float(row["valve_total_abs_mean"])) < 1.0e-3:
        return "B: PID is nonzero but valve_total remains small; valve limit/rate scaling is wrong"
    if abs(float(row["valve_total_abs_mean"])) > 1.0e-3 and abs(float(row["fd_delta_abs_mean"])) < 1.0e-6:
        return "C: valve_total is visible but FD delta is tiny; FD is the bottleneck"
    if abs(float(row["fd_delta_abs_mean"])) > 1.0e-6 and abs(float(row["actual_delta_abs_mean"])) < 1.0e-7:
        return "D/E: FD delta exists but adapter/PhysX motion is tiny"
    if desired > eps and measured / desired < 0.1:
        return "response attenuated >10x; inspect stage ratios"
    return "pass_or_inconclusive"


def _run_trial(env, cylinder_i: int, v_desired: float, steps: int, settle_steps: int) -> dict[str, Any]:
    uw = env.unwrapped if hasattr(env, "unwrapped") else env
    obs, _ = env.reset()
    zero_action = torch.zeros((uw.num_envs, uw.cfg.action_space), device=uw.device)
    target = torch.zeros((uw.num_envs, 4), device=uw.device)
    target[:, cylinder_i] = float(v_desired)

    samples: list[dict[str, float]] = []
    prev_x = None
    for step in range(steps + settle_steps):
        uw.set_debug_cylinder_velocity_target(target)
        cyl_q_before = uw.robot.data.joint_pos[:, uw._cylinder_joint_ids].detach().clone()
        cyl_qdot_before = uw.robot.data.joint_vel[:, uw._cylinder_joint_ids].detach().clone()
        x_before, _ = uw._cylinder_adapter.current_from_cylinder_joints(cyl_q_before, cyl_qdot_before)
        obs, reward, terminated, truncated, info = env.step(zero_action)
        cyl_q_after = uw.robot.data.joint_pos[:, uw._cylinder_joint_ids].detach().clone()
        cyl_qdot_after = uw.robot.data.joint_vel[:, uw._cylinder_joint_ids].detach().clone()
        x_after, xdot_after = uw._cylinder_adapter.current_from_cylinder_joints(cyl_q_after, cyl_qdot_after)
        actual_delta = x_after - x_before
        if step < settle_steps:
            continue
        err = uw._cyl_vel_desired - uw._cyl_vel_measured
        limits = uw.robot.data.soft_joint_pos_limits[:, uw._cylinder_joint_ids]
        clip = (
            torch.isclose(uw._cylinder_pos_target, limits[..., 0])
            | torch.isclose(uw._cylinder_pos_target, limits[..., 1])
        )
        samples.append({
            "measured": float(uw._cyl_vel_measured[:, cylinder_i].mean().detach().cpu()),
            "measured_abs": float(uw._cyl_vel_measured[:, cylinder_i].abs().mean().detach().cpu()),
            "err_sq": float((err[:, cylinder_i] ** 2).mean().detach().cpu()),
            "valve_ff_abs": float(uw._valve_ff[:, cylinder_i].abs().mean().detach().cpu()),
            "valve_pid_abs": float(uw._valve_pid[:, cylinder_i].abs().mean().detach().cpu()),
            "valve_total_abs": float(uw._valve_total[:, cylinder_i].abs().mean().detach().cpu()),
            "valve_sat": float((uw._valve_total[:, cylinder_i].abs() >= 0.95 * float(uw.cfg.valve_cmd_limit)).float().mean().detach().cpu()),
            "fd_delta_abs": float(uw._fd_delta_corrected[:, cylinder_i].abs().mean().detach().cpu()),
            "actual_delta_abs": float(actual_delta[:, cylinder_i].abs().mean().detach().cpu()),
            "clip": float(clip[:, cylinder_i].float().mean().detach().cpu()),
        })

    if not samples:
        raise RuntimeError("No samples collected; increase duration_s.")

    measured = np.array([s["measured"] for s in samples], dtype=np.float64)
    measured_abs = np.array([s["measured_abs"] for s in samples], dtype=np.float64)
    err_sq = np.array([s["err_sq"] for s in samples], dtype=np.float64)
    sign_accuracy = float(np.mean(np.sign(measured) == np.sign(float(v_desired)))) if abs(v_desired) > 0 else 1.0
    target_abs = abs(float(v_desired))
    threshold = 0.9 * target_abs
    rise_idx = np.where(np.abs(measured) >= threshold)[0]
    rise_time = float(rise_idx[0] * uw.step_dt) if len(rise_idx) else math.nan
    overshoot = float(max(0.0, measured_abs.max() - target_abs))
    steady_state_error = float(np.mean(measured[-max(1, len(measured) // 4):]) - float(v_desired))

    row = {
        "servo_mode": str(uw.cfg.low_level_servo_mode),
        "fd_usage": str(getattr(uw.cfg, "fd_servo_usage", "prediction_only")),
        "cylinder": int(cylinder_i),
        "v_desired_mps": float(v_desired),
        "measured_mean_mps": float(measured.mean()),
        "measured_abs_mean_mps": float(measured_abs.mean()),
        "tracking_rmse_mps": float(np.sqrt(err_sq.mean())),
        "sign_accuracy": sign_accuracy,
        "rise_time_s": rise_time,
        "overshoot_mps": overshoot,
        "steady_state_error_mps": steady_state_error,
        "valve_ff_abs_mean": float(np.mean([s["valve_ff_abs"] for s in samples])),
        "valve_pid_abs_mean": float(np.mean([s["valve_pid_abs"] for s in samples])),
        "valve_total_abs_mean": float(np.mean([s["valve_total_abs"] for s in samples])),
        "valve_total_saturation_frac": float(np.mean([s["valve_sat"] for s in samples])),
        "fd_delta_abs_mean": float(np.mean([s["fd_delta_abs"] for s in samples])),
        "actual_delta_abs_mean": float(np.mean([s["actual_delta_abs"] for s in samples])),
        "target_clip_frac": float(np.mean([s["clip"] for s in samples])),
        "attenuation_ratio": float(measured_abs.mean() / max(target_abs, 1.0e-9)),
    }
    row["failed_stage"] = _stage_failure(row)
    return row


def main() -> None:
    from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env import MulagUpperLevelEnv
    from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env_cfg import MulagUpperLevelEnvCfg
    from kevin_integration.utils.sim_memory import apply_kevin_sim_memory_optimizations

    output_dir = Path(args_cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = MulagUpperLevelEnvCfg()
    cfg.scene.num_envs = int(args_cli.num_envs)
    cfg.low_level_servo_mode = str(args_cli.servo_mode)
    cfg.cyl_vel_kp = [float(args_cli.kp)] * 4
    cfg.pos_pi_kp = [float(args_cli.kp)] * 4
    cfg.pos_pi_ki = [float(args_cli.ki)] * 4
    cfg.pos_pi_kd = [float(args_cli.kd)] * 4
    cfg.cyl_vel_ki = [float(args_cli.ki)] * 4
    cfg.cyl_vel_kd = [float(args_cli.kd)] * 4
    cfg.valve_cmd_limit = float(args_cli.valve_cmd_limit)
    cfg.max_abs_valve = max(float(args_cli.valve_cmd_limit), float(getattr(cfg, "max_abs_valve", 1.0)))
    cfg.cyl_vel_cmd_limit = float(args_cli.cyl_vel_cmd_limit)
    cfg.servo_tracking_delta_gain = float(args_cli.servo_tracking_delta_gain)
    cfg.servo_rate_limit = float(args_cli.servo_rate_limit)
    cfg.fd_output_gain = float(args_cli.fd_output_gain)
    cfg.sequence_fd_delta_gain = float(args_cli.sequence_fd_delta_gain)
    cfg.enable_scripted_vehicle_motion = False
    cfg.enable_startup_summary = True
    if getattr(args_cli, "device", None):
        cfg.device = args_cli.device
        cfg.sim.device = args_cli.device
    cfg.fd_servo_usage = str(args_cli.fd_usage)
    cfg.servo_fd_correction_gain = float(getattr(args_cli, "servo_fd_correction_gain", 0.0))
    apply_kevin_sim_memory_optimizations(cfg.sim)

    env = MulagUpperLevelEnv(cfg=cfg)
    steps = max(int(round(float(args_cli.duration_s) / float(env.step_dt))), 1)
    settle_steps = max(int(round(float(args_cli.settle_s) / float(env.step_dt))), 0)
    velocities = [float(x.strip()) for x in str(args_cli.velocities).split(",") if x.strip()]

    _write_physx_audit(env, output_dir / "physx_actuator_tracking_audit.md")

    rows: list[dict[str, Any]] = []
    cylinder_indices = [int(x.strip()) for x in str(args_cli.cylinders).split(",") if x.strip()]
    for cylinder_i in cylinder_indices:
        for v_desired in velocities:
            print(f"[servo-tune] cyl={cylinder_i} v={v_desired:+.4f} m/s", flush=True)
            row = _run_trial(env, cylinder_i, v_desired, steps, settle_steps)
            rows.append(row)
            print(
                "[servo-tune] "
                f"rmse={row['tracking_rmse_mps']:.5g} sign={row['sign_accuracy']:.2f} "
                f"atten={row['attenuation_ratio']:.3f} stage={row['failed_stage']}",
                flush=True,
            )

    csv_path = output_dir / "cylinder_velocity_servo_tuning.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    report_path = output_dir / "cylinder_velocity_servo_tuning_report.md"
    passed = [
        row for row in rows
        if float(row["sign_accuracy"]) >= 0.9
        and float(row["attenuation_ratio"]) >= 0.1
        and float(row["tracking_rmse_mps"]) <= max(0.02, abs(float(row["v_desired_mps"])))
    ]
    with report_path.open("w", encoding="utf-8") as stream:
        stream.write("# Cylinder Velocity Servo Tuning Report\n\n")
        stream.write(f"- mode: `{args_cli.mode}`\n")
        stream.write(f"- trials: `{len(rows)}`\n")
        stream.write(f"- pass_count: `{len(passed)}`\n")
        stream.write(f"- output_csv: `{csv_path}`\n")
        stream.write(f"- physx_audit: `{output_dir / 'physx_actuator_tracking_audit.md'}`\n\n")
        stream.write("## Failed Stage Summary\n\n")
        counts: dict[str, int] = {}
        for row in rows:
            counts[str(row["failed_stage"])] = counts.get(str(row["failed_stage"]), 0) + 1
        for name, count in sorted(counts.items(), key=lambda kv: kv[0]):
            stream.write(f"- {name}: {count}\n")
        stream.write("\n## Worst RMSE Trials\n\n")
        stream.write("| cylinder | v_desired | rmse | sign_accuracy | attenuation | failed_stage |\n")
        stream.write("|---:|---:|---:|---:|---:|---|\n")
        for row in sorted(rows, key=lambda r: float(r["tracking_rmse_mps"]), reverse=True)[:12]:
            stream.write(
                f"| {row['cylinder']} | {float(row['v_desired_mps']):+.4f} | "
                f"{float(row['tracking_rmse_mps']):.6f} | {float(row['sign_accuracy']):.3f} | "
                f"{float(row['attenuation_ratio']):.3f} | {row['failed_stage']} |\n"
            )

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()