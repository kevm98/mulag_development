#!/usr/bin/env python3
"""Stage 11 DLS-only pilot: one nominal DLS trajectory, full signal recording.

Run the deployed DLS teacher chain (Task reference -> EE cascade -> cylinder
DLS -> action port -> command ROM -> PhysX position drive) for one nominal
trajectory with no added excitation, recording the same signal set as the
Stage 8 master collector plus the DLS-produced 4-D action. The recorded
arrays are then fed through the Stage 10 identifiability diagnostics
(input rank/SVD/kappa, lagged persistent excitation, actuator->response
authority, state coverage).

Headless by default (AppLauncher). Modeled on the verified ground-arc gate
``mulag_plant/dls_solver/tests/ground_arc_tracking_gate.py``, which already
runs this exact teacher chain against the head-down mowing arc.

Usage:
    python dls_pilot.py --steps 900 --out ../results/dls_pilot
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from isaaclab.app import AppLauncher


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
AAM_ROOT = Path(__file__).resolve().parents[1]
if str(AAM_ROOT) not in sys.path:
    sys.path.insert(0, str(AAM_ROOT))


p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--steps", type=int, default=900, help="control steps after settle")
p.add_argument("--seed", type=int, default=17)
p.add_argument(
    "--arc", default=str(ROOT / "mulag_plant/models/mowing_arc_head_down.json")
)
p.add_argument("--speed_mps", type=float, default=0.18)
p.add_argument("--accel_time_s", type=float, default=1.0)
p.add_argument("--seed_lead_time_s", type=float, default=0.25)
p.add_argument("--cylinder_error_gain", type=float, default=0.18)
p.add_argument("--cylinder_velocity_feedforward", type=float, default=1.0)
p.add_argument("--action_smoothing", type=float, default=0.20)
p.add_argument("--ee_correction_gain", type=float, default=0.15)
p.add_argument("--ground_height_bias_m", type=float, default=-0.030)
p.add_argument("--preposition_steps", type=int, default=200)
p.add_argument("--settle_steps", type=int, default=60)
p.add_argument("--out", default=str(AAM_ROOT / "results/dls_pilot"))
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()

launcher = AppLauncher(a)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402

import mulag_plant.tasks  # noqa: E402,F401
from mulag_plant.dls_solver.cylinder import CylinderDLSConfig, CylinderDLSController  # noqa: E402
from mulag_plant.dls_solver.direct_drive import DirectCylinderDriver  # noqa: E402
from mulag_plant.dls_solver.ee_cascade import EECascadeController  # noqa: E402
from mulag_plant.trajectories.time_scaling import cruise_profile  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from aam_data import (  # noqa: E402
    COMMAND_SIGNALS,
    FLAG_SIGNALS,
    STATE_SIGNALS,
    UNAVAILABLE_SIGNALS,
    write_run,
)
from aam_data.arm_motion import ACTION_SCALE, coverage_stats  # noqa: E402
from tools.action_identifiability import (  # noqa: E402
    authority_diagnostics,
    input_diagnostics,
)


torch.manual_seed(int(a.seed))
np.random.seed(int(a.seed))

TASK = "RRLAB-Mulag-PPO2-ROM-PhysX-EETracking-PLAY-v0"


def main() -> None:
    cfg = parse_env_cfg(TASK, device=a.device, num_envs=1)
    cfg.seed = int(a.seed)
    cfg.enable_scripted_vehicle_motion = False
    cfg.episode_length_s = 100000.0
    cfg.ee_success_hold_steps = 100000
    cfg.low_level_servo_mode = "none"
    cfg.sim.physx.enable_external_forces_every_iteration = True
    env = gym.make(TASK, cfg=cfg).unwrapped

    payload = json.loads(Path(a.arc).expanduser().resolve().read_text())
    teacher = payload["cylinder_teacher"]
    theta = np.asarray([row["theta_deg"] for row in teacher], dtype=np.float64)
    goals = np.asarray([row["cutter_truck"] for row in teacher], dtype=np.float64)
    seeds = np.asarray([row["cylinder_cmd"] for row in teacher], dtype=np.float64)
    world_z = np.asarray([row["cutter_world_z"] for row in teacher], dtype=np.float64)
    order = np.argsort(theta)
    theta, goals, seeds, world_z = theta[order], goals[order], seeds[order], world_z[order]
    teacher_s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(goals, axis=0), axis=1))]
    arc_length = float(teacher_s[-1])

    def interpolate(distance_m: float) -> tuple[np.ndarray, np.ndarray, float]:
        s = float(np.clip(distance_m, 0.0, arc_length))
        goal = np.asarray(
            [np.interp(s, teacher_s, goals[:, i]) for i in range(3)], dtype=np.float32
        )
        seed = np.asarray(
            [np.interp(s, teacher_s, seeds[:, i]) for i in range(4)], dtype=np.float32
        )
        return goal, seed, float(np.interp(s, teacher_s, world_z))

    def hold_wheels(target: torch.Tensor) -> None:
        ids = env._wheel_joint_ids
        env.robot.write_joint_stiffness_to_sim(2.0e9, joint_ids=ids)
        env.robot.write_joint_damping_to_sim(1.0e7, joint_ids=ids)
        env.robot.set_joint_position_target(target, joint_ids=ids)
        env.robot.set_joint_velocity_target(torch.zeros_like(target), joint_ids=ids)

    def constrain_truck_xy(anchor_xy: torch.Tensor) -> None:
        base_xy = env.robot.data.body_link_state_w[:, env._truck_base_body_id(), :2]
        correction = anchor_xy - base_xy
        if float(torch.linalg.norm(correction, dim=1).max().item()) <= 0.001:
            return
        root_pose = env.robot.data.root_state_w[:, :7].detach().clone()
        root_pose[:, :2] += correction
        env.robot.write_root_pose_to_sim(root_pose)

    def grab(name: str, tensor: torch.Tensor) -> np.ndarray:
        return tensor[0].detach().cpu().numpy().astype(np.float32)

    env.reset(seed=int(a.seed))
    wheel_target = env.robot.data.joint_pos[:, env._wheel_joint_ids].detach().clone()
    hold_wheels(wheel_target)

    # Deploy to the nearest verified endpoint with a cosine ramp (no impulse).
    driver = DirectCylinderDriver(env)
    reset_x = driver.measured_x()[0].detach().cpu().numpy()
    if np.linalg.norm(reset_x - seeds[-1]) < np.linalg.norm(reset_x - seeds[0]):
        goals = goals[::-1].copy()
        seeds = seeds[::-1].copy()
        world_z = world_z[::-1].copy()
        teacher_s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(goals, axis=0), axis=1))]
        arc_length = float(teacher_s[-1])

    target_a = torch.as_tensor(seeds[0], device=env.device, dtype=torch.float32).reshape(1, 4)
    start_x = driver.measured_x().clone()
    for step in range(int(a.preposition_steps)):
        u = float(step + 1) / max(int(a.preposition_steps), 1)
        blend = 0.5 * (1.0 - math.cos(math.pi * u))
        deploy_target = start_x + blend * (target_a - start_x)
        hold_wheels(wheel_target)
        driver.step(deploy_target, substeps=2)

    scaling = env._cylinder_adapter.scaling
    q_a = scaling.clamp_usd_q(scaling.learned_x_to_usd_q(target_a))
    env._cylinder_pos_target[:] = q_a
    env._neutral_hold_cylinder_pos_target[:] = q_a
    env._prev_valve_cmd.zero_()
    env._current_valve_cmd.zero_()
    env._fd_delta_raw.zero_()
    env._fd_delta_corrected.zero_()
    env._prev_action.zero_()
    env._current_action.zero_()
    env._command_cylinder_rom.reset()

    ground_j = torch.tensor(
        json.loads(
            (ROOT / "mulag_plant/models/ee_jacobian_ground_truck3x4.json").read_text()
        )["jacobian"],
        device=env.device,
        dtype=torch.float32,
    )
    ground_j[:, 3] = 0.0  # cylinder 4 owns head pitch
    cylinder = CylinderDLSController(
        env,
        CylinderDLSConfig(
            invert_learned_pipeline=False,
            error_gain=float(a.cylinder_error_gain),
            velocity_damping=0.0,
            velocity_feedforward=float(a.cylinder_velocity_feedforward),
            max_desired_delta_m=0.004,
            max_action=1.0,
            action_smoothing=float(a.action_smoothing),
        ),
    )
    cascade = EECascadeController(
        env,
        cylinder_controller=cylinder,
        jacobian=ground_j,
        ee_feedback_gain=0.0,
        max_ik_step_m=0.004,
        max_cylinder_lead_m=0.025,
    )
    cascade.reset()
    for _ in range(int(a.settle_steps)):
        hold_wheels(wheel_target)
        cylinder.step(target_a)

    truck_start = env.robot.data.body_link_state_w[
        0, env._truck_base_body_id(), :3
    ].detach().clone()
    truck_anchor_xy = truck_start[:2].reshape(1, 2)

    zero_t = torch.zeros(1, device=env.device)
    _s0, _v0, _a0, one_way_s = cruise_profile(
        zero_t,
        length_m=arc_length,
        cruise_speed_mps=float(a.speed_mps),
        accel_time_s=float(a.accel_time_s),
        decel_time_s=float(a.accel_time_s),
    )

    # One A->B->A cycle if the requested step count allows, else a truncated
    # forward leg; record from control step 0 (after settle).
    steps = int(a.steps)
    goal_log: list[np.ndarray] = []
    action_log: list[np.ndarray] = []
    rom_tick_log: list[int] = []
    state = {k: [] for k in STATE_SIGNALS}
    cmd_buf = {k: [] for k in COMMAND_SIGNALS}
    flag_buf = {k: [] for k in FLAG_SIGNALS}
    time_s = []
    control_dt = float(env.step_dt)

    def snapshot_state(step_idx: int) -> None:
        robot = env.robot
        cyl_q = robot.data.joint_pos[:, env._cylinder_joint_ids]
        cyl_qdot = robot.data.joint_vel[:, env._cylinder_joint_ids]
        x, xdot = env._cylinder_adapter.current_from_cylinder_joints(cyl_q, cyl_qdot)
        ee_pos_w = robot.data.body_link_state_w[:, env._ee_body_id, :3]
        ee_quat_w = robot.data.body_link_state_w[:, env._ee_body_id, 3:7]
        vals = {
            "cylinder_extension": grab("x", x),
            "cylinder_velocity": grab("xdot", xdot),
            "dP": grab("dP", env._dP),
            "joint_position": robot.data.joint_pos[0, env._arm_joint_ids]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32),
            "joint_velocity": robot.data.joint_vel[0, env._arm_joint_ids]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32),
            "ee_pos_local": grab("ee", env._compute_ee_pos_local()),
            "ee_pos_w": grab("eew", ee_pos_w),
            "ee_quat_w": grab("eeq", ee_quat_w),
            "ee_lin_vel_w": robot.data.body_link_lin_vel_w[0, env._ee_body_id]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32),
            "ee_ang_vel_w": robot.data.body_link_ang_vel_w[0, env._ee_body_id]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32),
        }
        for k, v in vals.items():
            state[k].append(v)
        time_s.append(step_idx * control_dt)

    def snapshot_command(probe_action: torch.Tensor) -> None:
        valve = grab("valve", env._current_valve_cmd)
        nan4 = np.full(4, np.nan, dtype=np.float32)
        nan1 = np.full(1, np.nan, dtype=np.float32)
        nan3 = np.full(3, np.nan, dtype=np.float32)
        cmd_buf["probe_excitation"].append(grab("probe", probe_action))
        cmd_buf["controller_output"].append(grab("ctl", env._current_action))
        cmd_buf["desired_cylinder_position"].append(grab("ref", env._cylinder_ref))
        cmd_buf["desired_cylinder_velocity"].append(grab("vel", env._cylinder_vel_ref))
        cmd_buf["desired_delta_extension"].append(
            grab("dx", env._action_target_delta_x_learned)
        )
        cmd_buf["desired_delta_extension_velocity"].append(
            grab("dxv", env._action_target_delta_xdot_learned)
        )
        cmd_buf["command_before_limits"].append(
            grab("pre", env._action_target_delta_x_pre_gain_learned)
        )
        cmd_buf["command_after_limits"].append(
            grab("post", env._action_target_delta_x_learned)
        )
        cmd_buf["requested_valve_cmd"].append(valve)
        cmd_buf["applied_valve_cmd"].append(nan4)
        cmd_buf["rom_predicted_valve_cmd"].append(valve)
        cmd_buf["rom_effective_actuation"].append(valve)
        cmd_buf["delta_extension_per_step"].append(
            grab("ext", env._predicted_delta_extension)
        )
        cmd_buf["delta_isaac"].append(grab("isaac", env._delta_isaac))
        cmd_buf["cylinder_pos_target"].append(grab("target", env._cylinder_pos_target))
        cmd_buf["rom_tick"].append(
            np.array(
                [env._command_cylinder_rom.counter[0].item() == 1], dtype=np.float32
            )
        )
        q = env.robot.data.joint_pos[:, env._cylinder_joint_ids]
        lo = env.robot.data.soft_joint_pos_limits[:, env._cylinder_joint_ids, 0]
        hi = env.robot.data.soft_joint_pos_limits[:, env._cylinder_joint_ids, 1]
        eps = 1e-3
        near_limit = ((q - lo).abs() < eps) | ((hi - q).abs() < eps)
        flag_buf["cylinder_limit_flag"].append(
            near_limit[0].detach().cpu().numpy().astype(np.float32)
        )
        pre = env._action_target_delta_x_pre_gain_clip_mask
        post = env._action_target_delta_x_post_gain_clip_mask
        dxdot = env._action_target_delta_xdot_clip_mask
        flag_buf["command_saturation_flag"].append(
            (pre | post | dxdot)[0].detach().cpu().numpy().astype(np.float32)
        )
        flag_buf["valve_saturation_flag"].append(
            (np.abs(valve) >= 1.0 - 1e-6).astype(np.float32)
        )
        vals = torch.cat(
            (
                env._cylinder_x_current,
                env._cylinder_xdot_current,
                env._dP,
                env._delta_isaac,
                env._cylinder_pos_target,
            ),
            dim=-1,
        )
        finite = torch.isfinite(vals).all(dim=-1)
        flag_buf["invalid_physics_flag"].append(
            np.array([(~finite)[0].item()], dtype=np.float32)
        )
        flag_buf["dls_condition_number"].append(nan1)
        flag_buf["dls_damping"].append(nan1)
        flag_buf["contact_state"].append(nan1)
        flag_buf["contact_force"].append(nan3)

    snapshot_state(0)
    for step in range(steps):
        started = time.perf_counter()
        elapsed = step * control_dt
        cycle_t = elapsed % (2.0 * one_way_s)
        outward = cycle_t <= one_way_s
        stroke_t = cycle_t if outward else cycle_t - one_way_s
        profile_t = stroke_t if outward else one_way_s - stroke_t
        distance, speed, _accel, _ = cruise_profile(
            torch.tensor([profile_t], device=env.device),
            length_m=arc_length,
            cruise_speed_mps=float(a.speed_mps),
            accel_time_s=float(a.accel_time_s),
            decel_time_s=float(a.accel_time_s),
        )
        goal_np, _seed, _wz = interpolate(float(distance.item()))
        goal_np = goal_np.copy()
        goal_np[2] += float(a.ground_height_bias_m)
        direction = 1.0 if outward else -1.0
        seed_distance = float(distance.item()) + direction * abs(float(speed.item())) * float(
            a.seed_lead_time_s
        )
        _sg, seed_np, _sw = interpolate(seed_distance)
        seed_step_distance = seed_distance + direction * abs(float(speed.item())) * control_dt
        _ng, next_seed_np, _nw = interpolate(seed_step_distance)
        seed_velocity_np = (next_seed_np - seed_np) / max(control_dt, 1.0e-9)
        goal = torch.as_tensor(goal_np, device=env.device).reshape(1, 3)
        seed = torch.as_tensor(seed_np, device=env.device).reshape(1, 4)
        cylinder_goal = cascade.cylinder_goal_seeded(
            goal, seed, correction_gain=float(a.ee_correction_gain)
        )
        cylinder_goal[:, 3] = seed[:, 3]
        seed_velocity = torch.as_tensor(seed_velocity_np, device=env.device).reshape(1, 4)
        action, _sv = cylinder.action(cylinder_goal, goal_velocity=seed_velocity)
        env._cylinder_goal_x[:] = cylinder_goal
        constrain_truck_xy(truck_anchor_xy)
        hold_wheels(wheel_target)
        env.step(action)

        goal_log.append(goal_np.copy())
        action_log.append(action[0].detach().cpu().numpy().copy())
        snapshot_command(action)
        snapshot_state(step + 1)
        if step % 300 == 0 or step == steps - 1:
            print(
                f"[DLS-PILOT] step={step:05d}/{steps-1} "
                f"goal={np.round(goal_np, 3).tolist()} "
                f"|u|={np.abs(action[0].detach().cpu().numpy()).max():.4f}",
                flush=True,
            )

    U = np.asarray(action_log, dtype=np.float64)
    responses = {
        "dP": np.stack(state["dP"])[:-1],
        "velocity": np.stack(state["cylinder_velocity"])[:-1],
        "extension": np.stack(state["cylinder_extension"])[:-1],
    }
    lags = (1, 3, 6, 12)
    input_report = input_diagnostics(U, "dls_pilot", lags)
    authority_report = authority_diagnostics(U, responses, lags)

    # Coverage on the recorded state trajectory.
    traj_seeds = np.stack(state["cylinder_extension"])
    from types import SimpleNamespace

    pseudo = SimpleNamespace(
        name="dls_pilot",
        source="legacy_arc",
        dt=control_dt,
        goals=np.stack(state["ee_pos_local"]),
        seeds=traj_seeds,
        world_z=np.stack(state["ee_pos_w"])[:, 2],
        path_s=np.arange(len(traj_seeds), dtype=np.float32) * control_dt,
        action=U,
    )
    cov = coverage_stats(pseudo)

    arrays = {"time_s": np.asarray(time_s, dtype=np.float32)}
    for k in STATE_SIGNALS:
        arrays[k] = np.stack(state[k], axis=0)
    for k in COMMAND_SIGNALS:
        arrays[k] = np.stack(cmd_buf[k], axis=0)
    for k in FLAG_SIGNALS:
        arrays[k] = np.stack(flag_buf[k], axis=0)

    metadata = {
        "schema_version": "dls_pilot_v1",
        "run_id": 0,
        "domain": "nominal",
        "split": "pilot",
        "random_seed": int(a.seed),
        "physics_dt": float(env.cfg.sim.dt),
        "control_dt": control_dt,
        "rom_dt": float(env._command_cylinder_rom.sample_time_s),
        "task_name": TASK,
        "rom_checkpoint": str(env._command_cylinder_rom.checkpoint_path),
        "num_control_steps": steps,
        "dls_parameters": {
            "error_gain": float(a.cylinder_error_gain),
            "velocity_feedforward": float(a.cylinder_velocity_feedforward),
            "action_smoothing": float(a.action_smoothing),
            "ee_correction_gain": float(a.ee_correction_gain),
            "seed_lead_time_s": float(a.seed_lead_time_s),
            "speed_mps": float(a.speed_mps),
            "arc_length_m": arc_length,
            "one_way_s": one_way_s,
        },
        "start_time": datetime.now(timezone.utc).isoformat(),
        "end_time": datetime.now(timezone.utc).isoformat(),
        "availability": {
            name: (name not in UNAVAILABLE_SIGNALS)
            for name in (list(STATE_SIGNALS) + list(COMMAND_SIGNALS) + list(FLAG_SIGNALS))
        },
        "action_scale": ACTION_SCALE,
    }
    out = Path(a.out)
    write_run(out, metadata, arrays)

    report = {
        "metadata": metadata,
        "input": input_report,
        "authority": authority_report,
        "coverage": cov,
        "action_stats": {
            "rms": [float(v) for v in np.std(U, axis=0)],
            "max_abs": [float(v) for v in np.abs(U).max(axis=0)],
        },
    }
    (out / "identifiability.json").write_text(json.dumps(report, indent=2))

    print("\n[DLS-PILOT] input diagnostics:")
    print(f"  U shape={input_report['shape']} rank={input_report['rank']} "
          f"num_rank={input_report['effective_numerical_rank']} "
          f"kappa={input_report['condition_number']:.3g}")
    print("  singular values:", [f"{v:.4g}" for v in input_report["singular_values"]])
    for L in lags:
        lag = input_report.get(f"lagged_L{L}")
        if lag:
            print(f"  U_L L={L}: rank={lag['rank']} num_rank={lag['effective_numerical_rank']} "
                  f"kappa={lag['condition_number']:.3g}")
    print("  cross-channel correlation:")
    for row in input_report["cross_channel_correlation"]:
        print("   ", [f"{v:.2f}" if v == v else "  - " for v in row])
    print("\n[DLS-PILOT] authority (max abs lagged correlation):")
    for sig in ("dP", "velocity", "extension"):
        best = max(
            y["max_abs"]
            for u_row in authority_report[sig]["lagged_correlation"]
            for y in u_row.values()
        )
        print(f"  {sig}: {best:.3f}")
    io = authority_report.get("input_output_singular_values")
    if io:
        print("  io_sv:", [f"{v:.3g}" for v in io[:6]])
    print("\n[DLS-PILOT] coverage:")
    print(f"  extension span (m): {[round(v,4) for v in cov['extension_span_m']]}")
    print(f"  dominant cyl: {cov['dominant_cylinder']} frac={cov['dominant_span_frac']}")
    print(f"  both-sign velocity: {cov['velocity_both_signs']}")
    print(f"  direction reversals: {cov['direction_reversals']}")
    print(f"  world_z: {cov['world_z_min_m']:.3f} .. {cov['world_z_max_m']:.3f}")
    print(f"\n[DLS-PILOT] wrote {out}")

    env.close()


try:
    main()
finally:
    simulation_app.close()
