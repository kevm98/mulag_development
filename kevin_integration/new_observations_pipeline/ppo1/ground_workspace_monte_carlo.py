#!/usr/bin/env python3
"""Sample reachable tool-contact workspace and observe PPO1 ground trajectories."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isaaclab.app import AppLauncher


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("sample", "sample_dynamic", "observe", "sweep", "axis_probe", "authority_sweep"),
        default="sample",
    )
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--horizon_steps", type=int, default=10)
    parser.add_argument("--settle_steps", type=int, default=1)
    parser.add_argument("--joint_margin_frac", type=float, default=0.05)
    parser.add_argument("--z_tolerance", type=float, default=0.15)
    parser.add_argument("--min_arm_clearance", type=float, default=0.02)
    parser.add_argument("--tool_ground_clearance", type=float, default=0.03)
    parser.add_argument("--ground_z", type=float, default=0.0)
    parser.add_argument("--robot_spawn_z", type=float, default=-1.0, help=">=0 overrides the truck spawn height (grounding scan)")
    parser.add_argument("--trajectory_scale", type=float, default=0.8)
    parser.add_argument("--trajectory_margin", type=float, default=0.05)
    parser.add_argument("--trajectory_points", type=int, default=200)
    parser.add_argument("--ground_path_pacing", choices=("fixed", "capture"), default="fixed")
    parser.add_argument("--ground_path_capture_radius_m", type=float, default=0.20)
    parser.add_argument("--ground_path_capture_required_hold_steps", type=int, default=5)
    parser.add_argument(
        "--ground_initial_target_mode",
        choices=("path_start", "nearest_path", "local_offset"),
        default="path_start",
    )
    parser.add_argument("--ground_initial_target_offset_m", type=float, default=0.08)
    parser.add_argument("--ground_initial_target_capture_radius_m", type=float, default=0.03)
    parser.add_argument("--ground_anchor_path_to_current_tool", action="store_true")
    parser.add_argument("--ground_anchor_margin_m", type=float, default=0.05)
    parser.add_argument("--ground_anchor_max_span_m", type=float, default=0.30)
    parser.add_argument("--ground_anchor_static_sag_m", type=float, default=0.03)
    parser.add_argument("--ground_use_root_hold_frame", type=int, choices=(0, 1), default=-1)
    parser.add_argument("--ground_xy_condition_path_to_j_eff", type=int, choices=(0, 1), default=-1)
    parser.add_argument("--ground_xy_condition_path_perp_scale", type=float, default=-1.0)
    parser.add_argument("--ground_xy_condition_path_min_sv", type=float, default=-1.0)
    parser.add_argument("--ground_xy_conditioned_xy_clip_m", type=float, default=-1.0)
    parser.add_argument("--ground_xy_z_hold_gain", type=float, default=-1.0)
    parser.add_argument("--ground_xy_z_hold_clip_m", type=float, default=-1.0)
    parser.add_argument("--ground_xy_z_hold_deadband_m", type=float, default=-1.0)
    parser.add_argument("--ground_xy_z_hold_priority_threshold_m", type=float, default=-1.0)
    # Integrate cylinder command in reference space (target += dx) instead of
    # re-anchoring to the sagged measured position each step (fixes gravity ratchet).
    parser.add_argument("--reference_space_target", action="store_true")
    # Spawn the per-env mowable shelf (embankment) at the tool's low reachable band.
    parser.add_argument("--mowing_terrain", type=int, choices=(0, 1), default=-1)
    parser.add_argument("--mowing_terrain_collision", type=int, choices=(0, 1), default=-1)
    parser.add_argument(
        "--trajectory_mode",
        choices=("ground_straight_line", "ground_semicircle", "ground_mower_sweep"),
        default="ground_straight_line",
    )
    parser.add_argument("--trajectory_plane", choices=("ground_xy", "base_yz"), default="ground_xy")
    parser.add_argument(
        "--tool_contact_mode",
        choices=("geometry_aabb", "manual_offset", "ee_origin"),
        default="geometry_aabb",
    )
    parser.add_argument("--workspace_json", type=str, default="outputs/ppo1_horizon_idfd/reachable_workspace.json")
    parser.add_argument("--accepted_points_json", type=str, default="")
    parser.add_argument("--max_saved_points", type=int, default=2000)
    parser.add_argument("--progress_interval", type=int, default=512)
    parser.add_argument("--observe_steps", type=int, default=1000)
    parser.add_argument("--observe_policy", choices=("greedy", "zero", "random"), default="greedy")
    parser.add_argument("--greedy_gain", type=float, default=1.0)
    parser.add_argument("--ee_action_scale_m", type=float, default=-1.0)
    parser.add_argument("--ee_kp", type=float, default=-1.0)
    parser.add_argument("--ee_lam", type=float, default=-1.0)
    parser.add_argument("--ee_per_cyl_clip", type=float, default=-1.0)
    parser.add_argument("--ee_global_clip", type=float, default=-1.0)
    parser.add_argument("--ee_smooth_alpha", type=float, default=-1.0)
    parser.add_argument("--horizon_cmd_clip", type=float, default=-1.0)
    parser.add_argument("--pid_kp_scale", type=float, default=-1.0)
    parser.add_argument("--pid_kd_scale", type=float, default=-1.0)
    parser.add_argument("--pid_effort_scale", type=float, default=-1.0)
    parser.add_argument("--bias_runtime_ki_scale", type=float, default=-1.0)
    parser.add_argument("--force_rate_limit", type=float, default=-1.0)
    parser.add_argument(
        "--actuator_backend",
        choices=("implicit_position", "explicit_effort_pd", "explicit_effort_pid", "explicit_effort_pid_bias"),
        default="implicit_position",
    )
    # Re-enable per-step root teleport pinning (measured to pump energy into the
    # boom through tire contacts; keep off unless specifically studying it).
    parser.add_argument("--pin_root", action="store_true")
    # Override the PhysX-finetuned ID checkpoint (empty = wrapper default).
    parser.add_argument("--id_checkpoint", type=str, default="")
    # Runtime PhysX drive gain overrides for the 4 cylinder joints (implicit backend).
    # Written to sim after env creation; also sweepable via trial keys
    # "drive_stiffness_per_cylinder" / "drive_damping_per_cylinder".
    parser.add_argument("--drive_stiffness_per_cyl", type=float, nargs=4, default=None, metavar="N_PER_M")
    parser.add_argument("--drive_damping_per_cyl", type=float, nargs=4, default=None, metavar="NS_PER_M")
    parser.add_argument("--pid_kp_per_cyl", type=float, nargs=4, default=None, metavar="SCALE")
    parser.add_argument("--pid_kd_per_cyl", type=float, nargs=4, default=None, metavar="SCALE")
    parser.add_argument("--force_rate_limit_per_cyl", type=float, nargs=4, default=None, metavar="SCALE")
    # -- Dynamic hold sampling (mode=sample_dynamic) and sweep trials --
    parser.add_argument("--hold_seconds", type=float, default=3.0)
    parser.add_argument("--calibration_timeout_s", type=float, default=8.0)
    parser.add_argument("--pre_hold_settle_s", type=float, default=1.0)
    parser.add_argument("--post_calibration_settle_steps", type=int, default=30)
    parser.add_argument("--hold_cyl_err_max", type=float, default=0.01)
    parser.add_argument("--hold_tool_drift_max", type=float, default=0.05)
    parser.add_argument("--hold_min_tool_clearance", type=float, default=0.0)
    # Arm clearance uses link origins (not geometry AABBs); some chassis link origins sit
    # below the ground plane at rest, so the default -1.0 keeps this diagnostic-only.
    parser.add_argument("--hold_min_arm_clearance", type=float, default=-1.0)
    parser.add_argument(
        "--dynamic_workspace_json",
        type=str,
        default="outputs/ppo1_horizon_idfd/dynamic_hold_workspace.json",
    )
    parser.add_argument("--dynamic_records_json", type=str, default="")
    # Optional reference cloud (schema of --accepted_points_json: {"base": [[x,y,z],...]})
    # to report what fraction of it the dynamic points cover in base-frame YZ.
    parser.add_argument("--coverage_reference_points_json", type=str, default="")
    parser.add_argument("--coverage_tolerance_m", type=float, default=0.05)
    # Merge accepted points from a previous dynamic workspace JSON (its
    # accepted_points_base cloud) into this run before building the workspace.
    parser.add_argument("--merge_dynamic_json", type=str, default="")
    # -- PID/bias parameter sweep (mode=sweep) --
    parser.add_argument("--sweep_json", type=str, default="")
    parser.add_argument("--sweep_output_json", type=str, default="outputs/ppo1_horizon_idfd/pid_hold_sweep.json")
    # -- Observe convergence summary / saved plots --
    parser.add_argument("--summary_json", type=str, default="")
    parser.add_argument("--summary_warmup_steps", type=int, default=20)
    parser.add_argument("--save_plots_dir", type=str, default="")
    parser.add_argument("--pass_mean_tracking_err", type=float, default=0.05)
    parser.add_argument("--pass_max_tracking_err", type=float, default=0.15)
    # For zero-policy hold runs the plant-stability criterion is drift of the tool from
    # its own post-warmup position; tracking error vs the trajectory contains the static
    # offset between reset pose and trajectory start, which zero action cannot remove.
    parser.add_argument("--pass_max_tool_drift", type=float, default=0.05)
    parser.add_argument("--pass_max_cyl_err", type=float, default=0.01)
    parser.add_argument("--pass_min_tool_clearance", type=float, default=0.0)
    parser.add_argument("--pass_min_arm_clearance", type=float, default=-1.0)
    parser.add_argument("--enforce_pass", action="store_true")
    # -- Axis sign/authority probe (mode=axis_probe / authority_sweep) --
    parser.add_argument("--probe_steps", type=int, default=24, help="constant-action deploy steps per axis")
    parser.add_argument("--probe_min_motion_m", type=float, default=0.01, help="min |neutral-subtracted command response| to count as authority")
    parser.add_argument("--probe_max_neutral_drift_m", type=float, default=0.10, help="max zero-command drift before the neutral reference (and gate) fails")
    parser.add_argument("--probe_resettle_steps", type=int, default=40, help="hold steps after restoring the settled pose between axes")
    parser.add_argument("--authority_greedy_steps", type=int, default=48, help="steps for greedy-to-current-target in authority_sweep")
    parser.add_argument("--live_plots", action="store_true")
    parser.add_argument("--reset_debug", action="store_true")
    parser.add_argument("--hold_target", action="store_true")
    parser.add_argument("--plot_window", type=int, default=250)
    parser.add_argument("--plot_interval", type=int, default=2)
    parser.add_argument("--random_seed", type=int, default=7)
    parser.add_argument("--fast_exit", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    return parser


args_cli = _build_parser().parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
from isaaclab.utils.math import quat_apply

from kevin_integration.new_observations_pipeline.ppo1.horizon_idfd_env_wrapper import HorizonIDFDEnv
from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env_cfg import MulagUpperLevelEnvCfg


def _jsonable(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _print_json(label: str, payload: dict) -> None:
    print(f"{label} {json.dumps(_jsonable(payload), sort_keys=True)}", flush=True)


def _make_cfg(trajectory_mode: str = "none") -> MulagUpperLevelEnvCfg:
    cfg = MulagUpperLevelEnvCfg()
    cfg.scene.num_envs = int(args_cli.num_envs)
    cfg.sim.device = args_cli.device
    cfg.scene.replicate_physics = False
    if float(getattr(args_cli, "robot_spawn_z", -1.0)) >= 0.0:
        _p = cfg.robot_cfg.init_state.pos
        cfg.robot_cfg.init_state.pos = (_p[0], _p[1], float(args_cli.robot_spawn_z))
    cfg.ppo_stage = "full_mowing"
    cfg.ee_action_mode = "task_space_delta"
    cfg.ee_action_dim = 2 if trajectory_mode.startswith("ground_") and args_cli.trajectory_plane in ("base_yz", "ground_xy") else 3
    if trajectory_mode.startswith("ground_") and args_cli.trajectory_plane == "base_yz":
        # Sized so a max command gives dx_step ~= 0.3 mm/model-step, ~1 sigma of the
        # finetuned ID's training inputs (X_std ~3e-4). The previous 0.15 mm/horizon
        # values date from the unstable-plant era and starved ID/FD at 0.02-0.05
        # sigma, leaving only sign information after zero-baseline subtraction.
        cfg.ee_k_p = 0.15
        cfg.ee_per_cyl_clip_m = 0.003
        cfg.ee_global_clip_m = 0.006
        cfg.ee_smooth_alpha = 0.7
        cfg.horizon_delta_cmd_clip_m = 0.004
        cfg.bias_calibration_requires_static = False
        cfg.bias_calibration_failed_behavior = "use_anyway"
    if float(args_cli.ee_action_scale_m) >= 0.0:
        cfg.ee_action_scale_m = float(args_cli.ee_action_scale_m)
    if float(args_cli.ee_kp) >= 0.0:
        cfg.ee_k_p = float(args_cli.ee_kp)
    if float(args_cli.ee_lam) >= 0.0:
        cfg.ee_lam = float(args_cli.ee_lam)
    if float(args_cli.ee_per_cyl_clip) >= 0.0:
        cfg.ee_per_cyl_clip_m = float(args_cli.ee_per_cyl_clip)
    if float(args_cli.ee_global_clip) >= 0.0:
        cfg.ee_global_clip_m = float(args_cli.ee_global_clip)
    if float(args_cli.ee_smooth_alpha) >= 0.0:
        cfg.ee_smooth_alpha = float(args_cli.ee_smooth_alpha)
    if float(args_cli.horizon_cmd_clip) >= 0.0:
        cfg.horizon_delta_cmd_clip_m = float(args_cli.horizon_cmd_clip)
    if float(args_cli.pid_kp_scale) >= 0.0:
        cfg.explicit_pid_kp_scale = float(args_cli.pid_kp_scale)
    if float(args_cli.pid_kd_scale) >= 0.0:
        cfg.explicit_pid_kd_scale = float(args_cli.pid_kd_scale)
    if float(args_cli.pid_effort_scale) >= 0.0:
        cfg.explicit_pid_effort_scale = float(args_cli.pid_effort_scale)
    if float(args_cli.bias_runtime_ki_scale) >= 0.0:
        cfg.bias_runtime_ki_scale = float(args_cli.bias_runtime_ki_scale)
    if float(args_cli.force_rate_limit) >= 0.0:
        cfg.explicit_force_rate_limit_fraction = float(args_cli.force_rate_limit)
    if args_cli.pid_kp_per_cyl is not None:
        cfg.explicit_pid_kp_per_cylinder = [float(v) for v in args_cli.pid_kp_per_cyl]
    if args_cli.pid_kd_per_cyl is not None:
        cfg.explicit_pid_kd_per_cylinder = [float(v) for v in args_cli.pid_kd_per_cyl]
    if args_cli.force_rate_limit_per_cyl is not None:
        cfg.explicit_force_rate_limit_per_cylinder = [float(v) for v in args_cli.force_rate_limit_per_cyl]
    cfg.low_level_servo_mode = "none"
    cfg.lowlevel_actuator_backend = str(args_cli.actuator_backend)
    cfg.enable_scripted_vehicle_motion = False
    cfg.horizon_pin_root_pose = bool(args_cli.pin_root)
    if args_cli.id_checkpoint:
        cfg.id_checkpoint_override_path = str(args_cli.id_checkpoint)
    sampling_mode = args_cli.mode in ("sample", "sample_dynamic", "sweep")
    cfg.horizon_reset_settle_steps = 0 if sampling_mode else 1
    if (
        trajectory_mode.startswith("ground_")
        and args_cli.trajectory_plane == "base_yz"
        and args_cli.mode in ("observe", "axis_probe", "authority_sweep")
    ):
        cfg.horizon_reset_settle_steps = max(int(cfg.horizon_reset_settle_steps), 320)
    if args_cli.mode == "observe":
        # Observe is a validation/plotting mode; success resets make a held target
        # appear to jump because reset selects a new local waypoint. Training keeps
        # normal success terminations.
        cfg.horizon_success_required_hold_steps = int(args_cli.observe_steps) + 100
        cfg.horizon_min_success_steps = int(args_cli.observe_steps) + 100
    cfg.skip_j_eff_measurement = sampling_mode
    cfg.trajectory_mode = trajectory_mode
    cfg.trajectory_plane = args_cli.trajectory_plane
    cfg.trajectory_scale = float(args_cli.trajectory_scale)
    cfg.ground_trajectory_points = int(args_cli.trajectory_points)
    cfg.ground_path_pacing = str(args_cli.ground_path_pacing)
    cfg.ground_path_capture_radius_m = float(args_cli.ground_path_capture_radius_m)
    cfg.ground_path_capture_required_hold_steps = int(args_cli.ground_path_capture_required_hold_steps)
    cfg.ground_initial_target_mode = str(args_cli.ground_initial_target_mode)
    cfg.ground_initial_target_offset_m = float(args_cli.ground_initial_target_offset_m)
    cfg.ground_initial_target_capture_radius_m = float(args_cli.ground_initial_target_capture_radius_m)
    cfg.ground_anchor_path_to_current_tool = bool(args_cli.ground_anchor_path_to_current_tool)
    cfg.ground_anchor_margin_m = float(args_cli.ground_anchor_margin_m)
    cfg.ground_anchor_max_span_m = float(args_cli.ground_anchor_max_span_m)
    cfg.ground_anchor_static_sag_m = float(args_cli.ground_anchor_static_sag_m)
    if int(args_cli.ground_use_root_hold_frame) >= 0:
        cfg.ground_use_root_hold_frame = bool(int(args_cli.ground_use_root_hold_frame))
    if int(args_cli.ground_xy_condition_path_to_j_eff) >= 0:
        cfg.ground_xy_condition_path_to_j_eff = bool(int(args_cli.ground_xy_condition_path_to_j_eff))
    if float(args_cli.ground_xy_condition_path_perp_scale) >= 0.0:
        cfg.ground_xy_condition_path_perp_scale = float(args_cli.ground_xy_condition_path_perp_scale)
    if float(args_cli.ground_xy_condition_path_min_sv) >= 0.0:
        cfg.ground_xy_condition_path_min_sv = float(args_cli.ground_xy_condition_path_min_sv)
    if float(args_cli.ground_xy_z_hold_gain) >= 0.0:
        cfg.ground_xy_z_hold_gain = float(args_cli.ground_xy_z_hold_gain)
    if float(args_cli.ground_xy_conditioned_xy_clip_m) >= 0.0:
        cfg.ground_xy_conditioned_xy_clip_m = float(args_cli.ground_xy_conditioned_xy_clip_m)
    if float(args_cli.ground_xy_z_hold_clip_m) >= 0.0:
        cfg.ground_xy_z_hold_clip_m = float(args_cli.ground_xy_z_hold_clip_m)
    if float(args_cli.ground_xy_z_hold_deadband_m) >= 0.0:
        cfg.ground_xy_z_hold_deadband_m = float(args_cli.ground_xy_z_hold_deadband_m)
    if float(args_cli.ground_xy_z_hold_priority_threshold_m) >= 0.0:
        cfg.ground_xy_z_hold_priority_threshold_m = float(args_cli.ground_xy_z_hold_priority_threshold_m)
    cfg.horizon_reference_space_target = bool(args_cli.reference_space_target)
    if int(args_cli.mowing_terrain) >= 0:
        cfg.mowing_terrain_enable = bool(int(args_cli.mowing_terrain))
    if int(args_cli.mowing_terrain_collision) >= 0:
        cfg.mowing_terrain_collision = bool(int(args_cli.mowing_terrain_collision))
    if getattr(cfg, "mowing_terrain_enable", False):
        # Place the shelf just under the workspace's flat sweep band.
        try:
            import json as _json
            _ws = _json.load(open(args_cli.workspace_json))
            if "mowing_terrain_z" in _ws:
                cfg.mowing_terrain_z = float(_ws["mowing_terrain_z"])
            _slice = _ws.get("base_yz_slice", {})
            if "fixed_world_x" in _slice:
                cfg.mowing_terrain_local_x = float(_slice["fixed_world_x"])
            if "y_min" in _slice and "y_max" in _slice:
                cfg.mowing_terrain_local_y = 0.5 * (float(_slice["y_min"]) + float(_slice["y_max"]))
                cfg.mowing_terrain_half_y = max(0.5 * (float(_slice["y_max"]) - float(_slice["y_min"])) + 0.15, 0.3)
        except Exception:
            pass
    cfg.ground_workspace_json = args_cli.workspace_json
    cfg.tool_contact_mode = args_cli.tool_contact_mode
    cfg.tool_ground_clearance = float(args_cli.tool_ground_clearance)
    cfg.ground_z = float(args_cli.ground_z)
    cfg.min_arm_ground_clearance = float(args_cli.min_arm_clearance)
    cfg.visualize_target_trajectory = trajectory_mode.startswith("ground_")
    cfg.visualize_actual_ee_trail = trajectory_mode.startswith("ground_")
    cfg.ground_hold_target = bool(args_cli.hold_target)
    cfg.debug_draw_tool_contact_point = True
    cfg.debug_draw_tool_aabb = True
    cfg.enable_reset_debug = bool(args_cli.reset_debug)
    return cfg


def _write_cylinder_drive_gains(
    env: HorizonIDFDEnv,
    stiffness: list[float] | torch.Tensor | None = None,
    damping: list[float] | torch.Tensor | None = None,
) -> None:
    """Write PhysX implicit drive gains for the 4 cylinder joints at runtime."""
    uw = env._uw
    joint_ids = uw._cylinder_joint_ids
    if stiffness is not None:
        s = torch.as_tensor(stiffness, dtype=torch.float32, device=env.device)
        if s.ndim == 1:
            s = s.unsqueeze(0).expand(env.num_envs, -1)
        uw.robot.write_joint_stiffness_to_sim(s.clone(), joint_ids=joint_ids)
    if damping is not None:
        d = torch.as_tensor(damping, dtype=torch.float32, device=env.device)
        if d.ndim == 1:
            d = d.unsqueeze(0).expand(env.num_envs, -1)
        uw.robot.write_joint_damping_to_sim(d.clone(), joint_ids=joint_ids)


def _read_cylinder_drive_gains(env: HorizonIDFDEnv) -> tuple[torch.Tensor, torch.Tensor]:
    uw = env._uw
    joint_ids = uw._cylinder_joint_ids
    return (
        uw.robot.data.joint_stiffness[:, joint_ids].detach().clone(),
        uw.robot.data.joint_damping[:, joint_ids].detach().clone(),
    )


def _make_env(trajectory_mode: str = "none") -> HorizonIDFDEnv:
    torch.manual_seed(int(args_cli.random_seed))
    np.random.seed(int(args_cli.random_seed))
    env = HorizonIDFDEnv(
        _make_cfg(trajectory_mode),
        horizon_steps=int(args_cli.horizon_steps),
        device=args_cli.device,
    )
    env.reset()
    if args_cli.drive_stiffness_per_cyl is not None or args_cli.drive_damping_per_cyl is not None:
        _write_cylinder_drive_gains(
            env,
            stiffness=args_cli.drive_stiffness_per_cyl,
            damping=args_cli.drive_damping_per_cyl,
        )
        stiff_now, damp_now = _read_cylinder_drive_gains(env)
        _print_json(
            "[drive_gains]",
            {"stiffness": stiff_now[0].cpu().tolist(), "damping": damp_now[0].cpu().tolist()},
        )
    return env


def _write_sampled_joint_state(env: HorizonIDFDEnv, cylinder_q: torch.Tensor) -> None:
    uw = env._uw
    num_envs = env.num_envs
    env_ids = torch.arange(num_envs, device=env.device)
    joint_pos = uw.robot.data.default_joint_pos[:num_envs].detach().clone()
    joint_vel = torch.zeros_like(joint_pos)
    joint_pos[:, uw._cylinder_joint_ids] = cylinder_q
    tool_q = joint_pos[:, uw._tool_joint_id].detach().clone().unsqueeze(-1)
    env._root_hold_pose = uw.robot.data.default_root_state[:num_envs, :7].detach().clone()
    env._root_hold_pose[:, :3] += uw.scene.env_origins[:num_envs]
    env._root_hold_velocity = torch.zeros((num_envs, 6), device=env.device)
    uw.robot.write_root_pose_to_sim(env._root_hold_pose, env_ids)
    uw.robot.write_root_velocity_to_sim(env._root_hold_velocity, env_ids)
    uw.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
    uw.set_debug_manual_cylinder_targets(cylinder_q.detach().clone(), tool_position_target=tool_q)
    env._cylinder_hold_target = cylinder_q.detach().clone()
    env._tool_hold_target = tool_q.detach().clone()
    for _ in range(max(int(args_cli.settle_steps), 0)):
        env._pin_root_pose_if_enabled()
        env._env.step(torch.zeros((num_envs, 5), device=env.device))
        env._pin_root_pose_if_enabled()


def _sample_cylinder_positions(env: HorizonIDFDEnv) -> torch.Tensor:
    lo, hi = env._joint_limits()
    margin = torch.clamp((hi - lo) * float(args_cli.joint_margin_frac), min=0.0)
    lo = lo + margin
    hi = hi - margin
    u = torch.rand((env.num_envs, 4), device=env.device)
    return lo + u * (hi - lo).clamp(min=1.0e-6)


def _reset_explicit_bias_state(env: HorizonIDFDEnv) -> None:
    """Restart explicit_effort_pid_bias calibration from scratch for all envs."""
    uw = env._uw
    uw._explicit_pid_integral_error[:] = 0.0
    uw._explicit_force_bias[:] = 0.0
    uw._explicit_bias_calibrated[:] = False
    uw._explicit_bias_calib_step[:] = 0
    uw._explicit_force_bias_accum[:] = 0.0
    uw._explicit_bias_window_count[:] = 0
    uw._qd_filt[:] = 0.0
    uw._prev_force_cmd[:] = 0.0


def _arm_clearance_cheap(env: HorizonIDFDEnv) -> torch.Tensor:
    """Min arm body-link z above ground, from link origins (fast; not full geometry AABB)."""
    body_pos = env._uw.robot.data.body_link_state_w[:, :, :3]
    env_origins = env._uw.scene.env_origins[: env.num_envs].reshape(env.num_envs, 1, 3)
    body_pos_local = body_pos - env_origins
    body_names = list(getattr(env._uw.robot, "body_names", []))
    excluded_tokens = ("tool", "wheel", "tire", "tyre")
    arm_indices = [
        i for i, name in enumerate(body_names)
        if not any(token in str(name).lower() for token in excluded_tokens)
    ]
    arm_z = body_pos_local[:, arm_indices, 2] if arm_indices else body_pos_local[:, :, 2]
    return arm_z.amin(dim=1) - float(args_cli.ground_z)


def _run_hold_trial(env: HorizonIDFDEnv, cylinder_q: torch.Tensor) -> dict[str, torch.Tensor]:
    """Write a candidate cylinder pose and test whether explicit_effort_pid_bias can hold it.

    Sequence per candidate batch: write pose -> pin as manual cylinder target ->
    restart bias calibration -> wait for calibration -> retarget to the settled
    (calibrated) pose -> hold for --hold_seconds while recording stability metrics.
    Manual targets are re-asserted every step because a hidden base reset clears
    them globally; such resets are detected via episode_length_buf regression and
    the affected candidates are rejected.
    """
    uw = env._uw
    num_envs = env.num_envs
    device = env.device
    dt = float(uw.step_dt)
    zero_action = torch.zeros((num_envs, 5), device=device)

    _write_sampled_joint_state(env, cylinder_q)
    _reset_explicit_bias_state(env)

    reset_seen = torch.zeros(num_envs, dtype=torch.bool, device=device)
    nonfinite_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
    huge_transform_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
    limit_hit_seen = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def _step_once() -> None:
        nonlocal reset_seen, nonfinite_steps, huge_transform_steps, limit_hit_seen
        uw.set_debug_manual_cylinder_targets(
            env._cylinder_hold_target,
            tool_position_target=env._tool_hold_target,
        )
        env._pin_root_pose_if_enabled()
        ep_before = uw.episode_length_buf.detach().clone()
        env._env.step(zero_action)
        env._pin_root_pose_if_enabled()
        reset_seen |= uw.episode_length_buf < ep_before + 1
        body_pos = uw.robot.data.body_link_state_w[:, :, :3]
        env_origins = uw.scene.env_origins[:num_envs].reshape(num_envs, 1, 3)
        body_local = body_pos - env_origins
        finite = torch.isfinite(body_pos).all(dim=(1, 2))
        nonfinite_steps += (~finite).long()
        huge_transform_steps += (
            finite & (body_local.abs().amax(dim=(1, 2)) > 100.0)
        ).long()
        lo, hi = env._joint_limits()
        q_now = uw.robot.data.joint_pos[:, uw._cylinder_joint_ids]
        limit_hit_seen |= ((q_now <= lo + 1.0e-5) | (q_now >= hi - 1.0e-5)).any(dim=-1)

    # -- Phase 1: settle (bias calibration for the bias backend, plain settle otherwise) --
    use_bias = (
        str(getattr(env.cfg, "lowlevel_actuator_backend", "implicit_position"))
        == "explicit_effort_pid_bias"
    )
    if use_bias:
        calib_timeout_steps = max(int(math.ceil(float(args_cli.calibration_timeout_s) / dt)), 1)
        for _ in range(calib_timeout_steps):
            _step_once()
            if bool(uw._explicit_bias_calibrated.all()):
                break
        calibrated = uw._explicit_bias_calibrated.detach().clone()
    else:
        settle_steps = max(int(math.ceil(float(args_cli.pre_hold_settle_s) / dt)), 1)
        for _ in range(settle_steps):
            _step_once()
        calibrated = torch.ones(num_envs, dtype=torch.bool, device=device)

    # -- Phase 2: retarget hold target to the calibrated equilibrium pose --
    # NaN-poisoned envs keep their previous finite target (they are rejected later
    # via nonfinite_steps) so set_debug_manual_cylinder_targets never sees NaN.
    if bool(getattr(env.cfg, "bias_runtime_retarget_to_calibrated_pose", True)):
        cq = uw.robot.data.joint_pos[:, uw._cylinder_joint_ids].detach().clone()
        finite_rows = torch.isfinite(cq).all(dim=-1, keepdim=True)
        lo, hi = env._joint_limits()
        retarget = torch.clamp(torch.nan_to_num(cq, nan=0.0, posinf=0.0, neginf=0.0), lo, hi)
        env._cylinder_hold_target = torch.where(finite_rows, retarget, env._cylinder_hold_target)
        tool_q = uw.robot.data.joint_pos[:, uw._tool_joint_id].detach().clone().unsqueeze(-1)
        env._tool_hold_target = torch.where(
            torch.isfinite(tool_q),
            torch.nan_to_num(tool_q, nan=0.0, posinf=0.0, neginf=0.0),
            env._tool_hold_target,
        )
    for _ in range(max(int(args_cli.post_calibration_settle_steps), 0)):
        _step_once()
    tool_ref_w = torch.nan_to_num(
        env.get_tool_bottom_world_pos().detach().clone(), nan=1.0e6, posinf=1.0e6, neginf=-1.0e6
    )

    # -- Phase 3: hold and record stability metrics --
    # Limit hits during the settle transient don't matter — the candidate is the
    # settled/retargeted pose. Only limit contact during the hold rejects.
    limit_hit_seen[:] = False
    hold_steps = max(int(math.ceil(float(args_cli.hold_seconds) / dt)), 1)
    cyl_err_max = torch.zeros(num_envs, device=device)
    tool_drift_max = torch.zeros(num_envs, device=device)
    tool_clear_min = torch.full((num_envs,), float("inf"), device=device)
    arm_clear_min = torch.full((num_envs,), float("inf"), device=device)
    for _ in range(hold_steps):
        _step_once()
        q_now = uw.robot.data.joint_pos[:, uw._cylinder_joint_ids]
        cyl_err = torch.nan_to_num(
            (env._cylinder_hold_target - q_now).abs().amax(dim=-1), nan=1.0e6, posinf=1.0e6
        )
        cyl_err_max = torch.maximum(cyl_err_max, cyl_err)
        tool_w = torch.nan_to_num(
            env.get_tool_bottom_world_pos(), nan=1.0e6, posinf=1.0e6, neginf=-1.0e6
        )
        drift = torch.linalg.norm(tool_w - tool_ref_w, dim=-1)
        tool_drift_max = torch.maximum(tool_drift_max, drift)
        tool_clear_min = torch.minimum(tool_clear_min, tool_w[:, 2] - float(args_cli.ground_z))
        arm_clear_min = torch.minimum(
            arm_clear_min,
            torch.nan_to_num(_arm_clearance_cheap(env), nan=-1.0e6, neginf=-1.0e6),
        )

    tool_ref_b = env._world_to_base(tool_ref_w).detach()
    kinematic = _acceptance_components(env, tool_ref_w, tool_ref_b)
    hold_ok = (
        calibrated
        & ~reset_seen
        & ~limit_hit_seen
        & (nonfinite_steps == 0)
        & (huge_transform_steps == 0)
        & (cyl_err_max <= float(args_cli.hold_cyl_err_max))
        & (tool_drift_max <= float(args_cli.hold_tool_drift_max))
        & (tool_clear_min >= float(args_cli.hold_min_tool_clearance))
        & (arm_clear_min >= float(args_cli.hold_min_arm_clearance))
    )
    return {
        "accepted": hold_ok & kinematic["accepted"],
        "hold_ok": hold_ok,
        "kinematic_ok": kinematic["accepted"],
        "calibrated": calibrated,
        "reset_seen": reset_seen,
        "limit_hit_seen": limit_hit_seen,
        "nonfinite_steps": nonfinite_steps,
        "huge_transform_steps": huge_transform_steps,
        "cyl_err_max": cyl_err_max,
        "tool_drift_max": tool_drift_max,
        "tool_clear_min": tool_clear_min,
        "arm_clear_min": arm_clear_min,
        "tool_ref_w": tool_ref_w,
        "tool_ref_b": tool_ref_b,
        "cylinder_q_settled": uw.robot.data.joint_pos[:, uw._cylinder_joint_ids].detach().clone(),
    }


_HOLD_REJECT_REASONS = (
    ("nonfinite", lambda t: t["nonfinite_steps"] > 0),
    ("huge_transform", lambda t: t["huge_transform_steps"] > 0),
    ("reset_during_trial", lambda t: t["reset_seen"]),
    ("calibration_timeout", lambda t: ~t["calibrated"]),
    ("cylinder_limit_hit", lambda t: t["limit_hit_seen"]),
    ("cylinder_error", lambda t: t["cyl_err_max"] > float(args_cli.hold_cyl_err_max)),
    ("tool_drift", lambda t: t["tool_drift_max"] > float(args_cli.hold_tool_drift_max)),
    ("tool_clearance", lambda t: t["tool_clear_min"] < float(args_cli.hold_min_tool_clearance)),
    ("arm_clearance", lambda t: t["arm_clear_min"] < float(args_cli.hold_min_arm_clearance)),
    ("outside_workspace", lambda t: ~t["kinematic_ok"]),
)


def _tally_hold_rejections(trial: dict[str, torch.Tensor], counts: dict[str, int]) -> None:
    remaining = ~trial["accepted"]
    for name, mask_fn in _HOLD_REJECT_REASONS:
        mask = mask_fn(trial) & remaining
        counts[name] = counts.get(name, 0) + int(mask.sum().detach().cpu().item())
        remaining = remaining & ~mask


def _acceptance_components(env: HorizonIDFDEnv, tool_w: torch.Tensor, tool_b: torch.Tensor) -> dict[str, torch.Tensor]:
    wx = getattr(env.cfg, "horizon_workspace_x_range", (-0.5, 7.0))
    wy = getattr(env.cfg, "horizon_workspace_y_range", (-3.6, 3.0))
    wz = getattr(env.cfg, "horizon_workspace_z_range", (-0.4, 3.0))
    body_pos = env._uw.robot.data.body_link_state_w[:, :, :3]
    root_pos = env._uw.robot.data.root_pos_w
    env_origins = env._uw.scene.env_origins[: env.num_envs].reshape(env.num_envs, 1, 3)
    body_pos_local = body_pos - env_origins
    root_pos_local = root_pos - env_origins.squeeze(1)
    max_abs_body = body_pos_local.abs().amax(dim=(1, 2))
    body_finite = torch.isfinite(body_pos).all(dim=(1, 2))
    root_finite = torch.isfinite(root_pos_local).all(dim=1)
    finite = (
        torch.isfinite(tool_w).all(dim=-1)
        & torch.isfinite(tool_b).all(dim=-1)
        & body_finite
        & root_finite
    )
    body_names = list(getattr(env._uw.robot, "body_names", []))
    excluded_tokens = ("tool", "wheel", "tire", "tyre")
    arm_indices = [
        i for i, name in enumerate(body_names)
        if not any(token in str(name).lower() for token in excluded_tokens)
    ]
    if arm_indices:
        arm_z = body_pos_local[:, arm_indices, 2]
    else:
        arm_z = body_pos_local[:, :, 2]
    arm_clearance = arm_z.amin(dim=1) - float(args_cli.ground_z)
    inside_workspace = (
        (tool_b[:, 0] >= float(wx[0]))
        & (tool_b[:, 0] <= float(wx[1]))
        & (tool_b[:, 1] >= float(wy[0]))
        & (tool_b[:, 1] <= float(wy[1]))
        & (tool_b[:, 2] >= float(wz[0]))
        & (tool_b[:, 2] <= float(wz[1]))
    )
    # Keep Monte Carlo sampling fast and conservative about physics validity.
    # Full geometry/AABB arm clearance is intentionally not called here because
    # it is too expensive for the hot loop; observe/debug runs still report it.
    safe = max_abs_body <= 100.0
    return {
        "finite": finite,
        "inside_workspace": inside_workspace,
        "safe": safe,
        "accepted": finite & inside_workspace & safe,
        "arm_clearance": arm_clearance,
        "max_abs_body": max_abs_body,
    }


def _sampler_result_summary(workspace: dict) -> dict:
    summary_keys = (
        "status",
        "workspace_type",
        "hold_criteria",
        "controller_params",
        "samples_requested",
        "samples_evaluated",
        "accepted_points_count",
        "acceptance_rate",
        "reject_counts",
        "target_world_z",
        "target_world_z_source",
        "requested_target_world_z",
        "safe_ground_points_count",
        "selected_reachable_points_count",
        "largest_line_endpoints",
        "largest_line_length",
        "semicircle_center",
        "semicircle_radius",
        "mower_sweep_bounds",
        "base_yz_slice",
        "full_reachable_bounds",
        "accepted_base_bounds",
    )
    summary = {key: workspace[key] for key in summary_keys if key in workspace}
    if "accepted_reachable_points" in workspace:
        summary["accepted_reachable_points_saved"] = len(workspace["accepted_reachable_points"])
    return summary


def _farthest_pair_xy(points: np.ndarray) -> tuple[list[float], list[float], float]:
    if len(points) == 0:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0
    if len(points) == 1:
        p = points[0].tolist()
        return p, p, 0.0
    pool = points
    if len(pool) > 1500:
        idx = np.linspace(0, len(pool) - 1, 1500).astype(int)
        pool = pool[idx]
    xy = pool[:, :2]
    best_i = 0
    best_j = 0
    best_d2 = -1.0
    for i in range(len(pool)):
        d2 = np.sum((xy[i + 1 :] - xy[i]) ** 2, axis=1)
        if d2.size == 0:
            continue
        j_rel = int(np.argmax(d2))
        if float(d2[j_rel]) > best_d2:
            best_d2 = float(d2[j_rel])
            best_i = i
            best_j = i + 1 + j_rel
    return pool[best_i].tolist(), pool[best_j].tolist(), math.sqrt(max(best_d2, 0.0))


def _farthest_pair_yz(points: np.ndarray) -> tuple[list[float], list[float], float]:
    if len(points) == 0:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0
    if len(points) == 1:
        p = points[0].tolist()
        return p, p, 0.0
    pool = points
    if len(pool) > 1500:
        idx = np.linspace(0, len(pool) - 1, 1500).astype(int)
        pool = pool[idx]
    yz = pool[:, 1:3]
    best_i = 0
    best_j = 0
    best_d2 = -1.0
    for i in range(len(pool)):
        d2 = np.sum((yz[i + 1 :] - yz[i]) ** 2, axis=1)
        if d2.size == 0:
            continue
        j_rel = int(np.argmax(d2))
        if float(d2[j_rel]) > best_d2:
            best_d2 = float(d2[j_rel])
            best_i = i
            best_j = i + 1 + j_rel
    return pool[best_i].tolist(), pool[best_j].tolist(), math.sqrt(max(best_d2, 0.0))


def _shrunk_quantile_bounds(values: np.ndarray, *, lo_q: float, hi_q: float, margin: float) -> tuple[float, float]:
    lo = float(np.quantile(values, lo_q)) + float(margin)
    hi = float(np.quantile(values, hi_q)) - float(margin)
    if hi <= lo:
        mid = float(np.median(values))
        half = max(0.01, 0.25 * float(np.ptp(values)))
        lo = mid - half
        hi = mid + half
    return lo, hi


def _workspace_from_points(all_points: np.ndarray, target_z_request: float) -> dict:
    if len(all_points) == 0:
        raise RuntimeError("No reachable samples accepted. Increase --samples or loosen sampling filters.")
    z = all_points[:, 2]
    near = np.abs(z - target_z_request) <= float(args_cli.z_tolerance)
    if int(np.count_nonzero(near)) >= 8:
        selected = all_points[near]
        target_z = float(target_z_request)
        z_source = "requested_ground_slice"
    else:
        low_cut = float(np.quantile(z, 0.20))
        band = np.abs(z - low_cut) <= float(args_cli.z_tolerance)
        if int(np.count_nonzero(band)) < 8:
            order = np.argsort(np.abs(z - low_cut))
            selected = all_points[order[: max(8, min(len(all_points), 128))]]
        else:
            selected = all_points[band]
        target_z = float(np.median(selected[:, 2]))
        z_source = "lowest_reachable_band"
    p0, p1, line_len = _farthest_pair_xy(selected)
    x_min = float(np.min(selected[:, 0]))
    x_max = float(np.max(selected[:, 0]))
    y_min = float(np.min(selected[:, 1]))
    y_max = float(np.max(selected[:, 1]))
    all_x_min = float(np.min(all_points[:, 0]))
    all_x_max = float(np.max(all_points[:, 0]))
    all_y_min = float(np.min(all_points[:, 1]))
    all_y_max = float(np.max(all_points[:, 1]))
    all_z_min = float(np.min(all_points[:, 2]))
    all_z_max = float(np.max(all_points[:, 2]))
    yz_margin = float(args_cli.trajectory_margin)
    yz_y_min, yz_y_max = _shrunk_quantile_bounds(all_points[:, 1], lo_q=0.10, hi_q=0.90, margin=yz_margin)
    yz_z_min, yz_z_max = _shrunk_quantile_bounds(all_points[:, 2], lo_q=0.10, hi_q=0.90, margin=yz_margin)
    yz_z_min = max(yz_z_min, float(args_cli.ground_z) + float(args_cli.tool_ground_clearance))
    if yz_z_max <= yz_z_min:
        yz_mid = float(np.median(all_points[:, 2]))
        yz_z_min = max(float(args_cli.ground_z) + float(args_cli.tool_ground_clearance), yz_mid - 0.05)
        yz_z_max = yz_mid + 0.05
    yz_x = float(np.median(all_points[:, 0]))
    yz_line = np.asarray(
        [
            [yz_x, yz_y_min, yz_z_min],
            [yz_x, yz_y_max, yz_z_max],
        ],
        dtype=np.float64,
    )
    yz_p0, yz_p1, yz_len = _farthest_pair_yz(
        all_points[
            (all_points[:, 1] >= yz_y_min)
            & (all_points[:, 1] <= yz_y_max)
            & (all_points[:, 2] >= yz_z_min)
            & (all_points[:, 2] <= yz_z_max)
        ]
    )
    if yz_len > 1.0e-6:
        yz_line = np.asarray([[yz_x, yz_p0[1], yz_p0[2]], [yz_x, yz_p1[1], yz_p1[2]]], dtype=np.float64)
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    radius = max(0.01, 0.45 * min(max(x_max - x_min, 0.0), max(y_max - y_min, 0.0)))
    if radius <= 0.011:
        radius = max(0.01, 0.25 * math.sqrt(max(line_len, 0.0)))
    return {
        "ground_z": float(args_cli.ground_z),
        "tool_ground_clearance": float(args_cli.tool_ground_clearance),
        "requested_target_world_z": float(target_z_request),
        "target_world_z": target_z,
        "target_world_z_source": z_source,
        "controlled_frame": "base_frame",
        "controlled_axes": "base_frame_XYZ",
        "tracking_point": "tool_contact_bottom",
        "trajectory_margin_m": yz_margin,
        "full_reachable_bounds": {
            "x_min": all_x_min,
            "x_max": all_x_max,
            "y_min": all_y_min,
            "y_max": all_y_max,
            "z_min": all_z_min,
            "z_max": all_z_max,
        },
        "safe_x_min": x_min,
        "safe_x_max": x_max,
        "safe_y_min": y_min,
        "safe_y_max": y_max,
        "safe_z_min": all_z_min,
        "safe_z_max": all_z_max,
        "base_yz_slice": {
            "fixed_world_x": yz_x,
            "y_min": yz_y_min,
            "y_max": yz_y_max,
            "z_min": yz_z_min,
            "z_max": yz_z_max,
            "line_endpoints": yz_line.tolist(),
            "line_length": float(np.linalg.norm(yz_line[1, 1:3] - yz_line[0, 1:3])),
            "center": [yz_x, 0.5 * (yz_y_min + yz_y_max), 0.5 * (yz_z_min + yz_z_max)],
            "semicircle_radius": float(0.45 * min(max(yz_y_max - yz_y_min, 0.0), max(yz_z_max - yz_z_min, 0.0))),
        },
        "safe_ground_points_count": int(np.count_nonzero(near)),
        "selected_reachable_points_count": int(len(selected)),
        "largest_line_endpoints": [
            [float(p0[0]), float(p0[1]), target_z],
            [float(p1[0]), float(p1[1]), target_z],
        ],
        "largest_line_length": float(line_len),
        "semicircle_center": [float(cx), float(cy), target_z],
        "semicircle_radius": float(radius),
        "mower_sweep_bounds": {
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
        },
        "mower_sweep_amplitude": float(0.35 * max(y_max - y_min, 0.0)),
        "mower_sweep_length": float(max(x_max - x_min, line_len)),
        "accepted_reachable_points": selected[: int(args_cli.max_saved_points)].tolist(),
    }


def _canonical_world_from_base(env: HorizonIDFDEnv, base_points: np.ndarray) -> np.ndarray:
    """Map base-frame points into the canonical (env-0 default root pose) world frame."""
    base_t = torch.as_tensor(base_points, dtype=torch.float32, device=env.device)
    root_pos0 = torch.tensor(
        env.cfg.robot_cfg.init_state.pos,
        dtype=torch.float32,
        device=env.device,
    ).reshape(1, 3)
    root_quat0 = torch.tensor(
        getattr(env.cfg.robot_cfg.init_state, "rot", (1.0, 0.0, 0.0, 0.0)),
        dtype=torch.float32,
        device=env.device,
    ).reshape(1, 4)
    root_quat0 = root_quat0 / root_quat0.norm(dim=-1, keepdim=True).clamp(min=1.0e-8)
    return (
        root_pos0.expand(base_t.shape[0], -1)
        + quat_apply(root_quat0.expand(base_t.shape[0], -1), base_t)
    ).detach().cpu().numpy()


def run_sample() -> int:
    env = _make_env("none")
    accepted_world: list[np.ndarray] = []
    accepted_base: list[np.ndarray] = []
    reject_counts = {
        "nonfinite": 0,
        "outside_workspace": 0,
        "unsafe": 0,
    }
    total = 0
    start_time = time.time()
    next_progress = max(int(args_cli.progress_interval), env.num_envs)
    try:
        _print_json("[sampler/initial_geometry]", env.tool_geometry_report())
        while total < int(args_cli.samples):
            q = _sample_cylinder_positions(env)
            _write_sampled_joint_state(env, q)
            tool_w = env.get_tool_bottom_world_pos().detach()
            tool_b = env._world_to_base(tool_w).detach()
            components = _acceptance_components(env, tool_w, tool_b)
            mask = components["accepted"]
            reject_counts["nonfinite"] += int((~components["finite"]).sum().detach().cpu().item())
            reject_counts["outside_workspace"] += int((components["finite"] & ~components["inside_workspace"]).sum().detach().cpu().item())
            reject_counts["unsafe"] += int((components["finite"] & components["inside_workspace"] & ~components["safe"]).sum().detach().cpu().item())
            if mask.any():
                accepted_world.append(tool_w[mask].cpu().numpy())
                accepted_base.append(tool_b[mask].cpu().numpy())
            total += env.num_envs
            if total >= next_progress or total >= int(args_cli.samples):
                count = sum(len(a) for a in accepted_world)
                elapsed = max(time.time() - start_time, 1.0e-6)
                rate = min(total, int(args_cli.samples)) / elapsed
                print(
                    "[sampler] "
                    f"samples={min(total, int(args_cli.samples))}/{int(args_cli.samples)} "
                    f"accepted={count} "
                    f"rate={rate:.1f}_samples_s",
                    flush=True,
                )
                while next_progress <= total:
                    next_progress += max(int(args_cli.progress_interval), env.num_envs)
        world = np.concatenate(accepted_world, axis=0) if accepted_world else np.zeros((0, 3), dtype=np.float32)
        base = np.concatenate(accepted_base, axis=0) if accepted_base else np.zeros((0, 3), dtype=np.float32)
        target_request = float(args_cli.ground_z) + float(args_cli.tool_ground_clearance)
        if len(world) == 0:
            out = Path(args_cli.workspace_json)
            out.parent.mkdir(parents=True, exist_ok=True)
            result = {
                "status": "no_accepted_points",
                "samples_requested": int(args_cli.samples),
                "samples_evaluated": int(total),
                "accepted_points_count": 0,
                "acceptance_rate": 0.0,
                "reject_counts": reject_counts,
                "hint": "Loosen workspace/clearance filters or inspect sampled base-frame points with --accepted_points_json.",
            }
            out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            _print_json("[sampler/result]", _sampler_result_summary(result))
            print(f"[sampler/output] {out}", flush=True)
            return 2
        canonical_world = _canonical_world_from_base(env, base)
        workspace = _workspace_from_points(canonical_world, target_request)
        workspace["samples_requested"] = int(args_cli.samples)
        workspace["samples_evaluated"] = int(total)
        workspace["accepted_points_count"] = int(len(world))
        workspace["acceptance_rate"] = float(len(world) / max(total, 1))
        workspace["reject_counts"] = reject_counts
        workspace["accepted_base_bounds"] = {
            "x_min": float(np.min(base[:, 0])) if len(base) else None,
            "x_max": float(np.max(base[:, 0])) if len(base) else None,
            "y_min": float(np.min(base[:, 1])) if len(base) else None,
            "y_max": float(np.max(base[:, 1])) if len(base) else None,
            "z_min": float(np.min(base[:, 2])) if len(base) else None,
            "z_max": float(np.max(base[:, 2])) if len(base) else None,
        }
        out = Path(args_cli.workspace_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(workspace, indent=2) + "\n", encoding="utf-8")
        if args_cli.accepted_points_json:
            points_out = Path(args_cli.accepted_points_json)
            points_out.parent.mkdir(parents=True, exist_ok=True)
            points_out.write_text(
                json.dumps({"world_env0": canonical_world.tolist(), "world_raw_batched": world.tolist(), "base": base.tolist()}, indent=2) + "\n",
                encoding="utf-8",
            )
        _print_json("[sampler/result]", _sampler_result_summary(workspace))
        print(f"[sampler/output] {out}", flush=True)
        return 0
    finally:
        env.close()


def run_sample_dynamic() -> int:
    """Sample cylinder poses and keep only those explicit_effort_pid_bias can dynamically hold."""
    env = _make_env("none")
    accepted_world: list[np.ndarray] = []
    accepted_base: list[np.ndarray] = []
    records: list[dict] = []
    reject_counts: dict[str, int] = {}
    total = 0
    accepted_total = 0
    start_time = time.time()
    hold_criteria = {
        "hold_seconds": float(args_cli.hold_seconds),
        "calibration_timeout_s": float(args_cli.calibration_timeout_s),
        "post_calibration_settle_steps": int(args_cli.post_calibration_settle_steps),
        "hold_cyl_err_max": float(args_cli.hold_cyl_err_max),
        "hold_tool_drift_max": float(args_cli.hold_tool_drift_max),
        "hold_min_tool_clearance": float(args_cli.hold_min_tool_clearance),
        "hold_min_arm_clearance": float(args_cli.hold_min_arm_clearance),
    }
    try:
        _print_json("[dynamic_sampler/initial_geometry]", env.tool_geometry_report())
        _print_json("[dynamic_sampler/hold_criteria]", hold_criteria)
        while total < int(args_cli.samples):
            q = _sample_cylinder_positions(env)
            trial = _run_hold_trial(env, q)
            mask = trial["accepted"]
            _tally_hold_rejections(trial, reject_counts)
            if mask.any():
                accepted_world.append(trial["tool_ref_w"][mask].cpu().numpy())
                accepted_base.append(trial["tool_ref_b"][mask].cpu().numpy())
            if len(records) < int(args_cli.max_saved_points):
                for env_i in range(env.num_envs):
                    if len(records) >= int(args_cli.max_saved_points):
                        break
                    records.append({
                        "cylinder_q_written": q[env_i].detach().cpu().tolist(),
                        "cylinder_q_settled": trial["cylinder_q_settled"][env_i].cpu().tolist(),
                        "tool_ref_base": trial["tool_ref_b"][env_i].cpu().tolist(),
                        "accepted": bool(mask[env_i].detach().cpu().item()),
                        "calibrated": bool(trial["calibrated"][env_i].detach().cpu().item()),
                        "reset_during_trial": bool(trial["reset_seen"][env_i].detach().cpu().item()),
                        "cylinder_limit_hit": bool(trial["limit_hit_seen"][env_i].detach().cpu().item()),
                        "cyl_err_max": float(trial["cyl_err_max"][env_i].detach().cpu().item()),
                        "tool_drift_max": float(trial["tool_drift_max"][env_i].detach().cpu().item()),
                        "tool_clear_min": float(trial["tool_clear_min"][env_i].detach().cpu().item()),
                        "arm_clear_min": float(trial["arm_clear_min"][env_i].detach().cpu().item()),
                        "nonfinite_steps": int(trial["nonfinite_steps"][env_i].detach().cpu().item()),
                        "huge_transform_steps": int(trial["huge_transform_steps"][env_i].detach().cpu().item()),
                    })
            total += env.num_envs
            accepted_total = sum(len(a) for a in accepted_world)
            elapsed = max(time.time() - start_time, 1.0e-6)
            print(
                "[dynamic_sampler] "
                f"candidates={min(total, int(args_cli.samples))}/{int(args_cli.samples)} "
                f"accepted={accepted_total} "
                f"rate={min(total, int(args_cli.samples)) / elapsed:.2f}_candidates_s "
                f"rejects={json.dumps(reject_counts, sort_keys=True)}",
                flush=True,
            )
        world = np.concatenate(accepted_world, axis=0) if accepted_world else np.zeros((0, 3), dtype=np.float32)
        base = np.concatenate(accepted_base, axis=0) if accepted_base else np.zeros((0, 3), dtype=np.float32)
        merged_from_previous = 0
        if args_cli.merge_dynamic_json:
            prev = json.loads(Path(args_cli.merge_dynamic_json).read_text(encoding="utf-8"))
            prev_base = np.asarray(prev.get("accepted_points_base", []), dtype=np.float32)
            if len(prev_base):
                merged_from_previous = int(len(prev_base))
                base = np.concatenate([base, prev_base], axis=0) if len(base) else prev_base
                print(
                    f"[dynamic_sampler/merge] merged {merged_from_previous} accepted points "
                    f"from {args_cli.merge_dynamic_json}",
                    flush=True,
                )
        out = Path(args_cli.dynamic_workspace_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        if args_cli.dynamic_records_json:
            records_out = Path(args_cli.dynamic_records_json)
            records_out.parent.mkdir(parents=True, exist_ok=True)
            records_out.write_text(
                json.dumps({"hold_criteria": hold_criteria, "records": records}, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"[dynamic_sampler/records] {records_out}", flush=True)
        common_meta = {
            "workspace_type": "dynamic_hold",
            "lowlevel_actuator_backend": str(getattr(env.cfg, "lowlevel_actuator_backend", "")),
            "hold_criteria": hold_criteria,
            "samples_requested": int(args_cli.samples),
            "samples_evaluated": int(total),
            "accepted_points_count": int(len(base)),
            "accepted_points_this_run": int(len(world)),
            "merged_from_previous": int(merged_from_previous),
            "acceptance_rate": float(len(world) / max(total, 1)),
            "reject_counts": reject_counts,
            "controller_params": {
                "explicit_pid_kp_scale": float(getattr(env.cfg, "explicit_pid_kp_scale", 1.0)),
                "explicit_pid_kd_scale": float(getattr(env.cfg, "explicit_pid_kd_scale", 1.0)),
                "explicit_pid_effort_scale": float(getattr(env.cfg, "explicit_pid_effort_scale", 1.0)),
                "bias_runtime_ki_scale": float(getattr(env.cfg, "bias_runtime_ki_scale", 0.0)),
                "explicit_force_rate_limit_fraction": float(getattr(env.cfg, "explicit_force_rate_limit_fraction", 0.01)),
                "explicit_pid_kp_per_cylinder": list(getattr(env.cfg, "explicit_pid_kp_per_cylinder", [1.0] * 4)),
                "explicit_pid_kd_per_cylinder": list(getattr(env.cfg, "explicit_pid_kd_per_cylinder", [1.0] * 4)),
                "explicit_force_rate_limit_per_cylinder": list(
                    getattr(env.cfg, "explicit_force_rate_limit_per_cylinder", [1.0] * 4)
                ),
            },
        }
        if len(base) == 0:
            result = {
                "status": "no_dynamically_holdable_points",
                **common_meta,
                "hint": (
                    "No candidate pose survived the dynamic hold test. Loosen hold thresholds, "
                    "retune explicit PID/bias gains (--pid_kp_scale/--pid_kd_scale/--mode sweep), "
                    "or increase --samples."
                ),
            }
            out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            _print_json("[dynamic_sampler/result]", _sampler_result_summary(result))
            print(f"[dynamic_sampler/output] {out}", flush=True)
            return 2
        target_request = float(args_cli.ground_z) + float(args_cli.tool_ground_clearance)
        canonical_world = _canonical_world_from_base(env, base)
        workspace = _workspace_from_points(canonical_world, target_request)
        workspace.update(common_meta)
        # Full accepted cloud (base frame), independent of the z-band slice that
        # _workspace_from_points stores in accepted_reachable_points.
        workspace["accepted_points_base"] = base[: int(args_cli.max_saved_points)].tolist()
        if args_cli.coverage_reference_points_json:
            ref = json.loads(Path(args_cli.coverage_reference_points_json).read_text(encoding="utf-8"))
            ref_base = np.asarray(ref["base"], dtype=np.float64)
            tol = float(args_cli.coverage_tolerance_m)
            d2 = ((ref_base[:, None, 1:3] - base[None, :, 1:3].astype(np.float64)) ** 2).sum(-1)
            coverage = float((d2.min(axis=1) <= tol * tol).mean()) if len(base) else 0.0
            workspace["coverage_of_reference_yz"] = {
                "reference_points_json": str(args_cli.coverage_reference_points_json),
                "reference_points_count": int(len(ref_base)),
                "tolerance_m": tol,
                "coverage_fraction": coverage,
            }
            print(
                f"[dynamic_sampler/coverage] {coverage * 100:.1f}% of {len(ref_base)} reference YZ points "
                f"within {tol * 100:.0f} cm of a dynamic point",
                flush=True,
            )
        workspace["accepted_base_bounds"] = {
            "x_min": float(np.min(base[:, 0])),
            "x_max": float(np.max(base[:, 0])),
            "y_min": float(np.min(base[:, 1])),
            "y_max": float(np.max(base[:, 1])),
            "z_min": float(np.min(base[:, 2])),
            "z_max": float(np.max(base[:, 2])),
        }
        out.write_text(json.dumps(workspace, indent=2) + "\n", encoding="utf-8")
        _print_json("[dynamic_sampler/result]", _sampler_result_summary(workspace))
        print(f"[dynamic_sampler/output] {out}", flush=True)
        return 0
    finally:
        env.close()


def _load_sweep_trials() -> list[dict[str, float | list[float]]]:
    if args_cli.sweep_json:
        payload = json.loads(Path(args_cli.sweep_json).read_text(encoding="utf-8"))
        trials = payload["trials"] if isinstance(payload, dict) and "trials" in payload else payload
        if not isinstance(trials, list) or not all(isinstance(t, dict) for t in trials):
            raise RuntimeError(f"--sweep_json must contain a list of cfg-override dicts, got {type(trials)}")
        return trials
    return [
        {"explicit_pid_kp_scale": kp, "explicit_pid_kd_scale": kd}
        for kp in (1.0, 1.5, 2.0)
        for kd in (1.0, 1.5, 2.0)
    ]


def run_sweep() -> int:
    """Sweep explicit PID/bias parameters over a fixed-pose hold trial and rank results."""
    trials = _load_sweep_trials()
    if not trials:
        raise RuntimeError("Sweep trial list is empty; provide --sweep_json with a non-empty 'trials' list.")
    env = _make_env("none")
    try:
        uw = env._uw
        # Keys handled by writing PhysX drive gains to sim rather than via cfg.
        drive_keys = ("drive_stiffness_per_cylinder", "drive_damping_per_cylinder")
        # Keys the explicit-effort controller reads via getattr(cfg, key, default),
        # which therefore may not exist as declared config attributes yet.
        dynamic_key_defaults = {"explicit_pid_effort_scale": 1.0}
        for trial_params in trials:
            for key in trial_params:
                if key in drive_keys:
                    continue
                if not hasattr(env.cfg, key):
                    if key in dynamic_key_defaults:
                        setattr(env.cfg, key, dynamic_key_defaults[key])
                    else:
                        raise RuntimeError(f"Sweep parameter {key!r} is not a config attribute.")
        original = {
            key: getattr(env.cfg, key)
            for params in trials
            for key in params
            if key not in drive_keys
        }
        orig_stiffness, orig_damping = _read_cylinder_drive_gains(env)
        q_default = uw.robot.data.default_joint_pos[: env.num_envs, uw._cylinder_joint_ids].detach().clone()
        results: list[dict] = []
        for idx, params in enumerate(trials):
            for key, value in original.items():
                setattr(env.cfg, key, value)
            for key, value in params.items():
                if key not in drive_keys:
                    setattr(env.cfg, key, value)
            _write_cylinder_drive_gains(
                env,
                stiffness=params.get("drive_stiffness_per_cylinder", orig_stiffness),
                damping=params.get("drive_damping_per_cylinder", orig_damping),
            )
            trial = _run_hold_trial(env, q_default)
            accepted_fraction = float(trial["hold_ok"].float().mean().detach().cpu().item())
            entry = {
                "params": params,
                "hold_pass_fraction": accepted_fraction,
                "calibrated_fraction": float(trial["calibrated"].float().mean().detach().cpu().item()),
                "reset_fraction": float(trial["reset_seen"].float().mean().detach().cpu().item()),
                "limit_hit_fraction": float(trial["limit_hit_seen"].float().mean().detach().cpu().item()),
                "cyl_err_max_mean": float(trial["cyl_err_max"].mean().detach().cpu().item()),
                "cyl_err_max_worst": float(trial["cyl_err_max"].max().detach().cpu().item()),
                "tool_drift_max_mean": float(trial["tool_drift_max"].mean().detach().cpu().item()),
                "tool_drift_max_worst": float(trial["tool_drift_max"].max().detach().cpu().item()),
                "tool_clear_min": float(trial["tool_clear_min"].min().detach().cpu().item()),
                "arm_clear_min": float(trial["arm_clear_min"].min().detach().cpu().item()),
                "nonfinite_steps": int(trial["nonfinite_steps"].sum().detach().cpu().item()),
                "huge_transform_steps": int(trial["huge_transform_steps"].sum().detach().cpu().item()),
            }
            results.append(entry)
            _print_json(f"[sweep/trial {idx + 1}/{len(trials)}]", entry)
        for key, value in original.items():
            setattr(env.cfg, key, value)
        _write_cylinder_drive_gains(env, stiffness=orig_stiffness, damping=orig_damping)
        ranked = sorted(
            results,
            key=lambda r: (
                -r["hold_pass_fraction"],
                r["limit_hit_fraction"],
                r["reset_fraction"],
                1 if (r["nonfinite_steps"] > 0 or r["huge_transform_steps"] > 0) else 0,
                0 if r["tool_clear_min"] > 0.0 else 1,
                -r["calibrated_fraction"],
                r["tool_drift_max_mean"],
                r["cyl_err_max_mean"],
            ),
        )
        payload = {
            "fixed_pose": "default_joint_pos",
            "num_envs_per_trial": int(env.num_envs),
            "hold_seconds": float(args_cli.hold_seconds),
            "hold_cyl_err_max": float(args_cli.hold_cyl_err_max),
            "hold_tool_drift_max": float(args_cli.hold_tool_drift_max),
            "trials_ranked": ranked,
        }
        out = Path(args_cli.sweep_output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        _print_json("[sweep/best]", ranked[0])
        print(f"[sweep/output] {out}", flush=True)
        return 0 if ranked and ranked[0]["hold_pass_fraction"] > 0.0 else 2
    finally:
        env.close()


class LivePlotter:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.t: list[float] = []
        self.ref: list[np.ndarray] = []
        self.actual: list[np.ndarray] = []
        self.ref_vel: list[np.ndarray] = []
        self.actual_vel: list[np.ndarray] = []
        self._plt = None
        self._fig = None
        self._axes = None
        if enabled:
            import matplotlib.pyplot as plt

            plt.ion()
            self._plt = plt
            self._fig, self._axes = plt.subplots(2, 3, figsize=(13, 7), num="PPO1 tool tracking")
            self._fig.tight_layout()

    def append(self, t: float, ref: np.ndarray, actual: np.ndarray) -> None:
        self.t.append(float(t))
        self.ref.append(ref.copy())
        self.actual.append(actual.copy())
        if len(self.t) == 1:
            self.ref_vel.append(np.zeros(3))
            self.actual_vel.append(np.zeros(3))
        else:
            dt = max(self.t[-1] - self.t[-2], 1.0e-6)
            self.ref_vel.append((self.ref[-1] - self.ref[-2]) / dt)
            self.actual_vel.append((self.actual[-1] - self.actual[-2]) / dt)

    def draw(self) -> None:
        if not self.enabled or self._plt is None or self._axes is None:
            return
        window = int(args_cli.plot_window)
        t = np.asarray(self.t[-window:])
        ref = np.asarray(self.ref[-window:])
        actual = np.asarray(self.actual[-window:])
        ref_v = np.asarray(self.ref_vel[-window:])
        actual_v = np.asarray(self.actual_vel[-window:])
        labels = ("x", "y", "z")
        for i in range(3):
            ax = self._axes[0][i]
            ax.clear()
            ax.plot(t, ref[:, i], label="ref")
            ax.plot(t, actual[:, i], label="actual")
            ax.set_title(f"tool pos {labels[i]}")
            ax.grid(True)
            ax.legend(loc="best")
            av = self._axes[1][i]
            av.clear()
            av.plot(t, ref_v[:, i], label="ref vel")
            av.plot(t, actual_v[:, i], label="actual vel")
            av.set_title(f"tool vel {labels[i]}")
            av.grid(True)
            av.legend(loc="best")
        self._fig.canvas.draw_idle()
        self._plt.pause(0.001)

    def save(self, out_dir: Path, extra_series: dict[str, list[float]] | None = None) -> list[str]:
        if not self.t:
            return []
        if self._plt is None and "matplotlib.pyplot" not in sys.modules:
            import matplotlib

            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out_dir.mkdir(parents=True, exist_ok=True)
        saved: list[str] = []
        t = np.asarray(self.t)
        ref = np.asarray(self.ref)
        actual = np.asarray(self.actual)
        ref_v = np.asarray(self.ref_vel)
        actual_v = np.asarray(self.actual_vel)
        labels = ("x", "y", "z")
        fig, axes = plt.subplots(2, 3, figsize=(13, 7))
        for i in range(3):
            axes[0][i].plot(t, ref[:, i], label="ref")
            axes[0][i].plot(t, actual[:, i], label="actual")
            axes[0][i].set_title(f"tool pos {labels[i]}")
            axes[0][i].grid(True)
            axes[0][i].legend(loc="best")
            axes[1][i].plot(t, ref_v[:, i], label="ref vel")
            axes[1][i].plot(t, actual_v[:, i], label="actual vel")
            axes[1][i].set_title(f"tool vel {labels[i]}")
            axes[1][i].grid(True)
            axes[1][i].legend(loc="best")
        fig.tight_layout()
        path = out_dir / "tool_tracking_xyz.png"
        fig.savefig(str(path), dpi=140)
        plt.close(fig)
        saved.append(str(path))
        if extra_series:
            names = sorted(extra_series)
            fig, axes = plt.subplots(len(names), 1, figsize=(10, 2.6 * len(names)), squeeze=False)
            for row, name in enumerate(names):
                series = np.asarray(extra_series[name], dtype=np.float64)
                axes[row][0].plot(np.arange(len(series)), series)
                axes[row][0].set_title(name)
                axes[row][0].grid(True)
            fig.tight_layout()
            path = out_dir / "hold_convergence.png"
            fig.savefig(str(path), dpi=140)
            plt.close(fig)
            saved.append(str(path))
        return saved


def _observe_action(env: HorizonIDFDEnv) -> torch.Tensor:
    if args_cli.observe_policy == "zero":
        return torch.zeros((env.num_envs, env.num_actions), device=env.device)
    if args_cli.observe_policy == "random":
        return torch.empty((env.num_envs, env.num_actions), device=env.device).uniform_(-1.0, 1.0)
    return env.compute_ground_tracking_action(gain=float(args_cli.greedy_gain))


def run_observe() -> int:
    env = _make_env(args_cli.trajectory_mode)
    plotter = LivePlotter(bool(args_cli.live_plots))
    err_series: list[float] = []
    ref_series: list[np.ndarray] = []
    actual_series: list[np.ndarray] = []
    cyl_err_series: list[float] = []
    cyl_joint_vel_series: list[float] = []
    cyl_ext_vel_series: list[float] = []
    arm_joint_vel_series: list[float] = []
    root_motion_series: list[float] = []
    tool_clear_series: list[float] = []
    arm_clear_series: list[float] = []
    nan_steps = 0
    huge_transform_steps = 0
    wrapper_done_steps = 0
    hidden_base_resets = 0
    try:
        _print_json("[observe/geometry]", env.tool_geometry_report())
        _print_json("[observe/frame]", env.discover_controlled_frame())
        last = time.time()
        for step in range(int(args_cli.observe_steps)):
            ref = env.current_ground_target_world()[0].detach().cpu().numpy()
            actual = env.get_tool_bottom_world_pos()[0].detach().cpu().numpy()
            plotter.append(step * float(env._dt) * float(env._H), ref, actual)
            if step % max(int(args_cli.plot_interval), 1) == 0:
                plotter.draw()
            ep_before = int(env._uw.episode_length_buf[0].detach().cpu().item())
            _, _, terminated, truncated, extras = env.step(_observe_action(env))
            wrapper_done_steps += int(bool((terminated | truncated)[0].detach().cpu().item()))
            if int(env._uw.episode_length_buf[0].detach().cpu().item()) < ep_before:
                hidden_base_resets += 1
            step_ref = env.current_ground_target_world()[0]
            step_actual = env.get_tool_bottom_world_pos()[0]
            body_pos = env._uw.robot.data.body_link_state_w[0, :, :3]
            step_finite = bool(
                (torch.isfinite(step_ref).all() & torch.isfinite(step_actual).all() & torch.isfinite(body_pos).all())
                .detach()
                .cpu()
                .item()
            )
            if not step_finite:
                nan_steps += 1
            else:
                origin0 = env._uw.scene.env_origins[0]
                if float((body_pos - origin0).abs().max().detach().cpu().item()) > 100.0:
                    huge_transform_steps += 1
            err_series.append(
                float(torch.linalg.norm(torch.nan_to_num(step_actual - step_ref, nan=1.0e6)).detach().cpu().item())
            )
            ref_series.append(
                torch.nan_to_num(step_ref, nan=1.0e6, posinf=1.0e6, neginf=-1.0e6).detach().cpu().numpy()
            )
            actual_series.append(
                torch.nan_to_num(step_actual, nan=1.0e6, posinf=1.0e6, neginf=-1.0e6).detach().cpu().numpy()
            )
            cyl_qd = env._uw.robot.data.joint_vel[:, env._uw._cylinder_joint_ids]
            _, cyl_xd = env._uw._cylinder_adapter.current_from_cylinder_joints(
                env._uw.robot.data.joint_pos[:, env._uw._cylinder_joint_ids],
                cyl_qd,
            )
            arm_qd = env._uw.robot.data.joint_vel[:, env._uw._arm_joint_ids]
            cyl_joint_vel_series.append(
                float(torch.nan_to_num(cyl_qd[0].abs().max(), nan=1.0e6).detach().cpu().item())
            )
            cyl_ext_vel_series.append(
                float(torch.nan_to_num(cyl_xd[0].abs().max(), nan=1.0e6).detach().cpu().item())
            )
            arm_joint_vel_series.append(
                float(torch.nan_to_num(arm_qd[0].abs().max(), nan=1.0e6).detach().cpu().item())
            )
            root_metrics = env.ground_safety_metrics()
            root_motion_series.append(
                float(torch.nan_to_num(root_metrics["root_motion_norm"][0], nan=1.0e6).detach().cpu().item())
            )
            tool_clear_series.append(
                float(torch.nan_to_num(step_actual[2], nan=-1.0e6).detach().cpu().item()) - float(args_cli.ground_z)
            )
            arm_clear_series.append(
                float(torch.nan_to_num(_arm_clearance_cheap(env)[0], nan=-1.0e6).detach().cpu().item())
            )
            uw_now = env._uw
            if uw_now._debug_manual_cylinder_position_target is not None:
                cyl_q_now = uw_now.robot.data.joint_pos[:, uw_now._cylinder_joint_ids]
                cyl_err_series.append(
                    float(
                        torch.nan_to_num(
                            (uw_now._debug_manual_cylinder_position_target - cyl_q_now)[0].abs().max(), nan=1.0e6
                        )
                        .detach()
                        .cpu()
                        .item()
                    )
                )
            else:
                cyl_err_series.append(float("nan"))
            if time.time() - last > 1.0:
                metrics = env.ground_safety_metrics()
                report = env.control_pipeline_report()
                causes = extras.get("cause_masks", {})
                components = extras.get("cause_components", {})
                values = extras.get("cause_values", {})
                active_causes = [
                    name for name, mask in causes.items()
                    if bool(mask[0].detach().cpu().item())
                ]
                active_components = [
                    name for name, mask in components.items()
                    if bool(mask[0].detach().cpu().item())
                ]
                controlled_b = values.get("controlled_point_base")
                controlled_b0 = (
                    controlled_b[0].detach().cpu().tolist()
                    if isinstance(controlled_b, torch.Tensor) and controlled_b.ndim >= 2
                    else None
                )
                target_b0 = env._world_to_tracking_frame(env.current_ground_target_world())[0].detach().cpu().tolist()
                action_now = env._prev_action[0].detach().cpu().tolist()
                dx_des_vec = env._prev_delta_x_des[0].detach().cpu().tolist()
                fd_vec = env._prev_fd_delta[0].detach().cpu().tolist()
                meas_vec = env._prev_meas_delta[0].detach().cpu().tolist()
                cmd_vec = env._prev_cmd_delta[0].detach().cpu().tolist()
                uw = env._uw
                bias_cal = bool(uw._explicit_bias_calibrated[0].detach().cpu().item())
                bias_step = int(uw._explicit_bias_calib_step[0].detach().cpu().item())
                bias_count = int(uw._explicit_bias_window_count[0].detach().cpu().item())
                force_abs = float(uw._last_force_cmd[0].detach().abs().max().cpu().item())
                bias_abs = float(uw._explicit_force_bias[0].detach().abs().max().cpu().item())
                cyl_err_abs = 0.0
                cyl_err_vec: list[float] = []
                if uw._debug_manual_cylinder_position_target is not None:
                    cyl_q = uw.robot.data.joint_pos[:, uw._cylinder_joint_ids]
                    cyl_err_all = (uw._debug_manual_cylinder_position_target - cyl_q)[0].detach()
                    cyl_err_abs = float(cyl_err_all.abs().max().cpu().item())
                    cyl_err_vec = [round(float(v), 4) for v in cyl_err_all.cpu().tolist()]
                root_motion = float(metrics["root_motion_norm"][0].detach().cpu().item())
                cyl_joint_vel_abs = float(
                    torch.nan_to_num(
                        uw.robot.data.joint_vel[0, uw._cylinder_joint_ids].abs().max(), nan=1.0e6
                    ).detach().cpu().item()
                )
                _, cyl_xd_print = uw._cylinder_adapter.current_from_cylinder_joints(
                    uw.robot.data.joint_pos[:, uw._cylinder_joint_ids],
                    uw.robot.data.joint_vel[:, uw._cylinder_joint_ids],
                )
                cyl_ext_vel_abs = float(
                    torch.nan_to_num(cyl_xd_print[0].abs().max(), nan=1.0e6).detach().cpu().item()
                )
                arm_joint_vel_abs = float(
                    torch.nan_to_num(
                        uw.robot.data.joint_vel[0, uw._arm_joint_ids].abs().max(), nan=1.0e6
                    ).detach().cpu().item()
                )
                lim_lo, lim_hi = env._joint_limits()
                cyl_q_now0 = uw.robot.data.joint_pos[0, uw._cylinder_joint_ids]
                gap_lo = [round(float(v), 4) for v in (cyl_q_now0 - lim_lo[0]).detach().cpu().tolist()]
                gap_hi = [round(float(v), 4) for v in (lim_hi[0] - cyl_q_now0).detach().cpu().tolist()]
                print(
                    "[observe] "
                    f"step={step} err={metrics['tracking_error'][0].item():.4f} "
                    f"tool_clear={metrics['tool_bottom_clearance'][0].item():.4f} "
                    f"target_z={metrics['target_tool_bottom_world_z'][0].item():.4f} "
                    f"done={bool((terminated | truncated)[0].item())} "
                    f"causes={active_causes} "
                    f"components={active_components} "
                    f"target_b={[round(float(v), 4) for v in target_b0]} "
                    f"actual_b={None if controlled_b0 is None else [round(float(v), 4) for v in controlled_b0]} "
                    f"action={[round(float(v), 5) for v in action_now]} "
                    f"dx_des={[round(float(v), 6) for v in dx_des_vec]} "
                    f"fd={[round(float(v), 6) for v in fd_vec]} "
                    f"meas={[round(float(v), 6) for v in meas_vec]} "
                    f"cmd={[round(float(v), 6) for v in cmd_vec]} "
                    f"absmax=({report['last_delta_x_des_H_abs_max']:.6g},{report['last_delta_fd_H_abs_max']:.6g},{report['last_delta_cmd_H_abs_max']:.6g}) "
                    f"j_valid={report['jacobian_valid_mask']} "
                    f"bias_cal={bias_cal} "
                    f"bias_step={bias_step} "
                    f"bias_count={bias_count} "
                    f"force_abs={force_abs:.6g} "
                    f"bias_abs={bias_abs:.6g} "
                    f"cyl_err={cyl_err_abs:.6g} "
                    f"cyl_err_vec={cyl_err_vec} "
                    f"cyl_joint_vel_abs={cyl_joint_vel_abs:.6g} "
                    f"cyl_ext_vel_abs={cyl_ext_vel_abs:.6g} "
                    f"arm_joint_vel_abs={arm_joint_vel_abs:.6g} "
                    f"root_motion={root_motion:.4f} "
                    f"gap_lo={gap_lo} "
                    f"gap_hi={gap_hi}",
                    flush=True,
                )
                last = time.time()
        summary = _observe_summary(
            env,
            err_series=err_series,
            actual_series=actual_series,
            ref_series=ref_series,
            cyl_err_series=cyl_err_series,
            cyl_joint_vel_series=cyl_joint_vel_series,
            cyl_ext_vel_series=cyl_ext_vel_series,
            arm_joint_vel_series=arm_joint_vel_series,
            root_motion_series=root_motion_series,
            tool_clear_series=tool_clear_series,
            arm_clear_series=arm_clear_series,
            nan_steps=nan_steps,
            huge_transform_steps=huge_transform_steps,
            wrapper_done_steps=wrapper_done_steps,
            hidden_base_resets=hidden_base_resets,
        )
        _print_json("[observe/summary]", summary)
        if args_cli.summary_json:
            summary_out = Path(args_cli.summary_json)
            summary_out.parent.mkdir(parents=True, exist_ok=True)
            summary_out.write_text(json.dumps(_jsonable(summary), indent=2) + "\n", encoding="utf-8")
            print(f"[observe/summary_output] {summary_out}", flush=True)
        if args_cli.save_plots_dir:
            saved = plotter.save(
                Path(args_cli.save_plots_dir),
                extra_series={
                    "tracking_error_m": err_series,
                    "cylinder_target_error_m": cyl_err_series,
                    "tool_ground_clearance_m": tool_clear_series,
                    "arm_ground_clearance_m": arm_clear_series,
                },
            )
            for path in saved:
                print(f"[observe/plot_saved] {path}", flush=True)
        if args_cli.enforce_pass and not bool(summary["passed"]):
            return 3
        return 0
    finally:
        env.close()


def _observe_summary(
    env: HorizonIDFDEnv,
    *,
    err_series: list[float],
    ref_series: list[np.ndarray],
    actual_series: list[np.ndarray],
    cyl_err_series: list[float],
    cyl_joint_vel_series: list[float],
    cyl_ext_vel_series: list[float],
    arm_joint_vel_series: list[float],
    root_motion_series: list[float],
    tool_clear_series: list[float],
    arm_clear_series: list[float],
    nan_steps: int,
    huge_transform_steps: int,
    wrapper_done_steps: int,
    hidden_base_resets: int,
) -> dict:
    warmup = max(min(int(args_cli.summary_warmup_steps), max(len(err_series) - 1, 0)), 0)
    err = np.asarray(err_series[warmup:], dtype=np.float64)
    cyl = np.asarray(cyl_err_series[warmup:], dtype=np.float64)
    cyl = cyl[np.isfinite(cyl)]
    tool_clear = np.asarray(tool_clear_series[warmup:], dtype=np.float64)
    arm_clear = np.asarray(arm_clear_series[warmup:], dtype=np.float64)
    tail = err[-50:] if len(err) else err
    bias_cal = bool(env._uw._explicit_bias_calibrated[0].detach().cpu().item())
    actual = np.asarray(actual_series[warmup:], dtype=np.float64)
    ref = np.asarray(ref_series[warmup:], dtype=np.float64)
    cyl_joint_vel = np.asarray(cyl_joint_vel_series[warmup:], dtype=np.float64)
    cyl_ext_vel = np.asarray(cyl_ext_vel_series[warmup:], dtype=np.float64)
    arm_joint_vel = np.asarray(arm_joint_vel_series[warmup:], dtype=np.float64)
    root_motion = np.asarray(root_motion_series[warmup:], dtype=np.float64)
    if len(actual):
        drift = np.linalg.norm(actual - actual[0], axis=-1)
        tool_drift_mean = float(np.mean(drift))
        tool_drift_max = float(np.max(drift))
        actual_step = np.linalg.norm(np.diff(actual, axis=0), axis=-1) if len(actual) > 1 else np.asarray([])
        actual_path_length = float(np.sum(actual_step)) if len(actual_step) else 0.0
        z_jitter_std = float(np.std(actual[:, 2]))
        z_peak_to_peak = float(np.max(actual[:, 2]) - np.min(actual[:, 2]))
    else:
        tool_drift_mean = float("nan")
        tool_drift_max = float("nan")
        actual_path_length = float("nan")
        z_jitter_std = float("nan")
        z_peak_to_peak = float("nan")
    if len(ref):
        ref_step = np.linalg.norm(np.diff(ref, axis=0), axis=-1) if len(ref) > 1 else np.asarray([])
        ref_path_length = float(np.sum(ref_step)) if len(ref_step) else 0.0
        ref_net_motion = float(np.linalg.norm(ref[-1] - ref[0]))
    else:
        ref_path_length = float("nan")
        ref_net_motion = float("nan")
    is_zero_hold = str(args_cli.observe_policy) == "zero" and bool(args_cli.hold_target)
    metrics = {
        "steps": len(err_series),
        "warmup_steps_excluded": warmup,
        "observe_policy": str(args_cli.observe_policy),
        "actuator_backend": str(getattr(env.cfg, "lowlevel_actuator_backend", "implicit_position")),
        "trajectory_mode": str(args_cli.trajectory_mode),
        "hold_target": bool(args_cli.hold_target),
        "tracking_err_mean": float(np.mean(err)) if len(err) else float("nan"),
        "tracking_err_max": float(np.max(err)) if len(err) else float("nan"),
        "tracking_err_mean_last50": float(np.mean(tail)) if len(tail) else float("nan"),
        "tool_drift_mean": tool_drift_mean,
        "tool_drift_max": tool_drift_max,
        "ref_path_length": ref_path_length,
        "ref_net_motion": ref_net_motion,
        "actual_path_length": actual_path_length,
        "actual_z_jitter_std": z_jitter_std,
        "actual_z_peak_to_peak": z_peak_to_peak,
        "cyl_err_mean": float(np.mean(cyl)) if len(cyl) else float("nan"),
        "cyl_err_max": float(np.max(cyl)) if len(cyl) else float("nan"),
        "cyl_joint_vel_abs_max": float(np.max(cyl_joint_vel)) if len(cyl_joint_vel) else float("nan"),
        "cyl_joint_vel_abs_mean": float(np.mean(cyl_joint_vel)) if len(cyl_joint_vel) else float("nan"),
        "cyl_ext_vel_abs_max": float(np.max(cyl_ext_vel)) if len(cyl_ext_vel) else float("nan"),
        "cyl_ext_vel_abs_mean": float(np.mean(cyl_ext_vel)) if len(cyl_ext_vel) else float("nan"),
        "arm_joint_vel_abs_max": float(np.max(arm_joint_vel)) if len(arm_joint_vel) else float("nan"),
        "arm_joint_vel_abs_mean": float(np.mean(arm_joint_vel)) if len(arm_joint_vel) else float("nan"),
        "root_motion_max": float(np.max(root_motion)) if len(root_motion) else float("nan"),
        "root_motion_mean": float(np.mean(root_motion)) if len(root_motion) else float("nan"),
        "tool_clearance_min": float(np.min(tool_clear)) if len(tool_clear) else float("nan"),
        "arm_clearance_min": float(np.min(arm_clear)) if len(arm_clear) else float("nan"),
        "nan_steps": int(nan_steps),
        "huge_transform_steps": int(huge_transform_steps),
        "wrapper_done_steps": int(wrapper_done_steps),
        "hidden_base_resets": int(hidden_base_resets),
        "bias_calibrated_final": bias_cal,
    }
    thresholds = {
        "pass_mean_tracking_err": float(args_cli.pass_mean_tracking_err),
        "pass_max_tracking_err": float(args_cli.pass_max_tracking_err),
        "pass_max_tool_drift": float(args_cli.pass_max_tool_drift),
        "pass_max_cyl_err": float(args_cli.pass_max_cyl_err),
        "pass_min_tool_clearance": float(args_cli.pass_min_tool_clearance),
        "pass_min_arm_clearance": float(args_cli.pass_min_arm_clearance),
    }
    checks = {
        "cyl_err_max_ok": bool(len(cyl)) and metrics["cyl_err_max"] <= thresholds["pass_max_cyl_err"],
        "tool_clearance_ok": bool(len(tool_clear)) and metrics["tool_clearance_min"] >= thresholds["pass_min_tool_clearance"],
        "arm_clearance_ok": bool(len(arm_clear)) and metrics["arm_clearance_min"] >= thresholds["pass_min_arm_clearance"],
        "no_nan_ok": nan_steps == 0,
        "no_huge_transform_ok": huge_transform_steps == 0,
        "no_hidden_resets_ok": hidden_base_resets == 0,
    }
    if str(getattr(env.cfg, "lowlevel_actuator_backend", "implicit_position")) == "explicit_effort_pid_bias":
        checks["bias_calibrated_ok"] = bias_cal
    if is_zero_hold:
        # Plant-stability gate: zero action holds the settled pose, so gate on drift from
        # that pose; tracking error vs the trajectory retains the reset-pose offset and is
        # reported but not gated.
        checks["tool_drift_ok"] = bool(len(actual)) and tool_drift_max <= thresholds["pass_max_tool_drift"]
    else:
        if str(args_cli.trajectory_mode).startswith("ground_") and not bool(args_cli.hold_target):
            checks["reference_path_moves_ok"] = bool(np.isfinite(ref_path_length)) and ref_path_length > 0.02
        checks["tracking_err_mean_ok"] = (
            bool(len(err)) and metrics["tracking_err_mean"] <= thresholds["pass_mean_tracking_err"]
        )
        checks["tracking_err_max_ok"] = (
            bool(len(err)) and metrics["tracking_err_max"] <= thresholds["pass_max_tracking_err"]
        )
    return {
        **metrics,
        "gate_mode": "zero_hold_drift" if is_zero_hold else "trajectory_tracking",
        "thresholds": thresholds,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _probe_deploy_one_step(env, ee_unit_yz: torch.Tensor) -> dict:
    """Run ONE deploy step of a constant base-frame EE delta through the exact live
    pipeline (map_ee_to_cylinder -> ID/FD rollout -> fd_gains -> clip -> smooth ramp),
    returning the per-step intermediate signals. ee_unit_yz is [uy, uz] (base frame)."""
    uw = env._uw
    # Base-frame EE delta command: [x=0, y, z] * ee_action_scale (mirrors step()).
    ee_delta = torch.zeros((env.num_envs, 3), device=env._device)
    ee_delta[:, 1] = float(ee_unit_yz[0])
    ee_delta[:, 2] = float(ee_unit_yz[1])
    desired_ee_delta = ee_delta * env._ee_action_scale
    dx_des, _ = env._ee_controller.map_ee_to_cylinder(desired_ee_delta)
    cq0 = uw.robot.data.joint_pos[:, uw._cylinder_joint_ids].detach().clone()
    x0, xd0 = uw._cylinder_adapter.current_from_cylinder_joints(
        cq0, uw.robot.data.joint_vel[:, uw._cylinder_joint_ids]
    )
    fd = env._idfd_rollout(dx_des, x0, xd0)
    clip_m = float(getattr(env.cfg, "horizon_delta_cmd_clip_m", 0.02))
    raw_cmd = env._fd_gains * fd
    cmd = torch.clamp(raw_cmd, -clip_m, clip_m)
    neutral = torch.zeros(env.num_envs, dtype=torch.bool, device=env._device)
    ep_before = uw.episode_length_buf.detach().clone()
    final_target, guard_flags, terminated, truncated = env._smooth_ramp(cmd, cq0, neutral)
    cq1 = uw.robot.data.joint_pos[:, uw._cylinder_joint_ids].detach().clone()
    x1, _ = uw._cylinder_adapter.current_from_cylinder_joints(
        cq1, uw.robot.data.joint_vel[:, uw._cylinder_joint_ids]
    )
    metrics = env.ground_safety_metrics() if env._trajectory_mode.startswith("ground_") else {}
    body_pos = uw.robot.data.body_link_state_w[:, :, :3]
    finite_body = torch.isfinite(body_pos).all(dim=(1, 2))
    max_abs_body = torch.nan_to_num(body_pos, nan=1.0e6, posinf=1.0e6, neginf=-1.0e6).abs().amax(dim=(1, 2))
    nan_inf_count = (~torch.isfinite(body_pos)).sum(dim=(1, 2)).long()
    if metrics:
        max_abs_body = metrics["max_abs_body_world_position"].detach().clone()
        nan_inf_count = metrics["nan_inf_count"].detach().clone()
    cyl_target_error = torch.nan_to_num((final_target - cq1).abs().amax(dim=-1), nan=1.0e6, posinf=1.0e6)
    cmd_clip_ratio = cmd.abs().amax(dim=-1) / max(clip_m, 1.0e-9)
    raw_clip_ratio = raw_cmd.abs().amax(dim=-1) / max(clip_m, 1.0e-9)
    hidden_reset = uw.episode_length_buf < ep_before
    return {
        "desired_ee_delta": desired_ee_delta,
        "dx_des": dx_des,
        "fd": fd,
        "cmd": cmd,
        "raw_cmd": raw_cmd,
        "meas_cyl_delta_joint": cq1 - cq0,
        "meas_cyl_delta_h": x1 - x0,
        "cyl_target_error": cyl_target_error,
        "cmd_clip_ratio": cmd_clip_ratio,
        "raw_clip_ratio": raw_clip_ratio,
        "clip_fraction": (cmd - raw_cmd).abs().gt(1.0e-8).float().mean(dim=-1),
        "guard_fraction": guard_flags.float().mean(dim=-1),
        "terminated": terminated.detach().clone(),
        "truncated": truncated.detach().clone(),
        "hidden_reset": hidden_reset.detach().clone(),
        "nan_inf_count": nan_inf_count,
        "max_abs_body": max_abs_body,
        "huge_transform": finite_body & (max_abs_body > 100.0),
        "tool_clearance": metrics.get("tool_bottom_clearance", torch.full((env.num_envs,), float("nan"), device=env._device)),
        "arm_clearance": metrics.get("arm_geometry_clearance", torch.full((env.num_envs,), float("nan"), device=env._device)),
    }


def run_axis_probe() -> int:
    """Deterministic single-axis sign & authority probe.

    For each base-frame axis (+Y, -Y, +Z, -Z), command a constant unit EE delta
    through the exact active pipeline and measure the resulting tool/contact motion
    in the base frame. This isolates the low-level EE->tool response from PPO. The
    settled pose (and J_eff) is measured once at reset and restored before each axis
    so the four probes are comparable.

    Pass per axis: dot(commanded_dir, measured_tool_delta - neutral_drift) > 0
    (correct sign), |measured_tool_delta - neutral_drift| >= probe_min_motion_m
    (nontrivial authority), and no NaN / huge transform / cylinder-limit hit.
    """
    env = _make_env(args_cli.trajectory_mode)
    uw = env._uw
    _print_json("[axis_probe/frame]", env.discover_controlled_frame())

    J = env._ee_controller.jacobian
    J0 = J.detach().clone() if J is not None else None
    if J0 is not None:
        S = torch.linalg.svdvals(J0.float())[0]
        print(f"[axis_probe] J_eff (env0) singular_values={[round(float(v),4) for v in S.tolist()]} "
              f"cond={float(S[0]/S[-1].clamp(min=1e-9)):.1f}", flush=True)

    # Snapshot the settled state so each axis starts identically.
    env_ids = torch.arange(env.num_envs, device=env._device)
    saved_root = uw.robot.data.root_state_w.detach().clone()
    saved_qpos = uw.robot.data.joint_pos.detach().clone()
    saved_qvel = uw.robot.data.joint_vel.detach().clone()

    def _restore_and_resettle() -> None:
        uw.robot.write_root_state_to_sim(saved_root, env_ids)
        uw.robot.write_joint_state_to_sim(saved_qpos, saved_qvel, None, env_ids)
        env._ee_controller._prev_delta = torch.zeros_like(env._ee_controller._prev_delta)
        cyl_q = saved_qpos[:, uw._cylinder_joint_ids].detach().clone()
        env._cylinder_hold_target = cyl_q.clone()
        tool_q = saved_qpos[:, uw._tool_joint_id].detach().clone().unsqueeze(-1)
        env._tool_hold_target = tool_q.clone()
        for _ in range(max(int(args_cli.probe_resettle_steps), 0)):
            uw.set_debug_manual_cylinder_targets(cyl_q, tool_position_target=tool_q)
            env._env.step(torch.zeros((env.num_envs, 5), device=env._device))

    # Neutral MUST be first so its common-mode drift is available to subtract from
    # every commanded axis before the pass/fail verdict is formed.
    axes = [
        ("0 ", torch.tensor([0.0, 0.0], device=env._device)),
        ("+Y", torch.tensor([1.0, 0.0], device=env._device)),
        ("-Y", torch.tensor([-1.0, 0.0], device=env._device)),
        ("+Z", torch.tensor([0.0, 1.0], device=env._device)),
        ("-Z", torch.tensor([0.0, -1.0], device=env._device)),
    ]
    min_motion = float(args_cli.probe_min_motion_m)
    max_neutral_drift = float(args_cli.probe_max_neutral_drift_m)
    neutral_delta_b: torch.Tensor | None = None
    results = []

    def _diagnose(
        *,
        final_cmd_norm: float,
        final_cmd_to_clip: float,
        measured_cyl_norm: float,
        response_norm: float,
        sign_ok: bool,
        no_fault: bool,
    ) -> str:
        if not no_fault:
            return "physics_or_limit_fault"
        if final_cmd_norm < 1.0e-4:
            return "final_cmd_tiny_mapper_gain_or_j_eff_regularization"
        cyl_ratio = measured_cyl_norm / max(final_cmd_norm, 1.0e-9)
        if final_cmd_to_clip >= 0.95 and cyl_ratio < 0.25:
            return "command_clip_saturated_but_cylinders_do_not_follow_actuator_or_drive"
        if cyl_ratio < 0.25:
            return "final_cmd_present_but_measured_cylinder_motion_small_actuator_or_tracking"
        if response_norm < min_motion:
            return "cylinders_move_but_tool_response_small_workspace_or_j_eff_conditioning"
        if not sign_ok:
            return "tool_response_wrong_sign_frame_or_j_eff_issue"
        return "authority_present"

    for name, unit_yz in axes:
        _restore_and_resettle()
        x0_b = env._controlled_point_base_frame()[0].detach().clone()
        ee0_b = env._world_to_base(env.get_tool_origin_world_pos())[0].detach().clone()
        sum_desired_ee_delta = torch.zeros((env.num_envs, 3), device=env._device)
        sum_dx_des = torch.zeros((env.num_envs, 4), device=env._device)
        sum_fd = torch.zeros((env.num_envs, 4), device=env._device)
        sum_cmd = torch.zeros((env.num_envs, 4), device=env._device)
        sum_raw_cmd = torch.zeros((env.num_envs, 4), device=env._device)
        sum_meas_cyl_h = torch.zeros((env.num_envs, 4), device=env._device)
        cyl_target_err_max = torch.zeros(env.num_envs, device=env._device)
        cmd_clip_ratio_max = torch.zeros(env.num_envs, device=env._device)
        raw_clip_ratio_max = torch.zeros(env.num_envs, device=env._device)
        clip_fraction_max = torch.zeros(env.num_envs, device=env._device)
        guard_fraction_max = torch.zeros(env.num_envs, device=env._device)
        tool_clearance_min = torch.full((env.num_envs,), float("inf"), device=env._device)
        arm_clearance_min = torch.full((env.num_envs,), float("inf"), device=env._device)
        nan_inf_count_total = torch.zeros(env.num_envs, dtype=torch.long, device=env._device)
        huge_transform_count = torch.zeros(env.num_envs, dtype=torch.long, device=env._device)
        hidden_reset_count = torch.zeros(env.num_envs, dtype=torch.long, device=env._device)
        cq_start = uw.robot.data.joint_pos[:, uw._cylinder_joint_ids].detach().clone()
        nan_hit = huge_hit = limit_hit = False
        done_hit = False
        lim_lo, lim_hi = env._joint_limits()
        for _ in range(max(int(args_cli.probe_steps), 1)):
            step_sig = _probe_deploy_one_step(env, unit_yz)
            sum_desired_ee_delta += step_sig["desired_ee_delta"]
            sum_dx_des += step_sig["dx_des"]
            sum_fd += step_sig["fd"]
            sum_cmd += step_sig["cmd"]
            sum_raw_cmd += step_sig["raw_cmd"]
            sum_meas_cyl_h += step_sig["meas_cyl_delta_h"]
            cyl_target_err_max = torch.maximum(cyl_target_err_max, step_sig["cyl_target_error"])
            cmd_clip_ratio_max = torch.maximum(cmd_clip_ratio_max, step_sig["cmd_clip_ratio"])
            raw_clip_ratio_max = torch.maximum(raw_clip_ratio_max, step_sig["raw_clip_ratio"])
            clip_fraction_max = torch.maximum(clip_fraction_max, step_sig["clip_fraction"])
            guard_fraction_max = torch.maximum(guard_fraction_max, step_sig["guard_fraction"])
            tool_clearance_min = torch.minimum(
                tool_clearance_min,
                torch.nan_to_num(step_sig["tool_clearance"], nan=float("inf"), posinf=float("inf"), neginf=-1.0e6),
            )
            arm_clearance_min = torch.minimum(
                arm_clearance_min,
                torch.nan_to_num(step_sig["arm_clearance"], nan=float("inf"), posinf=float("inf"), neginf=-1.0e6),
            )
            nan_inf_count_total += step_sig["nan_inf_count"].long()
            huge_transform_count += step_sig["huge_transform"].long()
            hidden_reset_count += step_sig["hidden_reset"].long()
            done_hit = done_hit or bool((step_sig["terminated"] | step_sig["truncated"])[0].detach().cpu().item())
            cq_now = uw.robot.data.joint_pos[:, uw._cylinder_joint_ids]
            if bool(((cq_now <= lim_lo + 1e-4) | (cq_now >= lim_hi - 1e-4)).any().item()):
                limit_hit = True
            if bool((step_sig["nan_inf_count"] > 0).any().item()):
                nan_hit = True
            if bool(step_sig["huge_transform"].any().item()):
                huge_hit = True
        x1_b = env._controlled_point_base_frame()[0].detach().clone()
        ee1_b = env._world_to_base(env.get_tool_origin_world_pos())[0].detach().clone()
        meas_delta_b = (x1_b - x0_b)
        ee_delta_b = (ee1_b - ee0_b)
        meas_cyl_delta = (uw.robot.data.joint_pos[:, uw._cylinder_joint_ids][0] - cq_start[0])
        commanded_dir = torch.tensor([0.0, float(unit_yz[0]), float(unit_yz[1])], device=env._device)
        is_neutral = float(torch.linalg.norm(unit_yz).item()) < 1e-9
        dot = float(torch.dot(meas_delta_b, commanded_dir).item())
        meas_norm = float(torch.linalg.norm(meas_delta_b).item())
        cos = dot / max(meas_norm, 1e-9)
        pred_delta_b = None
        if J0 is not None:
            pred = torch.matmul(J0[0], sum_dx_des[0].unsqueeze(-1)).squeeze(-1)
            pred_delta_b = [round(float(v), 5) for v in pred.tolist()]
        no_fault = not (nan_hit or huge_hit or limit_hit)
        final_cmd_norm = float(torch.linalg.norm(sum_cmd[0]).item())
        measured_cyl_h_norm = float(torch.linalg.norm(sum_meas_cyl_h[0]).item())
        measured_cyl_joint_norm = float(torch.linalg.norm(meas_cyl_delta).item())
        rec = {
            "axis": name.strip(),
            "action_y": round(float(unit_yz[0]), 5),
            "action_z": round(float(unit_yz[1]), 5),
            "commanded_dir_base_xyz": [round(float(v), 3) for v in commanded_dir.tolist()],
            "sum_desired_ee_delta_base_xyz": [round(float(v), 5) for v in sum_desired_ee_delta[0].tolist()],
            "desired_ee_delta_norm_m": round(float(torch.linalg.norm(sum_desired_ee_delta[0]).item()), 6),
            "measured_tool_delta_base_xyz": [round(float(v), 5) for v in meas_delta_b.tolist()],
            "measured_ee_origin_delta_base_xyz": [round(float(v), 5) for v in ee_delta_b.tolist()],
            "measured_tool_delta_norm_m": round(meas_norm, 5),
            "raw_dot_commanded_measured": round(dot, 6),
            "raw_cos_alignment": round(cos, 4),
            "predicted_tool_delta_base_xyz_from_Jeff": pred_delta_b,
            "sum_dx_des_cyl": [round(float(v), 5) for v in sum_dx_des[0].tolist()],
            "sum_fd_cyl": [round(float(v), 5) for v in sum_fd[0].tolist()],
            "sum_cmd_cyl": [round(float(v), 5) for v in sum_cmd[0].tolist()],
            "sum_raw_cmd_cyl": [round(float(v), 5) for v in sum_raw_cmd[0].tolist()],
            "measured_cylinder_delta_h": [round(float(v), 5) for v in sum_meas_cyl_h[0].tolist()],
            "measured_cylinder_joint_delta": [round(float(v), 5) for v in meas_cyl_delta.tolist()],
            "dx_des_norm_m": round(float(torch.linalg.norm(sum_dx_des[0]).item()), 6),
            "fd_delta_norm_m": round(float(torch.linalg.norm(sum_fd[0]).item()), 6),
            "final_cmd_norm_m": round(final_cmd_norm, 6),
            "measured_cylinder_delta_norm_m": round(measured_cyl_h_norm, 6),
            "measured_cylinder_joint_delta_norm_m": round(measured_cyl_joint_norm, 6),
            "tool_delta_norm_over_final_cmd_norm": round(meas_norm / max(final_cmd_norm, 1.0e-9), 6),
            "cylinder_delta_norm_over_final_cmd_norm": round(measured_cyl_h_norm / max(final_cmd_norm, 1.0e-9), 6),
            "cylinder_tracking_error_max_m": round(float(cyl_target_err_max[0].item()), 6),
            "final_cmd_absmax_to_clip": round(float(cmd_clip_ratio_max[0].item()), 6),
            "raw_cmd_absmax_to_clip": round(float(raw_clip_ratio_max[0].item()), 6),
            "clip_fraction_max": round(float(clip_fraction_max[0].item()), 6),
            "guard_fraction_max": round(float(guard_fraction_max[0].item()), 6),
            "tool_clearance_min_m": round(float(tool_clearance_min[0].item()), 6),
            "arm_clearance_min_m": round(float(arm_clearance_min[0].item()), 6),
            "nan_inf_count_total": int(nan_inf_count_total[0].item()),
            "huge_transform_warning_count": int(huge_transform_count[0].item()),
            "hidden_reset_count": int(hidden_reset_count[0].item()),
            "terminated_or_truncated": bool(done_hit),
            "nan_hit": nan_hit, "huge_hit": huge_hit, "cylinder_limit_hit": limit_hit,
            "is_neutral_reference": is_neutral,
        }
        if is_neutral:
            # The neutral (zero-command) probe measures common-mode drift. Large
            # drift = the plant is not holding (e.g. the gravity ratchet), which
            # must fail the gate even though "no command" cannot have a sign.
            neutral_delta_b = meas_delta_b.detach().clone()
            drift_ok = meas_norm <= max_neutral_drift
            passed = drift_ok and no_fault
            rec["neutral_drift_norm_m"] = round(meas_norm, 5)
            rec["neutral_drift_ok"] = bool(drift_ok)
            rec["bottleneck_diagnosis"] = "zero_hold_drift_ok" if drift_ok and no_fault else "zero_hold_drift_or_physics_fault"
            rec["passed"] = bool(passed)
            print(
                f"[axis_probe] {name}: NEUTRAL drift={rec['measured_tool_delta_base_xyz']} "
                f"|drift|={meas_norm:.4f}m (<= {max_neutral_drift:.3f}? {drift_ok}) "
                f"ee_origin_delta={rec['measured_ee_origin_delta_base_xyz']} "
                f"-> {'PASS' if passed else 'FAIL'}",
                flush=True,
            )
        else:
            # Verdict uses the command response AFTER removing the neutral drift —
            # this is what makes the gate hard to fool: a pipeline that drifts the
            # tool the same way regardless of command (the failure we found) will
            # have a near-zero neutral-subtracted response and fail here even though
            # its RAW motion looks large and correctly signed.
            ref = neutral_delta_b if neutral_delta_b is not None else torch.zeros_like(meas_delta_b)
            resp = meas_delta_b - ref
            resp_dot = float(torch.dot(resp, commanded_dir).item())
            resp_norm = float(torch.linalg.norm(resp).item())
            resp_cos = resp_dot / max(resp_norm, 1e-9)
            sign_ok = resp_dot > 0.0
            authority_ok = resp_norm >= min_motion
            passed = sign_ok and authority_ok and no_fault
            rec["command_response_minus_neutral_xyz"] = [round(float(v), 5) for v in resp.tolist()]
            rec["command_response_dot"] = round(resp_dot, 6)
            rec["command_response_norm_m"] = round(resp_norm, 5)
            rec["command_response_cos"] = round(resp_cos, 4)
            rec["tool_response_over_final_cmd_norm"] = round(resp_norm / max(final_cmd_norm, 1.0e-9), 6)
            rec["sign_correct"] = bool(sign_ok)
            rec["authority_ok"] = bool(authority_ok)
            rec["bottleneck_diagnosis"] = _diagnose(
                final_cmd_norm=final_cmd_norm,
                final_cmd_to_clip=float(cmd_clip_ratio_max[0].item()),
                measured_cyl_norm=measured_cyl_h_norm,
                response_norm=resp_norm,
                sign_ok=sign_ok,
                no_fault=no_fault,
            )
            rec["passed"] = bool(passed)
            print(
                f"[axis_probe] {name}: cmd_response(-neutral)={rec['command_response_minus_neutral_xyz']} "
                f"|resp|={resp_norm:.4f}m dot={resp_dot:+.5f} cos={resp_cos:+.3f} "
                f"sign_ok={sign_ok} authority_ok={authority_ok} "
                f"(raw_delta={rec['measured_tool_delta_base_xyz']} raw_cos={cos:+.3f}) "
                f"-> {'PASS' if passed else 'FAIL'}",
                flush=True,
            )
        print(
            f"           dx_des_sum={rec['sum_dx_des_cyl']} cmd_sum={rec['sum_cmd_cyl']} "
            f"meas_cyl_h={rec['measured_cylinder_delta_h']} "
            f"meas_cyl_joint={rec['measured_cylinder_joint_delta']} "
            f"diag={rec['bottleneck_diagnosis']} pred_tool_delta={pred_delta_b}",
            flush=True,
        )
        results.append(rec)

    if str(args_cli.mode) == "authority_sweep":
        _restore_and_resettle()
        x0_b = env._controlled_point_base_frame()[0].detach().clone()
        target_b0 = env._world_to_tracking_frame(env.current_ground_target_world())[0].detach().clone()
        err0_b = target_b0 - x0_b
        sum_desired_ee_delta = torch.zeros((env.num_envs, 3), device=env._device)
        sum_dx_des = torch.zeros((env.num_envs, 4), device=env._device)
        sum_fd = torch.zeros((env.num_envs, 4), device=env._device)
        sum_cmd = torch.zeros((env.num_envs, 4), device=env._device)
        sum_raw_cmd = torch.zeros((env.num_envs, 4), device=env._device)
        sum_meas_cyl_h = torch.zeros((env.num_envs, 4), device=env._device)
        cyl_target_err_max = torch.zeros(env.num_envs, device=env._device)
        cmd_clip_ratio_max = torch.zeros(env.num_envs, device=env._device)
        raw_clip_ratio_max = torch.zeros(env.num_envs, device=env._device)
        clip_fraction_max = torch.zeros(env.num_envs, device=env._device)
        guard_fraction_max = torch.zeros(env.num_envs, device=env._device)
        tool_clearance_min = torch.full((env.num_envs,), float("inf"), device=env._device)
        arm_clearance_min = torch.full((env.num_envs,), float("inf"), device=env._device)
        nan_inf_count_total = torch.zeros(env.num_envs, dtype=torch.long, device=env._device)
        huge_transform_count = torch.zeros(env.num_envs, dtype=torch.long, device=env._device)
        hidden_reset_count = torch.zeros(env.num_envs, dtype=torch.long, device=env._device)
        cq_start = uw.robot.data.joint_pos[:, uw._cylinder_joint_ids].detach().clone()
        nan_hit = huge_hit = limit_hit = False
        done_hit = False
        lim_lo, lim_hi = env._joint_limits()
        action_sum = torch.zeros(2, device=env._device)
        steps = max(int(args_cli.authority_greedy_steps), 1)
        for _ in range(steps):
            tool_b = env._controlled_point_base_frame().detach()
            target_b = env._world_to_tracking_frame(env.current_ground_target_world()).detach()
            err_b = target_b - tool_b
            unit_yz = torch.stack(
                [
                    err_b[:, 1].mean() / max(float(env._ee_action_scale), 1.0e-9),
                    err_b[:, 2].mean() / max(float(env._ee_action_scale), 1.0e-9),
                ]
            )
            unit_yz = torch.clamp(unit_yz * float(args_cli.greedy_gain), -1.0, 1.0)
            action_sum += unit_yz
            step_sig = _probe_deploy_one_step(env, unit_yz)
            sum_desired_ee_delta += step_sig["desired_ee_delta"]
            sum_dx_des += step_sig["dx_des"]
            sum_fd += step_sig["fd"]
            sum_cmd += step_sig["cmd"]
            sum_raw_cmd += step_sig["raw_cmd"]
            sum_meas_cyl_h += step_sig["meas_cyl_delta_h"]
            cyl_target_err_max = torch.maximum(cyl_target_err_max, step_sig["cyl_target_error"])
            cmd_clip_ratio_max = torch.maximum(cmd_clip_ratio_max, step_sig["cmd_clip_ratio"])
            raw_clip_ratio_max = torch.maximum(raw_clip_ratio_max, step_sig["raw_clip_ratio"])
            clip_fraction_max = torch.maximum(clip_fraction_max, step_sig["clip_fraction"])
            guard_fraction_max = torch.maximum(guard_fraction_max, step_sig["guard_fraction"])
            tool_clearance_min = torch.minimum(
                tool_clearance_min,
                torch.nan_to_num(step_sig["tool_clearance"], nan=float("inf"), posinf=float("inf"), neginf=-1.0e6),
            )
            arm_clearance_min = torch.minimum(
                arm_clearance_min,
                torch.nan_to_num(step_sig["arm_clearance"], nan=float("inf"), posinf=float("inf"), neginf=-1.0e6),
            )
            nan_inf_count_total += step_sig["nan_inf_count"].long()
            huge_transform_count += step_sig["huge_transform"].long()
            hidden_reset_count += step_sig["hidden_reset"].long()
            done_hit = done_hit or bool((step_sig["terminated"] | step_sig["truncated"])[0].detach().cpu().item())
            cq_now = uw.robot.data.joint_pos[:, uw._cylinder_joint_ids]
            if bool(((cq_now <= lim_lo + 1e-4) | (cq_now >= lim_hi - 1e-4)).any().item()):
                limit_hit = True
            if bool((step_sig["nan_inf_count"] > 0).any().item()):
                nan_hit = True
            if bool(step_sig["huge_transform"].any().item()):
                huge_hit = True
        x1_b = env._controlled_point_base_frame()[0].detach().clone()
        target_b1 = env._world_to_tracking_frame(env.current_ground_target_world())[0].detach().clone()
        err1_b = target_b1 - x1_b
        tool_delta_b = x1_b - x0_b
        meas_cyl_delta = (uw.robot.data.joint_pos[:, uw._cylinder_joint_ids][0] - cq_start[0])
        final_cmd_norm = float(torch.linalg.norm(sum_cmd[0]).item())
        measured_cyl_h_norm = float(torch.linalg.norm(sum_meas_cyl_h[0]).item())
        measured_cyl_joint_norm = float(torch.linalg.norm(meas_cyl_delta).item())
        err0_norm = float(torch.linalg.norm(err0_b[[1, 2]]).item())
        err1_norm = float(torch.linalg.norm(err1_b[[1, 2]]).item())
        improvement = err0_norm - err1_norm
        no_fault = not (nan_hit or huge_hit or limit_hit)
        sign_ok = improvement > 0.0
        authority_ok = improvement >= min_motion
        passed = sign_ok and authority_ok and no_fault
        response_norm = float(torch.linalg.norm(tool_delta_b).item())
        rec = {
            "axis": "greedy_to_current_target",
            "action_y_mean": round(float((action_sum[0] / steps).item()), 5),
            "action_z_mean": round(float((action_sum[1] / steps).item()), 5),
            "target_base_yz_start": [round(float(v), 5) for v in target_b0[[1, 2]].tolist()],
            "tool_base_yz_start": [round(float(v), 5) for v in x0_b[[1, 2]].tolist()],
            "tool_base_yz_end": [round(float(v), 5) for v in x1_b[[1, 2]].tolist()],
            "tracking_error_yz_start_m": round(err0_norm, 6),
            "tracking_error_yz_end_m": round(err1_norm, 6),
            "tracking_error_yz_improvement_m": round(improvement, 6),
            "sum_desired_ee_delta_base_xyz": [round(float(v), 5) for v in sum_desired_ee_delta[0].tolist()],
            "desired_ee_delta_norm_m": round(float(torch.linalg.norm(sum_desired_ee_delta[0]).item()), 6),
            "measured_tool_delta_base_xyz": [round(float(v), 5) for v in tool_delta_b.tolist()],
            "measured_tool_delta_norm_m": round(response_norm, 5),
            "sum_dx_des_cyl": [round(float(v), 5) for v in sum_dx_des[0].tolist()],
            "sum_fd_cyl": [round(float(v), 5) for v in sum_fd[0].tolist()],
            "sum_cmd_cyl": [round(float(v), 5) for v in sum_cmd[0].tolist()],
            "sum_raw_cmd_cyl": [round(float(v), 5) for v in sum_raw_cmd[0].tolist()],
            "measured_cylinder_delta_h": [round(float(v), 5) for v in sum_meas_cyl_h[0].tolist()],
            "measured_cylinder_joint_delta": [round(float(v), 5) for v in meas_cyl_delta.tolist()],
            "dx_des_norm_m": round(float(torch.linalg.norm(sum_dx_des[0]).item()), 6),
            "fd_delta_norm_m": round(float(torch.linalg.norm(sum_fd[0]).item()), 6),
            "final_cmd_norm_m": round(final_cmd_norm, 6),
            "measured_cylinder_delta_norm_m": round(measured_cyl_h_norm, 6),
            "measured_cylinder_joint_delta_norm_m": round(measured_cyl_joint_norm, 6),
            "tool_delta_norm_over_final_cmd_norm": round(response_norm / max(final_cmd_norm, 1.0e-9), 6),
            "cylinder_delta_norm_over_final_cmd_norm": round(measured_cyl_h_norm / max(final_cmd_norm, 1.0e-9), 6),
            "cylinder_tracking_error_max_m": round(float(cyl_target_err_max[0].item()), 6),
            "final_cmd_absmax_to_clip": round(float(cmd_clip_ratio_max[0].item()), 6),
            "raw_cmd_absmax_to_clip": round(float(raw_clip_ratio_max[0].item()), 6),
            "clip_fraction_max": round(float(clip_fraction_max[0].item()), 6),
            "guard_fraction_max": round(float(guard_fraction_max[0].item()), 6),
            "tool_clearance_min_m": round(float(tool_clearance_min[0].item()), 6),
            "arm_clearance_min_m": round(float(arm_clearance_min[0].item()), 6),
            "nan_inf_count_total": int(nan_inf_count_total[0].item()),
            "huge_transform_warning_count": int(huge_transform_count[0].item()),
            "hidden_reset_count": int(hidden_reset_count[0].item()),
            "terminated_or_truncated": bool(done_hit),
            "nan_hit": nan_hit,
            "huge_hit": huge_hit,
            "cylinder_limit_hit": limit_hit,
            "sign_correct": bool(sign_ok),
            "authority_ok": bool(authority_ok),
            "bottleneck_diagnosis": _diagnose(
                final_cmd_norm=final_cmd_norm,
                final_cmd_to_clip=float(cmd_clip_ratio_max[0].item()),
                measured_cyl_norm=measured_cyl_h_norm,
                response_norm=response_norm,
                sign_ok=sign_ok,
                no_fault=no_fault,
            ),
            "passed": bool(passed),
        }
        print(
            f"[authority_sweep] greedy_to_current_target: err_yz={err0_norm:.4f}->{err1_norm:.4f} "
            f"improve={improvement:+.4f}m cmd_norm={final_cmd_norm:.6f} "
            f"meas_cyl_norm={measured_cyl_h_norm:.6f} tool_delta={response_norm:.4f} "
            f"diag={rec['bottleneck_diagnosis']} -> {'PASS' if passed else 'FAIL'}",
            flush=True,
        )
        results.append(rec)

    summary = {
        "mode": str(args_cli.mode),
        "trajectory_mode": str(args_cli.trajectory_mode),
        "trajectory_plane": str(args_cli.trajectory_plane),
        "ee_action_scale_m": float(env._ee_action_scale),
        "ee_per_cyl_clip_m": float(getattr(env.cfg, "ee_per_cyl_clip_m", -1.0)),
        "ee_global_clip_m": float(getattr(env.cfg, "ee_global_clip_m", -1.0)),
        "horizon_delta_cmd_clip_m": float(getattr(env.cfg, "horizon_delta_cmd_clip_m", -1.0)),
        "reference_space_target": bool(args_cli.reference_space_target),
        "probe_steps": int(args_cli.probe_steps),
        "authority_greedy_steps": int(args_cli.authority_greedy_steps) if str(args_cli.mode) == "authority_sweep" else 0,
        "probe_min_motion_m": float(args_cli.probe_min_motion_m),
        "probe_max_neutral_drift_m": float(args_cli.probe_max_neutral_drift_m),
        "verdict_basis": "command_response_minus_neutral_drift",
        "diagnostic_idfd_rollout_calls": int(getattr(env, "_diagnostic_idfd_rollout_calls", 0)),
        "diagnostic_id_forward_calls": int(getattr(env, "_diagnostic_id_forward_calls", 0)),
        "diagnostic_fd_forward_calls": int(getattr(env, "_diagnostic_fd_forward_calls", 0)),
        "last_horizon_target_source": str(getattr(env, "_last_horizon_target_source", "")),
        "axes": results,
        "passed": all(r["passed"] for r in results),
    }
    _print_json("[axis_probe/summary]", summary)
    if args_cli.summary_json:
        out = Path(args_cli.summary_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[axis_probe/summary_output] {out}", flush=True)
    if args_cli.enforce_pass and not summary["passed"]:
        return 3
    return 0


def main() -> int:
    try:
        if args_cli.mode == "sample":
            return run_sample()
        if args_cli.mode == "sample_dynamic":
            return run_sample_dynamic()
        if args_cli.mode == "sweep":
            return run_sweep()
        if args_cli.mode in ("axis_probe", "authority_sweep"):
            return run_axis_probe()
        if args_cli.mode == "observe":
            return run_observe()
        raise RuntimeError(f"Unsupported mode {args_cli.mode}")
    finally:
        if not args_cli.fast_exit:
            simulation_app.close()


if __name__ == "__main__":
    exit_code = main()
    print(f"[exit_code] {int(exit_code)}", flush=True)
    if args_cli.fast_exit:
        os._exit(int(exit_code))
    raise SystemExit(exit_code)
