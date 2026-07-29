#!/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
"""Deterministic held-out-terrain evaluation and control-path audit for PPO3.

Two modes:

The actor-mean rollout reports cut-height error over either the training verge
or the held-out rougher verge. These numbers are terrain-following quantities;
they are deliberately not labelled as PPO2 trajectory-tracking errors.

``--control-path-audit`` Runtime proof that the classical layers are out of the
                        loop.  The structural version of this lives in
                        ``tests/test_sole_control_path.py``; this one steps the
                        real environment and perturbs the cascade's baseline
                        goal between steps.  If the plant command changes, the
                        cascade is still controlling something.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("checkpoint", nargs="?", default=None)
p.add_argument("--num_envs", type=int, default=64)
p.add_argument("--seed", type=int, default=17)
p.add_argument("--steps", type=int, default=120)
p.add_argument("--stage", type=int, default=4)
p.add_argument("--eval_terrain", action="store_true")
p.add_argument(
    "--path_family",
    default="mixed",
    choices=(
        "mixed",
        "straight",
        "lateral_sine",
        "s_curve",
        "random_spline",
        "offset_return",
        "boom_sweep",
        "lateral_chirp",
        "double_s_curve",
        "lane_change_hold",
        "gaussian_slalom",
        "fourier_meander",
    ),
    help="Tracking path to visualize/evaluate; use one environment for an explicit family.",
)
p.add_argument(
    "--terrain_height_scale",
    type=float,
    default=0.0,
    help="Positive relief-scale override for visualization; zero uses the curriculum.",
)
p.add_argument(
    "--plot_tracking",
    action="store_true",
    help="Show one live 2D XY plot of the desired path and actual cutter trajectory.",
)
p.add_argument("--plot_update_interval", type=int, default=15)
p.add_argument("--plot_out", default="")
p.add_argument(
    "--visual_grid",
    action="store_true",
    help="Force the visual-only gray grid floor (interactive mode enables it automatically).",
)
p.add_argument("--trajectory_library_path", default="")
p.add_argument("--command_rom_checkpoint", default="")
p.add_argument("--sole_ik_hint", action="store_true")
p.add_argument(
    "--direct_hold_probe",
    action="store_true",
    help=(
        "DIAGNOSTIC ONLY: after reset, freeze the deployed PhysX joint targets "
        "and bypass command-ROM updates to measure the fixture/metric floor."
    ),
)
p.add_argument(
    "--hold_effort_floor_n",
    type=float,
    default=0.0,
    help="DIAGNOSTIC ONLY: minimum effort limit for the five held arm joints.",
)
p.add_argument(
    "--hold_damping_scale",
    type=float,
    default=1.0,
    help="DIAGNOSTIC ONLY: multiplier on damping of the five held arm joints.",
)
p.add_argument("--control-path-audit", dest="audit", action="store_true")
p.add_argument("--report", default="")
AppLauncher.add_app_launcher_args(p)
a, unknown = p.parse_known_args()
sys.argv = [sys.argv[0], *unknown]

if not a.audit and not a.checkpoint:
    p.error("a checkpoint is required unless --control-path-audit is used")
if a.plot_update_interval <= 0:
    p.error("--plot_update_interval must be positive")
if a.terrain_height_scale < 0.0:
    p.error("--terrain_height_scale cannot be negative")

launcher = AppLauncher(a)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab.utils.math import quat_apply, quat_apply_inverse  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import PPO3_SOLE_PhysX  # noqa: F401,E402
from PPO3_SOLE_PhysX.distill import assert_teacher_disabled  # noqa: E402
from PPO3_SOLE_PhysX.terrain_env_cfg import (  # noqa: E402
    TerrainSolePolicyPPORunnerCfg,
    apply_terrain_curriculum_stage,
    configure_terrain_gpu_physx_memory,
)
from PPO3_SOLE_PhysX.terrain_observations import verge_observation_dim  # noqa: E402
from PPO3_SOLE_PhysX.terrain_convergence import (  # noqa: E402
    GATE_SPECS,
    summarize_convergence,
)


def build_env():
    env_cfg = parse_env_cfg(
        PPO3_SOLE_PhysX.TERRAIN_TASK_ID, device=a.device, num_envs=a.num_envs
    )
    apply_terrain_curriculum_stage(env_cfg, int(a.stage))
    if a.sole_ik_hint:
        raise ValueError("--sole_ik_hint is not part of the terrain observation contract")
    env_cfg.sole_ik_hint = False
    env_cfg.observation_space = verge_observation_dim() + 1
    env_cfg.use_eval_terrain = bool(a.eval_terrain)
    env_cfg.terrain_base_seed = int(a.seed)
    env_cfg.terrain_height_scale_override = float(a.terrain_height_scale)
    env_cfg.enable_contact_tracking_demo = False
    env_cfg.tracking_path_family = str(a.path_family)
    env_cfg.enable_visual_grid_floor = bool(a.visual_grid) or not bool(a.headless)
    # Count raw terrain safety violations in the evaluator without triggering a
    # synchronized physical redeployment. Nonfinite state still terminates in
    # the environment; finite violation rate is accumulated explicitly below.
    env_cfg.defer_safety_terminations = True
    env_cfg.head_down_termination_alignment = float(
        GATE_SPECS["head_alignment_hard_min"].threshold
    )
    env_cfg.randomize_start_phase = False
    env_cfg.randomize_trajectory_shape = False
    env_cfg.trajectory_shape = str(env_cfg.terrain_deployment_trajectory_shape)
    if a.trajectory_library_path:
        env_cfg.trajectory_library_path = str(Path(a.trajectory_library_path).resolve())
    if a.command_rom_checkpoint:
        env_cfg.command_rom_checkpoint = str(Path(a.command_rom_checkpoint).resolve())
    configure_terrain_gpu_physx_memory(env_cfg, int(a.num_envs))
    # A gate number produced with the DLS teacher active is a distillation
    # diagnostic, not a sole-controller result. Refuse before spending the run.
    assert_teacher_disabled(env_cfg, context="PPO3 evaluation")
    env = gym.make(PPO3_SOLE_PhysX.TERRAIN_TASK_ID, cfg=env_cfg, render_mode=None)
    return env, env.unwrapped


class LiveTrackingPlot:
    """Live plan-view XY plot of environment zero's desired and actual paths."""

    def __init__(self, *, family: str, update_interval: int, output: Path):
        import matplotlib.pyplot as plt

        self.plt = plt
        self.family = family
        self.update_interval = int(update_interval)
        self.output = output
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.target_x_m: list[float] = []
        self.target_y_m: list[float] = []
        self.cutter_x_m: list[float] = []
        self.cutter_y_m: list[float] = []

        plt.ion()
        self.figure, self.axis = plt.subplots(figsize=(9, 6))
        self.figure.canvas.manager.set_window_title(
            f"PPO3 tracking — {family}"
        )
        (self.target_line,) = self.axis.plot(
            [], [], "#d62728", linewidth=2.0, label="desired XY path"
        )
        (self.actual_line,) = self.axis.plot(
            [], [], "#1f77b4", linewidth=1.5, label="actual cutter XY trajectory"
        )
        self.axis.set_xlabel("x [m]")
        self.axis.set_ylabel("y [m]")
        self.axis.set_aspect("equal", adjustable="datalim")
        self.axis.grid(True, alpha=0.3)
        self.axis.legend(loc="best")
        self.axis.set_title(f"PPO3 path tracking: {family}")
        self.figure.tight_layout()

    @torch.no_grad()
    def sample(self, raw, step: int) -> None:
        cutter_world = raw._compute_cutter_pos_w()[0]
        cutter_x = float(cutter_world[0] - raw.scene.env_origins[0, 0])
        cutter_y = float(cutter_world[1] - raw.scene.env_origins[0, 1])
        self.target_x_m.append(cutter_x)
        self.target_y_m.append(float(raw.tracking_line_y_local()[0]))
        self.cutter_x_m.append(cutter_x)
        self.cutter_y_m.append(cutter_y)
        if step % self.update_interval == 0:
            self.refresh()

    def refresh(self) -> None:
        self.target_line.set_data(self.target_x_m, self.target_y_m)
        self.actual_line.set_data(self.cutter_x_m, self.cutter_y_m)
        self.axis.relim()
        self.axis.autoscale_view()
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()

    def finalize(self) -> None:
        self.refresh()
        self.figure.savefig(self.output, dpi=180, bbox_inches="tight")
        print(f"[PPO3-PLOT] wrote {self.output}", flush=True)


@torch.no_grad()
def control_path_audit(raw, steps: int = 60) -> dict:
    """Perturb the cascade's output and confirm the plant does not notice.

    A structural test proves the symbol is absent from the source. This proves
    the behaviour: the same action sequence is replayed twice, once with the
    cascade's baseline goal corrupted by a large offset between every step. If
    PPO3 is genuinely the sole controller the two commanded target streams are
    bit-identical.
    """
    torch.manual_seed(int(a.seed))
    script = torch.tanh(
        torch.randn((steps, raw.num_envs, 4), device=raw.device) * 0.5
    )

    def rollout(corrupt: bool) -> torch.Tensor:
        raw.reset(seed=int(a.seed))
        commands = []
        for step in range(steps):
            if corrupt:
                raw._baseline_cylinder_goal += 0.25
            raw.step(script[step])
            commands.append(raw._current_action.detach().clone())
        return torch.stack(commands)

    clean = rollout(corrupt=False)
    corrupted = rollout(corrupt=True)
    deviation = float((clean - corrupted).abs().max())
    return {
        "steps": steps,
        "max_command_deviation": deviation,
        "cascade_is_out_of_the_loop": deviation == 0.0,
    }


@torch.no_grad()
def head_action_direction_probe(raw, steps: int = 12) -> dict:
    """Measure the sign from c4 action to the signed lateral head-axis tilt."""
    rows = {}
    for label, command in (("positive", 0.5), ("negative", -0.5)):
        raw.reset(seed=int(a.seed))
        start_axis = raw._head_down_axis_world().detach().clone()
        start_x = raw._refresh_cylinder_state()[0][:, 3].detach().clone()
        action = torch.zeros((raw.num_envs, 4), device=raw.device)
        action[:, 3] = command
        for _ in range(int(steps)):
            raw.step(action)
        final_axis = raw._head_down_axis_world()
        final_x = raw._refresh_cylinder_state()[0][:, 3]
        rows[label] = {
            "command": command,
            "mean_axis_y_start": float(start_axis[:, 1].mean()),
            "mean_axis_y_final": float(final_axis[:, 1].mean()),
            "mean_axis_y_delta": float(
                (final_axis[:, 1] - start_axis[:, 1]).mean()
            ),
            "mean_c4_extension_delta_m": float((final_x - start_x).mean()),
            "min_head_alignment": float((-final_axis[:, 2]).min()),
        }
    return {"steps_per_sign": int(steps), "responses": rows}


@torch.no_grad()
def evaluate_verge(raw, policy, steps: int, tracking_plot=None) -> dict:
    obs, _ = raw.reset(seed=int(a.seed))
    if a.hold_effort_floor_n > 0.0 or a.hold_damping_scale != 1.0:
        if not a.direct_hold_probe:
            raise ValueError("hold actuator overrides require --direct_hold_probe")
        held_joint_ids = torch.cat(
            (
                raw._cylinder_joint_ids,
                torch.as_tensor(
                    [raw._tool_joint_id], dtype=torch.long, device=raw.device
                ),
            )
        )
        if a.hold_effort_floor_n > 0.0:
            current_effort = raw.robot.data.joint_effort_limits[
                :, held_joint_ids
            ]
            effort_floor = torch.full_like(current_effort, a.hold_effort_floor_n)
            raw.robot.write_joint_effort_limit_to_sim(
                torch.maximum(current_effort, effort_floor),
                joint_ids=held_joint_ids,
            )
        if a.hold_damping_scale != 1.0:
            if a.hold_damping_scale <= 0.0:
                raise ValueError("--hold_damping_scale must be positive")
            current_damping = raw.robot.data.joint_damping[:, held_joint_ids]
            raw.robot.write_joint_damping_to_sim(
                current_damping * float(a.hold_damping_scale),
                joint_ids=held_joint_ids,
            )
    base_id = raw._truck_base_body_id()
    anchor_base_quat = raw.robot.data.body_link_state_w[:, base_id, 3:7]
    base_forward = torch.tensor(
        [[1.0, 0.0, 0.0]], device=raw.device
    ).expand(raw.num_envs, -1)
    anchor_heading = quat_apply(anchor_base_quat, base_forward)[:, :2]
    anchor_heading = anchor_heading / torch.linalg.norm(
        anchor_heading, dim=-1, keepdim=True
    ).clamp(min=1.0e-8)
    base_state = raw.robot.data.body_link_state_w[:, base_id]
    head_state = raw.robot.data.body_link_state_w[:, raw._ee_body_id]
    previous_head_origin_truck = quat_apply_inverse(
        base_state[:, 3:7], head_state[:, :3] - base_state[:, :3]
    )
    previous_head_origin_velocity = torch.zeros_like(previous_head_origin_truck)
    if a.direct_hold_probe:
        # Zero through the command ROM is not necessarily a true stationary
        # plant input: it still enters the learned target-update path.  Freeze
        # the already deployed joint targets so this evaluator-only ablation
        # measures the PhysX/fixture/kinematics floor without adding a controller
        # to the scored path.  The normal evaluator never takes this branch.
        deployed_cylinder_position = raw.robot.data.joint_pos[
            :, raw._cylinder_joint_ids
        ].detach().clone()
        deployed_arm_position = raw.robot.data.joint_pos[
            :, raw._arm_joint_ids
        ].detach().clone()
        initial_cylinder_target_error = float(
            (raw._cylinder_pos_target - deployed_cylinder_position).abs().max()
        )
        initial_joint_reference_error = float(
            (raw._joint_ref - deployed_arm_position).abs().max()
        )
        # Latch the pose PhysX actually reached, rather than retaining possibly
        # stale targets from before `_synchronize_deployed_robot_state` copied a
        # template articulation state across the clones.
        frozen_cylinder_target = deployed_cylinder_position
        frozen_joint_reference = deployed_arm_position

        def hold_deployed_targets() -> None:
            raw._cylinder_pos_target[:] = frozen_cylinder_target
            raw._joint_ref[:] = frozen_joint_reference
            raw._joint_vel_ref.zero_()
            raw._current_valve_cmd.zero_()
            raw._fd_delta_corrected.zero_()

        raw._run_cylinder_id_pipeline = hold_deployed_targets
    signed_errors, errors, accelerations, alignments = [], [], [], []
    tool_acceleration_vectors = []
    base_heading_errors, ride_heights = [], []
    head_origin_accelerations, head_angular_speeds = [], []
    cylinder_velocities, cylinder_effort_ratios = [], []
    head_axes = []
    laterals, scan_validity, contacts, actions = [], [], [], []
    cutter_contact_forces, wheel_support, wheel_forces = [], [], []
    applied_actions, shield_gains, head_extensions = [], [], []
    safety_terminations = 0
    for step in range(steps):
        # get_inference_policy returns the MLPModel; its forward consumes the
        # observation dict and returns the deterministic mean.
        action = policy(obs)
        obs, _, terminated, truncated, _ = raw.step(action)
        signed_error = raw.cut_height_error()
        signed_errors.append(signed_error)
        errors.append(signed_error.abs())
        tool_acceleration = raw._measured_tool_acceleration_truck.detach().clone()
        tool_acceleration_vectors.append(tool_acceleration)
        accelerations.append(torch.linalg.norm(tool_acceleration, dim=-1))
        head_axis = raw._head_down_axis_world()
        head_axes.append(head_axis)
        alignments.append(-head_axis[:, 2])
        base_state = raw.robot.data.body_link_state_w[:, base_id]
        current_head_state = raw.robot.data.body_link_state_w[:, raw._ee_body_id]
        current_head_origin_truck = quat_apply_inverse(
            base_state[:, 3:7], current_head_state[:, :3] - base_state[:, :3]
        )
        head_origin_velocity = (
            current_head_origin_truck - previous_head_origin_truck
        ) / max(float(raw.step_dt), 1.0e-6)
        head_origin_acceleration = (
            head_origin_velocity - previous_head_origin_velocity
        ) / max(float(raw.step_dt), 1.0e-6)
        head_origin_accelerations.append(head_origin_acceleration)
        previous_head_origin_truck = current_head_origin_truck.detach().clone()
        previous_head_origin_velocity = head_origin_velocity.detach().clone()
        head_angular_speeds.append(
            torch.linalg.norm(current_head_state[:, 10:13], dim=-1)
        )
        _, cylinder_velocity = raw._refresh_cylinder_state()
        cylinder_velocities.append(cylinder_velocity)
        cylinder_effort = raw.robot.data.applied_torque[:, raw._cylinder_joint_ids]
        cylinder_effort_limit = raw.robot.data.joint_effort_limits[
            :, raw._cylinder_joint_ids
        ].clamp(min=1.0e-6)
        cylinder_effort_ratios.append(
            (cylinder_effort / cylinder_effort_limit).abs()
        )
        heading = quat_apply(base_state[:, 3:7], base_forward)[:, :2]
        heading = heading / torch.linalg.norm(
            heading, dim=-1, keepdim=True
        ).clamp(min=1.0e-8)
        heading_cross = (
            anchor_heading[:, 0] * heading[:, 1]
            - anchor_heading[:, 1] * heading[:, 0]
        )
        heading_dot = torch.sum(anchor_heading * heading, dim=-1).clamp(
            min=-1.0, max=1.0
        )
        base_heading_errors.append(torch.atan2(heading_cross, heading_dot).abs())
        ride_heights.append(base_state[:, 2])
        laterals.append(raw.verge_lateral_error().abs())
        scan_validity.append(raw.scan_valid_fraction())
        contacts.append(raw.cutter_ground_contact().float())
        cutter_contact_forces.append(raw.cutter_terrain_contact_force_n())
        wheel_support.append(raw.wheel_road_contact_fraction())
        wheel_forces.append(raw.wheel_road_contact_forces_n())
        actions.append(action)
        applied_actions.append(raw._sole_applied.detach().clone())
        shield_gains.append(raw._sole_shield_gain.detach().clone())
        head_extensions.append(raw._refresh_cylinder_state()[0][:, 3].detach().clone())
        safety_terminations += int(raw.terrain_safety_violation().sum())
        if tracking_plot is not None:
            tracking_plot.sample(raw, step)
    signed_error_steps = torch.stack(signed_errors)
    error_steps = torch.stack(errors)
    acceleration_steps = torch.stack(accelerations)
    tool_acceleration_vector_steps = torch.stack(tool_acceleration_vectors)
    alignment_steps = torch.stack(alignments)
    base_heading_error_steps = torch.stack(base_heading_errors)
    ride_height_steps = torch.stack(ride_heights)
    head_origin_acceleration_steps = torch.stack(head_origin_accelerations)
    head_angular_speed_steps = torch.stack(head_angular_speeds)
    cylinder_velocity_steps = torch.stack(cylinder_velocities)
    cylinder_effort_ratio_steps = torch.stack(cylinder_effort_ratios)
    head_axis_steps = torch.stack(head_axes)
    lateral_steps = torch.stack(laterals)
    scan_steps = torch.stack(scan_validity)
    contact_steps = torch.stack(contacts)
    cutter_contact_force_steps = torch.stack(cutter_contact_forces)
    wheel_support_steps = torch.stack(wheel_support)
    wheel_force_steps = torch.stack(wheel_forces)
    action_steps = torch.stack(actions)
    applied_action_steps = torch.stack(applied_actions)
    shield_gain_steps = torch.stack(shield_gains)
    head_extension_steps = torch.stack(head_extensions)
    row = {
        "terrain": "held_out" if a.eval_terrain else "training",
        "steps": int(steps),
        "num_envs": int(raw.num_envs),
        "deployment_template_env": int(
            getattr(raw, "_last_deployment_template", -1)
        ),
        "mean_signed_cut_height_error_m": float(signed_error_steps.mean()),
        "mean_abs_cut_height_error_m": float(error_steps.mean()),
        "p95_abs_cut_height_error_m": float(torch.quantile(error_steps, 0.95)),
        "p95_tool_acceleration_mps2": float(
            torch.quantile(acceleration_steps, 0.95)
        ),
        "p95_abs_tool_acceleration_xyz_mps2": torch.quantile(
            tool_acceleration_vector_steps.abs().reshape(-1, 3), 0.95, dim=0
        ).tolist(),
        "p95_head_origin_acceleration_mps2": float(
            torch.quantile(
                torch.linalg.norm(head_origin_acceleration_steps, dim=-1), 0.95
            )
        ),
        "p95_abs_head_origin_acceleration_xyz_mps2": torch.quantile(
            head_origin_acceleration_steps.abs().reshape(-1, 3), 0.95, dim=0
        ).tolist(),
        "p95_head_angular_speed_radps": float(
            torch.quantile(head_angular_speed_steps, 0.95)
        ),
        "p95_abs_cylinder_velocity_mps": torch.quantile(
            cylinder_velocity_steps.abs().reshape(-1, 4), 0.95, dim=0
        ).tolist(),
        "p95_cylinder_effort_limit_ratio": torch.quantile(
            cylinder_effort_ratio_steps.reshape(-1, 4), 0.95, dim=0
        ).tolist(),
        "max_cylinder_effort_limit_ratio": cylinder_effort_ratio_steps.amax(
            dim=(0, 1)
        ).tolist(),
        "min_head_alignment": float(alignment_steps.min()),
        "p01_head_alignment": float(torch.quantile(alignment_steps, 0.01)),
        "max_abs_head_axis_x": float(head_axis_steps[..., 0].abs().max()),
        "max_abs_head_axis_y": float(head_axis_steps[..., 1].abs().max()),
        "p95_head_axis_horizontal_norm": float(
            torch.quantile(torch.linalg.norm(head_axis_steps[..., :2], dim=-1), 0.95)
        ),
        "max_truck_heading_error_deg": float(
            torch.rad2deg(base_heading_error_steps.max())
        ),
        "p95_truck_heading_error_deg": float(
            torch.rad2deg(torch.quantile(base_heading_error_steps, 0.95))
        ),
        "truck_ride_height_m_min": float(ride_height_steps.min()),
        "truck_ride_height_m_max": float(ride_height_steps.max()),
        "mean_abs_lateral_error_m": float(lateral_steps.mean()),
        "scan_valid_fraction": float(scan_steps.mean()),
        "ground_contact_fraction": float(contact_steps.mean()),
        "mean_cutter_contact_force_n": float(cutter_contact_force_steps.mean()),
        "p95_cutter_contact_force_n": float(
            torch.quantile(cutter_contact_force_steps, 0.95)
        ),
        "min_wheel_road_contact_duty_cycle": float(
            (
                wheel_force_steps
                > float(getattr(raw.cfg, "road_contact_force_threshold_n", 10.0))
            ).float().mean(dim=(0, 1)).min()
        ),
        "mean_wheel_road_contact_fraction": float(wheel_support_steps.mean()),
        "mean_requested_policy_action": action_steps.mean(dim=(0, 1)).tolist(),
        "mean_abs_requested_policy_action": action_steps.abs()
        .mean(dim=(0, 1))
        .tolist(),
        "mean_applied_action": applied_action_steps.mean(dim=(0, 1)).tolist(),
        "mean_abs_applied_action": applied_action_steps.abs()
        .mean(dim=(0, 1))
        .tolist(),
        "limit_shield_active_fraction": float((shield_gain_steps < 0.999).float().mean()),
        "head_cylinder_extension_min_m": float(head_extension_steps.min()),
        "head_cylinder_extension_max_m": float(head_extension_steps.max()),
        "max_truck_post_constraint_drift_m": float(
            raw._max_truck_post_constraint_drift.max()
        ),
        "safety_termination_rate": safety_terminations / max(steps * raw.num_envs, 1),
    }
    family_rows = []
    for family in sorted(set(raw._tracking_path_family_names)):
        family_mask = torch.as_tensor(
            [name == family for name in raw._tracking_path_family_names],
            dtype=torch.bool,
            device=raw.device,
        )
        family_rows.append(
            {
                "family": family,
                "num_envs": int(family_mask.sum()),
                "mean_abs_cut_height_error_m": float(
                    error_steps[:, family_mask].mean()
                ),
                "p95_abs_cut_height_error_m": float(
                    torch.quantile(error_steps[:, family_mask], 0.95)
                ),
                "mean_abs_lateral_error_m": float(
                    lateral_steps[:, family_mask].mean()
                ),
                "ground_contact_fraction": float(
                    contact_steps[:, family_mask].mean()
                ),
            }
        )
    row["path_family_results"] = family_rows
    row["worst_family_mean_abs_cut_height_error_m"] = max(
        item["mean_abs_cut_height_error_m"] for item in family_rows
    )
    row["worst_family_mean_abs_lateral_error_m"] = max(
        item["mean_abs_lateral_error_m"] for item in family_rows
    )
    row["worst_family_ground_contact_fraction"] = min(
        item["ground_contact_fraction"] for item in family_rows
    )
    if a.direct_hold_probe:
        row["initial_cylinder_target_error_max_usd"] = initial_cylinder_target_error
        row["initial_joint_reference_error_max"] = initial_joint_reference_error
        held_joint_ids = torch.cat(
            (
                raw._cylinder_joint_ids,
                torch.as_tensor(
                    [raw._tool_joint_id], dtype=torch.long, device=raw.device
                ),
            )
        )
        row["held_joint_effort_limits"] = raw.robot.data.joint_effort_limits[
            0, held_joint_ids
        ].tolist()
        row["held_joint_damping"] = raw.robot.data.joint_damping[
            0, held_joint_ids
        ].tolist()
    quarter_rows = []
    for quarter, index_chunks in enumerate(torch.tensor_split(torch.arange(steps), 4)):
        if index_chunks.numel() == 0:
            continue
        start = int(index_chunks[0])
        stop = int(index_chunks[-1]) + 1
        quarter_rows.append(
            {
                "quarter": quarter + 1,
                "start_step": start,
                "stop_step": stop,
                "mean_signed_cut_height_error_m": float(
                    signed_error_steps[start:stop].mean()
                ),
                "mean_abs_cut_height_error_m": float(error_steps[start:stop].mean()),
                "ground_contact_fraction": float(contact_steps[start:stop].mean()),
                "mean_cutter_contact_force_n": float(
                    cutter_contact_force_steps[start:stop].mean()
                ),
                "mean_wheel_road_contact_fraction": float(
                    wheel_support_steps[start:stop].mean()
                ),
                "mean_abs_lateral_error_m": float(lateral_steps[start:stop].mean()),
                "min_head_alignment": float(alignment_steps[start:stop].min()),
                "mean_policy_action": action_steps[start:stop]
                .mean(dim=(0, 1))
                .tolist(),
                "mean_applied_action": applied_action_steps[start:stop]
                .mean(dim=(0, 1))
                .tolist(),
            }
        )
    row["trajectory_quarters"] = quarter_rows
    if tracking_plot is not None:
        tracking_plot.finalize()
    return row


def main() -> None:
    env, raw = build_env()
    strip = raw._mow_strip_cfg()
    report: dict = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "checkpoint": str(a.checkpoint) if a.checkpoint else None,
        "stage": int(a.stage),
        "ik_hint_ablation": bool(a.sole_ik_hint),
        "direct_hold_probe": bool(a.direct_hold_probe),
        "hold_effort_floor_n": float(a.hold_effort_floor_n),
        "hold_damping_scale": float(a.hold_damping_scale),
        "terrain_provenance": {
            "curriculum_version": str(raw.cfg.terrain_curriculum_version),
            "split": "held_out_validation" if a.eval_terrain else "training",
            "base_seed": int(strip.seed),
            "per_env_seed_stride": int(raw.cfg.terrain_seed_stride),
            "randomize_per_env": bool(raw.cfg.randomize_terrain_per_env),
            "amplitude_start_m": float(strip.amplitude_start_m),
            "amplitude_end_m": float(strip.amplitude_end_m),
            "length_m": float(strip.length_m),
            "resolution_m": float(strip.resolution_m),
            "reset_seed": int(a.seed),
            "num_envs": int(raw.num_envs),
            "steps": int(a.steps),
            "path_family": str(a.path_family),
            "path_families_realized": sorted(
                set(raw._tracking_path_family_names)
            ),
            "height_scales_realized": sorted(
                set(float(value) for value in raw._terrain_height_scales)
            ),
        },
    }

    if a.audit:
        report["control_path_audit"] = control_path_audit(raw, steps=min(a.steps, 60))
        report["head_action_direction_probe"] = head_action_direction_probe(
            raw, steps=min(a.steps, 12)
        )
        print(json.dumps(report["control_path_audit"], indent=2), flush=True)
        print(
            json.dumps(report["head_action_direction_probe"], indent=2),
            flush=True,
        )

    if a.checkpoint:
        runner_cfg = TerrainSolePolicyPPORunnerCfg()
        runner_cfg.seed = int(a.seed)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)
        runner = OnPolicyRunner(
            wrapped, runner_cfg.to_dict(), log_dir=None, device=a.device
        )
        checkpoint_path = Path(a.checkpoint).resolve()
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        infos = payload.get("infos") or {}
        source = str(infos.get("source", ""))
        schema = str(infos.get("schema", ""))
        checkpoint_width = int(infos.get("observation_width", -1))
        if schema != "terrain_teacher_student_v1" or source not in {
            "PPO3 terrain teacher distillation v1",
            "PPO3 terrain PPO",
        }:
            raise RuntimeError(
                "evaluation accepts only terrain_teacher_student_v1 checkpoints; "
                f"got source={source!r}, schema={schema!r}"
            )
        if checkpoint_width != int(raw.cfg.observation_space):
            raise RuntimeError(
                f"checkpoint observation width {checkpoint_width} != "
                f"environment width {raw.cfg.observation_space}"
            )
        runner.load(
            str(checkpoint_path),
            load_cfg={
                "actor": True,
                "critic": True,
                "optimizer": False,
                "iteration": True,
                "rnd": False,
            },
        )
        report["checkpoint_contract"] = {
            "source": source,
            "schema": schema,
            "observation_width": checkpoint_width,
            "teacher_mode": str(raw.cfg.sole_teacher_mode),
            "direct_target_demo": bool(raw.cfg.enable_contact_tracking_demo),
        }
        policy = runner.get_inference_policy(device=a.device)

        tracking_plot = None
        if a.plot_tracking:
            plot_path = (
                Path(a.plot_out).expanduser().resolve()
                if a.plot_out
                else ROOT
                / "PPO3_SOLE_PhysX"
                / "plots"
                / f"{a.path_family}_tracking.png"
            )
            tracking_plot = LiveTrackingPlot(
                family=str(a.path_family),
                update_interval=int(a.plot_update_interval),
                output=plot_path,
            )
        row = evaluate_verge(
            raw, policy, int(a.steps), tracking_plot=tracking_plot
        )
        summary = summarize_convergence(row)
        report["verge"] = row
        report["gates"] = summary["gates"]
        report["normalized_gate_violations"] = summary[
            "normalized_gate_violations"
        ]
        report["score"] = summary["score"]
        report["passed"] = summary["passed"]
        print(
            json.dumps(
                {
                    "verge": row,
                    "gates": report["gates"],
                    "score": report["score"],
                    "passed": report["passed"],
                },
                indent=2,
            ),
            flush=True,
        )

    if a.report:
        path = Path(a.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2))
        print(f"[PPO3-SOLE] wrote {path}", flush=True)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
