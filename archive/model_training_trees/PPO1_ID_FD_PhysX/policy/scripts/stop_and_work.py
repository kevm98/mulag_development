"""Run the interlocked park-work-relocate workflow in Isaac Lab.

The Policy never acts while the truck moves.  Truck relocation is a
deterministic straight-line wheel controller with the arm held in a transport
pose.  At each station the wheels are stopped, settled, and position-locked
before a complete A->B->A arm cycle is allowed to start.
"""

from __future__ import annotations

import argparse
import enum
import math
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isaaclab.app import AppLauncher


p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--checkpoint", type=str, required=True)
p.add_argument("--cylinder4_residual_scale_m", type=float, default=0.00025)
p.add_argument(
    "--trajectory_library_path",
    default=str(
        ROOT / "PPO2_ROM_PhysX/models/plant_reachable_trajectory_library_v3_large.json"
    ),
)
p.add_argument("--path_speed_mps", type=float, default=0.18)
p.add_argument("--stations", type=int, default=5, help="Number of arm-work stations, including the initial pose.")
p.add_argument("--station_spacing_m", type=float, default=2.0)
p.add_argument("--drive_speed_mps", type=float, default=0.6)
p.add_argument("--drive_accel_s", type=float, default=2.0)
p.add_argument("--transport_settle_s", type=float, default=0.5)
p.add_argument("--transport_arm_stiffness_scale", type=float, default=2.0)
p.add_argument("--transport_arm_damping_scale", type=float, default=2.0)
p.add_argument("--settle_s", type=float, default=1.0)
p.add_argument("--transport_path_tolerance_m", type=float, default=0.002)
p.add_argument("--transport_min_clearance_m", type=float, default=0.02)
p.add_argument("--transport_max_arm_joint_error", type=float, default=1.0e-5)
p.add_argument("--transport_max_root_z_excursion_m", type=float, default=0.001)
p.add_argument("--transport_max_tool_z_excursion_m", type=float, default=0.001)
p.add_argument("--transport_max_tool_relative_excursion_m", type=float, default=0.001)
p.add_argument("--preposition_steps", type=int, default=300)
p.add_argument("--arm_settle_steps", type=int, default=120)
p.add_argument(
    "--cycle_completion_guard_steps",
    type=int,
    default=2,
    help="Stop a full cycle this many control steps before floating-point endpoint rollover.",
)
p.add_argument(
    "--arm_track_s",
    type=float,
    default=0.0,
    help="Test-only station duration; 0 runs the required complete A->B->A cycle.",
)
p.add_argument("--curriculum_stage", type=int, default=4, choices=[1, 2, 3, 4])
p.add_argument(
    "--station_shapes",
    "--trajectory_shapes",
    "--trajectory_shape",
    dest="station_shapes",
    nargs="+",
    default=["line,arc,sine,circle,square"],
    help=(
        "Shapes assigned to stations in repeating order. Accepts comma-separated "
        "values or space-separated values, for example: circle,line or circle line."
    ),
)
p.add_argument("--seed", type=int, default=17)
p.add_argument("--realtime", action="store_true")
p.add_argument(
    "--exit_on_complete",
    action="store_true",
    help="Close Isaac Sim when the requested stations complete.",
)
p.add_argument(
    "--plot_update_steps",
    type=int,
    default=30,
    help="Refresh the two live plots every N control steps.",
)
p.add_argument(
    "--disable_live_plots",
    action="store_true",
    help="Disable the EE and truck live plots. Headless mode always disables them.",
)
p.add_argument(
    "--disable_viewport_trajectory",
    action="store_true",
    help="Disable the cyan reference trajectory in the Isaac viewport.",
)
p.add_argument(
    "--viewport_line_sample_steps",
    type=int,
    default=3,
    help="Reference-overlay refresh cadence in policy steps.",
)
AppLauncher.add_app_launcher_args(p)
shape_args_provided = any(
    arg in {"--station_shapes", "--trajectory_shapes", "--trajectory_shape"}
    or arg.startswith("--station_shapes=")
    or arg.startswith("--trajectory_shapes=")
    or arg.startswith("--trajectory_shape=")
    for arg in sys.argv[1:]
)
stations_arg_provided = any(
    arg == "--stations" or arg.startswith("--stations=") for arg in sys.argv[1:]
)
a, unknown = p.parse_known_args()
sys.argv = [sys.argv[0], *unknown]
if a.stations < 1:
    p.error("--stations must be at least 1")
if a.station_spacing_m <= 0.0 or a.drive_speed_mps <= 0.0 or a.path_speed_mps <= 0.0:
    p.error("station spacing, drive speed, and path speed must be positive")
if min(
    a.drive_accel_s,
    a.transport_settle_s,
    a.transport_path_tolerance_m,
    a.transport_max_arm_joint_error,
    a.transport_max_root_z_excursion_m,
    a.transport_max_tool_z_excursion_m,
    a.transport_max_tool_relative_excursion_m,
) <= 0.0:
    p.error("transport timing and stress tolerances must be positive")
if a.transport_min_clearance_m < 0.0:
    p.error("--transport_min_clearance_m must be non-negative")
if min(a.transport_arm_stiffness_scale, a.transport_arm_damping_scale) < 1.0:
    p.error("transport arm gain scales must be at least 1.0")
if a.plot_update_steps < 1:
    p.error("--plot_update_steps must be at least 1")
if a.viewport_line_sample_steps < 1:
    p.error("--viewport_line_sample_steps must be at least 1")
station_shapes = tuple(
    part.strip()
    for token in a.station_shapes
    for part in str(token).split(",")
    if part.strip()
)
valid_shapes = {"line", "arc", "sine", "circle", "square"}
if not station_shapes or set(station_shapes) - valid_shapes:
    p.error(
        "--station_shapes/--trajectory_shapes must contain only "
        f"{sorted(valid_shapes)}"
    )
if shape_args_provided and not stations_arg_provided:
    a.stations = len(station_shapes)

launcher = AppLauncher(a)
simulation_app = launcher.app

import gymnasium as gym
import torch
from isaaclab.utils.math import quat_apply
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

import PPO1_ID_FD_PhysX.tasks  # noqa: F401,E402
from PPO1_ID_FD_PhysX.policy.env_cfg import apply_residual_curriculum_stage
from PPO1_ID_FD_PhysX.policy_4cyl.env_cfg import (
    FourCylinderResidualPolicyPPORunnerCfg,
)
from isaaclab_tasks.utils import parse_env_cfg

if not bool(a.headless):
    import carb
    import omni.debugdraw as omni_debug_draw
else:
    carb = None
    omni_debug_draw = None


class Mode(enum.Enum):
    ARM_TRACK = "ARM_TRACK"
    ARM_STOW = "ARM_STOW"
    DRIVE = "DRIVE"
    PARK_SETTLE = "PARK_SETTLE"
    COMPLETE = "COMPLETE"


class ViewportTrajectoryLines:
    """Draw only the current world-frame reference trajectory in cyan."""

    REFERENCE_COLOR = 0xFF00FFFF

    def __init__(self, *, sample_steps: int) -> None:
        if omni_debug_draw is None:
            raise RuntimeError("viewport trajectory lines require the Isaac GUI")
        self._draw = omni_debug_draw.get_debug_draw_interface()
        self._sample_steps = max(int(sample_steps), 1)
        self._step = 0
        self._reference_world: list[tuple[float, float, float]] = []

    @staticmethod
    def _point(
        position: tuple[float, float, float], *, color: int, width: float
    ):
        point = omni_debug_draw.SimplexPoint()
        point.position = carb.Float3(*position)
        point.color = int(color)
        point.width = float(width)
        return point

    def _redraw(self) -> None:
        segments = []
        for start, end in zip(
            self._reference_world[:-1], self._reference_world[1:], strict=True
        ):
            segments.append(
                self._point(start, color=self.REFERENCE_COLOR, width=4.0)
            )
            segments.append(
                self._point(end, color=self.REFERENCE_COLOR, width=4.0)
            )
        if segments:
            self._draw.draw_lines(segments)

    @torch.no_grad()
    def begin_station(self, raw) -> None:
        base = _base_state(raw)
        base_pos = base[0, :3]
        base_quat = base[0, 3:7]
        local = torch.as_tensor(raw._goals, dtype=torch.float32, device=raw.device)
        world = base_pos.reshape(1, 3) + quat_apply(
            base_quat.reshape(1, 4).expand(local.shape[0], -1), local
        )
        self._reference_world = [
            tuple(float(value) for value in row)
            for row in world.detach().cpu().tolist()
        ]
        self._step = 0
        self._redraw()

    @torch.no_grad()
    def sample_tool(self, raw) -> None:
        # Debug-draw lines are immediate-mode and must be resubmitted. The
        # measured tool is intentionally never drawn in the viewport.
        self._step += 1
        self._redraw()

    def close(self) -> None:
        self._reference_world.clear()


class LiveWorkflowPlots:
    """Exactly two live axes: tool tracking and straight truck transport."""

    def __init__(
        self,
        route_origin: torch.Tensor,
        route_forward: torch.Tensor,
        *,
        station_count: int,
        station_spacing_m: float,
        update_steps: int,
    ) -> None:
        import matplotlib.pyplot as plt

        self._plt = plt
        self._update_steps = max(1, int(update_steps))
        self._sample_count = 0
        self._enabled = True
        self._arm_tracking = False
        self._ee_reference: list[list[float]] = []
        self._ee_measured: list[list[float]] = []
        self._car_measured: list[list[float]] = []

        plt.ion()
        self.figure = plt.figure(figsize=(13.5, 6.2))
        try:
            self.figure.canvas.manager.set_window_title("MULAG live tracking")
        except AttributeError:
            pass
        self.ee_axis = self.figure.add_subplot(1, 2, 1)
        self.car_axis = self.figure.add_subplot(1, 2, 2)

        (self.ee_reference_line,) = self.ee_axis.plot(
            [], [], "--", color="tab:blue", linewidth=2.0, label="reference"
        )
        (self.ee_measured_line,) = self.ee_axis.plot(
            [], [], color="tab:orange", linewidth=1.8, label="measured"
        )
        self.ee_axis.set_xlabel("truck X [m]")
        self.ee_axis.set_ylabel("truck Y [m]")
        self.ee_axis.set_aspect("equal", adjustable="box")
        self.ee_axis.set_title("Mower-head tracking — waiting for trajectory")
        self.ee_axis.legend(loc="upper right")
        self.ee_axis.grid(True)

        origin = route_origin.detach().cpu().tolist()
        forward = route_forward.detach().cpu().tolist()
        station_x = [
            origin[0] + index * float(station_spacing_m) * forward[0]
            for index in range(int(station_count))
        ]
        station_y = [
            origin[1] + index * float(station_spacing_m) * forward[1]
            for index in range(int(station_count))
        ]
        self.car_axis.plot(
            station_x,
            station_y,
            "--o",
            color="tab:blue",
            linewidth=2.0,
            markersize=5,
            label="reference route/stations",
        )
        (self.car_measured_line,) = self.car_axis.plot(
            [], [], color="tab:orange", linewidth=1.8, label="measured truck"
        )
        route_span = max(
            max(station_x) - min(station_x),
            max(station_y) - min(station_y),
            1.0,
        )
        margin = max(0.35, 0.08 * route_span)
        self.car_axis.set_xlim(min(station_x) - margin, max(station_x) + margin)
        self.car_axis.set_ylim(min(station_y) - margin, max(station_y) + margin)
        self.car_axis.set_aspect("equal", adjustable="box")
        self.car_axis.set_xlabel("world X [m]")
        self.car_axis.set_ylabel("world Y [m]")
        self.car_axis.set_title("Truck route tracking")
        self.car_axis.legend(loc="best")
        self.car_axis.grid(True)
        self.figure.tight_layout()
        plt.show(block=False)
        self._draw(force=True)

    def begin_station(self, station: int, shape: str) -> None:
        """Show only the current station on the single EE axis."""
        self._arm_tracking = True
        self._ee_reference.clear()
        self._ee_measured.clear()
        self.ee_reference_line.set_data([], [])
        self.ee_measured_line.set_data([], [])
        self.ee_axis.set_title(f"Mower-head X–Y tracking — station {station}, {shape}")
        self._draw(force=True)

    def wait_for_trajectory(self) -> None:
        """Suppress mower-head data until physical deployment has completed."""
        self._arm_tracking = False
        self._ee_reference.clear()
        self._ee_measured.clear()
        self.ee_reference_line.set_data([], [])
        self.ee_measured_line.set_data([], [])
        self.ee_axis.set_title("Mower-head tracking — waiting for trajectory")
        self._draw(force=True)

    def _append_car(self, raw) -> None:
        xy = _base_state(raw)[0, :2].detach().cpu().tolist()
        self._car_measured.append([float(xy[0]), float(xy[1])])

    def sample_car(self, raw, *, force: bool = False) -> None:
        if not self._enabled:
            return
        self._sample_count += 1
        if not force and self._sample_count % self._update_steps != 0:
            return
        self._append_car(raw)
        self._draw(force=True)

    def sample_arm(self, raw) -> None:
        if not self._enabled or not self._arm_tracking:
            return
        self._sample_count += 1
        if self._sample_count % self._update_steps != 0:
            return
        if bool((raw._preposition_counter > 0).any()) or bool(
            (raw._settle_counter > 0).any()
        ):
            return
        reference = raw._original_reference_tool[0].detach().cpu().tolist()
        measured = raw._compute_ee_pos_local()[0].detach().cpu().tolist()
        self._ee_reference.append([float(value) for value in reference])
        self._ee_measured.append([float(value) for value in measured])
        self._append_car(raw)
        self._draw(force=True)

    def _draw(self, *, force: bool = False) -> None:
        if not self._enabled:
            return
        if not force and self._sample_count % self._update_steps != 0:
            return
        if not self._plt.fignum_exists(self.figure.number):
            self._enabled = False
            return

        if self._ee_reference:
            ref = self._ee_reference
            measured = self._ee_measured
            self.ee_reference_line.set_data(
                [row[0] for row in ref], [row[1] for row in ref]
            )
            self.ee_measured_line.set_data(
                [row[0] for row in measured], [row[1] for row in measured]
            )
            all_points = ref + measured
            limits = []
            for axis in range(2):
                low = min(row[axis] for row in all_points)
                high = max(row[axis] for row in all_points)
                margin = max(0.08, 0.08 * max(high - low, 0.1))
                limits.append((low - margin, high + margin))
            self.ee_axis.set_xlim(*limits[0])
            self.ee_axis.set_ylim(*limits[1])

        if self._car_measured:
            self.car_measured_line.set_data(
                [row[0] for row in self._car_measured],
                [row[1] for row in self._car_measured],
            )
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()

    def pump(self) -> None:
        """Keep the plot window responsive after the workflow completes."""
        self._draw(force=True)

    def close(self) -> None:
        if self._enabled and self._plt.fignum_exists(self.figure.number):
            self._plt.close(self.figure)
        self._enabled = False


def _announce(mode: Mode, station: int) -> None:
    print(f"[SUPERVISOR] mode={mode.value} station={station}", flush=True)


@torch.no_grad()
def _base_state(raw):
    return raw.robot.data.body_link_state_w[:, raw._truck_base_body_id()]


@torch.no_grad()
def _base_speed(raw) -> float:
    # body_link_state_w layout: position, quaternion, linear velocity, angular velocity.
    # Parking is planar: suspension bounce in world Z is allowed and must not
    # prevent the truck from entering ARM_TRACK.
    return float(torch.linalg.norm(_base_state(raw)[0, 7:9]).item())


@torch.no_grad()
def _heading_error_rad(raw, route_forward: torch.Tensor) -> float:
    base_q = _base_state(raw)[0, 3:7].reshape(1, 4)
    heading = quat_apply(
        base_q,
        torch.tensor([[1.0, 0.0, 0.0]], device=raw.device),
    )[0, :2]
    heading = heading / torch.linalg.norm(heading).clamp(min=1.0e-8)
    cross = route_forward[0] * heading[1] - route_forward[1] * heading[0]
    dot = torch.dot(route_forward, heading).clamp(min=-1.0, max=1.0)
    return float(torch.atan2(cross, dot).item())


@torch.no_grad()
def _latch_station_for_arm_tracking(raw) -> None:
    """Remove transport momentum once, then establish the work-station anchor.

    This mirrors a reset boundary without changing the dynamics during PPO
    tracking.  In particular, it does not freeze the arm or repeatedly rewrite
    articulation velocity after the policy has started.
    """
    root_velocity = raw.robot.data.root_state_w[:, 7:13].detach().clone()
    root_velocity[:, :2] = 0.0
    root_velocity[:, 5] = 0.0
    raw.robot.write_root_velocity_to_sim(root_velocity)
    for joint_ids in (raw._wheel_joint_ids, raw._steering_joint_ids):
        raw.robot.write_joint_velocity_to_sim(
            torch.zeros((raw.num_envs, len(joint_ids)), device=raw.device),
            joint_ids=joint_ids,
        )
    raw.sim.forward()
    raw.scene.update(float(raw.sim.get_physics_dt()))
    raw.park_truck_at_current_pose()


@torch.no_grad()
def _write_prescribed_truck_base_xy(
    raw,
    target_base_xy: torch.Tensor,
    *,
    transport_root_pose: torch.Tensor | None = None,
    transport_base_xy: torch.Tensor | None = None,
) -> None:
    """Place the chassis at target XY using one smooth transport reference.

    During DRIVE, ``transport_root_pose`` is the settled stowed pose and only
    its route-plane translation changes.  This avoids feeding suspension noise
    back into the arm-root orientation.  The no-reference form is reserved for
    one-shot station placement and preserves the current root pose.
    """
    target_base_xy = target_base_xy.reshape(raw.num_envs, 2)
    if (transport_root_pose is None) != (transport_base_xy is None):
        raise ValueError(
            "transport_root_pose and transport_base_xy must be supplied together"
        )
    if transport_root_pose is None:
        base_xy = _base_state(raw)[:, :2]
        root_pose = raw.robot.data.root_state_w[:, :7].detach().clone()
        root_pose[:, :2] += target_base_xy - base_xy
    else:
        root_pose = transport_root_pose.reshape(raw.num_envs, 7).clone()
        root_pose[:, :2] += target_base_xy - transport_base_xy.reshape(
            raw.num_envs, 2
        )
    raw.robot.write_root_pose_to_sim(root_pose)
    root_velocity = torch.zeros((raw.num_envs, 6), device=raw.device)
    raw.robot.write_root_velocity_to_sim(root_velocity)


@torch.no_grad()
def _direct_arm_step(
    raw,
    target_x: torch.Tensor,
    *,
    wheels_parked: bool,
    plots: LiveWorkflowPlots | None = None,
) -> None:
    """Advance one policy-rate interval using the trained plant contract."""
    mask = torch.ones(raw.num_envs, dtype=torch.bool, device=raw.device)
    raw._write_direct_learned_targets(mask, target_x)
    physics_dt = float(raw.sim.get_physics_dt())
    for _ in range(int(raw.cfg.decimation)):
        raw._apply_action()
        if wheels_parked:
            raw._hold_wheels()
        raw.scene.write_data_to_sim()
        raw.sim.step(render=False)
        raw.scene.update(physics_dt)
    if wheels_parked:
        # Match the environment used to train the checkpoint: correct chassis
        # XY once per policy interval, never once per physics substep.
        raw._constrain_truck_xy()
    if not bool(a.headless):
        raw.sim.render()
    if plots is not None:
        plots.sample_car(raw)


@torch.no_grad()
def _stow_arm(
    raw, transport_x: torch.Tensor, plots: LiveWorkflowPlots | None = None
) -> None:
    start_x, _ = raw._refresh_cylinder_state()
    steps = int(getattr(raw.cfg, "preposition_steps", 300))
    for step in range(steps):
        u = float(step + 1) / max(steps, 1)
        blend = 0.5 * (1.0 - math.cos(math.pi * u))
        _direct_arm_step(
            raw,
            start_x + blend * (transport_x - start_x),
            wheels_parked=True,
            plots=plots,
        )
    for _ in range(int(getattr(raw.cfg, "settle_steps", 120))):
        _direct_arm_step(
            raw,
            transport_x,
            wheels_parked=True,
            plots=plots,
        )
    print(
        f"[STOW] complete truck_drift={float(raw._truck_post_constraint_drift_xy.max().item()):.6f}m",
        flush=True,
    )


@torch.no_grad()
def _drive_to_next_station(
    raw,
    transport_x: torch.Tensor,
    route_origin: torch.Tensor,
    route_forward: torch.Tensor,
    station_index: int,
    plots: LiveWorkflowPlots | None = None,
) -> tuple[float, float]:
    """Drive straight with PPO disabled and an actuator-held arm rest pose."""
    transport_base_xy = _base_state(raw)[:, :2].detach().clone()
    transport_root_pose = raw.robot.data.root_state_w[:, :7].detach().clone()
    entry_tool_z = raw._compute_cutter_world_z().detach().clone()
    entry_tool_truck = raw._compute_ee_pos_local().detach().clone()
    entry_root_z = raw.robot.data.root_state_w[:, 2].detach().clone()
    entry_root_quat = raw.robot.data.root_state_w[:, 3:7].detach().clone()
    wheel_ids = raw._wheel_joint_ids
    steering_ids = raw._steering_joint_ids
    arm_ids = raw._arm_joint_ids
    moving_joint_ids = {
        *wheel_ids.detach().cpu().tolist(),
        *steering_ids.detach().cpu().tolist(),
    }
    # The hydraulic asset has ten passive revolute constraint DOFs in addition
    # to the five commanded arm joints. Holding only the commanded joints lets
    # those constraint DOFs articulate visibly during transport.
    transport_chain_ids = torch.tensor(
        [
            joint_id
            for joint_id in range(len(raw.robot.joint_names))
            if joint_id not in moving_joint_ids
        ],
        dtype=torch.long,
        device=raw.device,
    )
    transport_chain_pos = raw.robot.data.joint_pos[
        :, transport_chain_ids
    ].detach().clone()
    arm_rest_pos = raw.robot.data.joint_pos[:, arm_ids].detach().clone()

    def lock_transport_chain() -> None:
        raw.robot.write_joint_position_to_sim(
            transport_chain_pos, joint_ids=transport_chain_ids
        )
        raw.robot.write_joint_velocity_to_sim(
            torch.zeros_like(transport_chain_pos), joint_ids=transport_chain_ids
        )

    raw.release_truck_for_transport()
    start = _base_state(raw)[0, :3].detach().clone()
    forward = route_forward
    left = torch.stack((-forward[1], forward[0]))
    arm_stiffness = raw.robot.data.joint_stiffness[:, arm_ids].detach().clone()
    arm_damping = raw.robot.data.joint_damping[:, arm_ids].detach().clone()
    raw.robot.write_joint_stiffness_to_sim(
        arm_stiffness * float(a.transport_arm_stiffness_scale), joint_ids=arm_ids
    )
    raw.robot.write_joint_damping_to_sim(
        arm_damping * float(a.transport_arm_damping_scale), joint_ids=arm_ids
    )
    target_distance = float(station_index) * float(a.station_spacing_m)
    start_along = float(torch.dot(start[:2] - route_origin, forward).item())
    command_along = start_along
    # A straight relocation must start with exactly straight steering.  Holding
    # the previous slightly deformed steering pose caused tire scrub and shake.
    transport_steering = torch.zeros(
        (raw.num_envs, len(steering_ids)), device=raw.device
    )
    transport_wheels = raw.robot.data.joint_pos[:, wheel_ids].detach().clone()
    physics_dt = float(raw.sim.get_physics_dt())
    # Releasing the parking wheel gains changes the suspension load. Let that
    # transition settle while the chassis is still stationary, then define the
    # rest pose used during actual translation. This removes the old vertical
    # impulse at the beginning of DRIVE.
    initial_base_xy = start[:2].reshape(1, 2)
    prep_steps = max(1, int(math.ceil(float(a.transport_settle_s) / physics_dt)))
    prep_max_tool_z_excursion = 0.0
    prep_max_tool_relative_excursion = 0.0
    prep_max_root_z_excursion = 0.0
    prep_max_root_angle_rad = 0.0
    prep_max_chain_joint_error = 0.0
    for _ in range(prep_steps):
        raw._apply_action()
        raw.robot.set_joint_position_target(arm_rest_pos, joint_ids=arm_ids)
        raw.robot.write_joint_position_to_sim(arm_rest_pos, joint_ids=arm_ids)
        raw.robot.write_joint_velocity_to_sim(
            torch.zeros_like(arm_rest_pos), joint_ids=arm_ids
        )
        lock_transport_chain()
        raw.robot.write_joint_position_to_sim(transport_wheels, joint_ids=wheel_ids)
        raw.robot.write_joint_velocity_to_sim(
            torch.zeros_like(transport_wheels), joint_ids=wheel_ids
        )
        raw.robot.write_joint_position_to_sim(transport_steering, joint_ids=steering_ids)
        raw.robot.write_joint_velocity_to_sim(
            torch.zeros_like(transport_steering), joint_ids=steering_ids
        )
        _write_prescribed_truck_base_xy(
            raw,
            initial_base_xy,
            transport_root_pose=transport_root_pose,
            transport_base_xy=transport_base_xy,
        )
        raw.scene.write_data_to_sim()
        raw.sim.step(render=False)
        raw.scene.update(physics_dt)
        raw.robot.write_joint_position_to_sim(arm_rest_pos, joint_ids=arm_ids)
        raw.robot.write_joint_velocity_to_sim(
            torch.zeros_like(arm_rest_pos), joint_ids=arm_ids
        )
        lock_transport_chain()
        _write_prescribed_truck_base_xy(
            raw,
            initial_base_xy,
            transport_root_pose=transport_root_pose,
            transport_base_xy=transport_base_xy,
        )
        raw.sim.forward()
        raw.scene.update(physics_dt)
        prep_max_tool_z_excursion = max(
            prep_max_tool_z_excursion,
            float(torch.max(torch.abs(raw._compute_cutter_world_z() - entry_tool_z)).item()),
        )
        prep_max_tool_relative_excursion = max(
            prep_max_tool_relative_excursion,
            float(
                torch.linalg.norm(
                    raw._compute_ee_pos_local() - entry_tool_truck, dim=1
                ).max().item()
            ),
        )
        prep_max_root_z_excursion = max(
            prep_max_root_z_excursion,
            float(
                torch.max(
                    torch.abs(raw.robot.data.root_state_w[:, 2] - entry_root_z)
                ).item()
            ),
        )
        root_dot = torch.sum(
            raw.robot.data.root_state_w[:, 3:7] * entry_root_quat, dim=1
        ).abs().clamp(max=1.0)
        prep_max_root_angle_rad = max(
            prep_max_root_angle_rad,
            float((2.0 * torch.acos(root_dot)).max().item()),
        )
        prep_max_chain_joint_error = max(
            prep_max_chain_joint_error,
            float(
                torch.max(
                    torch.abs(
                        raw.robot.data.joint_pos[:, transport_chain_ids]
                        - transport_chain_pos
                    )
                ).item()
            ),
        )
        if not bool(a.headless):
            raw.sim.render()
    print(
        f"[DRIVE PREP] tool_dz={prep_max_tool_z_excursion:.4f}m "
        f"tool_rel={prep_max_tool_relative_excursion:.4f}m "
        f"root_dz={prep_max_root_z_excursion:.4f}m "
        f"root_angle={math.degrees(prep_max_root_angle_rad):.3f}deg "
        f"chain_error={prep_max_chain_joint_error:.6f}",
        flush=True,
    )
    # Measure the complete transition from the already-settled stowed pose;
    # recapturing here used to hide visible motion during DRIVE preparation.
    rest_tool_z = entry_tool_z
    rest_tool_truck = entry_tool_truck
    start = _base_state(raw)[0, :3].detach().clone()
    start_along = float(torch.dot(start[:2] - route_origin, forward).item())
    command_along = start_along
    leg_distance = max(target_distance - start_along, 0.0)
    # A quintic time law has zero velocity and acceleration at both ends.  Its
    # peak derivative is 1.875, so this duration respects drive_speed_mps.
    duration = max(
        1.875 * leg_distance / max(float(a.drive_speed_mps), 1.0e-6),
        2.0 * float(a.drive_accel_s),
        2.0 * float(raw.sim.get_physics_dt()),
    )
    physics_steps = max(2, int(math.ceil(duration / physics_dt)))
    final_along = 0.0
    final_lateral = 0.0
    max_path_error = 0.0
    min_tool_clearance = float("inf")
    max_tool_z_excursion = prep_max_tool_z_excursion
    max_tool_relative_excursion = prep_max_tool_relative_excursion
    max_arm_joint_error = 0.0
    max_chain_joint_error = prep_max_chain_joint_error
    max_root_z_excursion = prep_max_root_z_excursion
    max_heading_error = 0.0
    max_tool_z_step = 0
    mask = torch.ones(raw.num_envs, dtype=torch.bool, device=raw.device)
    raw._write_direct_learned_targets(mask, transport_x)
    for physics_step in range(physics_steps + 1):
        u = min(float(physics_step) / float(physics_steps), 1.0)
        smooth = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
        smooth_rate = 30.0 * u**2 * (1.0 - u) ** 2 / duration
        command_along = start_along + leg_distance * smooth
        speed = leg_distance * smooth_rate

        raw._apply_action()
        # The deterministic DRIVE fixture holds both commanded and passive arm
        # linkage DOFs. It is released before the next arm preposition begins.
        raw.robot.set_joint_position_target(arm_rest_pos, joint_ids=arm_ids)
        raw.robot.write_joint_position_to_sim(arm_rest_pos, joint_ids=arm_ids)
        raw.robot.write_joint_velocity_to_sim(
            torch.zeros_like(arm_rest_pos), joint_ids=arm_ids
        )
        lock_transport_chain()
        # Wheel contact is not a second transport controller.  Hold the wheel
        # and steering states while the root follows the prescribed route.
        raw.robot.write_joint_position_to_sim(transport_wheels, joint_ids=wheel_ids)
        raw.robot.write_joint_velocity_to_sim(
            torch.zeros_like(transport_wheels), joint_ids=wheel_ids
        )
        raw.robot.set_joint_position_target(transport_steering, joint_ids=steering_ids)
        raw.robot.write_joint_position_to_sim(transport_steering, joint_ids=steering_ids)
        raw.robot.write_joint_velocity_to_sim(
            torch.zeros_like(transport_steering), joint_ids=steering_ids
        )

        target_base_xy = (route_origin + command_along * forward).reshape(1, 2)
        _write_prescribed_truck_base_xy(
            raw,
            target_base_xy,
            transport_root_pose=transport_root_pose,
            transport_base_xy=transport_base_xy,
        )

        raw.scene.write_data_to_sim()
        raw.sim.step(render=False)
        raw.scene.update(physics_dt)
        raw.robot.set_joint_position_target(arm_rest_pos, joint_ids=arm_ids)
        raw.robot.write_joint_position_to_sim(arm_rest_pos, joint_ids=arm_ids)
        raw.robot.write_joint_velocity_to_sim(
            torch.zeros_like(arm_rest_pos), joint_ids=arm_ids
        )
        lock_transport_chain()
        # Reject every tire/contact/suspension displacement before rendering.
        _write_prescribed_truck_base_xy(
            raw,
            target_base_xy,
            transport_root_pose=transport_root_pose,
            transport_base_xy=transport_base_xy,
        )
        raw.sim.forward()
        raw.scene.update(physics_dt)
        raw.robot.write_joint_position_to_sim(transport_wheels, joint_ids=wheel_ids)
        raw.robot.write_joint_velocity_to_sim(
            torch.zeros_like(transport_wheels), joint_ids=wheel_ids
        )
        raw.robot.write_joint_position_to_sim(transport_steering, joint_ids=steering_ids)
        raw.robot.write_joint_velocity_to_sim(
            torch.zeros_like(transport_steering), joint_ids=steering_ids
        )
        if not bool(a.headless):
            raw.sim.render()
        if plots is not None and physics_step % int(raw.cfg.decimation) == 0:
            plots.sample_car(raw)

        position = _base_state(raw)[0, :2]
        delta = position - route_origin
        final_along = float(torch.dot(delta, forward).item())
        final_lateral = float(torch.dot(delta, left).item())
        path_error = float(torch.linalg.norm(position - target_base_xy[0]).item())
        max_path_error = max(max_path_error, path_error)
        tool_clearance = float(raw._compute_cutter_world_z().min().item())
        tool_z_excursion = float(
            torch.max(torch.abs(raw._compute_cutter_world_z() - rest_tool_z)).item()
        )
        tool_relative_excursion = float(
            torch.linalg.norm(
                raw._compute_ee_pos_local() - rest_tool_truck, dim=1
            ).max().item()
        )
        arm_joint_error = float(
            torch.max(torch.abs(raw.robot.data.joint_pos[:, arm_ids] - arm_rest_pos)).item()
        )
        chain_joint_error = float(
            torch.max(
                torch.abs(
                    raw.robot.data.joint_pos[:, transport_chain_ids]
                    - transport_chain_pos
                )
            ).item()
        )
        min_tool_clearance = min(min_tool_clearance, tool_clearance)
        if tool_z_excursion > max_tool_z_excursion:
            max_tool_z_excursion = tool_z_excursion
            max_tool_z_step = physics_step
        max_tool_relative_excursion = max(
            max_tool_relative_excursion, tool_relative_excursion
        )
        max_arm_joint_error = max(max_arm_joint_error, arm_joint_error)
        max_chain_joint_error = max(max_chain_joint_error, chain_joint_error)
        root_z_excursion = float(
            torch.max(
                torch.abs(
                    raw.robot.data.root_state_w[:, 2] - transport_root_pose[:, 2]
                )
            ).item()
        )
        max_root_z_excursion = max(max_root_z_excursion, root_z_excursion)
        max_heading_error = max(
            max_heading_error, abs(_heading_error_rad(raw, route_forward))
        )
        if physics_step % max(1, 60 * int(raw.cfg.decimation)) == 0:
            measured_x, _ = raw._refresh_cylinder_state()
            arm_hold_error = float(torch.max(torch.abs(measured_x - transport_x)).item())
            print(
                f"[DRIVE] along={final_along:.3f}/{target_distance:.3f}m "
                f"path_error={path_error:.5f}m command_speed={speed:.3f}m/s "
                f"arm_hold={arm_hold_error:.4f}m clearance={tool_clearance:+.3f}m "
                f"tool_dz={tool_z_excursion:.3f}m "
                f"tool_rel={tool_relative_excursion:.4f}m "
                f"root_dz={root_z_excursion:.4f}m "
                f"heading={math.degrees(_heading_error_rad(raw, route_forward)):+.3f}deg",
                flush=True,
            )

    # Finish at an exact zero-velocity endpoint before engaging the parking
    # fixture.  This avoids the old 0.12 m/s minimum-speed to instant-stop jump.
    final_base_xy = (route_origin + target_distance * forward).reshape(1, 2)
    _write_prescribed_truck_base_xy(
        raw,
        final_base_xy,
        transport_root_pose=transport_root_pose,
        transport_base_xy=transport_base_xy,
    )
    raw.robot.write_joint_stiffness_to_sim(arm_stiffness, joint_ids=arm_ids)
    raw.robot.write_joint_damping_to_sim(arm_damping, joint_ids=arm_ids)

    # Remove wheel motion before the parking fixture is latched.
    raw.robot.write_joint_velocity_to_sim(
        torch.zeros((raw.num_envs, len(wheel_ids)), device=raw.device), joint_ids=wheel_ids
    )
    raw.robot.set_joint_velocity_target(
        torch.zeros((raw.num_envs, len(wheel_ids)), device=raw.device), joint_ids=wheel_ids
    )
    if (
        max_path_error > float(a.transport_path_tolerance_m)
        or min_tool_clearance < float(a.transport_min_clearance_m)
        or max_arm_joint_error > float(a.transport_max_arm_joint_error)
        or max_chain_joint_error > float(a.transport_max_arm_joint_error)
        or max_root_z_excursion > float(a.transport_max_root_z_excursion_m)
        or max_tool_z_excursion > float(a.transport_max_tool_z_excursion_m)
        or max_tool_relative_excursion
        > float(a.transport_max_tool_relative_excursion_m)
    ):
        raise RuntimeError(
            "transport stress gate failed: "
            f"path_error={max_path_error:.6f}m/"
            f"{a.transport_path_tolerance_m:.6f}m "
            f"min_clearance={min_tool_clearance:.6f}m/"
            f"{a.transport_min_clearance_m:.6f}m "
            f"tool_z_excursion={max_tool_z_excursion:.6f}/"
            f"{a.transport_max_tool_z_excursion_m:.6f}m "
            f"tool_relative_excursion={max_tool_relative_excursion:.6f}/"
            f"{a.transport_max_tool_relative_excursion_m:.6f}m "
            f"at_step={max_tool_z_step} arm_joint_error={max_arm_joint_error:.6f}/"
            f"{a.transport_max_arm_joint_error:.6f} "
            f"chain_joint_error={max_chain_joint_error:.6f}/"
            f"{a.transport_max_arm_joint_error:.6f} "
            f"root_z_excursion={max_root_z_excursion:.6f}/"
            f"{a.transport_max_root_z_excursion_m:.6f}"
        )
    print(
        f"[DRIVE GATE] path_error={max_path_error:.6f}m "
        f"min_clearance={min_tool_clearance:.4f}m "
        f"tool_dz={max_tool_z_excursion:.4f}m "
        f"tool_rel={max_tool_relative_excursion:.4f}m "
        f"arm_joint_error={max_arm_joint_error:.5f} "
        f"chain_joint_error={max_chain_joint_error:.5f} "
        f"root_dz={max_root_z_excursion:.4f}m "
        f"heading={math.degrees(max_heading_error):.3f}deg "
        f"at_step={max_tool_z_step} PASS",
        flush=True,
    )
    return target_distance, 0.0


@torch.no_grad()
def _park_and_settle(
    raw,
    transport_x: torch.Tensor,
    target_base_xy: torch.Tensor,
    plots: LiveWorkflowPlots | None = None,
) -> None:
    # One-shot service-brake transition. Preserve suspension/vertical velocity,
    # but remove planar momentum before latching the parking fixture.
    root_velocity = raw.robot.data.root_state_w[:, 7:13].detach().clone()
    root_velocity[:, :2] = 0.0
    root_velocity[:, 5] = 0.0
    raw.robot.write_root_velocity_to_sim(root_velocity)
    target_base_xy = target_base_xy.reshape(raw.num_envs, 2)
    _write_prescribed_truck_base_xy(raw, target_base_xy)
    raw.sim.forward()
    raw.scene.update(float(raw.sim.get_physics_dt()))
    raw.park_truck_at_current_pose()
    # PPO2 passes the route waypoint explicitly into PARK_SETTLE.  Keep that
    # exact chassis waypoint as the station anchor instead of a stale cached
    # body state from the final transport frame.
    raw._truck_anchor_xy[:] = target_base_xy
    raw._truck_drift_xy.zero_()
    raw._max_truck_drift.zero_()
    raw._truck_post_constraint_drift_xy.zero_()
    raw._max_truck_post_constraint_drift.zero_()
    required = max(1, int(float(a.settle_s) / raw.step_dt))
    for _ in range(required):
        _direct_arm_step(
            raw,
            transport_x,
            wheels_parked=True,
            plots=plots,
        )


def main() -> None:
    task = "RRLAB-Mulag-PPO1-A-Policy-V3-PLAY-v0"
    cfg = parse_env_cfg(task, device=a.device, num_envs=1)
    cfg.seed = int(a.seed)
    apply_residual_curriculum_stage(cfg, int(a.curriculum_stage))
    cfg.action_space = 4
    cfg.observation_space = 42
    cfg.cylinder4_residual_scale_m = float(a.cylinder4_residual_scale_m)
    cfg.trajectory_library_path = str(
        Path(a.trajectory_library_path).expanduser().resolve()
    )
    cfg.path_speed_mps = float(a.path_speed_mps)
    cfg.trajectory_shape = station_shapes[0]
    cfg.randomize_trajectory_shape = False
    cfg.randomize_start_phase = False
    cfg.force_zero_policy = False
    cfg.episode_length_s = 200.0
    cfg.preposition_steps = int(a.preposition_steps)
    cfg.settle_steps = int(a.arm_settle_steps)

    gym_env = gym.make(task, cfg=cfg)
    raw = gym_env.unwrapped
    if not bool(a.headless):
        # ID-FD arm reactions are larger than ROM reactions. DirectRLEnv would
        # otherwise render the final physics substep before the parked chassis
        # pose correction. Render explicitly after each corrected control step.
        raw.cfg.sim.render_interval = 1_000_000_000
        raw.sim.render()
    agent_cfg = FourCylinderResidualPolicyPPORunnerCfg()
    agent_cfg.device = a.device
    env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    checkpoint = str(Path(a.checkpoint).expanduser().resolve())
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=raw.device)
    obs = env.get_observations()

    transport_x = raw._preposition_start_x.detach().clone()
    settled_base = _base_state(raw)[0].detach().clone()
    route_origin = settled_base[:2]
    route_q = settled_base[3:7].reshape(1, 4)
    raw.park_truck_at_current_pose()
    route_forward = quat_apply(
        route_q,
        torch.tensor([[1.0, 0.0, 0.0]], device=raw.device),
    )[0, :2]
    route_forward = route_forward / torch.linalg.norm(route_forward).clamp(min=1.0e-8)
    plots: LiveWorkflowPlots | None = None
    if not bool(a.headless) and not bool(a.disable_live_plots):
        plots = LiveWorkflowPlots(
            route_origin,
            route_forward,
            station_count=int(a.stations),
            station_spacing_m=float(a.station_spacing_m),
            update_steps=int(a.plot_update_steps),
        )
        plots.sample_car(raw, force=True)
        print("[PLOTS] live EE and truck plots enabled (exactly two axes)", flush=True)
    viewport_lines: ViewportTrajectoryLines | None = None
    if not bool(a.headless) and not bool(a.disable_viewport_trajectory):
        viewport_lines = ViewportTrajectoryLines(
            sample_steps=int(a.viewport_line_sample_steps)
        )
        print("[ARM] viewport line enabled: cyan=reference", flush=True)
    print(f"[SUPERVISOR] loaded {checkpoint}", flush=True)
    print(
        f"[SUPERVISOR] route origin=({float(route_origin[0]):.3f}, "
        f"{float(route_origin[1]):.3f}) source=settled_station_zero_chassis",
        flush=True,
    )
    print("[SUPERVISOR] interlock active: policy=ARM_TRACK only, wheels=DRIVE only", flush=True)

    try:
        for station in range(int(a.stations)):
            station_shape = station_shapes[station % len(station_shapes)]
            if station > 0:
                if plots is not None:
                    plots.wait_for_trajectory()
                _announce(Mode.ARM_STOW, station)
                _stow_arm(raw, transport_x, plots=plots)
                _announce(Mode.DRIVE, station)
                target_xy = (
                    route_origin
                    + float(station) * float(a.station_spacing_m) * route_forward
                )
                along, lateral = _drive_to_next_station(
                    raw,
                    transport_x,
                    route_origin,
                    route_forward,
                    station,
                    plots=plots,
                )
                print(f"[DRIVE] reached along={along:.3f}m lateral={lateral:+.3f}m", flush=True)
                _announce(Mode.PARK_SETTLE, station)
                _park_and_settle(
                    raw,
                    transport_x,
                    target_xy,
                    plots=plots,
                )
                raw.set_trajectory_shape(station_shape)
                raw.prepare_arm_cycle_at_station()
                _latch_station_for_arm_tracking(raw)
                obs = env.get_observations()
                if not bool(a.headless):
                    raw.sim.render()

            _announce(Mode.ARM_TRACK, station)
            if plots is not None:
                plots.begin_station(station, station_shape)
            if viewport_lines is not None:
                viewport_lines.begin_station(raw)
            print(
                f"[ARM] station={station} shape={station_shape} "
                f"speed={raw._effective_path_speed_mps():.3f}m/s",
                flush=True,
            )
            cycle_target_s = (
                float(raw._complete_cycle_time)
                if float(a.arm_track_s) <= 0.0
                else min(float(a.arm_track_s), float(raw._complete_cycle_time))
            )
            if float(a.arm_track_s) <= 0.0:
                cycle_target_s = max(
                    raw.step_dt,
                    cycle_target_s
                    - max(0, int(a.cycle_completion_guard_steps)) * raw.step_dt,
                )
            arm_step = 0
            arm_errors: list[torch.Tensor] = []
            arm_tools: list[torch.Tensor] = []
            while float(raw._scored_elapsed[0].item()) < cycle_target_s:
                if not simulation_app.is_running():
                    return
                started = time.perf_counter()
                # The environment mutates these buffers again during the
                # following deterministic stow/drive transition.  no_grad is
                # required here: inference_mode would permanently mark newly
                # assigned controller tensors as immutable inference tensors.
                with torch.no_grad():
                    action = policy(obs)
                    elapsed_before = float(raw._scored_elapsed[0].item())
                    obs, _, dones, extras = env.step(action)
                if viewport_lines is not None:
                    viewport_lines.sample_tool(raw)
                if not bool(a.headless):
                    raw.sim.render()
                if plots is not None:
                    plots.sample_arm(raw)
                if bool(dones.any()):
                    log = extras.get("log", {}) if isinstance(extras, dict) else {}
                    termination = {}
                    for key in (
                        "termination/limit",
                        "termination/nonfinite",
                        "termination/truck_drift",
                        "termination/head_down",
                        "termination/clearance",
                        "termination/timeout",
                    ):
                        value = log.get(key, 0.0)
                        if isinstance(value, torch.Tensor):
                            value = float(value.detach().max().item())
                        else:
                            value = float(value)
                        if value > 0.0:
                            termination[key.removeprefix("termination/")] = value
                    measured = {}
                    for key, value in log.items():
                        if not key.startswith("termination_value/"):
                            continue
                        if isinstance(value, torch.Tensor):
                            value = float(value.detach().max().item())
                        else:
                            value = float(value)
                        measured[key.removeprefix("termination_value/")] = value
                    raise RuntimeError(
                        "arm policy terminated before completing its station trajectory: "
                        f"station={station} elapsed={elapsed_before:.3f}/"
                        f"{cycle_target_s:.3f}s causes={termination or {'unknown': 1.0}} "
                        f"measured={measured}"
                    )
                tool_position = raw._compute_ee_pos_local()[0].detach().clone()
                arm_tools.append(tool_position)
                arm_errors.append(torch.linalg.norm(
                    raw._original_reference_tool[0] - tool_position
                ).detach())
                if arm_step % 300 == 0:
                    tool_error = float(torch.linalg.norm(
                        raw._original_reference_tool[0] - raw._compute_ee_pos_local()[0]
                    ).item())
                    print(
                        f"[ARM] station={station} elapsed="
                        f"{float(raw._scored_elapsed[0].item()):.2f}/"
                        f"{cycle_target_s:.2f}s error={tool_error:.4f}m "
                        f"clearance={float(raw._compute_cutter_world_z()[0].item()):+.4f}m "
                        f"head={float(raw._measure_head_down_alignment()[0].item()):.4f} "
                        f"drift={float(raw._truck_drift_xy[0].item()):.4f}m "
                        f"heading={math.degrees(_heading_error_rad(raw, route_forward)):+.3f}deg "
                        f"base_z={float(_base_state(raw)[0, 2].item()):.4f}m",
                        flush=True,
                    )
                arm_step += 1
                if a.realtime:
                    time.sleep(max(0.0, raw.step_dt - (time.perf_counter() - started)))
            error = float(torch.linalg.norm(
                raw._original_reference_tool[0] - raw._compute_ee_pos_local()[0]
            ).item())
            error_history = torch.stack(arm_errors)
            tool_history = torch.stack(arm_tools)
            velocity_history = torch.diff(tool_history, dim=0) / float(raw.step_dt)
            acceleration_history = torch.diff(
                velocity_history, dim=0
            ) / float(raw.step_dt)
            p95_acceleration = float(torch.quantile(
                torch.linalg.norm(acceleration_history, dim=-1), 0.95
            ).item()) if acceleration_history.shape[0] else float("inf")
            print(
                f"[ARM] station={station} cycle complete error={error:.4f}m "
                f"mean={float(error_history.mean().item()):.4f}m "
                f"p95={float(torch.quantile(error_history, 0.95).item()):.4f}m "
                f"acc95={p95_acceleration:.2f}m/s^2 "
                f"truck_speed={_base_speed(raw):.4f}m/s",
                flush=True,
            )

        _announce(Mode.COMPLETE, int(a.stations) - 1)
        if bool(a.exit_on_complete):
            return
        while simulation_app.is_running() and not bool(a.headless):
            if plots is not None:
                plots.pump()
            time.sleep(0.05)
    finally:
        if plots is not None:
            plots.close()
        if viewport_lines is not None:
            viewport_lines.close()
        env.close()


try:
    main()
except BaseException:
    traceback.print_exc()
    raise
finally:
    simulation_app.close()
