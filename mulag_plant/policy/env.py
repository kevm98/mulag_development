"""PPO Policy environment for the MULAG hydraulic mower arm.

The policy adds small residual corrections (cylinders 1-3 only) to the
deterministic head-down A->B->A arm controller. Cylinder 4 remains at the
head-down teacher value, and the truck remains completely stationary.

Zero policy action must reproduce the deterministic baseline.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from isaaclab.utils.math import quat_apply

from mulag_plant.mdp.actions import sanitize_cylinder_action
from mulag_plant.tasks.cylinder_env import PPO2ROMPhysXEnv
from mulag_plant.dls_solver.cylinder import CylinderDLSConfig, CylinderDLSController
from mulag_plant.dls_solver.ee_cascade import EECascadeController
from mulag_plant.trajectories.time_scaling import cruise_profile
from verge_task.trajectory_2d import Trajectory2D, from_xy_samples
from mulag_plant.policy.observations import (
    build_residual_observation,
    residual_observation_dim,
)
from mulag_plant.policy.actions import apply_residual_action
from mulag_plant.policy.rewards import (
    ResidualRewardTerms,
    compute_residual_reward,
)

ROOT = Path(__file__).resolve().parents[2]


class ResidualPolicyEnv(PPO2ROMPhysXEnv):
    """PPO Policy environment.

    Overrides observation, reward, termination, reset, and pre-physics step
    to implement the Policy contract while reusing the validated
    command-ROM lower-level controller pipeline.
    """

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self._residual_action_dim = int(getattr(cfg, "action_space", 3))
        if bool(getattr(cfg, "sole_head_joint_action", False)):
            # PPO3 appends a non-hydraulic Messerkopf_joint channel to the
            # policy action. It splits that column off before anything here
            # runs, so this cylinder pipeline still sees exactly 4 columns.
            # PPO2 never sets this flag, so its behaviour is unchanged.
            self._residual_action_dim -= 1
        if self._residual_action_dim not in (3, 4):
            raise ValueError(
                f"Policy requires 3 or 4 actions, got {self._residual_action_dim}"
            )

        # --- Trajectory data ---
        self._arc_path = Path(str(getattr(cfg, "arc_path", ""))).expanduser().resolve()
        self._trajectory_library_path = Path(
            str(getattr(cfg, "trajectory_library_path", ""))
        ).expanduser().resolve()
        self._trajectory_library: dict[str, dict] = {}
        self._trajectory_name = str(getattr(cfg, "trajectory_shape", "legacy_arc"))
        self._trajectory_speed_factor = 1.0
        self._trajectory_truck_motion_mode = "parked"
        self._trajectory_truck_speed_mps = 0.0
        self._trajectory_truck_motion_active = False
        self._trajectory_geometry: Trajectory2D | None = None
        self._trajectory_geometry_xy: torch.Tensor | None = None
        self._trajectory_geometry_tangent: torch.Tensor | None = None
        self._trajectory_geometry_s: torch.Tensor | None = None
        self._load_trajectory()

        # --- Residual action state ---
        self._residual_raw = torch.zeros(
            (self.num_envs, self._residual_action_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self._residual_filtered = torch.zeros_like(self._residual_raw)
        self._prev_residual_filtered = torch.zeros_like(self._residual_filtered)
        self._baseline_cylinder_goal = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self._baseline_cylinder_velocity = torch.zeros_like(self._baseline_cylinder_goal)
        self._corrected_cylinder_goal = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)

        # --- Tool observation history ---
        self._prev_tool_pos_truck: torch.Tensor | None = None
        self._measured_tool_velocity_truck = torch.zeros((self.num_envs, 3), device=self.device)
        self._previous_measured_tool_velocity_truck = torch.zeros_like(
            self._measured_tool_velocity_truck
        )
        self._measured_tool_acceleration_truck = torch.zeros_like(
            self._measured_tool_velocity_truck
        )

        # --- Reward state ---
        self._previous_error_norm = torch.zeros((self.num_envs,), device=self.device)

        # --- Head-down history ---
        self._prev_head_alignment = torch.zeros((self.num_envs,), device=self.device)
        self._head_alignment_rate = torch.zeros((self.num_envs,), device=self.device)

        # --- Trajectory state ---
        self._refresh_trajectory_timing()

        # Per-environment trajectory state
        self._traj_direction = torch.ones((self.num_envs,), device=self.device)
        self._traj_start_direction = torch.ones((self.num_envs,), device=self.device)
        self._traj_start_s = torch.zeros((self.num_envs,), device=self.device)
        self._traj_elapsed = torch.zeros((self.num_envs,), device=self.device)
        self._scored_elapsed = torch.zeros((self.num_envs,), device=self.device)
        self._traj_phase = torch.zeros((self.num_envs,), device=self.device)
        self._reference_timing_s = torch.zeros((self.num_envs,), device=self.device)
        self._reference_timing_speed = torch.zeros((self.num_envs,), device=self.device)
        self._reference_path_s = torch.zeros((self.num_envs,), device=self.device)
        self._reference_path_speed = torch.zeros((self.num_envs,), device=self.device)
        self._reference_path_accel = torch.zeros((self.num_envs,), device=self.device)

        # --- Preposition counters ---
        preposition_steps = int(getattr(cfg, "preposition_steps", 300))
        settle_steps = int(getattr(cfg, "settle_steps", 120))
        self._preposition_counter = torch.full(
            (self.num_envs,), preposition_steps, device=self.device, dtype=torch.long
        )
        self._settle_counter = torch.full(
            (self.num_envs,), settle_steps, device=self.device, dtype=torch.long
        )
        self._preposition_start_x = torch.zeros(
            (self.num_envs, 4), dtype=torch.float32, device=self.device
        )
        self._preposition_target_x = torch.zeros_like(self._preposition_start_x)
        self._preposition_total_steps = preposition_steps
        self._settle_total_steps = settle_steps

        # --- Truck stationarity ---
        self._truck_anchor_xy = torch.zeros((self.num_envs, 2), device=self.device)
        self._truck_drift_xy = torch.zeros((self.num_envs,), device=self.device)
        self._max_truck_drift = torch.zeros((self.num_envs,), device=self.device)
        self._truck_post_constraint_drift_xy = torch.zeros((self.num_envs,), device=self.device)
        self._max_truck_post_constraint_drift = torch.zeros((self.num_envs,), device=self.device)
        self._wheel_init_pos: torch.Tensor | None = None
        self._steering_init_pos: torch.Tensor | None = None
        self._truck_stationary_initialized = False

        # --- Cylinder DLS controllers ---
        self._cylinder_dls: CylinderDLSController | None = None
        self._cascade: EECascadeController | None = None
        self._setup_dls_controllers()

        # --- Reference tool position (kept invariant for reward) ---
        self._original_reference_tool = torch.zeros((self.num_envs, 3), device=self.device)
        self._original_reference_world_z = torch.zeros((self.num_envs,), device=self.device)

        print(
            f"[Policy] trajectory={self._trajectory_name} "
            f"length={self._arc_length:.3f}m "
            f"one_way={self._one_way_s:.2f}s "
            f"obs_dim={residual_observation_dim(self._residual_action_dim, trajectory_geometry=bool(getattr(cfg, 'trajectory_geometry_observation', False)))} "
            f"action_dim={self._residual_action_dim} "
            f"residual_scales_m={self._residual_scale_vector().detach().cpu().tolist()}",
            flush=True,
        )

    def _residual_scale_vector(self) -> torch.Tensor:
        scales = torch.full(
            (self._residual_action_dim,),
            float(getattr(self.cfg, "residual_scale_m", 0.002)),
            dtype=torch.float32,
            device=self.device,
        )
        if self._residual_action_dim == 4:
            scales[3] = float(getattr(self.cfg, "cylinder4_residual_scale_m", 0.0))
        return scales

    def _residual_filter_vector(self) -> torch.Tensor:
        beta = torch.full(
            (self._residual_action_dim,),
            float(getattr(self.cfg, "action_filter", 0.8)),
            dtype=torch.float32,
            device=self.device,
        )
        if self._residual_action_dim == 4:
            beta[3] = float(getattr(self.cfg, "cylinder4_action_filter", 0.9))
        return beta

    # ------------------------------------------------------------------
    # Trajectory loading
    # ------------------------------------------------------------------

    def _load_trajectory(self) -> None:
        if self._trajectory_library_path.is_file():
            library = json.loads(self._trajectory_library_path.read_text())
            self._trajectory_library = dict(library.get("shapes", {}))
        if self._trajectory_name != "legacy_arc":
            self._select_trajectory(self._trajectory_name)
            return

        payload = json.loads(self._arc_path.read_text())
        teacher = payload["cylinder_teacher"]
        if len(teacher) < 8:
            raise RuntimeError("head-down mowing arc does not contain a dense teacher path")

        theta = np.asarray([row["theta_deg"] for row in teacher], dtype=np.float64)
        goals = np.asarray([row["cutter_truck"] for row in teacher], dtype=np.float64)
        seeds = np.asarray([row["cylinder_cmd"] for row in teacher], dtype=np.float64)
        world_z = np.asarray([row["cutter_world_z"] for row in teacher], dtype=np.float64)
        order = np.argsort(theta)
        self._theta, self._goals, self._seeds, self._world_z = (
            theta[order], goals[order], seeds[order], world_z[order]
        )
        self._teacher_s = np.r_[
            0.0, np.cumsum(np.linalg.norm(np.diff(self._goals, axis=0), axis=1))
        ]
        self._geometry_s = self._teacher_s.copy()
        self._refresh_path_parameter_derivatives()
        self._arc_length = float(self._teacher_s[-1])
        if self._arc_length < 3.0:
            raise RuntimeError(f"Verified ground arc is only {self._arc_length:.3f} m")

    def _select_trajectory(self, name: str) -> None:
        if name == "legacy_arc":
            self._trajectory_name = name
            self._trajectory_speed_factor = 1.0
            self._trajectory_truck_motion_mode = "parked"
            self._trajectory_truck_speed_mps = 0.0
            payload = json.loads(self._arc_path.read_text())
            teacher = payload["cylinder_teacher"]
            theta = np.asarray([row["theta_deg"] for row in teacher], dtype=np.float64)
            order = np.argsort(theta)
            self._theta = theta[order]
            self._goals = np.asarray(
                [row["cutter_truck"] for row in teacher], dtype=np.float64
            )[order]
            self._seeds = np.asarray(
                [row["cylinder_cmd"] for row in teacher], dtype=np.float64
            )[order]
            self._world_z = np.asarray(
                [row["cutter_world_z"] for row in teacher], dtype=np.float64
            )[order]
            self._teacher_s = np.r_[
                0.0, np.cumsum(np.linalg.norm(np.diff(self._goals, axis=0), axis=1))
            ]
            self._geometry_s = self._teacher_s.copy()
        else:
            if name not in self._trajectory_library:
                raise ValueError(
                    f"trajectory {name!r} is unavailable; choose legacy_arc or "
                    f"{tuple(self._trajectory_library)}"
                )
            rows = self._trajectory_library[name]["samples"]
            if len(rows) < 81:
                raise RuntimeError(f"trajectory {name!r} is not densely sampled")
            self._trajectory_name = name
            self._trajectory_speed_factor = float(
                self._trajectory_library[name].get("speed_factor", 1.0)
            )
            execution = dict(self._trajectory_library[name].get("execution", {}))
            self._trajectory_truck_motion_mode = str(
                execution.get("truck_motion_mode", "parked")
            )
            if self._trajectory_truck_motion_mode not in {"parked", "creep"}:
                raise ValueError(
                    f"trajectory {name!r} has unsupported truck_motion_mode="
                    f"{self._trajectory_truck_motion_mode!r}"
                )
            self._trajectory_truck_speed_mps = max(
                0.0, float(execution.get("truck_speed_mps", 0.0))
            )
            self.cfg.enable_scripted_vehicle_motion = (
                self._trajectory_truck_motion_mode == "creep"
            )
            if self.cfg.enable_scripted_vehicle_motion:
                speed_kmh = 3.6 * self._trajectory_truck_speed_mps
                self.cfg.vehicle_speed_min_kmh = speed_kmh
                self.cfg.vehicle_speed_max_kmh = speed_kmh
            else:
                self.cfg.vehicle_speed_min_kmh = 0.0
                self.cfg.vehicle_speed_max_kmh = 0.0
            self._theta = np.arange(len(rows), dtype=np.float64)
            self._goals = np.asarray([row["cutter_truck"] for row in rows], dtype=np.float64)
            self._seeds = np.asarray([row["cylinder_cmd"] for row in rows], dtype=np.float64)
            self._world_z = np.asarray([row["cutter_world_z"] for row in rows], dtype=np.float64)
            self._geometry_s = np.asarray(
                [row["path_s_m"] for row in rows], dtype=np.float64
            )
            self._teacher_s = np.asarray(
                [row.get("timing_s_m", row["path_s_m"]) for row in rows],
                dtype=np.float64,
            )
        if (
            self._teacher_s[0] != 0.0
            or self._geometry_s[0] != 0.0
            or np.any(np.diff(self._teacher_s) <= 0.0)
            or np.any(np.diff(self._geometry_s) <= 0.0)
        ):
            raise ValueError(f"trajectory {name!r} path coordinates must increase from zero")
        self._refresh_path_parameter_derivatives()
        self._arc_length = float(self._teacher_s[-1])
        self._geometry_length = float(self._geometry_s[-1])
        if bool(getattr(self.cfg, "trajectory_geometry_observation", False)):
            closed = bool(self._trajectory_library.get(name, {}).get(
                "closed", name in ("circle", "rounded_square")
            ))
            self._trajectory_geometry = from_xy_samples(
                self._goals[:, :2], name=name, closed=closed
            )
            self._trajectory_geometry_xy = torch.as_tensor(
                self._trajectory_geometry.xy, dtype=torch.float32, device=self.device
            )
            self._trajectory_geometry_tangent = torch.as_tensor(
                self._trajectory_geometry.tangent_xy, dtype=torch.float32, device=self.device
            )
            self._trajectory_geometry_s = torch.as_tensor(
                self._trajectory_geometry.s_m, dtype=torch.float32, device=self.device
            )
        # Reachability is the authority for plant-backed references.  The old
        # arbitrary 2.5 m excitation floor forced the arc and true square back
        # outside the directly verified workspace.
        if self._arc_length < 1.8:
            raise RuntimeError(f"trajectory {name!r} is only {self._arc_length:.3f} m")

    def _refresh_trajectory_timing(self) -> None:
        zero_t = torch.zeros(1, device=self.device)
        _, _, _, self._one_way_s = cruise_profile(
            zero_t,
            length_m=self._arc_length,
            cruise_speed_mps=self._effective_path_speed_mps(),
            accel_time_s=float(getattr(self.cfg, "accel_time_s", 1.0)),
            decel_time_s=float(getattr(self.cfg, "accel_time_s", 1.0)),
        )
        self._complete_cycle_time = 2.0 * self._one_way_s

    def set_trajectory_shape(self, name: str) -> None:
        """Select a library shape between resets or parked work stations."""
        self._select_trajectory(str(name))
        self._refresh_trajectory_timing()
        print(
            f"[Policy] selected trajectory={self._trajectory_name} "
            f"height={float(np.median(self._goals[:, 2])):.3f}m "
            f"speed={self._effective_path_speed_mps():.3f}m/s "
            f"length={self._arc_length:.3f}m cycle={self._complete_cycle_time:.2f}s "
            f"truck={self._trajectory_truck_motion_mode} "
            f"truck_speed={self._trajectory_truck_speed_mps:.3f}m/s",
            flush=True,
        )

    @torch.no_grad()
    def _nearest_trajectory_geometry(self, tool_pos_truck: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return nearest-point delta and signed normal error in truck XY.

        The query is segment-based rather than nearest sampled vertex, so the
        result stays well behaved when a rounded corner is sampled coarsely.
        The old line task does not call this method and therefore keeps its
        exact observation/reward contract.
        """
        if self._trajectory_geometry_xy is None or self._trajectory_geometry_tangent is None:
            zeros = torch.zeros((self.num_envs, 2), device=self.device)
            return zeros, torch.zeros(self.num_envs, device=self.device)
        points = tool_pos_truck[:, :2]
        start = self._trajectory_geometry_xy
        if self._trajectory_geometry.closed:
            end = torch.roll(start, shifts=-1, dims=0)
            tangent = end - start
        else:
            start = start[:-1]
            end = self._trajectory_geometry_xy[1:]
            tangent = end - start
        length_sq = torch.sum(tangent * tangent, dim=-1).clamp_min(1.0e-10)
        delta = points[:, None, :] - start[None, :, :]
        alpha = (delta * tangent[None, :, :]).sum(dim=-1) / length_sq[None, :]
        alpha = alpha.clamp(0.0, 1.0)
        candidate = start[None, :, :] + alpha[..., None] * tangent[None, :, :]
        distance_sq = torch.sum((points[:, None, :] - candidate) ** 2, dim=-1)
        edge_index = torch.argmin(distance_sq, dim=-1)
        row = torch.arange(points.shape[0], device=self.device)
        nearest = candidate[row, edge_index]
        edge_tangent = tangent[edge_index] / torch.linalg.norm(
            tangent[edge_index], dim=-1, keepdim=True
        ).clamp_min(1.0e-8)
        normal = torch.stack((-edge_tangent[:, 1], edge_tangent[:, 0]), dim=-1)
        delta_to_path = nearest - points
        signed_normal = ((points - nearest) * normal).sum(dim=-1)
        return delta_to_path, signed_normal

    def _effective_path_speed_mps(self) -> float:
        return float(getattr(self.cfg, "path_speed_mps", 0.13)) * float(
            self._trajectory_speed_factor
        )

    def _interpolate_trajectory(self, distance_m: float):
        s = float(np.clip(distance_m, 0.0, self._arc_length))
        goal = np.asarray(
            [np.interp(s, self._teacher_s, self._goals[:, i]) for i in range(3)],
            dtype=np.float32,
        )
        seed = np.asarray(
            [np.interp(s, self._teacher_s, self._seeds[:, i]) for i in range(4)],
            dtype=np.float32,
        )
        wz = float(np.interp(s, self._teacher_s, self._world_z))
        return goal, seed, wz

    def _refresh_path_parameter_derivatives(self) -> None:
        """Cache ds/du and d2s/du2 for geometry s and timing coordinate u."""
        self._geometry_per_timing = np.gradient(self._geometry_s, self._teacher_s)
        self._geometry_per_timing2 = np.gradient(
            self._geometry_per_timing, self._teacher_s
        )
        self._geometry_length = float(self._geometry_s[-1])

    def _geometry_to_timing(self, geometry_s_m: np.ndarray) -> np.ndarray:
        return np.interp(geometry_s_m, self._geometry_s, self._teacher_s)

    def _timing_kinematics(
        self, timing_s_m: float, timing_speed: float, timing_accel: float
    ) -> tuple[float, float, float]:
        u = float(np.clip(timing_s_m, 0.0, self._arc_length))
        geometry_s = float(np.interp(u, self._teacher_s, self._geometry_s))
        ds_du = float(np.interp(u, self._teacher_s, self._geometry_per_timing))
        d2s_du2 = float(np.interp(u, self._teacher_s, self._geometry_per_timing2))
        geometry_speed = timing_speed * ds_du
        geometry_accel = timing_accel * ds_du + timing_speed**2 * d2s_du2
        return geometry_s, geometry_speed, geometry_accel

    # ------------------------------------------------------------------
    # DLS controller setup
    # ------------------------------------------------------------------

    def _setup_dls_controllers(self) -> None:
        ground_j_path = ROOT / "mulag_plant/models/ee_jacobian_ground_truck3x4.json"
        ground_j = torch.tensor(
            json.loads(ground_j_path.read_text())["jacobian"],
            device=self.device,
            dtype=torch.float32,
        )
        ground_j[:, 3] = 0.0  # cylinder 4 owns head pitch

        self._cylinder_dls = CylinderDLSController(
            self,
            CylinderDLSConfig(
                invert_learned_pipeline=False,
                error_gain=float(getattr(self.cfg, "dls_error_gain", 0.18)),
                velocity_damping=float(getattr(self.cfg, "dls_velocity_damping", 0.0)),
                velocity_feedforward=float(getattr(self.cfg, "dls_velocity_feedforward", 1.0)),
                measured_error_feedback=float(
                    getattr(self.cfg, "dls_measured_error_feedback", 0.0)
                ),
                max_desired_delta_m=0.004,
                max_action=1.0,
                action_smoothing=float(getattr(self.cfg, "dls_action_smoothing", 0.10)),
            ),
        )
        self._cascade = EECascadeController(
            self,
            cylinder_controller=self._cylinder_dls,
            jacobian=ground_j,
            ee_feedback_gain=0.0,
            max_ik_step_m=0.004,
            max_cylinder_lead_m=0.025,
            # Spend the redundant DOF of the 3x4 deployment solve on stroke
            # centering instead of leaving it at minimum-norm. Default 0 keeps
            # every stored result reproducible; see EECascadeController for the
            # measurement this exists for.
            null_space_centering_gain=float(
                getattr(self.cfg, "dls_null_space_centering_gain", 0.0)
            ),
            null_space_weights=tuple(
                getattr(
                    self.cfg, "dls_null_space_weights", (0.0, 0.0, 1.0, 0.0)
                )
            ),
            null_space_max_step_m=float(
                getattr(self.cfg, "dls_null_space_max_step_m", 0.004)
            ),
        )

    # ------------------------------------------------------------------
    # Truck stationarity
    # ------------------------------------------------------------------

    def _init_truck_stationarity(self) -> None:
        if self._truck_stationary_initialized:
            return
        self._wheel_init_pos = self.robot.data.joint_pos[
            :, self._wheel_joint_ids
        ].detach().clone()
        self._steering_init_pos = self.robot.data.joint_pos[
            :, self._steering_joint_ids
        ].detach().clone()
        truck_base = self.robot.data.body_link_state_w[:, self._truck_base_body_id(), :3]
        self._truck_anchor_xy = truck_base[:, :2].detach().clone()
        self._truck_stationary_initialized = True

    @torch.no_grad()
    def park_truck_at_current_pose(self) -> None:
        """Latch the current truck pose as the stationary arm-work station.

        This is the only supported transition into arm tracking.  It refreshes
        both the planar anchor and wheel-position hold, so the parking fixture
        never pulls the truck back toward a previous station.
        """
        base = self.robot.data.body_link_state_w[:, self._truck_base_body_id(), :3]
        self._truck_anchor_xy[:] = base[:, :2].detach()
        self._wheel_init_pos = self.robot.data.joint_pos[
            :, self._wheel_joint_ids
        ].detach().clone()
        self._steering_init_pos = self.robot.data.joint_pos[
            :, self._steering_joint_ids
        ].detach().clone()
        self._truck_drift_xy.zero_()
        self._max_truck_drift.zero_()
        self._truck_post_constraint_drift_xy.zero_()
        self._max_truck_post_constraint_drift.zero_()
        self._truck_stationary_initialized = True
        self._hold_wheels()

    @torch.no_grad()
    def release_truck_for_transport(self) -> None:
        """Release the wheel-position fixture before deterministic transport.

        The caller must not invoke the Policy ``step`` while released;
        `_pre_physics_step` deliberately re-parks the truck for arm safety.
        """
        # Mark the station latch released until `_park_and_settle()` establishes
        # the next station.  Policy stepping remains forbidden during transport.
        self._truck_stationary_initialized = False
        ids = self._wheel_joint_ids
        self.robot.write_joint_stiffness_to_sim(0.0, joint_ids=ids)
        self.robot.write_joint_damping_to_sim(1.0e6, joint_ids=ids)
        self.robot.set_joint_velocity_target(
            torch.zeros((self.num_envs, len(ids)), device=self.device), joint_ids=ids
        )
        steering_ids = self._steering_joint_ids
        self.robot.write_joint_stiffness_to_sim(10.0, joint_ids=steering_ids)
        self.robot.write_joint_damping_to_sim(1.0e9, joint_ids=steering_ids)
        self.robot.set_joint_position_target(
            torch.zeros((self.num_envs, len(steering_ids)), device=self.device),
            joint_ids=steering_ids,
        )

    @torch.no_grad()
    def _hold_wheels(self) -> None:
        ids = self._wheel_joint_ids
        self.robot.write_joint_stiffness_to_sim(2.0e9, joint_ids=ids)
        self.robot.write_joint_damping_to_sim(1.0e7, joint_ids=ids)
        if self._wheel_init_pos is not None:
            self.robot.set_joint_position_target(self._wheel_init_pos, joint_ids=ids)
        creep_active = (
            self._trajectory_truck_motion_active
            and self._trajectory_truck_motion_mode == "creep"
        )
        if not creep_active:
            self.robot.set_joint_velocity_target(
                torch.zeros((self.num_envs, len(ids)), device=self.device), joint_ids=ids
            )
        steering_ids = self._steering_joint_ids
        self.robot.write_joint_stiffness_to_sim(2.0e8, joint_ids=steering_ids)
        self.robot.write_joint_damping_to_sim(1.0e8, joint_ids=steering_ids)
        if self._steering_init_pos is not None:
            self.robot.set_joint_position_target(
                self._steering_init_pos, joint_ids=steering_ids
            )
            self.robot.write_joint_position_to_sim(
                self._steering_init_pos, joint_ids=steering_ids
            )
        self.robot.set_joint_velocity_target(
            torch.zeros((self.num_envs, len(steering_ids)), device=self.device),
            joint_ids=steering_ids,
        )
        self.robot.write_joint_velocity_to_sim(
            torch.zeros((self.num_envs, len(steering_ids)), device=self.device),
            joint_ids=steering_ids,
        )


    def _clearance_signal(self) -> torch.Tensor:
        """Cutter clearance against its reference.

        Default path (`clearance_frame_consistent=False`) is the historical one
        and is kept bit-exact for PPO2-A and ROM-C.

        The historical path is frame-inconsistent, measured 2026-07-25:
        `_compute_cutter_world_z` applies the tool offset without rotation, so in
        the head-down pose it places the cutter 1.58 m below the orientation-aware
        `_compute_cutter_pos_w`. The trajectory library's `cutter_world_z` column
        was generated with the same unrotated formula, so the two errors largely
        cancel -- but what survives is the truck's absolute ride height, which
        leaks one-for-one into the reported clearance (0.164 m measured, 82 % of
        the 0.200 m limit).

        The consistent path uses the orientation-aware cutter and derives the
        reference from the live truck pose plus the truck-frame commanded height,
        so ride height cancels and the signal is a genuine cutter-vs-reference
        error. See PPO2_ROM_B_PhysX/reports/rom_b_frame_decomposition.md.
        """

        if not bool(getattr(self.cfg, "clearance_frame_consistent", False)):
            cutter_z = self._compute_cutter_world_z()
            if bool(getattr(self.cfg, "clearance_relative_to_reference", False)):
                return cutter_z - self._original_reference_world_z
            return cutter_z
        base_z = self.robot.data.body_link_state_w[:, self._truck_base_body_id(), 2]
        reference_world_z = base_z + self._original_reference_tool[:, 2]
        return self._compute_cutter_pos_w()[:, 2] - reference_world_z

    @torch.no_grad()
    def _constrain_truck_xy(self) -> None:
        if (
            self._trajectory_truck_motion_active
            and self._trajectory_truck_motion_mode == "creep"
        ):
            # Chassis travel is intentional for open paths.  It is commanded
            # by the wheel controller at a fixed, low speed; the parked-truck
            # drift metric must not interpret that longitudinal travel as a
            # suspension failure.
            self._truck_drift_xy.zero_()
            self._max_truck_drift.zero_()
            self._truck_post_constraint_drift_xy.zero_()
            self._max_truck_post_constraint_drift.zero_()
            return
        base_xy = self.robot.data.body_link_state_w[:, self._truck_base_body_id(), :2]
        correction = self._truck_anchor_xy - base_xy
        drift_norm = torch.linalg.norm(correction, dim=1)
        self._truck_drift_xy = drift_norm
        self._max_truck_drift = torch.maximum(self._max_truck_drift, drift_norm)

        # Match the working PPO1 ID-FD parked fixture exactly.  The deadband
        # prevents continuous root rewrites from visually making the chassis
        # chase suspension reactions.
        mask = drift_norm > 0.001
        if not mask.any():
            self._truck_post_constraint_drift_xy = drift_norm
            self._max_truck_post_constraint_drift = torch.maximum(
                self._max_truck_post_constraint_drift, drift_norm
            )
            return
        root_pose = self.robot.data.root_state_w[:, :7].detach().clone()
        root_pose[mask, :2] += correction[mask]
        self.robot.write_root_pose_to_sim(root_pose)
        post_drift = drift_norm.clone()
        post_drift[mask] = 0.0
        self._truck_post_constraint_drift_xy = post_drift
        self._max_truck_post_constraint_drift = torch.maximum(
            self._max_truck_post_constraint_drift, post_drift
        )

    # ------------------------------------------------------------------
    # Head-down measurement
    # ------------------------------------------------------------------

    def _measure_head_down_alignment(self) -> torch.Tensor:
        q = self.robot.data.body_link_state_w[:, self._ee_body_id, 3:7]
        q = q / q.norm(dim=-1, keepdim=True).clamp(min=1.0e-8)
        axis_w = quat_apply(
            q,
            torch.tensor([[0.0, 0.0, 1.0]], device=self.device).expand(self.num_envs, -1),
        )
        return -axis_w[:, 2]

    # ------------------------------------------------------------------
    # Pre-physics step (core residual logic)
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor):
        cfg = self.cfg
        actions = torch.nan_to_num(
            actions.to(dtype=torch.float32, device=self.device),
            nan=0.0, posinf=0.0, neginf=0.0,
        )
        if tuple(actions.shape) != (int(self.num_envs), self._residual_action_dim):
            raise RuntimeError(
                "Residual policy action must be "
                f"({self.num_envs}, {self._residual_action_dim}), got {tuple(actions.shape)}"
            )

        force_zero = bool(getattr(cfg, "force_zero_policy", False))
        residual_scale_m = float(getattr(cfg, "residual_scale_m", 0.002))
        action_filter = self._residual_filter_vector()
        residual_scales = self._residual_scale_vector()

        # Deployment and settling use the same direct PhysX target path as the
        # validated ground-arc gate.  They are deliberately outside the learned
        # command-ROM controller: the policy is not scored and cannot act until the arm
        # is safely at its verified start pose.
        warmup_mask = (self._preposition_counter > 0) | (self._settle_counter > 0)
        if warmup_mask.any():
            raise RuntimeError("physical preposition must finish inside reset before policy stepping")

        # Step 1: compute baseline cylinder goal from trajectory for active envs.
        self._update_trajectory()

        # Step 2: apply residual correction
        self._prev_residual_filtered = self._residual_filtered.clone()
        scaling = self._cylinder_adapter.scaling
        learned_min = scaling.learned_min.to(self.device, torch.float32).reshape(1, 4)
        learned_max = scaling.learned_max.to(self.device, torch.float32).reshape(1, 4)

        min_override = getattr(cfg, "residual_cylinder_learned_min_override", None)
        max_override = getattr(cfg, "residual_cylinder_learned_max_override", None)
        if min_override is not None:
            learned_min = torch.as_tensor(
                min_override, device=self.device, dtype=torch.float32
            ).reshape(1, 4)
        if max_override is not None:
            learned_max = torch.as_tensor(
                max_override, device=self.device, dtype=torch.float32
            ).reshape(1, 4)

        effective_action = torch.zeros_like(actions) if force_zero else actions
        fallback_shapes = tuple(getattr(cfg, "residual_fallback_shapes", ()))
        if (
            not force_zero
            and bool(getattr(cfg, "trajectory_geometry_observation", False))
            and self._trajectory_name in fallback_shapes
        ):
            # The moving-path deployment contract is safety-first: use the
            # validated frozen DLS baseline on open creep paths.  PPO remains
            # active on parked closed-loop stations where its residual has
            # enough margin and cannot disturb vehicle motion.
            effective_action = torch.zeros_like(effective_action)
        if (
            not force_zero
            and bool(getattr(cfg, "residual_safety_guard", False))
            and bool(getattr(cfg, "trajectory_geometry_observation", False))
        ):
            # Learned residuals are an optional refinement around the
            # deterministic DLS/teacher command.  If the measured cutter is
            # already close to the clearance boundary or the head is losing
            # its downwards alignment, remove the residual before it can push
            # the plant into a termination.  The next step then uses the
            # frozen baseline and can recover without a reset.
            clearance_start = float(
                getattr(cfg, "residual_safety_clearance_start_m", 0.08)
            )
            clearance_scale = (
                1.0
                - torch.abs(self._clearance_signal()) / max(clearance_start, 1.0e-6)
            ).clamp(0.0, 1.0)
            head_start = float(getattr(cfg, "residual_safety_head_start", 0.990))
            head_scale = (
                (self._measure_head_down_alignment() - float(getattr(
                    cfg, "head_down_termination_alignment", 0.98
                )))
                / max(head_start - float(getattr(
                    cfg, "head_down_termination_alignment", 0.98
                )), 1.0e-6)
            ).clamp(0.0, 1.0)
            safety_scale = torch.minimum(clearance_scale, head_scale)
            effective_action = effective_action * safety_scale[:, None]
            # Do not let a filtered action survive the fallback transition.
            self._residual_filtered = self._residual_filtered * safety_scale[:, None]
        self._residual_raw, self._residual_filtered, self._corrected_cylinder_goal = (
            apply_residual_action(
                baseline_goal=self._baseline_cylinder_goal,
                previous_filtered_action=self._residual_filtered,
                policy_action=effective_action,
                residual_scale_m=residual_scale_m,
                filter_beta=action_filter,
                learned_min=learned_min,
                learned_max=learned_max,
                residual_scale_by_action_m=residual_scales,
            )
        )
        self._residual_filtered[warmup_mask] = 0.0
        self._corrected_cylinder_goal[warmup_mask] = self._baseline_cylinder_goal[warmup_mask]

        # Step 3/4: only active environments enter the production command-ROM stack.
        # The parent pipeline is vectorized over all envs, so compute it once and
        # restore direct deployment targets for warmup envs afterwards.
        action_4d, _sv = self._cylinder_dls.action(
            self._corrected_cylinder_goal,
            goal_velocity=self._baseline_cylinder_velocity,
        )
        self._cylinder_goal_x[:] = self._corrected_cylinder_goal.clone()
        self._ppo_raw_action = sanitize_cylinder_action(
            action_4d, num_envs=self.num_envs, action_dim=4
        ).to(self.device)
        self._ppo_raw_action[warmup_mask] = 0.0
        self._current_action = self._ppo_raw_action.clone()
        if (~warmup_mask).any():
            direct_q = self._cylinder_pos_target[warmup_mask].clone()
            direct_joint_ref = self._joint_ref[warmup_mask].clone()
            previous_target = self._cylinder_pos_target.detach().clone()
            self._run_cylinder_id_pipeline()
            target_beta = min(
                max(float(getattr(cfg, "hydraulic_target_filter_beta", 0.0)), 0.0), 1.0
            )
            if target_beta > 0.0:
                self._cylinder_pos_target[:] = (
                    target_beta * previous_target
                    + (1.0 - target_beta) * self._cylinder_pos_target
                )
                for column, ref_index in enumerate(self._cylinder_adapter.joint_reference_indices):
                    self._joint_ref[:, ref_index] = self._cylinder_pos_target[:, column]
                source = self._joint6_coupling_source_ref_index
                target = self._joint6_coupling_target_ref_index
                if source is not None and target is not None:
                    self._joint_ref[:, target] = (
                        float(getattr(cfg, "joint6_coupling_a", 0.76))
                        * self._joint_ref[:, source]
                        + float(getattr(cfg, "joint6_coupling_b", 0.0))
                    )
            self._cylinder_pos_target[warmup_mask] = direct_q
            self._joint_ref[warmup_mask] = direct_joint_ref
        else:
            self._zero_pipeline_diagnostics(warmup_mask)

        # Step 5: truck stationarity
        self._hold_wheels()
        self._constrain_truck_xy()

    @torch.no_grad()
    def _write_direct_learned_targets(
        self, mask: torch.Tensor, learned_target: torch.Tensor
    ) -> None:
        scaling = self._cylinder_adapter.scaling
        target = torch.clamp(
            learned_target,
            scaling.learned_min.to(self.device).reshape(1, 4),
            scaling.learned_max.to(self.device).reshape(1, 4),
        )
        q_goal = scaling.clamp_usd_q(scaling.learned_x_to_usd_q(target)).to(
            device=self.device, dtype=self._cylinder_pos_target.dtype
        )
        self._cylinder_pos_target[mask] = q_goal
        self._neutral_hold_cylinder_pos_target[mask] = q_goal
        for column, ref_index in enumerate(self._cylinder_adapter.joint_reference_indices):
            self._joint_ref[mask, ref_index] = q_goal[:, column]
        source = getattr(self, "_joint6_coupling_source_ref_index", None)
        target_ref = getattr(self, "_joint6_coupling_target_ref_index", None)
        if source is not None and target_ref is not None:
            self._joint_ref[mask, target_ref] = (
                float(getattr(self.cfg, "joint6_coupling_a", 0.76))
                * self._joint_ref[mask, source]
                + float(getattr(self.cfg, "joint6_coupling_b", 0.0))
            )

    @torch.no_grad()
    def _zero_pipeline_diagnostics(self, mask: torch.Tensor) -> None:
        self._current_valve_cmd[mask] = 0.0
        self._prev_valve_cmd[mask] = 0.0
        self._fd_delta_raw[mask] = 0.0
        self._fd_delta_corrected[mask] = 0.0

    @torch.no_grad()
    def _reset_control_state(self, env_ids: torch.Tensor) -> None:
        self._zero_pipeline_diagnostics(
            torch.arange(self.num_envs, device=self.device)[:, None].eq(env_ids).any(dim=1)
        )
        # The command ROM is recurrent, so a reset must clear its hidden state or
        # the new episode inherits the previous one's hydraulic history. Under a
        # ROM this is mandatory and a missing runtime is a real bug. The ID/FD
        # actuator is feed-forward and carries no state across a reset, so there
        # is nothing to clear -- raising there would only forbid a valid plant.
        if self._command_cylinder_rom is not None:
            self._command_cylinder_rom.reset(env_ids)
        elif str(getattr(self.cfg, "cylinder_dynamics_model", "")).lower() in {
            "command_rom",
            "base_rom",
            "rom",
        }:
            raise RuntimeError("PPO2 residual reset requires the command ROM runtime")
        self._cylinder_dls.reset(env_ids)
        self._cascade.reset(env_ids)

    @torch.no_grad()
    def _warm_command_rom_at_target(self, target: torch.Tensor) -> None:
        """Prime recurrent hydraulic history before the first scored transition.

        Physical deployment deliberately bypasses the learned actuator.  A
        command ROM reset at an arbitrary mid-trajectory pose therefore needs
        the same closed-loop hold used by the passing ROM ground-arc gate.
        These physics steps remain inside reset and never enter PPO rollouts.
        """
        steps = int(getattr(self.cfg, "rom_reset_warmup_steps", 120))
        if steps <= 0:
            return
        target = target.reshape(self.num_envs, 4)
        zero_velocity = torch.zeros_like(target)
        dt = float(self.sim.get_physics_dt())
        full_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        for _ in range(steps):
            action_4d, _ = self._cylinder_dls.action(
                target, goal_velocity=zero_velocity
            )
            self._cylinder_goal_x[:] = target
            self._ppo_raw_action = sanitize_cylinder_action(
                action_4d, num_envs=self.num_envs, action_dim=4
            ).to(self.device)
            self._current_action = self._ppo_raw_action.clone()
            self._run_cylinder_id_pipeline()
            for _ in range(int(getattr(self.cfg, "decimation", 2))):
                self._apply_action()
                self._hold_wheels()
                self.scene.write_data_to_sim()
                self.sim.step(render=False)
                self.scene.update(dt)
            self._constrain_truck_xy()
        # The hold is part of reset, not a policy action or scored residual.
        self._residual_raw.zero_()
        self._residual_filtered.zero_()
        self._prev_residual_filtered.zero_()
        self._write_direct_learned_targets(full_mask, target)
        # Flush the planar fixture pose into PhysX, then start scored drift
        # accounting from that anchored post-warmup state.  Deployment/warmup
        # reactions are outside the PPO episode by contract.
        self._constrain_truck_xy()
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.scene.update(dt)
        self._constrain_truck_xy()
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.scene.update(dt)
        # Deployment and ROM priming are outside the scored episode and can
        # shift suspension equilibrium.  The arm-work station begins here, so
        # latch the settled pose and wheel positions instead of retaining the
        # pre-deployment anchor.  Subsequent drift is still measured strictly
        # from this pose and remains subject to the 2 cm termination limit.
        self.park_truck_at_current_pose()

    @torch.no_grad()
    def _run_reset_preposition(self, env_ids: torch.Tensor) -> None:
        """Physically deploy during reset so PPO never observes ignored actions.

        RSL-RL must only receive transitions for which its action can affect the
        plant.  Therefore the validated cosine deployment is completed inside
        reset, using the same two physics substeps per control step as the
        deterministic gate, before the first observation is returned.
        """
        if env_ids.numel() != self.num_envs:
            raise RuntimeError(
                "Residual-policy resets are synchronized; physical preposition "
                f"received {env_ids.numel()} of {self.num_envs} environments."
            )
        dt = float(self.sim.get_physics_dt())
        start = self._preposition_start_x.clone()
        target = self._preposition_target_x.clone()
        for step in range(self._preposition_total_steps):
            u = float(step + 1) / max(self._preposition_total_steps, 1)
            blend = 0.5 * (1.0 - math.cos(math.pi * u))
            self._write_direct_learned_targets(
                torch.ones(self.num_envs, dtype=torch.bool, device=self.device),
                start + blend * (target - start),
            )
            for _ in range(int(getattr(self.cfg, "decimation", 2))):
                self._apply_action()
                self._hold_wheels()
                self.scene.write_data_to_sim()
                self.sim.step(render=False)
                self.scene.update(dt)
            self._constrain_truck_xy()

        full_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._write_direct_learned_targets(full_mask, target)
        for _ in range(self._settle_total_steps):
            for _ in range(int(getattr(self.cfg, "decimation", 2))):
                self._apply_action()
                self._hold_wheels()
                self.scene.write_data_to_sim()
                self.sim.step(render=False)
                self.scene.update(dt)
            self._constrain_truck_xy()

        self._preposition_counter[env_ids] = 0
        self._settle_counter[env_ids] = 0
        self._reset_control_state(env_ids)
        self._warm_command_rom_at_target(target)

    @torch.no_grad()
    def prepare_arm_cycle_at_station(self) -> None:
        """Start a fresh arm-only trajectory without resetting the truck pose.

        Transport is performed outside the RL environment step.  Once the
        supervisor has stopped and parked the truck, this method deploys the arm
        to the nearest verified trajectory endpoint and resets policy histories.
        It intentionally does not call the parent reset, which would teleport
        the truck back to the original scene pose.
        """
        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self._trajectory_truck_motion_active = False
        self.park_truck_at_current_pose()

        measured_x, _ = self._refresh_cylinder_state()
        endpoint_a = torch.as_tensor(
            self._seeds[0], dtype=torch.float32, device=self.device
        ).reshape(1, 4)
        endpoint_b = torch.as_tensor(
            self._seeds[-1], dtype=torch.float32, device=self.device
        ).reshape(1, 4)
        start_at_a = (
            torch.linalg.norm(measured_x - endpoint_a, dim=-1)
            <= torch.linalg.norm(measured_x - endpoint_b, dim=-1)
        )
        start_s = torch.where(
            start_at_a,
            torch.zeros((self.num_envs,), device=self.device),
            torch.full((self.num_envs,), self._arc_length, device=self.device),
        )
        start_direction = torch.where(
            start_at_a,
            torch.ones((self.num_envs,), device=self.device),
            -torch.ones((self.num_envs,), device=self.device),
        )

        target_seed = torch.zeros_like(measured_x)
        target_goal = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        target_world_z = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        for row, initial_s in enumerate(start_s.detach().cpu().tolist()):
            goal_np, seed_np, world_z = self._interpolate_trajectory(float(initial_s))
            target_goal[row] = torch.as_tensor(goal_np, device=self.device)
            target_seed[row] = torch.as_tensor(seed_np, device=self.device)
            target_world_z[row] = world_z

        self._traj_start_s[:] = start_s
        self._traj_start_direction[:] = start_direction
        self._traj_direction[:] = start_direction
        self._traj_elapsed.zero_()
        self._scored_elapsed.zero_()
        start_geometry_s = torch.as_tensor(
            np.interp(
                start_s.detach().cpu().numpy(), self._teacher_s, self._geometry_s
            ),
            dtype=torch.float32,
            device=self.device,
        )
        self._reference_path_s[:] = start_geometry_s
        self._reference_timing_s[:] = start_s
        self._reference_timing_speed.zero_()
        self._reference_path_speed.zero_()
        self._reference_path_accel.zero_()
        self._traj_phase[:] = torch.where(
            start_direction > 0.0,
            0.5 * start_geometry_s / self._geometry_length,
            0.5 + 0.5 * (1.0 - start_geometry_s / self._geometry_length),
        )
        self._preposition_start_x[:] = measured_x
        self._preposition_target_x[:] = target_seed
        self._preposition_counter.fill_(self._preposition_total_steps)
        self._settle_counter.fill_(self._settle_total_steps)
        self._original_reference_tool[:] = target_goal
        self._original_reference_world_z[:] = target_world_z

        self._residual_raw.zero_()
        self._residual_filtered.zero_()
        self._prev_residual_filtered.zero_()
        self._baseline_cylinder_goal.zero_()
        self._baseline_cylinder_velocity.zero_()
        self._corrected_cylinder_goal.zero_()
        self._measured_tool_velocity_truck.zero_()
        self._previous_measured_tool_velocity_truck.zero_()
        self._measured_tool_acceleration_truck.zero_()
        self._previous_error_norm.zero_()
        self._head_alignment_rate.zero_()
        self.episode_length_buf.zero_()

        self._run_reset_preposition(env_ids)
        self._trajectory_truck_motion_active = (
            self._trajectory_truck_motion_mode == "creep"
        )
        current_tool = self._compute_ee_pos_local().detach()
        if self._prev_tool_pos_truck is None:
            self._prev_tool_pos_truck = current_tool.clone()
        else:
            self._prev_tool_pos_truck[:] = current_tool
        self._prev_head_alignment[:] = self._measure_head_down_alignment()
        self._previous_error_norm[:] = torch.linalg.norm(
            self._original_reference_tool - current_tool, dim=-1
        )

    # ------------------------------------------------------------------
    # Trajectory update
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _update_trajectory(self) -> None:
        cfg = self.cfg
        seed_lead = float(getattr(cfg, "seed_lead_time_s", 0.25))
        dt = float(self.step_dt)

        not_warmup = (self._preposition_counter <= 0) & (self._settle_counter <= 0)
        self._traj_elapsed[not_warmup] += dt
        self._scored_elapsed[not_warmup] += dt

        if not not_warmup.any():
            return
        if not not_warmup.all():
            raise RuntimeError("Policy trajectory environments must advance synchronously")

        goals = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        seeds = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        seed_pitch = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        seed_velocities = torch.zeros_like(seeds)
        reference_world_z = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        for env_id in range(self.num_envs):
            if not not_warmup[env_id]:
                continue
            d_val, signed_speed, _signed_accel, direction = self._trajectory_state(
                env_id, float(self._traj_elapsed[env_id])
            )
            s_val = abs(signed_speed)
            geometry_s, geometry_speed, geometry_accel = self._timing_kinematics(
                d_val, signed_speed, _signed_accel
            )
            self._traj_direction[env_id] = direction
            self._reference_timing_s[env_id] = d_val
            self._reference_timing_speed[env_id] = signed_speed
            self._reference_path_s[env_id] = geometry_s
            self._reference_path_speed[env_id] = geometry_speed
            self._reference_path_accel[env_id] = geometry_accel

            goal_np, _seed_at_goal, _teacher_wz = self._interpolate_trajectory(d_val)
            goal_np = goal_np.copy()
            # The generated workspace library already encodes each shape's
            # absolute working height.  ``ground_height_bias_m`` belongs to
            # the legacy ground arc only; applying it here caused a 30 mm
            # reference/command discontinuity on the first scored step.
            height_bias = (
                float(getattr(cfg, "ground_height_bias_m", 0.0))
                if self._trajectory_name == "legacy_arc"
                else 0.0
            )
            goal_np[2] += height_bias
            seed_distance = d_val + direction * abs(s_val) * seed_lead
            _seed_goal, seed_np, _seed_wz = self._interpolate_trajectory(seed_distance)
            next_seed_distance = seed_distance + direction * abs(s_val) * dt
            _next_seed_goal, next_seed_np, _next_seed_wz = self._interpolate_trajectory(
                next_seed_distance
            )

            goals[env_id] = torch.as_tensor(goal_np, device=self.device)
            seed_pitch[env_id] = float(_seed_at_goal[3])
            seeds[env_id] = torch.as_tensor(seed_np, device=self.device)
            seed_velocities[env_id] = torch.as_tensor(
                (next_seed_np - seed_np) / max(dt, 1.0e-9), device=self.device
            )
            reference_world_z[env_id] = _teacher_wz + height_bias

            path_fraction = geometry_s / max(self._geometry_length, 1.0e-9)
            phase = 0.5 * path_fraction if direction > 0.0 else 0.5 + 0.5 * (1.0 - path_fraction)
            self._traj_phase[env_id] = phase

        self._original_reference_tool[:] = goals
        self._original_reference_world_z[:] = reference_world_z
        if self._trajectory_name == "legacy_arc":
            # Preserve the accepted legacy map-seeded controller exactly.
            cylinder_goal = self._cascade.cylinder_goal_seeded(
                goals,
                seeds,
                correction_gain=float(getattr(cfg, "ee_correction_gain", 0.0)),
            )
            cylinder_goal[:, 3] = seeds[:, 3]
        else:
            # v2 calibration seeds are endpoint/preposition hints only.  The
            # nearest workspace sample is not an inverse-kinematics solution
            # for an arbitrary 2-D point, so using it as the live goal creates
            # the large lateral errors seen on full-shape rollouts.  Use the
            # batch-stable full DLS solve for the free XYZ reference, then
            # retain the measured cylinder-4 head-down branch from the
            # workspace library.  (The optional reduced fixed-channel solve is
            # kept in EECascadeController for single-station diagnostics.)
            cylinder_goal = self._cascade.cylinder_goal(goals)
            cylinder_goal[:, 3] = seed_pitch
        self._baseline_cylinder_goal[:] = cylinder_goal
        velocity_beta = min(
            max(float(getattr(cfg, "dls_velocity_filter_beta", 0.0)), 0.0), 1.0
        )
        self._baseline_cylinder_velocity[:] = (
            velocity_beta * self._baseline_cylinder_velocity
            + (1.0 - velocity_beta) * seed_velocities
        )

    @torch.no_grad()
    def _trajectory_state(self, env_id: int, elapsed: float) -> tuple[float, float, float, float]:
        """Smooth path state from an arbitrary sampled start pose.

        The first segment always accelerates from rest toward the selected
        endpoint. Subsequent full-length legs use the validated A<->B cruise
        profile. This avoids a nonphysical velocity jump at randomized resets.
        """
        path_speed = self._effective_path_speed_mps()
        accel_time = float(getattr(self.cfg, "accel_time_s", 1.0))
        start_s = float(self._traj_start_s[env_id].item())
        start_direction = float(self._traj_start_direction[env_id].item())
        first_length = self._arc_length - start_s if start_direction > 0.0 else start_s
        zero_t = torch.zeros(1, device=self.device)
        _, _, _, first_time = cruise_profile(
            zero_t,
            length_m=max(first_length, 1.0e-6),
            cruise_speed_mps=path_speed,
            accel_time_s=accel_time,
            decel_time_s=accel_time,
        )
        if elapsed <= first_time:
            distance, speed, accel, _ = cruise_profile(
                torch.tensor([elapsed], device=self.device),
                length_m=max(first_length, 1.0e-6),
                cruise_speed_mps=path_speed,
                accel_time_s=accel_time,
                decel_time_s=accel_time,
            )
            coordinate = start_s + start_direction * min(float(distance.item()), first_length)
            return (
                float(np.clip(coordinate, 0.0, self._arc_length)),
                start_direction * float(speed.item()),
                start_direction * float(accel.item()),
                start_direction,
            )

        remaining = elapsed - first_time
        leg_index = int(remaining // self._one_way_s)
        leg_t = remaining - leg_index * self._one_way_s
        direction = -start_direction if leg_index % 2 == 0 else start_direction
        distance, speed, accel, _ = cruise_profile(
            torch.tensor([leg_t], device=self.device),
            length_m=self._arc_length,
            cruise_speed_mps=path_speed,
            accel_time_s=accel_time,
            decel_time_s=accel_time,
        )
        coordinate = float(distance.item()) if direction > 0.0 else self._arc_length - float(distance.item())
        return (
            float(np.clip(coordinate, 0.0, self._arc_length)),
            direction * float(speed.item()),
            direction * float(accel.item()),
            direction,
        )

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _get_observations(self) -> dict:
        cfg = self.cfg
        dt = max(float(self.step_dt), 1.0e-6)

        tool_pos_truck = self._compute_ee_pos_local()

        if self._prev_tool_pos_truck is not None:
            measured_velocity = (
                tool_pos_truck - self._prev_tool_pos_truck
            ) / dt
            self._measured_tool_acceleration_truck = (
                measured_velocity - self._previous_measured_tool_velocity_truck
            ) / dt
            self._measured_tool_velocity_truck = measured_velocity
            self._previous_measured_tool_velocity_truck = measured_velocity.detach().clone()
        self._prev_tool_pos_truck = tool_pos_truck.detach().clone()

        tool_error_truck = self._original_reference_tool - tool_pos_truck

        ref_vel = torch.zeros((self.num_envs, 3), device=self.device)
        ref_acc = torch.zeros((self.num_envs, 3), device=self.device)
        reference_preview_xy = torch.zeros((self.num_envs, 2), device=self.device)
        preview_time = max(float(getattr(self.cfg, "policy_preview_time_s", 0.25)), 0.0)
        for env_id in range(self.num_envs):
            path_distance = float(self._reference_timing_s[env_id].item())
            signed_speed = float(self._reference_path_speed[env_id].item())
            signed_timing_speed = float(
                self._reference_timing_speed[env_id].item()
            )
            signed_accel = float(self._reference_path_accel[env_id].item())
            goal_plus, _, _ = self._interpolate_trajectory(
                min(path_distance + 0.01, self._arc_length)
            )
            goal_minus, _, _ = self._interpolate_trajectory(
                max(path_distance - 0.01, 0.0)
            )
            tangent_dir = goal_plus - goal_minus
            tangent_norm = float(np.linalg.norm(tangent_dir))
            if tangent_norm > 1e-9:
                tangent_dir = tangent_dir / tangent_norm
            else:
                tangent_dir = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            ref_vel[env_id] = torch.as_tensor(
                signed_speed * tangent_dir,
                device=self.device, dtype=torch.float32,
            )
            ref_acc[env_id] = torch.as_tensor(
                signed_accel * tangent_dir,
                device=self.device, dtype=torch.float32,
            )
            direction = float(self._traj_direction[env_id].item())
            preview_step = max(
                abs(signed_timing_speed), self._effective_path_speed_mps()
            ) * preview_time
            preview_s = float(np.clip(
                path_distance + direction * preview_step,
                0.0,
                self._arc_length,
            ))
            preview_goal, _, _ = self._interpolate_trajectory(preview_s)
            current_goal, _, _ = self._interpolate_trajectory(path_distance)
            reference_preview_xy[env_id] = torch.as_tensor(
                preview_goal[:2] - current_goal[:2],
                device=self.device,
                dtype=torch.float32,
            )

        x, xdot = self._refresh_cylinder_state()
        self._cylinder_x_current = x.detach().clone()
        self._cylinder_xdot_current = xdot.detach().clone()
        cylinder_goal_error = self._corrected_cylinder_goal - x

        head_alignment = self._measure_head_down_alignment()
        self._head_alignment_rate = (
            head_alignment - self._prev_head_alignment
        ) / dt
        self._prev_head_alignment = head_alignment.detach().clone()

        clearance_signal = self._clearance_signal()

        nearest_point_delta_xy = None
        signed_normal_error_m = None
        if bool(getattr(cfg, "trajectory_geometry_observation", False)):
            nearest_point_delta_xy, signed_normal_error_m = self._nearest_trajectory_geometry(
                tool_pos_truck
            )

        obs = build_residual_observation(
            tool_error_truck=tool_error_truck,
            measured_tool_velocity_truck=self._measured_tool_velocity_truck,
            reference_tool_velocity_truck=ref_vel,
            reference_tool_acceleration_truck=ref_acc,
            measured_cylinder_extensions=x,
            cylinder_goal_error=cylinder_goal_error,
            measured_cylinder_velocities=xdot,
            previous_filtered_policy_action=self._prev_residual_filtered,
            current_valve_commands=self._current_valve_cmd,
            corrected_fd_delta=self._fd_delta_corrected,
            reference_preview_xy_truck=reference_preview_xy,
            trajectory_direction=self._traj_direction,
            cutter_bottom_world_clearance=clearance_signal,
            head_down_alignment=head_alignment,
            head_down_alignment_rate=self._head_alignment_rate,
            nearest_point_delta_xy=nearest_point_delta_xy,
            signed_normal_error_m=signed_normal_error_m,
        )
        return {"policy": obs}

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------

    def _get_rewards(self) -> torch.Tensor:
        cfg = self.cfg
        tool_pos_truck = self._compute_ee_pos_local()
        head_alignment = self._measure_head_down_alignment()
        clearance_signal = self._clearance_signal()
        signed_normal_error_m = None
        if bool(getattr(cfg, "trajectory_geometry_observation", False)):
            _, signed_normal_error_m = self._nearest_trajectory_geometry(tool_pos_truck)

        terms: ResidualRewardTerms = compute_residual_reward(
            original_reference_tool_position=self._original_reference_tool,
            measured_tool_position=tool_pos_truck,
            previous_error_norm=self._previous_error_norm,
            filtered_action=self._residual_filtered,
            previous_filtered_action=self._prev_residual_filtered,
            head_down_alignment=head_alignment,
            cutter_clearance=clearance_signal,
            valve_command=self._current_valve_cmd,
            tracking_std_m=float(getattr(cfg, "residual_tracking_std_m", 0.02)),
            tracking_weight=float(getattr(cfg, "residual_tracking_weight", 4.0)),
            progress_weight=float(getattr(cfg, "residual_progress_weight", 0.5)),
            action_magnitude_weight=float(getattr(cfg, "residual_action_magnitude_weight", -0.02)),
            action_rate_weight=float(getattr(cfg, "residual_action_rate_weight", -0.05)),
            head_penalty_weight=float(getattr(cfg, "residual_head_penalty_weight", -2.0)),
            ground_penalty_weight=float(getattr(cfg, "residual_ground_penalty_weight", -2.0)),
            saturation_penalty_weight=float(getattr(cfg, "residual_saturation_penalty_weight", -1.0)),
            head_threshold=float(getattr(cfg, "residual_head_threshold", 0.995)),
            ground_band_m=float(getattr(cfg, "residual_ground_band_m", 0.10)),
            signed_normal_error_m=signed_normal_error_m,
            normal_tracking_std_m=float(getattr(cfg, "trajectory_geometry_normal_std_m", 0.03)),
            normal_tracking_weight=float(getattr(cfg, "trajectory_geometry_normal_reward_weight", 0.0)),
        )
        active = (self._preposition_counter <= 0) & (self._settle_counter <= 0)
        acceleration_norm = torch.linalg.norm(
            self._measured_tool_acceleration_truck, dim=-1
        )
        acceleration_scale = max(float(getattr(
            cfg, "residual_tool_acceleration_scale_mps2", 30.0
        )), 1.0e-6)
        normalized_acceleration = torch.clamp(
            acceleration_norm / acceleration_scale, max=2.0
        )
        tracking_guard_scale = max(float(getattr(
            cfg, "residual_tracking_guard_scale_m", 0.04
        )), 1.0e-6)
        normalized_tracking_error = torch.clamp(
            terms.error_norm / tracking_guard_scale, max=2.0
        )
        tracking_guard = float(getattr(
            cfg, "residual_tracking_guard_weight", 0.0
        )) * normalized_tracking_error.square()
        acceleration_penalty = float(getattr(
            cfg, "residual_tool_acceleration_weight", 0.0
        )) * normalized_acceleration.square()
        cylinder4_penalty = torch.zeros_like(terms.reward)
        if self._residual_action_dim == 4:
            c4 = self._residual_filtered[:, 3]
            previous_c4 = self._prev_residual_filtered[:, 3]
            cylinder4_penalty = (
                float(getattr(cfg, "cylinder4_action_magnitude_weight", -0.5))
                * c4.square()
                + float(getattr(cfg, "cylinder4_action_rate_weight", -1.0))
                * (c4 - previous_c4).square()
            )
        reward = torch.where(
            active,
            terms.reward + tracking_guard + acceleration_penalty + cylinder4_penalty,
            torch.zeros_like(terms.reward),
        )
        self._previous_error_norm = torch.where(
            active, terms.error_norm, self._previous_error_norm
        )

        residual_scales = self._residual_scale_vector().reshape(1, -1)
        if torch.any(active):
            active_error = terms.error_norm[active]
            active_acceleration = acceleration_norm[active]
            active_head = head_alignment[active]
            active_residual = self._residual_filtered[active]
        else:
            # Preposition/settle samples are deliberately unscored. Reporting
            # zeros here keeps deployment transients out of PPO summaries.
            active_error = torch.zeros(1, device=self.device)
            active_acceleration = torch.zeros(1, device=self.device)
            active_head = torch.ones(1, device=self.device)
            active_residual = torch.zeros(
                (1, self._residual_action_dim), device=self.device
            )
        self._last_reward_terms = {
            "tracking/mean_error_m": torch.mean(active_error),
            "tracking/p95_error_m": torch.quantile(active_error, 0.95),
            "tracking/mean_abs_normal_error_m": (
                torch.mean(torch.abs(signed_normal_error_m[active]))
                if signed_normal_error_m is not None and torch.any(active)
                else torch.zeros((), device=self.device)
            ),
            "smoothness/p95_tool_acceleration_mps2": torch.quantile(
                active_acceleration, 0.95
            ),
            "safety/min_head_alignment": torch.min(active_head),
            "safety/max_truck_drift_m": torch.max(
                self._truck_post_constraint_drift_xy
            ),
            "policy/mean_abs_residual_m": torch.mean(
                torch.abs(active_residual) * residual_scales
            ),
        }
        if self._residual_action_dim == 4:
            self._last_reward_terms.update({
                "policy/cylinder4_mean_abs_residual_m": torch.mean(
                    torch.abs(active_residual[:, 3]) * residual_scales[0, 3]
                ),
                "policy/cylinder4_authority_m": residual_scales[0, 3],
            })
        self.extras.setdefault("log", {}).update(self._last_reward_terms)
        return reward

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        # `_pre_physics_step` clamps the parked truck before simulation. Clamp
        # again after the decimated physics steps so reward/observations see
        # the anchored pose rather than a one-step suspension reaction.
        self._constrain_truck_xy()
        # Warmup is an unscored deployment phase and must not consume the PPO
        # episode horizon.
        timeout = self._scored_elapsed >= float(self.cfg.episode_length_s)
        warmup = (self._preposition_counter > 0) | (self._settle_counter > 0)

        cylinder_q = self.robot.data.joint_pos[:, self._cylinder_joint_ids]
        cylinder_qdot = self.robot.data.joint_vel[:, self._cylinder_joint_ids]
        limits = self.robot.data.soft_joint_pos_limits[:, self._cylinder_joint_ids]
        limit_margin = float(getattr(cfg, "cylinder_limit_margin_m", 0.002))
        limit_violation = (
            (cylinder_q < limits[..., 0] - limit_margin)
            | (cylinder_q > limits[..., 1] + limit_margin)
        ).any(dim=-1)

        nonfinite = ~(
            torch.isfinite(cylinder_q).all(dim=-1)
            & torch.isfinite(cylinder_qdot).all(dim=-1)
            & torch.isfinite(self._current_valve_cmd).all(dim=-1)
            & torch.isfinite(self._fd_delta_corrected).all(dim=-1)
            & torch.isfinite(self._corrected_cylinder_goal).all(dim=-1)
        )

        truck_post_drift_exceeded = self._truck_post_constraint_drift_xy > float(
            getattr(cfg, "truck_drift_termination_m", 0.02)
        )
        truck_raw_reaction_exceeded = self._truck_drift_xy > float(
            getattr(cfg, "truck_raw_reaction_termination_m", 0.05)
        )
        truck_drift_exceeded = truck_post_drift_exceeded | truck_raw_reaction_exceeded

        head_alignment = self._measure_head_down_alignment()
        head_term = head_alignment < float(
            getattr(cfg, "head_down_termination_alignment", 0.95)
        )

        clearance_error = self._clearance_signal()
        hard_band = float(getattr(cfg, "cutter_clearance_hard_band_m", 0.15))
        clearance_violation = torch.abs(clearance_error) > hard_band

        safety_violation = (
            limit_violation
            | truck_drift_exceeded
            | head_term
            | clearance_violation
        )
        path_complete = getattr(self, "_truck_path_complete", None)
        if isinstance(path_complete, torch.Tensor):
            safety_violation &= ~path_complete
        # In large vectorized training runs, synchronizing a finite safety
        # violation from one replica used to reset every robot.  Each reset
        # performs physical deployment, settling, and ROM priming, so a single
        # noisy arc sample could turn a 15-second rollout into minutes without
        # an iteration log.  Keep all safety metrics/reward terms active and
        # defer only those finite resets to the regular synchronized timeout.
        # Non-finite state must always terminate immediately.
        if bool(getattr(cfg, "defer_safety_terminations", False)):
            terminated = nonfinite
        else:
            terminated = safety_violation | nonfinite
        terminated &= ~warmup
        if bool(getattr(cfg, "synchronize_resets", True)) and terminated.any():
            terminated = torch.ones_like(terminated)
        truncated = timeout

        log = self.extras.setdefault("log", {})
        log.update({
            "termination/limit": torch.mean(limit_violation.float()),
            "termination/nonfinite": torch.mean(nonfinite.float()),
            "termination/truck_drift": torch.mean(truck_drift_exceeded.float()),
            "termination/head_down": torch.mean(head_term.float()),
            "termination/clearance": torch.mean(clearance_violation.float()),
            "termination/timeout": torch.mean(timeout.float()),
            "termination_value/clearance_abs_m": torch.max(
                torch.abs(self._compute_cutter_world_z())
            ),
            "termination_value/clearance_error_abs_m": torch.max(torch.abs(clearance_error)),
            "termination_value/head_alignment_min": torch.min(head_alignment),
            "termination_value/truck_drift_m": torch.max(self._truck_drift_xy),
            "termination_value/truck_post_constraint_drift_m": torch.max(
                self._truck_post_constraint_drift_xy
            ),
            "termination_value/hard_clearance_band_m": torch.as_tensor(
                hard_band, device=self.device
            ),
            "termination_value/tool_error_m": torch.max(torch.linalg.norm(
                self._original_reference_tool - self._compute_ee_pos_local(), dim=-1
            )),
            "termination_value/truck_base_z_m": torch.max(
                self.robot.data.body_link_state_w[:, self._truck_base_body_id(), 2]
            ),
            "termination_value/path_s_m": torch.max(self._reference_path_s),
        })
        return terminated, truncated

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids: Sequence[int]):
        self._trajectory_truck_motion_active = False
        super()._reset_idx(env_ids)
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

        if bool(getattr(self.cfg, "randomize_trajectory_shape", False)):
            if env_ids.numel() != self.num_envs:
                raise RuntimeError("shape-randomized residual resets must be synchronized")
            exact = tuple(getattr(self.cfg, "trajectory_shapes", ()))
            procedural = tuple(getattr(self.cfg, "procedural_trajectory_shapes", ()))
            probability = min(max(float(getattr(
                self.cfg, "procedural_trajectory_probability", 0.0
            )), 0.0), 1.0)
            unavailable = sorted((set(exact) | set(procedural)) - set(self._trajectory_library))
            if not exact or unavailable:
                raise RuntimeError(
                    f"invalid exact={exact} procedural={procedural}; unavailable={unavailable}"
                )
            use_procedural = bool(procedural) and bool(
                torch.rand((), device=self.device).item() < probability
            )
            enabled = procedural if use_procedural else exact
            shape_index = int(torch.randint(len(enabled), (1,), device=self.device).item())
            self.set_trajectory_shape(enabled[shape_index])

        # Reset residual state
        self._residual_raw[env_ids] = 0.0
        self._residual_filtered[env_ids] = 0.0
        self._prev_residual_filtered[env_ids] = 0.0
        self._baseline_cylinder_goal[env_ids] = 0.0
        self._baseline_cylinder_velocity[env_ids] = 0.0
        self._corrected_cylinder_goal[env_ids] = 0.0

        # Reset observation history
        self._measured_tool_velocity_truck[env_ids] = 0.0
        self._previous_measured_tool_velocity_truck[env_ids] = 0.0
        self._measured_tool_acceleration_truck[env_ids] = 0.0
        if self._prev_tool_pos_truck is not None:
            self._prev_tool_pos_truck[env_ids] = 0.0

        # Reset reward state
        self._previous_error_norm[env_ids] = 0.0

        # Reset head-down history
        self._prev_head_alignment[env_ids] = 0.0
        self._head_alignment_rate[env_ids] = 0.0

        # Reset trajectory state
        n = env_ids.numel()
        measured_x, _ = self._refresh_cylinder_state()
        endpoint_a = torch.as_tensor(
            self._seeds[0], dtype=torch.float32, device=self.device
        ).reshape(1, 4)
        endpoint_b = torch.as_tensor(
            self._seeds[-1], dtype=torch.float32, device=self.device
        ).reshape(1, 4)
        distance_a = torch.linalg.norm(measured_x[env_ids] - endpoint_a, dim=-1)
        distance_b = torch.linalg.norm(measured_x[env_ids] - endpoint_b, dim=-1)
        start_at_a = distance_a <= distance_b
        randomize_phase = bool(getattr(self.cfg, "randomize_start_phase", False))
        if randomize_phase:
            # Sample the geometric circle phase, then map it to the optional
            # timing coordinate.  This keeps physical start poses bit-matched
            # between baseline and retimed candidates.
            geometry_start = (
                torch.rand((n,), device=self.device) * self._geometry_length
            )
            start_s = torch.as_tensor(
                self._geometry_to_timing(geometry_start.detach().cpu().numpy()),
                dtype=torch.float32,
                device=self.device,
            )
            start_direction = torch.where(
                torch.rand((n,), device=self.device) < 0.5,
                torch.ones((n,), device=self.device),
                -torch.ones((n,), device=self.device),
            )
        else:
            start_s = torch.where(
                start_at_a,
                torch.zeros((n,), device=self.device),
                torch.full((n,), self._arc_length, device=self.device),
            )
            start_direction = torch.where(
                start_at_a,
                torch.ones((n,), device=self.device),
                -torch.ones((n,), device=self.device),
            )
        self._traj_start_s[env_ids] = start_s
        self._traj_start_direction[env_ids] = start_direction
        self._traj_direction[env_ids] = start_direction
        self._traj_elapsed[env_ids] = 0.0
        self._scored_elapsed[env_ids] = 0.0
        self._reference_path_s[env_ids] = start_s
        self._reference_timing_s[env_ids] = start_s
        self._reference_timing_speed[env_ids] = 0.0
        self._reference_path_speed[env_ids] = 0.0
        self._reference_path_accel[env_ids] = 0.0
        start_geometry_s = torch.as_tensor(
            np.interp(
                start_s.detach().cpu().numpy(), self._teacher_s, self._geometry_s
            ),
            dtype=torch.float32,
            device=self.device,
        )
        self._reference_path_s[env_ids] = start_geometry_s
        self._traj_phase[env_ids] = torch.where(
            start_direction > 0.0,
            0.5 * start_geometry_s / self._geometry_length,
            0.5 + 0.5 * (1.0 - start_geometry_s / self._geometry_length),
        )

        target_seed = torch.zeros((n, 4), dtype=torch.float32, device=self.device)
        target_goal = torch.zeros((n, 3), dtype=torch.float32, device=self.device)
        target_world_z = torch.zeros((n,), dtype=torch.float32, device=self.device)
        for row, initial_s in enumerate(start_s.detach().cpu().tolist()):
            goal_np, seed_np, world_z = self._interpolate_trajectory(float(initial_s))
            target_goal[row] = torch.as_tensor(goal_np, device=self.device)
            target_seed[row] = torch.as_tensor(seed_np, device=self.device)
            target_world_z[row] = world_z

        self._preposition_start_x[env_ids] = measured_x[env_ids]
        self._preposition_target_x[env_ids] = target_seed

        # Reset preposition counters
        preposition_steps = int(getattr(self.cfg, "preposition_steps", 300))
        settle_steps = int(getattr(self.cfg, "settle_steps", 120))
        self._preposition_counter[env_ids] = preposition_steps
        self._settle_counter[env_ids] = settle_steps

        # Reset truck state
        self._truck_drift_xy[env_ids] = 0.0
        self._max_truck_drift[env_ids] = 0.0
        self._truck_post_constraint_drift_xy[env_ids] = 0.0
        self._max_truck_post_constraint_drift[env_ids] = 0.0
        if self._truck_stationary_initialized:
            truck_base = self.robot.data.body_link_state_w[
                :, self._truck_base_body_id(), :3
            ]
            self._truck_anchor_xy[env_ids] = truck_base[env_ids, :2].detach().clone()

        # Reset DLS controllers
        if self._cylinder_dls is not None:
            self._cylinder_dls.reset(env_ids)
        if self._cascade is not None:
            self._cascade.reset(env_ids)

        # Reset reference tool
        self._original_reference_tool[env_ids] = target_goal
        self._original_reference_world_z[env_ids] = target_world_z

        # Init truck stationarity on first reset
        self._init_truck_stationarity()
        self._run_reset_preposition(env_ids)
        self._trajectory_truck_motion_active = (
            self._trajectory_truck_motion_mode == "creep"
        )
        current_tool = self._compute_ee_pos_local()[env_ids].detach()
        if self._prev_tool_pos_truck is None:
            self._prev_tool_pos_truck = self._compute_ee_pos_local().detach().clone()
        else:
            self._prev_tool_pos_truck[env_ids] = current_tool
        self._prev_head_alignment[env_ids] = self._measure_head_down_alignment()[env_ids]
        self._previous_error_norm[env_ids] = torch.linalg.norm(
            self._original_reference_tool[env_ids] - current_tool, dim=-1
        )
