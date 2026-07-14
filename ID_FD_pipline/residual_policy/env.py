"""Residual PPO policy environment for the MULAG hydraulic mower arm.

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

from ID_FD_pipline.mdp.actions import sanitize_cylinder_action
from ID_FD_pipline.tasks.cylinder_env import IDFDCylinderSpaceEnv
from ID_FD_pipline.dls_solver.cylinder import CylinderDLSConfig, CylinderDLSController
from ID_FD_pipline.dls_solver.ee_cascade import EECascadeController
from ID_FD_pipline.trajectories.time_scaling import cruise_profile
from ID_FD_pipline.residual_policy.observations import (
    RESIDUAL_OBS_DIM,
    build_residual_observation,
)
from ID_FD_pipline.residual_policy.actions import apply_residual_action
from ID_FD_pipline.residual_policy.rewards import (
    ResidualRewardTerms,
    compute_residual_reward,
)

ROOT = Path(__file__).resolve().parents[2]


class ResidualPolicyEnv(IDFDCylinderSpaceEnv):
    """Residual PPO policy environment.

    Overrides observation, reward, termination, reset, and pre-physics step
    to implement the residual-policy contract while reusing the validated
    ID/FD lower-level controller pipeline.
    """

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)

        # --- Trajectory data ---
        self._arc_path = Path(str(getattr(cfg, "arc_path", ""))).expanduser().resolve()
        self._load_trajectory()

        # --- Residual action state ---
        self._residual_raw = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self._residual_filtered = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self._prev_residual_filtered = torch.zeros_like(self._residual_filtered)
        self._baseline_cylinder_goal = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self._corrected_cylinder_goal = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)

        # --- Tool observation history ---
        self._prev_tool_pos_truck: torch.Tensor | None = None
        self._measured_tool_velocity_truck = torch.zeros((self.num_envs, 3), device=self.device)

        # --- Reward state ---
        self._previous_error_norm = torch.zeros((self.num_envs,), device=self.device)

        # --- Head-down history ---
        self._prev_head_alignment = torch.zeros((self.num_envs,), device=self.device)
        self._head_alignment_rate = torch.zeros((self.num_envs,), device=self.device)

        # --- Trajectory state ---
        zero_t = torch.zeros(1, device=self.device)
        _, _, _, one_way_s = cruise_profile(
            zero_t,
            length_m=self._arc_length,
            cruise_speed_mps=float(getattr(cfg, "path_speed_mps", 0.13)),
            accel_time_s=float(getattr(cfg, "accel_time_s", 1.0)),
            decel_time_s=float(getattr(cfg, "accel_time_s", 1.0)),
        )
        self._one_way_s = one_way_s
        self._complete_cycle_time = 2.0 * one_way_s

        # Per-environment trajectory state
        self._traj_direction = torch.ones((self.num_envs,), device=self.device)
        self._traj_start_direction = torch.ones((self.num_envs,), device=self.device)
        self._traj_start_s = torch.zeros((self.num_envs,), device=self.device)
        self._traj_elapsed = torch.zeros((self.num_envs,), device=self.device)
        self._scored_elapsed = torch.zeros((self.num_envs,), device=self.device)
        self._traj_phase = torch.zeros((self.num_envs,), device=self.device)
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
        self._wheel_init_pos: torch.Tensor | None = None
        self._truck_stationary_initialized = False

        # --- Cylinder DLS controllers ---
        self._cylinder_dls: CylinderDLSController | None = None
        self._cascade: EECascadeController | None = None
        self._setup_dls_controllers()

        # --- Reference tool position (kept invariant for reward) ---
        self._original_reference_tool = torch.zeros((self.num_envs, 3), device=self.device)

        print(
            f"[ResidualPolicy] arc={self._arc_length:.3f}m "
            f"one_way={self._one_way_s:.2f}s "
            f"obs_dim={RESIDUAL_OBS_DIM} "
            f"residual_scale={float(getattr(cfg, 'residual_scale_m', 0.002)):.4f}m",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Trajectory loading
    # ------------------------------------------------------------------

    def _load_trajectory(self) -> None:
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
        self._arc_length = float(self._teacher_s[-1])
        if self._arc_length < 3.0:
            raise RuntimeError(f"Verified ground arc is only {self._arc_length:.3f} m")

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

    # ------------------------------------------------------------------
    # DLS controller setup
    # ------------------------------------------------------------------

    def _setup_dls_controllers(self) -> None:
        ground_j_path = ROOT / "ID_FD_pipline/models/ee_jacobian_ground_truck3x4.json"
        ground_j = torch.tensor(
            json.loads(ground_j_path.read_text())["jacobian"],
            device=self.device,
            dtype=torch.float32,
        )
        ground_j[:, 3] = 0.0  # cylinder 4 owns head pitch

        self._cylinder_dls = CylinderDLSController(
            self,
            CylinderDLSConfig(
                invert_learned_pipeline=True,
                error_gain=0.08,
                velocity_damping=0.0,
                velocity_feedforward=0.0,
                max_desired_delta_m=0.004,
                max_action=1.0,
                action_smoothing=0.5,
            ),
        )
        self._cascade = EECascadeController(
            self,
            cylinder_controller=self._cylinder_dls,
            jacobian=ground_j,
            ee_feedback_gain=0.0,
            max_ik_step_m=0.004,
            max_cylinder_lead_m=0.025,
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
        truck_base = self.robot.data.body_link_state_w[:, self._truck_base_body_id(), :3]
        self._truck_anchor_xy = truck_base[:, :2].detach().clone()
        self._truck_stationary_initialized = True

    @torch.no_grad()
    def _hold_wheels(self) -> None:
        ids = self._wheel_joint_ids
        self.robot.write_joint_stiffness_to_sim(2.0e9, joint_ids=ids)
        self.robot.write_joint_damping_to_sim(1.0e7, joint_ids=ids)
        if self._wheel_init_pos is not None:
            self.robot.set_joint_position_target(self._wheel_init_pos, joint_ids=ids)
        self.robot.set_joint_velocity_target(
            torch.zeros((self.num_envs, len(ids)), device=self.device), joint_ids=ids
        )

    @torch.no_grad()
    def _constrain_truck_xy(self) -> None:
        base_xy = self.robot.data.body_link_state_w[:, self._truck_base_body_id(), :2]
        correction = self._truck_anchor_xy - base_xy
        drift_norm = torch.linalg.norm(correction, dim=1)
        self._truck_drift_xy = drift_norm
        self._max_truck_drift = torch.maximum(self._max_truck_drift, drift_norm)

        mask = drift_norm > 0.001
        if not mask.any():
            return
        root_pose = self.robot.data.root_state_w[:, :7].detach().clone()
        root_pose[mask, :2] += correction[mask]
        self.robot.write_root_pose_to_sim(root_pose)

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
        if tuple(actions.shape) != (int(self.num_envs), 3):
            raise RuntimeError(
                f"Residual policy action must be ({self.num_envs}, 3), got {tuple(actions.shape)}"
            )

        force_zero = bool(getattr(cfg, "force_zero_policy", False))
        residual_scale_m = float(getattr(cfg, "residual_scale_m", 0.002))
        action_filter = float(getattr(cfg, "action_filter", 0.8))

        # Deployment and settling use the same direct PhysX target path as the
        # validated ground-arc gate.  They are deliberately outside the learned
        # ID/FD controller: the policy is not scored and cannot act until the arm
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
        self._residual_raw, self._residual_filtered, self._corrected_cylinder_goal = (
            apply_residual_action(
                baseline_goal=self._baseline_cylinder_goal,
                previous_filtered_action=self._residual_filtered,
                policy_action=effective_action,
                residual_scale_m=residual_scale_m,
                filter_beta=action_filter,
                learned_min=learned_min,
                learned_max=learned_max,
            )
        )
        self._residual_filtered[warmup_mask] = 0.0
        self._corrected_cylinder_goal[warmup_mask] = self._baseline_cylinder_goal[warmup_mask]

        # Step 3/4: only active environments enter the production ID/FD stack.
        # The parent pipeline is vectorized over all envs, so compute it once and
        # restore direct deployment targets for warmup envs afterwards.
        action_4d, _sv = self._cylinder_dls.action(self._corrected_cylinder_goal)
        self._cylinder_goal_x[:] = self._corrected_cylinder_goal.clone()
        self._ppo_raw_action = sanitize_cylinder_action(
            action_4d, num_envs=self.num_envs, action_dim=4
        ).to(self.device)
        self._ppo_raw_action[warmup_mask] = 0.0
        self._current_action = self._ppo_raw_action.clone()
        if (~warmup_mask).any():
            direct_q = self._cylinder_pos_target[warmup_mask].clone()
            direct_joint_ref = self._joint_ref[warmup_mask].clone()
            self._run_cylinder_id_pipeline()
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
        self._cylinder_id.reset(env_ids)
        self._cylinder_fd.reset(env_ids)
        self._cylinder_dls.reset(env_ids)
        self._cascade.reset(env_ids)

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
            raise RuntimeError("residual-policy trajectory environments must advance synchronously")

        goals = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        seeds = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        for env_id in range(self.num_envs):
            if not not_warmup[env_id]:
                continue
            d_val, signed_speed, _signed_accel, direction = self._trajectory_state(
                env_id, float(self._traj_elapsed[env_id])
            )
            s_val = abs(signed_speed)
            self._traj_direction[env_id] = direction
            self._reference_path_s[env_id] = d_val
            self._reference_path_speed[env_id] = signed_speed
            self._reference_path_accel[env_id] = _signed_accel

            goal_np, _seed_at_goal, _teacher_wz = self._interpolate_trajectory(d_val)
            seed_distance = d_val + direction * abs(s_val) * seed_lead
            _seed_goal, seed_np, _seed_wz = self._interpolate_trajectory(seed_distance)

            goals[env_id] = torch.as_tensor(goal_np, device=self.device)
            seeds[env_id] = torch.as_tensor(seed_np, device=self.device)

            path_fraction = d_val / max(self._arc_length, 1.0e-9)
            phase = 0.5 * path_fraction if direction > 0.0 else 0.5 + 0.5 * (1.0 - path_fraction)
            self._traj_phase[env_id] = phase

        self._original_reference_tool[:] = goals
        cylinder_goal = self._cascade.cylinder_goal_seeded(
            goals,
            seeds,
            correction_gain=float(getattr(cfg, "ee_correction_gain", 0.0)),
        )
        cylinder_goal[:, 3] = seeds[:, 3]
        self._baseline_cylinder_goal[:] = cylinder_goal

    @torch.no_grad()
    def _trajectory_state(self, env_id: int, elapsed: float) -> tuple[float, float, float, float]:
        """Smooth path state from an arbitrary sampled start pose.

        The first segment always accelerates from rest toward the selected
        endpoint. Subsequent full-length legs use the validated A<->B cruise
        profile. This avoids a nonphysical velocity jump at randomized resets.
        """
        path_speed = float(getattr(self.cfg, "path_speed_mps", 0.13))
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
        dt = max(float(self.step_dt), 1.0e-6)

        tool_pos_truck = self._compute_ee_pos_local()

        if self._prev_tool_pos_truck is not None:
            self._measured_tool_velocity_truck = (
                tool_pos_truck - self._prev_tool_pos_truck
            ) / dt
        self._prev_tool_pos_truck = tool_pos_truck.detach().clone()

        tool_error_truck = self._original_reference_tool - tool_pos_truck

        ref_vel = torch.zeros((self.num_envs, 3), device=self.device)
        ref_acc = torch.zeros((self.num_envs, 3), device=self.device)
        for env_id in range(self.num_envs):
            path_distance = float(self._reference_path_s[env_id].item())
            signed_speed = float(self._reference_path_speed[env_id].item())
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

        x, xdot = self._refresh_cylinder_state()
        self._cylinder_x_current = x.detach().clone()
        self._cylinder_xdot_current = xdot.detach().clone()
        cylinder_goal_error = self._corrected_cylinder_goal - x

        phase_angle = 2.0 * math.pi * self._traj_phase
        phase_sin_cos = torch.stack(
            [torch.sin(phase_angle), torch.cos(phase_angle)], dim=-1
        )

        head_alignment = self._measure_head_down_alignment()
        self._head_alignment_rate = (
            head_alignment - self._prev_head_alignment
        ) / dt
        self._prev_head_alignment = head_alignment.detach().clone()

        cutter_clearance = self._compute_cutter_world_z()

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
            trajectory_phase_sin_cos=phase_sin_cos,
            trajectory_direction=self._traj_direction,
            cutter_bottom_world_clearance=cutter_clearance,
            head_down_alignment=head_alignment,
            head_down_alignment_rate=self._head_alignment_rate,
        )
        return {"policy": obs}

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------

    def _get_rewards(self) -> torch.Tensor:
        cfg = self.cfg
        tool_pos_truck = self._compute_ee_pos_local()
        head_alignment = self._measure_head_down_alignment()
        cutter_clearance = self._compute_cutter_world_z()

        terms: ResidualRewardTerms = compute_residual_reward(
            original_reference_tool_position=self._original_reference_tool,
            measured_tool_position=tool_pos_truck,
            previous_error_norm=self._previous_error_norm,
            filtered_action=self._residual_filtered,
            previous_filtered_action=self._prev_residual_filtered,
            head_down_alignment=head_alignment,
            cutter_clearance=cutter_clearance,
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
        )
        active = (self._preposition_counter <= 0) & (self._settle_counter <= 0)
        reward = torch.where(active, terms.reward, torch.zeros_like(terms.reward))
        self._previous_error_norm = torch.where(
            active, terms.error_norm, self._previous_error_norm
        )

        self._last_reward_terms = {
            "reward/total": torch.mean(terms.reward),
            "reward/tracking": torch.mean(terms.tracking),
            "reward/progress": torch.mean(terms.progress),
            "reward/action_magnitude": torch.mean(terms.action_magnitude),
            "reward/action_rate": torch.mean(terms.action_rate),
            "reward/head_penalty": torch.mean(terms.head_penalty),
            "reward/ground_penalty": torch.mean(terms.ground_penalty),
            "reward/saturation_penalty": torch.mean(terms.saturation_penalty),
            "error/mean": torch.mean(terms.error_norm),
            "error/p95": torch.quantile(terms.error_norm, 0.95),
            "error/max": torch.max(terms.error_norm),
            "head/alignment_min": torch.min(head_alignment),
            "truck/drift_max": torch.max(self._max_truck_drift),
            "residual/mean_abs": torch.mean(torch.abs(self._residual_filtered)),
            "residual/max_abs": torch.max(torch.abs(self._residual_filtered)),
        }
        self.extras.setdefault("log", {}).update(self._last_reward_terms)
        return reward

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
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

        truck_drift_exceeded = self._truck_drift_xy > float(
            getattr(cfg, "truck_drift_termination_m", 0.02)
        )

        head_alignment = self._measure_head_down_alignment()
        head_term = head_alignment < float(
            getattr(cfg, "head_down_termination_alignment", 0.95)
        )

        cutter_z = self._compute_cutter_world_z()
        hard_band = float(getattr(cfg, "cutter_clearance_hard_band_m", 0.15))
        clearance_violation = torch.abs(cutter_z) > hard_band

        terminated = (
            limit_violation
            | nonfinite
            | truck_drift_exceeded
            | head_term
            | clearance_violation
        )
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
        })
        return terminated, truncated

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

        # Reset residual state
        self._residual_raw[env_ids] = 0.0
        self._residual_filtered[env_ids] = 0.0
        self._prev_residual_filtered[env_ids] = 0.0
        self._baseline_cylinder_goal[env_ids] = 0.0
        self._corrected_cylinder_goal[env_ids] = 0.0

        # Reset observation history
        self._measured_tool_velocity_truck[env_ids] = 0.0
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
            start_s = torch.rand((n,), device=self.device) * self._arc_length
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
        self._reference_path_speed[env_ids] = 0.0
        self._reference_path_accel[env_ids] = 0.0
        self._traj_phase[env_ids] = torch.where(
            start_direction > 0.0,
            0.5 * start_s / self._arc_length,
            0.5 + 0.5 * (1.0 - start_s / self._arc_length),
        )

        target_seed = torch.zeros((n, 4), dtype=torch.float32, device=self.device)
        target_goal = torch.zeros((n, 3), dtype=torch.float32, device=self.device)
        for row, initial_s in enumerate(start_s.detach().cpu().tolist()):
            goal_np, seed_np, _ = self._interpolate_trajectory(float(initial_s))
            target_goal[row] = torch.as_tensor(goal_np, device=self.device)
            target_seed[row] = torch.as_tensor(seed_np, device=self.device)

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

        # Init truck stationarity on first reset
        self._init_truck_stationarity()
        self._run_reset_preposition(env_ids)
        current_tool = self._compute_ee_pos_local()[env_ids].detach()
        if self._prev_tool_pos_truck is None:
            self._prev_tool_pos_truck = self._compute_ee_pos_local().detach().clone()
        else:
            self._prev_tool_pos_truck[env_ids] = current_tool
        self._prev_head_alignment[env_ids] = self._measure_head_down_alignment()[env_ids]
        self._previous_error_norm[env_ids] = torch.linalg.norm(
            self._original_reference_tool[env_ids] - current_tool, dim=-1
        )
