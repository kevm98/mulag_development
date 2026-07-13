"""RL residual on top of the verified map-seeded cascade controller.

Action (3): (dy, dz) correction on the nominal mapped-arc goal, plus a truck
speed command applied teacher_student-style (direct wheel velocity write --
drive-independent).  Zero action == the verified stationary controller baseline.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from ID_FD_pipline.dls_solver import CylinderDLSController, EECascadeController
from ID_FD_pipline.dls_solver.cylinder import CylinderDLSConfig
from ID_FD_pipline.dls_solver.ground_paths import GroundPathPlanner
from ID_FD_pipline.tasks.cylinder_env import IDFDCylinderSpaceEnv
from ID_FD_pipline.ppo1.observations import build_residual_observation
from ID_FD_pipline.ppo1.rewards import compute_residual_reward

_DEFAULT_MAP = Path(__file__).resolve().parents[1] / "models/ground_band_map.json"


class IDFDEEResidualEnv(IDFDCylinderSpaceEnv):
    """Small (dy, dz) goal correction + truck speed over the mapped-arc cascade."""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self._residual_scale_m = float(getattr(cfg, "residual_action_scale_m", 0.05))
        self._residual_action = torch.zeros((self.num_envs, 3), device=self.device)
        self._previous_residual_action = torch.zeros_like(self._residual_action)
        self._corrected_ee_goal_local = torch.zeros((self.num_envs, 3), device=self.device)
        self._previous_ee_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self._ee_velocity_local = torch.zeros((self.num_envs, 3), device=self.device)

        # Nominal reference = the physics-mapped ground arc (the branch the
        # passing long-trajectory gate tracks), pre-sampled at 2 cm spacing.
        planner = GroundPathPlanner(str(getattr(cfg, "residual_map", _DEFAULT_MAP)))
        path = planner.arc()
        self._path_goals = torch.as_tensor(path.goals, device=self.device, dtype=torch.float32)
        self._path_seeds = torch.as_tensor(path.cylinder_seeds, device=self.device, dtype=torch.float32)
        self._path_direct_targets = torch.as_tensor(path.direct_targets, device=self.device, dtype=torch.float32)
        self._path_step_m = float(path.length_m / max(len(path.goals) - 1, 1))
        self._path_len_m = float(path.length_m)
        self._nominal_seed = self._path_seeds[0].expand(self.num_envs, 4).clone()

        # Same calibrated controller stack as the passing gate.
        controller = CylinderDLSController(
            self,
            CylinderDLSConfig(
                invert_learned_pipeline=False,
                error_gain=0.10,
                max_desired_delta_m=0.01,
                max_action=1.0,
                action_smoothing=1.0,
            ),
        )
        self._ee_cascade = EECascadeController(self, cylinder_controller=controller)

    @torch.no_grad()
    def _update_residual_trajectory(self) -> None:
        """Advance the nominal goal + IK seed along the mapped arc."""
        settle = int(getattr(self.cfg, "residual_settle_steps", 150))
        speed = float(getattr(self.cfg, "residual_path_speed_mps", 0.045))
        progress_steps = torch.clamp(self.episode_length_buf - settle, min=0)
        distance = progress_steps.float() * float(self.step_dt) * speed
        idx = torch.clamp(
            (distance / max(self._path_step_m, 1.0e-9)).long(),
            0,
            self._path_goals.shape[0] - 1,
        )
        self._ee_goal_local = self._path_goals[idx]
        self._nominal_seed = self._path_seeds[idx]

    @torch.no_grad()
    def controller_targets(self, residual_action: torch.Tensor):
        """Pure mapping used by the zero-residual equivalence contract."""
        residual = torch.clamp(residual_action[:, :2], -1.0, 1.0) * self._residual_scale_m
        corrected = self._ee_goal_local.clone()
        corrected[:, 1:3] += residual
        cylinder_goal = self._ee_cascade.cylinder_goal_seeded(
            corrected,
            self._nominal_seed,
            correction_gain=float(getattr(self.cfg, "residual_ee_correction_gain", 0.40)),
        )
        return corrected, cylinder_goal

    @torch.no_grad()
    def _effective_truck_span(self) -> float:
        """Stage D: map curriculum phase to effective truck span."""
        phase = int(getattr(self.cfg, "residual_truck_curriculum_phase", 0))
        full_span = float(getattr(self.cfg, "residual_truck_span_kmh", 4.5))
        # Phase 0: truck disabled (arm-only). 1: slow. 2: full.
        phase_spans = {0: 0.0, 1: 2.0, 2: full_span}
        return phase_spans.get(phase, full_span)

    @torch.no_grad()
    def _apply_truck_speed(self, speed_action: torch.Tensor) -> None:
        """teacher_student-style direct wheel velocity write (drive-independent)."""
        base = float(getattr(self.cfg, "residual_truck_base_kmh", 0.0))
        span = self._effective_truck_span()
        vmax = float(getattr(self.cfg, "residual_truck_max_kmh", 6.0))
        kmh = torch.clamp(base + speed_action * span, 0.0, vmax)
        self._vehicle_speed_kmh = kmh
        radius = float(getattr(self.cfg, "vehicle_wheel_radius_m", 0.575))
        rad_s = (kmh / 3.6 / max(radius, 1.0e-6)).unsqueeze(-1)
        wheel = rad_s.expand(-1, len(self._wheel_joint_ids)).contiguous()
        self.robot.write_joint_velocity_to_sim(wheel, joint_ids=self._wheel_joint_ids)

    def _pre_physics_step(self, actions: torch.Tensor):
        action = torch.nan_to_num(actions.to(self.device, torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if action.shape != (self.num_envs, 3):
            raise RuntimeError(f"Residual action must be ({self.num_envs}, 3), got {tuple(action.shape)}")
        settle = int(getattr(self.cfg, "residual_settle_steps", 150))
        settling = self.episode_length_buf < settle
        self._update_residual_trajectory()
        self._previous_residual_action.copy_(self._residual_action)
        self._residual_action = torch.clamp(action, -1.0, 1.0)
        self._residual_action[settling] = 0.0
        corrected, cylinder_goal = self.controller_targets(self._residual_action)
        self._corrected_ee_goal_local = corrected
        cylinder_action, _ = self._ee_cascade.cylinder.action(cylinder_goal)
        self._cylinder_goal_x[:] = cylinder_goal
        self._ppo_raw_action = cylinder_action
        self._current_action = cylinder_action
        self._apply_truck_speed(self._residual_action[:, 2])
        self._run_cylinder_id_pipeline()
        if bool(settling.any()):
            # DIRECT pre-position phase (ported from the passing gate): override
            # the pipeline's PhysX target with the path-start direct target so the
            # plant descends to the mapped branch during the unscored settle.
            ids = torch.where(settling)[0]
            scaling = self._cylinder_adapter.scaling
            q_start = scaling.clamp_usd_q(
                scaling.learned_x_to_usd_q(self._path_direct_targets[0].expand(len(ids), 4))
            )
            self._cylinder_pos_target[ids] = q_start
        # HANDOFF at the settle->mow transition (the gate does exactly this;
        # skipping it left the integrator/valve state inconsistent -> blowup):
        handoff = self.episode_length_buf == settle
        if bool(handoff.any()):
            ids = torch.where(handoff)[0]
            measured_q = self.robot.data.joint_pos[:, self._cylinder_joint_ids].detach()
            self._cylinder_pos_target[ids] = measured_q[ids]
            self._prev_valve_cmd[ids] = 0.0
            self._ee_cascade.cylinder.previous_action[ids] = 0.0
            self._ee_cascade.reset(ids)

    def _get_observations(self) -> dict:
        ee = self._compute_ee_pos_local()
        dt = max(float(self.step_dt), 1.0e-6)
        self._ee_velocity_local = (ee - self._previous_ee_pos) / dt
        self._previous_ee_pos = ee.detach().clone()
        x, xdot = self._refresh_cylinder_state()
        yaw_rate = self.robot.data.root_ang_vel_w[:, 2]
        observation = build_residual_observation(
            ee_pos_local=ee,
            nominal_goal_local=self._ee_goal_local,
            ee_velocity_local=self._ee_velocity_local,
            cylinder_x=x,
            cylinder_xdot=xdot,
            vehicle_speed_kmh=self._vehicle_speed_kmh,
            vehicle_yaw_rate=yaw_rate,
            previous_residual_action=self._previous_residual_action,
            cylinder_goal_x=self._cylinder_goal_x,
            current_valve_cmd=self._current_valve_cmd,
            fd_delta_corrected=self._fd_delta_corrected,
        )
        return {"policy": observation}

    def _get_rewards(self) -> torch.Tensor:
        ee = self._compute_ee_pos_local()
        reward, error, reward_extras = compute_residual_reward(
            ee,
            self._ee_goal_local,
            error_scale=float(getattr(self.cfg, "residual_error_reward_scale", 20.0)),
            truck_kmh=self._vehicle_speed_kmh,
            progress_weight=float(getattr(self.cfg, "residual_progress_weight", 1.5)),
            progress_error_gate_m=float(getattr(self.cfg, "residual_progress_gate_m", 0.05)),
            truck_max_kmh=float(getattr(self.cfg, "residual_truck_max_kmh", 6.0)),
            progress_decay_m=float(getattr(self.cfg, "residual_progress_decay_m", 0.10)),
            action_smooth_weight=float(getattr(self.cfg, "residual_action_smooth_weight", 0.01)),
            current_residual_action=self._residual_action,
            previous_residual_action=self._previous_residual_action,
        )
        self.extras["log"] = {
            "residual/yz_error_mean": error.mean(),
            "residual/action_abs_mean": self._residual_action[:, :2].abs().mean(),
            "residual/truck_kmh_mean": self._vehicle_speed_kmh.mean(),
            **reward_extras,
        }
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        timeout = self.episode_length_buf >= self.max_episode_length
        cylinder_q = self.robot.data.joint_pos[:, self._cylinder_joint_ids]
        limits = self.robot.data.soft_joint_pos_limits[:, self._cylinder_joint_ids]
        nonfinite = ~(
            torch.isfinite(cylinder_q).all(dim=-1)
            & torch.isfinite(self._current_valve_cmd).all(dim=-1)
            & torch.isfinite(self._fd_delta_corrected).all(dim=-1)
        )
        limit = ((cylinder_q < limits[..., 0]) | (cylinder_q > limits[..., 1])).any(dim=-1)
        return limit | nonfinite, timeout

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        env_ids = env_ids.to(self.device, torch.long)
        ee = self._compute_ee_pos_local()
        self._residual_action[env_ids] = 0.0
        self._previous_residual_action[env_ids] = 0.0
        self._previous_ee_pos[env_ids] = ee[env_ids]
        self._ee_velocity_local[env_ids] = 0.0
        self._ee_cascade.reset(env_ids)
        self._update_residual_trajectory()
        self._corrected_ee_goal_local[env_ids] = self._ee_goal_local[env_ids]
