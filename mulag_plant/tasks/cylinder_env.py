from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.utils.math import quat_apply, quat_apply_inverse

from mulag_plant.mdp.actions import sanitize_cylinder_action
from mulag_plant.mdp.ee_observations import build_ee_tracking_observation
from mulag_plant.mdp.ee_rewards import compute_ee_tracking_reward
from mulag_plant.mdp.observations import CYLINDER_OBS_DIM, build_cylinder_observation
from mulag_plant.mdp.pipeline import pipeline_summary
from mulag_plant.mdp.rewards import compute_cylinder_reward
from mulag_plant.runtime.mulag_upper_level_env import MulagUpperLevelEnv


class PPO2ROMPhysXEnv(MulagUpperLevelEnv):
    """PPO2 task using the command ROM instead of the complete ID/FD pair.
    Supports both cylinder-space tracking and EE-space tracking via config.task_type."""

    #: Dynamics models this task is allowed to construct.
    #:
    #: PPO2 is the ROM task and stays ROM-only, which is what the guards below
    #: enforce. A subclass that deliberately runs this same task over a
    #: different actuator (PPO4 runs it over ID/FD) widens this set; everything
    #: else about the task, including every ROM-specific prohibition, then
    #: applies exactly as before. Keeping the default at command_rom means PPO2
    #: and PPO3 behaviour is bit-identical to before this attribute existed.
    _supported_dynamics_models: frozenset = frozenset({"command_rom"})

    #: Models that carry their own internal controller and therefore must not be
    #: combined with the ID/FD residual switches.
    _rom_dynamics_models: frozenset = frozenset({"command_rom", "base_rom", "rom"})

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        dynamics = str(cfg.cylinder_dynamics_model).lower()
        if dynamics not in self._supported_dynamics_models:
            raise ValueError(
                f"{type(self).__name__} supports cylinder_dynamics_model "
                f"{sorted(self._supported_dynamics_models)}; got {dynamics!r}."
            )
        if bool(getattr(cfg, "use_sequence_cylinder_models", False)):
            raise ValueError("PPO2-ROM-PhysX cannot construct sequence ID/FD models.")
        if str(getattr(cfg, "low_level_servo_mode", "none")).lower() != "none":
            raise ValueError("PPO2-ROM-PhysX uses the ROM's internal controller; external PID/servo must be off.")
        if dynamics in self._rom_dynamics_models:
            # ROM-only prohibitions. The ROM predicts its own valve, so an ID/FD
            # residual switch layered on top would silently mix two actuator
            # models. Under a genuine ID/FD actuator these same switches are the
            # validated contract, not a misconfiguration.
            forbidden_switches = (
                "id_residual_over_zero_target",
                "fd_residual_over_zero_valve",
                "paired_id_fd_residual",
            )
            enabled = [name for name in forbidden_switches if bool(getattr(cfg, name, False))]
            if enabled:
                raise ValueError(f"PPO2-ROM-PhysX forbids ID/FD switches: {enabled}")
            if float(getattr(cfg, "cylinder_idfd_reference_blend", 0.0)) != 0.0:
                raise ValueError("PPO2-ROM-PhysX forbids the ID/FD reference blend.")
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self._prev_action = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self._current_action = torch.zeros_like(self._prev_action)
        self._ppo_raw_action = torch.zeros_like(self._prev_action)
        self._cylinder_goal_x = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self._previous_goal_error_norm = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self._success_counter = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self._last_reward_terms = {}
        self._task_type = str(getattr(cfg, "task_type", "cylinder")).lower()
        if self._task_type == "ee_tracking":
            self._ee_goal_local = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
            self._ee_pos_local = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
            # Moving-goal trajectory state (base-frame path anchored at reset EE pos).
            self._traj_mode = str(getattr(cfg, "ee_trajectory_mode", "point")).lower()
            self._traj_anchor = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
            self._traj_u = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)  # primary dir
            self._traj_v = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)  # perp dir
        obs_dim = int(getattr(cfg, "observation_space", CYLINDER_OBS_DIM))
        action_dim = int(getattr(cfg, "action_space", 4))
        dynamics = str(getattr(cfg, "cylinder_dynamics_model", "command_rom")).lower()
        if dynamics in {"base_rom", "command_rom"}:
            label = "desired cylinder command -> internal valve -> hydraulic ROM" if dynamics == "command_rom" else "policy valve -> base hydraulic ROM"
            print(
                f"[PPO2-ROM-PhysX] {self._task_type} pipeline: policy_obs={obs_dim} "
                f"policy_action={action_dim} -> {label} -> PhysX | ID/FD disabled",
                flush=True,
            )
        else:
            summary = pipeline_summary()
            print(
                f"[ID_FD] {self._task_type} pipeline: "
                f"policy_obs={obs_dim} policy_action={action_dim} actuator_action=4 | "
                f"{summary.inverse_dynamics} | {summary.forward_dynamics} | "
                f"{summary.disabled_models}",
                flush=True,
            )

    def _compute_cutter_pos_w(self) -> torch.Tensor:
        """Live cutter position in world coordinates. Returns ``(N, 3)``.

        The controlled point is the CUTTER, not the Messerkopf body origin. The
        cutter geometry hangs 0.79 m BELOW that origin (measured:
        the measured local AABB z in [-0.792, -0.234], head
        0.558 m tall). Commanding the origin put every trajectory 0.79 m above the
        grass, which is why the arm never mowed: the origin's lowest reach is
        world z=0.664 m, but the CUTTER at that pose is at z=-0.13 m, i.e. on/below
        the ground. The offset is applied in the live EE-body frame. Keep this
        method as the single source of truth for world-height gates and control.
        """
        ee_pos_w = self.robot.data.body_link_state_w[:, self._ee_body_id, :3]
        offset = float(getattr(self.cfg, "tool_tip_local_offset_z", -0.79))
        if offset != 0.0:
            ee_quat_w = self.robot.data.body_link_state_w[:, self._ee_body_id, 3:7]
            ee_q = ee_quat_w / (torch.norm(ee_quat_w, dim=-1, keepdim=True).clamp(min=1e-8))
            tip_body = torch.zeros_like(ee_pos_w)
            tip_body[:, 2] = offset
            ee_pos_w = ee_pos_w + quat_apply(ee_q, tip_body)
        return torch.nan_to_num(ee_pos_w, nan=0.0, posinf=0.0, neginf=0.0)

    def _compute_cutter_world_z(self) -> torch.Tensor:
        """Cutter-bottom world height from the authored local-bound Z extent.

        ``tool_tip_local_offset_z`` came from ``ComputeLocalBound`` and only its
        Z component is physically trustworthy.  This intentionally preserves
        the passing mowing-clearance contract (body-origin world Z plus the
        local cutter-bottom Z) while ``_compute_cutter_pos_w`` remains the
        body-fixed control point used by the cached stepped Jacobian.
        """
        origin_z = self.robot.data.body_link_state_w[:, self._ee_body_id, 2]
        offset = float(getattr(self.cfg, "tool_tip_local_offset_z", -0.79))
        return torch.nan_to_num(origin_z + offset, nan=0.0, posinf=0.0, neginf=0.0)

    def _compute_ee_pos_local(self) -> torch.Tensor:
        """Live cutter position in the truck-chassis frame. Returns ``(N, 3)``."""
        ee_pos_w = self._compute_cutter_pos_w()

        # !!! Reference frame = TRUCK CHASSIS (base_link), NOT root_pos_w. !!!
        # The articulation ROOT is the `Drehzapfen` -- the ARM'S OWN SLEW BASE (verified:
        # root-frame calibration showed it coincides with Drehzapfen at d=0.0000 m. Using
        # the root meant every "base-frame" trajectory lived in a frame that ROTATES WITH
        # THE ARM, so:
        #   * EE-local x was IDENTICALLY 0 -> "base-X is null" / rank-2 EE Jacobian /
        #     BASE_YZ_PLUS_TRUCK_X were all ARTIFACTS of this frame, not kinematics;
        #   * slewing the Drehzapfen does not change the EE's coords in its own frame, so
        #     the PRIMARY MOWING DOF received no command -> the arm never moved and only
        #     the truck did.
        # In the truck frame the EE reads (5.48, -3.39, 0.95) -- a real 3-D position,
        # comparable to teacher_student's mowing commands (x 6.5-7.5, y 0.7-2.2, z 0.5).
        base_id = self._truck_base_body_id()
        base_pos_w = self.robot.data.body_link_state_w[:, base_id, :3]
        base_quat_w = self.robot.data.body_link_state_w[:, base_id, 3:7]
        base_q = base_quat_w / (torch.norm(base_quat_w, dim=-1, keepdim=True).clamp(min=1e-8))
        tcp_local = quat_apply_inverse(base_q, ee_pos_w - base_pos_w)
        return torch.nan_to_num(tcp_local, nan=0.0, posinf=0.0, neginf=0.0)

    def _truck_base_body_id(self) -> int:
        """Body index of the TRUCK CHASSIS (never the articulation root, which is the
        arm's Drehzapfen slew base)."""
        cached = getattr(self, "_truck_base_id_cache", None)
        if cached is not None:
            return cached
        names = self.robot.body_names
        for candidate in ("base_link", "Frontrahmen"):
            if candidate in names:
                self._truck_base_id_cache = names.index(candidate)
                return self._truck_base_id_cache
        raise RuntimeError(f"No truck chassis body found in {names}")

    def _sample_ee_goals(self, env_ids: torch.Tensor) -> None:
        """Sample EE position goals in base frame near current EE position."""
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        ee_now = self._ee_pos_local[env_ids]
        span = float(getattr(self.cfg, "ee_goal_delta_range_m", 0.15))
        delta = (2.0 * torch.rand((env_ids.numel(), 3), device=self.device) - 1.0) * span
        self._ee_goal_local[env_ids] = ee_now + delta
        err = torch.linalg.norm(self._ee_goal_local[env_ids] - ee_now, dim=-1)
        self._previous_goal_error_norm[env_ids] = err.detach()
        self._success_counter[env_ids] = 0

    def _step_dt(self) -> float:
        dt = getattr(self, "step_dt", None)
        if dt is None:
            dt = float(self.cfg.sim.dt) * float(getattr(self.cfg, "decimation", 1))
        return float(dt)

    def _sample_ee_trajectory(self, env_ids: torch.Tensor) -> None:
        """Anchor a base-frame path at the current EE pos with a random in-plane
        heading. The path lives in the base horizontal (x-y) plane (ground-parallel,
        mowing-like); u is the primary heading, v the in-plane perpendicular."""
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        n = env_ids.numel()
        self._traj_anchor[env_ids] = self._ee_pos_local[env_ids]
        theta = 2.0 * torch.pi * torch.rand((n,), device=self.device)
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        zeros = torch.zeros_like(cos_t)
        self._traj_u[env_ids] = torch.stack([cos_t, sin_t, zeros], dim=-1)
        self._traj_v[env_ids] = torch.stack([-sin_t, cos_t, zeros], dim=-1)
        self._success_counter[env_ids] = 0

    def _update_ee_trajectory_goal(self) -> None:
        """Advance _ee_goal_local along the anchored path at a fixed speed.
        s = elapsed path length = episode_step * step_dt * speed (per env)."""
        speed = float(getattr(self.cfg, "ee_trajectory_speed_mps", 0.06))
        s = (self.episode_length_buf.to(torch.float32) * self._step_dt() * speed).reshape(-1, 1)  # (N,1)
        u, v, anchor = self._traj_u, self._traj_v, self._traj_anchor
        if self._traj_mode == "line":
            length = float(getattr(self.cfg, "ee_trajectory_length_m", 0.30))
            frac = self._pingpong(s / max(length, 1e-6))  # in [0,1], back-and-forth
            self._ee_goal_local = anchor + frac * length * u
        elif self._traj_mode == "curve":
            radius = float(getattr(self.cfg, "ee_trajectory_radius_m", 0.15))
            ang = s / max(radius, 1e-6)  # arc-length -> angle
            self._ee_goal_local = anchor + radius * torch.sin(ang) * u + radius * (1.0 - torch.cos(ang)) * v
        elif self._traj_mode == "sweep":
            length = float(getattr(self.cfg, "ee_trajectory_length_m", 0.30))
            amp = float(getattr(self.cfg, "ee_trajectory_sweep_amp_m", 0.15))
            wavelength = float(getattr(self.cfg, "ee_trajectory_sweep_wavelength_m", 0.30))
            fwd = self._pingpong(s / max(length, 1e-6)) * length  # slow forward, bounded
            lat = amp * torch.sin(2.0 * torch.pi * s / max(wavelength, 1e-6))
            self._ee_goal_local = anchor + fwd * u + lat * v
        # "point" mode: _ee_goal_local is static (set by _sample_ee_goals); no-op.

    @staticmethod
    def _pingpong(p: torch.Tensor) -> torch.Tensor:
        """Triangle wave: maps path progress p>=0 to [0,1], back and forth."""
        frac = torch.remainder(p, 2.0)
        return torch.where(frac <= 1.0, frac, 2.0 - frac)

    def _sample_cylinder_goals(self, env_ids: torch.Tensor) -> None:
        """Sample cylinder extension goals near current extensions."""
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        x_now = self._cylinder_x_current[env_ids]
        span = float(getattr(self.cfg, "cylinder_goal_delta_range_m", 0.04))
        delta = (2.0 * torch.rand((env_ids.numel(), 4), device=self.device) - 1.0) * span
        self._cylinder_goal_x[env_ids] = x_now + delta
        err = torch.linalg.norm(self._cylinder_goal_x[env_ids] - x_now, dim=-1)
        self._previous_goal_error_norm[env_ids] = err.detach()
        self._success_counter[env_ids] = 0

    def _refresh_cylinder_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        cylinder_q = self.robot.data.joint_pos[:, self._cylinder_joint_ids]
        cylinder_qdot = self.robot.data.joint_vel[:, self._cylinder_joint_ids]
        if self._cylinder_adapter is None:
            return cylinder_q[:, :4], cylinder_qdot[:, :4]
        return self._cylinder_adapter.current_from_cylinder_joints(cylinder_q, cylinder_qdot)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._ppo_raw_action = sanitize_cylinder_action(actions, num_envs=self.num_envs, action_dim=4).to(self.device)
        self._current_action = self._ppo_raw_action.clone()
        self._run_cylinder_id_pipeline()

    def _get_observations(self) -> dict:
        if self._task_type == "ee_tracking":
            return self._get_observations_ee()
        return self._get_observations_cylinder()

    def _get_observations_cylinder(self) -> dict:
        x, xdot = self._refresh_cylinder_state()
        self._cylinder_x_current = x.detach().clone()
        self._cylinder_xdot_current = xdot.detach().clone()
        obs = build_cylinder_observation(
            cylinder_x=x,
            cylinder_xdot=xdot,
            cylinder_goal_x=self._cylinder_goal_x,
            prev_action=self._prev_action,
            prev_valve_cmd=self._prev_valve_cmd,
            fd_delta_x=self._fd_delta_corrected,
            vehicle_speed_kmh=self._vehicle_speed_kmh,
        )
        return {"policy": obs}

    def _get_observations_ee(self) -> dict:
        x, xdot = self._cylinder_adapter.current_from_cylinder_joints(
            self.robot.data.joint_pos[:, self._cylinder_joint_ids],
            self.robot.data.joint_vel[:, self._cylinder_joint_ids],
        )
        self._cylinder_x_current = x.detach().clone()
        self._cylinder_xdot_current = xdot.detach().clone()
        self._ee_pos_local = self._compute_ee_pos_local()
        if self._traj_mode != "point":
            self._update_ee_trajectory_goal()  # advance the moving goal once per step
        obs = build_ee_tracking_observation(
            ee_pos_local=self._ee_pos_local,
            ee_goal_local=self._ee_goal_local,
            cylinder_x=x,
            cylinder_xdot=xdot,
            prev_action=self._prev_action,
            prev_valve_cmd=self._prev_valve_cmd,
            fd_delta_x=self._fd_delta_corrected,
            vehicle_speed_kmh=self._vehicle_speed_kmh,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        if self._task_type == "ee_tracking":
            return self._get_rewards_ee_tracking()
        return self._get_rewards_cylinder()

    def _get_rewards_cylinder(self) -> torch.Tensor:
        x, _ = self._refresh_cylinder_state()
        terms = compute_cylinder_reward(
            cylinder_x=x,
            cylinder_goal_x=self._cylinder_goal_x,
            previous_goal_error_norm=self._previous_goal_error_norm,
            action=self._current_action,
            previous_action=self._prev_action,
            valve_cmd=self._current_valve_cmd,
            fd_delta_x=self._fd_delta_corrected,
            tracking_std_m=float(getattr(self.cfg, "cylinder_tracking_std_m", 0.04)),
            progress_weight=float(getattr(self.cfg, "cylinder_progress_weight", 4.0)),
            action_rate_weight=float(getattr(self.cfg, "cylinder_action_rate_weight", 0.02)),
            valve_weight=float(getattr(self.cfg, "cylinder_valve_weight", 0.01)),
            fd_delta_weight=float(getattr(self.cfg, "cylinder_fd_delta_weight", 0.1)),
        )
        self._previous_goal_error_norm = terms.goal_error_norm.detach().clone()
        within = terms.goal_error_norm < float(getattr(self.cfg, "cylinder_success_tolerance_m", 0.01))
        self._success_counter = torch.where(within, self._success_counter + 1, torch.zeros_like(self._success_counter))
        self._last_reward_terms = {
            "reward/cylinder_error": torch.mean(terms.goal_error_norm),
            "reward/tracking": torch.mean(terms.tracking),
            "reward/progress": torch.mean(terms.progress),
            "action/valve_abs": torch.mean(terms.valve_abs_mean),
            "action/rate_penalty": torch.mean(terms.action_rate),
            "model/fd_delta_abs": torch.mean(torch.abs(self._fd_delta_corrected)),
        }
        self.extras["log"] = self._last_reward_terms
        return terms.reward

    def _get_rewards_ee_tracking(self) -> torch.Tensor:
        self._ee_pos_local = self._compute_ee_pos_local()
        terms = compute_ee_tracking_reward(
            ee_pos_local=self._ee_pos_local,
            ee_goal_local=self._ee_goal_local,
            previous_goal_error_norm=self._previous_goal_error_norm,
            action=self._current_action,
            previous_action=self._prev_action,
            valve_cmd=self._current_valve_cmd,
            fd_delta_x=self._fd_delta_corrected,
            tracking_std_m=float(getattr(self.cfg, "ee_tracking_std_m", 0.05)),
            progress_weight=float(getattr(self.cfg, "ee_progress_weight", 10.0)),
            action_rate_weight=float(getattr(self.cfg, "ee_action_rate_weight", 0.02)),
            valve_weight=float(getattr(self.cfg, "ee_valve_weight", 0.01)),
            fd_delta_weight=float(getattr(self.cfg, "ee_fd_delta_weight", 0.1)),
        )
        self._previous_goal_error_norm = terms.goal_error_norm.detach().clone()
        within = terms.goal_error_norm < float(getattr(self.cfg, "ee_success_tolerance_m", 0.02))
        self._success_counter = torch.where(within, self._success_counter + 1, torch.zeros_like(self._success_counter))
        self._last_reward_terms = {
            "reward/ee_error": torch.mean(terms.goal_error_norm),
            "reward/tracking": torch.mean(terms.tracking),
            "reward/progress": torch.mean(terms.progress),
            "action/valve_abs": torch.mean(terms.valve_abs_mean),
            "action/rate_penalty": torch.mean(terms.action_rate),
            "model/fd_delta_abs": torch.mean(torch.abs(self._fd_delta_corrected)),
        }
        self.extras["log"] = self._last_reward_terms
        return terms.reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        timeout = self.episode_length_buf >= self.max_episode_length
        cylinder_q = self.robot.data.joint_pos[:, self._cylinder_joint_ids]
        cylinder_qdot = self.robot.data.joint_vel[:, self._cylinder_joint_ids]
        limits = self.robot.data.soft_joint_pos_limits[:, self._cylinder_joint_ids]
        limit_margin = float(getattr(self.cfg, "cylinder_limit_margin_m", 0.002))
        limit_violation = (
            (cylinder_q < limits[..., 0] - limit_margin)
            | (cylinder_q > limits[..., 1] + limit_margin)
        ).any(dim=-1)
        nonfinite = ~(
            torch.isfinite(cylinder_q).all(dim=-1)
            & torch.isfinite(cylinder_qdot).all(dim=-1)
            & torch.isfinite(self._current_valve_cmd).all(dim=-1)
            & torch.isfinite(self._fd_delta_corrected).all(dim=-1)
        )
        required = int(getattr(
            self.cfg,
            "ee_success_hold_steps" if self._task_type == "ee_tracking" else "cylinder_success_hold_steps",
            10,
        ))
        success = self._success_counter >= required
        terminated = limit_violation | nonfinite
        truncated = timeout | success
        log = self.extras.setdefault("log", {})
        log.update(
            {
                "termination/limit": torch.mean(limit_violation.float()),
                "termination/nonfinite": torch.mean(nonfinite.float()),
                "termination/success": torch.mean(success.float()),
                "termination/timeout": torch.mean(timeout.float()),
            }
        )
        return terminated, truncated

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if self._task_type == "ee_tracking":
            x, xdot = self._cylinder_adapter.current_from_cylinder_joints(
                self.robot.data.joint_pos[:, self._cylinder_joint_ids],
                self.robot.data.joint_vel[:, self._cylinder_joint_ids],
            )
        else:
            x, xdot = self._refresh_cylinder_state()
        self._cylinder_x_current[env_ids] = x[env_ids]
        self._cylinder_xdot_current[env_ids] = xdot[env_ids]
        self._prev_action[env_ids] = 0.0
        self._current_action[env_ids] = 0.0
        self._ppo_raw_action[env_ids] = 0.0
        if self._task_type == "ee_tracking":
            self._ee_pos_local = self._compute_ee_pos_local()
            if self._traj_mode != "point":
                self._sample_ee_trajectory(env_ids)
                self._update_ee_trajectory_goal()  # goal at s=0 (== anchor)
                err = torch.linalg.norm(self._ee_goal_local[env_ids] - self._ee_pos_local[env_ids], dim=-1)
                self._previous_goal_error_norm[env_ids] = err.detach()
            else:
                self._sample_ee_goals(env_ids)
        else:
            self._sample_cylinder_goals(env_ids)

    def run_learned_pipeline_preflight(self):
        stage = "ppo2_rom_physx_ee_tracking" if self._task_type == "ee_tracking" else "ppo2_rom_physx_cylinder"
        return {
            "passed": True,
            "preflight_stage": stage,
            "id_checkpoint": str(self._cylinder_id.checkpoint_path) if self._cylinder_id is not None else "",
            "fd_checkpoint": str(self._cylinder_fd.checkpoint_path) if self._cylinder_fd is not None else "",
            "rom_loaded": False,
            "sequence_loaded": False,
            "failure_category": None,
        }
