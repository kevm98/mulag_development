from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from kevin_integration.rl import AdaptiveRLLearnedController, AdaptiveRLLearnedControllerCfg, build_rl_obs
from kevin_integration.rl.action_adapter import clamp_and_sanitize
from kevin_integration.rl.reward_terms import (
    JointReachRewardWeights,
    joint_reach_reward,
    joint_reach_reward_debug,
    joint_safety_termination,
)
from kevin_integration.tasks.mulag_adaptive_rl_env_cfg import MulagAdaptiveRLJointReachEnvCfg


class MulagAdaptiveRLEnv(DirectRLEnv):
    cfg: MulagAdaptiveRLJointReachEnvCfg

    def __init__(self, cfg: MulagAdaptiveRLJointReachEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._state_joint_ids = self._find_joint_ids(self.cfg.state_joint_names)
        self._command_joint_ids = self._find_joint_ids(self.cfg.command_joint_names)
        if len(self._state_joint_ids) != 5:
            raise RuntimeError(f"Expected 5 state joints, got {len(self._state_joint_ids)}")
        if len(self._command_joint_ids) != 4:
            raise RuntimeError(f"Expected 4 command joints, got {len(self._command_joint_ids)}")
        self._q_tracking_joint_weights = torch.as_tensor(
            self.cfg.q_tracking_joint_weights, dtype=torch.float32, device=self.device
        )
        if self._q_tracking_joint_weights.shape != (5,):
            raise RuntimeError(
                f"q_tracking_joint_weights must have shape (5,), got {tuple(self._q_tracking_joint_weights.shape)}"
            )

        self._controller = AdaptiveRLLearnedController(
            AdaptiveRLLearnedControllerCfg(
                models_dir=self.cfg.models_dir,
                num_envs=self.num_envs,
                device=self.device,
                fallback_window=self.cfg.fallback_window,
                max_abs_valve=self.cfg.max_abs_valve,
                max_abs_torque=self.cfg.max_abs_torque,
                torque_adapter_mode=self.cfg.torque_adapter_mode,
                torque_adapter_preset=self.cfg.torque_adapter_preset,
                torque_adapter_scale=tuple(self.cfg.torque_adapter_scale),
                torque_adapter_bias=tuple(self.cfg.torque_adapter_bias),
                fd_torque_scale=self.cfg.fd_torque_scale,
                use_fd_residual_alpha=self.cfg.use_fd_residual_alpha,
                fd_residual_alpha=self.cfg.fd_residual_alpha,
                torque_rate_limit=self.cfg.torque_rate_limit,
                torque_lowpass_alpha=self.cfg.torque_lowpass_alpha,
                use_torque_rate_limit=self.cfg.use_torque_rate_limit,
                use_torque_lowpass=self.cfg.use_torque_lowpass,
                max_abs_pressure_delta=self.cfg.max_abs_pressure_delta,
                max_abs_fnet=self.cfg.max_abs_fnet,
                sd_output_mode=self.cfg.sd_output_mode,
            )
        )
        self._reward_weights = JointReachRewardWeights(
            q_tracking=self.cfg.rew_q_tracking,
            qdot=self.cfg.rew_qdot,
            action_smoothness=self.cfg.rew_action_smoothness,
            valve_effort=self.cfg.rew_valve_effort,
            progress=self.cfg.rew_progress,
            torque_effort=self.cfg.rew_torque_effort,
            alive=self.cfg.rew_alive,
            termination=self.cfg.rew_termination,
        )

        self._q_ref = torch.zeros((self.num_envs, 5), dtype=torch.float32, device=self.device)
        self._qdot_ref = torch.zeros((self.num_envs, 5), dtype=torch.float32, device=self.device)
        self._dP = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self._fnet = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self._prev_action = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self._current_action = torch.zeros_like(self._prev_action)
        self._action_delta = torch.zeros_like(self._prev_action)
        self._current_qddot_cmd = torch.zeros((self.num_envs, 5), dtype=torch.float32, device=self.device)
        self._prev_valve_cmd = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self._current_valve_cmd = torch.zeros_like(self._prev_valve_cmd)
        self._current_torque = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self._current_fd_simscape_output = torch.zeros_like(self._current_torque)
        self._current_tau_fd_adapted = torch.zeros_like(self._current_torque)
        self._current_tau_applied_raw = torch.zeros_like(self._current_torque)
        self._current_tau_isaac_filtered = torch.zeros_like(self._current_torque)
        self._current_tau_isaac_clamped = torch.zeros_like(self._current_torque)
        self._torque_clamp_fraction = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self._torque_adapter_clamp_fraction = torch.zeros_like(self._torque_clamp_fraction)
        self._torque_rate_limited_fraction = torch.zeros_like(self._torque_clamp_fraction)
        self._z_arm_hat = torch.zeros((self.num_envs, 24), dtype=torch.float32, device=self.device)
        self._prev_q_error_mean = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self._reward_debug: dict[str, torch.Tensor | str] = {}

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot
        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _find_joint_ids(self, joint_names: Sequence[str]) -> list[int]:
        ids: list[int] = []
        for joint_name in joint_names:
            joint_ids, _ = self.robot.find_joints(joint_name)
            if len(joint_ids) == 0:
                raise RuntimeError(f"Could not find joint: {joint_name}")
            ids.append(joint_ids[0])
        return ids

    def _read_joint_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.robot.data.joint_pos[:, self._state_joint_ids]
        qdot = self.robot.data.joint_vel[:, self._state_joint_ids]
        return q, qdot

    def _read_pressure_delta(self) -> torch.Tensor:
        if self.cfg.pressure_noise_std <= 0.0:
            return self._dP
        return self._dP + self.cfg.pressure_noise_std * torch.randn_like(self._dP)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._current_action = self.cfg.action_scale * clamp_and_sanitize(actions, 1.0)
        self._action_delta = self._current_action - self._prev_action
        self._current_qddot_cmd = self._controller.build_qddot_cmd(
            self._current_action,
            self.cfg.action_to_qddot_scale,
            self.cfg.qddot_cmd_clip,
        )

        q, qdot = self._read_joint_state()
        dP = self._read_pressure_delta()
        control_out = None

        if self.cfg.control_mode == "position_delta" and not self.cfg.use_fd_effort:
            if self.cfg.position_delta_source == "direct_action":
                self._current_valve_cmd = self._current_action.clone()
                self._zero_torque_state()
            elif self.cfg.position_delta_source == "id_valve":
                control_out = self._controller.apply_action_valve_only(
                    q,
                    qdot,
                    dP,
                    self._current_action,
                    self._prev_valve_cmd,
                    z_arm_hat=self._z_arm_hat,
                    action_to_qddot_scale=self.cfg.action_to_qddot_scale,
                    qddot_cmd_clip=self.cfg.qddot_cmd_clip,
                )
                self._current_qddot_cmd = control_out.qddot_cmd
                self._current_valve_cmd = control_out.valve_cmd
                self._zero_torque_state()
                self._z_arm_hat = control_out.z_arm_hat
            else:
                raise ValueError(f"Unsupported position_delta_source: {self.cfg.position_delta_source!r}")
        elif self.cfg.control_mode == "effort" or self.cfg.use_fd_effort:
            control_out = self._controller.apply_action_full_pipeline(
                q,
                qdot,
                dP,
                self._fnet,
                self._current_action,
                self._prev_valve_cmd,
                z_arm_hat=self._z_arm_hat,
                action_to_qddot_scale=self.cfg.action_to_qddot_scale,
                qddot_cmd_clip=self.cfg.qddot_cmd_clip,
                use_sd_feedback=self.cfg.use_sd_feedback,
            )
            self._current_qddot_cmd = control_out.qddot_cmd
            self._current_fd_simscape_output = control_out.fd_simscape_output
            self._current_tau_fd_adapted = control_out.tau_fd_adapted
            self._current_tau_applied_raw = control_out.tau_applied_raw
            self._current_tau_isaac_filtered = control_out.tau_isaac_filtered
            self._current_tau_isaac_clamped = control_out.tau_isaac_clamped
            self._torque_clamp_fraction = control_out.torque_clamp_fraction
            self._torque_adapter_clamp_fraction = control_out.torque_adapter_clamp_fraction
            self._torque_rate_limited_fraction = control_out.torque_rate_limited_fraction
            self._current_torque = control_out.tau_isaac_clamped
            self._current_valve_cmd = control_out.valve_cmd
            self._z_arm_hat = control_out.z_arm_hat
        else:
            raise ValueError(f"Unsupported control_mode: {self.cfg.control_mode!r}")

        if self.cfg.use_sd_feedback:
            if control_out is None or (self.cfg.control_mode != "effort" and not self.cfg.use_fd_effort):
                raise ValueError("use_sd_feedback=True requires the FD/SD effort path.")
            self._dP = clamp_and_sanitize(control_out.dP_next, self.cfg.max_abs_pressure_delta)
            self._fnet = clamp_and_sanitize(control_out.fnet_next, self.cfg.max_abs_fnet)
        else:
            self._dP = torch.zeros_like(self._dP)
            self._fnet = torch.zeros_like(self._fnet)

    def _apply_action(self) -> None:
        if self.cfg.control_mode == "position_delta" and not self.cfg.use_fd_effort:
            if self.cfg.position_delta_source == "direct_action":
                delta_cmd = self._current_action
            elif self.cfg.position_delta_source == "id_valve":
                delta_cmd = self._current_valve_cmd
            else:
                raise ValueError(f"Unsupported position_delta_source: {self.cfg.position_delta_source!r}")
            q_now = self.robot.data.joint_pos[:, self._command_joint_ids]
            q_target = q_now + self.cfg.position_scale * delta_cmd
            lower = self.robot.data.soft_joint_pos_limits[:, self._command_joint_ids, 0]
            upper = self.robot.data.soft_joint_pos_limits[:, self._command_joint_ids, 1]
            q_target = torch.clamp(q_target, lower, upper)
            self.robot.set_joint_position_target(q_target, joint_ids=self._command_joint_ids)
            return

        if self.cfg.control_mode == "effort" or self.cfg.use_fd_effort:
            self.robot.set_joint_effort_target(self._current_torque, joint_ids=self._command_joint_ids)
            return

        raise ValueError(f"Unsupported control_mode: {self.cfg.control_mode!r}")

    def _get_observations(self) -> dict:
        q, qdot = self._read_joint_state()
        dP = self._read_pressure_delta()
        self._z_arm_hat = self._controller.compute_z_arm(q, qdot, dP, self._prev_valve_cmd)
        obs = build_rl_obs(
            q, qdot, self._q_ref, self._qdot_ref, self._prev_action, self._prev_valve_cmd, self._z_arm_hat
        )
        if obs.shape[-1] != self.cfg.observation_space:
            raise RuntimeError(
                f"Observation size mismatch: got {obs.shape[-1]}, expected {self.cfg.observation_space}"
            )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        q, qdot = self._read_joint_state()
        current_q_error_mean = self._weighted_q_error_mean(q, self._q_ref)
        reward_weights = self._reward_weights_for_control_mode()
        if self.cfg.debug_reward_terms:
            reward, reward_debug = joint_reach_reward_debug(
                q,
                qdot,
                self._q_ref,
                self._action_delta,
                self._current_valve_cmd,
                self._current_torque,
                self.reset_terminated,
                reward_weights,
                action=self._current_action,
                prev_q_error_mean=self._prev_q_error_mean,
                torque_effort_weight=reward_weights.torque_effort,
                q_tracking_joint_weights=self._q_tracking_joint_weights,
            )
            self._reward_debug = self._mean_reward_debug(reward_debug)
            self._reward_debug["control_mode"] = self.cfg.control_mode
            self._reward_debug["position_delta_source"] = self.cfg.position_delta_source
            self._reward_debug["use_sd_feedback"] = str(self.cfg.use_sd_feedback)
            self._reward_debug["use_fd_effort"] = str(self.cfg.use_fd_effort)
            if self.cfg.log_full_pipeline_debug:
                self._add_full_pipeline_debug(self._reward_debug)
        else:
            reward = joint_reach_reward(
                q,
                qdot,
                self._q_ref,
                self._action_delta,
                self._current_valve_cmd,
                self._current_torque,
                self.reset_terminated,
                reward_weights,
                prev_q_error_mean=self._prev_q_error_mean,
                torque_effort_weight=reward_weights.torque_effort,
                q_tracking_joint_weights=self._q_tracking_joint_weights,
            )
            self._reward_debug = {}
        self._prev_q_error_mean = current_q_error_mean.detach().clone()
        self._prev_action = self._current_action.clone()
        self._prev_valve_cmd = self._current_valve_cmd.clone()
        return reward

    def _reward_weights_for_control_mode(self) -> JointReachRewardWeights:
        if self.cfg.control_mode == "position_delta" and not self.cfg.use_fd_effort:
            return replace(self._reward_weights, torque_effort=0.0)
        return self._reward_weights

    def _weighted_q_error_mean(self, q: torch.Tensor, q_ref: torch.Tensor) -> torch.Tensor:
        joint_weights = self._q_tracking_joint_weights.reshape(1, -1)
        return torch.sum(torch.abs(q - q_ref) * joint_weights, dim=-1) / torch.clamp(
            torch.sum(joint_weights), min=1.0e-8
        )

    def _mean_reward_debug(self, reward_debug: dict[str, torch.Tensor]) -> dict[str, torch.Tensor | str]:
        mean_debug: dict[str, torch.Tensor | str] = {}
        for name, value in reward_debug.items():
            if name == "q_error_abs_per_joint":
                mean_debug[name] = value.detach().mean(dim=0)
            else:
                mean_debug[name] = value.detach().mean()
        return mean_debug

    def _zero_torque_state(self) -> None:
        self._current_torque.zero_()
        self._current_fd_simscape_output.zero_()
        self._current_tau_fd_adapted.zero_()
        self._current_tau_applied_raw.zero_()
        self._current_tau_isaac_filtered.zero_()
        self._current_tau_isaac_clamped.zero_()
        self._torque_clamp_fraction.zero_()
        self._torque_adapter_clamp_fraction.zero_()
        self._torque_rate_limited_fraction.zero_()

    def _torque_ramp_scale(self) -> float:
        if self.cfg.torque_ramp_steps <= 0:
            return float(self.cfg.torque_safety_scale)
        step_count = float(getattr(self, "common_step_counter", 0))
        ramp = min(1.0, max(0.0, (step_count + 1.0) / float(self.cfg.torque_ramp_steps)))
        return float(self.cfg.torque_safety_scale) * ramp

    def _apply_torque_safety(self, torque: torch.Tensor) -> torch.Tensor:
        torque = clamp_and_sanitize(torque, self.cfg.max_abs_torque)
        torque = torque * self._torque_ramp_scale()
        return clamp_and_sanitize(torque, self.cfg.max_abs_torque)

    def _add_full_pipeline_debug(self, debug: dict[str, torch.Tensor | str]) -> None:
        debug["qddot_cmd_mean"] = torch.mean(torch.abs(self._current_qddot_cmd.detach()))
        debug["qddot_cmd_abs_per_joint"] = torch.mean(torch.abs(self._current_qddot_cmd.detach()), dim=0)
        debug["z_arm_mean"] = torch.mean(self._z_arm_hat.detach())
        debug["z_arm_std"] = torch.std(self._z_arm_hat.detach(), unbiased=False)
        debug["valve_max"] = torch.max(torch.abs(self._current_valve_cmd.detach()))
        debug["fd_simscape_mean"] = torch.mean(torch.abs(self._current_fd_simscape_output.detach()))
        debug["fd_simscape_max"] = torch.max(torch.abs(self._current_fd_simscape_output.detach()))
        debug["tau_fd_adapted_mean"] = torch.mean(torch.abs(self._current_tau_fd_adapted.detach()))
        debug["tau_fd_adapted_max"] = torch.max(torch.abs(self._current_tau_fd_adapted.detach()))
        debug["tau_applied_raw_mean"] = torch.mean(torch.abs(self._current_tau_applied_raw.detach()))
        debug["tau_applied_raw_max"] = torch.max(torch.abs(self._current_tau_applied_raw.detach()))
        debug["tau_isaac_filtered_mean"] = torch.mean(torch.abs(self._current_tau_isaac_filtered.detach()))
        debug["tau_isaac_filtered_max"] = torch.max(torch.abs(self._current_tau_isaac_filtered.detach()))
        debug["torque_mean"] = torch.mean(torch.abs(self._current_torque.detach()))
        debug["torque_max"] = torch.max(torch.abs(self._current_torque.detach()))
        debug["torque_adapter_mode"] = self.cfg.torque_adapter_mode
        debug["torque_adapter_preset"] = self.cfg.torque_adapter_preset
        debug["torque_adapter_scale"] = str(self.cfg.torque_adapter_scale)
        debug["use_fd_residual_alpha"] = str(self.cfg.use_fd_residual_alpha)
        debug["fd_residual_alpha"] = torch.tensor(float(self.cfg.fd_residual_alpha), device=self.device)
        debug["max_abs_torque"] = torch.tensor(float(self.cfg.max_abs_torque), device=self.device)
        effective_scale = [
            float(self.cfg.fd_residual_alpha) * float(value) if self.cfg.use_fd_residual_alpha else float(value)
            for value in self.cfg.torque_adapter_scale
        ]
        debug["effective_torque_adapter_scale"] = str(effective_scale)
        debug["torque_adapter_clamp_fraction"] = torch.mean(self._torque_adapter_clamp_fraction.detach())
        debug["torque_clamp_fraction"] = torch.mean(self._torque_clamp_fraction.detach())
        debug["torque_rate_limited_fraction"] = torch.mean(self._torque_rate_limited_fraction.detach())
        debug["torque_rate_limit"] = torch.tensor(float(self.cfg.torque_rate_limit), device=self.device)
        debug["torque_lowpass_alpha"] = torch.tensor(float(self.cfg.torque_lowpass_alpha), device=self.device)
        debug["dP_mean"] = torch.mean(torch.abs(self._dP.detach()))
        debug["fnet_mean"] = torch.mean(torch.abs(self._fnet.detach()))

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        q, qdot = self._read_joint_state()
        lower_limits = self.robot.data.soft_joint_pos_limits[:, self._state_joint_ids, 0]
        upper_limits = self.robot.data.soft_joint_pos_limits[:, self._state_joint_ids, 1]
        safety_terminated = joint_safety_termination(
            q,
            qdot,
            lower_limits,
            upper_limits,
            joint_limit_margin=self.cfg.joint_limit_margin,
            max_abs_joint_velocity=self.cfg.max_abs_joint_velocity,
        )
        finite_control = (
            torch.isfinite(self._current_torque).all(dim=-1)
            & torch.isfinite(self._current_fd_simscape_output).all(dim=-1)
            & torch.isfinite(self._current_tau_fd_adapted).all(dim=-1)
            & torch.isfinite(self._current_tau_applied_raw).all(dim=-1)
            & torch.isfinite(self._current_tau_isaac_filtered).all(dim=-1)
            & torch.isfinite(self._current_tau_isaac_clamped).all(dim=-1)
            & torch.isfinite(self._current_valve_cmd).all(dim=-1)
            & torch.isfinite(self._current_qddot_cmd).all(dim=-1)
            & torch.isfinite(self._z_arm_hat).all(dim=-1)
            & torch.isfinite(self._dP).all(dim=-1)
            & torch.isfinite(self._fnet).all(dim=-1)
        )
        terminated = safety_terminated | (~finite_control)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        default_root_pose, default_root_velocity = self._default_root_state(env_ids)
        default_root_pose[:, :3] += self.scene.env_origins[env_ids]

        self._sample_joint_targets(env_ids)
        self._qdot_ref[env_ids] = 0.0

        self._dP[env_ids] = 0.0
        self._fnet[env_ids] = 0.0
        self._prev_action[env_ids] = 0.0
        self._current_action[env_ids] = 0.0
        self._action_delta[env_ids] = 0.0
        self._current_qddot_cmd[env_ids] = 0.0
        self._prev_valve_cmd[env_ids] = 0.0
        self._current_valve_cmd[env_ids] = 0.0
        self._current_torque[env_ids] = 0.0
        self._current_fd_simscape_output[env_ids] = 0.0
        self._current_tau_fd_adapted[env_ids] = 0.0
        self._current_tau_applied_raw[env_ids] = 0.0
        self._current_tau_isaac_filtered[env_ids] = 0.0
        self._current_tau_isaac_clamped[env_ids] = 0.0
        self._torque_clamp_fraction[env_ids] = 0.0
        self._torque_adapter_clamp_fraction[env_ids] = 0.0
        self._torque_rate_limited_fraction[env_ids] = 0.0
        self._z_arm_hat[env_ids] = 0.0
        self._prev_q_error_mean[env_ids] = self._weighted_q_error_mean(
            joint_pos[:, self._state_joint_ids], self._q_ref[env_ids]
        )
        self._controller.reset(env_ids)

        self.robot.write_root_pose_to_sim(default_root_pose, env_ids)
        self.robot.write_root_velocity_to_sim(default_root_velocity, env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _default_root_state(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if hasattr(self.robot.data, "default_root_pose") and hasattr(self.robot.data, "default_root_vel"):
            root_pose = self.robot.data.default_root_pose[env_ids].clone()
            root_velocity = self.robot.data.default_root_vel[env_ids].clone()
            return root_pose, root_velocity

        root_state = self.robot.data.default_root_state[env_ids].clone()
        return root_state[:, :7], root_state[:, 7:]

    def _sample_joint_targets(self, env_ids: torch.Tensor) -> None:
        default_q = self.robot.data.default_joint_pos[env_ids][:, self._state_joint_ids]
        lower = self.robot.data.soft_joint_pos_limits[env_ids][:, self._state_joint_ids, 0]
        upper = self.robot.data.soft_joint_pos_limits[env_ids][:, self._state_joint_ids, 1]
        target_range = torch.tensor(self.cfg.target_joint_range, dtype=torch.float32, device=self.device).reshape(1, 5)

        has_limits = torch.isfinite(lower) & torch.isfinite(upper)
        has_limits = has_limits & ((upper - lower) > 2.0 * self.cfg.joint_limit_margin)
        fallback_lower = default_q - target_range
        fallback_upper = default_q + target_range
        safe_lower = torch.where(has_limits, lower + self.cfg.joint_limit_margin, fallback_lower)
        safe_upper = torch.where(has_limits, upper - self.cfg.joint_limit_margin, fallback_upper)
        sample_lower = torch.maximum(fallback_lower, safe_lower)
        sample_upper = torch.minimum(fallback_upper, safe_upper)
        sample_upper = torch.maximum(sample_upper, sample_lower + 1.0e-3)

        self._q_ref[env_ids] = sample_lower + torch.rand_like(default_q) * (sample_upper - sample_lower)
