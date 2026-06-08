from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from kevin_integration.controllers.reduced_hydraulic_actuator import (
    ReducedHydraulicActuatorROM,
    ReducedHydraulicActuatorROMCfg,
)
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
        self._q_tracking_reward_joint_weights = torch.as_tensor(
            getattr(self.cfg, "q_tracking_reward_joint_weights", self.cfg.q_tracking_joint_weights),
            dtype=torch.float32,
            device=self.device,
        )
        self._q_tracking_debug_joint_weights = torch.as_tensor(
            getattr(self.cfg, "q_tracking_debug_joint_weights", self.cfg.q_tracking_joint_weights),
            dtype=torch.float32,
            device=self.device,
        )
        self._q_tracking_joint_weights = self._q_tracking_reward_joint_weights
        if self._q_tracking_reward_joint_weights.shape != (5,):
            raise RuntimeError(
                "q_tracking_reward_joint_weights must have shape (5,), "
                f"got {tuple(self._q_tracking_reward_joint_weights.shape)}"
            )
        if self._q_tracking_debug_joint_weights.shape != (5,):
            raise RuntimeError(
                "q_tracking_debug_joint_weights must have shape (5,), "
                f"got {tuple(self._q_tracking_debug_joint_weights.shape)}"
            )
        self._joint5_coupling_source_joint = int(self.cfg.joint5_coupling_source_joint)
        self._joint5_coupling_target_joint = int(self.cfg.joint5_coupling_target_joint)
        if self.cfg.use_joint5_coupling:
            if self._joint5_coupling_source_joint not in range(5):
                raise RuntimeError(
                    "joint5_coupling_source_joint must be a zero-based index in [0, 4], "
                    f"got {self._joint5_coupling_source_joint}"
                )
            if self._joint5_coupling_target_joint not in range(5):
                raise RuntimeError(
                    "joint5_coupling_target_joint must be a zero-based index in [0, 4], "
                    f"got {self._joint5_coupling_target_joint}"
                )
            if self._joint5_coupling_source_joint == self._joint5_coupling_target_joint:
                raise RuntimeError("joint5 coupling source and target joints must be different")

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
                sd_nan_guard=self.cfg.sd_nan_guard,
                sd_clamp_outputs=self.cfg.sd_clamp_outputs,
            )
        )
        self._reduced_hydraulic_rom = ReducedHydraulicActuatorROM(
            num_envs=self.num_envs,
            cfg=self._make_rom_cfg(),
            num_cylinders=4,
            device=self.device,
        )
        self._reward_weights = JointReachRewardWeights(
            q_tracking=self.cfg.rew_q_tracking,
            qdot=self.cfg.rew_qdot,
            action_smoothness=self.cfg.rew_action_smoothness,
            action_rate=self.cfg.rew_action_rate,
            action_magnitude=self.cfg.rew_action_magnitude,
            action_saturation=self.cfg.rew_action_saturation,
            action_saturation_threshold=self.cfg.action_saturation_threshold,
            valve_effort=self.cfg.rew_valve_effort,
            progress=self.cfg.rew_progress,
            settle_tracking=self.cfg.rew_settle_tracking,
            joint1_settle_tracking=self.cfg.rew_joint1_settle_tracking,
            joint1_settle_error_deadband=self.cfg.joint1_settle_error_deadband,
            torque_effort=self.cfg.rew_torque_effort,
            alive=self.cfg.rew_alive,
            termination=self.cfg.rew_termination,
        )

        self._q_ref_commanded = torch.zeros((self.num_envs, 5), dtype=torch.float32, device=self.device)
        self._q_ref_target = torch.zeros_like(self._q_ref_commanded)
        self._q_ref_start = torch.zeros_like(self._q_ref_commanded)
        self._q_ref = self._q_ref_commanded
        self._qdot_ref = torch.zeros((self.num_envs, 5), dtype=torch.float32, device=self.device)
        self._q_ref_prev = torch.zeros_like(self._q_ref_commanded)
        self._q_ref_target_observed = torch.zeros_like(self._q_ref_commanded)
        self._q_ref_delta = torch.zeros_like(self._q_ref_commanded)
        self._q_ref_target_delta = torch.zeros_like(self._q_ref_commanded)
        self._q_ref_sampled_on_reset = torch.zeros_like(self._q_ref_commanded)
        self._q_ref_ramp_progress = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self._q_ref_target_scale = torch.ones((self.num_envs,), dtype=torch.float32, device=self.device)
        self._q_ref_curriculum_progress = torch.zeros_like(self._q_ref_target_scale)
        self._target_changed_flag = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self._target_change_step = torch.full((self.num_envs,), -1.0, dtype=torch.float32, device=self.device)
        self._last_reset_step_global = torch.full((self.num_envs,), -1.0, dtype=torch.float32, device=self.device)
        self._reset_due_to_timeout = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self._reset_due_to_termination = torch.zeros_like(self._reset_due_to_timeout)
        self._reset_due_to_manual_target_change = torch.zeros_like(self._reset_due_to_timeout)
        self._last_terminated_debug = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._last_time_out_debug = torch.zeros_like(self._last_terminated_debug)
        self._last_done_debug = torch.zeros_like(self._last_terminated_debug)
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
        self._rom_cylinder_pos_target = torch.zeros_like(self._current_torque)
        self._rom_pressure = torch.zeros_like(self._current_torque)
        self._rom_xdot_pred = torch.zeros_like(self._current_torque)
        self._rom_dx_pred = torch.zeros_like(self._current_torque)
        self._rom_valve_cmd_delayed = torch.zeros_like(self._current_torque)
        self._rom_valve_cmd_effective = torch.zeros_like(self._current_torque)
        self._rom_target_clamp_fraction = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self._rom_pressure_clamp_fraction = torch.zeros_like(self._rom_target_clamp_fraction)
        self._rom_velocity_clamp_fraction = torch.zeros_like(self._rom_target_clamp_fraction)
        self._rom_dx_clamp_fraction = torch.zeros_like(self._rom_target_clamp_fraction)
        self._rom_nan_guard_triggered = torch.zeros_like(self._rom_target_clamp_fraction)
        self._rom_output_zeroed_fraction = torch.zeros_like(self._rom_target_clamp_fraction)
        self._rom_actuator_pos_error_mean = torch.zeros_like(self._rom_target_clamp_fraction)
        self._rom_actuator_pos_error_abs_max = torch.zeros_like(self._rom_target_clamp_fraction)
        self._sd_q_next_pred = torch.zeros((self.num_envs, 5), dtype=torch.float32, device=self.device)
        self._sd_qdot_next_pred = torch.zeros_like(self._sd_q_next_pred)
        self._sd_dP_next_pred = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self._sd_fnet_next_pred = torch.zeros_like(self._sd_dP_next_pred)
        self._sd_output_nan_or_inf = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self._sd_output_clamped_fraction = torch.zeros_like(self._sd_output_nan_or_inf)
        self._sd_feedback_dP_delta_mean = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self._sd_feedback_dP_delta_max = torch.zeros_like(self._sd_feedback_dP_delta_mean)
        self._sd_feedback_fnet_delta_mean = torch.zeros_like(self._sd_feedback_dP_delta_mean)
        self._sd_feedback_fnet_delta_max = torch.zeros_like(self._sd_feedback_dP_delta_mean)
        self._sd_feedback_rate_limited_fraction = torch.zeros_like(self._sd_feedback_dP_delta_mean)
        self._sd_feedback_clamped_fraction = torch.zeros_like(self._sd_feedback_dP_delta_mean)
        self._sd_feedback_nan_or_inf = torch.zeros_like(self._sd_feedback_dP_delta_mean)
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

    def _make_rom_cfg(self) -> ReducedHydraulicActuatorROMCfg:
        return ReducedHydraulicActuatorROMCfg(
            output_mode=self.cfg.rom_output_mode,
            Abar=tuple(float(value) for value in self.cfg.rom_Abar),
            Vt=tuple(float(value) for value in self.cfg.rom_Vt),
            E=tuple(float(value) for value in self.cfg.rom_E),
            Kq=tuple(float(value) for value in self.cfg.rom_Kq),
            CL=tuple(float(value) for value in self.cfg.rom_CL),
            mu=tuple(float(value) for value in self.cfg.rom_mu),
            Fs=tuple(float(value) for value in self.cfg.rom_Fs),
            kf=tuple(float(value) for value in self.cfg.rom_kf),
            Pmax=tuple(float(value) for value in self.cfg.rom_Pmax),
            xdot_max=tuple(float(value) for value in self.cfg.rom_xdot_max),
            dx_max=tuple(float(value) for value in self.cfg.rom_dx_max),
            delay_steps=tuple(int(value) for value in self.cfg.rom_delay_steps),
            udz_pos=tuple(float(value) for value in self.cfg.rom_udz_pos),
            udz_neg=tuple(float(value) for value in self.cfg.rom_udz_neg),
            velocity_gain=tuple(float(value) for value in self.cfg.rom_velocity_gain),
            force_scale=tuple(float(value) for value in self.cfg.rom_force_scale),
            nan_guard=bool(self.cfg.rom_nan_guard),
            clamp_outputs=bool(self.cfg.rom_clamp_outputs),
        )

    def _read_joint_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.robot.data.joint_pos[:, self._state_joint_ids]
        qdot = self.robot.data.joint_vel[:, self._state_joint_ids]
        return q, qdot

    def get_debug_state(self, env_id: int = 0) -> dict[str, torch.Tensor | dict[str, torch.Tensor | str] | tuple[str, ...]]:
        """Return detached joint/controller state for live diagnostics."""
        if env_id < 0 or env_id >= self.num_envs:
            raise IndexError(f"env_id must be in [0, {self.num_envs}), got {env_id}")

        q, qdot = self._read_joint_state()
        q_ref = self._q_ref_commanded
        reward_debug: dict[str, torch.Tensor | str] = {}
        for name, value in self._reward_debug.items():
            if isinstance(value, torch.Tensor):
                reward_debug[name] = value.detach().clone()
            else:
                reward_debug[name] = value

        return {
            "joint_names": tuple(self.cfg.state_joint_names),
            "command_joint_names": tuple(self.cfg.command_joint_names),
            "q_actual": q[env_id].detach().clone(),
            "q_ref": q_ref[env_id].detach().clone(),
            "q_ref_commanded": q_ref[env_id].detach().clone(),
            "q_ref_target": self._q_ref_target[env_id].detach().clone(),
            "q_ref_start": self._q_ref_start[env_id].detach().clone(),
            "q_error": (q_ref[env_id] - q[env_id]).detach().clone(),
            "q_error_abs": torch.abs(q_ref[env_id] - q[env_id]).detach().clone(),
            "q_ref_prev": self._q_ref_prev[env_id].detach().clone(),
            "q_ref_delta": self._q_ref_delta[env_id].detach().clone(),
            "q_ref_delta_abs": torch.abs(self._q_ref_delta[env_id]).detach().clone(),
            "q_ref_ramp_progress": self._q_ref_ramp_progress[env_id].detach().clone(),
            "q_ref_target_scale": self._q_ref_target_scale[env_id].detach().clone(),
            "episode_step": self.episode_length_buf[env_id].detach().clone(),
            "target_changed_flag": self._target_changed_flag[env_id].detach().clone(),
            "qdot": qdot[env_id].detach().clone(),
            "action": self._current_action[env_id].detach().clone(),
            "qddot_cmd": self._current_qddot_cmd[env_id].detach().clone(),
            "valve_cmd": self._current_valve_cmd[env_id].detach().clone(),
            "rom_cylinder_pos_target": self._rom_cylinder_pos_target[env_id].detach().clone(),
            "rom_pressure": self._rom_pressure[env_id].detach().clone(),
            "rom_xdot_pred": self._rom_xdot_pred[env_id].detach().clone(),
            "rom_dx_pred": self._rom_dx_pred[env_id].detach().clone(),
            "rom_valve_cmd_delayed": self._rom_valve_cmd_delayed[env_id].detach().clone(),
            "rom_valve_cmd_effective": self._rom_valve_cmd_effective[env_id].detach().clone(),
            "rom_target_clamp_fraction": self._rom_target_clamp_fraction[env_id].detach().clone(),
            "rom_pressure_clamp_fraction": self._rom_pressure_clamp_fraction[env_id].detach().clone(),
            "rom_nan_guard_triggered": self._rom_nan_guard_triggered[env_id].detach().clone(),
            "use_joint5_coupling": str(self.cfg.use_joint5_coupling),
            "joint5_coupling_gain": torch.tensor(float(self.cfg.joint5_coupling_gain), device=self.device),
            "joint5_coupling_bias": torch.tensor(float(self.cfg.joint5_coupling_bias), device=self.device),
            "joint4_q_ref": q_ref[env_id, self._joint5_coupling_source_joint].detach().clone(),
            "joint5_q_ref": q_ref[env_id, self._joint5_coupling_target_joint].detach().clone(),
            "joint5_coupled_ref": self._coupled_joint5_reference(q_ref)[env_id].detach().clone(),
            "joint5_coupling_error": self._joint5_coupling_error(q_ref)[env_id].detach().clone(),
            "joint5_actual": q[env_id, self._joint5_coupling_target_joint].detach().clone(),
            "joint5_qdot": qdot[env_id, self._joint5_coupling_target_joint].detach().clone(),
            "joint5_tracking_reward_weight": self._q_tracking_reward_joint_weights[
                self._joint5_coupling_target_joint
            ].detach().clone(),
            "joint5_target_amplitude": torch.tensor(
                float(self.cfg.q_ref_target_amplitude[self._joint5_coupling_target_joint]), device=self.device
            ),
            "sd_q_next_pred": self._sd_q_next_pred[env_id].detach().clone(),
            "sd_qdot_next_pred": self._sd_qdot_next_pred[env_id].detach().clone(),
            "sd_dP_next_pred": self._sd_dP_next_pred[env_id].detach().clone(),
            "sd_fnet_next_pred": self._sd_fnet_next_pred[env_id].detach().clone(),
            "sd_output_nan_or_inf": self._sd_output_nan_or_inf[env_id].detach().clone(),
            "sd_output_clamped_fraction": self._sd_output_clamped_fraction[env_id].detach().clone(),
            "sd_feedback_rate_limited_fraction": self._sd_feedback_rate_limited_fraction[env_id].detach().clone(),
            "sd_feedback_clamped_fraction": self._sd_feedback_clamped_fraction[env_id].detach().clone(),
            "sd_feedback_nan_or_inf": self._sd_feedback_nan_or_inf[env_id].detach().clone(),
            "dP": self._dP[env_id].detach().clone(),
            "fnet": self._fnet[env_id].detach().clone(),
            "fd_tau_raw": self._current_fd_simscape_output[env_id].detach().clone(),
            "tau_fd_adapted": self._current_tau_fd_adapted[env_id].detach().clone(),
            "tau_applied_raw": self._current_tau_applied_raw[env_id].detach().clone(),
            "tau_isaac_filtered": self._current_tau_isaac_filtered[env_id].detach().clone(),
            "torque": self._current_torque[env_id].detach().clone(),
            "reward_debug": reward_debug,
        }

    def _read_pressure_delta(self) -> torch.Tensor:
        if self.cfg.pressure_noise_std <= 0.0:
            return self._dP
        return self._dP + self.cfg.pressure_noise_std * torch.randn_like(self._dP)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._update_q_ref_commanded()
        self._current_action = self.cfg.action_scale * clamp_and_sanitize(actions, 1.0)
        self._action_delta = self._current_action - self._prev_action
        self._current_qddot_cmd = self._controller.build_qddot_cmd(
            self._current_action,
            self.cfg.action_to_qddot_scale,
            self.cfg.qddot_cmd_clip,
        )

        q, qdot = self._read_joint_state()
        dP = self._read_pressure_delta()
        previous_dP = self._dP.detach().clone()
        previous_fnet = self._fnet.detach().clone()
        control_out = None

        if self.cfg.use_reduced_hydraulic_rom:
            self._pre_physics_step_reduced_hydraulic_rom(
                q=q,
                qdot=qdot,
                dP=dP,
                previous_dP=previous_dP,
                previous_fnet=previous_fnet,
            )
            return

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
                use_sd_feedback=False,
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

        run_sd_observer = self.cfg.use_sd_observer or self.cfg.use_sd_feedback
        if run_sd_observer:
            sd_observer = self._controller.run_sd_observer(
                q,
                qdot,
                dP,
                self._fnet,
                self._current_valve_cmd,
                self._z_arm_hat,
            )
            self._store_sd_observer_output(sd_observer)
        else:
            self._zero_sd_observer_state()

        if self.cfg.use_sd_feedback:
            if self.cfg.control_mode != "effort" and not self.cfg.use_fd_effort:
                raise ValueError("use_sd_feedback=True requires the FD/SD effort path.")
            self._apply_sd_feedback(
                previous_dP=previous_dP,
                previous_fnet=previous_fnet,
                dP_target=self._sd_dP_next_pred,
                fnet_target=self._sd_fnet_next_pred,
            )
        else:
            self._zero_sd_feedback_state()
            self._dP = torch.zeros_like(self._dP)
            self._fnet = torch.zeros_like(self._fnet)

    def _pre_physics_step_reduced_hydraulic_rom(
        self,
        *,
        q: torch.Tensor,
        qdot: torch.Tensor,
        dP: torch.Tensor,
        previous_dP: torch.Tensor,
        previous_fnet: torch.Tensor,
    ) -> None:
        if self.cfg.rom_output_mode != "velocity_to_position_target":
            raise ValueError(
                "use_reduced_hydraulic_rom=True only supports "
                "rom_output_mode='velocity_to_position_target'."
            )

        if self.cfg.rom_use_id_valve_cmd:
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
            self._z_arm_hat = control_out.z_arm_hat
        else:
            self._current_valve_cmd = clamp_and_sanitize(self._current_action, self.cfg.max_abs_valve)
        self._zero_torque_state()

        run_sd_observer = self.cfg.use_sd_observer or self.cfg.use_sd_feedback or self.cfg.rom_use_sd_feedback
        if run_sd_observer:
            sd_observer = self._controller.run_sd_observer(
                q,
                qdot,
                dP,
                self._fnet,
                self._current_valve_cmd,
                self._z_arm_hat,
            )
            self._store_sd_observer_output(sd_observer)
        else:
            self._zero_sd_observer_state()

        if self.cfg.rom_use_sd_feedback:
            self._apply_sd_feedback(
                previous_dP=previous_dP,
                previous_fnet=previous_fnet,
                dP_target=self._sd_dP_next_pred,
                fnet_target=self._sd_fnet_next_pred,
            )
            rom_pressure_input = self._dP
            rom_fnet_input = self._fnet
        else:
            self._zero_sd_feedback_state()
            rom_pressure_input = self._rom_pressure
            rom_fnet_input = torch.zeros_like(self._fnet)
            self._fnet = torch.zeros_like(self._fnet)

        x_cyl = self.robot.data.joint_pos[:, self._command_joint_ids]
        xdot_cyl = self.robot.data.joint_vel[:, self._command_joint_ids]
        step_dt = float(getattr(self, "step_dt", float(self.cfg.sim.dt) * float(self.cfg.decimation)))
        rom_dt = float(self.cfg.rom_dt) if float(self.cfg.rom_dt) > 0.0 else step_dt
        rom_out = self._reduced_hydraulic_rom.step(
            valve_cmd=self._current_valve_cmd,
            x_cyl=x_cyl,
            xdot_cyl=xdot_cyl,
            P_L=rom_pressure_input,
            Fnet=rom_fnet_input,
            dt=rom_dt,
        )
        self._rom_pressure = rom_out.P_L_next.detach().clone()
        self._dP = self._rom_pressure.detach().clone()
        self._rom_xdot_pred = rom_out.xdot_cyl_pred.detach().clone()
        self._rom_dx_pred = rom_out.dx_cyl_pred.detach().clone()
        self._rom_valve_cmd_delayed = rom_out.valve_cmd_delayed.detach().clone()
        self._rom_valve_cmd_effective = rom_out.valve_cmd_effective.detach().clone()
        self._rom_pressure_clamp_fraction = rom_out.debug["rom_pressure_clamp_fraction"].detach().clone()
        self._rom_velocity_clamp_fraction = rom_out.debug["rom_velocity_clamp_fraction"].detach().clone()
        self._rom_dx_clamp_fraction = rom_out.debug["rom_dx_clamp_fraction"].detach().clone()
        self._rom_nan_guard_triggered = rom_out.debug["rom_nan_guard_triggered"].detach().clone()
        self._rom_output_zeroed_fraction = rom_out.debug["rom_output_zeroed_fraction"].detach().clone()

        target_candidate = self._rom_cylinder_pos_target + self._rom_dx_pred
        target_candidate, invalid_target = self._replace_invalid_with_fallback(
            target_candidate,
            self._rom_cylinder_pos_target,
        )
        lower = self.robot.data.soft_joint_pos_limits[:, self._command_joint_ids, 0]
        upper = self.robot.data.soft_joint_pos_limits[:, self._command_joint_ids, 1]
        target_clamp_mask = (target_candidate < lower) | (target_candidate > upper)
        target_clamped = torch.clamp(target_candidate, lower, upper)
        self._rom_cylinder_pos_target = target_clamped.detach().clone()
        self._rom_target_clamp_fraction = torch.mean(target_clamp_mask.float(), dim=-1)
        self._rom_nan_guard_triggered = torch.maximum(
            self._rom_nan_guard_triggered,
            invalid_target.float(),
        )
        actuator_pos_error = self._rom_cylinder_pos_target - x_cyl
        self._rom_actuator_pos_error_mean = torch.mean(torch.abs(actuator_pos_error.detach()), dim=-1)
        self._rom_actuator_pos_error_abs_max = torch.max(torch.abs(actuator_pos_error.detach()), dim=-1).values

    def _apply_action(self) -> None:
        if self.cfg.use_reduced_hydraulic_rom:
            self.robot.set_joint_position_target(self._rom_cylinder_pos_target, joint_ids=self._command_joint_ids)
            return

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
            q, qdot, self._q_ref_commanded, self._qdot_ref, self._prev_action, self._prev_valve_cmd, self._z_arm_hat
        )
        if obs.shape[-1] != self.cfg.observation_space:
            raise RuntimeError(
                f"Observation size mismatch: got {obs.shape[-1]}, expected {self.cfg.observation_space}"
            )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        q, qdot = self._read_joint_state()
        current_q_error_mean = self._weighted_q_error_mean(q, self._q_ref_commanded)
        reward_weights = self._reward_weights_for_control_mode()
        is_settle_phase = self._q_ref_ramp_progress > float(self.cfg.settle_ramp_progress_threshold)
        if self.cfg.debug_reward_terms:
            reward, reward_debug = joint_reach_reward_debug(
                q,
                qdot,
                self._q_ref_commanded,
                self._action_delta,
                self._current_valve_cmd,
                self._current_torque,
                self.reset_terminated,
                reward_weights,
                action=self._current_action,
                prev_q_error_mean=self._prev_q_error_mean,
                torque_effort_weight=reward_weights.torque_effort,
                q_tracking_joint_weights=self._q_tracking_reward_joint_weights,
                is_settle_phase=is_settle_phase,
                use_settle_tracking_reward=self.cfg.use_settle_tracking_reward,
                use_joint1_settle_tracking_reward=self.cfg.use_joint1_settle_tracking_reward,
            )
            self._reward_debug = self._mean_reward_debug(reward_debug)
            self._reward_debug["control_mode"] = self.cfg.control_mode
            self._reward_debug["position_delta_source"] = self.cfg.position_delta_source
            self._reward_debug["use_sd_observer"] = str(self.cfg.use_sd_observer)
            self._reward_debug["use_sd_feedback"] = str(self.cfg.use_sd_feedback)
            self._reward_debug["sd_feedback_beta"] = torch.tensor(float(self.cfg.sd_feedback_beta), device=self.device)
            self._reward_debug["use_sd_feedback_blend"] = str(self.cfg.use_sd_feedback_blend)
            self._reward_debug["use_sd_feedback_rate_limit"] = str(self.cfg.use_sd_feedback_rate_limit)
            self._reward_debug["use_fd_effort"] = str(self.cfg.use_fd_effort)
            self._reward_debug["use_reduced_hydraulic_rom"] = str(self.cfg.use_reduced_hydraulic_rom)
            self._reward_debug["rom_enabled"] = str(self.cfg.use_reduced_hydraulic_rom)
            self._reward_debug["rom_output_mode"] = self.cfg.rom_output_mode
            self._reward_debug["rom_use_sd_feedback"] = str(self.cfg.rom_use_sd_feedback)
            self._reward_debug["rom_use_id_valve_cmd"] = str(self.cfg.rom_use_id_valve_cmd)
            self._add_reference_reset_debug(self._reward_debug, q)
            if self.cfg.log_full_pipeline_debug:
                self._add_full_pipeline_debug(self._reward_debug)
        else:
            reward = joint_reach_reward(
                q,
                qdot,
                self._q_ref_commanded,
                self._action_delta,
                self._current_valve_cmd,
                self._current_torque,
                self.reset_terminated,
                reward_weights,
                action=self._current_action,
                prev_q_error_mean=self._prev_q_error_mean,
                torque_effort_weight=reward_weights.torque_effort,
                q_tracking_joint_weights=self._q_tracking_reward_joint_weights,
                is_settle_phase=is_settle_phase,
                use_settle_tracking_reward=self.cfg.use_settle_tracking_reward,
                use_joint1_settle_tracking_reward=self.cfg.use_joint1_settle_tracking_reward,
            )
            self._reward_debug = {}
        progress_alpha = min(1.0, max(0.0, float(self.cfg.progress_ema_alpha)))
        self._prev_q_error_mean = (
            (1.0 - progress_alpha) * self._prev_q_error_mean + progress_alpha * current_q_error_mean
        ).detach().clone()
        self._prev_action = self._current_action.clone()
        self._prev_valve_cmd = self._current_valve_cmd.clone()
        return reward

    def _reward_weights_for_control_mode(self) -> JointReachRewardWeights:
        if self.cfg.control_mode == "position_delta" and not self.cfg.use_fd_effort:
            return replace(self._reward_weights, torque_effort=0.0)
        return self._reward_weights

    def _weighted_q_error_mean(self, q: torch.Tensor, q_ref: torch.Tensor) -> torch.Tensor:
        joint_weights = self._q_tracking_reward_joint_weights.reshape(1, -1)
        return torch.sum(torch.abs(q - q_ref) * joint_weights, dim=-1) / torch.clamp(
            torch.sum(joint_weights), min=1.0e-8
        )

    def _mean_reward_debug(self, reward_debug: dict[str, torch.Tensor]) -> dict[str, torch.Tensor | str]:
        mean_debug: dict[str, torch.Tensor | str] = {}
        per_joint_keys = {
            "q_error_abs_per_joint",
            "q_error_weighted_per_joint",
            "q_tracking_reward_per_joint",
            "q_tracking_reward_joint_weights",
        }
        for name, value in reward_debug.items():
            if name in per_joint_keys:
                mean_debug[name] = value.detach().mean(dim=0)
            else:
                mean_debug[name] = value.detach().mean()
        return mean_debug

    def _record_target_change(
        self,
        env_ids: torch.Tensor,
        previous_q_ref_target: torch.Tensor,
        *,
        manual_target_change: bool,
    ) -> None:
        current_q_ref_target = self._q_ref_target[env_ids].detach().clone()
        target_delta = current_q_ref_target - previous_q_ref_target
        target_delta_abs_max = torch.max(torch.abs(target_delta), dim=-1).values
        changed = target_delta_abs_max > float(self.cfg.target_change_threshold)
        step = torch.full_like(self._target_change_step[env_ids], float(getattr(self, "common_step_counter", 0)))

        self._q_ref_target_delta[env_ids] = target_delta
        self._q_ref_target_observed[env_ids] = current_q_ref_target
        self._target_changed_flag[env_ids] = changed.float()
        self._target_change_step[env_ids] = torch.where(changed, step, self._target_change_step[env_ids])
        self._reset_due_to_manual_target_change[env_ids] = float(manual_target_change) * changed.float()

    def _sync_external_reference_changes(self) -> None:
        q_ref_target_delta = self._q_ref_target.detach() - self._q_ref_target_observed.detach()
        q_ref_target_delta_abs_max = torch.max(torch.abs(q_ref_target_delta), dim=-1).values
        changed = q_ref_target_delta_abs_max > float(self.cfg.target_change_threshold)
        if not torch.any(changed):
            return
        env_ids = torch.nonzero(changed, as_tuple=False).flatten()
        previous_q_ref_target = self._q_ref_target_observed[env_ids].detach().clone()
        self._record_target_change(env_ids, previous_q_ref_target, manual_target_change=True)

    def _current_q_ref_curriculum_progress(self) -> float:
        if not self.cfg.use_q_ref_curriculum or self.cfg.q_ref_curriculum_steps <= 0:
            return 1.0
        step_count = float(getattr(self, "common_step_counter", 0))
        return min(1.0, max(0.0, step_count / float(self.cfg.q_ref_curriculum_steps)))

    def _current_q_ref_target_scale(self) -> tuple[float, float]:
        progress = self._current_q_ref_curriculum_progress()
        if not self.cfg.use_q_ref_curriculum:
            return float(self.cfg.q_ref_target_scale_final), progress
        initial = float(self.cfg.q_ref_target_scale_initial)
        final = float(self.cfg.q_ref_target_scale_final)
        return initial + progress * (final - initial), progress

    def _update_q_ref_commanded(self) -> None:
        previous_commanded = self._q_ref_commanded.detach().clone()
        if self.cfg.use_q_ref_ramp:
            step_dt = float(getattr(self, "step_dt", float(self.cfg.sim.dt) * float(self.cfg.decimation)))
            ramp_steps = max(float(self.cfg.q_ref_ramp_time_s) / max(step_dt, 1.0e-8), 1.0)
            target_change_step = torch.clamp(self._target_change_step, min=0.0)
            steps_since_target_change = torch.clamp(
                float(getattr(self, "common_step_counter", 0)) - target_change_step,
                min=0.0,
            )
            u = torch.clamp(steps_since_target_change / ramp_steps, 0.0, 1.0)
            if self.cfg.q_ref_ramp_profile == "smoothstep":
                s = 3.0 * torch.square(u) - 2.0 * torch.pow(u, 3)
            elif self.cfg.q_ref_ramp_profile == "linear":
                s = u
            else:
                raise ValueError(f"Unsupported q_ref_ramp_profile: {self.cfg.q_ref_ramp_profile!r}")
            commanded = self._q_ref_start + s.reshape(-1, 1) * (self._q_ref_target - self._q_ref_start)
            self._q_ref_ramp_progress = s.detach().clone()
        else:
            commanded = self._q_ref_target.detach().clone()
            self._q_ref_ramp_progress.fill_(1.0)

        self._q_ref_prev = previous_commanded
        self._q_ref_commanded = commanded.detach().clone()
        self._q_ref = self._q_ref_commanded
        self._q_ref_delta = self._q_ref_commanded - previous_commanded

    def _coupled_joint5_reference(self, reference: torch.Tensor) -> torch.Tensor:
        source_idx = self._joint5_coupling_source_joint
        return float(self.cfg.joint5_coupling_gain) * reference[:, source_idx] + float(self.cfg.joint5_coupling_bias)

    def _joint5_coupling_error(self, reference: torch.Tensor) -> torch.Tensor:
        target_idx = self._joint5_coupling_target_joint
        return reference[:, target_idx] - self._coupled_joint5_reference(reference)

    def _apply_joint5_coupling_to_reference(
        self,
        reference: torch.Tensor,
        *,
        lower: torch.Tensor | None = None,
        upper: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.cfg.use_joint5_coupling:
            return reference
        coupled_reference = reference.clone()
        target_idx = self._joint5_coupling_target_joint
        coupled_target = self._coupled_joint5_reference(coupled_reference)
        if lower is not None and upper is not None:
            coupled_target = torch.clamp(coupled_target, lower[:, target_idx], upper[:, target_idx])
        coupled_reference[:, target_idx] = coupled_target
        return coupled_reference

    def _add_reference_reset_debug(self, debug: dict[str, torch.Tensor | str], q: torch.Tensor) -> None:
        self._sync_external_reference_changes()
        q_ref = self._q_ref_commanded.detach()
        q_actual = q.detach()
        _, qdot = self._read_joint_state()
        qdot = qdot.detach()
        q_error_signed = q_ref - q_actual
        q_error_abs = torch.abs(q_error_signed)
        q_ref_delta = self._q_ref_delta.detach()
        q_ref_delta_abs = torch.abs(q_ref_delta)
        q_ref_target_delta_abs = torch.abs(self._q_ref_target_delta.detach())
        q_ref_delta_abs_max_per_env = torch.max(q_ref_delta_abs, dim=-1).values
        q_ref_target_delta_abs_max_per_env = torch.max(q_ref_target_delta_abs, dim=-1).values
        q_ref_jump_fraction = torch.mean(
            (q_ref_delta_abs > float(self.cfg.q_ref_jump_warning_threshold)).float(), dim=-1
        )
        current_step = torch.full_like(self._target_change_step, float(getattr(self, "common_step_counter", 0)))
        target_hold_steps = torch.where(
            self._target_change_step >= 0.0,
            torch.clamp(current_step - self._target_change_step, min=0.0),
            torch.zeros_like(self._target_change_step),
        )
        reset_buf = self._last_done_debug.float()
        if hasattr(self, "reset_buf"):
            reset_buf = getattr(self, "reset_buf").detach().float()

        debug["q_actual"] = torch.mean(q_actual, dim=0)
        debug["q_ref"] = torch.mean(q_ref, dim=0)
        debug["q_ref_commanded"] = torch.mean(q_ref, dim=0)
        debug["q_ref_target"] = torch.mean(self._q_ref_target.detach(), dim=0)
        debug["q_ref_start"] = torch.mean(self._q_ref_start.detach(), dim=0)
        debug["q_error_signed_per_joint"] = torch.mean(q_error_signed, dim=0)
        debug["q_error_abs_per_joint"] = torch.mean(q_error_abs, dim=0)
        debug["q_ref_prev"] = torch.mean(self._q_ref_prev.detach(), dim=0)
        debug["q_ref_delta_per_joint"] = torch.mean(q_ref_delta, dim=0)
        debug["q_ref_delta_abs_per_joint"] = torch.mean(q_ref_delta_abs, dim=0)
        debug["q_ref_delta_abs_max"] = torch.mean(q_ref_delta_abs_max_per_env)
        debug["q_ref_target_delta_abs_max"] = torch.mean(q_ref_target_delta_abs_max_per_env)
        debug["q_ref_jump_fraction"] = torch.mean(q_ref_jump_fraction)
        debug["q_ref_ramp_progress"] = torch.mean(self._q_ref_ramp_progress.detach())
        debug["q_ref_target_scale"] = torch.mean(self._q_ref_target_scale.detach())
        debug["q_ref_ramp_time_s"] = torch.tensor(float(self.cfg.q_ref_ramp_time_s), device=self.device)
        debug["q_ref_curriculum_progress"] = torch.mean(self._q_ref_curriculum_progress.detach())
        debug["joint1_q_actual"] = torch.mean(q_actual[:, 0])
        debug["joint1_q_ref"] = torch.mean(q_ref[:, 0])
        debug["joint1_q_error_signed"] = torch.mean(q_error_signed[:, 0])
        debug["joint1_q_error_abs"] = torch.mean(q_error_abs[:, 0])
        debug["joint1_q_ref_delta_abs"] = torch.mean(q_ref_delta_abs[:, 0])
        joint5_coupled_ref = self._coupled_joint5_reference(q_ref)
        joint5_coupling_error = self._joint5_coupling_error(q_ref)
        target_idx = self._joint5_coupling_target_joint
        source_idx = self._joint5_coupling_source_joint
        debug["use_joint5_coupling"] = str(self.cfg.use_joint5_coupling)
        debug["joint5_coupling_gain"] = torch.tensor(float(self.cfg.joint5_coupling_gain), device=self.device)
        debug["joint5_coupling_bias"] = torch.tensor(float(self.cfg.joint5_coupling_bias), device=self.device)
        debug["joint5_coupling_source_joint"] = torch.tensor(float(source_idx), device=self.device)
        debug["joint5_coupling_target_joint"] = torch.tensor(float(target_idx), device=self.device)
        debug["joint4_q_ref"] = torch.mean(q_ref[:, source_idx])
        debug["joint5_q_ref"] = torch.mean(q_ref[:, target_idx])
        debug["joint5_coupled_ref"] = torch.mean(joint5_coupled_ref)
        debug["joint5_coupling_error"] = torch.mean(joint5_coupling_error)
        debug["joint5_coupling_error_abs_max"] = torch.max(torch.abs(joint5_coupling_error))
        debug["joint5_actual"] = torch.mean(q_actual[:, target_idx])
        debug["joint5_qdot"] = torch.mean(qdot[:, target_idx])
        debug["joint5_tracking_reward_weight"] = self._q_tracking_reward_joint_weights[target_idx].detach().clone()
        debug["joint5_target_amplitude"] = torch.tensor(
            float(self.cfg.q_ref_target_amplitude[target_idx]), device=self.device
        )
        debug["episode_step"] = torch.mean(self.episode_length_buf.detach().float())
        debug["episode_length_buf"] = torch.mean(self.episode_length_buf.detach().float())
        debug["reset_buf_mean"] = torch.mean(reset_buf)
        debug["done_buf_mean"] = torch.mean(self._last_done_debug.detach().float())
        debug["terminated_frac"] = torch.mean(self._last_terminated_debug.detach().float())
        debug["time_out_frac"] = torch.mean(self._last_time_out_debug.detach().float())
        debug["target_changed_flag"] = torch.mean(self._target_changed_flag.detach())
        debug["target_change_step"] = torch.mean(self._target_change_step.detach())
        debug["target_hold_steps"] = torch.mean(target_hold_steps)
        debug["last_reset_step_global"] = torch.mean(self._last_reset_step_global.detach())
        debug["q_ref_sampled_on_reset"] = torch.mean(self._q_ref_sampled_on_reset.detach(), dim=0)
        debug["reset_due_to_timeout"] = torch.mean(self._reset_due_to_timeout.detach())
        debug["reset_due_to_termination"] = torch.mean(self._reset_due_to_termination.detach())
        debug["reset_due_to_manual_target_change"] = torch.mean(self._reset_due_to_manual_target_change.detach())

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

    def _zero_sd_observer_state(self) -> None:
        self._sd_q_next_pred.zero_()
        self._sd_qdot_next_pred.zero_()
        self._sd_dP_next_pred.zero_()
        self._sd_fnet_next_pred.zero_()
        self._sd_output_nan_or_inf.zero_()
        self._sd_output_clamped_fraction.zero_()

    def _zero_sd_feedback_state(self) -> None:
        self._sd_feedback_dP_delta_mean.zero_()
        self._sd_feedback_dP_delta_max.zero_()
        self._sd_feedback_fnet_delta_mean.zero_()
        self._sd_feedback_fnet_delta_max.zero_()
        self._sd_feedback_rate_limited_fraction.zero_()
        self._sd_feedback_clamped_fraction.zero_()
        self._sd_feedback_nan_or_inf.zero_()

    def _store_sd_observer_output(self, sd_observer) -> None:
        self._sd_q_next_pred = sd_observer.q_next_pred.detach().clone()
        self._sd_qdot_next_pred = sd_observer.qdot_next_pred.detach().clone()
        self._sd_dP_next_pred = sd_observer.dP_next_pred.detach().clone()
        self._sd_fnet_next_pred = sd_observer.fnet_next_pred.detach().clone()
        self._sd_output_nan_or_inf = sd_observer.output_nan_or_inf.detach().clone()
        self._sd_output_clamped_fraction = sd_observer.output_clamped_fraction.detach().clone()

    def _apply_sd_feedback(
        self,
        *,
        previous_dP: torch.Tensor,
        previous_fnet: torch.Tensor,
        dP_target: torch.Tensor,
        fnet_target: torch.Tensor,
    ) -> None:
        previous_dP = clamp_and_sanitize(previous_dP, self.cfg.max_abs_pressure_delta)
        previous_fnet = clamp_and_sanitize(previous_fnet, self.cfg.max_abs_fnet)
        dP_target, dP_target_invalid = self._replace_invalid_with_fallback(dP_target, previous_dP)
        fnet_target, fnet_target_invalid = self._replace_invalid_with_fallback(fnet_target, previous_fnet)

        beta = min(1.0, max(0.0, float(self.cfg.sd_feedback_beta)))
        if self.cfg.use_sd_feedback_blend:
            dP_candidate = (1.0 - beta) * previous_dP + beta * dP_target
            fnet_candidate = (1.0 - beta) * previous_fnet + beta * fnet_target
        else:
            dP_candidate = dP_target
            fnet_candidate = fnet_target

        dP_candidate, dP_candidate_invalid = self._replace_invalid_with_fallback(dP_candidate, previous_dP)
        fnet_candidate, fnet_candidate_invalid = self._replace_invalid_with_fallback(fnet_candidate, previous_fnet)

        dP_limited, dP_rate_mask = self._rate_limit_feedback(
            dP_candidate,
            previous_dP,
            float(self.cfg.sd_dP_rate_limit),
            enabled=self.cfg.use_sd_feedback_rate_limit,
        )
        fnet_limited, fnet_rate_mask = self._rate_limit_feedback(
            fnet_candidate,
            previous_fnet,
            float(self.cfg.sd_fnet_rate_limit),
            enabled=self.cfg.use_sd_feedback_rate_limit,
        )
        dP_limited, dP_limited_invalid = self._replace_invalid_with_fallback(dP_limited, previous_dP)
        fnet_limited, fnet_limited_invalid = self._replace_invalid_with_fallback(fnet_limited, previous_fnet)

        dP_clamp_mask = torch.abs(dP_limited) > max(float(self.cfg.max_abs_pressure_delta), 1.0e-8)
        fnet_clamp_mask = torch.abs(fnet_limited) > max(float(self.cfg.max_abs_fnet), 1.0e-8)
        dP_next = clamp_and_sanitize(dP_limited, self.cfg.max_abs_pressure_delta)
        fnet_next = clamp_and_sanitize(fnet_limited, self.cfg.max_abs_fnet)

        feedback_invalid = (
            dP_target_invalid
            | fnet_target_invalid
            | dP_candidate_invalid
            | fnet_candidate_invalid
            | dP_limited_invalid
            | fnet_limited_invalid
        )
        feedback_delta_dP = torch.abs(dP_next - previous_dP)
        feedback_delta_fnet = torch.abs(fnet_next - previous_fnet)
        rate_mask = torch.cat([dP_rate_mask, fnet_rate_mask], dim=-1)
        clamp_mask = torch.cat([dP_clamp_mask, fnet_clamp_mask], dim=-1)

        self._dP = dP_next.detach().clone()
        self._fnet = fnet_next.detach().clone()
        self._sd_feedback_dP_delta_mean = torch.mean(feedback_delta_dP.detach(), dim=-1)
        self._sd_feedback_dP_delta_max = torch.max(feedback_delta_dP.detach(), dim=-1).values
        self._sd_feedback_fnet_delta_mean = torch.mean(feedback_delta_fnet.detach(), dim=-1)
        self._sd_feedback_fnet_delta_max = torch.max(feedback_delta_fnet.detach(), dim=-1).values
        self._sd_feedback_rate_limited_fraction = torch.mean(rate_mask.float(), dim=-1)
        self._sd_feedback_clamped_fraction = torch.mean(clamp_mask.float(), dim=-1)
        self._sd_feedback_nan_or_inf = feedback_invalid.float()

    def _replace_invalid_with_fallback(
        self,
        value: torch.Tensor,
        fallback: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        finite = torch.isfinite(value)
        sanitized = torch.where(finite, value, fallback)
        invalid = (~finite).flatten(start_dim=1).any(dim=-1)
        return sanitized, invalid

    def _rate_limit_feedback(
        self,
        target: torch.Tensor,
        previous: torch.Tensor,
        limit: float,
        *,
        enabled: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not enabled:
            return target, torch.zeros_like(target, dtype=torch.bool)
        limit = max(float(limit), 0.0)
        delta = target - previous
        limited_delta = torch.clamp(delta, -limit, limit)
        rate_limited = torch.abs(delta) > limit
        return previous + limited_delta, rate_limited

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
        debug["joint1_qddot_cmd_abs"] = torch.mean(torch.abs(self._current_qddot_cmd.detach()[:, 0]))
        debug["joint1_valve_cmd_abs"] = torch.mean(torch.abs(self._current_valve_cmd.detach()[:, 0]))
        debug["joint1_fd_tau_raw_abs"] = torch.mean(torch.abs(self._current_fd_simscape_output.detach()[:, 0]))
        debug["joint1_tau_fd_adapted_abs"] = torch.mean(torch.abs(self._current_tau_fd_adapted.detach()[:, 0]))
        debug["joint1_tau_applied_abs"] = torch.mean(torch.abs(self._current_tau_applied_raw.detach()[:, 0]))
        debug["joint1_torque_abs"] = torch.mean(torch.abs(self._current_torque.detach()[:, 0]))
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
        debug["qddot_cmd_scale"] = torch.tensor(float(self.cfg.action_to_qddot_scale), device=self.device)
        debug["rew_q_tracking"] = torch.tensor(float(self.cfg.rew_q_tracking), device=self.device)
        debug["q_tracking_reward_joint_weights"] = self._q_tracking_reward_joint_weights.detach().clone()
        debug["q_tracking_debug_joint_weights"] = self._q_tracking_debug_joint_weights.detach().clone()
        debug["rew_action_saturation"] = torch.tensor(float(self.cfg.rew_action_saturation), device=self.device)
        debug["rew_action_magnitude"] = torch.tensor(float(self.cfg.rew_action_magnitude), device=self.device)
        debug["rew_action_rate"] = torch.tensor(float(self.cfg.rew_action_rate), device=self.device)
        debug["rew_progress"] = torch.tensor(float(self.cfg.rew_progress), device=self.device)
        debug["use_settle_tracking_reward"] = str(self.cfg.use_settle_tracking_reward)
        debug["settle_ramp_progress_threshold"] = torch.tensor(
            float(self.cfg.settle_ramp_progress_threshold), device=self.device
        )
        debug["rew_settle_tracking"] = torch.tensor(float(self.cfg.rew_settle_tracking), device=self.device)
        debug["use_joint1_settle_tracking_reward"] = str(self.cfg.use_joint1_settle_tracking_reward)
        debug["rew_joint1_settle_tracking"] = torch.tensor(
            float(self.cfg.rew_joint1_settle_tracking), device=self.device
        )
        debug["joint1_settle_error_deadband"] = torch.tensor(
            float(self.cfg.joint1_settle_error_deadband), device=self.device
        )
        debug["action_saturation_threshold"] = torch.tensor(
            float(self.cfg.action_saturation_threshold), device=self.device
        )
        debug["progress_ema_alpha"] = torch.tensor(float(self.cfg.progress_ema_alpha), device=self.device)
        debug["dP_mean"] = torch.mean(torch.abs(self._dP.detach()))
        debug["dP_abs_max"] = torch.max(torch.abs(self._dP.detach()))
        debug["fnet_mean"] = torch.mean(torch.abs(self._fnet.detach()))
        debug["fnet_abs_max"] = torch.max(torch.abs(self._fnet.detach()))
        debug["sd_feedback_beta"] = torch.tensor(float(self.cfg.sd_feedback_beta), device=self.device)
        debug["use_sd_feedback_blend"] = str(self.cfg.use_sd_feedback_blend)
        debug["use_sd_feedback_rate_limit"] = str(self.cfg.use_sd_feedback_rate_limit)
        debug["sd_feedback_dP_delta_mean"] = torch.mean(self._sd_feedback_dP_delta_mean.detach())
        debug["sd_feedback_dP_delta_max"] = torch.max(self._sd_feedback_dP_delta_max.detach())
        debug["sd_feedback_fnet_delta_mean"] = torch.mean(self._sd_feedback_fnet_delta_mean.detach())
        debug["sd_feedback_fnet_delta_max"] = torch.max(self._sd_feedback_fnet_delta_max.detach())
        debug["sd_feedback_rate_limited_fraction"] = torch.mean(self._sd_feedback_rate_limited_fraction.detach())
        debug["sd_feedback_clamped_fraction"] = torch.mean(self._sd_feedback_clamped_fraction.detach())
        debug["sd_feedback_nan_or_inf"] = torch.mean(self._sd_feedback_nan_or_inf.detach())
        if self.cfg.sd_observer_log:
            debug["sd_q_next_pred"] = torch.mean(self._sd_q_next_pred.detach(), dim=0)
            debug["sd_qdot_next_pred"] = torch.mean(self._sd_qdot_next_pred.detach(), dim=0)
            debug["sd_dP_next_pred"] = torch.mean(self._sd_dP_next_pred.detach(), dim=0)
            debug["sd_fnet_next_pred"] = torch.mean(self._sd_fnet_next_pred.detach(), dim=0)
            debug["sd_dP_mean"] = torch.mean(torch.abs(self._sd_dP_next_pred.detach()))
            debug["sd_dP_abs_max"] = torch.max(torch.abs(self._sd_dP_next_pred.detach()))
            debug["sd_fnet_mean"] = torch.mean(torch.abs(self._sd_fnet_next_pred.detach()))
            debug["sd_fnet_abs_max"] = torch.max(torch.abs(self._sd_fnet_next_pred.detach()))
            debug["sd_output_nan_or_inf"] = torch.mean(self._sd_output_nan_or_inf.detach())
            debug["sd_output_clamped_fraction"] = torch.mean(self._sd_output_clamped_fraction.detach())
        if self.cfg.rom_debug_log:
            self._add_rom_debug(debug)

    def _add_rom_debug(self, debug: dict[str, torch.Tensor | str]) -> None:
        x_cyl = self.robot.data.joint_pos[:, self._command_joint_ids].detach()
        target = self._rom_cylinder_pos_target.detach()
        debug["rom_enabled"] = str(self.cfg.use_reduced_hydraulic_rom)
        debug["rom_output_mode"] = self.cfg.rom_output_mode
        debug["rom_use_sd_feedback"] = str(self.cfg.rom_use_sd_feedback)
        debug["rom_use_id_valve_cmd"] = str(self.cfg.rom_use_id_valve_cmd)
        debug["rom_dt"] = torch.tensor(float(self.cfg.rom_dt), device=self.device)
        debug["rom_valve_cmd_mean"] = torch.mean(torch.abs(self._current_valve_cmd.detach()))
        debug["rom_valve_cmd_abs_max"] = torch.max(torch.abs(self._current_valve_cmd.detach()))
        debug["rom_dP_mean"] = torch.mean(torch.abs(self._dP.detach()))
        debug["rom_P_L_mean"] = torch.mean(self._rom_pressure.detach())
        debug["rom_P_L_abs_max"] = torch.max(torch.abs(self._rom_pressure.detach()))
        debug["rom_xdot_pred_mean"] = torch.mean(torch.abs(self._rom_xdot_pred.detach()))
        debug["rom_xdot_pred_abs_max"] = torch.max(torch.abs(self._rom_xdot_pred.detach()))
        debug["rom_dx_pred_mean"] = torch.mean(torch.abs(self._rom_dx_pred.detach()))
        debug["rom_dx_pred_abs_max"] = torch.max(torch.abs(self._rom_dx_pred.detach()))
        debug["rom_target_min"] = torch.min(target)
        debug["rom_target_max"] = torch.max(target)
        debug["rom_target_clamp_fraction"] = torch.mean(self._rom_target_clamp_fraction.detach())
        debug["rom_pressure_clamp_fraction"] = torch.mean(self._rom_pressure_clamp_fraction.detach())
        debug["rom_velocity_clamp_fraction"] = torch.mean(self._rom_velocity_clamp_fraction.detach())
        debug["rom_dx_clamp_fraction"] = torch.mean(self._rom_dx_clamp_fraction.detach())
        debug["rom_nan_guard_triggered"] = torch.mean(self._rom_nan_guard_triggered.detach())
        debug["rom_output_zeroed_fraction"] = torch.mean(self._rom_output_zeroed_fraction.detach())
        debug["rom_actuator_pos_error_mean"] = torch.mean(self._rom_actuator_pos_error_mean.detach())
        debug["rom_actuator_pos_error_abs_max"] = torch.max(self._rom_actuator_pos_error_abs_max.detach())
        debug["rom_cylinder_pos_target"] = torch.mean(target, dim=0)
        debug["rom_cylinder_actual_pos"] = torch.mean(x_cyl, dim=0)
        debug["rom_cylinder_pos_error"] = torch.mean(target - x_cyl, dim=0)
        debug["rom_xdot_pred"] = torch.mean(self._rom_xdot_pred.detach(), dim=0)
        debug["rom_dx_pred"] = torch.mean(self._rom_dx_pred.detach(), dim=0)
        debug["rom_pressure"] = torch.mean(self._rom_pressure.detach(), dim=0)

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
            & torch.isfinite(self._rom_cylinder_pos_target).all(dim=-1)
            & torch.isfinite(self._rom_pressure).all(dim=-1)
            & torch.isfinite(self._rom_xdot_pred).all(dim=-1)
            & torch.isfinite(self._rom_dx_pred).all(dim=-1)
        )
        terminated = safety_terminated | (~finite_control)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        self._last_terminated_debug = terminated.detach().clone()
        self._last_time_out_debug = time_out.detach().clone()
        self._last_done_debug = (terminated | time_out).detach().clone()
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        reset_due_to_timeout = self._last_time_out_debug[env_ids].detach().float().clone()
        reset_due_to_termination = self._last_terminated_debug[env_ids].detach().float().clone()
        previous_q_ref_target = self._q_ref_target[env_ids].detach().clone()

        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        default_root_pose, default_root_velocity = self._default_root_state(env_ids)
        default_root_pose[:, :3] += self.scene.env_origins[env_ids]

        q_reset = joint_pos[:, self._state_joint_ids].detach().clone()
        q_ref_center = q_reset
        if not self.cfg.q_ref_sample_relative_to_current:
            q_ref_center = self.robot.data.default_joint_pos[env_ids][:, self._state_joint_ids].detach().clone()

        q_ref_start = self._apply_joint5_coupling_to_reference(q_reset)

        self._q_ref_start[env_ids] = q_ref_start
        self._q_ref_commanded[env_ids] = q_ref_start
        self._q_ref_prev[env_ids] = q_ref_start
        self._q_ref_delta[env_ids] = 0.0
        self._q_ref[env_ids] = q_ref_start
        self._sample_joint_targets(env_ids, q_ref_center)
        self._q_ref_sampled_on_reset[env_ids] = self._q_ref_target[env_ids].detach().clone()
        self._last_reset_step_global[env_ids] = float(getattr(self, "common_step_counter", 0))
        self._reset_due_to_timeout[env_ids] = reset_due_to_timeout
        self._reset_due_to_termination[env_ids] = reset_due_to_termination
        self._record_target_change(env_ids, previous_q_ref_target, manual_target_change=False)
        self._q_ref_target_delta[env_ids] = self._q_ref_target[env_ids] - self._q_ref_start[env_ids]
        self._q_ref_target_observed[env_ids] = self._q_ref_target[env_ids].detach().clone()
        self._q_ref_ramp_progress[env_ids] = 0.0 if self.cfg.use_q_ref_ramp else 1.0
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
        self._rom_cylinder_pos_target[env_ids] = joint_pos[:, self._command_joint_ids]
        self._rom_pressure[env_ids] = 0.0
        self._rom_xdot_pred[env_ids] = 0.0
        self._rom_dx_pred[env_ids] = 0.0
        self._rom_valve_cmd_delayed[env_ids] = 0.0
        self._rom_valve_cmd_effective[env_ids] = 0.0
        self._rom_target_clamp_fraction[env_ids] = 0.0
        self._rom_pressure_clamp_fraction[env_ids] = 0.0
        self._rom_velocity_clamp_fraction[env_ids] = 0.0
        self._rom_dx_clamp_fraction[env_ids] = 0.0
        self._rom_nan_guard_triggered[env_ids] = 0.0
        self._rom_output_zeroed_fraction[env_ids] = 0.0
        self._rom_actuator_pos_error_mean[env_ids] = 0.0
        self._rom_actuator_pos_error_abs_max[env_ids] = 0.0
        self._sd_q_next_pred[env_ids] = 0.0
        self._sd_qdot_next_pred[env_ids] = 0.0
        self._sd_dP_next_pred[env_ids] = 0.0
        self._sd_fnet_next_pred[env_ids] = 0.0
        self._sd_output_nan_or_inf[env_ids] = 0.0
        self._sd_output_clamped_fraction[env_ids] = 0.0
        self._sd_feedback_dP_delta_mean[env_ids] = 0.0
        self._sd_feedback_dP_delta_max[env_ids] = 0.0
        self._sd_feedback_fnet_delta_mean[env_ids] = 0.0
        self._sd_feedback_fnet_delta_max[env_ids] = 0.0
        self._sd_feedback_rate_limited_fraction[env_ids] = 0.0
        self._sd_feedback_clamped_fraction[env_ids] = 0.0
        self._sd_feedback_nan_or_inf[env_ids] = 0.0
        self._torque_clamp_fraction[env_ids] = 0.0
        self._torque_adapter_clamp_fraction[env_ids] = 0.0
        self._torque_rate_limited_fraction[env_ids] = 0.0
        self._z_arm_hat[env_ids] = 0.0
        self._prev_q_error_mean[env_ids] = self._weighted_q_error_mean(
            joint_pos[:, self._state_joint_ids], self._q_ref_commanded[env_ids]
        )
        self._controller.reset(env_ids)
        self._reduced_hydraulic_rom.reset(env_ids)

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

    def _sample_joint_targets(self, env_ids: torch.Tensor, q_ref_center: torch.Tensor) -> None:
        default_q = self.robot.data.default_joint_pos[env_ids][:, self._state_joint_ids]
        lower = self.robot.data.soft_joint_pos_limits[env_ids][:, self._state_joint_ids, 0]
        upper = self.robot.data.soft_joint_pos_limits[env_ids][:, self._state_joint_ids, 1]
        target_range = torch.tensor(self.cfg.target_joint_range, dtype=torch.float32, device=self.device).reshape(1, 5)
        target_amplitude = torch.tensor(
            self.cfg.q_ref_target_amplitude, dtype=torch.float32, device=self.device
        ).reshape(1, 5)
        target_scale, curriculum_progress = self._current_q_ref_target_scale()
        self._q_ref_target_scale[env_ids] = float(target_scale)
        self._q_ref_curriculum_progress[env_ids] = float(curriculum_progress)

        has_limits = torch.isfinite(lower) & torch.isfinite(upper)
        has_limits = has_limits & ((upper - lower) > 2.0 * self.cfg.joint_limit_margin)
        fallback_lower = default_q - target_range
        fallback_upper = default_q + target_range
        safe_lower = torch.where(has_limits, lower + self.cfg.joint_limit_margin, fallback_lower)
        safe_upper = torch.where(has_limits, upper - self.cfg.joint_limit_margin, fallback_upper)
        sample_lower = q_ref_center - float(target_scale) * target_amplitude
        sample_upper = q_ref_center + float(target_scale) * target_amplitude
        sample_lower = torch.maximum(sample_lower, safe_lower)
        sample_upper = torch.minimum(sample_upper, safe_upper)
        sample_upper = torch.maximum(sample_upper, sample_lower + 1.0e-3)

        q_ref_target = sample_lower + torch.rand_like(default_q) * (sample_upper - sample_lower)
        q_ref_target = self._apply_joint5_coupling_to_reference(q_ref_target, lower=safe_lower, upper=safe_upper)
        self._q_ref_target[env_ids] = q_ref_target
