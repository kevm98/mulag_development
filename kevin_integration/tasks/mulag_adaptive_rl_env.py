from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply_inverse, subtract_frame_transforms

from kevin_integration.controllers.reduced_hydraulic_actuator import (
    ReducedHydraulicActuatorROM,
    ReducedHydraulicActuatorROMCfg,
)
from kevin_integration.rom_cylinders.cylinder_coordinate_adapter import (
    CylinderCoordinateMapping,
    load_coordinate_mapping,
    mapping_metadata_for_logging,
)
from kevin_integration.rom_cylinders.multirate_rom_wrapper import MultirateROMConfig, MultirateROMWrapper
from kevin_integration.rom_cylinders.rom_data_contract import CORE_INPUT_FEATURE_NAMES, HISTORY_LENGTH
from kevin_integration.rom_cylinders.valve_conditioned_causal_rom import ValveConditionedROMInference
from kevin_integration.rl import AdaptiveRLLearnedController, AdaptiveRLLearnedControllerCfg, build_rl_obs, build_task_space_rl_obs
from kevin_integration.rl.action_adapter import clamp_and_sanitize
from kevin_integration.rl.reward_terms import (
    JointReachRewardWeights,
    avoidable_clamp_penalty,
    fd_load_excess_penalty,
    final_pose_orientation_reward,
    final_pose_within_tolerance,
    hold_progress_reward,
    joint_reach_reward,
    joint_reach_reward_debug,
    joint_safety_termination,
    near_target_angular_settling_reward,
    near_target_linear_settling_reward,
    normalize_quaternion_wxyz,
    normalized_task_space_potential,
    potential_progress_shaping,
    quaternion_geodesic_error,
    update_consecutive_success_counter,
    valve_saturation_penalty,
)
from kevin_integration.tasks.asset_preflight_classifier import (
    ASSET_TRANSMISSION_PASS,
    AssetPreflightGates,
    classify_asset_transmission_preflight,
)
from kevin_integration.tasks.mechanical_constraint_resolver import resolve_mechanical_constraint_family_ids
from kevin_integration.tasks.mulag_adaptive_rl_env_cfg import MulagAdaptiveRLJointReachEnvCfg
from kevin_integration.tasks.task_space_reference import (
    BLOCKED_EE_BODY_RESOLUTION,
    BLOCKED_FD_ACTUATOR_CONFLICT,
    BLOCKED_FD_INPUT_CONTRACT,
    BLOCKED_FD_OUTPUT_NONFINITE,
    BLOCKED_TASKSPACE_COMMAND,
    BLOCKED_TASKSPACE_OBSERVATION,
    BLOCKED_TASKSPACE_REWARD,
    BLOCKED_VEHICLE_COMMAND,
    READY_FOR_PPO,
    TASK_SPACE_EE_BODY,
    TASK_SPACE_FRAME,
    TASK_SPACE_OBSERVATION_DIM,
    TASK_SPACE_POSE_ORDER,
    TASK_SPACE_QUATERNION_ORDER,
    validate_task_space_observation_contract,
)


class MulagAdaptiveRLEnv(DirectRLEnv):
    cfg: MulagAdaptiveRLJointReachEnvCfg
    _runtime_contract_printed = False
    _reward_contract_printed = False

    def __init__(self, cfg: MulagAdaptiveRLJointReachEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._mechanical_joint_ids = self._find_joint_ids(self.cfg.mechanical_joint_names)
        self._cylinder_joint_ids = self._find_joint_ids(self.cfg.cylinder_joint_names)
        # Backward-compatible aliases: "state" is the learned-model mechanical
        # state; "command" is the low-level cylinder/ROM writer state.
        self._state_joint_ids = self._mechanical_joint_ids
        self._command_joint_ids = self._cylinder_joint_ids
        self._vehicle_wheel_joint_ids = self._find_optional_joint_ids(self.cfg.vehicle_wheel_joint_names)
        self._ee_body_id = self._find_optional_body_id(getattr(self.cfg, "task_space_ee_body_name", TASK_SPACE_EE_BODY))
        if len(self._mechanical_joint_ids) != 5:
            raise RuntimeError(f"Expected 5 mechanical feature joints, got {len(self._mechanical_joint_ids)}")
        if len(self._cylinder_joint_ids) != 4:
            raise RuntimeError(f"Expected 4 cylinder joints, got {len(self._cylinder_joint_ids)}")
        self._assert_velocity_state_contract()
        self._assert_single_actuator_writer()
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
                max_abs_pressure_delta=getattr(
                    self.cfg,
                    "sd_max_abs_pressure_delta",
                    self.cfg.max_abs_pressure_delta,
                ),
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
        self._final_causal_rom: ValveConditionedROMInference | None = None
        self._final_causal_rom_wrapper: MultirateROMWrapper | None = None
        self._final_causal_rom_training_mean_dp = torch.zeros((4,), dtype=torch.float32, device=self.device)
        self._cylinder_coordinate_mapping: CylinderCoordinateMapping | None = None
        self._scripted_reference_phase_offset = torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)
        self._vehicle_commanded_speed_kmh = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self._vehicle_wheel_velocity_target = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self._vehicle_command_warned_missing_joints = False
        self._preflight_direct_target: torch.Tensor | None = None
        self._preflight_disable_ordinary_resets = False
        self._learned_preflight_scripted_action_active = False
        self._validate_control_mode()
        if self.cfg.use_final_causal_cylinder_rom:
            self._final_causal_rom, self._final_causal_rom_wrapper = self._make_final_causal_rom()
            self._final_causal_rom_training_mean_dp = self._training_mean_dp_from_final_rom()
            self._cylinder_coordinate_mapping = load_coordinate_mapping(
                self.cfg.cylinder_coordinate_mapping,
                mode=self.cfg.cylinder_coordinate_mapping_mode,
                allow_provisional_range_mapping=bool(self.cfg.allow_provisional_range_mapping),
                allow_provisional_nominal_reference=bool(self.cfg.allow_provisional_nominal_reference),
            )
            self._warn_if_provisional_mapping()
        self._print_runtime_contract_once()
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
        self._qddot_ref = torch.zeros_like(self._qdot_ref)
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
        self._reset_count = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self._last_reset_reason_code = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self._reset_due_to_timeout = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self._reset_due_to_termination = torch.zeros_like(self._reset_due_to_timeout)
        self._reset_due_to_manual_target_change = torch.zeros_like(self._reset_due_to_timeout)
        self._reset_due_to_target_reached = torch.zeros_like(self._reset_due_to_timeout)
        self._last_terminated_debug = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._last_time_out_debug = torch.zeros_like(self._last_terminated_debug)
        self._last_done_debug = torch.zeros_like(self._last_terminated_debug)
        self._learned_control_initialized = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._dP = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self._fnet = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self._prev_action = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self._current_action = torch.zeros_like(self._prev_action)
        self._action_delta = torch.zeros_like(self._prev_action)
        self._policy_action_std = torch.zeros((), dtype=torch.float32, device=self.device)
        # ROM 50ms policy-cadence hold buffer (one per env, action held across multirate window).
        self._rom_multirate_policy_step_counter = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self._rom_held_policy_action = None
        self._policy_action_std_min = torch.zeros_like(self._policy_action_std)
        self._policy_action_std_max = torch.zeros_like(self._policy_action_std)
        self._action_std_schedule_hold_fraction = torch.zeros_like(self._policy_action_std)
        self._action_std_schedule_blend = torch.zeros_like(self._policy_action_std)
        self._action_std_schedule_min_std = torch.zeros_like(self._policy_action_std)
        self._action_std_schedule_max_std = torch.zeros_like(self._policy_action_std)
        self._action_std_schedule_entropy_coef = torch.zeros_like(self._policy_action_std)
        self._action_std_schedule_applied = torch.zeros_like(self._policy_action_std)
        self._action_std_schedule_std_mean = torch.zeros_like(self._policy_action_std)
        self._action_std_schedule_std_max = torch.zeros_like(self._policy_action_std)
        self._current_qddot_cmd = torch.zeros((self.num_envs, 5), dtype=torch.float32, device=self.device)
        self._scripted_qddot_feedback = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self._scripted_qddot_feedforward = torch.zeros_like(self._scripted_qddot_feedback)
        self._scripted_qddot_raw = torch.zeros_like(self._scripted_qddot_feedback)
        self._scripted_action_preclip = torch.zeros_like(self._scripted_qddot_feedback)
        self._scripted_control_ramp = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self._prev_valve_cmd = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self._current_valve_cmd = torch.zeros_like(self._prev_valve_cmd)
        self._current_torque = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self._current_fd_simscape_output = torch.zeros_like(self._current_torque)
        self._current_tau_fd_adapted = torch.zeros_like(self._current_torque)
        self._current_tau_applied_raw = torch.zeros_like(self._current_torque)
        self._current_tau_isaac_filtered = torch.zeros_like(self._current_torque)
        self._current_tau_isaac_clamped = torch.zeros_like(self._current_torque)
        self._rom_cylinder_pos_target = torch.zeros_like(self._current_torque)
        self._rom_cylinder_pos_target_prev = torch.zeros_like(self._current_torque)
        self._rom_cylinder_pos_target_at_reset = torch.zeros_like(self._current_torque)
        self._rom_cylinder_measured_at_reset = torch.zeros_like(self._current_torque)
        self._rom_cylinder_cumulative_target_motion = torch.zeros_like(self._current_torque)
        self._rom_cylinder_cumulative_measured_motion = torch.zeros_like(self._current_torque)
        self._rom_cylinder_tracking_ratio = torch.zeros_like(self._current_torque)
        self._rom_pressure = torch.zeros_like(self._current_torque)
        self._rom_xdot_pred = torch.zeros_like(self._current_torque)
        self._rom_dx_pred = torch.zeros_like(self._current_torque)
        self._rom_valve_cmd_delayed = torch.zeros_like(self._current_torque)
        self._rom_valve_cmd_effective = torch.zeros_like(self._current_torque)
        self._final_rom_delta_extension_50ms = torch.zeros_like(self._current_torque)
        self._final_rom_delta_extension_kinematic_50ms = torch.zeros_like(self._current_torque)
        self._final_rom_delta_extension_correction_50ms = torch.zeros_like(self._current_torque)
        self._final_rom_delta_extension_10ms = torch.zeros_like(self._current_torque)
        self._final_rom_delta_isaac_50ms = torch.zeros_like(self._current_torque)
        self._final_rom_delta_isaac_10ms = torch.zeros_like(self._current_torque)
        self._final_rom_extension = torch.zeros_like(self._current_torque)
        self._final_rom_extension_velocity = torch.zeros_like(self._current_torque)
        self._prev_hydraulic_q = torch.zeros_like(self._current_torque)
        self._hydraulic_qdot_fd = torch.zeros_like(self._current_torque)
        self._hydraulic_target_velocity = torch.zeros_like(self._current_torque)
        self._hydraulic_rom_applied_rate = torch.zeros_like(self._current_torque)
        self._hydraulic_velocity_persistence = torch.zeros(
            (self.num_envs, 4), dtype=torch.long, device=self.device
        )
        self._hydraulic_velocity_raw_exceedance = torch.zeros(
            (self.num_envs, 4), dtype=torch.bool, device=self.device
        )
        self._hydraulic_velocity_confirmed_hard = torch.zeros_like(
            self._hydraulic_velocity_raw_exceedance
        )
        self._hydraulic_velocity_spike_ignored = torch.zeros_like(
            self._hydraulic_velocity_raw_exceedance
        )
        self._velocity_audit_physx_spike_count = torch.zeros((), dtype=torch.float32, device=self.device)
        self._velocity_audit_target_jump_count = torch.zeros((), dtype=torch.float32, device=self.device)
        self._velocity_audit_genuine_high_motion_count = torch.zeros((), dtype=torch.float32, device=self.device)
        self._velocity_audit_unit_mapping_mismatch_count = torch.zeros((), dtype=torch.float32, device=self.device)
        self._velocity_audit_nonfinite_count = torch.zeros((), dtype=torch.float32, device=self.device)
        self._final_rom_input_features_raw = torch.zeros((self.num_envs, 4, 17), dtype=torch.float32, device=self.device)
        self._final_rom_input_features_normalized = torch.zeros_like(self._final_rom_input_features_raw)
        self._final_rom_inverse_mapping_error = torch.zeros_like(self._current_torque)
        self._final_rom_range_violation_fraction = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self._final_rom_update_flag = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self._final_rom_history_ready = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self._final_rom_valve_ff_delta = torch.zeros_like(self._current_torque)
        self._rom_target_clamp_fraction = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self._rom_boundary_active = torch.zeros((self.num_envs, 4), dtype=torch.bool, device=self.device)
        self._rom_outward_request = torch.zeros_like(self._rom_boundary_active)
        self._rom_blocked_outward_increment = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self._rom_clamp_activation_fraction = torch.zeros_like(self._rom_target_clamp_fraction)
        self._rom_clamp_blocked_outward_mean = torch.zeros_like(self._rom_target_clamp_fraction)
        self._rom_pressure_clamp_fraction = torch.zeros_like(self._rom_target_clamp_fraction)
        self._rom_velocity_clamp_fraction = torch.zeros_like(self._rom_target_clamp_fraction)
        self._rom_dx_clamp_fraction = torch.zeros_like(self._rom_target_clamp_fraction)
        self._rom_nan_guard_triggered = torch.zeros_like(self._rom_target_clamp_fraction)
        self._rom_output_zeroed_fraction = torch.zeros_like(self._rom_target_clamp_fraction)
        self._rom_actuator_pos_error_mean = torch.zeros_like(self._rom_target_clamp_fraction)
        self._rom_actuator_pos_error_abs_max = torch.zeros_like(self._rom_target_clamp_fraction)
        self._rom_rate_limit_fraction = torch.zeros_like(self._rom_target_clamp_fraction)
        self._rom_ch4_damping_applied = torch.zeros_like(self._rom_target_clamp_fraction)
        self._ee_pose_command = torch.zeros((self.num_envs, 7), dtype=torch.float32, device=self.device)
        self._ee_pose_command[:, 3] = 1.0
        # Authoritative fixed final-pose target:
        # shape (num_envs, 7), [x, y, z, qw, qx, qy, qz], robot_base frame.
        self._final_pose_target_at_episode_start = self._ee_pose_command.detach().clone()
        self._current_ee_pose = torch.zeros_like(self._ee_pose_command)
        self._current_ee_pose[:, 3] = 1.0
        self._ee_position_error = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self._ee_orientation_error = torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)
        self._ee_linear_velocity_base = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.device
        )
        self._ee_angular_velocity_base = torch.zeros_like(self._ee_linear_velocity_base)
        self._ee_linear_speed = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self._ee_angular_speed = torch.zeros_like(self._ee_linear_speed)
        self._within_final_pose = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._final_pose_success_counter = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._final_pose_success_previous = torch.zeros_like(self._within_final_pose)
        self._new_final_pose_success = torch.zeros_like(self._within_final_pose)
        self._final_pose_time_to_success_steps = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._strict_position_condition = torch.zeros_like(self._within_final_pose)
        self._strict_orientation_condition = torch.zeros_like(self._within_final_pose)
        self._strict_linear_speed_condition = torch.zeros_like(self._within_final_pose)
        self._strict_angular_speed_condition = torch.zeros_like(self._within_final_pose)
        self._strict_all_except_hold_condition = torch.zeros_like(self._within_final_pose)
        self._final_pose_first_strict_settled_step = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._final_pose_last_strict_settled_step = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._final_pose_steps_remaining_after_max_hold = torch.zeros(
            (self.num_envs,), dtype=torch.long, device=self.device
        )
        self._final_pose_max_hold_steps_before_break = torch.zeros_like(
            self._final_pose_success_counter
        )
        self._final_pose_hold_break_reason_position = torch.zeros(
            (), dtype=torch.float32, device=self.device
        )
        self._final_pose_hold_break_reason_orientation = torch.zeros_like(
            self._final_pose_hold_break_reason_position
        )
        self._final_pose_hold_break_reason_linear_speed = torch.zeros_like(
            self._final_pose_hold_break_reason_position
        )
        self._final_pose_hold_break_reason_angular_speed = torch.zeros_like(
            self._final_pose_hold_break_reason_position
        )
        self._final_pose_hold_break_reason_target_changed = torch.zeros_like(
            self._final_pose_hold_break_reason_position
        )
        self._final_pose_hold_break_reason_timeout = torch.zeros_like(
            self._final_pose_hold_break_reason_position
        )
        self._final_pose_hold_break_reason_success_reached = torch.zeros_like(
            self._final_pose_hold_break_reason_position
        )
        self._final_pose_hold_break_reason_unknown = torch.zeros_like(
            self._final_pose_hold_break_reason_position
        )
        self._final_pose_max_hold_position_error_at_break = torch.zeros_like(
            self._final_pose_hold_break_reason_position
        )
        self._final_pose_max_hold_orientation_error_at_break = torch.zeros_like(
            self._final_pose_hold_break_reason_position
        )
        self._final_pose_max_hold_linear_speed_at_break = torch.zeros_like(
            self._final_pose_hold_break_reason_position
        )
        self._final_pose_max_hold_angular_speed_at_break = torch.zeros_like(
            self._final_pose_hold_break_reason_position
        )
        self._final_pose_max_hold_steps_before_break_total = torch.zeros_like(
            self._final_pose_hold_break_reason_position
        )
        self._final_pose_steps_remaining_after_max_hold_total = torch.zeros_like(
            self._final_pose_hold_break_reason_position
        )
        self._final_pose_first_step_entered_strict_settled = torch.full(
            (), -1.0, dtype=torch.float32, device=self.device
        )
        self._final_pose_last_step_entered_strict_settled = torch.full(
            (), -1.0, dtype=torch.float32, device=self.device
        )
        self._final_pose_entered_tolerance_count_total = torch.zeros((), dtype=torch.float32, device=self.device)
        self._final_pose_settling_hold_max = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self._final_pose_settling_hold_max_total = torch.zeros((), dtype=torch.float32, device=self.device)
        self._fd_tau_hat = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self._fd_tau_hat_normalized = torch.zeros_like(self._fd_tau_hat)
        self._fd_soft_limit_fraction = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self._fd_hard_limit_fraction = torch.zeros_like(self._fd_soft_limit_fraction)
        self._fd_normalized_abs_max = torch.zeros_like(self._fd_soft_limit_fraction)
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
        assert not self._z_arm_hat.is_inference(), "_z_arm_hat must be a normal mutable buffer."
        self._aam_latent_state_version = torch.zeros(
            (self.num_envs,), dtype=torch.long, device=self.device
        )
        self._observation_aam_state_version = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._id_aam_state_version = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._prev_q_error_mean = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self._reward_debug: dict[str, torch.Tensor | str] = {}

        # ── Curriculum state ──────────────────────────────────────────────────────
        # Explicit stage index (1-based, starts at 1). NOT derived from
        # common_step_counter; advanced only when performance gates are met.
        self._curriculum_stage_index = 1
        # Per-env effective max episode length steps (updated on curriculum change).
        self._stage_max_episode_length = int(
            self.cfg.curriculum_episode_length_s[0] / self.cfg.sim.dt
        )
        # Action-authority multiplier (stage-dependent, applied in _prepare_high_level_action).
        self._action_authority_multiplier = self.cfg.curriculum_action_authority[0]
        # Stage-level aggregate counters (scalar, accumulate across all envs).
        # Reset on stage advancement.
        self._curriculum_stage_episode_count = 0.0
        self._curriculum_stage_timeout_count = 0.0
        self._curriculum_stage_velocity_count = 0.0
        self._curriculum_stage_other_unsafe_count = 0.0
        self._curriculum_stage_position_error_sum = 0.0
        self._curriculum_stage_advance_count = 0
        self._curriculum_stage_success_count = 0.0
        self._curriculum_stage_near_pose_low_speed_count = 0.0
        self._curriculum_stage_max_hold_steps_sum = 0.0
        self._curriculum_stage_ready_windows = 0

        # ── Termination diagnostics (per-reason, per-env) ─────────────────────────
        # Per-step booleans (copied from latest _get_dones call).
        self._termination_joint_nonfinite = torch.zeros(
            (self.num_envs,), dtype=torch.bool, device=self.device
        )
        self._termination_lower_limit = torch.zeros_like(self._termination_joint_nonfinite)
        self._termination_upper_limit = torch.zeros_like(self._termination_joint_nonfinite)
        self._termination_cylinder_lower_limit = torch.zeros_like(self._termination_joint_nonfinite)
        self._termination_cylinder_upper_limit = torch.zeros_like(self._termination_joint_nonfinite)
        self._termination_velocity = torch.zeros_like(self._termination_joint_nonfinite)
        self._termination_control_nonfinite = torch.zeros_like(self._termination_joint_nonfinite)
        self._termination_success = torch.zeros_like(self._termination_joint_nonfinite)
        # Accumulated counters (per-env, across episodes).
        self._termination_count = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self._termination_joint_nonfinite_count = torch.zeros_like(self._termination_count)
        self._termination_lower_limit_count = torch.zeros_like(self._termination_count)
        self._termination_upper_limit_count = torch.zeros_like(self._termination_count)
        self._termination_velocity_count = torch.zeros_like(self._termination_count)
        self._termination_control_nonfinite_count = torch.zeros_like(self._termination_count)
        self._termination_timeout_count = torch.zeros_like(self._termination_count)
        self._termination_success_count = torch.zeros_like(self._termination_count)
        # Offending joint/channel (first violation, accumulated across episode).
        self._termination_offending_joint = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._termination_offending_channel = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )

        # ── Reward component episode sums (for extras["log"]) ─────────────────────
        self._episode_pos_tracking_sum = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self._episode_orient_tracking_sum = torch.zeros_like(self._episode_pos_tracking_sum)
        self._episode_progress_reward_sum = torch.zeros_like(self._episode_pos_tracking_sum)
        self._episode_action_rate_penalty_sum = torch.zeros_like(self._episode_pos_tracking_sum)
        self._episode_action_mag_penalty_sum = torch.zeros_like(self._episode_pos_tracking_sum)
        self._episode_valve_sat_penalty_sum = torch.zeros_like(self._episode_pos_tracking_sum)
        self._episode_clamp_penalty_sum = torch.zeros_like(self._episode_pos_tracking_sum)
        self._episode_fd_load_penalty_sum = torch.zeros_like(self._episode_pos_tracking_sum)
        self._episode_unsafe_penalty_sum = torch.zeros_like(self._episode_pos_tracking_sum)
        self._episode_linear_velocity_penalty_sum = torch.zeros_like(self._episode_pos_tracking_sum)
        self._episode_angular_velocity_penalty_sum = torch.zeros_like(self._episode_pos_tracking_sum)
        self._episode_settling_reward_sum = torch.zeros_like(self._episode_pos_tracking_sum)
        self._episode_near_target_linear_settling_sum = torch.zeros_like(self._episode_pos_tracking_sum)
        self._episode_near_target_angular_settling_sum = torch.zeros_like(self._episode_pos_tracking_sum)
        self._episode_hold_progress_reward_sum = torch.zeros_like(self._episode_pos_tracking_sum)
        self._episode_clamp_activation_sum = torch.zeros_like(self._episode_pos_tracking_sum)
        self._episode_valve_saturation_fraction_sum = torch.zeros_like(self._episode_pos_tracking_sum)
        self._episode_position_error_sum = torch.zeros_like(self._episode_pos_tracking_sum)
        self._episode_success_bonus_sum = torch.zeros_like(self._episode_pos_tracking_sum)

        # ── Persistent cumulative unsafe-penalty counters (monotonic, survive resets) ──
        self._unsafe_penalty_event_count_total = torch.zeros(
            (), dtype=torch.long, device=self.device
        )
        self._unsafe_penalty_value_total = torch.zeros(
            (), dtype=torch.float32, device=self.device
        )
        # ── Per-environment terminal unsafe-penalty snapshot buffers (consumed on reset) ──
        self._terminal_unsafe_penalty = torch.zeros(
            (self.num_envs,), dtype=torch.float32, device=self.device
        )
        self._terminal_unsafe_mask = torch.zeros(
            (self.num_envs,), dtype=torch.bool, device=self.device
        )

        # ── Progress reward buffer (per-env, stores previous position-error norm) ─
        self._prev_pos_error_norm = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self._prev_task_space_potential = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self._task_space_potential_valid = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        rolling_capacity = 1000
        self._rolling_1000_cursor = torch.zeros((), dtype=torch.long, device=self.device)
        self._rolling_1000_count = torch.zeros((), dtype=torch.long, device=self.device)
        self._rolling_1000_success = torch.zeros((rolling_capacity,), dtype=torch.float32, device=self.device)
        self._rolling_1000_timeout = torch.zeros_like(self._rolling_1000_success)
        self._rolling_1000_final_position_error = torch.zeros_like(self._rolling_1000_success)
        self._rolling_1000_final_orientation_error = torch.zeros_like(self._rolling_1000_success)
        self._rolling_1000_final_linear_speed = torch.zeros_like(self._rolling_1000_success)
        self._rolling_1000_final_angular_speed = torch.zeros_like(self._rolling_1000_success)
        self._rolling_1000_max_hold_steps = torch.zeros_like(self._rolling_1000_success)
        self._rolling_1000_clamp_activation_fraction = torch.zeros_like(self._rolling_1000_success)
        self._rolling_1000_valve_saturation_fraction = torch.zeros_like(self._rolling_1000_success)

        self._print_and_write_reward_contract_once()

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot
        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        self._reference_marker = None
        if bool(getattr(self.cfg, "reference_marker_debug_vis", False)):
            marker_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/KevinHydraulicROM/joint_reference")
            marker_cfg.markers["frame"].scale = (0.25, 0.25, 0.25)
            self._reference_marker = VisualizationMarkers(marker_cfg)
            self._reference_marker.set_visibility(True)

    def _find_joint_ids(self, joint_names: Sequence[str]) -> list[int]:
        ids: list[int] = []
        for joint_name in joint_names:
            joint_ids, _ = self.robot.find_joints(joint_name)
            if len(joint_ids) == 0:
                raise RuntimeError(f"Could not find joint: {joint_name}")
            ids.append(joint_ids[0])
        return ids

    def _find_optional_joint_ids(self, joint_names: Sequence[str]) -> list[int]:
        ids: list[int] = []
        missing: list[str] = []
        for joint_name in joint_names:
            joint_ids, _ = self.robot.find_joints(joint_name)
            if len(joint_ids) == 0:
                missing.append(joint_name)
            else:
                ids.append(joint_ids[0])
        if missing:
            print(
                "[WARN] Kevin RRLAB vehicle command disabled for missing optional wheel joints: "
                + ", ".join(missing),
                flush=True,
            )
            return []
        return ids

    def _find_optional_body_id(self, body_name: str) -> int | None:
        try:
            body_ids, _ = self.robot.find_bodies(body_name)
        except Exception:
            return None
        if len(body_ids) == 0:
            print(f"[WARN] Could not resolve optional end-effector body: {body_name}", flush=True)
            return None
        return int(body_ids[0])

    def _assert_velocity_state_contract(self) -> None:
        mechanical_names = tuple(self.cfg.mechanical_joint_names)
        cylinder_names = tuple(self.cfg.cylinder_joint_names)
        mechanical_pos_units = tuple(self.cfg.mechanical_joint_position_units)
        mechanical_vel_units = tuple(self.cfg.mechanical_joint_velocity_units)
        mechanical_types = tuple(self.cfg.mechanical_joint_types)
        cylinder_pos_units = tuple(self.cfg.cylinder_joint_position_units)
        cylinder_vel_units = tuple(self.cfg.cylinder_joint_velocity_units)
        if len(mechanical_names) != 5 or len(self._mechanical_joint_ids) != 5:
            raise RuntimeError("Mechanical learned-model state must resolve exactly five joints.")
        if len(cylinder_names) != 4 or len(self._cylinder_joint_ids) != 4:
            raise RuntimeError("Cylinder ROM/writer state must resolve exactly four joints.")
        if mechanical_pos_units != ("rad", "rad", "rad", "rad", "rad"):
            raise RuntimeError(f"Mechanical q units must be radians, got {mechanical_pos_units}.")
        if mechanical_vel_units != ("rad/s", "rad/s", "rad/s", "rad/s", "rad/s"):
            raise RuntimeError(f"Mechanical qdot units must be rad/s, got {mechanical_vel_units}.")
        if cylinder_pos_units != ("m", "m", "m", "m"):
            raise RuntimeError(f"Cylinder extension units must be metres, got {cylinder_pos_units}.")
        if cylinder_vel_units != ("m/s", "m/s", "m/s", "m/s"):
            raise RuntimeError(f"Cylinder extension velocity units must be m/s, got {cylinder_vel_units}.")
        if len(mechanical_types) != 5:
            raise RuntimeError(f"mechanical_joint_types must contain five entries, got {mechanical_types}.")
        self._mechanical_cylinder_joint_id_overlap = sorted(
            set(int(joint_id) for joint_id in self._mechanical_joint_ids)
            & set(int(joint_id) for joint_id in self._cylinder_joint_ids)
        )
        self._velocity_state_contract = {
            "mechanical_joint_names": mechanical_names,
            "mechanical_joint_ids": tuple(int(joint_id) for joint_id in self._mechanical_joint_ids),
            "mechanical_joint_types": mechanical_types,
            "mechanical_position_units": mechanical_pos_units,
            "mechanical_velocity_units": mechanical_vel_units,
            "cylinder_joint_names": cylinder_names,
            "cylinder_joint_ids": tuple(int(joint_id) for joint_id in self._cylinder_joint_ids),
            "cylinder_position_units": cylinder_pos_units,
            "cylinder_velocity_units": cylinder_vel_units,
            "overlap_joint_ids": tuple(self._mechanical_cylinder_joint_id_overlap),
        }

    def _assert_single_actuator_writer(self) -> None:
        use_rom_position_targets = bool(self.cfg.use_final_causal_cylinder_rom or self.cfg.use_reduced_hydraulic_rom)
        if use_rom_position_targets and bool(getattr(self.cfg, "use_fd_effort_actuation", False)):
            raise RuntimeError(
                "ROM position-target actuation and FD effort actuation "
                "cannot command the same hydraulic joints simultaneously."
            )
        if use_rom_position_targets and bool(getattr(self.cfg, "use_fd_effort", False)):
            raise RuntimeError(
                "ROM position-target actuation and FD effort actuation "
                "cannot command the same hydraulic joints simultaneously."
            )

    def _robot_joint_names(self) -> list[str]:
        names = getattr(self.robot, "joint_names", None)
        if names is None:
            names = getattr(self.robot.data, "joint_names", None)
        if names is None:
            raise RuntimeError("Could not read IsaacLab robot joint_names for Kevin mechanical family preflight.")
        return list(names)

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

    def _task_control_mode(self) -> str:
        if self.cfg.control_mode in {"scripted_reference_demo", "rl_policy", "joint_space_validation", "rl_task_space_tracking"}:
            return str(self.cfg.control_mode)
        return str(getattr(self.cfg, "task_control_mode", "rl_policy"))

    def _low_level_control_mode(self) -> str:
        if self.cfg.control_mode in {"position_delta", "effort"}:
            return str(self.cfg.control_mode)
        return str(getattr(self.cfg, "low_level_control_mode", "position_delta"))

    def _validate_control_mode(self) -> None:
        task_mode = self._task_control_mode()
        low_level_mode = self._low_level_control_mode()
        if task_mode not in {"scripted_reference_demo", "rl_policy", "joint_space_validation", "rl_task_space_tracking"}:
            raise ValueError(
                "control_mode must be one of 'joint_space_validation', 'rl_task_space_tracking', "
                "'scripted_reference_demo', or 'rl_policy' "
                f"for the hydraulic ROM task, got {self.cfg.control_mode!r}."
            )
        if low_level_mode not in {"position_delta", "effort"}:
            raise ValueError(f"low_level_control_mode must be 'position_delta' or 'effort', got {low_level_mode!r}.")

        # ── Low-level contract assertions for task-space tracking mode ──────────
        if task_mode == "rl_task_space_tracking":
            # PPO Action[4] = normalized desired qddot for 4 hydraulic cylinders.
            if len(self._command_joint_ids) != 4:
                raise RuntimeError(
                    f"Task-space tracking requires exactly 4 command joints, got {len(self._command_joint_ids)}."
                )
            # Observation dimension must match the task-space contract.
            obs_dim = int(getattr(self.cfg, "observation_space", 0))
            if obs_dim != TASK_SPACE_OBSERVATION_DIM:
                raise RuntimeError(
                    f"Task-space observation dimension must be {TASK_SPACE_OBSERVATION_DIM}, "
                    f"got {obs_dim}."
                )
            # AAM + ID pipeline: qddot → z_arm → ValveCmd.
            # SD observer and SD feedback must be disabled in the canonical task.
            if bool(self.cfg.use_sd_observer):
                raise RuntimeError(
                    "SD observer must be disabled in canonical task-space tracking mode "
                    "(use_sd_observer=False)."
                )
            if bool(self.cfg.use_sd_feedback):
                raise RuntimeError(
                    "SD feedback must be disabled in canonical task-space tracking mode "
                    "(use_sd_feedback=False)."
                )
            # ID must be the qddot-to-ValveCmd converter.
            if str(self.cfg.position_delta_source) != "id_valve":
                raise RuntimeError(
                    "Task-space tracking requires position_delta_source='id_valve' "
                    f"for ID-based qddot conversion, got {self.cfg.position_delta_source!r}."
                )
            # FD must be observer-only (never writes effort to actuators).
            if bool(getattr(self.cfg, "use_fd_effort", False)):
                raise RuntimeError(
                    "FD effort actuation must be disabled in task-space tracking mode "
                    "(use_fd_effort=False, FD is observer-only)."
                )

    def _make_final_causal_rom(self) -> tuple[ValveConditionedROMInference, MultirateROMWrapper]:
        export_dir = Path(self.cfg.final_causal_rom_export_dir)
        self._validate_final_causal_rom_export(export_dir)
        rom = ValveConditionedROMInference(export_dir, device=self.device)
        step_dt = float(getattr(self, "step_dt", float(self.cfg.sim.dt) * float(self.cfg.decimation)))
        config = MultirateROMConfig(
            control_dt=step_dt,
            rom_dt=float(self.cfg.final_causal_rom_update_period),
            input_dim=len(CORE_INPUT_FEATURE_NAMES),
            output_dim=4,
            history_length=HISTORY_LENGTH,
        )
        expected_ratio = int(self.cfg.final_causal_rom_multirate_ratio)
        if config.ratio != expected_ratio:
            raise RuntimeError(
                "Final causal ROM multirate mismatch: "
                f"configured ratio={expected_ratio}, actual ratio={config.ratio}."
            )
        wrapper = MultirateROMWrapper(
            rom.predict,
            num_envs=self.num_envs,
            config=config,
            device=self.device,
            dtype=torch.float32,
        )
        return rom, wrapper

    def _training_mean_dp_from_final_rom(self) -> torch.Tensor:
        if self._final_causal_rom is None:
            return torch.zeros((4,), dtype=torch.float32, device=self.device)
        dP_pa = self._final_causal_rom.model.training_mean_dp.detach().to(device=self.device, dtype=torch.float32)
        if not torch.isfinite(dP_pa).all():
            raise RuntimeError("Final ROM training-mean dP contains NaN/Inf.")
        return dP_pa

    def _validate_final_causal_rom_export(self, export_dir: Path) -> None:
        required = (
            "best_valve_conditioned_causal_rom.pth",
            "best_valve_conditioned_causal_rom_scaler.pth",
            "best_valve_conditioned_causal_rom_metadata.json",
        )
        missing = [name for name in required if not (export_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Final causal ROM export is missing files in {export_dir}: {missing}")
        metadata = json.loads((export_dir / "best_valve_conditioned_causal_rom_metadata.json").read_text(encoding="utf-8"))
        if metadata.get("model_version") != "valve_conditioned_causal_rom_v1":
            raise RuntimeError("Final ROM export must be the valve-conditioned causal ROM package.")
        if not bool(metadata.get("causality_flag", False)):
            raise RuntimeError("Final ROM export metadata is not marked causal.")
        if tuple(metadata.get("feature_order", ())) != CORE_INPUT_FEATURE_NAMES:
            raise RuntimeError("Final ROM export feature order does not match the runtime 26-feature contract.")
        if len(tuple(metadata.get("target_order", ()))) != 4:
            raise RuntimeError("Final ROM export must have exactly four cylinder outputs.")
        if abs(float(metadata.get("dt", 0.0)) - float(self.cfg.final_causal_rom_update_period)) > 1.0e-6:
            raise RuntimeError(
                "Final ROM export dt does not match final_causal_rom_update_period: "
                f"{metadata.get('dt')} vs {self.cfg.final_causal_rom_update_period}."
            )

    def _warn_if_provisional_mapping(self) -> None:
        mapping = self._cylinder_coordinate_mapping
        if mapping is None:
            return
        if mapping.mapping_validated and mapping.nominal_reference_confidence.upper() != "WARN":
            return
        print(
            "[WARN] Kevin hydraulic ROM task is using the configured nominal-sign cylinder mapping "
            f"with mapping_validated={mapping.mapping_validated} and "
            f"nominal_reference_confidence={mapping.nominal_reference_confidence}. "
            "This is serialized in the task config; no diagnostic CLI flag is required.",
            flush=True,
        )

    def _print_runtime_contract_once(self) -> None:
        if MulagAdaptiveRLEnv._runtime_contract_printed:
            return
        MulagAdaptiveRLEnv._runtime_contract_printed = True
        contract = getattr(self._controller, "runtime_contract", {})
        print("[INFO] Kevin active control contract:", flush=True)
        rows = [
            ("task_action", "4D normalized desired qddot for hydraulic families"),
            ("action_scale", str(self.cfg.action_to_qddot_scale)),
            ("qddot_clip", str(self.cfg.qddot_cmd_clip)),
            ("AAM", f"{contract.get('aam_input')} -> {contract.get('aam_output')} window={contract.get('aam_window')}"),
            ("ID", f"{contract.get('id_input')} -> {contract.get('id_output')} window={contract.get('id_window')}"),
            ("ID_qddot_std", str(contract.get("id_qddot_std"))),
            ("FD", f"{contract.get('fd_input')} -> {contract.get('fd_output')} window={contract.get('fd_window')}"),
            ("FD_output_units", str(contract.get("fd_output_units"))),
            ("FD_effort_writer", str(contract.get("fd_effort_actuation"))),
            (
                "SD",
                "ROM pressure observer enabled"
                if str(self.cfg.final_causal_rom_pressure_source) == "sd_observer"
                else ("disabled in active hydraulic ROM task" if not self.cfg.use_sd_observer else "passive observer enabled"),
            ),
            ("ROM", f"{len(CORE_INPUT_FEATURE_NAMES)} features, dt={self.cfg.final_causal_rom_update_period}s, ratio={self.cfg.final_causal_rom_multirate_ratio}"),
            (
                "ROM_command_coupling",
                "deployed valve-conditioned causal ROM explicitly uses ValveCmd, ValveCmd*abs(ValveCmd), and ValveCmd*dP",
            ),
            (
                "vehicle_command",
                f"{self.cfg.vehicle_command_mode}: {self.cfg.vehicle_speed_range_kmh} km/h "
                f"on {tuple(self.cfg.vehicle_wheel_joint_names)}",
            ),
            ("vehicle_joint_ids", str(self._vehicle_wheel_joint_ids)),
            ("dP_source", str(self.cfg.final_causal_rom_pressure_source)),
            ("coordinate_mapping", str(self.cfg.cylinder_coordinate_mapping_mode)),
            ("writer", "set_joint_position_target on four prismatic cylinder joints"),
            ("mechanical_joint_names", str(tuple(self.cfg.mechanical_joint_names))),
            ("mechanical_joint_ids", str(self._mechanical_joint_ids)),
            ("mechanical_units", "q=rad, qdot=rad/s"),
            ("cylinder_joint_names", str(tuple(self.cfg.cylinder_joint_names))),
            ("cylinder_joint_ids", str(self._cylinder_joint_ids)),
            ("cylinder_units", "extension=m, extension_velocity=m/s"),
            ("mechanical_cylinder_overlap_ids", str(self._mechanical_cylinder_joint_id_overlap)),
        ]
        for key, value in rows:
            print(f"[INFO]   {key:18s} {value}", flush=True)
        print("[INFO] AAM runtime contract:", flush=True)
        aam_rows = [
            ("active", contract.get("aam_active")),
            ("checkpoint", contract.get("aam_checkpoint")),
            ("frozen", contract.get("aam_frozen")),
            ("eval mode", contract.get("aam_eval_mode")),
            ("input feature width", contract.get("aam_input_width")),
            ("history length", contract.get("aam_window")),
            ("output width", contract.get("aam_output_width")),
            (
                "PPO observation connection",
                contract.get("aam_ppo_observation_connection"),
            ),
            (
                "ID conditioning connection",
                contract.get("aam_id_conditioning_connection"),
            ),
            ("history ValveCmd source", contract.get("aam_history_valve_source")),
        ]
        for key, value in aam_rows:
            print(f"[INFO]   {key}: {value}", flush=True)
        print(
            "[INFO]   input history shape: "
            f"({self.num_envs}, {contract.get('aam_window')}, 18)",
            flush=True,
        )
        print(
            f"[INFO]   output shape: ({self.num_envs}, 24)",
            flush=True,
        )

    def run_task_space_bridge_preflight(self) -> dict[str, object]:
        """Report PPO task-space readiness; PPO is the ee_pose-to-action bridge."""
        validate_task_space_observation_contract()
        ee_resolved = self._ee_body_id is not None
        wheels_resolved = (not bool(getattr(self.cfg, "enable_scripted_vehicle_motion", True))) or (
            len(self._vehicle_wheel_joint_ids) == len(self.cfg.vehicle_wheel_joint_names)
        )
        fd_conflict = bool(
            (self.cfg.use_final_causal_cylinder_rom or self.cfg.use_reduced_hydraulic_rom)
            and (getattr(self.cfg, "use_fd_effort_actuation", False) or getattr(self.cfg, "use_fd_effort", False))
        )
        obs_ok = int(getattr(self.cfg, "task_space_observation_space", TASK_SPACE_OBSERVATION_DIM)) == TASK_SPACE_OBSERVATION_DIM
        reward_ok = float(getattr(self.cfg, "rew_fd_load", 0.0)) >= 0.0
        fd_input_contract_ok = "fd_input" in getattr(self._controller, "runtime_contract", {})
        fd_output_finite = bool(
            (torch.isfinite(self._fd_tau_hat).all() & torch.isfinite(self._fd_tau_hat_normalized).all()).item()
        )
        final_pose_target = self._final_pose_target()
        normalize_quaternion_wxyz(
            final_pose_target[:, 3:7],
            name="final_pose_target quaternion",
            validate=True,
        )
        normalize_quaternion_wxyz(
            self._current_ee_pose[:, 3:7],
            name="measured end-effector quaternion",
            validate=True,
        )
        command_ok = bool(
            (
                torch.isfinite(final_pose_target).all()
                & torch.isfinite(self._ee_position_error).all()
                & torch.isfinite(self._ee_orientation_error).all()
            ).item()
        )
        if fd_conflict:
            decision = BLOCKED_FD_ACTUATOR_CONFLICT
        elif not fd_input_contract_ok:
            decision = BLOCKED_FD_INPUT_CONTRACT
        elif not fd_output_finite:
            decision = BLOCKED_FD_OUTPUT_NONFINITE
        elif not ee_resolved:
            decision = BLOCKED_EE_BODY_RESOLUTION
        elif not command_ok:
            decision = BLOCKED_TASKSPACE_COMMAND
        elif not obs_ok:
            decision = BLOCKED_TASKSPACE_OBSERVATION
        elif not reward_ok:
            decision = BLOCKED_TASKSPACE_REWARD
        elif not wheels_resolved:
            decision = BLOCKED_VEHICLE_COMMAND
        else:
            decision = READY_FOR_PPO
        passed = decision == READY_FOR_PPO
        return {
            "name": "task_space_bridge_preflight",
            "passed": passed,
            "preflight_stage": "Stage 3: PPO task-space observation/reward bridge",
            "preflight_decision": decision,
            "ee_body": TASK_SPACE_EE_BODY,
            "ee_body_id": self._ee_body_id,
            "reference_frame": TASK_SPACE_FRAME,
            "pose_order": TASK_SPACE_POSE_ORDER,
            "quaternion_order": TASK_SPACE_QUATERNION_ORDER,
            "authoritative_target_field": "_ee_pose_command",
            "mechanical_state_joint_order": tuple(self.cfg.state_joint_names),
            "cylinder_command_joint_order": tuple(self.cfg.command_joint_names),
            "task_space_observation_dim": int(getattr(self.cfg, "task_space_observation_space", -1)),
            "task_space_command_finite": command_ok,
            "fd_observer_enabled": bool(getattr(self.cfg, "use_fd_as_observer", True)),
            "fd_input_contract_ready": bool(fd_input_contract_ok),
            "fd_output_finite": fd_output_finite,
            "fd_effort_actuation_enabled": bool(getattr(self.cfg, "use_fd_effort_actuation", False) or getattr(self.cfg, "use_fd_effort", False)),
            "vehicle_stage_ready": wheels_resolved,
            "vehicle_wheel_joint_ids": tuple(self._vehicle_wheel_joint_ids),
            "blocker": "" if passed else decision,
        }

    def _read_mechanical_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        mechanical_q = self.robot.data.joint_pos[:, self._mechanical_joint_ids]
        mechanical_qdot = self.robot.data.joint_vel[:, self._mechanical_joint_ids]
        return mechanical_q, mechanical_qdot

    def _read_joint_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Backward-compatible alias for the 5D learned-model mechanical state."""
        return self._read_mechanical_state()

    def _read_cylinder_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        cylinder_extension = self.robot.data.joint_pos[:, self._cylinder_joint_ids]
        cylinder_extension_velocity = self.robot.data.joint_vel[:, self._cylinder_joint_ids]
        return cylinder_extension, cylinder_extension_velocity

    def _validated_hydraulic_velocity_limits_enabled(self) -> bool:
        if not bool(getattr(self.cfg, "enable_validated_hydraulic_velocity_termination", False)):
            return False
        hard = torch.as_tensor(
            self.cfg.hydraulic_hard_velocity_limits_mps,
            dtype=torch.float32,
            device=self.device,
        )
        if hard.shape != (4,):
            raise RuntimeError(f"hydraulic_hard_velocity_limits_mps must have shape (4,), got {tuple(hard.shape)}")
        return bool(torch.all(torch.isfinite(hard)) and torch.all(hard > 0.0))

    def _hydraulic_velocity_limit_tensor(self, field: str) -> torch.Tensor:
        limits = torch.as_tensor(getattr(self.cfg, field), dtype=torch.float32, device=self.device)
        if limits.shape != (4,):
            raise RuntimeError(f"{field} must have shape (4,), got {tuple(limits.shape)}")
        return limits.reshape(1, 4)

    def _mechanical_velocity_limit_tensor(self, field: str) -> torch.Tensor:
        limits = torch.as_tensor(getattr(self.cfg, field), dtype=torch.float32, device=self.device)
        if limits.shape != (5,):
            raise RuntimeError(f"{field} must have shape (5,), got {tuple(limits.shape)}")
        return limits.reshape(1, 5)

    def _validated_mechanical_velocity_limits_enabled(self) -> bool:
        if not bool(getattr(self.cfg, "enable_validated_mechanical_velocity_termination", False)):
            return False
        hard = self._mechanical_velocity_limit_tensor("mechanical_hard_velocity_limits")
        return bool(torch.all(torch.isfinite(hard)) and torch.all(hard > 0.0))

    def _update_hydraulic_velocity_audit(self, q_hyd: torch.Tensor, qdot_physx: torch.Tensor) -> torch.Tensor:
        if q_hyd.shape != (self.num_envs, 4) or qdot_physx.shape != (self.num_envs, 4):
            raise RuntimeError(
                "Hydraulic velocity audit expects q_hyd and qdot_physx shapes "
                f"({self.num_envs}, 4), got {tuple(q_hyd.shape)} and {tuple(qdot_physx.shape)}"
            )
        control_dt = float(self.cfg.sim.dt)
        if control_dt <= 0.0:
            raise RuntimeError(f"Control dt must be positive, got {control_dt}")

        self._hydraulic_qdot_fd = (q_hyd - self._prev_hydraulic_q) / control_dt
        self._hydraulic_target_velocity = (
            self._rom_cylinder_pos_target - self._rom_cylinder_pos_target_prev
        ) / control_dt
        self._hydraulic_rom_applied_rate = self._final_rom_delta_isaac_10ms / control_dt

        finite = (
            torch.isfinite(q_hyd)
            & torch.isfinite(qdot_physx)
            & torch.isfinite(self._hydraulic_qdot_fd)
            & torch.isfinite(self._hydraulic_target_velocity)
            & torch.isfinite(self._final_rom_extension_velocity)
            & torch.isfinite(self._hydraulic_rom_applied_rate)
        )
        finite_all = finite.all(dim=-1)

        if self._validated_hydraulic_velocity_limits_enabled():
            hard_limits = self._hydraulic_velocity_limit_tensor("hydraulic_hard_velocity_limits_mps")
            raw_exceedance = torch.abs(qdot_physx) > hard_limits
            corroborated = (
                (torch.abs(self._hydraulic_qdot_fd) > hard_limits)
                | (torch.abs(self._hydraulic_target_velocity) > hard_limits)
                | (torch.abs(self._final_rom_extension_velocity) > hard_limits)
                | (torch.abs(self._hydraulic_rom_applied_rate) > hard_limits)
            )
        else:
            diagnostic_threshold = abs(float(getattr(self.cfg, "unvalidated_velocity_diagnostic_threshold", 6.0)))
            hard_limits = torch.full((1, 4), diagnostic_threshold, dtype=torch.float32, device=self.device)
            raw_exceedance = torch.abs(qdot_physx) > hard_limits
            corroborated = torch.zeros_like(raw_exceedance)

        self._hydraulic_velocity_raw_exceedance = raw_exceedance.detach().clone()
        persistence_next = torch.where(
            raw_exceedance,
            self._hydraulic_velocity_persistence + 1,
            torch.zeros_like(self._hydraulic_velocity_persistence),
        )
        self._hydraulic_velocity_persistence = persistence_next
        required_steps = max(int(getattr(self.cfg, "hydraulic_velocity_persistence_steps", 3)), 1)
        confirmed = (
            raw_exceedance
            & corroborated
            & finite
            & (self._hydraulic_velocity_persistence >= required_steps)
            & bool(self._validated_hydraulic_velocity_limits_enabled())
        )
        self._hydraulic_velocity_confirmed_hard = confirmed.detach().clone()
        self._hydraulic_velocity_spike_ignored = (
            raw_exceedance & ~confirmed & finite
        ).detach().clone()

        fd_ok = torch.abs(self._hydraulic_qdot_fd) <= hard_limits
        target_ok = torch.abs(self._hydraulic_target_velocity) <= hard_limits
        rom_ok = torch.abs(self._final_rom_extension_velocity) <= hard_limits
        applied_ok = torch.abs(self._hydraulic_rom_applied_rate) <= hard_limits
        physx_spike = raw_exceedance & fd_ok & target_ok & rom_ok & applied_ok & finite
        target_jump = raw_exceedance & ~target_ok & ~applied_ok & finite
        genuine_high_motion = raw_exceedance & ~fd_ok & (~target_ok | ~rom_ok | ~applied_ok) & finite
        nonfinite_event = raw_exceedance & ~finite

        self._velocity_audit_physx_spike_count += physx_spike.float().sum()
        self._velocity_audit_target_jump_count += target_jump.float().sum()
        self._velocity_audit_genuine_high_motion_count += genuine_high_motion.float().sum()
        self._velocity_audit_nonfinite_count += nonfinite_event.float().sum()
        self._velocity_audit_unit_mapping_mismatch_count += torch.zeros((), dtype=torch.float32, device=self.device)

        self._prev_hydraulic_q = q_hyd.detach().clone()
        self._rom_cylinder_pos_target_prev = self._rom_cylinder_pos_target.detach().clone()
        return ~finite_all

    def export_velocity_audit_events(self) -> list[dict[str, object]]:
        """Return bounded CPU-readable velocity audit summary for external diagnostics."""
        event_mask = self._hydraulic_velocity_raw_exceedance.detach()
        env_ids, channels = torch.where(event_mask)
        max_events = min(int(env_ids.numel()), 32)
        records: list[dict[str, object]] = []
        for i in range(max_events):
            env_id = int(env_ids[i].detach().cpu())
            channel = int(channels[i].detach().cpu())
            records.append(
                {
                    "env_id": env_id,
                    "simulation_step": int(getattr(self, "common_step_counter", 0)),
                    "offending_hydraulic_channel": channel,
                    "articulation_joint_name": self.cfg.command_joint_names[channel],
                    "articulation_dof_id": int(self._command_joint_ids[channel]),
                    "joint_type": "hydraulic_prismatic_cylinder",
                    "q_after_event": float(self.robot.data.joint_pos[env_id, self._command_joint_ids[channel]].detach().cpu()),
                    "q_before_event": float(self._prev_hydraulic_q[env_id, channel].detach().cpu()),
                    "qdot_physx": float(self.robot.data.joint_vel[env_id, self._command_joint_ids[channel]].detach().cpu()),
                    "qdot_fd": float(self._hydraulic_qdot_fd[env_id, channel].detach().cpu()),
                    "target_velocity": float(self._hydraulic_target_velocity[env_id, channel].detach().cpu()),
                    "rom_extension_velocity": float(self._final_rom_extension_velocity[env_id, channel].detach().cpu()),
                    "rom_applied_rate": float(self._hydraulic_rom_applied_rate[env_id, channel].detach().cpu()),
                    "valve_cmd": float(self._current_valve_cmd[env_id, channel].detach().cpu()),
                    "ppo_action": float(self._current_action[env_id, channel].detach().cpu()),
                    "qddot_command": float(self._current_qddot_cmd[env_id, channel].detach().cpu()),
                    "ee_linear_speed": float(self._ee_linear_speed[env_id].detach().cpu()),
                    "ee_angular_speed": float(self._ee_angular_speed[env_id].detach().cpu()),
                }
            )
        return records

    def _reward_contract_summary(self) -> dict[str, object]:
        """Serializable summary of the task-space reward contract."""
        return {
            "environment_contract_version": str(self.cfg.kevin_fixed_pose_contract_version),
            "reward_contract_version": str(self.cfg.reward_contract_version),
            "components": [
                {
                    "name": "position_tracking",
                    "formula": (
                        "position_tracking_reward_weight * "
                        "exp(-position_tracking_exponential_scale * final_position_error_m)"
                    ),
                    "weight": float(self.cfg.position_tracking_reward_weight),
                    "normalization": "metres, bounded exponential",
                    "minimum": 0.0,
                    "maximum": float(self.cfg.position_tracking_reward_weight),
                    "activation_condition": "every task-space reward step",
                    "source_tensor": "_ee_position_error",
                },
                {
                    "name": "orientation_tracking",
                    "formula": "rew_task_orient_tracking * exp(-(orientation_error_rad / tolerance_rad)^2)",
                    "weight": float(self.cfg.rew_task_orient_tracking),
                    "normalization": "quaternion geodesic radians / final_pose_orientation_tolerance_rad",
                    "minimum": 0.0,
                    "maximum": float(self.cfg.rew_task_orient_tracking),
                    "activation_condition": "enabled curriculum orientation stage",
                    "source_tensor": "_ee_orientation_error",
                },
                {
                    "name": "potential_progress",
                    "formula": "rew_task_progress * clamp(gamma * phi_next - phi_current)",
                    "weight": float(self.cfg.rew_task_progress),
                    "normalization": "phi = -(wp * pos_err / pos_scale + wo * orient_err / orient_scale)",
                    "minimum": -float(self.cfg.rew_task_progress_clip) * float(self.cfg.rew_task_progress),
                    "maximum": float(self.cfg.rew_task_progress_clip) * float(self.cfg.rew_task_progress),
                    "activation_condition": "after first non-reset transition",
                    "source_tensor": "_prev_task_space_potential, _ee_position_error, _ee_orientation_error",
                },
                {
                    "name": "near_pose_velocity",
                    "formula": "-weight * pose_gate * tanh((speed / scale)^2)",
                    "weight": {
                        "linear": float(self.cfg.rew_final_pose_linear_velocity),
                        "angular": float(self.cfg.rew_final_pose_angular_velocity),
                    },
                    "normalization": "end-effector speed / configured shaping scale",
                    "minimum": -(
                        float(self.cfg.rew_final_pose_linear_velocity)
                        + float(self.cfg.rew_final_pose_angular_velocity)
                    ),
                    "maximum": 0.0,
                    "activation_condition": "smooth near-pose gate",
                    "source_tensor": "_ee_linear_speed, _ee_angular_speed",
                },
                {
                    "name": "settling",
                    "formula": "rew_final_pose_settling * pose_gate * mean(exp(-(speed / scale)^2))",
                    "weight": float(self.cfg.rew_final_pose_settling),
                    "normalization": "end-effector speed / configured settling scale",
                    "minimum": 0.0,
                    "maximum": float(self.cfg.rew_final_pose_settling),
                    "activation_condition": "smooth near-pose gate",
                    "source_tensor": "_ee_linear_speed, _ee_angular_speed",
                },
                {
                    "name": "near_target_linear_settling",
                    "formula": "weight * near_pose_gate * tanh((linear_speed_tolerance - linear_speed) / speed_margin)",
                    "weight": float(self.cfg.rew_near_target_linear_settling),
                    "normalization": "linear speed margin around final_pose_linear_speed_tolerance_mps",
                    "minimum": -float(self.cfg.rew_near_target_linear_settling),
                    "maximum": float(self.cfg.rew_near_target_linear_settling),
                    "activation_condition": "smooth position/orientation near-target gate",
                    "source_tensor": "_ee_linear_speed, _ee_position_error, _ee_orientation_error",
                },
                {
                    "name": "hold_progress",
                    "formula": "rew_final_pose_hold_progress * within_final_pose * (hold_counter / required_hold_steps)^2",
                    "weight": float(self.cfg.rew_final_pose_hold_progress),
                    "normalization": "consecutive strict settled steps / final_pose_success_hold_steps",
                    "minimum": 0.0,
                    "maximum": float(self.cfg.rew_final_pose_hold_progress),
                    "activation_condition": "strict pose and speed tolerances are met",
                    "source_tensor": "_final_pose_success_counter, _within_final_pose",
                },
                {
                    "name": "valve_saturation",
                    "formula": "-rew_valve_saturation_soft * mean(tanh(relu(abs(applied_valve)-sat_start)/(1-sat_start))^2)",
                    "weight": float(self.cfg.rew_valve_saturation_soft),
                    "normalization": "applied ValveCmd excess above valve_saturation_start",
                    "minimum": -float(self.cfg.rew_valve_saturation_soft),
                    "maximum": 0.0,
                    "activation_condition": "abs(applied ValveCmd) > valve_saturation_start",
                    "source_tensor": "_current_valve_cmd",
                },
                {
                    "name": "avoidable_clamp",
                    "formula": "-rew_clamp_outward * mean(outward_violation * bounded_loss(abs(blocked_increment)/increment_scale))",
                    "weight": float(self.cfg.rew_clamp_outward),
                    "normalization": "blocked outward cylinder increment / rom_clamp_documented_increment_scale",
                    "minimum": -float(self.cfg.rew_clamp_outward),
                    "maximum": 0.0,
                    "activation_condition": "boundary active and request points outward",
                    "source_tensor": "_rom_boundary_active, _rom_outward_request, _rom_blocked_outward_increment",
                },
                {
                    "name": "fd_load",
                    "formula": "-rew_fd_load * mean(bounded_loss(relu(abs(fd_norm)-soft_threshold)/soft_threshold))",
                    "weight": float(self.cfg.rew_fd_load),
                    "normalization": "FD normalized observer output above fd_load_soft_threshold",
                    "minimum": -float(self.cfg.rew_fd_load),
                    "maximum": 0.0,
                    "activation_condition": "abs(_fd_tau_hat_normalized) > fd_load_soft_threshold",
                    "source_tensor": "_fd_tau_hat_normalized",
                },
                {
                    "name": "strict_success_bonus",
                    "formula": "final_pose_success_bonus_value on first transition reaching required consecutive hold",
                    "weight": float(self.cfg.final_pose_success_bonus_value),
                    "normalization": "one-shot terminal event",
                    "minimum": 0.0,
                    "maximum": float(self.cfg.final_pose_success_bonus_value),
                    "activation_condition": "final_pose_success_hold_steps consecutive strict settled steps",
                    "source_tensor": "_new_final_pose_success",
                },
            ],
        }

    def _print_and_write_reward_contract_once(self) -> None:
        if MulagAdaptiveRLEnv._reward_contract_printed:
            return
        contract = self._reward_contract_summary()
        print(
            "[INFO] Kevin reward contract: "
            f"env={contract['environment_contract_version']} "
            f"reward={contract['reward_contract_version']}",
            flush=True,
        )
        log_dir = getattr(self.cfg, "log_dir", None)
        if log_dir:
            path = Path(log_dir) / "reward_contract.json"
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(contract, indent=2, sort_keys=True), encoding="utf-8")
            except OSError as exc:
                print(f"[WARN] Failed to write reward contract to {path}: {exc}", flush=True)
        MulagAdaptiveRLEnv._reward_contract_printed = True

    def _record_rom_clamp_telemetry(
        self,
        boundary_active: torch.Tensor,
        outward_request: torch.Tensor,
        blocked_outward_increment: torch.Tensor,
    ) -> None:
        self._rom_boundary_active = boundary_active.detach().clone()
        self._rom_outward_request = outward_request.detach().clone()
        self._rom_blocked_outward_increment = blocked_outward_increment.detach().clone()
        self._rom_clamp_activation_fraction = torch.mean(
            (boundary_active & outward_request).float(), dim=-1
        )
        self._rom_clamp_blocked_outward_mean = torch.mean(
            (boundary_active & outward_request).float() * torch.abs(blocked_outward_increment.detach()),
            dim=-1,
        )

    def _update_rolling_1000_metrics(
        self,
        *,
        success: torch.Tensor,
        timeout: torch.Tensor,
        final_position_error: torch.Tensor,
        final_orientation_error: torch.Tensor,
        final_linear_speed: torch.Tensor,
        final_angular_speed: torch.Tensor,
        max_hold_steps: torch.Tensor,
        clamp_activation_fraction: torch.Tensor,
        valve_saturation_fraction: torch.Tensor,
    ) -> None:
        n = int(success.numel())
        if n == 0:
            return
        capacity = int(self._rolling_1000_success.numel())
        offsets = torch.arange(n, device=self.device, dtype=torch.long)
        indices = (self._rolling_1000_cursor + offsets) % capacity
        self._rolling_1000_success[indices] = success.detach().float()
        self._rolling_1000_timeout[indices] = timeout.detach().float()
        self._rolling_1000_final_position_error[indices] = final_position_error.detach().float()
        self._rolling_1000_final_orientation_error[indices] = final_orientation_error.detach().float()
        self._rolling_1000_final_linear_speed[indices] = final_linear_speed.detach().float()
        self._rolling_1000_final_angular_speed[indices] = final_angular_speed.detach().float()
        self._rolling_1000_max_hold_steps[indices] = max_hold_steps.detach().float()
        self._rolling_1000_clamp_activation_fraction[indices] = clamp_activation_fraction.detach().float()
        self._rolling_1000_valve_saturation_fraction[indices] = valve_saturation_fraction.detach().float()
        self._rolling_1000_cursor = (self._rolling_1000_cursor + n) % capacity
        self._rolling_1000_count = torch.clamp(self._rolling_1000_count + n, max=capacity)

    def _rolling_1000_mean(self, values: torch.Tensor) -> torch.Tensor:
        count = torch.clamp(self._rolling_1000_count, min=1)
        mask = torch.arange(values.numel(), device=self.device) < count
        return torch.sum(values * mask.float()) / count.to(dtype=torch.float32)

    def _rolling_1000_max(self, values: torch.Tensor) -> torch.Tensor:
        count = torch.clamp(self._rolling_1000_count, min=1)
        mask = torch.arange(values.numel(), device=self.device) < count
        active_values = torch.where(mask, values, torch.zeros_like(values))
        return torch.max(active_values)

    def _rolling_1000_log(self) -> dict[str, object]:
        return {
            "rolling_1000/success_rate": self._rolling_1000_mean(self._rolling_1000_success),
            "rolling_1000/timeout_rate": self._rolling_1000_mean(self._rolling_1000_timeout),
            "rolling_1000/final_position_error": self._rolling_1000_mean(
                self._rolling_1000_final_position_error
            ),
            "rolling_1000/final_orientation_error": self._rolling_1000_mean(
                self._rolling_1000_final_orientation_error
            ),
            "rolling_1000/final_linear_speed": self._rolling_1000_mean(
                self._rolling_1000_final_linear_speed
            ),
            "rolling_1000/final_angular_speed": self._rolling_1000_mean(
                self._rolling_1000_final_angular_speed
            ),
            "rolling_1000/max_hold_steps": self._rolling_1000_max(self._rolling_1000_max_hold_steps),
            "rolling_1000/clamp_activation_fraction": self._rolling_1000_mean(
                self._rolling_1000_clamp_activation_fraction
            ),
            "rolling_1000/valve_saturation_fraction": self._rolling_1000_mean(
                self._rolling_1000_valve_saturation_fraction
            ),
        }

    @property
    def _previous_action(self) -> torch.Tensor:
        return self._prev_action

    def _final_pose_target(self) -> torch.Tensor:
        """Return the authoritative robot-base final pose target."""
        final_pose_target = self._ee_pose_command
        if final_pose_target.shape != (self.num_envs, 7):
            raise RuntimeError(
                "Final pose target must have shape "
                f"({self.num_envs}, 7), got "
                f"{tuple(final_pose_target.shape)}"
            )
        return final_pose_target

    def _sample_task_space_command(self, env_ids: torch.Tensor) -> None:
        """Sample a reachable reset-time final pose from a safe robot configuration."""
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return

        sampled_joint_pos = self._q_ref_target[env_ids].detach().clone()
        sampled_joint_pos = self._apply_joint5_coupling_to_reference(
            sampled_joint_pos,
            lower=self.robot.data.soft_joint_pos_limits[env_ids][:, self._state_joint_ids, 0],
            upper=self.robot.data.soft_joint_pos_limits[env_ids][:, self._state_joint_ids, 1],
        )
        joint_pos_for_fk = self.robot.data.joint_pos[env_ids].detach().clone()
        joint_vel_for_fk = torch.zeros_like(joint_pos_for_fk)
        joint_pos_for_fk[:, self._state_joint_ids] = sampled_joint_pos
        self.robot.write_joint_state_to_sim(joint_pos_for_fk, joint_vel_for_fk, None, env_ids)
        self.robot.update(0.0)
        self._update_task_space_state()

        final_pose_target = self._final_pose_target()
        final_pose_target[env_ids] = self._current_ee_pose[env_ids].detach().clone()

    def _update_task_space_state(self) -> None:
        final_pose_target = self._final_pose_target()
        if self._ee_body_id is None:
            self._current_ee_pose[:, :3] = 0.0
            self._current_ee_pose[:, 3:] = torch.as_tensor(
                [1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device
            ).reshape(1, 4)
            self._ee_position_error.zero_()
            self._ee_orientation_error.zero_()
            self._ee_linear_velocity_base.zero_()
            self._ee_angular_velocity_base.zero_()
            self._ee_linear_speed.zero_()
            self._ee_angular_speed.zero_()
            return
        ee_state_w = self.robot.data.body_link_state_w[:, self._ee_body_id]
        root_state_w = self.robot.data.root_link_state_w
        ee_pos_w = ee_state_w[:, :3]
        ee_quat_w = ee_state_w[:, 3:7]
        ee_linear_velocity_w = ee_state_w[:, 7:10]
        ee_angular_velocity_w = ee_state_w[:, 10:13]
        root_pos_w = root_state_w[:, :3]
        root_quat_w = root_state_w[:, 3:7]
        root_linear_velocity_w = root_state_w[:, 7:10]
        root_angular_velocity_w = root_state_w[:, 10:13]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
        target_quaternion = normalize_quaternion_wxyz(
            final_pose_target[:, 3:7],
            name="final_pose_target quaternion",
            validate=False,
        )
        current_quaternion = normalize_quaternion_wxyz(
            ee_quat_b,
            name="measured end-effector quaternion",
            validate=False,
        )
        root_to_ee_w = ee_pos_w - root_pos_w
        ee_linear_velocity_relative_w = (
            ee_linear_velocity_w
            - root_linear_velocity_w
            - torch.cross(root_angular_velocity_w, root_to_ee_w, dim=-1)
        )
        ee_angular_velocity_relative_w = ee_angular_velocity_w - root_angular_velocity_w
        ee_linear_velocity_base = quat_apply_inverse(
            root_quat_w, ee_linear_velocity_relative_w
        )
        ee_angular_velocity_base = quat_apply_inverse(
            root_quat_w, ee_angular_velocity_relative_w
        )
        self._current_ee_pose[:, :3] = ee_pos_b.detach()
        self._current_ee_pose[:, 3:] = current_quaternion.detach()
        target_position = final_pose_target[:, 0:3]
        current_position = self._current_ee_pose[:, 0:3]
        position_error = target_position - current_position
        orientation_error = quaternion_geodesic_error(
            target_quaternion,
            current_quaternion,
            validate=False,
        )
        self._ee_position_error = position_error.detach()
        self._ee_orientation_error = orientation_error.reshape(-1, 1).detach()
        self._ee_linear_velocity_base = ee_linear_velocity_base.detach()
        self._ee_angular_velocity_base = ee_angular_velocity_base.detach()
        self._ee_linear_speed = torch.linalg.vector_norm(
            self._ee_linear_velocity_base, dim=-1
        )
        self._ee_angular_speed = torch.linalg.vector_norm(
            self._ee_angular_velocity_base, dim=-1
        )

    def _target_changed_this_episode(self) -> torch.Tensor:
        return torch.any(
            self._final_pose_target() != self._final_pose_target_at_episode_start,
            dim=-1,
        )

    def _clear_global_hold_break_reasons(self) -> None:
        self._final_pose_hold_break_reason_position.zero_()
        self._final_pose_hold_break_reason_orientation.zero_()
        self._final_pose_hold_break_reason_linear_speed.zero_()
        self._final_pose_hold_break_reason_angular_speed.zero_()
        self._final_pose_hold_break_reason_target_changed.zero_()
        self._final_pose_hold_break_reason_timeout.zero_()
        self._final_pose_hold_break_reason_success_reached.zero_()
        self._final_pose_hold_break_reason_unknown.zero_()

    def _update_global_hold_diagnostic_without_break(
        self,
        *,
        candidate_mask: torch.Tensor,
        hold_steps: torch.Tensor,
        position_error_norm: torch.Tensor,
        orientation_error: torch.Tensor,
        linear_speed: torch.Tensor,
        angular_speed: torch.Tensor,
    ) -> None:
        if not bool(torch.any(candidate_mask)):
            return
        candidate_steps = torch.where(candidate_mask, hold_steps, torch.zeros_like(hold_steps))
        best_steps = torch.max(candidate_steps)
        if best_steps <= self._final_pose_hold_diagnostic_best_steps():
            return
        best_env = torch.argmax(candidate_steps).reshape(())
        self._final_pose_settling_hold_max_total = best_steps.detach().float().clone()
        self._clear_global_hold_break_reasons()
        success_reached = best_steps >= int(self.cfg.final_pose_success_hold_steps)
        self._final_pose_hold_break_reason_success_reached.copy_(
            success_reached.detach().float()
        )
        self._final_pose_hold_break_reason_unknown.copy_((~success_reached).detach().float())
        self._final_pose_max_hold_position_error_at_break.copy_(position_error_norm[best_env].detach())
        self._final_pose_max_hold_orientation_error_at_break.copy_(orientation_error[best_env].detach())
        self._final_pose_max_hold_linear_speed_at_break.copy_(linear_speed[best_env].detach())
        self._final_pose_max_hold_angular_speed_at_break.copy_(angular_speed[best_env].detach())
        self._final_pose_max_hold_steps_before_break_total.copy_(best_steps.detach().float())
        self._final_pose_steps_remaining_after_max_hold_total.copy_(
            self._final_pose_steps_remaining_after_max_hold[best_env].detach().float()
        )
        self._final_pose_first_step_entered_strict_settled.copy_(
            self._final_pose_first_strict_settled_step[best_env].detach().float()
        )
        self._final_pose_last_step_entered_strict_settled.copy_(
            self._final_pose_last_strict_settled_step[best_env].detach().float()
        )

    def _record_hold_break_diagnostic(
        self,
        *,
        candidate_mask: torch.Tensor,
        hold_steps_before_break: torch.Tensor,
        reason_position: torch.Tensor,
        reason_orientation: torch.Tensor,
        reason_linear_speed: torch.Tensor,
        reason_angular_speed: torch.Tensor,
        reason_target_changed: torch.Tensor,
        reason_timeout: torch.Tensor,
        position_error_norm: torch.Tensor,
        orientation_error: torch.Tensor,
        linear_speed: torch.Tensor,
        angular_speed: torch.Tensor,
    ) -> None:
        if not bool(torch.any(candidate_mask)):
            return
        candidate_steps = torch.where(
            candidate_mask,
            hold_steps_before_break,
            torch.zeros_like(hold_steps_before_break),
        )
        best_steps = torch.max(candidate_steps)
        if best_steps < self._final_pose_hold_diagnostic_best_steps():
            return
        best_env = torch.argmax(candidate_steps).reshape(())
        self._final_pose_settling_hold_max_total = torch.maximum(
            self._final_pose_settling_hold_max_total,
            best_steps.detach().float(),
        )
        self._clear_global_hold_break_reasons()
        self._final_pose_hold_break_reason_position.copy_(reason_position[best_env].detach().float())
        self._final_pose_hold_break_reason_orientation.copy_(reason_orientation[best_env].detach().float())
        self._final_pose_hold_break_reason_linear_speed.copy_(reason_linear_speed[best_env].detach().float())
        self._final_pose_hold_break_reason_angular_speed.copy_(reason_angular_speed[best_env].detach().float())
        self._final_pose_hold_break_reason_target_changed.copy_(reason_target_changed[best_env].detach().float())
        self._final_pose_hold_break_reason_timeout.copy_(reason_timeout[best_env].detach().float())
        success_reached = best_steps >= int(self.cfg.final_pose_success_hold_steps)
        self._final_pose_hold_break_reason_success_reached.copy_(
            success_reached.detach().float()
        )
        explained = (
            reason_position[best_env]
            | reason_orientation[best_env]
            | reason_linear_speed[best_env]
            | reason_angular_speed[best_env]
            | reason_target_changed[best_env]
            | reason_timeout[best_env]
            | success_reached
        )
        self._final_pose_hold_break_reason_unknown.copy_((~explained).detach().float())
        self._final_pose_max_hold_position_error_at_break.copy_(position_error_norm[best_env].detach())
        self._final_pose_max_hold_orientation_error_at_break.copy_(orientation_error[best_env].detach())
        self._final_pose_max_hold_linear_speed_at_break.copy_(linear_speed[best_env].detach())
        self._final_pose_max_hold_angular_speed_at_break.copy_(angular_speed[best_env].detach())
        self._final_pose_max_hold_steps_before_break_total.copy_(best_steps.detach().float())
        self._final_pose_steps_remaining_after_max_hold_total.copy_(
            self._final_pose_steps_remaining_after_max_hold[best_env].detach().float()
        )
        self._final_pose_first_step_entered_strict_settled.copy_(
            self._final_pose_first_strict_settled_step[best_env].detach().float()
        )
        self._final_pose_last_step_entered_strict_settled.copy_(
            self._final_pose_last_strict_settled_step[best_env].detach().float()
        )

    def _final_pose_hold_diagnostic_best_steps(self) -> torch.Tensor:
        return self._final_pose_settling_hold_max_total.to(dtype=torch.long)

    def _record_hold_timeout_break(self, time_out: torch.Tensor) -> None:
        position_error_norm = torch.linalg.vector_norm(self._ee_position_error, dim=-1)
        orientation_error = self._ee_orientation_error.reshape(-1)
        current_hold_steps = self._final_pose_success_counter.detach().clone()
        timeout_break = time_out & (current_hold_steps > 0) & ~self._termination_success
        false_reason = torch.zeros_like(timeout_break)
        self._record_hold_break_diagnostic(
            candidate_mask=timeout_break,
            hold_steps_before_break=current_hold_steps,
            reason_position=false_reason,
            reason_orientation=false_reason,
            reason_linear_speed=false_reason,
            reason_angular_speed=false_reason,
            reason_target_changed=self._target_changed_this_episode() & timeout_break,
            reason_timeout=timeout_break,
            position_error_norm=position_error_norm,
            orientation_error=orientation_error,
            linear_speed=self._ee_linear_speed.reshape(-1),
            angular_speed=self._ee_angular_speed.reshape(-1),
        )

    def _update_final_pose_success_state(self, unsafe_termination: torch.Tensor) -> None:
        position_error_norm = torch.linalg.vector_norm(self._ee_position_error, dim=-1)
        orientation_error = self._ee_orientation_error.reshape(-1)
        strict_position = position_error_norm <= float(self.cfg.final_pose_position_tolerance_m)
        strict_orientation = orientation_error <= float(self.cfg.final_pose_orientation_tolerance_rad)
        strict_linear_speed = self._ee_linear_speed <= float(
            self.cfg.final_pose_linear_speed_tolerance_mps
        )
        strict_angular_speed = self._ee_angular_speed <= float(
            self.cfg.final_pose_angular_speed_tolerance_radps
        )
        strict_all_except_hold = (
            strict_position
            & strict_orientation
            & strict_linear_speed
            & strict_angular_speed
        )
        within_final_pose = final_pose_within_tolerance(
            position_error_norm,
            orientation_error,
            self._ee_linear_speed,
            self._ee_angular_speed,
            position_tolerance_m=float(self.cfg.final_pose_position_tolerance_m),
            orientation_tolerance_rad=float(self.cfg.final_pose_orientation_tolerance_rad),
            linear_speed_tolerance_mps=float(self.cfg.final_pose_linear_speed_tolerance_mps),
            angular_speed_tolerance_radps=float(
                self.cfg.final_pose_angular_speed_tolerance_radps
            ),
        )
        if not bool(torch.equal(strict_all_except_hold, within_final_pose)):
            raise RuntimeError("Strict final-pose predicate mismatch.")
        previous_counter = self._final_pose_success_counter.detach().clone()
        current_step = self.episode_length_buf.detach().to(dtype=torch.long)
        first_strict = strict_all_except_hold & (self._final_pose_first_strict_settled_step < 0)
        self._final_pose_first_strict_settled_step = torch.where(
            first_strict,
            current_step,
            self._final_pose_first_strict_settled_step,
        )
        self._final_pose_last_strict_settled_step = torch.where(
            strict_all_except_hold,
            current_step,
            self._final_pose_last_strict_settled_step,
        )
        self._strict_position_condition = strict_position.detach().clone()
        self._strict_orientation_condition = strict_orientation.detach().clone()
        self._strict_linear_speed_condition = strict_linear_speed.detach().clone()
        self._strict_angular_speed_condition = strict_angular_speed.detach().clone()
        self._strict_all_except_hold_condition = strict_all_except_hold.detach().clone()
        self._within_final_pose = strict_all_except_hold.detach().clone()
        self._final_pose_success_counter = update_consecutive_success_counter(
            self._final_pose_success_counter,
            self._within_final_pose,
        )
        new_env_max = self._final_pose_success_counter > self._final_pose_settling_hold_max
        remaining_steps = torch.clamp(
            int(self._stage_max_episode_length) - current_step - 1,
            min=0,
        )
        self._final_pose_steps_remaining_after_max_hold = torch.where(
            new_env_max,
            remaining_steps,
            self._final_pose_steps_remaining_after_max_hold,
        )
        self._final_pose_settling_hold_max = torch.maximum(
            self._final_pose_settling_hold_max,
            self._final_pose_success_counter,
        )
        self._update_global_hold_diagnostic_without_break(
            candidate_mask=new_env_max,
            hold_steps=self._final_pose_success_counter.detach().clone(),
            position_error_norm=position_error_norm,
            orientation_error=orientation_error,
            linear_speed=self._ee_linear_speed.reshape(-1),
            angular_speed=self._ee_angular_speed.reshape(-1),
        )
        hold_broke = (previous_counter > 0) & ~strict_all_except_hold
        target_changed = self._target_changed_this_episode()
        self._final_pose_max_hold_steps_before_break = torch.where(
            hold_broke & (previous_counter >= self._final_pose_max_hold_steps_before_break),
            previous_counter,
            self._final_pose_max_hold_steps_before_break,
        )
        self._record_hold_break_diagnostic(
            candidate_mask=hold_broke,
            hold_steps_before_break=previous_counter,
            reason_position=hold_broke & ~strict_position,
            reason_orientation=hold_broke & ~strict_orientation,
            reason_linear_speed=hold_broke & ~strict_linear_speed,
            reason_angular_speed=hold_broke & ~strict_angular_speed,
            reason_target_changed=hold_broke & target_changed,
            reason_timeout=torch.zeros_like(hold_broke),
            position_error_norm=position_error_norm,
            orientation_error=orientation_error,
            linear_speed=self._ee_linear_speed.reshape(-1),
            angular_speed=self._ee_angular_speed.reshape(-1),
        )
        final_pose_success = (
            self._final_pose_success_counter
            >= int(self.cfg.final_pose_success_hold_steps)
        )
        self._new_final_pose_success = (
            final_pose_success & ~self._final_pose_success_previous
        )
        first_success_step = torch.full_like(
            self._final_pose_time_to_success_steps,
            -1,
        )
        first_success_step[:] = self.episode_length_buf
        self._final_pose_time_to_success_steps = torch.where(
            self._new_final_pose_success
            & (self._final_pose_time_to_success_steps < 0),
            first_success_step,
            self._final_pose_time_to_success_steps,
        )
        self._termination_success = final_pose_success.detach().clone()
        self._final_pose_success_previous = final_pose_success.detach().clone()

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
        controller_debug = self._controller.debug_state()

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
            "reset_count": self._reset_count[env_id].detach().clone(),
            "reset_reason": self._reset_reason_name(float(self._last_reset_reason_code[env_id].detach().cpu())),
            "reset_reason_code": self._last_reset_reason_code[env_id].detach().clone(),
            "timeout": self._last_time_out_debug[env_id].detach().float().clone(),
            "unsafe_termination": self._last_terminated_debug[env_id].detach().float().clone(),
            "target_reached_termination": self._reset_due_to_target_reached[env_id].detach().clone(),
            "target_changed_flag": self._target_changed_flag[env_id].detach().clone(),
            "qdot": qdot[env_id].detach().clone(),
            "qdot_ref": self._qdot_ref[env_id].detach().clone(),
            "qddot_ref": self._qddot_ref[env_id].detach().clone(),
            "ee_pose_command": self._ee_pose_command[env_id].detach().clone(),
            "current_ee_pose": self._current_ee_pose[env_id].detach().clone(),
            "ee_position_error": self._ee_position_error[env_id].detach().clone(),
            "ee_orientation_error": self._ee_orientation_error[env_id].detach().clone(),
            "vehicle_command_mode": str(self.cfg.vehicle_command_mode),
            "vehicle_wheel_joint_names": tuple(self.cfg.vehicle_wheel_joint_names),
            "vehicle_wheel_joint_ids": tuple(self._vehicle_wheel_joint_ids),
            "vehicle_speed_kmh": self._vehicle_commanded_speed_kmh[env_id].detach().clone(),
            "vehicle_wheel_velocity_target": self._vehicle_wheel_velocity_target[env_id].detach().clone(),
            "root_pos_w": self.robot.data.root_pos_w[env_id].detach().clone(),
            "root_quat_w": self.robot.data.root_quat_w[env_id].detach().clone(),
            "root_lin_vel_w": self.robot.data.root_lin_vel_w[env_id].detach().clone(),
            "root_ang_vel_w": self.robot.data.root_ang_vel_w[env_id].detach().clone(),
            "action": self._current_action[env_id].detach().clone(),
            "qddot_cmd": self._current_qddot_cmd[env_id].detach().clone(),
            "id_input_raw": controller_debug["id_input_raw"][env_id].detach().clone(),
            "id_input_normalized": controller_debug["id_input_normalized"][env_id].detach().clone(),
            "id_input_normalized_min": controller_debug["id_input_normalized_min"][env_id].detach().clone(),
            "id_input_normalized_max": controller_debug["id_input_normalized_max"][env_id].detach().clone(),
            "id_input_normalized_abs_max": controller_debug["id_input_normalized_abs_max"][env_id].detach().clone(),
            "z_arm_norm": controller_debug["z_arm_norm"][env_id].detach().clone(),
            "z_arm_runtime_std": controller_debug["z_arm_std"][env_id].detach().clone(),
            "scripted_qddot_feedback": self._scripted_qddot_feedback[env_id].detach().clone(),
            "scripted_qddot_feedforward": self._scripted_qddot_feedforward[env_id].detach().clone(),
            "scripted_qddot_raw": self._scripted_qddot_raw[env_id].detach().clone(),
            "scripted_action_preclip": self._scripted_action_preclip[env_id].detach().clone(),
            "scripted_control_ramp": self._scripted_control_ramp[env_id].detach().clone(),
            "valve_cmd": self._current_valve_cmd[env_id].detach().clone(),
            "rom_cylinder_pos_target": self._rom_cylinder_pos_target[env_id].detach().clone(),
            "rom_cylinder_pos_target_at_reset": self._rom_cylinder_pos_target_at_reset[env_id].detach().clone(),
            "rom_cylinder_cumulative_target_motion": self._rom_cylinder_cumulative_target_motion[
                env_id
            ].detach().clone(),
            "rom_cylinder_cumulative_measured_motion": self._rom_cylinder_cumulative_measured_motion[
                env_id
            ].detach().clone(),
            "rom_cylinder_tracking_ratio": self._rom_cylinder_tracking_ratio[env_id].detach().clone(),
            "rom_pressure": self._rom_pressure[env_id].detach().clone(),
            "final_rom_command_coupling": "ValveCmd-conditioned deployed correction",
            "rom_xdot_pred": self._rom_xdot_pred[env_id].detach().clone(),
            "rom_dx_pred": self._rom_dx_pred[env_id].detach().clone(),
            "final_rom_delta_extension_50ms": self._final_rom_delta_extension_50ms[env_id].detach().clone(),
            "final_rom_delta_extension_kinematic_50ms": self._final_rom_delta_extension_kinematic_50ms[
                env_id
            ].detach().clone(),
            "final_rom_delta_extension_correction_50ms": self._final_rom_delta_extension_correction_50ms[
                env_id
            ].detach().clone(),
            "final_rom_delta_extension_10ms": self._final_rom_delta_extension_10ms[env_id].detach().clone(),
            "final_rom_delta_isaac_50ms": self._final_rom_delta_isaac_50ms[env_id].detach().clone(),
            "final_rom_delta_isaac_10ms": self._final_rom_delta_isaac_10ms[env_id].detach().clone(),
            "final_rom_extension": self._final_rom_extension[env_id].detach().clone(),
            "final_rom_extension_velocity": self._final_rom_extension_velocity[env_id].detach().clone(),
            "final_rom_input_features_raw": self._final_rom_input_features_raw[env_id].detach().clone(),
            "final_rom_input_features_normalized": self._final_rom_input_features_normalized[env_id].detach().clone(),
            "final_rom_input_normalized_min": torch.min(
                self._final_rom_input_features_normalized[env_id].detach()
            ).clone(),
            "final_rom_input_normalized_max": torch.max(
                self._final_rom_input_features_normalized[env_id].detach()
            ).clone(),
            "final_rom_update_flag": self._final_rom_update_flag[env_id].detach().clone(),
            "final_rom_history_ready": self._final_rom_history_ready[env_id].detach().clone(),
            "final_rom_mapping_scale": (
                self._cylinder_coordinate_mapping.scale_tensor(self._final_rom_delta_isaac_10ms).reshape(-1)
                .detach()
                .clone()
                if self._cylinder_coordinate_mapping is not None
                else torch.ones((4,), dtype=torch.float32, device=self.device)
            ),
            "final_rom_mapping_sign": (
                self._cylinder_coordinate_mapping.sign_tensor(self._final_rom_delta_isaac_10ms).reshape(-1)
                .detach()
                .clone()
                if self._cylinder_coordinate_mapping is not None
                else torch.ones((4,), dtype=torch.float32, device=self.device)
            ),
            "rom_valve_cmd_delayed": self._rom_valve_cmd_delayed[env_id].detach().clone(),
            "rom_valve_cmd_effective": self._rom_valve_cmd_effective[env_id].detach().clone(),
            "rom_target_clamp_fraction": self._rom_target_clamp_fraction[env_id].detach().clone(),
            "rom_pressure_clamp_fraction": self._rom_pressure_clamp_fraction[env_id].detach().clone(),
            "rom_nan_guard_triggered": self._rom_nan_guard_triggered[env_id].detach().clone(),
            "rom_rate_limit_fraction": self._rom_rate_limit_fraction[env_id].detach().clone(),
            "rom_ch4_damping_applied": self._rom_ch4_damping_applied[env_id].detach().clone(),
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
            "fd_tau_hat": self._fd_tau_hat[env_id].detach().clone(),
            "fd_tau_hat_normalized": self._fd_tau_hat_normalized[env_id].detach().clone(),
            "fd_soft_limit_fraction": self._fd_soft_limit_fraction[env_id].detach().clone(),
            "fd_hard_limit_fraction": self._fd_hard_limit_fraction[env_id].detach().clone(),
            "fd_normalized_abs_max": self._fd_normalized_abs_max[env_id].detach().clone(),
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

    def _scripted_demo_elapsed(self) -> torch.Tensor:
        step_dt = float(getattr(self, "step_dt", float(self.cfg.sim.dt) * float(self.cfg.decimation)))
        return self.episode_length_buf.to(dtype=torch.float32) * step_dt

    def _hold_reference_at_current_state(self, q: torch.Tensor) -> None:
        previous_commanded = self._q_ref_commanded.detach().clone()
        q_ref = self._apply_joint5_coupling_to_reference(q.detach().clone())
        self._q_ref_prev = previous_commanded
        self._q_ref_commanded = q_ref.detach().clone()
        self._q_ref_target = self._q_ref_commanded.detach().clone()
        self._q_ref = self._q_ref_commanded
        self._qdot_ref.zero_()
        self._qddot_ref.zero_()
        self._q_ref_delta = self._q_ref_commanded - previous_commanded
        self._q_ref_target_delta = self._q_ref_delta.detach().clone()
        self._q_ref_target_observed = self._q_ref_target.detach().clone()
        self._q_ref_ramp_progress.zero_()
        self._q_ref_target_scale.fill_(1.0)
        self._q_ref_curriculum_progress.fill_(1.0)
        self._target_changed_flag.zero_()

    def _maybe_initialize_learned_control(self, q: torch.Tensor, qdot: torch.Tensor, dP: torch.Tensor) -> None:
        if (self._task_control_mode() == "scripted_reference_demo"
                or self._learned_preflight_scripted_action_active):
            elapsed = self._scripted_demo_elapsed()
            ready_mask = elapsed >= float(self.cfg.scripted_demo_settle_time_s)
        else:
            ready_mask = torch.ones((self.num_envs,), dtype=torch.bool, device=self.device)
        init_mask = ready_mask & (~self._learned_control_initialized)
        if not bool(init_mask.any()):
            return
        env_ids = torch.nonzero(init_mask, as_tuple=False).reshape(-1)
        self._initialize_learned_control_from_current_state(env_ids, q[env_ids], qdot[env_ids], dP[env_ids])

    def _initialize_learned_control_from_current_state(
        self,
        env_ids: torch.Tensor,
        q: torch.Tensor,
        qdot: torch.Tensor,
        dP: torch.Tensor,
    ) -> None:
        zero_valve = torch.zeros((env_ids.numel(), 4), dtype=torch.float32, device=self.device)
        zero_qddot = torch.zeros((env_ids.numel(), 5), dtype=torch.float32, device=self.device)
        self._controller.reset(env_ids)
        z_arm = self._controller.warm_start_histories(
            q,
            qdot,
            dP,
            zero_valve,
            qddot_cmd=zero_qddot,
            env_ids=env_ids,
        )
        self._z_arm_hat[env_ids] = z_arm.detach().clone()
        q_cyl = self.robot.data.joint_pos[:, self._command_joint_ids].detach()[env_ids]
        qdot_cyl = self.robot.data.joint_vel[:, self._command_joint_ids].detach()[env_ids]
        self._rom_cylinder_pos_target[env_ids] = q_cyl.detach().clone()
        self._rom_cylinder_pos_target_at_reset[env_ids] = q_cyl.detach().clone()
        self._rom_cylinder_measured_at_reset[env_ids] = q_cyl.detach().clone()
        self._rom_cylinder_cumulative_target_motion[env_ids] = 0.0
        self._rom_cylinder_cumulative_measured_motion[env_ids] = 0.0
        self._rom_cylinder_tracking_ratio[env_ids] = 0.0
        if self._final_causal_rom_wrapper is not None:
            if str(self.cfg.final_causal_rom_pressure_source) == "sd_observer":
                pressure_limit = float(getattr(self.cfg, "final_causal_rom_pressure_clamp", 0.0))
                rom_pressure_input = clamp_and_sanitize(dP.detach().clone(), pressure_limit if pressure_limit > 0.0 else None)
            else:
                rom_pressure_input = self._final_causal_rom_pressure_input(dP)
            features = self._final_causal_rom_feature_row(
                q,
                qdot,
                rom_pressure_input,
                zero_valve,
                q_cyl=q_cyl,
                qdot_cyl=qdot_cyl,
                env_ids=env_ids,
            )
            history = features.reshape(env_ids.numel(), 1, -1).repeat(1, HISTORY_LENGTH, 1)
            self._final_causal_rom_wrapper.prime_history(history, env_ids)
        self._learned_control_initialized[env_ids] = True

    def _update_aam_latent(
        self,
        q: torch.Tensor,
        qdot: torch.Tensor,
        dP: torch.Tensor,
    ) -> None:
        # _prev_valve_cmd is the command that was supplied to FD and ROM on
        # the preceding control step, so it is the applied ValveCmd history.
        previous_applied_valve_cmd = self._prev_valve_cmd
        aam_step_features = torch.cat(
            (
                q,
                qdot,
                dP,
                previous_applied_valve_cmd,
            ),
            dim=-1,
        )
        expected_features_shape = (self.num_envs, 18)
        if tuple(aam_step_features.shape) != expected_features_shape:
            raise RuntimeError(
                "AAM step features must have shape "
                f"({self.num_envs}, 18), got "
                f"{tuple(aam_step_features.shape)}"
            )
        with torch.no_grad():
            z_arm_hat_new = self._controller.update_aam_latent(aam_step_features)
        expected_latent_shape = (self.num_envs, 24)
        if tuple(z_arm_hat_new.shape) != expected_latent_shape:
            raise RuntimeError(
                "AAM latent must have shape "
                f"({self.num_envs}, 24), got "
                f"{tuple(z_arm_hat_new.shape)}"
            )
        if not torch.isfinite(z_arm_hat_new).all():
            raise RuntimeError("AAM produced non-finite latent values.")
        self._z_arm_hat.copy_(z_arm_hat_new)
        self._aam_latent_state_version += 1

    def _id_conditioning_latent(self) -> torch.Tensor:
        if not torch.equal(
            self._observation_aam_state_version,
            self._aam_latent_state_version,
        ):
            raise RuntimeError(
                "ID received an AAM latent version that does not match the PPO observation."
            )
        self._id_aam_state_version.copy_(self._aam_latent_state_version)
        return self._z_arm_hat

    def _prepare_high_level_action(self, actions: torch.Tensor, q: torch.Tensor, qdot: torch.Tensor) -> None:
        task_mode = self._task_control_mode()
        use_scripted_action = (
            task_mode in {"scripted_reference_demo", "joint_space_validation"}
            or self._learned_preflight_scripted_action_active
        )
        if task_mode == "rl_task_space_tracking":
            self._update_task_space_state()
        if use_scripted_action:
            elapsed = self._scripted_demo_elapsed()
            settling = elapsed < float(self.cfg.scripted_demo_settle_time_s)
            if bool(settling.any()):
                self._hold_reference_at_current_state(q)
            if bool((~settling).any()):
                self._update_scripted_reference()
            action = self._scripted_reference_action(q, qdot)
            ramp_time = max(float(self.cfg.scripted_demo_control_ramp_time_s), 1.0e-6)
            ramp_u = torch.clamp((elapsed - float(self.cfg.scripted_demo_settle_time_s)) / ramp_time, 0.0, 1.0)
            ramp = 3.0 * torch.square(ramp_u) - 2.0 * torch.pow(ramp_u, 3)
            self._scripted_control_ramp = ramp.detach().clone()
            action = action * ramp.reshape(-1, 1)
            action = torch.where(settling.reshape(-1, 1), torch.zeros_like(action), action)
        elif task_mode in {"rl_policy", "rl_task_space_tracking"}:
            self._update_q_ref_commanded()
            action = clamp_and_sanitize(actions, 1.0)
        else:
            raise ValueError(f"Unsupported task control_mode: {task_mode!r}")

        self._current_action = self.cfg.action_scale * action
        # Apply curriculum stage-dependent action-authority multiplier ONLY for
        # rl_task_space_tracking mode and NOT during preflight.  This reduces PPO
        # action magnitude in early stages, preventing aggressive qddot commands
        # that would violate ROM / Virtual-Maxx linearity.
        apply_authority = (
            self._task_control_mode() == "rl_task_space_tracking"
            and not self._learned_preflight_scripted_action_active
        )
        if apply_authority:
            self._current_action = self._current_action * self._action_authority_multiplier
        self._action_delta = self._current_action - self._prev_action
        self._current_qddot_cmd = self._controller.build_qddot_cmd(
            self._current_action,
            self.cfg.action_to_qddot_scale,
            self.cfg.qddot_cmd_clip,
        )
        self._update_reference_marker()

    def _update_reference_marker(self) -> None:
        marker = getattr(self, "_reference_marker", None)
        if marker is None:
            return
        base_offset = torch.as_tensor(
            self.cfg.reference_marker_base_offset, dtype=torch.float32, device=self.device
        ).reshape(1, 3)
        joint_scale = torch.as_tensor(
            self.cfg.reference_marker_joint_scale, dtype=torch.float32, device=self.device
        ).reshape(1, 3)
        encoded_reference = base_offset + joint_scale * self._q_ref_commanded[:, :3]
        positions = self.robot.data.root_pos_w + encoded_reference
        orientations = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        orientations[:, 0] = 1.0
        marker.visualize(positions, orientations)

    def step(self, actions):
        """Override to hold policy actions across the ROM multirate window.

        The final causal ROM predicts at 50ms but the Isaac control step is 10ms.
        Without action holding, the policy provides 5 valve commands per ROM cycle
        but the ROM only sees the first one (wrapper inserts history only on step 0).
        The feedforward path applies all 5, creating a ROM-vs-ff mismatch.

        This override enforces 50ms policy cadence: the action from step 0 is held
        for the full 5-step ROM window. The PPO rollout buffer receives the held
        action via in-place tensor modification (same-device no-op .to() path).

        Mathematical justification (Chen & Francis, 1995; Bamieh & Pearson, 1992):
        For a multirate system with slow model period T_s and fast control period
        T_f = T_s / N, the lifted single-rate representation at period T_s with
        augmented input u_0..u_{N-1} is equivalent to the multirate system. By
        constraining u_i = u_0 for i > 0, we reduce to a standard single-rate
        control problem at period T_s without introducing additional learned components.
        """
        if (self.cfg.use_final_causal_cylinder_rom
                and self._final_causal_rom_wrapper is not None
                and self.num_envs > 0):
            # Read wrapper counter BEFORE the step (it tracks pending cycle position)
            rom_counter = int(self._final_causal_rom_wrapper.rom_step_counter[0].item())
            if rom_counter != 0 and self._rom_held_policy_action is not None:
                # Hold step: overwrite actions tensor in-place with held action
                if isinstance(actions, torch.Tensor):
                    actions[:] = self._rom_held_policy_action
            else:
                # ROM-update step: save the new action as the held action
                self._rom_held_policy_action = actions.clone()
        return super().step(actions)

    def _apply_rrlab_vehicle_command(self) -> None:
        mode = str(getattr(self.cfg, "vehicle_command_mode", "disabled"))
        if not bool(getattr(self.cfg, "enable_scripted_vehicle_motion", True)):
            self._vehicle_commanded_speed_kmh.zero_()
            self._vehicle_wheel_velocity_target.zero_()
            return
        if mode in {"", "disabled", "none"}:
            return
        if mode != "rrlab_obstacle_scripted_wheel_velocity":
            raise ValueError(f"Unsupported vehicle_command_mode: {mode!r}")
        if len(self._vehicle_wheel_joint_ids) == 0:
            if not self._vehicle_command_warned_missing_joints:
                print("[WARN] Skipping RRLAB vehicle command because no wheel joint IDs were resolved.", flush=True)
                self._vehicle_command_warned_missing_joints = True
            return

        # Curriculum stage-aware vehicle speed: stationary in early stages,
        # scripted wheel-velocity range in later stages.
        stage = self._curriculum_stage()
        stationary_stages = int(getattr(self.cfg, "curriculum_vehicle_stationary_stages", 0))
        if stage <= stationary_stages:
            self._vehicle_commanded_speed_kmh.zero_()
            self._vehicle_wheel_velocity_target.zero_()
            return

        resample_mask = self._vehicle_commanded_speed_kmh <= 0.0
        if bool(resample_mask.any()):
            env_ids = torch.nonzero(resample_mask, as_tuple=False).reshape(-1)
            idx = stage - 1
            lower = float(self.cfg.curriculum_vehicle_speed_min_kmh[idx])
            upper = float(self.cfg.curriculum_vehicle_speed_max_kmh[idx])
            sampled_speed_kmh = lower + (upper - lower) * torch.rand((env_ids.numel(),), device=self.device)
            omega = (sampled_speed_kmh / 3.6) / float(self.cfg.vehicle_wheel_radius_m)
            self._vehicle_commanded_speed_kmh[env_ids] = sampled_speed_kmh
            self._vehicle_wheel_velocity_target[env_ids] = omega.reshape(-1, 1).expand(
                -1, len(self._vehicle_wheel_joint_ids)
            )

        self.robot.write_joint_velocity_to_sim(
            self._vehicle_wheel_velocity_target,
            joint_ids=self._vehicle_wheel_joint_ids,
        )

    def _update_scripted_reference(self) -> None:
        previous_commanded = self._q_ref_commanded.detach().clone()
        step_dt = float(getattr(self, "step_dt", float(self.cfg.sim.dt) * float(self.cfg.decimation)))
        t = self.episode_length_buf.to(dtype=torch.float32).reshape(-1, 1) * step_dt + self._scripted_reference_phase_offset
        amplitude = torch.as_tensor(self.cfg.scripted_reference_amplitude, dtype=torch.float32, device=self.device).reshape(1, 5)
        frequency = torch.as_tensor(
            self.cfg.scripted_reference_frequency_hz, dtype=torch.float32, device=self.device
        ).reshape(1, 5)
        phase = torch.as_tensor(self.cfg.scripted_reference_phase_rad, dtype=torch.float32, device=self.device).reshape(1, 5)
        center = self.robot.data.default_joint_pos[:, self._state_joint_ids].detach()
        angle = 2.0 * math.pi * frequency * t + phase
        q_ref = center + amplitude * torch.sin(angle)
        omega = 2.0 * math.pi * frequency
        qdot_ref = amplitude * omega * torch.cos(angle)
        qddot_ref = -amplitude * torch.square(omega) * torch.sin(angle)

        lower = self.robot.data.soft_joint_pos_limits[:, self._state_joint_ids, 0]
        upper = self.robot.data.soft_joint_pos_limits[:, self._state_joint_ids, 1]
        has_limits = torch.isfinite(lower) & torch.isfinite(upper)
        safe_lower = torch.where(has_limits, lower + self.cfg.joint_limit_margin, center - torch.abs(amplitude))
        safe_upper = torch.where(has_limits, upper - self.cfg.joint_limit_margin, center + torch.abs(amplitude))
        q_ref = torch.clamp(q_ref, safe_lower, safe_upper)
        q_ref = self._apply_joint5_coupling_to_reference(q_ref, lower=safe_lower, upper=safe_upper)
        if self.cfg.use_joint5_coupling:
            target_idx = self._joint5_coupling_target_joint
            source_idx = self._joint5_coupling_source_joint
            qdot_ref[:, target_idx] = float(self.cfg.joint5_coupling_gain) * qdot_ref[:, source_idx]
            qddot_ref[:, target_idx] = float(self.cfg.joint5_coupling_gain) * qddot_ref[:, source_idx]

        self._q_ref_prev = previous_commanded
        self._q_ref_commanded = q_ref.detach().clone()
        self._q_ref_target = self._q_ref_commanded
        self._q_ref = self._q_ref_commanded
        self._qdot_ref = qdot_ref.detach().clone()
        self._qddot_ref = qddot_ref.detach().clone()
        self._q_ref_delta = self._q_ref_commanded - previous_commanded
        self._q_ref_target_delta = self._q_ref_delta.detach().clone()
        self._q_ref_target_observed = self._q_ref_target.detach().clone()
        self._q_ref_ramp_progress.fill_(1.0)
        self._q_ref_target_scale.fill_(1.0)
        self._q_ref_curriculum_progress.fill_(1.0)
        self._target_changed_flag = (torch.max(torch.abs(self._q_ref_delta), dim=-1).values > 1.0e-6).float()
        self._target_change_step = torch.full_like(
            self._target_change_step,
            float(getattr(self, "common_step_counter", 0)),
        )

    def _scripted_reference_action(self, q: torch.Tensor, qdot: torch.Tensor) -> torch.Tensor:
        kp = torch.as_tensor(self.cfg.scripted_reference_kp, dtype=torch.float32, device=self.device).reshape(1, 4)
        kd = torch.as_tensor(self.cfg.scripted_reference_kd, dtype=torch.float32, device=self.device).reshape(1, 4)
        q_err = self._q_ref_commanded[:, :4] - q[:, :4]
        qdot_err = self._qdot_ref[:, :4] - qdot[:, :4]
        feedback = kp * q_err + kd * qdot_err
        feedforward = self._qddot_ref[:, :4]
        qddot_cmd = feedback + feedforward
        action = self._controller.qddot_to_action(qddot_cmd, self.cfg.action_to_qddot_scale)
        self._scripted_qddot_feedback = feedback.detach().clone()
        self._scripted_qddot_feedforward = feedforward.detach().clone()
        self._scripted_qddot_raw = qddot_cmd.detach().clone()
        self._scripted_action_preclip = action.detach().clone()
        return clamp_and_sanitize(action, float(self.cfg.scripted_reference_action_clip))

    def _resolve_fd_limit(self, value: float | str, normalizer: torch.Tensor, multiplier: float) -> torch.Tensor:
        if isinstance(value, str):
            if value == "fd_output_std":
                return normalizer
            if value == "fd_output_3std":
                return 3.0 * normalizer
            raise ValueError(f"Unsupported FD torque limit spec: {value!r}")
        return torch.full_like(normalizer, max(float(value), 1.0e-8) * multiplier)

    def _run_fd_observer_only(self, q: torch.Tensor, qdot: torch.Tensor, dP: torch.Tensor, valve_cmd: torch.Tensor) -> None:
        if not bool(getattr(self.cfg, "use_fd_model", True)) or not bool(getattr(self.cfg, "use_fd_as_observer", True)):
            self._current_fd_simscape_output.zero_()
            self._current_tau_fd_adapted.zero_()
            self._fd_tau_hat.zero_()
            self._fd_tau_hat_normalized.zero_()
            self._fd_soft_limit_fraction.zero_()
            self._fd_hard_limit_fraction.zero_()
            self._fd_normalized_abs_max.zero_()
            return
        fd_tau = self._controller.run_fd_observer(q, qdot, dP, valve_cmd)
        normalizer = self._controller.fd_output_normalizer(like=fd_tau)
        normalized = fd_tau / normalizer
        soft_limit = self._resolve_fd_limit(self.cfg.fd_torque_soft_limit, normalizer, 1.0)
        hard_limit = self._resolve_fd_limit(self.cfg.fd_torque_hard_limit, normalizer, 1.0)
        self._current_fd_simscape_output = fd_tau.detach().clone()
        self._current_tau_fd_adapted = fd_tau.detach().clone()
        self._fd_tau_hat = fd_tau.detach().clone()
        self._fd_tau_hat_normalized = normalized.detach().clone()
        self._fd_soft_limit_fraction = torch.mean((torch.abs(fd_tau) > soft_limit).float(), dim=-1)
        self._fd_hard_limit_fraction = torch.mean((torch.abs(fd_tau) > hard_limit).float(), dim=-1)
        self._fd_normalized_abs_max = torch.max(torch.abs(normalized), dim=-1).values.detach().clone()
        self._current_torque.zero_()
        self._current_tau_applied_raw.zero_()
        self._current_tau_isaac_filtered.zero_()
        self._current_tau_isaac_clamped.zero_()
        self._torque_clamp_fraction.zero_()
        self._torque_adapter_clamp_fraction.zero_()
        self._torque_rate_limited_fraction.zero_()

    def _pre_physics_step(self, actions: torch.Tensor) -> None:

        action_std_by_dim = torch.std(actions, dim=0, unbiased=False)
        self._policy_action_std = torch.mean(action_std_by_dim).detach()
        self._policy_action_std_min = torch.min(action_std_by_dim).detach()
        self._policy_action_std_max = torch.max(action_std_by_dim).detach()
        if self._preflight_direct_target is not None:
            self._current_action.zero_()
            self._action_delta.zero_()
            self._current_qddot_cmd.zero_()
            self._current_valve_cmd.zero_()
            self._zero_torque_state()
            self._zero_sd_observer_state()
            self._zero_sd_feedback_state()
            return

        mechanical_q, mechanical_qdot = self._read_mechanical_state()
        dP = self._read_pressure_delta()
        self._maybe_initialize_learned_control(mechanical_q, mechanical_qdot, dP)
        self._prepare_high_level_action(actions, mechanical_q, mechanical_qdot)
        previous_dP = self._dP.detach().clone()
        previous_fnet = self._fnet.detach().clone()
        control_out = None

        if self.cfg.use_final_causal_cylinder_rom:
            self._pre_physics_step_final_causal_rom(
                q=mechanical_q,
                qdot=mechanical_qdot,
                dP=dP,
                previous_dP=previous_dP,
                previous_fnet=previous_fnet,
            )
            return

        if self.cfg.use_reduced_hydraulic_rom:
            self._pre_physics_step_reduced_hydraulic_rom(
                q=mechanical_q,
                qdot=mechanical_qdot,
                dP=dP,
                previous_dP=previous_dP,
                previous_fnet=previous_fnet,
            )
            return

        low_level_mode = self._low_level_control_mode()
        if low_level_mode == "position_delta" and not self.cfg.use_fd_effort:
            if self.cfg.position_delta_source == "direct_action":
                self._current_valve_cmd = self._current_action.clone()
                self._zero_torque_state()
            elif self.cfg.position_delta_source == "id_valve":
                control_out = self._controller.apply_action_valve_only(
                    mechanical_q,
                    mechanical_qdot,
                    dP,
                    self._current_action,
                    self._prev_valve_cmd,
                    z_arm_hat=self._id_conditioning_latent(),
                    action_to_qddot_scale=self.cfg.action_to_qddot_scale,
                    qddot_cmd_clip=self.cfg.qddot_cmd_clip,
                )
                self._current_qddot_cmd = control_out.qddot_cmd
                self._current_valve_cmd = control_out.valve_cmd
                self._zero_torque_state()
            else:
                raise ValueError(f"Unsupported position_delta_source: {self.cfg.position_delta_source!r}")
        elif low_level_mode == "effort" or self.cfg.use_fd_effort:
            control_out = self._controller.apply_action_full_pipeline(
                mechanical_q,
                mechanical_qdot,
                dP,
                self._fnet,
                self._current_action,
                self._prev_valve_cmd,
                z_arm_hat=self._id_conditioning_latent(),
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
        else:
            raise ValueError(f"Unsupported low_level_control_mode: {low_level_mode!r}")

        run_sd_observer = self.cfg.use_sd_observer or self.cfg.use_sd_feedback
        if run_sd_observer:
            sd_observer = self._controller.run_sd_observer(
                mechanical_q,
                mechanical_qdot,
                dP,
                self._fnet,
                self._current_valve_cmd,
                self._z_arm_hat,
            )
            self._store_sd_observer_output(sd_observer)
        else:
            self._zero_sd_observer_state()

        if self.cfg.use_sd_feedback:
            if low_level_mode != "effort" and not self.cfg.use_fd_effort:
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

    def _pre_physics_step_final_causal_rom(
        self,
        *,
        q: torch.Tensor,
        qdot: torch.Tensor,
        dP: torch.Tensor,
        previous_dP: torch.Tensor,
        previous_fnet: torch.Tensor,
    ) -> None:
        if self._final_causal_rom_wrapper is None:
            raise RuntimeError("use_final_causal_cylinder_rom=True but the final causal ROM wrapper is not initialized.")
        if len(self._mechanical_joint_ids) != 5 or len(self._cylinder_joint_ids) != 4:
            raise RuntimeError("Final causal ROM requires separate 5D mechanical and 4D cylinder state selectors.")

        if self.cfg.rom_use_id_valve_cmd:
            control_out = self._controller.apply_action_valve_only(
                q,
                qdot,
                dP,
                self._current_action,
                self._prev_valve_cmd,
                z_arm_hat=self._id_conditioning_latent(),
                action_to_qddot_scale=self.cfg.action_to_qddot_scale,
                qddot_cmd_clip=self.cfg.qddot_cmd_clip,
            )
            self._current_qddot_cmd = control_out.qddot_cmd
            self._current_valve_cmd = control_out.valve_cmd
        else:
            self._current_valve_cmd = clamp_and_sanitize(self._current_action, self.cfg.max_abs_valve)
        self._run_fd_observer_only(q, qdot, dP, self._current_valve_cmd)

        rom_pressure_source = str(self.cfg.final_causal_rom_pressure_source)
        run_sd_observer = (
            self.cfg.use_sd_observer
            or self.cfg.use_sd_feedback
            or self.cfg.rom_use_sd_feedback
            or rom_pressure_source == "sd_observer"
        )
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
        else:
            self._zero_sd_feedback_state()
            rom_pressure_input = self._final_causal_rom_pressure_input(dP)
            self._dP = rom_pressure_input.detach().clone()
            self._fnet = torch.zeros_like(self._fnet)

        q_cyl = self.robot.data.joint_pos[:, self._command_joint_ids].detach()
        qdot_cyl = self.robot.data.joint_vel[:, self._command_joint_ids].detach()
        features = self._final_causal_rom_feature_row(
            q,
            qdot,
            rom_pressure_input,
            self._current_valve_cmd,
            q_cyl=q_cyl,
            qdot_cyl=qdot_cyl,
        )
        if self._final_causal_rom is not None:
            raw_correction_features = self._final_causal_rom.model.cylinder_features(features)
            normalized_correction_features = (
                raw_correction_features - self._final_causal_rom.model.feature_mean.to(raw_correction_features.device)
            ) / self._final_causal_rom.model.feature_std.to(raw_correction_features.device)
            self._final_rom_input_features_raw = raw_correction_features.detach().clone()
            self._final_rom_input_features_normalized = normalized_correction_features.detach().clone()

        delta_extension_10ms = self._final_causal_rom_wrapper.step(features)
        wrapper_debug = self._final_causal_rom_wrapper.debug_state()
        delta_extension_50ms = wrapper_debug["predicted_delta_extension_50ms"].to(
            device=self.device, dtype=torch.float32
        )
        delta_extension_kinematic_50ms = wrapper_debug["predicted_delta_extension_kinematic_50ms"].to(
            device=self.device, dtype=torch.float32
        )
        delta_extension_correction_50ms = wrapper_debug["predicted_delta_extension_correction_50ms"].to(
            device=self.device, dtype=torch.float32
        )
        update_flag = wrapper_debug["rom_update_flag"].to(device=self.device)
        history_ready = wrapper_debug["history_ready"].to(device=self.device)

        ratio = float(self._final_causal_rom_wrapper.substeps_per_rom_step)
        exact_sum_error = torch.max(torch.abs(delta_extension_10ms * ratio - delta_extension_50ms))
        if float(exact_sum_error.detach().cpu()) > 1.0e-4:
            raise RuntimeError(
                "Final ROM multirate attenuation contract failed: "
                f"10ms_delta * {ratio:g} differs from 50ms_delta by {float(exact_sum_error.detach().cpu()):.6g}."
            )
        sim_limit = abs(float(self.cfg.final_causal_rom_extension_delta_clamp))
        sim_10ms_limit = sim_limit / max(ratio, 1.0)
        sim_clamp_mask = torch.zeros_like(delta_extension_10ms, dtype=torch.bool)
        if sim_limit > 0.0:
            sim_clamp_mask = torch.abs(delta_extension_10ms) > sim_10ms_limit
            delta_extension_10ms = torch.clamp(delta_extension_10ms, -sim_10ms_limit, sim_10ms_limit)

        delta_isaac_10ms = self._simscape_delta_to_isaac_delta(delta_extension_10ms)
        delta_isaac_50ms = self._simscape_delta_to_isaac_delta(delta_extension_50ms)
        isaac_limit = abs(float(self.cfg.final_causal_rom_applied_delta_clamp))
        isaac_clamp_mask = torch.zeros_like(delta_isaac_10ms, dtype=torch.bool)
        if isaac_limit > 0.0:
            isaac_clamp_mask = torch.abs(delta_isaac_10ms) > isaac_limit
            delta_isaac_10ms = torch.clamp(delta_isaac_10ms, -isaac_limit, isaac_limit)

        # ── Per-cylinder velocity rate limiting (Change 5) ────────────────
        rate_limit_mask = torch.zeros_like(delta_isaac_10ms, dtype=torch.bool)
        rom_dt = float(self.cfg.rom_dt)
        rom_velocity_limits = torch.as_tensor(
            self.cfg.rom_xdot_max, dtype=torch.float32, device=self.device
        ).reshape(1, 4)
        max_delta_per_step = rom_velocity_limits * rom_dt
        if rom_dt > 0.0 and bool(torch.any(rom_velocity_limits > 0.0)):
            rate_limit_mask = torch.abs(delta_isaac_10ms) > max_delta_per_step
            delta_isaac_10ms = torch.where(
                delta_isaac_10ms > 0.0,
                torch.min(delta_isaac_10ms, max_delta_per_step),
                torch.max(delta_isaac_10ms, -max_delta_per_step),
            )


        # ── Direct valve feedforward (ROM evidence fix) ─────────────────
        valve_ff_gain = float(getattr(self.cfg, "final_causal_rom_valve_feedforward_gain", 0.0))
        if valve_ff_gain > 0.0:
            position_scale = float(self.cfg.position_scale)
            valve_ff_delta = valve_ff_gain * position_scale * self._current_valve_cmd
            delta_isaac_10ms = delta_isaac_10ms + valve_ff_delta
            self._final_rom_valve_ff_delta = valve_ff_delta.detach().clone()
        else:
            self._final_rom_valve_ff_delta = torch.zeros_like(delta_isaac_10ms)

        if rom_dt > 0.0 and bool(torch.any(rom_velocity_limits > 0.0)):
            post_ff_rate_limit_mask = torch.abs(delta_isaac_10ms) > max_delta_per_step
            delta_isaac_10ms = torch.where(
                delta_isaac_10ms > 0.0,
                torch.min(delta_isaac_10ms, max_delta_per_step),
                torch.max(delta_isaac_10ms, -max_delta_per_step),
            )
            rate_limit_mask = rate_limit_mask | post_ff_rate_limit_mask
        # ── Boundary conditional integration (Change 6) ───────────────────
        target_candidate = self._rom_cylinder_pos_target + delta_isaac_10ms
        target_candidate, invalid_target = self._replace_invalid_with_fallback(
            target_candidate,
            self._rom_cylinder_pos_target,
        )
        lower = self.robot.data.soft_joint_pos_limits[:, self._command_joint_ids, 0]
        upper = self.robot.data.soft_joint_pos_limits[:, self._command_joint_ids, 1]
        at_upper = target_candidate >= upper
        moving_outward_upper = delta_isaac_10ms > 0.0
        at_lower = target_candidate <= lower
        moving_outward_lower = delta_isaac_10ms < 0.0
        boundary_blocked = (at_upper & moving_outward_upper) | (at_lower & moving_outward_lower)
        blocked_outward_increment = torch.where(
            boundary_blocked,
            delta_isaac_10ms,
            torch.zeros_like(delta_isaac_10ms),
        )
        delta_effective = torch.where(
            ~boundary_blocked,
            delta_isaac_10ms,
            torch.zeros_like(delta_isaac_10ms),
        )
        self._rom_cylinder_pos_target = self._rom_cylinder_pos_target + delta_effective
        self._rom_cylinder_pos_target = torch.clamp(self._rom_cylinder_pos_target, lower, upper)
        self._rom_target_clamp_fraction = torch.mean(boundary_blocked.float(), dim=-1)
        self._record_rom_clamp_telemetry(
            boundary_active=at_upper | at_lower,
            outward_request=moving_outward_upper | moving_outward_lower,
            blocked_outward_increment=blocked_outward_increment,
        )
        self._rom_nan_guard_triggered = torch.maximum(self._rom_nan_guard_triggered, invalid_target.float())

        # ── Measured-velocity channel 4 damping (Change 7) ────────────────
        ch4_damping_gain = float(getattr(self.cfg, "channel4_velocity_damping_gain", 0.0))
        ch4_damping_applied = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        if ch4_damping_gain > 0.0 and ch4_damping_gain <= 1.0:
            qdot_cyl_meas = self.robot.data.joint_vel[:, self._command_joint_ids].detach()
            ch4_damping = -ch4_damping_gain * qdot_cyl_meas[:, 3:4] * float(self.cfg.rom_dt)
            self._rom_cylinder_pos_target[:, 3:4] = (
                self._rom_cylinder_pos_target[:, 3:4] + ch4_damping
            )
            self._rom_cylinder_pos_target = torch.clamp(self._rom_cylinder_pos_target, lower, upper)
            ch4_damping_applied = torch.norm(ch4_damping, dim=-1)

        self._final_rom_delta_extension_50ms = delta_extension_50ms.detach().clone()
        self._final_rom_delta_extension_kinematic_50ms = delta_extension_kinematic_50ms.detach().clone()
        self._final_rom_delta_extension_correction_50ms = delta_extension_correction_50ms.detach().clone()
        self._final_rom_delta_extension_10ms = delta_extension_10ms.detach().clone()
        self._final_rom_delta_isaac_50ms = delta_isaac_50ms.detach().clone()
        self._final_rom_delta_isaac_10ms = delta_isaac_10ms.detach().clone()
        self._final_rom_update_flag = update_flag.float().detach().clone()
        self._final_rom_history_ready = history_ready.float().detach().clone()
        self._rom_pressure = rom_pressure_input.detach().clone()
        self._rom_xdot_pred = self._final_rom_extension_velocity.detach().clone()
        self._rom_dx_pred = delta_isaac_10ms.detach().clone()
        self._rom_valve_cmd_delayed = self._current_valve_cmd.detach().clone()
        self._rom_valve_cmd_effective = self._current_valve_cmd.detach().clone()
        self._rom_pressure_clamp_fraction.zero_()
        self._rom_velocity_clamp_fraction = torch.mean(rate_limit_mask.float(), dim=-1)
        self._rom_dx_clamp_fraction = torch.mean((sim_clamp_mask | isaac_clamp_mask).float(), dim=-1)
        self._rom_rate_limit_fraction = torch.mean(rate_limit_mask.float(), dim=-1)
        self._rom_ch4_damping_applied = ch4_damping_applied.detach().clone()
        self._rom_nan_guard_triggered = (~torch.isfinite(delta_isaac_10ms).all(dim=-1)).float()
        self._rom_output_zeroed_fraction = torch.mean((torch.abs(delta_isaac_10ms) <= 1.0e-12).float(), dim=-1)
        actuator_pos_error = self._rom_cylinder_pos_target - q_cyl
        self._rom_actuator_pos_error_mean = torch.mean(torch.abs(actuator_pos_error.detach()), dim=-1)
        self._rom_actuator_pos_error_abs_max = torch.max(torch.abs(actuator_pos_error.detach()), dim=-1).values
        self._rom_cylinder_cumulative_target_motion = (
            self._rom_cylinder_pos_target - self._rom_cylinder_pos_target_at_reset
        ).detach().clone()
        self._rom_cylinder_cumulative_measured_motion = (q_cyl - self._rom_cylinder_measured_at_reset).detach().clone()
        denominator = torch.where(
            torch.abs(self._rom_cylinder_cumulative_target_motion) > 1.0e-9,
            self._rom_cylinder_cumulative_target_motion,
            torch.full_like(self._rom_cylinder_cumulative_target_motion, float("nan")),
        )
        self._rom_cylinder_tracking_ratio = (
            self._rom_cylinder_cumulative_measured_motion / denominator
        ).detach().clone()

    def _final_causal_rom_pressure_input(self, dP: torch.Tensor) -> torch.Tensor:
        source = str(self.cfg.final_causal_rom_pressure_source)
        if source == "zero":
            pressure = torch.zeros_like(dP)
        elif source == "training_mean":
            pressure = self._final_causal_rom_training_mean_dp.reshape(1, 4).expand_as(dP).clone()
        elif source == "environment":
            pressure = dP.detach().clone()
        elif source == "sd_observer":
            pressure = self._sd_dP_next_pred.detach().clone()
        else:
            raise ValueError(f"Unsupported final_causal_rom_pressure_source: {source!r}")
        limit = float(getattr(self.cfg, "final_causal_rom_pressure_clamp", 0.0))
        if limit > 0.0:
            pressure = clamp_and_sanitize(pressure, limit)
        else:
            pressure = clamp_and_sanitize(pressure, None)
        return pressure

    def _mapped_cylinder_state(
        self,
        q_isaac: torch.Tensor,
        qdot_isaac: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mapping = self._cylinder_coordinate_mapping
        if mapping is None:
            zeros = torch.zeros_like(q_isaac)
            flags = torch.zeros_like(q_isaac, dtype=torch.bool)
            return q_isaac, qdot_isaac, zeros, flags
        extension = mapping.isaac_position_to_simscape_extension(q_isaac)
        velocity = mapping.isaac_velocity_to_simscape_velocity(qdot_isaac)
        inverse_error = mapping.inverse_mapping_error(q_isaac)
        range_flags = mapping.isaac_limit_violation(q_isaac)
        return extension.float(), velocity.float(), inverse_error.float(), range_flags

    def _simscape_delta_to_isaac_delta(self, delta_extension: torch.Tensor) -> torch.Tensor:
        mapping = self._cylinder_coordinate_mapping
        if mapping is None:
            return delta_extension
        return mapping.simscape_delta_to_isaac_delta(delta_extension).float()

    def _final_causal_rom_feature_row(
        self,
        q: torch.Tensor,
        qdot: torch.Tensor,
        rom_pressure_input: torch.Tensor,
        valve_cmd: torch.Tensor,
        *,
        q_cyl: torch.Tensor | None = None,
        qdot_cyl: torch.Tensor | None = None,
        env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if q_cyl is None:
            q_cyl = self.robot.data.joint_pos[:, self._command_joint_ids].detach()
        if qdot_cyl is None:
            qdot_cyl = self.robot.data.joint_vel[:, self._command_joint_ids].detach()
        extension, extension_velocity, inverse_error, range_flags = self._mapped_cylinder_state(q_cyl, qdot_cyl)
        if env_ids is None:
            env_ids = torch.arange(extension.shape[0], dtype=torch.long, device=self.device)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)
        self._final_rom_extension[env_ids] = extension.detach().clone()
        self._final_rom_extension_velocity[env_ids] = extension_velocity.detach().clone()
        self._final_rom_inverse_mapping_error[env_ids] = inverse_error.detach().clone()
        self._final_rom_range_violation_fraction[env_ids] = torch.mean(range_flags.float(), dim=-1)
        # ROM feature order: ValveCmd_1..4, dP_1..4, Extension_1..4,
        # ExtensionVelocity_1..4, mechanical JointPos_1..5, mechanical JointVel_1..5.
        features = torch.cat(
            [
                valve_cmd,
                rom_pressure_input,
                extension,
                extension_velocity,
                q,
                qdot,
            ],
            dim=-1,
        )
        if features.shape[-1] != len(CORE_INPUT_FEATURE_NAMES):
            raise RuntimeError(
                f"Final ROM feature dimension mismatch: got {features.shape[-1]}, "
                f"expected {len(CORE_INPUT_FEATURE_NAMES)}."
            )
        if not torch.isfinite(features).all():
            raise FloatingPointError("Final ROM feature vector contains NaN/Inf.")
        return features

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
                z_arm_hat=self._id_conditioning_latent(),
                action_to_qddot_scale=self.cfg.action_to_qddot_scale,
                qddot_cmd_clip=self.cfg.qddot_cmd_clip,
            )
            self._current_qddot_cmd = control_out.qddot_cmd
            self._current_valve_cmd = control_out.valve_cmd
        else:
            self._current_valve_cmd = clamp_and_sanitize(self._current_action, self.cfg.max_abs_valve)
        self._run_fd_observer_only(q, qdot, dP, self._current_valve_cmd)

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
        target_delta = target_candidate - self._rom_cylinder_pos_target
        at_upper = target_candidate >= upper
        moving_outward_upper = target_delta > 0.0
        at_lower = target_candidate <= lower
        moving_outward_lower = target_delta < 0.0
        outward_violation = (at_upper & moving_outward_upper) | (at_lower & moving_outward_lower)
        blocked_outward_increment = torch.where(
            outward_violation,
            target_delta,
            torch.zeros_like(target_delta),
        )
        target_clamped = torch.clamp(target_candidate, lower, upper)
        self._rom_cylinder_pos_target = target_clamped.detach().clone()
        self._rom_target_clamp_fraction = torch.mean(target_clamp_mask.float(), dim=-1)
        self._record_rom_clamp_telemetry(
            boundary_active=at_upper | at_lower,
            outward_request=moving_outward_upper | moving_outward_lower,
            blocked_outward_increment=blocked_outward_increment,
        )
        self._rom_nan_guard_triggered = torch.maximum(
            self._rom_nan_guard_triggered,
            invalid_target.float(),
        )
        actuator_pos_error = self._rom_cylinder_pos_target - x_cyl
        self._rom_actuator_pos_error_mean = torch.mean(torch.abs(actuator_pos_error.detach()), dim=-1)
        self._rom_actuator_pos_error_abs_max = torch.max(torch.abs(actuator_pos_error.detach()), dim=-1).values

    def _apply_action(self) -> None:
        if self._preflight_direct_target is not None:
            self.robot.set_joint_position_target(self._preflight_direct_target, joint_ids=self._command_joint_ids)
            return

        self._apply_rrlab_vehicle_command()

        if self.cfg.use_final_causal_cylinder_rom:
            self.robot.set_joint_position_target(self._rom_cylinder_pos_target, joint_ids=self._command_joint_ids)
            return

        if self.cfg.use_reduced_hydraulic_rom:
            self.robot.set_joint_position_target(self._rom_cylinder_pos_target, joint_ids=self._command_joint_ids)
            return

        low_level_mode = self._low_level_control_mode()
        if low_level_mode == "position_delta" and not self.cfg.use_fd_effort:
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

        if low_level_mode == "effort" or self.cfg.use_fd_effort:
            self.robot.set_joint_effort_target(self._current_torque, joint_ids=self._command_joint_ids)
            return

        raise ValueError(f"Unsupported low_level_control_mode: {low_level_mode!r}")

    def _get_observations(self) -> dict:
        q, qdot = self._read_joint_state()
        cylinder_q, cylinder_qdot = self._read_cylinder_state()
        dP = self._read_pressure_delta()
        self._maybe_initialize_learned_control(q, qdot, dP)
        self._update_aam_latent(q, qdot, dP)
        if self._task_control_mode() == "rl_task_space_tracking":
            self._update_task_space_state()
            obs = build_task_space_rl_obs(
                self._final_pose_target(),
                self._current_ee_pose,
                self._ee_position_error,
                self._ee_orientation_error,
                q,
                qdot,
                cylinder_q,
                cylinder_qdot,
                self._prev_action,
                self._prev_valve_cmd,
                self._z_arm_hat,
                self._fd_tau_hat if bool(getattr(self.cfg, "use_fd_in_policy_observation", True)) else torch.zeros_like(self._fd_tau_hat),
                self._vehicle_commanded_speed_kmh.reshape(-1, 1),
            )
            expected_obs = int(getattr(self.cfg, "task_space_observation_space", TASK_SPACE_OBSERVATION_DIM))
            if tuple(obs.shape) != (self.num_envs, TASK_SPACE_OBSERVATION_DIM):
                raise RuntimeError(
                    "Task-space observation must have shape "
                    f"({self.num_envs}, {TASK_SPACE_OBSERVATION_DIM}), got "
                    f"{tuple(obs.shape)}"
                )
            if not torch.equal(obs[:, 41:65], self._z_arm_hat):
                raise RuntimeError(
                    "AAM latent must occupy task-space observation indices 41:65."
                )
        else:
            obs = build_rl_obs(
                q, qdot, self._q_ref_commanded, self._qdot_ref, self._prev_action, self._prev_valve_cmd, self._z_arm_hat
            )
            expected_obs = int(self.cfg.observation_space)
        if obs.shape[-1] != expected_obs:
            raise RuntimeError(
                f"Observation size mismatch: got {obs.shape[-1]}, expected {expected_obs}"
            )
        self._observation_aam_state_version.copy_(self._aam_latent_state_version)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        q, qdot = self._read_joint_state()
        if self._task_control_mode() == "rl_task_space_tracking":
            return self._get_task_space_rewards(q, qdot)
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
            self._reward_debug["control_mode"] = self._task_control_mode()
            self._reward_debug["low_level_control_mode"] = self._low_level_control_mode()
            self._reward_debug["position_delta_source"] = self.cfg.position_delta_source
            self._reward_debug["use_sd_observer"] = str(self.cfg.use_sd_observer)
            self._reward_debug["use_sd_feedback"] = str(self.cfg.use_sd_feedback)
            self._reward_debug["sd_feedback_beta"] = torch.tensor(float(self.cfg.sd_feedback_beta), device=self.device)
            self._reward_debug["use_sd_feedback_blend"] = str(self.cfg.use_sd_feedback_blend)
            self._reward_debug["use_sd_feedback_rate_limit"] = str(self.cfg.use_sd_feedback_rate_limit)
            self._reward_debug["use_fd_effort"] = str(self.cfg.use_fd_effort)
            self._reward_debug["use_reduced_hydraulic_rom"] = str(self.cfg.use_reduced_hydraulic_rom)
            self._reward_debug["use_final_causal_cylinder_rom"] = str(self.cfg.use_final_causal_cylinder_rom)
            self._reward_debug["rom_enabled"] = str(
                self.cfg.use_reduced_hydraulic_rom or self.cfg.use_final_causal_cylinder_rom
            )
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

    def _get_task_space_rewards(self, q: torch.Tensor, qdot: torch.Tensor) -> torch.Tensor:
        """Compute curriculum-aware task-space tracking reward with diagnostics."""
        self._update_curriculum_state()
        self._update_task_space_state()

        pos_error_norm = torch.linalg.norm(self._ee_position_error, dim=-1)
        orient_error = self._ee_orientation_error.reshape(-1)
        stage = self._curriculum_stage()
        W_pos = float(self.cfg.position_tracking_reward_weight)
        pos_tracking_scale = float(getattr(self.cfg, "position_tracking_exponential_scale", 4.0))

        # ── Bounded position tracking reward [0, 1] ──
        pos_tracking = torch.exp(-pos_tracking_scale * pos_error_norm)
        pos_tracking_weighted = W_pos * pos_tracking

        # ── Dense quaternion-geodesic orientation tracking ──
        orient_enabled = (
            not bool(self.cfg.enable_automatic_curriculum)
            or bool(self.cfg.curriculum_orientation_enabled[stage - 1])
        )
        if orient_enabled:
            orient_tracking_unweighted = final_pose_orientation_reward(
                orient_error,
                float(self.cfg.final_pose_orientation_tolerance_rad),
            )
            orient_tracking = (
                float(self.cfg.rew_task_orient_tracking)
                * orient_tracking_unweighted
            )
        else:
            orient_tracking_unweighted = torch.zeros_like(orient_error)
            orient_tracking = torch.zeros_like(orient_error)

        success_bonus = torch.where(
            self._new_final_pose_success,
            torch.full_like(
                pos_tracking,
                float(self.cfg.final_pose_success_bonus_value),
            ),
            torch.zeros_like(pos_tracking),
        )

        # ── Potential-based progress reward ──
        # positive = potential improves between consecutive non-reset states
        current_potential = normalized_task_space_potential(
            pos_error_norm,
            orient_error,
            position_scale_m=float(self.cfg.taskspace_progress_position_scale_m),
            orientation_scale_rad=float(self.cfg.taskspace_progress_orientation_scale_rad),
            position_weight=float(self.cfg.taskspace_progress_position_weight),
            orientation_weight=float(self.cfg.taskspace_progress_orientation_weight),
        )
        progress_raw = potential_progress_shaping(
            self._prev_task_space_potential,
            current_potential,
            gamma=float(self.cfg.taskspace_progress_gamma),
            weight=1.0,
            reset_mask=~self._task_space_potential_valid,
        )
        progress_clip = float(getattr(self.cfg, "rew_task_progress_clip", 0.2))
        progress_clamped = torch.clamp(progress_raw, -progress_clip, progress_clip)
        progress_reward = float(self.cfg.rew_task_progress) * progress_clamped

        position_gate = torch.exp(
            -torch.square(pos_error_norm / 0.20)
        )
        orientation_gate = torch.exp(
            -torch.square(orient_error / 0.1745329252)
        )
        pose_gate = position_gate * orientation_gate
        ee_linear_speed = self._ee_linear_speed.reshape(-1)
        ee_angular_speed = self._ee_angular_speed.reshape(-1)
        linear_velocity_penalty = (
            -float(self.cfg.rew_final_pose_linear_velocity)
            * W_pos
            * pose_gate
            * torch.tanh(
                torch.square(
                    ee_linear_speed / float(self.cfg.final_pose_linear_shaping_scale_mps)
                )
            )
        )
        angular_velocity_penalty = (
            -float(self.cfg.rew_final_pose_angular_velocity)
            * W_pos
            * pose_gate
            * torch.tanh(
                torch.square(
                    ee_angular_speed / float(self.cfg.final_pose_angular_shaping_scale_radps)
                )
            )
        )
        linear_settling_score = torch.exp(
            -torch.square(
                ee_linear_speed / float(self.cfg.final_pose_linear_settling_scale_mps)
            )
        )
        angular_settling_score = torch.exp(
            -torch.square(
                ee_angular_speed / float(self.cfg.final_pose_angular_settling_scale_radps)
            )
        )
        settling_reward = (
            float(self.cfg.rew_final_pose_settling)
            * W_pos
            * pose_gate
            * 0.5
            * (
                linear_settling_score
                + angular_settling_score
            )
        )
        near_target_linear_settling = near_target_linear_settling_reward(
            pos_error_norm,
            orient_error,
            ee_linear_speed,
            position_tolerance_m=float(self.cfg.final_pose_position_tolerance_m),
            orientation_tolerance_rad=float(self.cfg.final_pose_orientation_tolerance_rad),
            linear_speed_tolerance_mps=float(self.cfg.final_pose_linear_speed_tolerance_mps),
            position_gate_scale=float(self.cfg.near_target_linear_settling_position_gate_scale),
            orientation_gate_scale=float(self.cfg.near_target_linear_settling_orientation_gate_scale),
            speed_margin_mps=float(self.cfg.near_target_linear_settling_speed_margin_mps),
            sharpness=float(getattr(self.cfg, "near_target_linear_settling_sharpness", 1.0)),
            weight=float(self.cfg.rew_near_target_linear_settling) * W_pos,
        )
        near_target_angular_settling = near_target_angular_settling_reward(
            pos_error_norm,
            orient_error,
            ee_angular_speed,
            position_tolerance_m=float(self.cfg.final_pose_position_tolerance_m),
            orientation_tolerance_rad=float(self.cfg.final_pose_orientation_tolerance_rad),
            angular_speed_tolerance_radps=float(self.cfg.final_pose_angular_speed_tolerance_radps),
            position_gate_scale=float(self.cfg.near_target_angular_settling_position_gate_scale),
            orientation_gate_scale=float(self.cfg.near_target_angular_settling_orientation_gate_scale),
            speed_margin_radps=float(self.cfg.near_target_angular_settling_speed_margin_radps),
            weight=float(self.cfg.rew_near_target_angular_settling) * W_pos,
        )
        hold_reward = hold_progress_reward(
            self._final_pose_success_counter,
            required_hold_steps=int(self.cfg.final_pose_success_hold_steps),
            weight=float(self.cfg.rew_final_pose_hold_progress) * W_pos,
            within_final_pose=self._within_final_pose,
        )

        # ── Penalties ──
        action_delta = (
            self._current_action
            - self._previous_action
        )
        action_rate_penalty = (
            -0.01
            * W_pos
            * torch.mean(
                torch.square(action_delta),
                dim=-1,
            )
        )
        action_mag_penalty = (
            -0.0025
            * W_pos
            * torch.mean(
                torch.square(
                    self._current_action
                ),
                dim=-1,
            )
        )
        valve_sat_penalty, valve_sat = valve_saturation_penalty(
            self._current_valve_cmd,
            sat_start=float(self.cfg.valve_saturation_start),
            weight=float(self.cfg.rew_valve_saturation_soft) * W_pos,
        )
        clamp_penalty, clamp_activation_fraction, clamp_blocked_outward_mean = avoidable_clamp_penalty(
            self._rom_boundary_active,
            self._rom_outward_request,
            self._rom_blocked_outward_increment,
            increment_scale=float(self.cfg.rom_clamp_documented_increment_scale),
            weight=float(self.cfg.rew_clamp_outward) * W_pos,
        )
        fd_load_penalty, fd_excess_fraction = fd_load_excess_penalty(
            self._fd_tau_hat_normalized,
            soft_threshold=float(self.cfg.fd_load_soft_threshold),
            weight=float(getattr(self.cfg, "rew_fd_load", 0.0)) * W_pos,
        )
        # ── Unsafe terminal penalty (reconnected; uses current _termination_* masks) ──
        unsafe_termination = (
            self._termination_velocity
            | self._termination_lower_limit
            | self._termination_upper_limit
            | self._termination_cylinder_lower_limit
            | self._termination_cylinder_upper_limit
            | self._termination_joint_nonfinite
            | self._termination_control_nonfinite
        )
        unsafe_termination = (
            unsafe_termination
            & ~self._termination_success
            & ~self.reset_time_outs
        )
        if unsafe_termination.shape != (self.num_envs,):
            raise RuntimeError(
                f"unsafe_termination mask must have shape ({self.num_envs},), "
                f"got {tuple(unsafe_termination.shape)}"
            )
        unsafe_penalty = torch.where(
            unsafe_termination,
            torch.full_like(
                pos_tracking,
                -10.0 * W_pos,
            ),
            torch.zeros_like(pos_tracking),
        )

        # ── Debug telemetry for unsafe penalty ──
        velocity_mask_count = self._termination_velocity.sum()
        unsafe_mask_count = unsafe_termination.sum()
        unsafe_penalty_nonzero_count = (unsafe_penalty != 0.0).sum()
        unsafe_penalty_min = unsafe_penalty.min()

        # ── Persistent cumulative unsafe-penalty counters (monotonic, survive resets) ──
        self._unsafe_penalty_event_count_total += unsafe_termination.sum().long()
        self._unsafe_penalty_value_total += unsafe_penalty.sum()
        # ── Update terminal snapshot buffers for rows that just went unsafe ──
        unsafe_ids = torch.where(unsafe_termination)[0]
        if unsafe_ids.numel() > 0:
            self._terminal_unsafe_penalty[unsafe_ids] = unsafe_penalty[unsafe_ids].detach()
            self._terminal_unsafe_mask[unsafe_ids] = unsafe_termination[unsafe_ids].detach()

        # ── Total reward ──
        reward = (
            pos_tracking_weighted
            + orient_tracking
            + progress_reward
            + action_rate_penalty
            + action_mag_penalty
            + linear_velocity_penalty
            + angular_velocity_penalty
            + settling_reward
            + near_target_linear_settling
            + near_target_angular_settling
            + hold_reward
            + valve_sat_penalty
            + clamp_penalty
            + fd_load_penalty
            + unsafe_penalty
            + success_bonus
        )

        # ── Update progress buffer ──
        self._prev_pos_error_norm = pos_error_norm.detach().clone()
        self._prev_task_space_potential = current_potential.detach().clone()
        self._task_space_potential_valid.fill_(True)

        # ── Accumulate episode sums ──
        self._episode_pos_tracking_sum += pos_tracking_weighted.detach()
        self._episode_orient_tracking_sum += orient_tracking.detach()
        self._episode_progress_reward_sum += progress_reward.detach()
        self._episode_action_rate_penalty_sum += action_rate_penalty.detach()
        self._episode_action_mag_penalty_sum += action_mag_penalty.detach()
        self._episode_valve_sat_penalty_sum += valve_sat_penalty.detach()
        self._episode_clamp_penalty_sum += clamp_penalty.detach()
        self._episode_fd_load_penalty_sum += fd_load_penalty.detach()
        self._episode_unsafe_penalty_sum += unsafe_penalty.detach()
        self._episode_linear_velocity_penalty_sum += linear_velocity_penalty.detach()
        self._episode_angular_velocity_penalty_sum += angular_velocity_penalty.detach()
        self._episode_settling_reward_sum += settling_reward.detach()
        self._episode_near_target_linear_settling_sum += near_target_linear_settling.detach()
        self._episode_near_target_angular_settling_sum += near_target_angular_settling.detach()
        self._episode_hold_progress_reward_sum += hold_reward.detach()
        self._episode_clamp_activation_sum += clamp_activation_fraction.detach()
        self._episode_valve_saturation_fraction_sum += valve_sat.detach()
        self._episode_position_error_sum += pos_error_norm.detach()
        self._episode_success_bonus_sum += success_bonus.detach()

        positive_total = (
            pos_tracking_weighted
            + orient_tracking
            + torch.relu(progress_reward)
            + settling_reward
            + torch.relu(near_target_linear_settling)
            + torch.relu(near_target_angular_settling)
            + hold_reward
            + success_bonus
        )
        negative_total_abs = (
            torch.abs(action_rate_penalty)
            + torch.abs(action_mag_penalty)
            + torch.abs(linear_velocity_penalty)
            + torch.abs(angular_velocity_penalty)
            + torch.relu(-near_target_linear_settling)
            + torch.relu(-near_target_angular_settling)
            + torch.abs(valve_sat_penalty)
            + torch.abs(clamp_penalty)
            + torch.abs(fd_load_penalty)
            + torch.abs(unsafe_penalty)
            + torch.relu(-progress_reward)
        )
        tracking_positive = pos_tracking_weighted + orient_tracking + torch.relu(progress_reward)
        actuator_penalty_abs = (
            torch.abs(valve_sat_penalty)
            + torch.abs(clamp_penalty)
            + torch.abs(fd_load_penalty)
        )
        positive_total_mean = torch.mean(positive_total.detach())
        negative_total_abs_mean = torch.mean(negative_total_abs.detach())
        negative_to_positive_ratio = negative_total_abs_mean / torch.clamp(
            positive_total_mean,
            min=1.0e-6,
        )
        tracking_fraction = torch.mean(
            tracking_positive.detach() / torch.clamp(positive_total.detach(), min=1.0e-6)
        )
        actuator_penalty_fraction = torch.mean(
            actuator_penalty_abs.detach() / torch.clamp(negative_total_abs.detach(), min=1.0e-6)
        )
        settling_fraction = torch.mean(
            (
                settling_reward
                + torch.relu(near_target_linear_settling)
                + torch.relu(near_target_angular_settling)
                + hold_reward
            ).detach() / torch.clamp(positive_total.detach(), min=1.0e-6)
        )

        target_changed = self._target_changed_this_episode()

        # ── Populate reward_debug ──
        if self.cfg.debug_reward_terms:
            self._reward_debug = {
                "reward_mode": "rl_task_space_tracking",
                "curriculum_stage": float(stage),
                "action_authority_multiplier": self._action_authority_multiplier,
                "ee_position_error_norm": torch.mean(pos_error_norm.detach()),
                "ee_orientation_error": torch.mean(orient_error.detach()),
                "ee_position_tracking_reward": torch.mean(pos_tracking_weighted.detach()),
                "ee_orientation_tracking_reward_unweighted": torch.mean(
                    orient_tracking_unweighted.detach()
                ),
                "ee_orientation_tracking_reward": torch.mean(orient_tracking.detach()),
                "final_pose_success_bonus": torch.mean(success_bonus.detach()),
                "progress_reward": torch.mean(progress_reward.detach()),
                "action_rate_penalty": torch.mean(action_rate_penalty.detach()),
                "action_magnitude_penalty": torch.mean(action_mag_penalty.detach()),
                "linear_velocity_penalty": torch.mean(linear_velocity_penalty.detach()),
                "angular_velocity_penalty": torch.mean(angular_velocity_penalty.detach()),
                "settling_reward": torch.mean(settling_reward.detach()),
                "near_target_linear_settling": torch.mean(near_target_linear_settling.detach()),
                "near_target_angular_settling": torch.mean(near_target_angular_settling.detach()),
                "hold_progress_reward": torch.mean(hold_reward.detach()),
                "valve_saturation_penalty": torch.mean(valve_sat_penalty.detach()),
                "rom_clamp_penalty": torch.mean(clamp_penalty.detach()),
                "rom_clamp_activation_fraction": torch.mean(clamp_activation_fraction.detach()),
                "rom_clamp_blocked_outward_mean": torch.mean(clamp_blocked_outward_mean.detach()),
                "fd_load_penalty": torch.mean(fd_load_penalty.detach()),
                "fd_load_excess_fraction": torch.mean(fd_excess_fraction.detach()),
                "fd_soft_limit_fraction": torch.mean(self._fd_soft_limit_fraction.detach()),
                "fd_hard_limit_fraction": torch.mean(self._fd_hard_limit_fraction.detach()),
                "fd_normalized_abs_max": torch.mean(self._fd_normalized_abs_max.detach()),
                "unsafe_penalty": torch.mean(unsafe_penalty.detach()),
            }
            self._add_reference_reset_debug(self._reward_debug, q)
            if self.cfg.log_full_pipeline_debug:
                self._add_full_pipeline_debug(self._reward_debug)
        else:
            self._reward_debug = {}

        # ── Populate extras["log"] for RSL-RL TensorBoard logging ──
        log = {}
        log["curriculum/stage"] = float(stage)
        log["curriculum/action_authority"] = self._action_authority_multiplier
        log["curriculum/episode_length"] = float(self._stage_max_episode_length) * float(self.cfg.sim.dt)
        log["curriculum/automatic_advancement_enabled"] = float(
            bool(self.cfg.enable_automatic_curriculum)
        )
        log["policy/action_std_mean"] = self._policy_action_std
        log["policy/action_std_min"] = self._policy_action_std_min
        log["policy/action_std_max"] = self._policy_action_std_max
        log["policy/action_std_schedule_hold_fraction"] = self._action_std_schedule_hold_fraction
        log["policy/action_std_schedule_blend"] = self._action_std_schedule_blend
        log["policy/action_std_schedule_min_std"] = self._action_std_schedule_min_std
        log["policy/action_std_schedule_max_std"] = self._action_std_schedule_max_std
        log["policy/action_std_schedule_entropy_coef"] = self._action_std_schedule_entropy_coef
        log["policy/action_std_schedule_applied"] = self._action_std_schedule_applied
        log["policy/action_std_schedule_std_mean"] = self._action_std_schedule_std_mean
        log["policy/action_std_schedule_std_max"] = self._action_std_schedule_std_max
        log["reward/pos_tracking_mean"] = torch.mean(pos_tracking_weighted.detach())
        log["reward/orient_tracking_mean"] = torch.mean(orient_tracking.detach())
        log["reward/final_pose_success_bonus_mean"] = torch.mean(success_bonus.detach())
        log["reward/progress_mean"] = torch.mean(progress_reward.detach())
        log["reward/action_rate_penalty_mean"] = torch.mean(action_rate_penalty.detach())
        log["reward/action_mag_penalty_mean"] = torch.mean(action_mag_penalty.detach())
        log["reward/linear_velocity_penalty_mean"] = torch.mean(linear_velocity_penalty.detach())
        log["reward/angular_velocity_penalty_mean"] = torch.mean(angular_velocity_penalty.detach())
        log["reward/settling_reward_mean"] = torch.mean(settling_reward.detach())
        log["reward/near_target_linear_settling_mean"] = torch.mean(near_target_linear_settling.detach())
        log["reward/near_target_angular_settling_mean"] = torch.mean(near_target_angular_settling.detach())
        log["reward/hold_progress_reward_mean"] = torch.mean(hold_reward.detach())
        log["settling_reward_mean"] = torch.mean(settling_reward.detach())
        log["hold_progress_reward_mean"] = torch.mean(hold_reward.detach())
        log["reward/valve_sat_penalty_mean"] = torch.mean(valve_sat_penalty.detach())
        log["reward/valve_saturation_fraction"] = torch.mean(valve_sat.detach())
        log["reward/clamp_penalty_mean"] = torch.mean(clamp_penalty.detach())
        log["reward/clamp_activation_fraction"] = torch.mean(clamp_activation_fraction.detach())
        log["reward/clamp_blocked_outward_mean"] = torch.mean(clamp_blocked_outward_mean.detach())
        log["reward/fd_load_penalty_mean"] = torch.mean(fd_load_penalty.detach())
        log["reward/fd_load_excess_fraction"] = torch.mean(fd_excess_fraction.detach())
        log["reward/unsafe_penalty_mean"] = torch.mean(unsafe_penalty.detach())
        log["reward/pos_error_norm_mean"] = torch.mean(pos_error_norm.detach())
        log["reward/orient_error_mean"] = torch.mean(orient_error.detach())
        log["reward_balance/positive_total_mean"] = positive_total_mean
        log["reward_balance/negative_total_abs_mean"] = negative_total_abs_mean
        log["reward_balance/negative_to_positive_ratio"] = negative_to_positive_ratio
        log["reward_balance/tracking_fraction"] = tracking_fraction
        log["reward_balance/actuator_penalty_fraction"] = actuator_penalty_fraction
        log["reward_balance/settling_fraction"] = settling_fraction
        log["episode/pos_tracking_sum"] = torch.mean(self._episode_pos_tracking_sum)
        log["episode/orient_tracking_sum"] = torch.mean(self._episode_orient_tracking_sum)
        log["episode/progress_reward_sum"] = torch.mean(self._episode_progress_reward_sum)
        log["episode/action_rate_penalty_sum"] = torch.mean(self._episode_action_rate_penalty_sum)
        log["episode/action_mag_penalty_sum"] = torch.mean(self._episode_action_mag_penalty_sum)
        log["episode/valve_sat_penalty_sum"] = torch.mean(self._episode_valve_sat_penalty_sum)
        log["episode/clamp_penalty_sum"] = torch.mean(self._episode_clamp_penalty_sum)
        log["episode/fd_load_penalty_sum"] = torch.mean(self._episode_fd_load_penalty_sum)
        log["debug/velocity_mask_count"] = velocity_mask_count.float()
        log["debug/unsafe_mask_count"] = unsafe_mask_count.float()
        log["debug/unsafe_penalty_nonzero_count"] = unsafe_penalty_nonzero_count.float()
        log["debug/unsafe_penalty_min"] = unsafe_penalty_min
        log["debug/unsafe_penalty_event_count_total"] = self._unsafe_penalty_event_count_total.float()
        log["debug/unsafe_penalty_value_total"] = self._unsafe_penalty_value_total
        log["episode/unsafe_penalty_sum"] = torch.mean(self._episode_unsafe_penalty_sum)
        log["episode/linear_velocity_penalty_sum"] = torch.mean(
            self._episode_linear_velocity_penalty_sum
        )
        log["episode/angular_velocity_penalty_sum"] = torch.mean(
            self._episode_angular_velocity_penalty_sum
        )
        log["episode/settling_reward_sum"] = torch.mean(
            self._episode_settling_reward_sum
        )
        log["episode/near_target_linear_settling_sum"] = torch.mean(
            self._episode_near_target_linear_settling_sum
        )
        log["episode/near_target_angular_settling_sum"] = torch.mean(
            self._episode_near_target_angular_settling_sum
        )
        log["episode/hold_progress_reward_sum"] = torch.mean(
            self._episode_hold_progress_reward_sum
        )
        log["episode/clamp_activation_fraction_sum"] = torch.mean(
            self._episode_clamp_activation_sum
        )
        log["episode/valve_saturation_fraction_sum"] = torch.mean(
            self._episode_valve_saturation_fraction_sum
        )
        log["episode/final_pose_success_bonus_sum"] = torch.mean(
            self._episode_success_bonus_sum
        )
        log["final_pose/position_error_mean"] = torch.mean(pos_error_norm.detach())
        log["final_pose/position_error_max"] = torch.max(pos_error_norm.detach())
        log["final_pose/orientation_error_mean_rad"] = torch.mean(orient_error.detach())
        log["final_pose/orientation_error_mean_deg"] = torch.rad2deg(
            torch.mean(orient_error.detach())
        )
        log["final_pose/orientation_error_max_rad"] = torch.max(orient_error.detach())
        log["final_pose/linear_speed_mean"] = torch.mean(self._ee_linear_speed.detach())
        log["final_pose/angular_speed_mean"] = torch.mean(self._ee_angular_speed.detach())
        strict_position = pos_error_norm <= float(self.cfg.final_pose_position_tolerance_m)
        strict_orientation = orient_error <= float(self.cfg.final_pose_orientation_tolerance_rad)
        strict_linear_speed = self._ee_linear_speed <= float(
            self.cfg.final_pose_linear_speed_tolerance_mps
        )
        strict_angular_speed = self._ee_angular_speed <= float(
            self.cfg.final_pose_angular_speed_tolerance_radps
        )
        strict_all_except_hold = (
            strict_position
            & strict_orientation
            & strict_linear_speed
            & strict_angular_speed
        )
        near_position = strict_position
        near_orientation = strict_orientation
        near_pose = near_position & near_orientation
        low_speed = strict_linear_speed & strict_angular_speed
        near_pose_low_speed = near_pose & low_speed
        near_pose_high_speed = near_pose & ~low_speed
        self._final_pose_entered_tolerance_count_total += near_pose.float().sum()
        self._final_pose_settling_hold_max = torch.maximum(
            self._final_pose_settling_hold_max,
            self._final_pose_success_counter,
        )
        self._final_pose_settling_hold_max_total = torch.maximum(
            self._final_pose_settling_hold_max_total,
            torch.max(self._final_pose_settling_hold_max.float()),
        )
        log["final_pose/near_position_fraction"] = torch.mean(near_position.float())
        log["final_pose/near_orientation_fraction"] = torch.mean(near_orientation.float())
        log["final_pose/near_pose_fraction"] = torch.mean(near_pose.float())
        log["final_pose/near_pose_high_speed_fraction"] = torch.mean(near_pose_high_speed.float())
        log["final_pose/near_pose_low_speed_fraction"] = torch.mean(near_pose_low_speed.float())
        log["final_pose/strict_position_fraction"] = torch.mean(strict_position.float())
        log["final_pose/strict_orientation_fraction"] = torch.mean(strict_orientation.float())
        log["final_pose/strict_linear_speed_fraction"] = torch.mean(strict_linear_speed.float())
        log["final_pose/strict_angular_speed_fraction"] = torch.mean(strict_angular_speed.float())
        log["final_pose/strict_all_except_hold_fraction"] = torch.mean(strict_all_except_hold.float())
        log["final_pose/entered_tolerance_count_total"] = self._final_pose_entered_tolerance_count_total
        log["final_pose/settling_hold_max_mean"] = torch.mean(self._final_pose_settling_hold_max.float())
        log["final_pose/settling_hold_max_total"] = self._final_pose_settling_hold_max_total
        log["final_pose/hold_break_reason_position"] = self._final_pose_hold_break_reason_position
        log["final_pose/hold_break_reason_orientation"] = self._final_pose_hold_break_reason_orientation
        log["final_pose/hold_break_reason_linear_speed"] = self._final_pose_hold_break_reason_linear_speed
        log["final_pose/hold_break_reason_angular_speed"] = self._final_pose_hold_break_reason_angular_speed
        log["final_pose/hold_break_reason_target_changed"] = self._final_pose_hold_break_reason_target_changed
        log["final_pose/hold_break_reason_timeout"] = self._final_pose_hold_break_reason_timeout
        log["final_pose/hold_break_reason_success_reached"] = self._final_pose_hold_break_reason_success_reached
        log["final_pose/hold_break_reason_unknown"] = self._final_pose_hold_break_reason_unknown
        log["final_pose/max_hold_position_error_at_break"] = (
            self._final_pose_max_hold_position_error_at_break
        )
        log["final_pose/max_hold_orientation_error_at_break"] = (
            self._final_pose_max_hold_orientation_error_at_break
        )
        log["final_pose/max_hold_linear_speed_at_break"] = (
            self._final_pose_max_hold_linear_speed_at_break
        )
        linear_speed_margin_at_break = (
            float(self.cfg.final_pose_linear_speed_tolerance_mps)
            - self._final_pose_max_hold_linear_speed_at_break
        )
        log["final_pose/linear_speed_margin_at_break"] = linear_speed_margin_at_break
        log["final_pose/linear_speed_break_excess"] = torch.relu(
            -linear_speed_margin_at_break
        )
        log["final_pose/max_hold_angular_speed_at_break"] = (
            self._final_pose_max_hold_angular_speed_at_break
        )
        angular_speed_margin_at_break = (
            float(self.cfg.final_pose_angular_speed_tolerance_radps)
            - self._final_pose_max_hold_angular_speed_at_break
        )
        log["final_pose/angular_speed_margin_at_break"] = angular_speed_margin_at_break
        log["final_pose/angular_speed_break_excess"] = torch.relu(
            -angular_speed_margin_at_break
        )
        log["final_pose/max_hold_steps_before_break"] = (
            self._final_pose_max_hold_steps_before_break_total
        )
        log["final_pose/steps_remaining_after_max_hold"] = (
            self._final_pose_steps_remaining_after_max_hold_total
        )
        log["final_pose/first_step_entered_strict_settled"] = (
            self._final_pose_first_step_entered_strict_settled
        )
        log["final_pose/last_step_entered_strict_settled"] = (
            self._final_pose_last_step_entered_strict_settled
        )
        log["final_pose/success_count_total"] = torch.sum(self._termination_success_count)
        log["final_pose/success_events_this_iteration"] = self._new_final_pose_success.sum().float()
        log["final_pose/within_tolerance_fraction"] = torch.mean(
            self._within_final_pose.float()
        )
        log["final_pose/success_fraction"] = torch.mean(
            self._termination_success.float()
        )
        log["final_pose/success_hold_counter_mean"] = torch.mean(
            self._final_pose_success_counter.float()
        )
        log["final_pose/target_changed_fraction"] = torch.mean(target_changed.float())
        log["final_pose/target_finite_fraction"] = torch.mean(
            torch.isfinite(self._final_pose_target()).all(dim=-1).float()
        )
        aam_history = self._controller.aam_history
        z_arm_hat = self._z_arm_hat
        log["adaptation/z_arm_hat_mean"] = torch.mean(z_arm_hat)
        log["adaptation/z_arm_hat_std"] = torch.std(
            z_arm_hat, unbiased=False
        )
        log["adaptation/z_arm_hat_abs_max"] = torch.max(torch.abs(z_arm_hat))
        log["adaptation/z_arm_hat_finite_fraction"] = torch.mean(
            torch.isfinite(z_arm_hat).float()
        )
        log["adaptation/history_abs_mean"] = torch.mean(torch.abs(aam_history))
        log["adaptation/history_valve_abs_mean"] = torch.mean(
            torch.abs(aam_history[:, :, 14:18])
        )
        log["adaptation/aam_active"] = torch.ones(
            (), dtype=torch.float32, device=self.device
        )
        log["adaptation/aam_frozen"] = torch.as_tensor(
            float(self._controller.aam_frozen),
            dtype=torch.float32,
            device=self.device,
        )
        self._populate_extras_log(log)

        self._prev_q_error_mean = self._weighted_q_error_mean(q, self._q_ref_commanded).detach()
        self._prev_action = self._current_action.detach().clone()
        self._prev_valve_cmd = self._current_valve_cmd.detach().clone()
        return reward

    def _populate_extras_log(self, log: dict[str, object]) -> None:
        """Write diagnostics dict into self.extras["log"] for RSL-RL TensorBoard."""
        try:
            if not hasattr(self, "extras") or not isinstance(self.extras, dict):
                return
            if "log" not in self.extras:
                self.extras["log"] = {}
            self.extras["log"].update(log)
        except Exception:
            pass  # non-critical; best-effort logging

    def _reward_weights_for_control_mode(self) -> JointReachRewardWeights:
        if self._low_level_control_mode() == "position_delta" and not self.cfg.use_fd_effort:
            return replace(self._reward_weights, torque_effort=0.0)
        return self._reward_weights

    # ── Curriculum helpers ─────────────────────────────────────────────────────────

    def _curriculum_stage(self) -> int:
        """Return current curriculum stage (1–4).

        This is the EXPLICITLY stored stage index, NOT derived from
        common_step_counter.  Advancement requires performance gates to be met
        (see _evaluate_stage_advancement).
        """
        return int(self._curriculum_stage_index)

    def _evaluate_stage_advancement(self) -> None:
        """Check performance gates and advance stage by at most 1 level."""
        goal_curriculum_enabled = bool(getattr(self.cfg, "enable_goal_difficulty_curriculum", False))
        if not (bool(self.cfg.enable_automatic_curriculum) or goal_curriculum_enabled):
            return
        if self._curriculum_stage_index >= 4:
            return  # already at maximum stage

        n = max(self.num_envs, 1)
        episodes_per_env = self._curriculum_stage_episode_count / n
        mean_position_error = (
            self._curriculum_stage_position_error_sum
            / max(self._curriculum_stage_episode_count, 1.0)
        )
        timeout_rate = (
            self._curriculum_stage_timeout_count
            / max(self._curriculum_stage_episode_count, 1.0)
        )
        velocity_rate = (
            self._curriculum_stage_velocity_count
            / max(self._curriculum_stage_episode_count, 1.0)
        )
        other_unsafe = (
            self._curriculum_stage_other_unsafe_count
            / max(self._curriculum_stage_episode_count, 1.0)
        )

        if goal_curriculum_enabled:
            completed = max(self._curriculum_stage_episode_count, 1.0)
            success_rate = self._curriculum_stage_success_count / completed
            near_pose_low_speed_rate = self._curriculum_stage_near_pose_low_speed_count / completed
            mean_max_hold_steps = self._curriculum_stage_max_hold_steps_sum / completed
            stage_ready = (
                self._curriculum_stage_episode_count >= float(self.cfg.goal_curriculum_min_completed_episodes)
                and success_rate >= float(self.cfg.goal_curriculum_advance_success_rate)
                and near_pose_low_speed_rate >= float(self.cfg.goal_curriculum_advance_near_pose_low_speed_rate)
                and mean_max_hold_steps >= float(self.cfg.goal_curriculum_advance_mean_max_hold_steps)
            )
            if stage_ready:
                self._curriculum_stage_ready_windows += 1
            else:
                self._curriculum_stage_ready_windows = 0
            stage_ready = self._curriculum_stage_ready_windows >= max(
                int(getattr(self.cfg, "goal_curriculum_hysteresis_windows", 2)),
                1,
            )
        else:
            success_rate = self._curriculum_stage_success_count / max(self._curriculum_stage_episode_count, 1.0)
            stage_ready = (
                episodes_per_env >= float(self.cfg.curriculum_min_completed_episodes_per_env)
                and mean_position_error <= float(self.cfg.curriculum_max_mean_position_error)
                and velocity_rate <= float(self.cfg.curriculum_max_velocity_termination_rate)
                and other_unsafe <= float(self.cfg.curriculum_max_other_unsafe_rate)
                and timeout_rate >= float(self.cfg.curriculum_min_timeout_rate)
                and success_rate >= float(getattr(self.cfg, "curriculum_advance_min_success_rate", 0.0))
            )
        # Hysteresis: require consecutive ready windows before advancing.
        if not goal_curriculum_enabled:
            if stage_ready:
                self._curriculum_stage_ready_windows += 1
            else:
                self._curriculum_stage_ready_windows = 0
            stage_ready = self._curriculum_stage_ready_windows >= max(
                int(getattr(self.cfg, "goal_curriculum_hysteresis_windows", 2)),
                1,
            )
        if stage_ready:
            self._curriculum_stage_index += 1
            self._curriculum_stage_advance_count += 1
            # Reset stage-level aggregate counters for the new stage.
            self._curriculum_stage_episode_count = 0.0
            self._curriculum_stage_timeout_count = 0.0
            self._curriculum_stage_velocity_count = 0.0
            self._curriculum_stage_other_unsafe_count = 0.0
            self._curriculum_stage_position_error_sum = 0.0
            self._curriculum_stage_success_count = 0.0
            self._curriculum_stage_near_pose_low_speed_count = 0.0
            self._curriculum_stage_max_hold_steps_sum = 0.0
            self._curriculum_stage_ready_windows = 0

    def _update_curriculum_state(self) -> None:
        """Refresh curriculum-derived state (stage, action authority, episode length)."""
        if not (
            bool(self.cfg.enable_automatic_curriculum)
            or bool(getattr(self.cfg, "enable_goal_difficulty_curriculum", False))
        ):
            self._curriculum_stage_index = 1
        stage = self._curriculum_stage()
        authority = self.cfg.curriculum_action_authority
        self._action_authority_multiplier = float(authority[stage - 1])
        stage_ep_len_s = float(self.cfg.curriculum_episode_length_s[stage - 1])
        self._stage_max_episode_length = int(stage_ep_len_s / self.cfg.sim.dt)

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
        if bool(getattr(self.cfg, "enable_goal_difficulty_curriculum", False)):
            scales = list(getattr(self.cfg, "goal_curriculum_stage_scales", [1.0]))
            stage = min(max(self._curriculum_stage(), 1), len(scales))
            return float(scales[stage - 1]), float(stage) / float(max(len(scales), 1))
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
        debug["qdot_ref"] = torch.mean(self._qdot_ref.detach(), dim=0)
        debug["qddot_ref"] = torch.mean(self._qddot_ref.detach(), dim=0)
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
        debug["vehicle_command_mode"] = str(self.cfg.vehicle_command_mode)
        debug["enable_scripted_vehicle_motion"] = str(getattr(self.cfg, "enable_scripted_vehicle_motion", True))
        debug["vehicle_speed_kmh"] = torch.mean(self._vehicle_commanded_speed_kmh.detach())
        debug["vehicle_wheel_velocity_target"] = torch.mean(self._vehicle_wheel_velocity_target.detach(), dim=0)
        debug["vehicle_speed_range_kmh"] = str(self.cfg.vehicle_speed_range_kmh)
        debug["vehicle_wheel_joint_ids"] = str(self._vehicle_wheel_joint_ids)
        debug["fd_tau_hat"] = torch.mean(self._fd_tau_hat.detach(), dim=0)
        debug["fd_tau_hat_normalized"] = torch.mean(self._fd_tau_hat_normalized.detach(), dim=0)
        debug["fd_soft_limit_fraction"] = torch.mean(self._fd_soft_limit_fraction.detach())
        debug["fd_hard_limit_fraction"] = torch.mean(self._fd_hard_limit_fraction.detach())
        debug["fd_normalized_abs_max"] = torch.mean(self._fd_normalized_abs_max.detach())
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
        debug["reset_count"] = torch.mean(self._reset_count.detach())
        debug["reset_reason_code"] = torch.mean(self._last_reset_reason_code.detach())
        debug["reset_buf_mean"] = torch.mean(reset_buf)
        debug["done_buf_mean"] = torch.mean(self._last_done_debug.detach().float())
        debug["terminated_frac"] = torch.mean(self._last_terminated_debug.detach().float())
        debug["time_out_frac"] = torch.mean(self._last_time_out_debug.detach().float())
        debug["timeout"] = torch.mean(self._last_time_out_debug.detach().float())
        debug["unsafe_termination"] = torch.mean(self._last_terminated_debug.detach().float())
        debug["target_reached_termination"] = torch.mean(self._reset_due_to_target_reached.detach())
        debug["target_changed_flag"] = torch.mean(self._target_changed_flag.detach())
        debug["target_change_step"] = torch.mean(self._target_change_step.detach())
        debug["target_hold_steps"] = torch.mean(target_hold_steps)
        debug["last_reset_step_global"] = torch.mean(self._last_reset_step_global.detach())
        debug["q_ref_sampled_on_reset"] = torch.mean(self._q_ref_sampled_on_reset.detach(), dim=0)
        debug["reset_due_to_timeout"] = torch.mean(self._reset_due_to_timeout.detach())
        debug["reset_due_to_termination"] = torch.mean(self._reset_due_to_termination.detach())
        debug["reset_due_to_manual_target_change"] = torch.mean(self._reset_due_to_manual_target_change.detach())
        debug["reset_due_to_target_reached"] = torch.mean(self._reset_due_to_target_reached.detach())

    def _zero_torque_state(self) -> None:
        self._current_torque.zero_()
        self._current_fd_simscape_output.zero_()
        self._current_tau_fd_adapted.zero_()
        self._fd_tau_hat.zero_()
        self._fd_tau_hat_normalized.zero_()
        self._fd_soft_limit_fraction.zero_()
        self._fd_hard_limit_fraction.zero_()
        self._fd_normalized_abs_max.zero_()
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

    def _mechanical_constraint_family_ids(self) -> dict[str, list[int]]:
        resolution = resolve_mechanical_constraint_family_ids(self._robot_joint_names())
        for family, missing in resolution.missing_optional.items():
            print(
                "[WARN] Kevin mechanical preflight optional constraint joints missing "
                f"for {family}: {missing}",
                flush=True,
            )
        print("[INFO] Kevin mechanical preflight resolved family joint IDs:", flush=True)
        for family, ids in resolution.family_ids.items():
            names = resolution.family_names[family]
            print(f"[INFO]   {family}: {list(zip(names, ids))}", flush=True)
        return resolution.family_ids

    def _mechanical_family_positions(self, families: dict[str, list[int]]) -> dict[str, torch.Tensor]:
        positions: dict[str, torch.Tensor] = {}
        for family, ids in families.items():
            if ids:
                positions[family] = self.robot.data.joint_pos[:, ids].detach().clone()
            else:
                positions[family] = torch.zeros((self.num_envs, 0), dtype=torch.float32, device=self.device)
        return positions

    def _mechanical_family_motion(
        self,
        before: dict[str, torch.Tensor],
        after: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        motion: dict[str, float] = {}
        for family, q0 in before.items():
            q1 = after[family]
            if q0.numel() == 0 or q1.numel() == 0:
                motion[family] = float("nan")
            else:
                motion[family] = float(torch.max(torch.abs(q1[0] - q0[0])).detach().cpu())
        return motion

    @staticmethod
    def _root_roll_pitch_from_quat_wxyz(quat_wxyz: torch.Tensor) -> torch.Tensor:
        w = quat_wxyz[:, 0]
        x = quat_wxyz[:, 1]
        y = quat_wxyz[:, 2]
        z = quat_wxyz[:, 3]
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp)
        sinp = torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0)
        pitch = torch.asin(sinp)
        return torch.stack((roll, pitch), dim=-1)

    def _wheel_suspension_state(self) -> dict[str, dict[str, float]]:
        names = self._robot_joint_names()
        state: dict[str, dict[str, float]] = {}
        for joint_id, name in enumerate(names):
            lowered = name.lower()
            if "wheel" not in lowered and "suspension" not in lowered:
                continue
            state[name] = {
                "pos": float(self.robot.data.joint_pos[0, joint_id].detach().cpu()),
                "vel": float(self.robot.data.joint_vel[0, joint_id].detach().cpu()),
            }
        return state

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
        controller_debug = self._controller.debug_state()
        debug["qddot_cmd_mean"] = torch.mean(torch.abs(self._current_qddot_cmd.detach()))
        debug["qddot_cmd_abs_per_joint"] = torch.mean(torch.abs(self._current_qddot_cmd.detach()), dim=0)
        debug["id_input_raw"] = torch.mean(controller_debug["id_input_raw"].detach(), dim=0)
        debug["id_input_normalized"] = torch.mean(controller_debug["id_input_normalized"].detach(), dim=0)
        debug["id_input_normalized_min"] = torch.mean(controller_debug["id_input_normalized_min"].detach())
        debug["id_input_normalized_max"] = torch.mean(controller_debug["id_input_normalized_max"].detach())
        debug["id_input_normalized_abs_max"] = torch.mean(controller_debug["id_input_normalized_abs_max"].detach())
        debug["qddot_ref"] = torch.mean(self._qddot_ref.detach(), dim=0)
        debug["scripted_qddot_feedback"] = torch.mean(self._scripted_qddot_feedback.detach(), dim=0)
        debug["scripted_qddot_feedforward"] = torch.mean(self._scripted_qddot_feedforward.detach(), dim=0)
        debug["scripted_qddot_raw"] = torch.mean(self._scripted_qddot_raw.detach(), dim=0)
        debug["scripted_action_preclip"] = torch.mean(self._scripted_action_preclip.detach(), dim=0)
        debug["scripted_control_ramp"] = torch.mean(self._scripted_control_ramp.detach())
        debug["vehicle_speed_kmh"] = torch.mean(self._vehicle_commanded_speed_kmh.detach())
        debug["vehicle_wheel_velocity_target"] = torch.mean(self._vehicle_wheel_velocity_target.detach(), dim=0)
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
        debug["fd_tau_hat"] = torch.mean(self._fd_tau_hat.detach(), dim=0)
        debug["fd_tau_hat_normalized"] = torch.mean(self._fd_tau_hat_normalized.detach(), dim=0)
        debug["fd_soft_limit_fraction"] = torch.mean(self._fd_soft_limit_fraction.detach())
        debug["fd_hard_limit_fraction"] = torch.mean(self._fd_hard_limit_fraction.detach())
        debug["fd_normalized_abs_max"] = torch.mean(self._fd_normalized_abs_max.detach())
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
        qddot_scale = self._controller.action_scale_vector(
            self.cfg.action_to_qddot_scale,
            like=self._current_action,
        ).reshape(-1)
        debug["qddot_cmd_scale"] = torch.mean(qddot_scale.detach())
        debug["qddot_cmd_scale_per_joint"] = qddot_scale.detach().clone()
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
        lower = self.robot.data.soft_joint_pos_limits[:, self._command_joint_ids, 0]
        upper = self.robot.data.soft_joint_pos_limits[:, self._command_joint_ids, 1]
        usable_range = torch.clamp(upper - lower, min=1.0e-9)
        target_motion_pct = 100.0 * torch.abs(self._rom_cylinder_cumulative_target_motion.detach()) / usable_range
        measured_motion_pct = 100.0 * torch.abs(self._rom_cylinder_cumulative_measured_motion.detach()) / usable_range
        debug["rom_enabled"] = str(self.cfg.use_reduced_hydraulic_rom or self.cfg.use_final_causal_cylinder_rom)
        debug["rom_type"] = (
            "final_causal_linear_correction"
            if self.cfg.use_final_causal_cylinder_rom
            else ("reduced_hydraulic_placeholder" if self.cfg.use_reduced_hydraulic_rom else "none")
        )
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
        debug["rom_rate_limit_fraction"] = torch.mean(self._rom_rate_limit_fraction.detach())
        debug["rom_ch4_damping_applied_mean"] = torch.mean(self._rom_ch4_damping_applied.detach())
        debug["rom_dx_clamp_fraction"] = torch.mean(self._rom_dx_clamp_fraction.detach())
        debug["rom_nan_guard_triggered"] = torch.mean(self._rom_nan_guard_triggered.detach())
        debug["rom_output_zeroed_fraction"] = torch.mean(self._rom_output_zeroed_fraction.detach())
        debug["rom_actuator_pos_error_mean"] = torch.mean(self._rom_actuator_pos_error_mean.detach())
        debug["rom_actuator_pos_error_abs_max"] = torch.max(self._rom_actuator_pos_error_abs_max.detach())
        debug["rom_cylinder_pos_target"] = torch.mean(target, dim=0)
        debug["rom_cylinder_actual_pos"] = torch.mean(x_cyl, dim=0)
        debug["rom_cylinder_pos_error"] = torch.mean(target - x_cyl, dim=0)
        debug["rom_cylinder_cumulative_target_motion"] = torch.mean(
            self._rom_cylinder_cumulative_target_motion.detach(), dim=0
        )
        debug["rom_cylinder_cumulative_measured_motion"] = torch.mean(
            self._rom_cylinder_cumulative_measured_motion.detach(), dim=0
        )
        debug["rom_cylinder_target_motion_pct_range"] = torch.mean(target_motion_pct.detach(), dim=0)
        debug["rom_cylinder_measured_motion_pct_range"] = torch.mean(measured_motion_pct.detach(), dim=0)
        debug["rom_cylinder_tracking_ratio"] = torch.nan_to_num(
            torch.mean(self._rom_cylinder_tracking_ratio.detach(), dim=0),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        debug["rom_xdot_pred"] = torch.mean(self._rom_xdot_pred.detach(), dim=0)
        debug["rom_dx_pred"] = torch.mean(self._rom_dx_pred.detach(), dim=0)
        debug["rom_pressure"] = torch.mean(self._rom_pressure.detach(), dim=0)
        if self.cfg.use_final_causal_cylinder_rom:
            debug["final_rom_command_coupling"] = "ValveCmd-conditioned deployed correction"
            debug["final_rom_update_flag"] = torch.mean(self._final_rom_update_flag.detach())
            debug["final_rom_history_ready"] = torch.mean(self._final_rom_history_ready.detach())
            debug["final_rom_delta_extension_50ms"] = torch.mean(self._final_rom_delta_extension_50ms.detach(), dim=0)
            debug["final_rom_delta_extension_kinematic_50ms"] = torch.mean(
                self._final_rom_delta_extension_kinematic_50ms.detach(), dim=0
            )
            debug["final_rom_delta_extension_correction_50ms"] = torch.mean(
                self._final_rom_delta_extension_correction_50ms.detach(), dim=0
            )
            debug["final_rom_delta_extension_10ms"] = torch.mean(self._final_rom_delta_extension_10ms.detach(), dim=0)
            debug["final_rom_delta_isaac_50ms"] = torch.mean(self._final_rom_delta_isaac_50ms.detach(), dim=0)
            debug["final_rom_delta_isaac_10ms"] = torch.mean(self._final_rom_delta_isaac_10ms.detach(), dim=0)
            debug["final_rom_extension"] = torch.mean(self._final_rom_extension.detach(), dim=0)
            debug["final_rom_extension_velocity"] = torch.mean(self._final_rom_extension_velocity.detach(), dim=0)
            debug["final_rom_input_features_raw"] = torch.mean(self._final_rom_input_features_raw.detach(), dim=0)
            debug["final_rom_input_normalized_min"] = torch.min(self._final_rom_input_features_normalized.detach())
            debug["final_rom_input_normalized_max"] = torch.max(self._final_rom_input_features_normalized.detach())
            debug["final_rom_inverse_mapping_error_abs_max"] = torch.max(
                torch.abs(self._final_rom_inverse_mapping_error.detach())
            )
            debug["final_rom_range_violation_fraction"] = torch.mean(self._final_rom_range_violation_fraction.detach())
            if self._cylinder_coordinate_mapping is not None:
                metadata = mapping_metadata_for_logging(self._cylinder_coordinate_mapping, x_cyl)
                debug["final_rom_mapping_mode"] = str(metadata["mapping_mode"])
                debug["final_rom_mapping_validated"] = str(metadata["mapping_validated"])
                debug["final_rom_mapping_reference_confidence"] = str(metadata["nominal_reference_confidence"])
                debug["final_rom_mapping_scale"] = torch.as_tensor(
                    metadata["scale"], dtype=torch.float32, device=self.device
                )
                debug["final_rom_mapping_sign"] = torch.as_tensor(
                    metadata["sign"], dtype=torch.float32, device=self.device
                )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        mechanical_q, mechanical_qdot = self._read_mechanical_state()
        q = mechanical_q
        qdot = mechanical_qdot
        lower_limits = self.robot.data.soft_joint_pos_limits[:, self._state_joint_ids, 0]
        upper_limits = self.robot.data.soft_joint_pos_limits[:, self._state_joint_ids, 1]
        cylinder_extension, cylinder_extension_velocity = self._read_cylinder_state()
        cylinder_lower_limits = self.robot.data.soft_joint_pos_limits[:, self._cylinder_joint_ids, 0]
        cylinder_upper_limits = self.robot.data.soft_joint_pos_limits[:, self._cylinder_joint_ids, 1]

        # ── Per-reason termination diagnostics ──
        joint_nonfinite = ~(torch.isfinite(mechanical_q).all(dim=-1) & torch.isfinite(mechanical_qdot).all(dim=-1))
        lower_limit = torch.any(mechanical_q < lower_limits - self.cfg.joint_limit_margin, dim=-1)
        upper_limit = torch.any(mechanical_q > upper_limits + self.cfg.joint_limit_margin, dim=-1)
        cylinder_lower_limit = torch.any(cylinder_extension < cylinder_lower_limits - self.cfg.joint_limit_margin, dim=-1)
        cylinder_upper_limit = torch.any(cylinder_extension > cylinder_upper_limits + self.cfg.joint_limit_margin, dim=-1)
        cylinder_state_invalid = self._update_hydraulic_velocity_audit(
            cylinder_extension.detach(),
            cylinder_extension_velocity.detach(),
        )
        if self._validated_mechanical_velocity_limits_enabled():
            mechanical_hard_limits = self._mechanical_velocity_limit_tensor("mechanical_hard_velocity_limits")
            mechanical_hard_exceedance = torch.abs(mechanical_qdot) > mechanical_hard_limits
        else:
            mechanical_hard_limits = torch.zeros((1, 5), dtype=torch.float32, device=self.device)
            mechanical_hard_exceedance = torch.zeros_like(mechanical_qdot, dtype=torch.bool)
        velocity_violation = torch.any(mechanical_hard_exceedance, dim=-1)

        safety_terminated = (
            joint_nonfinite
            | lower_limit
            | upper_limit
            | cylinder_lower_limit
            | cylinder_upper_limit
            | cylinder_state_invalid
            | velocity_violation
        )

        finite_control = (
            torch.isfinite(self._current_torque).all(dim=-1)
            & torch.isfinite(self._current_fd_simscape_output).all(dim=-1)
            & torch.isfinite(self._current_tau_fd_adapted).all(dim=-1)
            & torch.isfinite(self._fd_tau_hat).all(dim=-1)
            & torch.isfinite(self._fd_tau_hat_normalized).all(dim=-1)
            & torch.isfinite(self._fd_soft_limit_fraction)
            & torch.isfinite(self._fd_hard_limit_fraction)
            & torch.isfinite(self._fd_normalized_abs_max)
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
            & torch.isfinite(self._final_rom_delta_extension_50ms).all(dim=-1)
            & torch.isfinite(self._final_rom_delta_extension_kinematic_50ms).all(dim=-1)
            & torch.isfinite(self._final_rom_delta_extension_correction_50ms).all(dim=-1)
            & torch.isfinite(self._final_rom_delta_extension_10ms).all(dim=-1)
            & torch.isfinite(self._final_rom_delta_isaac_50ms).all(dim=-1)
            & torch.isfinite(self._final_rom_delta_isaac_10ms).all(dim=-1)
            & torch.isfinite(self._final_rom_extension).all(dim=-1)
            & torch.isfinite(self._final_rom_extension_velocity).all(dim=-1)
        )
        if self._task_control_mode() == "scripted_reference_demo" and bool(self.cfg.scripted_demo_disable_safety_termination):
            safety_terminated = torch.zeros_like(safety_terminated)
            joint_nonfinite = torch.zeros_like(joint_nonfinite)
            lower_limit = torch.zeros_like(lower_limit)
            upper_limit = torch.zeros_like(upper_limit)
            velocity_violation = torch.zeros_like(velocity_violation)

        unsafe_terminated = safety_terminated | (~finite_control)
        if self._task_control_mode() == "rl_task_space_tracking":
            self._update_task_space_state()
            self._update_final_pose_success_state(unsafe_terminated)
        else:
            self._within_final_pose.zero_()
            self._new_final_pose_success.zero_()
            self._termination_success.zero_()
        terminated = unsafe_terminated | self._termination_success

        # ── Stage-aware episode timeout ──
        time_out = self.episode_length_buf >= self._stage_max_episode_length - 1
        time_out = time_out & ~terminated
        if self._preflight_disable_ordinary_resets:
            time_out = torch.zeros_like(time_out)
        if self._task_control_mode() == "rl_task_space_tracking":
            self._record_hold_timeout_break(time_out)

        # ── Record offending joint (first joint violating a limit) ──
        offending_joint = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        offending_channel = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        # Check each joint for limit violations
        for j in range(5):
            joint_violation = (
                (q[:, j] < lower_limits[:, j] - self.cfg.joint_limit_margin)
                | (q[:, j] > upper_limits[:, j] + self.cfg.joint_limit_margin)
                | (~torch.isfinite(q[:, j]))
                | (~torch.isfinite(qdot[:, j]))
                | mechanical_hard_exceedance[:, j]
            )
            offending_joint = torch.where(
                joint_violation & (offending_joint < 0),
                torch.full_like(offending_joint, j),
                offending_joint,
            )
        # Check each control channel for non-finite values
        control_tensors_4d = [
            self._current_torque,
            self._current_fd_simscape_output,
            self._current_tau_fd_adapted,
            self._fd_tau_hat,
            self._current_tau_applied_raw,
            self._current_tau_isaac_filtered,
            self._current_tau_isaac_clamped,
            self._current_valve_cmd,
            self._dP,
            self._fnet,
            self._rom_cylinder_pos_target,
            self._rom_pressure,
            self._rom_xdot_pred,
            self._rom_dx_pred,
            self._final_rom_delta_extension_50ms,
            self._final_rom_delta_extension_kinematic_50ms,
            self._final_rom_delta_extension_correction_50ms,
            self._final_rom_delta_extension_10ms,
            self._final_rom_delta_isaac_50ms,
            self._final_rom_delta_isaac_10ms,
            self._final_rom_extension,
            self._final_rom_extension_velocity,
            self._current_qddot_cmd,
            self._z_arm_hat,
        ]
        control_nonfinite = ~finite_control
        for t_idx, t in enumerate(control_tensors_4d):
            if t.dim() == 2 and t.shape[-1] == 4:
                for c in range(4):
                    channel_nonfinite = ~torch.isfinite(t[:, c])
                    offending_channel = torch.where(
                        channel_nonfinite & (offending_channel < 0),
                        torch.full_like(offending_channel, c + t_idx * 4),
                        offending_channel,
                    )

        # ── Store per-reason diagnostics ──
        self._termination_joint_nonfinite = joint_nonfinite.detach().clone()
        self._termination_lower_limit = lower_limit.detach().clone()
        self._termination_upper_limit = upper_limit.detach().clone()
        self._termination_cylinder_lower_limit = cylinder_lower_limit.detach().clone()
        self._termination_cylinder_upper_limit = cylinder_upper_limit.detach().clone()
        self._termination_velocity = velocity_violation.detach().clone()
        self._termination_control_nonfinite = control_nonfinite.detach().clone()
        self._termination_offending_joint = offending_joint.detach().clone()
        self._termination_offending_channel = offending_channel.detach().clone()

        # ── Accumulate counters ──
        was_terminated = terminated.detach()
        self._termination_count += was_terminated.float()
        self._termination_joint_nonfinite_count += joint_nonfinite.float()
        self._termination_lower_limit_count += lower_limit.float()
        self._termination_upper_limit_count += upper_limit.float()
        self._termination_velocity_count += velocity_violation.float()
        self._termination_control_nonfinite_count += control_nonfinite.float()
        self._termination_timeout_count += time_out.float()
        self._termination_success_count += self._new_final_pose_success.float()

        # ── Populate extras["log"] ──
        term_log: dict[str, object] = {}
        term_log["termination/joint_nonfinite"] = torch.mean(joint_nonfinite.float())
        term_log["termination/lower_limit"] = torch.mean(lower_limit.float())
        term_log["termination/upper_limit"] = torch.mean(upper_limit.float())
        term_log["termination/cylinder_lower_limit"] = torch.mean(cylinder_lower_limit.float())
        term_log["termination/cylinder_upper_limit"] = torch.mean(cylinder_upper_limit.float())
        term_log["termination/velocity"] = torch.mean(velocity_violation.float())
        mechanical_soft_limits = self._mechanical_velocity_limit_tensor("mechanical_soft_velocity_limits")
        mechanical_soft_exceedance = (
            (mechanical_soft_limits > 0.0)
            & (torch.abs(mechanical_qdot) > mechanical_soft_limits)
        )
        cylinder_old_threshold_crossing = (
            torch.abs(cylinder_extension_velocity)
            > abs(float(getattr(self.cfg, "unvalidated_velocity_diagnostic_threshold", 6.0)))
        )
        term_log["mechanical_velocity/abs_mean_per_joint"] = torch.mean(torch.abs(mechanical_qdot.detach()), dim=0)
        term_log["mechanical_velocity/abs_max_per_joint"] = torch.max(torch.abs(mechanical_qdot.detach()), dim=0).values
        term_log["mechanical_velocity/soft_exceedance_fraction_per_joint"] = torch.mean(
            mechanical_soft_exceedance.float(), dim=0
        )
        term_log["mechanical_velocity/hard_exceedance_fraction_per_joint"] = torch.mean(
            mechanical_hard_exceedance.float(), dim=0
        )
        term_log["cylinder_velocity/abs_mean_per_channel"] = torch.mean(
            torch.abs(cylinder_extension_velocity.detach()), dim=0
        )
        term_log["cylinder_velocity/abs_max_per_channel"] = torch.max(
            torch.abs(cylinder_extension_velocity.detach()), dim=0
        ).values
        term_log["cylinder_velocity/old_6_threshold_crossing_fraction"] = torch.mean(
            cylinder_old_threshold_crossing.float()
        )
        term_log["cylinder_velocity/diagnostic_only"] = torch.ones((), dtype=torch.float32, device=self.device)
        term_log["termination/velocity_raw_exceedance"] = torch.mean(
            torch.any(self._hydraulic_velocity_raw_exceedance, dim=-1).float()
        )
        term_log["termination/velocity_confirmed_hard"] = torch.mean(
            torch.any(self._hydraulic_velocity_confirmed_hard, dim=-1).float()
        )
        term_log["termination/velocity_spike_ignored"] = torch.mean(
            torch.any(self._hydraulic_velocity_spike_ignored, dim=-1).float()
        )
        term_log["termination/velocity_persistence_max"] = torch.max(
            self._hydraulic_velocity_persistence.float()
        )
        term_log["velocity_audit/physx_spike_count"] = self._velocity_audit_physx_spike_count
        term_log["velocity_audit/target_jump_count"] = self._velocity_audit_target_jump_count
        term_log["velocity_audit/genuine_high_motion_count"] = self._velocity_audit_genuine_high_motion_count
        term_log["velocity_audit/unit_mapping_mismatch_count"] = self._velocity_audit_unit_mapping_mismatch_count
        term_log["velocity_audit/nonfinite_count"] = self._velocity_audit_nonfinite_count
        term_log["termination/control_nonfinite"] = torch.mean(control_nonfinite.float())
        term_log["termination/success"] = torch.mean(self._termination_success.float())
        term_log["termination/unsafe"] = torch.mean(unsafe_terminated.float())
        term_log["termination/any"] = torch.mean(terminated.float())
        term_log["termination/timeout"] = torch.mean(time_out.float())
        term_log["termination/offending_joint_dist"] = torch.mean(
            (offending_joint >= 0).float()
        )
        term_log["termination/acc_joint_nonfinite"] = torch.mean(
            self._termination_joint_nonfinite_count
        )
        term_log["termination/acc_lower_limit"] = torch.mean(
            self._termination_lower_limit_count
        )
        term_log["termination/acc_upper_limit"] = torch.mean(
            self._termination_upper_limit_count
        )
        term_log["termination/acc_velocity"] = torch.mean(self._termination_velocity_count)
        term_log["termination/acc_control_nonfinite"] = torch.mean(
            self._termination_control_nonfinite_count
        )
        term_log["termination/acc_timeout"] = torch.mean(self._termination_timeout_count)
        term_log["termination/acc_success"] = torch.mean(self._termination_success_count)
        self._populate_extras_log(term_log)

        self._last_terminated_debug = unsafe_terminated.detach().clone()
        self._last_time_out_debug = time_out.detach().clone()
        self._last_done_debug = (terminated | time_out).detach().clone()
        return terminated, time_out

    @staticmethod
    def _reset_reason_name(code: float) -> str:
        names = {
            0.0: "startup_or_manual",
            1.0: "timeout",
            2.0: "unsafe_termination",
            3.0: "manual_target_change",
            4.0: "final_pose_success",
        }
        return names.get(float(code), "unknown")

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        reset_due_to_timeout = self._last_time_out_debug[env_ids].detach().float().clone()
        reset_due_to_termination = self._last_terminated_debug[env_ids].detach().float().clone()
        final_pose_success_snapshot = self._termination_success[env_ids].detach().float().clone()
        previous_q_ref_target = self._q_ref_target[env_ids].detach().clone()
        previous_reset_count = self._reset_count[env_ids].detach().clone()

        # ── Capture terminal snapshots BEFORE super()._reset_idx clears buffers ──
        episode_length_snapshot = self.episode_length_buf[env_ids].detach().float().clone()
        final_position_error_snapshot = torch.linalg.vector_norm(
            self._ee_position_error[env_ids], dim=-1
        ).detach().clone()
        final_orientation_error_snapshot = (
            self._ee_orientation_error[env_ids].reshape(-1).detach().clone()
        )
        final_linear_speed_snapshot = self._ee_linear_speed[env_ids].detach().clone()
        final_angular_speed_snapshot = self._ee_angular_speed[env_ids].detach().clone()
        final_success_counter_snapshot = (
            self._final_pose_success_counter[env_ids].detach().float().clone()
        )
        final_max_hold_snapshot = (
            self._final_pose_settling_hold_max[env_ids].detach().float().clone()
        )
        near_pose_low_speed_snapshot = (
            (final_position_error_snapshot <= float(self.cfg.final_pose_position_tolerance_m))
            & (final_orientation_error_snapshot <= float(self.cfg.final_pose_orientation_tolerance_rad))
            & (final_linear_speed_snapshot <= float(self.cfg.final_pose_linear_speed_tolerance_mps))
            & (final_angular_speed_snapshot <= float(self.cfg.final_pose_angular_speed_tolerance_radps))
        ).float()
        episode_denominator = torch.clamp(episode_length_snapshot, min=1.0)
        mean_position_error_snapshot = (
            self._episode_position_error_sum[env_ids].detach().clone()
            / episode_denominator
        )
        clamp_activation_snapshot = (
            self._episode_clamp_activation_sum[env_ids].detach().clone()
            / episode_denominator
        )
        valve_saturation_snapshot = (
            self._episode_valve_saturation_fraction_sum[env_ids].detach().clone()
            / episode_denominator
        )
        time_to_success_snapshot = (
            self._final_pose_time_to_success_steps[env_ids].detach().float().clone()
        )
        final_unsafe_penalty_snapshot = (
            self._terminal_unsafe_penalty[env_ids].detach().clone()
        )
        final_unsafe_mask_snapshot = (
            self._terminal_unsafe_mask[env_ids].detach().clone()
        )
        termination_reason_snapshot = torch.zeros_like(reset_due_to_timeout)
        termination_reason_snapshot = torch.where(
            reset_due_to_timeout > 0.5,
            torch.ones_like(termination_reason_snapshot),
            termination_reason_snapshot,
        )
        termination_reason_snapshot = torch.where(
            reset_due_to_termination > 0.5,
            torch.full_like(termination_reason_snapshot, 2.0),
            termination_reason_snapshot,
        )
        termination_reason_snapshot = torch.where(
            final_pose_success_snapshot > 0.5,
            torch.full_like(termination_reason_snapshot, 4.0),
            termination_reason_snapshot,
        )
        terminal_log: dict[str, object] = {}
        terminal_log["Episode/final_pose_success"] = torch.mean(final_pose_success_snapshot)
        terminal_log["Episode/final_position_error"] = torch.mean(final_position_error_snapshot)
        terminal_log["Episode/final_orientation_error_rad"] = torch.mean(
            final_orientation_error_snapshot
        )
        terminal_log["Episode/final_orientation_error_deg"] = torch.rad2deg(
            torch.mean(final_orientation_error_snapshot)
        )
        terminal_log["Episode/final_linear_speed"] = torch.mean(final_linear_speed_snapshot)
        terminal_log["Episode/final_angular_speed"] = torch.mean(final_angular_speed_snapshot)
        terminal_log["Episode/final_success_hold_steps"] = torch.mean(
            final_success_counter_snapshot
        )
        terminal_log["Episode/max_hold_steps"] = torch.mean(final_max_hold_snapshot)
        terminal_log["Episode/mean_position_error"] = torch.mean(mean_position_error_snapshot)
        terminal_log["Episode/clamp_activation_fraction"] = torch.mean(clamp_activation_snapshot)
        terminal_log["Episode/valve_saturation_fraction"] = torch.mean(valve_saturation_snapshot)
        terminal_log["Episode/time_to_success_steps"] = torch.mean(time_to_success_snapshot)
        terminal_log["Episode/final_episode_length"] = torch.mean(episode_length_snapshot)
        terminal_log["Episode/final_termination_reason"] = torch.mean(
            termination_reason_snapshot
        )
        terminal_log["Episode/final_unsafe_penalty"] = torch.mean(final_unsafe_penalty_snapshot)
        terminal_log["Episode/final_unsafe_mask"] = torch.mean(final_unsafe_mask_snapshot.float())
        self._update_rolling_1000_metrics(
            success=final_pose_success_snapshot,
            timeout=reset_due_to_timeout,
            final_position_error=final_position_error_snapshot,
            final_orientation_error=final_orientation_error_snapshot,
            final_linear_speed=final_linear_speed_snapshot,
            final_angular_speed=final_angular_speed_snapshot,
            max_hold_steps=final_max_hold_snapshot,
            clamp_activation_fraction=clamp_activation_snapshot,
            valve_saturation_fraction=valve_saturation_snapshot,
        )
        terminal_log.update(self._rolling_1000_log())
        self._populate_extras_log(terminal_log)

        # ── Aggregate curriculum performance from completed episodes ──
        n_done = env_ids.numel()
        unsafe_mask = reset_due_to_termination > 0.5
        timeout_mask = reset_due_to_timeout > 0.5
        velocity_done = self._termination_velocity[env_ids].detach().float().clone()
        other_unsafe_done = unsafe_mask.float() * (1.0 - velocity_done)

        self._curriculum_stage_episode_count += float(n_done)
        self._curriculum_stage_timeout_count += float(timeout_mask.sum().item())
        self._curriculum_stage_velocity_count += float(velocity_done.sum().item())
        self._curriculum_stage_other_unsafe_count += float(other_unsafe_done.sum().item())
        self._curriculum_stage_position_error_sum += float(mean_position_error_snapshot.sum().item())
        self._curriculum_stage_success_count += float(final_pose_success_snapshot.sum().item())
        self._curriculum_stage_near_pose_low_speed_count += float(near_pose_low_speed_snapshot.sum().item())
        self._curriculum_stage_max_hold_steps_sum += float(final_max_hold_snapshot.sum().item())

        # ── Evaluate and log curriculum advancement ──
        old_stage = self._curriculum_stage_index
        self._evaluate_stage_advancement()
        stage_advanced = self._curriculum_stage_index > old_stage

        # ── Curriculum telemetry in extras["log"] ──
        n = max(self.num_envs, 1)
        curriculum_log: dict[str, object] = {}
        curriculum_log["curriculum/stage"] = float(self._curriculum_stage_index)
        curriculum_log["curriculum/action_authority"] = float(
            self._action_authority_multiplier
        )
        curriculum_log["curriculum/episode_length"] = (
            float(self._stage_max_episode_length) * float(self.cfg.sim.dt)
        )
        curriculum_log["curriculum/automatic_advancement_enabled"] = float(
            bool(self.cfg.enable_automatic_curriculum)
        )
        curriculum_log["curriculum/advance_count"] = float(self._curriculum_stage_advance_count)
        curriculum_log["curriculum/stage_ready"] = float(stage_advanced)
        curriculum_log["curriculum/episodes_per_env"] = self._curriculum_stage_episode_count / n
        curriculum_log["curriculum/mean_position_error"] = (
            self._curriculum_stage_position_error_sum
            / max(self._curriculum_stage_episode_count, 1.0)
        )
        curriculum_log["curriculum/velocity_termination_rate"] = (
            self._curriculum_stage_velocity_count
            / max(self._curriculum_stage_episode_count, 1.0)
        )
        curriculum_log["curriculum/other_unsafe_rate"] = (
            self._curriculum_stage_other_unsafe_count
            / max(self._curriculum_stage_episode_count, 1.0)
        )
        curriculum_log["curriculum/timeout_rate"] = (
            self._curriculum_stage_timeout_count
            / max(self._curriculum_stage_episode_count, 1.0)
        )
        curriculum_log["curriculum/goal_difficulty_enabled"] = float(
            bool(getattr(self.cfg, "enable_goal_difficulty_curriculum", False))
        )
        curriculum_log["curriculum/stage_success_rate"] = (
            self._curriculum_stage_success_count
            / max(self._curriculum_stage_episode_count, 1.0)
        )
        curriculum_log["curriculum/stage_near_pose_low_speed_rate"] = (
            self._curriculum_stage_near_pose_low_speed_count
            / max(self._curriculum_stage_episode_count, 1.0)
        )
        curriculum_log["curriculum/stage_mean_max_hold_steps"] = (
            self._curriculum_stage_max_hold_steps_sum
            / max(self._curriculum_stage_episode_count, 1.0)
        )
        curriculum_log["curriculum/stage_ready_windows"] = float(self._curriculum_stage_ready_windows)
        self._populate_extras_log(curriculum_log)

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
        self._qdot_ref[env_ids] = 0.0
        self._qddot_ref[env_ids] = 0.0
        self._sample_joint_targets(env_ids, q_ref_center)
        self._q_ref_sampled_on_reset[env_ids] = self._q_ref_target[env_ids].detach().clone()
        self._last_reset_step_global[env_ids] = float(getattr(self, "common_step_counter", 0))
        self._reset_count[env_ids] = previous_reset_count + 1.0
        self._reset_due_to_timeout[env_ids] = reset_due_to_timeout
        self._reset_due_to_termination[env_ids] = reset_due_to_termination
        reason_code = torch.zeros_like(reset_due_to_timeout)
        reason_code = torch.where(reset_due_to_timeout > 0.5, torch.ones_like(reason_code), reason_code)
        reason_code = torch.where(reset_due_to_termination > 0.5, torch.full_like(reason_code, 2.0), reason_code)
        reason_code = torch.where(
            final_pose_success_snapshot > 0.5,
            torch.full_like(reason_code, 4.0),
            reason_code,
        )
        self._last_reset_reason_code[env_ids] = reason_code
        self._record_target_change(env_ids, previous_q_ref_target, manual_target_change=False)
        self._q_ref_target_delta[env_ids] = self._q_ref_target[env_ids] - self._q_ref_start[env_ids]
        self._q_ref_target_observed[env_ids] = self._q_ref_target[env_ids].detach().clone()
        self._q_ref_ramp_progress[env_ids] = 0.0 if self.cfg.use_q_ref_ramp else 1.0
        self._qdot_ref[env_ids] = 0.0
        # Refresh the measured robot-base pose before the Stage-1 sampler uses
        # it as the center of the existing local reachable target range.
        self._update_task_space_state()
        self._sample_task_space_command(env_ids)
        final_pose_target = self._final_pose_target()
        self._final_pose_target_at_episode_start[env_ids] = final_pose_target[env_ids]

        pressure_source = str(getattr(self.cfg, "final_causal_rom_pressure_source", "training_mean"))
        if self._final_causal_rom is not None and pressure_source in {"training_mean", "sd_observer"}:
            reset_dP = self._final_causal_rom_training_mean_dp.reshape(1, 4).expand(env_ids.numel(), 4)
            self._dP[env_ids] = reset_dP
        else:
            reset_dP = torch.zeros((env_ids.numel(), 4), dtype=torch.float32, device=self.device)
            self._dP[env_ids] = reset_dP
        self._fnet[env_ids] = 0.0
        self._prev_action[env_ids] = 0.0
        self._current_action[env_ids] = 0.0
        self._rom_held_policy_action = None  # Force fresh action on next ROM-update step after reset
        self._action_delta[env_ids] = 0.0
        self._vehicle_commanded_speed_kmh[env_ids] = 0.0
        self._vehicle_wheel_velocity_target[env_ids] = 0.0
        self._current_qddot_cmd[env_ids] = 0.0
        self._scripted_qddot_feedback[env_ids] = 0.0
        self._scripted_qddot_feedforward[env_ids] = 0.0
        self._scripted_qddot_raw[env_ids] = 0.0
        self._scripted_action_preclip[env_ids] = 0.0
        self._scripted_control_ramp[env_ids] = 0.0
        self._prev_valve_cmd[env_ids] = 0.0
        self._current_valve_cmd[env_ids] = 0.0
        self._current_torque[env_ids] = 0.0
        self._current_fd_simscape_output[env_ids] = 0.0
        self._current_tau_fd_adapted[env_ids] = 0.0
        self._current_tau_applied_raw[env_ids] = 0.0
        self._current_tau_isaac_filtered[env_ids] = 0.0
        self._current_tau_isaac_clamped[env_ids] = 0.0
        self._rom_cylinder_pos_target[env_ids] = joint_pos[:, self._command_joint_ids]
        self._rom_cylinder_pos_target_prev[env_ids] = joint_pos[:, self._command_joint_ids]
        self._rom_cylinder_pos_target_at_reset[env_ids] = joint_pos[:, self._command_joint_ids]
        self._rom_cylinder_measured_at_reset[env_ids] = joint_pos[:, self._command_joint_ids]
        self._rom_cylinder_cumulative_target_motion[env_ids] = 0.0
        self._rom_cylinder_cumulative_measured_motion[env_ids] = 0.0
        self._rom_cylinder_tracking_ratio[env_ids] = 0.0
        self._rom_pressure[env_ids] = reset_dP
        self._rom_xdot_pred[env_ids] = 0.0
        self._rom_dx_pred[env_ids] = 0.0
        self._rom_valve_cmd_delayed[env_ids] = 0.0
        self._rom_valve_cmd_effective[env_ids] = 0.0
        self._final_rom_delta_extension_50ms[env_ids] = 0.0
        self._final_rom_delta_extension_kinematic_50ms[env_ids] = 0.0
        self._final_rom_delta_extension_correction_50ms[env_ids] = 0.0
        self._final_rom_delta_extension_10ms[env_ids] = 0.0
        self._final_rom_delta_isaac_50ms[env_ids] = 0.0
        self._final_rom_delta_isaac_10ms[env_ids] = 0.0
        self._final_rom_extension[env_ids] = 0.0
        self._final_rom_extension_velocity[env_ids] = 0.0
        self._prev_hydraulic_q[env_ids] = joint_pos[:, self._command_joint_ids]
        self._hydraulic_qdot_fd[env_ids] = 0.0
        self._hydraulic_target_velocity[env_ids] = 0.0
        self._hydraulic_rom_applied_rate[env_ids] = 0.0
        self._hydraulic_velocity_persistence[env_ids] = 0
        self._hydraulic_velocity_raw_exceedance[env_ids] = False
        self._hydraulic_velocity_confirmed_hard[env_ids] = False
        self._hydraulic_velocity_spike_ignored[env_ids] = False

        # ── Reset curriculum / termination / reward tracking state ─────────────
        self._termination_joint_nonfinite[env_ids] = False
        self._termination_lower_limit[env_ids] = False
        self._termination_upper_limit[env_ids] = False
        self._termination_cylinder_lower_limit[env_ids] = False
        self._termination_cylinder_upper_limit[env_ids] = False
        self._termination_velocity[env_ids] = False
        self._termination_control_nonfinite[env_ids] = False
        self._termination_success[env_ids] = False
        self._within_final_pose[env_ids] = False
        self._final_pose_success_counter[env_ids] = 0
        self._final_pose_success_previous[env_ids] = False
        self._new_final_pose_success[env_ids] = False
        self._final_pose_time_to_success_steps[env_ids] = -1
        self._strict_position_condition[env_ids] = False
        self._strict_orientation_condition[env_ids] = False
        self._strict_linear_speed_condition[env_ids] = False
        self._strict_angular_speed_condition[env_ids] = False
        self._strict_all_except_hold_condition[env_ids] = False
        self._final_pose_first_strict_settled_step[env_ids] = -1
        self._final_pose_last_strict_settled_step[env_ids] = -1
        self._final_pose_steps_remaining_after_max_hold[env_ids] = 0
        self._final_pose_max_hold_steps_before_break[env_ids] = 0
        self._final_pose_settling_hold_max[env_ids] = 0
        # NOTE: accumulated counters (_termination_*_count) are deliberately NOT
        # reset — they accumulate across episodes for lifetime diagnostics.
        self._termination_offending_joint[env_ids] = -1
        self._termination_offending_channel[env_ids] = -1
        self._episode_pos_tracking_sum[env_ids] = 0.0
        self._episode_orient_tracking_sum[env_ids] = 0.0
        self._episode_progress_reward_sum[env_ids] = 0.0
        self._episode_action_rate_penalty_sum[env_ids] = 0.0
        self._episode_action_mag_penalty_sum[env_ids] = 0.0
        self._episode_valve_sat_penalty_sum[env_ids] = 0.0
        self._episode_clamp_penalty_sum[env_ids] = 0.0
        self._episode_fd_load_penalty_sum[env_ids] = 0.0
        self._episode_unsafe_penalty_sum[env_ids] = 0.0
        self._episode_linear_velocity_penalty_sum[env_ids] = 0.0
        self._episode_angular_velocity_penalty_sum[env_ids] = 0.0
        self._episode_settling_reward_sum[env_ids] = 0.0
        self._episode_near_target_linear_settling_sum[env_ids] = 0.0
        self._episode_near_target_angular_settling_sum[env_ids] = 0.0
        self._episode_hold_progress_reward_sum[env_ids] = 0.0
        self._episode_clamp_activation_sum[env_ids] = 0.0
        self._episode_valve_saturation_fraction_sum[env_ids] = 0.0
        self._episode_position_error_sum[env_ids] = 0.0
        self._episode_success_bonus_sum[env_ids] = 0.0
        self._prev_pos_error_norm[env_ids] = 0.0
        self._prev_task_space_potential[env_ids] = 0.0
        self._task_space_potential_valid[env_ids] = False

        # ── Clear consumed terminal unsafe-penalty snapshot buffers ──
        self._terminal_unsafe_penalty[env_ids] = 0.0
        self._terminal_unsafe_mask[env_ids] = False

        self._final_rom_input_features_raw[env_ids] = 0.0
        self._final_rom_input_features_normalized[env_ids] = 0.0
        self._final_rom_inverse_mapping_error[env_ids] = 0.0
        self._final_rom_range_violation_fraction[env_ids] = 0.0
        self._final_rom_update_flag[env_ids] = 0.0
        self._final_rom_history_ready[env_ids] = 0.0
        self._rom_target_clamp_fraction[env_ids] = 0.0
        self._rom_pressure_clamp_fraction[env_ids] = 0.0
        self._rom_velocity_clamp_fraction[env_ids] = 0.0
        self._rom_dx_clamp_fraction[env_ids] = 0.0
        self._rom_nan_guard_triggered[env_ids] = 0.0
        self._rom_output_zeroed_fraction[env_ids] = 0.0
        self._rom_actuator_pos_error_mean[env_ids] = 0.0
        self._rom_actuator_pos_error_abs_max[env_ids] = 0.0
        self._rom_rate_limit_fraction[env_ids] = 0.0
        self._rom_ch4_damping_applied[env_ids] = 0.0
        self._rom_boundary_active[env_ids] = False
        self._rom_outward_request[env_ids] = False
        self._rom_blocked_outward_increment[env_ids] = 0.0
        self._rom_clamp_activation_fraction[env_ids] = 0.0
        self._rom_clamp_blocked_outward_mean[env_ids] = 0.0
        self._fd_tau_hat[env_ids] = 0.0
        self._fd_tau_hat_normalized[env_ids] = 0.0
        self._fd_soft_limit_fraction[env_ids] = 0.0
        self._fd_hard_limit_fraction[env_ids] = 0.0
        self._fd_normalized_abs_max[env_ids] = 0.0
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
        self._aam_latent_state_version[env_ids] = 0
        self._observation_aam_state_version[env_ids] = -1
        self._id_aam_state_version[env_ids] = -1
        self._learned_control_initialized[env_ids] = False
        self._prev_q_error_mean[env_ids] = self._weighted_q_error_mean(
            joint_pos[:, self._state_joint_ids], self._q_ref_commanded[env_ids]
        )
        self._controller.reset(env_ids)
        self._reduced_hydraulic_rom.reset(env_ids)
        if self._final_causal_rom_wrapper is not None:
            self._final_causal_rom_wrapper.reset(env_ids)
        if self.cfg.scripted_reference_reset_phase_randomization and self.num_envs > 1:
            self._scripted_reference_phase_offset[env_ids] = torch.rand(
                (env_ids.numel(), 1), dtype=torch.float32, device=self.device
            ) * float(self.cfg.episode_length_s)
        else:
            self._scripted_reference_phase_offset[env_ids] = 0.0

        self.robot.write_root_pose_to_sim(default_root_pose, env_ids)
        self.robot.write_root_velocity_to_sim(default_root_velocity, env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        if bool(getattr(self.cfg, "verbose_reset_logging", False)):
            max_logs = max(int(getattr(self.cfg, "reset_diagnostic_log_limit", 8)), 0)
            for local_index, env_id in enumerate(env_ids.detach().cpu().tolist()):
                if local_index >= max_logs:
                    break
                code = float(reason_code[local_index].detach().cpu())
                print(
                    "[RESET] "
                    f"env={env_id} "
                    f"count={int(self._reset_count[env_id].detach().cpu().item())} "
                    f"reason={self._reset_reason_name(code)} "
                    f"episode_step={int(episode_length_snapshot[local_index].detach().cpu().item())} "
                    f"timeout={bool(reset_due_to_timeout[local_index].detach().cpu().item() > 0.5)} "
                    f"unsafe_termination={bool(reset_due_to_termination[local_index].detach().cpu().item() > 0.5)}",
                    flush=True,
                )

    def run_asset_transmission_preflight(
        self,
        *,
        steps: int = 80,
        settle_steps: int | None = None,
        pulse_m: float = 0.01,
        min_target_motion_m: float = 1.0e-4,
        min_cylinder_motion_m: float = 0.002,
        min_mechanical_motion: float = 1.0e-4,
        max_root_z_motion_m: float | None = None,
        max_root_roll_pitch_motion_rad: float | None = None,
        max_pre_pulse_root_vertical_velocity_mps: float | None = None,
        disable_ordinary_resets: bool = False,
    ) -> dict[str, object]:
        if self.num_envs < 1:
            raise RuntimeError("Asset preflight requires at least one environment.")
        settle_steps = int(self.cfg.asset_preflight_settle_steps if settle_steps is None else settle_steps)
        max_root_z_motion_m = float(
            self.cfg.asset_preflight_max_pulse_root_z_motion_m
            if max_root_z_motion_m is None
            else max_root_z_motion_m
        )
        max_root_roll_pitch_motion_rad = float(
            self.cfg.asset_preflight_max_pulse_roll_pitch_motion_rad
            if max_root_roll_pitch_motion_rad is None
            else max_root_roll_pitch_motion_rad
        )
        max_pre_pulse_root_vertical_velocity_mps = float(
            self.cfg.asset_preflight_max_pre_pulse_root_vertical_velocity_mps
            if max_pre_pulse_root_vertical_velocity_mps is None
            else max_pre_pulse_root_vertical_velocity_mps
        )
        root_before_reset = self.robot.data.root_pos_w.detach().clone()
        self.reset()
        root_after_reset = self.robot.data.root_pos_w.detach().clone()
        reset_count_after_reset = self._reset_count.detach().clone()
        episode_after_reset = self.episode_length_buf.detach().clone()
        timeout_after_reset = self._last_time_out_debug.detach().clone()
        unsafe_after_reset = self._last_terminated_debug.detach().clone()
        wheel_suspension_after_reset = self._wheel_suspension_state()

        mechanical_families = self._mechanical_constraint_family_ids()
        hold_target = self.robot.data.joint_pos[:, self._command_joint_ids].detach().clone()
        zeros = torch.zeros((self.num_envs, self.cfg.action_space), dtype=torch.float32, device=self.device)
        previous_disable_ordinary_resets = self._preflight_disable_ordinary_resets
        self._preflight_disable_ordinary_resets = bool(disable_ordinary_resets)
        self._preflight_direct_target = hold_target.detach().clone()
        try:
            for _ in range(max(0, settle_steps)):
                self.step(zeros)
        finally:
            self._preflight_direct_target = None
            self._preflight_disable_ordinary_resets = previous_disable_ordinary_resets

        root_after_settle = self.robot.data.root_pos_w.detach().clone()
        root_before_pulse = root_after_settle.detach().clone()
        root_velocity_before_pulse = self.robot.data.root_lin_vel_w.detach().clone()
        root_orientation_before_pulse = self.robot.data.root_quat_w.detach().clone()
        settled_cylinder_positions = self.robot.data.joint_pos[:, self._command_joint_ids].detach().clone()
        settled_mechanical_positions = self._mechanical_family_positions(mechanical_families)
        reset_count_before_pulse = self._reset_count.detach().clone()
        episode_before_pulse = self.episode_length_buf.detach().clone()
        timeout_before_pulse = self._last_time_out_debug.detach().clone()
        unsafe_before_pulse = self._last_terminated_debug.detach().clone()
        wheel_suspension_after_settle = self._wheel_suspension_state()

        lower = self.robot.data.soft_joint_pos_limits[:, self._command_joint_ids, 0]
        upper = self.robot.data.soft_joint_pos_limits[:, self._command_joint_ids, 1]
        direction = torch.where(settled_cylinder_positions + pulse_m <= upper, 1.0, -1.0)
        target = torch.clamp(settled_cylinder_positions + direction * abs(float(pulse_m)), lower, upper)
        self._preflight_direct_target = target.detach().clone()
        previous_disable_ordinary_resets = self._preflight_disable_ordinary_resets
        self._preflight_disable_ordinary_resets = bool(disable_ordinary_resets)
        try:
            for _ in range(max(1, int(steps))):
                self.step(zeros)
        finally:
            self._preflight_direct_target = None
            self._preflight_disable_ordinary_resets = previous_disable_ordinary_resets

        q1 = self.robot.data.joint_pos[:, self._command_joint_ids].detach().clone()
        root_after_pulse = self.robot.data.root_pos_w.detach().clone()
        root_velocity_after_pulse = self.robot.data.root_lin_vel_w.detach().clone()
        root_orientation_after_pulse = self.robot.data.root_quat_w.detach().clone()
        mechanical1 = self._mechanical_family_positions(mechanical_families)
        mechanical_motion = self._mechanical_family_motion(settled_mechanical_positions, mechanical1)
        cylinder_motion = torch.abs(q1 - settled_cylinder_positions)
        target_motion = torch.abs(target - settled_cylinder_positions)
        reset_count_after_pulse = self._reset_count.detach().clone()
        wheel_suspension_after_pulse = self._wheel_suspension_state()
        reset_during_settle = bool(torch.any(reset_count_before_pulse != reset_count_after_reset).detach().cpu())
        reset_during_pulse = bool(torch.any(reset_count_after_pulse != reset_count_before_pulse).detach().cpu())
        reset_during_preflight = reset_during_settle or reset_during_pulse
        reset_to_settle_delta = root_after_settle - root_after_reset
        settle_window_delta = root_after_settle - root_after_reset
        pulse_window_delta = root_after_pulse - root_before_pulse
        pulse_root_motion = torch.linalg.norm(pulse_window_delta, dim=-1)
        pulse_root_z_motion = torch.abs(pulse_window_delta[:, 2])
        root_vertical_velocity_before_pulse = root_velocity_before_pulse[:, 2]
        root_vertical_velocity_after_pulse = root_velocity_after_pulse[:, 2]
        root_roll_pitch_before = self._root_roll_pitch_from_quat_wxyz(root_orientation_before_pulse)
        root_roll_pitch_after = self._root_roll_pitch_from_quat_wxyz(root_orientation_after_pulse)
        root_roll_pitch_delta = torch.max(torch.abs(root_roll_pitch_after - root_roll_pitch_before), dim=-1).values
        finite = bool(
            torch.isfinite(target).all()
            and torch.isfinite(q1).all()
            and torch.isfinite(root_after_pulse).all()
            and torch.isfinite(root_velocity_before_pulse).all()
            and all(math.isfinite(value) for value in mechanical_motion.values())
        )
        gates = AssetPreflightGates(
            min_target_motion_m=float(min_target_motion_m),
            min_cylinder_motion_m=float(min_cylinder_motion_m),
            min_mechanical_motion=float(min_mechanical_motion),
            max_pulse_root_z_motion_m=float(max_root_z_motion_m),
            max_pulse_roll_pitch_motion_rad=float(max_root_roll_pitch_motion_rad),
            max_pre_pulse_root_vertical_velocity_mps=float(max_pre_pulse_root_vertical_velocity_mps),
        )
        decision = classify_asset_transmission_preflight(
            finite=finite,
            reset_during_preflight=reset_during_preflight,
            pulse_root_z_motion_m=float(pulse_root_z_motion[0].detach().cpu()),
            pulse_root_roll_pitch_motion_rad=float(root_roll_pitch_delta[0].detach().cpu()),
            root_vertical_velocity_before_pulse_mps=float(root_vertical_velocity_before_pulse[0].detach().cpu()),
            target_motion_m=target_motion[0].detach().cpu().tolist(),
            cylinder_motion_m=cylinder_motion[0].detach().cpu().tolist(),
            mechanical_family_motion=mechanical_motion,
            gates=gates,
        )
        pass_mask = decision == ASSET_TRANSMISSION_PASS
        stage = "Stage 2: direct actuator transmission"
        result = {
            "name": "asset_transmission_preflight",
            "passed": bool(pass_mask),
            "preflight_stage": stage,
            "preflight_decision": decision,
            "writer": "direct preflight set_joint_position_target only",
            "command_joint_names": tuple(self.cfg.command_joint_names),
            "root_position_source": "robot.data.root_pos_w",
            "root_frame": "world",
            "env_origin": self.scene.env_origins[0].detach().cpu().tolist(),
            "requested_pulse_m": float(pulse_m),
            "target_motion_m": target_motion[0].detach().cpu().tolist(),
            "cylinder_motion_m": cylinder_motion[0].detach().cpu().tolist(),
            "cylinder_tracking_ratio": (
                cylinder_motion[0] / torch.clamp(target_motion[0], min=1.0e-9)
            ).detach().cpu().tolist(),
            "mechanical_family_motion": mechanical_motion,
            "root_before_reset": root_before_reset[0].detach().cpu().tolist(),
            "root_after_reset": root_after_reset[0].detach().cpu().tolist(),
            "root_after_settle": root_after_settle[0].detach().cpu().tolist(),
            "root_before_pulse": root_before_pulse[0].detach().cpu().tolist(),
            "root_after_pulse": root_after_pulse[0].detach().cpu().tolist(),
            "root_reset_delta": (root_after_reset - root_before_reset)[0].detach().cpu().tolist(),
            "reset_to_settle_root_delta": reset_to_settle_delta[0].detach().cpu().tolist(),
            "settle_window_root_delta": settle_window_delta[0].detach().cpu().tolist(),
            "pulse_window_root_delta": pulse_window_delta[0].detach().cpu().tolist(),
            "pulse_window_root_motion_m": float(pulse_root_motion[0].detach().cpu()),
            "pulse_window_root_z_motion_m": float(pulse_root_z_motion[0].detach().cpu()),
            "pulse_window_root_roll_pitch_delta_rad": float(root_roll_pitch_delta[0].detach().cpu()),
            "root_vertical_velocity_before_pulse_mps": float(root_vertical_velocity_before_pulse[0].detach().cpu()),
            "root_vertical_velocity_after_pulse_mps": float(root_vertical_velocity_after_pulse[0].detach().cpu()),
            "settled_cylinder_positions": settled_cylinder_positions[0].detach().cpu().tolist(),
            "settled_mechanical_family_sizes": {name: len(ids) for name, ids in mechanical_families.items()},
            "reset_count_after_reset": reset_count_after_reset.detach().cpu().tolist(),
            "reset_count_before_pulse": reset_count_before_pulse.detach().cpu().tolist(),
            "reset_count_after_pulse": reset_count_after_pulse.detach().cpu().tolist(),
            "episode_length_after_reset": episode_after_reset.detach().cpu().tolist(),
            "episode_length_before_pulse": episode_before_pulse.detach().cpu().tolist(),
            "timeout_after_reset": timeout_after_reset.detach().cpu().tolist(),
            "unsafe_after_reset": unsafe_after_reset.detach().cpu().tolist(),
            "timeout_before_pulse": timeout_before_pulse.detach().cpu().tolist(),
            "unsafe_before_pulse": unsafe_before_pulse.detach().cpu().tolist(),
            "wheel_suspension_after_reset": wheel_suspension_after_reset,
            "wheel_suspension_after_settle": wheel_suspension_after_settle,
            "wheel_suspension_after_pulse": wheel_suspension_after_pulse,
            "reset_during_settle": reset_during_settle,
            "reset_during_pulse": reset_during_pulse,
            "reset_during_preflight": reset_during_preflight,
            "finite": finite,
            "base_stability_gates": {
                "max_pulse_root_z_motion_m": float(max_root_z_motion_m),
                "max_pulse_roll_pitch_motion_rad": float(max_root_roll_pitch_motion_rad),
                "max_pre_pulse_root_vertical_velocity_mps": float(max_pre_pulse_root_vertical_velocity_mps),
            },
            "failure_category": "" if bool(pass_mask) else decision,
        }
        print(
            "[PREFLIGHT] asset_root_motion_report "
            f"reset_to_settle_root_delta={result['reset_to_settle_root_delta']} "
            f"settle_window_root_delta={result['settle_window_root_delta']} "
            f"pulse_window_root_delta={result['pulse_window_root_delta']}",
            flush=True,
        )
        return result

    def run_learned_pipeline_preflight(
        self,
        *,
        steps: int = 700,
        min_reference_motion: float = 0.02,
        min_qddot_abs_mean: float = 0.05,
        min_valve_abs_mean: float = 0.01,
        min_rom_delta_abs_mean: float = 1.0e-5,
        min_target_motion_m: float = 1.0e-4,
        min_cylinder_motion_m: float = 1.0e-5,
        min_mechanical_motion: float = 1.0e-5,
        max_clamp_fraction: float = 0.50,
    ) -> dict[str, object]:
        if not self.cfg.use_final_causal_cylinder_rom:
            raise RuntimeError("Learned-pipeline preflight requires use_final_causal_cylinder_rom=True.")
        previous_flag = self._learned_preflight_scripted_action_active
        self._learned_preflight_scripted_action_active = True
        self.reset()
        q0 = self.robot.data.joint_pos[:, self._command_joint_ids].detach().clone()
        target0 = self._rom_cylinder_pos_target.detach().clone()
        mechanical_families = self._mechanical_constraint_family_ids()
        mechanical0 = self._mechanical_family_positions(mechanical_families)
        zeros = torch.zeros((self.num_envs, self.cfg.action_space), dtype=torch.float32, device=self.device)
        valve_samples: list[torch.Tensor] = []
        qddot_samples: list[torch.Tensor] = []
        reference_samples: list[torch.Tensor] = []
        action_samples: list[torch.Tensor] = []
        rom_samples: list[torch.Tensor] = []
        rom_delta_50ms_samples: list[torch.Tensor] = []
        clamp_samples: list[torch.Tensor] = []
        finite = True
        try:
            for _ in range(max(1, int(steps))):
                self.step(zeros)
                reference_samples.append(self._q_ref_commanded[:, :4].detach().clone())
                qddot_samples.append(torch.abs(self._current_qddot_cmd[:, :4].detach()))
                action_samples.append(torch.abs(self._current_action.detach()))
                valve_samples.append(torch.abs(self._current_valve_cmd.detach()).mean(dim=-1))
                rom_samples.append(torch.abs(self._final_rom_delta_isaac_10ms.detach()).mean(dim=-1))
                rom_delta_50ms_samples.append(torch.abs(self._final_rom_delta_isaac_50ms.detach()).mean(dim=-1))
                clamp_samples.append(self._rom_target_clamp_fraction.detach())
                finite = finite and bool(
                    torch.isfinite(self._current_valve_cmd).all()
                    and torch.isfinite(self._final_rom_delta_isaac_10ms).all()
                    and torch.isfinite(self._rom_cylinder_pos_target).all()
                )
        finally:
            self._learned_preflight_scripted_action_active = previous_flag
        q1 = self.robot.data.joint_pos[:, self._command_joint_ids].detach().clone()
        target1 = self._rom_cylinder_pos_target.detach().clone()
        mechanical1 = self._mechanical_family_positions(mechanical_families)
        mechanical_motion = self._mechanical_family_motion(mechanical0, mechanical1)
        if reference_samples:
            reference_stack = torch.stack(reference_samples)
            reference_motion = torch.max(reference_stack[:, 0, :], dim=0).values - torch.min(
                reference_stack[:, 0, :], dim=0
            ).values
        else:
            reference_motion = torch.zeros((4,), dtype=torch.float32, device=self.device)
        qddot_abs_mean_per_joint = (
            torch.stack(qddot_samples).mean(dim=(0, 1)) if qddot_samples else torch.zeros((4,), device=self.device)
        )
        action_abs_mean_per_joint = (
            torch.stack(action_samples).mean(dim=(0, 1)) if action_samples else torch.zeros((4,), device=self.device)
        )
        valve_abs_mean = torch.stack(valve_samples).mean().item() if valve_samples else 0.0
        rom_delta_abs_mean = torch.stack(rom_samples).mean().item() if rom_samples else 0.0
        rom_delta_50ms_abs_mean = torch.stack(rom_delta_50ms_samples).mean().item() if rom_delta_50ms_samples else 0.0
        clamp_fraction = torch.stack(clamp_samples).mean().item() if clamp_samples else 1.0
        cylinder_motion = torch.abs(q1 - q0)
        target_motion = torch.abs(target1 - target0)
        mechanical_motion_ok = all(
            math.isfinite(value) and value >= float(min_mechanical_motion) for value in mechanical_motion.values()
        )
        passed = (
            finite
            and float(torch.max(reference_motion).item()) >= float(min_reference_motion)
            and float(torch.mean(qddot_abs_mean_per_joint).item()) >= float(min_qddot_abs_mean)
            and valve_abs_mean >= float(min_valve_abs_mean)
            and rom_delta_abs_mean >= float(min_rom_delta_abs_mean)
            and float(torch.max(target_motion[0]).item()) >= float(min_target_motion_m)
            and float(torch.max(cylinder_motion[0]).item()) >= float(min_cylinder_motion_m)
            and mechanical_motion_ok
            and clamp_fraction <= float(max_clamp_fraction)
        )
        if not finite:
            category = "BLOCKED_ID_NORMALIZATION"
            stage = "Stage 4: ID only"
        elif float(torch.max(reference_motion).item()) < float(min_reference_motion):
            category = "BLOCKED_REFERENCE_TOO_SMALL"
            stage = "Stage 3: reference generation"
        elif float(torch.mean(qddot_abs_mean_per_joint).item()) < float(min_qddot_abs_mean):
            category = "BLOCKED_REFERENCE_TOO_SMALL"
            stage = "Stage 3: reference generation"
        elif valve_abs_mean < float(min_valve_abs_mean):
            category = "BLOCKED_ID_COMMAND_TOO_SMALL"
            stage = "Stage 4: ID only"
        elif rom_delta_abs_mean < float(min_rom_delta_abs_mean):
            category = "BLOCKED_ROM_OUTPUT_TOO_SMALL"
            stage = "Stage 5: ROM only"
        elif float(torch.max(target_motion[0]).item()) < float(min_target_motion_m):
            category = "BLOCKED_TARGET_ACCUMULATION"
            stage = "Stage 6: full chain"
        elif float(torch.max(cylinder_motion[0]).item()) < float(min_cylinder_motion_m):
            category = "BLOCKED_DRIVE_AUTHORITY"
            stage = "Stage 6: full chain"
        elif not mechanical_motion_ok:
            category = "BLOCKED_LINKAGE_TRANSMISSION"
            stage = "Stage 6: full chain"
        elif clamp_fraction > float(max_clamp_fraction):
            category = "BLOCKED_TARGET_ACCUMULATION"
            stage = "Stage 6: full chain"
        else:
            category = "READY_FOR_PPO"
            stage = "Stage 6: full chain"
        return {
            "name": "learned_pipeline_preflight",
            "passed": bool(passed),
            "preflight_stage": stage,
            "preflight_decision": category,
            "failure_category": "" if bool(passed) else category,
            "reference_motion_rad": reference_motion.detach().cpu().tolist(),
            "qddot_abs_mean_per_joint": qddot_abs_mean_per_joint.detach().cpu().tolist(),
            "action_abs_mean_per_joint": action_abs_mean_per_joint.detach().cpu().tolist(),
            "static_rom_command_coupling": "ValveCmd-conditioned deployed correction",
            "valve_abs_mean": float(valve_abs_mean),
            "rom_delta_isaac_10ms_abs_mean": float(rom_delta_abs_mean),
            "rom_delta_isaac_50ms_abs_mean": float(rom_delta_50ms_abs_mean),
            "target_motion_m": target_motion[0].detach().cpu().tolist(),
            "cylinder_motion_m": cylinder_motion[0].detach().cpu().tolist(),
            "mechanical_family_motion": mechanical_motion,
            "target_clamp_fraction": float(clamp_fraction),
            "finite": bool(finite),
        }

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
