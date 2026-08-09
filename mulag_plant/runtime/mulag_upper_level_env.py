from __future__ import annotations

import csv
import json
import torch
import math
from pathlib import Path
from typing import Sequence

from isaaclab.assets import Articulation
import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import (
    combine_frame_transforms,
    quat_apply_inverse,
    quat_mul,
    euler_xyz_from_quat,
)

from mulag_plant.runtime.cylinder_models_runtime import (
    SequenceFDRuntime,
    SequenceIDRuntime,
    CylinderFDRuntime,
    CylinderIDRuntime,
    CylinderROMRuntime,
)
from mulag_plant.runtime.base_rom_runtime import BaseCylinderROMRuntime
from mulag_plant.runtime.command_rom_runtime import CommandCylinderROMRuntime
from mulag_plant.runtime.cylinder_space_scaling import CylinderSpaceScaling
from mulag_plant.runtime.joint_to_cylinder_adapter import (
    CylinderReference,
    JointToCylinderAdapter,
    PPOActionToCylinderTargetAdapter,
)
from mulag_plant.runtime.mulag_upper_level_env_cfg import (
    MulagUpperLevelEnvCfg,
    ARM_JOINT_DIM,
    DISTANCE_TO_GOAL_DIM,
    UNIMOG_POSE_DIM,
    UNIMOG_POSE_DOT_DIM,
    OBS_DIM,
    HISTORY_STEPS as HISTORY,
)

from mulag_plant.runtime.rrlab_reference_conventions import (
    kmh_to_wheel_rad_per_sec,
    obstacle_speed_range_rad_per_sec,
)
from mulag_plant.runtime.cylinder_coordinate_adapter import load_coordinate_mapping


class MulagUpperLevelEnv(DirectRLEnv):

    cfg: "MulagUpperLevelEnvCfg"

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # -- Resolve arm joint IDs --
        self._arm_joint_ids = self._find_joint_ids(self.cfg.arm_joint_names)
        self._cylinder_joint_ids = self._find_joint_ids(self.cfg.cylinder_joint_names)
        self._arm_joint_count = len(tuple(self.cfg.arm_joint_names))
        self._tool_joint_id = int(self._arm_joint_ids[-1].detach().cpu().item())
        self._joint6_coupling_source_ref_index: int | None = None
        self._joint6_coupling_target_ref_index: int | None = None
        self._joint6_coupling_source_joint_id: int | None = None
        self._joint6_coupling_target_joint_id: int | None = None
        self._joint6_coupling_source_name = ""
        self._joint6_coupling_target_name = ""
        self._resolve_joint6_coupling()

        # -- Resolve wheel joint IDs --
        self._wheel_joint_ids = self._find_joint_ids(self.cfg.vehicle_wheel_joint_names)
        # -- Resolve steering joint IDs (hold at 0 for straight driving) --
        self._steering_joint_ids = self._find_joint_ids([
            "Wheel_Front_Left_Steering_joint",
            "Wheel_Front_Right_Steering_joint",
        ])

        # -- Observation buffers with history --
        self._obs_joint_pos_hist = torch.zeros(
            (self.num_envs, HISTORY, ARM_JOINT_DIM), device=self.device
        )
        self._obs_joint_vel_hist = torch.zeros(
            (self.num_envs, HISTORY, ARM_JOINT_DIM), device=self.device
        )
        self._obs_distance_to_goal_hist = torch.zeros(
            (self.num_envs, HISTORY, DISTANCE_TO_GOAL_DIM), device=self.device
        )

        # -- Task-space command state --
        self._command_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._command_quat_w = torch.zeros((self.num_envs, 4), device=self.device)
        self._command_quat_w[:, 0] = 1.0
        self._command_resample_timer = torch.zeros(self.num_envs, device=self.device)
        self._command_resample_interval = torch.zeros(self.num_envs, device=self.device)

        # -- Vehicle speed --
        self._vehicle_speed_kmh = torch.zeros(self.num_envs, device=self.device)
        self._vehicle_speed_resample_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # -- Reward state (hysteresis) --

        # -- Prev action for smoothing --
        self._prev_action = torch.zeros((self.num_envs, 5), device=self.device)
        self._current_action = torch.zeros_like(self._prev_action)
        self._ppo_raw_action = torch.zeros_like(self._prev_action)

        # -- Cylinder-space learned pipeline state --
        self._joint_ref = torch.zeros((self.num_envs, self._arm_joint_count), device=self.device)
        self._joint_vel_ref = torch.zeros_like(self._joint_ref)
        self._cylinder_ref = torch.zeros((self.num_envs, 4), device=self.device)
        self._cylinder_vel_ref = torch.zeros_like(self._cylinder_ref)
        self._cylinder_x_current = torch.zeros_like(self._cylinder_ref)
        self._cylinder_xdot_current = torch.zeros_like(self._cylinder_ref)
        self._action_hydraulic = torch.zeros_like(self._cylinder_ref)
        self._action_ignored_passive = torch.zeros((self.num_envs, 1), device=self.device)
        self._action_delta_q_usd_ref = torch.zeros_like(self._cylinder_ref)
        self._action_target_delta_x_pre_gain_learned = torch.zeros_like(self._cylinder_ref)
        self._action_target_delta_x_post_gain_learned = torch.zeros_like(self._cylinder_ref)
        self._action_target_delta_x_safety_limit_learned = torch.zeros_like(self._cylinder_ref)
        self._action_target_delta_x_learned = torch.zeros_like(self._cylinder_ref)
        self._action_target_delta_x_pre_gain_clip_mask = torch.zeros_like(self._cylinder_ref, dtype=torch.bool)
        self._action_target_delta_x_post_gain_clip_mask = torch.zeros_like(self._cylinder_ref, dtype=torch.bool)
        self._action_target_delta_xdot_raw_learned = torch.zeros_like(self._cylinder_ref)
        self._action_target_delta_xdot_learned = torch.zeros_like(self._cylinder_ref)
        self._action_target_delta_xdot_clip_mask = torch.zeros_like(self._cylinder_ref, dtype=torch.bool)
        self._target_delta_xdot_clip_counts = torch.zeros(4, dtype=torch.long, device=self.device)
        self._target_delta_xdot_step_count = 0
        self._target_delta_xdot_clip_log_count = 0
        self._cylinder_pos_target = torch.zeros((self.num_envs, 4), device=self.device)
        self._neutral_hold_cylinder_pos_target = torch.zeros((self.num_envs, 4), device=self.device)
        self._explicit_pid_integral_error = torch.zeros((self.num_envs, 4), device=self.device)
        self._explicit_force_bias = torch.zeros((self.num_envs, 4), device=self.device)
        self._explicit_bias_calibrated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._explicit_bias_calib_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._explicit_force_bias_accum = torch.zeros((self.num_envs, 4), device=self.device)
        self._last_force_cmd = torch.zeros((self.num_envs, 4), device=self.device)
        self._qd_filt = torch.zeros((self.num_envs, 4), device=self.device)
        self._prev_force_cmd = torch.zeros((self.num_envs, 4), device=self.device)
        self._explicit_bias_window_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._bias_shapes_printed = False
        self._neutral_hold_prev_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._prev_valve_cmd = torch.zeros((self.num_envs, 4), device=self.device)
        self._id_prev_valve_cmd_input = torch.zeros_like(self._prev_valve_cmd)
        self._current_valve_cmd = torch.zeros_like(self._prev_valve_cmd)
        self._fd_baseline_valve_cmd = torch.zeros_like(self._prev_valve_cmd)
        self._id_input_tensor = torch.zeros((self.num_envs, 24), device=self.device)
        self._dP = torch.zeros((self.num_envs, 4), device=self.device)
        self._force = torch.zeros_like(self._dP)
        self._hydraulic_dP_training_mean = torch.zeros((1, 4), device=self.device)
        self._hydraulic_force_training_mean = torch.zeros((1, 4), device=self.device)
        self._id_target_delta_xdot_mean = torch.zeros((1, 4), device=self.device)
        self._id_target_delta_xdot_std = torch.ones((1, 4), device=self.device)
        self._id_target_delta_xdot_clip_min = torch.full((1, 4), -float("inf"), device=self.device)
        self._id_target_delta_xdot_clip_max = torch.full((1, 4), float("inf"), device=self.device)
        self._physics_dt = float(self.cfg.sim.dt)
        self._control_dt = float(self.cfg.sim.dt) * float(self.cfg.decimation)
        # The ID's TargetDeltaExtension / TargetDeltaExtensionVelocity features are defined
        # on the MODEL's sample time (the Simscape data is a constant 0.05 s), not on the
        # control step. Building them at control_dt = 1/60 s makes both features wrong: TDX is
        # 3x too small, and TDX/dt - xdot misses the data's TDXV by 407-729% of its std
        # (measured: ppo1/tests/wiring_audit_offline.py). 0 keeps the legacy behaviour.
        self._learned_model_dt = (
            float(getattr(self.cfg, "learned_model_dt_s", 0.0)) or self._control_dt
        )
        self._predicted_extension_velocity = torch.zeros_like(self._dP)
        self._raw_delta_extension = torch.zeros_like(self._dP)
        self._raw_delta_extension_velocity = torch.zeros_like(self._dP)
        self._raw_delta_dP = torch.zeros_like(self._dP)
        self._predicted_delta_extension = torch.zeros_like(self._dP)
        self._delta_isaac = torch.zeros_like(self._dP)
        self._runtime_debug_csv_header_written = False
        self._debug_manual_cylinder_position_target: torch.Tensor | None = None
        self._debug_manual_tool_position_target: torch.Tensor | None = None
        self._neutral_hold_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._fd_delta_scale = 1.0
        self._pipeline_debug_step = 0
        self._cyl_vel_error_integral = torch.zeros_like(self._cylinder_ref)
        self._servo_position_delta = torch.zeros_like(self._cylinder_ref)
        self._servo_tracking_delta = torch.zeros_like(self._cylinder_ref)
        self._servo_fd_gain = torch.ones_like(self._cylinder_ref)
        self._tracking_e_int = torch.zeros_like(self._cylinder_ref)
        self._tracking_e_prev = torch.zeros_like(self._cylinder_ref)
        self._fined_tuned_id_model = None
        self._fined_tuned_id_X_mean = None
        self._fined_tuned_id_X_std = None
        self._pos_pi_x_cmd = torch.zeros_like(self._cylinder_ref)
        self._pos_pi_e_int = torch.zeros_like(self._cylinder_ref)
        self._pos_pi_e_prev = torch.zeros_like(self._cylinder_ref)
        self._cyl_vel_error_prev = torch.zeros_like(self._cylinder_ref)
        self._cyl_vel_desired = torch.zeros_like(self._cylinder_ref)
        self._cyl_vel_measured = torch.zeros_like(self._cylinder_ref)
        self._valve_ff = torch.zeros_like(self._cylinder_ref)
        self._valve_pid = torch.zeros_like(self._cylinder_ref)
        self._valve_total = torch.zeros_like(self._cylinder_ref)
        self._fd_delta_raw = torch.zeros_like(self._cylinder_ref)
        self._fd_delta_corrected = torch.zeros_like(self._cylinder_ref)
        self._debug_cylinder_velocity_target: torch.Tensor | None = None
        self._debug_valve_cmd: torch.Tensor | None = None
        self._cyl_vel_gravity_bias = torch.zeros_like(self._cylinder_ref)
        self._gravity_bias_measured = False

        # -- EE body id --
        self._ee_body_id: int | None = None
        for i, name in enumerate(self.robot.body_names):
            if name == self.cfg.task_space_ee_body_name:
                self._ee_body_id = i
                break
        if self._ee_body_id is None:
            raise RuntimeError(f"End-effector body '{self.cfg.task_space_ee_body_name}' not found in robot body names: {self.robot.body_names}")

        # -- Arm base (Drehzapfen) body id for tcp_ahead reward --
        self._arm_base_body_id: int | None = None
        for i, name in enumerate(self.robot.body_names):
            if name == "Drehzapfen":
                self._arm_base_body_id = i
                break

        if bool(getattr(self.cfg, "enable_verbose_startup", False)):
            self._print_articulation_joint_map()
        self._cylinder_adapter: JointToCylinderAdapter | None = None
        self._action_to_cylinder_target_adapter: PPOActionToCylinderTargetAdapter | None = None
        self._cylinder_space_scaling: CylinderSpaceScaling | None = None
        self._cylinder_id: CylinderIDRuntime | None = None
        self._cylinder_rom: CylinderROMRuntime | None = None
        self._base_cylinder_rom: BaseCylinderROMRuntime | None = None
        self._command_cylinder_rom: CommandCylinderROMRuntime | None = None
        self._cylinder_fd: CylinderFDRuntime | None = None
        self._sequence_id: SequenceIDRuntime | None = None
        self._sequence_fd: SequenceFDRuntime | None = None
        self._sequence_model_dt: float = 0.05
        self._sequence_decimation: int = 1
        self._sequence_tick_counter: int = 0
        self._sequence_step_counter: int = 0
        self._sequence_tick_active: bool = True
        self._fd_delta_plan: torch.Tensor = torch.zeros((0, 4))
        self._fd_correction_mode = str(getattr(self.cfg, "fd_correction_mode", "none")).lower()
        self._fd_correction_matrix = torch.eye(4, dtype=torch.float32, device=self.device)
        self._fd_correction_bias = torch.zeros((4,), dtype=torch.float32, device=self.device)
        self._fd_valve_residual_matrix = torch.zeros((4, 4), dtype=torch.float32, device=self.device)
        if bool(getattr(self.cfg, "use_cylinder_id_pipeline", False)):
            self._setup_cylinder_id_pipeline()

        # -- Print observation and reward info at startup --
        if bool(getattr(self.cfg, "enable_startup_summary", True)):
            self._print_startup_info()

    def _load_fd_correction(self) -> None:
        mode = str(getattr(self.cfg, "fd_correction_mode", "none")).lower()
        self._fd_correction_mode = mode
        self._fd_correction_matrix = torch.eye(4, dtype=torch.float32, device=self.device)
        self._fd_correction_bias = torch.zeros((4,), dtype=torch.float32, device=self.device)
        self._fd_valve_residual_matrix = torch.zeros((4, 4), dtype=torch.float32, device=self.device)
        if mode in {"none", ""}:
            return
        if mode == "fine_tuned_checkpoint":
            print("[MulagUpperLevelEnv] FD correction mode: fine_tuned_checkpoint", flush=True)
            return
        if mode not in {"affine_calibration", "valve_residual_calibration"}:
            raise ValueError(
                "fd_correction_mode must be 'none', 'affine_calibration', "
                "'valve_residual_calibration', or "
                f"'fine_tuned_checkpoint', got {mode!r}."
            )

        path = Path(str(getattr(self.cfg, "fd_calibration_path", ""))).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"fd_correction_mode={mode!r} requires fd_calibration_path, got {path}"
            )
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
        elif path.suffix.lower() == ".csv":
            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            if not rows:
                raise ValueError(f"FD calibration CSV has no rows: {path}")
            row = rows[0]
            data = {
                "matrix": [
                    [float(row.get(f"M_{i}{j}", 1.0 if i == j else 0.0)) for j in range(4)]
                    for i in range(4)
                ],
                "bias": [float(row.get(f"b_{i}", 0.0)) for i in range(4)],
            }
        else:
            data = torch.load(path, map_location="cpu", weights_only=False)
        matrix = data.get("matrix", data.get("M"))
        bias = data.get("bias", data.get("b", [0.0, 0.0, 0.0, 0.0]))
        matrix_t = torch.as_tensor(matrix, dtype=torch.float32, device=self.device)
        bias_t = torch.as_tensor(bias, dtype=torch.float32, device=self.device)
        if matrix_t.shape != (4, 4) or bias_t.shape != (4,):
            raise ValueError(
                f"FD affine calibration must contain matrix [4,4] and bias [4], got {matrix_t.shape}, {bias_t.shape}"
            )
        self._fd_correction_matrix = matrix_t
        self._fd_correction_bias = bias_t
        valve_matrix = data.get("valve_matrix", data.get("K"))
        if valve_matrix is not None:
            valve_matrix_t = torch.as_tensor(valve_matrix, dtype=torch.float32, device=self.device)
            if valve_matrix_t.shape != (4, 4):
                raise ValueError(f"FD valve residual matrix must have shape [4,4], got {valve_matrix_t.shape}")
            self._fd_valve_residual_matrix = valve_matrix_t
        else:
            self._fd_valve_residual_matrix = torch.zeros((4, 4), dtype=torch.float32, device=self.device)
        print(f"[MulagUpperLevelEnv] FD correction loaded mode={mode}: {path}", flush=True)

    def _apply_fd_delta_correction(self, delta_extension: torch.Tensor, valve_cmd: torch.Tensor | None = None) -> torch.Tensor:
        if self._fd_correction_mode not in {"affine_calibration", "valve_residual_calibration"}:
            return delta_extension
        corrected = delta_extension @ self._fd_correction_matrix.T + self._fd_correction_bias
        if self._fd_correction_mode == "valve_residual_calibration":
            if valve_cmd is None:
                raise RuntimeError("valve_residual_calibration requires valve_cmd.")
            corrected = corrected + valve_cmd @ self._fd_valve_residual_matrix.T
        return torch.nan_to_num(corrected, nan=0.0, posinf=0.0, neginf=0.0)

    def _low_level_servo_mode(self) -> str:
        mode = str(getattr(self.cfg, "low_level_servo_mode", "none")).lower()
        valid = {
            "none",
            "cylinder_velocity_pid",
            "id_ff_plus_cylinder_velocity_pid",
            "position_ramp_pi",
            "direct_position_delta",
            "fd_tracking_servo",
            "id_pid_fd_tracking_servo",
        }
        if mode not in valid:
            raise ValueError(f"low_level_servo_mode must be one of {sorted(valid)}, got {mode!r}.")
        return mode

    def _low_level_servo_enabled(self) -> bool:
        return self._low_level_servo_mode() != "none"


    def _fd_servo_usage_mode(self) -> str:
        mode = str(getattr(self.cfg, "fd_servo_usage", "prediction_only")).lower()
        valid = {"prediction_only", "prediction_plus_feedback_delta", "bypass_for_diagnostic_only"}
        if mode not in valid:
            raise ValueError(f"fd_servo_usage must be one of {sorted(valid)}, got {mode!r}.")
        return mode

    def _apply_fd_servo_bypass(
        self,
        delta_extension: torch.Tensor,
        reference,
        step_dt: float,
    ) -> torch.Tensor:
        """Apply fd_servo_usage bypass when low_level_servo is enabled.

        Returns modified delta_extension.

        - prediction_only: no modification (FD output used directly).
        - prediction_plus_feedback_delta: adds Kx * (v_des - v_meas) * dt.
        - bypass_for_diagnostic_only: replaces with v_des * dt (DIAGNOSTIC ONLY).
        """
        usage = self._fd_servo_usage_mode()
        if not self._low_level_servo_enabled():
            return delta_extension
        if usage == "prediction_only":
            return delta_extension
        if usage == "bypass_for_diagnostic_only":
            v_des = self._cyl_vel_desired.to(device=self.device, dtype=torch.float32)
            bypass = v_des * float(step_dt)
            self._fd_delta_corrected = bypass.detach().clone()
            return bypass
        if usage == "prediction_plus_feedback_delta":
            vel_error = self._cyl_vel_desired - self._cyl_vel_measured
            correction_gain = float(getattr(self.cfg, "servo_fd_correction_gain", 0.0))
            if correction_gain != 0.0:
                corrected = delta_extension + correction_gain * vel_error * float(step_dt)
                self._fd_delta_corrected = corrected.detach().clone()
                return corrected
            # fall through to existing servo_tracking_delta_gain path in the caller
            return delta_extension
        return delta_extension
    def _load_fined_tuned_id_model(self):
        if self._fined_tuned_id_model is not None:
            return
        override = str(getattr(self.cfg, "id_checkpoint_override_path", ""))
        if not override:
            return
        from pathlib import Path
        ckpt_path = Path(override).expanduser().resolve()
        if not ckpt_path.is_file():
            return
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        from finetune_id_physx_dynamic import VelocityToValveMLP
        self._fined_tuned_id_model = VelocityToValveMLP().to(self.device)
        self._fined_tuned_id_model.load_state_dict(ckpt["model_state_dict"])
        self._fined_tuned_id_model.eval()
        self._fined_tuned_id_X_mean = torch.from_numpy(ckpt["X_mean"].astype("float32")).to(self.device)
        self._fined_tuned_id_X_std = torch.from_numpy(ckpt["X_std"].astype("float32")).to(self.device)

    def _compute_fined_tuned_id_valve(self, v_des, x_cur, xdot_cur, dt):
        self._load_fined_tuned_id_model()
        if self._fined_tuned_id_model is None:
            return torch.zeros_like(v_des)
        dd = v_des * dt
        id_in = torch.cat([dd, v_des, x_cur, xdot_cur], dim=-1)
        id_in_n = (id_in - self._fined_tuned_id_X_mean) / self._fined_tuned_id_X_std
        with torch.no_grad():
            valve = self._fined_tuned_id_model(id_in_n)
        return torch.nan_to_num(valve, nan=0.0, posinf=0.0, neginf=0.0)

    def _cfg_cyl_vector(self, name: str, default: float) -> torch.Tensor:
        value = getattr(self.cfg, name, [default, default, default, default])
        tensor = torch.as_tensor(value, dtype=torch.float32, device=self.device).reshape(-1)
        if tensor.numel() == 1:
            tensor = tensor.repeat(4)
        if tensor.numel() != 4:
            raise ValueError(f"{name} must contain one or four values, got {tensor.numel()}.")
        return tensor.reshape(1, 4)

    def set_debug_cylinder_velocity_target(self, target_mps: torch.Tensor | None) -> None:
        """Override the upper-level action with a direct desired cylinder velocity."""
        if target_mps is None:
            self._debug_cylinder_velocity_target = None
            return
        target = torch.as_tensor(target_mps, dtype=torch.float32, device=self.device)
        if target.shape == (4,):
            target = target.reshape(1, 4).expand(self.num_envs, -1).clone()
        if target.shape != (self.num_envs, 4):
            raise ValueError(
                f"debug cylinder velocity target must have shape (4,) or ({self.num_envs}, 4), got {tuple(target.shape)}."
            )
        self._debug_cylinder_velocity_target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)


    def set_debug_valve_cmd(self, valve_cmd: torch.Tensor | None) -> None:
        """Override valve command directly (bypasses ID and servo for FD data collection)."""
        if valve_cmd is None:
            self._debug_valve_cmd = None
            return
        vc = torch.as_tensor(valve_cmd, dtype=torch.float32, device=self.device)
        if vc.shape == (4,):
            vc = vc.reshape(1, 4).expand(self.num_envs, -1).clone()
        if vc.shape != (self.num_envs, 4):
            raise ValueError(f"debug valve cmd must have shape (4,) or ({self.num_envs}, 4), got {tuple(vc.shape)}.")
        self._debug_valve_cmd = torch.nan_to_num(vc, nan=0.0, posinf=0.0, neginf=0.0)

    def clear_debug_valve_cmd(self) -> None:
        self._debug_valve_cmd = None
    def clear_debug_cylinder_velocity_target(self) -> None:
        self._debug_cylinder_velocity_target = None

    def _maybe_override_reference_for_debug_velocity(
        self,
        reference: CylinderReference,
    ) -> CylinderReference:
        if self._debug_cylinder_velocity_target is None:
            return reference
        limit = float(getattr(self.cfg, "cyl_vel_cmd_limit", 0.0))
        desired = self._debug_cylinder_velocity_target.to(device=self.device, dtype=torch.float32)
        if limit > 0.0:
            desired = torch.clamp(desired, -limit, limit)
        target_delta_x = desired * float(self._learned_model_dt)
        return CylinderReference(
            x_current=reference.x_current,
            xdot_current=reference.xdot_current,
            x_ref=reference.x_current + target_delta_x,
            xdot_ref=desired,
            target_delta_x=target_delta_x,
            target_delta_xdot=desired - reference.xdot_current,
        )

    def _apply_cylinder_velocity_servo(
        self,
        *,
        reference: CylinderReference,
        valve_ff: torch.Tensor,
        neutral_hold_mask: torch.Tensor,
        step_dt: float,
    ) -> torch.Tensor:
        mode = self._low_level_servo_mode()

        # -- Direct measured-error position increment -----------------------
        # The rank-promoted FD is retained for diagnostics, but its residual
        # valve effect is too small for large excursions and its full output
        # incorrectly injects state dynamics into the position-target
        # integrator.  In this mode the upstream cylinder controller owns the
        # closed loop and action_target already is the desired bounded learned
        # cylinder increment. PhysX remains the plant.
        if mode == "direct_position_delta":
            dx_cmd = reference.target_delta_x.to(device=self.device, dtype=torch.float32)
            clip_m = max(float(getattr(self.cfg, "direct_position_delta_clip_m", 0.0015)), 0.0)
            if clip_m > 0.0:
                dx_cmd = torch.clamp(dx_cmd, -clip_m, clip_m)
            if neutral_hold_mask.any():
                dx_cmd = dx_cmd.clone()
                dx_cmd[neutral_hold_mask] = 0.0
            self._servo_position_delta = dx_cmd.detach().clone()
            self._cyl_vel_desired = dx_cmd.detach().clone() / max(float(step_dt), 1.0e-9)
            self._cyl_vel_measured = reference.xdot_current.detach().clone()
            self._valve_ff.zero_()
            self._valve_pid.zero_()
            self._valve_total.zero_()
            return self._valve_total

        # -- Position-ramp PI mode: directly produces position deltas, bypasses ID/FD --
        if mode == "position_ramp_pi":
            x_meas = reference.x_current.to(device=self.device, dtype=torch.float32)
            x_meas = torch.nan_to_num(x_meas, nan=0.0, posinf=0.0, neginf=0.0)
            desired = reference.xdot_ref.to(device=self.device, dtype=torch.float32)
            desired = torch.nan_to_num(desired, nan=0.0, posinf=0.0, neginf=0.0)
            cmd_limit = float(getattr(self.cfg, "cyl_vel_cmd_limit", 0.0))
            if cmd_limit > 0.0:
                desired = torch.clamp(desired, -cmd_limit, cmd_limit)
            v_meas = reference.xdot_current.to(device=self.device, dtype=torch.float32)
            v_meas = torch.nan_to_num(v_meas, nan=0.0, posinf=0.0, neginf=0.0)

            # Integrate position ramp from desired velocity

            # Lazy-init: sync x_cmd with current position on first call after reset
            if self._pos_pi_x_cmd.abs().sum() < 1e-12:
                self._pos_pi_x_cmd = x_meas.detach().clone()
            self._pos_pi_x_cmd = self._pos_pi_x_cmd + desired * float(step_dt)

            # Position PI control
            e_pos = self._pos_pi_x_cmd - x_meas
            self._pos_pi_e_int = self._pos_pi_e_int + e_pos * float(step_dt)
            pi_ilim = max(float(getattr(self.cfg, "pos_pi_integral_limit_m", 0.01)), 0.0)
            if pi_ilim > 0:
                self._pos_pi_e_int = torch.clamp(self._pos_pi_e_int, -pi_ilim, pi_ilim)
            de_pos = (e_pos - self._pos_pi_e_prev) / max(float(step_dt), 1e-9)
            self._pos_pi_e_prev = e_pos.detach().clone()

            kp_p = self._cfg_cyl_vector("pos_pi_kp", 0.5)
            ki_p = self._cfg_cyl_vector("pos_pi_ki", 0.2)
            kd_p = self._cfg_cyl_vector("pos_pi_kd", 0.0)
            dx_cmd = kp_p * e_pos + ki_p * self._pos_pi_e_int + kd_p * de_pos
            dclip_m = max(float(getattr(self.cfg, "pos_pi_delta_clip_m", 0.005)), 0.0)
            if dclip_m > 0:
                dx_cmd = torch.clamp(dx_cmd, -dclip_m, dclip_m)

            if neutral_hold_mask.any():
                dx_cmd = dx_cmd.clone()
                self._pos_pi_e_int = self._pos_pi_e_int.clone()
                dx_cmd[neutral_hold_mask] = 0.0

            # Store output and bridge to downstream pipeline
            self._servo_position_delta = dx_cmd.detach().clone()
            self._cyl_vel_desired = desired.detach().clone()
            self._cyl_vel_measured = v_meas.detach().clone()
            self._cyl_vel_error_prev = (desired - v_meas).detach().clone()
            self._valve_ff.zero_()
            self._valve_pid.zero_()
            self._valve_total.zero_()
            return self._valve_total  # zero — position delta replaces FD output
        # -- FD tracking servo: uses fine-tuned ID + original FD + tracking correction --
        if mode in ("fd_tracking_servo", "id_pid_fd_tracking_servo"):
            desired = reference.xdot_ref.to(device=self.device, dtype=torch.float32)
            measured = reference.xdot_current.to(device=self.device, dtype=torch.float32)
            desired = torch.nan_to_num(desired, nan=0.0, posinf=0.0, neginf=0.0)
            measured = torch.nan_to_num(measured, nan=0.0, posinf=0.0, neginf=0.0)
            cmd_limit = float(getattr(self.cfg, "cyl_vel_cmd_limit", 0.0))
            if cmd_limit > 0.0:
                desired = torch.clamp(desired, -cmd_limit, cmd_limit)

            # Fine-tuned ID feedforward: desired_velocity -> valve command
            valve_ff = self._compute_fined_tuned_id_valve(
                desired, reference.x_current, reference.xdot_current, float(step_dt)
            )

            # Valve-space PID correction
            e_v = desired - measured
            deadband = max(float(getattr(self.cfg, "servo_deadband_mps", 0.0)), 0.0)
            if deadband > 0.0:
                e_v = torch.where(e_v.abs() <= deadband, torch.zeros_like(e_v), e_v)
            self._tracking_e_int = self._tracking_e_int + e_v * float(step_dt)
            ilim = max(float(getattr(self.cfg, "tracking_integral_limit", 0.5)), 0.0)
            if ilim > 0.0:
                self._tracking_e_int = torch.clamp(self._tracking_e_int, -ilim, ilim)
            de_v = (e_v - self._tracking_e_prev) / max(float(step_dt), 1e-9)
            self._tracking_e_prev = e_v.detach().clone()

            kp_v = float(getattr(self.cfg, "tracking_kp", 40.0))
            ki_v = float(getattr(self.cfg, "tracking_ki", 15.0))
            kd_v = float(getattr(self.cfg, "tracking_kd", 2.0))
            valve_pid = kp_v * e_v + ki_v * self._tracking_e_int + kd_v * de_v

            if mode == "fd_tracking_servo":
                valve_pid = torch.zeros_like(valve_pid)  # PID disabled

            valve_limit = float(getattr(self.cfg, "valve_cmd_limit", 1.0))
            valve_total = torch.clamp(valve_ff + valve_pid, -valve_limit, valve_limit)

            if neutral_hold_mask.any():
                valve_ff = valve_ff.clone()
                valve_pid = valve_pid.clone()
                valve_total = valve_total.clone()
                valve_ff[neutral_hold_mask] = 0.0
                valve_pid[neutral_hold_mask] = 0.0
                valve_total[neutral_hold_mask] = 0.0

            # Store tracking delta for downstream correction
            fd_g = float(getattr(self.cfg, "fd_gain", 1.0))
            tr_g = float(getattr(self.cfg, "tracking_gain", 0.0))
            self._servo_fd_gain.fill_(fd_g)
            self._servo_tracking_delta = tr_g * e_v * float(step_dt)
            tclip = max(float(getattr(self.cfg, "tracking_delta_clip_m", 0.01)), 0.0)
            if tclip > 0.0:
                self._servo_tracking_delta = torch.clamp(self._servo_tracking_delta, -tclip, tclip)

            self._cyl_vel_desired = desired.detach().clone()
            self._cyl_vel_measured = measured.detach().clone()
            self._cyl_vel_error_prev = e_v.detach().clone()
            self._valve_ff = valve_ff.detach().clone()
            self._valve_pid = valve_pid.detach().clone()
            self._valve_total = valve_total.detach().clone()
            return valve_total

        desired = reference.xdot_ref.to(device=self.device, dtype=torch.float32)
        measured = reference.xdot_current.to(device=self.device, dtype=torch.float32)
        desired = torch.nan_to_num(desired, nan=0.0, posinf=0.0, neginf=0.0)
        measured = torch.nan_to_num(measured, nan=0.0, posinf=0.0, neginf=0.0)
        cmd_limit = float(getattr(self.cfg, "cyl_vel_cmd_limit", 0.0))
        if cmd_limit > 0.0:
            desired = torch.clamp(desired, -cmd_limit, cmd_limit)


        error = desired - measured
        deadband = max(float(getattr(self.cfg, "servo_deadband_mps", 0.0)), 0.0)
        if deadband > 0.0:
            error = torch.where(error.abs() <= deadband, torch.zeros_like(error), error)

        self._cyl_vel_error_integral = self._cyl_vel_error_integral + error * float(step_dt)
        integral_limit = max(float(getattr(self.cfg, "cyl_vel_integral_limit", 0.0)), 0.0)
        if integral_limit > 0.0:
            self._cyl_vel_error_integral = torch.clamp(
                self._cyl_vel_error_integral,
                -integral_limit,
                integral_limit,
            )
        derivative = (error - self._cyl_vel_error_prev) / max(float(step_dt), 1.0e-9)

        kp = self._cfg_cyl_vector("cyl_vel_kp", 0.0)
        ki = self._cfg_cyl_vector("cyl_vel_ki", 0.0)
        kd = self._cfg_cyl_vector("cyl_vel_kd", 0.0)
        valve_pid = kp * error + ki * self._cyl_vel_error_integral + kd * derivative
        valve_ff = torch.zeros_like(valve_ff) if mode == "cylinder_velocity_pid" else valve_ff

        valve_limit = float(getattr(self.cfg, "valve_cmd_limit", getattr(self.cfg, "max_abs_valve", 1.0)))
        valve_limit = min(abs(valve_limit), abs(float(getattr(self.cfg, "max_abs_valve", valve_limit))))
        valve_total_preclip = valve_ff + valve_pid
        valve_total = torch.clamp(valve_total_preclip, -valve_limit, valve_limit)

        rate_limit = max(float(getattr(self.cfg, "servo_rate_limit", 0.0)), 0.0)
        if rate_limit > 0.0:
            delta = torch.clamp(valve_total - self._prev_valve_cmd, -rate_limit, rate_limit)
            valve_total = self._prev_valve_cmd + delta
            valve_total = torch.clamp(valve_total, -valve_limit, valve_limit)

        if neutral_hold_mask.any():
            valve_ff = valve_ff.clone()
            valve_pid = valve_pid.clone()
            valve_total = valve_total.clone()
            error = error.clone()
            # Keep integral on neutral hold — it is the anti-gravity memory
            valve_ff[neutral_hold_mask] = 0.0
            valve_pid[neutral_hold_mask] = 0.0
            valve_total[neutral_hold_mask] = 0.0
            error[neutral_hold_mask] = 0.0

        self._cyl_vel_desired = desired.detach().clone()
        self._cyl_vel_measured = measured.detach().clone()
        self._cyl_vel_error_prev = error.detach().clone()
        self._valve_ff = valve_ff.detach().clone()
        self._valve_pid = valve_pid.detach().clone()
        self._valve_total = valve_total.detach().clone()
        self._gravity_bias_measured = True
        return valve_total


    def measure_gravity_drift(self, steps: int = 50) -> torch.Tensor:
        """Measure cylinder velocity drift under zero command (gravity bias).

        Returns gravity bias [1, 4] in learned cylinder velocity units.
        To compensate, add this bias to desired velocity: v_des_corrected = v_des - bias.
        """
        saved_target = self._debug_cylinder_velocity_target
        saved_servo = self.cfg.low_level_servo_mode
        zero_target = torch.zeros((self.num_envs, 4), device=self.device)
        self.set_debug_cylinder_velocity_target(zero_target)
        self.cfg.low_level_servo_mode = "none"

        velocities = []
        for _step in range(steps):
            obs, _reward, _term, _trunc, _info = self.step(
                torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
            )
            cyl_qdot = self.robot.data.joint_vel[:, self._cylinder_joint_ids]
            _x_current, xdot_current = self._cylinder_adapter.current_from_cylinder_joints(
                torch.zeros_like(cyl_qdot), cyl_qdot
            )
            velocities.append(xdot_current.detach().clone())

        self.cfg.low_level_servo_mode = saved_servo
        self.set_debug_cylinder_velocity_target(saved_target)

        avg_velocity = torch.stack(velocities, dim=0).mean(dim=0)
        self._cyl_vel_gravity_bias = -avg_velocity.detach().clone()
        self._gravity_bias_measured = True
        return self._cyl_vel_gravity_bias.clone()
    def _find_joint_ids(self, names):
        ids = []
        for n in names:
            found = False
            for i, jn in enumerate(self.robot.joint_names):
                if jn == n:
                    ids.append(i)
                    found = True
                    break
            if not found:
                raise RuntimeError(f"Joint '{n}' not found in robot joint names: {self.robot.joint_names}")
        return torch.tensor(ids, dtype=torch.long, device=self.device)

    def set_debug_manual_cylinder_targets(
        self,
        cylinder_position_target: torch.Tensor,
        *,
        tool_position_target: torch.Tensor | None = None,
    ) -> None:
        """Pin cylinder position targets for diagnostics that bypass PPO/ID/FD.

        This is intentionally gated by config so production training cannot
        accidentally bypass the PPO1 -> ID -> FD -> PhysX path.
        """

        if not bool(getattr(self.cfg, "allow_debug_manual_cylinder_targets", False)):
            raise RuntimeError("Debug manual cylinder targets are disabled by config.")
        target = cylinder_position_target.to(device=self.device, dtype=torch.float32)
        if target.shape != (self.num_envs, 4):
            raise ValueError(
                f"cylinder_position_target must have shape ({self.num_envs}, 4), got {tuple(target.shape)}."
            )
        if not torch.isfinite(target).all():
            raise FloatingPointError("cylinder_position_target contains NaN/Inf.")
        self._debug_manual_cylinder_position_target = target.detach().clone()
        self._cylinder_pos_target = target.detach().clone()

        if tool_position_target is None:
            self._debug_manual_tool_position_target = None
            return
        tool_target = tool_position_target.to(device=self.device, dtype=torch.float32)
        if tool_target.shape != (self.num_envs, 1):
            raise ValueError(
                f"tool_position_target must have shape ({self.num_envs}, 1), got {tuple(tool_target.shape)}."
            )
        if not torch.isfinite(tool_target).all():
            raise FloatingPointError("tool_position_target contains NaN/Inf.")
        self._debug_manual_tool_position_target = tool_target.detach().clone()

    def clear_debug_manual_cylinder_targets(self) -> None:
        self._debug_manual_cylinder_position_target = None
        self._debug_manual_tool_position_target = None

    def _resolve_joint6_coupling(self) -> None:
        if not bool(getattr(self.cfg, "enable_joint6_coupling", False)):
            return

        arm_names = tuple(self.cfg.arm_joint_names)
        cylinder_names = tuple(self.cfg.cylinder_joint_names)
        if len(cylinder_names) != 4:
            raise RuntimeError(
                "Joint6 coupling expects exactly four cylinder-controlled joints. "
                f"Got {len(cylinder_names)}: {cylinder_names}."
            )

        source_name = str(getattr(self.cfg, "joint6_coupling_source_joint_name", "") or cylinder_names[3])
        target_name = str(getattr(self.cfg, "joint6_coupling_target_joint_name", ""))
        if not target_name:
            passive_candidates = [name for name in arm_names if name not in cylinder_names]
            if len(passive_candidates) != 1:
                raise RuntimeError(
                    "Cannot infer joint6 coupling target without guessing. Set "
                    "joint6_coupling_target_joint_name explicitly. "
                    f"Non-cylinder arm candidates: {passive_candidates}."
                )
            target_name = passive_candidates[0]

        if source_name not in cylinder_names:
            raise RuntimeError(
                "Joint6 coupling source must be one of the four cylinder-controlled joints. "
                f"Got {source_name!r}, cylinder joints are {cylinder_names}."
            )
        if source_name not in arm_names:
            raise RuntimeError(f"Joint6 coupling source {source_name!r} is not in arm_joint_names: {arm_names}.")
        if target_name not in arm_names:
            raise RuntimeError(f"Joint6 coupling target {target_name!r} is not in arm_joint_names: {arm_names}.")
        if target_name in cylinder_names:
            raise RuntimeError(
                "Joint6 coupling target must not be cylinder-controlled. "
                f"Got {target_name!r}, cylinder joints are {cylinder_names}."
            )

        self._joint6_coupling_source_name = source_name
        self._joint6_coupling_target_name = target_name
        self._joint6_coupling_source_ref_index = arm_names.index(source_name)
        self._joint6_coupling_target_ref_index = arm_names.index(target_name)
        source_joint_tensor = self._arm_joint_ids[self._joint6_coupling_source_ref_index]
        target_joint_tensor = self._arm_joint_ids[self._joint6_coupling_target_ref_index]
        self._joint6_coupling_source_joint_id = int(source_joint_tensor.detach().cpu().item())
        self._joint6_coupling_target_joint_id = int(target_joint_tensor.detach().cpu().item())
        self._tool_joint_id = self._joint6_coupling_target_joint_id

    def _usd_joint_type_by_name(self) -> tuple[dict[str, str], bool]:
        usd_path = getattr(getattr(self.cfg.robot_cfg, "spawn", None), "usd_path", None)
        if not usd_path:
            return {}, False
        try:
            from pxr import Usd
        except Exception as exc:
            print(f"[MulagUpperLevelEnv] USD joint type scan unavailable: {exc}", flush=True)
            return {}, False
        stage = Usd.Stage.Open(str(usd_path))
        if stage is None:
            print(f"[MulagUpperLevelEnv] Could not open robot USD for joint type scan: {usd_path}", flush=True)
            return {}, False
        joint_types: dict[str, str] = {}
        found_mimic_or_gear = False
        for prim in stage.Traverse():
            type_name = prim.GetTypeName()
            name = prim.GetName()
            if "Joint" in type_name or name.endswith("_joint"):
                joint_types[name] = type_name
                for attr in prim.GetAttributes():
                    if any(key in attr.GetName().lower() for key in ("mimic", "gear")):
                        found_mimic_or_gear = True
                        print(
                            f"[MulagUpperLevelEnv] USD mimic/gear attr on {prim.GetPath()}: "
                            f"{attr.GetName()}={attr.Get()}",
                            flush=True,
                        )
                for rel in prim.GetRelationships():
                    if any(key in rel.GetName().lower() for key in ("mimic", "gear")):
                        found_mimic_or_gear = True
                        print(
                            f"[MulagUpperLevelEnv] USD mimic/gear rel on {prim.GetPath()}: "
                            f"{rel.GetName()}={[str(x) for x in rel.GetTargets()]}",
                            flush=True,
                        )
        if not found_mimic_or_gear:
            print("[MulagUpperLevelEnv] No USD mimic/gear attributes or relationships found in joint prims.", flush=True)
        return joint_types, found_mimic_or_gear


    def _debug_interval_due(self) -> bool:
        if bool(getattr(self.cfg, "pipeline_debug_prints", False)) and not bool(
            getattr(self.cfg, "enable_pipeline_debug", False)
        ):
            interval = int(getattr(self.cfg, "pipeline_debug_interval", 1000))
        else:
            interval = int(getattr(self.cfg, "debug_log_interval", 1000))
        interval = max(interval, 1)
        return self._pipeline_debug_step % interval == 0

    def _selected_debug_env_ids(self) -> list[int]:
        configured = tuple(getattr(self.cfg, "debug_env_ids", (0,)) or (0,))
        max_count = max(int(getattr(self.cfg, "debug_max_envs_to_print", 1)), 1)
        selected: list[int] = []
        for env_id in configured:
            try:
                eid = int(env_id)
            except Exception:
                continue
            if 0 <= eid < self.num_envs and eid not in selected:
                selected.append(eid)
            if len(selected) >= max_count:
                break
        return selected or [0]

    def _format_debug_list(self, tensor: torch.Tensor, env_id: int) -> list[float | bool]:
        row = tensor[env_id].detach().cpu().reshape(-1).tolist()
        return row

    def _print_startup_info(self):
        """Print a compact startup summary; verbose details are opt-in."""
        cfg = self.cfg
        print("[MulagUpperLevelEnv] startup summary", flush=True)
        print(f"  task/env: {self.__class__.__name__}", flush=True)
        print(f"  num_envs: {self.num_envs}", flush=True)
        print(f"  device: {self.device}", flush=True)
        print(f"  action_space: {cfg.action_space}", flush=True)
        print(f"  observation_space: {cfg.observation_space}", flush=True)
        print(f"  action_scale: {cfg.action_scale}", flush=True)
        print(f"  model_dt/control_dt/decimation: {self._learned_model_dt:.9g} / {self._control_dt:.9g} / {self._sequence_decimation}", flush=True)
        use_sequence = bool(getattr(cfg, "use_sequence_cylinder_models", False))
        print(f"  sequence ID enabled: {use_sequence and self._sequence_id is not None}", flush=True)
        print(f"  sequence FD enabled: {use_sequence and self._sequence_fd is not None}", flush=True)
        print("  active cylinders:", flush=True)
        for name, joint_id in zip(cfg.cylinder_joint_names, self._cylinder_joint_ids.detach().cpu().tolist()):
            print(f"    {name} -> joint[{int(joint_id)}]", flush=True)
        if self._joint6_coupling_target_ref_index is not None:
            print(
                "  passive joint handling: "
                f"{self._joint6_coupling_target_name} coupled outside ID/FD from {self._joint6_coupling_source_name}",
                flush=True,
            )
        else:
            print("  passive joint handling: none configured", flush=True)
        if not bool(getattr(cfg, "enable_verbose_startup", False)):
            return
        print("  verbose startup details:", flush=True)
        print(f"    episode_length_s: {cfg.episode_length_s}", flush=True)
        if bool(getattr(cfg, "use_cylinder_id_pipeline", False)):
            print(f"    cylinder pipeline: enabled", flush=True)
            print(f"    dynamics_model: {cfg.cylinder_dynamics_model}", flush=True)
            print(f"    ID checkpoint: {self._cylinder_id.checkpoint_path if self._cylinder_id else 'N/A'}", flush=True)
            print(f"    ID input_dim: {self._cylinder_id.input_dim if self._cylinder_id else 'N/A'}", flush=True)
            print(f"    ID target_mode: {self._cylinder_id.target_mode if self._cylinder_id else 'N/A'}", flush=True)
            if self._cylinder_fd is not None:
                print(f"    FD checkpoint: {self._cylinder_fd.checkpoint_path}", flush=True)
                print(f"    FD sample_time_s: {self._cylinder_fd.sample_time_s}", flush=True)
                print(f"    PhysX dt: {self._physics_dt:.9g}", flush=True)
                print(f"    Env control dt: {self._control_dt:.9g}", flush=True)
                print(f"    Learned model dt: {self._learned_model_dt:.9g}", flush=True)
                print(f"    FD temporal scale: {self._control_dt / max(self._cylinder_fd.sample_time_s, 1.0e-6):.9g}", flush=True)
        print(f"    observation components:", flush=True)
        print(f"    joint_pos_history: {ARM_JOINT_DIM}x{HISTORY} = {ARM_JOINT_DIM * HISTORY} dims", flush=True)
        print(f"    joint_vel_history: {ARM_JOINT_DIM}x{HISTORY} = {ARM_JOINT_DIM * HISTORY} dims", flush=True)
        print(f"    distance_to_goal_history: {DISTANCE_TO_GOAL_DIM}x{HISTORY} = {DISTANCE_TO_GOAL_DIM * HISTORY} dims", flush=True)
        print(f"    joint_torques: {ARM_JOINT_DIM} dims", flush=True)
        print(f"    last_action: {ARM_JOINT_DIM} dims", flush=True)
        print(f"    unimog_pose: {UNIMOG_POSE_DIM} dims", flush=True)
        print(f"    unimog_pose_dot: {UNIMOG_POSE_DOT_DIM} dims", flush=True)
        total = ARM_JOINT_DIM * HISTORY + ARM_JOINT_DIM * HISTORY + DISTANCE_TO_GOAL_DIM * HISTORY + ARM_JOINT_DIM + ARM_JOINT_DIM + UNIMOG_POSE_DIM + UNIMOG_POSE_DOT_DIM
        print(f"    Total: {total} dims", flush=True)
        print(f"    reward terms:", flush=True)
        print(f"    rew_position_tracking_linear: {cfg.rew_position_tracking_linear}", flush=True)
        print(f"    rew_position_tracking_tanh: {cfg.rew_position_tracking_tanh}", flush=True)
        print(f"    rew_tcp_ahead: {cfg.rew_tcp_ahead}", flush=True)
        print(f"    rew_yaw_tracking: {cfg.rew_yaw_tracking}", flush=True)
        print(f"    rew_finish_progress: {cfg.rew_finish_progress}", flush=True)
        print(f"    rew_action_rate: {cfg.rew_action_rate}", flush=True)
        print(f"    rew_joint_vel: {cfg.rew_joint_vel}", flush=True)
    def _print_articulation_joint_map(self):
        joint_types, _ = self._usd_joint_type_by_name()
        limits = self.robot.data.soft_joint_pos_limits
        vel_limits = self.robot.data.joint_vel_limits
        effort_limits = self.robot.data.joint_effort_limits
        print("[MulagUpperLevelEnv] Available articulation DOFs:", flush=True)
        for index, name in enumerate(self.robot.joint_names):
            limit = limits[0, index].detach().cpu().tolist() if limits.ndim == 3 else [float("nan"), float("nan")]
            if vel_limits.ndim == 2:
                vel_limit = float(vel_limits[0, index].detach().cpu())
            else:
                vel_limit = float(vel_limits[index].detach().cpu())
            if effort_limits.ndim == 2:
                effort_limit = float(effort_limits[0, index].detach().cpu())
            else:
                effort_limit = float(effort_limits[index].detach().cpu())
            joint_type = joint_types.get(name, "unknown")
            print(
                f"[MulagUpperLevelEnv]   dof[{index:02d}] name={name} type={joint_type} "
                f"limits=[{float(limit[0]):.6g}, {float(limit[1]):.6g}] "
                f"vel_limit={vel_limit:.6g} effort_limit={effort_limit:.6g}",
                flush=True,
            )
        print("[MulagUpperLevelEnv] Explicit arm joint map:", flush=True)
        for name, joint_id in zip(self.cfg.arm_joint_names, self._arm_joint_ids.detach().cpu().tolist()):
            print(f"[MulagUpperLevelEnv]   arm {name} -> joint[{int(joint_id)}]", flush=True)
        print("[MulagUpperLevelEnv] Active 4-cylinder joint map:", flush=True)
        for name, joint_id in zip(self.cfg.cylinder_joint_names, self._cylinder_joint_ids.detach().cpu().tolist()):
            print(f"[MulagUpperLevelEnv]   cylinder {name} -> joint[{int(joint_id)}]", flush=True)
        if self._joint6_coupling_target_ref_index is not None:
            print("[MulagUpperLevelEnv] Passive joint6 coupling:", flush=True)
            print(
                f"[MulagUpperLevelEnv]   source/fourth cylinder: {self._joint6_coupling_source_name} "
                f"ref_index={self._joint6_coupling_source_ref_index} "
                f"dof={self._joint6_coupling_source_joint_id}",
                flush=True,
            )
            print(
                f"[MulagUpperLevelEnv]   target/passive joint: {self._joint6_coupling_target_name} "
                f"ref_index={self._joint6_coupling_target_ref_index} "
                f"dof={self._joint6_coupling_target_joint_id}",
                flush=True,
            )
            print(
                f"[MulagUpperLevelEnv]   q6 = {float(self.cfg.joint6_coupling_a)} * q4 "
                f"+ {float(self.cfg.joint6_coupling_b)}",
                flush=True,
            )

    def _setup_cylinder_id_pipeline(self):
        mapping = load_coordinate_mapping(
            self.cfg.cylinder_coordinate_mapping,
            mode=self.cfg.cylinder_coordinate_mapping_mode,
            allow_provisional_range_mapping=bool(self.cfg.allow_provisional_range_mapping),
            allow_provisional_nominal_reference=bool(self.cfg.allow_provisional_nominal_reference),
        )
        usd_limits = self.robot.data.soft_joint_pos_limits[0, self._cylinder_joint_ids].detach()
        if bool(getattr(self.cfg, "enable_cylinder_space_scaling", True)):
            self._cylinder_space_scaling = CylinderSpaceScaling.from_coordinate_mapping(
                mapping=mapping,
                cylinder_names=self.cfg.cylinder_joint_names,
                dof_indices=self._cylinder_joint_ids,
                usd_limits=usd_limits,
            )
        else:
            self._cylinder_space_scaling = None
        joint_reference_indices: list[int] = []
        for cylinder_name in self.cfg.cylinder_joint_names:
            if cylinder_name not in self.cfg.arm_joint_names:
                raise RuntimeError(
                    "Cylinder-space adapter requires every configured cylinder joint to appear in "
                    f"arm_joint_names. Missing {cylinder_name!r}."
                )
            joint_reference_indices.append(tuple(self.cfg.arm_joint_names).index(cylinder_name))
        self._cylinder_adapter = JointToCylinderAdapter(
            mapping=mapping if bool(getattr(self.cfg, "enable_cylinder_space_scaling", True)) else None,
            scaling=self._cylinder_space_scaling,
            joint_reference_indices=joint_reference_indices,
        )
        self._validate_cylinder_usd_scaling()

        dynamics_model = str(self.cfg.cylinder_dynamics_model).lower()
        rom_only = dynamics_model in {"base_rom", "command_rom"}
        # The base hydraulic ROM is intentionally independent of both inverse
        # and forward ID/FD models. Its valve command comes from the measured
        # cylinder velocity servo below.
        if rom_only and bool(getattr(self.cfg, "use_sequence_cylinder_models", False)):
            raise ValueError("ROM-only dynamics cannot enable sequence ID/FD models.")

        # -- Sequence model branch (HistoryGRU ID + FD) --
        use_sequence = bool(getattr(self.cfg, "use_sequence_cylinder_models", False)) and not rom_only

        if use_sequence:
            # Resolve sequence checkpoint paths
            id_ckpt_rel = str(getattr(self.cfg, "sequence_id_checkpoint", ""))
            fd_ckpt_rel = str(getattr(self.cfg, "sequence_fd_checkpoint", ""))
            models_root = Path(str(self.cfg.cylinder_models_root)).expanduser()
            id_ckpt_path = (models_root / id_ckpt_rel).resolve()
            fd_ckpt_path = (models_root / fd_ckpt_rel).resolve()
            fd_override = str(getattr(self.cfg, "fd_checkpoint_override", ""))
            if fd_override:
                fd_ckpt_path = Path(fd_override).expanduser().resolve()
                print(f"[MulagUpperLevelEnv] FD checkpoint overridden: {fd_ckpt_path}", flush=True)

            self._sequence_id = SequenceIDRuntime(
                checkpoint_path=id_ckpt_path,
                num_envs=self.num_envs,
                device=self.device,
                verbose=bool(getattr(self.cfg, "enable_verbose_startup", False)),
            )
            if dynamics_model == "fd":
                self._sequence_fd = SequenceFDRuntime(
                    checkpoint_path=fd_ckpt_path,
                    num_envs=self.num_envs,
                    device=self.device,
                    verbose=bool(getattr(self.cfg, "enable_verbose_startup", False)),
                )
                self._load_fd_correction()
            elif dynamics_model == "rom":
                self._cylinder_rom = CylinderROMRuntime(
                    models_root=self.cfg.cylinder_models_root,
                    checkpoint=self.cfg.rom_cylinder_checkpoint,
                    num_envs=self.num_envs,
                    device=self.device,
                )
            elif dynamics_model in {"direct_valve", "direct"}:
                pass
            else:
                raise ValueError(
                    "cylinder_dynamics_model must be 'base_rom', 'rom', 'fd', or 'direct_valve', "
                    f"got {self.cfg.cylinder_dynamics_model!r}."
                )

            # Determine sequence model dt
            seq_dt_cfg = str(getattr(self.cfg, "sequence_model_dt", "auto_from_metadata"))
            if seq_dt_cfg == "auto_from_metadata":
                self._sequence_model_dt = float(self._sequence_id.sample_time_s)
            else:
                self._sequence_model_dt = max(float(seq_dt_cfg), 1.0e-6)

            self._physics_dt = float(self.cfg.sim.dt)
            self._control_dt = max(
                float(getattr(self, "step_dt", self._physics_dt * float(self.cfg.decimation))), 1.0e-6
            )
            self._sequence_decimation = max(
                round(self._sequence_model_dt / self._control_dt), 1
            )
            self._learned_model_dt = self._sequence_model_dt

            # Keep _cylinder_id None for safety; diagnostics use _sequence_id
            self._cylinder_id = None
            self._cylinder_fd = None

            if self._sequence_fd is not None and bool(getattr(self.cfg, "enable_verbose_startup", False)):
                print(
                    "[MulagUpperLevelEnv]   Sequence FD delta scale: "
                    f"control_dt={self._control_dt:.6g}, model_dt={self._sequence_model_dt:.6g}, "
                    f"decimation={self._sequence_decimation}, "
                    f"scale_factor={self._control_dt / self._sequence_model_dt:.6g}",
                    flush=True,
                )
            if bool(getattr(self.cfg, "enable_verbose_startup", False)):
                print(
                    f"[MulagUpperLevelEnv]   sequence ID H={self._sequence_id.history_length}, "
                    f"model_dt={self._sequence_model_dt:.6g} s",
                    flush=True,
                )
        else:
            # -- Non-sequence learned dynamics pipeline --
            if rom_only:
                if self._low_level_servo_mode() != "none":
                    raise ValueError(
                        "ROM-only dynamics require low_level_servo_mode='none'; "
                        "the command ROM contains its valve controller internally."
                    )
                control_dt = max(
                    float(getattr(self, "step_dt", float(self.cfg.sim.dt) * float(self.cfg.decimation))),
                    1.0e-6,
                )
                if dynamics_model == "command_rom":
                    self._command_cylinder_rom = CommandCylinderROMRuntime(
                        checkpoint=self.cfg.command_rom_checkpoint,
                        num_envs=self.num_envs,
                        device=self.device,
                        control_dt_s=control_dt,
                    )
                    self._learned_model_dt = self._command_cylinder_rom.sample_time_s
                else:
                    self._base_cylinder_rom = BaseCylinderROMRuntime(
                        checkpoint=self.cfg.base_rom_checkpoint,
                        num_envs=self.num_envs,
                        device=self.device,
                        control_dt_s=control_dt,
                    )
                    self._learned_model_dt = self._base_cylinder_rom.sample_time_s
            else:
                self._cylinder_id = CylinderIDRuntime(
                    models_root=self.cfg.cylinder_models_root,
                    checkpoint=self.cfg.id_cylinder_checkpoint,
                    num_envs=self.num_envs,
                    device=self.device,
                )
                self._id_input_tensor = torch.zeros(
                    (self.num_envs, self._cylinder_id.input_dim), device=self.device
                )
            if dynamics_model == "fd":
                self._cylinder_fd = CylinderFDRuntime(
                    models_root=self.cfg.cylinder_models_root,
                    checkpoint=self.cfg.fd_cylinder_checkpoint,
                    device=self.device,
                )
                self._load_fd_correction()
            elif dynamics_model == "rom":
                self._cylinder_rom = CylinderROMRuntime(
                    models_root=self.cfg.cylinder_models_root,
                    checkpoint=self.cfg.rom_cylinder_checkpoint,
                    num_envs=self.num_envs,
                    device=self.device,
                )
            elif dynamics_model == "base_rom":
                pass
            elif dynamics_model == "command_rom":
                pass
            elif dynamics_model in {"direct_valve", "direct"}:
                pass
            else:
                raise ValueError(
                    "cylinder_dynamics_model must be 'command_rom', 'base_rom', 'rom', 'fd', or 'direct_valve', "
                    f"got {self.cfg.cylinder_dynamics_model!r}."
                )
        self._configure_id_feature_ranges()
        self._action_to_cylinder_target_adapter = PPOActionToCylinderTargetAdapter(
            mapping=mapping if bool(getattr(self.cfg, "enable_cylinder_space_scaling", True)) else None,
            scaling=self._cylinder_space_scaling,
            action_scale=float(self.cfg.action_scale),
            action_scale_per_axis=getattr(
                self.cfg, "cylinder_action_scale_per_axis_m", None
            ),
            target_dxdot_mode=str(getattr(self.cfg, "target_dxdot_mode", "cmd_minus_xdot")),
            target_dxdot_gain=float(getattr(self.cfg, "target_dxdot_gain", 1.0)),
            target_dx_gain=float(getattr(self.cfg, "target_dx_gain", 1.0)),
            learned_model_dt=self._learned_model_dt,
            target_delta_xdot_clip_min=self._id_target_delta_xdot_clip_min
            if bool(getattr(self.cfg, "clip_target_delta_xdot_to_training_range", True))
            else None,
            target_delta_xdot_clip_max=self._id_target_delta_xdot_clip_max
            if bool(getattr(self.cfg, "clip_target_delta_xdot_to_training_range", True))
            else None,
        )
        self._apply_hydraulic_feature_mode()
        if bool(getattr(self.cfg, "enable_verbose_startup", False)):
            print("[MulagUpperLevelEnv] Cylinder-space learned pipeline enabled:", flush=True)
            if use_sequence:
                print(f"[MulagUpperLevelEnv]   Sequence ID checkpoint: {self._sequence_id.checkpoint_path}", flush=True)
                if self._sequence_fd is not None:
                    print(f"[MulagUpperLevelEnv]   Sequence FD checkpoint: {self._sequence_fd.checkpoint_path}", flush=True)
                print(f"[MulagUpperLevelEnv]   Sequence ID H: {self._sequence_id.history_length}", flush=True)
            elif self._base_cylinder_rom is not None:
                print(
                    f"[MulagUpperLevelEnv]   Base ROM checkpoint: "
                    f"{self._base_cylinder_rom.checkpoint_path}",
                    flush=True,
                )
            elif self._command_cylinder_rom is not None:
                print(
                    f"[MulagUpperLevelEnv]   Command ROM checkpoint: "
                    f"{self._command_cylinder_rom.checkpoint_path}", flush=True,
                )
            else:
                print(f"[MulagUpperLevelEnv]   ID checkpoint: {self._cylinder_id.checkpoint_path}", flush=True)
            print(f"[MulagUpperLevelEnv]   physics_dt: {self._physics_dt:.9g}", flush=True)
            print(f"[MulagUpperLevelEnv]   control_dt: {self._control_dt:.9g}", flush=True)
            print(f"[MulagUpperLevelEnv]   learned_model_dt used by ID targets: {self._learned_model_dt:.9g}", flush=True)
            print(
                "[MulagUpperLevelEnv]   target_delta_xdot clipping: "
                f"{bool(getattr(self.cfg, 'clip_target_delta_xdot_to_training_range', True))} "
                f"range={self._id_target_delta_xdot_clip_min.detach().cpu().tolist()}.."
                f"{self._id_target_delta_xdot_clip_max.detach().cpu().tolist()}",
                flush=True,
            )
            print(
                f"[MulagUpperLevelEnv]   hydraulic feature mode: {self._hydraulic_feature_mode()} "
                f"dP_mean={self._hydraulic_dP_training_mean.detach().cpu().tolist()} "
                f"force_mean={self._hydraulic_force_training_mean.detach().cpu().tolist()}",
                flush=True,
            )
            if self._cylinder_rom is not None:
                print(f"[MulagUpperLevelEnv]   ROM checkpoint: {self._cylinder_rom.checkpoint_path}", flush=True)
            if use_sequence:
                if self._sequence_fd is not None:
                    print(f"[MulagUpperLevelEnv]   Sequence FD sample_time_s: {self._sequence_fd.sample_time_s}", flush=True)
            elif self._cylinder_fd is not None:
                print(f"[MulagUpperLevelEnv]   FD checkpoint: {self._cylinder_fd.checkpoint_path}", flush=True)
                print(f"[MulagUpperLevelEnv]   FD sample_time_s: {self._cylinder_fd.sample_time_s}", flush=True)
                fd_gain = float(getattr(self.cfg, "cylinder_fd_delta_gain", 1.0))
                if bool(getattr(self.cfg, "strict_cylinder_usd_scaling", True)) and abs(fd_gain - 1.0) > 1.0e-9:
                    raise RuntimeError(
                        "strict_cylinder_usd_scaling=True requires cylinder_fd_delta_gain=1.0. "
                        f"Got {fd_gain}. Disable strict_cylinder_usd_scaling only for deliberate diagnostics."
                    )
                if bool(getattr(self.cfg, "cylinder_fd_scale_delta_by_sample_time", False)):
                    fd_sample_time = float(getattr(self.cfg, "cylinder_fd_sample_time_s", 0.0))
                    if fd_sample_time <= 0.0:
                        fd_sample_time = float(self._cylinder_fd.sample_time_s)
                    print(
                        "[MulagUpperLevelEnv]   FD temporal delta scale "
                        f"(control_dt / fd_sample_time): {self._control_dt / max(fd_sample_time, 1.0e-6):.9g}",
                        flush=True,
                    )
                    print(
                        "[MulagUpperLevelEnv]   FD temporal scaling changes the integration timestep only; "
                        "USD spatial scaling remains the cylinder mapping above.",
                        flush=True,
                    )
            print(f"[MulagUpperLevelEnv]   dynamics model: {dynamics_model}", flush=True)

    def _feature_stats(self, runtime, names: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        feature_names = list(getattr(runtime, "input_feature_names", []))
        if not feature_names:
            raise RuntimeError(f"{runtime.__class__.__name__} checkpoint does not expose input_feature_names.")
        indices = []
        for name in names:
            if name not in feature_names:
                raise RuntimeError(
                    f"{runtime.__class__.__name__} checkpoint is missing feature {name!r}. "
                    f"Available features: {feature_names}"
                )
            indices.append(feature_names.index(name))
        index = torch.as_tensor(indices, device=self.device, dtype=torch.long)
        x_mean = runtime.normalizers["x_mean"].to(device=self.device, dtype=torch.float32).reshape(-1)
        x_std = runtime.normalizers["x_std"].to(device=self.device, dtype=torch.float32).reshape(-1)
        return x_mean.index_select(0, index).reshape(1, -1), x_std.index_select(0, index).reshape(1, -1)

    def _configure_id_feature_ranges(self) -> None:
        dP_names = [f"dP_{i}" for i in range(1, 5)]
        force_names = [f"Force_{i}" for i in range(1, 5)]
        target_vel_names = [f"TargetDeltaExtensionVelocity_{i}" for i in range(1, 5)]
        if self._command_cylinder_rom is not None:
            mean = self._command_cylinder_rom.model.input_mean.to(self.device).reshape(-1)
            std = self._command_cylinder_rom.model.input_std.to(self.device).reshape(-1)
            self._hydraulic_dP_training_mean = mean[8:12].reshape(1, 4)
            self._hydraulic_force_training_mean.zero_()
            self._id_target_delta_xdot_mean = mean[20:24].reshape(1, 4)
            self._id_target_delta_xdot_std = std[20:24].reshape(1, 4)
        elif self._base_cylinder_rom is not None:
            mean = self._base_cylinder_rom.model.input_mean.to(
                device=self.device, dtype=torch.float32
            ).reshape(-1)
            self._hydraulic_dP_training_mean = mean[12:16].reshape(1, 4)
            self._hydraulic_force_training_mean = mean[16:20].reshape(1, 4)
            self._id_target_delta_xdot_mean.zero_()
            self._id_target_delta_xdot_std.fill_(1.0)
        elif self._sequence_id is not None:
            feature_names = list(getattr(self._sequence_id, "current_feature_names", []))
            if not feature_names:
                raise RuntimeError("Sequence ID checkpoint does not expose current_feature_names.")
            indices = []
            for name in target_vel_names:
                if name not in feature_names:
                    raise RuntimeError(
                        f"Sequence ID checkpoint is missing feature {name!r}. "
                        f"Available current features: {feature_names}"
                    )
                indices.append(feature_names.index(name))
            index = torch.as_tensor(indices, device=self.device, dtype=torch.long)
            self._id_target_delta_xdot_mean = self._sequence_id.current_mean.reshape(-1).index_select(0, index).reshape(1, -1)
            self._id_target_delta_xdot_std = self._sequence_id.current_std.reshape(-1).index_select(0, index).reshape(1, -1)
            self._hydraulic_dP_training_mean.zero_()
            self._hydraulic_force_training_mean.zero_()
        elif self._cylinder_id is not None:
            self._hydraulic_dP_training_mean, _ = self._feature_stats(self._cylinder_id, dP_names)
            self._hydraulic_force_training_mean, _ = self._feature_stats(self._cylinder_id, force_names)
            self._id_target_delta_xdot_mean, self._id_target_delta_xdot_std = self._feature_stats(
                self._cylinder_id,
                target_vel_names,
            )
        else:
            return
        clip_std = max(float(getattr(self.cfg, "target_delta_xdot_clip_std", 3.0)), 0.0)
        self._id_target_delta_xdot_clip_min = self._id_target_delta_xdot_mean - clip_std * self._id_target_delta_xdot_std
        self._id_target_delta_xdot_clip_max = self._id_target_delta_xdot_mean + clip_std * self._id_target_delta_xdot_std

    def _hydraulic_feature_mode(self) -> str:
        mode = str(getattr(self.cfg, "cylinder_hydraulic_feature_mode", "training_mean")).lower()
        if mode not in {"zero", "training_mean", "fd_dp_state"}:
            raise ValueError(
                "cylinder_hydraulic_feature_mode must be 'zero', 'training_mean', or 'fd_dp_state', "
                f"got {mode!r}."
            )
        return mode

    def _apply_hydraulic_feature_mode(self, env_ids: torch.Tensor | None = None) -> None:
        mode = self._hydraulic_feature_mode()
        if env_ids is None:
            env_slice = slice(None)
            count = self.num_envs
        else:
            env_slice = env_ids
            count = int(env_ids.numel())
        if mode == "zero":
            self._dP[env_slice] = 0.0
            self._force[env_slice] = 0.0
        elif mode == "training_mean":
            self._dP[env_slice] = self._hydraulic_dP_training_mean.expand(count, -1)
            self._force[env_slice] = self._hydraulic_force_training_mean.expand(count, -1)
        elif mode == "fd_dp_state":
            if env_ids is not None:
                self._dP[env_slice] = self._hydraulic_dP_training_mean.expand(count, -1)
                self._force[env_slice] = self._hydraulic_force_training_mean.expand(count, -1)
            elif str(getattr(self.cfg, "cylinder_dynamics_model", "fd")).lower() != "base_rom":
                self._force[env_slice] = self._hydraulic_force_training_mean.expand(count, -1)

    def _update_hydraulic_dp_after_model(self, delta_dP: torch.Tensor) -> None:
        mode = self._hydraulic_feature_mode()
        if mode == "fd_dp_state":
            self._dP = self._dP + delta_dP
        elif mode == "training_mean":
            self._dP = self._hydraulic_dP_training_mean.expand(self.num_envs, -1).clone()
        elif mode == "zero":
            self._dP.zero_()

    def _validate_cylinder_usd_scaling(self) -> None:
        if self._cylinder_adapter is None:
            return
        mapping = self._cylinder_adapter.mapping
        if mapping is None:
            if bool(getattr(self.cfg, "enable_verbose_startup", False)):
                print(
                    "[MulagUpperLevelEnv] Cylinder USD scaling contract: no mapping configured; "
                    "learned cylinder values are assumed to already be USD prismatic meters.",
                    flush=True,
                )
            return

        limits = self.robot.data.soft_joint_pos_limits[0, self._cylinder_joint_ids].detach()
        usd_min = limits[:, 0].to(device=self.device, dtype=torch.float32)
        usd_max = limits[:, 1].to(device=self.device, dtype=torch.float32)
        map_min = mapping.isaac_min.to(device=self.device, dtype=torch.float32)
        map_max = mapping.isaac_max.to(device=self.device, dtype=torch.float32)
        limit_error = torch.maximum((map_min - usd_min).abs(), (map_max - usd_max).abs())
        limit_tolerance = float(getattr(self.cfg, "cylinder_usd_limit_tolerance_m", 1.0e-5))

        usd_min_row = usd_min.reshape(1, 4)
        usd_max_row = usd_max.reshape(1, 4)
        if self._cylinder_space_scaling is not None:
            x_at_usd_min = self._cylinder_space_scaling.usd_q_to_learned_x(usd_min_row).reshape(-1)
            x_at_usd_max = self._cylinder_space_scaling.usd_q_to_learned_x(usd_max_row).reshape(-1)
            delta_scale = self._cylinder_space_scaling.learned_delta_to_usd_delta(
                torch.ones((1, 4), device=self.device, dtype=torch.float32)
            ).reshape(-1)
        else:
            x_at_usd_min = mapping.isaac_position_to_simscape_extension(usd_min_row).reshape(-1)
            x_at_usd_max = mapping.isaac_position_to_simscape_extension(usd_max_row).reshape(-1)
            delta_scale = mapping.simscape_delta_to_isaac_delta(
                torch.ones((1, 4), device=self.device, dtype=torch.float32)
            ).reshape(-1)
        delta_scale_error = (delta_scale.abs() - 1.0).abs()
        delta_scale_tolerance = float(getattr(self.cfg, "cylinder_usd_delta_scale_tolerance", 1.0e-4))

        if bool(getattr(self.cfg, "enable_verbose_startup", False)):
            print("[MulagUpperLevelEnv] Cylinder USD scaling contract:", flush=True)
            print(
                f"[MulagUpperLevelEnv]   enable_cylinder_space_scaling="
                f"{bool(getattr(self.cfg, 'enable_cylinder_space_scaling', True))}",
                flush=True,
            )
            print(
                "[MulagUpperLevelEnv]   learned FD/ROM DeltaExtension is metres in Simscape cylinder space; "
                "USD targets are prismatic joint metres.",
                flush=True,
            )
            print(
                "[MulagUpperLevelEnv]   conversion used for PhysX: "
                "q_target_usd = q_current_usd + simscape_delta_to_isaac_delta(delta_extension_m)",
                flush=True,
            )
            for i, name in enumerate(self.cfg.cylinder_joint_names):
                print(
                    f"[MulagUpperLevelEnv]   {name} dof={int(self._cylinder_joint_ids[i].detach().cpu())} "
                    f"usd_limits=[{float(usd_min[i]):.9g}, {float(usd_max[i]):.9g}] m "
                    f"mapping_limits=[{float(map_min[i]):.9g}, {float(map_max[i]):.9g}] m "
                    f"learned_x_at_usd_limits=[{float(x_at_usd_min[i]):.9g}, {float(x_at_usd_max[i]):.9g}] m "
                    f"delta_q_per_delta_x={float(delta_scale[i]):.9g} m/m "
                    f"limit_error={float(limit_error[i]):.3g}",
                    flush=True,
                )

            if self._cylinder_space_scaling is not None:
                print("[MulagUpperLevelEnv] Explicit cylinder-space scaling adapter:", flush=True)
                for line in self._cylinder_space_scaling.summary_lines():
                    print(f"[MulagUpperLevelEnv]   {line}", flush=True)
            else:
                print(
                    "[MulagUpperLevelEnv] WARNING: cylinder-space scaling adapter is disabled; "
                    "raw USD prismatic coordinates will be used for diagnostics only.",
                    flush=True,
                )

        if bool(getattr(self.cfg, "strict_cylinder_usd_scaling", True)) and torch.any(limit_error > limit_tolerance):
            raise RuntimeError(
                "Cylinder coordinate mapping limits do not match the USD/articulation limits. "
                f"max_error={float(limit_error.max()):.6g} m, tolerance={limit_tolerance:.6g} m. "
                "Update cylinder_coordinate_mapping.json before training."
            )
        if bool(getattr(self.cfg, "strict_cylinder_usd_delta_scale", True)) and torch.any(
            delta_scale_error > delta_scale_tolerance
        ):
            raise RuntimeError(
                "Cylinder delta mapping is not a pure +/-1 metre-to-metre conversion into USD prismatic joints. "
                f"delta_q_per_delta_x={delta_scale.detach().cpu().tolist()}, "
                f"tolerance={delta_scale_tolerance:.6g}. "
                "Disable strict_cylinder_usd_delta_scale only if a validated non-unit mechanism mapping is intended."
            )

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        ground_color = tuple(getattr(self.cfg, "ground_color", (0.0, 0.0, 0.0)))
        # Ground must cover the whole env grid: origins span ~spacing*sqrt(N)/2 and
        # the truck spawns ~20 m from its origin. A fixed 200 m plate ends at x=±100,
        # and at 64 envs the far-corner trucks straddled that edge and fell off the
        # world (observed: envs 62/63 in endless free fall -> invalid J_eff).
        _n = int(getattr(self.scene.cfg, "num_envs", 1))
        _spacing = float(getattr(self.scene.cfg, "env_spacing", 20.0))
        _grid_half_span = _spacing * (math.ceil(math.sqrt(max(_n, 1))) / 2.0)
        _ground_size = max(200.0, 2.0 * (_grid_half_span + 60.0))
        ground_cfg = sim_utils.MeshCuboidCfg(
            size=(_ground_size, _ground_size, 0.02),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.02,  # 2cm contact offset to prevent penetration
                rest_offset=0.0,      # Contact starts at surface
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=float(getattr(self.cfg, "terrain_static_friction", 1.0)),
                dynamic_friction=float(getattr(self.cfg, "terrain_dynamic_friction", 1.0)),
                restitution=float(getattr(self.cfg, "terrain_restitution", 0.0)),
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=ground_color),
        )
        ground_cfg.func("/World/ground", ground_cfg, translation=(0.0, 0.0, -0.01))
        if bool(getattr(self.cfg, "enable_mathbox_ground", True)):
            self._spawn_mathbox_ground_visuals()
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot
        if bool(getattr(self.cfg, "mowing_terrain_enable", False)):
            self._spawn_mowing_terrain()
        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _spawn_mowing_terrain(self) -> None:
        """Per-env mowable shelf at the tool's lowest reachable band (embankment).
        The main ground stays at z=0 for the wheels; this shelf sits beside the
        truck where the arm reaches. Visual-only unless mowing_terrain_collision."""
        half_x = float(getattr(self.cfg, "mowing_terrain_half_x", 0.7))
        half_y = float(getattr(self.cfg, "mowing_terrain_half_y", 0.9))
        thickness = float(getattr(self.cfg, "mowing_terrain_thickness", 0.04))
        z_top = float(getattr(self.cfg, "mowing_terrain_z", 1.08))
        local = (
            float(getattr(self.cfg, "mowing_terrain_local_x", -19.528)),
            float(getattr(self.cfg, "mowing_terrain_local_y", 1.38)),
            z_top - 0.5 * thickness,
        )
        color = tuple(getattr(self.cfg, "mowing_terrain_color", (0.20, 0.45, 0.18)))
        with_collision = bool(getattr(self.cfg, "mowing_terrain_collision", False))
        kwargs = dict(
            size=(2.0 * half_x, 2.0 * half_y, thickness),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        )
        if with_collision:
            kwargs["collision_props"] = sim_utils.CollisionPropertiesCfg(contact_offset=0.02, rest_offset=0.0)
            kwargs["physics_material"] = sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
            )
        shelf_cfg = sim_utils.MeshCuboidCfg(**kwargs)
        origins = self.scene.env_origins.detach().cpu().numpy()
        origin0 = origins[0]
        for env_id in range(origins.shape[0]):
            pos = (
                float(origins[env_id][0] - origin0[0] + local[0]),
                float(origins[env_id][1] - origin0[1] + local[1]),
                float(local[2]),
            )
            shelf_cfg.func(f"/World/mowing_terrain/shelf_{env_id}", shelf_cfg, translation=pos)

    def _spawn_mathbox_ground_visuals(self) -> None:
        line_color = tuple(getattr(self.cfg, "lane_line_color", (1.0, 1.0, 1.0)))
        line_width = float(getattr(self.cfg, "lane_line_width_m", 0.08))
        lane_half_width = 1.5
        z = 0.006

        def spawn_visual_line(path: str, size: tuple[float, float, float], pos: tuple[float, float, float],
                              color: tuple[float, float, float]) -> None:
            line_cfg = sim_utils.MeshCuboidCfg(
                size=size,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )
            line_cfg.func(path, line_cfg, translation=pos)

        # Lane markings follow the simple mowing-lane convention used by the
        # Kevin tasks: sampled target y stays in a lane while the vehicle moves in +X.
        for idx, y in enumerate((-lane_half_width, 0.0, lane_half_width)):
            spawn_visual_line(
                f"/World/ground/lane_line_{idx}",
                (120.0, line_width, 0.006),
                (0.0, y, z),
                line_color,
            )

        grid_color = (0.28, 0.28, 0.28)
        for idx, x in enumerate(range(-60, 61, 5)):
            spawn_visual_line(
                f"/World/ground/grid_x_{idx}",
                (line_width * 0.5, 20.0, 0.004),
                (float(x), 0.0, z * 0.8),
                grid_color,
            )
        for idx, y in enumerate(range(-10, 11, 2)):
            if y in (-2, 0, 2):
                continue
            spawn_visual_line(
                f"/World/ground/grid_y_{idx}",
                (120.0, line_width * 0.5, 0.004),
                (0.0, float(y), z * 0.8),
                grid_color,
            )

    # =========================================================================
    # Observations
    # =========================================================================

    def _get_observations(self) -> dict:
        self._compute_observations()
        obs = self._flatten_observations()
        return {"policy": obs}

    def _compute_observations(self):
        robot = self.robot
        env_origins = self.scene.env_origins

        # Push history buffers
        self._obs_joint_pos_hist[:, :-1] = self._obs_joint_pos_hist[:, 1:].clone()
        self._obs_joint_vel_hist[:, :-1] = self._obs_joint_vel_hist[:, 1:].clone()
        self._obs_distance_to_goal_hist[:, :-1] = self._obs_distance_to_goal_hist[:, 1:].clone()

        # -- joint_pos (normalized to [-1,1]) --
        jpos = robot.data.joint_pos[:, self._arm_joint_ids]
        limits = robot.data.soft_joint_pos_limits[:, self._arm_joint_ids]
        jpos_norm = 2.0 * (jpos - limits[..., 0]) / (limits[..., 1] - limits[..., 0] + 1e-8) - 1.0
        jpos_norm = torch.clamp(jpos_norm, -1.0, 1.0)
        self._obs_joint_pos_hist[:, -1] = jpos_norm

        # -- joint_vel --
        jvel = robot.data.joint_vel[:, self._arm_joint_ids]
        self._obs_joint_vel_hist[:, -1] = jvel

        # -- distance_to_goal --
        dist_to_goal = self._compute_distance_to_goal()
        self._obs_distance_to_goal_hist[:, -1] = dist_to_goal

        # -- joint_torques (normalized by effort limits) --
        tau = robot.data.applied_torque[:, self._arm_joint_ids]
        tau_limits = robot.data.joint_effort_limits[:, self._arm_joint_ids]
        tau_limits = torch.clamp(tau_limits, min=1e-6)
        tau_norm = tau / tau_limits
        tau_norm = torch.clamp(tau_norm, -1.0, 1.0)
        self._obs_joint_torques = tau_norm

        # -- last_action --
        self._obs_last_action = self._prev_action.clone()

        # -- unimog_pose (normalized) --
        local_pos = robot.data.root_pos_w - env_origins
        root_q = robot.data.root_quat_w
        root_q = root_q / (torch.norm(root_q, dim=-1, keepdim=True).clamp(min=1e-8))
        _, _, yaw = euler_xyz_from_quat(root_q)
        x_norm = (local_pos[:, 0] - 0.0) / 25.0
        y_norm = (local_pos[:, 1] - 4.3) / 4.0  # centre at spawn y
        x_norm = torch.clamp(x_norm, -1.5, 1.5)
        y_norm = torch.clamp(y_norm, -1.5, 1.5)
        self._obs_unimog_pose = torch.stack([x_norm, y_norm, torch.sin(yaw), torch.cos(yaw)], dim=-1)

        # -- unimog_pose_dot --
        lin_vel = robot.data.root_lin_vel_w
        ang_vel = robot.data.root_ang_vel_w
        self._obs_unimog_pose_dot = torch.stack([lin_vel[:, 0], lin_vel[:, 1], ang_vel[:, 2]], dim=-1)

        # -- Update sequence model histories with post-physics state --
        self._update_sequence_histories()

    def _apply_fd_delta_plan_substep(self) -> None:
        """Between model ticks: apply one substep from the stored FD plan."""
        decimation = max(self._sequence_decimation, 1)
        delta_per_step = self._fd_delta_plan / float(decimation)
        self._raw_delta_extension = delta_per_step.detach().clone()
        self._fd_delta_raw = delta_per_step.detach().clone()
        self._fd_delta_corrected = delta_per_step.detach().clone()
        self._fd_delta_scale = 1.0 / float(decimation)
        cylinder_q = self.robot.data.joint_pos[:, self._cylinder_joint_ids]
        if self._cylinder_adapter is not None:
            delta_isaac = self._cylinder_adapter.simscape_delta_to_isaac_delta(delta_per_step)
        else:
            delta_isaac = delta_per_step
        self._raw_delta_extension_velocity = torch.zeros_like(delta_per_step)
        self._raw_delta_dP = torch.zeros_like(delta_per_step)
        self._predicted_delta_extension = delta_per_step.detach().clone()
        self._delta_isaac = delta_isaac.detach().clone()
        lower = self.robot.data.soft_joint_pos_limits[:, self._cylinder_joint_ids, 0]
        upper = self.robot.data.soft_joint_pos_limits[:, self._cylinder_joint_ids, 1]
        target_base = (
            self._cylinder_pos_target.detach().clone()
            if bool(getattr(self.cfg, "cylinder_target_integrates_from_previous_target", False))
            else cylinder_q
        )
        self._cylinder_pos_target = torch.clamp(target_base + delta_isaac, lower, upper)
        if self._neutral_hold_mask.any():
            neutral_hold_target_mode = str(getattr(self.cfg, "cylinder_neutral_hold_target_mode", "previous_target"))
            if neutral_hold_target_mode in {"current", "current_position", "current_joint_position"}:
                self._cylinder_pos_target[self._neutral_hold_mask] = torch.clamp(
                    self._neutral_hold_cylinder_pos_target[self._neutral_hold_mask],
                    lower[self._neutral_hold_mask],
                    upper[self._neutral_hold_mask],
                )
        self._maybe_print_pipeline_debug()

    def _update_sequence_histories(self) -> None:
        if self._sequence_id is None:
            return
        # Only push new history rows on model ticks
        if not self._sequence_tick_active:
            return
        cylinder_q = self.robot.data.joint_pos[:, self._cylinder_joint_ids]
        cylinder_qdot = self.robot.data.joint_vel[:, self._cylinder_joint_ids]
        if self._cylinder_adapter is not None:
            x_current, xdot_current = self._cylinder_adapter.current_from_cylinder_joints(
                cylinder_q, cylinder_qdot
            )
        else:
            x_current = cylinder_q[:, :4]
            xdot_current = cylinder_qdot[:, :4]
        # Push into ID history: [Extension, ExtensionVelocity, ValveCmd]
        self._sequence_id.push_history(
            extension=x_current,
            extension_velocity=xdot_current,
            valve_cmd=self._current_valve_cmd,  # becomes PreviousValveCmd next step
        )

    def _flatten_observations(self) -> torch.Tensor:
        parts = [
            self._obs_joint_pos_hist.reshape(self.num_envs, -1),
            self._obs_joint_vel_hist.reshape(self.num_envs, -1),
            self._obs_distance_to_goal_hist.reshape(self.num_envs, -1),
            self._obs_joint_torques,
            self._obs_last_action,
            self._obs_unimog_pose,
            self._obs_unimog_pose_dot,
        ]
        obs = torch.cat(parts, dim=-1)
        obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return obs

    # -- Observation helper functions --

    def _compute_distance_to_goal(self) -> torch.Tensor:
        """XY distance from TCP to command in base frame. Returns (N, 2)."""
        robot = self.robot
        ee_pos_w = robot.data.body_link_state_w[:, self._ee_body_id, :3]
        base_pos_w = robot.data.root_pos_w
        base_quat_w = robot.data.root_quat_w

        base_q = base_quat_w / (torch.norm(base_quat_w, dim=-1, keepdim=True).clamp(min=1e-8))
        tcp_pos_local = quat_apply_inverse(
            base_q, ee_pos_w - base_pos_w
        )
        target_xy = self._command_pos_w[:, :2]
        current_xy_local = tcp_pos_local[:, :2]
        diff = target_xy - current_xy_local
        diff = torch.nan_to_num(diff, nan=0.0)
        return diff

    # =========================================================================
    # Rewards
    # =========================================================================

    def _get_rewards(self) -> torch.Tensor:
        """Core mowing rewards — no collision avoidance, just position/height/orientation tracking."""
        cfg = self.cfg
        robot = self.robot
        env_origins = self.scene.env_origins

        ee_pos_w = robot.data.body_link_state_w[:, self._ee_body_id, :3]
        ee_quat_w = robot.data.body_link_state_w[:, self._ee_body_id, 3:7]
        base_pos_w = robot.data.root_pos_w
        base_quat_w = robot.data.root_quat_w
        jpos = robot.data.joint_pos[:, self._arm_joint_ids]
        jvel = robot.data.joint_vel[:, self._arm_joint_ids]

        # Normalize quaternion (prevent NaN from zero-quat after reset)
        base_quat_w = base_quat_w / (torch.norm(base_quat_w, dim=-1, keepdim=True).clamp(min=1e-8))

        reward = torch.zeros(self.num_envs, device=self.device)

        # -- TCP position in base frame --
        tcp_pos_local = quat_apply_inverse(base_quat_w, ee_pos_w - base_pos_w)
        target_xy = self._command_pos_w[:, :2]
        pos_err = torch.norm(tcp_pos_local[:, :2] - target_xy, dim=-1)

        # 1. Position tracking (linear penalty)
        reward += cfg.rew_position_tracking_linear * (pos_err - cfg.position_tracking_offset)
        # 2. Position tracking (tanh bonus)
        reward += cfg.rew_position_tracking_tanh * (1.0 - torch.tanh(pos_err / cfg.position_tracking_std))

        # 5. Yaw tracking (absolute error, offset-tolerant)
        _, _, cmd_yaw = euler_xyz_from_quat(self._command_quat_w)
        _, _, ee_yaw = euler_xyz_from_quat(ee_quat_w / (torch.norm(ee_quat_w, dim=-1, keepdim=True).clamp(min=1e-8)))
        yaw_err = torch.abs(torch.atan2(torch.sin(ee_yaw - cmd_yaw), torch.cos(ee_yaw - cmd_yaw)))
        yaw_offset = cfg.yaw_tracking_offset_deg * (torch.pi / 180.0)
        reward += cfg.rew_yaw_tracking * torch.clamp(yaw_err - yaw_offset, min=0.0)

        # 7. TCP ahead of arm base
        if self._arm_base_body_id is not None:
            arm_base_pos_w = robot.data.body_link_state_w[:, self._arm_base_body_id, :3]
            dist_x = ee_pos_w[:, 0] - arm_base_pos_w[:, 0]
            ahead = torch.where(dist_x > cfg.tcp_ahead_threshold, 10.0, dist_x - cfg.tcp_ahead_threshold)
            reward += cfg.rew_tcp_ahead * ahead

        # 6. Progress along lane (+X)
        dx = base_pos_w[:, 0] - env_origins[:, 0] + 25.0
        progress = torch.clamp(dx / max(cfg.finish_traj_length_x, 1e-6), 0.0, 1.0)
        reward += cfg.rew_finish_progress * progress

        # 7. Finish bonus
        finished = progress >= 1.0
        reward += finished.float() * cfg.finish_bonus_weight

        # 8. Action rate smoothing
        action_delta = self._current_action - self._prev_action
        reward += cfg.rew_action_rate * torch.sum(action_delta ** 2, dim=-1)

        # 9. Joint velocity penalty
        reward += cfg.rew_joint_vel * torch.sum(jvel ** 2, dim=-1)

        reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
        return reward

    # =========================================================================
    # Dones
    # =========================================================================

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        robot = self.robot
        cfg = self.cfg
        env_origins = self.scene.env_origins

        # Timeout
        timeout = self.episode_length_buf >= self.max_episode_length

        # Joint limit violation
        jpos = robot.data.joint_pos[:, self._arm_joint_ids]
        limits = robot.data.soft_joint_pos_limits[:, self._arm_joint_ids]
        joint_limit_violation = (
            (jpos < limits[..., 0] - cfg.joint_limit_margin)
            | (jpos > limits[..., 1] + cfg.joint_limit_margin)
        ).any(dim=-1)
        horizon_owns_resets = bool(getattr(cfg, "disable_base_workspace_termination", False))
        if horizon_owns_resets:
            joint_limit_violation = torch.zeros_like(joint_limit_violation)

        # Flying (roll/pitch)
        root_q = robot.data.root_quat_w
        root_q = root_q / (torch.norm(root_q, dim=-1, keepdim=True).clamp(min=1e-8))
        _, roll, pitch = euler_xyz_from_quat(root_q)
        flying = (torch.abs(roll) > cfg.unimog_flying_roll_pitch_threshold) | (
            torch.abs(pitch) > cfg.unimog_flying_roll_pitch_threshold
        )
        if horizon_owns_resets:
            flying = torch.zeros_like(flying)

        # Lateral deviation (vehicle drifts off mowing lane)
        local_y = robot.data.root_pos_w[:, 1] - env_origins[:, 1] - 4.3  # centre at y=4.3
        lateral_deviation = torch.abs(local_y) > cfg.unimog_lateral_deviation_threshold

        # TCP facing backwards (cutter yaw error > threshold)
        _, _, cmd_yaw = euler_xyz_from_quat(self._command_quat_w)
        ee_q = robot.data.body_link_state_w[:, self._ee_body_id, 3:7]
        ee_q = ee_q / (torch.norm(ee_q, dim=-1, keepdim=True).clamp(min=1e-8))
        _, _, ee_yaw = euler_xyz_from_quat(ee_q)
        yaw_err = torch.abs(torch.atan2(torch.sin(ee_yaw - cmd_yaw), torch.cos(ee_yaw - cmd_yaw)))
        tcp_face_backwards = yaw_err > cfg.tcp_face_backwards_yaw_threshold
        if horizon_owns_resets:
            lateral_deviation = torch.zeros_like(lateral_deviation)
            tcp_face_backwards = torch.zeros_like(tcp_face_backwards)
            timeout = torch.zeros_like(timeout)

        # End of lane (x-direction progress)
        dx = robot.data.root_pos_w[:, 0] - env_origins[:, 0] + 25.0
        end_of_lane = dx >= cfg.finish_traj_length_x
        if horizon_owns_resets:
            end_of_lane = torch.zeros_like(end_of_lane)

        # NaN/inf guard
        nonfinite = ~(
            torch.isfinite(jpos).all(dim=-1)
            & torch.isfinite(robot.data.joint_vel[:, self._arm_joint_ids]).all(dim=-1)
            & torch.isfinite(robot.data.root_pos_w).all(dim=-1)
        )

        terminated = joint_limit_violation | flying | lateral_deviation | tcp_face_backwards | nonfinite
        truncated = timeout | end_of_lane
        self._last_done_cause_masks = {
            "timeout": timeout.detach().clone(),
            "joint_limit": joint_limit_violation.detach().clone(),
            "base_flying": flying.detach().clone(),
            "workspace_violation": lateral_deviation.detach().clone() | tcp_face_backwards.detach().clone(),
            "NaN/nonfinite": nonfinite.detach().clone(),
            "success": end_of_lane.detach().clone(),
        }

        return terminated, truncated

    # =========================================================================
    # Actions
    # =========================================================================

    def _pre_physics_step(self, actions: torch.Tensor):
        """Convert PPO actions into either direct joint targets or cylinder targets."""
        self._ppo_raw_action = torch.nan_to_num(actions.to(device=self.device, dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if self._ppo_raw_action.shape != (self.num_envs, self.cfg.action_space):
            raise RuntimeError(
                f"PPO action must have shape ({self.num_envs}, {self.cfg.action_space}), "
                f"got {tuple(self._ppo_raw_action.shape)}."
            )
        if self._debug_manual_cylinder_position_target is not None:
            self._current_action = torch.zeros_like(self._ppo_raw_action)
            self._action_hydraulic.zero_()
            self._action_target_delta_x_learned.zero_()
            self._action_target_delta_xdot_learned.zero_()
            self._neutral_hold_mask[:] = True
            self._neutral_hold_prev_mask[:] = True
            return
        if bool(getattr(self.cfg, "use_cylinder_id_pipeline", False)):
            self._current_action = torch.clamp(self._ppo_raw_action, -1.0, 1.0)
            self._run_cylinder_id_pipeline()
        else:
            self._current_action = self._ppo_raw_action.clone()

    def _joint_reference_from_action(self) -> tuple[torch.Tensor, torch.Tensor]:
        q_now = self.robot.data.joint_pos[:, self._arm_joint_ids]
        step_dt = max(float(getattr(self, "step_dt", float(self.cfg.sim.dt) * float(self.cfg.decimation))), 1.0e-6)
        action_width = min(self._current_action.shape[-1], q_now.shape[-1])
        q_ref = q_now.clone()
        q_ref[:, :action_width] = q_now[:, :action_width] + self._current_action[:, :action_width] * float(self.cfg.action_scale)
        limits = self.robot.data.soft_joint_pos_limits[:, self._arm_joint_ids]
        q_ref = torch.clamp(q_ref, limits[..., 0], limits[..., 1])
        qdot_ref = (q_ref - q_now) / step_dt
        velocity_clip = float(getattr(self.cfg, "joint_reference_velocity_clip", 0.0))
        if velocity_clip > 0.0:
            qdot_ref = torch.clamp(qdot_ref, -velocity_clip, velocity_clip)
        q_ref, qdot_ref = self._apply_joint6_coupling_to_reference(q_ref, qdot_ref)
        return q_ref, qdot_ref

    def _apply_joint6_coupling_to_reference(
        self,
        q_ref: torch.Tensor,
        qdot_ref: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._joint6_coupling_source_ref_index is None or self._joint6_coupling_target_ref_index is None:
            return q_ref, qdot_ref

        source = self._joint6_coupling_source_ref_index
        target = self._joint6_coupling_target_ref_index
        a = float(self.cfg.joint6_coupling_a)
        b = float(self.cfg.joint6_coupling_b)

        q_ref = q_ref.clone()
        qdot_ref = qdot_ref.clone()
        q_ref[:, target] = a * q_ref[:, source] + b
        qdot_ref[:, target] = a * qdot_ref[:, source]

        limits = self.robot.data.soft_joint_pos_limits[:, self._arm_joint_ids]
        q_ref[:, target] = torch.clamp(q_ref[:, target], limits[:, target, 0], limits[:, target, 1])
        velocity_clip = float(getattr(self.cfg, "joint_reference_velocity_clip", 0.0))
        if velocity_clip > 0.0:
            qdot_ref[:, target] = torch.clamp(qdot_ref[:, target], -velocity_clip, velocity_clip)
        return q_ref, qdot_ref

    def _run_cylinder_id_pipeline(self) -> None:
        dynamics_model = str(self.cfg.cylinder_dynamics_model).lower()
        rom_only = dynamics_model in {"base_rom", "command_rom"}
        use_sequence = bool(getattr(self.cfg, "use_sequence_cylinder_models", False))
        if not use_sequence and self._cylinder_adapter is None:
            raise RuntimeError("Cylinder action pipeline is not initialized.")
        if not use_sequence and not rom_only and self._cylinder_id is None:
            raise RuntimeError("use_cylinder_id_pipeline=True but the cylinder ID pipeline is not initialized.")
        if use_sequence and self._sequence_id is None:
            raise RuntimeError("use_sequence_cylinder_models=True but SequenceIDRuntime is not initialized.")
        if self._action_to_cylinder_target_adapter is None:
            raise RuntimeError("Cylinder action-to-target adapter is not initialized.")

        # --- Decimation: determine if this step is a sequence model tick ---
        self._sequence_tick_counter += 1
        if use_sequence:
            decimation = max(self._sequence_decimation, 1)
            is_model_tick = (self._sequence_tick_counter % decimation == 1)
            if self._low_level_servo_enabled():
                is_model_tick = True
            self._sequence_tick_active = is_model_tick
        elif rom_only:
            # The command ROM predicts its valve internally. The base-ROM
            # experiment retains direct policy valves for comparison only.
            if dynamics_model == "command_rom":
                self._current_valve_cmd = self._command_cylinder_rom.previous_predicted_valve.clone()
            else:
                self._current_valve_cmd = torch.clamp(
                    self._action_hydraulic, -float(self.cfg.max_abs_valve), float(self.cfg.max_abs_valve)
                )
            self._fd_baseline_valve_cmd.zero_()
            self._id_input_tensor.zero_()
        else:
            is_model_tick = True
            self._sequence_tick_active = True

        # --- Non-model-tick fast path (distribute stored FD plan) ---
        if use_sequence and not is_model_tick:
            self._apply_fd_delta_plan_substep()
            return

        # --- Model tick: full ID -> FD pipeline below ---
        self._joint_ref, self._joint_vel_ref = self._joint_reference_from_action()
        self._apply_hydraulic_feature_mode()
        cylinder_q = self.robot.data.joint_pos[:, self._cylinder_joint_ids]
        cylinder_qdot = self.robot.data.joint_vel[:, self._cylinder_joint_ids]
        previous_cylinder_pos_target = self._cylinder_pos_target.detach().clone()
        x_current, xdot_current = self._cylinder_adapter.current_from_cylinder_joints(cylinder_q, cylinder_qdot)
        action_target = self._action_to_cylinder_target_adapter.convert(
            action=self._current_action,
            x_current=x_current,
            xdot_current=xdot_current,
        )
        reference = self._maybe_override_reference_for_debug_velocity(action_target.reference)
        self._action_hydraulic = action_target.hydraulic_action.detach().clone()
        if action_target.ignored_passive_action is None:
            self._action_ignored_passive.zero_()
        else:
            self._action_ignored_passive = action_target.ignored_passive_action.detach().clone()
        self._action_delta_q_usd_ref = action_target.delta_q_usd_ref.detach().clone()
        self._action_target_delta_x_pre_gain_learned = action_target.target_delta_x_pre_gain.detach().clone()
        self._action_target_delta_x_post_gain_learned = action_target.target_delta_x_post_gain.detach().clone()
        self._action_target_delta_x_safety_limit_learned = action_target.target_delta_x_safety_limit.detach().clone()
        self._action_target_delta_x_learned = reference.target_delta_x.detach().clone()
        self._action_target_delta_x_pre_gain_clip_mask = action_target.target_delta_x_pre_gain_clip_mask.detach().clone()
        self._action_target_delta_x_post_gain_clip_mask = action_target.target_delta_x_post_gain_clip_mask.detach().clone()
        self._action_target_delta_xdot_raw_learned = action_target.target_delta_xdot_raw.detach().clone()
        self._action_target_delta_xdot_learned = reference.target_delta_xdot.detach().clone()
        self._action_target_delta_xdot_clip_mask = action_target.target_delta_xdot_clip_mask.detach().clone()
        if self._action_target_delta_xdot_clip_mask.any() and self._debug_cylinder_velocity_target is None:
            clip_per_cylinder = self._action_target_delta_xdot_clip_mask.sum(dim=0).to(dtype=torch.long)
            self._target_delta_xdot_clip_counts += clip_per_cylinder
            if bool(getattr(self.cfg, "log_target_delta_xdot_clipping", True)):
                max_logs = int(getattr(self.cfg, "target_delta_xdot_clip_log_first_n", 20))
                if self._target_delta_xdot_clip_log_count < max_logs:
                    selected = [
                        env_id for env_id in self._selected_debug_env_ids()
                        if bool(self._action_target_delta_xdot_clip_mask[env_id].any().detach().cpu())
                    ]
                    clipped_count = int(self._action_target_delta_xdot_clip_mask.any(dim=-1).sum().detach().cpu().item())
                    detail = ""
                    if selected:
                        idx = torch.as_tensor(selected, dtype=torch.long, device=self.device)
                        detail = (
                            f" selected_env_ids={selected} "
                            f"raw={self._action_target_delta_xdot_raw_learned[idx].detach().cpu().tolist()} "
                            f"clipped={self._action_target_delta_xdot_learned[idx].detach().cpu().tolist()}"
                        )
                    print(
                        "[MulagUpperLevelEnv] TargetDeltaExtensionVelocity clipped "
                        f"count={clipped_count}/{self.num_envs}{detail}",
                        flush=True,
                    )
                    self._target_delta_xdot_clip_log_count += 1
        self._target_delta_xdot_step_count += self.num_envs
        self._cylinder_x_current = reference.x_current.detach().clone()
        self._cylinder_xdot_current = reference.xdot_current.detach().clone()
        hold_deadband = float(getattr(self.cfg, "cylinder_neutral_hold_deadband_m", 0.0))
        hold_velocity_deadband = float(getattr(self.cfg, "cylinder_neutral_hold_velocity_deadband_mps", 0.0))
        action_deadband = float(getattr(self.cfg, "ppo_neutral_action_deadband", 0.0))
        neutral_hold_mask = (
            (self._action_hydraulic.abs().amax(dim=-1) <= action_deadband)
            | (
                (reference.target_delta_x.abs().amax(dim=-1) <= hold_deadband)
                & (reference.target_delta_xdot.abs().amax(dim=-1) <= hold_velocity_deadband)
            )
        )
        self._neutral_hold_mask = neutral_hold_mask.detach().clone()
        neutral_hold_target_mode = str(getattr(self.cfg, "cylinder_neutral_hold_target_mode", "previous_target"))
        if neutral_hold_target_mode in {"current", "current_position", "current_joint_position"}:
            entering_neutral = neutral_hold_mask & (~self._neutral_hold_prev_mask)
            if entering_neutral.any():
                self._neutral_hold_cylinder_pos_target[entering_neutral] = cylinder_q[entering_neutral].detach().clone()
        self._neutral_hold_prev_mask = neutral_hold_mask.detach().clone()
        dP_before = self._dP.clone()
        self._cylinder_ref = reference.x_ref.detach().clone()
        self._cylinder_vel_ref = reference.xdot_ref.detach().clone()
        self._id_prev_valve_cmd_input = self._prev_valve_cmd.detach().clone()

        if use_sequence:
            # Sequence ID: predict DeltaValveCmd from history + target features
            seq_id_out = self._sequence_id.predict_delta_valve(
                target_delta_x=reference.target_delta_x,
                target_delta_xdot=reference.target_delta_xdot,
                previous_valve_cmd=self._id_prev_valve_cmd_input,
            )
            self._current_valve_cmd = seq_id_out.valve_cmd
            self._current_valve_cmd = torch.clamp(
                self._current_valve_cmd, -float(self.cfg.max_abs_valve), float(self.cfg.max_abs_valve)
            )
            if neutral_hold_mask.any():
                self._current_valve_cmd[neutral_hold_mask] = 0.0
        elif rom_only:
            # Both ROM variants bypass inverse-model inference.  The command ROM
            # predicts its valve internally below; the experimental base ROM uses
            # the direct valve action prepared above.
            pass
        else:
            # Legacy ID: build_id_input -> cylinder_id.predict
            self._id_input_tensor = self._cylinder_adapter.build_id_input(
                reference,
                dP=self._dP,
                force=self._force,
                prev_valve_cmd=self._id_prev_valve_cmd_input,
                input_dim=self._cylinder_id.input_dim,
            )
            self._current_valve_cmd = self._cylinder_id.predict(
                self._id_input_tensor,
                prev_valve_cmd=self._id_prev_valve_cmd_input,
            )
            if bool(getattr(self.cfg, "id_residual_over_zero_target", False)):
                zero_target_input = self._id_input_tensor.detach().clone()
                zero_target_input[:, 16:24] = 0.0
                zero_target_valve = self._cylinder_id.predict(
                    zero_target_input,
                    prev_valve_cmd=self._id_prev_valve_cmd_input,
                )
                if bool(getattr(self.cfg, "paired_id_fd_residual", False)):
                    self._fd_baseline_valve_cmd = zero_target_valve.detach().clone()
                else:
                    self._current_valve_cmd = self._current_valve_cmd - zero_target_valve
            self._current_valve_cmd = torch.clamp(
                self._current_valve_cmd, -float(self.cfg.max_abs_valve), float(self.cfg.max_abs_valve)
            )
            if bool(getattr(self.cfg, "paired_id_fd_residual", False)):
                self._fd_baseline_valve_cmd = torch.clamp(
                    self._fd_baseline_valve_cmd,
                    -float(self.cfg.max_abs_valve),
                    float(self.cfg.max_abs_valve),
                )
            if neutral_hold_mask.any():
                self._current_valve_cmd[neutral_hold_mask] = 0.0
                if bool(getattr(self.cfg, "paired_id_fd_residual", False)):
                    self._fd_baseline_valve_cmd[neutral_hold_mask] = 0.0
                self._cylinder_id.reset(torch.where(neutral_hold_mask)[0])

        step_dt = max(float(getattr(self, "step_dt", float(self.cfg.sim.dt) * float(self.cfg.decimation))), 1.0e-6)
        self._valve_ff = self._current_valve_cmd.detach().clone()
        self._valve_pid.zero_()
        self._valve_total = self._current_valve_cmd.detach().clone()
        self._cyl_vel_desired = reference.xdot_ref.detach().clone()
        self._cyl_vel_measured = reference.xdot_current.detach().clone()
        if self._low_level_servo_enabled():
            self._current_valve_cmd = self._apply_cylinder_velocity_servo(
                reference=reference,
                valve_ff=self._current_valve_cmd,
                neutral_hold_mask=neutral_hold_mask,
                step_dt=step_dt,
            )
        # -- Debug valve command override (for FD data collection) --
        if self._debug_valve_cmd is not None:
            self._current_valve_cmd = self._debug_valve_cmd.to(device=self.device, dtype=torch.float32).detach().clone()
            self._valve_ff = self._current_valve_cmd.detach().clone()
            self._valve_pid.zero_()
            self._valve_total = self._current_valve_cmd.detach().clone()
        if dynamics_model == "command_rom":
            if self._command_cylinder_rom is None:
                raise RuntimeError("cylinder_dynamics_model='command_rom' but runtime is not initialized.")
            command = torch.cat((reference.target_delta_x, reference.target_delta_xdot), dim=-1)
            rom_out = self._command_cylinder_rom.step(
                command=command,
                extension=reference.x_current,
                extension_velocity=reference.xdot_current,
                dP=self._dP,
                system_factors=getattr(self, "_command_rom_system_factors", None),
            )
            self._current_valve_cmd = rom_out.predicted_valve.detach().clone()
            delta_extension = rom_out.delta_extension
            self._raw_delta_extension = delta_extension.detach().clone()
            self._fd_delta_raw = delta_extension.detach().clone()
            if bool(getattr(self.cfg, "command_rom_direction_guard", False)):
                command_delta = reference.target_delta_x
                ratio = max(
                    float(getattr(self.cfg, "command_rom_command_max_ratio", 3.0)),
                    0.0,
                )
                # Command-ROM displacement is distributed across its three
                # 60 Hz control steps.  Scale the model-interval guard by the
                # same factor; otherwise the three guarded increments sum to
                # three times the intended 50 ms authority.
                ratio /= float(max(self._command_cylinder_rom.steps_per_tick, 1))
                same_direction = delta_extension * command_delta > 0.0
                limit = ratio * command_delta.abs()
                delta_extension = torch.where(
                    same_direction,
                    torch.sign(command_delta) * torch.minimum(delta_extension.abs(), limit),
                    torch.zeros_like(delta_extension),
                )
            self._fd_delta_corrected = delta_extension.detach().clone()
            self._raw_delta_extension_velocity = (
                rom_out.extension_velocity_next - reference.xdot_current
            ).detach().clone()
            self._raw_delta_dP = rom_out.delta_dP.detach().clone()
            self._fd_delta_scale = 1.0
            self._predicted_extension_velocity = rom_out.extension_velocity_next.detach().clone()
            self._update_hydraulic_dp_after_model(rom_out.delta_dP)
            dp_limit = max(float(getattr(self.cfg, "command_rom_dp_abs_limit_pa", 1.5e8)), 0.0)
            if dp_limit > 0.0:
                self._dP = torch.clamp(self._dP, -dp_limit, dp_limit)
        elif dynamics_model == "base_rom":
            if self._base_cylinder_rom is None:
                raise RuntimeError("cylinder_dynamics_model='base_rom' but base ROM runtime is not initialized.")
            rom_out = self._base_cylinder_rom.step(
                valve_cmd=self._current_valve_cmd,
                extension=reference.x_current,
                extension_velocity=reference.xdot_current,
                dP=self._dP,
                force=self._force,
            )
            delta_extension = rom_out.delta_extension
            self._raw_delta_extension = delta_extension.detach().clone()
            self._fd_delta_raw = delta_extension.detach().clone()
            self._fd_delta_corrected = delta_extension.detach().clone()
            self._raw_delta_extension_velocity = (
                rom_out.extension_velocity_next - reference.xdot_current
            ).detach().clone()
            self._raw_delta_dP = rom_out.delta_dP.detach().clone()
            self._fd_delta_scale = 1.0
            self._predicted_extension_velocity = rom_out.extension_velocity_next.detach().clone()
            self._update_hydraulic_dp_after_model(rom_out.delta_dP)
            if self._hydraulic_feature_mode() == "fd_dp_state":
                self._force = rom_out.predicted_force.detach().clone()
        elif dynamics_model == "rom":
            if self._cylinder_rom is None:
                raise RuntimeError("cylinder_dynamics_model='rom' but ROM runtime is not initialized.")
            rom_out = self._cylinder_rom.step(
                valve_cmd=self._current_valve_cmd,
                extension=reference.x_current,
                extension_velocity=reference.xdot_current,
                dP=self._dP,
                force=self._force,
                dt=step_dt,
            )
            delta_extension = rom_out.delta_extension
            self._raw_delta_extension = delta_extension.detach().clone()
            self._fd_delta_raw = delta_extension.detach().clone()
            self._fd_delta_corrected = delta_extension.detach().clone()
            self._raw_delta_extension_velocity.zero_()
            self._raw_delta_dP = rom_out.delta_dP.detach().clone()
            self._fd_delta_scale = 1.0
            self._predicted_extension_velocity = rom_out.extension_velocity_next.detach().clone()
            self._update_hydraulic_dp_after_model(rom_out.delta_dP)
        elif dynamics_model == "fd":
            if use_sequence:
                if self._sequence_fd is None:
                    raise RuntimeError("use_sequence_cylinder_models=True but SequenceFDRuntime is not initialized.")
                # Push current valve_cmd + state into FD history before prediction
                self._sequence_fd.push_current_state(
                    valve_cmd=self._current_valve_cmd,
                    extension=reference.x_current,
                    extension_velocity=reference.xdot_current,
                )
                fd_out = self._sequence_fd.predict_delta_extension()
                fd_delta_raw = fd_out.delta_extension_raw
                if self._low_level_servo_enabled() and not bool(getattr(self.cfg, "servo_use_fd_prediction", True)):
                    fd_delta_raw = torch.zeros_like(fd_delta_raw)
                self._fd_delta_raw = fd_delta_raw.detach().clone()
                if bool(getattr(self.cfg, "sequence_fd_residual_over_zero_valve", False)):
                    baseline_history = self._sequence_fd.last_seq_raw.detach().clone()
                    baseline_history[:, -1, :4] = 0.0
                    fd_baseline = self._sequence_fd.predict_delta_extension_from_history(baseline_history)
                    fd_delta_raw = fd_delta_raw - fd_baseline.delta_extension_raw
                    self._sequence_fd.last_seq_raw = self._sequence_fd._step_history.detach().clone()
                    self._sequence_fd.last_delta_extension_raw = fd_delta_raw.detach().clone()
                    self._sequence_fd.last_delta_extension_scaled = fd_delta_raw.detach().clone()
                fd_delta_raw = self._apply_fd_delta_correction(fd_delta_raw, self._current_valve_cmd)
                self._fd_delta_corrected = fd_delta_raw.detach().clone()
                # FD model-step delta (one sequence_model_dt = 0.05 s).
                # Sequence checkpoints are trained in cylinder-extension space; this gain is a calibrated
                # task-level authority knob and defaults to 1.0 to preserve the raw model contract.
                seq_fd_gain = float(getattr(self.cfg, "sequence_fd_delta_gain", 1.0))
                delta_extension_model = fd_delta_raw * seq_fd_gain
                fd_output_gain_val = float(getattr(self.cfg, "fd_output_gain", 1.0))
                delta_extension_model = delta_extension_model * fd_output_gain_val
                if self._low_level_servo_enabled():
                    correction_gain = float(getattr(self.cfg, "servo_tracking_delta_gain", 0.0))
                    if correction_gain != 0.0:
                        vel_error = self._cyl_vel_desired - self._cyl_vel_measured
                        delta_extension_model = delta_extension_model + correction_gain * vel_error * self._sequence_model_dt
                self._fd_delta_corrected = delta_extension_model.detach().clone()
                self._raw_delta_extension = delta_extension_model.detach().clone()
                self._raw_delta_extension_velocity = torch.zeros_like(delta_extension_model)
                self._raw_delta_dP = torch.zeros_like(delta_extension_model)
                delta_extension_velocity = torch.zeros_like(delta_extension_model)
                delta_dP = torch.zeros_like(delta_extension_model)
                self._predicted_extension_velocity = (
                    reference.xdot_current + delta_extension_velocity
                ).detach().clone()

                # Store the model-step delta as the plan for the decimation interval
                self._fd_delta_plan = delta_extension_model.detach().clone()

                # Apply one substep: delta_per_step = model_delta / decimation
                decimation = max(self._sequence_decimation, 1)
                delta_extension = delta_extension_model / float(decimation)
                self._fd_delta_scale = 1.0 / float(decimation)

                self._sequence_step_counter += 1
                if bool(getattr(self.cfg, "enable_sequence_tick_debug", False)) or bool(
                    getattr(self.cfg, "enable_sequence_runtime_debug", False)
                ):
                    interval_name = "debug_log_interval"
                    if bool(getattr(self.cfg, "enable_sequence_runtime_debug", False)) and not bool(
                        getattr(self.cfg, "enable_sequence_tick_debug", False)
                    ):
                        interval_name = "sequence_runtime_log_interval"
                    interval = max(int(getattr(self.cfg, interval_name, 1000)), 1)
                    if self._sequence_step_counter % interval != 0:
                        return_debug_tick = False
                    else:
                        return_debug_tick = True
                else:
                    return_debug_tick = False
                if return_debug_tick:
                    env_id = self._selected_debug_env_ids()[0]
                    print(
                        f"[MulagUpperLevelEnv][seq-fd] TICK step={self._sequence_step_counter}"
                        f" tick={self._sequence_tick_counter} decimation={decimation}"
                        f" env={env_id}"
                        f" delta_model={delta_extension_model[env_id].detach().cpu().tolist()}"
                        f" delta_per_step={delta_extension[env_id].detach().cpu().tolist()}",
                        flush=True,
                    )
            else:
                if self._cylinder_fd is None:
                    raise RuntimeError("cylinder_dynamics_model='fd' but FD runtime is not initialized.")
                fd_out = self._cylinder_fd.step(
                    valve_cmd=self._current_valve_cmd,
                    extension=reference.x_current,
                    extension_velocity=reference.xdot_current,
                    dP=self._dP,
                    force=self._force,
                )
                paired_residual = bool(getattr(self.cfg, "paired_id_fd_residual", False))
                if paired_residual:
                    fd_zero = self._cylinder_fd.step(
                        valve_cmd=self._fd_baseline_valve_cmd,
                        extension=reference.x_current,
                        extension_velocity=reference.xdot_current,
                        dP=self._dP,
                        force=self._force,
                    )
                    fd_out = type(fd_out)(
                        delta_extension=fd_out.delta_extension - fd_zero.delta_extension,
                        delta_extension_velocity=(
                            fd_out.delta_extension_velocity - fd_zero.delta_extension_velocity
                        ),
                        delta_dP=fd_out.delta_dP - fd_zero.delta_dP,
                        raw_output=fd_out.raw_output - fd_zero.raw_output,
                    )
                elif bool(getattr(self.cfg, "fd_residual_over_zero_valve", False)):
                    fd_zero = self._cylinder_fd.step(
                        valve_cmd=torch.zeros_like(self._current_valve_cmd),
                        extension=reference.x_current,
                        extension_velocity=reference.xdot_current,
                        dP=self._dP,
                        force=self._force,
                    )
                    fd_out = type(fd_out)(
                        delta_extension=fd_out.delta_extension - fd_zero.delta_extension,
                        delta_extension_velocity=(
                            fd_out.delta_extension_velocity - fd_zero.delta_extension_velocity
                        ),
                        delta_dP=fd_out.delta_dP - fd_zero.delta_dP,
                        raw_output=fd_out.raw_output - fd_zero.raw_output,
                    )
                if paired_residual or bool(getattr(self.cfg, "fd_residual_over_zero_valve", False)):
                    residual_gain = self._cfg_cyl_vector(
                        "cylinder_fd_residual_output_gain", 1.0
                    )
                    fd_out = type(fd_out)(
                        delta_extension=fd_out.delta_extension * residual_gain,
                        delta_extension_velocity=fd_out.delta_extension_velocity * residual_gain,
                        delta_dP=fd_out.delta_dP * residual_gain,
                        raw_output=fd_out.raw_output,
                    )
                if self._low_level_servo_enabled() and not bool(getattr(self.cfg, "servo_use_fd_prediction", True)):
                    fd_out = type(fd_out)(
                        delta_extension=torch.zeros_like(fd_out.delta_extension),
                        delta_extension_velocity=torch.zeros_like(fd_out.delta_extension_velocity),
                        delta_dP=torch.zeros_like(fd_out.delta_dP),
                        raw_output=fd_out.raw_output,
                    )
                self._raw_delta_extension = fd_out.delta_extension.detach().clone()
                self._fd_delta_raw = fd_out.delta_extension.detach().clone()
                self._raw_delta_extension_velocity = fd_out.delta_extension_velocity.detach().clone()
                self._raw_delta_dP = fd_out.delta_dP.detach().clone()
                fd_scale = float(getattr(self.cfg, "cylinder_fd_delta_gain", 1.0))
                if bool(getattr(self.cfg, "cylinder_fd_scale_delta_by_sample_time", False)):
                    fd_sample_time = float(getattr(self.cfg, "cylinder_fd_sample_time_s", 0.0))
                    if fd_sample_time <= 0.0:
                        fd_sample_time = float(getattr(self._cylinder_fd, "sample_time_s", step_dt))
                    fd_scale *= step_dt / max(fd_sample_time, 1.0e-6)
                self._fd_delta_scale = fd_scale
                delta_extension = fd_out.delta_extension * fd_scale
                reference_blend = float(getattr(self.cfg, "cylinder_idfd_reference_blend", 0.0))
                reference_blend = min(max(reference_blend, 0.0), 1.0)
                if reference_blend > 0.0:
                    delta_extension = (
                        (1.0 - reference_blend) * delta_extension
                        + reference_blend * reference.target_delta_x
                    )
                if bool(getattr(self.cfg, "cylinder_fd_command_direction_guard", False)):
                    # Paired residuals are intended to represent the causal
                    # response to target_delta_x.  Keep the network's predicted
                    # magnitude, but reject recursive state-leakage components
                    # that reverse that command or exceed a modest envelope.
                    # This is a safety contract, not a direct-drive bypass: a
                    # zero learned response remains zero.
                    command_delta = reference.target_delta_x
                    same_direction = delta_extension * command_delta >= 0.0
                    ratio = max(float(getattr(self.cfg, "cylinder_fd_command_max_ratio", 1.5)), 0.0)
                    limit = ratio * command_delta.abs()
                    delta_extension = torch.where(
                        same_direction,
                        torch.sign(delta_extension) * torch.minimum(delta_extension.abs(), limit),
                        torch.zeros_like(delta_extension),
                    )
                if self._low_level_servo_enabled():
                    correction_gain = float(getattr(self.cfg, "servo_tracking_delta_gain", 0.0))
                    if correction_gain != 0.0:
                        vel_error = self._cyl_vel_desired - self._cyl_vel_measured
                        delta_extension = delta_extension + correction_gain * vel_error * step_dt
                self._fd_delta_corrected = delta_extension.detach().clone()
                delta_extension_velocity = fd_out.delta_extension_velocity * fd_scale
                delta_dP = fd_out.delta_dP * fd_scale
            self._predicted_extension_velocity = (reference.xdot_current + delta_extension_velocity).detach().clone()
            self._update_hydraulic_dp_after_model(delta_dP)
        elif dynamics_model in {"direct_valve", "direct"}:
            delta_extension = self._current_valve_cmd * float(self.cfg.direct_valve_position_scale)
            self._raw_delta_extension = delta_extension.detach().clone()
            self._fd_delta_raw = delta_extension.detach().clone()
            self._fd_delta_corrected = delta_extension.detach().clone()
            self._raw_delta_extension_velocity.zero_()
            self._raw_delta_dP.zero_()
            self._fd_delta_scale = 1.0
            self._predicted_extension_velocity = delta_extension / step_dt
        else:
            raise ValueError(f"Unsupported cylinder_dynamics_model: {self.cfg.cylinder_dynamics_model!r}")

        # --- FD tracking servo: apply fd_gain + tracking delta correction ---
        servo_mode = self._low_level_servo_mode()
        if servo_mode in ("fd_tracking_servo", "id_pid_fd_tracking_servo"):
            delta_extension = self._servo_fd_gain * delta_extension + self._servo_tracking_delta
            self._fd_delta_corrected = delta_extension.detach().clone()
        elif servo_mode in ("position_ramp_pi", "direct_position_delta"):
            delta_extension = self._servo_position_delta
            self._raw_delta_extension = self._servo_position_delta.detach().clone()
            self._fd_delta_raw = self._servo_position_delta.detach().clone()
            self._fd_delta_corrected = self._servo_position_delta.detach().clone()
            self._raw_delta_extension_velocity.zero_()
            self._raw_delta_dP.zero_()
            self._fd_delta_scale = 1.0
            self._predicted_extension_velocity = (reference.xdot_current +
                self._servo_position_delta / max(float(step_dt), 1e-9)).detach().clone()




            if use_sequence:
                self._fd_delta_plan[neutral_hold_mask] = 0.0
                self._raw_delta_extension[neutral_hold_mask] = 0.0
            self._predicted_extension_velocity[neutral_hold_mask] = reference.xdot_current[neutral_hold_mask]
            self._dP[neutral_hold_mask] = dP_before[neutral_hold_mask]

        clip_field = "cylinder_fd_delta_clip_m" if dynamics_model == "fd" else "cylinder_rom_delta_clip_m"
        delta_clip = float(getattr(self.cfg, clip_field, 0.0))
        if delta_clip > 0.0:
            delta_extension = torch.clamp(delta_extension, -delta_clip, delta_clip)
        if self._cylinder_space_scaling is not None and dynamics_model == "rom":
            delta_isaac = self._cylinder_space_scaling.learned_velocity_to_usd_delta(
                self._predicted_extension_velocity,
                step_dt,
            )
        else:
            delta_isaac = self._cylinder_adapter.simscape_delta_to_isaac_delta(delta_extension)
        self._predicted_delta_extension = delta_extension.detach().clone()
        self._delta_isaac = delta_isaac.detach().clone()
        lower = self.robot.data.soft_joint_pos_limits[:, self._cylinder_joint_ids, 0]
        upper = self.robot.data.soft_joint_pos_limits[:, self._cylinder_joint_ids, 1]
        integrates_from_prev = bool(getattr(self.cfg, "cylinder_target_integrates_from_previous_target", False))
        target_base = previous_cylinder_pos_target if integrates_from_prev else cylinder_q
        new_target = target_base + delta_isaac
        # Leaky integration (anti-runaway): pull the reference target gently back
        # toward the measured joint pos so a persistent action bias settles at a
        # bounded offset (delta_isaac / leak) instead of integrating to the limit.
        leak = float(getattr(self.cfg, "cylinder_target_leak", 0.0))
        if integrates_from_prev and leak > 0.0:
            new_target = new_target + leak * (cylinder_q - target_base)
        # Anti-windup: the reference integrator must not run arbitrarily far ahead of the
        # plant. The drive is a position servo with a ~0.2 s lag, so while the arm is
        # still catching up the integrator keeps adding delta and the target sails past
        # the goal -- the arm then overshoots by exactly the accumulated lead. Bounding
        # the lead bounds the overshoot without touching the learned FD delta itself.
        #
        # PER-AXIS, because one scalar conflates two different jobs. Measured
        # 2026-08-04 on the PPO3 16/17 checkpoint, evaluation only, no
        # retraining: raising this scalar 0.018 -> 0.050 cut lateral error on
        # the sustained-offset families by half (lane_change_hold 0.4833 ->
        # 0.2310, boom_sweep 0.3745 -> 0.1602) -- the clamp really was bounding
        # the standing effort those paths need. But the SAME relaxation applied
        # to cylinder 4 drove head alignment 0.9507 -> 0.4636, because c4 is
        # head pitch and acts only in the Y-Z plane, so a head excursion is
        # unrecoverable (the same asymmetry `--c4_exploration_std` exists for).
        # 0.100 on all axes is worse again (0.7070): the cap is NOT monotonic,
        # since the overshoot it bounds grows with the lead it allows.
        #
        # `cylinder_target_max_lead_per_axis_m` therefore overrides the scalar
        # per cylinder when supplied. Default None keeps the scalar behaviour
        # bit-for-bit, so every stored PPO1/PPO2/PPO3 result stands.
        lead_m = float(getattr(self.cfg, "cylinder_target_max_lead_m", 0.0))
        per_axis = getattr(self.cfg, "cylinder_target_max_lead_per_axis_m", None)
        if per_axis is not None:
            lead_row = torch.as_tensor(
                [float(value) for value in per_axis],
                dtype=new_target.dtype,
                device=new_target.device,
            )
            if lead_row.numel() != new_target.shape[-1]:
                raise ValueError(
                    "cylinder_target_max_lead_per_axis_m must have one entry per "
                    f"cylinder ({new_target.shape[-1]}), got {lead_row.numel()}"
                )
            if bool((lead_row <= 0.0).any()):
                raise ValueError(
                    "cylinder_target_max_lead_per_axis_m entries must be positive; "
                    "use the scalar cylinder_target_max_lead_m = 0 to disable"
                )
            lead_usd = self._cylinder_adapter.simscape_delta_to_isaac_delta(
                lead_row.expand_as(new_target).contiguous()
            ).abs()
            new_target = torch.clamp(new_target, cylinder_q - lead_usd, cylinder_q + lead_usd)
        elif lead_m > 0.0:
            lead_usd = self._cylinder_adapter.simscape_delta_to_isaac_delta(
                torch.full_like(new_target, lead_m)
            ).abs()
            new_target = torch.clamp(new_target, cylinder_q - lead_usd, cylinder_q + lead_usd)
        self._cylinder_pos_target = torch.clamp(new_target, lower, upper)
        if neutral_hold_mask.any():
            if neutral_hold_target_mode in {"current", "current_position", "current_joint_position"}:
                neutral_hold_target = self._neutral_hold_cylinder_pos_target[neutral_hold_mask]
            else:
                neutral_hold_target = previous_cylinder_pos_target[neutral_hold_mask]
            self._cylinder_pos_target[neutral_hold_mask] = torch.clamp(
                neutral_hold_target,
                lower[neutral_hold_mask],
                upper[neutral_hold_mask],
            )
        if self._joint6_coupling_source_ref_index is not None and self._joint6_coupling_target_ref_index is not None:
            source = self._joint6_coupling_source_ref_index
            target = self._joint6_coupling_target_ref_index
            a = float(self.cfg.joint6_coupling_a)
            b = float(self.cfg.joint6_coupling_b)
            self._joint_ref[:, source] = self._cylinder_pos_target[:, 3]
            self._joint_ref[:, target] = a * self._joint_ref[:, source] + b
            self._joint_vel_ref[:, source] = 0.0
            self._joint_vel_ref[:, target] = 0.0
        self._dP = torch.nan_to_num(self._dP, nan=0.0, posinf=0.0, neginf=0.0)
        self._maybe_print_pipeline_debug()

    def _runtime_debug_row(self, env_id: int) -> dict[str, object]:
        cylinder_q = self.robot.data.joint_pos[:, self._cylinder_joint_ids]
        cylinder_limits = self.robot.data.soft_joint_pos_limits[:, self._cylinder_joint_ids]
        lower = cylinder_limits[..., 0]
        upper = cylinder_limits[..., 1]
        valve_abs = self._current_valve_cmd[env_id].detach().abs()
        max_abs_valve = float(getattr(self.cfg, "max_abs_valve", 1.0))
        return {
            "step": int(self._pipeline_debug_step),
            "env_id": int(env_id),
            "action": self._format_debug_list(self._ppo_raw_action, env_id),
            "clipped_action": self._format_debug_list(self._current_action, env_id),
            "target_dx_pre_gain": self._format_debug_list(self._action_target_delta_x_pre_gain_learned, env_id),
            "target_dx_post_gain": self._format_debug_list(self._action_target_delta_x_post_gain_learned, env_id),
            "target_delta_x": self._format_debug_list(self._action_target_delta_x_learned, env_id),
            "target_dx_safety_limit": self._format_debug_list(self._action_target_delta_x_safety_limit_learned, env_id),
            "target_delta_xdot": self._format_debug_list(self._action_target_delta_xdot_learned, env_id),
            "valve_cmd": self._format_debug_list(self._current_valve_cmd, env_id),
            "cyl_vel_desired": self._format_debug_list(self._cyl_vel_desired, env_id),
            "cyl_vel_measured": self._format_debug_list(self._cyl_vel_measured, env_id),
            "valve_ff": self._format_debug_list(self._valve_ff, env_id),
            "valve_pid": self._format_debug_list(self._valve_pid, env_id),
            "valve_total": self._format_debug_list(self._valve_total, env_id),
            "fd_delta_raw": self._format_debug_list(self._fd_delta_raw, env_id),
            "fd_delta_corrected": self._format_debug_list(self._fd_delta_corrected, env_id),
            "fd_delta": self._format_debug_list(self._raw_delta_extension, env_id),
            "cylinder_q": self._format_debug_list(cylinder_q, env_id),
            "cylinder_target": self._format_debug_list(self._cylinder_pos_target, env_id),
            "valve_saturation_flag": bool((valve_abs >= 0.999 * max_abs_valve).any().detach().cpu()),
            "target_dx_pre_gain_clamp_flag": bool(
                self._action_target_delta_x_pre_gain_clip_mask[env_id].any().detach().cpu()
            ),
            "target_dx_post_gain_clamp_flag": bool(
                self._action_target_delta_x_post_gain_clip_mask[env_id].any().detach().cpu()
            ),
            "target_delta_xdot_clip_flag": bool(self._action_target_delta_xdot_clip_mask[env_id].any().detach().cpu()),
            "physx_target_clamp_flag": bool(
                (
                    torch.isclose(self._cylinder_pos_target[env_id], lower[env_id])
                    | torch.isclose(self._cylinder_pos_target[env_id], upper[env_id])
                ).any().detach().cpu()
            ),
            "neutral_hold_flag": bool(self._neutral_hold_mask[env_id].detach().cpu()),
        }

    def _write_runtime_debug_csv(self, rows: list[dict[str, object]]) -> None:
        if not rows:
            return
        path = Path(str(getattr(self.cfg, "debug_csv_path", "outputs/cylinder_diagnostics/runtime_pipeline_debug.csv"))).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0].keys())
        write_header = (not self._runtime_debug_csv_header_written) or (not path.exists())
        with path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)
        self._runtime_debug_csv_header_written = True

    def _maybe_print_pipeline_debug(self) -> None:
        self._pipeline_debug_step += 1
        if not self._debug_interval_due():
            return
        env_ids = self._selected_debug_env_ids()
        if bool(getattr(self.cfg, "write_debug_csv", False)):
            self._write_runtime_debug_csv([self._runtime_debug_row(env_id) for env_id in env_ids])
        if not (
            bool(getattr(self.cfg, "enable_pipeline_debug", False))
            or bool(getattr(self.cfg, "pipeline_debug_prints", False))
        ):
            return
        for env_id in env_ids:
            self._print_pipeline_debug(env_id)

    def _print_pipeline_debug(self, env_id: int) -> None:
        print(f"[MulagUpperLevelEnv][pipeline-debug] step={self._pipeline_debug_step} env={env_id}", flush=True)
        print(f"  PPO raw action[{env_id}]: {self._ppo_raw_action[env_id].detach().cpu().tolist()}", flush=True)
        print(f"  PPO clipped hydraulic action[{env_id}]: {self._action_hydraulic[env_id].detach().cpu().tolist()}", flush=True)
        print(f"  ignored passive action[{env_id}]: {self._action_ignored_passive[env_id].detach().cpu().tolist()}", flush=True)
        print(f"  action delta_q_usd_ref[{env_id}]: {self._action_delta_q_usd_ref[env_id].detach().cpu().tolist()}", flush=True)
        print(f"  joint reference[{env_id}]: {self._joint_ref[env_id].detach().cpu().tolist()}", flush=True)
        print(f"  joint velocity reference[{env_id}]: {self._joint_vel_ref[env_id].detach().cpu().tolist()}", flush=True)
        print(f"  cylinder reference[{env_id}]: {self._cylinder_ref[env_id].detach().cpu().tolist()}", flush=True)
        print(f"  cylinder velocity reference[{env_id}]: {self._cylinder_vel_ref[env_id].detach().cpu().tolist()}", flush=True)
        cylinder_q = self.robot.data.joint_pos[:, self._cylinder_joint_ids]
        cylinder_qdot = self.robot.data.joint_vel[:, self._cylinder_joint_ids]
        cylinder_limits = self.robot.data.soft_joint_pos_limits[:, self._cylinder_joint_ids]
        target_error = self._cylinder_pos_target - cylinder_q
        applied_force = self.robot.data.applied_torque[:, self._cylinder_joint_ids]
        effort_limits = torch.clamp(
            self.robot.data.joint_effort_limits[:, self._cylinder_joint_ids].abs(),
            min=1.0,
        )
        effort_ratio = applied_force.abs() / effort_limits
        print(f"  q_usd current[{env_id}]: {cylinder_q[env_id].detach().cpu().tolist()}", flush=True)
        print(f"  x_learned current[{env_id}]: {self._cylinder_x_current[env_id].detach().cpu().tolist()}", flush=True)
        print(f"  qdot_usd current[{env_id}]: {cylinder_qdot[env_id].detach().cpu().tolist()}", flush=True)
        print(f"  xdot_learned current[{env_id}]: {self._cylinder_xdot_current[env_id].detach().cpu().tolist()}", flush=True)
        print(f"  target_delta_x_pre_gain_learned[{env_id}]: {self._action_target_delta_x_pre_gain_learned[env_id].detach().cpu().tolist()}", flush=True)
        print(f"  target_delta_x_post_gain_learned[{env_id}]: {self._action_target_delta_x_post_gain_learned[env_id].detach().cpu().tolist()}", flush=True)
        print(f"  target_delta_x_safety_limit_learned[{env_id}]: {self._action_target_delta_x_safety_limit_learned[env_id].detach().cpu().tolist()}", flush=True)
        print(f"  target_delta_x_learned[{env_id}]: {self._action_target_delta_x_learned[env_id].detach().cpu().tolist()}", flush=True)
        print(f"  target_delta_x_pre_gain_clip_mask[{env_id}]: {self._action_target_delta_x_pre_gain_clip_mask[env_id].detach().cpu().tolist()}", flush=True)
        print(f"  target_delta_x_post_gain_clip_mask[{env_id}]: {self._action_target_delta_x_post_gain_clip_mask[env_id].detach().cpu().tolist()}", flush=True)
        print(f"  target_delta_xdot_raw_learned[{env_id}]: {self._action_target_delta_xdot_raw_learned[env_id].detach().cpu().tolist()}", flush=True)
        print(f"  target_delta_xdot_clipped_learned[{env_id}]: {self._action_target_delta_xdot_learned[env_id].detach().cpu().tolist()}", flush=True)
        print(f"  target_delta_xdot_clip_mask[{env_id}]: {self._action_target_delta_xdot_clip_mask[env_id].detach().cpu().tolist()}", flush=True)
        print(f"  servo mode: {self._low_level_servo_mode()}", flush=True)
        print(f"  cyl_vel_desired[{env_id}]: {self._cyl_vel_desired[env_id].detach().cpu().tolist()}", flush=True)
        print(f"  cyl_vel_measured[{env_id}]: {self._cyl_vel_measured[env_id].detach().cpu().tolist()}", flush=True)
        print(f"  valve_ff[{env_id}]: {self._valve_ff[env_id].detach().cpu().tolist()}", flush=True)
        print(f"  valve_pid[{env_id}]: {self._valve_pid[env_id].detach().cpu().tolist()}", flush=True)
        print(f"  valve_total[{env_id}]: {self._valve_total[env_id].detach().cpu().tolist()}", flush=True)
        if self._cylinder_id is not None and self._cylinder_id.last_input_normalized is not None:
            id_norm = self._cylinder_id.last_input_normalized[env_id].detach().cpu()
            print(f"  ID normalized TargetDeltaExtension[{env_id}]: {id_norm[16:20].tolist()}", flush=True)
            print(f"  ID normalized TargetDeltaExtensionVelocity[{env_id}]: {id_norm[20:24].tolist()}", flush=True)
        print(f"  dP feature[{env_id}]: {self._dP[env_id].detach().cpu().tolist()}", flush=True)
        print(f"  Force feature[{env_id}]: {self._force[env_id].detach().cpu().tolist()}", flush=True)
        print(
            f"  cylinder q limits Isaac[{env_id}]: "
            f"{cylinder_limits[env_id].detach().cpu().tolist()}",
            flush=True,
        )
        print(f"  cylinder target error Isaac[{env_id}]: {target_error[env_id].detach().cpu().tolist()}", flush=True)
        neutral_hold = bool(self._neutral_hold_mask[env_id].detach().cpu())
        print(f"  neutral hold active[{env_id}]: {neutral_hold}", flush=True)
        if self._joint6_coupling_source_ref_index is not None and self._joint6_coupling_target_ref_index is not None:
            source = self._joint6_coupling_source_ref_index
            target = self._joint6_coupling_target_ref_index
            print(
                f"  q4_ref/q6_ref[{env_id}]: {float(self._joint_ref[env_id, source].detach().cpu()):.9g} -> "
                f"{float(self._joint_ref[env_id, target].detach().cpu()):.9g}",
                flush=True,
            )
            print(
                f"  q4dot_ref/q6dot_ref[{env_id}]: {float(self._joint_vel_ref[env_id, source].detach().cpu()):.9g} -> "
                f"{float(self._joint_vel_ref[env_id, target].detach().cpu()):.9g}",
                flush=True,
            )
        print(f"  ID input tensor shape: {tuple(self._id_input_tensor.shape)}", flush=True)
        print(f"  previous ValveCmd used by ID[{env_id}]: {self._id_prev_valve_cmd_input[env_id].detach().cpu().tolist()}", flush=True)
        print(f"  ID output valve command[{env_id}]: {self._current_valve_cmd[env_id].detach().cpu().tolist()}", flush=True)
        if str(self.cfg.cylinder_dynamics_model).lower() == "rom":
            print(
                f"  ROM output velocity learned[{env_id}]: {self._predicted_extension_velocity[env_id].detach().cpu().tolist()}",
                flush=True,
            )
        if str(self.cfg.cylinder_dynamics_model).lower() == "fd":
            print(
                f"  FD raw delta extension simscape[{env_id}]: {self._raw_delta_extension[env_id].detach().cpu().tolist()}",
                flush=True,
            )
            print(f"  FD raw model delta[{env_id}]: {self._fd_delta_raw[env_id].detach().cpu().tolist()}", flush=True)
            print(f"  FD corrected delta[{env_id}]: {self._fd_delta_corrected[env_id].detach().cpu().tolist()}", flush=True)
            print(f"  FD delta scale: {float(self._fd_delta_scale):.9g}", flush=True)
        print(f"  learned delta_x[{env_id}]: {self._predicted_delta_extension[env_id].detach().cpu().tolist()}", flush=True)
        print(f"  final delta_q_usd to PhysX[{env_id}]: {self._delta_isaac[env_id].detach().cpu().tolist()}", flush=True)
        print(f"  PhysX cylinder position target[{env_id}]: {self._cylinder_pos_target[env_id].detach().cpu().tolist()}", flush=True)
        print(f"  applied cylinder force/torque[{env_id}]: {applied_force[env_id].detach().cpu().tolist()}", flush=True)
        print(f"  cylinder effort ratio[{env_id}]: {effort_ratio[env_id].detach().cpu().tolist()}", flush=True)
        if self._cylinder_rom is not None:
            print(
                f"  ROM input/output shape: {self._cylinder_rom.last_input_shape} -> {self._cylinder_rom.last_output_shape}",
                flush=True,
            )

    def _ensure_batched_cylinder_target(self, q_des):
        """Ensure cylinder target tensor has shape [B, 4].

        Handles:
          [4]       -> expand to [B, 4]
          [1, 4]    -> expand to [B, 4]
          [B, 4]    -> pass through
        """
        import torch as _torch
        B = self.num_envs
        q_des = _torch.as_tensor(q_des, device=self.device, dtype=_torch.float32)
        if q_des.ndim == 1:
            if q_des.numel() != 4:
                raise RuntimeError(f"Expected [4], got {q_des.shape}")
            q_des = q_des.unsqueeze(0).expand(B, -1).clone()
        elif q_des.ndim == 2:
            if q_des.shape == (1, 4) and B > 1:
                q_des = q_des.expand(B, -1).clone()
            elif q_des.shape != (B, 4):
                raise RuntimeError(f"Bad q_des shape: {q_des.shape}, expected {(B, 4)}")
        else:
            raise RuntimeError(f"Bad q_des ndim: {q_des.ndim}, shape={q_des.shape}")
        return q_des

    def _explicit_effort_target(self, q_des: torch.Tensor) -> None:
        """Apply explicit PD/PID/bias effort to track cylinder position targets.

        Backends:
          explicit_effort_pd:  force = kp*error - kd*qdot
          explicit_effort_pid: force = kp*error + ki*integral - kd*qdot (with anti-windup)
          explicit_effort_pid_bias:
            Calibration phase: PID with ki_calib, capture mean force_cmd as bias
            Runtime phase:     force = bias + kp*error - kd*qdot
        """
        cfg = self.cfg
        backend = str(getattr(cfg, "lowlevel_actuator_backend", "implicit_position"))
        # --- shape guard ---
        q_des = self._ensure_batched_cylinder_target(q_des)
        B_sh = self.num_envs
        use_pid = (backend in ("explicit_effort_pid", "explicit_effort_pid_bias"))
        use_bias = (backend == "explicit_effort_pid_bias")

        # --- debug: print bias shapes once ---
        if getattr(cfg, "debug_bias_shapes", False) and use_bias:
            if not getattr(self, "_bias_shapes_printed", False):
                self._bias_shapes_printed = True
                print(f"[BIAS_SHAPES] num_envs={self.num_envs}", flush=True)
                print(f"[BIAS_SHAPES] q_des.shape={q_des.shape}", flush=True)
                print(f"[BIAS_SHAPES] _explicit_force_bias.shape={self._explicit_force_bias.shape}", flush=True)
                print(f"[BIAS_SHAPES] _explicit_bias_calibrated.shape={self._explicit_bias_calibrated.shape}", flush=True)
                print(f"[BIAS_SHAPES] _explicit_bias_calib_step.shape={self._explicit_bias_calib_step.shape}", flush=True)
                print(f"[BIAS_SHAPES] cylinder_pos.shape={self.robot.data.joint_pos[:, self._cylinder_joint_ids].shape}", flush=True)
                print(f"[BIAS_SHAPES] cylinder_vel.shape={self.robot.data.joint_vel[:, self._cylinder_joint_ids].shape}", flush=True)
                if self._debug_manual_cylinder_position_target is not None:
                    print(f"[BIAS_SHAPES] _debug_manual_cylinder_position_target.shape={self._debug_manual_cylinder_position_target.shape}", flush=True)

        kps = float(getattr(cfg, "explicit_pid_kp_scale" if use_pid else "explicit_pd_gain_scale", 1.0))
        kds = float(getattr(cfg, "explicit_pid_kd_scale" if use_pid else "explicit_pd_damping_scale", 1.0))
        ess = float(getattr(cfg, "explicit_pid_effort_scale" if use_pid else "explicit_pd_effort_scale", 1.0))

        kp = torch.tensor([
            getattr(cfg, "explicit_pd_kp_drehzapfen", 1.5e7),
            getattr(cfg, "explicit_pd_kp_ausleger_i", 6.0e6),
            getattr(cfg, "explicit_pd_kp_ausleger_ii", 1.0e6),
            getattr(cfg, "explicit_pd_kp_messerkopf", 1.0e6),
        ], device=self.device, dtype=torch.float32) * kps
        kp = kp * torch.as_tensor(
            getattr(cfg, "explicit_pid_kp_per_cylinder", [1.0, 1.0, 1.0, 1.0]),
            device=self.device,
            dtype=torch.float32,
        )

        kd = torch.tensor([
            getattr(cfg, "explicit_pd_kd_drehzapfen", 3.0e6),
            getattr(cfg, "explicit_pd_kd_ausleger_i", 1.8e6),
            getattr(cfg, "explicit_pd_kd_ausleger_ii", 1.5e6),
            getattr(cfg, "explicit_pd_kd_messerkopf", 1.5e6),
        ], device=self.device, dtype=torch.float32) * kds
        kd = kd * torch.as_tensor(
            getattr(cfg, "explicit_pid_kd_per_cylinder", [1.0, 1.0, 1.0, 1.0]),
            device=self.device,
            dtype=torch.float32,
        )

        effort_lim = torch.tensor([
            getattr(cfg, "explicit_pd_effort_limit_drehzapfen", 1.0e8),
            getattr(cfg, "explicit_pd_effort_limit_ausleger_i", 1.0e7),
            getattr(cfg, "explicit_pd_effort_limit_ausleger_ii", 1.0e6),
            getattr(cfg, "explicit_pd_effort_limit_messerkopf", 1.0e5),
        ], device=self.device, dtype=torch.float32) * ess

        cyl_ids = self._cylinder_joint_ids
        q = self.robot.data.joint_pos[:, cyl_ids]
        qd = self.robot.data.joint_vel[:, cyl_ids]
        # Velocity filtering for D term
        vf_beta = float(getattr(cfg, "explicit_pd_velocity_filter_beta", 0.8))
        self._qd_filt = vf_beta * self._qd_filt + (1.0 - vf_beta) * qd
        qd_use = self._qd_filt  # use filtered velocity for D term
        error = q_des - q
        dt = max(float(getattr(self, "step_dt", float(cfg.sim.dt) * float(cfg.decimation))), 1.0e-6)

        if use_bias:
            # ---- Bias calibration: accumulate force_cmd over window ----
            calib_time_s = float(getattr(cfg, "bias_calibration_time_s", 2.0))
            window_s = float(getattr(cfg, "bias_average_window_s", 0.5))
            calib_steps = int(calib_time_s / (dt / float(cfg.decimation)))
            window_steps = int(window_s / dt)

            # Check calibration state BEFORE incrementing
            in_calib = self._explicit_bias_calib_step < calib_steps
            self._explicit_bias_calib_step += 1

            # Calibration phase: PID only (use calibration ki, not runtime ki)
            if in_calib.any():
                self._explicit_pid_integral_error[in_calib] += error[in_calib] * dt
                i_limit = float(getattr(cfg, "explicit_pid_integral_limit_m", 0.01))
                self._explicit_pid_integral_error = torch.clamp(
                    self._explicit_pid_integral_error, -i_limit, i_limit)
                ki_calib_scale = float(getattr(cfg, "bias_calibration_ki_scale", 3.0))
                ki = torch.tensor([
                    getattr(cfg, "explicit_pid_ki_drehzapfen", 1.0e6),
                    getattr(cfg, "explicit_pid_ki_ausleger_i", 5.0e5),
                    getattr(cfg, "explicit_pid_ki_ausleger_ii", 1.0e5),
                    getattr(cfg, "explicit_pid_ki_messerkopf", 1.0e5),
                ], device=self.device, dtype=torch.float32) * ki_calib_scale
                force_raw = (kp.unsqueeze(0) * error
                             + ki.unsqueeze(0) * self._explicit_pid_integral_error
                             - kd.unsqueeze(0) * qd_use)
                # Force rate limiting
                rate_frac = float(getattr(cfg, "explicit_force_rate_limit_fraction", 0.01))
                rate_scale = torch.as_tensor(
                    getattr(cfg, "explicit_force_rate_limit_per_cylinder", [1.0, 1.0, 1.0, 1.0]),
                    device=self.device,
                    dtype=torch.float32,
                )
                max_delta = rate_frac * rate_scale.unsqueeze(0) * effort_lim.unsqueeze(0)
                force_delta = force_raw - self._prev_force_cmd
                force_delta = torch.clamp(force_delta, -max_delta, max_delta)
                force_raw_limited = self._prev_force_cmd + force_delta
                force_cmd = torch.clamp(force_raw_limited, -effort_lim.unsqueeze(0), effort_lim.unsqueeze(0))

                # Accumulate force_cmd over final window for bias capture
                in_window = in_calib & (self._explicit_bias_calib_step >= calib_steps - window_steps)
                requires_static = bool(getattr(cfg, "bias_calibration_requires_static", True))
                static_vel_thresh = float(getattr(cfg, "bias_static_velocity_threshold_mps", 0.02))
                static_err_thresh = float(getattr(cfg, "bias_static_error_threshold_m", 0.02))
                if requires_static:
                    is_static = (qd_use.abs().mean(dim=-1) < static_vel_thresh) & (error.abs().mean(dim=-1) < static_err_thresh)
                    is_static_4d = is_static.unsqueeze(-1).expand(-1, 4)
                    in_window_use = in_window.unsqueeze(-1).expand_as(is_static_4d) & is_static_4d
                    count_mask = in_window & is_static
                else:
                    in_window_use = in_window.unsqueeze(-1).expand_as(error)
                    count_mask = in_window
                if in_window_use.any():
                    self._explicit_force_bias_accum[in_window_use] += force_cmd[in_window_use]
                if count_mask.any():
                    self._explicit_bias_window_count[count_mask] += 1

            # Transition: capture bias from accumulated force_cmd
            just_finished = ~in_calib & ~self._explicit_bias_calibrated
            if just_finished.any():
                counts = self._explicit_bias_window_count[just_finished].clamp(min=1)
                self._explicit_force_bias[just_finished] = (
                    self._explicit_force_bias_accum[just_finished] / counts.unsqueeze(-1))
                self._explicit_pid_integral_error[just_finished] = 0.0
                self._explicit_bias_calibrated[just_finished] = True

            # Runtime phase: bias + P+D (no integral, or very small ki)
            runtime_ki_scale = float(getattr(cfg, "bias_runtime_ki_scale", 0.0))
            force_raw = self._explicit_force_bias + kp.unsqueeze(0) * error - kd.unsqueeze(0) * qd_use
            if runtime_ki_scale > 0:
                self._explicit_pid_integral_error += error * dt
                i_limit = float(getattr(cfg, "explicit_pid_integral_limit_m", 0.01))
                self._explicit_pid_integral_error = torch.clamp(
                    self._explicit_pid_integral_error, -i_limit, i_limit)
                ki_runtime = torch.tensor([
                    getattr(cfg, "explicit_pid_ki_drehzapfen", 1.0e6),
                    getattr(cfg, "explicit_pid_ki_ausleger_i", 5.0e5),
                    getattr(cfg, "explicit_pid_ki_ausleger_ii", 1.0e5),
                    getattr(cfg, "explicit_pid_ki_messerkopf", 1.0e5),
                ], device=self.device, dtype=torch.float32) * runtime_ki_scale
                force_raw = force_raw + ki_runtime.unsqueeze(0) * self._explicit_pid_integral_error
            force_cmd = torch.clamp(force_raw, -effort_lim.unsqueeze(0), effort_lim.unsqueeze(0))

            # Online slow bias update (when nearly static)
            update_rate = float(getattr(cfg, "bias_online_update_rate", 0.005))
            err_thresh = float(getattr(cfg, "bias_online_error_threshold_m", 0.001))
            vel_thresh = float(getattr(cfg, "bias_online_velocity_threshold_mps", 0.01))
            static = (error.abs() < err_thresh) & (qd_use.abs() < vel_thresh)
            static = static.bool()
            calibrated = self._explicit_bias_calibrated.bool()
            if calibrated.ndim == 1:
                calibrated = calibrated.unsqueeze(-1).expand_as(static)
            elif calibrated.ndim == 2 and calibrated.shape != static.shape:
                raise RuntimeError(
                    f"Bad calibrated shape: {calibrated.shape}, static shape: {static.shape}"
                )
            update_mask = static & calibrated
            if static.any() and self._explicit_bias_calibrated.any():
                # belt-and-suspenders shape check
                if update_mask.shape != (B_sh, 4):
                    raise RuntimeError(
                        f"Bad update_mask shape: {update_mask.shape}, expected {(B_sh, 4)}"
                    )
                self._explicit_force_bias[update_mask] = (
                    (1.0 - update_rate) * self._explicit_force_bias[update_mask]
                    + update_rate * force_cmd[update_mask])

        elif use_pid:
            # Standard PID with anti-windup
            self._explicit_pid_integral_error += error * dt
            i_limit = float(getattr(cfg, "explicit_pid_integral_limit_m", 0.01))
            self._explicit_pid_integral_error = torch.clamp(
                self._explicit_pid_integral_error, -i_limit, i_limit)
            kis = float(getattr(cfg, "explicit_pid_ki_scale", 1.0))
            ki = torch.tensor([
                getattr(cfg, "explicit_pid_ki_drehzapfen", 1.0e6),
                getattr(cfg, "explicit_pid_ki_ausleger_i", 5.0e5),
                getattr(cfg, "explicit_pid_ki_ausleger_ii", 1.0e5),
                getattr(cfg, "explicit_pid_ki_messerkopf", 1.0e5),
            ], device=self.device, dtype=torch.float32) * kis
            force_raw = (kp.unsqueeze(0) * error
                         + ki.unsqueeze(0) * self._explicit_pid_integral_error
                         - kd.unsqueeze(0) * qd_use)
            force_cmd = torch.clamp(force_raw, -effort_lim.unsqueeze(0), effort_lim.unsqueeze(0))
            # Anti-windup: freeze integral when saturated
            saturated = (force_raw.abs() >= effort_lim.unsqueeze(0))
            pushing_same = (torch.sign(force_raw) == torch.sign(error))
            freeze = saturated & pushing_same
            self._explicit_pid_integral_error = torch.where(
                freeze, self._explicit_pid_integral_error - error * dt, self._explicit_pid_integral_error)
        else:
            # Pure PD
            force_cmd = torch.clamp(kp.unsqueeze(0) * error - kd.unsqueeze(0) * qd_use,
                                    -effort_lim.unsqueeze(0), effort_lim.unsqueeze(0))

        force_cmd = torch.nan_to_num(force_cmd, nan=0.0, posinf=0.0, neginf=0.0)
        self.robot.set_joint_effort_target(force_cmd, joint_ids=cyl_ids)

        # Store for logging
        self._last_force_cmd = force_cmd.detach().clone()
        self._prev_force_cmd = force_cmd.detach().clone()
        return force_cmd, error
    def _apply_action(self):
        """Apply either learned cylinder targets or the legacy direct joint targets."""
        if self._debug_manual_cylinder_position_target is not None:
            if str(getattr(self.cfg, "lowlevel_actuator_backend", "implicit_position")) in ("explicit_effort_pd", "explicit_effort_pid", "explicit_effort_pid_bias"):
                self._explicit_effort_target(self._debug_manual_cylinder_position_target)
            else:
                self.robot.set_joint_position_target(
                    self._debug_manual_cylinder_position_target,
                    joint_ids=self._cylinder_joint_ids,
                )
            if self._debug_manual_tool_position_target is not None:
                self.robot.set_joint_position_target(
                    self._debug_manual_tool_position_target,
                    joint_ids=[self._tool_joint_id],
                )
        elif bool(getattr(self.cfg, "use_cylinder_id_pipeline", False)):
            if str(getattr(self.cfg, "cylinder_actuation_mode", "position")) == "velocity":
                v_des_usd = self._cylinder_adapter.simscape_delta_to_isaac_delta(
                    self._cyl_vel_desired * float(self._learned_model_dt)
                ) / max(float(self._learned_model_dt), 1e-9)
                vel_scale = float(getattr(self.cfg, "cylinder_velocity_target_scale", 1.0))
                self.robot.set_joint_velocity_target(
                    v_des_usd * vel_scale, joint_ids=self._cylinder_joint_ids
                )
            else:
                if str(getattr(self.cfg, "lowlevel_actuator_backend", "implicit_position")) in ("explicit_effort_pd", "explicit_effort_pid", "explicit_effort_pid_bias"):
                    self._explicit_effort_target(self._cylinder_pos_target)
                else:
                    self.robot.set_joint_position_target(self._cylinder_pos_target, joint_ids=self._cylinder_joint_ids)
            if self._joint6_coupling_target_ref_index is not None and self._joint6_coupling_target_joint_id is not None:
                tool_target = self._joint_ref[:, self._joint6_coupling_target_ref_index : self._joint6_coupling_target_ref_index + 1]
                self.robot.set_joint_position_target(tool_target, joint_ids=[self._joint6_coupling_target_joint_id])
            elif str(getattr(self.cfg, "tool_joint_control_mode", "direct_position")) == "direct_position":
                tool_target = self._joint_ref[:, -1:]
                self.robot.set_joint_position_target(tool_target, joint_ids=[self._tool_joint_id])
        else:
            q_now = self.robot.data.joint_pos[:, self._arm_joint_ids]
            action_width = min(self._current_action.shape[-1], q_now.shape[-1])
            arm_targets = q_now.clone()
            arm_targets[:, :action_width] = q_now[:, :action_width] + self._current_action[:, :action_width] * self.cfg.action_scale
            limits = self.robot.data.soft_joint_pos_limits[:, self._arm_joint_ids]
            arm_targets = torch.clamp(arm_targets, limits[..., 0], limits[..., 1])
            arm_vel_targets = (arm_targets - q_now) / max(float(getattr(self, "step_dt", float(self.cfg.sim.dt) * float(self.cfg.decimation))), 1.0e-6)
            arm_targets, _ = self._apply_joint6_coupling_to_reference(arm_targets, arm_vel_targets)
            self.robot.set_joint_position_target(arm_targets, joint_ids=self._arm_joint_ids)

        # Steering: hold at the configured scripted target (zero by default).
        # Stop-and-mow relocation updates this scalar with a small heading/lateral
        # feedback correction; all existing tasks retain straight-ahead zero.
        steering_override = getattr(self, "_scripted_steering_target", None)
        if steering_override is None:
            steering_target = torch.full(
                (self.num_envs, 2),
                float(getattr(self.cfg, "vehicle_steering_target_rad", 0.0)),
                device=self.device,
            )
        else:
            steering_target = torch.as_tensor(
                steering_override, device=self.device, dtype=torch.float32
            ).reshape(self.num_envs, 2)
        self.robot.set_joint_position_target(
            steering_target,
            joint_ids=self._steering_joint_ids,
        )

        # Wheels: match the student-teacher task and command wheel angular velocity directly.
        if self.cfg.enable_scripted_vehicle_motion:
            speed_rad_per_s = (
                self._vehicle_speed_kmh / 3.6 / self.cfg.vehicle_wheel_radius_m
            )
            speed_rad_per_s = speed_rad_per_s * float(
                getattr(self.cfg, "vehicle_wheel_speed_gain", 1.0)
            )
            wheel_vel = speed_rad_per_s.unsqueeze(-1).expand(-1, len(self._wheel_joint_ids))
            left_scale = float(getattr(self.cfg, "vehicle_left_wheel_speed_scale", 1.0))
            right_scale = float(getattr(self.cfg, "vehicle_right_wheel_speed_scale", 1.0))
            if len(self._wheel_joint_ids) == 4:
                wheel_scale = torch.tensor(
                    [left_scale, right_scale, left_scale, right_scale],
                    device=self.device,
                    dtype=wheel_vel.dtype,
                ).reshape(1, 4)
                wheel_vel = wheel_vel * wheel_scale
        else:
            wheel_vel = torch.zeros((self.num_envs, len(self._wheel_joint_ids)), device=self.device)
        self.robot.write_joint_velocity_to_sim(wheel_vel, joint_ids=self._wheel_joint_ids)

        # Update previous action for next step (critical for action rate penalty and observations)
        self._prev_action = self._current_action.clone()
        self._prev_valve_cmd = self._current_valve_cmd.clone()

        # Resample task-space command
        self._command_resample_timer += self.step_dt
        resample_cmd = self._command_resample_timer >= self._command_resample_interval
        if resample_cmd.any():
            self._resample_command(torch.where(resample_cmd)[0])

        # Update vehicle speed periodically
        self._vehicle_speed_resample_counter += 1
        resample = self._vehicle_speed_resample_counter >= self.cfg.vehicle_speed_resample_step_threshold
        if resample.any():
            new_speed = (
                torch.rand(self.num_envs, device=self.device)
                * (self.cfg.vehicle_speed_max_kmh - self.cfg.vehicle_speed_min_kmh)
                + self.cfg.vehicle_speed_min_kmh
            )
            self._vehicle_speed_kmh = torch.where(resample, new_speed, self._vehicle_speed_kmh)
            self._vehicle_speed_resample_counter = torch.where(
                resample, torch.zeros_like(self._vehicle_speed_resample_counter), self._vehicle_speed_resample_counter
            )

    # =========================================================================
    # Reset
    # =========================================================================

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)

        # Write root pose and joint state to simulation
        default_root_pose = self.robot.data.default_root_state[env_ids, :7].clone()
        default_root_pose[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(default_root_pose, env_ids)
        self.robot.write_root_velocity_to_sim(
            torch.zeros_like(self.robot.data.default_root_state[env_ids, 7:13]), env_ids
        )
        self.robot.write_joint_state_to_sim(
            self.robot.data.default_joint_pos[env_ids].clone(),
            self.robot.data.default_joint_vel[env_ids].clone(),
            None, env_ids,
        )


        if bool(getattr(self.cfg, "enable_reset_debug", False)):
            selected = set(self._selected_debug_env_ids())
            reset_env_ids = [int(eid) for eid in env_ids.detach().cpu().tolist() if int(eid) in selected]
            for eid in reset_env_ids:
                root_pos = self.robot.data.root_pos_w[eid].detach().cpu().tolist()
                cyl_pos = self.robot.data.joint_pos[eid, self._cylinder_joint_ids].detach().cpu().tolist()
                cyl_qd = self.robot.data.joint_vel[eid, self._cylinder_joint_ids].detach().cpu().tolist()
                print(f"[MulagUpperLevelEnv][reset] env[{eid}] root_pos_w={root_pos} cylinder_q={cyl_pos} cylinder_qdot={cyl_qd}", flush=True)
        # Reset history buffers
        self._obs_joint_pos_hist[env_ids] = 0.0
        self._obs_joint_vel_hist[env_ids] = 0.0
        self._obs_distance_to_goal_hist[env_ids] = 0.0

        # Reset hysteresis state

        # Reset PPO/action and learned cylinder pipeline state.
        default_joint_pos = self.robot.data.default_joint_pos[env_ids]
        default_joint_vel = self.robot.data.default_joint_vel[env_ids]
        default_arm_pos = default_joint_pos[:, self._arm_joint_ids]
        default_arm_vel = default_joint_vel[:, self._arm_joint_ids]
        default_cylinder_pos = default_joint_pos[:, self._cylinder_joint_ids]
        default_cylinder_vel = default_joint_vel[:, self._cylinder_joint_ids]

        self._prev_action[env_ids] = 0.0
        self._current_action[env_ids] = 0.0
        self._ppo_raw_action[env_ids] = 0.0
        self._action_hydraulic[env_ids] = 0.0
        self._action_ignored_passive[env_ids] = 0.0
        self._action_delta_q_usd_ref[env_ids] = 0.0
        self._action_target_delta_x_pre_gain_learned[env_ids] = 0.0
        self._action_target_delta_x_post_gain_learned[env_ids] = 0.0
        self._action_target_delta_x_safety_limit_learned[env_ids] = 0.0
        self._action_target_delta_x_learned[env_ids] = 0.0
        self._action_target_delta_x_pre_gain_clip_mask[env_ids] = False
        self._action_target_delta_x_post_gain_clip_mask[env_ids] = False
        self._action_target_delta_xdot_raw_learned[env_ids] = 0.0
        self._action_target_delta_xdot_learned[env_ids] = 0.0
        self._action_target_delta_xdot_clip_mask[env_ids] = False
        self._joint_ref[env_ids] = default_arm_pos
        self._joint_vel_ref[env_ids] = default_arm_vel
        self._cylinder_pos_target[env_ids] = default_cylinder_pos
        self._neutral_hold_cylinder_pos_target[env_ids] = default_cylinder_pos
        self._prev_valve_cmd[env_ids] = 0.0
        self._id_prev_valve_cmd_input[env_ids] = 0.0
        self._current_valve_cmd[env_ids] = 0.0
        self._fd_baseline_valve_cmd[env_ids] = 0.0
        self._id_input_tensor[env_ids] = 0.0
        self._apply_hydraulic_feature_mode(env_ids)
        self._predicted_extension_velocity[env_ids] = 0.0
        self._raw_delta_extension[env_ids] = 0.0
        self._raw_delta_extension_velocity[env_ids] = 0.0
        self._raw_delta_dP[env_ids] = 0.0
        self._predicted_delta_extension[env_ids] = 0.0
        self._delta_isaac[env_ids] = 0.0
        self._cyl_vel_error_integral[env_ids] = 0.0
        self._servo_position_delta[env_ids] = 0.0
        self._servo_tracking_delta[env_ids] = 0.0
        self._tracking_e_int[env_ids] = 0.0
        self._tracking_e_prev[env_ids] = 0.0
        self._pos_pi_x_cmd[env_ids] = 0.0
        self._pos_pi_e_int[env_ids] = 0.0
        self._pos_pi_e_prev[env_ids] = 0.0
        self._cyl_vel_error_prev[env_ids] = 0.0
        self._cyl_vel_desired[env_ids] = 0.0
        self._cyl_vel_measured[env_ids] = 0.0
        self._valve_ff[env_ids] = 0.0
        self._valve_pid[env_ids] = 0.0
        self._valve_total[env_ids] = 0.0
        self._fd_delta_raw[env_ids] = 0.0
        self._fd_delta_corrected[env_ids] = 0.0
        self.clear_debug_manual_cylinder_targets()
        self._explicit_pid_integral_error[env_ids] = 0.0
        self._explicit_force_bias[env_ids] = 0.0
        self._explicit_bias_calibrated[env_ids] = False
        self._explicit_bias_calib_step[env_ids] = 0
        self._explicit_force_bias_accum[env_ids] = 0.0
        self._explicit_bias_window_count[env_ids] = 0
        self._qd_filt[env_ids] = 0.0
        self._prev_force_cmd[env_ids] = 0.0
        self.clear_debug_cylinder_velocity_target()
        self._neutral_hold_mask[env_ids] = False
        self._neutral_hold_prev_mask[env_ids] = False
        if self._cylinder_adapter is not None:
            x_current, xdot_current = self._cylinder_adapter.current_from_cylinder_joints(
                default_cylinder_pos,
                default_cylinder_vel,
            )
            self._cylinder_ref[env_ids] = x_current
            self._cylinder_vel_ref[env_ids] = xdot_current
            self._cylinder_x_current[env_ids] = x_current
            self._cylinder_xdot_current[env_ids] = xdot_current
        else:
            self._cylinder_ref[env_ids] = 0.0
            self._cylinder_vel_ref[env_ids] = 0.0
            self._cylinder_x_current[env_ids] = 0.0
            self._cylinder_xdot_current[env_ids] = 0.0
        if self._cylinder_id is not None:
            self._cylinder_id.reset(env_ids)
        if self._cylinder_rom is not None:
            self._cylinder_rom.reset(env_ids)
        if self._base_cylinder_rom is not None:
            self._base_cylinder_rom.reset(env_ids)
        if self._command_cylinder_rom is not None:
            self._command_cylinder_rom.reset(env_ids)
        if self._cylinder_fd is not None:
            self._cylinder_fd.reset(env_ids)
        # Sequence model history initialization from reset state
        if self._sequence_id is not None:
            self._sequence_id.reset(env_ids)
            # Fill histories with reset state (zero valve cmd)
            zero_valve = torch.zeros((self.num_envs, 4), device=self.device)
            self._sequence_id.init_history_from_state(
                env_ids,
                extension=self._cylinder_x_current,
                extension_velocity=self._cylinder_xdot_current,
                previous_valve_cmd=zero_valve,
            )
        if self._sequence_fd is not None:
            self._sequence_fd.reset(env_ids)
            zero_valve = torch.zeros((self.num_envs, 4), device=self.device)
            self._sequence_fd.init_history_from_state(
                env_ids,
                valve_cmd=zero_valve,
                extension=self._cylinder_x_current,
                extension_velocity=self._cylinder_xdot_current,
            )
        # Reset sequence decimation state
        if self._sequence_id is not None or self._sequence_fd is not None:
            self._sequence_tick_counter = 0
            self._sequence_step_counter = 0
            self._sequence_tick_active = True
            self._fd_delta_plan = torch.zeros(
                (self.num_envs, 4), device=self.device, dtype=torch.float32
            )

        # Resample command
        self._resample_command(env_ids)

        # Resample vehicle speed
        self._vehicle_speed_kmh[env_ids] = (
            torch.rand(len(env_ids), device=self.device)
            * (self.cfg.vehicle_speed_max_kmh - self.cfg.vehicle_speed_min_kmh)
            + self.cfg.vehicle_speed_min_kmh
        )
        self._vehicle_speed_resample_counter[env_ids] = 0

    def _resample_command(self, env_ids: torch.Tensor):
        """Sample new task-space position command (XY in base frame).
        Yaw = 0 (aligned with vehicle forward, matching default cutter pose).
        """
        n = len(env_ids)
        rx = self.cfg.command_pos_x_range
        ry = self.cfg.command_pos_y_range
        new_x = torch.rand(n, device=self.device) * (rx[1] - rx[0]) + rx[0]
        new_y = torch.rand(n, device=self.device) * (ry[1] - ry[0]) + ry[0]
        self._command_pos_w[env_ids, 0] = new_x
        self._command_pos_w[env_ids, 1] = new_y
        self._command_pos_w[env_ids, 2] = self.cfg.command_pos_z

        # Identity quaternion (w=1, x=y=z=0) — matches default cutter pose
        self._command_quat_w[env_ids, 0] = 1.0
        self._command_quat_w[env_ids, 1] = 0.0
        self._command_quat_w[env_ids, 2] = 0.0
        self._command_quat_w[env_ids, 3] = 0.0

        tr = self.cfg.command_resample_time_s_range
        self._command_resample_interval[env_ids] = (
            torch.rand(n, device=self.device) * (tr[1] - tr[0]) + tr[0]
        )
        self._command_resample_timer[env_ids] = 0.0



    # =========================================================================
    # Preflight stubs (train script compatibility — no hydraulic to validate)
    # =========================================================================

    def run_asset_transmission_preflight(self, *, steps=80, settle_steps=None, pulse_m=0.01,
                                          disable_ordinary_resets=False, **kwargs):
        """Stub: upper-level control has no hydraulic transmission to validate."""
        return {
            "passed": True,
            "preflight_stage": "asset_transmission_upper_level",
            "preflight_decision": "skipped",
            "reset_to_settle_root_delta": [0.0, 0.0, 0.0],
            "reset_during_settle": False,
            "reset_during_pulse": False,
            "failure_category": None,
        }

    def run_learned_pipeline_preflight(self):
        """Import-level learned-pipeline check for the upper-level PPO action path."""
        pipeline_enabled = bool(getattr(self.cfg, "use_cylinder_id_pipeline", False))
        return {
            "passed": True,
            "preflight_stage": "learned_pipeline_upper_level",
            "preflight_decision": "ready" if pipeline_enabled else "skipped",
            "pipeline_enabled": pipeline_enabled,
            "dynamics_model": str(getattr(self.cfg, "cylinder_dynamics_model", "")),
            "id_checkpoint": str(self._cylinder_id.checkpoint_path) if self._cylinder_id is not None else "",
            "rom_checkpoint": str(self._cylinder_rom.checkpoint_path) if self._cylinder_rom is not None else "",
            "fd_checkpoint": str(self._cylinder_fd.checkpoint_path) if self._cylinder_fd is not None else "",
            "failure_category": None,
        }
