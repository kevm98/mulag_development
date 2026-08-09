"""PPO3: the policy is the only controller in the scored loop.

PPO2's scored control path was four layers deep::

    trajectory -> EE Jacobian cascade -> PPO residual (+-2 mm) -> cylinder DLS
                                                              -> ROM-A -> PhysX

PPO3 deletes the two classical layers::

    trajectory -> PPO command -> ROM-A -> PhysX

The cascade and the DLS controller are still constructed by the parent class and
are still used, but only inside ``_reset_idx``: the cosine preposition, the
settle, and ``_warm_command_rom_at_target`` all run before the first scored
transition and are outside the PPO episode by the same contract that already put
the direct-PhysX deployment there.  ``test_sole_control_path`` asserts that
nothing the cascade computes reaches the plant during a scored step.

The plant is ROM-A, the canonical command-conditioned ROM
(``plant_reachable_v1_finetune_blend25/best_rom.pt``).  It consumes a desired
motion command and predicts its own valve internally, so the policy's action is
a cylinder command rather than a raw valve; the ROM's valve head is part of the
plant, not part of the controller.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from mulag_plant.mdp.actions import sanitize_cylinder_action
from mulag_plant.policy.env import ResidualPolicyEnv
from verge_task.actions import apply_sole_action
from verge_task.distill import TEACHER_MODES
from verge_task.observations import (
    build_sole_observation,
    finite_difference_kinematics,
    sole_observation_dim,
)
from verge_task.rewards import compute_sole_reward


class SolePolicyEnv(ResidualPolicyEnv):
    """Sole-controller task over ROM-A."""

    _sole_applied: torch.Tensor | None = None

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        # 4 hydraulic cylinders, plus Messerkopf_joint when it is driven as a
        # real joint rather than slaved to c4. Any other width is a mismatched
        # checkpoint or a mis-set config, and must fail loudly rather than
        # silently truncate the command.
        expected_actions = 5 if bool(getattr(cfg, "sole_head_joint_action", False)) else 4
        if int(getattr(cfg, "action_space", 4)) != expected_actions:
            raise ValueError(
                f"PPO3 requires {expected_actions} actions, got "
                f"{getattr(cfg, 'action_space', None)}"
            )
        super().__init__(cfg, render_mode=render_mode, **kwargs)

        self._sole_ik_hint = bool(getattr(cfg, "sole_ik_hint", False))
        expected_obs = self._expected_observation_dim()
        configured_obs = int(getattr(cfg, "observation_space", expected_obs))
        if configured_obs != expected_obs:
            raise ValueError(
                f"observation_space={configured_obs} but sole_ik_hint="
                f"{self._sole_ik_hint} requires {expected_obs}"
            )

        zeros = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self._sole_raw = zeros.clone()
        self._sole_filtered = zeros.clone()
        self._sole_applied = zeros.clone()
        self._prev_sole_applied = zeros.clone()
        self._sole_shield_gain = torch.ones_like(zeros)
        head_zeros = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._head_action_raw = head_zeros.clone()
        self._head_action_filtered = head_zeros.clone()
        self._head_action_applied = head_zeros.clone()
        self._prev_head_action_applied = head_zeros.clone()
        self._head_joint_target = torch.full_like(
            head_zeros, float(getattr(cfg, "sole_head_neutral_position_rad", 0.0))
        )
        self._tool_kinematics_common_step: int | None = None

        self._teacher_mode = str(getattr(cfg, "sole_teacher_mode", "off")).lower()
        if self._teacher_mode not in TEACHER_MODES:
            raise ValueError(
                f"sole_teacher_mode must be one of {TEACHER_MODES}, "
                f"got {self._teacher_mode!r}"
            )
        self._teacher_action = zeros.clone()
        self._teacher_execute_mask = torch.zeros(
            (self.num_envs,), dtype=torch.bool, device=self.device
        )

        scaling = self._cylinder_adapter.scaling
        self._learned_min = scaling.learned_min.to(self.device, torch.float32).reshape(1, 4)
        self._learned_max = scaling.learned_max.to(self.device, torch.float32).reshape(1, 4)
        self._learned_span = torch.clamp(self._learned_max - self._learned_min, min=1.0e-6)
        self._usd_per_extension_scale = scaling.scale.to(
            self.device, torch.float32
        ).reshape(1, 4)
        self._action_to_learned_sign = torch.sign(
            self._usd_per_extension_scale
        )

        # Report the actuator this env ACTUALLY built, not a hard-coded label.
        # PPO4 runs this same class over ID/FD, and a banner that always says
        # "ROM-A" makes a log look like a ROM run when it is not -- precisely the
        # mislabelling the actuator tagging elsewhere exists to prevent.
        _dynamics = str(getattr(cfg, "cylinder_dynamics_model", "command_rom")).lower()
        _plant_label = {
            "command_rom": "ROM-A command ROM",
            "base_rom": "base ROM",
            "rom": "ROM",
            "fd": "ID -> FD",
            "direct_valve": "direct valve",
        }.get(_dynamics, _dynamics)
        _path_label = "policy -> ID -> FD -> PhysX" if _dynamics == "fd" else (
            "policy -> ROM -> PhysX"
        )
        print(
            f"[PPO3-SOLE] obs={expected_obs} action={expected_actions} | plant={_plant_label} | "
            f"scored adaptive path = {_path_label} (no EE cascade, no "
            f"cylinder DLS) | action_scale={float(cfg.action_scale):.4f} m "
            f"rate<={float(getattr(cfg, 'sole_max_action_rate', 0.2)):.3f}/step "
            f"shield_margin={float(getattr(cfg, 'sole_limit_shield_margin_m', 0.0)):.3f} m"
            + (
                " | *** IK-HINT ABLATION: the cascade's cylinder error is in the "
                "observation. This run is an ablation and is not a sole-controller "
                "result. ***"
                if self._sole_ik_hint
                else ""
            )
            + (
                f" | *** DISTILLATION: teacher_mode={self._teacher_mode}. The DLS "
                "teacher is active. No result from this run is a sole-controller "
                "result. ***"
                if self._teacher_mode != "off"
                else ""
            ),
            flush=True,
        )

    # ------------------------------------------------------------------
    # Cylinder limit geometry
    # ------------------------------------------------------------------

    def _expected_observation_dim(self) -> int:
        """Observation width this environment produces.

        Overridable: the verge-mowing subclass emits a different observation
        (ray-cast height profile instead of the trajectory preview), so the
        check has to ask the subclass rather than assume the sole layout.
        """
        return sole_observation_dim(ik_hint=self._sole_ik_hint)

    def _cylinder_headroom(
        self, measured: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Metres of travel remaining below and above the measured extension."""
        low = torch.clamp(measured - self._learned_min, min=0.0)
        high = torch.clamp(self._learned_max - measured, min=0.0)
        return low, high

    def _commanded_target_lead(self, measured: torch.Tensor) -> torch.Tensor:
        """Learned-space gap between the commanded PhysX target and the plant."""
        scaling = self._cylinder_adapter.scaling
        target_learned = scaling.usd_q_to_learned_x(self._cylinder_pos_target)
        return target_learned - measured

    def _task_action_shield(
        self, applied_action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Task-specific safety projection; the base sole task needs none."""
        return applied_action, torch.ones_like(applied_action)

    # ------------------------------------------------------------------
    # Distillation teacher (training loop only; never a deployed layer)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_teacher_action(self) -> torch.Tensor:
        """The command the deleted cylinder DLS would issue for this state.

        Queried only when ``sole_teacher_mode != "off"``. Calling it advances the
        DLS controller's own smoothing state, which is correct and required: the
        teacher must track along whatever trajectory the *student* is producing,
        which is what makes this DAgger rather than plain behaviour cloning.
        """
        action_4d, _ = self._cylinder_dls.action(
            self._baseline_cylinder_goal,
            goal_velocity=self._baseline_cylinder_velocity,
        )
        teacher = sanitize_cylinder_action(
            action_4d, num_envs=self.num_envs, action_dim=4
        ).to(self.device)

        if not bool(getattr(self.cfg, "sole_head_joint_action", False)):
            return teacher

        # The DLS inverts the CYLINDER Jacobian and knows nothing about
        # Messerkopf_joint -- it is not hydraulic and c4 only sets the boom
        # attitude the head hangs from. Returning the 4-wide command here would
        # crash the DAgger assignment against a 5-wide student action, and
        # padding it with zeros would be worse than crashing: the teacher would
        # actively demonstrate "never move the head", teaching the student to
        # abandon the joint this whole change exists to control.
        #
        # q5 rotates around the cutter's own local Z. It has exactly zero
        # authority over the down-axis tilt, so the retired levelling label was
        # physically wrong. The hydraulic DLS has no q5 demonstration; use the
        # configured neutral yaw target as the honest distillation label and let
        # PPO learn deviations from the yaw/parallel-sensor rewards.
        neutral = float(getattr(self.cfg, "sole_head_neutral_position_rad", 0.0))
        scale = max(abs(float(getattr(self.cfg, "sole_head_action_scale_rad", 1.40))), 1.0e-6)
        head_action = torch.full(
            (self.num_envs,), neutral / scale, dtype=teacher.dtype, device=self.device
        ).clamp(-1.0, 1.0)
        return torch.cat(
            [teacher, torch.nan_to_num(head_action).reshape(-1, 1)], dim=-1
        )

    def set_teacher_execute_mask(self, mask: torch.Tensor) -> None:
        """Choose which environments the teacher drives this step (DAgger beta)."""
        if self._teacher_mode == "off":
            raise RuntimeError(
                "sole_teacher_mode='off'; the teacher cannot be given control"
            )
        self._teacher_execute_mask = mask.to(
            device=self.device, dtype=torch.bool
        ).reshape(self.num_envs)

    @property
    def teacher_action(self) -> torch.Tensor:
        """Last teacher label. Zero unless the teacher is enabled."""
        return self._teacher_action

    # ------------------------------------------------------------------
    # Pre-physics step: the entire scored control path
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor):
        cfg = self.cfg
        actions = torch.nan_to_num(
            actions.to(dtype=torch.float32, device=self.device),
            nan=0.0, posinf=0.0, neginf=0.0,
        )
        expected_actions = (
            5 if bool(getattr(cfg, "sole_head_joint_action", False)) else 4
        )
        if tuple(actions.shape) != (int(self.num_envs), expected_actions):
            raise RuntimeError(
                f"Sole-controller action must be "
                f"({self.num_envs}, {expected_actions}), "
                f"got {tuple(actions.shape)}"
            )

        warmup_mask = (self._preposition_counter > 0) | (self._settle_counter > 0)
        if warmup_mask.any():
            raise RuntimeError(
                "physical preposition must finish inside reset before policy stepping"
            )

        # Advance the reference. This also refreshes the cascade's baseline goal,
        # which is retained for logging and for the ik_hint ablation only; in the
        # sole-controller contract it never reaches the plant.
        self._update_trajectory()

        measured, _ = self._refresh_cylinder_state()
        headroom_low, headroom_high = self._cylinder_headroom(measured)

        effective = (
            torch.zeros_like(actions)
            if bool(getattr(cfg, "force_zero_policy", False))
            else actions
        )
        # Split off the dedicated q5 channel. It is a direct position command
        # for Messerkopf_joint and must never enter the four-cylinder actuator
        # model. c4 remains the sole command for the hydraulic Schwenk linkage.
        if bool(getattr(cfg, "sole_head_joint_action", False)):
            self._head_action_raw = torch.clamp(
                torch.nan_to_num(
                    effective[:, 4], nan=0.0, posinf=0.0, neginf=0.0
                ),
                -1.0,
                1.0,
            )
            effective = effective[:, :4].clone()
            beta = min(max(float(getattr(cfg, "sole_head_action_filter", 0.75)), 0.0), 1.0)
            filtered = beta * self._head_action_filtered + (1.0 - beta) * self._head_action_raw
            max_rate = max(float(getattr(cfg, "sole_head_max_action_rate", 0.10)), 0.0)
            self._prev_head_action_applied = self._head_action_applied.clone()
            self._head_action_filtered = filtered
            self._head_action_applied = torch.clamp(
                filtered,
                self._head_action_applied - max_rate,
                self._head_action_applied + max_rate,
            )
            neutral = float(getattr(cfg, "sole_head_neutral_position_rad", 0.0))
            scale = abs(float(getattr(cfg, "sole_head_action_scale_rad", 1.40)))
            target = neutral + scale * self._head_action_applied
            limits = self.robot.data.soft_joint_pos_limits[:, int(self._tool_joint_id)]
            self._head_joint_target = torch.clamp(target, limits[:, 0], limits[:, 1])
        else:
            self._head_action_raw.zero_()
        self._prev_sole_applied = self._sole_applied.clone()
        output = apply_sole_action(
            policy_action=effective,
            previous_applied_action=self._sole_applied,
            filter_beta=float(getattr(cfg, "sole_action_filter", 0.75)),
            max_action_rate=float(getattr(cfg, "sole_max_action_rate", 0.20)),
            headroom_low_m=headroom_low,
            headroom_high_m=headroom_high,
            limit_shield_margin_m=float(
                getattr(cfg, "sole_limit_shield_margin_m", 0.0)
            ),
            action_to_extension_sign=self._action_to_learned_sign,
            action_authority=getattr(cfg, "sole_action_authority", (1.0,) * 4),
        )
        self._sole_raw = output.raw
        self._sole_filtered = output.filtered
        self._sole_applied = output.applied
        self._sole_shield_gain = output.shield_gain
        self._sole_applied, task_shield_gain = self._task_action_shield(
            self._sole_applied
        )
        self._sole_shield_gain = self._sole_shield_gain * task_shield_gain

        # The policy action IS the plant command. No DLS inversion, no cascade.
        self._ppo_raw_action = sanitize_cylinder_action(
            self._sole_applied, num_envs=self.num_envs, action_dim=4
        ).to(self.device)
        self._current_action = self._ppo_raw_action.clone()

        # ---- Everything below this line runs AFTER the plant command is final.
        # The distillation teacher may be read here and, in "execute" mode, may
        # replace the command -- but only because a distillation rollout is not a
        # sole-controller rollout. In the default "off" mode this is dead code,
        # and evaluation refuses to run in any other mode. The ordering is a
        # tested contract: see tests/test_sole_control_path.py.
        if self._teacher_mode != "off":
            self._teacher_action = self._compute_teacher_action()
            if self._teacher_mode == "execute" and self._teacher_execute_mask.any():
                mask = self._teacher_execute_mask
                # The teacher is five-wide when q5 is direct. Its first four
                # columns replace the hydraulic command; its fifth replaces the
                # independent joint command.
                teacher_cylinders = self._teacher_action[:, :4]
                self._current_action[mask] = teacher_cylinders[mask]
                self._ppo_raw_action[mask] = teacher_cylinders[mask]
                self._sole_applied[mask] = teacher_cylinders[mask]
                if (
                    getattr(self, "_head_action_raw", None) is not None
                    and self._teacher_action.shape[-1] > 4
                ):
                    self._head_action_raw[mask] = self._teacher_action[mask, 4]
                    self._head_action_filtered[mask] = self._teacher_action[mask, 4]
                    self._head_action_applied[mask] = self._teacher_action[mask, 4]
                    neutral = float(getattr(cfg, "sole_head_neutral_position_rad", 0.0))
                    scale = abs(float(getattr(cfg, "sole_head_action_scale_rad", 1.40)))
                    target = neutral + scale * self._head_action_applied
                    limits = self.robot.data.soft_joint_pos_limits[:, int(self._tool_joint_id)]
                    self._head_joint_target = torch.clamp(target, limits[:, 0], limits[:, 1])

        # Parent bookkeeping expects these to describe the commanded cylinder
        # state; keep them meaningful so inherited logging and the inherited
        # finiteness termination stay valid.
        self._corrected_cylinder_goal[:] = measured + self._commanded_target_lead(
            measured
        )
        self._cylinder_goal_x[:] = self._corrected_cylinder_goal
        self._residual_filtered[:] = self._sole_applied
        self._prev_residual_filtered[:] = self._prev_sole_applied

        previous_target = self._cylinder_pos_target.detach().clone()
        self._run_cylinder_id_pipeline()
        target_beta = min(
            max(float(getattr(cfg, "hydraulic_target_filter_beta", 0.0)), 0.0),
            1.0,
        )
        if target_beta > 0.0:
            # The inherited stage configuration deliberately specifies the same
            # 0.5 plant-target low-pass used by PPO2.  The sole-controller path
            # previously skipped it, so an otherwise smoothed PPO action still
            # jumped the high-gain PhysX target in one step.  This is actuator
            # dynamics, not a tracking controller: it filters only the target
            # the command ROM already produced and contributes no task command.
            self._cylinder_pos_target[:] = (
                target_beta * previous_target
                + (1.0 - target_beta) * self._cylinder_pos_target
            )
            for column, ref_index in enumerate(
                self._cylinder_adapter.joint_reference_indices
            ):
                self._joint_ref[:, ref_index] = self._cylinder_pos_target[:, column]
            source = self._joint6_coupling_source_ref_index
            target = self._joint6_coupling_target_ref_index
            if source is not None and target is not None:
                self._joint_ref[:, target] = (
                    float(getattr(cfg, "joint6_coupling_a", 0.76))
                    * self._joint_ref[:, source]
                    + float(getattr(cfg, "joint6_coupling_b", 0.0))
                )

        self._hold_wheels()
        self._constrain_truck_xy()

    @torch.no_grad()
    def _apply_action(self):
        """Write the direct q5 target after all hydraulic references are built."""
        if bool(getattr(self.cfg, "sole_head_joint_action", False)) and hasattr(
            self, "_head_joint_target"
        ):
            target_ref = getattr(self, "_joint6_coupling_target_ref_index", None)
            if target_ref is None:
                raise RuntimeError("direct q5 control requires Messerkopf_joint")
            self._joint_ref[:, int(target_ref)] = self._head_joint_target
        super()._apply_action()

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _update_measured_tool_kinematics(self) -> torch.Tensor:
        """Refresh truck-frame kinematics at most once per environment step.

        DirectRLEnv asks for rewards before observations.  Terrain rewards call
        this method first so the acceleration cost belongs to the transition
        that produced it; the later observation call must then reuse that
        sample rather than finite-difference the same pose a second time.
        """
        tool_pos_truck = self._compute_ee_pos_local()
        common_step = int(getattr(self, "common_step_counter", -1))
        if getattr(self, "_tool_kinematics_common_step", None) == common_step:
            return tool_pos_truck

        dt = max(float(self.step_dt), 1.0e-6)
        measured_velocity, measured_acceleration = finite_difference_kinematics(
            tool_pos_truck,
            self._prev_tool_pos_truck,
            self._previous_measured_tool_velocity_truck,
            dt,
        )
        self._measured_tool_velocity_truck = measured_velocity
        self._measured_tool_acceleration_truck = measured_acceleration
        self._previous_measured_tool_velocity_truck = (
            measured_velocity.detach().clone()
        )
        self._prev_tool_pos_truck = tool_pos_truck.detach().clone()
        self._tool_kinematics_common_step = common_step
        return tool_pos_truck

    def _get_observations(self) -> dict:
        if self._sole_applied is None:
            return super()._get_observations()

        dt = max(float(self.step_dt), 1.0e-6)

        tool_pos_truck = self._update_measured_tool_kinematics()

        ref_vel, ref_acc, preview_xy = self._reference_kinematics()

        x, xdot = self._refresh_cylinder_state()
        self._cylinder_x_current = x.detach().clone()
        self._cylinder_xdot_current = xdot.detach().clone()
        headroom_low, headroom_high = self._cylinder_headroom(x)

        head_alignment = self._measure_head_down_alignment()
        self._head_alignment_rate = (
            head_alignment - self._prev_head_alignment
        ) / dt
        self._prev_head_alignment = head_alignment.detach().clone()

        hint = None
        if self._sole_ik_hint:
            hint = self._baseline_cylinder_goal - x

        obs = build_sole_observation(
            tool_error_truck=self._original_reference_tool - tool_pos_truck,
            measured_tool_velocity_truck=self._measured_tool_velocity_truck,
            reference_tool_velocity_truck=ref_vel,
            reference_tool_acceleration_truck=ref_acc,
            measured_cylinder_extensions=x,
            measured_cylinder_velocities=xdot,
            cylinder_headroom_low=headroom_low / self._learned_span,
            cylinder_headroom_high=headroom_high / self._learned_span,
            previous_applied_action=self._sole_applied,
            current_valve_commands=self._current_valve_cmd,
            rom_delta_extension=self._fd_delta_corrected,
            commanded_target_lead=self._commanded_target_lead(x),
            reference_preview_xy_truck=preview_xy,
            trajectory_direction=self._traj_direction,
            cutter_bottom_world_clearance=self._clearance_signal(),
            head_down_alignment=head_alignment,
            head_down_alignment_rate=self._head_alignment_rate,
            ik_hint_cylinder_error=hint,
        )
        return {"policy": obs}

    def _reference_kinematics(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reference velocity, acceleration and XY preview in the truck frame.

        Extracted verbatim in behaviour from the parent observation builder so
        PPO2 and PPO3 share reference semantics; only the assembly differs.
        """
        import numpy as np

        ref_vel = torch.zeros((self.num_envs, 3), device=self.device)
        ref_acc = torch.zeros((self.num_envs, 3), device=self.device)
        preview_xy = torch.zeros((self.num_envs, 2), device=self.device)
        preview_time = max(float(getattr(self.cfg, "policy_preview_time_s", 0.25)), 0.0)

        for env_id in range(self.num_envs):
            path_distance = float(self._reference_timing_s[env_id].item())
            signed_speed = float(self._reference_path_speed[env_id].item())
            signed_timing_speed = float(self._reference_timing_speed[env_id].item())
            signed_accel = float(self._reference_path_accel[env_id].item())

            goal_plus, _, _ = self._interpolate_trajectory(
                min(path_distance + 0.01, self._arc_length)
            )
            goal_minus, _, _ = self._interpolate_trajectory(
                max(path_distance - 0.01, 0.0)
            )
            tangent = goal_plus - goal_minus
            norm = float(np.linalg.norm(tangent))
            tangent = (
                tangent / norm
                if norm > 1e-9
                else np.array([1.0, 0.0, 0.0], dtype=np.float32)
            )
            ref_vel[env_id] = torch.as_tensor(
                signed_speed * tangent, device=self.device, dtype=torch.float32
            )
            ref_acc[env_id] = torch.as_tensor(
                signed_accel * tangent, device=self.device, dtype=torch.float32
            )

            direction = float(self._traj_direction[env_id].item())
            preview_step = max(
                abs(signed_timing_speed), self._effective_path_speed_mps()
            ) * preview_time
            preview_s = float(
                np.clip(path_distance + direction * preview_step, 0.0, self._arc_length)
            )
            preview_goal, _, _ = self._interpolate_trajectory(preview_s)
            current_goal, _, _ = self._interpolate_trajectory(path_distance)
            preview_xy[env_id] = torch.as_tensor(
                preview_goal[:2] - current_goal[:2],
                device=self.device,
                dtype=torch.float32,
            )
        return ref_vel, ref_acc, preview_xy

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------

    def _get_rewards(self) -> torch.Tensor:
        if self._sole_applied is None:
            return super()._get_rewards()

        cfg = self.cfg
        tool_pos_truck = self._compute_ee_pos_local()
        head_alignment = self._measure_head_down_alignment()
        clearance_signal = self._clearance_signal()
        measured, _ = self._refresh_cylinder_state()
        headroom_low, headroom_high = self._cylinder_headroom(measured)
        acceleration_norm = torch.linalg.norm(
            self._measured_tool_acceleration_truck, dim=-1
        )

        terms = compute_sole_reward(
            original_reference_tool_position=self._original_reference_tool,
            measured_tool_position=tool_pos_truck,
            previous_error_norm=self._previous_error_norm,
            applied_action=self._sole_applied,
            previous_applied_action=self._prev_sole_applied,
            head_down_alignment=head_alignment,
            cutter_clearance=clearance_signal,
            valve_command=self._current_valve_cmd,
            cylinder_headroom_low_m=headroom_low,
            cylinder_headroom_high_m=headroom_high,
            tool_acceleration_norm=acceleration_norm,
            tracking_std_m=float(getattr(cfg, "residual_tracking_std_m", 0.02)),
            tracking_weight=float(getattr(cfg, "residual_tracking_weight", 4.0)),
            progress_weight=float(getattr(cfg, "residual_progress_weight", 0.5)),
            action_magnitude_weight=float(
                getattr(cfg, "residual_action_magnitude_weight", -0.02)
            ),
            action_rate_weight=float(
                getattr(cfg, "sole_rate_penalty_weight", -0.10)
            ),
            head_penalty_weight=float(
                getattr(cfg, "residual_head_penalty_weight", -2.0)
            ),
            ground_penalty_weight=float(
                getattr(cfg, "residual_ground_penalty_weight", -2.0)
            ),
            saturation_penalty_weight=float(
                getattr(cfg, "residual_saturation_penalty_weight", -1.0)
            ),
            head_threshold=float(getattr(cfg, "residual_head_threshold", 0.995)),
            ground_band_m=float(getattr(cfg, "residual_ground_band_m", 0.10)),
            limit_penalty_weight=float(
                getattr(cfg, "sole_limit_penalty_weight", -1.0)
            ),
            limit_margin_m=float(getattr(cfg, "sole_limit_margin_m", 0.02)),
            rate_penalty_weight=float(getattr(cfg, "sole_rate_penalty_weight", -0.10)),
            acceleration_weight=float(
                getattr(cfg, "sole_acceleration_weight", -0.20)
            ),
            acceleration_scale_mps2=float(
                getattr(cfg, "sole_acceleration_scale_mps2", 30.0)
            ),
            tracking_guard_weight=float(
                getattr(cfg, "residual_tracking_guard_weight", 0.0)
            ),
            tracking_guard_scale_m=float(
                getattr(cfg, "residual_tracking_guard_scale_m", 0.04)
            ),
        )

        active = (self._preposition_counter <= 0) & (self._settle_counter <= 0)
        reward = torch.where(active, terms.reward, torch.zeros_like(terms.reward))
        self._previous_error_norm = torch.where(
            active, terms.error_norm, self._previous_error_norm
        )

        if torch.any(active):
            active_error = terms.error_norm[active]
            active_acceleration = acceleration_norm[active]
            active_head = head_alignment[active]
            active_action = self._sole_applied[active]
            active_shield = self._sole_shield_gain[active]
            active_proximity = terms.limit_proximity[active]
        else:
            one = torch.ones(1, device=self.device)
            active_error = torch.zeros(1, device=self.device)
            active_acceleration = torch.zeros(1, device=self.device)
            active_head = one
            active_action = torch.zeros((1, 4), device=self.device)
            active_shield = torch.ones((1, 4), device=self.device)
            active_proximity = torch.zeros((1, 4), device=self.device)

        self._last_reward_terms = {
            "tracking/mean_error_m": torch.mean(active_error),
            "tracking/p95_error_m": torch.quantile(active_error, 0.95),
            "smoothness/p95_tool_acceleration_mps2": torch.quantile(
                active_acceleration, 0.95
            ),
            "safety/min_head_alignment": torch.min(active_head),
            "safety/max_truck_drift_m": torch.max(
                self._truck_post_constraint_drift_xy
            ),
            "sole/mean_abs_action": torch.mean(torch.abs(active_action)),
            "sole/action_saturation_fraction": torch.mean(
                (torch.abs(active_action) > 0.99).float()
            ),
            "sole/mean_shield_gain": torch.mean(active_shield),
            "sole/shield_active_fraction": torch.mean((active_shield < 1.0).float()),
            "sole/max_limit_proximity": torch.max(active_proximity),
            "sole/limit_penalty": torch.mean(terms.limit_penalty),
            "sole/rate_penalty": torch.mean(terms.rate_penalty),
            "sole/acceleration_penalty": torch.mean(terms.acceleration_penalty),
        }
        self.extras.setdefault("log", {}).update(self._last_reward_terms)
        return reward

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        if self._sole_applied is None:
            return
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self._sole_raw[env_ids] = 0.0
        self._sole_filtered[env_ids] = 0.0
        self._sole_applied[env_ids] = 0.0
        self._prev_sole_applied[env_ids] = 0.0
        self._sole_shield_gain[env_ids] = 1.0
        self._head_action_raw[env_ids] = 0.0
        self._head_action_filtered[env_ids] = 0.0
        self._head_action_applied[env_ids] = 0.0
        self._prev_head_action_applied[env_ids] = 0.0
        if bool(getattr(self.cfg, "sole_head_joint_action", False)):
            neutral = float(getattr(self.cfg, "sole_head_neutral_position_rad", 0.0))
            limits = self.robot.data.soft_joint_pos_limits[env_ids, int(self._tool_joint_id)]
            self._head_joint_target[env_ids] = torch.clamp(
                torch.full_like(self._head_joint_target[env_ids], neutral),
                limits[:, 0], limits[:, 1],
            )
