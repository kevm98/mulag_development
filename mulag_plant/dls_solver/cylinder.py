"""Reusable online 4x4 cylinder DLS controller."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CylinderDLSConfig:
    # These were throttled so hard that the controller could never reach a setpoint
    # more than a few cm away: max_desired_delta_m=5e-4 caps cylinder motion at
    # 0.5 mm/step (30 mm/s), and max_action=0.2 gave away 80% of the actuator.
    # Measured symptom: |cyl_ref - cyl_meas| stuck at 0.07-0.13 m regardless of path
    # speed. The old <0.01 m gate only ever tested small, static, settled setpoints.
    jac_eps: float = 0.005
    error_gain: float = 0.08          # was 0.02
    max_desired_delta_m: float = 1.5e-3  # was 5.0e-4 (0.5 -> 1.5 mm/step)
    damping_smax_ratio: float = 0.10
    max_action: float = 0.5           # was 0.2
    action_smoothing: float = 0.2
    invert_learned_pipeline: bool = True

    # Proportional-only control of a TARGET INTEGRATOR in series with a lagging plant
    # is unconditionally overshooting: the controller keeps pushing the target while the
    # plant is still catching up, so by the time the measurement reaches the goal the
    # target is already far past it. Measured on the repaired plant: +-5 mm steps ended
    # 15-125 mm on the WRONG side of the goal, every cylinder, every direction.
    # `velocity_damping` is the missing derivative term (it subtracts the motion already
    # in flight), and `velocity_feedforward` lets a moving reference command its own speed
    # instead of being dragged along by position error alone.
    velocity_damping: float = 0.0      # [s] multiplies measured xdot
    velocity_feedforward: float = 0.0  # [0..1] fraction of the reference's own velocity
    measured_error_feedback: float = 0.0


class CylinderDLSController:
    """Close the learned FD target integrator on a cylinder extension setpoint."""

    def __init__(self, env, config: CylinderDLSConfig | None = None):
        self.env = env
        self.config = config or CylinderDLSConfig()
        self.previous_action = torch.zeros((env.num_envs, 4), dtype=torch.float32, device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self.previous_action.zero_()
        else:
            self.previous_action[env_ids.to(device=self.env.device, dtype=torch.long)] = 0.0

    @torch.no_grad()
    def target_x(self) -> torch.Tensor:
        zero_velocity = torch.zeros_like(self.env._cylinder_pos_target)
        x, _ = self.env._cylinder_adapter.current_from_cylinder_joints(
            self.env._cylinder_pos_target, zero_velocity
        )
        return x

    @torch.no_grad()
    def predict_fd_delta(self, action: torch.Tensor) -> torch.Tensor:
        env = self.env
        x, xdot = env._refresh_cylinder_state()
        converted = env._action_to_cylinder_target_adapter.convert(
            action=action, x_current=x, xdot_current=xdot
        )
        previous_valve = env._prev_valve_cmd.detach().clone()
        model_input = env._cylinder_adapter.build_id_input(
            converted.reference,
            dP=env._dP,
            force=env._force,
            prev_valve_cmd=previous_valve,
            input_dim=env._cylinder_id.input_dim,
        )
        valve = env._cylinder_id.predict(model_input, prev_valve_cmd=previous_valve)
        baseline_valve = torch.zeros_like(valve)
        if bool(getattr(env.cfg, "id_residual_over_zero_target", False)):
            zero_input = model_input.detach().clone()
            zero_input[:, 16:24] = 0.0
            zero_valve = env._cylinder_id.predict(zero_input, prev_valve_cmd=previous_valve)
            if bool(getattr(env.cfg, "paired_id_fd_residual", False)):
                baseline_valve = zero_valve
            else:
                valve = valve - zero_valve
        valve = torch.clamp(valve, -float(env.cfg.max_abs_valve), float(env.cfg.max_abs_valve))
        baseline_valve = torch.clamp(
            baseline_valve, -float(env.cfg.max_abs_valve), float(env.cfg.max_abs_valve)
        )
        fd = env._cylinder_fd.step(
            valve_cmd=valve, extension=x, extension_velocity=xdot, dP=env._dP, force=env._force
        ).delta_extension
        paired_residual = bool(getattr(env.cfg, "paired_id_fd_residual", False))
        if paired_residual:
            fd_zero = env._cylinder_fd.step(
                valve_cmd=baseline_valve,
                extension=x,
                extension_velocity=xdot,
                dP=env._dP,
                force=env._force,
            ).delta_extension
            fd = fd - fd_zero
        elif bool(getattr(env.cfg, "fd_residual_over_zero_valve", False)):
            fd_zero = env._cylinder_fd.step(
                valve_cmd=torch.zeros_like(valve),
                extension=x,
                extension_velocity=xdot,
                dP=env._dP,
                force=env._force,
            ).delta_extension
            fd = fd - fd_zero
        if paired_residual or bool(getattr(env.cfg, "fd_residual_over_zero_valve", False)):
            residual_gain = env._cfg_cyl_vector(
                "cylinder_fd_residual_output_gain", 1.0
            )
            fd = fd * residual_gain
        scale = float(getattr(env.cfg, "cylinder_fd_delta_gain", 1.0))
        if bool(getattr(env.cfg, "cylinder_fd_scale_delta_by_sample_time", False)):
            fd_dt = float(getattr(env.cfg, "cylinder_fd_sample_time_s", 0.0))
            if fd_dt <= 0.0:
                fd_dt = float(env._cylinder_fd.sample_time_s)
            scale *= float(env.step_dt) / max(fd_dt, 1.0e-6)
        fd = fd * scale
        reference_blend = float(getattr(env.cfg, "cylinder_idfd_reference_blend", 0.0))
        reference_blend = min(max(reference_blend, 0.0), 1.0)
        if reference_blend > 0.0:
            fd = (1.0 - reference_blend) * fd + reference_blend * converted.reference.target_delta_x
        return fd

    @torch.no_grad()
    def predict_base_rom_delta(self, action: torch.Tensor) -> torch.Tensor:
        """Predict PPO-valve -> ROM displacement without advancing ROM history."""
        env = self.env
        if env._base_cylinder_rom is None:
            raise RuntimeError("Base ROM runtime is not initialized.")
        x, xdot = env._refresh_cylinder_state()
        valve = torch.clamp(action, -float(env.cfg.max_abs_valve), float(env.cfg.max_abs_valve))
        return env._base_cylinder_rom.predict_delta(
            valve_cmd=valve,
            extension=x,
            extension_velocity=xdot,
            dP=env._dP,
            force=env._force,
        )

    @torch.no_grad()
    def predict_learned_delta(self, action: torch.Tensor) -> torch.Tensor:
        dynamics = str(getattr(self.env.cfg, "cylinder_dynamics_model", "fd")).lower()
        if dynamics == "command_rom":
            env = self.env
            x, xdot = env._refresh_cylinder_state()
            converted = env._action_to_cylinder_target_adapter.convert(
                action=action, x_current=x, xdot_current=xdot
            )
            command = torch.cat(
                (converted.reference.target_delta_x, converted.reference.target_delta_xdot), dim=-1
            )
            return env._command_cylinder_rom.predict_delta(
                command=command, extension=x, extension_velocity=xdot, dP=env._dP
            )
        if dynamics == "base_rom":
            return self.predict_base_rom_delta(action)
        return self.predict_fd_delta(action)

    @torch.no_grad()
    def local_model(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        columns = []
        batch = self.env.num_envs
        midpoint = torch.zeros((batch, 4), dtype=torch.float32, device=self.env.device)
        for column in range(4):
            plus = torch.zeros((batch, 4), dtype=torch.float32, device=self.env.device)
            minus = torch.zeros_like(plus)
            plus[:, column] = self.config.jac_eps
            minus[:, column] = -self.config.jac_eps
            y_plus = self.predict_learned_delta(plus)
            y_minus = self.predict_learned_delta(minus)
            columns.append((y_plus - y_minus) / (2.0 * self.config.jac_eps))
            midpoint += (y_plus + y_minus) / 8.0
        jacobian = torch.stack(columns, dim=2)
        # CUDA's batched SVD can terminate the Isaac process on the very
        # ill-conditioned action->FD Jacobian seen off the tiny old gate
        # (condition O(1e3)).  Eigenvalues of J J^T give the same singular
        # values for damping/diagnostics without that fragile solver path.
        # Keep the tiny diagnostic decomposition on CPU.  cuSOLVER has been
        # observed to abort the entire Kit process (without a Python exception)
        # for these ill-conditioned 4x4 batches.
        singular_values = torch.stack(
            [torch.linalg.svdvals(matrix.detach().cpu()) for matrix in jacobian], dim=0
        ).to(device=jacobian.device, dtype=jacobian.dtype)
        return jacobian, midpoint, singular_values

    @torch.no_grad()
    def _desired_delta(
        self, error: torch.Tensor, xdot_measured: torch.Tensor,
        goal_velocity: torch.Tensor | None,
    ) -> torch.Tensor:
        """Per-step cylinder motion the plant should make: PD + velocity feedforward.

        Returns a delta in metres-per-control-step, which is the unit the FD model and
        the target integrator both speak.
        """
        dt = float(self.env.step_dt)
        desired = self.config.error_gain * error
        if self.config.velocity_damping != 0.0:
            desired = desired - self.config.velocity_damping * xdot_measured * dt
        if self.config.velocity_feedforward != 0.0 and goal_velocity is not None:
            desired = desired + self.config.velocity_feedforward * goal_velocity * dt
        return torch.clamp(desired, -self.config.max_desired_delta_m, self.config.max_desired_delta_m)

    @torch.no_grad()
    def action(
        self,
        cylinder_goal: torch.Tensor,
        goal_velocity: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        goal = cylinder_goal.reshape(self.env.num_envs, 4)
        # Close the loop on the MEASURED cylinder state, not on the commanded
        # target. Using target_x() (env._cylinder_pos_target) made this an OPEN loop
        # on the plant: the controller drove the *target* to the goal, saw its error
        # go to zero, and stopped acting while the actual arm was still ~0.1 m behind
        # (measured |cyl_ref - cyl_meas| = 0.07-0.13 m, independent of path speed).
        measured, xdot_measured = self.env._refresh_cylinder_state()
        error = goal - measured
        direct_target_action = (
            self.env._low_level_servo_mode() in ("position_ramp_pi", "direct_position_delta")
            or not self.config.invert_learned_pipeline
        )
        if direct_target_action:
            # The environment's position-ramp servo is already a closed-loop
            # cylinder controller and deliberately bypasses the under-authority
            # ID/FD residual. Feed it a bounded cylinder-space increment rather
            # than inverting action->FD a second time.
            # Both backends integrate increments into a persistent PhysX target.
            # Close that inner integrator on target error so it cannot wind past
            # the goal while the fast physical plant catches the held target.
            control_error = goal - self.target_x()
            if self.config.measured_error_feedback != 0.0:
                control_error = control_error + self.config.measured_error_feedback * error
            desired_learned = self._desired_delta(control_error, xdot_measured, goal_velocity)
            scaling = self.env._cylinder_adapter.scaling
            desired_usd = scaling.learned_delta_to_usd_delta(desired_learned)
            action_scale = max(float(self.env.cfg.action_scale), 1.0e-9)
            raw = torch.clamp(
                desired_usd / action_scale,
                -self.config.max_action,
                self.config.max_action,
            )
            alpha = self.config.action_smoothing
            action = alpha * raw + (1.0 - alpha) * self.previous_action
            self.previous_action = action.detach().clone()
            singular_values = torch.ones(
                (self.env.num_envs, 4), device=self.env.device, dtype=torch.float32
            )
            return action, singular_values
        jacobian, midpoint, singular_values = self.local_model()
        reference_blend = float(getattr(self.env.cfg, "cylinder_idfd_reference_blend", 0.0))
        if reference_blend > 0.0:
            # Reference-anchored ID/FD is required to behave like DirectCylinderDriver:
            # smoothly move the commanded PhysX target to the cylinder goal, then stop.
            # Closing this integrator on measured error instead drives the target past the
            # goal to compensate linkage deflection, so it can never match direct-drive
            # semantics and eventually winds against the target-lead clamp.
            control_error = goal - self.target_x()
        else:
            control_error = error
        desired = self._desired_delta(control_error, xdot_measured, goal_velocity) - midpoint
        damping = torch.clamp(
            self.config.damping_smax_ratio * singular_values[:, 0], min=1.0e-7
        )
        # Damped least squares: J^T (J J^T + lambda^2 I)^-1 desired.
        # This is algebraically equivalent to the SVD form. Keep the tiny 4x4
        # solve on CPU too: cuSOLVER can terminate Kit on this matrix family.
        raw_rows = []
        identity_cpu = torch.eye(4, dtype=torch.float32)
        for matrix, rhs, damp in zip(jacobian, desired, damping, strict=True):
            matrix_cpu = matrix.detach().cpu().float()
            rhs_cpu = rhs.detach().cpu().float().unsqueeze(-1)
            lhs_cpu = matrix_cpu @ matrix_cpu.transpose(0, 1)
            lhs_cpu = lhs_cpu + float(damp.item()) ** 2 * identity_cpu
            dual_cpu = torch.linalg.solve(lhs_cpu, rhs_cpu)
            raw_rows.append((matrix_cpu.transpose(0, 1) @ dual_cpu).squeeze(-1))
        raw = torch.stack(raw_rows, dim=0).to(device=jacobian.device, dtype=jacobian.dtype)
        raw = torch.clamp(raw, -self.config.max_action, self.config.max_action)
        alpha = self.config.action_smoothing
        action = alpha * raw + (1.0 - alpha) * self.previous_action
        self.previous_action = action.detach().clone()
        return action, singular_values

    @torch.no_grad()
    def step(self, cylinder_goal: torch.Tensor):
        action, singular_values = self.action(cylinder_goal)
        self.env._cylinder_goal_x[:] = cylinder_goal.reshape(self.env.num_envs, 4)
        transition = self.env.step(action)
        return transition, action, singular_values
