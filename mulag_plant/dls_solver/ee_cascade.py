"""Full 3-D (truck-frame X/Y/Z) EE DLS over the verified cylinder DLS controller."""

from __future__ import annotations

import torch

from .cylinder import CylinderDLSController
from .kinematics import ee_xyz_jacobian_stepped




class EECascadeController:
    def __init__(
        self,
        env,
        *,
        cylinder_controller: CylinderDLSController | None = None,
        jacobian: torch.Tensor | None = None,
        ee_feedback_gain: float = 0.12,
        ee_damping_ratio: float = 0.03,
        velocity_feedforward: float = 1.0,
        jacobian_refresh_every: int = 10,  # cheap re-anchor cadence (no jacobian re-measure)
        ik_gain: float = 0.7,
        ik_iters: int = 3,
        max_ik_step_m: float = 0.01,
        max_cylinder_lead_m: float = 0.01,
        null_space_centering_gain: float = 0.0,
        null_space_weights: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 0.0),
        null_space_max_step_m: float = 0.004,
    ):
        self.env = env
        self.cylinder = cylinder_controller or CylinderDLSController(env)
        self.ee_damping_ratio = float(ee_damping_ratio)
        self.velocity_feedforward = float(velocity_feedforward)
        self.jacobian_refresh_every = int(jacobian_refresh_every)
        self.ik_gain = float(ik_gain)
        self.ik_iters = int(ik_iters)
        self.max_ik_step_m = float(max_ik_step_m)
        self.max_cylinder_lead_m = float(max_cylinder_lead_m)
        # Null-space stroke centering, used ONLY by the redundant 3x4 solve in
        # `cylinder_goal` -- i.e. the deployment/reset pose. The run-time
        # privileged teacher solves a SQUARE 3x3 system (cylinders 1-3, c4 held
        # for head pitch; see verge_task/terrain_env.py) and therefore has no
        # null space at all: nothing can be projected there.
        #
        # Measured 2026-08-15, AAM init, mesh 1337, combined_test: cylinder 3
        # runs from a mean stroke fraction of 0.37 out to its MAX stop in the
        # p01 tail, while cylinders 1 and 2 hold 0.08-0.10 m of unused travel.
        # Because the null space here is one-dimensional, the only free choice
        # is a scalar along it -- c4 cannot be protected independently, it moves
        # by whatever that direction assigns it. That is why the step is bounded
        # and why head alignment has to be re-measured after any change.
        #
        # Gain 0 reproduces the previous minimum-norm behaviour exactly.
        self.null_space_centering_gain = float(null_space_centering_gain)
        self.null_space_max_step_m = float(null_space_max_step_m)
        self.null_space_weights = torch.as_tensor(
            [float(v) for v in null_space_weights],
            dtype=torch.float32,
            device=env.device,
        ).reshape(1, 4)
        self._ee_ref = torch.zeros((env.num_envs, 3), device=env.device)
        self._cyl_ref = torch.zeros((env.num_envs, 4), device=env.device)
        self._steps_since_jac = 10**9  # force a refresh on first use
        self._have_prev_goal = False
        self._jacobian_fixed = jacobian is not None
        if jacobian is None:
            jacobian = ee_xyz_jacobian_stepped(env)[0]
        self.jacobian = jacobian.to(device=env.device, dtype=torch.float32).reshape(3, 4)
        self._set_pinv(self.jacobian)
        self.ee_feedback_gain = float(ee_feedback_gain)
        self.anchor_ee = torch.zeros((env.num_envs, 3), device=env.device)
        self.anchor_x = torch.zeros((env.num_envs, 4), device=env.device)
        scaling = env._cylinder_adapter.scaling
        self.learned_min = scaling.learned_min.to(env.device, torch.float32).reshape(1, 4)
        self.learned_max = scaling.learned_max.to(env.device, torch.float32).reshape(1, 4)

    @torch.no_grad()
    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.env.num_envs, device=self.env.device)
        env_ids = env_ids.to(device=self.env.device, dtype=torch.long)
        ee = self.env._compute_ee_pos_local()
        x, _ = self.env._refresh_cylinder_state()
        self.anchor_ee[env_ids] = ee[env_ids]
        self.anchor_x[env_ids] = x[env_ids]
        self._steps_since_jac = 10**9  # re-linearize at the new pose
        self._have_prev_goal = False
        self._cyl_ref[env_ids] = x[env_ids]  # seed the reference at the measured state
        self.cylinder.reset(env_ids)
        if not self._jacobian_fixed:
            self.refresh_jacobian()  # once, at the settled pose (stepped rollouts
            # after a reset with active wheel velocity drives have aborted Kit
            # silently -- callers with a moving truck must pass a fixed jacobian)
        self._resync_reference()
        self._steps_since_jac = 0

    def _set_pinv(self, jacobian: torch.Tensor) -> None:
        self.jacobian = jacobian.reshape(3, 4)
        # SVD on CPU: cuSOLVER has aborted the whole Kit process (silently, exit 0)
        # on small ill-conditioned matrices -- documented in cylinder.py too.
        j_cpu = self.jacobian.detach().cpu()
        u, s, vh = torch.linalg.svd(j_cpu, full_matrices=False)
        damping = float(self.ee_damping_ratio) * s[0]
        pinv = vh.transpose(0, 1) @ torch.diag(s / (s.square() + damping.square())) @ u.transpose(0, 1)
        self.pinv = pinv.to(self.jacobian.device, self.jacobian.dtype)
        # Exact null-space projector for the 3x4 task. NOT `I - J^+ J`: the
        # damped inverse makes that full rank, and a step through it leaked
        # ~10 mm of EE motion in a 5-sample check against ~0 for the exact form.
        # The undamped SVD gives the null direction outright as the fourth right
        # singular vector, so N = n n^T is rank 1 and exactly idempotent, and
        # the secondary objective cannot disturb the primary task at all.
        null_direction = torch.linalg.svd(j_cpu, full_matrices=True)[2][3]
        self.null_direction = null_direction.to(
            self.jacobian.device, self.jacobian.dtype
        )
        self.null_projector = torch.outer(
            self.null_direction, self.null_direction
        )

    def _pinv_for_columns(self, columns: list[int]) -> torch.Tensor:
        """Return a damped pseudo-inverse for a selected cylinder subset.

        The 2-D deployment path keeps cylinder 4 at the measured head-down
        extension.  Solving the full 3x4 inverse and overwriting that channel
        afterwards leaves the XYZ solve inconsistent, especially in Z.  The
        fixed-channel path therefore solves the square 3x3 problem for the
        remaining cylinders instead.
        """
        jacobian = self.jacobian[:, columns]
        j_cpu = jacobian.detach().cpu()
        u, s, vh = torch.linalg.svd(j_cpu, full_matrices=False)
        damping = float(self.ee_damping_ratio) * s[0]
        pinv = vh.transpose(0, 1) @ torch.diag(
            s / (s.square() + damping.square())
        ) @ u.transpose(0, 1)
        return pinv.to(device=jacobian.device, dtype=jacobian.dtype)

    @torch.no_grad()
    def _null_space_step(self, reference: torch.Tensor) -> torch.Tensor:
        """Null-space motion pulling each cylinder toward the middle of its stroke.

        Gradient projection in the sense of Liegeois: the primary task fixes the
        EE, and the leftover degree of freedom is spent on a secondary objective
        instead of being left at whatever the minimum-norm solution happens to
        pick. The objective here is stroke centering, weighted per cylinder, so
        the deployment pose starts with travel in reserve where it is needed.

        Returns zeros when the gain is 0, so the caller is bit-for-bit unchanged.
        """
        if self.null_space_centering_gain <= 0.0:
            return torch.zeros_like(reference)
        span = torch.clamp(self.learned_max - self.learned_min, min=1.0e-6)
        middle = 0.5 * (self.learned_min + self.learned_max)
        # Dimensionless in [-0.5, 0.5]; positive means "this cylinder is below
        # its mid-stroke and would have to extend to reach the middle".
        centering = (middle - reference) / span
        gradient = self.null_space_weights * centering * span
        step = self.null_space_centering_gain * (
            gradient @ self.null_projector.transpose(0, 1)
        )
        return torch.clamp(
            step, -self.null_space_max_step_m, self.null_space_max_step_m
        )

    @torch.no_grad()
    def refresh_jacobian(self) -> None:
        """Measure the TRUE stepped-physics Jacobian (constraint loop solved).
        EXPENSIVE; only call at a settled pose (reset). Mid-run snapshot/restore
        rollouts have aborted Kit silently."""
        self._set_pinv(ee_xyz_jacobian_stepped(self.env)[0])

    @torch.no_grad()
    def _resync_reference(self) -> None:
        """Cheap re-anchor of the reference to the measured EE. No sim writes and
        NO jacobian re-measure -- safe every few steps."""
        measured_x, _ = self.env._refresh_cylinder_state()
        measured_ee = self.env._compute_ee_pos_local()
        lead = self._cyl_ref - measured_x
        self._ee_ref = measured_ee + lead @ self.jacobian.transpose(0, 1)

    @torch.no_grad()
    def cylinder_goal(self, ee_goal_local: torch.Tensor) -> torch.Tensor:
        """Solve IK in REFERENCE space (Newton on the forward kinematics), then let
        the verified cylinder DLS controller track that solution.

        Do NOT close this loop on the MEASURED EE: it lags and carries ~1 mm/step of
        plant jitter, which fed straight into the reference and made the commanded
        cylinder setpoints wrong -- EE error came out 6-8x larger than the cylinder
        controller's own <0.01 m tracking error. FK is exact, so the reference
        converges to the true cylinder configuration for the goal, and the plant is
        left doing the one thing it is already verified to do well: track a
        cylinder setpoint.
        """
        if self._steps_since_jac >= self.jacobian_refresh_every:
            self._resync_reference()
            self._steps_since_jac = 0
        self._steps_since_jac += 1

        goal = ee_goal_local.reshape(-1, 3)
        measured_feedback = self.ee_feedback_gain * (goal - self.env._compute_ee_pos_local())
        for iteration in range(self.ik_iters):
            error = goal - self._ee_ref
            if iteration == 0:
                error = error + measured_feedback
            delta_x = self.ik_gain * (error @ self.pinv.transpose(0, 1))
            delta_x = torch.clamp(delta_x, -self.max_ik_step_m, self.max_ik_step_m)
            delta_x = delta_x + self._null_space_step(self._cyl_ref)
            new_ref = torch.clamp(self._cyl_ref + delta_x, self.learned_min, self.learned_max)
            applied = new_ref - self._cyl_ref
            self._cyl_ref = new_ref
            # first-order FK prediction; periodically corrected by _resync_reference
            self._ee_ref = self._ee_ref + applied @ self.jacobian.transpose(0, 1)
        # Fast Newton iterations may solve a distant EE goal immediately, but
        # the inner cylinder plant cannot teleport to that solution. Bound the
        # cascade backlog while retaining the exact pose-adaptive IK direction.
        measured_x, _ = self.env._refresh_cylinder_state()
        limited_ref = torch.clamp(
            self._cyl_ref,
            measured_x - self.max_cylinder_lead_m,
            measured_x + self.max_cylinder_lead_m,
        )
        lead_correction = limited_ref - self._cyl_ref
        self._cyl_ref = limited_ref
        self._ee_ref = self._ee_ref + lead_correction @ self.jacobian.transpose(0, 1)
        return self._cyl_ref

    @torch.no_grad()
    def cylinder_goal_fixed_channel(
        self,
        ee_goal_local: torch.Tensor,
        *,
        fixed_index: int,
        fixed_value: float | torch.Tensor,
    ) -> torch.Tensor:
        """Solve XYZ while holding one cylinder at a prescribed extension.

        This is used for the mowing/head-down trajectory library.  Cylinder 4
        controls the head-pitch branch and must remain at the calibrated
        downwards extension; overwriting it after a full 3x4 solve would
        reintroduce a vertical tracking error that the inverse solve never saw.
        """
        if fixed_index < 0 or fixed_index >= 4:
            raise ValueError(f"fixed_index must be in [0, 3], got {fixed_index}")
        free = [column for column in range(4) if column != fixed_index]
        if self._steps_since_jac >= self.jacobian_refresh_every:
            self._resync_reference()
            self._steps_since_jac = 0
        self._steps_since_jac += 1

        goal = ee_goal_local.reshape(-1, 3)
        if isinstance(fixed_value, torch.Tensor):
            fixed = fixed_value.to(
                device=self.env.device, dtype=torch.float32
            ).reshape(self.env.num_envs)
        else:
            fixed = torch.full(
                (self.env.num_envs,), float(fixed_value), device=self.env.device,
                dtype=torch.float32,
            )
        fixed = torch.clamp(
            fixed, self.learned_min[:, fixed_index], self.learned_max[:, fixed_index]
        )
        self._cyl_ref[:, fixed_index] = fixed
        measured_x, _ = self.env._refresh_cylinder_state()
        measured_ee = self.env._compute_ee_pos_local()
        lead = self._cyl_ref - measured_x
        self._ee_ref = measured_ee + lead @ self.jacobian.transpose(0, 1)
        pinv = self._pinv_for_columns(free)
        feedback = self.ee_feedback_gain * (goal - measured_ee)
        for iteration in range(self.ik_iters):
            error = goal - self._ee_ref
            if iteration == 0:
                error = error + feedback
            delta_x = self.ik_gain * (error @ pinv.transpose(0, 1))
            delta_x = torch.clamp(delta_x, -self.max_ik_step_m, self.max_ik_step_m)
            old_free = self._cyl_ref[:, free].clone()
            new_free = torch.clamp(
                old_free + delta_x,
                self.learned_min[:, free],
                self.learned_max[:, free],
            )
            applied = new_free - old_free
            self._cyl_ref[:, free] = new_free
            self._ee_ref = self._ee_ref + applied @ self.jacobian[:, free].transpose(0, 1)

        # Bound only the free-cylinder backlog.  The fixed pitch channel is a
        # deliberate absolute target and is tracked by the inner DLS loop.
        measured_x, _ = self.env._refresh_cylinder_state()
        limited_free = torch.clamp(
            self._cyl_ref[:, free],
            measured_x[:, free] - self.max_cylinder_lead_m,
            measured_x[:, free] + self.max_cylinder_lead_m,
        )
        correction = limited_free - self._cyl_ref[:, free]
        self._cyl_ref[:, free] = limited_free
        self._ee_ref = self._ee_ref + correction @ self.jacobian[:, free].transpose(0, 1)
        self._cyl_ref[:, fixed_index] = fixed
        return self._cyl_ref

    @torch.no_grad()
    def cylinder_goal_seeded(
        self,
        ee_goal_local: torch.Tensor,
        cylinder_seed: torch.Tensor,
        *,
        correction_gain: float = 0.12,
    ) -> torch.Tensor:
        """Track a physics-mapped IK branch with a small measured-EE correction.

        The map seed is the primary inverse-kinematics solution.  A bounded DLS
        correction removes interpolation/model error without asking the fixed
        one-shot Jacobian to solve the metres-long workspace from scratch.
        """
        goal = ee_goal_local.reshape(-1, 3)
        seed = torch.clamp(cylinder_seed.reshape(-1, 4), self.learned_min, self.learned_max)
        measured_ee = self.env._compute_ee_pos_local()
        correction = float(correction_gain) * ((goal - measured_ee) @ self.pinv.transpose(0, 1))
        correction = torch.clamp(correction, -self.max_ik_step_m, self.max_ik_step_m)
        requested = torch.clamp(seed + correction, self.learned_min, self.learned_max)
        measured_x, _ = self.env._refresh_cylinder_state()
        self._cyl_ref = torch.clamp(
            requested,
            measured_x - self.max_cylinder_lead_m,
            measured_x + self.max_cylinder_lead_m,
        )
        self._ee_ref = goal
        return self._cyl_ref

    @torch.no_grad()
    def action(self, ee_goal_local: torch.Tensor):
        cylinder_goal = self.cylinder_goal(ee_goal_local)
        action, singular_values = self.cylinder.action(cylinder_goal)
        return action, cylinder_goal, singular_values
