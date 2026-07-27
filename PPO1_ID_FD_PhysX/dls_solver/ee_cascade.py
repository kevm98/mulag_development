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
