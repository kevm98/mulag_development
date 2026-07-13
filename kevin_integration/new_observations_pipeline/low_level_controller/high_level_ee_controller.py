"""High-level EE-to-cylinder controller.

Measures the effective Jacobian J_eff = d(EE_base) / d(delta_x_des_H)
through the full IDFD -> PhysX deployment chain, then uses a damped
least-squares controller with temporal smoothing.

PPO never sees cylinder internals.
"""
from __future__ import annotations
import torch


class HighLevelEEController:
    def __init__(self, device="cuda:0", eps=0.001, k_p=0.15, lam=1e-2,
                 per_cyl_clip=0.0015, global_clip=0.003, smooth_alpha=0.7):
        self._device = device
        self._eps = float(eps)
        self._k_p = float(k_p)
        self._lam = float(lam)
        self._per_cyl_clip = float(per_cyl_clip)
        self._global_clip = float(global_clip)
        self._smooth_alpha = float(smooth_alpha)
        self._J_eff = None
        self._condition_number = None
        self._singular_values = None
        self._prev_delta = None

    def measure_effective_jacobian(self, deploy_fn, B):
        """Measure J_eff through full deployment chain.

        deploy_fn(delta_x_des_H[B,4]) -> (ee_after[B,3], restore_fn)
        Uses central difference with perturbation eps on each cylinder.
        """
        J = torch.zeros((B, 3, 4), device=self._device)
        for c in range(4):
            dx_pos = torch.zeros((B, 4), device=self._device)
            dx_pos[:, c] = self._eps
            ee_pos, restore_pos = deploy_fn(dx_pos)
            restore_pos()

            dx_neg = torch.zeros((B, 4), device=self._device)
            dx_neg[:, c] = -self._eps
            ee_neg, restore_neg = deploy_fn(dx_neg)
            restore_neg()

            J[:, :, c] = (ee_pos - ee_neg) / (2.0 * self._eps)

        self._J_eff = J
        S = torch.linalg.svdvals(J.float())
        self._singular_values = S
        self._condition_number = S[:, 0] / S[:, -1].clamp(min=1e-9)
        self._prev_delta = torch.zeros((B, 4), device=self._device)
        return J, self._condition_number, S

    def set_jacobian(self, J: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Replace the stored Jacobian (e.g. after cross-env outlier filtering) and
        recompute singular values / condition numbers. Returns (cond, S)."""
        self._J_eff = J
        S = torch.linalg.svdvals(J.float())
        self._singular_values = S
        self._condition_number = S[:, 0] / S[:, -1].clamp(min=1e-9)
        return self._condition_number, S

    def map_ee_to_cylinder(self, desired_ee_delta, **_ignored):
        """Damped least-squares with temporal smoothing (Whitney resolved-rate +
        Nakamura DLS). Plain, full-3-row solve — the standard, well-behaved form.

        `**_ignored` absorbs legacy axis_indices/secondary kwargs from callers that
        experimented with task-priority/nullspace variants; those regressed the
        low-level response (wrong-signed, oscillating) and were reverted 2026-07-09.
        """
        if self._J_eff is None:
            raise RuntimeError("Jacobian not measured.")
        B = desired_ee_delta.shape[0]
        J = self._J_eff
        ee = desired_ee_delta
        JT = J.transpose(1, 2)
        JJT = torch.bmm(J, JT)
        eye = torch.eye(3, device=self._device).unsqueeze(0).expand(B, -1, -1)
        A = JJT + self._lam * eye
        w = torch.linalg.solve(A.float(), ee.unsqueeze(-1).float()).to(J.dtype).squeeze(-1)
        delta_raw = self._k_p * torch.bmm(JT, w.unsqueeze(-1)).squeeze(-1)

        delta = torch.clamp(delta_raw, -self._per_cyl_clip, self._per_cyl_clip)
        norms = torch.norm(delta, dim=-1, keepdim=True).clamp(min=1e-9)
        scale = torch.clamp(self._global_clip / norms, max=1.0)
        delta = delta * scale

        if self._prev_delta is not None and self._prev_delta.shape[0] == B:
            delta = self._smooth_alpha * self._prev_delta + (1.0 - self._smooth_alpha) * delta
        self._prev_delta = delta.detach().clone()

        clipped = (delta.abs() >= self._per_cyl_clip).float().mean(dim=-1)
        delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
        return delta, clipped

    @property
    def condition_number(self):
        return self._condition_number

    @property
    def singular_values(self):
        return self._singular_values

    @property
    def jacobian(self):
        return self._J_eff

    def effective_rank(self, tol=0.01):
        if self._singular_values is None:
            return None
        max_sv = self._singular_values[:, 0]
        return int((self._singular_values > tol * max_sv.unsqueeze(-1)).sum(dim=-1).max().item())

    def jacobian_report(self, env_id=0):
        if self._J_eff is None:
            return "Jacobian not measured."
        J = self._J_eff[env_id].detach().cpu().numpy()
        S = self._singular_values[env_id].detach().cpu().numpy() if self._singular_values is not None else None
        cond = self._condition_number[env_id].item() if self._condition_number is not None else float("inf")
        rank = self.effective_rank()
        finite = bool(torch.isfinite(self._J_eff).all().item())
        return (
            f"J_eff [3x4]:\n  {J}\n"
            f"singular values: {S}\n"
            f"condition number: {cond:.2f}\n"
            f"effective rank (tol=0.01): {rank}\n"
            f"finite: {finite}"
        )
