"""Plant-RESPONSE observer: temporal summary of commanded-versus-achieved motion.

Measured 2026-07-24/25 on `model_53`:

* The ROM-B factors move the plant by ~14 mm of cylinder divergence (p95 27 mm)
  over a few seconds, so the randomization is physically real.
* Closed-loop tracking error is nonetheless flat across the whole envelope
  (0.0307-0.0314 m), because the DLS/cascade inner loop rejects the disturbance.
* The policy observes a single 42-dimensional frame with no memory.

"This plant is sluggish" is not expressible in one frame: it is a statement about
commanded-versus-achieved motion *over time*. A feedforward policy therefore
cannot perform the implicit system identification that domain randomization is
supposed to teach, no matter how wide the factor ranges are.

It is deliberately named a plant-*response* observer, not a factor estimator.
Cylinder goal error is driven by trajectory, load, controller state and the
policy's own action as well as by the hydraulics, so these features are a summary
of plant response under the current conditions - not an estimate of e_comp or
seal leakage. Claiming adaptation requires the ablations listed in
reports/rom_b_observer_validation.md, not the presence of this block.

This observer supplies a temporal statistic without a recurrent
policy and without touching the actor architecture. It tracks the cylinder goal
error (observation slice 16:20, `corrected_cylinder_goal - measured_extension`)
through two exponential filters:

* a slow filter (default 2.0 s) whose steady value reflects the *plant* - a leaky
  or soft cylinder sits at a persistent goal offset;
* a fast filter (default 0.25 s), whose difference from the slow one reflects
  *motion* transients rather than plant health.

Publishing `[slow, fast - slow]` gives the policy a persistent-lag estimate and a
transient term, 8 dimensions in total. Both are functions of signals the policy
already sees; no hidden factor is exposed, so the actor and critic stay
factor-blind.
"""

from __future__ import annotations

import torch


PLANT_LAG_OBS_DIM = 8


class PlantLagObserver:
    """Two-timescale exponential estimate of persistent cylinder goal error."""

    def __init__(
        self,
        *,
        num_envs: int,
        device: str | torch.device,
        control_dt_s: float,
        fast_tau_s: float = 0.25,
        slow_tau_s: float = 2.0,
        scale: float = 100.0,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if control_dt_s <= 0.0:
            raise ValueError("control_dt_s must be positive")
        if not 0.0 < fast_tau_s < slow_tau_s:
            raise ValueError("require 0 < fast_tau_s < slow_tau_s")
        self.num_envs = int(num_envs)
        self.device = device
        self.control_dt_s = float(control_dt_s)
        # Discrete smoothing factors, clamped so a long dt cannot overshoot.
        self.fast_alpha = min(self.control_dt_s / float(fast_tau_s), 1.0)
        self.slow_alpha = min(self.control_dt_s / float(slow_tau_s), 1.0)
        # Cylinder goal error is millimetre-scale; normalize so the appended
        # features have a comparable magnitude to the rest of the observation.
        self.scale = float(scale)
        self.fast = torch.zeros(self.num_envs, 4, device=device)
        self.slow = torch.zeros(self.num_envs, 4, device=device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self.fast.zero_()
            self.slow.zero_()
            return
        ids = env_ids.to(device=self.fast.device, dtype=torch.long).reshape(-1)
        if ids.numel():
            self.fast[ids] = 0.0
            self.slow[ids] = 0.0

    @torch.no_grad()
    def update(
        self, cylinder_goal_error: torch.Tensor, active: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Advance the filters and return the `(N, 8)` observation block.

        `active` gates the update so deployment and settling transients do not
        pollute the plant estimate; inactive environments hold their last value.
        """

        error = cylinder_goal_error.detach().to(self.fast.dtype)
        if error.shape != self.fast.shape:
            raise ValueError(
                f"cylinder_goal_error must have shape {tuple(self.fast.shape)}, "
                f"got {tuple(error.shape)}"
            )
        fast = self.fast + self.fast_alpha * (error - self.fast)
        slow = self.slow + self.slow_alpha * (error - self.slow)
        if active is not None:
            gate = active.reshape(-1, 1).to(self.fast.dtype)
            fast = gate * fast + (1.0 - gate) * self.fast
            slow = gate * slow + (1.0 - gate) * self.slow
        self.fast = fast
        self.slow = slow
        return torch.nan_to_num(
            torch.cat((self.slow * self.scale, (self.fast - self.slow) * self.scale), dim=-1),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    def features(self) -> torch.Tensor:
        """Current block without advancing the filters."""

        return torch.cat(
            (self.slow * self.scale, (self.fast - self.slow) * self.scale), dim=-1
        )


def pad_state_dict_for_wider_observation(
    checkpoint_state: dict[str, torch.Tensor],
    target_state: dict[str, torch.Tensor],
    *,
    added_dims: int = PLANT_LAG_OBS_DIM,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    """Adapt a narrower checkpoint onto a wider-observation model.

    Any tensor whose trailing dimension is short by exactly `added_dims` is
    zero-padded, except running-variance and running-std buffers, which are
    padded with ones so the new channels normalize to themselves rather than
    dividing by zero.

    Because the appended features start at zero on reset *and* their input
    weights start at zero, the widened policy is numerically identical to the
    checkpoint immediately after transfer. Training then decides whether the new
    inputs earn their place. Without this, adding the observer would force a cold
    start and throw away `model_53`.
    """

    if added_dims <= 0:
        raise ValueError("added_dims must be positive")
    adapted: dict[str, torch.Tensor] = {}
    report: dict[str, object] = {
        "added_dims": int(added_dims),
        "padded": [],
        "copied": 0,
        "missing_from_checkpoint": [],
        "shape_mismatch": [],
    }
    for key, target in target_state.items():
        source = checkpoint_state.get(key)
        if source is None:
            report["missing_from_checkpoint"].append(key)
            adapted[key] = target.clone()
            continue
        if source.shape == target.shape:
            adapted[key] = source.clone()
            report["copied"] = int(report["copied"]) + 1
            continue
        same_leading = source.shape[:-1] == target.shape[:-1]
        widened_by = target.shape[-1] - source.shape[-1]
        if not (same_leading and widened_by == added_dims):
            report["shape_mismatch"].append(
                f"{key}: checkpoint {tuple(source.shape)} vs model {tuple(target.shape)}"
            )
            adapted[key] = target.clone()
            continue
        lowered = key.lower()
        fill = 1.0 if ("var" in lowered or "std" in lowered) else 0.0
        pad = torch.full(
            (*source.shape[:-1], added_dims),
            fill,
            dtype=source.dtype,
            device=source.device,
        )
        adapted[key] = torch.cat((source, pad), dim=-1)
        report["padded"].append(
            f"{key}: {tuple(source.shape)} -> {tuple(adapted[key].shape)} fill={fill}"
        )
    return adapted, report
