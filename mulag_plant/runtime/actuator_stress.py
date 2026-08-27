"""Actuator-boundary stress for the deployment plant, in the displacement domain.

Applied to the realised cylinder increment immediately before it updates the
PhysX target -- the same location as ``actuator_authority_alpha``:

    R      : delta_x from the command ROM -> [ chain ] -> target
    IF/AIF : delta_x from ID->FD          -> [ chain ] -> target

That location is the ONLY actuator boundary all three stacks share. The
hydraulic factor draw reaches the command ROM alone (``CylinderFD.step`` takes
no factor argument), and ``PlantPerturbation`` reaches the ID->FD valve alone
(the ROM emits a valve rather than consuming one), so neither can carry a
common robustness axis. This can.

Chain order, each stage an exact identity at its nominal setting:

    realised delta_x
      -> deadband        (insensitivity,  d = 0     -> identity)
      -> authority       (alpha = 1                 -> identity)
      -> first-order lag (tau = 0                   -> identity)
      -> FIFO delay      (delay = 0                 -> identity)
      -> rate limit      (r = inf                   -> identity)
      -> applied delta_x

UNITS. This is the displacement domain, not the valve domain. ``deadband_m``
and ``rate_limit_m`` are metres and metres-per-step; the valve-domain twin in
``experiments/RMA_PPO/robustness/stress.py`` takes them in valve units on
[-1, 1]. The two are not numerically interchangeable, so severities calibrated
against one boundary must be recalibrated for the other. ``lag_tau_s`` and
``delay_steps`` are time-domain and do transfer unchanged.

Nominal invariance is a hard requirement: with every field at its default the
chain returns its input bit-for-bit, so enabling the feature cannot perturb an
unstressed run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


NOMINAL_DEADBAND_M = 0.0
NOMINAL_RATE_LIMIT_M = float("inf")


@dataclass(frozen=True)
class ActuatorStress:
    """One point in displacement-domain stress space. Defaults are nominal."""

    authority: float = 1.0                          # alpha, multiplicative
    lag_tau_s: float = 0.0                          # first-order response time [s]
    delay_steps: int = 0                            # pure transport delay [control ticks]
    deadband_m: float = NOMINAL_DEADBAND_M          # increment insensitivity [m]
    rate_limit_m: float = NOMINAL_RATE_LIMIT_M      # max |d(delta_x)| per tick [m]

    @property
    def is_nominal(self) -> bool:
        return (
            self.authority == 1.0
            and self.lag_tau_s == 0.0
            and self.delay_steps == 0
            and self.deadband_m == NOMINAL_DEADBAND_M
            and math.isinf(self.rate_limit_m)
        )

    def validate(self, *, max_delay_steps: int) -> "ActuatorStress":
        if not 0.0 <= float(self.authority) <= 10.0:
            raise ValueError("authority must lie in [0, 10]")
        if float(self.lag_tau_s) < 0.0:
            raise ValueError("lag tau must be non-negative")
        if int(self.delay_steps) != self.delay_steps or not 0 <= int(self.delay_steps) <= max_delay_steps:
            raise ValueError(f"delay must be an integer in [0, {max_delay_steps}]")
        if float(self.deadband_m) < 0.0:
            raise ValueError("deadband must be non-negative")
        if float(self.rate_limit_m) <= 0.0:
            raise ValueError("rate limit must be positive")
        return self

    def label(self) -> str:
        if self.is_nominal:
            return "nominal"
        bits = []
        if self.authority != 1.0:
            bits.append(f"alpha={self.authority:g}")
        if self.lag_tau_s != 0.0:
            bits.append(f"tau={self.lag_tau_s:g}s")
        if self.delay_steps != 0:
            bits.append(f"delay={self.delay_steps:d}")
        if self.deadband_m != NOMINAL_DEADBAND_M:
            bits.append(f"deadband={self.deadband_m:g}m")
        if not math.isinf(self.rate_limit_m):
            bits.append(f"rate={self.rate_limit_m:g}m")
        return " ".join(bits)

    def as_dict(self) -> dict:
        return {
            "authority": float(self.authority),
            "lag_tau_s": float(self.lag_tau_s),
            "delay_steps": int(self.delay_steps),
            "deadband_m": float(self.deadband_m),
            "rate_limit_m": None if math.isinf(self.rate_limit_m) else float(self.rate_limit_m),
        }


class ActuatorStressChain:
    """Stateful displacement-domain stress chain; exact identity at nominal."""

    def __init__(self, stress: ActuatorStress, *, dt_s: float, num_envs: int,
                 num_cylinders: int = 4, max_delay_steps: int = 40,
                 device: str | torch.device = "cpu") -> None:
        if dt_s <= 0.0 or num_cylinders <= 0 or max_delay_steps < 0:
            raise ValueError("invalid actuator-stress configuration")
        self.stress = stress.validate(max_delay_steps=max_delay_steps)
        self.dt_s = float(dt_s)
        self.num_envs = int(num_envs)
        self.num_cylinders = int(num_cylinders)
        self.max_delay_steps = int(max_delay_steps)
        self.device = torch.device(device)
        self._lag: torch.Tensor | None = None
        self._queue: torch.Tensor | None = None
        self._last_out: torch.Tensor | None = None
        # Delivered-stress accounting for the Stage 0 positive control.
        self.delivered_deadband_zeroed = 0
        self.delivered_rate_clipped = 0
        self.delivered_steps = 0
        self.reset()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        shape = (self.num_envs, self.num_cylinders)
        if self._lag is None or env_ids is None:
            self._lag = torch.zeros(shape, dtype=torch.float32, device=self.device)
            self._queue = torch.zeros((self.max_delay_steps + 1, *shape),
                                      dtype=torch.float32, device=self.device)
            self._last_out = torch.zeros(shape, dtype=torch.float32, device=self.device)
            self.delivered_deadband_zeroed = 0
            self.delivered_rate_clipped = 0
            self.delivered_steps = 0
            return
        ids = env_ids.to(self.device, dtype=torch.long).reshape(-1)
        self._lag[ids] = 0.0
        self._queue[:, ids] = 0.0
        self._last_out[ids] = 0.0

    @torch.no_grad()
    def step(self, delta_extension: torch.Tensor) -> torch.Tensor:
        """Map the realised increment to the applied increment."""
        raw = torch.as_tensor(delta_extension, dtype=torch.float32, device=self.device)
        if raw.ndim != 2 or raw.shape[1] != self.num_cylinders:
            raise ValueError(f"delta_extension must have shape (N, {self.num_cylinders})")
        if not torch.isfinite(raw).all():
            raise FloatingPointError("actuator stress received NaN/Inf")
        assert self._lag is not None and self._queue is not None and self._last_out is not None
        stress = self.stress

        # -- deadband: symmetric, continuous, identity at d = 0 --------------
        d = float(stress.deadband_m)
        if d > 0.0:
            self.delivered_deadband_zeroed += int((raw.abs() <= d).sum())
            value = torch.sign(raw) * torch.clamp(raw.abs() - d, min=0.0)
        else:
            value = raw

        # -- authority -------------------------------------------------------
        if stress.authority != 1.0:
            value = value * float(stress.authority)

        # -- first-order lag --------------------------------------------------
        # tau = 0 assigns straight through. Writing that as the incremental
        # update with beta = 1 is algebraically identical but NOT exact in
        # float32 once the lag state is nonzero, which would break nominal
        # invariance at the 1e-7 level.
        tau = float(stress.lag_tau_s)
        if tau <= 0.0:
            self._lag = value.clone()
        else:
            beta = 1.0 - math.exp(-self.dt_s / tau)
            self._lag = self._lag + beta * (value - self._lag)
        value = self._lag.clone()

        # -- pure FIFO transport delay ---------------------------------------
        self._queue = torch.cat((self._queue[1:], value.unsqueeze(0)), dim=0)
        value = self._queue[self.max_delay_steps - int(stress.delay_steps)]

        # -- realised rate limit ----------------------------------------------
        r = float(stress.rate_limit_m)
        if math.isfinite(r):
            change = value - self._last_out
            self.delivered_rate_clipped += int((change.abs() > r + 1.0e-12).sum())
            value = self._last_out + torch.clamp(change, -r, r)

        self._last_out = value.detach().clone()
        self.delivered_steps += int(raw.shape[0]) * self.num_cylinders
        return value

    def diagnostics(self) -> dict:
        steps = max(self.delivered_steps, 1)
        return {
            "stress": self.stress.as_dict(),
            "label": self.stress.label(),
            "delivered_steps": self.delivered_steps,
            "deadband_zeroed_fraction": self.delivered_deadband_zeroed / steps,
            "rate_clipped_fraction": self.delivered_rate_clipped / steps,
        }
