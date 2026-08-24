"""Plant-side stress mechanisms for the frozen-PPO robustness envelope.

Every mechanism acts at the actuator boundary, between the frozen ID model's
valve command and the frozen SD transition.  The controller is never touched.

Chain order (nominal setting of each stage is an exact identity):

    commanded valve
      -> deadband        (command insensitivity, d = 0 -> identity)
      -> authority       (alpha = 1 -> identity)
      -> first-order lag (tau = 0 -> identity)
      -> FIFO delay      (delay = 0 -> identity)
      -> rate limit      (r = inf -> identity)
      -> applied valve

Deadband and rate limit are new; authority/lag/delay reproduce the existing
``DeploymentActuation`` semantics exactly so earlier results stay comparable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

NOMINAL_DEADBAND = 0.0
NOMINAL_RATE_LIMIT = float("inf")


@dataclass(frozen=True)
class Stress:
    """One point in plant-stress space.  Defaults are the nominal plant."""

    authority: float = 1.0            # alpha, multiplicative actuator authority
    lag_tau_s: float = 0.0            # first-order hydraulic response time [s]
    delay_steps: int = 0              # pure transport delay [control ticks]
    deadband: float = NOMINAL_DEADBAND        # valve-command insensitivity
    rate_limit: float = NOMINAL_RATE_LIMIT    # max |du| per control tick

    @property
    def is_nominal(self) -> bool:
        return (
            self.authority == 1.0
            and self.lag_tau_s == 0.0
            and self.delay_steps == 0
            and self.deadband == NOMINAL_DEADBAND
            and math.isinf(self.rate_limit)
        )

    def validate(self, *, max_delay_steps: int) -> "Stress":
        if not 0.0 <= float(self.authority) <= 1.0:
            raise ValueError("authority must lie in [0, 1]")
        if float(self.lag_tau_s) < 0.0:
            raise ValueError("lag tau must be non-negative")
        if int(self.delay_steps) != self.delay_steps or not 0 <= int(self.delay_steps) <= max_delay_steps:
            raise ValueError(f"delay must be an integer in [0, {max_delay_steps}]")
        if float(self.deadband) < 0.0:
            raise ValueError("deadband must be non-negative")
        if float(self.rate_limit) <= 0.0:
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
        if self.deadband != NOMINAL_DEADBAND:
            bits.append(f"deadband={self.deadband:g}")
        if not math.isinf(self.rate_limit):
            bits.append(f"rate={self.rate_limit:g}")
        return " ".join(bits)

    def as_dict(self) -> dict:
        return {
            "authority": float(self.authority),
            "lag_tau_s": float(self.lag_tau_s),
            "delay_steps": int(self.delay_steps),
            "deadband": float(self.deadband),
            "rate_limit": None if math.isinf(self.rate_limit) else float(self.rate_limit),
        }


class StressedActuation:
    """Stateful actuator-boundary stress chain, exact identity at nominal."""

    def __init__(self, *, dt_s: float, action_dim: int = 4, max_delay_steps: int = 40,
                 device: str | torch.device = "cpu"):
        if dt_s <= 0.0 or action_dim <= 0 or max_delay_steps < 0:
            raise ValueError("invalid actuation configuration")
        self.dt_s = float(dt_s)
        self.action_dim = int(action_dim)
        self.max_delay_steps = int(max_delay_steps)
        self.device = torch.device(device)
        self._lag: torch.Tensor | None = None
        self._queue: torch.Tensor | None = None
        self._last_out: torch.Tensor | None = None
        # Delivered-stress accounting, for the positive-control checks.
        self.delivered_rate_clipped = 0
        self.delivered_deadband_zeroed = 0
        self.delivered_steps = 0

    def reset(self, batch: int) -> None:
        shape = (batch, self.action_dim)
        self._lag = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self._queue = torch.zeros((self.max_delay_steps + 1, *shape), dtype=torch.float32, device=self.device)
        self._last_out = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.delivered_rate_clipped = 0
        self.delivered_deadband_zeroed = 0
        self.delivered_steps = 0

    def step(self, commanded: torch.Tensor, stress: Stress) -> torch.Tensor:
        raw = torch.as_tensor(commanded, dtype=torch.float32, device=self.device)
        if raw.ndim != 2 or raw.shape[1] != self.action_dim:
            raise ValueError(f"commanded valve must have shape (N, {self.action_dim})")
        batch = int(raw.shape[0])
        if self._lag is None or self._lag.shape != raw.shape:
            self.reset(batch)
        assert self._queue is not None and self._last_out is not None

        # -- deadband: symmetric, continuous, identity at d = 0 --------------
        d = float(stress.deadband)
        if d > 0.0:
            magnitude = torch.clamp(raw.abs() - d, min=0.0)
            self.delivered_deadband_zeroed += int((raw.abs() <= d).sum())
            value = torch.sign(raw) * magnitude
        else:
            value = raw

        # -- authority -------------------------------------------------------
        value = value * float(stress.authority)

        # -- first-order lag ---------------------------------------------------
        # tau = 0 assigns the input straight through.  Writing it as the
        # incremental update with beta = 1 is algebraically identical but NOT
        # exact in float32 once the lag state is nonzero, which would break the
        # nominal-invariance requirement at the 1e-7 level.
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

        # -- realized actuator rate limit -------------------------------------
        r = float(stress.rate_limit)
        if math.isfinite(r):
            step_change = value - self._last_out
            self.delivered_rate_clipped += int((step_change.abs() > r + 1.0e-9).sum())
            value = self._last_out + torch.clamp(step_change, -r, r)

        self._last_out = value.detach().clone()
        self.delivered_steps += batch * self.action_dim
        return value
