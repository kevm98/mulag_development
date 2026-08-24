"""Context and deployment-actuation contracts for RMA preparation."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Iterable

import torch

CONTEXT_NAMES = (
    "actuator_authority_alpha",
    "hydraulic_response_time_tau",
    "actuator_delay_steps",
    "mass_scale",
)
CONTEXT_DIM = len(CONTEXT_NAMES)


@dataclass(frozen=True)
class DeploymentContext:
    """Physical deployment context; no legacy six-factor coordinates."""

    actuator_authority_alpha: float = 1.0
    hydraulic_response_time_tau: float = 0.0
    actuator_delay_steps: int = 0
    mass_scale: float = 1.0
    # COM is deliberately optional and is not part of the policy context by
    # default.  If enabled later, it must be a native rigid-body field.
    com_z: float | None = None

    def validate(self, *, max_delay_steps: int = 64) -> "DeploymentContext":
        if not 0.0 <= float(self.actuator_authority_alpha) <= 1.0:
            raise ValueError("actuator_authority_alpha must be in [0, 1]")
        if float(self.hydraulic_response_time_tau) < 0.0:
            raise ValueError("hydraulic_response_time_tau must be non-negative")
        if int(self.actuator_delay_steps) != self.actuator_delay_steps:
            raise ValueError("actuator_delay_steps must be an integer")
        if not 0 <= int(self.actuator_delay_steps) <= int(max_delay_steps):
            raise ValueError("actuator_delay_steps exceeds the configured FIFO")
        if float(self.mass_scale) <= 0.0:
            raise ValueError("mass_scale must be positive")
        return self

    def as_tensor(self, *, device: str | torch.device = "cpu") -> torch.Tensor:
        self.validate()
        return torch.tensor(
            [
                self.actuator_authority_alpha,
                self.hydraulic_response_time_tau,
                float(self.actuator_delay_steps),
                self.mass_scale,
            ],
            dtype=torch.float32,
            device=device,
        )


@dataclass(frozen=True)
class ContextNormalizer:
    """Explicit physical-to-policy normalization, kept separate from PPO stats."""

    mean: tuple[float, float, float, float] = (1.0, 0.08, 1.0, 1.0)
    std: tuple[float, float, float, float] = (0.5, 0.08, 1.0, 0.25)

    def tensors(self, *, device: str | torch.device = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
        mean = torch.as_tensor(self.mean, dtype=torch.float32, device=device)
        std = torch.as_tensor(self.std, dtype=torch.float32, device=device)
        if mean.numel() != CONTEXT_DIM or std.numel() != CONTEXT_DIM or bool((std <= 0).any()):
            raise ValueError("context normalizer must have four positive-std entries")
        return mean, std


def normalized_context(
    context: DeploymentContext | torch.Tensor,
    normalizer: ContextNormalizer = ContextNormalizer(),
) -> torch.Tensor:
    """Return the four direct normalized context values used by PPO."""

    raw = context.as_tensor() if isinstance(context, DeploymentContext) else torch.as_tensor(context, dtype=torch.float32)
    if raw.shape[-1] != CONTEXT_DIM:
        raise ValueError(f"context must end in width {CONTEXT_DIM}, got {tuple(raw.shape)}")
    mean, std = normalizer.tensors(device=raw.device)
    return (raw - mean) / std


@dataclass(frozen=True)
class ActuationConfig:
    """Discrete wrapper settings; values are intentionally conservative."""

    dt_s: float = 1.0 / 60.0
    action_dim: int = 4
    max_delay_steps: int = 64


@dataclass(frozen=True)
class NativeRigidBodyWiring:
    """Static declaration of native PhysX mass/COM wiring.

    The wrapper never fakes mass or COM in an observation.  A future runtime
    must apply these fields to the native rigid body and expose the resulting
    values only through the declared context path.
    """

    mass_field: str = "articulation.data.default_mass"
    com_field: str = "articulation.data.default_com_pos_b"
    mass_scale_source: str = "native_physx_rigid_body"
    com_source: str = "native_physx_rigid_body"


def verify_native_mass_com_wiring(wiring: NativeRigidBodyWiring) -> None:
    """Reject surrogate mass/COM channels before a simulator run is allowed."""

    if wiring.mass_scale_source != "native_physx_rigid_body":
        raise ValueError("mass_scale must be wired to the native PhysX rigid body")
    if wiring.com_source != "native_physx_rigid_body":
        raise ValueError("com_z must be wired to the native PhysX rigid body")
    if not wiring.mass_field or not wiring.com_field:
        raise ValueError("native mass and COM field paths are required")


def _as_batch_context(context: DeploymentContext | torch.Tensor, batch: int, *, device: torch.device) -> torch.Tensor:
    raw = context.as_tensor(device=device) if isinstance(context, DeploymentContext) else torch.as_tensor(context, dtype=torch.float32, device=device)
    if raw.shape[-1] != CONTEXT_DIM:
        raise ValueError(f"context must end in width {CONTEXT_DIM}, got {tuple(raw.shape)}")
    raw = raw.reshape(-1, CONTEXT_DIM)
    if raw.shape[0] == 1:
        raw = raw.expand(batch, -1)
    if raw.shape[0] != batch:
        raise ValueError(f"context batch {raw.shape[0]} does not match action batch {batch}")
    return raw


class DeploymentActuation:
    """Authority -> stable first-order lag -> pure FIFO delay.

    This is an opt-in mathematical wrapper around the existing command path.
    It does not replace ID, FD, ROM, or PhysX.  At the identity context
    (alpha=1, tau=0, delay=0, mass=1) the output is exactly the raw command.
    """

    def __init__(self, config: ActuationConfig = ActuationConfig(), *, device: str | torch.device = "cpu"):
        if config.dt_s <= 0 or config.action_dim <= 0 or config.max_delay_steps < 0:
            raise ValueError("invalid actuation configuration")
        self.config = config
        self.device = torch.device(device)
        self._lag_state: torch.Tensor | None = None
        self._delay_queue: torch.Tensor | None = None

    def reset(self, batch: int | None = None) -> None:
        self._lag_state = None
        self._delay_queue = None
        if batch is not None:
            shape = (batch, self.config.action_dim)
            self._lag_state = torch.zeros(shape, dtype=torch.float32, device=self.device)
            self._delay_queue = torch.zeros(
                (self.config.max_delay_steps + 1, batch, self.config.action_dim),
                dtype=torch.float32,
                device=self.device,
            )

    def step(self, raw_action: torch.Tensor, context: DeploymentContext | torch.Tensor) -> torch.Tensor:
        raw = torch.as_tensor(raw_action, dtype=torch.float32, device=self.device)
        if raw.ndim != 2 or raw.shape[1] != self.config.action_dim:
            raise ValueError(f"raw action must have shape (N, {self.config.action_dim})")
        batch = int(raw.shape[0])
        ctx = _as_batch_context(context, batch, device=self.device)
        if self._lag_state is None or self._lag_state.shape != raw.shape:
            self.reset(batch)
        authority = torch.clamp(ctx[:, 0:1], 0.0, 1.0)
        authorized = raw * authority
        tau = torch.clamp(ctx[:, 1:2], min=0.0)
        # exp(-dt/tau) is stable and exact in the discrete-time sense.  tau=0
        # is the explicit bypass used by the identity test.
        beta = torch.where(
            tau > 0.0,
            1.0 - torch.exp(-float(self.config.dt_s) / torch.clamp(tau, min=1.0e-8)),
            torch.ones_like(tau),
        )
        self._lag_state = self._lag_state + beta * (authorized - self._lag_state)
        lagged = self._lag_state.clone()

        assert self._delay_queue is not None
        self._delay_queue = torch.cat((self._delay_queue[1:], lagged.unsqueeze(0)), dim=0)
        delays = torch.round(ctx[:, 2]).to(torch.long).clamp(0, self.config.max_delay_steps)
        row = torch.arange(batch, device=self.device)
        delayed = self._delay_queue[self.config.max_delay_steps - delays, row]
        return delayed


# Descriptive alias used by experiment notes and future integration code.
ActuationStressWrapper = DeploymentActuation


def causal_mismatch_features(obs_t: torch.Tensor, action_t: torch.Tensor, obs_tp1: torch.Tensor) -> torch.Tensor:
    """Build estimator inputs from causal on-policy signals only."""

    obs_t = torch.as_tensor(obs_t, dtype=torch.float32)
    action_t = torch.as_tensor(action_t, dtype=torch.float32, device=obs_t.device)
    obs_tp1 = torch.as_tensor(obs_tp1, dtype=torch.float32, device=obs_t.device)
    if obs_t.ndim != 2 or action_t.ndim != 2 or obs_tp1.shape != obs_t.shape or action_t.shape[0] != obs_t.shape[0]:
        raise ValueError("causal features require aligned 2-D obs/action batches")
    return torch.cat((obs_t, action_t, obs_tp1 - obs_t), dim=-1)
