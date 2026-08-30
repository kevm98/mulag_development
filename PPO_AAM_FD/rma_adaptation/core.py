"""RMA / A-RMA adaptation contract for the PPO -> FD -> PhysX stack.

This module was specified by `experiments/RMA_PPO/config.json`, its runbook and
`PPO_AAM_FD/tests/test_rma_adaptation_cpu.py`, but never written -- the test
file imports it and therefore could not even be collected. The design work is
not being redone here; this is the implementation the design was waiting for.

Design decisions inherited from that specification, and why they are right for
this stack:

* **The context is four-wide**: `(actuator_authority_alpha,
  hydraulic_response_time_tau, actuator_delay_steps, mass_scale)`. It is NOT the
  six ROM factors. Four of those six -- flow, compliance, seal, pressure -- are
  inert here because `CylinderFD.step()` takes no factor argument, and E4
  measured that feeding a policy near-noise scores *worse* than feeding it a
  constant. Every axis kept here is one that measurably moves the plant.
* **The stress acts between commanded and applied valve**, so the policy, the ID
  model and the AAM all act on the command and only the response changes. That
  commanded/response mismatch is the sole signal the estimator may use.
* **The estimator never sees a context label.** `assert_no_context_leakage`
  enforces it on the feature schema, because an estimator trained on the answer
  is the failure this whole design exists to avoid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import torch
from torch import nn

# The privileged context, in a fixed order. Everything that serialises or
# normalises the context reads this, so the order cannot drift between phases.
CONTEXT_NAMES: tuple[str, ...] = (
    "actuator_authority_alpha",
    "hydraulic_response_time_tau",
    "actuator_delay_steps",
    "mass_scale",
)
CONTEXT_DIM = len(CONTEXT_NAMES)

# Normalisation constants: centre at nominal, scale by the admitted range, so a
# nominal plant is the zero vector and s=1 on an axis is order 1.
_CONTEXT_CENTRE = torch.tensor([1.0, 0.0, 0.0, 1.0])
_CONTEXT_SCALE = torch.tensor([0.75, 0.12, 3.0, 0.5])


def normalized_context(context: torch.Tensor) -> torch.Tensor:
    """Map a raw context vector to normalised coordinates (nominal -> 0)."""
    centre = _CONTEXT_CENTRE.to(context.device, context.dtype)
    scale = _CONTEXT_SCALE.to(context.device, context.dtype)
    return (context - centre) / scale


@dataclass
class DeploymentContext:
    """One environment's plant, in the four admitted axes."""

    actuator_authority_alpha: float = 1.0
    hydraulic_response_time_tau: float = 0.0
    actuator_delay_steps: int = 0
    mass_scale: float = 1.0

    def as_tensor(self) -> torch.Tensor:
        return torch.tensor(
            [
                float(self.actuator_authority_alpha),
                float(self.hydraulic_response_time_tau),
                float(self.actuator_delay_steps),
                float(self.mass_scale),
            ],
            dtype=torch.float32,
        )

    @property
    def is_nominal(self) -> bool:
        return (
            self.actuator_authority_alpha == 1.0
            and self.hydraulic_response_time_tau == 0.0
            and int(self.actuator_delay_steps) == 0
            and self.mass_scale == 1.0
        )


@dataclass
class ActuationConfig:
    dt_s: float = 0.05
    max_delay_steps: int = 8
    num_cylinders: int = 4


class DeploymentActuation:
    """Commanded valve -> applied valve, as authority -> lag -> delay.

    The order matters and is fixed: authority is a static gain, lag is a
    first-order hold on the scaled command, and delay is a pure transport FIFO
    on the lagged signal. Reordering changes the composed response.

    At nominal every stage is an exact identity -- not an approximate one -- so
    a nominal run is bit-identical to an unstressed run and the identity smoke
    test is meaningful.
    """

    def __init__(self, config: ActuationConfig | None = None) -> None:
        self.config = config or ActuationConfig()
        self._lag_state: torch.Tensor | None = None
        self._buffer: torch.Tensor | None = None
        self._write = 0

    def reset(self) -> None:
        self._lag_state = None
        self._buffer = None
        self._write = 0

    def step(self, command: torch.Tensor, context: DeploymentContext) -> torch.Tensor:
        u = command * float(context.actuator_authority_alpha)

        tau = float(context.hydraulic_response_time_tau)
        if tau > 0.0:
            if self._lag_state is None or self._lag_state.shape != u.shape:
                self._lag_state = torch.zeros_like(u)
            # Exact discretisation of a first-order hold, not a Euler step: the
            # step response must stay bounded for any dt/tau.
            beta = 1.0 - torch.exp(torch.tensor(-self.config.dt_s / tau, dtype=u.dtype))
            self._lag_state = self._lag_state + beta * (u - self._lag_state)
            u = self._lag_state

        delay = int(context.actuator_delay_steps)
        if delay <= 0:
            return u
        if delay > self.config.max_delay_steps:
            raise ValueError(
                f"delay {delay} exceeds max_delay_steps {self.config.max_delay_steps}"
            )
        depth = self.config.max_delay_steps + 1
        if self._buffer is None or self._buffer.shape[1:] != u.shape:
            self._buffer = torch.zeros((depth, *u.shape), dtype=u.dtype, device=u.device)
            self._write = 0
        self._buffer[self._write] = u
        read = (self._write - delay) % depth
        out = self._buffer[read].clone()
        self._write = (self._write + 1) % depth
        return out


class ContextMode(Enum):
    """How the policy's context slot is filled at runtime.

    The ablation arms are modes, not ad-hoc call sites, so an evaluation cannot
    silently run a different substitution than it reports.
    """

    BASELINE = "baseline"
    PRIVILEGED = "privileged"
    MEAN_CONTEXT = "mean_context"
    SHUFFLED_CONTEXT = "shuffled_context"
    AAM_ESTIMATED = "estimated"
    AAM_FROZEN = "frozen"
    AAM_ESTIMATED_FINETUNED = "estimated_finetuned"


def assemble_policy_observation(
    base_observation: torch.Tensor, mode: ContextMode, **streams: torch.Tensor
) -> torch.Tensor:
    """Concatenate the base observation with the context this mode selects.

    BASELINE returns the base observation untouched -- it is the no-context
    control and must not be given a zero-filled slot, which would be a different
    network input width and therefore a different experiment.
    """
    if mode is ContextMode.BASELINE:
        return base_observation
    key = mode.value
    if key not in streams:
        raise ValueError(f"mode {mode.name} requires the '{key}' stream; got {sorted(streams)}")
    context = streams[key]
    if context.shape[0] != base_observation.shape[0]:
        raise ValueError(
            f"context batch {context.shape[0]} != observation batch {base_observation.shape[0]}"
        )
    return torch.cat((base_observation, context), dim=-1)


# Any feature name that is (or trivially reveals) a context axis. The estimator
# must infer the plant from the response, never read it.
_FORBIDDEN = set(CONTEXT_NAMES) | {
    "context", "privileged", "mass", "mass_scale", "alpha", "authority",
    "tau", "delay", "delay_steps", "e", "z", "z_true",
}


def assert_no_context_leakage(feature_names) -> None:
    """Raise if any estimator input names a privileged quantity."""
    leaked = sorted({n for n in feature_names if str(n).strip().lower() in _FORBIDDEN})
    if leaked:
        raise ValueError(
            f"estimator feature schema leaks privileged context: {leaked}. "
            "The adaptation module must infer the plant from the commanded/response "
            "mismatch alone."
        )


def causal_mismatch_features(
    obs_t: torch.Tensor, action_t: torch.Tensor, obs_tp1: torch.Tensor
) -> torch.Tensor:
    """(obs_t, action_t, obs_tp1 - obs_t) -- strictly causal, no privileged term.

    The third block is the mismatch: what the plant did with the command. A pair
    of identical commands on two different plants differs here and nowhere else,
    so this is the only place plant identity can enter.
    """
    if obs_t.shape != obs_tp1.shape:
        raise ValueError(f"obs_t {tuple(obs_t.shape)} != obs_tp1 {tuple(obs_tp1.shape)}")
    return torch.cat((obs_t, action_t, obs_tp1 - obs_t), dim=-1)


class OnPolicyHistoryRecorder:
    """Collects the causal transition schema during a rollout.

    On-policy by construction: RMA Phase 2 regressed offline on the teacher's
    state distribution suffers covariate shift, because at deployment the module
    sees the states its own estimation errors produce.
    """

    KEYS = ("obs_t", "action_t", "obs_tp1", "reward_t", "done_t")

    def __init__(self, obs_dim: int, action_dim: int) -> None:
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self._rows: list[dict[str, torch.Tensor]] = []

    def append(self, obs_t, action_t, obs_tp1, reward_t, done_t) -> None:
        self._rows.append(
            {
                "obs_t": torch.as_tensor(obs_t, dtype=torch.float32).reshape(-1),
                "action_t": torch.as_tensor(action_t, dtype=torch.float32).reshape(-1),
                "obs_tp1": torch.as_tensor(obs_tp1, dtype=torch.float32).reshape(-1),
                "reward_t": torch.as_tensor([float(reward_t)], dtype=torch.float32),
                "done_t": torch.as_tensor([float(bool(done_t))], dtype=torch.float32),
            }
        )

    def __len__(self) -> int:
        return len(self._rows)

    def tensors(self) -> dict[str, torch.Tensor]:
        if not self._rows:
            return {k: torch.zeros(0) for k in self.KEYS}
        return {k: torch.stack([r[k] for r in self._rows]) for k in self.KEYS}

    def estimator_features(self) -> torch.Tensor:
        t = self.tensors()
        return causal_mismatch_features(t["obs_t"], t["action_t"], t["obs_tp1"])


class AAMEstimator(nn.Module):
    """phi: causal mismatch features -> context estimate."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64,
                 context_dim: int = CONTEXT_DIM) -> None:
        super().__init__()
        self.feature_dim = 2 * int(obs_dim) + int(action_dim)
        self.net = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, int(context_dim)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class AAMTrainer:
    def __init__(self, estimator: AAMEstimator, lr: float = 1.0e-3) -> None:
        self.estimator = estimator
        self.optimizer = torch.optim.Adam(estimator.parameters(), lr=lr)

    def train_step(self, features: torch.Tensor, target: torch.Tensor) -> float:
        prediction = self.estimator(features)
        loss = nn.functional.mse_loss(prediction, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return float(loss.detach())


def delta_adapt(without_adaptation: torch.Tensor, with_adaptation: torch.Tensor) -> torch.Tensor:
    """Error removed by adaptation. Positive means adaptation helped."""
    return without_adaptation - with_adaptation


def recovery_fraction(errors: torch.Tensor, stressed: float, nominal: float) -> torch.Tensor:
    """Share of the stress-induced gap that adaptation recovers, clipped to [0,1].

    G1 in one number: adaptation can only recover headroom a degradation
    creates, so the denominator is the gap the stress opened, not the error.
    """
    gap = float(stressed) - float(nominal)
    if gap <= 0.0:
        raise ValueError(f"stressed ({stressed}) must exceed nominal ({nominal})")
    return ((float(stressed) - errors) / gap).clamp(0.0, 1.0).mean()


def stress_thresholds() -> dict[str, float]:
    """Severities admitted by the Stage-0 screens.

    Delay leads: `experiments/RMA_PPO/robustness/memory.md` records it as the
    only axis with NO analytic breaking point -- shifting a command in time does
    not change its magnitude, so saturation never leaves its 21.2% cylinder-1
    floor. Authority is demoted: a 7.5% cut already puts 30% of rows past the
    command limits, so its knee measures saturation, not robustness.
    """
    return {
        "delay_onset_steps": 2.0,
        "deadband_onset": 0.05,
        "lag_onset_s": 0.05,
        "authority_onset_alpha": 0.9,
        "authority_breaking_alpha": 0.925,
        "nominal_c1_clip_fraction": 0.212,
    }


@dataclass
class NativeRigidBodyWiring:
    """How mass and COM reach the simulator."""

    mass_scale_source: str = "native_physx_set_masses"
    com_source: str = "native_physx_set_coms"
    body_name: str = "Messerkopf"


def verify_native_mass_com_wiring(wiring: NativeRigidBodyWiring) -> None:
    """Mass and COM must be written into PhysX, never merely observed.

    A context column the policy can read but that never changes the simulated
    body is an inert factor, and inert factors are what made the six ROM columns
    worthless on this stack.
    """
    if not wiring.mass_scale_source.startswith("native_physx"):
        raise ValueError(
            f"mass_scale must be applied through native PhysX rigid-body properties, "
            f"got {wiring.mass_scale_source!r}"
        )
    if not wiring.com_source.startswith("native_physx"):
        raise ValueError(
            f"com must be applied through native PhysX rigid-body properties, "
            f"got {wiring.com_source!r}"
        )
