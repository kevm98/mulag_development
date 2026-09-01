"""Declarative, no-training configuration for the three RMA phases."""

from __future__ import annotations

from dataclasses import dataclass, field

from .core import CONTEXT_NAMES
from .modes import ContextMode


@dataclass(frozen=True)
class Phase1Config:
    enabled: bool = False
    context_names: tuple[str, ...] = CONTEXT_NAMES
    privileged_context_observation: bool = True
    context_dim: int = 4
    randomization_stages: tuple[str, ...] = ("nominal", "authority", "lag", "delay", "mass", "combined")
    authority_alpha_range: tuple[float, float] = (0.25, 1.0)
    tau_s_range: tuple[float, float] = (0.0, 0.12)
    delay_steps_range: tuple[int, int] = (0, 3)
    mass_scale_range: tuple[float, float] = (0.85, 1.15)
    runtime_mode: ContextMode = ContextMode.PRIVILEGED


@dataclass(frozen=True)
class Phase2Config:
    enabled: bool = False
    history_ticks: int = 20
    causal_feature_names: tuple[str, ...] = ("obs_t", "action_t", "obs_tp1_minus_obs_t")
    context_dim: int = 4
    hidden_dim: int = 64
    source: str = "ppo_on_policy_rollouts_only"
    train_on_historical_classical_dataset: bool = False


@dataclass(frozen=True)
class Phase3Config:
    enabled: bool = False
    runtime_mode: ContextMode = ContextMode.AAM_ESTIMATED_FINETUNED
    fine_tune_iterations: int = 0
    freeze_aam: bool = True
    privileged_gate_required: bool = True


@dataclass(frozen=True)
class RMAExperimentConfig:
    phase1: Phase1Config = field(default_factory=Phase1Config)
    phase2: Phase2Config = field(default_factory=Phase2Config)
    phase3: Phase3Config = field(default_factory=Phase3Config)
    modes: tuple[ContextMode, ...] = tuple(ContextMode)


DEFAULT_RMA_CONFIG = RMAExperimentConfig()
