"""On-policy history and a deliberately untrained CPU AAM skeleton.

NOTE ON NAMING: the estimator/trainer here are CPU smoke-test scaffolding and
are NOT the deployed adaptation encoder. The real encoder (phi) is
``core.AAMEstimator``. These were previously both named ``AAMEstimator``, so an
import could silently return either one depending on the module path; they are
now ``SmokeAAMEstimator`` / ``SmokeAAMTrainer``, with the old names kept as
aliases for backwards compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .core import CONTEXT_DIM, causal_mismatch_features
from .modes import assert_no_context_leakage


@dataclass(frozen=True)
class HistorySchema:
    version: str = "rma_on_policy_v1"
    fields: tuple[str, ...] = ("obs_t", "action_t", "obs_tp1", "reward_t", "done_t")


class OnPolicyHistoryRecorder:
    """Records only causal PPO rollout rows; it has no true-context field."""

    schema = HistorySchema()

    def __init__(self, *, obs_dim: int, action_dim: int = 4):
        if obs_dim <= 0 or action_dim <= 0:
            raise ValueError("observation and action dimensions must be positive")
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.rows: list[dict[str, torch.Tensor]] = []
        assert_no_context_leakage(self.schema.fields)

    def append(self, obs_t: torch.Tensor, action_t: torch.Tensor, obs_tp1: torch.Tensor, reward_t: float | torch.Tensor, done_t: bool | torch.Tensor) -> None:
        obs_t = torch.as_tensor(obs_t, dtype=torch.float32).reshape(-1)
        action_t = torch.as_tensor(action_t, dtype=torch.float32).reshape(-1)
        obs_tp1 = torch.as_tensor(obs_tp1, dtype=torch.float32).reshape(-1)
        if obs_t.numel() != self.obs_dim or obs_tp1.numel() != self.obs_dim or action_t.numel() != self.action_dim:
            raise ValueError("history row width does not match recorder schema")
        self.rows.append({
            "obs_t": obs_t.clone(), "action_t": action_t.clone(),
            "obs_tp1": obs_tp1.clone(),
            "reward_t": torch.as_tensor(reward_t, dtype=torch.float32).reshape(()).clone(),
            "done_t": torch.as_tensor(done_t, dtype=torch.bool).reshape(()).clone(),
        })

    def __len__(self) -> int:
        return len(self.rows)

    def tensors(self) -> dict[str, torch.Tensor]:
        if not self.rows:
            return {name: torch.empty((0,), dtype=torch.float32) for name in self.schema.fields}
        return {name: torch.stack([row[name] for row in self.rows]) for name in self.schema.fields}

    def estimator_features(self) -> torch.Tensor:
        data = self.tensors()
        return causal_mismatch_features(data["obs_t"], data["action_t"], data["obs_tp1"])


class SmokeAAMEstimator(nn.Module):
    """CPU smoke-test GRU skeleton; intentionally uninitialized. NOT the deployed
    encoder -- see ``core.AAMEstimator`` for phi."""

    def __init__(self, *, obs_dim: int, action_dim: int = 4, hidden_dim: int = 64, context_dim: int = CONTEXT_DIM):
        super().__init__()
        if context_dim != CONTEXT_DIM:
            raise ValueError(f"RMA context width is fixed at {CONTEXT_DIM}")
        input_dim = 2 * int(obs_dim) + int(action_dim)
        self.gru = nn.GRU(input_dim, int(hidden_dim), batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(int(hidden_dim)), nn.Linear(int(hidden_dim), context_dim))

    def forward(self, features: torch.Tensor, hidden: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim == 2:
            features = features.unsqueeze(1)
        sequence, hidden = self.gru(features, hidden)
        return self.head(sequence[:, -1]), hidden


class SmokeAAMTrainer:
    """CPU smoke trainer for synthetic tensors, never an old-dataset loader."""

    def __init__(self, estimator: SmokeAAMEstimator, *, learning_rate: float = 1.0e-3):
        self.estimator = estimator.to("cpu")
        self.optimizer = torch.optim.Adam(self.estimator.parameters(), lr=learning_rate)

    def train_step(self, features: torch.Tensor, target_context: torch.Tensor) -> float:
        features = torch.as_tensor(features, dtype=torch.float32, device="cpu")
        target_context = torch.as_tensor(target_context, dtype=torch.float32, device="cpu")
        if target_context.ndim != 2 or target_context.shape[-1] != CONTEXT_DIM:
            raise ValueError("synthetic target context must be (N, 4)")
        prediction, _ = self.estimator(features)
        loss = torch.mean((prediction - target_context) ** 2)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return float(loss.detach())


# Backwards-compatible aliases (see module docstring).
AAMEstimator = SmokeAAMEstimator
AAMTrainer = SmokeAAMTrainer
