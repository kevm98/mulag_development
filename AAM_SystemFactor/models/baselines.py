"""Baseline models for the AAM SystemFactor."""

import numpy as np
import torch
import torch.nn as nn

from AAM_SystemFactor.data.factor_targets import NOMINAL_TARGET


class NominalBaseline:
    """Always predicts the nominal factor vector: [0, 0, 0, 1, 1, 1, 1]."""

    def __init__(self):
        self.nominal = NOMINAL_TARGET.copy()

    def predict(self, *args, **kwargs) -> np.ndarray:
        """Return nominal targets broadcast to batch size."""
        batch_size = 1
        if args and isinstance(args[0], np.ndarray):
            batch_size = args[0].shape[0]
        elif args and isinstance(args[0], torch.Tensor):
            batch_size = args[0].shape[0]
        return np.tile(self.nominal, (batch_size, 1))

    def to(self, device):
        return self


class AggregateMLP(nn.Module):
    """Aggregate-feature regressor: MLP on causal history summary stats.

    Uses five summary statistics per feature channel:
        mean, std, min, max, end_value
    Input: (B, 20 * 5) = (B, 100)
    """

    def __init__(self, input_features: int = 20, n_stats: int = 5, output_size: int = 7):
        super().__init__()
        input_dim = input_features * n_stats
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.SiLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, output_size),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    @staticmethod
    def compute_summary_stats(x: torch.Tensor) -> torch.Tensor:
        """Compute summary stats from (B, T, F) windows.

        Returns (B, F * 5): mean, std, min, max, last_value.
        """
        B, T, F = x.shape
        mean = x.mean(dim=1)
        std = x.std(dim=1)
        vmin = x.min(dim=1).values
        vmax = x.max(dim=1).values
        last = x[:, -1, :]
        return torch.cat([mean, std, vmin, vmax, last], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stats = self.compute_summary_stats(x)
        return self.net(stats)
