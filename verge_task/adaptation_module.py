"""The RMA adaptation module phi, in an import-safe module.

Kept out of ``PPO_AAM_FD/scripts/train_phase2_dagger.py`` deliberately: that
script calls ``AppLauncher`` at import time, so importing phi from it would
launch a second simulation app. Phase 2 (training) and Phase 3 (deployment of
the frozen module) both import from here.
"""

from __future__ import annotations

import torch
from torch import nn


class AdaptationModule(nn.Module):
    """phi: flattened (state, action) history -> context estimate.

    A 1-D temporal convolution over the history, as in RMA, rather than a flat
    MLP: the plant signature is a temporal pattern (how the response lags and
    scales the command), and a conv shares weights across time instead of
    relearning it at every offset.
    """

    def __init__(self, feature_dim: int, history: int, context_dim: int) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.history = int(history)
        self.encoder = nn.Sequential(
            nn.Linear(self.feature_dim, 64), nn.ELU(),
        )
        self.temporal = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=5, stride=1), nn.ELU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=1), nn.ELU(),
        )
        with torch.no_grad():
            probe = torch.zeros(1, 64, self.history)
            flat = self.temporal(probe).flatten(1).shape[1]
        self.head = nn.Sequential(nn.Linear(flat, 64), nn.ELU(), nn.Linear(64, int(context_dim)))

    def forward(self, hist: torch.Tensor) -> torch.Tensor:
        # hist: (B, history, feature_dim)
        z = self.encoder(hist)                 # (B, history, 64)
        z = self.temporal(z.transpose(1, 2))   # (B, 64, history')
        return self.head(z.flatten(1))
