"""Payload-conditioned FiLM modulation of a frozen policy's hidden layer.

Why this exists. Training only the four context COLUMNS gives the policy one
degree of freedom: measured 2026-09-06, the reachable behaviours form a roughly
one-dimensional family in which cut height and ground contact trade against
lateral tracking. Cut passed in 12 of 12 configurations and lateral in 8, and
the four lateral failures are all where the frozen policy still holds high
ground contact, so there is no contact left to buy the lateral back with.
Different training signals only slide the policy along that curve: an eightfold
lateral reward price did not bend it, and A-RMA landed at the opposite end of it.

FiLM gives a second degree of freedom without unfreezing the controller. The
context sets a per-unit scale and shift on one hidden layer,

    h <- h * (1 + gamma(c)) + beta(c)

which can express a different function of the payload for each hidden unit
rather than one shared direction in input space.

**Zero-initialised, so it is an exact identity at construction.** With the FiLM
layer's weight and bias both zero, gamma = beta = 0 and h is unchanged, so the
wrapped policy reproduces the frozen one bit for bit for ANY context value. That
is the same property the context-column route relies on, and it means every arm
of an ablation starts as the identical policy.

Trainable parameters: `n_ctx * 2 * hidden + 2 * hidden`, which is 2560 for a
4-wide context and a 256-unit layer, against 59,588 in the actor.
"""

from __future__ import annotations

import torch
from torch import nn


class FiLMWrap(nn.Module):
    """Wrap an rsl_rl actor's ``mlp`` with payload-conditioned FiLM."""

    def __init__(self, base_mlp: nn.Sequential, n_ctx: int = 4,
                 split_at: int = 2) -> None:
        super().__init__()
        # Split after the first Linear+activation pair, so the modulation acts
        # on a representation that has already mixed the observation rather than
        # on the raw inputs, where it would collapse back to a scaling of the
        # input columns.
        self.head = nn.Sequential(*list(base_mlp)[:split_at])
        self.tail = nn.Sequential(*list(base_mlp)[split_at:])
        hidden = None
        for layer in list(base_mlp)[:split_at]:
            if isinstance(layer, nn.Linear):
                hidden = int(layer.out_features)
        if hidden is None:
            raise RuntimeError("FiLMWrap found no Linear before the split")
        self.n_ctx = int(n_ctx)
        self.hidden = hidden
        self.film = nn.Linear(self.n_ctx, 2 * hidden)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context = x[:, -self.n_ctx:]
        h = self.head(x)
        gamma, beta = self.film(context).chunk(2, dim=-1)
        return self.tail(h * (1.0 + gamma) + beta)

    def film_parameters(self):
        return self.film.parameters()
