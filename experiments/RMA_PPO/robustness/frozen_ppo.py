"""Loader for the frozen verge PPO actor, independent of RSL-RL/Isaac.

The checkpoint is the promoted PPO4_1 sole policy.  Nothing here trains or
modifies it: weights, observation normalizer and action std are all frozen.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

REPO = Path("/Volumes/DATA/rrlab_mulag")
SOURCE = REPO / "logs/rsl_rl/ppo4_1_sole/speed_range_2_7_pathfix_v2_idfd/model_promoted_2_7kmh.pt"


class FrozenActor(nn.Module):
    """RSL-RL compatible MLP actor without depending on Isaac/RSL packages."""

    def __init__(self, state: dict[str, torch.Tensor]):
        super().__init__()
        self.width = int(state["mlp.0.weight"].shape[1])
        self.mlp = nn.Sequential(
            nn.Linear(self.width, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(),
            nn.Linear(64, 4),
        )
        self.mlp.load_state_dict({
            "0.weight": state["mlp.0.weight"], "0.bias": state["mlp.0.bias"],
            "2.weight": state["mlp.2.weight"], "2.bias": state["mlp.2.bias"],
            "4.weight": state["mlp.4.weight"], "4.bias": state["mlp.4.bias"],
            "6.weight": state["mlp.6.weight"], "6.bias": state["mlp.6.bias"],
        })
        self.register_buffer("obs_mean", state["obs_normalizer._mean"].float().reshape(1, -1).clone())
        self.register_buffer("obs_std", state["obs_normalizer._std"].float().reshape(1, -1).clone())
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def z(self, obs: torch.Tensor) -> torch.Tensor:
        """Per-channel standard score of an observation under the frozen normalizer."""
        return (obs - self.obs_mean) / torch.clamp(self.obs_std, min=1.0e-6)

    @torch.no_grad()
    def mean(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.z(obs))


def load_frozen_actor(checkpoint: Path = SOURCE) -> FrozenActor:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return FrozenActor(payload["actor_state_dict"])
