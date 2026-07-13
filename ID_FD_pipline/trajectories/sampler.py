"""Batched trajectory sampler that wraps a shape generator with speed control.

The sampler advances arc-length progress at a configured path speed and calls
the underlying shape generator at every step.  It is the primary interface
used by both deterministic gates and the PPO environment.
"""

from __future__ import annotations

from typing import Callable

import torch

from .types import CutterTrajectorySample


class TrajectorySampler:
    """Batched progress-driven trajectory sampler.

    Maintains per-environment path-progress state and calls the supplied
    ``shape_fn(path_distance_m, anchor=..., ...) -> CutterTrajectorySample``
    at every step.

    The caller is responsible for resetting ``path_progress`` when an episode
    or work phase restarts.
    """

    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        shape_fn: Callable[..., CutterTrajectorySample],
        *,
        path_speed_mps: float = 0.05,
        step_dt: float = 1.0 / 60.0,
        anchor: torch.Tensor | None = None,
        **shape_kwargs,
    ):
        self.num_envs = int(num_envs)
        self.device = device
        self._path_speed_mps = float(path_speed_mps)
        self._step_dt = float(step_dt)
        self._shape_fn = shape_fn
        self._shape_kwargs = shape_kwargs

        self.path_progress = torch.zeros(num_envs, device=device)
        self.accumulated_path = torch.zeros(num_envs, device=device)

        if anchor is not None:
            self.anchor = anchor.to(device, torch.float32)
        else:
            self.anchor = torch.zeros(num_envs, 3, device=device, dtype=torch.float32)

    def set_anchor(self, anchor: torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        """Update the truck-frame anchor position for selected environments."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.anchor[env_ids] = anchor[env_ids].to(self.device, torch.float32)

    def set_speed(self, speed_mps: float) -> None:
        """Change the path-advance speed."""
        self._path_speed_mps = float(speed_mps)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Zero the progress counters for selected environments."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.path_progress[env_ids] = 0.0
        self.accumulated_path[env_ids] = 0.0

    def advance(self, env_ids: torch.Tensor | None = None) -> None:
        """Increment path progress by one time step."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        delta = self._path_speed_mps * self._step_dt
        self.path_progress[env_ids] += delta
        self.accumulated_path[env_ids] += delta

    def sample(self) -> CutterTrajectorySample:
        """Sample the trajectory at the current path progress.

        Returns:
            ``CutterTrajectorySample`` with batch size ``num_envs``.
        """
        return self._shape_fn(
            self.path_progress,
            anchor=self.anchor,
            **self._shape_kwargs,
        )

    def step(self) -> CutterTrajectorySample:
        """Advance progress then sample (convenience for per-step calls)."""
        self.advance()
        return self.sample()
