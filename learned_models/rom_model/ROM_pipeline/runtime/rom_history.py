from __future__ import annotations

import torch


class ROMHistory:
    def __init__(self, num_envs: int, history_length: int, input_dim: int, *, device: str | torch.device = "cpu") -> None:
        self.buffer = torch.zeros((num_envs, history_length, input_dim), dtype=torch.float32, device=device)
        self.count = torch.zeros(num_envs, dtype=torch.long, device=device)

    @property
    def ready(self) -> torch.Tensor:
        return self.count >= self.buffer.shape[1]

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self.buffer.zero_()
            self.count.zero_()
        else:
            self.buffer[env_ids] = 0.0
            self.count[env_ids] = 0

    def append(self, row: torch.Tensor) -> None:
        if row.shape != (self.buffer.shape[0], self.buffer.shape[2]):
            raise ValueError("ROM history row has the wrong shape.")
        if not torch.isfinite(row).all():
            raise ValueError("ROM history row contains NaN/Inf.")
        self.buffer[:, :-1] = self.buffer[:, 1:].clone()
        self.buffer[:, -1] = row
        self.count.clamp_max_(self.buffer.shape[1] - 1).add_(1)

    def prime(self, row: torch.Tensor) -> None:
        if row.shape != (self.buffer.shape[0], self.buffer.shape[2]):
            raise ValueError("ROM history prime row has the wrong shape.")
        self.buffer[:] = row.unsqueeze(1)
        self.count[:] = self.buffer.shape[1]
