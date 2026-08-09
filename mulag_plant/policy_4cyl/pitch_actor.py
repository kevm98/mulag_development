"""Actor with a frozen three-cylinder teacher and an independent pitch branch."""

from __future__ import annotations

import torch
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP


class _TeacherPitchMLP(torch.nn.Module):
    """Concatenate the accepted three actions with a dedicated fourth action."""

    def __init__(
        self,
        *,
        observation_dim: int,
        teacher_hidden_dims: list[int] | tuple[int, ...],
        pitch_hidden_dims: list[int] | tuple[int, ...],
        pitch_obs_indices: list[int] | tuple[int, ...],
        activation: str,
    ) -> None:
        super().__init__()
        indices = torch.as_tensor(pitch_obs_indices, dtype=torch.long)
        if indices.ndim != 1 or indices.numel() == 0:
            raise ValueError("pitch observation indices must be a non-empty vector")
        if int(indices.min()) < 0 or int(indices.max()) >= int(observation_dim):
            raise ValueError(
                f"pitch observation index outside [0, {observation_dim}): "
                f"{indices.tolist()}"
            )
        self.register_buffer("pitch_obs_indices", indices, persistent=True)
        self.teacher_mlp = MLP(
            observation_dim, 3, teacher_hidden_dims, activation
        )
        self.pitch_mlp = MLP(
            int(indices.numel()), 1, pitch_hidden_dims, activation
        )
        # Exact zero keeps the transferred policy identical to model_249 at
        # initialization. Hidden layers become trainable after the output row's
        # first PPO update.
        final = self.pitch_mlp[-1]
        if not isinstance(final, torch.nn.Linear):
            raise TypeError("pitch MLP must end in a linear layer")
        torch.nn.init.zeros_(final.weight)
        torch.nn.init.zeros_(final.bias)

    def forward(self, normalized_observation: torch.Tensor) -> torch.Tensor:
        teacher = self.teacher_mlp(normalized_observation)
        pitch_input = torch.index_select(
            normalized_observation, -1, self.pitch_obs_indices
        )
        pitch = self.pitch_mlp(pitch_input)
        return torch.cat((teacher, pitch), dim=-1)


class FrozenTeacherPitchActor(MLPModel):
    """RSL-RL actor whose fourth action has its own pitch-specific MLP."""

    def __init__(
        self,
        obs,
        obs_groups,
        obs_set,
        output_dim,
        hidden_dims=(256, 128, 64),
        activation="elu",
        obs_normalization=True,
        distribution_cfg=None,
        pitch_hidden_dims=(64, 32),
        pitch_obs_indices=(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
            15, 19, 23, 27, 31, 35, 36, 37, 38, 39, 40, 41,
        ),
    ) -> None:
        if int(output_dim) != 4:
            raise ValueError(f"pitch actor requires four outputs, got {output_dim}")
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=distribution_cfg,
        )
        self.mlp = _TeacherPitchMLP(
            observation_dim=self.obs_dim,
            teacher_hidden_dims=hidden_dims,
            pitch_hidden_dims=pitch_hidden_dims,
            pitch_obs_indices=pitch_obs_indices,
            activation=activation,
        )

    def update_normalization(self, obs) -> None:
        """Keep accepted teacher normalization bitwise frozen."""
        return None
