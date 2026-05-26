from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from kevin_integration.controllers.learned_models import (
    AAMModel,
    ForwardDynamicsModel,
    InverseDynamicsModel,
)
from kevin_integration.rl.action_adapter import clamp_and_sanitize
from kevin_integration.rl.observation_builder import build_aam_input, build_fd_input, build_id_input


class TorchSequenceBuffer:
    def __init__(self, *, num_envs: int, window_size: int, feature_dim: int, device: torch.device):
        self.data = torch.zeros((num_envs, window_size, feature_dim), dtype=torch.float32, device=device)

    def append(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[0] != self.data.shape[0] or value.shape[-1] != self.data.shape[-1]:
            raise ValueError(
                f"Expected value shape ({self.data.shape[0]}, {self.data.shape[-1]}), got {tuple(value.shape)}"
            )
        self.data = torch.roll(self.data, shifts=-1, dims=1)
        self.data[:, -1, :] = value
        return self.data

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> None:
        if env_ids is None:
            self.data.zero_()
            return
        self.data[env_ids] = 0.0


@dataclass(frozen=True)
class AdaptiveRLLearnedControllerCfg:
    models_dir: str | Path
    num_envs: int
    device: str
    fallback_window: int = 10
    activation: str = "relu"
    dropout: float = 0.0
    max_abs_valve: float = 1.0
    max_abs_torque: float = 5000.0


class AdaptiveRLLearnedController:
    def __init__(self, cfg: AdaptiveRLLearnedControllerCfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        models_dir = Path(cfg.models_dir)

        self.aam = AAMModel(
            checkpoint_path=models_dir / "aam.pth",
            activation=cfg.activation,
            dropout=cfg.dropout,
            map_location="cpu",
            device=str(self.device),
        )
        self.id = InverseDynamicsModel(
            checkpoint_path=models_dir / "id.pth",
            activation=cfg.activation,
            dropout=cfg.dropout,
            map_location="cpu",
            device=str(self.device),
        )
        self.fd = ForwardDynamicsModel(
            checkpoint_path=models_dir / "fd.pth",
            scaler_path=models_dir / "fd_scaler.pth",
            activation=cfg.activation,
            dropout=cfg.dropout,
            map_location="cpu",
            device=str(self.device),
        )
        self._freeze_model(self.aam)
        self._freeze_model(self.id)
        self._freeze_model(self.fd)

        aam_window = self.aam.window_size or cfg.fallback_window
        id_window = self.id.window_size or cfg.fallback_window
        fd_window = self.fd.window_size or cfg.fallback_window

        self._aam_history = TorchSequenceBuffer(
            num_envs=cfg.num_envs, window_size=aam_window, feature_dim=18, device=self.device
        )
        self._id_history = TorchSequenceBuffer(
            num_envs=cfg.num_envs, window_size=id_window, feature_dim=19, device=self.device
        )
        self._fd_history = TorchSequenceBuffer(
            num_envs=cfg.num_envs, window_size=fd_window, feature_dim=18, device=self.device
        )

    @staticmethod
    def _freeze_model(wrapper) -> None:
        wrapper.model.requires_grad_(False)
        wrapper.model.eval()

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> None:
        self._aam_history.reset(env_ids)
        self._id_history.reset(env_ids)
        self._fd_history.reset(env_ids)

    def compute_z_arm(
        self,
        q: torch.Tensor,
        qdot: torch.Tensor,
        dP: torch.Tensor,
        prev_valve_cmd: torch.Tensor,
        *,
        update_history: bool = True,
    ) -> torch.Tensor:
        aam_x_t = build_aam_input(q, qdot, dP, prev_valve_cmd).to(self.device)
        aam_seq = self._aam_history.append(aam_x_t) if update_history else self._aam_history.data
        return clamp_and_sanitize(self.aam(aam_seq), None)

    def apply_action(
        self,
        q: torch.Tensor,
        qdot: torch.Tensor,
        dP: torch.Tensor,
        rl_action: torch.Tensor,
        *,
        z_arm_hat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = q.to(self.device)
        qdot = qdot.to(self.device)
        dP = dP.to(self.device)
        rl_action = rl_action.to(self.device)
        z_arm_hat = z_arm_hat.to(self.device)

        id_x_t = build_id_input(q, qdot, rl_action, dP)
        id_seq = self._id_history.append(id_x_t)
        valve_cmd = clamp_and_sanitize(self.id(id_seq, z_arm_hat), self.cfg.max_abs_valve)

        fd_x_t = build_fd_input(q, qdot, dP, valve_cmd)
        fd_seq = self._fd_history.append(fd_x_t)
        torque = clamp_and_sanitize(self.fd(fd_seq), self.cfg.max_abs_torque)

        return torque, valve_cmd, z_arm_hat
