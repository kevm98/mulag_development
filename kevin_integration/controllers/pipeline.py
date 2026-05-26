from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .learned_models import AAMModel, ForwardDynamicsModel, InverseDynamicsModel, _require_torch
from .sequence_buffer import SequenceBuffer, SequenceSpec


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for the learned control pipeline.

    The feature vectors are treated as *already assembled* according to the training convention.
    Your integration code is responsible for building these vectors in the same ordering as used during training.
    """

    aam_window: Optional[int] = None
    id_window: Optional[int] = None
    fd_window: Optional[int] = None
    fallback_window: int = 10

    aam_dim: int = 18
    id_dim: int = 19
    fd_dim: int = 18

    activation: str = "relu"
    dropout: float = 0.0
    device: str = "cpu"


class LearnedControlPipeline:
    """Implements the learned-model portion of the diagram (AAM -> ID -> FD).

    This is intentionally independent of IsaacLab task code and only operates on numeric vectors.
    """

    def __init__(self, *, models_dir: str | Path, cfg: PipelineConfig = PipelineConfig()):
        torch = _require_torch()
        self._torch = torch
        self._device = torch.device(cfg.device)

        self.cfg = cfg
        models_dir = Path(models_dir)

        self.aam = AAMModel(
            checkpoint_path=models_dir / "aam.pth",
            activation=cfg.activation,
            dropout=cfg.dropout,
            map_location="cpu",
            device=str(self._device),
        )
        self.id = InverseDynamicsModel(
            checkpoint_path=models_dir / "id.pth",
            activation=cfg.activation,
            dropout=cfg.dropout,
            map_location="cpu",
            device=str(self._device),
        )
        self.fd = ForwardDynamicsModel(
            checkpoint_path=models_dir / "fd.pth",
            scaler_path=models_dir / "fd_scaler.pth",
            activation=cfg.activation,
            dropout=cfg.dropout,
            map_location="cpu",
            device=str(self._device),
        )

        aam_window = cfg.aam_window or self.aam.window_size or cfg.fallback_window
        id_window = cfg.id_window or self.id.window_size or cfg.fallback_window
        fd_window = cfg.fd_window or self.fd.window_size or cfg.fallback_window

        self._aam_buf = SequenceBuffer(SequenceSpec(aam_window, cfg.aam_dim))
        self._id_buf = SequenceBuffer(SequenceSpec(id_window, cfg.id_dim))
        self._fd_buf = SequenceBuffer(SequenceSpec(fd_window, cfg.fd_dim))

    def reset(self) -> None:
        self._aam_buf.reset()
        self._id_buf.reset()
        self._fd_buf.reset()

    def step(
        self,
        *,
        aam_x_t: np.ndarray,
        id_x_t: np.ndarray,
        fd_x_t: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Advance one control step.

        Required inputs:
        - aam_x_t: (18,) = [q(5), qdot(5), dP(4), valve_cmd(4)]
        - id_x_t: (19,) = [q(5), qdot(5), qddot(5), dP(4)]

        Optional:
        - fd_x_t: (18,) = [q(5), qdot(5), dP(4), valve_cmd(4)]
        """

        self._aam_buf.append(aam_x_t)
        self._id_buf.append(id_x_t)
        if fd_x_t is not None:
            self._fd_buf.append(fd_x_t)

        # Build padded sequences
        aam_seq = self._aam_buf.as_array(pad=True)
        id_seq = self._id_buf.as_array(pad=True)

        torch = self._torch
        aam_seq_t = torch.as_tensor(aam_seq, dtype=torch.float32, device=self._device).unsqueeze(0)
        id_seq_t = torch.as_tensor(id_seq, dtype=torch.float32, device=self._device).unsqueeze(0)

        z_arm_hat = self.aam(aam_seq_t)  # (1, 24)
        valve_cmd = self.id(id_seq_t, z_arm_hat)  # (1, 4)

        out: Dict[str, Any] = {
            "z_arm_hat": z_arm_hat.squeeze(0).detach().cpu().numpy(),
            "valve_cmd": valve_cmd.squeeze(0).detach().cpu().numpy(),
        }

        if fd_x_t is not None:
            fd_seq = self._fd_buf.as_array(pad=True)
            fd_seq_t = torch.as_tensor(fd_seq, dtype=torch.float32, device=self._device).unsqueeze(0)
            torque = self.fd(fd_seq_t)  # (1, 4)
            out["torque"] = torque.squeeze(0).detach().cpu().numpy()

        return out

    def predict_torque(self, fd_x_t: np.ndarray) -> np.ndarray:
        """Run FD on the current `[q(5), qdot(5), dP(4), valve_cmd(4)]` feature vector."""
        self._fd_buf.append(fd_x_t)
        fd_seq = self._fd_buf.as_array(pad=True)
        fd_seq_t = self._torch.as_tensor(fd_seq, dtype=self._torch.float32, device=self._device).unsqueeze(0)
        torque = self.fd(fd_seq_t)
        return torque.squeeze(0).detach().cpu().numpy()
