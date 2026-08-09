from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .rollout_loss import LossWeights


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 17
    device: str = "cuda:0"
    batch_size: int = 512
    validation_batch_size: int = 1024
    rollout_batch_size: int = 128
    excitation_source_weight: float = 1.0
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-5
    gradient_clip_norm: float = 5.0
    num_workers: int = 4
    stage_epochs: tuple[int, ...] = (30, 30, 30, 40)
    stage_rollout_horizons: tuple[int, ...] = (1, 5, 20, 50)
    validation_horizons: tuple[int, ...] = (1, 5, 10, 20, 50, 100)
    scheduled_sampling_start: float = 0.0
    scheduled_sampling_end: float = 0.75
    early_stopping_patience: int = 20
    loss_weights: LossWeights = field(default_factory=LossWeights)

    def save(self, path: str | Path) -> None:
        payload = asdict(self)
        Path(path).write_text(json.dumps(payload, indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "TrainingConfig":
        payload = json.loads(Path(path).read_text())
        payload["stage_epochs"] = tuple(payload["stage_epochs"])
        payload["stage_rollout_horizons"] = tuple(payload["stage_rollout_horizons"])
        payload["validation_horizons"] = tuple(payload["validation_horizons"])
        payload["loss_weights"] = LossWeights(**payload["loss_weights"])
        return cls(**payload)
