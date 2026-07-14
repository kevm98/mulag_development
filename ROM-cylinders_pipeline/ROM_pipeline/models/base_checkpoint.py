from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ..data.base_rom_schema import BASE_ROM_SCHEMA, BaseROMSchema
from .base_hydraulic_rom import BaseHydraulicROM, BaseHydraulicROMConfig


BASE_CHECKPOINT_FORMAT_VERSION = 1


def save_base_checkpoint(
    path: str | Path,
    model: BaseHydraulicROM,
    *,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    cylinder_limits: torch.Tensor,
    training_metadata: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "checkpoint_format_version": BASE_CHECKPOINT_FORMAT_VERSION,
        "model_version": model.schema.version,
        "model_config": model.config.to_dict(),
        "schema": model.schema.to_dict(),
        "state_dict": model.state_dict(),
        "normalization": {
            "input_mean": model.input_mean.detach().cpu(),
            "input_std": model.input_std.detach().cpu(),
            "target_mean": target_mean.detach().cpu(),
            "target_std": target_std.detach().cpu(),
        },
        "cylinder_limits": cylinder_limits.detach().cpu(),
        "training_metadata": training_metadata,
    }, path)


def load_base_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    schema: BaseROMSchema = BASE_ROM_SCHEMA,
) -> tuple[BaseHydraulicROM, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if int(payload.get("checkpoint_format_version", -1)) != BASE_CHECKPOINT_FORMAT_VERSION:
        raise ValueError("Unsupported base ROM checkpoint format.")
    stored = payload["schema"]
    if tuple(stored["input_feature_names"]) != schema.input_feature_names:
        raise ValueError("Base ROM checkpoint input ordering does not match runtime.")
    if tuple(stored["cylinder_order"]) != schema.cylinder_order or tuple(stored["joint_order"]) != schema.joint_order:
        raise ValueError("Base ROM checkpoint cylinder/joint ordering does not match runtime.")
    if float(stored["sample_time_s"]) != schema.sample_time_s or int(stored["history_length"]) != schema.history_length:
        raise ValueError("Base ROM checkpoint timing does not match runtime.")
    normalization = payload["normalization"]
    model = BaseHydraulicROM(
        BaseHydraulicROMConfig(**payload["model_config"]),
        schema=schema,
        input_mean=normalization["input_mean"],
        input_std=normalization["input_std"],
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device).eval(), payload
