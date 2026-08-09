from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from ..data.rom_schema import DEFAULT_SCHEMA, ROMSchema
from .hydraulic_rom import HydraulicROM, HydraulicROMConfig


CHECKPOINT_FORMAT_VERSION = 1


def save_checkpoint(
    path: str | Path,
    model: HydraulicROM,
    *,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    cylinder_limits: torch.Tensor,
    training_metadata: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
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
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    schema: ROMSchema = DEFAULT_SCHEMA,
) -> tuple[HydraulicROM, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if int(payload.get("checkpoint_format_version", -1)) != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("Unsupported ROM checkpoint format.")
    stored = payload["schema"]
    if tuple(stored["input_feature_names"]) != schema.input_feature_names:
        raise ValueError("Checkpoint input feature ordering does not match runtime.")
    if tuple(stored["cylinder_order"]) != schema.cylinder_order:
        raise ValueError("Checkpoint actuator ordering does not match runtime.")
    if float(stored["sample_time_s"]) != schema.sample_time_s or int(stored["history_length"]) != schema.history_length:
        raise ValueError("Checkpoint timing contract does not match runtime.")
    normalization = payload["normalization"]
    model = HydraulicROM(
        HydraulicROMConfig(**payload["model_config"]),
        schema=schema,
        input_mean=normalization["input_mean"],
        input_std=normalization["input_std"],
    )
    incompatible = model.load_state_dict(payload["state_dict"], strict=False)
    allowed_missing = {
        "command_extension_correction_head.0.weight",
        "command_extension_correction_head.0.bias",
        "command_extension_correction_head.2.weight",
        "command_extension_correction_head.2.bias",
    }
    unexpected = set(incompatible.unexpected_keys)
    missing = set(incompatible.missing_keys)
    if unexpected or missing - allowed_missing:
        raise ValueError(
            f"Checkpoint model state is incompatible: missing={sorted(missing)} "
            f"unexpected={sorted(unexpected)}"
        )
    model.to(device).eval()
    return model, payload
