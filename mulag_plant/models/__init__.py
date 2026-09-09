"""PPO2 ROM checkpoint references.

This package deliberately does not vendor or resolve any ID/FD checkpoints.
"""

from __future__ import annotations

from pathlib import Path


ROM_ROOT = Path(__file__).resolve().parents[2] / "learned_models/rom_model"  # ROM_pipeline model defs + checkpoints live here
COMMAND_ROM_CHECKPOINT = (
    ROM_ROOT
    / "plant_reachable_v1_finetune_blend25/best_rom.pt"
)


__all__ = ["ROM_ROOT", "COMMAND_ROM_CHECKPOINT"]
