from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .rom_schema import CYLINDER_ORDER, _features


@dataclass(frozen=True)
class BaseROMSchema:
    """Valve-conditioned base ROM contract derived from the Simscape logs."""

    version: str = "hydraulic_base_rom_cylinder_only_v2"
    sample_time_s: float = 0.05
    history_length: int = 6
    cylinder_order: tuple[str, ...] = CYLINDER_ORDER
    input_feature_names: tuple[str, ...] = (
        _features("ValveCmd", 4)
        + _features("Extension", 4)
        + _features("ExtensionVelocity", 4)
        + _features("dP", 4)
        + _features("Force", 4)
    )
    target_feature_names: tuple[str, ...] = (
        _features("DeltaDP", 4)
        + _features("DeltaExtensionVelocity", 4)
        + _features("DeltaExtension", 4)
        + _features("ForceNext", 4)
    )
    auxiliary_feature_names: tuple[str, ...] = ()

    @property
    def input_dim(self) -> int:
        return len(self.input_feature_names)

    @property
    def command_dim(self) -> int:
        return 4

    @property
    def target_dim(self) -> int:
        return len(self.target_feature_names)

    @property
    def valve_slice(self) -> slice:
        return slice(0, 4)

    @property
    def command_slice(self) -> slice:
        return self.valve_slice

    @property
    def extension_slice(self) -> slice:
        return slice(4, 8)

    @property
    def velocity_slice(self) -> slice:
        return slice(8, 12)

    @property
    def pressure_slice(self) -> slice:
        return slice(12, 16)

    @property
    def force_slice(self) -> slice:
        return slice(16, 20)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(input_dim=self.input_dim, command_dim=self.command_dim, target_dim=self.target_dim)
        return payload

    def validate(self) -> None:
        if self.input_dim != 20 or self.command_dim != 4 or self.target_dim != 16:
            raise ValueError("The base ROM v2 contract must be 20 cylinder-space inputs and 16 targets.")
        if self.history_length != 6 or self.sample_time_s != 0.05:
            raise ValueError("The verified base ROM contract is six samples at 0.05 s.")


BASE_ROM_SCHEMA = BaseROMSchema()
