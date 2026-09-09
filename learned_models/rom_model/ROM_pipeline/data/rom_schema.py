from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


CYLINDER_ORDER = (
    "Drehzapfen_joint",
    "Ausleger_I_joint",
    "Ausleger_II_joint",
    "Messerkopf_Schwenk_joint",
)
def _features(prefix: str, count: int) -> tuple[str, ...]:
    return tuple(f"{prefix}_{index}" for index in range(1, count + 1))


@dataclass(frozen=True)
class ROMSchema:
    """Canonical raw feature and target contract.

    Histories have shape ``[batch, history_length, input_dim]``. The current
    desired command is present in the final history row and is also accepted as
    a separate runtime argument so callers cannot accidentally reuse a stale
    command.
    """

    version: str = "hydraulic_command_cylinder_only_rom_v2"
    sample_time_s: float = 0.05
    history_length: int = 6
    cylinder_order: tuple[str, ...] = CYLINDER_ORDER
    state_feature_names: tuple[str, ...] = (
        _features("Extension", 4)
        + _features("ExtensionVelocity", 4)
        + _features("dP", 4)
        + _features("PreviousValveCmd", 4)
    )
    command_feature_names: tuple[str, ...] = (
        _features("DesiredDeltaExtension", 4)
        + _features("DesiredDeltaExtensionVelocity", 4)
    )
    auxiliary_feature_names: tuple[str, ...] = _features("ValveCmd", 4)
    target_feature_names: tuple[str, ...] = (
        _features("PredictedValveCmd", 4)
        + _features("DeltaDP", 4)
        + _features("DeltaExtensionVelocity", 4)
        + _features("DeltaExtension", 4)
    )

    @property
    def input_feature_names(self) -> tuple[str, ...]:
        return self.state_feature_names + self.command_feature_names

    @property
    def input_dim(self) -> int:
        return len(self.input_feature_names)

    @property
    def command_dim(self) -> int:
        return len(self.command_feature_names)

    @property
    def target_dim(self) -> int:
        return len(self.target_feature_names)

    @property
    def extension_slice(self) -> slice:
        return slice(0, 4)

    @property
    def velocity_slice(self) -> slice:
        return slice(4, 8)

    @property
    def pressure_slice(self) -> slice:
        return slice(8, 12)

    @property
    def previous_valve_slice(self) -> slice:
        return slice(12, 16)

    @property
    def command_slice(self) -> slice:
        return slice(16, 24)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["input_feature_names"] = self.input_feature_names
        payload["input_dim"] = self.input_dim
        payload["command_dim"] = self.command_dim
        payload["target_dim"] = self.target_dim
        return payload

    def validate(self) -> None:
        if len(self.cylinder_order) != 4 or len(set(self.cylinder_order)) != 4:
            raise ValueError("The ROM requires four uniquely ordered cylinders.")
        if self.input_dim != 24 or self.command_dim != 8 or self.target_dim != 16:
            raise ValueError("Unexpected ROM dimensions; the v2 contract is 24 -> 16 with an 8D command.")
        if self.history_length < 2 or self.sample_time_s <= 0.0:
            raise ValueError("History length and sample time must be positive.")


DEFAULT_SCHEMA = ROMSchema()
