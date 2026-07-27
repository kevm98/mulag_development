from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


CYLINDER_ORDER: tuple[str, ...] = (
    "Drehzapfen_joint",
    "Ausleger_I_joint",
    "Ausleger_II_joint",
    "Messerkopf_Schwenk_joint",
)


@dataclass(frozen=True)
class CylinderCoordinateMapping:
    mapping_mode: str
    scale: torch.Tensor
    offset: torch.Tensor
    sign: torch.Tensor
    simscape_nominal_extension: torch.Tensor
    isaac_nominal_position: torch.Tensor
    isaac_min: torch.Tensor
    isaac_max: torch.Tensor
    sign_verified: tuple[bool, bool, bool, bool]
    mapping_validated: bool
    source: str
    nominal_reference_confidence: str
    reference_sample_count: int | None
    reference_pose_match_error: float | None

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        *,
        mode: str = "affine_validated",
        allow_provisional_range_mapping: bool = False,
        allow_provisional_nominal_reference: bool = False,
        min_abs_scale: float = 1.0e-12,
    ) -> "CylinderCoordinateMapping":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("version") != "cylinder_coordinate_mapping_v1":
            raise ValueError(f"Unsupported cylinder coordinate mapping version: {data.get('version')!r}")
        if mode not in {"nominal_sign_offset", "affine_validated", "provisional", "validated"}:
            raise ValueError("Unsupported coordinate mapping mode.")
        if mode == "validated":
            mode = "affine_validated"
        source = str(data.get("source", "unresolved"))
        mapping_validated = bool(data.get("mapping_validated", False))
        reference_sample_count: int | None = None
        reference_pose_match_error: float | None = None
        if mode == "affine_validated" and not mapping_validated:
            raise ValueError("Validated coordinate mapping requested, but mapping_validated is false.")
        if mode == "provisional" and not allow_provisional_range_mapping:
            raise ValueError("Provisional coordinate mapping requires --allow_provisional_range_mapping.")
        if mode == "provisional" and source != "provisional_range_mapping":
            raise ValueError(f"Provisional mode requires source='provisional_range_mapping', got {source!r}.")
        if mode == "nominal_sign_offset":
            if data.get("mapping_mode") != "nominal_sign_offset":
                raise ValueError("nominal_sign_offset mode requires mapping_mode='nominal_sign_offset' in mapping JSON.")
            if not data.get("nominal_reference_complete", False):
                raise ValueError("nominal_sign_offset mode requires nominal_reference_complete=true.")
            if not all(bool(data.get("units", {}).get(key) == "m") for key in ("simscape_extension", "isaac_position")):
                raise ValueError("nominal_sign_offset mode requires metre units for Simscape extension and Isaac position.")

        cylinders = data.get("cylinders", {})
        scale: list[float] = [0.0] * 4
        offset: list[float] = [0.0] * 4
        sign: list[float] = [0.0] * 4
        simscape_nominal: list[float] = [0.0] * 4
        isaac_nominal: list[float] = [0.0] * 4
        isaac_min: list[float] = [0.0] * 4
        isaac_max: list[float] = [0.0] * 4
        sign_verified: list[bool] = [False] * 4
        confidence_values: list[str] = []
        reference_source_path: str | None = None
        for expected_index, name in enumerate(CYLINDER_ORDER):
            entry = cylinders.get(name)
            if not isinstance(entry, dict):
                raise ValueError(f"Missing mapping entry for {name}.")
            rom_index = int(entry.get("rom_index", -1))
            if rom_index != expected_index:
                raise ValueError(f"Mapping entry {name} has rom_index={rom_index}, expected {expected_index}.")
            verified = bool(entry.get("sign_verified", entry.get("direction_verified", False)))
            if mode == "nominal_sign_offset":
                raw_sign = entry.get("sign")
                raw_simscape_nominal = entry.get("simscape_nominal_extension")
                raw_isaac_nominal = entry.get("isaac_nominal_position")
                if raw_sign not in (-1, 1, -1.0, 1.0):
                    raise ValueError(f"Mapping entry {name} is missing a verified +/-1 sign.")
                if raw_simscape_nominal is None:
                    raise ValueError(f"Mapping entry {name} is missing simscape_nominal_extension.")
                if raw_isaac_nominal is None:
                    raise ValueError(f"Mapping entry {name} is missing isaac_nominal_position.")
                if not verified:
                    raise ValueError(f"nominal_sign_offset mode requires sign_verified for {name}.")
                confidence = str(entry.get("nominal_reference_confidence", "") or "UNKNOWN")
                confidence_values.append(confidence)
                reference_source_path = reference_source_path or entry.get("nominal_reference_source")
                value_sign = float(raw_sign)
                value_simscape_nominal = float(raw_simscape_nominal)
                value_isaac_nominal = float(raw_isaac_nominal)
                if not torch.isfinite(torch.tensor([value_sign, value_simscape_nominal, value_isaac_nominal])).all():
                    raise ValueError(f"Mapping entry {name} has non-finite nominal-sign values.")
                sign[expected_index] = value_sign
                simscape_nominal[expected_index] = value_simscape_nominal
                isaac_nominal[expected_index] = value_isaac_nominal
                scale[expected_index] = value_sign
                offset[expected_index] = value_isaac_nominal - value_sign * value_simscape_nominal
            else:
                raw_scale = entry.get("scale")
                raw_offset = entry.get("offset")
                if raw_scale is None or raw_offset is None:
                    raise ValueError(f"Mapping entry {name} is missing scale/offset.")
                value_scale = float(raw_scale)
                value_offset = float(raw_offset)
                if not torch.isfinite(torch.tensor([value_scale, value_offset])).all():
                    raise ValueError(f"Mapping entry {name} has non-finite scale/offset.")
                if abs(value_scale) <= min_abs_scale:
                    raise ValueError(f"Mapping entry {name} has zero or near-zero scale.")
                if mode == "affine_validated" and not verified:
                    raise ValueError(f"Validated coordinate mapping requires sign_verified/direction_verified for {name}.")
                scale[expected_index] = value_scale
                offset[expected_index] = value_offset
                sign[expected_index] = 1.0 if value_scale > 0.0 else -1.0
                simscape_nominal[expected_index] = 0.0
                isaac_nominal[expected_index] = value_offset
            isaac_min[expected_index] = float(entry.get("isaac_min"))
            isaac_max[expected_index] = float(entry.get("isaac_max"))
            sign_verified[expected_index] = verified
        nominal_reference_confidence = "UNKNOWN"
        if confidence_values:
            nominal_reference_confidence = "WARN" if any(value.upper() == "WARN" for value in confidence_values) else confidence_values[0]
        if mode == "nominal_sign_offset":
            if reference_source_path is not None:
                reference_path = Path(reference_source_path)
                if not reference_path.is_absolute():
                    reference_path = Path(path).resolve().parents[2] / reference_path
                if reference_path.exists():
                    reference_data = json.loads(reference_path.read_text(encoding="utf-8"))
                    reference_sample_count = int(reference_data["sample_count"]) if "sample_count" in reference_data else None
                    reference_pose_match_error = (
                        float(reference_data["pose_match_error"]) if "pose_match_error" in reference_data else None
                    )
                    nominal_reference_confidence = str(reference_data.get("confidence", nominal_reference_confidence))
            if (nominal_reference_confidence.upper() == "WARN" or not mapping_validated) and not allow_provisional_nominal_reference:
                raise ValueError(
                    "nominal_sign_offset reference is WARN or nonvalidated; rerun with "
                    "--allow_provisional_nominal_reference for the manual diagnostic only."
                )
        return cls(
            mapping_mode=mode,
            scale=torch.tensor(scale, dtype=torch.float64),
            offset=torch.tensor(offset, dtype=torch.float64),
            sign=torch.tensor(sign, dtype=torch.float64),
            simscape_nominal_extension=torch.tensor(simscape_nominal, dtype=torch.float64),
            isaac_nominal_position=torch.tensor(isaac_nominal, dtype=torch.float64),
            isaac_min=torch.tensor(isaac_min, dtype=torch.float64),
            isaac_max=torch.tensor(isaac_max, dtype=torch.float64),
            sign_verified=tuple(sign_verified),  # type: ignore[arg-type]
            mapping_validated=mapping_validated,
            source=source,
            nominal_reference_confidence=nominal_reference_confidence,
            reference_sample_count=reference_sample_count,
            reference_pose_match_error=reference_pose_match_error,
        )

    @classmethod
    def from_values(
        cls,
        *,
        scale: list[float] | tuple[float, ...],
        offset: list[float] | tuple[float, ...],
        isaac_min: list[float] | tuple[float, ...] | None = None,
        isaac_max: list[float] | tuple[float, ...] | None = None,
        direction_verified: tuple[bool, bool, bool, bool] = (True, True, True, True),
        mapping_validated: bool = True,
        source: str = "synthetic_test",
    ) -> "CylinderCoordinateMapping":
        if len(scale) != 4 or len(offset) != 4:
            raise ValueError("scale and offset must contain exactly four cylinder values.")
        mapping = cls(
            mapping_mode="affine_validated" if mapping_validated else "provisional",
            scale=torch.tensor(scale, dtype=torch.float64),
            offset=torch.tensor(offset, dtype=torch.float64),
            sign=torch.sign(torch.tensor(scale, dtype=torch.float64)),
            simscape_nominal_extension=torch.zeros(4, dtype=torch.float64),
            isaac_nominal_position=torch.tensor(offset, dtype=torch.float64),
            isaac_min=torch.tensor(isaac_min or [-float("inf")] * 4, dtype=torch.float64),
            isaac_max=torch.tensor(isaac_max or [float("inf")] * 4, dtype=torch.float64),
            sign_verified=direction_verified,
            mapping_validated=mapping_validated,
            source=source,
            nominal_reference_confidence="PASS" if mapping_validated else "UNKNOWN",
            reference_sample_count=None,
            reference_pose_match_error=None,
        )
        mapping._validate_static()
        return mapping

    @classmethod
    def from_nominal_sign_values(
        cls,
        *,
        sign: list[int] | tuple[int, ...],
        simscape_nominal_extension: list[float] | tuple[float, ...],
        isaac_nominal_position: list[float] | tuple[float, ...],
        isaac_min: list[float] | tuple[float, ...] | None = None,
        isaac_max: list[float] | tuple[float, ...] | None = None,
        sign_verified: tuple[bool, bool, bool, bool] = (True, True, True, True),
        mapping_validated: bool = False,
        source: str = "synthetic_test",
    ) -> "CylinderCoordinateMapping":
        if len(sign) != 4 or len(simscape_nominal_extension) != 4 or len(isaac_nominal_position) != 4:
            raise ValueError("sign, simscape_nominal_extension, and isaac_nominal_position must contain four values.")
        sign_tensor = torch.tensor(sign, dtype=torch.float64)
        if not torch.all((sign_tensor == 1.0) | (sign_tensor == -1.0)):
            raise ValueError("sign must be +/-1 for all cylinders.")
        x_nom = torch.tensor(simscape_nominal_extension, dtype=torch.float64)
        q_nom = torch.tensor(isaac_nominal_position, dtype=torch.float64)
        if not torch.isfinite(x_nom).all() or not torch.isfinite(q_nom).all():
            raise ValueError("nominal values must be finite.")
        mapping = cls(
            mapping_mode="nominal_sign_offset",
            scale=sign_tensor.clone(),
            offset=q_nom - sign_tensor * x_nom,
            sign=sign_tensor,
            simscape_nominal_extension=x_nom,
            isaac_nominal_position=q_nom,
            isaac_min=torch.tensor(isaac_min or [-float("inf")] * 4, dtype=torch.float64),
            isaac_max=torch.tensor(isaac_max or [float("inf")] * 4, dtype=torch.float64),
            sign_verified=sign_verified,
            mapping_validated=mapping_validated,
            source=source,
            nominal_reference_confidence="PASS" if mapping_validated else "WARN",
            reference_sample_count=None,
            reference_pose_match_error=None,
        )
        mapping._validate_static()
        return mapping

    def _validate_static(self) -> None:
        for name, value in (
            ("scale", self.scale),
            ("offset", self.offset),
            ("sign", self.sign),
            ("simscape_nominal_extension", self.simscape_nominal_extension),
            ("isaac_nominal_position", self.isaac_nominal_position),
        ):
            if value.shape != (4,):
                raise ValueError(f"{name} tensor must have shape [4].")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} must be finite.")
        if torch.any(torch.abs(self.scale) <= 1.0e-12):
            raise ValueError("scale must be nonzero for all cylinders.")
        if self.mapping_mode == "nominal_sign_offset" and not torch.all((self.sign == 1.0) | (self.sign == -1.0)):
            raise ValueError("nominal_sign_offset requires sign values of +/-1.")

    def _as_tensor(self, value: torch.Tensor, *, name: str) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor.")
        if value.shape[-1] != 4:
            raise ValueError(f"{name} must have four cylinders in the final dimension, got {tuple(value.shape)}.")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} contains NaN/Inf.")
        return value

    def _scale_like(self, value: torch.Tensor) -> torch.Tensor:
        return self.scale.to(device=value.device, dtype=value.dtype)

    def _offset_like(self, value: torch.Tensor) -> torch.Tensor:
        return self.offset.to(device=value.device, dtype=value.dtype)

    def _sign_like(self, value: torch.Tensor) -> torch.Tensor:
        return self.sign.to(device=value.device, dtype=value.dtype)

    def _simscape_nominal_like(self, value: torch.Tensor) -> torch.Tensor:
        return self.simscape_nominal_extension.to(device=value.device, dtype=value.dtype)

    def _isaac_nominal_like(self, value: torch.Tensor) -> torch.Tensor:
        return self.isaac_nominal_position.to(device=value.device, dtype=value.dtype)

    def scale_tensor(self, like: torch.Tensor | None = None) -> torch.Tensor:
        return self.scale.clone() if like is None else self._scale_like(like)

    def offset_tensor(self, like: torch.Tensor | None = None) -> torch.Tensor:
        return self.offset.clone() if like is None else self._offset_like(like)

    def sign_tensor(self, like: torch.Tensor | None = None) -> torch.Tensor:
        return self.sign.clone() if like is None else self._sign_like(like)

    def simscape_nominal_tensor(self, like: torch.Tensor | None = None) -> torch.Tensor:
        return self.simscape_nominal_extension.clone() if like is None else self._simscape_nominal_like(like)

    def isaac_nominal_tensor(self, like: torch.Tensor | None = None) -> torch.Tensor:
        return self.isaac_nominal_position.clone() if like is None else self._isaac_nominal_like(like)

    def isaac_position_to_simscape_extension(self, q_isaac: torch.Tensor) -> torch.Tensor:
        q = self._as_tensor(q_isaac, name="q_isaac")
        if self.mapping_mode == "nominal_sign_offset":
            return self._simscape_nominal_like(q) + self._sign_like(q) * (q - self._isaac_nominal_like(q))
        return (q - self._offset_like(q)) / self._scale_like(q)

    def isaac_velocity_to_simscape_velocity(self, qdot_isaac: torch.Tensor) -> torch.Tensor:
        qdot = self._as_tensor(qdot_isaac, name="qdot_isaac")
        if self.mapping_mode == "nominal_sign_offset":
            return self._sign_like(qdot) * qdot
        return qdot / self._scale_like(qdot)

    def simscape_extension_to_isaac_position(self, x_simscape: torch.Tensor) -> torch.Tensor:
        x = self._as_tensor(x_simscape, name="x_simscape")
        if self.mapping_mode == "nominal_sign_offset":
            return self._isaac_nominal_like(x) + self._sign_like(x) * (x - self._simscape_nominal_like(x))
        return self._scale_like(x) * x + self._offset_like(x)

    def simscape_velocity_to_isaac_velocity(self, xdot_simscape: torch.Tensor) -> torch.Tensor:
        xdot = self._as_tensor(xdot_simscape, name="xdot_simscape")
        if self.mapping_mode == "nominal_sign_offset":
            return self._sign_like(xdot) * xdot
        return self._scale_like(xdot) * xdot

    def simscape_delta_to_isaac_delta(self, delta_x_simscape: torch.Tensor) -> torch.Tensor:
        delta_x = self._as_tensor(delta_x_simscape, name="delta_x_simscape")
        if self.mapping_mode == "nominal_sign_offset":
            return self._sign_like(delta_x) * delta_x
        return self._scale_like(delta_x) * delta_x

    def isaac_delta_to_simscape_delta(self, delta_q_isaac: torch.Tensor) -> torch.Tensor:
        delta_q = self._as_tensor(delta_q_isaac, name="delta_q_isaac")
        if self.mapping_mode == "nominal_sign_offset":
            return self._sign_like(delta_q) * delta_q
        return delta_q / self._scale_like(delta_q)

    def inverse_mapping_error(self, q_isaac: torch.Tensor) -> torch.Tensor:
        q = self._as_tensor(q_isaac, name="q_isaac")
        reconstructed = self.simscape_extension_to_isaac_position(self.isaac_position_to_simscape_extension(q))
        return torch.abs(reconstructed - q)

    def isaac_limit_violation(self, q_isaac: torch.Tensor) -> torch.Tensor:
        q = self._as_tensor(q_isaac, name="q_isaac")
        lower = self.isaac_min.to(device=q.device, dtype=q.dtype)
        upper = self.isaac_max.to(device=q.device, dtype=q.dtype)
        return (q < lower) | (q > upper)


def load_coordinate_mapping(
    path: str | Path,
    *,
    mode: str,
    allow_provisional_range_mapping: bool = False,
    allow_provisional_nominal_reference: bool = False,
) -> CylinderCoordinateMapping:
    return CylinderCoordinateMapping.from_json(
        path,
        mode=mode,
        allow_provisional_range_mapping=allow_provisional_range_mapping,
        allow_provisional_nominal_reference=allow_provisional_nominal_reference,
    )


def mapping_metadata_for_logging(mapping: CylinderCoordinateMapping | None, like: torch.Tensor) -> dict[str, Any]:
    if mapping is None:
        nan = float("nan")
        return {
            "scale": [nan, nan, nan, nan],
            "offset": [nan, nan, nan, nan],
            "sign": [nan, nan, nan, nan],
            "simscape_nominal_extension": [nan, nan, nan, nan],
            "isaac_nominal_position": [nan, nan, nan, nan],
            "sign_verified": [False, False, False, False],
            "mapping_validated": False,
            "mapping_mode": "none",
            "source": "none",
            "nominal_reference_confidence": "",
            "reference_sample_count": "",
            "reference_pose_match_error": "",
        }
    return {
        "scale": [float(v) for v in mapping.scale_tensor(like).detach().cpu().reshape(-1).tolist()],
        "offset": [float(v) for v in mapping.offset_tensor(like).detach().cpu().reshape(-1).tolist()],
        "sign": [float(v) for v in mapping.sign_tensor(like).detach().cpu().reshape(-1).tolist()],
        "simscape_nominal_extension": [
            float(v) for v in mapping.simscape_nominal_tensor(like).detach().cpu().reshape(-1).tolist()
        ],
        "isaac_nominal_position": [float(v) for v in mapping.isaac_nominal_tensor(like).detach().cpu().reshape(-1).tolist()],
        "sign_verified": list(mapping.sign_verified),
        "mapping_validated": bool(mapping.mapping_validated),
        "mapping_mode": mapping.mapping_mode,
        "source": mapping.source,
        "nominal_reference_confidence": mapping.nominal_reference_confidence,
        "reference_sample_count": mapping.reference_sample_count,
        "reference_pose_match_error": mapping.reference_pose_match_error,
    }
