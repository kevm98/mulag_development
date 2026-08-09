from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from PPO1_ID_FD_PhysX.runtime.cylinder_coordinate_adapter import CylinderCoordinateMapping


@dataclass(frozen=True)
class CylinderSpaceScaling:
    """Affine boundary adapter between learned cylinder space and USD joint space.

    Learned model convention:
        x_learned, xdot_learned, delta_x_learned are Simscape cylinder
        extension coordinates in metres.

    Isaac/PhysX convention:
        q_usd, qdot_usd, delta_q_usd are USD prismatic joint coordinates in
        metres.

    Mapping:
        q_usd = scale * x_learned + offset
    """

    cylinder_names: tuple[str, ...]
    dof_indices: torch.Tensor
    scale: torch.Tensor
    offset: torch.Tensor
    learned_at_usd_min: torch.Tensor
    learned_at_usd_max: torch.Tensor
    learned_min: torch.Tensor
    learned_max: torch.Tensor
    usd_min: torch.Tensor
    usd_max: torch.Tensor

    @classmethod
    def from_coordinate_mapping(
        cls,
        *,
        mapping: CylinderCoordinateMapping,
        cylinder_names: Sequence[str],
        dof_indices: torch.Tensor,
        usd_limits: torch.Tensor,
    ) -> "CylinderSpaceScaling":
        names = tuple(str(name) for name in cylinder_names)
        if len(names) != 4:
            raise ValueError(f"CylinderSpaceScaling requires exactly four cylinders, got {len(names)}.")
        if dof_indices.shape[-1] != 4:
            raise ValueError(f"dof_indices must contain four DOF ids, got {tuple(dof_indices.shape)}.")
        if usd_limits.shape[-2:] != (4, 2):
            raise ValueError(f"usd_limits must have shape (..., 4, 2), got {tuple(usd_limits.shape)}.")

        usd_min = usd_limits[..., 0].reshape(-1, 4)[0].detach().to(device="cpu", dtype=torch.float64)
        usd_max = usd_limits[..., 1].reshape(-1, 4)[0].detach().to(device="cpu", dtype=torch.float64)
        if not torch.isfinite(usd_min).all() or not torch.isfinite(usd_max).all():
            raise ValueError("USD cylinder limits must be finite.")
        if torch.any(usd_min >= usd_max):
            raise ValueError(f"USD cylinder limits must satisfy min < max, got {usd_limits}.")

        scale = mapping.scale_tensor().detach().to(device="cpu", dtype=torch.float64)
        if scale.shape != (4,) or torch.any(torch.abs(scale) <= 1.0e-12):
            raise ValueError(f"Cylinder scale must have four nonzero values, got {scale}.")

        learned_at_usd_min = mapping.isaac_position_to_simscape_extension(usd_min.reshape(1, 4)).reshape(4)
        learned_at_usd_max = mapping.isaac_position_to_simscape_extension(usd_max.reshape(1, 4)).reshape(4)
        learned_at_usd_min = learned_at_usd_min.detach().to(device="cpu", dtype=torch.float64)
        learned_at_usd_max = learned_at_usd_max.detach().to(device="cpu", dtype=torch.float64)

        # Required construction: offset = q_usd_min - scale * x_learned_at_usd_min.
        offset = usd_min - scale * learned_at_usd_min
        learned_min = torch.minimum(learned_at_usd_min, learned_at_usd_max)
        learned_max = torch.maximum(learned_at_usd_min, learned_at_usd_max)
        out = cls(
            cylinder_names=names,
            dof_indices=dof_indices.detach().to(dtype=torch.long).cpu(),
            scale=scale.cpu(),
            offset=offset.cpu(),
            learned_at_usd_min=learned_at_usd_min.cpu(),
            learned_at_usd_max=learned_at_usd_max.cpu(),
            learned_min=learned_min.cpu(),
            learned_max=learned_max.cpu(),
            usd_min=usd_min.cpu(),
            usd_max=usd_max.cpu(),
        )
        out.validate()
        return out

    @classmethod
    def identity(
        cls,
        *,
        cylinder_names: Sequence[str],
        dof_indices: torch.Tensor,
        usd_limits: torch.Tensor,
    ) -> "CylinderSpaceScaling":
        usd_min = usd_limits[..., 0].reshape(-1, 4)[0].detach().to(device="cpu", dtype=torch.float64)
        usd_max = usd_limits[..., 1].reshape(-1, 4)[0].detach().to(device="cpu", dtype=torch.float64)
        return cls(
            cylinder_names=tuple(str(name) for name in cylinder_names),
            dof_indices=dof_indices.detach().to(dtype=torch.long).cpu(),
            scale=torch.ones(4, dtype=torch.float64),
            offset=torch.zeros(4, dtype=torch.float64),
            learned_at_usd_min=usd_min.cpu(),
            learned_at_usd_max=usd_max.cpu(),
            learned_min=usd_min.cpu(),
            learned_max=usd_max.cpu(),
            usd_min=usd_min.cpu(),
            usd_max=usd_max.cpu(),
        )

    def validate(self) -> None:
        for name, value in (
            ("scale", self.scale),
            ("offset", self.offset),
            ("learned_at_usd_min", self.learned_at_usd_min),
            ("learned_at_usd_max", self.learned_at_usd_max),
            ("learned_min", self.learned_min),
            ("learned_max", self.learned_max),
            ("usd_min", self.usd_min),
            ("usd_max", self.usd_max),
        ):
            if value.shape != (4,):
                raise ValueError(f"{name} must have shape (4,), got {tuple(value.shape)}.")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} contains NaN/Inf.")
        if torch.any(torch.abs(self.scale) <= 1.0e-12):
            raise ValueError("Cylinder scale contains zero or near-zero values.")
        if torch.any(self.usd_min >= self.usd_max):
            raise ValueError("USD cylinder limits must satisfy min < max.")
        if torch.any(self.learned_min >= self.learned_max):
            raise ValueError("Learned cylinder limits must satisfy min < max.")

    def _as_tensor(self, value: torch.Tensor, *, name: str) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor.")
        if value.shape[-1] != 4:
            raise ValueError(f"{name} must have final dimension 4, got {tuple(value.shape)}.")
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"{name} contains NaN/Inf.")
        return value

    def _scale_like(self, value: torch.Tensor) -> torch.Tensor:
        return self.scale.to(device=value.device, dtype=value.dtype)

    def _offset_like(self, value: torch.Tensor) -> torch.Tensor:
        return self.offset.to(device=value.device, dtype=value.dtype)

    def _usd_min_like(self, value: torch.Tensor) -> torch.Tensor:
        return self.usd_min.to(device=value.device, dtype=value.dtype)

    def _usd_max_like(self, value: torch.Tensor) -> torch.Tensor:
        return self.usd_max.to(device=value.device, dtype=value.dtype)

    def usd_q_to_learned_x(self, q_usd: torch.Tensor) -> torch.Tensor:
        q = self._as_tensor(q_usd, name="q_usd")
        return (q - self._offset_like(q)) / self._scale_like(q)

    def usd_qdot_to_learned_xdot(self, qdot_usd: torch.Tensor) -> torch.Tensor:
        qdot = self._as_tensor(qdot_usd, name="qdot_usd")
        return qdot / self._scale_like(qdot)

    def usd_delta_to_learned_delta(self, delta_q_usd: torch.Tensor) -> torch.Tensor:
        delta = self._as_tensor(delta_q_usd, name="delta_q_usd")
        return delta / self._scale_like(delta)

    def learned_delta_to_usd_delta(self, delta_x: torch.Tensor) -> torch.Tensor:
        delta = self._as_tensor(delta_x, name="delta_x")
        return self._scale_like(delta) * delta

    def learned_velocity_to_usd_delta(self, v_learned: torch.Tensor, dt: float) -> torch.Tensor:
        vel = self._as_tensor(v_learned, name="v_learned")
        if dt < 0.0:
            raise ValueError(f"dt must be non-negative, got {dt}.")
        return self._scale_like(vel) * vel * float(dt)

    def learned_x_to_usd_q(self, x_learned: torch.Tensor) -> torch.Tensor:
        x = self._as_tensor(x_learned, name="x_learned")
        return self._scale_like(x) * x + self._offset_like(x)

    def clamp_usd_q(self, q_usd: torch.Tensor) -> torch.Tensor:
        q = self._as_tensor(q_usd, name="q_usd")
        return torch.clamp(q, self._usd_min_like(q), self._usd_max_like(q))

    def summary_lines(self) -> list[str]:
        lines: list[str] = []
        for index, name in enumerate(self.cylinder_names):
            lines.append(
                f"{name} dof={int(self.dof_indices[index])} "
                f"scale={float(self.scale[index]):.9g} "
                f"offset={float(self.offset[index]):.9g} "
                f"learned_at_usd_limits=[{float(self.learned_at_usd_min[index]):.9g}, "
                f"{float(self.learned_at_usd_max[index]):.9g}] m "
                f"learned_limits=[{float(self.learned_min[index]):.9g}, "
                f"{float(self.learned_max[index]):.9g}] m "
                f"usd_limits=[{float(self.usd_min[index]):.9g}, "
                f"{float(self.usd_max[index]):.9g}] m"
            )
        return lines
