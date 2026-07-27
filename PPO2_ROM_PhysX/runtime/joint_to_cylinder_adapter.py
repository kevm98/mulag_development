from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from PPO2_ROM_PhysX.runtime.cylinder_space_scaling import CylinderSpaceScaling
from PPO2_ROM_PhysX.runtime.cylinder_coordinate_adapter import CylinderCoordinateMapping


@dataclass(frozen=True)
class CylinderReference:
    x_current: torch.Tensor
    xdot_current: torch.Tensor
    x_ref: torch.Tensor
    xdot_ref: torch.Tensor
    target_delta_x: torch.Tensor
    target_delta_xdot: torch.Tensor


@dataclass(frozen=True)
class PPOActionCylinderTarget:
    raw_action: torch.Tensor
    clipped_action: torch.Tensor
    hydraulic_action: torch.Tensor
    ignored_passive_action: torch.Tensor | None
    delta_q_usd_ref: torch.Tensor
    target_delta_x_pre_gain: torch.Tensor
    target_delta_x_post_gain: torch.Tensor
    target_delta_x_safety_limit: torch.Tensor
    target_delta_x_pre_gain_clip_mask: torch.Tensor
    target_delta_x_post_gain_clip_mask: torch.Tensor
    target_delta_xdot_raw: torch.Tensor
    target_delta_xdot_clip_mask: torch.Tensor
    reference: CylinderReference


class PPOActionToCylinderTargetAdapter:
    """Convert PPO action increments into learned cylinder-space ID targets.

    PPO1 still exposes a 5-D arm action for compatibility.  Only the first four
    entries correspond to the active hydraulic cylinders used by ID/FD.  The
    optional fifth/passive action is reported for diagnostics but is not passed
    into the learned hydraulic pipeline.
    """

    def __init__(
        self,
        *,
        scaling: CylinderSpaceScaling | None,
        mapping: CylinderCoordinateMapping | None = None,
        action_scale: float,
        learned_model_dt: float | None = None,
        control_dt: float | None = None,
        action_clip: float = 1.0,
        target_delta_xdot_clip_min: torch.Tensor | None = None,
        target_delta_xdot_clip_max: torch.Tensor | None = None,
        target_dxdot_mode: str = "cmd_minus_xdot",
        target_dxdot_gain: float = 1.0,
        target_dx_gain: float = 1.0,
    ) -> None:
        self.scaling = scaling
        self.mapping = mapping
        self.action_scale = float(action_scale)
        dt = learned_model_dt if learned_model_dt is not None else control_dt
        if dt is None:
            raise ValueError("PPOActionToCylinderTargetAdapter requires learned_model_dt or control_dt.")
        self.learned_model_dt = max(float(dt), 1.0e-9)
        self.action_clip = abs(float(action_clip))
        self.target_delta_xdot_clip_min = None if target_delta_xdot_clip_min is None else target_delta_xdot_clip_min.detach().clone()
        self.target_delta_xdot_clip_max = None if target_delta_xdot_clip_max is None else target_delta_xdot_clip_max.detach().clone()
        self.target_dxdot_mode = str(target_dxdot_mode)
        self.target_dxdot_gain = float(target_dxdot_gain)
        self.target_dx_gain = float(target_dx_gain)

    def _usd_delta_to_learned_delta(self, delta_q_usd: torch.Tensor) -> torch.Tensor:
        if self.scaling is not None:
            return self.scaling.usd_delta_to_learned_delta(delta_q_usd).float()
        if self.mapping is not None:
            return self.mapping.isaac_delta_to_simscape_delta(delta_q_usd).float()
        return delta_q_usd

    def _learned_delta_safety_limit_like(self, target_delta_x: torch.Tensor) -> torch.Tensor:
        usd_limit = torch.full_like(target_delta_x, self.action_scale)
        learned_limit = self._usd_delta_to_learned_delta(usd_limit).abs()
        return torch.clamp(learned_limit, min=1.0e-12)

    def convert(
        self,
        *,
        action: torch.Tensor,
        x_current: torch.Tensor,
        xdot_current: torch.Tensor,
    ) -> PPOActionCylinderTarget:
        if action.shape[-1] < 4:
            raise ValueError(f"PPO action must contain at least four hydraulic entries, got {tuple(action.shape)}.")
        if action.shape[-1] > 5:
            raise ValueError(f"PPO action must be 4-D or 5-D for this adapter, got {tuple(action.shape)}.")
        JointToCylinderAdapter._check_width("x_current", x_current, 4)
        JointToCylinderAdapter._check_width("xdot_current", xdot_current, 4)

        raw_action = torch.nan_to_num(action.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
        clipped_action = torch.clamp(raw_action, -self.action_clip, self.action_clip)
        hydraulic_action = clipped_action[..., :4]
        ignored_passive_action = clipped_action[..., 4:5] if clipped_action.shape[-1] == 5 else None
        delta_q_usd_ref = hydraulic_action * self.action_scale

        target_delta_x_pre_gain = self._usd_delta_to_learned_delta(delta_q_usd_ref)
        target_delta_x_safety_limit = self._learned_delta_safety_limit_like(target_delta_x_pre_gain)
        target_delta_x_pre_gain_clip_mask = (
            target_delta_x_pre_gain.abs() >= target_delta_x_safety_limit * (1.0 - 1.0e-6)
        )
        target_delta_x_post_gain = target_delta_x_pre_gain * self.target_dx_gain
        target_delta_x = torch.clamp(
            target_delta_x_post_gain,
            -target_delta_x_safety_limit,
            target_delta_x_safety_limit,
        )
        target_delta_x_post_gain_clip_mask = target_delta_x != target_delta_x_post_gain

        desired_next_xdot = target_delta_x / self.learned_model_dt
        desired_next_xdot = desired_next_xdot * self.target_dxdot_gain
        mode = self.target_dxdot_mode
        if mode == "cmd_only":
            target_delta_xdot_raw = desired_next_xdot
        elif mode == "cmd_minus_xdot_clipped":
            v_cmd = desired_next_xdot
            v_fb = torch.clamp(xdot_current, -0.5 * v_cmd.abs(), 0.5 * v_cmd.abs())
            target_delta_xdot_raw = v_cmd - v_fb
        else:  # "cmd_minus_xdot" (old behavior)
            target_delta_xdot_raw = desired_next_xdot - xdot_current
        target_delta_xdot = target_delta_xdot_raw
        clip_mask = torch.zeros_like(target_delta_xdot_raw, dtype=torch.bool)
        if self.target_delta_xdot_clip_min is not None and self.target_delta_xdot_clip_max is not None:
            lo = self.target_delta_xdot_clip_min.to(device=target_delta_xdot_raw.device, dtype=target_delta_xdot_raw.dtype)
            hi = self.target_delta_xdot_clip_max.to(device=target_delta_xdot_raw.device, dtype=target_delta_xdot_raw.dtype)
            target_delta_xdot = torch.clamp(target_delta_xdot_raw, lo, hi)
            clip_mask = target_delta_xdot != target_delta_xdot_raw
        reference = CylinderReference(
            x_current=x_current,
            xdot_current=xdot_current,
            x_ref=x_current + target_delta_x,
            xdot_ref=desired_next_xdot,
            target_delta_x=target_delta_x,
            target_delta_xdot=target_delta_xdot,
        )
        return PPOActionCylinderTarget(
            raw_action=raw_action.detach().clone(),
            clipped_action=clipped_action.detach().clone(),
            hydraulic_action=hydraulic_action.detach().clone(),
            ignored_passive_action=None if ignored_passive_action is None else ignored_passive_action.detach().clone(),
            delta_q_usd_ref=delta_q_usd_ref.detach().clone(),
            target_delta_x_pre_gain=target_delta_x_pre_gain.detach().clone(),
            target_delta_x_post_gain=target_delta_x_post_gain.detach().clone(),
            target_delta_x_safety_limit=target_delta_x_safety_limit.detach().clone(),
            target_delta_x_pre_gain_clip_mask=target_delta_x_pre_gain_clip_mask.detach().clone(),
            target_delta_x_post_gain_clip_mask=target_delta_x_post_gain_clip_mask.detach().clone(),
            target_delta_xdot_raw=target_delta_xdot_raw.detach().clone(),
            target_delta_xdot_clip_mask=clip_mask.detach().clone(),
            reference=reference,
        )


class JointToCylinderAdapter:
    """Adapter from high-level joint references to cylinder-space ID features.

    Runtime convention:
    - Direct joint references can still be converted for diagnostics/legacy use.
    - The live PPO1 -> ID path uses PPOActionToCylinderTargetAdapter so ID sees
      learned cylinder target deltas rather than raw actions or USD references.
    - The first four configured reference joints correspond to the four hydraulic
      cylinder command coordinates.
    - If a CylinderCoordinateMapping is provided, Isaac joint coordinates are
      converted into Simscape cylinder extension coordinates before ID.
    """

    def __init__(
        self,
        *,
        mapping: CylinderCoordinateMapping | None,
        scaling: CylinderSpaceScaling | None = None,
        joint_reference_indices: Sequence[int],
    ) -> None:
        if len(joint_reference_indices) != 4:
            raise ValueError(
                "joint_reference_indices must contain exactly four hydraulic joint indices, "
                f"got {len(joint_reference_indices)}."
            )
        self.mapping = mapping
        self.scaling = scaling
        self.joint_reference_indices = tuple(int(index) for index in joint_reference_indices)

    def current_from_cylinder_joints(
        self,
        cylinder_q: torch.Tensor,
        cylinder_qdot: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._check_width("cylinder_q", cylinder_q, 4)
        self._check_width("cylinder_qdot", cylinder_qdot, 4)
        if self.scaling is not None:
            return (
                self.scaling.usd_q_to_learned_x(cylinder_q).float(),
                self.scaling.usd_qdot_to_learned_xdot(cylinder_qdot).float(),
            )
        if self.mapping is None:
            return cylinder_q, cylinder_qdot
        return (
            self.mapping.isaac_position_to_simscape_extension(cylinder_q).float(),
            self.mapping.isaac_velocity_to_simscape_velocity(cylinder_qdot).float(),
        )

    def reference_from_joint_reference(
        self,
        q_ref: torch.Tensor,
        qdot_ref: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if q_ref.shape[-1] <= max(self.joint_reference_indices):
            raise ValueError(
                "q_ref does not contain all configured hydraulic reference indices: "
                f"shape={tuple(q_ref.shape)}, indices={self.joint_reference_indices}."
            )
        if qdot_ref.shape[-1] <= max(self.joint_reference_indices):
            raise ValueError(
                "qdot_ref does not contain all configured hydraulic reference indices: "
                f"shape={tuple(qdot_ref.shape)}, indices={self.joint_reference_indices}."
            )
        index = torch.as_tensor(self.joint_reference_indices, device=q_ref.device, dtype=torch.long)
        q_cyl_ref = q_ref.index_select(dim=-1, index=index)
        qdot_cyl_ref = qdot_ref.index_select(dim=-1, index=index)
        if self.scaling is not None:
            return (
                self.scaling.usd_q_to_learned_x(q_cyl_ref).float(),
                self.scaling.usd_qdot_to_learned_xdot(qdot_cyl_ref).float(),
            )
        if self.mapping is None:
            return q_cyl_ref, qdot_cyl_ref
        return (
            self.mapping.isaac_position_to_simscape_extension(q_cyl_ref).float(),
            self.mapping.isaac_velocity_to_simscape_velocity(qdot_cyl_ref).float(),
        )

    def convert(
        self,
        *,
        q_ref: torch.Tensor,
        qdot_ref: torch.Tensor,
        cylinder_q: torch.Tensor,
        cylinder_qdot: torch.Tensor,
    ) -> CylinderReference:
        x_current, xdot_current = self.current_from_cylinder_joints(cylinder_q, cylinder_qdot)
        x_ref, xdot_ref = self.reference_from_joint_reference(q_ref, qdot_ref)
        return CylinderReference(
            x_current=x_current,
            xdot_current=xdot_current,
            x_ref=x_ref,
            xdot_ref=xdot_ref,
            target_delta_x=x_ref - x_current,
            target_delta_xdot=xdot_ref - xdot_current,
        )

    def build_id_input(
        self,
        reference: CylinderReference,
        *,
        dP: torch.Tensor,
        force: torch.Tensor | None = None,
        prev_valve_cmd: torch.Tensor | None = None,
        input_dim: int = 24,
    ) -> torch.Tensor:
        self._check_width("dP", dP, 4)
        if force is None:
            force = torch.zeros_like(dP)
        self._check_width("force", force, 4)
        parts = [
            reference.x_current,
            reference.xdot_current,
            dP,
            force,
            reference.target_delta_x,
            reference.target_delta_xdot,
        ]
        if input_dim == 28:
            if prev_valve_cmd is None:
                prev_valve_cmd = torch.zeros_like(dP)
            self._check_width("prev_valve_cmd", prev_valve_cmd, 4)
            parts.append(prev_valve_cmd)
        elif input_dim != 24:
            raise ValueError(f"Cylinder ID input_dim must be 24 or 28, got {input_dim}.")
        out = torch.cat(parts, dim=-1)
        self._check_width("id_input", out, input_dim)
        if not torch.isfinite(out).all():
            raise FloatingPointError("Cylinder ID input contains NaN/Inf.")
        return out

    def simscape_delta_to_isaac_delta(self, delta_extension: torch.Tensor) -> torch.Tensor:
        self._check_width("delta_extension", delta_extension, 4)
        if self.scaling is not None:
            return self.scaling.learned_delta_to_usd_delta(delta_extension).float()
        if self.mapping is None:
            return delta_extension
        return self.mapping.simscape_delta_to_isaac_delta(delta_extension).float()

    @staticmethod
    def _check_width(name: str, tensor: torch.Tensor, expected: int) -> None:
        if tensor.shape[-1] != expected:
            raise ValueError(f"{name} must have final dimension {expected}, got {tuple(tensor.shape)}.")
        if not torch.isfinite(tensor).all():
            raise FloatingPointError(f"{name} contains NaN/Inf.")
