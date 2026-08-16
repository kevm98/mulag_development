"""Wide verge environment with validated per-cylinder authority."""

from __future__ import annotations

import torch

from verge_task.terrain_env import TerrainSolePolicyEnv


class TerrainC4AuthorityEnv(TerrainSolePolicyEnv):
    """Verge sole controller with the per-axis c4 authority experiment."""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)

        per_axis_lead = tuple(
            float(value)
            for value in getattr(
                cfg, "cylinder_target_max_lead_per_axis_m", ()
            )
        )
        per_axis_scale = tuple(
            float(value)
            for value in getattr(
                cfg, "cylinder_action_scale_per_axis_m", ()
            )
        )
        if len(per_axis_lead) != 4 or len(per_axis_scale) != 4:
            raise RuntimeError(
                "PPO3-1 requires both per-axis lead and per-axis action scale "
                f"(4 values each); got lead={per_axis_lead}, scale={per_axis_scale}"
            )
        c4_stroke_m = 0.170
        if per_axis_lead[3] >= c4_stroke_m:
            raise RuntimeError(
                "PPO3-1 requires a c4 lead smaller than c4's own stroke "
                f"({c4_stroke_m} m), or the anti-windup clamp can never bind; "
                f"got {per_axis_lead}"
            )
        c1_steps_to_cap = per_axis_lead[0] / max(per_axis_scale[0], 1.0e-9)
        c4_steps_to_cap = per_axis_lead[3] / max(per_axis_scale[3], 1.0e-9)
        if c4_steps_to_cap * 4.0 < c1_steps_to_cap:
            raise RuntimeError(
                "PPO3-1 requires c4 to reach its anti-windup cap no more than "
                f"4x faster than c1-c3; c4 takes {c4_steps_to_cap:.1f} steps "
                f"against {c1_steps_to_cap:.1f} for c1. Lower "
                "c4_action_scale_gain or raise cylinder_target_lead_c4_m; got "
                f"lead={per_axis_lead} scale={per_axis_scale}"
            )
        self._per_axis_lead = per_axis_lead
        self._per_axis_scale = per_axis_scale

        adapter_scale = getattr(
            getattr(self, "_action_to_cylinder_target_adapter", None),
            "_action_scale_row",
            None,
        )
        if adapter_scale is None:
            raise RuntimeError(
                "PPO3-1 adapter did not receive the per-axis action scale; "
                "cylinder_action_scale_per_axis_m was not forwarded"
            )
        adapter_row = adapter_scale.detach().cpu().reshape(-1).tolist()
        for expected, actual in zip(per_axis_scale, adapter_row):
            if abs(float(expected) - float(actual)) > 1.0e-9:
                raise RuntimeError(
                    f"PPO3-1 adapter action scale {adapter_row} does not match "
                    f"config {per_axis_scale}"
                )

        dynamics_model = str(
            getattr(self.cfg, "cylinder_dynamics_model", "rom")
        ).lower()
        if dynamics_model == "fd":
            actuator_label = "ID -> FD"
        elif dynamics_model == "rom":
            actuator_label = "ROM-A"
        else:
            actuator_label = dynamics_model or "unknown"

        print(
            "[PPO3-1-SOLE] c4 authority active: "
            f"lead={per_axis_lead} m  action_scale={per_axis_scale} m/unit  "
            f"sole_controller=TRUE  actuator={actuator_label}",
            flush=True,
        )

    @torch.no_grad()
    def c4_authority_report(self) -> dict:
        """Machine-readable summary of the active c4 authority."""
        return {
            "cylinder_target_max_lead_per_axis_m": list(self._per_axis_lead),
            "cylinder_action_scale_per_axis_m": list(self._per_axis_scale),
            "action_space": int(self.cfg.action_space),
            "head_joint_action": bool(self.cfg.sole_head_joint_action),
            "teacher_mode": str(self.cfg.sole_teacher_mode),
            "actuator": (
                "idfd"
                if str(getattr(self.cfg, "cylinder_dynamics_model", "rom")).lower()
                == "fd"
                else "rom"
            ),
        }
