"""PPO3-1 environment: the shared verge task with c4 authority asserted.

The runtime behaviour is entirely inherited from PPO3's
``TerrainSolePolicyEnv`` -- same sensors, observation, rewards, safety logic,
terrain, path families and ROM-A actuator. This subclass exists to (a) give
the new gym task a distinct entry point and (b) fail loudly if the per-axis
c4 authority is not actually active, so a misconfigured run cannot silently
train the plain PPO3 task under a PPO3-1 label.
"""

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
        if per_axis_lead[3] <= per_axis_lead[0]:
            raise RuntimeError(
                "PPO3-1 requires c4 lead strictly larger than c1-c3 lead; got "
                f"{per_axis_lead}"
            )
        if per_axis_scale[3] <= per_axis_scale[0]:
            raise RuntimeError(
                "PPO3-1 requires c4 action scale strictly larger than c1-c3; "
                f"got {per_axis_scale}"
            )
        self._per_axis_lead = per_axis_lead
        self._per_axis_scale = per_axis_scale

        # The adapter reads the per-axis scale at construction. Verify the
        # active adapter actually saw the c4 boost rather than silently falling
        # back to the scalar.
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

        print(
            "[PPO3-1-SOLE] c4 authority active: "
            f"lead={per_axis_lead} m  action_scale={per_axis_scale} m/unit  "
            f"sole_controller=TRUE  actuator=ROM-A",
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
        }
