"""PPO4_1 environment: PPO3_1's task and reward, ID/FD actuator.

Deliberately thin, for the same reason `PPO4_ID_FD_PhysX/terrain_env.py` is:
every physical and task behaviour is PPO3_1's, so this class exists only to
(a) assert at construction that the ID/FD plant is the one actually running,
and (b) give the package its own entry point for gym registration.

The assertion is not ceremony. The inherited runtime falls back to ROM-A
whenever `cylinder_dynamics_model` is anything it does not recognise, and a
silent fallback here would produce PPO3_1 numbers under a PPO4_1 label -- which
would not just be wrong, it would corrupt the one comparison this package exists
to make.
"""

from __future__ import annotations

from PPO_ROM.env import Ppo31SoleEnv


class Ppo41IDFDEnv(Ppo31SoleEnv):
    """Verge line tracking, nine-term reward, ID -> FD actuator."""

    # PPO2's task class is ROM-only by default and refuses to construct
    # anything else. Widening it here is the single line that lets the identical
    # task run over a different actuator. PPO2, PPO3 and PPO3_1 keep the strict
    # default, so this cannot loosen any of them.
    _supported_dynamics_models = frozenset({"fd"})

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        dynamics = str(getattr(cfg, "cylinder_dynamics_model", "")).lower()
        if dynamics != "fd":
            raise ValueError(
                "PPO4_1 is the ID/FD task: cylinder_dynamics_model must be "
                f"'fd', got {dynamics!r}. Constructing this env over ROM-A "
                "would report PPO3_1 behaviour as a PPO4_1 result and silently "
                "invalidate the actuator comparison."
            )
        if not bool(getattr(cfg, "use_cylinder_id_pipeline", False)):
            raise ValueError(
                "PPO4_1 requires use_cylinder_id_pipeline=True; without it the "
                "ID stage is bypassed and FD is driven from an unmapped "
                "command."
            )
        super().__init__(cfg, render_mode=render_mode, **kwargs)
