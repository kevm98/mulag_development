"""PPO4 environment: PPO3's verge task, ID/FD actuator.

Deliberately thin. Every physical and task behaviour is PPO3's, so this class
exists only to (a) assert at construction that the ID/FD plant is actually the
one running, and (b) give the package its own entry point for gym registration.

The assertion is not ceremony. The inherited runtime falls back to ROM-A
whenever `cylinder_dynamics_model` is anything it does not recognise, and a
silent fallback here would produce PPO3 numbers under a PPO4 label -- exactly
the class of error this repository keeps finding after the fact.
"""

from __future__ import annotations

from PPO3_SOLE_PhysX.terrain_env import TerrainSolePolicyEnv


class TerrainIDFDEnv(TerrainSolePolicyEnv):
    """Sole-controller arm over ID/FD, constant-speed truck, verge strip."""

    # PPO2's task class is ROM-only by default and refuses to construct anything
    # else. Widening it here is the single line that lets the identical task run
    # over a different actuator. PPO2 and PPO3 keep the strict default, so this
    # cannot loosen either of them.
    _supported_dynamics_models = frozenset({"fd"})

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        dynamics = str(getattr(cfg, "cylinder_dynamics_model", "")).lower()
        if dynamics != "fd":
            raise ValueError(
                "PPO4 requires cylinder_dynamics_model='fd'; got "
                f"{dynamics!r}. PPO4 exists to measure the ID/FD actuator, and "
                "running it over ROM-A would report PPO3 behaviour under a PPO4 "
                "name."
            )
        if not bool(getattr(cfg, "use_cylinder_id_pipeline", False)):
            raise ValueError(
                "PPO4 requires use_cylinder_id_pipeline=True; the FD branch "
                "reads the ID output."
            )
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        # Verify against the constructed runtime, not the config: the config is
        # a request, these attributes are what the plant actually built.
        if getattr(self, "_cylinder_id", None) is None:
            raise RuntimeError(
                "PPO4: the ID runtime did not initialize; the actuator would "
                "silently not be the one under test."
            )
        # No ROM object may exist in this package, not merely be unused. A
        # constructed-but-idle ROM would still be loaded from disk, would appear
        # in any state dump, and is the kind of leftover that later gets read as
        # evidence that PPO4 runs ROM-A.
        rom_runtimes = {
            "_command_cylinder_rom": getattr(self, "_command_cylinder_rom", None),
            "_base_cylinder_rom": getattr(self, "_base_cylinder_rom", None),
            "_cylinder_rom": getattr(self, "_cylinder_rom", None),
        }
        live = sorted(name for name, value in rom_runtimes.items() if value is not None)
        if live:
            raise RuntimeError(
                f"PPO4 constructed ROM runtimes {live}. This package is the "
                "ID/FD pipeline and must contain no ROM model."
            )
        print(
            "[PPO4-IDFD] plant verified: policy -> ID -> FD -> PhysX "
            "(command ROM is NOT in the scored path)",
            flush=True,
        )


__all__ = ["TerrainIDFDEnv"]
