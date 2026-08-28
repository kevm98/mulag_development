"""Compatibility shim for the Isaac Lab ``SimulationCfg.physx`` -> ``physics`` rename.

Isaac Lab reorganised the simulation config so that the PhysX settings object
hangs off ``SimulationCfg.physics`` (typed ``PhysicsCfg | None``, still
defaulting to ``PhysxCfg()``) rather than ``SimulationCfg.physx``.  PhysX is
still the backend on both sides of the rename -- Newton is opt-in via
``use_newton_actuators`` or an explicit ``NewtonManagerCfg`` -- so the settings
this repo applies mean the same thing before and after.

The repo reads and writes ``cfg.sim.physx.*`` in a number of places (runtime
config, memory sizing, tools, tests).  Rather than edit each call site and
break the Legion box, which runs a build that still has ``physx``, install a
read-only ``physx`` alias on the newer class so every existing access keeps
working.  Only a getter is needed: every call site either reads a field or
mutates the object the attribute returns.

Importing this module is idempotent and is a no-op on builds that already
expose ``physx``.
"""

from __future__ import annotations

from isaaclab.sim import SimulationCfg

__all__ = ["PHYSX_FIELD", "install_physx_alias"]

# Which attribute actually carries the PhysX settings on this build.
PHYSX_FIELD = "physx" if "physx" in getattr(SimulationCfg, "__annotations__", {}) else "physics"


def install_physx_alias() -> None:
    """Alias ``SimulationCfg.physx`` onto ``.physics`` when the build lacks it."""
    if PHYSX_FIELD == "physx" or "physx" in vars(SimulationCfg):
        return

    def _physx(self):
        return self.physics

    SimulationCfg.physx = property(_physx)


install_physx_alias()
