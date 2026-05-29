from __future__ import annotations

from typing import Any


DEFAULT_RENDER_INTERVAL = 2

GPU_PHYSX_MEMORY_SETTINGS: dict[str, int] = {
    "gpu_max_rigid_contact_count": 2**16,
    "gpu_found_lost_pairs_capacity": 2**16,
    "gpu_total_aggregate_pairs_capacity": 2**16,
    "gpu_collision_stack_size": 2**20,
    "gpu_heap_capacity": 2**22,
    "gpu_temp_buffer_capacity": 2**20,
}


def apply_kevin_sim_memory_optimizations(
    sim_cfg: Any,
    *,
    render_interval: int = DEFAULT_RENDER_INTERVAL,
    verbose: bool = False,
) -> None:
    """Apply the low-memory SimulationCfg/PhysX settings used by the working standalone controller."""
    if hasattr(sim_cfg, "render_interval"):
        current = getattr(sim_cfg, "render_interval", None)
        if current is None or int(current) < int(render_interval):
            if verbose:
                print(f"[INFO] Setting sim.render_interval = {render_interval}")
            setattr(sim_cfg, "render_interval", int(render_interval))

    physx = getattr(sim_cfg, "physx", sim_cfg)
    for name, value in GPU_PHYSX_MEMORY_SETTINGS.items():
        if hasattr(physx, name):
            if verbose:
                print(f"[INFO] Setting sim.physx.{name} = {value}")
            setattr(physx, name, value)
