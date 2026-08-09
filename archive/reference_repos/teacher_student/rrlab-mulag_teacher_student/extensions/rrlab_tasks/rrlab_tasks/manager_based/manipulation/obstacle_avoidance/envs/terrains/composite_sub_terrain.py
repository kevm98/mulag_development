from __future__ import annotations
import numpy as np
import trimesh
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .composite_sub_terrain_cfg import CompositeSubTerrainCfg

def composite_terrain(
    difficulty: float, cfg: CompositeSubTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a composite terrain by calling component terrain functions and concatenating meshes."""

    if len(cfg.components) == 0:
        raise ValueError("CompositeSubTerrainCfg.components is empty.")

    meshes_all: list[trimesh.Trimesh] = []
    origins: list[np.ndarray] = []

    # Use a local RNG for deterministic component randomness (stones/trees placement etc.)
    # This avoids global np.random interference.
    rng = np.random.default_rng(getattr(cfg, "seed", 0))

    for i, comp in enumerate(cfg.components):
        comp_i = comp.copy()  # keep original untouched

        # Ensure all components share the same sub-terrain size unless you intentionally want otherwise
        comp_i.size = cfg.size

        # If components use random placement (your repeated_objects_terrain uses np.random),
        # you need deterministic seeding here.
        # Best practice: change those generators to use rng instead of np.random.
        # Minimal workaround: set global seed per component call:
        seed_i = (getattr(cfg, "seed", 0) + 10007 * i) % (2**32 - 1)
        np.random.seed(seed_i)

        meshes_i, origin_i = comp_i.function(difficulty, comp_i)
        meshes_all.extend(meshes_i)
        origins.append(np.asarray(origin_i, dtype=float))

    # Origin policy
    if cfg.origin_source == "first":
        origin = origins[0]
    elif cfg.origin_source == "average":
        origin = np.mean(np.stack(origins, axis=0), axis=0)
    elif cfg.origin_source == "max_z_center":
        # pick the origin with highest z (useful if obstacles set a platform height)
        origin = origins[int(np.argmax([o[2] for o in origins]))]
    else:
        origin = origins[0]

    # Return as list-of-meshes; IsaacLab later concatenates them anyway.
    return meshes_all, origin
