from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.utils import configclass
from typing import Callable, Literal
from dataclasses import field
from .composite_sub_terrain import composite_terrain

@configclass
class CompositeSubTerrainCfg(SubTerrainBaseCfg):
    """A sub-terrain that composes multiple sub-terrains into one.

    Composition is typically:
    - choose exactly one "base" heightfield (flat/pyramid/wave) OR allow multiple tiled bases
    - overlay additional mesh terrains (stones/trees/boxes/etc.)
    """

    # Put multiple component terrain configs here (any subclass of SubTerrainBaseCfg)
    components: list[SubTerrainBaseCfg] = field(default_factory=list)

    # How to combine components
    compose_mode: Literal["overlay"] = "overlay"
    # Optional: which component provides the origin
    origin_source: Literal["first", "max_z_center", "average"] = "first"

    # IMPORTANT: set function to our composer
    function: Callable = None  # will be set in __post_init__

    def __post_init__(self):
        # bind the generation function
        self.function = composite_terrain