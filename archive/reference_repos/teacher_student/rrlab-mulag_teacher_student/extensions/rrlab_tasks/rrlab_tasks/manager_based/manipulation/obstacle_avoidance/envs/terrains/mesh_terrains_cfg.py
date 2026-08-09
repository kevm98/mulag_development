import warnings
from dataclasses import MISSING
from typing import Literal

from .mesh_terrains import repeated_objects_terrain_without_ground_plane
import isaaclab.terrains.trimesh.utils as mesh_utils_terrains
from isaaclab.utils import configclass

from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg


@configclass
class MeshRepeatedObjectsTerrainWithoutGroundPlaneCfg(SubTerrainBaseCfg):
    """Base configuration for a terrain with repeated objects."""

    @configclass
    class ObjectCfg:
        """Configuration of repeated objects."""

        num_objects: int = MISSING
        """The number of objects to add to the terrain."""
        height: float = MISSING
        """The height (along z) of the object (in m)."""

    function = repeated_objects_terrain_without_ground_plane

    object_type: Literal["cylinder", "box", "cone"] | callable = MISSING
    """The type of object to generate.

    The type can be a string or a callable. If it is a string, the function will look for a function called
    ``make_{object_type}`` in the current module scope. If it is a callable, the function will
    use the callable to generate the object.
    """

    object_params_start: ObjectCfg = MISSING
    """The object curriculum parameters at the start of the curriculum."""

    object_params_end: ObjectCfg = MISSING
    """The object curriculum parameters at the end of the curriculum."""

    max_height_noise: float | None = None
    """"This parameter is deprecated, but stated here to support backward compatibility"""

    abs_height_noise: tuple[float, float] = (0.0, 0.0)
    """The minimum and maximum amount of additive noise for the height of the objects. Default is set to 0.0, which is no noise."""

    rel_height_noise: tuple[float, float] = (1.0, 1.0)
    """The minimum and maximum amount of multiplicative noise for the height of the objects. Default is set to 1.0, which is no noise."""

    platform_width: float = 1.0
    """The width of the cylindrical platform at the center of the terrain. Defaults to 1.0."""

    platform_height: float = -1.0
    """The height of the platform. Defaults to -1.0.

    If the value is negative, the height is the same as the object height.
    """

    def __post_init__(self):
        if self.max_height_noise is not None:
            warnings.warn(
                "MeshRepeatedObjectsTerrainCfg: max_height_noise:float is deprecated and support will be removed in the"
                " future. Use abs_height_noise:list[float] instead."
            )
            self.abs_height_noise = (-self.max_height_noise, self.max_height_noise)


@configclass
class MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg(MeshRepeatedObjectsTerrainWithoutGroundPlaneCfg):
    """Configuration for a terrain with repeated boxes."""

    @configclass
    class ObjectCfg(MeshRepeatedObjectsTerrainWithoutGroundPlaneCfg.ObjectCfg):
        """Configuration for repeated boxes."""

        size: tuple[float, float] = MISSING
        """The width (along x) and length (along y) of the box (in m)."""
        max_yx_angle: float = 0.0
        """The maximum angle along the y and x axis. Defaults to 0.0."""
        degrees: bool = True
        """Whether the angle is in degrees. Defaults to True."""

    object_type = mesh_utils_terrains.make_box

    object_params_start: ObjectCfg = MISSING
    """The box curriculum parameters at the start of the curriculum."""

    object_params_end: ObjectCfg = MISSING
    """The box curriculum parameters at the end of the curriculum."""