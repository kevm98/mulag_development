from __future__ import annotations

import numpy as np
import scipy.spatial.transform as tf
import torch
import trimesh
from typing import TYPE_CHECKING

from isaaclab.terrains.trimesh.utils import *  # noqa: F401, F403
from isaaclab.terrains.trimesh.utils import make_border, make_plane

if TYPE_CHECKING:
    from . import mesh_terrains_cfg

def repeated_objects_terrain_without_ground_plane(
    difficulty: float, cfg: mesh_terrains_cfg.MeshRepeatedObjectsTerrainWithoutGroundPlaneCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with a set of repeated objects EXTENDED VERSION.

    The terrain has a ground with a platform in the middle. The objects are randomly placed on the
    terrain s.t. they do not overlap with the platform.

    Depending on the object type, the objects are generated with different parameters. The objects
    The types of objects that can be generated are: ``"cylinder"``, ``"box"``, ``"cone"``.

    The object parameters are specified in the configuration as curriculum parameters. The difficulty
    is used to linearly interpolate between the minimum and maximum values of the parameters.

    .. image:: ../../_static/terrains/trimesh/repeated_objects_cylinder_terrain.jpg
       :width: 30%

    .. image:: ../../_static/terrains/trimesh/repeated_objects_box_terrain.jpg
       :width: 30%

    .. image:: ../../_static/terrains/trimesh/repeated_objects_pyramid_terrain.jpg
       :width: 30%

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).

    Raises:
        ValueError: If the object type is not supported. It must be either a string or a callable.
    """

    # import the object functions -- this is done here to avoid circular imports
    from .mesh_terrains_cfg import (
        MeshRepeatedObjectsTerrainWithoutGroundPlaneCfg,
    )
    # if object type is a string, get the function: make_{object_type}
    if isinstance(cfg.object_type, str):
        object_func = globals().get(f"make_{cfg.object_type}")
    else:
        object_func = cfg.object_type
    if not callable(object_func):
        raise ValueError(f"Attribute 'object_type' must be str or callable. Received: {object_func}")
    
    # Resolve the terrain configuration
    # -- pass parameters to make calling simpler
    cp_0 = cfg.object_params_start
    cp_1 = cfg.object_params_end
    
    num_objects = cp_0.num_objects + int(difficulty * (cp_1.num_objects - cp_0.num_objects))
    height = cp_0.height + difficulty * (cp_1.height - cp_0.height)
    platform_height = cfg.platform_height if cfg.platform_height >= 0.0 else height
    
    # -- object specific parameters
    # note: SIM114 requires duplicated logical blocks under a single body.
    if isinstance(cfg, MeshRepeatedObjectsTerrainWithoutGroundPlaneCfg):
        cp_0: MeshRepeatedObjectsTerrainWithoutGroundPlaneCfg.ObjectCfg
        cp_1: MeshRepeatedObjectsTerrainWithoutGroundPlaneCfg.ObjectCfg
        object_kwargs = {
            "length": cp_0.size[0] + difficulty * (cp_1.size[0] - cp_0.size[0]),
            "width": cp_0.size[1] + difficulty * (cp_1.size[1] - cp_0.size[1]),
            "max_yx_angle": cp_0.max_yx_angle + difficulty * (cp_1.max_yx_angle - cp_0.max_yx_angle),
            "degrees": cp_0.degrees,
        }
    else:
        raise ValueError(f"Unknown terrain configuration: {cfg}")

    
    border_x_start = 11.0
    border_x_end   = 11.0

    if (cp_0.height == 7.0): 
        #print("[Terrain Gen]: Generating Tree subterrain...")
        border_y_start = 5.3
        border_y_end   = 4.2 # y-direction (unimog side)
    elif (cp_0.height == 1.0): # stones can spawn closer to unimog
        #print("[Terrain Gen]: Generating Stone subterrain...")
        border_y_start = 5.3
        border_y_end   = 2.8 # y-direction (unimog side) smaller value -> closer to unimog
    else: # experimentation obstacle pyramid
        #print("[Terrain Gen]: Generating Experiment subterrain...")
        border_y_start = 5.3
        border_y_end   = 4.2 # y-direction (unimog side)  



    min_distance = 2.7    # min_distance of objects to each other
    max_retries = 40000     # timeout protection

    spawn_x_min = border_x_start
    spawn_x_max = cfg.size[0] - border_x_end
    
    spawn_y_min = border_y_start
    spawn_y_max = cfg.size[1] - border_y_end


    if spawn_x_max <= spawn_x_min or spawn_y_max <= spawn_y_min: # param check
        raise ValueError(
            f"Spawn area is invalid (negative size). \n"
            f"Terrain Size: {cfg.size}\n"
            f"X Range: {spawn_x_min} to {spawn_x_max}\n"
            f"Y Range: {spawn_y_min} to {spawn_y_max}\n"
            f"Check your border values!"
        )

    meshes_list = list()
    platform_clearance = 0.1
    origin = np.asarray((0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.5 * platform_height))
    
    platform_corners = np.asarray([
        [origin[0] - cfg.platform_width / 2, origin[1] - cfg.platform_width / 2],
        [origin[0] + cfg.platform_width / 2, origin[1] + cfg.platform_width / 2],
    ])
    platform_corners[0, :] *= 1 - platform_clearance
    platform_corners[1, :] *= 1 + platform_clearance

    object_centers = np.zeros((num_objects, 3))
    mask_objects_left = np.ones((num_objects,), dtype=bool)
    current_retry = 0

    while np.any(mask_objects_left):
        # Timeout Check
        if current_retry > max_retries:
            placed = num_objects - mask_objects_left.sum()
            print(f"[Terrain Gen Warning] Could only place {placed}/{num_objects} objects due to lack of space.")
            print("[Terrain Gen Warning] Randomly trying again...")
            valid_indices = ~mask_objects_left # try with remaining objects
            object_centers = object_centers[valid_indices]
            break 
            
        current_retry += 1

        indices_to_place = np.where(mask_objects_left)[0]
        num_to_place = len(indices_to_place)

        object_centers[indices_to_place, 0] = np.random.uniform(spawn_x_min, spawn_x_max, num_to_place)
        object_centers[indices_to_place, 1] = np.random.uniform(spawn_y_min, spawn_y_max, num_to_place)
        
        # Validate candidates
        for idx in indices_to_place:
            pos = object_centers[idx]
            
            # A. Platform Check
            in_plat_x = (pos[0] >= platform_corners[0, 0]) and (pos[0] <= platform_corners[1, 0])
            in_plat_y = (pos[1] >= platform_corners[0, 1]) and (pos[1] <= platform_corners[1, 1])
            if in_plat_x and in_plat_y:
                continue # Invalid, try again next loop

            valid_mask = ~mask_objects_left
            
            if np.any(valid_mask):
                existing = object_centers[valid_mask]
                # distance to other objects (nur XY)
                dists = np.linalg.norm(existing[:, :2] - pos[:2], axis=1)
                if np.any(dists < min_distance):
                    continue

            mask_objects_left[idx] = False

   
    for center in object_centers:
        abs_height_noise = np.random.uniform(cfg.abs_height_noise[0], cfg.abs_height_noise[1])
        rel_height_noise = np.random.uniform(cfg.rel_height_noise[0], cfg.rel_height_noise[1])
        ob_height = height * rel_height_noise + abs_height_noise
        
        if ob_height > 0.0:
            object_mesh = object_func(center=center, height=ob_height, **object_kwargs)
            meshes_list.append(object_mesh)


    # ground_plane = make_plane(cfg.size, height=0.0, center_zero=False)
    # meshes_list.append(ground_plane)
    
    dim = (cfg.platform_width, cfg.platform_width, 0.5 * platform_height)
    pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.25 * platform_height)
    platform = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
    meshes_list.append(platform)

    return meshes_list, origin