# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
import isaaclab.terrains as terrain_gen
from .terrains.mesh_terrains_cfg import MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg
from .terrains.composite_sub_terrain_cfg import CompositeSubTerrainCfg

UNIMOG_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(45.0, 11.0),  # Size of each terrain patch in meters (overwrites sub_terrains size)
    border_width=20.0,  # Border width around terrain
    num_rows=7,  # more terrains crashes collision detection between unimog and terrain
    num_cols=20,  
    horizontal_scale=0.1,  
    vertical_scale=0.005, 
    slope_threshold=0.75,   
    seed=42,
    curriculum=True, 
    color_scheme="height",  
    difficulty_range=(0.1, 0.9),
    use_cache=False,
    
    sub_terrains={
        # # straight path
        # "flat_path": terrain_gen.MeshPlaneTerrainCfg(
        #     proportion=0.05
        # ),

        # # very small step pyramid
        # "small_py": terrain_gen.HfPyramidSlopedTerrainCfg(
        #     proportion=0.05,
        #     slope_range=(0.03, 0.08),
        #     platform_width= 0.01,
        #     border_width=3.0,
        #     horizontal_scale=0.1,
        #     vertical_scale=0.005    
        # ),

        # # # very small step pyramid (inv)
        # "small_py_inv": terrain_gen.HfPyramidSlopedTerrainCfg(
        #     proportion=0.05,
        #     slope_range=(0.03, 0.08),
        #     platform_width= 0.01,
        #     inverted=True,
        #     border_width=3.0,
        #     horizontal_scale=0.1,
        #     vertical_scale=0.005    
        # ),
        
        # # wave terrain
        # "wave_terrain": terrain_gen.HfWaveTerrainCfg(
        #     proportion=0.05,
        #     border_width=3.0,
        #     horizontal_scale=0.1,
        #     vertical_scale=0.005,
        #     amplitude_range=(0.2, 0.36),
        #     num_waves=1
        # ),
        # random obstacle terrain (stones)
        "obj_terrain_stone": terrain_gen.MeshRepeatedBoxesTerrainCfg( 
            platform_width=0.0,
            proportion=0.05,
            object_params_start=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                num_objects=3,
                size=(0.25, 0.25),
                height=1,
                max_yx_angle=0.0,
                degrees=True
            ),
            object_params_end=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                num_objects=6,
                size=(0.3, 0.3),
                height=1,
                max_yx_angle=0.0,
                degrees=True
            )
        ),
        # # random obstacle terrain (trees)
        # "obj_terrain_tree": terrain_gen.MeshRepeatedBoxesTerrainCfg( 
        #     platform_width=0.0,
        #     proportion=0.05,
        #     object_params_start=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
        #         num_objects=2,
        #         size=(0.25, 0.25),
        #         height=7,
        #         max_yx_angle=0.0,
        #         degrees=True
        #     ),
        #     object_params_end=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
        #         num_objects=5,
        #         size=(0.25, 0.25),
        #         height=7,
        #         max_yx_angle=0.0,
        #         degrees=True
        #     )
        # ),
        # # composite terrain (flat + stones + trees)
        # "composite_terrain_1": CompositeSubTerrainCfg(
        #     proportion=0.05,
        #     components=[
        #         terrain_gen.MeshRepeatedBoxesTerrainCfg( 
        #             platform_width=0.0,
        #             # proportion=0.15,
        #             object_params_start=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
        #                 num_objects=3,
        #                 size=(0.25, 0.25),
        #                 height=1,
        #                 max_yx_angle=0.0,
        #                 degrees=True
        #             ),
        #             object_params_end=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
        #                 num_objects=6,
        #                 size=(0.34, 0.34),
        #                 height=1.2,
        #                 max_yx_angle=0.0,
        #                 degrees=True
        #             )
        #         ),
        #         terrain_gen.MeshRepeatedBoxesTerrainCfg( 
        #             platform_width=0.0,
        #             # proportion=0.15,
        #             object_params_start=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
        #                 num_objects=2,
        #                 size=(0.25, 0.25),
        #                 height=7,
        #                 max_yx_angle=0.0,
        #                 degrees=True
        #             ),   
        #             object_params_end=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
        #                 num_objects=4,
        #                 size=(0.25, 0.25),
        #                 height=7,
        #                 max_yx_angle=0.0,
        #                 degrees=True
        #             )
        #         )]
        # ),


        # composite terrain (pyramid + stones)
        "composite_terrain_2": CompositeSubTerrainCfg(
            proportion=0.05,
            components=[
                terrain_gen.HfPyramidSlopedTerrainCfg(
                    # proportion=0.25,
                    slope_range=(0.03, 0.065),
                    platform_width= 0.01,
                    border_width=3.0,
                    horizontal_scale=0.1,
                    vertical_scale=0.005    
                ),
                terrain_gen.MeshRepeatedBoxesTerrainCfg( 
                    platform_width=0.0,
                    # proportion=0.15,
                    object_params_start=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                        num_objects=3,
                        size=(0.25, 0.25),
                        height=1,
                        max_yx_angle=0.0,
                        degrees=True
                    ),
                    object_params_end=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                        num_objects=6,
                        size=(0.34, 0.34),
                        height=1.2,
                        max_yx_angle=0.0,
                        degrees=True
                    )
                )]
        ),

        # # composite terrain (pyramid + trees)
        # "composite_terrain_3": CompositeSubTerrainCfg(
        #     proportion=0.05,
        #     components=[
        #         terrain_gen.HfPyramidSlopedTerrainCfg(
        #             # proportion=0.25,
        #             slope_range=(0.03, 0.065),
        #             platform_width= 0.01,
        #             border_width=3.0,
        #             horizontal_scale=0.1,
        #             vertical_scale=0.005    
        #         ),
        #         terrain_gen.MeshRepeatedBoxesTerrainCfg( 
        #             platform_width=0.0,
        #             # proportion=0.15,
        #             object_params_start=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
        #                 num_objects=2,
        #                 size=(0.25, 0.25),
        #                 height=7,
        #                 max_yx_angle=0.0,
        #                 degrees=True
        #             ),   
        #             object_params_end=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
        #                 num_objects=4,
        #                 size=(0.25, 0.25),
        #                 height=7,
        #                 max_yx_angle=0.0,
        #                 degrees=True
        #             )
        #         )]
        # ),

        # # composite terrain (pyramid + stones + trees)
        # "composite_terrain_4": CompositeSubTerrainCfg(
        #     proportion=0.05,
        #     components=[
        #         terrain_gen.HfPyramidSlopedTerrainCfg(
        #             # proportion=0.25,
        #             slope_range=(0.03, 0.065),
        #             platform_width= 0.01,
        #             border_width=3.0,
        #             horizontal_scale=0.1,
        #             vertical_scale=0.005    
        #         ),
        #         terrain_gen.MeshRepeatedBoxesTerrainCfg( 
        #             platform_width=0.0,
        #             # proportion=0.15,
        #             object_params_start=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
        #                 num_objects=3,
        #                 size=(0.25, 0.25),
        #                 height=1,
        #                 max_yx_angle=0.0,
        #                 degrees=True
        #             ),
        #             object_params_end=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
        #                 num_objects=6,
        #                 size=(0.34, 0.34),
        #                 height=1.2,
        #                 max_yx_angle=0.0,
        #                 degrees=True
        #             )
        #         ),
        #         terrain_gen.MeshRepeatedBoxesTerrainCfg( 
        #             platform_width=0.0,
        #             # proportion=0.15,
        #             object_params_start=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
        #                 num_objects=2,
        #                 size=(0.25, 0.25),
        #                 height=7,
        #                 max_yx_angle=0.0,
        #                 degrees=True
        #             ),   
        #             object_params_end=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
        #                 num_objects=4,
        #                 size=(0.25, 0.25),
        #                 height=7,
        #                 max_yx_angle=0.0,
        #                 degrees=True
        #             )
        #         )]
        # ),

        # composite terrain (inv pyramid + stones)
        "composite_terrain_5": CompositeSubTerrainCfg(
            proportion=0.05,
            components=[
                terrain_gen.HfPyramidSlopedTerrainCfg(
                    # proportion=0.25,
                    slope_range=(0.03, 0.065),
                    platform_width= 0.01,
                    inverted=True,
                    border_width=3.0,
                    horizontal_scale=0.1,
                    vertical_scale=0.005    
                ),
                MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg( 
                    platform_width=0.0,
                    # proportion=0.15,
                    object_params_start=MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg.ObjectCfg(
                        num_objects=3,
                        size=(0.25, 0.25),
                        height=1,
                        max_yx_angle=0.0,
                        degrees=True
                    ),
                    object_params_end=MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg.ObjectCfg(
                        num_objects=6,
                        size=(0.34, 0.34),
                        height=1.2,
                        max_yx_angle=0.0,
                        degrees=True
                    )
                )]
        ),

        # # composite terrain (inv pyramid + trees)
        # "composite_terrain_6": CompositeSubTerrainCfg(
        #     proportion=0.05,
        #     components=[
        #         terrain_gen.HfPyramidSlopedTerrainCfg(
        #             # proportion=0.25,
        #             slope_range=(0.03, 0.065),
        #             platform_width= 0.01,
        #             inverted=True,
        #             border_width=3.0,
        #             horizontal_scale=0.1,
        #             vertical_scale=0.005    
        #         ),
        #         MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg( 
        #             platform_width=0.0,
        #             # proportion=0.15,
        #             object_params_start=MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg.ObjectCfg(
        #                 num_objects=2,
        #                 size=(0.25, 0.25),
        #                 height=7,
        #                 max_yx_angle=0.0,
        #                 degrees=True
        #             ),   
        #             object_params_end=MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg.ObjectCfg(
        #                 num_objects=4,
        #                 size=(0.25, 0.25),
        #                 height=7,
        #                 max_yx_angle=0.0,
        #                 degrees=True
        #             )
        #         )]
        # ),

        # # composite terrain (inv pyramid + stones + trees)
        # "composite_terrain_7": CompositeSubTerrainCfg(
        #     proportion=0.05,
        #     components=[
        #         terrain_gen.HfPyramidSlopedTerrainCfg(
        #             # proportion=0.25,
        #             slope_range=(0.03, 0.065),
        #             platform_width= 0.01,
        #             inverted=True,
        #             border_width=3.0,
        #             horizontal_scale=0.1,
        #             vertical_scale=0.005    
        #         ),
        #         MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg( 
        #             platform_width=0.0,
        #             # proportion=0.15,
        #             object_params_start=MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg.ObjectCfg(
        #                 num_objects=3,
        #                 size=(0.25, 0.25),
        #                 height=1,
        #                 max_yx_angle=0.0,
        #                 degrees=True
        #             ),
        #             object_params_end=MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg.ObjectCfg(
        #                 num_objects=6,
        #                 size=(0.34, 0.34),
        #                 height=1.2,
        #                 max_yx_angle=0.0,
        #                 degrees=True
        #             )
        #         ),
        #         MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg( 
        #             platform_width=0.0,
        #             # proportion=0.15,
        #             object_params_start=MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg.ObjectCfg(
        #                 num_objects=2,
        #                 size=(0.25, 0.25),
        #                 height=7,
        #                 max_yx_angle=0.0,
        #                 degrees=True
        #             ),   
        #             object_params_end=MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg.ObjectCfg(
        #                 num_objects=4,
        #                 size=(0.25, 0.25),
        #                 height=7,
        #                 max_yx_angle=0.0,
        #                 degrees=True
        #             )
        #         )]
        # ),


        # composite terrain (wave + stones)
        "composite_terrain_8": CompositeSubTerrainCfg(
            proportion=0.05,
            components=[
                terrain_gen.HfWaveTerrainCfg(
                    border_width=3.0,
                    horizontal_scale=0.1,
                    vertical_scale=0.005,
                    amplitude_range=(0.2, 0.36),
                    num_waves=1,
                ),
                MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg( 
                    platform_width=0.0,
                    # proportion=0.15,
                    object_params_start=MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg.ObjectCfg(
                        num_objects=3,
                        size=(0.25, 0.25),
                        height=1,
                        max_yx_angle=0.0,
                        degrees=True
                    ),
                    object_params_end=MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg.ObjectCfg(
                        num_objects=6,
                        size=(0.34, 0.34),
                        height=1.2,
                        max_yx_angle=0.0,
                        degrees=True
                    )
                )]
        ),
        # # composite terrain (wave + trees)
        # "composite_terrain_9": CompositeSubTerrainCfg(
        #     proportion=0.05,
        #     components=[
        #         terrain_gen.HfWaveTerrainCfg(
        #             # proportion=0.2,
        #             border_width=3.0,
        #             horizontal_scale=0.1,
        #             vertical_scale=0.005,
        #             amplitude_range=(0.2, 0.36),
        #             num_waves=1
        #         ),
        #         MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg( 
        #             platform_width=0.0,
        #             # proportion=0.15,
        #             object_params_start=MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg.ObjectCfg(
        #                 num_objects=2,
        #                 size=(0.25, 0.25),
        #                 height=7,
        #                 max_yx_angle=0.0,
        #                 degrees=True
        #             ),   
        #             object_params_end=MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg.ObjectCfg(
        #                 num_objects=4,
        #                 size=(0.25, 0.25),
        #                 height=7,
        #                 max_yx_angle=0.0,
        #                 degrees=True
        #             )
        #         )]
        # ),
        # # composite terrain (wave + stones + trees)
        # "composite_terrain_10": CompositeSubTerrainCfg(
        #     proportion=0.05,
        #     components=[
        #         terrain_gen.HfWaveTerrainCfg(
        #             # proportion=0.2,
        #             border_width=3.0,
        #             horizontal_scale=0.1,
        #             vertical_scale=0.005,
        #             amplitude_range=(0.2, 0.36),
        #             num_waves=1
        #         ),
        #         MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg( 
        #             platform_width=0.0,
        #             # proportion=0.15,
        #             object_params_start=MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg.ObjectCfg(
        #                 num_objects=3,
        #                 size=(0.25, 0.25),
        #                 height=1,
        #                 max_yx_angle=0.0,
        #                 degrees=True
        #             ),
        #             object_params_end=MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg.ObjectCfg(
        #                 num_objects=6,
        #                 size=(0.34, 0.34),
        #                 height=1.2,
        #                 max_yx_angle=0.0,
        #                 degrees=True
        #             )
        #         ),
        #         MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg( 
        #             platform_width=0.0,
        #             # proportion=0.15,
        #             object_params_start=MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg.ObjectCfg(
        #                 num_objects=2,
        #                 size=(0.25, 0.25),
        #                 height=7,
        #                 max_yx_angle=0.0,
        #                 degrees=True
        #             ),   
        #             object_params_end=MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg.ObjectCfg(
        #                 num_objects=4,
        #                 size=(0.25, 0.25),
        #                 height=7,
        #                 max_yx_angle=0.0,
        #                 degrees=True
        #             )
        #         )]
        # ),

        # # composite terrain (wave + pyramid)
        # "composite_terrain_11": CompositeSubTerrainCfg(
        #     proportion=0.05,
        #     components=[
        #         terrain_gen.HfWaveTerrainCfg(
        #             # proportion=0.2,
        #             border_width=3.0,
        #             horizontal_scale=0.1,
        #             vertical_scale=0.005,
        #             amplitude_range=(0.2, 0.36),
        #             num_waves=1
        #         ),
        #         terrain_gen.HfPyramidSlopedTerrainCfg(
        #             # proportion=0.25,
        #             slope_range=(0.03, 0.065),
        #             platform_width= 0.01,
        #             border_width=3.0,
        #             horizontal_scale=0.1,
        #             vertical_scale=0.005    
        #         )]
        # ),

        # composite terrain (wave + pyramid + stones)
        "composite_terrain_12": CompositeSubTerrainCfg(
            proportion=0.05,
            components=[
                terrain_gen.HfWaveTerrainCfg(
                    # proportion=0.2,
                    border_width=3.0,
                    horizontal_scale=0.1,
                    vertical_scale=0.005,
                    amplitude_range=(0.2, 0.36),
                    num_waves=1
                ),
                terrain_gen.HfPyramidSlopedTerrainCfg(
                    # proportion=0.25,
                    slope_range=(0.03, 0.065),
                    platform_width= 0.01,
                    border_width=3.0,
                    horizontal_scale=0.1,
                    vertical_scale=0.005    
                ),
                MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg( 
                    platform_width=0.0,
                    # proportion=0.15,
                    object_params_start=MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg.ObjectCfg(
                        num_objects=3,
                        size=(0.25, 0.25),
                        height=1,
                        max_yx_angle=0.0,
                        degrees=True
                    ),
                    object_params_end=MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg.ObjectCfg(
                        num_objects=6,
                        size=(0.34, 0.34),
                        height=1.2,
                        max_yx_angle=0.0,
                        degrees=True
                    )
                )]
        ),
        # # composite terrain (wave + pyramid + trees)
        # "composite_terrain_13": CompositeSubTerrainCfg(
        #     proportion=0.05,
        #     components=[
        #         terrain_gen.HfWaveTerrainCfg(
        #             # proportion=0.2,
        #             border_width=3.0,
        #             horizontal_scale=0.1,
        #             vertical_scale=0.005,
        #             amplitude_range=(0.2, 0.36),
        #             num_waves=1
        #         ),
        #         terrain_gen.HfPyramidSlopedTerrainCfg(
        #             # proportion=0.25,
        #             slope_range=(0.03, 0.065),
        #             platform_width= 0.01,
        #             border_width=3.0,
        #             horizontal_scale=0.1,
        #             vertical_scale=0.005    
        #         ),
        #         MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg( 
        #             platform_width=0.0,
        #             # proportion=0.15,
        #             object_params_start=MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg.ObjectCfg(
        #                 num_objects=2,
        #                 size=(0.25, 0.25),
        #                 height=7,
        #                 max_yx_angle=0.0,
        #                 degrees=True
        #             ),   
        #             object_params_end=MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg.ObjectCfg(
        #                 num_objects=4,
        #                 size=(0.25, 0.25),
        #                 height=7,
        #                 max_yx_angle=0.0,
        #                 degrees=True
        #             )
        #         )]
        # ),
        # # composite terrain (wave + pyramid + stones + trees)
        # "composite_terrain_14": CompositeSubTerrainCfg(
        #     proportion=0.05,
        #     components=[
        #         terrain_gen.HfWaveTerrainCfg(
        #             # proportion=0.2,
        #             border_width=3.0,
        #             horizontal_scale=0.1,
        #             vertical_scale=0.005,
        #             amplitude_range=(0.2, 0.36),
        #             num_waves=1
        #         ),
        #         terrain_gen.HfPyramidSlopedTerrainCfg(
        #             # proportion=0.25,
        #             slope_range=(0.03, 0.065),
        #             platform_width= 0.01,
        #             border_width=3.0,
        #             horizontal_scale=0.1,
        #             vertical_scale=0.005    
        #         ),
        #         MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg( 
        #             platform_width=0.0,
        #             # proportion=0.15,
        #             object_params_start=MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg.ObjectCfg(
        #                 num_objects=3,
        #                 size=(0.25, 0.25),
        #                 height=1,
        #                 max_yx_angle=0.0,
        #                 degrees=True
        #             ),
        #             object_params_end=MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg.ObjectCfg(
        #                 num_objects=6,
        #                 size=(0.34, 0.34),
        #                 height=1.2,
        #                 max_yx_angle=0.0,
        #                 degrees=True
        #             )
        #         ),
        #         MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg( 
        #             platform_width=0.0,
        #             # proportion=0.15,
        #             object_params_start=MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg.ObjectCfg(
        #                 num_objects=2,
        #                 size=(0.25, 0.25),
        #                 height=7,
        #                 max_yx_angle=0.0,
        #                 degrees=True
        #             ),   
        #             object_params_end=MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg.ObjectCfg(
        #                 num_objects=4,
        #                 size=(0.25, 0.25),
        #                 height=7,
        #                 max_yx_angle=0.0,
        #                 degrees=True
        #             )
        #         )], 
        # ),
    }
)


UNIMOG_TERRAIN_EVAL_CFG = TerrainGeneratorCfg(
    size=(45.0, 11.0),  # Size of each terrain patch in meters (overwrites sub_terrains size)
    border_width=20.0,  # Border width around terrain
    num_rows=7,  
    num_cols=20,  
    horizontal_scale=0.1,  
    vertical_scale=0.005, 
    slope_threshold=0.75,  
    seed=42,  
    curriculum=True, 
    color_scheme="random",  
    difficulty_range=(0.5, 0.5),
    use_cache=False,
    
    sub_terrains={
        # random obstacle terrain (experimentation)
        #"obj_terrain_exp": terrain_gen.MeshRepeatedPyramidsTerrainCfg( 
        #    platform_width=0.0,
        #    proportion=0.1,
        #    object_params_start=terrain_gen.MeshRepeatedPyramidsTerrainCfg.ObjectCfg(
        #        num_objects=2,
        #        radius=0.2,
        #        height=1.3,
        #        max_yx_angle=12.0,
        #        degrees=True
        #    ),
        #    object_params_end=terrain_gen.MeshRepeatedPyramidsTerrainCfg.ObjectCfg(
        #        num_objects=4,
        #        radius=0.4,
        #        height=2.1,
        #        max_yx_angle=29.0,
        #        degrees=True
        #    )
        #),
        # straight path
        "flat_path": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.05
        ),

        # very small step pyramid
        "small_py": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.05,
            slope_range=(0.03, 0.08),
            platform_width= 0.01,
            border_width=3.0,
            horizontal_scale=0.1,
            vertical_scale=0.005    
        ),

        # # very small step pyramid (inv)
        "small_py_inv": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.05,
            slope_range=(0.03, 0.08),
            platform_width= 0.01,
            inverted=True,
            border_width=3.0,
            horizontal_scale=0.1,
            vertical_scale=0.005    
        ),
        
        # wave terrain
        "wave_terrain": terrain_gen.HfWaveTerrainCfg(
            proportion=0.05,
            border_width=3.0,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            amplitude_range=(0.2, 0.36),
            num_waves=1
        ),
        # random obstacle terrain (stones)
        "obj_terrain_stone": terrain_gen.MeshRepeatedBoxesTerrainCfg( 
            platform_width=0.0,
            proportion=0.05,
            object_params_start=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                num_objects=3,
                size=(0.25, 0.25),
                height=1,
                max_yx_angle=0.0,
                degrees=True
            ),
            object_params_end=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                num_objects=6,
                size=(0.3, 0.3),
                height=1,
                max_yx_angle=0.0,
                degrees=True
            )
        ),
        # random obstacle terrain (trees)
        "obj_terrain_tree": terrain_gen.MeshRepeatedBoxesTerrainCfg( 
            platform_width=0.0,
            proportion=0.05,
            object_params_start=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                num_objects=2,
                size=(0.25, 0.25),
                height=7,
                max_yx_angle=0.0,
                degrees=True
            ),
            object_params_end=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                num_objects=5,
                size=(0.25, 0.25),
                height=7,
                max_yx_angle=0.0,
                degrees=True
            )
        ),
        # composite terrain (flat + stones + trees)
        "composite_terrain_1": CompositeSubTerrainCfg(
            proportion=0.05,
            components=[
                terrain_gen.MeshRepeatedBoxesTerrainCfg( 
                    platform_width=0.0,
                    # proportion=0.15,
                    object_params_start=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                        num_objects=3,
                        size=(0.25, 0.25),
                        height=1,
                        max_yx_angle=0.0,
                        degrees=True
                    ),
                    object_params_end=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                        num_objects=6,
                        size=(0.34, 0.34),
                        height=1.2,
                        max_yx_angle=0.0,
                        degrees=True
                    )
                ),
                terrain_gen.MeshRepeatedBoxesTerrainCfg( 
                    platform_width=0.0,
                    # proportion=0.15,
                    object_params_start=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                        num_objects=2,
                        size=(0.25, 0.25),
                        height=7,
                        max_yx_angle=0.0,
                        degrees=True
                    ),   
                    object_params_end=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                        num_objects=4,
                        size=(0.25, 0.25),
                        height=7,
                        max_yx_angle=0.0,
                        degrees=True
                    )
                )]
        ),


        # composite terrain (pyramid + stones)
        "composite_terrain_2": CompositeSubTerrainCfg(
            proportion=0.05,
            components=[
                terrain_gen.HfPyramidSlopedTerrainCfg(
                    # proportion=0.25,
                    slope_range=(0.03, 0.065),
                    platform_width= 0.01,
                    border_width=3.0,
                    horizontal_scale=0.1,
                    vertical_scale=0.005    
                ),
                terrain_gen.MeshRepeatedBoxesTerrainCfg( 
                    platform_width=0.0,
                    # proportion=0.15,
                    object_params_start=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                        num_objects=3,
                        size=(0.25, 0.25),
                        height=1,
                        max_yx_angle=0.0,
                        degrees=True
                    ),
                    object_params_end=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                        num_objects=6,
                        size=(0.34, 0.34),
                        height=1.2,
                        max_yx_angle=0.0,
                        degrees=True
                    )
                )]
        ),

        # composite terrain (pyramid + trees)
        "composite_terrain_3": CompositeSubTerrainCfg(
            proportion=0.05,
            components=[
                terrain_gen.HfPyramidSlopedTerrainCfg(
                    # proportion=0.25,
                    slope_range=(0.03, 0.065),
                    platform_width= 0.01,
                    border_width=3.0,
                    horizontal_scale=0.1,
                    vertical_scale=0.005    
                ),
                terrain_gen.MeshRepeatedBoxesTerrainCfg( 
                    platform_width=0.0,
                    # proportion=0.15,
                    object_params_start=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                        num_objects=2,
                        size=(0.25, 0.25),
                        height=7,
                        max_yx_angle=0.0,
                        degrees=True
                    ),   
                    object_params_end=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                        num_objects=4,
                        size=(0.25, 0.25),
                        height=7,
                        max_yx_angle=0.0,
                        degrees=True
                    )
                )]
        ),

        # composite terrain (pyramid + stones + trees)
        "composite_terrain_4": CompositeSubTerrainCfg(
            proportion=0.05,
            components=[
                terrain_gen.HfPyramidSlopedTerrainCfg(
                    # proportion=0.25,
                    slope_range=(0.03, 0.065),
                    platform_width= 0.01,
                    border_width=3.0,
                    horizontal_scale=0.1,
                    vertical_scale=0.005    
                ),
                terrain_gen.MeshRepeatedBoxesTerrainCfg( 
                    platform_width=0.0,
                    # proportion=0.15,
                    object_params_start=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                        num_objects=3,
                        size=(0.25, 0.25),
                        height=1,
                        max_yx_angle=0.0,
                        degrees=True
                    ),
                    object_params_end=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                        num_objects=6,
                        size=(0.34, 0.34),
                        height=1.2,
                        max_yx_angle=0.0,
                        degrees=True
                    )
                ),
                terrain_gen.MeshRepeatedBoxesTerrainCfg( 
                    platform_width=0.0,
                    # proportion=0.15,
                    object_params_start=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                        num_objects=2,
                        size=(0.25, 0.25),
                        height=7,
                        max_yx_angle=0.0,
                        degrees=True
                    ),   
                    object_params_end=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                        num_objects=4,
                        size=(0.25, 0.25),
                        height=7,
                        max_yx_angle=0.0,
                        degrees=True
                    )
                )]
        ),

        # composite terrain (inv pyramid + stones)
        "composite_terrain_5": CompositeSubTerrainCfg(
            proportion=0.05,
            components=[
                terrain_gen.HfPyramidSlopedTerrainCfg(
                    # proportion=0.25,
                    slope_range=(0.03, 0.065),
                    platform_width= 0.01,
                    inverted=True,
                    border_width=3.0,
                    horizontal_scale=0.1,
                    vertical_scale=0.005    
                ),
                MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg( 
                    platform_width=0.0,
                    # proportion=0.15,
                    object_params_start=MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg.ObjectCfg(
                        num_objects=3,
                        size=(0.25, 0.25),
                        height=1,
                        max_yx_angle=0.0,
                        degrees=True
                    ),
                    object_params_end=MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg.ObjectCfg(
                        num_objects=6,
                        size=(0.34, 0.34),
                        height=1.2,
                        max_yx_angle=0.0,
                        degrees=True
                    )
                )]
        ),

        # composite terrain (inv pyramid + trees)
        "composite_terrain_6": CompositeSubTerrainCfg(
            proportion=0.05,
            components=[
                terrain_gen.HfPyramidSlopedTerrainCfg(
                    # proportion=0.25,
                    slope_range=(0.03, 0.065),
                    platform_width= 0.01,
                    inverted=True,
                    border_width=3.0,
                    horizontal_scale=0.1,
                    vertical_scale=0.005    
                ),
                MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg( 
                    platform_width=0.0,
                    # proportion=0.15,
                    object_params_start=MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg.ObjectCfg(
                        num_objects=2,
                        size=(0.25, 0.25),
                        height=7,
                        max_yx_angle=0.0,
                        degrees=True
                    ),   
                    object_params_end=MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg.ObjectCfg(
                        num_objects=4,
                        size=(0.25, 0.25),
                        height=7,
                        max_yx_angle=0.0,
                        degrees=True
                    )
                )]
        ),

        # composite terrain (inv pyramid + stones + trees)
        "composite_terrain_7": CompositeSubTerrainCfg(
            proportion=0.05,
            components=[
                terrain_gen.HfPyramidSlopedTerrainCfg(
                    # proportion=0.25,
                    slope_range=(0.03, 0.065),
                    platform_width= 0.01,
                    inverted=True,
                    border_width=3.0,
                    horizontal_scale=0.1,
                    vertical_scale=0.005    
                ),
                MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg( 
                    platform_width=0.0,
                    # proportion=0.15,
                    object_params_start=MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg.ObjectCfg(
                        num_objects=3,
                        size=(0.25, 0.25),
                        height=1,
                        max_yx_angle=0.0,
                        degrees=True
                    ),
                    object_params_end=MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg.ObjectCfg(
                        num_objects=6,
                        size=(0.34, 0.34),
                        height=1.2,
                        max_yx_angle=0.0,
                        degrees=True
                    )
                ),
                MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg( 
                    platform_width=0.0,
                    # proportion=0.15,
                    object_params_start=MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg.ObjectCfg(
                        num_objects=2,
                        size=(0.25, 0.25),
                        height=7,
                        max_yx_angle=0.0,
                        degrees=True
                    ),   
                    object_params_end=MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg.ObjectCfg(
                        num_objects=4,
                        size=(0.25, 0.25),
                        height=7,
                        max_yx_angle=0.0,
                        degrees=True
                    )
                )]
        ),


        # composite terrain (wave + stones)
        "composite_terrain_8": CompositeSubTerrainCfg(
            proportion=0.05,
            components=[
                terrain_gen.HfWaveTerrainCfg(
                    border_width=3.0,
                    horizontal_scale=0.1,
                    vertical_scale=0.005,
                    amplitude_range=(0.2, 0.36),
                    num_waves=1,
                ),
                MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg( 
                    platform_width=0.0,
                    # proportion=0.15,
                    object_params_start=MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg.ObjectCfg(
                        num_objects=3,
                        size=(0.25, 0.25),
                        height=1,
                        max_yx_angle=0.0,
                        degrees=True
                    ),
                    object_params_end=MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg.ObjectCfg(
                        num_objects=6,
                        size=(0.34, 0.34),
                        height=1.2,
                        max_yx_angle=0.0,
                        degrees=True
                    )
                )]
        ),
        # composite terrain (wave + trees)
        "composite_terrain_9": CompositeSubTerrainCfg(
            proportion=0.05,
            components=[
                terrain_gen.HfWaveTerrainCfg(
                    # proportion=0.2,
                    border_width=3.0,
                    horizontal_scale=0.1,
                    vertical_scale=0.005,
                    amplitude_range=(0.2, 0.36),
                    num_waves=1
                ),
                MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg( 
                    platform_width=0.0,
                    # proportion=0.15,
                    object_params_start=MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg.ObjectCfg(
                        num_objects=2,
                        size=(0.25, 0.25),
                        height=7,
                        max_yx_angle=0.0,
                        degrees=True
                    ),   
                    object_params_end=MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg.ObjectCfg(
                        num_objects=4,
                        size=(0.25, 0.25),
                        height=7,
                        max_yx_angle=0.0,
                        degrees=True
                    )
                )]
        ),
        # composite terrain (wave + stones + trees)
        "composite_terrain_10": CompositeSubTerrainCfg(
            proportion=0.05,
            components=[
                terrain_gen.HfWaveTerrainCfg(
                    # proportion=0.2,
                    border_width=3.0,
                    horizontal_scale=0.1,
                    vertical_scale=0.005,
                    amplitude_range=(0.2, 0.36),
                    num_waves=1
                ),
                MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg( 
                    platform_width=0.0,
                    # proportion=0.15,
                    object_params_start=MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg.ObjectCfg(
                        num_objects=3,
                        size=(0.25, 0.25),
                        height=1,
                        max_yx_angle=0.0,
                        degrees=True
                    ),
                    object_params_end=MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg.ObjectCfg(
                        num_objects=6,
                        size=(0.34, 0.34),
                        height=1.2,
                        max_yx_angle=0.0,
                        degrees=True
                    )
                ),
                MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg( 
                    platform_width=0.0,
                    # proportion=0.15,
                    object_params_start=MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg.ObjectCfg(
                        num_objects=2,
                        size=(0.25, 0.25),
                        height=7,
                        max_yx_angle=0.0,
                        degrees=True
                    ),   
                    object_params_end=MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg.ObjectCfg(
                        num_objects=4,
                        size=(0.25, 0.25),
                        height=7,
                        max_yx_angle=0.0,
                        degrees=True
                    )
                )]
        ),

        # composite terrain (wave + pyramid)
        "composite_terrain_11": CompositeSubTerrainCfg(
            proportion=0.05,
            components=[
                terrain_gen.HfWaveTerrainCfg(
                    # proportion=0.2,
                    border_width=3.0,
                    horizontal_scale=0.1,
                    vertical_scale=0.005,
                    amplitude_range=(0.2, 0.36),
                    num_waves=1
                ),
                terrain_gen.HfPyramidSlopedTerrainCfg(
                    # proportion=0.25,
                    slope_range=(0.03, 0.065),
                    platform_width= 0.01,
                    border_width=3.0,
                    horizontal_scale=0.1,
                    vertical_scale=0.005    
                )]
        ),

        # composite terrain (wave + pyramid + stones)
        "composite_terrain_12": CompositeSubTerrainCfg(
            proportion=0.05,
            components=[
                terrain_gen.HfWaveTerrainCfg(
                    # proportion=0.2,
                    border_width=3.0,
                    horizontal_scale=0.1,
                    vertical_scale=0.005,
                    amplitude_range=(0.2, 0.36),
                    num_waves=1
                ),
                terrain_gen.HfPyramidSlopedTerrainCfg(
                    # proportion=0.25,
                    slope_range=(0.03, 0.065),
                    platform_width= 0.01,
                    border_width=3.0,
                    horizontal_scale=0.1,
                    vertical_scale=0.005    
                ),
                MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg( 
                    platform_width=0.0,
                    # proportion=0.15,
                    object_params_start=MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg.ObjectCfg(
                        num_objects=3,
                        size=(0.25, 0.25),
                        height=1,
                        max_yx_angle=0.0,
                        degrees=True
                    ),
                    object_params_end=MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg.ObjectCfg(
                        num_objects=6,
                        size=(0.34, 0.34),
                        height=1.2,
                        max_yx_angle=0.0,
                        degrees=True
                    )
                )]
        ),
        # composite terrain (wave + pyramid + trees)
        "composite_terrain_13": CompositeSubTerrainCfg(
            proportion=0.05,
            components=[
                terrain_gen.HfWaveTerrainCfg(
                    # proportion=0.2,
                    border_width=3.0,
                    horizontal_scale=0.1,
                    vertical_scale=0.005,
                    amplitude_range=(0.2, 0.36),
                    num_waves=1
                ),
                terrain_gen.HfPyramidSlopedTerrainCfg(
                    # proportion=0.25,
                    slope_range=(0.03, 0.065),
                    platform_width= 0.01,
                    border_width=3.0,
                    horizontal_scale=0.1,
                    vertical_scale=0.005    
                ),
                MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg( 
                    platform_width=0.0,
                    # proportion=0.15,
                    object_params_start=MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg.ObjectCfg(
                        num_objects=2,
                        size=(0.25, 0.25),
                        height=7,
                        max_yx_angle=0.0,
                        degrees=True
                    ),   
                    object_params_end=MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg.ObjectCfg(
                        num_objects=4,
                        size=(0.25, 0.25),
                        height=7,
                        max_yx_angle=0.0,
                        degrees=True
                    )
                )]
        ),
        # composite terrain (wave + pyramid + stones + trees)
        "composite_terrain_14": CompositeSubTerrainCfg(
            proportion=0.05,
            components=[
                terrain_gen.HfWaveTerrainCfg(
                    # proportion=0.2,
                    border_width=3.0,
                    horizontal_scale=0.1,
                    vertical_scale=0.005,
                    amplitude_range=(0.2, 0.36),
                    num_waves=1
                ),
                terrain_gen.HfPyramidSlopedTerrainCfg(
                    # proportion=0.25,
                    slope_range=(0.03, 0.065),
                    platform_width= 0.01,
                    border_width=3.0,
                    horizontal_scale=0.1,
                    vertical_scale=0.005    
                ),
                MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg( 
                    platform_width=0.0,
                    # proportion=0.15,
                    object_params_start=MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg.ObjectCfg(
                        num_objects=3,
                        size=(0.25, 0.25),
                        height=1,
                        max_yx_angle=0.0,
                        degrees=True
                    ),
                    object_params_end=MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg.ObjectCfg(
                        num_objects=6,
                        size=(0.34, 0.34),
                        height=1.2,
                        max_yx_angle=0.0,
                        degrees=True
                    )
                ),
                MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg( 
                    platform_width=0.0,
                    # proportion=0.15,
                    object_params_start=MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg.ObjectCfg(
                        num_objects=2,
                        size=(0.25, 0.25),
                        height=7,
                        max_yx_angle=0.0,
                        degrees=True
                    ),   
                    object_params_end=MeshRepeatedBoxesTerrainWithoutGroundPlaneCfg.ObjectCfg(
                        num_objects=4,
                        size=(0.25, 0.25),
                        height=7,
                        max_yx_angle=0.0,
                        degrees=True
                    )
                )]
        ),

       
    }
)

