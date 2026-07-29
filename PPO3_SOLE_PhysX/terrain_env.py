"""PPO3 verge mowing: truck on the road, arm over the embankment.

The MULAG FME 500 is a verge mower. The truck drives a straight line at constant
speed on a flat road, and the arm reaches out to the side to cut ground that
rises and falls. Only the cutter has to track terrain; the wheels never leave a
level surface.

Three changes relative to ``SolePolicyEnv``. The ROM-A plant, the action chain,
the limit shield and the reward core are untouched.

1.  **The truck drives.** ``enable_scripted_vehicle_motion`` writes wheel angular
    velocity directly, the same mechanism teacher_student uses in
    ``mdp/randomization.py`` (km/h -> m/s -> rad/s through a 0.575 m wheel
    radius). The truck is NOT in the action space: teacher_student's entire
    ``ActionsCfg`` is a single ``arm_action``, and keeping the vehicle scripted
    leaves the problem as "track the ground despite the truck's motion".

2.  **The truck is no longer pinned.** ``SolePolicyEnv`` inherits
    ``_hold_wheels`` (stiffness 2e9, wheel position targets) and
    ``_constrain_truck_xy`` (rewrites the root pose every step, terminates at
    1 mm of drift). The position hold directly fights the velocity write, and
    the root-pose rewrite would teleport a driving truck backwards, so both are
    replaced. Lateral drift off the driving line is still measured.

3.  **A verge strip is added beside the road.** The road itself stays exactly as
    the parent builds it -- a flat plate at z=0, with the truck at its proven
    ``init_state.pos`` z of 0.05. An earlier version drove the truck over
    generated terrain instead; that spawned the chassis inside a hillside
    (measured 0.60 km/h against 4.00 commanded), and the resulting impact through
    the stiff arm produced ``cylinder_q contains NaN/Inf``. Keeping the wheels on
    a level road removes that entire failure mode.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import torch
from pxr import Sdf

import isaaclab.sim as sim_utils
from isaaclab.sensors import ContactSensor, ContactSensorCfg, RayCaster, RayCasterCfg
from isaaclab.sensors.ray_caster import patterns
from isaaclab.terrains.utils import create_prim_from_mesh
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_inv, quat_mul

from PPO3_SOLE_PhysX.env import SolePolicyEnv
from PPO3_SOLE_PhysX.actions import apply_sole_action, directional_tilt_shield
from PPO3_SOLE_PhysX.terrain_observations import (
    HEIGHT_PROFILE_BINS,
    PATH_PREVIEW_BINS,
    build_verge_observation,
    collapse_scan_width,
    nonfinite_row_mask,
)
from PPO3_SOLE_PhysX.terrain_rewards import (
    bounded_cost_progress,
    compute_verge_reward,
    sanitize_terminal_reward,
)
from PPO3_SOLE_PhysX.terrain import (
    MOW_STRIP_CFG,
    MOW_STRIP_EVAL_CFG,
    TRACKING_PATH_FAMILIES,
    build_mow_strip_mesh,
    build_tracking_line_mesh,
    terrain_height_scale_for_environment,
    tracking_line_center_y_at,
)

MOW_STRIP_ROOT = "/World/MowStrip"
ROAD_ROOT = "/World/TruckRoad"
TRACKING_LINE_ROOT = "/World/MowTrackingLine"
VISUAL_FLOOR_ROOT = "/World/PPO3VisualFloor"
VISUAL_GRID_ROOT = "/World/PPO3VisualGrid"


def _path_aligned_grid_pattern(cfg, device: str):
    """Grid translated in the fixed drive frame, independent of chassis attitude."""
    starts, directions = patterns.grid_pattern(cfg, device)
    starts = starts + torch.as_tensor(cfg.center, dtype=starts.dtype, device=device)
    return starts, directions


@configclass
class PathAlignedGridPatternCfg(patterns.GridPatternCfg):
    """Ray grid whose centre is an offset from the moving chassis position."""

    func: Callable = _path_aligned_grid_pattern
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)


class TerrainSolePolicyEnv(SolePolicyEnv):
    """Sole-controller arm over ROM-A, constant-speed truck, verge strip."""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self._terrain_held_joint_ids = torch.cat(
            (
                self._cylinder_joint_ids,
                torch.as_tensor(
                    [self._tool_joint_id], dtype=torch.long, device=self.device
                ),
            )
        )
        self._terrain_nominal_arm_damping = self.robot.data.joint_damping[
            :, self._terrain_held_joint_ids
        ].detach().clone()
        self._terrain_nominal_arm_stiffness = self.robot.data.joint_stiffness[
            :, self._terrain_held_joint_ids
        ].detach().clone()
        self._terrain_damping_logged = False
        self._terrain_step_nonfinite = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._previous_abs_cut_error = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self._previous_abs_lateral_error = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self._contact_demo_ready = False
        self._install_terrain_deployment_target()
        speed = float(getattr(cfg, "vehicle_speed_min_kmh", 0.0))
        strip = self._mow_strip_cfg()
        print(
            f"[PPO3-VERGE] truck DRIVING at {speed:.2f} km/h ({speed / 3.6:.3f} m/s) "
            "on the DARK ASPHALT FLAT ROAD, scripted (not policy-controlled) | chassis XY/Z/yaw "
            "locked to the straight road, wheel-road contact measured | green verge strip "
            f"{strip.length_m:.0f}x{strip.width_m:.0f} m at y={strip.center_y_m:.2f}, "
            f"irregular broad hills up to {max(strip.hill_heights_m):.2f} m, "
            f"background rise 0 -> {strip.elevation_rise_end_m:.2f} m, smooth variation "
            f"{strip.amplitude_start_m:.3f} -> {strip.amplitude_end_m:.2f} m "
            "along the drive | target cut height "
            f"{float(getattr(cfg, 'target_cut_height_m', 0.05)):.3f} m",
            flush=True,
        )

    @torch.no_grad()
    def _apply_terrain_arm_drive_tuning(self) -> None:
        """Apply the measured terrain-stage damping to the held arm joints.

        A matched frozen-pose screen measured 64.02 m/s² p95 with the inherited
        damping, 104.37 at 2x, and 38.39 at 0.5x.  Effort-limit changes up to
        2 MN were bit-identical, so force limits stay untouched. Deployment uses
        inherited damping because its finite preposition/settle was validated
        there; this tuning starts only after the final synchronized pose.
        """
        if not hasattr(self, "_terrain_nominal_arm_damping"):
            return
        scale = float(getattr(self.cfg, "terrain_arm_damping_scale", 1.0))
        if not 0.0 < scale <= 1.0:
            raise ValueError(
                f"terrain_arm_damping_scale must be in (0, 1], got {scale}"
            )
        self.robot.write_joint_damping_to_sim(
            self._terrain_nominal_arm_damping * scale,
            joint_ids=self._terrain_held_joint_ids,
        )
        if not self._terrain_damping_logged:
            print(
                f"[PPO3-VERGE] scored terrain arm damping scale={scale:.3f} "
                "(deployment remains at stock damping)",
                flush=True,
            )
            self._terrain_damping_logged = True

    @torch.no_grad()
    def _restore_terrain_arm_deployment_damping(self) -> None:
        """Restore the validated stock drive before physical preposition."""
        if not hasattr(self, "_terrain_nominal_arm_damping"):
            return
        self.robot.write_joint_damping_to_sim(
            self._terrain_nominal_arm_damping,
            joint_ids=self._terrain_held_joint_ids,
        )

    def _install_terrain_deployment_target(self) -> None:
        """Make the measured verge pose both endpoints of reset deployment.

        The parent chooses whichever trajectory endpoint is closer to the spawn
        cylinder state. Replacing both endpoint seeds preserves its validated
        cosine deployment and ROM warm-up while removing an irrelevant random
        stored-path phase from the terrain task.
        """
        target = tuple(
            float(value)
            for value in getattr(
                self.cfg, "terrain_deployment_cylinder_target_m", ()
            )
        )
        if len(target) != 4:
            raise ValueError(
                "terrain_deployment_cylinder_target_m must contain 4 extensions"
            )
        self._seeds[0, :] = target
        self._seeds[-1, :] = target
        print(
            f"[PPO3-VERGE] deterministic deployment target={list(target)}",
            flush=True,
        )

    def _init_kinematic_drive(self) -> None:
        """Latch the pose the truck drives from, once it has settled."""
        if getattr(self, "_kinematic_drive_ready", False):
            return
        self._truck_travel_m = torch.zeros(
            (self.num_envs,), dtype=torch.float32, device=self.device
        )
        self._last_advance_step = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._drive_anchor_root = self.robot.data.root_state_w[:, :7].detach().clone()
        self._drive_anchor_base_xy = self.robot.data.body_link_state_w[
            :, self._truck_base_body_id(), :2
        ].detach().clone()
        self._drive_anchor_base_heading_xy = self._truck_base_heading_xy()
        self._drive_anchor_base_quat = self.robot.data.body_link_state_w[
            :, self._truck_base_body_id(), 3:7
        ].detach().clone()
        self._kinematic_drive_ready = True

    def _truck_base_heading_xy(self) -> torch.Tensor:
        """Normalized world-XY projection of chassis local +X."""
        base_quat = self.robot.data.body_link_state_w[
            :, self._truck_base_body_id(), 3:7
        ]
        forward = torch.tensor(
            [[1.0, 0.0, 0.0]], dtype=base_quat.dtype, device=self.device
        ).expand(self.num_envs, -1)
        heading = quat_apply(base_quat, forward)[:, :2]
        return heading / torch.linalg.norm(
            heading, dim=-1, keepdim=True
        ).clamp(min=1.0e-8)

    def _expected_observation_dim(self) -> int:
        from PPO3_SOLE_PhysX.terrain_observations import verge_observation_dim

        # +1 for the trailing scan-validity channel.
        return verge_observation_dim() + 1

    def _mow_strip_cfg(self):
        held_out = bool(getattr(self.cfg, "use_eval_terrain", False))
        base = MOW_STRIP_EVAL_CFG if held_out else MOW_STRIP_CFG
        # Held-out geometry remains fixed at seed 1337. Training entry points
        # deliberately vary this seed between resumed blocks so the policy does
        # not replay the same 512 static meshes forever.
        seed = (
            int(base.seed)
            if held_out
            else int(getattr(self.cfg, "terrain_base_seed", base.seed))
        )
        resolution = float(
            getattr(self.cfg, "mow_strip_resolution_m", base.resolution_m)
        )
        length = float(getattr(self.cfg, "mow_strip_length_m", base.length_m))
        width = float(getattr(self.cfg, "mow_strip_width_m", base.width_m))
        return replace(
            base,
            seed=seed,
            resolution_m=resolution,
            length_m=length,
            width_m=width,
        )

    def _cutter_prim_path(self) -> str:
        """Full USD path expression of the cutter body across all envs.

        This asset nests bodies under ``Rigid_Bodies/Arm_Links/``. teacher_student's
        copy of the Unimog exposes them directly under ``/Robot``, so its sensor
        paths do not transfer -- sensors attach by USD path, not by articulation
        body name.
        """
        subpath = str(
            getattr(self.cfg, "cutter_prim_subpath", self.cfg.cutter_body_name)
        ).strip("/")
        return f"{self.cfg.robot_cfg.prim_path}/{subpath}"

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------

    def _setup_scene(self):
        # The parent creates a world-sized flat plate.  That cannot remain under
        # the verge: valleys below z=0 would collide with the hidden plate rather
        # than the green terrain. Replace it with a narrow asphalt road under the
        # tyres and a physically separate green collision mesh under the cutter.
        super()._setup_scene()
        self._configure_dark_visual_theme()
        self._disable_parent_ground()
        if bool(getattr(self.cfg, "enable_visual_grid_floor", False)):
            self._spawn_visual_grid_floor()
        self._spawn_flat_road()
        self._spawn_mow_strip()
        self._add_ground_sensors()

    def _configure_dark_visual_theme(self) -> None:
        """Use Isaac's dark viewport background without darkening the robot.

        The parent scene's uniform dome light is visible to primary camera rays,
        which is why the empty world renders as solid white. Hiding the dome
        from primary rays restores the default dark renderer background while
        keeping the same dome illumination and therefore the trained scene's
        object visibility. This is visual-only and changes no physics.
        """
        if not bool(getattr(self.cfg, "dark_visual_background", True)):
            return
        light = sim_utils.get_prim_at_path("/World/Light")
        attribute = light.GetAttribute("visibleInPrimaryRay")
        if not attribute.IsValid():
            attribute = light.CreateAttribute(
                "visibleInPrimaryRay", Sdf.ValueTypeNames.Bool
            )
        attribute.Set(False)
        print(
            "[PPO3-VISUAL] Isaac dark background enabled; dome illumination retained",
            flush=True,
        )

    def _spawn_visual_grid_floor(self) -> None:
        """Spawn a non-colliding gray Isaac-style floor with metric grid lines."""
        import numpy as np
        import trimesh

        length = float(getattr(self.cfg, "visual_grid_length_m", 50.0))
        width = float(getattr(self.cfg, "visual_grid_width_m", 18.0))
        spacing = float(getattr(self.cfg, "visual_grid_spacing_m", 2.0))
        line_width = float(getattr(self.cfg, "visual_grid_line_width_m", 0.025))
        center_x = float(getattr(self.cfg, "visual_grid_center_x_m", -20.0))
        center_y = float(getattr(self.cfg, "visual_grid_center_y_m", 3.0))
        if min(length, width, spacing, line_width) <= 0.0:
            raise ValueError("visual grid dimensions and spacing must be positive")

        floor_pieces = []
        line_pieces = []
        for origin in self.scene.env_origins.detach().cpu().numpy():
            floor = trimesh.creation.box(extents=(length, width, 0.02))
            floor.apply_translation(
                (origin[0] + center_x, origin[1] + center_y, origin[2] - 0.04)
            )
            floor_pieces.append(floor)
            for x in np.arange(
                center_x - 0.5 * length,
                center_x + 0.5 * length + 0.5 * spacing,
                spacing,
            ):
                line = trimesh.creation.box(extents=(line_width, width, 0.006))
                line.apply_translation(
                    (origin[0] + x, origin[1] + center_y, origin[2] - 0.026)
                )
                line_pieces.append(line)
            for y in np.arange(
                center_y - 0.5 * width,
                center_y + 0.5 * width + 0.5 * spacing,
                spacing,
            ):
                line = trimesh.creation.box(extents=(length, line_width, 0.006))
                line.apply_translation(
                    (origin[0] + center_x, origin[1] + y, origin[2] - 0.026)
                )
                line_pieces.append(line)

        create_prim_from_mesh(
            VISUAL_FLOOR_ROOT,
            trimesh.util.concatenate(floor_pieces),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=tuple(
                    getattr(self.cfg, "visual_floor_color", (0.25, 0.26, 0.27))
                ),
                roughness=1.0,
            ),
        )
        create_prim_from_mesh(
            VISUAL_GRID_ROOT,
            trimesh.util.concatenate(line_pieces),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=tuple(
                    getattr(self.cfg, "visual_grid_color", (0.58, 0.59, 0.60))
                ),
                roughness=0.9,
            ),
        )
        for root in (VISUAL_FLOOR_ROOT, VISUAL_GRID_ROOT):
            mesh_prim = sim_utils.get_prim_at_path(f"{root}/mesh")
            sim_utils.define_collision_properties(
                mesh_prim.GetPrimPath(),
                sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            )
        print(
            f"[PPO3-VISUAL] gray grid floor enabled ({spacing:.1f} m spacing, visual-only)",
            flush=True,
        )

    def _disable_parent_ground(self) -> None:
        ground = sim_utils.get_prim_at_path("/World/ground")
        sim_utils.set_prim_visibility(ground, False)
        ground_mesh = sim_utils.get_prim_at_path("/World/ground/geometry/mesh")
        sim_utils.define_collision_properties(
            ground_mesh.GetPrimPath(),
            sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        )

    def _spawn_flat_road(self) -> None:
        """Create one merged, straight asphalt road under all truck paths."""
        import trimesh

        start_x = float(getattr(self.cfg, "road_start_x_m", -34.0))
        length = float(getattr(self.cfg, "road_length_m", 24.0))
        center_y = float(getattr(self.cfg, "road_center_y_m", 4.30))
        width = float(getattr(self.cfg, "road_width_m", 3.60))
        if length <= 0.0 or width <= 0.0:
            raise ValueError("road length and width must be positive")
        pieces = []
        for origin in self.scene.env_origins.detach().cpu().numpy():
            piece = trimesh.creation.box(extents=(length, width, 0.02))
            piece.apply_translation(
                (
                    float(origin[0] + start_x + 0.5 * length),
                    float(origin[1] + center_y),
                    float(origin[2] - 0.01),
                )
            )
            pieces.append(piece)
        mesh = trimesh.util.concatenate(pieces)
        create_prim_from_mesh(
            ROAD_ROOT,
            mesh,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=float(
                    getattr(self.cfg, "terrain_static_friction", 1.0)
                ),
                dynamic_friction=float(
                    getattr(self.cfg, "terrain_dynamic_friction", 1.0)
                ),
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=tuple(
                    getattr(self.cfg, "road_color", (0.11, 0.12, 0.13))
                ),
                roughness=float(getattr(self.cfg, "road_roughness", 0.92)),
                metallic=0.0,
            ),
        )
        print(
            f"[PPO3-ROAD] dark asphalt flat road: x=[{start_x:.2f}, "
            f"{start_x + length:.2f}] y=[{center_y - 0.5 * width:.2f}, "
            f"{center_y + 0.5 * width:.2f}] per environment",
            flush=True,
        )

    def _spawn_mow_strip(self) -> None:
        """Lay one verge strip alongside each environment's driving line.

        Spawned after ``clone_environments``, so the prims are created per
        environment explicitly rather than cloned -- the same approach the
        parent's ``_spawn_mowing_terrain`` uses for its side shelf.
        """
        import numpy as np
        import trimesh

        strip_cfg = self._mow_strip_cfg()
        origins = self.scene.env_origins.detach().cpu().numpy()
        vary_seed = bool(getattr(self.cfg, "randomize_terrain_per_env", True))
        seed_stride = int(getattr(self.cfg, "terrain_seed_stride", 7919))

        # ONE merged mesh for all environments, not one prim per environment.
        # RayCasterCfg.mesh_prim_paths resolves to a single mesh, so per-env prims
        # meant only strip_0 was ever ray-cast against: with 4 envs the height
        # scan reported exactly 0.25 valid, i.e. one env seeing ground and three
        # seeing nothing. Merging keeps every environment visible to the sensor.
        pieces = []
        line_pieces = []
        path_y_samples = []
        path_family_names = []
        terrain_height_scales = []
        requested_family = str(
            getattr(self.cfg, "tracking_path_family", "mixed")
        )
        if requested_family != "mixed" and requested_family not in TRACKING_PATH_FAMILIES:
            raise ValueError(
                f"tracking_path_family must be 'mixed' or one of "
                f"{TRACKING_PATH_FAMILIES}, got {requested_family!r}"
            )
        family_offset = int(getattr(self.cfg, "tracking_path_family_offset", 0))
        configured_height_scales = tuple(
            float(value)
            for value in getattr(
                self.cfg, "terrain_height_scales", (0.50, 0.70, 0.85, 1.00)
            )
        )
        height_override = float(
            getattr(self.cfg, "terrain_height_scale_override", 0.0)
        )
        for env_index, origin in enumerate(origins):
            family = (
                TRACKING_PATH_FAMILIES[
                    (env_index + family_offset) % len(TRACKING_PATH_FAMILIES)
                ]
                if requested_family == "mixed"
                else requested_family
            )
            env_strip_cfg = (
                replace(
                    strip_cfg,
                    seed=strip_cfg.seed + env_index * seed_stride,
                    tracking_path_family=family,
                )
                if vary_seed
                else replace(strip_cfg, tracking_path_family=family)
            )
            # Height difficulty is independent of lateral family. Because 11
            # families and four scales are coprime, consecutive environments
            # cover every family x height pair exactly once per 44 envs.
            height_scale = terrain_height_scale_for_environment(
                configured_height_scales, env_index, height_override
            )
            env_strip_cfg = replace(
                env_strip_cfg,
                elevation_rise_end_m=(
                    env_strip_cfg.elevation_rise_end_m * height_scale
                ),
                hill_heights_m=tuple(
                    value * height_scale for value in env_strip_cfg.hill_heights_m
                ),
                amplitude_start_m=env_strip_cfg.amplitude_start_m * height_scale,
                amplitude_end_m=env_strip_cfg.amplitude_end_m * height_scale,
                cross_slope_end_rad=(
                    env_strip_cfg.cross_slope_end_rad * height_scale
                ),
            )
            terrain_height_scales.append(height_scale)
            piece = build_mow_strip_mesh(env_strip_cfg)
            piece.apply_translation(
                (float(origin[0]), float(origin[1]), float(origin[2]))
            )
            pieces.append(piece)
            line = build_tracking_line_mesh(
                env_strip_cfg,
                width_m=float(getattr(self.cfg, "tracking_line_width_m", 0.07)),
                z_offset_m=float(
                    getattr(self.cfg, "tracking_line_z_offset_m", 0.012)
                ),
            )
            line.apply_translation(
                (float(origin[0]), float(origin[1]), float(origin[2]))
            )
            line_pieces.append(line)
            path_x = env_strip_cfg.start_x_m + np.linspace(
                0.0,
                env_strip_cfg.length_m,
                max(
                    int(
                        round(
                            env_strip_cfg.length_m
                            / env_strip_cfg.resolution_m
                        )
                    )
                    + 1,
                    2,
                ),
            )
            path_y_samples.append(
                tracking_line_center_y_at(env_strip_cfg, path_x)
            )
            path_family_names.append(family)
        mesh = trimesh.util.concatenate(pieces)
        line_mesh = trimesh.util.concatenate(line_pieces)
        self._tracking_path_x_min = float(path_x[0])
        self._tracking_path_x_max = float(path_x[-1])
        self._tracking_path_y_samples = torch.as_tensor(
            np.stack(path_y_samples), dtype=torch.float32, device=self.device
        )
        self._tracking_path_family_names = tuple(path_family_names)
        self._terrain_height_scales = tuple(terrain_height_scales)

        physics_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=float(getattr(self.cfg, "terrain_static_friction", 1.0)),
            dynamic_friction=float(getattr(self.cfg, "terrain_dynamic_friction", 1.0)),
            restitution=float(getattr(self.cfg, "terrain_restitution", 0.0)),
        )
        visual_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=tuple(getattr(self.cfg, "mow_strip_color", (0.20, 0.45, 0.18)))
        )

        create_prim_from_mesh(
            MOW_STRIP_ROOT,
            mesh,
            physics_material=physics_material,
            visual_material=visual_material,
        )
        if not bool(getattr(self.cfg, "mow_strip_collision", False)):
            mesh_prim = sim_utils.get_prim_at_path(f"{MOW_STRIP_ROOT}/mesh")
            sim_utils.define_collision_properties(
                mesh_prim.GetPrimPath(),
                sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            )
        create_prim_from_mesh(
            TRACKING_LINE_ROOT,
            line_mesh,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=tuple(
                    getattr(self.cfg, "tracking_line_color", (0.95, 0.08, 0.05))
                )
            ),
        )
        line_prim = sim_utils.get_prim_at_path(f"{TRACKING_LINE_ROOT}/mesh")
        sim_utils.define_collision_properties(
            line_prim.GetPrimPath(),
            sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        )
        family_counts = {
            family: path_family_names.count(family)
            for family in sorted(set(path_family_names))
        }
        height_scale_counts = {
            f"{scale:.2f}": terrain_height_scales.count(scale)
            for scale in sorted(set(terrain_height_scales))
        }
        print(
            f"[PPO3-VERGE] verge strip: {len(mesh.faces):,} triangles across "
            f"{len(origins)} environments (single merged mesh, "
            f"{'per-env seeds' if vary_seed else 'shared seed'}; paths="
            f"{family_counts}; height_scales={height_scale_counts})",
            flush=True,
        )

    def _add_ground_sensors(self) -> None:
        """Ray-cast height scan and contact sensing on the cutter.

        The height scan is what replaces the trajectory library: the reference
        comes from the measured surface, so the policy generalizes to ground it
        has never seen. Rays are cast only against the verge strip, not the road,
        so the scan reads the surface being mowed.
        """
        # Mounted on the CHASSIS, not the cutter. The Messerkopf's local +z points
        # world-DOWN in the head-down pose (alignment 0.9988), so a (0,0,+2) ray
        # origin offset on that body is not reliably above the ground -- the same
        # rotation trap that made _compute_cutter_pos_w disagree with
        # _compute_cutter_world_z by 1.58 m.
        #
        # The offset places the scan over the cutter's nominal track, measured at
        # truck-frame (6.20, -2.82). Anchoring to the vehicle rather than the arm
        # is also the more faithful model: a real machine carries the sensor on
        # the truck, not on the moving tool.
        #
        # RayCaster applies its OffsetCfg through the body's FULL orientation
        # before ray_alignment is considered. On this USD, base_link's frame can
        # rotate during deployment even while its XY translation stays on the
        # scripted line. The old OffsetCfg therefore swung the scan sideways:
        # measured ray Y changed from [-0.50, 0.49] to [6.25, 8.10] m while the
        # verge ends at y=3.38 m, dropping validity from 1.0 to 0.0. Put the
        # horizontal/vertical displacement in the world-aligned pattern instead.
        # The sensor still follows base_link translation, but chassis attitude
        # can no longer rotate the scan away from path B.
        scan_offset = tuple(
            getattr(self.cfg, "height_scan_base_offset", (6.20, -2.82, 2.0))
        )
        heading = torch.as_tensor(
            tuple(getattr(self.cfg, "drive_direction_xy", (1.0, 0.0))),
            dtype=torch.float64,
        )
        heading = heading / heading.norm().clamp(min=1.0e-8)
        left = torch.stack((-heading[1], heading[0]))
        path_offset_xy = scan_offset[0] * heading + scan_offset[1] * left
        path_offset = (
            float(path_offset_xy[0]),
            float(path_offset_xy[1]),
            float(scan_offset[2]),
        )
        self._height_scanner = RayCaster(
            RayCasterCfg(
                prim_path=f"{self.cfg.robot_cfg.prim_path}/Rigid_Bodies/base_link",
                offset=RayCasterCfg.OffsetCfg(),
                mesh_prim_paths=[MOW_STRIP_ROOT],
                ray_alignment="world",
                pattern_cfg=PathAlignedGridPatternCfg(
                    resolution=float(getattr(self.cfg, "height_scan_resolution_m", 0.25)),
                    size=(
                        float(getattr(self.cfg, "height_scan_length_m", 2.0)),
                        float(getattr(self.cfg, "height_scan_width_m", 0.5)),
                    ),
                    # Width must be the inner loop: verge_height_profile groups
                    # each contiguous width row into one along-track bin.
                    ordering="yx",
                    center=path_offset,
                ),
                debug_vis=False,
            )
        )
        self.scene.sensors["height_scanner"] = self._height_scanner

        self._cutter_contact = ContactSensor(
            ContactSensorCfg(
                prim_path=self._cutter_prim_path(),
                update_period=0.0,
                history_length=3,
                track_contact_points=True,
                max_contact_data_count_per_prim=128,
                filter_prim_paths_expr=[f"{MOW_STRIP_ROOT}/mesh"],
                debug_vis=False,
            )
        )
        self.scene.sensors["cutter_contact"] = self._cutter_contact

        self._wheel_road_contacts = []
        for wheel_name in (
            "wheel_rear_left",
            "wheel_rear_right",
            "wheel_front_left",
            "wheel_front_right",
        ):
            sensor = ContactSensor(
                ContactSensorCfg(
                    prim_path=(
                        f"{self.cfg.robot_cfg.prim_path}/Rigid_Bodies/Wheels/"
                        f"{wheel_name}"
                    ),
                    update_period=0.0,
                    history_length=3,
                    track_contact_points=True,
                    max_contact_data_count_per_prim=32,
                    filter_prim_paths_expr=[f"{ROAD_ROOT}/mesh"],
                    debug_vis=False,
                )
            )
            self._wheel_road_contacts.append(sensor)
            self.scene.sensors[f"road_contact_{wheel_name}"] = sensor

    def _pre_physics_step(self, actions):
        """Advance the drive clock exactly once per environment step.

        Travel is accumulated HERE, not in _constrain_truck_xy, because this
        method runs exactly once per env step while _constrain_truck_xy runs
        several times. Deriving travel from the parent's `_scored_elapsed`
        instead looked clean but broke under the production deployment ramp:
        `_reset_idx` zeroes that counter, so travel stayed near zero and the
        truck covered 1.67 m of a commanded 5.55 m.
        """
        contact_demo = bool(
            getattr(self.cfg, "enable_contact_tracking_demo", False)
        )
        super()._pre_physics_step(
            torch.zeros_like(actions) if contact_demo else actions
        )
        if contact_demo:
            self._apply_contact_tracking_demo_targets()
        self._init_kinematic_drive()
        speed_kmh = getattr(self, "_vehicle_speed_kmh", None)
        if speed_kmh is None:
            return
        active = (self._preposition_counter <= 0) & (self._settle_counter <= 0)
        step = (speed_kmh / 3.6) * float(self.step_dt)
        self._truck_travel_m = self._truck_travel_m + torch.where(
            active, step, torch.zeros_like(step)
        )

    @torch.no_grad()
    def _initialize_contact_tracking_demo(self, *, apply_compliance: bool = True) -> None:
        """Latch the privileged terrain/contact controller state.

        ``apply_compliance`` is reserved for the visual physics demonstration.
        The training teacher must use exactly the same articulation gains as the
        teacher-free student, otherwise distillation would change the plant at
        hand-off.
        """
        env_ids = torch.arange(self.num_envs, device=self.device)
        # The production hydraulic-lock gains are intentionally very stiff and
        # turn small terrain-height errors into impact/separation cycles.  A
        # real adaptive mower head has compliance.  Soften only this non-PPO
        # physics demonstration; normal training continues to use the original
        # articulation gains.
        compliance_scale = torch.as_tensor(
            getattr(
                self.cfg,
                "contact_demo_arm_stiffness_scale",
                (1.0, 0.15, 0.15, 0.25, 0.25),
            ),
            dtype=self._terrain_nominal_arm_stiffness.dtype,
            device=self.device,
        )
        if compliance_scale.numel() != self._terrain_held_joint_ids.numel():
            raise ValueError(
                "contact_demo_arm_stiffness_scale must contain one value for "
                "each of the four arm cylinders plus the passive head joint"
            )
        if apply_compliance:
            self.robot.write_joint_stiffness_to_sim(
                self._terrain_nominal_arm_stiffness * compliance_scale.reshape(1, -1),
                joint_ids=self._terrain_held_joint_ids,
            )
        self._cascade.reset(env_ids)
        self._cascade.ee_feedback_gain = 0.35
        self._cascade.jacobian_refresh_every = 1
        # Cylinder 4 is reserved for cutter-head pitch. The old full 3x4
        # pseudoinverse assigned part of XYZ correction to c4 and then replaced
        # that channel with the fixed head target below, silently discarding the
        # correction. Solve translation with c1-c3 only so the privileged
        # teacher can actually hold a lateral line while height changes.
        j_xyz = self._cascade.jacobian[:, :3].detach().cpu()
        u, singular_values, vh = torch.linalg.svd(j_xyz, full_matrices=False)
        damping = 0.03 * singular_values[0]
        pinv_xyz = (
            vh.transpose(0, 1)
            @ torch.diag(
                singular_values
                / (singular_values.square() + damping.square())
            )
            @ u.transpose(0, 1)
        )
        self._contact_demo_xyz_pinv = pinv_xyz.to(
            device=self.device, dtype=torch.float32
        )
        ee = self._compute_ee_pos_local().detach()
        measured_x, _ = self._refresh_cylinder_state()
        self._contact_demo_anchor_ee = ee.clone()
        self._contact_demo_anchor_ground = (
            self.ground_height_under_cutter().detach().clone()
        )
        self._contact_demo_vertical_offset = torch.full(
            (self.num_envs,),
            float(
                getattr(
                    self.cfg,
                    "contact_demo_initial_vertical_offset_m",
                    -0.075,
                )
            ),
            dtype=torch.float32,
            device=self.device,
        )
        self._contact_demo_lateral_error = (
            self.verge_lateral_error().detach().clone()
        )
        self._contact_demo_anchor_line_y = (
            self.tracking_line_y_local().detach().clone()
        )
        self._contact_demo_c4_target = measured_x[:, 3].detach().clone()
        self._contact_demo_contact_memory = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._contact_demo_acquired = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._contact_demo_acquired_offset = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self._contact_demo_ready = True
        print(
            "[PPO3-CONTACT-DEMO] non-PPO controller enabled: terrain-height "
            "feed-forward + cutter/terrain force feedback",
            flush=True,
        )

    @torch.no_grad()
    def _apply_contact_tracking_demo_targets(self) -> None:
        """Keep the cutter on the terrain centre line with contact feedback.

        This controller is intentionally outside PPO and is enabled only by the
        visualization/physics probe.  It proves that the road, collision mesh,
        contact sensor and arm mechanism can realize the requested task before
        any learned policy is allowed to train.
        """
        target = self._compute_privileged_contact_target(apply_compliance=True)
        full_mask = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._write_direct_learned_targets(full_mask, target)
        self._current_action.zero_()
        self._ppo_raw_action.zero_()
        self._sole_applied.zero_()

    @torch.no_grad()
    def _compute_privileged_contact_target(
        self, *, apply_compliance: bool
    ) -> torch.Tensor:
        """Return the adaptive teacher's desired learned cylinder extensions.

        This is the one privileged controller shared by the visual proof and
        terrain distillation.  It may read the exact terrain height and contact
        force.  Only the visual proof writes this target directly; the training
        teacher converts it into the student's four-action interface below.
        """
        if not self._contact_demo_ready:
            self._initialize_contact_tracking_demo(
                apply_compliance=apply_compliance
            )

        force = self.cutter_terrain_contact_force_n()
        threshold = float(
            getattr(self.cfg, "cutter_contact_force_threshold_n", 10.0)
        )
        target_force = float(
            getattr(self.cfg, "contact_demo_target_force_n", 250.0)
        )
        deadband = float(
            getattr(self.cfg, "contact_demo_force_deadband_n", 100.0)
        )
        search_step = float(
            getattr(self.cfg, "contact_demo_search_step_m", 0.0010)
        )
        force_step = float(
            getattr(self.cfg, "contact_demo_force_step_m", 0.00025)
        )
        in_contact = force > threshold
        memory_steps = int(
            getattr(self.cfg, "contact_demo_memory_steps", 12)
        )
        self._contact_demo_contact_memory = torch.where(
            in_contact,
            torch.full_like(self._contact_demo_contact_memory, memory_steps),
            torch.clamp(self._contact_demo_contact_memory - 1, min=0),
        )
        newly_acquired = in_contact & ~self._contact_demo_acquired
        self._contact_demo_acquired |= in_contact
        no_contact = (
            ~self._contact_demo_acquired
            & ~in_contact
            & (self._contact_demo_contact_memory == 0)
        )
        # Once contact is acquired, keep regulating it.  The previous masks
        # included ``~self._contact_demo_acquired`` after the flag had already
        # been latched above, so both the low- and high-force feedback branches
        # were unreachable.  That left only the coarse reseat search and made
        # the head repeatedly strike and leave the terrain.
        low_force = (
            self._contact_demo_acquired
            & in_contact
            & (force < target_force - deadband)
        )
        high_force = (
            self._contact_demo_acquired
            & (force > target_force + deadband)
        )
        correction = torch.zeros_like(force)
        correction = torch.where(
            no_contact, torch.full_like(correction, -search_step), correction
        )
        preload = float(getattr(self.cfg, "contact_demo_preload_m", 0.01))
        correction = torch.where(
            newly_acquired,
            torch.full_like(correction, -abs(preload)),
            correction,
        )
        correction = torch.where(
            low_force,
            torch.full_like(correction, -force_step),
            correction,
        )
        correction = torch.where(
            high_force,
            torch.clamp(
                force_step * force / max(target_force, 1.0),
                min=force_step,
                max=0.002,
            ),
            correction,
        )
        reseat_step = float(
            getattr(self.cfg, "contact_demo_reseat_step_m", 0.00025)
        )
        max_reseat = float(
            getattr(self.cfg, "contact_demo_max_reseat_m", 0.04)
        )
        support_lost = (
            self._contact_demo_acquired
            & ~in_contact
            & (self._contact_demo_contact_memory == 0)
            & (
                self._contact_demo_vertical_offset
                > self._contact_demo_acquired_offset - max_reseat
            )
        )
        correction = torch.where(
            support_lost,
            torch.full_like(correction, -abs(reseat_step)),
            correction,
        )
        hard_force = float(
            getattr(self.cfg, "contact_demo_hard_force_n", 50000.0)
        )
        relief_step = float(
            getattr(
                self.cfg,
                "contact_demo_hard_force_relief_step_m",
                0.001,
            )
        )
        correction = torch.where(
            force > hard_force,
            torch.full_like(correction, abs(relief_step)),
            correction,
        )
        max_offset = float(
            getattr(self.cfg, "contact_demo_max_vertical_offset_m", 0.30)
        )
        max_up_offset = float(
            getattr(self.cfg, "contact_demo_max_upward_offset_m", -0.05)
        )
        self._contact_demo_vertical_offset = torch.clamp(
            self._contact_demo_vertical_offset + correction,
            min=-max_offset,
            max=max_up_offset,
        )
        self._contact_demo_acquired_offset = torch.where(
            newly_acquired,
            self._contact_demo_vertical_offset.detach(),
            self._contact_demo_acquired_offset,
        )

        ground_delta = (
            self.ground_height_under_cutter()
            - self._contact_demo_anchor_ground
        )
        goal = self._contact_demo_anchor_ee.clone()
        # The road heading is fixed, so truck-frame lateral and world lateral
        # have the same sign. Remove the initial offset to sit over the red line.
        goal[:, 1] = (
            self._contact_demo_anchor_ee[:, 1]
            - self._contact_demo_lateral_error
            + self.tracking_line_y_local()
            - self._contact_demo_anchor_line_y
            + float(getattr(self.cfg, "contact_demo_lateral_load_bias_m", -0.03))
        )
        goal[:, 2] = (
            self._contact_demo_anchor_ee[:, 2]
            + ground_delta
            + self._contact_demo_vertical_offset
        )
        measured_ee = self._compute_ee_pos_local()
        measured_x, _ = self._refresh_cylinder_state()
        dls_gain = float(getattr(self.cfg, "contact_demo_dls_gain", 0.20))
        max_cylinder_step = float(
            getattr(self.cfg, "contact_demo_max_cylinder_step_m", 0.0015)
        )
        task_error = goal - measured_ee
        task_error[:, 1] *= float(
            getattr(self.cfg, "contact_demo_lateral_error_weight", 3.0)
        )
        learned_delta = torch.zeros_like(measured_x)
        learned_delta[:, :3] = dls_gain * (
            task_error @ self._contact_demo_xyz_pinv.transpose(0, 1)
        )
        learned_delta = torch.clamp(
            learned_delta,
            min=-max_cylinder_step,
            max=max_cylinder_step,
        )
        target = measured_x + learned_delta

        # Cylinder 4 maintains the head-down orientation while the other three
        # follow terrain height. Its signed response direction was measured in
        # the live mechanism and is encoded by terrain_c4_action_to_axis_y_sign.
        # Keep c4 on the stable deployed head pose during the physics proof.
        # PPO later receives signed head-axis feedback for adaptive c4 control.
        target[:, 3] = self._contact_demo_c4_target

        return target

    @torch.no_grad()
    def _compute_teacher_action(self) -> torch.Tensor:
        """Privileged adaptive teacher expressed through the student interface.

        The teacher first computes the desired cylinder extension target, then
        maps its one-step learned-space correction to the same normalized
        command consumed by ROM-A.  Finally it runs the student's EMA, slew and
        safety shields.  Therefore ``teacher_action`` is a reachable *applied*
        command, while ``terrain_teacher_raw_action`` is the exact BC label.
        No direct PhysX target write is used in teacher mode.
        """
        target = self._compute_privileged_contact_target(
            apply_compliance=False
        )
        measured, _ = self._refresh_cylinder_state()
        desired_learned_delta = target - measured
        desired_usd_delta = (
            self._cylinder_adapter.scaling.learned_delta_to_usd_delta(
                desired_learned_delta
            )
        )
        action_scale = max(float(self.cfg.action_scale), 1.0e-9)
        raw_limit = min(
            max(float(getattr(self.cfg, "terrain_teacher_max_raw_action", 0.60)), 0.0),
            1.0,
        )
        raw_teacher = torch.clamp(
            desired_usd_delta / action_scale, -raw_limit, raw_limit
        )
        head_axis = self._head_down_axis_world()
        # atan2 keeps the error directional even beyond 90 degrees. Using only
        # axis_y becomes zero again when the head is upside down and allowed a
        # long-horizon controller to mistake the inverted pose for the goal.
        signed_tilt = torch.atan2(head_axis[:, 1], -head_axis[:, 2])
        tilt_response_sign = float(
            getattr(self.cfg, "terrain_c4_action_to_axis_y_sign", 1.0)
        )
        head_gain = float(getattr(self.cfg, "terrain_teacher_head_gain", 4.0))
        raw_teacher[:, 3] = torch.clamp(
            raw_teacher[:, 3]
            - head_gain * signed_tilt / tilt_response_sign,
            -raw_limit,
            raw_limit,
        )

        low, high = self._cylinder_headroom(measured)
        output = apply_sole_action(
            policy_action=raw_teacher,
            # ``SolePolicyEnv`` computes the student's candidate first and
            # latches the action that actually drove the previous transition in
            # ``_prev_sole_applied``.  Teacher reachability must start from that
            # real previous command, not from the just-computed student guess.
            previous_applied_action=self._prev_sole_applied,
            filter_beta=float(getattr(self.cfg, "sole_action_filter", 0.75)),
            max_action_rate=float(getattr(self.cfg, "sole_max_action_rate", 0.20)),
            headroom_low_m=low,
            headroom_high_m=high,
            limit_shield_margin_m=float(
                getattr(self.cfg, "sole_limit_shield_margin_m", 0.0)
            ),
            action_to_extension_sign=self._action_to_learned_sign,
            action_authority=getattr(
                self.cfg, "sole_action_authority", (1.0,) * 4
            ),
        )
        applied, task_gain = self._task_action_shield(output.applied)
        self._terrain_teacher_raw_action = raw_teacher.detach().clone()
        self._terrain_teacher_shield_gain = (
            output.shield_gain * task_gain
        ).detach().clone()
        return applied

    @property
    def terrain_teacher_raw_action(self) -> torch.Tensor:
        """Exact raw actor label for the last privileged teacher query."""
        if self._teacher_mode == "off":
            raise RuntimeError("terrain teacher labels are disabled")
        return self._terrain_teacher_raw_action

    # ------------------------------------------------------------------
    # Truck: driving, not parked
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _hold_wheels(self) -> None:
        """Spin the road-supported wheels at the scripted rolling rate.

        The chassis is advanced kinematically (see ``_constrain_truck_xy``), so
        the wheel-road contacts are physical while propulsion remains scripted.
        Three attempts at driving through the
        wheels are recorded in git history; measured against a 4.00 km/h command
        they gave 0.135 km/h (parking brake), 0.300 km/h (velocity drive, one
        wheel of four obeying), and 0.135 km/h again (teacher_student's gains).
        Since the vehicle is scripted and never part of the policy, none of that
        fidelity buys anything the task needs.
        """
        ids = self._wheel_joint_ids
        speed_kmh = getattr(self, "_vehicle_speed_kmh", None)
        if speed_kmh is None:
            return
        radius = max(float(getattr(self.cfg, "vehicle_wheel_radius_m", 0.575)), 1e-6)
        omega = (speed_kmh / 3.6 / radius).reshape(-1, 1).expand(-1, len(ids))
        self.robot.write_joint_velocity_to_sim(omega, joint_ids=ids)
        # `_steering_init_pos` is latched after the articulation has settled and
        # can already contain visible toe/steer deflection.  Straight driving
        # means the authored steering-joint zero, not that displaced snapshot.
        steering_target = torch.full(
            (self.num_envs, len(self._steering_joint_ids)),
            float(getattr(self.cfg, "vehicle_steering_target_rad", 0.0)),
            dtype=self.robot.data.joint_pos.dtype,
            device=self.device,
        )
        self.robot.set_joint_position_target(
            steering_target, joint_ids=self._steering_joint_ids
        )
        self.robot.write_joint_position_to_sim(
            steering_target, joint_ids=self._steering_joint_ids
        )
        self.robot.write_joint_velocity_to_sim(
            torch.zeros_like(steering_target), joint_ids=self._steering_joint_ids
        )

    @torch.no_grad()
    def _constrain_truck_xy(self) -> None:
        """Advance the chassis along the road at exactly the commanded speed.

        The parent pins the truck by rewriting the root pose every step. This is
        that same fixture with a constant translation added, which is why it sits
        closer to PPO2's validated behaviour than driving through the wheels
        would: every PPO2 gate was measured with the chassis held this way.

        Consequences, stated plainly: wheel-road contact is simulated and
        measured, while the fixture rejects arm-induced chassis translation and
        rotation. This buys an exactly straight constant-speed road path without
        pretending that an airborne wheel is grounded.

        Advance happens only outside warmup, so the reset preposition still
        deploys the arm from a stationary truck.
        """
        self._init_kinematic_drive()

        # Correct the root by the CHASSIS error, never set the root absolutely.
        # The articulation root is the Drehzapfen -- the arm's own slew base --
        # not the chassis, so pinning the root to an absolute pose lets the
        # chassis swing around it as the arm moves: measured 6.36 m of lateral
        # drift while the cutter never advanced in x. Driving the correction off
        # base_link error is what the parent does, and it is idempotent, so
        # repeated calls converge on the target instead of accumulating.

        heading = torch.as_tensor(
            tuple(getattr(self.cfg, "drive_direction_xy", (1.0, 0.0))),
            dtype=torch.float32,
            device=self.device,
        ).reshape(1, 2)
        heading = heading / heading.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        desired_base_xy = (
            self._drive_anchor_base_xy
            + self._truck_travel_m.reshape(-1, 1) * heading
        )
        base_state = self.robot.data.body_link_state_w[
            :, self._truck_base_body_id()
        ]
        base_xy = base_state[:, :2]
        correction = desired_base_xy - base_xy
        base_z = self.robot.data.body_link_state_w[
            :, self._truck_base_body_id(), 2
        ]
        target_base_z = float(
            getattr(self.cfg, "truck_ride_height_target_m", 0.1446)
        )

        root_pose = self.robot.data.root_state_w[:, :7].detach().clone()
        base_heading = self._truck_base_heading_xy()
        anchor_heading = self._drive_anchor_base_heading_xy
        heading_cross = (
            anchor_heading[:, 0] * base_heading[:, 1]
            - anchor_heading[:, 1] * base_heading[:, 0]
        )
        heading_dot = torch.sum(anchor_heading * base_heading, dim=-1).clamp(
            min=-1.0, max=1.0
        )
        heading_error = torch.atan2(heading_cross, heading_dot)
        self._truck_heading_error_rad = heading_error.abs()

        # Translation/yaw alone still allowed arm reactions to roll and pitch the
        # truck enough to unload tyres. Rotate the articulation root around
        # base_link back to the complete latched road attitude. The articulation
        # root is Drehzapfen, so the lever arm must rotate with the quaternion or
        # the correction itself would move the chassis.
        base_quat = base_state[:, 3:7]
        base_quat = base_quat / base_quat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        anchor_quat = self._drive_anchor_base_quat
        anchor_quat = anchor_quat / anchor_quat.norm(
            dim=-1, keepdim=True
        ).clamp(min=1e-8)
        delta_q = quat_mul(anchor_quat, quat_inv(base_quat))
        root_relative_to_base = root_pose[:, :3] - base_state[:, :3]
        rotated_relative = quat_apply(delta_q, root_relative_to_base)
        root_pose[:, :3] = base_state[:, :3] + rotated_relative
        root_pose[:, 3:7] = quat_mul(delta_q, root_pose[:, 3:7])
        root_pose[:, :2] = root_pose[:, :2] + correction
        # The truck is scripted on a flat road, so hold the validated ride
        # height just as strictly as the straight XY path. Large cloned grids
        # otherwise settle at different suspension heights (0.229 m at 512 vs
        # 0.139 m at 2 envs), injecting an irrelevant base-height curriculum
        # into the arm task.
        root_pose[:, 2] = root_pose[:, 2] + (target_base_z - base_z)
        self.robot.write_root_pose_to_sim(root_pose)

        # Report the true translational velocity so anything reading root
        # velocity (observations, diagnostics) sees a moving vehicle rather than
        # a teleporting one.
        velocity = torch.zeros((self.num_envs, 6), device=self.device)
        speed_kmh = getattr(self, "_vehicle_speed_kmh", None)
        if speed_kmh is not None:
            velocity[:, :2] = (speed_kmh / 3.6).reshape(-1, 1) * heading
        self.robot.write_root_velocity_to_sim(velocity)

        # Lateral drift is identically zero by construction; keep the fields the
        # inherited termination logic reads.
        zero = torch.zeros((self.num_envs,), device=self.device)
        self._truck_drift_xy = zero
        self._truck_post_constraint_drift_xy = zero

    # ------------------------------------------------------------------
    # Ground queries
    # ------------------------------------------------------------------

    def _finite_hits(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Ray-hit heights and a validity mask.

        Missed rays come back non-finite. They are returned as a mask rather than
        silently replaced with zero: an earlier version scrubbed them with
        ``nan_to_num``, which turned "the sensor saw nothing" into "the ground is
        at z=0" and produced entirely plausible-looking numbers that were wrong.
        """
        z = self._height_scanner.data.ray_hits_w[..., 2]
        return z, torch.isfinite(z)

    def ground_height_under_cutter(self) -> torch.Tensor:
        """World z of the verge directly beneath the cutter. Returns ``(N,)``.

        The scan starts at the cutter and extends forward along path B. Use the
        first along-track row, collapsed across scan width, rather than the
        median of the entire 2 m preview. Scoring against the preview median
        makes the arm react to ground roughly 1 m ahead instead of ground under
        the cutter.

        Falls back to the strip's base height where the cutter row missed;
        ``scan_valid_fraction`` reports how often that happens, so a caller can
        tell a real measurement from a fallback.
        """
        z, valid = self._finite_hits()
        base = float(self._mow_strip_cfg().base_z_m)
        safe = torch.where(valid, z, torch.full_like(z, base))
        height_rows = collapse_scan_width(safe)
        valid_rows = collapse_scan_width(valid.float()) > 0.0
        cutter_height = height_rows[:, 0]
        cutter_valid = valid_rows[:, 0]
        return torch.where(
            cutter_valid, cutter_height, torch.full_like(cutter_height, base)
        )

    def scan_valid_fraction(self) -> torch.Tensor:
        """Fraction of height-scan rays that actually hit the verge. ``(N,)``."""
        _, valid = self._finite_hits()
        return valid.float().mean(dim=1)

    def ground_height_scan(self) -> torch.Tensor:
        """Verge height relative to the cutter, per ray. ``(N, R)``.

        Positive means the ground is below the cutter.
        """
        z, valid = self._finite_hits()
        base = float(self._mow_strip_cfg().base_z_m)
        safe = torch.where(valid, z, torch.full_like(z, base))
        cutter_z = self._cutter_world_z().reshape(-1, 1)
        return cutter_z - safe

    def _cutter_world_z(self) -> torch.Tensor:
        """World height of the cutting disc. ``(N,)``.

        Uses ``_compute_cutter_world_z`` (origin z plus the unrotated tool
        offset), NOT ``_compute_cutter_pos_w``. Measured 2026-07-28 on the
        deployed arm: origin at +0.944 m, unrotated cutter at +0.154 m, rotated
        cutter at +1.733 m, head-down alignment 0.9988.

        In the head-down pose the body's local +z points world-DOWN, so rotating
        the local -0.79 m tool offset flips it to +0.79 m and puts the cutter
        above the head. The 1.579 m gap between the two is exactly the
        frame inconsistency PPO2 documents in
        ``ResidualPolicyEnv._clearance_signal``.

        The unrotated convention is the one PPO1/PPO2 are built on -- the
        trajectory library's ``cutter_world_z`` column, the clearance gates and
        every validated mowing result use it -- so PPO3 uses it too. Mixing
        conventions would make PPO3's tracking numbers incomparable to the PPO2-A
        gates they are meant to be measured against.
        """
        return self._compute_cutter_world_z()

    def cutter_height_above_ground(self) -> torch.Tensor:
        """Cutter height above the verge beneath it. ``(N,)``.

        This is the quantity the mowing task regulates to ``target_cut_height_m``.
        """
        return self._cutter_world_z() - self.ground_height_under_cutter()

    def cutter_ground_contact(self) -> torch.Tensor:
        """Whether the cutter physically touches the green terrain mesh."""
        return self.cutter_terrain_contact_force_n() > float(
            getattr(self.cfg, "cutter_contact_force_threshold_n", 1.0)
        )

    @staticmethod
    def _filtered_contact_force_per_body(sensor) -> torch.Tensor:
        """Return maximum filtered force magnitude per sensed body."""
        force_history = sensor.data.force_matrix_w_history
        if force_history is not None:
            # (env, history, sensed body, filtered body, xyz).  DirectRLEnv has
            # two physics substeps per control step, so reading only the newest
            # sample can incorrectly call a real rolling/contact interaction
            # airborne when it occurred on the preceding substep.
            magnitude = torch.linalg.norm(force_history, dim=-1)
            return magnitude.amax(dim=(1, 3))
        force_matrix = sensor.data.force_matrix_w
        if force_matrix is not None:
            magnitude = torch.linalg.norm(force_matrix, dim=-1)
            while magnitude.ndim > 2:
                magnitude = magnitude.amax(dim=-1)
            return magnitude
        forces = sensor.data.net_forces_w
        if forces is None:
            raise RuntimeError("contact sensor has no force data")
        return torch.linalg.norm(forces, dim=-1)

    def cutter_terrain_contact_force_n(self) -> torch.Tensor:
        """Filtered cutter-to-green-terrain interaction force, in newtons."""
        per_body = self._filtered_contact_force_per_body(self._cutter_contact)
        return per_body.amax(dim=-1)

    def wheel_road_contact_forces_n(self) -> torch.Tensor:
        """Filtered yellow-road contact force for each of the four tyres."""
        return torch.stack(
            [
                self._filtered_contact_force_per_body(sensor).amax(dim=-1)
                for sensor in self._wheel_road_contacts
            ],
            dim=-1,
        )

    def wheel_road_contact_fraction(self) -> torch.Tensor:
        """Fraction of tyres supported by the asphalt road in each environment."""
        threshold = float(
            getattr(self.cfg, "road_contact_force_threshold_n", 10.0)
        )
        return (self.wheel_road_contact_forces_n() > threshold).float().mean(dim=-1)

    def terrain_safety_violation(self) -> torch.Tensor:
        """Raw per-environment safety predicate without synchronized reset side effects."""
        cfg = self.cfg
        cylinder_q = self.robot.data.joint_pos[:, self._cylinder_joint_ids]
        cylinder_qdot = self.robot.data.joint_vel[:, self._cylinder_joint_ids]
        limits = self.robot.data.soft_joint_pos_limits[:, self._cylinder_joint_ids]
        margin = float(getattr(cfg, "cylinder_limit_margin_m", 0.002))
        limit_violation = (
            (cylinder_q < limits[..., 0] - margin)
            | (cylinder_q > limits[..., 1] + margin)
        ).any(dim=-1)
        nonfinite = self._terrain_nonfinite_mask()
        truck_violation = (
            self._truck_post_constraint_drift_xy
            > float(getattr(cfg, "truck_drift_termination_m", 0.02))
        ) | (
            self._truck_drift_xy
            > float(getattr(cfg, "truck_raw_reaction_termination_m", 0.05))
        )
        head_violation = self._measure_head_down_alignment() < float(
            getattr(cfg, "head_down_termination_alignment", 0.98)
        )
        # The stored-path parent's absolute-world-Z clearance is not a terrain
        # safety signal. Use deviation from the sensed cut-height target here.
        clearance_violation = self.cut_height_error().abs() > float(
            getattr(cfg, "cutter_clearance_hard_band_m", 0.20)
        )
        warmup = (self._preposition_counter > 0) | (self._settle_counter > 0)
        return (
            limit_violation
            | nonfinite
            | truck_violation
            | head_violation
            | clearance_violation
        ) & ~warmup

    def _terrain_nonfinite_mask(self) -> torch.Tensor:
        """Detect corrupt raw simulation/control state before scrubbed features hide it."""
        tensors = [
            self.robot.data.root_state_w,
            self.robot.data.body_link_state_w,
            self.robot.data.joint_pos,
            self.robot.data.joint_vel,
        ]
        for name in (
            "_sole_applied",
            "_prev_sole_applied",
            "_current_valve_cmd",
            "_fd_delta_corrected",
            "_corrected_cylinder_goal",
            "_cylinder_pos_target",
            "_measured_tool_acceleration_truck",
            "_previous_measured_tool_velocity_truck",
            "_prev_tool_pos_truck",
            "_prev_cutter_world_z",
            "_prev_head_alignment",
            "_previous_abs_cut_error",
            "_previous_abs_lateral_error",
            "_truck_travel_m",
            "_drive_anchor_root",
            "_drive_anchor_base_xy",
            "_drive_anchor_base_heading_xy",
            "_drive_anchor_base_quat",
        ):
            value = getattr(self, name, None)
            if isinstance(value, torch.Tensor):
                tensors.append(value)
        if bool(getattr(self.cfg, "mow_strip_collision", False)):
            for sensor in (self._cutter_contact, *self._wheel_road_contacts):
                forces = getattr(sensor.data, "net_forces_w", None)
                if isinstance(forces, torch.Tensor):
                    tensors.append(forces)
        return nonfinite_row_mask(*tensors)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Make any raw terrain-state corruption terminal even in deferred mode.

        Physical preposition currently requires synchronized reset IDs, so an
        exceptional non-finite replica deliberately recovers the full vectorized
        scene. This is expensive but safe and, unlike a poisoned PPO batch,
        resumable. Ordinary finite safety excursions remain deferred.
        """
        terminated, truncated = super()._get_dones()
        nonfinite = self._terrain_nonfinite_mask() | getattr(
            self,
            "_terrain_step_nonfinite",
            torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
        )
        self._terrain_step_nonfinite = nonfinite
        terminated = terminated | nonfinite
        if bool(getattr(self.cfg, "synchronize_resets", True)) and nonfinite.any():
            terminated = torch.ones_like(terminated)
        self.extras.setdefault("log", {})["termination/terrain_nonfinite"] = (
            nonfinite.float().mean()
        )
        return terminated, truncated

    def truck_ride_height(self) -> torch.Tensor:
        """Chassis height above the road. ``(N,)``.

        The road is a flat plate at z=0, so this is simply the chassis height and
        needs no ray cast. It is the grounding check: a floating truck's value
        grows, a sunk one's goes negative.
        """
        return self.robot.data.body_link_state_w[:, self._truck_base_body_id(), 2]

    @torch.no_grad()
    def _run_terrain_deployment_correction(self, env_ids: torch.Tensor) -> None:
        """Finish reset at the sensed cut height, using reset-only DLS.

        The legacy arc safely unfolds the mechanism but its final world height
        depends on the chassis fixture and terrain. This correction is outside
        the scored loop and makes the reset contract explicit: cutter at sensed
        ground + target height, with the head orientation left to cylinder 4.
        """
        if env_ids.numel() != self.num_envs or self._cascade is None:
            return
        steps = int(getattr(self.cfg, "terrain_deployment_correction_steps", 120))
        settle_steps = int(getattr(self.cfg, "terrain_deployment_settle_steps", 30))
        if steps <= 0:
            return

        self._cascade.reset(env_ids)
        target_cut = float(getattr(self.cfg, "target_cut_height_m", 0.05))
        full_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        dt = float(self.sim.get_physics_dt())
        target = self._refresh_cylinder_state()[0]

        for _ in range(steps):
            ee_goal = self._compute_ee_pos_local().detach().clone()
            cut_correction = torch.clamp(
                target_cut - self.cutter_height_above_ground(), -0.02, 0.02
            )
            ee_goal[:, 2] = ee_goal[:, 2] + cut_correction
            target = self._cascade.cylinder_goal(ee_goal)
            self._write_direct_learned_targets(full_mask, target)
            for _ in range(int(getattr(self.cfg, "decimation", 2))):
                self._apply_action()
                self._hold_wheels()
                self.scene.write_data_to_sim()
                self.sim.step(render=False)
                self.scene.update(dt)
            self._constrain_truck_xy()

        self._write_direct_learned_targets(full_mask, target)
        for _ in range(settle_steps):
            for _ in range(int(getattr(self.cfg, "decimation", 2))):
                self._apply_action()
                self._hold_wheels()
                self.scene.write_data_to_sim()
                self.sim.step(render=False)
                self.scene.update(dt)
            self._constrain_truck_xy()

        self._preposition_target_x[env_ids] = target[env_ids]
        self._reset_control_state(env_ids)
        self._warm_command_rom_at_target(target)

    @torch.no_grad()
    def _synchronize_deployed_robot_state(self, env_ids: torch.Tensor) -> None:
        """Copy one stable deployed articulation state across cloned origins."""
        if (
            env_ids.numel() != self.num_envs
            or self.num_envs <= 1
            or not bool(getattr(self.cfg, "synchronize_deployed_robot_state", True))
        ):
            return
        # Never choose the numerically "best" clone.  Doing that made the reset
        # pose depend on vectorization width: 16-, 32-, 128- and 512-env runs
        # selected different winners, so capacity probes and held-out gates did
        # not evaluate the same initial mechanism state.  A fixed clone makes
        # deployment deterministic across training/evaluation batch sizes.
        template = int(
            getattr(self.cfg, "synchronized_deployment_template_env", 0)
        )
        if not 0 <= template < self.num_envs:
            raise ValueError(
                "synchronized_deployment_template_env="
                f"{template} outside [0, {self.num_envs})"
            )
        self._last_deployment_template = template
        self._last_deployment_start_s = float(self._traj_start_s[template])
        self._last_deployment_target_x = self._preposition_target_x[
            template
        ].detach().clone()

        root_pose = self.robot.data.root_state_w[:, :7].detach().clone()
        template_relative_pos = (
            root_pose[template, :3] - self.scene.env_origins[template]
        )
        root_pose[:, :3] = self.scene.env_origins + template_relative_pos
        root_pose[:, 3:7] = root_pose[template, 3:7]
        self.robot.write_root_pose_to_sim(root_pose)

        joint_pos = self.robot.data.joint_pos[template].detach().reshape(1, -1).expand(
            self.num_envs, -1
        ).clone()
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel)
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.scene.update(float(self.sim.get_physics_dt()))
        self._constrain_truck_xy()

    def _reset_idx(self, env_ids):
        import torch as _t
        if not isinstance(env_ids, _t.Tensor):
            env_ids = _t.as_tensor(env_ids, dtype=_t.long, device=self.device)

        self._restore_terrain_arm_deployment_damping()
        self._contact_demo_ready = False

        # The parent reset runs the physical deployment sequence, which calls
        # our overridden ``_constrain_truck_xy`` several times.  Clear the old
        # episode's travel *before* that sequence: otherwise those calls target
        # ``old_anchor + old_travel`` and translate the freshly reset robot (and
        # its ray scanner) farther down the static verge on every reset.
        drive_was_ready = getattr(self, "_kinematic_drive_ready", False)
        if drive_was_ready:
            self._truck_travel_m[env_ids] = 0.0

        super()._reset_idx(env_ids)
        # The first parent deployment calls _constrain_truck_xy(), which creates
        # the drive state. Re-check after super rather than returning based on
        # the pre-reset snapshot; otherwise the first explicit reset skips the
        # terrain correction, synchronization, and anchor latch entirely.
        if not getattr(self, "_kinematic_drive_ready", False):
            return
        if getattr(self, "_sole_applied", None) is not None:
            self._run_terrain_deployment_correction(env_ids)
            self._synchronize_deployed_robot_state(env_ids)
        # A new episode starts from the pose the arm was just deployed at.
        self._drive_anchor_root[env_ids] = self.robot.data.root_state_w[env_ids, :7].detach().clone()
        self._drive_anchor_base_xy[env_ids] = self.robot.data.body_link_state_w[
            env_ids, self._truck_base_body_id(), :2
        ].detach().clone()
        self._drive_anchor_base_heading_xy[env_ids] = self._truck_base_heading_xy()[
            env_ids
        ].detach().clone()
        self._drive_anchor_base_quat[env_ids] = self.robot.data.body_link_state_w[
            env_ids, self._truck_base_body_id(), 3:7
        ].detach().clone()
        self._apply_terrain_arm_drive_tuning()
        self._relatch_terrain_temporal_state(env_ids)
        if hasattr(self, "_terrain_step_nonfinite"):
            self._terrain_step_nonfinite[env_ids] = False

    def _relatch_terrain_temporal_state(self, env_ids: torch.Tensor) -> None:
        """Start derivative observations from the final synchronized reset pose."""
        tool_pos = self._compute_ee_pos_local().detach()
        if self._prev_tool_pos_truck is None:
            self._prev_tool_pos_truck = tool_pos.clone()
        else:
            self._prev_tool_pos_truck[env_ids] = tool_pos[env_ids]
        self._measured_tool_velocity_truck[env_ids] = 0.0
        self._previous_measured_tool_velocity_truck[env_ids] = 0.0
        self._measured_tool_acceleration_truck[env_ids] = 0.0
        # Relatching is already the kinematics sample for this step. Mark it as
        # current so DirectRLEnv's post-reset observation cannot differentiate
        # the synchronized pose a second time and manufacture a reset spike.
        self._tool_kinematics_common_step = int(
            getattr(self, "common_step_counter", -1)
        )

        cutter_z = self._cutter_world_z().detach()
        previous_z = getattr(self, "_prev_cutter_world_z", None)
        if previous_z is None:
            self._prev_cutter_world_z = cutter_z.clone()
        else:
            self._prev_cutter_world_z[env_ids] = cutter_z[env_ids]

        alignment = self._measure_head_down_alignment().detach()
        self._prev_head_alignment[env_ids] = alignment[env_ids]
        self._head_alignment_rate[env_ids] = 0.0
        if not hasattr(self, "_previous_abs_cut_error"):
            self._previous_abs_cut_error = torch.zeros(
                self.num_envs, dtype=torch.float32, device=self.device
            )
        if not hasattr(self, "_previous_abs_lateral_error"):
            self._previous_abs_lateral_error = torch.zeros(
                self.num_envs, dtype=torch.float32, device=self.device
            )
        self._previous_abs_cut_error[env_ids] = self.cut_height_error()[
            env_ids
        ].abs()
        self._previous_abs_lateral_error[env_ids] = self.verge_lateral_error()[
            env_ids
        ].abs()

    # ------------------------------------------------------------------
    # Verge-mowing task: reference, observation, reward
    # ------------------------------------------------------------------

    def verge_height_profile(self, bins: int = HEIGHT_PROFILE_BINS) -> torch.Tensor:
        """Verge height ahead of the cutter, relative to it. ``(N, bins)``.

        The scan grid is (length x width); the width is collapsed by median so a
        single stone under one ray cannot swing the profile, leaving one height
        per along-track position.
        """
        rel = self.ground_height_scan()
        return collapse_scan_width(rel, bins=bins)

    def cut_height_error(self) -> torch.Tensor:
        """Physical cutter-to-ground contact-gap error. ``(N,)``.

        The legacy unrotated cutter reference lies about 0.26 m inside the
        collision geometry when the real mower head is supported by the verge.
        Treating that coordinate as a literal zero-height point made valid
        physical contact look like a 26 cm tracking failure.  Contact is the
        task contract, so a measured interaction is zero error.  Off ground,
        the legacy height relative to its calibrated contact reference provides
        a continuous positive search signal.
        """
        reference = float(
            getattr(self.cfg, "cutter_contact_reference_height_m", -0.30)
        )
        geometric_gap = torch.relu(
            self.cutter_height_above_ground() - reference
        )
        return torch.where(
            self.cutter_ground_contact(),
            torch.zeros_like(geometric_gap),
            geometric_gap,
        )

    def verge_lateral_error(self) -> torch.Tensor:
        """Cutter offset from the projected curved tracking line. ``(N,)``."""
        cutter_y = self._compute_cutter_pos_w()[:, 1] - self.scene.env_origins[:, 1]
        return cutter_y - self.tracking_line_y_local()

    def tracking_line_y_local(self) -> torch.Tensor:
        """Interpolate the exact per-environment path used by the red mesh."""
        cutter_x = (
            self._compute_cutter_pos_w()[:, 0] - self.scene.env_origins[:, 0]
        )
        return self._tracking_line_y_at_x_local(cutter_x)

    def _tracking_line_y_at_x_local(self, x_local: torch.Tensor) -> torch.Tensor:
        """Vectorized linear interpolation for ``(N,)`` or ``(N,K)`` x values."""
        squeeze = x_local.ndim == 1
        if squeeze:
            x_local = x_local[:, None]
        samples = self._tracking_path_y_samples
        count = samples.shape[1]
        u = (x_local - self._tracking_path_x_min) / max(
            self._tracking_path_x_max - self._tracking_path_x_min, 1.0e-8
        ) * (count - 1)
        lower = torch.floor(u).to(torch.long).clamp(0, count - 1)
        upper = (lower + 1).clamp(0, count - 1)
        alpha = (u - lower.to(u.dtype)).clamp(0.0, 1.0)
        y0 = torch.gather(samples, 1, lower)
        y1 = torch.gather(samples, 1, upper)
        result = y0 + alpha * (y1 - y0)
        return result.squeeze(1) if squeeze else result

    def tracking_line_preview_local(self) -> torch.Tensor:
        """Path offsets ahead of the cutter for policy anticipation."""
        cutter_x = (
            self._compute_cutter_pos_w()[:, 0] - self.scene.env_origins[:, 0]
        )
        distances = torch.tensor(
            (0.5, 1.0, 2.0), dtype=cutter_x.dtype, device=self.device
        )
        preview_y = self._tracking_line_y_at_x_local(
            cutter_x[:, None] + distances[None, :]
        )
        return preview_y - self.tracking_line_y_local()[:, None]

    def _head_down_axis_world(self) -> torch.Tensor:
        """Cutter local +Z in world coordinates; desired value is (0, 0, -1)."""
        q = self.robot.data.body_link_state_w[:, self._ee_body_id, 3:7]
        q = q / q.norm(dim=-1, keepdim=True).clamp(min=1.0e-8)
        local_axis = torch.tensor(
            [[0.0, 0.0, 1.0]], device=self.device, dtype=q.dtype
        ).expand(self.num_envs, -1)
        return quat_apply(q, local_axis)

    def _task_action_shield(
        self, applied_action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prevent c4 from continuing to drive the cutter away from world-down."""
        head_axis = self._head_down_axis_world()
        signed_head_angle = torch.atan2(head_axis[:, 1], -head_axis[:, 2])
        return directional_tilt_shield(
            applied_action,
            signed_tilt=signed_head_angle,
            action_channel=3,
            action_to_tilt_sign=float(
                getattr(self.cfg, "terrain_c4_action_to_axis_y_sign", 1.0)
            ),
            soft_limit=float(
                getattr(self.cfg, "terrain_head_tilt_shield_soft", 0.08)
            ),
            hard_limit=float(
                getattr(self.cfg, "terrain_head_tilt_shield_hard", 0.16)
            ),
        )

    def _get_observations(self) -> dict:
        if getattr(self, "_sole_applied", None) is None:
            return super()._get_observations()

        dt = max(float(self.step_dt), 1.0e-6)
        self._update_measured_tool_kinematics()
        cutter_z = self._cutter_world_z()
        previous = getattr(self, "_prev_cutter_world_z", None)
        vertical_velocity = (
            (cutter_z - previous) / dt
            if previous is not None
            else torch.zeros_like(cutter_z)
        )
        self._prev_cutter_world_z = cutter_z.detach().clone()

        x, xdot = self._refresh_cylinder_state()
        self._cylinder_x_current = x.detach().clone()
        self._cylinder_xdot_current = xdot.detach().clone()
        low, high = self._cylinder_headroom(x)

        head_axis_world = self._head_down_axis_world()
        alignment = -head_axis_world[:, 2]
        self._head_alignment_rate = (alignment - self._prev_head_alignment) / dt
        self._prev_head_alignment = alignment.detach().clone()

        speed = getattr(self, "_vehicle_speed_kmh", None)
        speed_mps = (
            speed / 3.6
            if speed is not None
            else torch.zeros_like(alignment)
        )

        obs = build_verge_observation(
            height_profile=self.verge_height_profile(),
            cut_height_error=self.cut_height_error(),
            lateral_error=self.verge_lateral_error(),
            cutter_vertical_velocity=vertical_velocity,
            measured_cylinder_extensions=x,
            measured_cylinder_velocities=xdot,
            cylinder_headroom_low=low / self._learned_span,
            cylinder_headroom_high=high / self._learned_span,
            previous_applied_action=self._sole_applied,
            current_valve_commands=self._current_valve_cmd,
            rom_delta_extension=self._fd_delta_corrected,
            commanded_target_lead=self._commanded_target_lead(x),
            head_down_alignment=alignment,
            head_down_alignment_rate=self._head_alignment_rate,
            truck_speed_mps=speed_mps,
            scan_valid_fraction=self.scan_valid_fraction(),
            head_down_axis_xy=head_axis_world[:, :2],
            cutter_contact_force_normalized=torch.clamp(
                self.cutter_terrain_contact_force_n()
                / max(
                    float(
                        getattr(
                            self.cfg,
                            "cutter_contact_force_observation_scale_n",
                            1000.0,
                        )
                    ),
                    1.0,
                ),
                min=0.0,
                max=10.0,
            ),
            path_preview=self.tracking_line_preview_local(),
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        if getattr(self, "_sole_applied", None) is None:
            return super()._get_rewards()

        cfg = self.cfg
        # DirectRLEnv computes rewards before observations. Refresh here so the
        # acceleration penalty is attached to the action/transition that caused
        # it; _get_observations reuses this step's cached sample.
        self._update_measured_tool_kinematics()
        x, _ = self._refresh_cylinder_state()
        low, high = self._cylinder_headroom(x)
        acceleration = torch.linalg.norm(self._measured_tool_acceleration_truck, dim=-1)
        contact_gap = self.cut_height_error()
        lateral_error = self.verge_lateral_error()
        alignment = self._measure_head_down_alignment()
        scan_confidence = self.scan_valid_fraction()

        terms = compute_verge_reward(
            cut_height=contact_gap,
            target_cut_height_m=0.0,
            lateral_error=lateral_error,
            applied_action=self._sole_applied,
            previous_applied_action=self._prev_sole_applied,
            head_down_alignment=alignment,
            cutter_contact=self.cutter_ground_contact(),
            cylinder_headroom_low_m=low,
            cylinder_headroom_high_m=high,
            tool_acceleration_norm=acceleration,
            scan_valid_fraction=scan_confidence,
            cut_tracking_std_m=float(getattr(cfg, "cut_tracking_std_m", 0.03)),
            cut_tracking_weight=float(getattr(cfg, "cut_tracking_weight", 5.0)),
            lateral_std_m=float(getattr(cfg, "lateral_tracking_std_m", 0.30)),
            lateral_weight=float(getattr(cfg, "lateral_tracking_weight", 0.5)),
            action_magnitude_weight=float(
                getattr(
                    cfg,
                    "terrain_action_magnitude_weight",
                    getattr(cfg, "residual_action_magnitude_weight", -0.02),
                )
            ),
            action_rate_weight=float(
                getattr(cfg, "sole_rate_penalty_weight", -0.10)
            ),
            head_penalty_weight=float(
                getattr(cfg, "terrain_head_penalty_weight", -20.0)
            ),
            head_threshold=float(getattr(cfg, "terrain_head_threshold", 0.995)),
            head_penalty_scale=float(
                getattr(cfg, "terrain_head_penalty_scale", 0.02)
            ),
            head_penalty_max_deviation=float(
                getattr(cfg, "terrain_head_penalty_max_deviation", 2.0)
            ),
            clearance_barrier_weight=float(
                getattr(cfg, "terrain_clearance_barrier_weight", -2.0)
            ),
            clearance_margin_m=float(
                getattr(cfg, "terrain_clearance_margin_m", 0.02)
            ),
            clearance_barrier_scale_m=float(
                getattr(cfg, "terrain_clearance_barrier_scale_m", 0.02)
            ),
            clearance_barrier_max_deviation=float(
                getattr(cfg, "terrain_clearance_barrier_max_deviation", 5.0)
            ),
            contact_reward_weight=float(
                getattr(cfg, "terrain_contact_reward_weight", 2.0)
            ),
            limit_penalty_weight=float(
                getattr(cfg, "sole_limit_penalty_weight", -1.0)
            ),
            limit_margin_m=float(getattr(cfg, "sole_limit_margin_m", 0.02)),
            acceleration_weight=float(
                getattr(cfg, "sole_acceleration_weight", -0.20)
            ),
            acceleration_scale_mps2=float(
                getattr(cfg, "sole_acceleration_scale_mps2", 30.0)
            ),
            acceleration_max_ratio=float(
                getattr(cfg, "terrain_acceleration_max_ratio", 10.0)
            ),
        )

        active = (self._preposition_counter <= 0) & (self._settle_counter <= 0)
        dense_cut = -float(getattr(cfg, "dense_cut_error_weight", 5.0)) * torch.clamp(
            terms.cut_height_error.abs(), max=1.0
        ) * scan_confidence
        dense_lateral = -float(
            getattr(cfg, "dense_lateral_error_weight", 1.0)
        ) * torch.clamp(lateral_error.abs(), max=3.0)

        cut_cost = terms.cut_height_error.abs()
        lateral_cost = lateral_error.abs()
        head_threshold = float(getattr(cfg, "terrain_head_threshold", 0.995))
        head_cost = torch.relu(head_threshold - alignment)
        previous_head_cost = torch.relu(
            head_threshold - self._prev_head_alignment
        )
        cut_progress = scan_confidence * bounded_cost_progress(
            self._previous_abs_cut_error,
            cut_cost,
            scale=float(getattr(cfg, "terrain_cut_progress_scale_m", 0.02)),
            weight=float(getattr(cfg, "terrain_cut_progress_weight", 1.0)),
        )
        lateral_progress = bounded_cost_progress(
            self._previous_abs_lateral_error,
            lateral_cost,
            scale=float(getattr(cfg, "terrain_lateral_progress_scale_m", 0.01)),
            weight=float(getattr(cfg, "terrain_lateral_progress_weight", 0.5)),
        )
        head_progress = bounded_cost_progress(
            previous_head_cost,
            head_cost,
            scale=float(getattr(cfg, "terrain_head_progress_scale", 0.002)),
            weight=float(getattr(cfg, "terrain_head_progress_weight", 0.25)),
        )
        shaped_reward = (
            terms.reward
            + dense_cut
            + dense_lateral
            + cut_progress
            + lateral_progress
            + head_progress
        )
        shaped_reward = torch.where(
            active, shaped_reward, torch.zeros_like(shaped_reward)
        )
        reward, reward_nonfinite = sanitize_terminal_reward(
            shaped_reward,
            invalid_mask=getattr(
                self,
                "_terrain_step_nonfinite",
                torch.zeros_like(active),
            ),
            terminal_reward=float(
                getattr(cfg, "terrain_nonfinite_terminal_reward", -100.0)
            ),
        )
        self._terrain_step_nonfinite = reward_nonfinite

        valid_active = active & ~reward_nonfinite
        measured_cut = valid_active & (scan_confidence > 0.0)
        self._previous_abs_cut_error = torch.where(
            measured_cut,
            cut_cost.detach(),
            self._previous_abs_cut_error,
        )
        self._previous_abs_lateral_error = torch.where(
            valid_active,
            lateral_cost.detach(),
            self._previous_abs_lateral_error,
        )
        if torch.any(valid_active):
            err = terms.cut_height_error[valid_active].abs()
            self._last_reward_terms = {
                "reward/mean_total": reward[valid_active].mean(),
                "mow/mean_abs_cut_height_error_m": torch.mean(err),
                "mow/p95_cut_height_error_m": torch.quantile(err, 0.95),
                "mow/mean_cut_height_m": torch.mean(
                    contact_gap[valid_active]
                ),
                "mow/scan_valid_fraction": torch.mean(
                    scan_confidence[valid_active]
                ),
                "mow/ground_contact_fraction": torch.mean(
                    self.cutter_ground_contact()[valid_active].float()
                ),
                "mow/mean_cutter_contact_force_n": torch.mean(
                    self.cutter_terrain_contact_force_n()[valid_active]
                ),
                "mow/p95_cutter_contact_force_n": torch.quantile(
                    self.cutter_terrain_contact_force_n()[valid_active], 0.95
                ),
                "truck/mean_wheel_road_contact_fraction": torch.mean(
                    self.wheel_road_contact_fraction()[valid_active]
                ),
                "truck/mean_ride_height_m": self.truck_ride_height()[
                    valid_active
                ].mean(),
                "truck/max_post_constraint_drift_m": self._truck_post_constraint_drift_xy[
                    valid_active
                ].max(),
                "truck/max_heading_error_deg": torch.rad2deg(
                    self._truck_heading_error_rad[valid_active].abs().max()
                ),
                "truck/mean_speed_kmh": self._vehicle_speed_kmh[
                    valid_active
                ].mean(),
                "mow/mean_abs_lateral_error_m": torch.mean(
                    lateral_error[valid_active].abs()
                ),
                "mow/p95_abs_lateral_error_m": torch.quantile(
                    lateral_error[valid_active].abs(), 0.95
                ),
                "safety/min_head_alignment": torch.min(
                    alignment[valid_active]
                ),
                "safety/p95_tool_acceleration_mps2": torch.quantile(
                    acceleration[valid_active], 0.95
                ),
                "reward/nonfinite_fraction": reward_nonfinite.float().mean(),
                "reward/mean_head_penalty": terms.head_penalty[valid_active].mean(),
                "reward/mean_clearance_barrier": terms.clearance_barrier[
                    valid_active
                ].mean(),
                "reward/mean_acceleration_penalty": terms.acceleration_penalty[
                    valid_active
                ].mean(),
                "reward/mean_cut_progress": cut_progress[valid_active].mean(),
                "reward/mean_lateral_progress": lateral_progress[
                    valid_active
                ].mean(),
                "reward/mean_head_progress": head_progress[valid_active].mean(),
                "action/mean_abs_raw": self._sole_raw[valid_active].abs().mean(),
                "action/mean_abs_applied": self._sole_applied[
                    valid_active
                ].abs().mean(),
                "action/mean_abs_rate": (
                    self._sole_applied[valid_active]
                    - self._prev_sole_applied[valid_active]
                ).abs().mean(),
                "action/raw_saturation_fraction": (
                    self._sole_raw[valid_active].abs() >= 0.999
                ).float().mean(),
                "action/shield_active_fraction": (
                    self._sole_shield_gain[valid_active] < 0.999
                ).float().mean(),
                "cylinder/min_low_headroom_m": low[valid_active].min(),
                "cylinder/min_high_headroom_m": high[valid_active].min(),
            }
            for channel in range(4):
                self._last_reward_terms[
                    f"action/c{channel + 1}_mean"
                ] = self._sole_applied[valid_active, channel].mean()
                self._last_reward_terms[
                    f"action/c{channel + 1}_mean_abs"
                ] = self._sole_applied[valid_active, channel].abs().mean()
            for family in sorted(set(self._tracking_path_family_names)):
                family_mask = torch.as_tensor(
                    [name == family for name in self._tracking_path_family_names],
                    dtype=torch.bool,
                    device=self.device,
                ) & valid_active
                if bool(family_mask.any()):
                    self._last_reward_terms[
                        f"family/{family}_mean_cut_error_m"
                    ] = terms.cut_height_error[family_mask].abs().mean()
                    self._last_reward_terms[
                        f"family/{family}_mean_lateral_error_m"
                    ] = lateral_error[family_mask].abs().mean()
                    self._last_reward_terms[
                        f"family/{family}_contact_fraction"
                    ] = self.cutter_ground_contact()[family_mask].float().mean()
            for scale in sorted(set(self._terrain_height_scales)):
                scale_mask = torch.as_tensor(
                    [value == scale for value in self._terrain_height_scales],
                    dtype=torch.bool,
                    device=self.device,
                ) & valid_active
                if bool(scale_mask.any()):
                    label = f"{scale:.2f}".replace(".", "p")
                    self._last_reward_terms[
                        f"height_scale/{label}_mean_cut_error_m"
                    ] = terms.cut_height_error[scale_mask].abs().mean()
                    self._last_reward_terms[
                        f"height_scale/{label}_mean_lateral_error_m"
                    ] = lateral_error[scale_mask].abs().mean()
                    self._last_reward_terms[
                        f"height_scale/{label}_contact_fraction"
                    ] = self.cutter_ground_contact()[scale_mask].float().mean()
            self.extras.setdefault("log", {}).update(self._last_reward_terms)
        return reward
