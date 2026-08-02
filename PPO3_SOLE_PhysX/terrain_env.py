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
from PPO3_SOLE_PhysX.actions import (
    apply_sole_action,
    directional_tilt_shield,
    limit_shield,
    nullspace_head_restoring_overlay,
)
from PPO3_SOLE_PhysX.terrain_observations import (
    HEIGHT_PROFILE_BINS,
    PATH_PREVIEW_BINS,
    build_verge_observation,
    collapse_scan_width,
    nonfinite_row_mask,
)
from PPO3_SOLE_PhysX.terrain_rewards import (
    bounded_cost_progress,
    compute_contact_safety_reward,
    compute_verge_reward,
    lower_tail_weights,
    sanitize_terminal_reward,
    update_failure_memory,
)
from PPO3_SOLE_PhysX.terrain import (
    MOW_STRIP_CFG,
    MOW_STRIP_EVAL_CFG,
    TRACKING_PATH_FAMILIES,
    apply_relief_profile,
    build_mow_strip_mesh,
    build_tracking_line_mesh,
    decoration_seed,
    terrain_height_scale_for_environment,
    tracking_line_center_y_at,
)

MOW_STRIP_ROOT = "/World/MowStrip"
ROAD_ROOT = "/World/TruckRoad"
TRACKING_LINE_ROOT = "/World/MowTrackingLine"
VISUAL_FLOOR_ROOT = "/World/PPO3VisualFloor"
VISUAL_GRID_ROOT = "/World/PPO3VisualGrid"
ROAD_APRON_ROOT = "/World/PPO3RoadApron"
ROAD_MARKING_ROOT = "/World/PPO3RoadMarkings"
VERGE_APRON_ROOT = "/World/PPO3VergeApron"


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
        # Compliance in the vertical load path, at the only place in this plant
        # where a spring can actually be inserted: the cylinder drives.
        #
        # The head-tool bounce is ejection off a rigid contact -- 94.4 % of
        # separations leave the ground moving UPWARD at +0.92 m/s mean after
        # 53 kN median force spikes (reports/contact_loss_policy_v5.json). With
        # the hydraulic lock at ~5e6 N/m, contact force IS commanded position
        # (5.6 mm penetration = 28 kN), so riding gently would need
        # sub-millimetre regulation at 60 Hz that the command path cannot
        # deliver.
        #
        # PhysX compliant contacts were tried first and are the cleaner physical
        # model, but they are NOT honoured here: sweeping
        # terrain_compliant_contact_stiffness 1e6 -> 1e5 (a 10x range that
        # should have cut force ~10x) moved median peak force only 50.4 -> 46.9
        # kN and made ejection worse. Softening the drive is the mechanism that
        # demonstrably takes effect, because it is an articulation drive gain.
        #
        # Damping is deliberately NOT scaled with stiffness. Holding damping
        # while cutting stiffness by N raises the damping ratio by sqrt(N), so
        # the same knob that adds compliance also dissipates the impact energy
        # instead of returning it as ejection velocity.
        stiffness_scale = float(
            getattr(self.cfg, "terrain_arm_stiffness_scale", 1.0)
        )
        if not 0.0 < stiffness_scale <= 1.0:
            raise ValueError(
                "terrain_arm_stiffness_scale must be in (0, 1], got "
                f"{stiffness_scale}"
            )
        # Compliance belongs on the CYLINDERS ONLY. `_terrain_held_joint_ids`
        # also contains the tool joint, which is Messerkopf_joint -- the head
        # orientation joint. Scaling it along with the cylinders was measured to
        # cost exactly what it should: min_head_alignment 0.9500 -> 0.9150
        # against a 0.950 gate, because a softer head joint holds the head down
        # less firmly. The head joint carries orientation, not the vertical
        # impact load, so softening it buys none of the bounce reduction and
        # pays for it in the one gate that had zero margin.
        head_scale = float(
            getattr(self.cfg, "terrain_head_joint_stiffness_scale", 1.0)
        )
        if not 0.0 < head_scale <= 1.0:
            raise ValueError(
                "terrain_head_joint_stiffness_scale must be in (0, 1], got "
                f"{head_scale}"
            )
        if stiffness_scale != 1.0 or head_scale != 1.0:
            per_joint = torch.full(
                (self._terrain_held_joint_ids.numel(),),
                stiffness_scale,
                dtype=self._terrain_nominal_arm_stiffness.dtype,
                device=self._terrain_nominal_arm_stiffness.device,
            )
            head_positions = (
                self._terrain_held_joint_ids == int(self._tool_joint_id)
            ).nonzero(as_tuple=True)[0]
            per_joint[head_positions] = head_scale
            self.robot.write_joint_stiffness_to_sim(
                self._terrain_nominal_arm_stiffness * per_joint.reshape(1, -1),
                joint_ids=self._terrain_held_joint_ids,
            )
        if not self._terrain_damping_logged:
            print(
                f"[PPO3-VERGE] scored terrain arm damping scale={scale:.3f} "
                f"stiffness scale={stiffness_scale:.4f} "
                "(deployment remains at stock drive)",
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
        # Stiffness too: the deployment ramp was validated at the stock drive,
        # and unfolding a compliant arm to the deployment pose is a different
        # (and unvalidated) problem from mowing with one.
        self.robot.write_joint_stiffness_to_sim(
            self._terrain_nominal_arm_stiffness,
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
        # How far `_truck_travel_m` advanced in the current control step, and
        # how many physics substeps still have to run before the truck should
        # have covered it. Together they let the fixture aim at where the truck
        # belongs *at this substep* instead of where it belongs at the end of
        # the step. See `_pin_target_travel_m`.
        self._truck_step_advance_m = torch.zeros(
            (self.num_envs,), dtype=torch.float32, device=self.device
        )
        self._pin_substeps_pending = 0
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
        # Follows terrain_model_error_observation: 54 without the
        # actuator-model-error channels, 66 with them at depth 1, 90 at depth
        # 3, and 96 with the contact-state block on top.
        return (
            verge_observation_dim(
                model_error=bool(
                    getattr(self.cfg, "terrain_model_error_observation", False)
                ),
                model_error_history=int(
                    getattr(self.cfg, "terrain_model_error_history", 1)
                ),
                contact_state=bool(
                    getattr(self.cfg, "terrain_contact_observation", False)
                ),
            )
            + 1
        )

    def _mow_strip_cfg(self):
        held_out = bool(getattr(self.cfg, "use_eval_terrain", False))
        base = MOW_STRIP_EVAL_CFG if held_out else MOW_STRIP_CFG
        # Held-out geometry remains fixed at seed 1337. Training entry points
        # deliberately vary this seed between resumed blocks so the policy does
        # not replay the same 512 static meshes forever.
        #
        # `terrain_eval_seed_override` is the ONE way to move the held-out mesh,
        # and it exists because `--seed` cannot: with the mesh pinned, the start
        # phase fixed and the actor evaluated at its mean, a held-out rollout is
        # fully deterministic and every seed returns bit-identical numbers.
        # Measured directly -- seeds 17/909/1337/4242 produced identical event
        # counts to the digit across all four runs of the pre-fix diagnostic.
        # A "multi-seed" claim made by varying `--seed` on held-out terrain is
        # therefore four copies of one measurement, and the override is what
        # makes it a real robustness check. Zero keeps the pinned gate mesh.
        eval_override = int(getattr(self.cfg, "terrain_eval_seed_override", 0))
        if held_out:
            seed = eval_override if eval_override > 0 else int(base.seed)
        else:
            seed = int(getattr(self.cfg, "terrain_base_seed", base.seed))
        resolution = float(
            getattr(self.cfg, "mow_strip_resolution_m", base.resolution_m)
        )
        length = float(getattr(self.cfg, "mow_strip_length_m", base.length_m))
        width = float(getattr(self.cfg, "mow_strip_width_m", base.width_m))
        amplitude_scale = float(
            getattr(
                self.cfg,
                "terrain_tracking_amplitude_scale",
                base.tracking_path_amplitude_scale,
            )
        )
        return apply_relief_profile(
            replace(
                base,
                seed=seed,
                resolution_m=resolution,
                length_m=length,
                width_m=width,
                tracking_path_amplitude_scale=amplitude_scale,
            ),
            str(getattr(self.cfg, "terrain_relief_profile", "gentle")),
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
        if bool(getattr(self.cfg, "enable_road_scenery", False)):
            self._spawn_road_scenery()
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
        """Spawn a non-colliding dark-grass floor with subtle grid lines."""
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
            f"[PPO3-VISUAL] grass grid floor enabled ({spacing:.1f} m spacing, visual-only)",
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

    def _lawn_apron_mesh(
        self, *, x0: float, x1: float, y0: float, y1: float, z: float, seed_offset: int
    ):
        """Undulating grass sheet for the roadside banks.

        Visual only, so it is sampled far coarser than the verge and carries no
        collider. Deterministic per environment and per bank, which keeps a
        screenshot reproducible.
        """
        import numpy as np
        import trimesh

        step = max(float(getattr(self.cfg, "verge_apron_resolution_m", 0.45)), 0.05)
        relief = float(getattr(self.cfg, "verge_apron_relief_m", 0.09))
        nx = max(int(round((x1 - x0) / step)) + 1, 2)
        ny = max(int(round((y1 - y0) / step)) + 1, 2)
        x = np.linspace(x0, x1, nx)
        y = np.linspace(y0, y1, ny)
        xx, yy = np.meshgrid(x, y, indexing="ij")

        rng = np.random.default_rng(
            decoration_seed(
                int(getattr(self.cfg, "terrain_base_seed", 42)), int(seed_offset)
            )
        )
        ph = rng.uniform(0.0, 2.0 * np.pi, size=6)
        # Layered sinusoids rather than per-vertex noise: a real lawn reads as
        # broad swells with finer texture on top, and this stays continuous so
        # the surface never shows facet seams under the viewport light.
        zz = z + relief * (
            0.55 * np.sin(2.0 * np.pi * xx / 7.0 + ph[0])
            + 0.30 * np.sin(2.0 * np.pi * yy / 3.1 + ph[1])
            + 0.20 * np.sin(2.0 * np.pi * xx / 2.3 + ph[2])
            + 0.14 * np.sin(2.0 * np.pi * yy / 1.4 + ph[3])
            + 0.10 * np.sin(2.0 * np.pi * (xx + yy) / 1.1 + ph[4])
            + 0.07 * np.sin(2.0 * np.pi * (xx - yy) / 0.7 + ph[5])
        )
        vertices = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
        i, j = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), indexing="ij")
        v00 = (i * ny + j).ravel()
        v01 = v00 + 1
        v10 = v00 + ny
        v11 = v10 + 1
        faces = np.vstack(
            (
                np.column_stack((v00, v10, v11)),
                np.column_stack((v00, v11, v01)),
            )
        )
        return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    def _spawn_road_scenery(self) -> None:
        """Dress the calibrated road as a marked two-lane carriageway.

        Visual only, and deliberately additive: the colliding road from
        ``_spawn_flat_road`` keeps its measured width, position and friction, and
        the wheel-road contact sensors keep resolving against it. Everything here
        has collision disabled, so physics and every gate metric are unchanged.

        Three merged prims (asphalt apron, white markings, grass apron) rather
        than one prim per piece -- the same reason ``_spawn_mow_strip`` merges.
        Off by default; ``evaluate.py``/``probe_drive.py`` enable it only for
        interactive viewing, where ``--num_envs 1`` makes the cost negligible.
        """
        import numpy as np
        import trimesh

        start_x = float(getattr(self.cfg, "road_start_x_m", -40.0))
        length = float(getattr(self.cfg, "road_length_m", 40.0))
        road_center_y = float(getattr(self.cfg, "road_center_y_m", 4.30))
        road_width = float(getattr(self.cfg, "road_width_m", 3.60))
        lanes = max(int(getattr(self.cfg, "road_visual_lanes", 2)), 1)
        edge_width = float(getattr(self.cfg, "road_edge_line_width_m", 0.14))
        dash_width = float(getattr(self.cfg, "road_lane_dash_width_m", 0.12))
        dash_length = float(getattr(self.cfg, "road_lane_dash_length_m", 2.0))
        dash_gap = float(getattr(self.cfg, "road_lane_dash_gap_m", 3.0))
        apron_m = float(getattr(self.cfg, "verge_visual_apron_m", 3.2))
        if min(edge_width, dash_width, dash_length, dash_gap) <= 0.0:
            raise ValueError("road marking dimensions must be positive")

        # The colliding lane is the one under the tyres; extra lanes are added on
        # the far side of it so nothing is ever placed between the verge and the
        # wheels, where it could occlude the height scan.
        near_edge_y = road_center_y - 0.5 * road_width
        far_edge_y = near_edge_y + lanes * road_width
        # Take the verge placement from the strip config actually used to build
        # the colliding mesh, so the grass and shoulder stay aligned if the strip
        # is ever moved or resized.
        strip_cfg = self._mow_strip_cfg()
        strip_center_y = float(strip_cfg.center_y_m)
        strip_width = float(strip_cfg.width_m)
        strip_outer_y = strip_center_y - 0.5 * strip_width

        asphalt, markings, grass = [], [], []
        # Sit each layer a hair above the one below it: apron under the markings,
        # both just under the colliding road so no z-fight with the tyre surface.
        apron_z, marking_z = -0.012, -0.004
        for origin in self.scene.env_origins.detach().cpu().numpy():
            ox, oy, oz = float(origin[0]), float(origin[1]), float(origin[2])
            mid_x = ox + start_x + 0.5 * length

            if lanes > 1:
                extra = (lanes - 1) * road_width
                slab = trimesh.creation.box(extents=(length, extra, 0.02))
                slab.apply_translation(
                    (mid_x, oy + far_edge_y - 0.5 * extra, oz + apron_z)
                )
                asphalt.append(slab)

            for edge_y in (near_edge_y + 0.5 * edge_width,
                           far_edge_y - 0.5 * edge_width):
                line = trimesh.creation.box(extents=(length, edge_width, 0.012))
                line.apply_translation((mid_x, oy + edge_y, oz + marking_z))
                markings.append(line)

            period = dash_length + dash_gap
            for lane in range(1, lanes):
                lane_y = near_edge_y + lane * road_width
                for dash_x in np.arange(start_x, start_x + length - dash_length, period):
                    dash = trimesh.creation.box(
                        extents=(dash_length, dash_width, 0.012)
                    )
                    dash.apply_translation(
                        (
                            ox + dash_x + 0.5 * dash_length,
                            oy + lane_y,
                            oz + marking_z,
                        )
                    )
                    markings.append(dash)

            # Grass outboard of the mowed verge, and a matching bank beyond the
            # far kerb, so the carriageway reads as a road between two verges.
            uneven = bool(getattr(self.cfg, "uneven_verge_apron", True))
            for centre, band in (
                (strip_outer_y - 0.5 * apron_m, apron_m),
                (far_edge_y + 0.5 * apron_m, apron_m),
            ):
                if uneven:
                    bank = self._lawn_apron_mesh(
                        x0=ox + start_x,
                        x1=ox + start_x + length,
                        y0=oy + centre - 0.5 * band,
                        y1=oy + centre + 0.5 * band,
                        z=oz + apron_z,
                        seed_offset=int(round(centre * 1000.0)),
                    )
                else:
                    bank = trimesh.creation.box(extents=(length, band, 0.02))
                    bank.apply_translation((mid_x, oy + centre, oz + apron_z))
                grass.append(bank)
            # Close the narrow gap between the verge mesh and the near kerb.
            shoulder = (near_edge_y) - (strip_center_y + 0.5 * strip_width)
            if shoulder > 1.0e-3:
                fill = trimesh.creation.box(extents=(length, shoulder, 0.02))
                fill.apply_translation(
                    (mid_x, oy + near_edge_y - 0.5 * shoulder, oz + apron_z)
                )
                grass.append(fill)

        for root, pieces, material in (
            (
                ROAD_APRON_ROOT,
                asphalt,
                sim_utils.PreviewSurfaceCfg(
                    diffuse_color=tuple(
                        getattr(self.cfg, "road_color", (0.11, 0.12, 0.13))
                    ),
                    roughness=float(getattr(self.cfg, "road_roughness", 0.92)),
                    metallic=0.0,
                ),
            ),
            (
                ROAD_MARKING_ROOT,
                markings,
                sim_utils.PreviewSurfaceCfg(
                    diffuse_color=tuple(
                        getattr(
                            self.cfg, "road_marking_color", (0.90, 0.90, 0.88)
                        )
                    ),
                    roughness=0.6,
                ),
            ),
        ):
            if not pieces:
                continue
            create_prim_from_mesh(
                root, trimesh.util.concatenate(pieces), visual_material=material
            )
            mesh_prim = sim_utils.get_prim_at_path(f"{root}/mesh")
            sim_utils.define_collision_properties(
                mesh_prim.GetPrimPath(),
                sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            )

        # Reuse the exact apron triangles but sort them into broad, irregular
        # color regions.  This is material variation only: no vertex is moved,
        # no triangle is added, and every resulting prim remains visual-only.
        # The low-frequency field avoids the artificial single-color carpet
        # look while keeping the result deterministic across screenshots.
        palette = tuple(
            getattr(
                self.cfg,
                "verge_visual_palette",
                (
                    getattr(self.cfg, "verge_visual_color", (0.08, 0.27, 0.05)),
                ),
            )
        )
        if not palette:
            raise ValueError("verge_visual_palette must contain at least one color")
        grass_by_color: list[list[trimesh.Trimesh]] = [
            [] for _ in range(len(palette))
        ]
        # The last palette entry is soil. Wide outer thresholds keep that
        # region sparse while the four green shades occupy most of the lawn.
        thresholds = np.linspace(-1.05, 1.05, len(palette) - 1)
        for piece in grass:
            centers = piece.triangles_center
            color_field = (
                np.sin(0.31 * centers[:, 0] + 0.73 * centers[:, 1])
                + 0.55 * np.sin(0.83 * centers[:, 0] - 0.29 * centers[:, 1])
            )
            color_ids = np.digitize(color_field, thresholds)
            for color_id in range(len(palette)):
                faces = piece.faces[color_ids == color_id]
                if len(faces) == 0:
                    continue
                grass_by_color[color_id].append(
                    trimesh.Trimesh(
                        vertices=piece.vertices.copy(),
                        faces=faces.copy(),
                        process=False,
                    )
                )
        for color_id, (color, color_pieces) in enumerate(
            zip(palette, grass_by_color, strict=True)
        ):
            if not color_pieces:
                continue
            root = (
                VERGE_APRON_ROOT
                if color_id == 0
                else f"{VERGE_APRON_ROOT}Shade{color_id + 1}"
            )
            create_prim_from_mesh(
                root,
                trimesh.util.concatenate(color_pieces),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=tuple(color), roughness=1.0
                ),
            )
            mesh_prim = sim_utils.get_prim_at_path(f"{root}/mesh")
            sim_utils.define_collision_properties(
                mesh_prim.GetPrimPath(),
                sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            )
        print(
            f"[PPO3-SCENERY] {lanes}-lane marked road with "
            f"{len(palette)}-tone grass verges "
            f"(visual-only, y=[{strip_outer_y - apron_m:.2f}, "
            f"{far_edge_y + apron_m:.2f}])",
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
        path_start_phases = []
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
        family_schedule = tuple(
            str(value)
            for value in getattr(self.cfg, "tracking_path_family_schedule", ())
        )
        start_phase_schedule = tuple(
            float(value)
            for value in getattr(
                self.cfg, "tracking_path_start_phase_schedule", ()
            )
        )
        unknown_scheduled = sorted(set(family_schedule) - set(TRACKING_PATH_FAMILIES))
        if unknown_scheduled:
            raise ValueError(
                "tracking_path_family_schedule contains unknown families: "
                f"{unknown_scheduled}"
            )
        if start_phase_schedule and len(start_phase_schedule) != len(family_schedule):
            raise ValueError(
                "tracking_path_start_phase_schedule must be empty or match "
                "tracking_path_family_schedule length"
            )
        if any(value > 1.0 for value in start_phase_schedule):
            raise ValueError(
                "tracking path start phases must be <= 1.0; use a negative "
                "entry to preserve the authored reference"
            )
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
            if requested_family != "mixed":
                family = requested_family
            elif family_schedule:
                family = family_schedule[
                    (env_index + family_offset) % len(family_schedule)
                ]
            else:
                family = TRACKING_PATH_FAMILIES[
                    (env_index + family_offset) % len(TRACKING_PATH_FAMILIES)
                ]
            env_strip_cfg = (
                replace(
                    strip_cfg,
                    seed=strip_cfg.seed + env_index * seed_stride,
                    tracking_path_family=family,
                )
                if vary_seed
                else replace(strip_cfg, tracking_path_family=family)
            )
            if start_phase_schedule:
                start_phase = start_phase_schedule[
                    (env_index + family_offset) % len(start_phase_schedule)
                ]
                if start_phase >= 0.0:
                    phase_anchor_x = float(
                        getattr(
                            self.cfg,
                            "tracking_path_phase_anchor_x_m",
                            -27.12,
                        )
                    )
                    env_strip_cfg = replace(
                        env_strip_cfg,
                        tracking_reference_x_m=(
                            phase_anchor_x
                            - start_phase * env_strip_cfg.tracking_active_length_m
                        ),
                    )
            else:
                start_phase = -1.0
            path_start_phases.append(start_phase)
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
        self._tracking_path_start_phases = tuple(path_start_phases)
        self._terrain_height_scales = tuple(terrain_height_scales)

        # Compliance in the vertical load path. Measured cause of the head-tool
        # bounce (reports/contact_loss_policy_v5.json): 94.4 % of separations
        # leave the ground moving UPWARD at +0.92 m/s mean (max 3.99), after
        # force spikes of 53 kN median / 162 kN max. The tool is not being left
        # behind by falling ground -- it is being catapulted off a rigid contact.
        #
        # With rigid contacts the series stiffness is the arm's own hydraulic
        # lock (~5e6 N/m), so force IS position: 0.40 mm of penetration is 2 kN
        # and the measured 5.6 mm mean is 28 kN. Riding gently would need
        # sub-millimetre regulation at 60 Hz while the terrain moves 1.5 mm per
        # control step, with no force feedback in the loop -- authority the
        # plant does not have, which is why no reward shaping moved this.
        #
        # An implicit contact spring puts a real compliance in series with that
        # lock and gives the damping somewhere to dissipate the stored energy
        # instead of returning it as ejection velocity. This is the simulation
        # equivalent of the floating/suspended cutter head a real verge mower
        # uses, and it acts only at the tool/ground interface, so the boom's own
        # dynamics are untouched.
        #
        # Zero stiffness disables the model and restores rigid contacts exactly.
        compliant_stiffness = float(
            getattr(self.cfg, "terrain_compliant_contact_stiffness", 0.0)
        )
        compliant_damping = float(
            getattr(self.cfg, "terrain_compliant_contact_damping", 0.0)
        )
        if compliant_stiffness < 0.0 or compliant_damping < 0.0:
            raise ValueError(
                "terrain_compliant_contact_stiffness/damping must be >= 0, got "
                f"{compliant_stiffness}/{compliant_damping}"
            )
        physics_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=float(getattr(self.cfg, "terrain_static_friction", 1.0)),
            dynamic_friction=float(getattr(self.cfg, "terrain_dynamic_friction", 1.0)),
            restitution=float(getattr(self.cfg, "terrain_restitution", 0.0)),
            compliant_contact_stiffness=compliant_stiffness,
            compliant_contact_damping=compliant_damping,
        )
        print(
            "[PPO3-VERGE] contact model: "
            + (
                f"COMPLIANT stiffness={compliant_stiffness:.4g} N/m "
                f"damping={compliant_damping:.4g} Ns/m"
                if compliant_stiffness > 0.0
                else "rigid (compliant contacts disabled)"
            ),
            flush=True,
        )
        visual_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=tuple(
                getattr(self.cfg, "mow_strip_color", (0.06, 0.30, 0.045))
            ),
            roughness=1.0,
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
        focused_phase_count = sum(value >= 0.0 for value in path_start_phases)
        print(
            f"[PPO3-VERGE] verge strip: {len(mesh.faces):,} triangles across "
            f"{len(origins)} environments (single merged mesh, "
            f"{'per-env seeds' if vary_seed else 'shared seed'}; paths="
            f"{family_counts}; height_scales={height_scale_counts}; "
            f"focused_start_phases={focused_phase_count})",
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
        # Start a fresh per-step window for the chassis-reaction peak. Zeroing
        # here rather than after the reward keeps the accumulator aligned with
        # exactly one control step's worth of substep corrections.
        self._ensure_chassis_reaction_buffers()
        self._chassis_lateral_reaction_m.zero_()
        self._chassis_attitude_reaction_rad.zero_()

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
        advance = torch.where(active, step, torch.zeros_like(step))
        self._truck_travel_m = self._truck_travel_m + advance
        # Remember the advance and open the substep window. Envs still in
        # preposition/settle advance by zero, so their interpolated target
        # equals their held target and the fixture cannot drag them backwards.
        self._truck_step_advance_m = advance.detach()
        self._pin_substeps_pending = max(int(getattr(self.cfg, "decimation", 2)), 1)

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
        # q5's measured authority is the world Y-Z tilt: a held positive q5
        # command drives axis_y negative. The held-out failure is likewise
        # dominated by axis_y (0.654 max versus 0.064 axis_x), so train the
        # policy on the direction it can actually correct.
        signed_tilt = torch.atan2(head_axis[:, 1], -head_axis[:, 2])
        tilt_response_sign = float(
            getattr(self.cfg, "terrain_c4_action_to_axis_y_sign", 1.0)
        )
        head_gain = float(getattr(self.cfg, "terrain_teacher_head_gain", 4.0))
        head_mix = max(
            abs(float(getattr(self.cfg, "sole_head_action_to_c4_gain", 0.5))),
            1.0e-6,
        )
        head_command = torch.clamp(
            -head_gain * signed_tilt / (tilt_response_sign * head_mix),
            -1.0,
            1.0,
        )
        mixed_teacher = raw_teacher.clone()
        mixed_teacher[:, 3] = torch.clamp(
            mixed_teacher[:, 3] + head_mix * head_command,
            -raw_limit,
            raw_limit,
        )

        low, high = self._cylinder_headroom(measured)
        output = apply_sole_action(
            policy_action=mixed_teacher,
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
        self._terrain_teacher_shield_gain = (
            output.shield_gain * task_gain
        ).detach().clone()

        if bool(getattr(self.cfg, "sole_head_joint_action", False)):
            # q5 is the explicit decomposition of the teacher's head correction
            # from its general cylinder-4 command. The environment mixes it
            # back into c4 before ID/FD, so student and teacher use the same
            # physical hydraulic path.
            head_command = torch.nan_to_num(head_command).reshape(-1, 1)
            raw_teacher = torch.cat([raw_teacher, head_command], dim=-1)
            applied = torch.cat([applied, head_command], dim=-1)

        self._terrain_teacher_raw_action = raw_teacher.detach().clone()
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
    def _ensure_chassis_reaction_buffers(self) -> None:
        """Allocate the chassis-reaction accumulators on first use.

        Allocated lazily rather than in ``__init__`` because the base class
        already calls ``_constrain_truck_xy`` during its own construction, so an
        ``__init__`` assignment would run too late and raise on the first step.
        """
        if getattr(self, "_chassis_lateral_reaction_m", None) is None:
            self._chassis_lateral_reaction_m = torch.zeros(
                self.num_envs, device=self.device
            )
            self._chassis_attitude_reaction_rad = torch.zeros(
                self.num_envs, device=self.device
            )

    def _pin_target_travel_m(self) -> torch.Tensor:
        """Where the chassis belongs at THIS substep, not at the end of the step.

        ``_pre_physics_step`` advances ``_truck_travel_m`` by a whole control
        step before any physics runs, but the fixture pins the chassis several
        times inside that step (once pre-physics, once per decimation substep,
        once in the inherited ``_get_dones``). Every one of those pins used the
        end-of-step target, so the sequence measured at 4 km/h was:

            pin +18.52 mm (teleport a full step ahead of the physics)
            substep integrates  +8.1 mm
            pin  -8.10 mm (drag it back to the same end-of-step target)
            substep integrates  +8.1 mm
            pin  -8.10 mm

        i.e. a +-8 mm longitudinal square wave at 60-120 Hz on top of a correct
        18.52 mm/step mean. Nothing measured it: ``p95_tool_acceleration`` is a
        truck-frame finite difference and a root teleport moves chassis and tool
        rigidly together, so it cancels exactly; the chassis-reaction gates keep
        only the component perpendicular to the heading; and ``truck_drift`` is
        assigned literal zero after the correction. Under the cutter the shake
        is not invisible at all -- 8 mm of longitudinal travel over 0.15 m verge
        facets changes the ground height beneath a rigidly-pinned tool, and that
        modulates contact penetration directly.

        Subtracting the not-yet-simulated share of the advance makes each pin
        aim at the position the truck should actually hold at that instant, so
        the correction collapses to the friction gap between commanded and
        integrated travel. The mean speed is unchanged: the window closes at
        ``_get_dones`` with the full advance applied.
        """
        if not bool(
            getattr(self.cfg, "terrain_kinematic_drive_substep_interpolation", True)
        ):
            return self._truck_travel_m
        pending = int(getattr(self, "_pin_substeps_pending", 0))
        advance = getattr(self, "_truck_step_advance_m", None)
        if pending <= 0 or advance is None:
            return self._truck_travel_m
        decimation = max(int(getattr(self.cfg, "decimation", 2)), 1)
        remaining = min(pending, decimation) / decimation
        return self._truck_travel_m - advance * remaining

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
            + self._pin_target_travel_m().reshape(-1, 1) * heading
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

        # Arm-induced chassis reaction, sampled BEFORE the correction is written.
        # Sampling after is what made `truck_drift` read 0.0000 while the chassis
        # was visibly swaying: the fixture had already removed the evidence.
        # `correction` is the residual chassis position error, so its component
        # perpendicular to the drive heading is exactly the lateral sway, and
        # `delta_q` is the full attitude error, so roll and pitch are counted and
        # not just the yaw already in `heading_error`.
        #
        # Several calls per env step under per-substep pinning: keep the worst,
        # because a single hard shove is what is visible and what unloads a tyre.
        lateral_reaction = (
            correction[:, 0] * (-heading[:, 1]) + correction[:, 1] * heading[:, 0]
        ).abs()
        attitude_reaction = 2.0 * torch.acos(delta_q[:, 0].abs().clamp(max=1.0))
        self._ensure_chassis_reaction_buffers()
        self._chassis_lateral_reaction_m = torch.maximum(
            self._chassis_lateral_reaction_m, lateral_reaction.detach()
        )
        self._chassis_attitude_reaction_rad = torch.maximum(
            self._chassis_attitude_reaction_rad, attitude_reaction.detach()
        )

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
        # Partial vertical correction. teacher_student drives a FREE chassis on
        # velocity-driven wheels and lets gravity load all four tyres; PPO3 pins
        # the pose instead, and a hard z pin is exactly what leaves one tyre
        # hovering (min duty 0.1947 against a 0.7527 mean, i.e. three wheels near
        # 0.94 and one chattering). Freeing z entirely was measured and recovers
        # the wheel gate outright (0.5178) but costs smoothness (accel 138.88),
        # because the vertical pin is what supplies it. A gain in (0,1) is a
        # first-order lag toward the target height: the chassis may settle onto
        # its tyres while large vertical excursions are still suppressed.
        # 1.0 reproduces every previously measured number exactly.
        ride_gain = float(getattr(self.cfg, "truck_ride_height_gain", 1.0))
        root_pose[:, 2] = root_pose[:, 2] + ride_gain * (target_base_z - base_z)
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

    # ------------------------------------------------------------------
    # Contact-support state machine
    # ------------------------------------------------------------------

    def _ensure_contact_safety_buffers(self) -> None:
        """Allocate the contact-support state once the scene exists."""
        if getattr(self, "_contact_acquired", None) is not None:
            return
        n = self.num_envs
        zeros_b = torch.zeros(n, dtype=torch.bool, device=self.device)
        zeros_l = torch.zeros(n, dtype=torch.long, device=self.device)
        zeros_f = torch.zeros(n, dtype=torch.float32, device=self.device)

        # --- machine state ---
        self._contact_acquired = zeros_b.clone()
        self._contact_acquisition_steps = zeros_l.clone()
        self._contact_loss_steps = zeros_l.clone()
        self._previous_contact_force_n = zeros_f.clone()
        self._previous_contact_state = zeros_b.clone()
        self._previous_cutter_z = torch.full_like(zeros_f, float("nan"))
        self._airborne_gap_m = zeros_f.clone()

        # --- per-step event flags (consumed by reward, dones and metrics) ---
        self._takeoff_event = zeros_b.clone()
        self._impact_ejection_event = zeros_b.clone()
        self._hard_impact_event = zeros_b.clone()
        self._visible_lift_event = zeros_b.clone()
        # Previous LEVEL, so the counters below can count rising edges.
        self._hard_impact_level = zeros_b.clone()
        self._visible_lift_level = zeros_b.clone()
        self._hard_impact_onset = zeros_b.clone()
        self._visible_lift_onset = zeros_b.clone()
        self._contact_loss_event = zeros_b.clone()

        # --- cached per-step measurements ---
        self._contact_state = zeros_b.clone()
        self._contact_force_n = zeros_f.clone()
        self._contact_force_rate_nps = zeros_f.clone()
        self._geometric_gap_m = zeros_f.clone()
        self._cutter_vertical_velocity_mps = zeros_f.clone()
        self._contact_support = zeros_b.clone()

        # --- episode accumulators, reported by the evaluator ---
        self._contact_loss_event_count = zeros_l.clone()
        self._takeoff_event_count = zeros_l.clone()
        self._impact_ejection_event_count = zeros_l.clone()
        self._hard_impact_event_count = zeros_l.clone()
        self._visible_lift_event_count = zeros_l.clone()
        self._hard_impact_steps = zeros_l.clone()
        self._visible_lift_steps = zeros_l.clone()
        self._post_acquisition_steps = zeros_l.clone()
        self._post_acquisition_contact_steps = zeros_l.clone()
        self._max_consecutive_airborne_steps = zeros_l.clone()
        self._max_airborne_gap_m = zeros_f.clone()

    def _reset_contact_safety(self, env_ids: torch.Tensor) -> None:
        """Clear every contact-support variable for the given environments.

        A stale ``_contact_acquired`` across a reset would let a fresh episode
        inherit the previous one's support claim and immediately terminate
        during its own deployment ramp.
        """
        self._ensure_contact_safety_buffers()
        for name in (
            "_contact_acquired",
            "_previous_contact_state",
            "_takeoff_event",
            "_impact_ejection_event",
            "_hard_impact_event",
            "_visible_lift_event",
            "_hard_impact_level",
            "_visible_lift_level",
            "_hard_impact_onset",
            "_visible_lift_onset",
            "_contact_loss_event",
            "_contact_state",
            "_contact_support",
        ):
            getattr(self, name)[env_ids] = False
        for name in (
            "_contact_acquisition_steps",
            "_contact_loss_steps",
            "_contact_loss_event_count",
            "_takeoff_event_count",
            "_impact_ejection_event_count",
            "_hard_impact_event_count",
            "_visible_lift_event_count",
            "_hard_impact_steps",
            "_visible_lift_steps",
            "_post_acquisition_steps",
            "_post_acquisition_contact_steps",
            "_max_consecutive_airborne_steps",
        ):
            getattr(self, name)[env_ids] = 0
        for name in (
            "_previous_contact_force_n",
            "_airborne_gap_m",
            "_contact_force_n",
            "_contact_force_rate_nps",
            "_geometric_gap_m",
            "_cutter_vertical_velocity_mps",
            "_max_airborne_gap_m",
        ):
            getattr(self, name)[env_ids] = 0.0
        # NaN, not zero: the first post-reset sample must produce no vertical
        # velocity at all. Seeding with 0.0 would differentiate the deployed
        # cutter height against the world origin and manufacture a takeoff.
        self._previous_cutter_z[env_ids] = float("nan")
        # The per-step cache marker is deliberately NOT cleared here.
        #
        # `_reset_idx` runs between `_get_rewards` and `_get_observations`
        # within one control step. Clearing the marker would make the
        # observation call perform a SECOND update for that step -- for every
        # environment, not just the reset ones. Today that second update
        # cannot raise an event (the reset clears `_contact_acquired` and
        # `_previous_contact_state`, and resets here are synchronized), so it
        # is harmless by accident rather than by construction. Leaving the
        # marker alone means the observation reads the state this reset just
        # cleared, which is what it should see, and the next control step gets
        # its own single update from `_get_dones`.

    @torch.no_grad()
    def _update_contact_safety(self) -> None:
        """Advance the contact-support state machine, exactly once per step.

        ``DirectRLEnv.step`` increments ``common_step_counter`` and then calls
        ``_get_dones``, ``_get_rewards``, ``_reset_idx`` and
        ``_get_observations`` in that order. All four consumers call this
        method, so it caches on the counter -- otherwise one contact-loss
        transition would be counted three times and the termination and the
        penalty would disagree about how long support had been missing.

        Nothing here is a controller. It only observes the PhysX contact
        sensor and the cutter's own kinematics; no target, valve command or
        joint reference is written anywhere in this method.
        """
        self._ensure_contact_safety_buffers()
        marker = int(getattr(self, "common_step_counter", -1))
        if getattr(self, "_contact_safety_step", None) == marker:
            return
        self._contact_safety_step = marker

        cfg = self.cfg
        dt = max(float(self.step_dt), 1.0e-6)
        force = self.cutter_terrain_contact_force_n().detach()
        contact = force > float(
            getattr(cfg, "cutter_contact_force_threshold_n", 10.0)
        )
        cutter_z = self._cutter_world_z().detach()
        reference = float(
            getattr(cfg, "cutter_contact_reference_height_m", -0.30)
        )
        # Height of the cutter above the calibrated ground-support coordinate.
        # Contact-independent by construction: this is geometry, so a missed
        # contact sample cannot make it read zero, and a stuck contact sample
        # cannot make it read large.
        geometric_gap = torch.relu(
            (cutter_z - self.ground_height_under_cutter().detach()) - reference
        )

        previous_z = self._previous_cutter_z
        vertical_velocity = torch.where(
            torch.isfinite(previous_z),
            (cutter_z - previous_z) / dt,
            torch.zeros_like(cutter_z),
        )

        # Deployment, preposition, reset correction and settling are not scored
        # and must never acquire support or raise an event -- the arm is being
        # driven to its pose by the reset ramp, not by the policy.
        active = (self._preposition_counter <= 0) & (self._settle_counter <= 0)

        acquire_steps = max(int(getattr(cfg, "terrain_contact_acquire_steps", 3)), 1)
        grace = max(int(getattr(cfg, "terrain_contact_loss_grace_steps", 2)), 0)
        max_gap = float(getattr(cfg, "terrain_max_airborne_gap_m", 0.005))
        takeoff_v = float(
            getattr(cfg, "terrain_takeoff_velocity_threshold_mps", 0.02)
        )
        hard_force = float(getattr(cfg, "terrain_hard_impact_force_n", 30000.0))
        ejection_force = float(
            getattr(cfg, "terrain_impact_ejection_force_n", 0.0)
        )
        if ejection_force <= 0.0:
            ejection_force = hard_force

        # --- hysteresis: acquisition ---
        run = torch.where(
            contact & active,
            self._contact_acquisition_steps + 1,
            torch.zeros_like(self._contact_acquisition_steps),
        )
        self._contact_acquisition_steps = run
        newly_acquired = active & (run >= acquire_steps)
        self._contact_acquired = self._contact_acquired | newly_acquired

        # --- hysteresis: loss ---
        loss_steps = torch.where(
            contact | ~active,
            torch.zeros_like(self._contact_loss_steps),
            self._contact_loss_steps + 1,
        )
        self._contact_loss_steps = loss_steps
        # Support is still claimed while inside the grace window, so a single
        # missed physics substep does not destroy the episode during training.
        self._contact_support = contact | (
            self._contact_acquired & (loss_steps <= grace) & (loss_steps > 0)
        )

        # --- events ---
        separated = active & self._contact_acquired & self._previous_contact_state & ~contact
        takeoff = separated & (vertical_velocity > takeoff_v)
        impact_ejection = takeoff & (self._previous_contact_force_n > ejection_force)
        hard_impact = active & (force > hard_force)
        # A visible lift is a GEOMETRIC claim about a cutter that is genuinely
        # unsupported -- not a single sensor miss. Requiring the loss to have
        # outlived the grace window is what makes "zero visible lifts" a
        # statement about the tool and not about the sensor.
        unsupported = active & self._contact_acquired & ~contact & (loss_steps > grace)
        visible_lift = unsupported & (geometric_gap > max_gap)

        self._contact_loss_event = separated
        self._takeoff_event = takeoff
        self._impact_ejection_event = impact_ejection
        self._hard_impact_event = hard_impact
        self._visible_lift_event = visible_lift
        self._airborne_gap_m = torch.where(
            unsupported, geometric_gap, torch.zeros_like(geometric_gap)
        )

        # --- accumulators ---
        # Counters count EVENTS, i.e. rising edges.
        #
        # `hard_impact` and `visible_lift` are per-step LEVELS -- the force is
        # extreme now, the tool is off the ground now -- because that is what
        # the reward and the termination need. Accumulating a level gives a
        # sample count wearing an event name: the first measured run reported
        # "1559 hard impacts" in 30 720 samples, which is a 5 % duty cycle, not
        # 1559 distinct impacts. Separations are already edges (they require
        # `_previous_contact_state`), so they are counted directly.
        hard_impact_onset = hard_impact & ~self._hard_impact_level
        visible_lift_onset = visible_lift & ~self._visible_lift_level
        self._hard_impact_level = hard_impact
        self._visible_lift_level = visible_lift
        self._hard_impact_onset = hard_impact_onset
        self._visible_lift_onset = visible_lift_onset

        self._contact_loss_event_count += separated.long()
        self._takeoff_event_count += takeoff.long()
        self._impact_ejection_event_count += impact_ejection.long()
        self._hard_impact_event_count += hard_impact_onset.long()
        self._visible_lift_event_count += visible_lift_onset.long()
        # Duty cycles, kept alongside the counts because the two answer
        # different questions: "how often does it happen" vs "how much of the
        # run is spent in it".
        self._hard_impact_steps += hard_impact.long()
        self._visible_lift_steps += visible_lift.long()
        scored = active & self._contact_acquired
        self._post_acquisition_steps += scored.long()
        self._post_acquisition_contact_steps += (scored & contact).long()
        self._max_consecutive_airborne_steps = torch.maximum(
            self._max_consecutive_airborne_steps,
            torch.where(scored, loss_steps, torch.zeros_like(loss_steps)),
        )
        self._max_airborne_gap_m = torch.maximum(
            self._max_airborne_gap_m, self._airborne_gap_m
        )

        # --- cached measurements for reward/observation ---
        self._contact_force_rate_nps = (force - self._previous_contact_force_n) / dt
        self._contact_state = contact
        self._contact_force_n = force
        self._geometric_gap_m = geometric_gap
        self._cutter_vertical_velocity_mps = vertical_velocity

        self._previous_contact_force_n = force.clone()
        self._previous_contact_state = contact.clone()
        self._previous_cutter_z = cutter_z.clone()

    def contact_safety_terminal(self) -> torch.Tensor:
        """Environments whose contact behaviour makes the episode a failure.

        Deliberately separate from ``terrain_safety_violation``: that predicate
        is deferred (``defer_safety_terminations``) so exploratory excursions do
        not redeploy the whole vectorized scene, and deferring a lift would
        leave the policy free to finish the path after visibly leaving the
        verge -- which is the exact behaviour this is meant to forbid.
        """
        self._update_contact_safety()
        cfg = self.cfg
        active = (self._preposition_counter <= 0) & (self._settle_counter <= 0)
        grace = max(int(getattr(cfg, "terrain_contact_loss_grace_steps", 2)), 0)
        terminal = torch.zeros_like(self._contact_acquired)
        if bool(getattr(cfg, "terrain_terminate_on_contact_loss", False)):
            terminal = terminal | (
                self._contact_acquired & (self._contact_loss_steps > grace)
            )
            terminal = terminal | self._visible_lift_event
        if bool(getattr(cfg, "terrain_terminate_on_hard_impact", False)):
            terminal = terminal | self._hard_impact_event
        repeats = int(getattr(cfg, "terrain_max_takeoff_events", 0))
        if repeats > 0:
            terminal = terminal | (self._impact_ejection_event_count >= repeats)
        return terminal & active

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
        # This is the FIRST consumer of the state machine each step, so the
        # update happens here and the reward and observation reuse it.
        contact_terminal = self.contact_safety_terminal()
        self._contact_safety_terminated = contact_terminal
        terminated = terminated | contact_terminal
        # Resets in this task are ALL-OR-NOTHING. `_run_reset_preposition`
        # raises outright when it is handed fewer than num_envs environments
        # ("Residual-policy resets are synchronized"), because the physical
        # deployment ramp steps the whole scene. A contact termination that
        # fired on a subset would therefore crash the run rather than reset
        # those environments, so it has to synchronize exactly like every
        # other termination the parent produces.
        #
        # The cost is real and is why `terrain_terminate_on_contact_loss`
        # defaults to False: one environment losing support resets all of
        # them, and the pre-fix policy loses support ~430 times per
        # 32-env/960-step rollout. Termination is the belt-and-braces; the
        # seven-term contact reward is the signal that has to do the work,
        # and it is built so floating is strictly worse than staying planted
        # without needing an episode to end (see
        # terrain_rewards.airborne_task_reward_bound).
        synchronize = bool(getattr(self.cfg, "synchronize_resets", True))
        if synchronize and (nonfinite.any() or contact_terminal.any()):
            terminated = torch.ones_like(terminated)
        log = self.extras.setdefault("log", {})
        log["termination/terrain_nonfinite"] = nonfinite.float().mean()
        log["termination/contact_safety"] = contact_terminal.float().mean()
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

    @torch.no_grad()
    def _randomize_actuator(self, env_ids) -> None:
        """Scatter the cylinder drive gains per episode.

        The third lever for model error, and the one teacher_student uses
        (`randomize_joint_parameters` at reset): instead of telling the policy
        what the actuator error is, make the actuator vary so the policy has to
        be robust to it. Cheap -- a reset-time CPU write, no extra sensor and no
        per-step cost.

        Applied to the cylinders only. The head joint is left alone: it carries
        orientation rather than the vertical load, and softening it was measured
        to cost head alignment for nothing.
        """
        span = float(getattr(self.cfg, "terrain_actuator_randomization", 0.0))
        if span <= 0.0 or not hasattr(self, "_terrain_nominal_arm_stiffness"):
            return
        ids = self._cylinder_joint_ids
        nominal_k = self._terrain_nominal_arm_stiffness[:, : len(ids)]
        nominal_d = self._terrain_nominal_arm_damping[:, : len(ids)]
        base = float(getattr(self.cfg, "terrain_arm_stiffness_scale", 1.0))
        low, high = 1.0 - span, 1.0 + span
        shape = (self.num_envs, len(ids))
        k_scale = torch.empty(shape, device=self.device).uniform_(low, high)
        d_scale = torch.empty(shape, device=self.device).uniform_(low, high)
        self.robot.write_joint_stiffness_to_sim(
            nominal_k * base * k_scale, joint_ids=ids
        )
        self.robot.write_joint_damping_to_sim(nominal_d * d_scale, joint_ids=ids)

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
            # Clear the advance with it. A reset env's travel restarts at zero,
            # so a carried-over advance would make the deployment pins aim
            # behind the anchor for the rest of the step.
            self._truck_step_advance_m[env_ids] = 0.0

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
        self._randomize_actuator(env_ids)
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
        if hasattr(self, "_head_tail_failure_memory"):
            self._head_tail_failure_memory[env_ids] = 0
        if not hasattr(self, "_head_down_failed"):
            self._head_down_failed = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
        self._head_down_failed[env_ids] = False
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
        # Contact support is an episode-scoped claim. Clearing it here, in the
        # same place every other temporal latch is cleared, is what stops a new
        # episode inheriting the previous one's acquisition and terminating on
        # its own deployment ramp.
        self._reset_contact_safety(env_ids)

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

    def _head_joint_signed_tilt(self) -> torch.Tensor:
        """Signed tilt in the plane the moving Messerkopf joint can correct."""
        head_axis = self._head_down_axis_world()
        axis_index = int(getattr(self.cfg, "terrain_head_tilt_axis_index", 0))
        local_axis = torch.zeros_like(head_axis)
        local_axis[:, axis_index] = 1.0
        quat = self.robot.data.body_link_state_w[:, self._ee_body_id, 3:7]
        quat = quat / quat.norm(dim=-1, keepdim=True).clamp(min=1.0e-8)
        joint_axis = quat_apply(quat, local_axis)
        world_down = torch.zeros_like(head_axis)
        world_down[:, 2] = -1.0
        target_p = world_down - (
            world_down * joint_axis
        ).sum(-1, keepdim=True) * joint_axis
        head_p = head_axis - (
            head_axis * joint_axis
        ).sum(-1, keepdim=True) * joint_axis
        cross = torch.cross(head_p, target_p, dim=-1)
        return -torch.atan2(
            (cross * joint_axis).sum(-1),
            (head_p * target_p).sum(-1),
        )

    @torch.no_grad()
    def _apply_action(self):
        """Level the cutter head before the inherited joint targets are written.

        ``Messerkopf_joint`` carries no cylinder and is not hydraulic, yet the
        inherited pipeline commands it as ``joint6_coupling_a * q4`` with a fixed
        ratio. The mechanism's real ratio is pose-dependent -- the inherited
        comment records 0.50-0.76 across tests -- because the joint angle that
        points the head at world-down changes as c1-c3 move the boom. A fixed
        ratio is therefore wrong by a pose-varying amount, and measurement bears
        that out: the joint misses its reference by 0.2096 rad at p99, 43x c4's
        own joint, with drive torque pinned at any ceiling it is given.

        Deleting the ratio is worse, not better. Commanding a constant 0.0 rad
        instead (teacher-student gains, stiffness 6000) drove held-out minimum
        head alignment to -0.6956, and 300000 stiffness to 0.6268, against
        0.9628 for the inherited coupling -- the response is not even monotonic
        in stiffness, because a constant cannot express a pose-dependent angle.

        So keep the ratio as feedforward and close the loop around it: measure
        the head's actual tilt and correct the reference until the head is level.
        That makes the effective ratio adapt with pose, which is what the
        mechanism does, without discarding the term that currently holds the
        head. The correction is bounded so a bad measurement cannot command the
        joint across its travel, and it is recomputed from the coupling source
        every substep rather than accumulated onto the reference -- ``_apply_action``
        runs once per decimation step while the coupling is written once per
        environment step, so an incremental update would integrate.

        Disabled by default (gain 0.0): with the gain at zero this reproduces the
        inherited targets exactly.
        """
        gain = float(getattr(self.cfg, "terrain_messerkopf_level_gain", 0.0))
        target = getattr(self, "_joint6_coupling_target_ref_index", None)
        source = getattr(self, "_joint6_coupling_source_ref_index", None)
        policy_correction = getattr(self, "_head_joint_correction", None)
        if (
            (gain != 0.0 or policy_correction is not None)
            and target is not None
            and source is not None
        ):
            head_axis = self._head_down_axis_world()
            # Same projection the c4 tilt shield uses: the signed tilt in the
            # plane this joint actually acts in. Out-of-plane (X) excursions are
            # unreachable for it -- measured authority is 3.8 % of the Y axis --
            # so correcting on the full off-down angle would chase a component
            # the joint cannot move.
            if bool(getattr(self.cfg, "terrain_head_tilt_arm_frame", False)):
                # Exact one-DOF solve about the joint's own rotation axis, which
                # travels with the arm instead of staying pinned to world Y-Z.
                #
                # The world projection below is blind whenever the arm slews:
                # world-X tilt is in-plane tilt rotated by the slew angle, so a
                # correction measured in a fixed world plane stops seeing the
                # error exactly when the arm turns. That is what the "X is
                # unsteerable" reading was: reports/head_axis_authority.json was
                # sampled at alignment 0.978-0.997, a nearly vertical down-axis,
                # and a vertical vector is invariant to yaw -- so the probe could
                # not have detected slew-coupled authority even if it existed.
                #
                # Rotating the head about `axis` by this angle brings its
                # down-axis as close to world-down as that single joint allows.
                axis_index = int(
                    getattr(self.cfg, "terrain_head_tilt_axis_index", 0)
                )
                local = torch.zeros(
                    (self.num_envs, 3), device=self.device, dtype=head_axis.dtype
                )
                local[:, axis_index] = 1.0
                quat = self.robot.data.body_link_state_w[:, self._ee_body_id, 3:7]
                quat = quat / quat.norm(dim=-1, keepdim=True).clamp(min=1.0e-8)
                axis = quat_apply(quat, local)
                # NOTE: named down_target, not target -- `target` is already the
                # coupling joint's reference index in this scope and shadowing it
                # indexes _joint_ref with a float tensor.
                down_target = torch.zeros_like(head_axis)
                down_target[:, 2] = -1.0
                # Component of each vector in the plane the joint can reach.
                target_p = (
                    down_target
                    - (down_target * axis).sum(-1, keepdim=True) * axis
                )
                head_p = head_axis - (head_axis * axis).sum(-1, keepdim=True) * axis
                cross = torch.cross(head_p, target_p, dim=-1)
                signed_tilt = -torch.atan2(
                    (cross * axis).sum(-1), (head_p * target_p).sum(-1)
                )
            else:
                signed_tilt = torch.atan2(head_axis[:, 1], -head_axis[:, 2])
            sign = float(
                getattr(self.cfg, "terrain_messerkopf_level_sign", -1.0)
            )
            bound = float(
                getattr(self.cfg, "terrain_messerkopf_level_max_rad", 0.15)
            )
            # Progressive response. A single linear gain cannot serve both ends
            # of the tilt range: the swept pair 0.5/0.08 saturates at 0.16 rad,
            # but the worst held-out sample sits at 18.5 deg (0.323 rad), so the
            # correction is pinned exactly where the hard-min gate is decided,
            # while simply widening it oscillates in the common small-tilt case
            # (clamp 0.15 measured min alignment 0.9407 against 0.9482 at 0.08).
            # The quadratic term is negligible near level and dominant in the
            # tail, which is the only region the hard-min gate reads.
            quadratic = float(
                getattr(self.cfg, "terrain_messerkopf_level_quadratic_gain", 0.0)
            )
            response = gain * signed_tilt + quadratic * signed_tilt * signed_tilt.abs()
            correction = torch.clamp(sign * response, -bound, bound)
            # Optional low-pass. The correction is recomputed every decimation
            # substep, so an unfiltered term injects its own high-frequency
            # motion into the tool -- and p95_tool_acceleration is a gate.
            beta = float(
                getattr(self.cfg, "terrain_messerkopf_level_filter", 0.0)
            )
            if beta > 0.0:
                previous = getattr(self, "_messerkopf_level_filtered", None)
                if previous is None or previous.shape != correction.shape:
                    previous = torch.zeros_like(correction)
                correction = beta * previous + (1.0 - beta) * correction
                self._messerkopf_level_filtered = correction.detach().clone()
            a = float(getattr(self.cfg, "joint6_coupling_a", 0.76))
            b = float(getattr(self.cfg, "joint6_coupling_b", 0.0))
            # One reference, three contributions: the coupling FEEDFORWARD that
            # makes the head track the boom it hangs from, the measured-tilt
            # levelling CORRECTION, and the policy's own integrated correction.
            # Dropping any one of them was measured worse -- no levelling gives
            # min_head_alignment 0.8662, a frozen (non-tracking) reference gives
            # 0.8369, and both together give 0.9296.
            if policy_correction is None:
                policy_correction = torch.zeros_like(correction)
            # Feedforward source: COMMANDED q4, or MEASURED q4.
            #
            # The head hangs off the boom, so its levelling feedforward is only
            # right if it points at where the boom actually IS. Under ROM-A the
            # commanded and measured q4 agree closely, so using the command is
            # harmless. Under a LEARNED actuator (PPO4's ID/FD) they do not:
            # the model has a systematic tracking error, so a command-based
            # feedforward aims the head at a boom pose that was never reached,
            # and the resulting head-alignment offset is injected downstream of
            # the policy every substep -- which is why PPO could not move
            # p01_head_alignment off 0.79 in ten blocks (~500 updates) while
            # every other metric improved.
            #
            # Reading the measured joint closes that loop and lets the head
            # compensate actuator model error structurally, instead of asking
            # the policy to learn a correction for an error it cannot observe.
            source_angle = self._joint_ref[:, source]
            if bool(
                getattr(self.cfg, "terrain_head_feedforward_from_measured", False)
            ):
                source_joint = getattr(
                    self, "_joint6_coupling_source_joint_id", None
                )
                if source_joint is not None:
                    source_angle = self.robot.data.joint_pos[:, int(source_joint)]

            # Pose-dependent head reference (the fix PPO3's own notes ask for:
            # "a pose-dependent reference that solves for the Messerkopf angle
            # putting the head at world-down given the current boom attitude").
            #
            # Measured on ID/FD, 30 080 scored samples: the a*q4 feedforward is
            # wrong by 0.296 rad mean / 0.783 rad max, while q4 itself spans
            # only 0.033-0.096 rad and the head tilt spans 0.68 rad. The head
            # angle is therefore NOT a function of q4, and re-fitting a/b cannot
            # help -- a least-squares fit over that range returns a=11.5, which
            # is fitting noise.
            #
            # Because the levelling term is PROPORTIONAL, it trims but cannot
            # null a standing offset, which is why releasing its clamp topped
            # out at p01 0.821 and cost 14/17 -> 11/17.
            #
            # Anchoring on the head's own measured angle removes the offset by
            # construction: the reference becomes "where the head is now, minus
            # how far it is currently off world-down".
            if bool(getattr(self.cfg, "terrain_head_pose_solve", False)):
                tool_joint = getattr(self, "_tool_joint_id", None)
                if tool_joint is not None:
                    measured_head = self.robot.data.joint_pos[:, int(tool_joint)]
                    solve_gain = float(
                        getattr(self.cfg, "terrain_head_pose_solve_gain", 1.0)
                    )
                    self._joint_ref[:, target] = (
                        measured_head
                        - solve_gain * signed_tilt
                        + policy_correction
                    )
                    self._terrain_head_level_correction = -solve_gain * signed_tilt
                    super()._apply_action()
                    if bool(
                        getattr(self.cfg, "terrain_constrain_truck_per_substep", True)
                    ):
                        self._constrain_truck_xy()
                    self._pin_substeps_pending = max(
                        int(getattr(self, "_pin_substeps_pending", 0)) - 1, 0
                    )
                    return
            self._joint_ref[:, target] = (
                a * source_angle
                + b
                + correction
                + policy_correction
            )
            self._terrain_head_level_correction = correction
        super()._apply_action()
        # Re-pin the chassis every decimation substep. The inherited fixture runs
        # once per control step, so with decimation=2 two physics substeps elapse
        # in which arm reaction translates and rotates the truck before the next
        # correction snaps it back -- visible as a small sideways sway when the
        # cutter strikes terrain. max_truck_post_constraint_drift_m reads 0.0000
        # because it is sampled AFTER the correction, so no gate could see this.
        # The correction is idempotent (it drives off base_link error), so the
        # extra call costs a pose write and cannot accumulate.
        if bool(getattr(self.cfg, "terrain_constrain_truck_per_substep", True)):
            self._constrain_truck_xy()
        # One physics substep is about to run, so consume one substep of the
        # advance. Decremented after the pin (the pin above serves the substep
        # that has NOT run yet) and outside the flag, so the inherited
        # `_get_dones` pin sees a closed window either way. Clamped at zero
        # because the reset deployment loops also call `_apply_action` with the
        # window already closed and travel frozen.
        self._pin_substeps_pending = max(
            int(getattr(self, "_pin_substeps_pending", 0)) - 1, 0
        )

    def _task_action_shield(
        self, applied_action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Block worsening c4 motion and optionally add XYZ-neutral recovery."""
        head_axis = self._head_down_axis_world()
        signed_head_angle = torch.atan2(head_axis[:, 1], -head_axis[:, 2])
        # NOTE: atan2(y, -z) is the tilt projected into the Y-Z plane, so this
        # shield is blind to an X-Z excursion (see the blind-axis tests in
        # tests/test_sole_actions.py). Fading on the true off-down angle instead
        # was measured to be strictly worse: on the same checkpoint and held-out
        # set it drove min_head_alignment 0.881 -> -0.168 and max_abs_head_axis_y
        # 0.474 -> 0.979. The total angle is dominated by X, which c4 cannot
        # affect, so it zeroed c4 authority almost permanently while the
        # worsening sign kept chattering with y -- destroying the very authority
        # that holds the head down in c4's own plane. Out-of-plane excursions are
        # the head barrier's job, not the shield's; it sees alignment directly
        # and is direction-agnostic.
        shielded, shield_gain = directional_tilt_shield(
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
        restore_gain = float(
            getattr(self.cfg, "terrain_head_nullspace_restore_gain", 0.0)
        )
        if restore_gain <= 0.0:
            self._terrain_head_nullspace_overlay = torch.zeros_like(shielded)
            return shielded, shield_gain

        preserve_xyz = bool(
            getattr(self.cfg, "terrain_head_restore_preserve_xyz", True)
        )
        direction = getattr(self, "_terrain_head_nullspace_direction", None)
        if not preserve_xyz:
            direction = torch.tensor(
                [0.0, 0.0, 0.0, 1.0],
                device=shielded.device,
                dtype=shielded.dtype,
            )
        elif direction is None:
            # The fixed ground-pose Jacobian maps learned cylinder extension to
            # cutter XYZ. Convert it to policy-action coordinates, then take its
            # one-dimensional nullspace. Normalizing c4 to +1 makes the existing
            # measured action-to-tilt sign directly applicable.
            jacobian = self._cascade.jacobian.to(
                device=shielded.device, dtype=shielded.dtype
            )
            action_jacobian = jacobian * self._action_to_learned_sign.reshape(1, 4)
            direction = torch.linalg.svd(action_jacobian, full_matrices=True).Vh[-1]
            if direction[3] < 0.0:
                direction = -direction
            direction = direction / direction[3].clamp(min=1.0e-6)
            self._terrain_head_nullspace_direction = direction.detach().clone()

        _, overlay = nullspace_head_restoring_overlay(
            shielded,
            signed_tilt=signed_head_angle,
            action_nullspace=direction,
            action_to_tilt_sign=float(
                getattr(self.cfg, "terrain_c4_action_to_axis_y_sign", 1.0)
            ),
            deadband_rad=float(
                getattr(self.cfg, "terrain_head_nullspace_restore_deadband_rad", 0.08)
            ),
            gain=restore_gain,
            max_c4_action=float(
                getattr(self.cfg, "terrain_head_nullspace_restore_max_action", 0.30)
            ),
        )
        measured, _ = self._refresh_cylinder_state()
        low, high = self._cylinder_headroom(measured)
        safe_overlay, overlay_gain = limit_shield(
            overlay,
            headroom_low_m=low,
            headroom_high_m=high,
            margin_m=float(getattr(self.cfg, "sole_limit_shield_margin_m", 0.0)),
            action_to_extension_sign=self._action_to_learned_sign,
        )
        restored = torch.clamp(shielded + safe_overlay, -1.0, 1.0)
        self._terrain_head_nullspace_overlay = safe_overlay.detach().clone()
        active_overlay = overlay.abs() > 1.0e-8
        combined_gain = shield_gain * torch.where(
            active_overlay, overlay_gain, torch.ones_like(overlay_gain)
        )
        return restored, combined_gain

    @torch.no_grad()
    def _model_error_channels(self, measured_x: torch.Tensor) -> torch.Tensor | None:
        """Twelve columns describing how wrong the actuator model currently is.

        Returns ``None`` only in legacy compatibility mode.

        The three groups answer three different questions, and a learned
        actuator gets all three wrong in different ways:

        * tracking residual -- did the cylinder arrive where it was commanded?
        * prediction residual -- did it move by what the model predicted it
          would move LAST step? This is the only channel that carries the
          model's own error signal, and it needs the stored previous
          prediction, which is why a memoryless policy cannot derive it.
        * effort -- what load is the actuator working against? Contact load is
          where a model fitted on free motion degrades first.
        """
        if not bool(getattr(self.cfg, "terrain_model_error_observation", False)):
            return None
        # Per-cylinder stroke span, a tensor -- not a scalar. The rest of the
        # observation already normalizes headroom by it the same way.
        span = getattr(self, "_learned_span", None)
        if span is None:
            span = torch.ones_like(measured_x)
        span = torch.as_tensor(span, dtype=measured_x.dtype, device=measured_x.device)
        span = torch.where(span.abs() < 1.0e-6, torch.ones_like(span), span)
        target = getattr(self, "_cylinder_pos_target", None)
        if target is None:
            target = measured_x
        tracking = (measured_x - target) / span

        previous_x = getattr(self, "_prev_measured_x_model_error", None)
        previous_pred = getattr(self, "_prev_predicted_delta_model_error", None)
        if previous_x is None or previous_x.shape != measured_x.shape:
            realized = torch.zeros_like(measured_x)
        else:
            realized = measured_x - previous_x
        if previous_pred is None or previous_pred.shape != measured_x.shape:
            previous_pred = torch.zeros_like(measured_x)
        prediction_residual = (realized - previous_pred) / span

        self._prev_measured_x_model_error = measured_x.detach().clone()
        predicted = getattr(self, "_fd_delta_corrected", None)
        self._prev_predicted_delta_model_error = (
            torch.zeros_like(measured_x)
            if predicted is None
            else predicted.detach().clone()
        )

        scale = float(getattr(self.cfg, "terrain_model_error_effort_scale_n", 2.0e5))
        effort = self.robot.data.applied_torque[:, self._cylinder_joint_ids]
        effort = torch.clamp(effort / max(scale, 1.0e-6), -5.0, 5.0)

        current = torch.cat(
            (
                torch.clamp(tracking, -5.0, 5.0),
                torch.clamp(prediction_residual, -5.0, 5.0),
                effort,
            ),
            dim=-1,
        )

        # Optional history stack. A single-step residual says how wrong the
        # model is NOW; it says nothing about whether the error is growing, or
        # how it tracks load. teacher_student reaches the same conclusion from
        # the other direction and puts history_length=3 on its rate-bearing
        # terms (height map, joint pos/vel, distance-to-goal). Stacking only
        # these 12 channels keeps the cost trivial -- depth 3 adds 24 inputs to
        # an MLP, versus the depth camera that repo also uses, which will not
        # fit alongside 512 environments on a 6 GB card.
        depth = max(int(getattr(self.cfg, "terrain_model_error_history", 1)), 1)
        if depth <= 1:
            return current
        buffer = getattr(self, "_model_error_history", None)
        if buffer is None or buffer.shape != (depth, *current.shape):
            buffer = current.unsqueeze(0).repeat(depth, 1, 1)
        else:
            buffer = torch.roll(buffer, shifts=1, dims=0)
            buffer[0] = current
        self._model_error_history = buffer.detach()
        return buffer.permute(1, 0, 2).reshape(current.shape[0], -1)

    @torch.no_grad()
    def _contact_state_channels(self) -> torch.Tensor | None:
        """Six columns describing the physical contact state. See
        ``terrain_observations.build_verge_observation`` for the ordering and
        for why the pre-existing clipped force column is not enough."""
        if not bool(getattr(self.cfg, "terrain_contact_observation", False)):
            return None
        self._update_contact_safety()
        cfg = self.cfg
        force_scale = max(
            float(getattr(cfg, "cutter_contact_force_observation_scale_n", 1000.0)),
            1.0,
        )
        rate_scale = max(
            float(getattr(cfg, "terrain_contact_force_rate_scale_nps", 1.0e6)), 1.0
        )
        gap_scale = max(
            float(getattr(cfg, "terrain_contact_gap_observation_scale_m", 0.02)),
            1.0e-6,
        )
        grace = max(int(getattr(cfg, "terrain_contact_loss_grace_steps", 2)), 1)

        contact = self._contact_state.float()
        # log1p, not a clip: the measured force spans 0 to 170 kN and a linear
        # channel clipped at 10 kN makes the whole operating range one value.
        log_force = torch.log1p(
            torch.clamp(self._contact_force_n, min=0.0) / force_scale
        )
        force_rate = torch.clamp(
            self._contact_force_rate_nps / rate_scale, -5.0, 5.0
        )
        gap = torch.clamp(self._airborne_gap_m / gap_scale, 0.0, 10.0)
        acquired = self._contact_acquired.float()
        loss_steps = torch.clamp(
            self._contact_loss_steps.float() / float(grace), 0.0, 10.0
        )
        return torch.stack(
            (contact, log_force, force_rate, gap, acquired, loss_steps), dim=-1
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
            model_error=self._model_error_channels(x),
            model_error_history=int(
                getattr(self.cfg, "terrain_model_error_history", 1)
            ),
            contact_state=self._contact_state_channels(),
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
        # Cached: `_get_dones` already ran the state machine for this step, so
        # this returns immediately and reward, termination and metrics all read
        # the SAME transition.
        self._update_contact_safety()
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
            head_penalty_far_weight=float(
                getattr(cfg, "terrain_head_penalty_far_weight", -0.5)
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
            chassis_lateral_reaction_m=self._chassis_lateral_reaction_m,
            chassis_attitude_reaction_rad=self._chassis_attitude_reaction_rad,
            chassis_reaction_weight=float(
                getattr(cfg, "terrain_chassis_reaction_weight", 0.0)
            ),
            chassis_lateral_scale_m=float(
                getattr(cfg, "terrain_chassis_lateral_scale_m", 0.005)
            ),
            chassis_attitude_scale_rad=float(
                getattr(cfg, "terrain_chassis_attitude_scale_rad", 0.005)
            ),
            chassis_reaction_max_ratio=float(
                getattr(cfg, "terrain_chassis_reaction_max_ratio", 5.0)
            ),
        )

        active = (self._preposition_counter <= 0) & (self._settle_counter <= 0)
        contact_terms = None
        if bool(getattr(cfg, "terrain_contact_safety_reward", False)):
            contact_terms = compute_contact_safety_reward(
                contact=self._contact_state,
                contact_acquired=self._contact_acquired,
                contact_force_n=self._contact_force_n,
                airborne_gap_m=self._airborne_gap_m,
                cutter_vertical_velocity_mps=self._cutter_vertical_velocity_mps,
                hard_impact=self._hard_impact_event,
                active=active,
                support_weight=float(
                    getattr(cfg, "terrain_contact_support_weight", 4.0)
                ),
                loss_weight=float(getattr(cfg, "terrain_contact_loss_weight", 20.0)),
                force_weight=float(
                    getattr(cfg, "terrain_contact_force_weight", 2.0)
                ),
                force_target_n=float(
                    getattr(cfg, "terrain_contact_force_target_n", 6000.0)
                ),
                force_std_n=float(
                    getattr(cfg, "terrain_contact_force_std_n", 8000.0)
                ),
                excess_weight=float(
                    getattr(cfg, "terrain_contact_excess_weight", 2.0)
                ),
                force_soft_limit_n=float(
                    getattr(cfg, "terrain_contact_force_soft_limit_n", 20000.0)
                ),
                excess_max_ratio=float(
                    getattr(cfg, "terrain_contact_excess_max_ratio", 5.0)
                ),
                hard_impact_weight=float(
                    getattr(cfg, "terrain_contact_hard_impact_weight", 20.0)
                ),
                takeoff_weight=float(
                    getattr(cfg, "terrain_contact_takeoff_weight", 4.0)
                ),
                takeoff_velocity_scale_mps=float(
                    getattr(cfg, "terrain_contact_takeoff_velocity_scale_mps", 0.10)
                ),
                takeoff_max_ratio=float(
                    getattr(cfg, "terrain_contact_takeoff_max_ratio", 5.0)
                ),
                weak_support_force_n=float(
                    getattr(cfg, "terrain_contact_weak_support_force_n", 500.0)
                ),
                gap_weight=float(getattr(cfg, "terrain_contact_gap_weight", 8.0)),
                gap_scale_m=float(
                    getattr(cfg, "terrain_contact_gap_scale_m", 0.005)
                ),
                gap_max_ratio=float(
                    getattr(cfg, "terrain_contact_gap_max_ratio", 4.0)
                ),
            )
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
        tail_weights = lower_tail_weights(
            alignment,
            active_mask=active,
            fraction=float(getattr(cfg, "terrain_head_tail_fraction", 0.0)),
            multiplier=float(getattr(cfg, "terrain_head_tail_multiplier", 1.0)),
        )
        weighted_head_penalty = terms.head_penalty * tail_weights
        if not hasattr(self, "_head_tail_failure_memory"):
            self._head_tail_failure_memory = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )
        memory_steps = int(getattr(cfg, "terrain_head_tail_memory_steps", 0))
        self._head_tail_failure_memory = update_failure_memory(
            self._head_tail_failure_memory,
            trigger=alignment
            < float(getattr(cfg, "terrain_head_tail_memory_trigger", 0.975)),
            active_mask=active,
            horizon_steps=memory_steps,
        )
        tail_memory_penalty = float(
            getattr(cfg, "terrain_head_tail_memory_penalty", 0.0)
        ) * (self._head_tail_failure_memory > 0).float()
        task_reward = (
            terms.cut_tracking
            + terms.lateral
            + terms.contact_reward
            + dense_cut
            + dense_lateral
            + cut_progress
            + lateral_progress
        )
        # Head-down is a precondition for being paid to track, not a competing
        # objective. Terminating on a violation is not affordable here: any
        # termination synchronizes (PPO2 policy/env.py) and _run_reset_preposition
        # steps the whole scene, so at 512 envs a per-step head check would reset
        # continuously. Latching the failure for the rest of the episode buys the
        # same guarantee for free -- once the head leaves the cone that env earns
        # no tracking reward again until reset, so no rollout can ever profit from
        # mowing with a tilted head. The safety barriers stay live, so recovering
        # still pays and the policy keeps a gradient back toward world-down.
        required = float(
            getattr(self.cfg, "terrain_head_down_required_alignment", 0.98)
        )
        if not hasattr(self, "_head_down_failed"):
            self._head_down_failed = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
        # Hysteresis release. The latch above is monotone within an episode and
        # is cleared only at reset, so a single transient dip zeroes task reward
        # for every remaining step -- up to the full 16 s horizon for one bad
        # contact. Measured consequence across five blocks on the rough verge
        # (v3 2-4, v4 1, v5 1): the latched fraction spikes at the same
        # phase-locked hard windows, Loss/value goes 1-3k -> 80-230k as returns
        # become bimodal, and every one of those blocks scored worse than the
        # checkpoint it resumed from. The only promoted block held latch mean
        # 0.0026; every collapse ran 0.032-0.109. Softening the head barrier and
        # zeroing the entropy bonus were both tried against this and both failed
        # to prevent it (entropy did cut latch mean 0.109 -> 0.032 and the score
        # 6.89 -> 2.19, so it is kept, but it is not the mechanism).
        #
        # Releasing at a strictly higher alignment than `required` preserves the
        # guarantee that matters -- an env earns nothing while tilted, so no
        # rollout can profit from mowing with a tilted head -- while making the
        # cost of a dip proportional to its duration instead of to whenever it
        # happened to occur in the episode. The gap between the two thresholds
        # is what stops a head oscillating across the boundary from collecting
        # partial reward on the upswing.
        release = float(
            getattr(self.cfg, "terrain_head_down_latch_release_alignment", 0.97)
        )
        if bool(getattr(self.cfg, "terrain_head_down_latches_failure", True)):
            self._head_down_failed |= active & (alignment < required)
            if release > required:
                recovered = self._head_down_failed & active & (alignment >= release)
                self._head_down_failed &= ~recovered
                self._head_down_latch_released = recovered
            else:
                self._head_down_latch_released = torch.zeros_like(
                    self._head_down_failed
                )
            task_reward = torch.where(
                self._head_down_failed, torch.zeros_like(task_reward), task_reward
            )
        else:
            self._head_down_latch_released = torch.zeros_like(self._head_down_failed)

        # terms.reward less its three task components; these stay live whether or
        # not the head has failed, so barriers keep shaping and recovery pays.
        chassis_penalty = terms.chassis_reaction_penalty
        if chassis_penalty is None:
            chassis_penalty = torch.zeros_like(terms.limit_penalty)
        safety_reward = (
            terms.smoothness
            + weighted_head_penalty
            + tail_memory_penalty
            + terms.clearance_barrier
            + terms.limit_penalty
            + terms.acceleration_penalty
            # Kept in the safety group, not the task group: a latched head must
            # not stop paying for keeping the truck steady, or an env that dips
            # once is free to shove the chassis for the rest of the episode.
            + chassis_penalty
        )
        if contact_terms is not None:
            # Safety group, for the same reason: an env whose head has latched
            # must still be paid for staying planted and charged for lifting,
            # or a single head dip licenses it to fly for the rest of the
            # episode with the contact penalty switched off.
            safety_reward = safety_reward + contact_terms.total
        shaped_reward = safety_reward + task_reward + head_progress
        shaped_reward = torch.where(
            active, shaped_reward, torch.zeros_like(shaped_reward)
        )
        # A contact-safety termination must not be able to pay. Applied after
        # the `active` mask so a warm-up step cannot receive it, and before the
        # non-finite sanitizer so the two cannot disagree about the same row.
        contact_terminal = getattr(self, "_contact_safety_terminated", None)
        if contact_terminal is not None and bool(contact_terminal.any()):
            shaped_reward = torch.where(
                contact_terminal & active,
                torch.full_like(
                    shaped_reward,
                    float(getattr(cfg, "terrain_lift_terminal_reward", -100.0)),
                ),
                shaped_reward,
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
                # Pre-correction chassis error: the sway the fixture erases
                # before `truck/max_post_constraint_drift_m` can see it.
                "truck/mean_chassis_lateral_reaction_mm": 1000.0
                * self._chassis_lateral_reaction_m[valid_active].mean(),
                "truck/p95_chassis_lateral_reaction_mm": 1000.0
                * torch.quantile(
                    self._chassis_lateral_reaction_m[valid_active], 0.95
                ),
                "truck/p95_chassis_attitude_reaction_deg": torch.rad2deg(
                    torch.quantile(
                        self._chassis_attitude_reaction_rad[valid_active], 0.95
                    )
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
                "safety/head_down_failed_fraction": self._head_down_failed[
                    valid_active
                ].float().mean(),
                # Envs whose head physically recovered this step and had their
                # task reward restored. Under the old monotone latch this is
                # exactly the population that stayed unpaid for the rest of the
                # episode, so a non-trivial value here measures the mechanism
                # directly rather than inferring it from the collapse.
                "safety/head_down_latch_released_fraction": getattr(
                    self,
                    "_head_down_latch_released",
                    torch.zeros_like(self._head_down_failed),
                )[valid_active].float().mean(),
                "safety/max_abs_head_axis_x": torch.max(
                    self._head_down_axis_world()[valid_active, 0].abs()
                ),
                "safety/p95_tool_acceleration_mps2": torch.quantile(
                    acceleration[valid_active], 0.95
                ),
                "reward/nonfinite_fraction": reward_nonfinite.float().mean(),
                "reward/mean_head_penalty": weighted_head_penalty[
                    valid_active
                ].mean(),
                "reward/head_tail_weighted_fraction": (
                    tail_weights[valid_active] > 1.0
                ).float().mean(),
                "reward/head_tail_memory_fraction": (
                    self._head_tail_failure_memory[valid_active] > 0
                ).float().mean(),
                "reward/mean_head_tail_memory_penalty": tail_memory_penalty[
                    valid_active
                ].mean(),
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
                # Contact-safety instrumentation. `ground_contact_fraction`
                # above is a DUTY CYCLE and reads 0.9486 on a policy measured
                # 45 consecutive steps and 297 mm off the verge, so it cannot
                # be the signal training watches for lift.
                "contact/acquired_fraction": self._contact_acquired[
                    valid_active
                ].float().mean(),
                "contact/loss_event_fraction": self._contact_loss_event[
                    valid_active
                ].float().mean(),
                "contact/takeoff_event_fraction": self._takeoff_event[
                    valid_active
                ].float().mean(),
                "contact/impact_ejection_event_fraction": self._impact_ejection_event[
                    valid_active
                ].float().mean(),
                "contact/hard_impact_event_fraction": self._hard_impact_event[
                    valid_active
                ].float().mean(),
                "contact/visible_lift_event_fraction": self._visible_lift_event[
                    valid_active
                ].float().mean(),
                "contact/mean_airborne_gap_m": self._airborne_gap_m[
                    valid_active
                ].mean(),
                "contact/max_airborne_gap_m": self._airborne_gap_m[
                    valid_active
                ].max(),
                "contact/max_consecutive_airborne_steps": self._max_consecutive_airborne_steps[
                    valid_active
                ].float().max(),
                "contact/mean_force_rate_nps": self._contact_force_rate_nps[
                    valid_active
                ].abs().mean(),
                "contact/max_positive_cutter_vertical_velocity_mps": self._cutter_vertical_velocity_mps[
                    valid_active
                ].max(),
            }
            if contact_terms is not None:
                self._last_reward_terms.update(
                    {
                        "reward/contact_total": contact_terms.total[
                            valid_active
                        ].mean(),
                        "reward/contact_support": contact_terms.support[
                            valid_active
                        ].mean(),
                        "reward/contact_loss": contact_terms.contact_loss[
                            valid_active
                        ].mean(),
                        "reward/contact_force_regulation": contact_terms.force_regulation[
                            valid_active
                        ].mean(),
                        "reward/contact_excess_force": contact_terms.excess_force[
                            valid_active
                        ].mean(),
                        "reward/contact_hard_impact": contact_terms.hard_impact[
                            valid_active
                        ].mean(),
                        "reward/contact_takeoff": contact_terms.takeoff[
                            valid_active
                        ].mean(),
                        "reward/contact_airborne_gap": contact_terms.airborne_gap[
                            valid_active
                        ].mean(),
                    }
                )
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
