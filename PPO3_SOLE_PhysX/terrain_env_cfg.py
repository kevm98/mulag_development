"""Configuration for PPO3 on terrain with a driving truck."""

from __future__ import annotations

from pathlib import Path

from isaaclab.utils import configclass

from PPO3_SOLE_PhysX.env_cfg import (
    SolePolicyEnvCfg,
    SolePolicyPPORunnerCfg,
    apply_sole_curriculum_stage,
)
from PPO2_ROM_PhysX.runtime.sim_memory import apply_kevin_sim_memory_optimizations


def configure_terrain_gpu_physx_memory(cfg, num_envs: int) -> None:
    """Scale GPU PhysX contact buffers to the actual vectorization width.

    ``DirectRLEnvCfg.__post_init__`` runs before ``parse_env_cfg`` applies the
    CLI ``--num_envs`` override, so the inherited helper otherwise sizes every
    scene as one environment. At 512 environments this overflowed
    ``collisionStackSize`` and PhysX dropped the very cutter/road contacts the
    task scores. This must run after parsing and before ``gym.make``.
    """
    count = max(int(num_envs), 1)
    apply_kevin_sim_memory_optimizations(
        cfg.sim, num_envs=count, verbose=False
    )
    physx = cfg.sim.physx
    print(
        "[PPO3-PHYSX-MEMORY] "
        f"num_envs={count} "
        f"contacts={int(physx.gpu_max_rigid_contact_count)} "
        f"pairs={int(physx.gpu_found_lost_pairs_capacity)} "
        f"aggregate_pairs={int(physx.gpu_total_aggregate_pairs_capacity)} "
        f"collision_stack={int(physx.gpu_collision_stack_size)} "
        f"heap={int(physx.gpu_heap_capacity)} "
        f"temp={int(physx.gpu_temp_buffer_capacity)}",
        flush=True,
    )


def apply_terrain_curriculum_stage(cfg, stage: int) -> None:
    """Ramp tracking precision and episode horizon toward the full path.

    Short early episodes keep exploratory velocity commands from leaving the
    arm at a limit for most of a 12-second rollout. Stage 4 restores the full
    production horizon used by convergence evaluation.
    """
    apply_sole_curriculum_stage(cfg, stage)
    stage = int(stage)
    if stage == 1:
        cfg.cut_tracking_std_m = 0.20
        cfg.lateral_tracking_std_m = 2.0
        cfg.episode_length_s = 2.0
    elif stage == 2:
        cfg.cut_tracking_std_m = 0.12
        cfg.lateral_tracking_std_m = 1.2
        cfg.episode_length_s = 4.0
    elif stage == 3:
        cfg.cut_tracking_std_m = 0.07
        cfg.lateral_tracking_std_m = 0.60
        cfg.episode_length_s = 8.0
    elif stage == 4:
        cfg.cut_tracking_std_m = 0.03
        cfg.lateral_tracking_std_m = 0.30
        cfg.episode_length_s = 16.0
        # Kept explicit for terrain A/B studies. The 0.85 filter improved head
        # but barely changed acceleration (234.8 -> 233.5 m/s^2) and regressed
        # strikes/lateral, so production remains at the learned 0.70 contract.
        cfg.sole_action_filter = float(
            getattr(cfg, "terrain_action_filter", 0.70)
        )


@configclass
class TerrainSolePolicyEnvCfg(SolePolicyEnvCfg):
    """Sole-controller arm, constant-speed truck, generated mowing terrain."""

    # The inherited flag samples a phase of PPO2's stored arm trajectory during
    # reset.  PPO3 has no scored stored path: every episode must deploy to the
    # same mower pose, with diversity coming from the per-environment terrain.
    # Leaving this enabled made the chosen synchronized reset template depend on
    # the process seed (measured 1.89 m lateral error for seed 10018).
    randomize_start_phase: bool = False
    # Measured from the best full-physics legacy-arc deployment candidate across
    # 64 environments (seed 17). It places the cutter at env-local y=1.446 m for
    # path B at y=1.380 m, with head alignment above 0.98. TerrainEnv installs
    # this as both reset endpoints so deployment is deterministic and does not
    # depend on random phase sampling or the number of parallel environments.
    terrain_deployment_trajectory_shape: str = "legacy_arc"
    terrain_deployment_cylinder_target_m: tuple = (
        0.33238252997398376,
        0.5770000219345093,
        0.46299999952316284,
        0.5889909863471985,
    )
    # Cylinder 4 owns cutter-head pitch and must adapt as the first three
    # cylinders change the boom attitude. A fixed c4 pose passed short reset
    # checks but fell to 0.92 world-down alignment over the full terrain path.
    # Give PPO all four axes and expose signed head tilt in the observation;
    # the strict head barrier and corrected limit shield provide safety shaping.
    sole_action_authority: tuple = (1.0, 1.0, 1.0, 1.0)
    trajectory_library_path: str = str(
        Path(__file__).resolve().parents[1]
        / "PPO2_ROM_PhysX/models/plant_reachable_trajectory_library_v3_large.json"
    )
    # Make the task itself select canonical ROM-A. Entry-point flags may still
    # override this, but omitting a flag must never silently select command_run_001.
    command_rom_checkpoint: str = str(
        Path(__file__).resolve().parents[1]
        / "ROM-cylinders_pipeline/checkpoints/plant_reachable_v1_finetune_blend25/best_rom.pt"
    )

    # --- Vehicle ---
    # Scripted, not policy-controlled. teacher_student samples 3-8 km/h per
    # episode; a single constant speed is the simpler starting point and is what
    # "the truck moving at a constant speed" asks for. Set min < max to sample.
    enable_scripted_vehicle_motion: bool = True
    vehicle_speed_min_kmh: float = 4.0
    vehicle_speed_max_kmh: float = 4.0
    vehicle_steering_target_rad: float = 0.0
    drive_direction_xy: tuple = (1.0, 0.0)
    """Unit heading the chassis is advanced along, in env-local XY."""
    truck_ride_height_target_m: float = -0.015
    """Base-link height that places the authored tyre bottoms on the road.

    This is an authored frame coordinate, not physical clearance. At -0.015 m
    all four 0.575 m-radius tyre bottoms measure within 0.3 mm of road z=0;
    filtered wheel-road forces verify support instead of inferring it from the
    misleading base-link origin.
    """

    road_start_x_m: float = -40.0
    road_length_m: float = 40.0
    road_center_y_m: float = 4.30
    road_width_m: float = 3.60
    road_color: tuple = (0.11, 0.12, 0.13)
    road_roughness: float = 0.92
    """Dark, rough asphalt-like road, physically separate from the green verge."""
    dark_visual_background: bool = True
    """Hide the bright dome from camera rays while retaining its illumination."""
    enable_visual_grid_floor: bool = False
    visual_floor_color: tuple = (0.30, 0.31, 0.31)
    visual_grid_color: tuple = (0.52, 0.53, 0.54)
    visual_grid_length_m: float = 70.0
    visual_grid_width_m: float = 26.0
    visual_grid_spacing_m: float = 2.0
    visual_grid_line_width_m: float = 0.025
    visual_grid_center_x_m: float = -20.0
    visual_grid_center_y_m: float = 3.0
    road_contact_force_threshold_n: float = 10.0

    wheel_drive_mode: str = "kinematic"
    """"teacher_student" | "velocity_drive".

    "teacher_student" replicates their working setup exactly: leave the asset
    gains alone (stiffness 6e8, damping 2e6) and only write joint velocity each
    step, never a wheel position target. The position target is what turns those
    gains into PPO2's parking brake.

    "velocity_drive" is the textbook alternative (stiffness 0, damping, velocity
    target). Measured 2026-07-28 it drove exactly one wheel of four correctly --
    Rear_Left tracked 1.93 rad/s while Rear_Right free-spun to 22.5 and both
    front wheels turned backwards -- so it is retained only for comparison.
    """
    wheel_drive_damping: float = 2.0e6
    """Velocity-drive damping once the parking brake is released.

    Matches teacher_student's `unimog_wheels_drive` damping. With stiffness 0 this
    is the gain on (target_velocity - velocity), i.e. how hard the wheels chase
    the commanded speed.
    """

    # --- Terrain ---
    use_eval_terrain: bool = False
    terrain_curriculum_version: str = "path_height_v2"
    terrain_base_seed: int = 42
    """Training mesh seed; entry points vary it between resumed PPO blocks."""
    randomize_terrain_per_env: bool = True
    """Give each parallel environment a distinct deterministic verge phase."""
    terrain_seed_stride: int = 7919
    """Prime stride separating per-environment terrain seeds."""
    terrain_static_friction: float = 1.0
    terrain_dynamic_friction: float = 1.0
    terrain_restitution: float = 0.0

    # --- Verge strip ---
    mow_strip_color: tuple = (0.08, 0.55, 0.08)
    mow_strip_collision: bool = True
    """The green verge is a real collider; contact cannot be faked geometrically."""
    mow_strip_width_m: float = 1.8
    mow_strip_resolution_m: float = 0.25
    """Collision/ray mesh spacing.

    Matches the height-scan spacing and still gives ten samples across the
    shortest 2.5 m terrain wavelength. At 512 environments this reduces the
    merged mesh from 24.6 million to about 3.9 million triangles.
    """
    mow_strip_length_m: float = 26.0
    """Per-environment strip length.

    A 16 s episode at 4 km/h covers 17.8 m; 26 m includes the lead-in and exit
    margin while allowing a complete 16 m lateral trajectory.
    """
    tracking_line_color: tuple = (0.95, 0.08, 0.05)
    tracking_line_width_m: float = 0.07
    tracking_line_z_offset_m: float = 0.012
    """Visual-only line projected onto the exact terrain centre path."""
    tracking_path_family: str = "mixed"
    """Cycle every path family across parallel environments."""
    tracking_path_family_offset: int = 0
    terrain_height_scales: tuple = (0.50, 0.70, 0.85, 1.00)
    """Balanced relief levels crossed independently with every path family."""
    terrain_height_scale_override: float = 0.0
    """Positive visualization/probe override; zero uses the balanced curriculum."""

    # --- Ground sensing ---
    cutter_body_name: str = "Messerkopf"
    cutter_prim_subpath: str = "Rigid_Bodies/Arm_Links/Messerkopf"
    """USD path of the cutter body relative to the robot prim.

    This asset nests bodies under ``Rigid_Bodies/Arm_Links/``; teacher_student's
    copy of the Unimog exposes them directly under ``/Robot``, so its sensor
    paths ("{ENV_REGEX_NS}/Robot/Messerkopf") do not transfer. Sensors attach by
    USD path, not by articulation body name, so this has to be the real path.
    """
    height_scan_base_offset: tuple = (6.20, -2.82, 2.0)
    """Height-scan centre in the fixed drive frame, 1 m ahead of the cutter.

    The origin follows chassis translation but deliberately ignores chassis
    attitude, preventing articulation motion from swinging the scan off the
    verge. With the 2 m grid, its rear row is over the cutter and the remaining
    rows preview path B. X is forward along ``drive_direction_xy`` and Y is
    path-left.
    """
    height_scan_offset_z: float = 2.0
    """Ray origin height above the cutter frame. Must clear the tallest relief."""
    height_scan_length_m: float = 2.0
    height_scan_width_m: float = 0.5
    height_scan_resolution_m: float = 0.25
    cutter_contact_force_threshold_n: float = 10.0
    cutter_contact_force_observation_scale_n: float = 1000.0

    # Non-PPO contact-tracking controller used only to validate and visualize
    # the physical task before training is allowed to resume.
    enable_contact_tracking_demo: bool = False
    contact_demo_arm_stiffness_scale: tuple = (1.0, 1.0, 1.0, 1.0, 1.0)
    """Per-joint compliance for the non-PPO physical contact demonstration.

    Keep the controlled arm geometry stiff enough to remain on the red line;
    collision depenetration supplies the contact compliance.
    """
    contact_demo_target_force_n: float = 2000.0
    contact_demo_force_deadband_n: float = 1500.0
    contact_demo_initial_vertical_offset_m: float = -0.11
    contact_demo_preload_m: float = 0.003
    contact_demo_search_step_m: float = 0.0005
    contact_demo_force_step_m: float = 0.00025
    contact_demo_max_vertical_offset_m: float = 0.30
    contact_demo_max_upward_offset_m: float = -0.05
    """Never let force relief lift the nominal cutter goal clear of the verge."""
    contact_demo_memory_steps: int = 3
    contact_demo_reseat_step_m: float = 0.00015
    contact_demo_max_reseat_m: float = 0.06
    contact_demo_hard_force_n: float = 30000.0
    contact_demo_hard_force_relief_step_m: float = 0.001
    contact_demo_dls_gain: float = 0.30
    contact_demo_lateral_error_weight: float = 8.0
    contact_demo_lateral_load_bias_m: float = 0.0
    """Compensate the measured outward deflection under cutter contact load."""
    contact_demo_max_cylinder_step_m: float = 0.0030
    contact_demo_line_tolerance_m: float = 0.30
    """Maximum cutter-centre error; comfortably inside the physical head width."""
    terrain_teacher_max_raw_action: float = 0.50
    """Teacher command cap in the student's normalized ROM-A action space."""
    terrain_teacher_head_gain: float = 1.0
    """Closed-loop c4 gain on signed cutter-axis tilt during distillation."""

    # --- Mowing target ---
    target_cut_height_m: float = 0.0
    cutter_contact_reference_height_m: float = -0.30
    """Legacy unrotated cutter height observed near physical ground support.

    This is used only to shape the positive gap while the contact sensor says
    the cutter is airborne. Real cutter/verge interaction, not this coordinate,
    defines zero tracking error.
    """
    terrain_deployment_correction_steps: int = 0
    terrain_deployment_settle_steps: int = 30
    synchronize_deployed_robot_state: bool = True
    synchronized_deployment_template_env: int = 0
    """Fixed clone copied after deployment; keeps 16/128/512-env starts identical."""
    terrain_arm_damping_scale: float = 1.0
    """Stock hydraulic damping retained for stable physical ground contact."""
    cut_tracking_std_m: float = 0.03
    cut_tracking_weight: float = 5.0
    """Commanded cutter height above the sensed terrain surface."""
    lateral_tracking_std_m: float = 0.30
    lateral_tracking_weight: float = 2.0
    terrain_head_penalty_weight: float = -20.0
    terrain_head_threshold: float = 0.995
    terrain_head_penalty_scale: float = 0.02
    terrain_head_penalty_max_deviation: float = 2.0
    terrain_head_tilt_shield_soft: float = 0.08
    terrain_head_tilt_shield_hard: float = 0.16
    terrain_c4_action_to_axis_y_sign: float = 1.0
    # Start warning one centimetre below the 5 cm mowing target, well before a
    # geometric strike. The initial 2 cm / -2 barrier left held-out quarters 1
    # and 2 systematically below target and the strike rate flat at 32.6%.
    terrain_clearance_barrier_weight: float = 0.0
    terrain_clearance_margin_m: float = 0.0
    terrain_clearance_barrier_scale_m: float = 0.02
    terrain_clearance_barrier_max_deviation: float = 5.0
    terrain_contact_reward_weight: float = 2.0
    terrain_acceleration_max_ratio: float = 10.0
    """Bound acceleration shaping at 10 x scale while retaining signal to 300 m/s²."""
    terrain_nonfinite_terminal_reward: float = -100.0
    """Finite reward used for a corrupt transition before synchronized recovery."""
    dense_cut_error_weight: float = 5.0
    dense_lateral_error_weight: float = 3.0
    terrain_cut_progress_weight: float = 1.0
    terrain_cut_progress_scale_m: float = 0.02
    terrain_lateral_progress_weight: float = 0.5
    terrain_lateral_progress_scale_m: float = 0.01
    terrain_head_progress_weight: float = 0.25
    terrain_head_progress_scale: float = 0.002
    # A terrain action is a desired cylinder motion, not a pose residual.  A
    # persistent non-zero mean therefore keeps driving the arm.  Model 372 paid
    # only ~7e-4 reward for its measured c1/c3 DC commands under the inherited
    # -0.02 coefficient; -1 makes settling visible without overwhelming task or
    # safety terms (about -0.04 at its observed mean-square action).
    terrain_action_magnitude_weight: float = -1.0
    terrain_action_filter: float = 0.70
    # The convergence gate is 40 m/s^2 while block 2 remained at 231 m/s^2.
    # These are still bounded by terrain_acceleration_max_ratio, but strong
    # enough to compete with task reward through the observed range.
    sole_acceleration_weight: float = -0.50
    sole_rate_penalty_weight: float = -0.30
    # A non-zero constant is a cylinder-velocity command, not a pose offset.
    # The earlier (-0.25, -0.25, 0, 0) seed improved a 20-step action screen but
    # kept driving the arm for the full 721-step episode: the first two measured
    # returns regressed from -1895 to -1996. Zero is the verified fixed point of
    # the plant and lets exploration discover state-dependent corrections.
    terrain_initial_action_bias: tuple = (0.0, 0.0, 0.0, 0.0)

    # --- Truck gates ---
    # The inherited 1 mm / 50 mm gates assume a pinned chassis and are
    # meaningless for a driving truck. Lateral drift off the mowing line is
    # still measured (see TerrainSolePolicyEnv._constrain_truck_xy); these are
    # the thresholds for that, not for forward travel.
    truck_drift_termination_m: float = 1.5
    truck_raw_reaction_termination_m: float = 2.0
    head_down_termination_alignment: float = 0.98

    # With hundreds of synchronized replicas, one finite safety excursion from
    # an exploratory policy would redeploy every arm (540 reset physics steps),
    # turning a rollout into near-continuous reset work. Safety rewards, metrics,
    # and held-out gates remain active; only non-finite state resets immediately.
    defer_safety_terminations: bool = True

    # --- Spawn ---
    # The MULAG articulation root sits ~1.11 m ABOVE the wheels, so the root z is
    # not the ride height. PPO1_ID_FD overrides init_state.pos z to 1.26 for this
    # reason.
    #
    # PPO2 uses 0.05 and still reports base_link at +0.149 -- but only because its
    # parked fixture holds the chassis there: wheel stiffness 2e9 plus a root-pose
    # rewrite every step. PPO3 removes both so the truck can drive, and without
    # that scaffolding 0.05 lets it settle to base_link = -1.197 m, i.e. sunk
    # through the road (measured 2026-07-28, stable across 50+ steps, so a resting
    # state rather than free fall).
    truck_spawn_z: float = 1.26

    # NOT replicated: teacher_student's initial arm joint angles
    # (Drehzapfen 0.1, Ausleger_I -0.02, rest 0). Two reasons. The asset's
    # init_state.joint_pos is the IsaacLab default {".*": 0.0}, and adding named
    # joints alongside that wildcard makes the match ambiguous -- IsaacLab raises
    # "Multiple matches for 'Drehzapfen_joint'". More importantly it would not
    # matter: _run_reset_preposition drives the arm to its deployment pose during
    # every reset, so the spawn pose is overwritten before the first scored step.
    # The spawn HEIGHT above is a different story -- that one decides whether the
    # truck lands on the terrain or inside it.

    def __post_init__(self):
        super().__post_init__()
        # Terrain patches are large; a wide env grid needs terrain to match.
        # See TerrainSolePolicyEnv._setup_scene, which checks coverage.
        self.scene.num_envs = 16
        # The 26 m verge meshes must never overlap adjacent X columns: overlap
        # would let one environment's cutter contact another environment's
        # independently generated height profile.
        self.scene.env_spacing = 28.0
        from PPO3_SOLE_PhysX.terrain_observations import verge_observation_dim
        self.observation_space = verge_observation_dim() + 1

        pos = self.robot_cfg.init_state.pos
        self.robot_cfg.init_state.pos = (pos[0], pos[1], float(self.truck_spawn_z))
        # Contact should settle instead of ejecting the cutter at the asset's
        # inherited 5 m/s depenetration cap. This remains a real PhysX contact;
        # it only limits the solver's separation velocity after overlap.
        self.robot_cfg.spawn.rigid_props.max_depenetration_velocity = 0.20




@configclass
class TerrainSolePolicyEnvCfg_PLAY(TerrainSolePolicyEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.randomize_start_phase = False
        self.enable_visual_grid_floor = True


@configclass
class TerrainSolePolicyPPORunnerCfg(SolePolicyPPORunnerCfg):
    experiment_name = "ppo3_sole_terrain"
    # 512 envs x 32 steps = 16,384 transitions/update. The inherited 128-step
    # horizon makes one update exceed two minutes on the 6 GB laptop RTX 3060
    # without adding useful batch diversity.
    num_steps_per_env = 32

    def __post_init__(self):
        super().__post_init__()
        self.actor.distribution_cfg.init_std = 0.10
        self.algorithm.entropy_coef = 0.001
