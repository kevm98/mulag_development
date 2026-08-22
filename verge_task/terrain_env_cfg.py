"""Configuration for PPO3 on terrain with a driving truck."""

from __future__ import annotations

from pathlib import Path

from isaaclab.utils import configclass

from verge_task.env_cfg import (
    SolePolicyEnvCfg,
    SolePolicyPPORunnerCfg,
    apply_sole_curriculum_stage,
)
from mulag_plant.runtime.sim_memory import apply_kevin_sim_memory_optimizations


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


def configure_floating_cutter(cfg) -> None:
    """Select and configure the passive floating-cutter articulation asset."""

    if not bool(getattr(cfg, "terrain_floating_cutter", False)):
        return
    from isaaclab.actuators import ImplicitActuatorCfg

    base_usd = Path(str(cfg.robot_cfg.spawn.usd_path))
    floating_usd = base_usd.with_name("unimog_mulag_fme500_float_cutter.usda")
    if not floating_usd.is_file():
        raise FileNotFoundError(f"floating-cutter asset is missing: {floating_usd}")
    stiffness = float(cfg.terrain_floating_cutter_stiffness_npm)
    damping = float(cfg.terrain_floating_cutter_damping_nspm)
    effort = float(cfg.terrain_floating_cutter_effort_limit_n)
    if stiffness <= 0.0 or damping < 0.0 or effort <= 0.0:
        raise ValueError(
            "floating-cutter stiffness/effort must be positive and damping "
            f"non-negative; got {stiffness}/{damping}/{effort}"
        )
    cfg.robot_cfg.spawn.usd_path = str(floating_usd)
    cfg.robot_cfg.actuators["cutter_float_passive"] = ImplicitActuatorCfg(
        joint_names_expr=["CutterFloat_joint"],
        stiffness=stiffness,
        damping=damping,
        effort_limit_sim=effort,
        velocity_limit_sim=1.0,
    )
    # All task-space/contact measurements now follow the physical collider body
    # downstream of the suspension. Its orientation remains that of Messerkopf
    # because the added degree of freedom is prismatic.
    cfg.task_space_ee_body_name = "CutterFloat"
    cfg.cutter_body_name = "CutterFloat"
    cfg.cutter_prim_subpath = "Rigid_Bodies/Arm_Links/CutterFloat"


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
    elif stage == 5:
        # Rough-lawn verge. Identical precision, horizon and action contract to
        # stage 4 -- the ONLY difference is the ground. That keeps stage 4 usable
        # as a fixed regression set: a stage-5 score is not comparable with a
        # stage-4 one, but a stage-4 re-run of a stage-5 checkpoint is.
        cfg.cut_tracking_std_m = 0.03
        cfg.lateral_tracking_std_m = 0.30
        cfg.episode_length_s = 16.0
        cfg.sole_action_filter = float(
            getattr(cfg, "terrain_action_filter", 0.70)
        )
        cfg.terrain_relief_profile = "lawn_rough"
        # 0.25 m sampling has a 0.5 m Nyquist wavelength and cannot carry the
        # 0.6 m fine band at all. 0.15 m resolves both new components; the verge
        # is the only mesh that changes, so the cost is bounded.
        cfg.mow_strip_resolution_m = 0.15
        cfg.uneven_verge_apron = True
    elif stage == 6:
        # FLAT GROUND WITH RANDOM RAISED CHUNKS, and a much wider tracking path.
        #
        # Stage 5 asks the arm to follow a surface that undulates everywhere,
        # which makes cut height the dominant difficulty and leaves the lateral
        # path exercising barely a third of the verge. Stage 6 inverts that: the
        # ground is flat at road level for long runs with discrete raised
        # sections between, and the tracking line swings across most of the
        # available half-strip. Tracking is the objective; the chunks are the
        # disturbance.
        #
        # The patch layout is drawn PER ENVIRONMENT from its own terrain seed
        # (see terrain.generate_flat_patches), so a policy trained here sees a
        # different number of chunks in different places at different heights in
        # every parallel environment and on every resumed block. That is the
        # robustness this stage exists for -- it cannot be obtained by
        # memorising one mesh.
        cfg.cut_tracking_std_m = 0.03
        cfg.lateral_tracking_std_m = 0.30
        # Episode duration is COUPLED to truck speed and therefore belongs with
        # the task's tracking band, not here. The arm has to traverse the whole
        # authored path: 16 m at 4 km/h is 14.4 s, but at 2 km/h it is 28.8 s,
        # and a task that halves the speed while keeping a 16 s episode simply
        # never shows the policy the far half of the path. Shortening the PATH
        # to fit instead is the trap -- substituting active_length = v * T into
        # the demand gives 2*pi*A*f/T, with the speed cancelled entirely, so a
        # shorter path at lower speed has exactly the slope it started with.
        if not bool(getattr(cfg, "terrain_tracking_band_owned_by_task", False)):
            cfg.episode_length_s = 16.0
        cfg.sole_action_filter = float(
            getattr(cfg, "terrain_action_filter", 0.70)
        )
        cfg.terrain_relief_profile = "flat_patches"
        cfg.mow_strip_resolution_m = 0.15
        cfg.uneven_verge_apron = True
        # The whole point of this stage. The families author 0.10-0.24 m of
        # lateral swing; 4.4x takes the mean peak-to-peak from ~0.20 m to
        # ~0.88 m against the widened 1.34 m half-strip, so the arm works
        # across most of the verge instead of a third of it. Still bounded by
        # the same smooth tanh saturation, so it cannot leave the green.
        #
        # The frequency stays at 1.0, and that is deliberate after measuring the
        # alternative. Widening costs lateral RATE (peak lateral speed goes as
        # amplitude x frequency), so halving the frequency to pay for the wider
        # amplitude looks free -- and is not. Families are authored to complete
        # their cycle over q in [0, 1] and end at zero: `offset_return` is
        # `sin(pi*f*q)**2`, which at f=0.5 ends at its PEAK instead of zero, and
        # the endpoint window then slams it shut over the last 10% of the path.
        # Measured with `probe_path_feasibility.py`: that took offset_return's
        # peak lateral demand from 0.083 m/s to 0.948 m/s -- an 11x INCREASE
        # from a change meant to reduce it. Sub-unit frequency scaling is only
        # safe for families whose shape is periodic in q, which is not all of
        # them.
        #
        # The width was bought out of the `lateral` gate's margin on the
        # prediction that "roughly doubling the demand still lands near 0.18 m".
        # THE 8-GATE RUN THAT WAS SUPPOSED TO CONFIRM THAT FALSIFIED IT, and the
        # amplitude is corrected here as a result. Measured 2026-08-11:
        #
        # `probe_path_feasibility.py` (pure geometry, no simulator, 3 meshes x
        # 11 families = 33 cells) counts the cells whose required lateral cutter
        # speed exceeds the cylinder rate ceiling at the scripted 4 km/h drive:
        #
        #     amplitude   6.6   5.5   4.4   3.3   2.6   2.2   1.8   1.6   1.4
        #     cells over   26    24    24    20    15     7     3     0     0
        #     worst step % 55.9  43.6  33.9  22.3  17.9  14.5   7.5   0     0
        #
        # EVERY RUNG of the shipped 2.2-6.6 ladder was over the ceiling, the top
        # one on 56% of steps. Tracking error at those amplitudes is accumulated
        # by construction: the reference moves laterally faster than the
        # cylinders can follow, so no policy -- and not the privileged DLS
        # teacher -- can remove it. Confirmed on the simulator: scored on stage 6
        # at 32 envs, the teacher FAILS 5 of 8 gates (worst-family lateral
        # 0.3027 m, worst-family ground contact 0.4021, min cylinder headroom
        # exactly 0.0000, i.e. the arm sitting on its own travel limits), while
        # the same teacher on stage 5 -- which runs at the class-default
        # amplitude 1.0 -- passes all eight. The stage-5/stage-6 gap that was
        # read as "stage 6 is harder terrain" was almost entirely this ladder.
        #
        # The ceiling is OPTIMISTIC (it assumes a lateral Jacobian of 1.0 and the
        # whole cylinder budget spent on lateral motion rather than on holding
        # height and head angle), so the honest limit is at or below the 1.4-1.8
        # crossing, not above it.
        #
        # THE FEASIBLE BAND WAS ALREADY DERIVED, AND THIS FUNCTION WAS ERASING
        # IT. `Ppo31SoleEnvCfg.__post_init__` sets amplitude 0.7 with the ladder
        # (0.30, 0.45, 0.60, 0.75) and sweep gain 0.5, chosen against a
        # three-knob sweep so every family sits at or under 75% of the ceiling.
        # That runs at config construction; this function runs AFTER it, in
        # train.py, distill_terrain.py and evaluate_gates.py alike, and the two
        # assignments below put the infeasible ladder back every time. The
        # precedence was an accident of call order, not a decision -- so the
        # values here are now the same measured band, and a task that pins its
        # own is left alone (see `terrain_tracking_band_owned_by_task`).
        if not bool(getattr(cfg, "terrain_tracking_band_owned_by_task", False)):
            cfg.terrain_tracking_amplitude_scale = 0.7
        # AND IT IS NOT PINNED THERE. A policy generalises over the distribution
        # it was trained on and essentially nothing else -- this repo measured a
        # student distilled at a PINNED 2.2 delivering 0.303 m of lateral error
        # on amplitude-0.7 terrain that physically permits 0.058 m, five times
        # worse than the plant allows, purely from being asked a question it was
        # never trained on (see `terrain_parameter_for_environment`). Shipping a
        # single wider amplitude would repeat that mistake at a new number.
        #
        # Each environment therefore draws its own width from this ladder, which
        # spans narrower-than-old (2.2) to wider-than-new (6.6) so the trained
        # band brackets deployment instead of sitting at its edge. FIVE values
        # against the FOUR `terrain_height_scales` is deliberate: the lengths are
        # coprime, so all 20 (amplitude, height) pairs appear every 20
        # environments at stride 1, decorrelating width from roughness.
        #
        # Amplitude is the safe axis to randomise because it is applied AFTER
        # the endpoint window, so every family still starts and ends at zero
        # offset no matter how it is scaled. Frequency is NOT safe to jitter the
        # same way: it moves cycle counts off multiples of 0.5, the family stops
        # landing on zero at q=1, and the endpoint window closes the gap
        # discontinuously -- measured at 11x the lateral rate demand. Leave
        # `terrain_tracking_frequency_jitter` at 0.0 unless every family is
        # re-authored to be periodic under scaling.
        #
        # Same band as the scalar above, and same ownership rule. Four rungs
        # against the three `terrain_tracking_height_sweep_gains` are coprime, so
        # all 12 (amplitude, gain) pairs appear every 12 environments at stride 1.
        if not bool(getattr(cfg, "terrain_tracking_band_owned_by_task", False)):
            cfg.terrain_tracking_amplitude_scales = (0.30, 0.45, 0.60, 0.75)


@configclass
class TerrainSolePolicyEnvCfg(SolePolicyEnvCfg):
    """Sole-controller arm, constant-speed truck, generated mowing terrain."""

    # --- deployment-domain factor contract -------------------------------
    # Disabled by default so historical nominal checkpoints remain bitwise
    # comparable. Phase-1 evaluation enables this explicitly and latches one
    # audited e=(flow, compliance, seal, pressure, mass_scale, com_z) row per
    # environment for the entire episode.
    deployment_factor_enabled: bool = False
    deployment_factor_domain: str = "nominal"
    deployment_factor_key: str = ""
    deployment_factor_seed: int = 17
    deployment_factor_manifest: str = str(
        Path(__file__).resolve().parents[1]
        / "AAM_module/datasets/rom_deployment_stage31_v1_frozen/raw/manifest.csv"
    )
    deployment_factor_log: bool = True

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
    # Four hydraulic cylinders are the physical action interface. The former
    # fifth output was only mixed back into c4 and is retained as an explicit
    # compatibility mode for existing checkpoints.
    sole_head_joint_action: bool = False
    terrain_head_reference_per_substep: bool = True
    """Assemble the head joint reference in _apply_action, every substep.

    The head hangs off a boom that moves every substep. Writing its reference
    once per control step (decimation=2) tracks that boom half as tightly and
    measured min_head_alignment 0.9296 -> 0.8662. The policy still integrates
    its correction once per control step, as its action is defined; only the
    reference assembly moves.
    """
    sole_action_authority: tuple = (1.0, 1.0, 1.0, 1.0)
    trajectory_library_path: str = str(
        Path(__file__).resolve().parents[1]
        / "mulag_plant/models/plant_reachable_trajectory_library_v3_large.json"
    )
    # Make the task itself select canonical ROM-A. Entry-point flags may still
    # override this, but omitting a flag must never silently select command_run_001.
    command_rom_checkpoint: str = str(
        Path(__file__).resolve().parents[1]
        / "rom_model/plant_reachable_v1_finetune_blend25/best_rom.pt"
    )

    # --- Vehicle ---
    # Scripted, not policy-controlled. teacher_student samples 3-8 km/h per
    # episode; a single constant speed is the simpler starting point and is what
    # "the truck moving at a constant speed" asks for. Set min < max to sample.
    enable_scripted_vehicle_motion: bool = True
    vehicle_speed_min_kmh: float = 4.0
    vehicle_speed_max_kmh: float = 4.0
    vehicle_speed_per_episode: bool = True
    """Sample one constant scripted speed at reset; never resample mid-episode."""
    vehicle_speed_sampling: str = "uniform"
    """Reset distribution: uniform, or active_time_balanced for PPO training."""
    vehicle_training_segment_duration_s: float = 0.0
    """Equal active duration for speed-randomized training; zero scores all 26 m."""
    vehicle_path_completion_margin_s: float = 0.5
    """Extra global horizon beyond the slowest speed's exact path duration."""
    vehicle_path_completion_tolerance_m: float = 1.0e-4
    vehicle_terminate_when_all_paths_complete: bool = True
    vehicle_steering_target_rad: float = 0.0
    drive_direction_xy: tuple = (1.0, 0.0)
    """Unit heading the chassis is advanced along, in env-local XY."""
    # Held at -0.015: the authored tyre bottoms sit within 0.3 mm of road z=0,
    # i.e. the truck actually rests on the road. Preloading to -0.020 to buy the
    # wheel_road_support gate was measured and is wrong twice over -- wheel duty
    # cycle FELL 0.1973 -> 0.1446, and p95_cut_height_error read exactly 0.0000
    # because the sunk chassis presses the cutter into the terrain and saturates
    # the error. A gate that reads a perfect zero is measuring nothing.
    truck_ride_height_target_m: float = -0.015
    """Base-link height that places the authored tyre bottoms on the road.

    This is an authored frame coordinate, not physical clearance. At -0.015 m
    all four 0.575 m-radius tyre bottoms measure within 0.3 mm of road z=0;
    filtered wheel-road forces verify support instead of inferring it from the
    misleading base-link origin.
    """

    truck_ride_height_gain: float = 1.0
    """Fraction of the ride-height error corrected per fixture call.

    1.0 is the hard pin every existing score was measured with. Values in (0,1)
    make it a first-order lag, so the chassis can settle onto its tyres instead
    of being held at a pose where one of them hovers -- the teacher_student repo
    drives a FREE chassis on velocity-driven wheels and lets gravity do this,
    which is the arrangement `min_wheel_road_contact_duty_cycle` was written to
    measure. Freeing z completely recovers that gate (0.5178) but costs
    smoothness (accel 138.88), so the vertical hold itself is load-bearing.
    """

    road_start_x_m: float = -40.0
    road_length_m: float = 40.0
    road_center_y_m: float = 4.30
    road_width_m: float = 3.60
    road_color: tuple = (0.11, 0.12, 0.13)
    road_roughness: float = 0.92
    enable_road_edge_markings: bool = False
    road_edge_marking_inset_m: float = 0.45
    road_edge_marking_width_m: float = 0.11
    road_edge_marking_color: tuple = (0.82, 0.82, 0.76)
    """Dark, rough asphalt-like road, physically separate from the green verge."""
    dark_visual_background: bool = False
    """Opt into a black renderer background; the default visible dome is white."""
    enable_visual_grid_floor: bool = False
    visual_floor_color: tuple = (0.085, 0.22, 0.045)
    """Dark grass base below the wider visual banks."""
    visual_grid_color: tuple = (0.13, 0.285, 0.065)
    visual_grid_length_m: float = 70.0
    visual_grid_width_m: float = 26.0
    visual_grid_spacing_m: float = 2.0
    visual_grid_line_width_m: float = 0.025
    visual_grid_center_x_m: float = -20.0
    visual_grid_center_y_m: float = 3.0

    # --- Visual-only road dressing (TerrainSolePolicyEnv._spawn_road_scenery) ---
    # Presentation geometry for interactive viewing. All of it has collision
    # disabled and none of it sits between the verge and the wheels, so the
    # colliding road, the wheel-road contact sensors and the height scan are
    # untouched -- physics and every gate metric are identical with this on or
    # off. Default False so training and headless evaluation pay nothing;
    # evaluate.py and probe_drive.py turn it on for interactive runs only.
    enable_road_scenery: bool = False
    road_visual_lanes: int = 2
    """Lanes drawn. Extra lanes extend away from the verge, never toward it."""
    road_marking_color: tuple = (0.90, 0.90, 0.88)
    road_edge_line_width_m: float = 0.14
    road_lane_dash_width_m: float = 0.12
    road_lane_dash_length_m: float = 2.0
    road_lane_dash_gap_m: float = 3.0
    road_scenery_edge_inset_m: float = 0.28
    road_gravel_shoulder_width_m: float = 0.34
    road_gravel_shoulder_color: tuple = (0.10, 0.085, 0.060)
    verge_visual_apron_m: float = 6.0
    """Grass width added outboard of the mowed verge and beyond the far kerb."""
    verge_apron_overlap_m: float = 0.45
    """Display-only overlap beneath adjacent meshes, preventing visible cracks."""
    verge_visual_color: tuple = (0.052, 0.18, 0.025)
    verge_visual_palette: tuple = (
        (0.022, 0.095, 0.010),
        (0.035, 0.135, 0.015),
        (0.052, 0.180, 0.025),
        (0.072, 0.145, 0.030),
        (0.125, 0.068, 0.024),
    )
    """Muted grass, dry growth and sparse soil shared with the mow-strip skin."""
    uneven_verge_apron: bool = True
    """Displace the roadside grass into lawn relief instead of flat slabs."""
    verge_apron_relief_m: float = 0.09
    verge_apron_resolution_m: float = 0.45
    """Apron mesh step. Visual-only, so this stays coarse relative to the verge."""

    enable_verge_grass_tufts: bool = True
    verge_grass_tufts_per_env: int = 150
    verge_grass_tracking_clearance_m: float = 0.24
    verge_grass_edge_clearance_m: float = 0.18
    verge_grass_height_range_m: tuple = (0.16, 0.52)
    verge_grass_width_range_m: tuple = (0.09, 0.26)
    verge_grass_palette: tuple = (
        (0.025, 0.10, 0.012),
        (0.042, 0.145, 0.018),
        (0.070, 0.19, 0.028),
        (0.10, 0.16, 0.035),
    )
    verge_bank_grass_tufts_per_side: int = 240
    verge_seam_grass_tufts_per_side: int = 90
    verge_bushes_per_side: int = 8
    verge_bush_height_range_m: tuple = (0.38, 0.82)
    verge_bush_width_range_m: tuple = (0.38, 0.90)
    verge_bush_palette: tuple = (
        (0.018, 0.075, 0.008),
        (0.032, 0.11, 0.014),
        (0.052, 0.145, 0.020),
    )
    verge_twigs_per_side: int = 28
    verge_logs_per_side: int = 4
    verge_strip_twigs_per_env: int = 16
    verge_strip_logs_per_env: int = 2
    verge_twig_length_range_m: tuple = (0.22, 0.65)
    verge_log_length_range_m: tuple = (0.70, 1.60)
    verge_wood_palette: tuple = (
        (0.11, 0.055, 0.018),
        (0.16, 0.080, 0.025),
        (0.21, 0.115, 0.040),
    )
    """Clustered grass, sparse shrubs and wood debris; visual-only/non-colliding."""

    # Relief class of the mowed verge. "gentle" is the stage 1-4 surface every
    # existing score was measured on; "lawn_rough" is stage 5. Set by
    # apply_terrain_curriculum_stage, not by hand.
    terrain_relief_profile: str = "gentle"
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
    # Per-cylinder anti-windup lead caps, overriding the scalar
    # `cylinder_target_max_lead_m` when set. None keeps the scalar behaviour
    # bit-for-bit, so every stored report stands.
    #
    # Measured 2026-08-04 on the PPO3 16/17 checkpoint, EVALUATION ONLY (no
    # retraining): raising the scalar 0.018 -> 0.050 halved lateral error on the
    # sustained-offset families (lane_change_hold 0.4833 -> 0.2310, boom_sweep
    # 0.3745 -> 0.1602), confirming the clamp was bounding the standing effort
    # those paths need -- but the same relaxation on c4 (head pitch) drove head
    # alignment 0.9507 -> 0.4636. Hence per axis.
    cylinder_target_max_lead_per_axis_m: tuple | None = None
    terrain_static_friction: float = 1.0
    terrain_dynamic_friction: float = 1.0
    terrain_restitution: float = 0.0

    # --- Verge strip ---
    # Exactly `verge_visual_color`, so the tracked strip is not a visibly
    # different green slab laid on the lawn -- the mowed band and the grass
    # around it read as one continuous surface. Kept as its own field because
    # the strip is a real collider and the apron is visual-only; they are
    # separate meshes that happen to share a colour.
    mow_strip_color: tuple = (0.052, 0.18, 0.025)
    mow_strip_use_vertex_palette: bool = False
    """Enable the zero-geometry-cost natural grass vertex palette for viewing."""
    mow_strip_visual_palette: tuple = (
        (0.022, 0.095, 0.010),
        (0.035, 0.135, 0.015),
        (0.052, 0.180, 0.025),
        (0.072, 0.145, 0.030),
        (0.125, 0.068, 0.024),
    )
    """The exact apron palette: muted grass, dry growth and sparse soil."""
    mow_strip_soil_field_threshold: float = 0.95
    """Produces about five percent sparse soil accents on the reference mesh."""
    mow_strip_visual_shoulder_transition_m: float = 0.30
    """Visual-only easing into the clean road edge; collision stays unchanged."""
    mow_strip_visual_skirt_depth_m: float = 0.35
    """Deep display-only sidewall that hides open height-field edges."""
    mow_strip_collision: bool = True
    """The green verge is a real collider; contact cannot be faked geometrically."""
    terrain_tracking_height_sweep_gain: float = 1.5
    """How strongly ``boom_sweep`` couples its lateral offset to terrain height.

    ``boom_sweep`` is the ONLY family whose offset is a function of the ground:
    ``raw_offset = gain * (height - reference_height)``. Every other family
    draws a fixed random amplitude. That coupling is why it alone gets harder
    as the terrain gets rougher, and why it demanded 2.9-4.6x the cylinder rate
    ceiling at the shipped settings while the privileged DLS teacher failed it
    too. Lowering this brings one infeasible family inside the plant's
    capability without making the other ten easier.
    """
    terrain_tracking_amplitude_scale: float = 1.0
    """Widen every tracking path family's lateral excursion by this factor.

    The families author 0.10-0.24 m of lateral swing against 0.74 m of
    available half-strip, so the arm was being exercised over a third of the
    room the verge actually offers. Raising this gives the end effector a wider
    working band. Bounded by the same smooth tanh saturation, so it can never
    push the path off the green; widen `mow_strip_width_m` alongside it if the
    scaled amplitude starts to compress against that bound.
    """
    terrain_tracking_frequency_scale: float = 1.0
    """Frequency multiplier for cyclic paths. 1.0 preserves stored reports."""
    terrain_tracking_frequency_jitter: float = 0.0
    """Seeded per-strip frequency variation. 0.0 preserves stored reports."""
    terrain_composite_trajectory: bool = False
    """Opt-in multi-manoeuvre path; disabled for checkpoint reproducibility."""
    terrain_composite_families: tuple = (
        "straight",
        "s_curve",
        "gaussian_slalom",
        "lane_change_hold",
    )
    terrain_tracking_reference_x_m: float = -25.4
    """Environment-local x coordinate where the active tracking path begins."""
    terrain_tracking_active_length_m: float = 16.0
    """Along-track distance occupied by the path or composed segments."""
    mow_strip_start_x_m: float = -32.0
    """Environment-local x coordinate where the colliding green strip begins."""
    mow_strip_width_m: float = 3.0
    """Width of the green verge strip.

    Widened from 1.8 m alongside the 4.4x path amplitude. The path is bounded
    by `half_available = 0.5 * width - 0.16`, so 1.8 m allowed only 0.74 m of
    excursion either side and the scaled path would have spent most of its
    travel compressed against that tanh saturation -- which flattens the very
    peaks the widening is meant to produce. 3.0 m gives 1.34 m of half-strip,
    enough that the ~0.88 m peak-to-peak path stays in the linear region with
    green either side of the cutter.
    """
    mow_strip_resolution_m: float = 0.25
    """Collision/ray mesh spacing.

    Matches the height-scan spacing and still gives ten samples across the
    shortest 2.5 m terrain wavelength. At 512 environments this reduces the
    merged mesh from 24.6 million to about 3.9 million triangles.
    """
    mow_strip_length_m: float = 26.0
    """Per-environment strip length.

    The ROM/ID-FD wide task overrides the path reference so this entire 26 m
    strip is active rather than retaining the legacy lead-in and exit margins.
    """
    # Keep the projected reference visually part of the verge instead of a red
    # debug stripe.  A slightly lighter green remains readable against the lawn
    # while matching the surrounding scene palette.
    tracking_line_color: tuple = (0.92, 0.92, 0.88)
    tracking_line_width_m: float = 0.07
    tracking_line_z_offset_m: float = 0.012
    """Visual-only line projected onto the exact terrain centre path."""
    tracking_path_family: str = "mixed"
    """Cycle every path family across parallel environments."""
    tracking_path_family_offset: int = 0
    # Scheduled step in the COMMANDED tracking line, in metres of lateral
    # offset, applied after `tracking_reference_step_time_s` seconds of scored
    # travel. This changes what the controller is ASKED to do rather than the
    # plant it acts on, so the latency it exposes belongs to the control stack.
    # Zero leaves the commanded line untouched.
    # Magnitude of the deployment-factor randomisation, as a multiple of the
    # manifest draw. 0 is the nominal plant, 1 reproduces the manifest, 2
    # doubles each deviation. Only the magnitude changes; the direction of each
    # draw is fixed, so a sweep over this is not confounded with visiting
    # different plants.
    deployment_factor_scale: float = 1.0
    tracking_reference_step_m: float = 0.0
    tracking_reference_step_time_s: float = 0.0
    tracking_path_family_schedule: tuple = ()
    """Optional training-only weighted family schedule; empty stays balanced."""
    tracking_path_start_phase_schedule: tuple = ()
    """Optional phase-at-spawn schedule aligned with the family schedule.

    Negative entries preserve the authored straight lead-in.  Entries in
    ``[0, 1]`` shift that environment's path reference so the deployed cutter
    begins at the requested normalized path phase.  This changes only the
    training geometry; held-out evaluation leaves the schedule empty.
    """
    tracking_path_phase_anchor_x_m: float = -27.12
    """Measured deployed cutter x used to place a requested start phase."""
    # The GATE terrain. Held-out geometry is pinned at seed 1337 precisely so a
    # stage-5 score means the same thing across runs, and this tuple is read for
    # every environment including evaluation -- widening it here silently moved
    # the gate set (measured: the 15/15 checkpoint re-scored 12/15 / 0.0897 when
    # 0.35 and 1.15 were added). Training gets the wider distribution from
    # scripts/train.py instead, which is the only place that may change it.
    terrain_height_scales: tuple = (0.50, 0.70, 0.85, 1.00)
    terrain_tracking_amplitude_scales: tuple = ()
    """Per-environment ladder of path widths; empty falls back to the scalar.

    Declared here so the field is part of the contract rather than materialised
    by assignment in a stage function. Stage 6 fills it; see the comment there
    for why the length is coprime with `terrain_height_scales`.
    """
    terrain_tracking_band_owned_by_task: bool = False
    """The TASK config owns the tracking band; the stage function must not touch it.

    `apply_terrain_curriculum_stage` runs AFTER a task cfg's ``__post_init__``,
    in every entry point, so a band pinned by the task was silently overwritten
    by the stage's own values. That is how the feasibility-derived amplitude
    0.7 / (0.30, 0.45, 0.60, 0.75) in `Ppo31SoleEnvCfg` came to be replaced by
    the infeasible (2.2, 3.3, 4.4, 5.5, 6.6) on every single run -- a precedence
    accident, never a decision. Setting this True makes the task's ownership
    explicit and greppable instead of depending on who assigns last.
    """
    """Balanced relief levels crossed independently with every path family."""
    terrain_height_scale_override: float = 0.0
    """Positive visualization/probe override; zero uses the balanced curriculum."""
    terrain_eval_seed_override: int = 0
    """Move the HELD-OUT mesh off its pinned seed 1337. Zero keeps the gate mesh.

    The held-out geometry is pinned so a stage-5 score means the same thing
    across runs, which is correct -- but it also means `--seed` cannot vary a
    held-out evaluation at all. With the mesh pinned, `randomize_start_phase`
    False and the actor evaluated at its mean, the rollout is deterministic:
    the pre-fix diagnostic measured seeds 17, 909, 1337 and 4242 returning
    bit-identical event counts, gaps and forces across all four runs
    (reports/contact_safe_v1/PREFIX_SUMMARY.md). Robustness across meshes has
    to be requested explicitly, and this is how. Never set it on a gate run."""

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
    terrain_parallel_sensor_observation: bool = True
    """Append twin cutter-ray readings and direct-q5 state when q5 is enabled."""
    terrain_parallel_sensor_left_offset: tuple = (0.4, -0.5, 0.5)
    terrain_parallel_sensor_right_offset: tuple = (0.4, 0.6, 0.5)
    terrain_parallel_reward_weight: float = 1.0
    terrain_parallel_reward_std_m: float = 0.02
    """Teacher-student terrain-parallel reward: 1-tanh(|left-right|/std)."""
    terrain_floating_cutter: bool = False
    """Use the opt-in passive prismatic cutter suspension.

    This changes the plant and is therefore never enabled implicitly for an
    existing checkpoint.  The policy still owns exactly the same five commands;
    ``CutterFloat_joint`` is a fixed passive spring-damper with no state-based
    target writer.
    """
    # Measured operating point, not a guess. Zero-action 2x240 screens on the
    # held-out terrain (experiments/floating_cutter_flight/) swept stiffness,
    # damping and the drive cap:
    #
    #   plant                     lift eject takeoff  hard  peak kN  gap m
    #   rigid (no suspension)        2     0       5    14    123.0  0.049
    #   k500 c40  f80                0     0       1    18     45.9  0.000
    #   k800 c40  f80                0     0       1    10     67.2  0.000
    #   k800 c40  f15                0     0       0    13     70.5  0.000
    #   k800 c40  f20                0     0       0     4     34.5  0.000
    #   k800 c40  f25                0     0       0     9     49.3  0.000
    #
    # k800/c40/f20 dominates every other point tried and turns 7 of the 9
    # contact-safety gates green under ZERO action; only `hard_impacts` (4) and
    # `contact_force_peak` (34.5 kN of 30 kN) still fail, and both are impact
    # impulse rather than drive force, so they are the policy's to remove.
    # Dropping to f15 makes the cutter reach its lower stop and the peak gets
    # WORSE (70.5 kN) -- the cap is non-monotonic, do not sweep it blindly.
    # Deflection stayed within -34.6/+1.3 mm with both travel-stop fractions at
    # 0.0, so the suspension is operating inside its stroke, not on its limits.
    terrain_floating_cutter_stiffness_npm: float = 800_000.0
    terrain_floating_cutter_damping_nspm: float = 40_000.0
    terrain_floating_cutter_effort_limit_n: float = 20_000.0
    terrain_head_yaw_target_rad: float = 0.0
    """Current cylinder asset's calibrated neutral cutter yaw.

    The teacher-student USD uses a body frame rotated by pi and commands world
    yaw pi. Copying that number here left the current asset at a 3.12 rad error
    even with q5=0. Direct measurement shows this asset's neutral yaw is zero.
    """
    terrain_head_yaw_reward_weight: float = 0.5
    """Weight of the bounded linear wrapped-yaw reward."""
    terrain_head_action_penalty_weight: float = -0.01
    terrain_head_action_rate_weight: float = -0.02
    cutter_contact_force_threshold_n: float = 10.0
    cutter_contact_force_observation_scale_n: float = 1000.0

    # Non-PPO contact-tracking controller used only to validate and visualize
    # the physical task before training is allowed to resume.
    enable_contact_tracking_demo: bool = False
    contact_demo_arm_stiffness_scale: tuple = (1.0, 1.0, 1.0, 1.0, 1.0)
    """Per-joint compliance for the non-PPO physical contact demonstration.

    Keep the controlled arm geometry stiff enough to remain on the green line;
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
    terrain_teacher_head_gain: float = 4.0
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
    # Raised 5.0 -> 8.0 with lateral 2.0 -> 3.0 to prioritize tracking. The
    # plant fixes left margin to spend: held-out p95 cut is 0.0498 of a 0.080
    # gate and p95 acceleration 111.6 of 120, so tracking can be weighted harder
    # without pushing acceleration back through its gate. Safety terms (head
    # barrier, acceleration penalty, limit shield) are deliberately unchanged --
    # they are what makes the head hold.
    cut_tracking_weight: float = 8.0
    """Commanded cutter height above the sensed terrain surface."""
    lateral_tracking_std_m: float = 0.30
    lateral_tracking_weight: float = 3.0
    terrain_head_penalty_weight: float = -20.0
    terrain_head_threshold: float = 0.995
    # DO NOT soften this pairing again without re-reading this note.
    #
    # The clamped quadratic saturates at alignment 0.955, just above the 0.95
    # latch, which looks like a defect: the latch-adjacent band is a -80 step
    # with a 160x slope drop beyond it. Widening the band to spread that step
    # out (scale 0.05, far_weight -2.0) was tried on 2026-07-30 and measured
    # STRICTLY WORSE on every axis, same resumed checkpoint, same phase-locked
    # windows (lawn_rough_stage5_v4 block 1 vs v3 blocks 2-4):
    #
    #     barrier          latch mean   latch max   value loss max
    #     0.02 / -0.5        0.046-0.066  0.24-0.31   108k-153k
    #     0.05 / -2.0        0.109        0.570       231k
    #
    # The reason is that widening the band weakens the near field, which is
    # where the restoring work actually happens: the cost at alignment 0.975
    # falls -20.0 -> -3.2 and at the 0.95 latch boundary -80.1 -> -16.2, a ~5x
    # loss of gradient exactly where the head must be held. Far-field
    # smoothness is not what keeps the head up; near-field steepness is. The
    # only block ever promoted on this terrain (v3 block 1: latch mean 0.0026,
    # value loss max 33k) ran this original pairing.
    terrain_head_penalty_scale: float = 0.02
    terrain_head_penalty_max_deviation: float = 2.0
    # Answers an inversion cascade seen in expanded_paths_height_v2_very_long
    # (one env at -0.643 spreading past the 1st percentile within two blocks):
    # restores a finite gradient out of the saturated plateau so a flipped head
    # no longer costs exactly what a marginal one does. At alignment=-1 it adds
    # -48.9. Deliberately shallow -- see the note above for why steepening it
    # to -2.0 backfired.
    terrain_head_penalty_far_weight: float = -8.0
    # Worst-case (CVaR) weighting for the two gates that read a MINIMUM over
    # path families rather than a mean. Both default off so the shipped reward
    # is bit-identical unless explicitly enabled. See the derivation at the
    # task_reward assembly in terrain_env.py.
    terrain_contact_tail_fraction: float = 0.0
    terrain_contact_tail_multiplier: float = 1.0
    terrain_lateral_tail_fraction: float = 0.0
    terrain_lateral_tail_multiplier: float = 1.0
    terrain_head_tail_fraction: float = 0.0
    """Training-only CVaR fraction; zero preserves the standard reward."""
    terrain_head_tail_multiplier: float = 1.0
    terrain_head_tail_memory_trigger: float = 0.975
    terrain_head_tail_memory_steps: int = 0
    terrain_head_tail_memory_penalty: float = 0.0
    # Head-down is a precondition for earning tracking reward. An env that drops
    # below this alignment while tracking earns no cut/lateral/contact reward for
    # the rest of the episode.
    #
    # Set to the held-out head_alignment_hard_min gate (0.950, ~18 degrees off
    # world-down) so training enforces exactly what convergence measures. Do not
    # raise this to head_down_termination_alignment (0.98) without checking the
    # latch rate first: a latch costs the whole remaining episode, not one step,
    # and the seed checkpoint already runs its 1st percentile at 0.920, so a 0.98
    # requirement would latch most episodes early and starve the tracking
    # gradient. Watch safety/head_down_failed_fraction and tighten from there.
    # Pose-dependent cutter-head levelling. See TerrainSolePolicyEnv._apply_action
    # for why a fixed joint6_coupling ratio is wrong and why removing it is worse.
    #
    # Held-out sweep, 128 envs x 960 steps, same checkpoint, NO retraining
    # (v5/model_best.pt; reports/heldout_messerkopf_level_*.json):
    #
    #   gain  clamp   min_align  p95_cut  p95_accel   score   passing
    #   0.00    --      0.9628    0.0817    130.94   0.1130     13
    #   0.50   0.15     0.9407    0.0729    128.66   0.0819     13
    #   0.50   0.08     0.9482    0.0739    129.31   0.0795     13   <- chosen
    #   0.50   0.05     0.9498    0.0772    132.13   0.1013     13
    #   0.50   0.04     0.9371    0.0757    138.50   0.1677     13
    #   0.25   0.15     0.9579    0.0990    134.50   0.3585     13
    #
    # NEITHER knob is monotone -- gain 0.25 is worse than both 0.0 and 0.5, and
    # clamp 0.04 is worse than 0.05. This is a feedback loop around a contact-
    # loaded joint; do not bisect it, sweep it. No clamp reaches the 0.950
    # hard-min alignment gate, so the choice trades that gate (0.9482, short by
    # 0.5%) for p95_cut_height_error, which levelling fixes outright
    # (0.0817 -> 0.0739 against a 0.080 gate). Equal gate count, 30% better
    # score, so terrain_convergence's own ranking selects it.
    #
    # Gain 0.0 reproduces the inherited targets bit-for-bit (verified: score
    # 0.11299147332708048 exactly), so this stays reversible in one number.
    # Disabled so direct construction matches train and evaluation.
    terrain_messerkopf_level_gain: float = 0.0
    terrain_messerkopf_level_sign: float = -1.0
    """Sign that makes the correction oppose the measured tilt, not amplify it."""
    terrain_messerkopf_level_max_rad: float = 0.05
    """Bound on the correction so a bad tilt reading cannot sweep the joint.

    Widened from 0.08 only because the quadratic term below now shapes the
    response: with a purely linear gain this bound was what stopped small-tilt
    oscillation, and 0.15 measured worse than 0.08 for exactly that reason.
    """
    terrain_messerkopf_level_quadratic_gain: float = 0.0
    """Tilt-squared term. Negligible near level, dominant in the tail.

    The hard-min alignment gate is decided by the single worst sample across
    128 x 960 -- measured 0.9482, an 18.5 deg tilt -- while p01 already passes
    at 0.9823. Only the tail needs more authority, so adding it linearly would
    disturb the swept small-tilt behaviour for no benefit.
    """
    # MEASURED WORSE - leave False. The frame reasoning is correct (a 25 deg
    # tilt at 70 deg slew reads 9.06 deg in world frame vs 25.00 deg in the
    # joint frame, so the correction fades exactly where training collapses),
    # but delivering it costs the hard-min alignment gate: 14/15 / 0.0109 with
    # min alignment 0.9396 against 0.950, twice measured. The joint cannot
    # supply the correction the solve asks for without disturbing what already
    # holds. Kept switchable because the diagnosis is worth preserving.
    terrain_constrain_truck_per_substep: bool = True
    """Re-pin the chassis every physics substep instead of every control step.

    The inherited fixture runs once per control step while decimation=2, so two
    physics substeps elapse in which arm reaction moves the truck before the
    next correction snaps it back -- a small visible sideways sway when the
    cutter strikes terrain. max_truck_post_constraint_drift_m reads 0.0000
    regardless because it samples AFTER the correction, so no gate sees it.

    TRUE removes the sway and, because chassis motion feeds straight into the
    arm, also cuts tool acceleration hard: 111.57 -> 91.71 m/s^2, an 18 %
    reduction and the smoothest measured. It costs the wheel gate:
    min_wheel_road_contact_duty_cycle 0.2107 -> 0.1947 against 0.20, i.e. 14/15.

    Measured alternatives, none of which recover both:
      depen 0.08  -> wheel 0.2289 PASS, align 0.9280 FAIL  (14/15)
      depen 0.10  -> wheel 0.2347 PASS, align 0.9396 FAIL  (14/15)
      pin XY+attitude only, restoring z -> wheel 0.5178 PASS but acceleration
        138.88 FAIL and cut 0.1111 FAIL (12/15): the VERTICAL pin is what
        supplies the smoothness, not the lateral one.

    Default False keeps the verified 15/15. Set True to trade one 2.6 %-margin
    gate for visibly smoother motion and no sway.

    NOTE: every number above was measured while each of those pins aimed at the
    END-of-step travel target, which teleported the chassis +18.52/-8.10/-8.10 mm
    per control step. See terrain_kinematic_drive_substep_interpolation."""
    terrain_arm_stiffness_scale: float = 0.35
    """Compliance in the vertical load path: scale on the cylinder drive gains.

    1.0 is the stock hydraulic lock and reproduces every stored report.

    0.35 measured on the held-out verge (128 envs x 960 steps, v5/model_best,
    reports/compliant_arm_035_stiffhead_heldout.json vs
    reports/drive_sawtooth_baseline_16of17.json):

        metric                          rigid    0.35     gate
        p95_tool_acceleration           91.71   48.69     <=120
        p95_cut_height_error_m         0.0631  0.0000     <=0.080
        mean_cutter_contact_force_n     28022   16904       --
        ground_contact_fraction        0.9350  0.9468     >=0.90
        safety_termination_rate        0.0133  0.0067     <=0.05
        min_wheel_road_duty            0.1947  0.2128     >=0.20   PASS now
        min_head_alignment             0.9500  0.9165     >=0.950  FAIL now
        gates                           16/17   16/17

    And on the bounce itself, which is what this is for (64 envs x 480 steps,
    reports/contact_loss_stiff_035.json vs contact_loss_policy_v5.json):
    ejection velocity at separation 0.92 -> 0.30 m/s, peak force before
    separation 53 -> 17 kN, ballistic hop height 43 -> 4.6 mm, at an unchanged
    separation rate (9.1 -> 9.2 per 1000 samples).

    The response is NOT monotonic and this is not a knob to bisect: 0.5 leaves a
    19.7 mm hop, 0.2 and 0.1 buy gentleness by losing contact (0.885/0.853), 0.05
    is worse than 0.2 on force, and 0.01 collapses contact to 0.347 because the
    cylinders can no longer hold the boom against gravity. 0.35 is the only value
    measured that improves contact AND smoothness together.

    Known cost: `head_alignment_hard_min` 0.9500 -> 0.9165. p01 alignment is
    essentially unchanged (0.9854 -> 0.9840), so this is the extreme tail, not a
    systematic loss of head control. Note the rigid checkpoint already measures
    0.9111 / 0.9185 / 0.9396 on other terrain seeds (FINAL_17of17_varied17,
    FINAL_varied_seed909, FINAL_varied_seed17), so this gate is marginal for v5
    on every mesh and its 0.9500 here is the top of that spread. Recovering it
    wants a fine-tune on the compliant plant, which v5 has never seen.

    This is the knob that actually inserts a spring between the tool and the
    world. `terrain_compliant_contact_stiffness` below is the cleaner physical
    model and was tried first, but PhysX does not honour it in this scene --
    sweeping it 1e6 -> 1e5 N/m, a 10x range that should have cut contact force
    roughly 10x, moved median peak force only 50.4 -> 46.9 kN.

    Applied to the cylinder joints plus the tool joint, and only during scored
    driving: `_restore_terrain_arm_deployment_damping` puts the stock drive back
    for the reset ramp, which was validated stiff.

    Damping is deliberately left at nominal rather than scaled alongside.
    Holding damping while cutting stiffness by N raises the damping ratio by
    sqrt(N), so one knob buys both the compliance and the dissipation that stops
    stored contact energy coming back as ejection velocity."""
    terrain_model_error_history: int = 1
    """Stack this many past model-error samples. 1 = current sample only.

    teacher_student puts history_length=3 on its rate-bearing observation terms
    for the same reason: a snapshot cannot express a trend. Depth 3 costs 24
    extra MLP inputs here. Its depth camera is deliberately NOT copied -- it
    will not run alongside 512 environments on a 6 GB card."""
    terrain_actuator_randomization: float = 0.0
    """Per-episode uniform scatter on cylinder stiffness/damping, e.g. 0.2 = +-20 %.

    The robustness route to model-error tolerance rather than the observability
    one: vary the plant so the policy cannot rely on an exact actuator. Costs a
    reset-time write and nothing per step."""
    terrain_model_error_observation: bool = True
    """Expose ACTUATOR MODEL ERROR to the policy (12 extra observation columns).

    The policy is a memoryless MLP. It already sees the actuator model's
    prediction and the current measured extension, but forming the error needs
    two steps of history, so it cannot compute it. Under ROM-A that costs
    nothing -- the ROM reaches its command. Under PPO4's learned ID/FD it is the
    difference between a policy that can compensate model error and one that
    cannot even see it.

    Legacy 54-value checkpoints must first be migrated with
    ``scripts/migrate_shared_policy.py``."""
    terrain_head_pose_solve: bool = False
    """Command the head from its own measured angle minus its measured tilt.

    Replaces the `a * q4 + b` coupling with a pose-dependent solve. Measured on
    ID/FD: the coupling is wrong by 0.296 rad mean / 0.783 rad max while q4
    spans only 0.06 rad, so the head angle is not a function of q4 and the
    proportional levelling term cannot null the resulting standing offset.

    Default False keeps PPO3's validated coupling, which its notes record as
    crude but the only configuration measured to hold the head there."""
    terrain_head_pose_solve_gain: float = 1.0
    """How much of the measured tilt to remove per step in the pose solve."""
    terrain_head_feedforward_from_measured: bool = True
    """Build the head levelling feedforward from MEASURED q4, not commanded q4.

    The head is slaved to the boom by `joint6_coupling_a * q4`. Taking q4 from
    the command assumes the actuator reaches what it was told. ROM-A effectively
    does; a learned ID/FD actuator does not, and the residual becomes a fixed
    head-alignment offset the policy cannot correct because it is applied after
    the policy's own contribution, every substep.

    Default False preserves every stored PPO3 number. PPO4 sets it True."""
    terrain_head_joint_stiffness_scale: float = 1.0
    """Stiffness scale for Messerkopf_joint alone. Keep at 1.0.

    `terrain_arm_stiffness_scale` applies to the held joints, and that set
    includes the tool joint -- which is Messerkopf_joint, the head ORIENTATION
    joint, not part of the vertical impact load path. Softening it with the
    cylinders was measured to cost exactly what it should: min_head_alignment
    0.9500 -> 0.9150 against a 0.950 gate, while buying none of the bounce
    reduction, which comes from the cylinders carrying the boom."""
    terrain_compliant_contact_stiffness: float = 0.0
    """PhysX implicit contact-spring stiffness on the verge, N/m. 0 = rigid.

    The head-tool bounce is ejection off a rigid contact, not a tracking
    failure: 94.4 % of separations leave the ground moving UPWARD (+0.92 m/s
    mean, 3.99 max) after force spikes of 53 kN median and 162 kN max, on ground
    that is on average RISING (reports/contact_loss_policy_v5.json, 269 events
    in 29 440 scored samples).

    With rigid contacts the series stiffness is the cylinders' hydraulic lock
    (~5e6 N/m), so contact force is purely commanded position:

        0.40 mm penetration ->   2 kN
        5.60 mm             ->  28 kN   (measured mean)
       32.50 mm             -> 162 kN   (measured max)

    while the terrain under the cutter moves 1.5 mm per control step. Holding a
    gentle contact therefore demands sub-millimetre regulation at 60 Hz through
    an EMA-filtered, slew-limited velocity command with no force feedback. That
    authority does not exist, which is why contact fraction never moved under
    any reward or physics change tried.

    This adds a real spring in series with that lock, at the tool/ground
    interface only, leaving the boom's own dynamics alone -- the simulation
    equivalent of the floating cutter head a real verge mower carries."""
    terrain_compliant_contact_damping: float = 0.0
    """Damping of the contact spring, Ns/m. Only used when stiffness > 0.

    This is what actually kills the bounce. Stiffness alone stores the impact
    energy and returns it; damping dissipates it, so the head settles onto the
    verge instead of being thrown off it."""
    terrain_kinematic_drive_substep_interpolation: bool = False
    """Aim each chassis pin at this substep's travel, not the end of the step.

    `_pre_physics_step` advances `_truck_travel_m` by a whole control step before
    any physics runs, while the fixture pins the chassis up to four times inside
    that step. Every pin therefore targeted a position up to 18.52 mm (4 km/h x
    1/60 s) ahead of where the truck belonged at that instant, and the substeps
    in between kept integrating the truck past it. Measured live from the PhysX
    root transform, per control step:

        pin phase 0 (pre-physics)      0.00 mm
        pin phase 1 (substep A)      +18.52 mm   <- teleport a full step ahead
        pin phase 2 (substep B)       -8.10 mm   <- dragged back
        pin phase 3 (_get_dones)      -8.10 mm

    A +-8 mm longitudinal square wave at 60-120 Hz, on a correct 18.52 mm/step
    mean. No gate could see it: p95_tool_acceleration is a TRUCK-FRAME finite
    difference and a root teleport moves chassis and tool rigidly together, so it
    cancels identically; the chassis-reaction gates keep only the component
    perpendicular to the heading; truck_drift is assigned literal zero after the
    correction. It is not invisible under the cutter, where 8 mm of longitudinal
    travel across 0.15 m verge facets modulates the ground height beneath a
    rigidly pinned tool.

    TRUE subtracts the not-yet-simulated share of the advance, so each pin aims
    where the truck should be at that substep and the correction collapses to the
    commanded-vs-integrated friction gap. Mean speed is unchanged -- the window
    closes at `_get_dones` with the full advance applied. FALSE restores the
    end-of-step target every stored report in reports/ was measured under.

    Measured (lawn_rough_stage5_v5/model_best.pt, 128 envs x 960 held-out steps,
    reports/drive_sawtooth_{baseline_16of17,interpolated_15of17}.json). FALSE
    reproduces the stored FINAL_16of17 score to every digit (0.026448503), which
    is the control proving this refactor is inert when disabled:

        metric                       FALSE      TRUE     gate
        min_wheel_road_duty         0.1947    0.2263    >=0.20   <- FIXED
        max_chassis_lateral_mm     121.13    101.13       --
        p95_cut_height_error_m      0.0631    0.1039    <=0.080  <- BROKE
        min_head_alignment          0.9500    0.9248    >=0.950  <- BROKE
        p95_tool_acceleration      91.71    116.41      <=120
        ground_contact_fraction     0.9350    0.9180    >=0.90
        gates passed                 16/17     15/17

    DEFAULT FALSE IS NOT A JUDGEMENT THAT THE END-OF-STEP TARGET IS CORRECT --
    it is not; an 8 mm 120 Hz teleport is a defect. The default protects the one
    checkpoint this package ships, which was TRAINED against that teleport and is
    co-adapted to it: enabling the fix is an off-distribution plant change for
    it, and it degrades accordingly. Note the failure axis -- head alignment and
    cut error, i.e. tracking -- not the chassis quantities the fix targets, which
    all improve. Whether the corrected fixture supports a BETTER policy is
    unmeasured and needs a fine-tune, not another evaluation of this checkpoint.
    Turn this on for the next training run; do not turn it on under v5."""
    # Set to 0.0 once terrain_constrain_truck_per_substep=True drove the measured
    # reaction to the floor: p95 17.72 mm / 0.206 deg against a zero-action floor
    # of 16.40 mm / 0.202 deg. There is nothing left for a policy to remove, and
    # penalizing a quantity already at its floor only pays the policy to disengage
    # the tool -- measured twice (v3/v4: sway 54.77 -> ~39 mm bought with contact
    # 0.9363 -> 0.9149 and a flipped head). The two chassis gates stay as
    # regression barriers.
    terrain_chassis_reaction_weight: float = 0.0
    """Penalize arm-induced chassis reaction (the sway), measured pre-correction.

    The fixture corrects the chassis pose every step, so every post-correction
    drift metric reads 0.0000 and the sway was invisible to all 15 gates --
    ``truck_drift`` is gated at 1 mm against a quantity assigned literal zero in
    ``_constrain_truck_xy``, so it cannot fail. The residual error the fixture
    has to remove IS the reaction the arm fed into the truck, and unlike the
    fixture it is something the policy can act on.

    Measured on the 15/15 checkpoint (v5, 128 x 960 held-out, 2026-07-31):
    lateral p95 54.77 mm, max 192.91 mm; attitude p95 0.629 deg, max 2.094 deg.
    Five centimetres of chassis slide, entirely unpenalized and entirely
    ungated -- which is why every physics-side attempt to remove the visible
    sway had to fight the plant instead of the policy.

    Complements ``terrain_constrain_truck_per_substep`` rather than replacing
    it: that flag suppresses sway in the plant at the cost of the wheel gate,
    this teaches the arm not to cause it in the first place."""
    terrain_chassis_lateral_scale_m: float = 0.020
    """Normalizer for lateral chassis reaction.

    Chosen so the measured p95 (54.77 mm) sits at ratio 2.74, i.e. inside the
    quadratic's responsive region and well clear of the clamp -- a scale small
    enough to put today's behaviour at the ceiling would leave no gradient
    exactly where the mass of the distribution is."""
    terrain_chassis_attitude_scale_rad: float = 0.004
    """Normalizer for chassis attitude reaction.

    0.004 rad = 0.229 deg puts the measured attitude p95 (0.629 deg) at ratio
    2.745, matching the lateral channel's 2.74 at the same operating point, so
    neither channel silently dominates the other at the starting policy."""
    terrain_chassis_reaction_max_ratio: float = 5.0
    """Clamp on the normalized reaction, so one impact cannot dominate returns.

    5.0 = 100 mm lateral / 1.15 deg attitude, above the measured p95 and below
    the measured max, so ordinary sway is shaped and a single strike is bounded
    at -6.25 rather than being allowed to swamp a ~10 task reward."""
    terrain_head_tilt_arm_frame: bool = False
    # Active c4 recovery along the four-cylinder Cartesian nullspace. Unlike
    # the directional shield, this contributes a restoring command; unlike a
    # c4-only correction, c1-c3 cancel its first-order cutter XYZ motion. Off in
    # PPO3 until an actuator-specific held-out evaluation enables it.
    terrain_head_nullspace_restore_gain: float = 1.0
    terrain_head_nullspace_restore_deadband_rad: float = 0.01
    terrain_head_nullspace_restore_max_action: float = 0.60
    terrain_head_restore_preserve_xyz: bool = True
    """Measure head tilt about the joint's own axis instead of world Y-Z.

    World-X tilt is in-plane tilt rotated by the arm's slew, so the world
    projection under-reads the real error as the arm turns: a 25 deg tilt at
    70 deg slew reads 9.06 deg in world frame and 25.00 deg in the joint frame.
    The levelling correction therefore fades exactly where training collapses,
    and the unmeasured remainder appears as world-X -- the axis that looked
    unsteerable. At evaluation poses (near-level, near-zero slew) the two agree
    to the digit, so this is free on the gates: measured 15/15 / 0.0000 either
    way, with identical align 0.9621 / cut 0.0498 / accel 111.57.
    """
    terrain_head_tilt_axis_index: int = 0
    """Head-local axis the tilt joint rotates about (0=X, 1=Y)."""
    terrain_messerkopf_level_filter: float = 0.5
    """EMA on the correction; p95_tool_acceleration is a gate and the term is
    recomputed every decimation substep, so unfiltered it injects tool motion."""
    """Bound on the correction so a bad tilt reading cannot sweep the joint."""

    terrain_head_down_required_alignment: float = 0.95
    terrain_head_down_latches_failure: bool = True
    # Alignment at which a latched env is paid again. Must exceed
    # `terrain_head_down_required_alignment`; the 0.02 gap is the hysteresis
    # that prevents a head oscillating across the latch boundary from earning
    # on the upswing. Set equal to or below `required` to restore the old
    # monotone, episode-long latch. See the note in
    # TerrainSolePolicyEnv._get_rewards for the five-block measurement that
    # motivated this.
    terrain_head_down_latch_release_alignment: float = 0.97
    terrain_head_tilt_shield_soft: float = 0.30
    terrain_head_tilt_shield_hard: float = 0.80
    terrain_c4_action_to_axis_y_sign: float = -1.0
    # Start warning one centimetre below the 5 cm mowing target, well before a
    # geometric strike. The initial 2 cm / -2 barrier left held-out quarters 1
    # and 2 systematically below target and the strike rate flat at 32.6%.
    terrain_clearance_barrier_weight: float = 0.0
    terrain_clearance_margin_m: float = 0.0
    terrain_clearance_barrier_scale_m: float = 0.02
    terrain_clearance_barrier_max_deviation: float = 5.0
    terrain_contact_reward_weight: float = 4.0
    """Pay for keeping the cutter planted on the verge.

    Raised from 2.0 on 2026-07-31. The residual ~6.5 % skipping
    (ground_contact_fraction 0.9363, worst family s_curve 0.9193) survived every
    physics change tried -- depenetration velocity, ride-height preload, solver
    iteration count, and per-substep chassis pinning all left it within 0.002 --
    which is the signature of an objective problem, not a simulation one. At 2.0
    a cutter 2 cm off the ground still collects 5.13 of the 8.0 cut-tracking
    term, so floating through a rise was close to free; at 4.0 staying planted
    is worth 12.0 against 5.13. Binary on measured PhysX contact, so it pays for
    being in contact and cannot be farmed by pressing harder -- the cut-height
    term and its gate still bound that direction."""
    # ------------------------------------------------------------------
    # Contact safety (shared by PPO3/ROM-A and PPO4/ID-FD)
    #
    # Measured pre-fix on the held-out verge, stage 5, s_curve, height scale
    # 1.0, 960 steps x 32 envs, seed 17, checkpoint shared_v2_rom/model_migrated
    # (reports/contact_safe_v1/prefix_ppo3_s_curve_seed17.json):
    #
    #     ground_contact_fraction            0.9486   <- PASSES the 0.90 gate
    #     contact-loss events                   428
    #     visible lifts (gap > 5 mm)            334
    #     hard impacts (> 30 kN)                286
    #     max consecutive airborne steps         45   (0.75 s in the air)
    #     max airborne gap                   0.297 m
    #     upward separations                  73.4 %
    #     max upward cutter velocity        4.41 m/s
    #     max contact force                   119 kN
    #
    # The first and last lines are the whole problem: the shipped gate reads a
    # PASS while the cutter spends three quarters of a second 30 cm off the
    # verge. `ground_contact_fraction` is a DUTY CYCLE, and a duty cycle cannot
    # distinguish 428 short flights from continuous light support.
    # ------------------------------------------------------------------
    terrain_contact_acquire_steps: int = 3
    """Consecutive contact steps before the episode claims physical support.

    Support has to be claimed before its loss can be a failure, and it must not
    be claimable during the reset ramp -- the arm is being driven to its pose
    there, so contact then says nothing about the policy."""
    terrain_contact_loss_grace_steps: int = 2
    """Consecutive missing-contact steps tolerated before support is dropped.

    Curriculum: 4 (stage A) -> 2 (stage B) -> 1 (stage C) -> 0 at strict
    evaluation. Not zero during early training on purpose: the contact sensor
    is read per physics substep and one missed sample must not delete an
    episode, or the policy is trained against sensor noise rather than against
    the tool leaving the ground."""
    terrain_max_airborne_gap_m: float = 0.005
    """Geometric gap above the calibrated support height that counts as a lift.

    Geometric, not sensor-derived, so 'no visible lift' is a claim about where
    the cutter IS. 5 mm against a measured pre-fix maximum of 297 mm."""
    terrain_takeoff_velocity_threshold_mps: float = 0.02
    """Upward cutter velocity at separation that makes it a takeoff."""
    terrain_hard_impact_force_n: float = 30000.0
    """Force above which a sample is a hard impact. Pre-fix maximum 119 kN."""
    terrain_impact_ejection_force_n: float = 30000.0
    """Force in the sample BEFORE a takeoff that makes it an ejection."""
    terrain_max_takeoff_events: int = 0
    """Impact/ejection events per episode before termination. 0 disables."""

    terrain_terminate_on_contact_loss: bool = False
    """Terminate when support is lost for longer than the grace window.

    Default False, and NOT only for reproducibility. Resets in this task are
    ALL-OR-NOTHING: `_run_reset_preposition` raises outright when handed fewer
    than num_envs environments, because the physical deployment ramp steps the
    whole scene. A contact termination therefore has to synchronize like every
    other termination the parent produces -- one environment lifting resets all
    of them -- and the pre-fix policy lifts ~430 times per 32-env/960-step
    rollout, which at 256 training environments is continuous reset work. That
    is the same trap `defer_safety_terminations` was added to escape.

    So the seven-term contact reward is the signal that has to do the work
    (`airborne_task_reward_bound` is asserted against the contact-loss penalty
    precisely so that floating is strictly worse than staying planted WITHOUT
    needing the episode to end), and this switch is the belt-and-braces for a
    policy that is already close. Turn it on late, not early.

    Never fires during preposition, reset correction or settling -- see
    `contact_safety_terminal`."""
    terrain_contact_loss_latches_failure: bool = False
    """Once support is visibly lost, earn NO task reward for the rest of the
    episode -- instead of ending it.

    This is the affordable form of the requirement "a policy must not be
    rewarded for completing the path after visibly lifting", and it is the
    pattern this codebase already uses for exactly this problem. From
    `_get_rewards` on the head-down latch: *"Terminating on a violation is not
    affordable here: any termination synchronizes and _run_reset_preposition
    steps the whole scene, so at 512 envs a per-step head check would reset
    continuously. Latching the failure for the rest of the episode buys the
    same guarantee for free."*

    MEASURED, on this task, with `terrain_terminate_on_contact_loss = True` at
    256 environments, grace 4, boom_sweep-heavy Stage B: **mean episode length
    10-11 steps** against a 960-step horizon, at a per-step contact-termination
    rate of only 0.0039. One environment in 256 lifting resets all 256, so the
    policy never saw more than ~1 % of an episode and every reset paid a full
    deployment ramp. Training was impossible, not merely slow.

    The latch gives the same incentive -- finishing the path after a visible
    lift pays nothing -- at zero reset cost, and unlike termination it keeps
    the safety barriers live so recovery still has a gradient.

    Released by `terrain_contact_loss_latch_release_steps` of continuous
    re-acquired support, so a policy that recovers is not punished for the
    whole remaining episode. Set that to 0 for a monotone episode-long latch."""
    terrain_contact_loss_latch_release_steps: int = 60
    """Consecutive in-contact steps that clear the latch. 0 = never."""
    terrain_terminate_on_hard_impact: bool = False
    terrain_lift_terminal_reward: float = -100.0
    """Finite terminal penalty for a contact-safety termination.

    Large enough that finishing the path after lifting cannot pay, finite so it
    cannot destabilise the value targets the way an infinite penalty would."""

    # --- contact-safety reward group (terrain_rewards.compute_contact_safety_reward) ---
    terrain_contact_safety_reward: bool = False
    """Enable the seven-term contact reward. Off reproduces historical scores."""
    terrain_contact_support_weight: float = 4.0
    """Pay for physical support. Replaces `terrain_contact_reward_weight`, which
    the contact-safe configuration sets to 0.0 so support is not paid twice."""
    terrain_contact_loss_weight: float = 20.0
    """Penalty per step of lost support after acquisition.

    Must exceed `airborne_task_reward_bound` (cut 8.0 + lateral 3.0 + progress
    1.5 = 12.5 here) or floating through a rise stays profitable. Asserted in
    tests/test_contact_safety.py rather than left as a comment."""
    terrain_contact_force_weight: float = 2.0
    terrain_contact_force_target_n: float = 6000.0
    """Force the band peaks at.

    NOT the 2 kN design figure. Measured stable-contact force runs at 18.9 kN
    mean / 7.2 kN p05, so a band centred on 2 kN with any sane width is flat
    across the entire region the policy occupies -- the same reward-saturation
    defect §3.5 of HANDOVER_PPO3_PPO4.md records for the head penalty, where
    alignment 0.9477, 0.80 and 0.66 all scored -40.00 identically. 6 kN sits at
    the light end of measured stable contact, so it is both gentler than today
    and inside the reachable set."""
    terrain_contact_force_std_n: float = 8000.0
    """Band width. Wide enough that the measured 18.9 kN mean retains gradient."""
    terrain_contact_excess_weight: float = 2.0
    terrain_contact_force_soft_limit_n: float = 20000.0
    """Force above which the quadratic excess penalty engages."""
    terrain_contact_excess_max_ratio: float = 5.0
    terrain_contact_hard_impact_weight: float = 20.0
    terrain_contact_takeoff_weight: float = 4.0
    terrain_contact_takeoff_velocity_scale_mps: float = 0.10
    terrain_contact_takeoff_max_ratio: float = 5.0
    terrain_contact_weak_support_force_n: float = 500.0
    terrain_contact_gap_weight: float = 8.0
    terrain_contact_gap_scale_m: float = 0.005
    terrain_contact_gap_max_ratio: float = 4.0

    terrain_contact_observation: bool = False
    """Expose the six contact-state channels to the policy (Phase 5).

    Changes the observation width, so a legacy checkpoint must be migrated with
    ``scripts/migrate_shared_policy.py`` before it can be loaded."""
    terrain_head_frame_observation: bool = False
    """Expose the JOINT-FRAME head tilt and the arm slew to the policy.

    The policy currently sees an unsigned alignment cosine and the WORLD-frame
    head axis. Neither is the error the head joint can act on. Measured on
    stage 6 at 2.2x amplitude (reports/stage6/head_frame/ppo3_amp2.2.json),
    the correctable tilt is 1.545x the world-frame measure on the samples
    where alignment collapses, against 1.048x when healthy, and
    corr(|slew|, joint_tilt - world_tilt) = +0.814.

    Adds 2 columns, so a narrower checkpoint must be migrated."""
    terrain_contact_force_rate_scale_nps: float = 1.0e6
    """Normalizer for the force-rate observation channel.

    Measured p95 |dF/dt| is 1.18e6 N/s on ROM-A and 1.12e6 on ID/FD, so 1e6
    puts the mass of the distribution near unit magnitude and the +-5 clip at
    5e6, above the measured 6.9e6/8.6e6 maxima only in the extreme tail."""
    terrain_contact_gap_observation_scale_m: float = 0.02
    """Normalizer for the airborne-gap observation channel.

    20 mm = 4x the 5 mm visible-lift tolerance, so the channel is well inside
    its clip across the whole region where recovery is still possible."""

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
    terrain_action_filter: float = 0.55
    # Calibrated to the original stage-5 p95_tool_acceleration gate value of
    # 80 m/s^2, not to the free-space 30 m/s^2 scale inherited from env_cfg.
    # The gate was later re-derived to 120 (see terrain_convergence.py) after
    # the zero-action floor measured 100.06; the reward scale deliberately
    # stays at 80 -- it prices the trainable, command-attributable share and
    # is the only pairing measured to improve the held-out score (v3 block 1,
    # 0.673 -> 0.659 under the old gate).
    #
    # Measured stage 5, lawn_rough block 1 (reports/heldout_block_0001.json):
    # the -0.50 / 30 pairing cost only -2.47 at 148.7 m/s^2 and PPO simply
    # bought acceleration with tracking -- p95 acceleration rose 125.9 -> 148.7
    # m/s^2 while lateral error *improved* 0.078 -> 0.071 m, and the held-out
    # score regressed 0.673 -> 1.316. Both tracking gates retain wide margin
    # (mean cut 0.012 of 0.040 m, lateral 0.071 of 0.30 m), so there is room to
    # spend tracking on smoothness; this pairing roughly doubles both the cost
    # at the observed operating point and the marginal gradient.
    # Still bounded by terrain_acceleration_max_ratio, now saturating at
    # 800 m/s^2 rather than 300.
    sole_acceleration_scale_mps2: float = 80.0
    sole_acceleration_weight: float = -2.5
    # Held at -0.30. Raising this to -0.60 alongside the acceleration change
    # above collapsed the cutter head: stage 5 v2 block 1 measured
    # max_abs_head_axis_x 0.861 (5x the 0.16 hard tilt shield), min head
    # alignment 0.337 and p01 0.717, against 0.039 / 0.904 / 0.965 for the same
    # checkpoint under -0.30. Cylinder 4 owns head pitch, so an action-rate tax
    # falls hardest on exactly the fast c4 corrections that keep the head down
    # as boom attitude shifts over the verge, and PPO chose head drift over
    # paying the rate cost. The acceleration term above reaches the same
    # smoothness objective without that coupling.
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
        configure_floating_cutter(self)
        # Terrain patches are large; a wide env grid needs terrain to match.
        # See TerrainSolePolicyEnv._setup_scene, which checks coverage.
        self.scene.num_envs = 16
        # The 26 m verge meshes must never overlap adjacent X columns: overlap
        # would let one environment's cutter contact another environment's
        # independently generated height profile.
        self.scene.env_spacing = 28.0
        from verge_task.terrain_observations import verge_observation_dim
        self.observation_space = (
            verge_observation_dim(
                model_error=bool(self.terrain_model_error_observation),
                model_error_history=int(self.terrain_model_error_history),
                contact_state=bool(self.terrain_contact_observation),
            )
            + 1
        )

        pos = self.robot_cfg.init_state.pos
        self.robot_cfg.init_state.pos = (pos[0], pos[1], float(self.truck_spawn_z))

        # Cutter-head holding torque. Overridden here rather than in
        # mulag_upper_level_env_cfg because PPO2 has trained checkpoints against
        # the inherited 20000 and must not move underneath them.
        #
        # `Messerkopf_joint` carries no cylinder -- it is slaved to c4 at
        # joint6_coupling_a=0.76 -- and at the inherited ceiling it cannot hold
        # the reference the control pipeline writes. Measured on the stage-5
        # verge with zero commanded action (reports/head_joint_tracking_*.json,
        # 64 envs x 600 steps): tracking error 0.0674 rad mean / 0.2096 rad p99
        # / 0.2435 rad max, and torque pinned at the ceiling 23.1 % of samples --
        # against 0.0016 / 0.0130 and 8.5 % for c4's own Schwenk joint, a 43x
        # worse error on the one joint nothing drives. The head's true
        # orientation therefore departs up to 12 degrees from the commanded one,
        # and the departure lives on the axis the cylinders cannot steer: the
        # measured authority Jacobian (reports/head_axis_authority.json) puts
        # head-X authority at 3.8 % of head-Y, so an X excursion is not
        # recoverable and must instead be prevented.
        #
        # This raises only the two head joints. The large cylinders keep their
        # inherited caps: those were cut from 1e7-1e8 deliberately because arm
        # reaction dragged the free-standing truck ~0.5 m ("mulag starts
        # flying"), and that budget is sized against the truck's friction, not
        # the head's. Verify with truck_drift and truck_heading, which both
        # measure exactly 0.0000 today against 0.001 m / 0.1 deg thresholds and
        # will expose any reaction-force regression immediately.
        # `Messerkopf_joint` carries no cylinder and is NOT hydraulic, so the
        # inherited model -- slaved to c4 at joint6_coupling_a=0.76 with
        # hydraulic-lock gains (stiffness 5.0e6, damping 1.12e6) -- is wrong in
        # principle. Measured on the stage-5 verge it misses its reference by
        # 0.2096 rad p99 (43x c4's own joint) with drive torque pinned at any
        # ceiling it is given; raising that ceiling 320000 -> 960000 cut
        # saturation 23.1 % -> 0.97 % and changed the held-out score by nothing
        # at all (bit-identical 0.11299147332708048).
        #
        # Decoupling it and using the teacher-student gains (mulag_arm_5:
        # stiffness/damping 6000, an independent joint) was measured and is
        # WORSE, and so is every interpolation tried:
        #
        #     stiffness   min_head_align   p95_accel   p95_cut   score   fails
        #     5.0e6 (0.76)      0.9628       130.9     0.0817   0.1130     2
        #     6.0e3 (indep)    -0.6956       109.4     0.0805   2.4268     6
        #     3.0e5 (indep)     0.6268       132.1     0.1205   1.6734     7
        #
        # The response is not monotonic in stiffness, so this is not a gain to
        # bisect. The reason is that a decoupled joint here is commanded to a
        # CONSTANT 0.0 rad, and the angle that points the tool at world-down is
        # pose-dependent -- it changes as c1/c2/c3 move the boom. The 0.76
        # coupling is a crude stand-in for that dependence (the inherited comment
        # records the true ratio as "pose-dependent, 0.50-0.76 across tests"),
        # and crude beats absent: every decoupled variant loses head alignment.
        #
        # The real fix is a pose-dependent reference that solves for the
        # Messerkopf angle putting the head at world-down given the current boom
        # attitude, not a constant and not a fixed ratio. Until that exists the
        # inherited coupling stays, because it is the only configuration
        # measured to hold the head.
        #
        # Independently useful finding for whoever does that work: the two gates
        # eight training blocks could never move -- p95_tool_acceleration and
        # p95_cut_height_error -- BOTH improved the moment the joint stopped
        # fighting its command (130.9 -> 109.4, gate 120, and 0.0817 -> 0.0805,
        # gate 0.080). They are reachable through the head model, not through
        # reward shaping.
        # Contact should settle instead of ejecting the cutter at the asset's
        # inherited 5 m/s depenetration cap. This remains a real PhysX contact;
        # it only limits the solver's separation velocity after overlap.
        # 0.05, down from 0.20. Measured cause of the last failing gate
        # (reports/tool_accel_policy_blk1.json, 128 envs x 960 steps): split by
        # cutter contact, the in-contact population already PASSES at p95 110.4
        # with 9.8 % over the 120 gate, while the 7.1 % free-swing population
        # measures p95 188.7 with 57.3 % over. The worst 5 % of accelerations
        # are 63.7 % in contact against a 92.9 % baseline, so contact loss is
        # what populates the tail. The cutter is being ejected off terrain and
        # accelerating in free space; capping the depenetration speed attacks
        # that directly rather than smoothing a command stream that is only
        # 42 % of the variance (command-only R2 0.425).
        self.robot_cfg.spawn.rigid_props.max_depenetration_velocity = 0.07
        # NOT solver_position_iteration_count: setting it to 16 here was measured
        # bit-identical to the 8 in the asset, so it does not take effect at this
        # layer. The cutter bounce was never a contact-resolution problem anyway
        # -- it was chassis sway feeding motion into the arm, fixed by
        # terrain_constrain_truck_per_substep.




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
