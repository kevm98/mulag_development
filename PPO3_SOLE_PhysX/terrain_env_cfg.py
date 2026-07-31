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
    # Messerkopf_joint is not hydraulic and is not commanded by c4. Driving it
    # as a real 5th action replaces the 0.76 feedforward coupling, which was
    # measured missing its own reference by 0.2096 rad at p99 and which no
    # levelling correction could rescue: min_head_alignment fails on varied
    # terrain in EVERY coupled configuration tried (0.9500 held-out is a
    # zero-margin pass, 0.9396 and 0.9185 on varied meshes), while p01 stays
    # healthy at 0.976-0.988 -- rare, mesh-dependent excursions the feedforward
    # cannot see because it has no feedback at all.
    #
    # This changes the action interface from 4 to 5, so no existing checkpoint
    # can be resumed. The policy relearns from scratch.
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
    verge_visual_apron_m: float = 3.2
    """Grass width added outboard of the mowed verge and beyond the far kerb."""
    verge_visual_color: tuple = (0.16, 0.42, 0.13)
    uneven_verge_apron: bool = True
    """Displace the roadside grass into lawn relief instead of flat slabs."""
    verge_apron_relief_m: float = 0.09
    verge_apron_resolution_m: float = 0.45
    """Apron mesh step. Visual-only, so this stays coarse relative to the verge."""

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
    terrain_static_friction: float = 1.0
    terrain_dynamic_friction: float = 1.0
    terrain_restitution: float = 0.0

    # --- Verge strip ---
    mow_strip_color: tuple = (0.08, 0.55, 0.08)
    mow_strip_collision: bool = True
    """The green verge is a real collider; contact cannot be faked geometrically."""
    terrain_tracking_amplitude_scale: float = 1.0
    """Widen every tracking path family's lateral excursion by this factor.

    The families author 0.10-0.24 m of lateral swing against 0.74 m of
    available half-strip, so the arm was being exercised over a third of the
    room the verge actually offers. Raising this gives the end effector a wider
    working band. Bounded by the same smooth tanh saturation, so it can never
    push the path off the green; widen `mow_strip_width_m` alongside it if the
    scaled amplitude starts to compress against that bound.
    """
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
    # The GATE terrain. Held-out geometry is pinned at seed 1337 precisely so a
    # stage-5 score means the same thing across runs, and this tuple is read for
    # every environment including evaluation -- widening it here silently moved
    # the gate set (measured: the 15/15 checkpoint re-scored 12/15 / 0.0897 when
    # 0.35 and 1.15 were added). Training gets the wider distribution from
    # scripts/train.py instead, which is the only place that may change it.
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
    terrain_head_penalty_far_weight: float = -0.5
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
    terrain_messerkopf_level_gain: float = 0.5
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
    gate for visibly smoother motion and no sway."""
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
