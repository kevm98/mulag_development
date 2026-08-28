from __future__ import annotations

from pathlib import Path

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
try:
    from isaaclab.sim import PhysxCfg
except ImportError:
    # Isaac Lab moved PhysxCfg into isaaclab_physx.physics on some builds
    # (e.g. 3.0.0-beta2); same fields, same defaults.
    from isaaclab_physx.physics import PhysxCfg

# Restores cfg.sim.physx on builds that renamed the field to .physics, so the
# repo's existing physx accesses keep working. See physx_compat for why this is
# still the same PhysX backend either way.
from mulag_plant.runtime.physx_compat import PHYSX_FIELD as _PHYSX_FIELD
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mulag_plant.runtime.sim_memory import apply_kevin_sim_memory_optimizations
from mulag_plant.runtime.rrlab_reference_conventions import (
    OBSTACLE_ARM_JOINT_NAMES,
    OBSTACLE_DRIVING_SPEED_RANGE_KMH,
    OBSTACLE_DRIVING_SPEED_RESAMPLE_STEP_THRESHOLD,
    OBSTACLE_DRIVING_SPEED_UPDATE_INTERVAL_S,
    OBSTACLE_HYDRAULIC_COMMAND_JOINT_NAMES,
    OBSTACLE_WHEEL_JOINT_NAMES,
    OBSTACLE_WHEEL_RADIUS_M,
)
from rrlab_assets import MULAG_CFG

_PIPELINE_DIR = Path(__file__).resolve().parents[1]
_MODELS_DIR = str(_PIPELINE_DIR / "models")
_MODELS_CYLINDER_DIR = str(_PIPELINE_DIR / "models" / "vendor_models")
_CYLINDER_MAPPING_PATH = str(Path(__file__).resolve().parent / "cylinder_coordinate_mapping.json")

# -- Observation dimension constants -------------------------------------------------
ARM_JOINT_DIM = 5
DISTANCE_TO_GOAL_DIM = 2
UNIMOG_POSE_DIM = 4
UNIMOG_POSE_DOT_DIM = 3
HISTORY_STEPS = 3

# Compact observation: only live data (no placeholder zeros for missing sensors)
# joint_pos(5*h), joint_vel(5*h), distance_to_goal(2*h), joint_torques(5),
# last_action(5), unimog_pose(4), unimog_pose_dot(3)
OBS_DIM = (
    ARM_JOINT_DIM * HISTORY_STEPS  # 15
    + ARM_JOINT_DIM * HISTORY_STEPS  # 15
    + DISTANCE_TO_GOAL_DIM * HISTORY_STEPS  # 6
    + ARM_JOINT_DIM  # 5
    + ARM_JOINT_DIM  # 5
    + UNIMOG_POSE_DIM  # 4
    + UNIMOG_POSE_DOT_DIM  # 3
)  # = 53


@configclass
class MulagUpperLevelEnvCfg(DirectRLEnvCfg):
    # -- Core --
    decimation = 2
    episode_length_s = 34.0
    action_scale = 0.02  # PPO reference increment; large enough for visible cylinder motion.
    action_space = 5
    observation_space = OBS_DIM
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=2,
        **{
            _PHYSX_FIELD: PhysxCfg(
                min_position_iteration_count=16,  # Increased from default 1 for better contact
                max_position_iteration_count=32,  # Increased from default 255 (effective max)
                min_velocity_iteration_count=2,   # Increased from default 0 for stability
                max_velocity_iteration_count=2,   # Keep same as min
                bounce_threshold_velocity=0.2,    # Threshold for bounce
                friction_offset_threshold=0.04,   # Friction offset
                friction_correlation_distance=0.025, # Friction correlation
            )
        }
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=169, env_spacing=20.0, replicate_physics=False)
    robot_cfg: ArticulationCfg = MULAG_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    models_dir = _MODELS_DIR

    # -- Arm joint names and end-effector --
    arm_joint_names: tuple[str, ...] = tuple(OBSTACLE_ARM_JOINT_NAMES)
    cylinder_joint_names: tuple[str, ...] = tuple(OBSTACLE_HYDRAULIC_COMMAND_JOINT_NAMES)
    task_space_ee_body_name = "Messerkopf"

    # -- PPO2 -> cylinder command -> hydraulic ROM -> PhysX pipeline --
    use_cylinder_id_pipeline = True
    cylinder_models_root = "disabled"
    id_cylinder_checkpoint = "disabled"
    id_checkpoint_override_path = ""
    rom_cylinder_checkpoint = "disabled"
    base_rom_checkpoint = "disabled"
    command_rom_checkpoint = str(
        Path(__file__).resolve().parents[2]
        / "rom_model/command_run_001/best_rom.pt"
    )
    fd_cylinder_checkpoint = "disabled"
    cylinder_dynamics_model = "command_rom"
    cylinder_coordinate_mapping = _CYLINDER_MAPPING_PATH
    cylinder_coordinate_mapping_mode = "nominal_sign_offset"
    enable_cylinder_space_scaling = True
    allow_provisional_nominal_reference = True
    allow_provisional_range_mapping = False
    strict_cylinder_usd_scaling = True
    strict_cylinder_usd_delta_scale = True
    cylinder_usd_limit_tolerance_m = 1.0e-5
    cylinder_usd_delta_scale_tolerance = 1.0e-4
    max_abs_valve = 1.0
    # Plant-side perturbation for the AAM online-adaptation experiments.
    # Applied at the FD boundary, AFTER the controller has produced its
    # command, so the controller never sees these values and the AAM must
    # infer their effect from the commanded/response mismatch. Off by default.
    plant_perturbation_enabled = False
    plant_perturbation_alpha_range = (1.0, 1.0)      # valve effectiveness
    plant_perturbation_delay_steps_range = (0, 0)    # transport delay, control steps
    plant_perturbation_step_time_s = None            # mid-episode step change
    plant_perturbation_step_alpha_scale = 1.0
    plant_perturbation_step_delay_steps = None
    plant_perturbation_seed = 20260816

    # Actuator-authority degradation: the fraction of the realised cylinder
    # motion that reaches the PhysX target. Applied at the same point on every
    # stack (after the actuator model, before the target update), so it is the
    # one stress axis that means the same thing for the command ROM and for
    # ID->FD. 1.0 is unperturbed. See _actuator_authority_alpha().
    actuator_authority_alpha = 1.0
    # Mid-episode change: from actuator_authority_step_time_s onwards the factor
    # becomes actuator_authority_alpha_after. None disables the step, and a sham
    # arm sets the two values equal so the schedule fires without a plant change.
    actuator_authority_alpha_after = None
    actuator_authority_step_time_s = None
    # Displacement-domain actuator stress, applied at the SAME boundary as
    # actuator_authority_alpha and therefore on every stack. Off by default;
    # at these defaults the chain is a bit-for-bit identity, so enabling the
    # feature cannot move an unstressed run. deadband/rate limit are METRES --
    # this is the delta_x domain, not the valve domain, so severities
    # calibrated on the valve do not transfer. See runtime/actuator_stress.py.
    actuator_stress_enabled = False
    actuator_stress_deadband_m = 0.0
    actuator_stress_lag_tau_s = 0.0
    actuator_stress_delay_steps = 0
    actuator_stress_rate_limit_m = None      # None -> unlimited
    actuator_stress_max_delay_steps = 8
    # ROM-side plant perturbation. The command ROM produces the valve rather
    # than consuming it, so valve effectiveness does not apply; its per-env
    # plant channel is system_factors (num_envs, 7) ordered
    # [compliance, flow, pressure, seal, seal, seal, seal].
    # Ranges/step values are keyed by NAME: compliance | flow | pressure | seal.
    rom_factor_perturbation_enabled = False
    rom_factor_perturbation_ranges = None        # e.g. {"seal": (0.0, 0.8)}
    rom_factor_perturbation_step_time_s = None   # mid-episode step change
    rom_factor_perturbation_step_values = None   # e.g. {"seal": 0.8}
    # Per-timestep trace for the online-adaptation figures. None disables it.
    # Distance-binned reports cannot resolve the AAM warmup or a mid-episode
    # step change, so those figures are built from this trace instead.
    adaptation_trace_path = None
    adaptation_trace_max_steps = 20000
    cylinder_rom_delta_clip_m = 0.02
    command_rom_dp_abs_limit_pa = 1.5e8
    command_rom_direction_guard = False
    command_rom_command_max_ratio = 3.0
    cylinder_fd_delta_clip_m = 0.02
    cylinder_fd_scale_delta_by_sample_time = True
    cylinder_fd_sample_time_s = 0.0  # 0 means use checkpoint metadata.
    cylinder_fd_delta_gain = 1.0
    # Blend the paired learned prediction with the requested cylinder-space increment:
    #   applied_delta = (1-blend)*FD(ID(target)-ID(zero)) + blend*target_delta_x
    # 0 preserves the historical pure learned path.  A nonzero value makes ID/FD a learned
    # residual around an intuitive, full-rank cylinder command instead of allowing model
    # cross-coupling to own all position-target authority.
    cylinder_idfd_reference_blend = 0.0
    # Optional output calibration applied only to FD(valve)-FD(zero valve).
    # Kept separate from spatial/unit scaling because valve authority is
    # strongly anisotropic across the four cylinders.
    cylinder_fd_residual_output_gain = [1.0, 1.0, 1.0, 1.0]
    # Optional causal safety envelope for paired ID/FD rollout.  Observational
    # checkpoints can predict a large or sign-reversed increment after their
    # own output changes the next state.  The envelope retains the learned
    # magnitude but prevents recursive target-integrator runaway.
    cylinder_fd_command_direction_guard = False
    cylinder_fd_command_max_ratio = 1.5
    cylinder_learned_model_dt_s = 0.0  # 0 means use FD checkpoint sample_time_s when available.
    clip_target_delta_xdot_to_training_range = True
    target_delta_xdot_clip_std = 3.0
    log_target_delta_xdot_clipping = True
    target_delta_xdot_clip_log_first_n = 20
    cylinder_hydraulic_feature_mode = "fd_dp_state"
    cylinder_neutral_hold_deadband_m = 1.0e-5
    cylinder_neutral_hold_velocity_deadband_mps = 1.0e-5
    ppo_neutral_action_deadband = 1.0e-6
    direct_valve_position_scale = 0.005
    joint_reference_velocity_clip = 0.8
    tool_joint_control_mode = "direct_position"
    # "position": position targets (implicit actuator drives).
    # "velocity": velocity targets -- maps learned vel to USD vel via velocity scale.
    cylinder_actuation_mode = "position"
    cylinder_velocity_target_scale = 1.0
    enable_joint6_coupling = True
    # Measured mechanical coupling q5 ~= 0.76 * q4 (extensive_debug test 7; the
    # ratio is pose-dependent, 0.50-0.76 across tests). 1.0 made the passive-joint
    # drive fight the mechanism by ~24% of q4's travel.
    joint6_coupling_a = 0.76
    joint6_coupling_b = 0.0
    joint6_coupling_source_joint_name = ""
    joint6_coupling_target_joint_name = ""
    enable_console_debug: bool = False
    enable_sequence_tick_debug: bool = False
    enable_reset_debug: bool = False
    enable_pipeline_debug: bool = False
    enable_startup_summary: bool = True
    enable_verbose_startup: bool = False
    debug_log_interval: int = 1000
    debug_env_ids: tuple[int, ...] = (0,)
    debug_max_envs_to_print: int = 1
    write_debug_csv: bool = False
    debug_csv_path: str = "outputs/cylinder_diagnostics/runtime_pipeline_debug.csv"
    console_log_level: str = "summary"
    pipeline_debug_prints = False
    pipeline_debug_first_steps = 5
    pipeline_debug_interval = 1000
    allow_debug_manual_cylinder_targets = True

    # Sequence ID/FD is explicitly unavailable in PPO2-ROM-PhysX.
    use_sequence_cylinder_models = False
    sequence_id_checkpoint = "disabled"
    sequence_fd_checkpoint = "disabled"
    sequence_history_length = "auto_from_metadata"
    sequence_model_dt = "auto_from_metadata"
    enable_sequence_runtime_debug = False
    sequence_runtime_log_interval = 1000
    sequence_model_decimation = "auto"
    sequence_history_update_mode = "model_dt"
    sequence_command_hold_mode = "distribute_fd_delta"
    sequence_fd_delta_gain = 1.0
    sequence_fd_residual_over_zero_valve = False
    paired_id_fd_residual = False
    fd_output_gain = 1.0
    fd_residual_blend = 0.0
    fd_correction_mode = "none"
    fd_calibration_path = ""
    fd_checkpoint_override = ""

    # -- Lower-level cylinder velocity servo --
    # "none": existing ID -> FD feedforward path.
    # "cylinder_velocity_pid": PID valve command only.
    # "id_ff_plus_cylinder_velocity_pid": learned ID feedforward plus PID residual.
    low_level_servo_mode = "none"
    cyl_vel_kp = [6.0, 6.0, 6.0, 6.0]
    cyl_vel_ki = [2.0, 2.0, 2.0, 2.0]
    cyl_vel_kd = [0.0, 0.0, 0.0, 0.0]
    cyl_vel_integral_limit = 0.2
    cyl_vel_cmd_limit = 0.05
    valve_cmd_limit = 1.0
    servo_deadband_mps = 1.0e-4
    servo_rate_limit = 0.0
    servo_use_fd_prediction = True
    servo_tracking_delta_gain = 0.0

    # -- FD servo usage mode (paper-matched pipeline) --
    # "prediction_only":                delta_x_cmd = FD(valve_total)
    # "prediction_plus_feedback_delta": delta_x_cmd = FD(valve_total) + Kx * e * dt
    # "bypass_for_diagnostic_only":     delta_x_cmd = v_des * dt (DIAGNOSTIC ONLY)
    fd_servo_usage = "prediction_only"

    # -- FD correction gain (used when fd_servo_usage = "prediction_plus_feedback_delta") --
    servo_fd_correction_gain = 0.0

    # -- Servo step dt override (0 = use env step_dt) --
    servo_dt = 0.0

    # -- FD tracking servo gains (deployment mode: fd_tracking_servo / id_pid_fd_tracking_servo) --
    fd_gain = 1.0  # multiplier on original FD delta output
    tracking_gain = 0.0  # velocity error -> delta correction factor
    tracking_delta_clip_m = 0.01  # clip on tracking delta correction
    tracking_kp = 40.0  # valve-space P gain for velocity tracking PID
    tracking_ki = 15.0  # valve-space I gain
    tracking_kd = 2.0  # valve-space D gain
    tracking_integral_limit = 0.5  # anti-windup on valve integral

    # -- Position-ramp PI gains (used when low_level_servo_mode = "position_ramp_pi") --
    pos_pi_kp = [0.5, 0.5, 0.5, 0.5]
    pos_pi_ki = [0.2, 0.2, 0.2, 0.2]
    pos_pi_kd = [0.0, 0.0, 0.0, 0.0]
    pos_pi_integral_limit_m = 0.01
    pos_pi_delta_clip_m = 0.005

    # -- Scripted vehicle motion --
    enable_scripted_vehicle_motion = True
    vehicle_speed_min_kmh = 3.0  # moving (teacher-student range) (was OBSTACLE_DRIVING_SPEED_RANGE_KMH[0])
    vehicle_speed_max_kmh = 8.0  # moving (teacher-student range) (was OBSTACLE_DRIVING_SPEED_RANGE_KMH[1])
    vehicle_wheel_joint_names = tuple(OBSTACLE_WHEEL_JOINT_NAMES)
    vehicle_wheel_radius_m = OBSTACLE_WHEEL_RADIUS_M
    vehicle_wheel_speed_gain = 1.0
    vehicle_speed_update_interval_s = OBSTACLE_DRIVING_SPEED_UPDATE_INTERVAL_S
    vehicle_speed_resample_step_threshold = OBSTACLE_DRIVING_SPEED_RESAMPLE_STEP_THRESHOLD

    # -- Task-space command ranges (in base frame, reachable workspace) --
    command_pos_x_range = (4.0, 6.0)  # forward reach in base frame
    command_pos_y_range = (-1.5, 1.5)  # lateral reach in base frame
    command_pos_z = 0.3  # low to ground in base frame
    command_yaw_rad = 0.0     # match default initial cutter yaw (0 rad = aligned with vehicle forward)
    command_resample_time_s_range = (10.0, 12.0)

    # -- Horizon PPO reset/hold and workspace guards --
    horizon_pin_root_pose = False
    horizon_reset_settle_steps = 0
    horizon_zero_action_deadband = 1.0e-5
    horizon_delta_cmd_clip_m = 0.02
    horizon_cylinder_limit_margin = 0.002
    enable_horizon_workspace_guard = True
    enable_horizon_cab_guard = True
    horizon_workspace_x_range = (-0.5, 7.0)
    horizon_workspace_y_range = (-3.6, 3.0)
    horizon_workspace_z_range = (-0.4, 3.0)
    horizon_cab_x_range = (-2.5, 1.0)
    horizon_cab_y_range = (-1.35, 1.35)
    horizon_cab_z_range = (-0.1, 2.4)
    horizon_guard_penalty_weight = 0.5
    horizon_hydraulic_consistency_weight = 5.0
    horizon_failure_termination_penalty = 20.0
    horizon_success_bonus = 20.0
    horizon_success_tolerance_m = 0.05
    horizon_success_required_hold_steps = 10
    horizon_min_success_steps = 30
    # -- Stage-0 local reach curriculum --
    ppo_stage = "local_reach_stage0"  # "local_reach_stage0" or "full_mowing"
    stage0_level = 0  # 0, 1, or 2 (curriculum progression)
    # --- Level 0: easiest ---
    stage0_l0_target_box_dx = (-0.05, 0.05)
    stage0_l0_target_box_dy = (-0.05, 0.05)
    stage0_l0_target_box_dz = (-0.02, 0.02)
    stage0_l0_success_tolerance_m = 0.10
    stage0_l0_required_hold_steps = 3
    stage0_l0_min_episode_steps = 10
    # --- Level 1: medium ---
    stage0_l1_target_box_dx = (-0.10, 0.10)
    stage0_l1_target_box_dy = (-0.10, 0.10)
    stage0_l1_target_box_dz = (-0.03, 0.03)
    stage0_l1_success_tolerance_m = 0.08
    stage0_l1_required_hold_steps = 5
    stage0_l1_min_episode_steps = 20
    # --- Level 2: hardest (original) ---
    stage0_l2_target_box_dx = (-0.15, 0.15)
    stage0_l2_target_box_dy = (-0.15, 0.15)
    stage0_l2_target_box_dz = (-0.05, 0.05)
    stage0_l2_success_tolerance_m = 0.08
    stage0_l2_required_hold_steps = 5
    stage0_l2_min_episode_steps = 20
    # --- Shared Stage-0 params ---
    stage0_max_episode_steps = 100
    # -- Workspace violation: soft penalty for Stage 0 --
    workspace_violation_soft = True   # penalty, not immediate termination
    workspace_violation_penalty = -5.0  # per-horizon penalty when violating
    workspace_violation_done_after_consecutive_steps = 20  # increased from 10
    severe_cab_collision_done = True
    nonfinite_done = True
    # -- Guard/clipping penalty weights --
    horizon_guard_penalty_weight = 5.0   # increased from 0.5
    horizon_clip_penalty_weight = 3.0    # new
    # -- Reward config --
    horizon_success_bonus = 5.0   # reduced from 20.0
    horizon_failure_termination_penalty = 20.0
    # -- EE task-space action mode --
    ee_action_mode = "task_space_delta"  # "task_space_delta" or "cylinder_space"
    ee_action_scale_m = 0.05  # max EE displacement per horizon step
    ee_action_dim = 3  # dx, dy, dz in base frame
    ee_jacobian_eps = 0.001  # perturbation for J_eff measurement
    ee_k_p = 0.15  # proportional gain for damped LS
    ee_lam = 1e-2  # damping for damped LS
    ee_per_cyl_clip_m = 0.0015  # per-cylinder delta clip
    ee_global_clip_m = 0.003  # global norm clip
    ee_smooth_alpha = 0.7  # EMA smoothing factor
    # -- Target sampling for high-level EE reach --
    target_offset_radius_min_m = 0.06
    target_offset_radius_max_m = 0.12
    target_offset_z_range_m = (-0.02, 0.02)

    # -- Ground trajectory debug/training targets --
    trajectory_mode = "none"  # "none", "ground_semicircle", "ground_straight_line", "ground_mower_sweep"
    trajectory_plane = "ground_xy"  # "ground_xy" or "base_yz"
    ground_workspace_json = "outputs/ppo1_horizon_idfd/ground_workspace.json"
    ground_z = 0.0
    tool_contact_mode = "geometry_aabb"  # "geometry_aabb", "manual_offset", or "ee_origin"
    tool_ground_clearance = 0.03
    min_arm_ground_clearance = 0.02
    ee_frame_clearance_when_tool_offset_unknown = 0.05
    tool_tip_local_offset = (0.0, 0.0, 0.0)
    tool_tip_local_offset_x = 0.0
    tool_tip_local_offset_y = 0.0
    tool_tip_local_offset_z = 0.0
    use_tool_tip_offset = False
    trajectory_scale = 0.8
    visualize_target_trajectory = False
    visualize_actual_ee_trail = False
    debug_draw_tool_aabb = False
    debug_draw_tool_contact_point = False
    ground_trajectory_points = 200
    ground_actual_trail_length = 100
    # PPO trajectory training: episode timeout in wrapper steps (0 = no timeout,
    # legacy diagnostic behavior) and whether the shared path index wraps around
    # instead of clamping at the final point.
    ground_max_episode_steps = 0
    ground_path_loop = False
    ground_path_pacing = "fixed"  # "fixed" or "capture"
    ground_path_capture_radius_m = 0.20
    ground_path_capture_required_hold_steps = 5
    # "path_start": legacy path[0].
    # "nearest_path": begin at the path point closest to the current tool/contact point.
    # "local_offset": begin with a small local waypoint from the current tool/contact
    # point toward the nearest path point, then advance onto the full path.
    ground_initial_target_mode = "path_start"
    ground_initial_target_offset_m = 0.08
    ground_initial_target_capture_radius_m = 0.03
    # -- Mowing terrain shelf (embankment) --
    # OBSOLETED 2026-07-09 by grounding the truck (spawn 1.11 -> 0.05): with the
    # wheels on the ground the tool reaches ~ground level, so the REAL ground plane
    # (z=0) is the mowing surface and no shelf is needed. Kept for the raised-bank
    # scenario; default OFF. If enabled, place it at the tool's reachable band.
    mowing_terrain_enable = False
    mowing_terrain_z = 0.05                      # world z of the shelf top
    mowing_terrain_local_x = -19.528            # arm-reach center, relative to env-0 origin
    mowing_terrain_local_y = 1.38
    mowing_terrain_half_x = 0.7
    mowing_terrain_half_y = 0.9
    mowing_terrain_thickness = 0.04
    mowing_terrain_color = (0.20, 0.45, 0.18)
    mowing_terrain_collision = False
    ground_anchor_path_to_current_tool = False
    ground_anchor_margin_m = 0.05
    ground_anchor_max_span_m = 0.30
    ground_anchor_static_sag_m = 0.03
    # Measure/control the tool in the LIVE base frame (False). The frozen root-hold
    # frame variant (True) corrupted the empirical J_eff measurement -> wrong-signed
    # DLS directions -> 7-13x clip saturation -> the "vibrating arm" symptom.
    # axis_probe A/B 2026-07-09: frame ON = +Y FAIL (cos -0.75), violent 0.5-1.1 m
    # responses; frame OFF = ALL axes PASS (+Y cos +1.00, symmetric +-0.15 m).
    # De-rocking is fine for plots/diagnostics but must never sit inside the
    # measurement/control loop.
    ground_use_root_hold_frame = False
    ground_xy_enable_z_hold = True
    ground_xy_condition_z_first = True
    ground_xy_condition_path_to_j_eff = True
    ground_xy_condition_path_perp_scale = 0.15
    ground_xy_condition_path_min_sv = 1.0e-5
    ground_xy_conditioned_xy_clip_m = 0.003
    ground_xy_z_hold_gain = 0.35
    ground_xy_z_hold_clip_m = 0.012
    ground_xy_z_hold_deadband_m = 0.015
    ground_xy_z_hold_priority_threshold_m = 0.04

    # -- Low-level actuation backend --
    # "implicit_position": PhysX implicit drives close the cylinder position loop
    # (unconditionally stable at the configured stiffness); hydraulic realism comes
    # from the ID/FD models upstream, force saturation from effort_limit_sim.
    # "explicit_effort_pid_bias": hand-written explicit force PID + calibrated bias.
    # Measured 2026-07-07: dynamically unstable on this plant (limit cycles, ~0.35-1.7 m
    # tool drift, cylinder-limit hits within 3 s across a 22-trial gain sweep) — keep
    # only for effort-level experiments, not for training.
    lowlevel_actuator_backend = "implicit_position"
    explicit_pd_kp_drehzapfen = 1.5e7
    explicit_pd_kp_ausleger_i = 6.0e6
    explicit_pd_kp_ausleger_ii = 1.0e6
    explicit_pd_kp_messerkopf = 1.0e6
    explicit_pd_kd_drehzapfen = 1.0e6
    explicit_pd_kd_ausleger_i = 6.0e5
    explicit_pd_kd_ausleger_ii = 5.0e5
    explicit_pd_kd_messerkopf = 5.0e5
    explicit_pd_effort_limit_drehzapfen = 1.0e8
    explicit_pd_effort_limit_ausleger_i = 1.0e7
    explicit_pd_effort_limit_ausleger_ii = 1.0e6
    explicit_pd_effort_limit_messerkopf = 1.0e5
    explicit_pd_gain_scale = 1.0
    explicit_pd_damping_scale = 1.0
    explicit_pd_effort_scale = 1.0

    # -- PID integral gains (for explicit_effort_pid backend) --
    explicit_pid_ki_drehzapfen = 1.0e6
    explicit_pid_ki_ausleger_i = 5.0e5
    explicit_pid_ki_ausleger_ii = 1.0e5
    explicit_pid_ki_messerkopf = 1.0e5
    explicit_pid_ki_scale = 1.0
    explicit_pid_integral_limit_m = 0.01  # max integral windup per cylinder (meters)
    # -- Bias calibration (for explicit_effort_pid_bias backend) --
    bias_calibration_ki_scale = 3.0  # ki during calibration phase
    bias_runtime_ki_scale = 0.0  # ki during runtime (0 = frozen bias)
    bias_calibration_time_s = 2.0  # PID calibration duration
    bias_average_window_s = 0.5  # force_cmd averaging window
    bias_calibration_requires_static = True  # only capture bias when arm is settled
    bias_static_velocity_threshold_mps = 0.02  # max velocity for static check
    bias_static_error_threshold_m = 0.02  # max error for static check
    bias_calibration_failed_behavior = "extend"  # "extend" or "retry" or "use_anyway"
    bias_runtime_retarget_to_calibrated_pose = True  # hold calibrated equilibrium

    # -- Debug --
    debug_bias_shapes: bool = False  # print bias tensor shapes at startup and first update

    # -- Online bias adaptation --
    bias_online_update_rate = 0.005  # slow online adaptation
    bias_online_error_threshold_m = 0.001
    bias_online_velocity_threshold_mps = 0.01

    # -- Force rate limiting --
    explicit_force_rate_limit_fraction = 0.01  # per-step max force delta / effort_limit

    # -- Velocity filtering for D term --
    explicit_pd_velocity_filter_beta = 0.8  # EMA filter for qdot

    # -- Per-cylinder gain overrides (1.0 = use global scale) --
    explicit_pid_kp_per_cylinder = [1.0, 1.0, 1.0, 1.0]
    explicit_pid_kd_per_cylinder = [1.0, 1.0, 1.0, 1.0]
    explicit_force_rate_limit_per_cylinder = [1.0, 1.0, 1.0, 1.0]

    # -- Global scales --
    explicit_pid_kp_scale = 1.0
    explicit_pid_kd_scale = 1.0

    # -- Terrain / physics --
    enable_mathbox_ground: bool = True
    ground_color = (0.0, 0.0, 0.0)
    lane_line_color = (1.0, 1.0, 1.0)
    lane_line_width_m = 0.08
    # Optional visual-only road/verge dressing for the arm-workstation tasks.
    # The physical ground remains the same collision surface; these layers are
    # merged meshes placed a few millimetres above it for clear visualization.
    enable_visual_road_verge: bool = False
    visual_road_start_x_m = -38.0
    visual_road_length_m = 36.0
    visual_road_center_y_m = 4.30
    visual_road_width_m = 3.60
    visual_road_color = (0.11, 0.12, 0.13)
    visual_road_marking_color = (0.90, 0.90, 0.88)
    visual_verge_start_y_m = -3.20
    visual_verge_end_y_m = 2.48
    visual_verge_color = (0.08, 0.28, 0.055)
    visual_verge_secondary_color = (0.12, 0.36, 0.075)
    disable_heavy_terrain = True
    terrain_static_friction = 1.0
    terrain_dynamic_friction = 1.0
    terrain_restitution = 0.0

    # -- Reward weights (simplified — core mowing only, no collision avoidance) --
    rew_position_tracking_linear = -12.0
    rew_position_tracking_tanh = 30.0
    rew_tcp_ahead = 2.0
    rew_yaw_tracking = -15.0
    rew_finish_progress = 10.0
    rew_action_rate = -0.06
    rew_joint_vel = -0.7

    # -- Reward params --
    position_tracking_offset = 0.4
    position_tracking_std = 0.4
    yaw_tracking_offset_deg = 30.0    # tolerance before yaw penalty starts
    tcp_ahead_threshold = -0.1
    finish_traj_length_x = 40.0
    finish_bonus_weight = 2.0

    # -- Termination --
    joint_limit_margin = 0.15  # relaxed from 0.05 for debugging
    unimog_flying_roll_pitch_threshold = 1.0  # relaxed from ~28 deg to ~57 deg
    unimog_lateral_deviation_threshold = 8.0  # relaxed from 4m to 8m
    tcp_face_backwards_yaw_threshold = 3.14   # pi — only terminate if fully backwards
    tcp_above_ground_height_threshold = 3.0   # relaxed from 1.8m to 3m

    # -- Stubs for train script compatibility (all unused in upper-level mode) --
    control_mode = "joint_space_validation"
    low_level_control_mode = "position_delta"
    reference_mode = "joint_space_validation"
    position_delta_source = "direct_action"
    use_sd_observer = False
    use_sd_feedback = False
    sd_feedback_beta = 0.01
    sd_dP_rate_limit = 100000.0
    sd_fnet_rate_limit = 10000.0
    use_reduced_hydraulic_rom = False
    use_final_causal_cylinder_rom = False
    final_causal_rom_export_dir = ""
    rom_output_mode = "velocity_to_position_target"
    rom_use_sd_feedback = False
    # Command ROM is not valve-commanded. It receives desired cylinder
    # extension/velocity and predicts the internal valve itself. ID-FD has a
    # separate valve interface in PPO_ID_FD/env_cfg.py.
    rom_use_id_valve_cmd = False
    rom_dt = 0.01
    rom_xdot_max = [0.8, 0.8, 0.8, 0.8]
    rom_dx_max = [0.0005, 0.0005, 0.0005, 0.0005]
    use_fd_effort = False
    use_fd_residual_alpha = False
    fd_residual_alpha = 0.002
    max_abs_torque = 5.0
    torque_ramp_steps = 1000
    torque_safety_scale = 1.0
    torque_adapter_mode = "scale_bias"
    torque_adapter_preset = "conservative"
    torque_adapter_scale = [0.0001, 0.001, 0.0005, 0.0001]
    torque_adapter_bias = [0.0, 0.0, 0.0, 0.0]
    action_to_qddot_scale = 0.5
    qddot_cmd_clip = 5.0
    use_q_ref_ramp = False
    q_ref_ramp_time_s = 2.0
    q_ref_ramp_profile = "smoothstep"
    use_q_ref_curriculum = False
    q_ref_target_scale_initial = 0.25
    q_ref_target_scale_final = 1.0
    q_ref_curriculum_steps = 20000
    enable_goal_difficulty_curriculum = False
    goal_curriculum_stage_scales = [0.1, 0.25, 0.6, 1.0]
    task_space_observation_space = OBS_DIM
    use_joint5_coupling = False
    joint5_coupling_source_joint = 3
    joint5_coupling_target_joint = 4
    joint5_coupling_gain = 1.0
    joint5_coupling_bias = 0.0
    q_tracking_reward_joint_weights = [1.0, 1.0, 1.0, 1.0, 0.0]
    q_tracking_debug_joint_weights = [1.0, 1.0, 1.0, 1.0, 0.0]
    q_tracking_joint_weights = [1.0, 1.0, 1.0, 1.0, 0.0]
    kevin_fixed_pose_contract_version = ""
    reward_contract_version = ""
    rew_q_tracking = 0.0
    rew_qdot = 0.0
    rew_alive = 0.0
    rew_termination = 0.0
    rew_fd_load = 0.0
    rew_progress = 0.0
    rew_action_saturation = 0.0
    action_saturation_threshold = 0.9
    rew_action_magnitude = 0.0
    progress_ema_alpha = 0.1
    use_settle_tracking_reward = False
    settle_ramp_progress_threshold = 0.95
    rew_settle_tracking = 0.0
    use_joint1_settle_tracking_reward = False
    rew_joint1_settle_tracking = 0.0
    joint1_settle_error_deadband = 0.005
    log_full_pipeline_debug = False
    debug_reward_terms = False
    verbose_reset_logging = False

    def __post_init__(self):
        apply_kevin_sim_memory_optimizations(self.sim)

        # Override prim_path and add PD gains (local to this task, does not modify MULAG_CFG)
        self.robot_cfg = self.robot_cfg.replace(prim_path="/World/envs/env_.*/Robot")
        from isaaclab.actuators import ImplicitActuatorCfg
        # Implicit drive stiffness models hydraulic lock (closed valve ~= rigid oil
        # column). Stiffness values from the 2026-07-07 drive-stiffness bracket
        # (outputs/ppo1_horizon_idfd/drive_stiffness_bracket_results.json): the x5
        # ladder minimizes hold drift; softer sags under gravity, stiffer excites
        # the solver. Effort limits and velocity limits follow the teacher-student
        # asset tuning (~1e5 effort scale, velocity_limit 1.0): with the earlier
        # 1e7-1e8 limits the arm's reaction forces dragged the free-standing truck
        # ~0.5 m during greedy tracking ("mulag starts flying" in teacher-student
        # notes). The truck must dominate because the arm is force-capped at
        # hydraulic scale, not because the base is pinned. Measured masses: truck
        # 6.8 t, arm 542 kg -- effort caps sized so the arm cannot beat the
        # truck's friction budget (mu*m*g ~ 7e4 N lateral, ~1e5 N*m yaw).
        self.robot_cfg.actuators = {
            "drehzapfen_cylinder": ImplicitActuatorCfg(
                joint_names_expr=["Drehzapfen_joint"],
                stiffness=75000000.0,
                damping=2240000.0,
                effort_limit_sim=50000.0,
                velocity_limit_sim=1.0,
            ),
            "ausleger_i_cylinder": ImplicitActuatorCfg(
                joint_names_expr=["Ausleger_I_joint"],
                stiffness=30000000.0,
                damping=1340000.0,
                effort_limit_sim=100000.0,
                velocity_limit_sim=1.0,
            ),
            "ausleger_ii_cylinder": ImplicitActuatorCfg(
                joint_names_expr=["Ausleger_II_joint"],
                stiffness=5000000.0,
                damping=1120000.0,
                effort_limit_sim=50000.0,
                velocity_limit_sim=1.0,
            ),
            "messerkopf_schwenk_cylinder": ImplicitActuatorCfg(
                joint_names_expr=["Messerkopf_Schwenk_joint"],
                stiffness=5000000.0,
                damping=1120000.0,
                effort_limit_sim=20000.0,
                velocity_limit_sim=1.0,
            ),
            "messerkopf_passive_coupled": ImplicitActuatorCfg(
                joint_names_expr=["Messerkopf_joint"],
                stiffness=5000000.0,
                damping=1120000.0,
                effort_limit_sim=20000.0,
                velocity_limit_sim=1.0,
            ),
            # Parking-brake wheels (teacher-student values): the truck resists the
            # arm's reaction forces through braked wheels + tire friction. For the
            # scripted-driving stage these gains must be switched to velocity-drive
            # (stiffness 0, high damping) before wheel velocity targets can act.
            "mulag_wheels": ImplicitActuatorCfg(
                joint_names_expr=[
                    "Wheel_Front_Left_joint", "Wheel_Front_Right_joint",
                    "Wheel_Rear_Left_joint", "Wheel_Rear_Right_joint",
                ],
                stiffness=600000000.0, damping=2000000.0,
                effort_limit_sim=100000.0,
            ),
            "mulag_steering": ImplicitActuatorCfg(
                joint_names_expr=[
                    "Wheel_Front_Left_Steering_joint", "Wheel_Front_Right_Steering_joint",
                ],
                stiffness=10.0, damping=1000000000.0,
                effort_limit_sim=100000.0,
            ),
        }
        # The USD is authored with its LOWEST geometry (wheels) at root z=0, and the
        # articulation is held at spawn height (it does not fall to ground contact).
        # A spawn-height sweep (2026-07-09) showed lowest_geom_z == spawn_z exactly and
        # tool_bottom_z == spawn_z + 0.468, so the old 1.11 floated the whole truck
        # 1.1 m in the air and lifted the arm out of ground reach. Spawn just above 0
        # so the wheels sit on the ground and the tool reaches ~ground level.
        self.robot_cfg.init_state.pos = (-25.4, 4.3, 0.05)


@configclass
class MulagUpperLevelEnvCfg_PLAY(MulagUpperLevelEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1


@configclass
class MulagUpperLevelPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 2000
    save_interval = 100
    experiment_name = "mulag_upper_level"
    run_name = ""
    resume = False
    empirical_normalization = True
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.15),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 256, 128],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    def __post_init__(self):
        for model_cfg in (self.actor, self.critic):
            for attr_name in ("stochastic", "init_noise_std", "noise_std_type", "state_dependent_std"):
                if attr_name in getattr(model_cfg, "__dict__", {}):
                    delattr(model_cfg, attr_name)
