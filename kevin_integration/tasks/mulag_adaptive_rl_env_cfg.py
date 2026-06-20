from __future__ import annotations

from pathlib import Path

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from kevin_integration.tasks.rrlab_reference_conventions import (
    OBSTACLE_ARM_JOINT_NAMES,
    OBSTACLE_DRIVING_SPEED_RANGE_KMH,
    OBSTACLE_DRIVING_SPEED_RESAMPLE_STEP_THRESHOLD,
    OBSTACLE_DRIVING_SPEED_UPDATE_INTERVAL_S,
    OBSTACLE_HYDRAULIC_COMMAND_JOINT_NAMES,
    OBSTACLE_WHEEL_JOINT_NAMES,
    OBSTACLE_WHEEL_RADIUS_M,
)
from kevin_integration.tasks.task_space_reference import TASK_SPACE_OBSERVATION_DIM
from kevin_integration.utils.sim_memory import apply_kevin_sim_memory_optimizations
from rrlab_assets import MULAG_CFG


_MODELS_DIR = str(Path(__file__).resolve().parents[1] / "models")
_ROM_DIR = Path(__file__).resolve().parents[1] / "rom_cylinders"


@configclass
class MulagAdaptiveRLJointReachEnvCfg(DirectRLEnvCfg):
    decimation = 1
    episode_length_s = 10.0
    action_scale = 1.0
    action_space = 4
    observation_space = 52
    task_space_observation_space = TASK_SPACE_OBSERVATION_DIM
    state_space = 0
    # Legacy low-level values ("position_delta", "effort") are still accepted by
    # MulagAdaptiveRLEnv. The hydraulic ROM task below uses this field for its
    # end-to-end mode: "joint_space_validation" or "rl_task_space_tracking".
    control_mode = "position_delta"
    low_level_control_mode = "position_delta"
    position_delta_source = "id_valve"
    use_sd_observer = True
    use_sd_feedback = False
    sd_observer_log = True
    sd_feedback_beta = 0.01
    use_sd_feedback_blend = True
    use_sd_feedback_rate_limit = True
    use_fd_effort = False
    use_fd_model = True
    use_fd_as_observer = True
    use_fd_in_policy_observation = True
    use_fd_safety_limits = True
    use_fd_auxiliary_reward = True
    use_fd_effort_actuation = False
    fd_torque_soft_limit = "fd_output_std"
    fd_torque_hard_limit = "fd_output_3std"
    rew_fd_load = 0.01
    position_scale = 0.005
    action_to_qddot_scale = 0.5
    qddot_cmd_clip = 5.0
    torque_ramp_steps = 1000
    torque_safety_scale = 1.0
    log_full_pipeline_debug = True
    debug_reward_terms = True
    asset_preflight_settle_steps = 300
    asset_preflight_max_pulse_root_z_motion_m = 0.02
    asset_preflight_max_pulse_roll_pitch_motion_rad = 0.03490658503988659
    asset_preflight_max_pre_pulse_root_vertical_velocity_mps = 0.05

    sim: SimulationCfg = SimulationCfg(dt=1 / 100, render_interval=2)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=20.0, replicate_physics=False)
    robot_cfg: ArticulationCfg = MULAG_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    models_dir = _MODELS_DIR
    fallback_window = 10
    max_abs_valve = 1.0
    torque_adapter_mode = "scale_bias"
    torque_adapter_preset = "conservative"
    torque_adapter_scale = [0.0001, 0.001, 0.0005, 0.0001]
    torque_adapter_bias = [0.0, 0.0, 0.0, 0.0]
    fd_torque_scale = 1.0e6
    use_fd_residual_alpha = True
    fd_residual_alpha = 0.002
    max_abs_torque = 5.0
    torque_rate_limit = 1.0
    torque_lowpass_alpha = 0.2
    use_torque_rate_limit = True
    use_torque_lowpass = True
    max_abs_pressure_delta = 1.0e7
    max_abs_fnet = 1.0e7
    sd_dP_rate_limit = 100000.0
    sd_fnet_rate_limit = 10000.0
    sd_output_mode = "state"
    sd_nan_guard = True
    sd_clamp_outputs = True
    pressure_noise_std = 0.0

    use_reduced_hydraulic_rom = False
    use_final_causal_cylinder_rom = False
    final_causal_rom_export_dir = str(_ROM_DIR / "exported_valve_conditioned_causal_rom")
    final_causal_rom_checkpoint = str(_ROM_DIR / "exported_valve_conditioned_causal_rom" / "best_valve_conditioned_causal_rom.pth")
    final_causal_rom_scaler = str(_ROM_DIR / "exported_valve_conditioned_causal_rom" / "best_valve_conditioned_causal_rom_scaler.pth")
    final_causal_rom_metadata = str(_ROM_DIR / "exported_valve_conditioned_causal_rom" / "best_valve_conditioned_causal_rom_metadata.json")
    final_causal_rom_update_period = 0.05
    final_causal_rom_multirate_ratio = 5
    final_causal_rom_target_clamp = True
    final_causal_rom_debug = True
    final_causal_rom_extension_delta_clamp = 0.04
    final_causal_rom_applied_delta_clamp = 0.008
    final_causal_rom_pressure_source = "training_mean"
    cylinder_coordinate_mapping = str(_ROM_DIR / "cylinder_coordinate_mapping.json")
    cylinder_coordinate_mapping_mode = "nominal_sign_offset"
    allow_provisional_range_mapping = False
    allow_provisional_nominal_reference = False
    rom_output_mode = "velocity_to_position_target"
    rom_use_sd_feedback = False
    rom_use_id_valve_cmd = True
    rom_dt = 0.01
    rom_nan_guard = True
    rom_clamp_outputs = True
    rom_debug_log = True
    # Initial safe placeholders only. Identify these from Simscape/full-order data before training with ROM mode.
    rom_Abar = [1.0e-3, 1.0e-3, 1.0e-3, 1.0e-3]
    rom_Vt = [1.0e-3, 1.0e-3, 1.0e-3, 1.0e-3]
    rom_E = [1.0e8, 1.0e8, 1.0e8, 1.0e8]
    rom_Kq = [1.0e-8, 1.0e-8, 1.0e-8, 1.0e-8]
    rom_CL = [0.0, 0.0, 0.0, 0.0]
    rom_mu = [10.0, 10.0, 10.0, 10.0]
    rom_Fs = [10.0, 10.0, 10.0, 10.0]
    rom_kf = [100.0, 100.0, 100.0, 100.0]
    rom_Pmax = [1.0e7, 1.0e7, 1.0e7, 1.0e7]
    rom_xdot_max = [0.02, 0.02, 0.02, 0.02]
    rom_dx_max = [5.0e-4, 5.0e-4, 5.0e-4, 5.0e-4]
    rom_delay_steps = [0, 0, 0, 0]
    rom_udz_pos = [0.0, 0.0, 0.0, 0.0]
    rom_udz_neg = [0.0, 0.0, 0.0, 0.0]
    rom_velocity_gain = [0.02, 0.02, 0.02, 0.02]
    rom_force_scale = [1000.0, 1000.0, 1000.0, 1000.0]
    channel4_velocity_damping_gain = 0.0

    # ── Curriculum configuration (task-space tracking stages) ──────────────────────
    # The canonical fixed-final-pose experiment remains at Stage 1. The
    # curriculum implementation is retained for explicit future experiments.
    enable_automatic_curriculum = False
    enable_goal_difficulty_curriculum = False
    goal_curriculum_min_completed_episodes = 256
    goal_curriculum_advance_success_rate = 0.20
    goal_curriculum_advance_near_pose_low_speed_rate = 0.10
    goal_curriculum_advance_mean_max_hold_steps = 10.0
    goal_curriculum_hysteresis_windows = 2
    goal_curriculum_stage_scales = [0.25, 0.60, 1.0, 1.0]
    # Stage boundaries in cumulative env steps: S1: 0-50k, S2: 50k-150k,
    # S3: 150k-300k, S4: >300k.
    curriculum_stage_steps = [0, 50000, 150000, 300000]
    # Stage-aware episode durations (seconds).
    curriculum_episode_length_s = [15.0, 15.0, 30.0, 45.0]
    # Action-authority multiplier applied before qddot conversion.
    curriculum_action_authority = [0.25, 0.50, 0.75, 1.0]
    # Vehicle speed ranges per stage [min_kmh, max_kmh]. First N stages
    # (curriculum_vehicle_stationary_stages) force 0 km/h regardless.
    curriculum_vehicle_speed_min_kmh = [0.0, 0.0, 0.0, 3.0]
    curriculum_vehicle_speed_max_kmh = [0.0, 0.0, 0.0, 8.0]
    curriculum_vehicle_stationary_stages = 3  # S1, S2, S3 are stationary

    # Task-space command ranges per stage [[x_min, x_max], ...].
    # S1: tight centred on nominal rest pose; S2–S4 progressively expand.
    curriculum_command_x_range = [[6.8, 7.2], [6.6, 7.4], [6.5, 7.5], [6.5, 7.5]]
    curriculum_command_y_range = [[1.2, 1.8], [1.0, 2.0], [0.8, 2.2], [0.7, 2.2]]
    curriculum_command_z_range = [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]
    # Orientation reward enabled per stage.
    curriculum_orientation_enabled = [False, False, True, True]

    # ── Curriculum performance gates (for stage advancement) ───────────────────
    # Minimum completed episodes per environment before stage advancement.
    curriculum_min_completed_episodes_per_env = 5.0
    # Maximum mean position error (m) for stage advancement.
    curriculum_max_mean_position_error = 0.50
    # Maximum velocity-termination rate for stage advancement.
    curriculum_max_velocity_termination_rate = 0.10
    # Maximum other unsafe termination rate (lower_limit, upper_limit, nonfinite).
    curriculum_max_other_unsafe_rate = 0.01
    # Minimum timeout (successful completion) rate for stage advancement.
    curriculum_min_timeout_rate = 0.80
    # Minimum final-pose success rate for stage advancement (standard curriculum).
    # Prevents advancement from one rare isolated success.
    curriculum_advance_min_success_rate = 0.05
    # Unsafe terminal penalty (negative value applied on unsafe termination).
    # Set to -taskspace_unsafe_penalty_weight by default in MulagHydraulicROMTaskSpaceEnvCfg.
    unsafe_terminal_penalty = -5.0

    # Task-space reward weights (stage-independent base values).
    kevin_fixed_pose_contract_version = "kevin_fixed_pose_task_space_v2"
    reward_contract_version = "task_space_reward_balance_v2"
    rew_task_pos_tracking = 1.0       # bounded [0,1] exponential position tracking reward
    position_tracking_reward_weight = 1.0
    rew_task_orient_tracking = 0.25   # dense quaternion-geodesic orientation tracking
    rew_task_progress = 0.5           # potential progress shaping scale
    rew_task_progress_clip = 0.2      # max absolute progress-reward contribution
    taskspace_progress_position_scale_m = 1.0
    taskspace_progress_orientation_scale_rad = 3.141592653589793
    taskspace_progress_position_weight = 1.0
    taskspace_progress_orientation_weight = 0.25
    taskspace_progress_gamma = 0.99
    rew_final_pose_linear_velocity = 0.15
    rew_final_pose_angular_velocity = 0.08
    rew_final_pose_settling = 0.20
    rew_near_target_linear_settling = 0.35
    rew_near_target_angular_settling = 0.35
    rew_final_pose_hold_progress = 1.00
    rew_clamp_outward = 0.25
    rom_clamp_documented_increment_scale = 0.008
    valve_saturation_start = 0.90
    rew_valve_saturation_soft = 0.15
    fd_load_soft_threshold = 1.0
    final_pose_position_tolerance_m = 0.05
    final_pose_orientation_tolerance_rad = 0.0872664626
    final_pose_linear_speed_tolerance_mps = 0.05
    final_pose_angular_speed_tolerance_radps = 0.10
    final_pose_success_hold_steps = 50
    final_pose_success_bonus_value = 10.0

    mechanical_joint_names = list(OBSTACLE_ARM_JOINT_NAMES)
    mechanical_joint_position_units = ["rad", "rad", "rad", "rad", "rad"]
    mechanical_joint_velocity_units = ["rad/s", "rad/s", "rad/s", "rad/s", "rad/s"]
    mechanical_joint_types = ["revolute", "revolute", "revolute", "revolute", "revolute_follower"]
    cylinder_joint_names = list(OBSTACLE_HYDRAULIC_COMMAND_JOINT_NAMES)
    cylinder_joint_position_units = ["m", "m", "m", "m"]
    cylinder_joint_velocity_units = ["m/s", "m/s", "m/s", "m/s"]
    # Backward-compatible field names used by older tests/tools.
    state_joint_names = list(OBSTACLE_ARM_JOINT_NAMES)
    command_joint_names = list(OBSTACLE_HYDRAULIC_COMMAND_JOINT_NAMES)

    # Mirrors RRLAB-Obstacle-Avoidance-Mulag-v0: vehicle motion is scripted by
    # wheel velocity targets, separate from PPO/ID/ROM arm control.
    enable_scripted_vehicle_motion = True
    vehicle_speed_min_kmh = OBSTACLE_DRIVING_SPEED_RANGE_KMH[0]
    vehicle_speed_max_kmh = OBSTACLE_DRIVING_SPEED_RANGE_KMH[1]
    vehicle_command_mode = "rrlab_obstacle_scripted_wheel_velocity"
    vehicle_wheel_joint_names = list(OBSTACLE_WHEEL_JOINT_NAMES)
    vehicle_speed_range_kmh = list(OBSTACLE_DRIVING_SPEED_RANGE_KMH)
    vehicle_wheel_radius_m = OBSTACLE_WHEEL_RADIUS_M
    vehicle_speed_update_interval_s = OBSTACLE_DRIVING_SPEED_UPDATE_INTERVAL_S
    vehicle_speed_resample_step_threshold = OBSTACLE_DRIVING_SPEED_RESAMPLE_STEP_THRESHOLD
    reference_mode = "joint_space_validation"
    task_space_ee_body_name = "Messerkopf"
    target_joint_range = [0.35, 0.35, 0.35, 0.30, 0.25]
    use_q_ref_ramp = True
    q_ref_ramp_time_s = 2.0
    q_ref_ramp_profile = "smoothstep"
    use_q_ref_curriculum = True
    q_ref_target_scale_initial = 0.25
    q_ref_target_scale_final = 1.0
    q_ref_curriculum_steps = 20000
    q_ref_target_amplitude = [0.08, 0.08, 0.08, 0.05, 0.0]
    q_ref_sample_relative_to_current = True
    use_joint5_coupling = True
    joint5_coupling_source_joint = 3
    joint5_coupling_target_joint = 4
    joint5_coupling_gain = 1.0
    joint5_coupling_bias = 0.0
    q_tracking_reward_joint_weights = [1.5, 1.0, 1.0, 1.0, 0.0]
    q_tracking_debug_joint_weights = [1.0, 1.0, 1.0, 1.0, 0.0]
    q_tracking_joint_weights = [1.0, 1.0, 1.0, 1.0, 0.0]
    target_change_threshold = 1.0e-5
    q_ref_jump_warning_threshold = 0.02

    reference_marker_debug_vis = False
    reference_marker_base_offset = [4.0, 1.5, 2.0]
    reference_marker_joint_scale = [4.0, 4.0, 4.0]

    scripted_reference_amplitude = [0.08, 0.075, 0.07, 0.045, 0.0]
    scripted_reference_frequency_hz = [0.105, 0.091, 0.112, 0.137, 0.137]
    scripted_reference_phase_rad = [0.0, 1.1, 2.2, 0.6, 0.6]
    scripted_reference_kp = [30.0, 22.0, 85.0, 160.0]
    scripted_reference_kd = [8.0, 7.0, 18.0, 24.0]
    scripted_reference_action_clip = 0.85
    scripted_reference_reset_phase_randomization = True
    scripted_demo_settle_time_s = 1.0
    scripted_demo_control_ramp_time_s = 2.0
    scripted_demo_disable_safety_termination = True
    verbose_reset_logging = False
    reset_diagnostic_log_limit = 8
    print_reset_telemetry = False

    joint_limit_margin = 0.02
    # Historical diagnostic threshold only. It is not a validated MulaG
    # hydraulic safety limit and must not be used for hard termination.
    unvalidated_velocity_diagnostic_threshold = 6.0
    enable_validated_hydraulic_velocity_termination = False
    hydraulic_soft_velocity_limits_mps = [0.0, 0.0, 0.0, 0.0]
    hydraulic_hard_velocity_limits_mps = [0.0, 0.0, 0.0, 0.0]
    hydraulic_velocity_persistence_steps = 3
    hydraulic_velocity_limits_source = "blocked_missing_pytorch_hydraulic_rom_dataset_causal_mat"
    enable_validated_mechanical_velocity_termination = False
    mechanical_soft_velocity_limits = [0.0, 0.0, 0.0, 0.0, 0.0]
    mechanical_hard_velocity_limits = [0.0, 0.0, 0.0, 0.0, 0.0]
    mechanical_velocity_limits_source = "working_collision_task_has_no_arm_hard_velocity_termination"
    final_pose_linear_shaping_scale_mps = 0.05
    final_pose_angular_shaping_scale_radps = 0.10
    final_pose_linear_settling_scale_mps = 0.05
    final_pose_angular_settling_scale_radps = 0.10
    near_target_linear_settling_position_gate_scale = 1.5
    near_target_linear_settling_orientation_gate_scale = 1.5
    near_target_linear_settling_speed_margin_mps = 0.03
    near_target_angular_settling_position_gate_scale = 1.5
    near_target_angular_settling_orientation_gate_scale = 1.5
    near_target_angular_settling_speed_margin_radps = 0.05

    rew_q_tracking = 3.0
    rew_qdot = 0.25
    rew_progress = 2.0
    progress_ema_alpha = 0.1
    use_settle_tracking_reward = True
    settle_ramp_progress_threshold = 0.95
    rew_settle_tracking = 2.0
    use_joint1_settle_tracking_reward = True
    rew_joint1_settle_tracking = 3.0
    joint1_settle_error_deadband = 0.005
    rew_action_smoothness = 0.0
    rew_action_rate = 0.05
    rew_action_magnitude = 0.02
    rew_action_saturation = 0.5
    action_saturation_threshold = 0.90
    rew_valve_effort = 0.01
    rew_torque_effort = 0.0
    rew_alive = 0.1
    rew_termination = 5.0

    def __post_init__(self):
        self.robot_cfg = MULAG_CFG.replace(prim_path="/World/envs/env_.*/Robot")
        self.robot_cfg.init_state.pos = (-25.4, 4.3, 0.0)
        apply_kevin_sim_memory_optimizations(self.sim)


@configclass
class MulagAdaptiveRLJointReachEnvCfg_PLAY(MulagAdaptiveRLJointReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1


@configclass
class MulagAdaptiveRLPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1000
    save_interval = 50
    experiment_name = "mulag_adaptive_rl_joint_reach"
    run_name = ""
    resume = False
    empirical_normalization = True
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[128, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.5),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[128, 128],
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


@configclass
class MulagHydraulicROMJointTrackingEnvCfg(MulagAdaptiveRLJointReachEnvCfg):
    control_mode = "joint_space_validation"
    low_level_control_mode = "position_delta"
    position_delta_source = "id_valve"
    use_reduced_hydraulic_rom = False
    use_final_causal_cylinder_rom = True
    use_sd_observer = False
    use_sd_feedback = False
    rom_use_id_valve_cmd = True
    rom_use_sd_feedback = False
    rom_dt = 0.01
    episode_length_s = 300.0
    action_to_qddot_scale = "id_qddot_std"
    qddot_cmd_clip = "id_qddot_std"
    q_ref_target_amplitude = [0.10, 0.09, 0.09, 0.055, 0.0]
    q_ref_target_scale_initial = 0.5
    q_ref_target_scale_final = 1.0
    rew_q_tracking = 4.0
    rew_progress = 1.0
    rew_action_rate = 0.02
    rew_valve_effort = 0.02
    rew_torque_effort = 0.0
    # The current mapping JSON is nominal/sign-verified but not second-pose
    # validated. Keeping this as an explicit serialized task default avoids a
    # diagnostic CLI flag while making the remaining validation debt visible.
    allow_provisional_nominal_reference = True


@configclass
class MulagHydraulicROMJointTrackingEnvCfg_PLAY(MulagHydraulicROMJointTrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1


@configclass
class MulagHydraulicROMJointTrackingPPORunnerCfg(MulagAdaptiveRLPPORunnerCfg):
    experiment_name = "kevin_hydraulic_rom_joint_tracking"
    max_iterations = 1000


@configclass
class MulagHydraulicROMTaskSpaceEnvCfg(MulagAdaptiveRLJointReachEnvCfg):
    control_mode = "rl_task_space_tracking"
    low_level_control_mode = "position_delta"
    position_delta_source = "id_valve"
    use_reduced_hydraulic_rom = False
    use_final_causal_cylinder_rom = True
    use_sd_observer = False
    use_sd_feedback = False
    rom_use_id_valve_cmd = True
    rom_use_sd_feedback = False
    rom_dt = 0.01
    # Base episode length = S4 maximum (45 s). Stage-aware override in
    # _get_dones uses curriculum_episode_length_s for S1–S3.
    episode_length_s = 45.0
    action_to_qddot_scale = "id_qddot_std"
    qddot_cmd_clip = "id_qddot_std"
    reference_mode = "rl_task_space_tracking"
    enable_scripted_vehicle_motion = True
    observation_space = TASK_SPACE_OBSERVATION_DIM
    allow_provisional_nominal_reference = True
    # Joint-space reward terms are unused in task-space mode.
    rew_q_tracking = 0.0
    rew_joint1_settle_tracking = 0.0
    use_joint1_settle_tracking_reward = False
    use_settle_tracking_reward = False
    rew_torque_effort = 0.0
    # Task-space curriculum defaults (can be overridden per run).
    enable_automatic_curriculum = False
    curriculum_stage_steps = [0, 50000, 150000, 300000]
    curriculum_episode_length_s = [15.0, 15.0, 30.0, 45.0]
    curriculum_action_authority = [0.25, 0.50, 0.75, 1.0]
    curriculum_vehicle_speed_min_kmh = [0.0, 0.0, 0.0, 3.0]
    curriculum_vehicle_speed_max_kmh = [0.0, 0.0, 0.0, 8.0]
    curriculum_vehicle_stationary_stages = 3
    curriculum_command_x_range = [[6.8, 7.2], [6.6, 7.4], [6.5, 7.5], [6.5, 7.5]]
    curriculum_command_y_range = [[1.2, 1.8], [1.0, 2.0], [0.8, 2.2], [0.7, 2.2]]
    curriculum_command_z_range = [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]
    curriculum_orientation_enabled = [False, False, True, True]
    rew_task_pos_tracking = 1.0
    position_tracking_reward_weight = 1.0
    rew_task_orient_tracking = 0.25
    rew_task_progress = 0.5
    rew_task_progress_clip = 0.2
    # FD load penalty weight (auxiliary load from observer-only FD).
    rew_fd_load = 0.01
    # Unsafe termination penalty weight (5.0, NOT hardcoded 100).
    taskspace_unsafe_penalty_weight = 5.0


@configclass
class MulagHydraulicROMTaskSpaceEnvCfg_PLAY(MulagHydraulicROMTaskSpaceEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1


@configclass
class MulagHydraulicROMTaskSpacePPORunnerCfg(MulagAdaptiveRLPPORunnerCfg):
    experiment_name = "kevin_hydraulic_rom_task_space"
    max_iterations = 1000
    num_steps_per_env = 128
    enable_action_std_hold_schedule = True
    action_std_schedule_start_hold_fraction = 0.50
    action_std_schedule_end_hold_fraction = 0.98
    action_std_schedule_min_std = 0.12
    action_std_schedule_max_std_initial = 0.55
    action_std_schedule_max_std_final = 0.25
    action_std_schedule_entropy_final = 0.001
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.03,
        max_grad_norm=1.0,
    )
    actor = RslRlMLPModelCfg(
        hidden_dims=[128, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.20),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[128, 128],
        activation="elu",
        obs_normalization=True,
    )
