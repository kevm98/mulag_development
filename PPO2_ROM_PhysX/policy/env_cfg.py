"""Configuration for the PPO Policy environment.

Extends the base PPO2ROMPhysXEnvCfg with Policy-specific settings
and provides a dedicated PPO runner configuration.
"""

from __future__ import annotations

from pathlib import Path

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlPpoAlgorithmCfg

from PPO2_ROM_PhysX.tasks.env_cfg import PPO2ROMPhysXEnvCfg
from PPO2_ROM_PhysX.runtime.mulag_upper_level_env_cfg import (
    MulagUpperLevelEnvCfg_PLAY,
    MulagUpperLevelPPORunnerCfg,
)
from PPO2_ROM_PhysX.policy.observations import RESIDUAL_OBS_DIM


def apply_residual_curriculum_stage(cfg, stage: int) -> None:
    """Apply the trajectory speed and residual authority for a saved policy stage."""
    stage = int(stage)
    if stage == 1:
        cfg.path_speed_mps = 0.10
        cfg.residual_scale_m = 0.001
    elif stage == 2:
        cfg.path_speed_mps = 0.13
        cfg.residual_scale_m = 0.002
    elif stage == 3:
        cfg.path_speed_mps = 0.13
        cfg.residual_scale_m = 0.004
    elif stage == 4:
        # Canonical plant-reachable PPO profile.  Keep residual authority small:
        # the deterministic ROM baseline is already within 2-3 cm and larger
        # corrections were the source of the old oscillatory policy.
        cfg.path_speed_mps = 0.18
        cfg.residual_scale_m = 0.002
        cfg.trajectory_shape = "line"
        cfg.randomize_trajectory_shape = True
        cfg.seed_lead_time_s = 0.25
        cfg.hydraulic_target_filter_beta = 0.5
        cfg.ee_correction_gain = 0.25
        cfg.command_rom_direction_guard = True
        cfg.procedural_trajectory_probability = 0.0
        cfg.clearance_relative_to_reference = True
        cfg.cutter_clearance_hard_band_m = 0.20
    else:
        raise ValueError(f"curriculum stage must be 1, 2, 3, or 4; got {stage}")
    cfg.curriculum_stage = stage
    cfg.force_zero_policy = False


@configclass
class ResidualPolicyEnvCfg(PPO2ROMPhysXEnvCfg):
    """Residual policy environment configuration.

    The policy adds small residual corrections (cylinders 1-3 only) to the
    deterministic head-down A->B->A arm controller.
    """

    # --- Policy action dimension ---
    # 3 action dims for cylinders 1-3; cylinder 4 stays at head-down value.
    action_space = 3
    observation_space = RESIDUAL_OBS_DIM

    # The ROM trajectory gate is validated with the smaller EE/reference-scale
    # command.  The 0.02 cylinder-development scale can drive the recurrent ROM
    # target integrator far outside the mowing envelope during random-phase
    # residual resets.
    action_scale: float = 0.003

    # --- Residual scaling ---
    residual_scale_m: float = 0.002       # +-2 mm default
    action_filter: float = 0.8            # EMA filter for residual action
    residual_scale_min_m: float = 0.001   # 1 mm for curriculum stage 1
    residual_scale_max_m: float = 0.004   # 4 mm for curriculum stage 3

    # --- Trajectory ---
    arc_path: str = str(
        Path(__file__).resolve().parents[1]
        / "models/mowing_arc_head_down.json"
    )
    trajectory_library_path: str = str(
        Path(__file__).resolve().parents[1]
        / "models/plant_reachable_trajectory_library_v1.json"
    )
    # Stages 1-3 retain the exact trajectory used to train the existing
    # checkpoint. Stage 4 switches to the direct-PhysX-verified shape library.
    trajectory_shape: str = "legacy_arc"
    trajectory_shapes: tuple[str, ...] = ("line", "arc", "sine", "circle", "square")
    procedural_trajectory_shapes: tuple[str, ...] = (
        "proc_line_compact_low",
        "proc_line_offset_high",
        "proc_arc_compact",
        "proc_arc_offset",
        "proc_sine_compact",
        "proc_sine_offset",
        "proc_circle_inner",
        "proc_circle_offset",
        "proc_square_inner",
        "proc_square_offset",
    )
    procedural_trajectory_probability: float = 0.0
    randomize_trajectory_shape: bool = False
    path_speed_mps: float = 0.13
    accel_time_s: float = 1.0
    policy_preview_time_s: float = 0.25
    seed_lead_time_s: float = 0.25
    ground_height_bias_m: float = -0.03
    dls_error_gain: float = 0.18
    dls_velocity_damping: float = 0.0
    dls_velocity_feedforward: float = 1.0
    dls_measured_error_feedback: float = 0.0
    dls_velocity_filter_beta: float = 0.0
    dls_action_smoothing: float = 0.20
    # Low-pass the 60 Hz cylinder target stream.  A value of 0.5 removes the
    # visible step-to-step chatter without adding the tracking lag seen at 0.75.
    hydraulic_target_filter_beta: float = 0.5
    # A ROM tick spans three 60 Hz control steps.  The ROM remains the sole
    # learned actuator model, while this causal envelope prevents recurrent
    # state leakage from commanding a cylinder opposite to the requested
    # motion or moving it under an exact hold command.
    command_rom_direction_guard: bool = False
    command_rom_command_max_ratio: float = 3.0
    ee_correction_gain: float = 0.25
    cylinder4_head_down_extension: float = 0.588991
    # Twelve-second training episodes start at random path positions/directions
    # and accelerate from rest, so PPO sees both directions and the full arc.
    randomize_start_phase: bool = True
    synchronize_resets: bool = True
    # Vectorized training may defer finite safety terminations until the normal
    # episode timeout.  The violations are still logged and penalized, but one
    # outlier can no longer force every replica through the expensive physical
    # deployment reset in the middle of a rollout.  Playback and evaluation
    # keep strict terminations because this defaults to False.
    defer_safety_terminations: bool = False

    # --- Curriculum ---
    curriculum_stage: int = 0
    force_zero_policy: bool = False

    # --- Episode ---
    episode_length_s: float = 12.0

    # --- Reward weights (configurable) ---
    residual_tracking_std_m: float = 0.02
    residual_tracking_weight: float = 4.0
    # Optional non-saturating quadratic guard. The Gaussian tracking reward is
    # deliberately local and has almost no gradient on the larger-error parts
    # of some workspace shapes; this guard is enabled only by constrained
    # experiments that must not trade one shape against another.
    residual_tracking_guard_weight: float = 0.0
    residual_tracking_guard_scale_m: float = 0.04
    residual_progress_weight: float = 0.5
    residual_action_magnitude_weight: float = -0.02
    residual_action_rate_weight: float = -0.05
    # Penalize physical tool acceleration in addition to action rate.  The
    # normalized bounded form is applied in env.py so isolated PhysX spikes do
    # not dominate the tracking objective.
    residual_tool_acceleration_weight: float = 0.0
    residual_tool_acceleration_scale_mps2: float = 30.0
    residual_head_penalty_weight: float = -2.0
    residual_ground_penalty_weight: float = -2.0
    residual_saturation_penalty_weight: float = -1.0
    residual_head_threshold: float = 0.995
    residual_ground_band_m: float = 0.10
    # Legacy stages retain absolute mowing-ground clearance semantics. Mixed
    # height stage 4 measures safety deviation from the commanded height.
    clearance_relative_to_reference: bool = False
    # Opt-in frame-consistent clearance. False keeps the historical (frame
    # inconsistent) signal so PPO2-A and ROM-C are bit-exact; True uses the
    # orientation-aware cutter and a live-truck-pose reference so ride height
    # cancels. See PPO2_ROM_B_PhysX/reports/rom_b_frame_decomposition.md.
    clearance_frame_consistent: bool = False

    # --- Safety limits ---
    residual_cylinder_learned_min_override: list[float] | None = None
    residual_cylinder_learned_max_override: list[float] | None = None

    # --- Truck ---
    enable_scripted_vehicle_motion: bool = False
    vehicle_speed_min_kmh: float = 0.0
    vehicle_speed_max_kmh: float = 0.0

    # --- Termination thresholds ---
    truck_drift_termination_m: float = 0.02
    # A larger pre-constraint value is a suspension/arm reaction, not retained
    # chassis travel.  Still reject extreme impulses instead of silently
    # relying on the planar fixture.
    truck_raw_reaction_termination_m: float = 0.05
    head_down_termination_alignment: float = 0.95
    cutter_clearance_hard_band_m: float = 0.15

    # --- Preposition ---
    preposition_steps: int = 300
    settle_steps: int = 120
    # Direct deployment bypasses hydraulics. Prime and settle the recurrent ROM
    # at the deployed cylinder state before the first scored/policy transition.
    rom_reset_warmup_steps: int = 120

    # --- Cylinder DLS ---
    tracking_tolerance_m: float = 0.05
    ground_tolerance_m: float = 0.10
    head_down_min_alignment: float = 0.95

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        # Match PPO1's smooth deterministic plant integration while retaining
        # the command ROM as the only learned hydraulic backend.
        self.sim.physx.enable_external_forces_every_iteration = True
        self.robot_cfg.spawn.articulation_props.solver_velocity_iteration_count = 2
        # Ensure observation dimension matches
        self.observation_space = RESIDUAL_OBS_DIM
        self.episode_length_s = 12.0


@configclass
class ResidualPolicyEnvCfg_PLAY(ResidualPolicyEnvCfg, MulagUpperLevelEnvCfg_PLAY):
    """Single-environment config for play/evaluation."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.randomize_start_phase = False
        # The measured teacher reaches +0.094 m and the deterministic gate
        # reaches +0.09401 m. Keep the ±0.10 m reward penalty unchanged, but
        # allow transient suspension recovery before playback hard-terminates.
        # Training retains the stricter ±0.15 m safety curriculum.
        self.cutter_clearance_hard_band_m = 0.20


@configclass
class ResidualPolicyPPORunnerCfg(MulagUpperLevelPPORunnerCfg):
    """PPO runner configuration for the Policy.

    Initial settings:
      actor:  [256, 128, 64] with ELU activation
      critic: [256, 128, 64] with ELU activation
      initial std: 0.05
      learning rate: 3e-4
      gamma: 0.99
      lambda: 0.95
      clip: 0.2
      entropy coefficient: 0.001
      learning epochs: 5
      mini-batches: 4
      desired KL: 0.02
      max grad norm: 1.0
      empirical normalization: enabled
    """

    experiment_name = "ppo2_rom_residual"
    num_steps_per_env = 128
    max_iterations = 2000
    save_interval = 100
    empirical_normalization = True

    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.05),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
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
        desired_kl=0.02,
        max_grad_norm=1.0,
    )
