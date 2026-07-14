"""Configuration for the residual PPO policy environment.

Extends the base IDFDCylinderSpaceEnvCfg with residual-policy-specific settings
and provides a dedicated PPO runner configuration.
"""

from __future__ import annotations

from pathlib import Path

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlPpoAlgorithmCfg

from ID_FD_pipline.tasks.env_cfg import IDFDCylinderSpaceEnvCfg
from ID_FD_pipline.runtime.mulag_upper_level_env_cfg import (
    MulagUpperLevelEnvCfg_PLAY,
    MulagUpperLevelPPORunnerCfg,
)
from ID_FD_pipline.residual_policy.observations import RESIDUAL_OBS_DIM


@configclass
class ResidualPolicyEnvCfg(IDFDCylinderSpaceEnvCfg):
    """Residual policy environment configuration.

    The policy adds small residual corrections (cylinders 1-3 only) to the
    deterministic head-down A->B->A arm controller.
    """

    # --- Policy action dimension ---
    # 3 action dims for cylinders 1-3; cylinder 4 stays at head-down value.
    action_space = 3
    observation_space = RESIDUAL_OBS_DIM

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
    path_speed_mps: float = 0.13
    accel_time_s: float = 1.0
    seed_lead_time_s: float = 0.25
    cylinder4_head_down_extension: float = 0.588991
    # Twelve-second training episodes start at random path positions/directions
    # and accelerate from rest, so PPO sees both directions and the full arc.
    randomize_start_phase: bool = True
    synchronize_resets: bool = True

    # --- Curriculum ---
    curriculum_stage: int = 0
    force_zero_policy: bool = False

    # --- Episode ---
    episode_length_s: float = 12.0

    # --- Reward weights (configurable) ---
    residual_tracking_std_m: float = 0.02
    residual_tracking_weight: float = 4.0
    residual_progress_weight: float = 0.5
    residual_action_magnitude_weight: float = -0.02
    residual_action_rate_weight: float = -0.05
    residual_head_penalty_weight: float = -2.0
    residual_ground_penalty_weight: float = -2.0
    residual_saturation_penalty_weight: float = -1.0
    residual_head_threshold: float = 0.995
    residual_ground_band_m: float = 0.10

    # --- Safety limits ---
    residual_cylinder_learned_min_override: list[float] | None = None
    residual_cylinder_learned_max_override: list[float] | None = None

    # --- Truck ---
    enable_scripted_vehicle_motion: bool = False
    vehicle_speed_min_kmh: float = 0.0
    vehicle_speed_max_kmh: float = 0.0

    # --- Termination thresholds ---
    truck_drift_termination_m: float = 0.02
    head_down_termination_alignment: float = 0.95
    cutter_clearance_hard_band_m: float = 0.15

    # --- Preposition ---
    preposition_steps: int = 300
    settle_steps: int = 120

    # --- Cylinder DLS ---
    ee_correction_gain: float = 0.0
    tracking_tolerance_m: float = 0.05
    ground_tolerance_m: float = 0.10
    head_down_min_alignment: float = 0.95

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
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


@configclass
class ResidualPolicyPPORunnerCfg(MulagUpperLevelPPORunnerCfg):
    """PPO runner configuration for the residual policy.

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

    experiment_name = "id_fd_residual"
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
