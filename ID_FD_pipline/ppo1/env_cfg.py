"""Environment and PPO configuration for the stop-and-mow residual policy.

DeepSeek rebuild: state-machine-driven arm-only mowing with bounded PPO residual.
"""

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlPpoAlgorithmCfg

from ID_FD_pipline.ppo1.observations import EE_RESIDUAL_OBS_DIM
from ID_FD_pipline.tasks.env_cfg import IDFDCylinderSpaceEnvCfg
from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env_cfg import (
    MulagUpperLevelEnvCfg_PLAY,
    MulagUpperLevelPPORunnerCfg,
)


@configclass
class IDFDEEResidualEnvCfg(IDFDCylinderSpaceEnvCfg):
    """Bounded Y/Z residual + speed scale over stop-and-mow deterministic controller."""

    task_type: str = "ee_tracking"
    action_space = 3  # (dy, dz, speed_scale)
    observation_space = EE_RESIDUAL_OBS_DIM
    action_scale = 0.02
    episode_length_s = 60.0
    enable_scripted_vehicle_motion = False
    low_level_servo_mode = "none"

    # --- Stop-and-mow configuration ---
    residual_mow_shape: str = "arc"          # "arc" | "line" | "bowed_curve" | "sine_sweep"
    residual_lateral_span_m: float = 1.00    # unique lateral span
    residual_forward_bow_m: float = 0.14     # bowed_curve forward bow
    residual_forward_amplitude_m: float = 0.06  # sine_sweep forward amplitude
    residual_sweep_lobes: int = 3            # sine_sweep lobes per stroke

    # --- Arc (Drehzapfen-centered slewing arc) ---
    residual_arc_center_x: float = 4.327      # Drehzapfen X in truck base_link
    residual_arc_center_y: float = -3.224     # Drehzapfen Y in truck base_link
    residual_arc_radius_m: float = 1.286      # constant horizontal radius
    residual_arc_theta_a_deg: float = -26.7   # start angle [deg]
    residual_arc_theta_b_deg: float = 9.3     # end angle [deg]
    residual_min_accumulated_path_m: float = 5.0  # min accumulated cutter travel

    # --- Speed ---
    residual_path_speed_mps: float = 0.05    # m/s along the path

    # --- PPO residual bounds ---
    residual_action_scale_m: float = 0.03    # ±0.03 m lateral/height
    residual_action_smooth_weight: float = 0.01

    # --- Reward ---
    residual_error_reward_scale: float = 20.0
    residual_progress_weight: float = 1.5
    residual_progress_gate_m: float = 0.05   # unused for arm-only
    residual_progress_decay_m: float = 0.10

    # --- Settle / deploy / stow ---
    residual_settle_steps: int = 300
    residual_deploy_hold_steps: int = 60
    residual_stow_steps: int = 300

    # --- Ground following ---
    residual_target_clearance_m: float = 0.05
    residual_max_height_correction_m: float = 0.12
    residual_height_correction_gain: float = 0.5

    def __post_init__(self):
        super().__post_init__()
        self.observation_space = EE_RESIDUAL_OBS_DIM
        self.action_space = 3


@configclass
class IDFDEEResidualEnvCfg_PLAY(IDFDEEResidualEnvCfg, MulagUpperLevelEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1


@configclass
class IDFDEEResidualPPORunnerCfg(MulagUpperLevelPPORunnerCfg):
    experiment_name = "id_fd_ee_yz_residual"
    num_steps_per_env = 64
    max_iterations = 500
    save_interval = 25
    empirical_normalization = True
    actor = RslRlMLPModelCfg(
        hidden_dims=[128, 64],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.10),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[128, 64],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.02,
        max_grad_norm=1.0,
    )


__all__ = [
    "IDFDEEResidualEnvCfg",
    "IDFDEEResidualEnvCfg_PLAY",
    "IDFDEEResidualPPORunnerCfg",
]
