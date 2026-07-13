"""Environment and PPO configuration owned by the residual policy.
Stages A-D: enriched obs, restructured reward, truck curriculum.
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
    """Two-dimensional Y/Z residual over the verified classical cascade."""

    task_type: str = "ee_tracking"
    action_space = 3  # (dy, dz, truck-speed)
    observation_space = EE_RESIDUAL_OBS_DIM
    action_scale = 0.02
    episode_length_s = 35.0
    # Truck speed is a POLICY action (teacher_student-style direct wheel write);
    # scripted driving off. Zero action = parked truck = stationary baseline.
    enable_scripted_vehicle_motion = False
    # Same learned ID/FD backend as the passing trajectory gate.
    low_level_servo_mode = "none"

    residual_action_scale_m = 0.05
    residual_error_reward_scale = 20.0
    residual_settle_steps = 400  # unscored cascade descent to the mapped path start (~1 cm/step measured)
    # Nominal path = the PHYSICS-MAPPED ground arc (same branch the passing
    # long-trajectory gate tracks), sampled at a fixed path speed.
    residual_path_speed_mps = 0.045
    residual_ee_correction_gain = 0.40
    # action[2] in [-1,1] -> truck speed kmh = clamp(base + a*span, 0, max)
    residual_truck_base_kmh = 0.0
    residual_truck_span_kmh = 4.5
    residual_truck_max_kmh = 6.0
    # Stage D: truck-speed curriculum. Phase 0=disabled, 1=slow(2.0kmh), 2=full(4.5kmh).
    residual_truck_curriculum_phase: int = 0
    # Stage B: ungated progress reward with exponential weight
    residual_progress_weight = 1.5
    residual_progress_gate_m = 0.05      # unused now (kept for compatibility)
    # Stage B: action-smoothing penalty scale
    residual_action_smooth_weight = 0.01
    # Stage B: progress decay constant (larger = wider credit window)
    residual_progress_decay_m = 0.10

    def __post_init__(self):
        super().__post_init__()
        self.observation_space = EE_RESIDUAL_OBS_DIM
        self.action_space = 3  # (dy, dz, truck-speed)


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
    # Stage A: larger net, wider init, exploration pressure, higher LR
    actor = RslRlMLPModelCfg(
        hidden_dims=[128, 64],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.15),
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
        desired_kl=0.008,
        max_grad_norm=1.0,
    )


__all__ = [
    "IDFDEEResidualEnvCfg",
    "IDFDEEResidualEnvCfg_PLAY",
    "IDFDEEResidualPPORunnerCfg",
]
