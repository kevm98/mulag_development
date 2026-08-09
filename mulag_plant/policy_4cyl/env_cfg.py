"""Configuration for the constrained four-cylinder residual experiment."""

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlPpoAlgorithmCfg

from mulag_plant.policy.env_cfg import ResidualPolicyEnvCfg
from mulag_plant.policy.observations import FOUR_CYLINDER_RESIDUAL_OBS_DIM
from mulag_plant.runtime.mulag_upper_level_env_cfg import MulagUpperLevelEnvCfg_PLAY
from mulag_plant.runtime.mulag_upper_level_env_cfg import MulagUpperLevelPPORunnerCfg


@configclass
class PitchBranchActorCfg(RslRlMLPModelCfg):
    """Configuration for the independent cylinder-4 pitch branch."""

    class_name: str = (
        "mulag_plant.policy_4cyl.pitch_actor:FrozenTeacherPitchActor"
    )
    pitch_hidden_dims: list[int] = [64, 32]
    # Tool/reference dynamics; cylinder-4 extension, goal error, velocity,
    # previous action, valve and ROM increment; preview/direction; clearance;
    # head alignment and alignment rate.
    pitch_obs_indices: list[int] = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
        15, 19, 23, 27, 31, 35, 36, 37, 38, 39, 40, 41,
    ]


@configclass
class FourCylinderResidualPolicyEnvCfg(ResidualPolicyEnvCfg):
    """Four-action variant with tightly bounded mower-pitch authority."""

    action_space = 4
    observation_space = FOUR_CYLINDER_RESIDUAL_OBS_DIM

    # Cylinders 1-3 retain the accepted ±2 mm residual. Cylinder 4 starts at
    # zero in the training script and is unlocked to no more than ±0.5 mm.
    cylinder4_residual_scale_m: float = 0.0
    cylinder4_residual_scale_max_m: float = 0.0005
    cylinder4_action_filter: float = 0.95

    # Equal action penalties would encourage the policy to tilt the mower as a
    # positional shortcut. Penalize the fourth channel explicitly and protect
    # the measured orientation/clearance envelope.
    cylinder4_action_magnitude_weight: float = -0.25
    cylinder4_action_rate_weight: float = -1.50
    residual_action_rate_weight: float = -0.20
    residual_tracking_std_m: float = 0.04
    residual_tracking_guard_weight: float = -2.0
    residual_tracking_guard_scale_m: float = 0.04
    residual_tool_acceleration_weight: float = -0.50
    residual_tool_acceleration_scale_mps2: float = 30.0
    residual_head_penalty_weight: float = -8.0
    residual_ground_penalty_weight: float = -4.0
    residual_head_threshold: float = 0.997
    head_down_termination_alignment: float = 0.98
    cutter_clearance_hard_band_m: float = 0.20
    clearance_relative_to_reference: bool = True

    def __post_init__(self):
        super().__post_init__()
        self.action_space = 4
        self.observation_space = FOUR_CYLINDER_RESIDUAL_OBS_DIM


@configclass
class FourCylinderResidualPolicyEnvCfg_PLAY(
    FourCylinderResidualPolicyEnvCfg, MulagUpperLevelEnvCfg_PLAY
):
    """Single-environment four-cylinder playback configuration."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.randomize_start_phase = False


@configclass
class FourCylinderResidualPolicyPPORunnerCfg(MulagUpperLevelPPORunnerCfg):
    """Conservative PPO settings for the cylinder-4 unlock experiment."""

    experiment_name = "ppo2_a_policy_v3"
    num_steps_per_env = 128
    max_iterations = 250
    save_interval = 10
    empirical_normalization = True

    actor = PitchBranchActorCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.02),
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
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-5,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
