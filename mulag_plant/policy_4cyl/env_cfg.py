"""Configuration for the constrained four-cylinder residual experiment."""

from __future__ import annotations

from pathlib import Path

from isaaclab.utils import configclass
from rrlab_compat.rsl_rl import RslRlMLPModelCfg, RslRlPpoAlgorithmCfg

from mulag_plant.policy.env_cfg import ResidualPolicyEnvCfg
from mulag_plant.policy.observations import FOUR_CYLINDER_RESIDUAL_OBS_DIM, residual_observation_dim
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
    clearance_frame_consistent: bool = True
    # Deployment guard: the learned correction may only operate inside the
    # measured head-down clearance envelope.  Outside the start band the
    # frozen DLS/teacher command is used until the plant recovers.
    residual_safety_guard: bool = True
    residual_safety_clearance_start_m: float = 0.08
    residual_safety_head_start: float = 0.990
    # Moving open paths are the most sensitive to small residual-induced
    # clearance transients.  Keep their approved frozen DLS baseline online;
    # the learned residual remains enabled for parked closed-loop stations.
    residual_fallback_shapes: tuple[str, ...] = ("line", "sine_chain")

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


@configclass
class FourCylinderTrajectory2DEnvCfg(FourCylinderResidualPolicyEnvCfg):
    """New standalone 2-D shape task; the historical 42-D task is unchanged."""

    trajectory_library_path: str = str(
        Path(__file__).resolve().parents[1] / "models/trajectory_2d_library_v2.json"
    )
    trajectory_shape: str = "line"
    trajectory_shapes: tuple[str, ...] = (
        "line", "sine_chain", "circle", "rounded_square"
    )
    procedural_trajectory_shapes: tuple[str, ...] = ()
    randomize_trajectory_shape: bool = False
    trajectory_geometry_observation: bool = True
    trajectory_geometry_normal_reward_weight: float = 2.0
    trajectory_geometry_normal_std_m: float = 0.03
    # Shape changes are arm-only station transitions.  Give the direct
    # deployment controller enough physical time to move from the previous
    # learned endpoint to the next measured workspace seed before scoring.
    preposition_steps: int = 600
    settle_steps: int = 180
    # Match the original verge-mower presentation: a wide green working strip
    # beside a marked asphalt road.  Both are visual-only overlays on the
    # already validated flat collision ground.
    enable_visual_road_verge: bool = True
    visual_road_start_x_m: float = -38.0
    visual_road_length_m: float = 36.0
    visual_verge_start_y_m: float = -3.20
    visual_verge_end_y_m: float = 2.48
    observation_space = residual_observation_dim(4, trajectory_geometry=True)

    def __post_init__(self):
        super().__post_init__()
        self.trajectory_geometry_observation = True
        self.observation_space = residual_observation_dim(4, trajectory_geometry=True)


@configclass
class FourCylinderTrajectory2DEnvCfg_PLAY(
    FourCylinderTrajectory2DEnvCfg, MulagUpperLevelEnvCfg_PLAY
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.randomize_start_phase = False


@configclass
class FourCylinderTrajectory2DPPORunnerCfg(FourCylinderResidualPolicyPPORunnerCfg):
    experiment_name = "ppo2_a_trajectory_2d"
