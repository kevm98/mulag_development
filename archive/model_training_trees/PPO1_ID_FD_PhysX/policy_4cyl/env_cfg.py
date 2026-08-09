"""Four-cylinder residual configuration for PPO1 -> ID -> FD -> PhysX."""

from __future__ import annotations

from isaaclab.utils import configclass

from PPO1_ID_FD_PhysX.policy.env_cfg import ResidualPolicyEnvCfg
from PPO1_ID_FD_PhysX.policy.observations import (
    FOUR_CYLINDER_RESIDUAL_OBS_DIM,
)
from PPO1_ID_FD_PhysX.runtime.mulag_upper_level_env_cfg import (
    MulagUpperLevelEnvCfg_PLAY,
)
from PPO2_ROM_PhysX.policy_4cyl.env_cfg import (
    FourCylinderResidualPolicyPPORunnerCfg as _SharedFourCylinderRunnerCfg,
)


@configclass
class FourCylinderResidualPolicyEnvCfg(ResidualPolicyEnvCfg):
    """ID-FD V3 environment with bounded PPO authority on all cylinders."""

    action_space = 4
    observation_space = FOUR_CYLINDER_RESIDUAL_OBS_DIM
    cylinder4_residual_scale_m: float = 0.00025
    cylinder4_residual_scale_max_m: float = 0.0005
    cylinder4_action_filter: float = 0.95
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
    """Single-environment playback configuration."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.randomize_start_phase = False


@configclass
class FourCylinderResidualPolicyPPORunnerCfg(_SharedFourCylinderRunnerCfg):
    """Shared four-output actor/critic with an ID-FD experiment namespace."""

    experiment_name = "ppo1_a_policy_v3"
