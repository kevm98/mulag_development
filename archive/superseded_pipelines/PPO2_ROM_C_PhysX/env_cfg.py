"""Configuration for PPO2-ROM_C (locked robust ROM-B + online AAM)."""

from __future__ import annotations

from pathlib import Path

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlPpoAlgorithmCfg

from PPO2_ROM_B_PhysX.env_cfg import PPO2ROMBPolicyEnvCfg
from PPO2_ROM_PhysX.policy_4cyl.env_cfg import (
    FourCylinderResidualPolicyPPORunnerCfg,
    PitchBranchActorCfg,
)
from PPO2_ROM_PhysX.runtime.mulag_upper_level_env_cfg import MulagUpperLevelEnvCfg_PLAY


ROOT = Path(__file__).resolve().parents[1]
AAM_OBS_DIM = 5
PPO2_C_OBS_DIM = 47


@configclass
class PPO2CAAMActorCfg(PitchBranchActorCfg):
    """ROM-B actor topology with all five AAM estimates in both branches."""

    pitch_obs_indices: list[int] = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
        15, 19, 23, 27, 31, 35, 36, 37, 38, 39, 40, 41,
        42, 43, 44, 45, 46,
    ]


@configclass
class PPO2CAAMPolicyEnvCfg(PPO2ROMBPolicyEnvCfg):
    """Training configuration with hidden per-environment ROM factors."""

    observation_space = PPO2_C_OBS_DIM
    action_space = 4
    aam_checkpoint: str = str(
        ROOT / "AAM_SystemFactor/checkpoints/aam_system_factor_runtime.pt"
    )
    aam_sample_time_s: float = 0.05
    aam_ema_beta: float = 0.90

    def __post_init__(self):
        super().__post_init__()
        self.observation_space = PPO2_C_OBS_DIM
        self.action_space = 4


@configclass
class PPO2CAAMPolicyEnvCfg_PLAY(
    PPO2CAAMPolicyEnvCfg, MulagUpperLevelEnvCfg_PLAY
):
    """Single-environment nominal playback (AAM remains online)."""

    rom_b_mode: str = "nominal"

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.randomize_start_phase = False
        self.rom_b_mode = "nominal"


@configclass
class PPO2CAAMPPORunnerCfg(FourCylinderResidualPolicyPPORunnerCfg):
    """Low-rate adaptation of the new AAM input columns."""

    experiment_name = "ppo2_rom_c_aam_policy"
    num_steps_per_env = 128
    max_iterations = 500
    save_interval = 10
    empirical_normalization = True
    actor = PPO2CAAMActorCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.01),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64], activation="elu", obs_normalization=True
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-6,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
