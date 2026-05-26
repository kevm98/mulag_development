from __future__ import annotations

from pathlib import Path

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from rrlab_assets import MULAG_CFG


_MODELS_DIR = str(Path(__file__).resolve().parents[1] / "models")


@configclass
class MulagAdaptiveRLJointReachEnvCfg(DirectRLEnvCfg):
    decimation = 1
    episode_length_s = 5.0
    action_scale = 1.0
    action_space = 4
    observation_space = 62
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1 / 100, render_interval=decimation)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=8, env_spacing=20.0, replicate_physics=True)
    robot_cfg: ArticulationCfg = MULAG_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    models_dir = _MODELS_DIR
    fallback_window = 10
    max_abs_valve = 1.0
    max_abs_torque = 5000.0
    max_abs_pressure_delta = 1.0e7
    max_abs_fnet = 1.0e7
    sd_output_mode = "state"
    pressure_noise_std = 0.0

    state_joint_names = [
        "Drehzapfen_joint",
        "Ausleger_I_joint",
        "Ausleger_II_joint",
        "Messerkopf_Schwenk_joint",
        "Messerkopf_joint",
    ]
    command_joint_names = [
        "Drehzapfen_joint",
        "Ausleger_I_joint",
        "Ausleger_II_joint",
        "Messerkopf_Schwenk_joint",
    ]
    target_joint_range = [0.35, 0.35, 0.35, 0.30, 0.25]

    joint_limit_margin = 0.02
    max_abs_joint_velocity = 6.0

    rew_q_tracking = 8.0
    rew_qdot = 0.25
    rew_action_smoothness = 0.05
    rew_valve_effort = 0.01
    rew_torque_effort = 1.0e-6
    rew_alive = 0.1
    rew_termination = 5.0


@configclass
class MulagAdaptiveRLJointReachEnvCfg_PLAY(MulagAdaptiveRLJointReachEnvCfg):
    def __post_init__(self):
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
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_hidden_dims=[128, 128],
        critic_hidden_dims=[128, 128],
        activation="elu",
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
