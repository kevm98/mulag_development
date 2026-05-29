from __future__ import annotations

from pathlib import Path

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from kevin_integration.utils.sim_memory import apply_kevin_sim_memory_optimizations
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
    control_mode = "position_delta"
    position_delta_source = "id_valve"
    use_sd_feedback = False
    use_fd_effort = False
    position_scale = 0.005
    action_to_qddot_scale = 1.0
    qddot_cmd_clip = 5.0
    torque_ramp_steps = 1000
    torque_safety_scale = 1.0
    log_full_pipeline_debug = True
    debug_reward_terms = True

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
    q_tracking_joint_weights = [1.0, 1.0, 1.0, 1.0, 0.2]

    joint_limit_margin = 0.02
    max_abs_joint_velocity = 6.0

    rew_q_tracking = 8.0
    rew_qdot = 0.25
    rew_progress = 1.0
    rew_action_smoothness = 0.05
    rew_valve_effort = 0.01
    rew_torque_effort = 1.0e-6
    rew_alive = 0.1
    rew_termination = 5.0

    def __post_init__(self):
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
