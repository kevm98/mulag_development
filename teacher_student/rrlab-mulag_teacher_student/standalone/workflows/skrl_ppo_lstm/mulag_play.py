import argparse
import sys

import gymnasium as gym
import os
import random
import time
import torch

import skrl
from packaging import version

from isaaclab.app import AppLauncher
from lstm_model import Shared
from lstm_model import SharedRNN
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.ppo import PPO
from skrl.agents.torch.ppo import PPO_RNN
from skrl.agents.torch.ppo import PPO_DEFAULT_CONFIG
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.resources.schedulers.torch import KLAdaptiveRL


# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default=None,
    help=(
        "Name of the RL agent configuration entry point. Defaults to None, in which case the argument "
        "--algorithm is used to determine the default agent configuration entry point."
    ),
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

# check for minimum supported skrl version
SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import rrlab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)

# config shortcuts
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, experiment_cfg: dict):
    """Play with skrl agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

        # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    experiment_cfg["seed"] = args_cli.seed if args_cli.seed is not None else experiment_cfg["seed"]
    env_cfg.seed = experiment_cfg["seed"]

    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # get checkpoint path
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("skrl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # get environment (step) dt for real-time evaluation
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    device = env.device

    # instantiate a memory as rollout buffer
    memory = RandomMemory(memory_size=512, num_envs=env.num_envs, device=device)

    # instantiate the agent's models
    #models = {}
    #models["policy"] = Shared(env.observation_space, env.action_space, device)
    #models["value"] = models["policy"]  # same instance: shared model

    models = {}
    models["policy"] = SharedRNN(env.observation_space, env.action_space, device, num_envs=env.num_envs)
    models["value"] = models["policy"]

    # configure and instantiate the agent
    cfg = PPO_DEFAULT_CONFIG.copy()
    cfg["rollouts"] = 512
    cfg["learning_epochs"] = 4
    cfg["mini_batches"] = 32 
    cfg["discount_factor"] = 0.99
    cfg["lambda"] = 0.95

    cfg["learning_rate"] = 3.0e-4
    cfg["learning_rate_scheduler"] = None
    #cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.01}

    cfg["grad_norm_clip"] = 1.0
    cfg["ratio_clip"] = 0.2
    cfg["value_clip"] = 0.2
    cfg["clip_predicted_values"] = True
    cfg["entropy_loss_scale"] = 0.005
    cfg["value_loss_scale"] = 1.0

    cfg["kl_threshold"] = 0.0
    cfg["random_timesteps"] = 0
    cfg["learning_starts"] = 0
    cfg["time_limit_bootstrap"] = True
    cfg["mixed_precision"] = True

    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
    cfg["value_preprocessor"] = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}

    # logging to TensorBoard and write checkpoints (in timesteps)
    cfg["experiment"]["directory"] = log_root_path
    cfg["experiment"]["experiment_name"] = os.path.basename(log_dir)
    cfg["experiment"]["write_interval"] = 0
    cfg["experiment"]["checkpoint_interval"] = 0

    agent = PPO_RNN(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)
    

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    agent.load(resume_path)

    # configure and instantiate the RL trainer
    cfg_trainer = {"timesteps": 6000000, "environment_info": "log"}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

    # start evaluation
    trainer.eval()

    # ---------------------- Non RNN Evaluation ------------------------------
    ## set agent to evaluation mode
    #agent.set_running_mode("eval")
#
    ## reset environment
    #obs, _ = env.reset()
    #timestep = 0
    ## simulate environment
    #while simulation_app.is_running():
    #    start_time = time.time()
#
    #    # run everything in inference mode
    #    with torch.inference_mode():
    #        # agent stepping
    #        outputs = agent.act(obs, timestep=0, timesteps=0)
    #        # - multi-agent (deterministic) actions
    #        if hasattr(env, "possible_agents"):
    #            actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
    #        # - single-agent (deterministic) actions
    #        else:
    #            actions = outputs[-1].get("mean_actions", outputs[0])
    #        # env stepping
    #        obs, _, _, _, _ = env.step(actions)
    #    if args_cli.video:
    #        timestep += 1
    #        # exit the play loop after recording one video
    #        if timestep == args_cli.video_length:
    #            break
#
    #    # time delay for real-time evaluation
    #    sleep_time = dt - (time.time() - start_time)
    #    if args_cli.real_time and sleep_time > 0:
    #        time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

# python standalone/workflows/skrl_ppo_lstm/mulag_play.py --task RRLAB-Obstacle-Avoidance-Mulag-v0 --num_envs 16 --enable_cameras --checkpoint
# python standalone/workflows/skrl_ppo_lstm/mulag_train.py --task RRLAB-Obstacle-Avoidance-Mulag-v0 --num_envs 140 --enable_cameras --headless 