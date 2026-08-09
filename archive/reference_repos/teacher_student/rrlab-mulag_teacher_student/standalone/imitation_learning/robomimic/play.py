# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play and evaluate a trained policy from robomimic.

This script loads a robomimic policy and plays it in an Isaac Lab environment.

Args:
    task: Name of the environment.
    checkpoint: Path to the robomimic policy checkpoint.
    horizon: If provided, override the step horizon of each rollout.
    num_rollouts: If provided, override the number of rollouts.
    seed: If provided, overeride the default random seed.
    norm_factor_min: If provided, minimum value of the action space normalization factor.
    norm_factor_max: If provided, maximum value of the action space normalization factor.
"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate robomimic policy for Isaac Lab environment.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Pytorch model checkpoint to load.")
parser.add_argument("--horizon", type=int, default=5000, help="Step horizon of each rollout.")
parser.add_argument("--num_rollouts", type=int, default=1, help="Number of rollouts.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=101, help="Random seed.")
parser.add_argument(
    "--norm_factor_min", type=float, default=None, help="Optional: minimum value of the normalization factor."
)
parser.add_argument(
    "--norm_factor_max", type=float, default=None, help="Optional: maximum value of the normalization factor."
)
parser.add_argument("--enable_pinocchio", default=False, action="store_true", help="Enable Pinocchio.")


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version installed by IsaacLab and not the one installed by Isaac Sim
    # pinocchio is required by the Pink IK controllers and the GR1T2 retargeter
    import pinocchio  # noqa: F401

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import copy
import gymnasium as gym
import numpy as np
import random
import torch

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import rrlab_tasks

if args_cli.enable_pinocchio:
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401

from isaaclab_tasks.utils import parse_env_cfg


def rollout(policy, env, success_term, horizon, device):
    """Rollout robomimic policy in a vectorized Isaac Lab env and compute per-env results.

    Returns:
        success_flags_np: (num_envs,) bool
        done_flags_np:    (num_envs,) bool
        steps_np:         (num_envs,) int  (steps taken until done or horizon)
    """
    policy.start_episode()
    obs_dict, _ = env.reset()

    num_envs = getattr(env, "num_envs", None) or 1
    action_dim = env.action_space.shape[1]

    done = torch.zeros(num_envs, dtype=torch.bool, device=device)
    success_flags = torch.zeros(num_envs, dtype=torch.bool, device=device)
    steps = torch.zeros(num_envs, dtype=torch.int32, device=device)

    for _ in range(horizon):
        obs = copy.deepcopy(obs_dict["policy"])

        # Image preprocessing: ensure (B,C,H,W) float in [0,1]
        if hasattr(env.cfg, "image_obs_list"):
            for image_name in env.cfg.image_obs_list:
                if image_name in obs:
                    img = obs[image_name]
                    if not torch.is_tensor(img):
                        img = torch.as_tensor(img)

                    if img.dtype != torch.float32:
                        img = img.float()

                    # If uint8 0..255 -> normalize
                    if img.max() > 1.0:
                        img = img / 255.0
                    img = img.clamp(0.0, 1.0)

                    # If channel-last (B,H,W,C) -> (B,C,H,W)
                    if img.ndim == 4 and img.shape[-1] in (1, 3) and img.shape[1] not in (1, 3):
                        img = img.permute(0, 3, 1, 2).contiguous()

                    obs[image_name] = img

        # Robomimic policy inference (expects batched obs dict)
        actions_np = policy(obs)

        # Optional action unnormalization
        if args_cli.norm_factor_min is not None and args_cli.norm_factor_max is not None:
            actions_np = ((actions_np + 1) * (args_cli.norm_factor_max - args_cli.norm_factor_min)) / 2 + args_cli.norm_factor_min

        actions = torch.from_numpy(actions_np).to(device=device)

        # Ensure (num_envs, action_dim)
        if actions.ndim == 1:
            actions = actions.view(1, -1)
        if actions.shape[0] == 1 and num_envs > 1:
            # Fallback: replicate same action to all envs
            actions = actions.repeat(num_envs, 1)
        else:
            actions = actions.view(num_envs, action_dim)

        # Step env
        obs_dict, _, terminated, truncated, _ = env.step(actions)

        terminated = torch.as_tensor(terminated, device=device, dtype=torch.bool)
        truncated = torch.as_tensor(truncated, device=device, dtype=torch.bool)
        step_done = terminated | truncated

        # Count steps for envs that are not done yet (inclusive step count)
        steps += (~done).to(torch.int32)

        # Update done mask
        done = done | step_done

        # Success per env (assumed shape (num_envs,))
        success = success_term.func(env, **success_term.params)
        success = torch.as_tensor(success, device=device, dtype=torch.bool)
        success_flags = success_flags | success

        # Stop when all envs finished
        if torch.all(done):
            break

    return success_flags.detach().cpu().numpy(), done.detach().cpu().numpy(), steps.detach().cpu().numpy()


def main():
    """Run a trained robomimic policy in Isaac Lab with multi-env evaluation."""
    num_envs = args_cli.num_envs if args_cli.num_envs is not None else 1
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    # robomimic expects dict obs (not concatenated)
    env_cfg.observations.policy.concatenate_terms = False

    # Use custom horizon instead of env time_out termination
    env_cfg.terminations.time_out = None

    # Disable recorders for speed
    env_cfg.recorders = None

    # Extract success term (we'll call it manually)
    success_term = env_cfg.terminations.success
    env_cfg.terminations.success = None

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # Seeds
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    random.seed(args_cli.seed)
    env.seed(args_cli.seed)

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    all_success = []
    all_done = []
    all_steps = []

    for trial in range(args_cli.num_rollouts):
        print(f"[INFO] Starting trial {trial}")

        policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=args_cli.checkpoint, device=device)

        success_flags, done_flags, steps = rollout(
            policy=policy,
            env=env,
            success_term=success_term,
            horizon=args_cli.horizon,
            device=device,
        )

        all_success.append(success_flags)
        all_done.append(done_flags)
        all_steps.append(steps)

        print(f"[INFO] Trial {trial} per-env success: {success_flags.astype(int).tolist()}")
        print(f"[INFO] Trial {trial} per-env done:    {done_flags.astype(int).tolist()}")
        print(f"[INFO] Trial {trial} per-env steps:   {steps.tolist()}")
        print(f"[INFO] Trial {trial} success rate:    {success_flags.mean():.3f}\n")

    all_success = np.concatenate(all_success, axis=0)
    all_done = np.concatenate(all_done, axis=0)
    all_steps = np.concatenate(all_steps, axis=0)

    print("\n========== Summary ==========")
    print(f"Total env episodes evaluated: {len(all_success)}")
    print(f"Total successes: {int(all_success.sum())} / {len(all_success)}")
    print(f"Overall success rate: {all_success.mean():.4f}")
    print(f"Done rate (terminated/truncated): {all_done.mean():.4f}")
    print(f"Mean steps: {all_steps.mean():.1f} | Median steps: {np.median(all_steps):.1f} | Max steps: {all_steps.max()}")

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
