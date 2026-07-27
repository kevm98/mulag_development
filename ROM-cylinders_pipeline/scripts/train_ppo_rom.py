#!/usr/bin/env python3
"""Train PPO with the shared controller -> command ROM -> PhysX actuator."""

from __future__ import annotations

import argparse
import sys
import traceback
from datetime import datetime
from pathlib import Path


ROM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROM_ROOT.parent
for path in (REPO_ROOT, ROM_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--max_iterations", type=int, default=2000)
parser.add_argument("--seed", type=int, default=17)
parser.add_argument("--checkpoint", type=str)
AppLauncher.add_app_launcher_args(parser)
args, unknown = parser.parse_known_args()
sys.argv = [sys.argv[0], *unknown]

launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from ROM_pipeline.isaac import TASK_ID  # noqa: E402
from ROM_pipeline.isaac.env_cfg import ROMCylinderSpacePPORunnerCfg  # noqa: E402


def main() -> None:
    cfg = parse_env_cfg(TASK_ID, device=args.device, num_envs=int(args.num_envs))
    cfg.seed = int(args.seed)
    checkpoint = Path(cfg.command_rom_checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Finish base ROM training before PPO launch: {checkpoint}")

    agent_cfg = ROMCylinderSpacePPORunnerCfg()
    agent_cfg.seed = int(args.seed)
    agent_cfg.device = args.device
    agent_cfg.max_iterations = int(args.max_iterations)
    log_dir = REPO_ROOT / "logs/rsl_rl/rom_cylinder_space" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[PPO-ROM] task={TASK_ID} checkpoint={checkpoint} "
        f"envs={args.num_envs} iterations={args.max_iterations} log={log_dir}",
        flush=True,
    )

    gym_env = gym.make(TASK_ID, cfg=cfg)
    try:
        env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device)
        if args.checkpoint:
            runner.load(str(Path(args.checkpoint).expanduser().resolve()))
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    finally:
        gym_env.close()


try:
    main()
except BaseException:
    traceback.print_exc()
    raise
finally:
    simulation_app.close()
