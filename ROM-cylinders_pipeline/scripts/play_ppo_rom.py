#!/usr/bin/env python3
"""Run a trained PPO -> desired cylinder command -> ROM -> PhysX policy."""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path


ROM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROM_ROOT.parent
for path in (REPO_ROOT, ROM_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint")
parser.add_argument("--seed", type=int, default=17)
parser.add_argument("--realtime", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args, unknown = parser.parse_known_args()
if not args.checkpoint:
    parser.error("--checkpoint is required")
sys.argv = [sys.argv[0], *unknown]

launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from ROM_pipeline.isaac import PLAY_TASK_ID  # noqa: E402
from ROM_pipeline.isaac.env_cfg import ROMCylinderSpacePPORunnerCfg  # noqa: E402


def main() -> None:
    cfg = parse_env_cfg(PLAY_TASK_ID, device=args.device, num_envs=1)
    cfg.seed = int(args.seed)
    gym_env = gym.make(PLAY_TASK_ID, cfg=cfg)
    raw = gym_env.unwrapped
    agent_cfg = ROMCylinderSpacePPORunnerCfg()
    agent_cfg.device = args.device
    env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    checkpoint = str(Path(args.checkpoint).expanduser().resolve())
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=raw.device)
    obs, _ = env.reset()
    step = 0
    try:
        while simulation_app.is_running():
            started = time.perf_counter()
            with torch.inference_mode():
                action = policy(obs)
                obs, _, dones, _ = env.step(action)
            if step % 30 == 0:
                error = torch.linalg.norm(raw._cylinder_goal_x - raw._cylinder_x_current, dim=-1).mean()
                print(
                    f"[PPO-ROM] step={step} mean_error={float(error):.5f}m "
                    f"valve={raw._current_valve_cmd[0].detach().cpu().tolist()}",
                    flush=True,
                )
            if dones.any():
                print("[PPO-ROM] episode reset", flush=True)
            step += 1
            if args.realtime:
                time.sleep(max(0.0, float(raw.step_dt) - (time.perf_counter() - started)))
    finally:
        env.close()


try:
    main()
except BaseException:
    traceback.print_exc()
    raise
finally:
    simulation_app.close()
