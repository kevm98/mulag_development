"""Visualize a trained residual policy or its exact zero-action baseline."""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--checkpoint", type=str, default=None)
p.add_argument("--zero_policy", action="store_true")
p.add_argument("--realtime", action="store_true")
p.add_argument("--seed", type=int, default=17)
AppLauncher.add_app_launcher_args(p)
a, unknown = p.parse_known_args()
sys.argv = [sys.argv[0], *unknown]
if not a.zero_policy and not a.checkpoint:
    p.error("--checkpoint is required unless --zero_policy is selected")

launcher = AppLauncher(a)
simulation_app = launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import ID_FD_pipline.tasks  # noqa: F401,E402
from ID_FD_pipline.residual_policy.env_cfg import ResidualPolicyPPORunnerCfg
from isaaclab_tasks.utils import parse_env_cfg


def main() -> None:
    task = "Kevin-Mulag-IDFD-Residual-PLAY-v0"
    cfg = parse_env_cfg(task, device=a.device, num_envs=1)
    cfg.seed = int(a.seed)
    cfg.force_zero_policy = bool(a.zero_policy)
    cfg.episode_length_s = 200.0
    gym_env = gym.make(task, cfg=cfg)
    raw = gym_env.unwrapped
    agent_cfg = ResidualPolicyPPORunnerCfg()
    agent_cfg.device = a.device
    env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)

    policy = None
    if not a.zero_policy:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        checkpoint = str(Path(a.checkpoint).expanduser().resolve())
        runner.load(checkpoint)
        policy = runner.get_inference_policy(device=raw.device)
        print(f"[PLAY] loaded {checkpoint}", flush=True)

    obs, _ = env.reset()
    dt = float(raw.step_dt)
    step = 0
    try:
        while simulation_app.is_running():
            started = time.perf_counter()
            with torch.inference_mode():
                action = (
                    torch.zeros((1, 3), device=raw.device)
                    if policy is None
                    else policy(obs)
                )
                obs, _, dones, _ = env.step(action)
            if step % 30 == 0:
                error = torch.linalg.norm(
                    raw._original_reference_tool[0] - raw._compute_ee_pos_local()[0]
                ).item()
                print(
                    f"step={step:05d} error={error:.4f}m "
                    f"residual={raw._residual_filtered[0].cpu().numpy().round(5)} "
                    f"head={raw._measure_head_down_alignment()[0].item():.5f} "
                    f"drift={raw._truck_drift_xy[0].item():.6f}m",
                    flush=True,
                )
            if dones.any():
                print("[PLAY] episode reset", flush=True)
            step += 1
            if a.realtime:
                time.sleep(max(0.0, dt - (time.perf_counter() - started)))
    finally:
        env.close()


try:
    main()
except BaseException:
    traceback.print_exc()
    raise
finally:
    simulation_app.close()
