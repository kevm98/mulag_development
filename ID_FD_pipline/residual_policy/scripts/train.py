"""Train the stationary-truck residual PPO policy with RSL-RL."""

from __future__ import annotations

import argparse
import sys
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--num_envs", type=int, default=64)
p.add_argument("--max_iterations", type=int, default=2000)
p.add_argument("--seed", type=int, default=17)
p.add_argument("--curriculum_stage", type=int, default=1, choices=[1, 2, 3])
p.add_argument("--residual_scale_m", type=float, default=None)
p.add_argument("--skip_contract_check", action="store_true")
p.add_argument("--checkpoint", type=str, default=None)
AppLauncher.add_app_launcher_args(p)
a, unknown = p.parse_known_args()
sys.argv = [sys.argv[0], *unknown]

launcher = AppLauncher(a)
simulation_app = launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import ID_FD_pipline.tasks  # noqa: F401,E402
from ID_FD_pipline.residual_policy.env_cfg import ResidualPolicyPPORunnerCfg
from isaaclab_tasks.utils import parse_env_cfg


def _apply_curriculum(cfg) -> None:
    stage = int(a.curriculum_stage)
    if stage == 1:
        cfg.path_speed_mps, cfg.residual_scale_m = 0.10, 0.001
    elif stage == 2:
        cfg.path_speed_mps, cfg.residual_scale_m = 0.13, 0.002
    else:
        cfg.path_speed_mps, cfg.residual_scale_m = 0.13, 0.004
    if a.residual_scale_m is not None:
        cfg.residual_scale_m = float(a.residual_scale_m)
    cfg.curriculum_stage = stage
    cfg.force_zero_policy = False


@torch.no_grad()
def _zero_action_preflight(raw_env) -> None:
    """Exercise deployment plus scored tracking before PPO gets control."""
    raw_env.cfg.force_zero_policy = True
    hard_band = float(raw_env.cfg.cutter_clearance_hard_band_m)
    # Measure the complete short envelope before deciding. Early auto-reset
    # destroys the failing state and only reports how many envs crossed it.
    raw_env.cfg.cutter_clearance_hard_band_m = 10.0
    raw_env.reset(seed=int(a.seed))
    phase = raw_env._traj_phase.detach()
    initial_clearance = raw_env._compute_cutter_world_z().detach()
    initial_head = raw_env._measure_head_down_alignment().detach()
    print(
        f"[PREFLIGHT] initial phase=[{phase.min().item():.3f}, {phase.max().item():.3f}] "
        f"clearance=[{initial_clearance.min().item():.5f}, "
        f"{initial_clearance.max().item():.5f}]m "
        f"head_min={initial_head.min().item():.6f}",
        flush=True,
    )
    steps = 100
    zero = torch.zeros((raw_env.num_envs, 3), device=raw_env.device)
    max_clearance = torch.zeros(raw_env.num_envs, device=raw_env.device)
    max_tool_error = torch.zeros_like(max_clearance)
    for _ in range(steps):
        _, _, terminated, _, extras = raw_env.step(zero)
        if terminated.any():
            raw_env.cfg.cutter_clearance_hard_band_m = hard_band
            log = extras.get("log", {})
            diagnostic = {
                key: float(value.detach().cpu().item()) if torch.is_tensor(value) else value
                for key, value in log.items()
                if key.startswith("termination/")
            }
            raise RuntimeError(
                "zero-action preflight terminated during tracking: "
                f"{diagnostic}"
            )
        max_clearance = torch.maximum(max_clearance, raw_env._compute_cutter_world_z().abs())
        max_tool_error = torch.maximum(
            max_tool_error,
            torch.linalg.norm(
                raw_env._original_reference_tool - raw_env._compute_ee_pos_local(), dim=-1
            ),
        )
    raw_env.cfg.cutter_clearance_hard_band_m = hard_band
    max_residual = float(raw_env._residual_filtered.abs().max().item())
    max_drift = float(raw_env._max_truck_drift.max().item())
    if max_residual > 1.0e-7 or max_drift > 0.02:
        raise RuntimeError(
            f"zero-action contract failed: residual={max_residual:.3e}, drift={max_drift:.6f}m"
        )
    measured_clearance = float(max_clearance.max().item())
    measured_error = float(max_tool_error.max().item())
    print(
        f"[PREFLIGHT] envelope clearance_max={measured_clearance:.5f}m "
        f"tool_error_max={measured_error:.5f}m", flush=True
    )
    if measured_clearance > hard_band:
        raise RuntimeError(
            f"zero-action clearance envelope {measured_clearance:.5f}m exceeds "
            f"hard band {hard_band:.5f}m"
        )
    print(f"[PREFLIGHT] PASS residual={max_residual:.3e} drift={max_drift:.6f}m", flush=True)
    raw_env.cfg.force_zero_policy = False
    raw_env.reset(seed=int(a.seed))


def main() -> None:
    task = "Kevin-Mulag-IDFD-Residual-v0"
    cfg = parse_env_cfg(task, device=a.device, num_envs=int(a.num_envs))
    cfg.seed = int(a.seed)
    _apply_curriculum(cfg)

    agent_cfg = ResidualPolicyPPORunnerCfg()
    agent_cfg.seed = int(a.seed)
    agent_cfg.device = a.device
    agent_cfg.max_iterations = int(a.max_iterations)

    log_dir = ROOT / "logs/rsl_rl/id_fd_residual" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    cfg.log_dir = str(log_dir)
    print(
        f"[TRAIN] stage={a.curriculum_stage} scale={cfg.residual_scale_m:.4f}m "
        f"speed={cfg.path_speed_mps:.3f}m/s log={log_dir}", flush=True
    )

    gym_env = gym.make(task, cfg=cfg)
    try:
        if not a.skip_contract_check:
            _zero_action_preflight(gym_env.unwrapped)
        env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device)
        if a.checkpoint:
            runner.load(str(Path(a.checkpoint).expanduser().resolve()))
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=False)
        print(f"[TRAIN] complete iterations={agent_cfg.max_iterations} log={log_dir}", flush=True)
    finally:
        gym_env.close()


try:
    main()
except BaseException:
    traceback.print_exc()
    raise
finally:
    simulation_app.close()
