#!/usr/bin/env python3
"""Score a trained residual arm, deterministically, and dump per-env errors.

`evaluate_gates.py` cannot load these checkpoints: its actor is a plain mlp at
the base observation width, while these are 1470 wide with a residual spliced in
at ``mlp``. So the arms are scored here, with the SAME plant configuration in
both arms and per-env errors written out for paired analysis.

Deterministic actions (the distribution mean, no sampling): the comparison is
between two policies, and sampling noise would only widen both arms' intervals.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

p = argparse.ArgumentParser()
p.add_argument("base_checkpoint", type=Path)
p.add_argument("--residual", type=Path, required=True)
p.add_argument("--report", type=Path, required=True)
p.add_argument("--num_envs", type=int, default=176)
p.add_argument("--steps", type=int, default=900)
p.add_argument("--mesh", type=int, default=1337)
p.add_argument("--seed", type=int, default=17)
p.add_argument("--history", type=int, default=20)
p.add_argument("--latent_dim", type=int, default=8)
p.add_argument("--alpha", type=float, default=0.5)
p.add_argument("--stage", type=int, default=4)
p.add_argument("--task", type=str, default="RRLAB-Mulag-Verge-ID-AAM-FD-v0")
p.add_argument("--no_history", action="store_true")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()

launcher = AppLauncher(a)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402

import mulag_plant.tasks  # noqa: F401,E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from PPO_AAM_FD.env_cfg import PpoIDAAMFDPPORunnerCfg  # noqa: E402
from PPO_ROM.terrain_env_cfg import apply_c4_authority_stage  # noqa: E402
from verge_task.residual_adapter import ResidualMLP  # noqa: E402


def main() -> int:
    env_cfg = parse_env_cfg(a.task, device=a.device, num_envs=a.num_envs)
    apply_c4_authority_stage(env_cfg, int(a.stage))
    env_cfg.terrain_base_seed = int(a.mesh)
    base_width = int(env_cfg.observation_space)
    action_dim = int(env_cfg.action_space)
    base_dim = base_width - 4 if base_width in (70, 72) else base_width

    env = gym.make(a.task, cfg=env_cfg, render_mode=None)
    raw = env.unwrapped
    runner_cfg = PpoIDAAMFDPPORunnerCfg()
    wrapped = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)

    sd = torch.load(a.residual, map_location=a.device, weights_only=False)["actor_state_dict"]
    import torch.nn as nn
    bm = nn.Sequential(
        nn.Linear(sd["mlp.base_mlp.0.weight"].shape[1], sd["mlp.base_mlp.0.weight"].shape[0]), nn.ELU(),
        nn.Linear(sd["mlp.base_mlp.2.weight"].shape[1], sd["mlp.base_mlp.2.weight"].shape[0]), nn.ELU(),
        nn.Linear(sd["mlp.base_mlp.4.weight"].shape[1], sd["mlp.base_mlp.4.weight"].shape[0]), nn.ELU(),
        nn.Linear(sd["mlp.base_mlp.6.weight"].shape[1], sd["mlp.base_mlp.6.weight"].shape[0]),
    ).to(a.device)
    residual = ResidualMLP(bm, base_dim=base_dim, base_width=base_width,
                           action_dim=action_dim, history=int(a.history),
                           latent_dim=int(a.latent_dim), alpha=float(a.alpha)).to(a.device)
    residual.load_state_dict(
        {k[len("mlp."):]: v for k, v in sd.items() if k.startswith("mlp.")}, strict=False)
    residual.eval()

    # rsl_rl's EmpiricalNormalization is (x - mean) / (std + eps) with
    # eps = 1e-2 -- NOT a division by std alone. These checkpoints have channels
    # with std as low as 8.685e-4, where the two differ by 12.5x, so omitting eps
    # feeds the policy wildly mis-scaled low-variance channels. That is what made
    # the frozen base read 175 mm here against 7.6 mm through the gate.
    NORM_EPS = 1.0e-2
    mean = sd["obs_normalizer._mean"].to(a.device).reshape(-1)
    std = sd["obs_normalizer._std"].to(a.device).reshape(-1)

    n = int(a.num_envs)
    hist = torch.zeros((n, int(a.history), base_dim + action_dim), device=a.device)
    prev = torch.zeros((n, action_dim), device=a.device)

    obs_td = wrapped.get_observations()
    key = next(iter(obs_td.keys()))
    lat_sum = torch.zeros(n, device=a.device)
    cut_sum = torch.zeros(n, device=a.device)
    active_count = torch.zeros(n, device=a.device)
    count = 0

    for _ in range(int(a.steps)):
        base_obs = obs_td[key]
        step_feat = torch.nan_to_num(
            torch.cat((base_obs[:, :base_dim], prev), dim=-1), nan=0.0,
            posinf=0.0, neginf=0.0)
        hist = torch.roll(hist, shifts=-1, dims=1)
        hist[:, -1, :] = step_feat
        flat = torch.zeros_like(hist.flatten(1)) if a.no_history else hist.flatten(1)
        wide = torch.cat((base_obs, flat), dim=-1)
        with torch.no_grad():
            full_mean = torch.cat((mean[:base_width],
                                   torch.zeros(flat.shape[1], device=a.device)))
            full_std = torch.cat((std[:base_width],
                                  torch.ones(flat.shape[1], device=a.device)))
            normalized = (wide - full_mean) / (full_std + NORM_EPS)
            action = residual(normalized)
            action = torch.clamp(action, -1.0, 1.0)
        obs_td, _, dones, _ = wrapped.step(action)
        prev = torch.nan_to_num(action.detach().clone(), nan=0.0)
        if dones is not None and bool(dones.any()):
            mask = dones.bool()
            hist[mask] = 0.0
            prev[mask] = 0.0
        # Count ONLY active steps, matching `roll_mesh`'s
        # `active = ~raw._truck_path_complete`. After a replica finishes its
        # path it parks, and its error against the reference becomes large and
        # meaningless; averaging those steps in inflated the frozen base from
        # 7.6 mm to 324 mm and made the arms incomparable to every other number
        # in this project.
        active = (~raw._truck_path_complete).float()
        lat_sum += raw.verge_lateral_error().abs() * active
        cut_sum += raw.cut_height_error().abs() * active
        active_count += active

    report = {
        "residual": str(a.residual),
        "no_history": bool(a.no_history),
        "mesh": int(a.mesh),
        "num_envs": n,
        "steps": int(a.steps),
        "mean_active_steps": float(active_count.mean()),
        "alpha": float(a.alpha),
        "per_env_lateral_m": (lat_sum / active_count.clamp_min(1.0)).cpu().tolist(),
        "per_env_cut_error_m": (cut_sum / active_count.clamp_min(1.0)).cpu().tolist(),
    }
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, indent=2))
    print(f"[eval-residual] wrote {a.report}", flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    raise SystemExit(code)
