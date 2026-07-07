#!/usr/bin/env python3
"""Behavior cloning: train actor to mimic greedy mapper."""
from __future__ import annotations
import argparse, sys, os
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="BC train actor on greedy mapper.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_samples", type=int, default=500)
parser.add_argument("--num_epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--output", type=str, default="outputs/bc_actor_yz.pt")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env_cfg import MulagUpperLevelEnvCfg
from kevin_integration.new_observations_pipeline.ppo1.stage0_yz_env_wrapper import Stage0YZEnvWrapper

cfg = MulagUpperLevelEnvCfg()
env = Stage0YZEnvWrapper(cfg, horizon_steps=10, action_scale_m=0.05, device=args_cli.device)
B = env.num_envs

print(f"Collecting {args_cli.num_samples} (obs, greedy_action) pairs...", flush=True)

data_obs = []; data_act = []

for i in range(int(args_cli.num_samples)):
    if i % 100 == 0: print(f"  sample {i}/{args_cli.num_samples}", flush=True)
    env.reset()
    for step in range(5):  # collect multiple states per episode
        obs = env._get_obs()[0].detach().cpu().numpy()
        ee = env._tcp_yz()[0]
        err = env._target_yz[0] - ee
        greedy_act = torch.clamp(err / env._action_scale, -1.0, 1.0).detach().cpu().numpy()
        data_obs.append(obs); data_act.append(greedy_act)
        # Step with greedy to get next state
        obs_td, reward, dones, extras = env.step(torch.from_numpy(greedy_act).float().unsqueeze(0).to(args_cli.device))
        if dones[0].item(): break

data_obs = torch.tensor(np.array(data_obs), dtype=torch.float32)
data_act = torch.tensor(np.array(data_act), dtype=torch.float32)
print(f"Collected {len(data_obs)} pairs", flush=True)

# Build actor network (same as PPO actor: 6->64->32->2 with ELU)
class ActorMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(6, 64), torch.nn.ELU(),
            torch.nn.Linear(64, 32), torch.nn.ELU(),
            torch.nn.Linear(32, 2),
        )
    def forward(self, x): return self.mlp(x)

model = ActorMLP()
opt = torch.optim.Adam(model.parameters(), lr=float(args_cli.lr))
loss_fn = torch.nn.MSELoss()

n_epochs = int(args_cli.num_epochs)
bs = int(args_cli.batch_size)
N = len(data_obs)

print(f"Training {n_epochs} epochs, batch={bs}...", flush=True)
for epoch in range(n_epochs):
    perm = torch.randperm(N)
    total_loss = 0.0
    for i in range(0, N, bs):
        idx = perm[i:i+bs]
        pred = model(data_obs[idx])
        loss = loss_fn(pred, data_act[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        total_loss += loss.item() * len(idx)
    total_loss /= N
    if epoch % 20 == 0:
        print(f"  epoch {epoch}: loss={total_loss:.6f}", flush=True)

print(f"Final loss: {total_loss:.6f}", flush=True)

# Evaluate BC actor
print(f"\nEvaluating BC actor...", flush=True)
env.reset()
with torch.no_grad():
    for _ in range(10):
        obs = env._get_obs()
        bc_act = model(obs).clamp(-1.0, 1.0)
        obs_td, reward, dones, extras = env.step(bc_act)
        if dones[0].item():
            env.reset()

# Compare vs greedy on 20 episodes
dist_bc = []; dist_greedy = []; cosines = []
for ep in range(20):
    env.reset()
    for step in range(10):
        obs = env._get_obs()
        ee = env._tcp_yz()[0]
        err = env._target_yz[0] - ee
        g_act = torch.clamp(err / env._action_scale, -1.0, 1.0)
        with torch.no_grad():
            b_act = model(obs).clamp(-1.0, 1.0)
        cos = float(torch.nn.functional.cosine_similarity(
            g_act.flatten(), b_act.flatten(), dim=0).item())
        cosines.append(cos)
        # Run greedy
        env._prev_action = g_act
        dc = torch.clamp((env._J_pinv @ (g_act * env._action_scale).T).T, -0.005, 0.005)
        cqh = env._uw.robot.data.joint_pos[:, env._cyl_ids].clone()
        lim = env._uw.robot.data.soft_joint_pos_limits[:, env._cyl_ids]
        cqt = torch.clamp(cqh + dc, lim[..., 0], lim[..., 1])
        for _ in range(10):
            env._pin_root(); env._uw.set_debug_manual_cylinder_targets(cqt, tool_position_target=env._tool_target)
            env._env.step(torch.zeros((1,5), device=args_cli.device))
        g_dist = torch.norm(env._tcp_yz()[0] - env._target_yz[0]).item()
        dist_greedy.append(g_dist)

        # Reset and run BC
        env.reset()
        dc = torch.clamp((env._J_pinv @ (b_act * env._action_scale).T).T, -0.005, 0.005)
        cqh = env._uw.robot.data.joint_pos[:, env._cyl_ids].clone()
        cqt = torch.clamp(cqh + dc, lim[..., 0], lim[..., 1])
        for _ in range(10):
            env._pin_root(); env._uw.set_debug_manual_cylinder_targets(cqt, tool_position_target=env._tool_target)
            env._env.step(torch.zeros((1,5), device=args_cli.device))
        b_dist = torch.norm(env._tcp_yz()[0] - env._target_yz[0]).item()
        dist_bc.append(b_dist)

print(f"  BC mean_dist: {np.mean(dist_bc):.4f}m  Greedy mean_dist: {np.mean(dist_greedy):.4f}m", flush=True)
print(f"  Mean action cosine: {np.mean(cosines):.3f}", flush=True)

# Save
output_path = Path(args_cli.output); output_path.parent.mkdir(parents=True, exist_ok=True)
torch.save({"actor_state_dict": model.state_dict()}, output_path)
print(f"Saved BC actor to {output_path}", flush=True)

env.close(); simulation_app.close()
