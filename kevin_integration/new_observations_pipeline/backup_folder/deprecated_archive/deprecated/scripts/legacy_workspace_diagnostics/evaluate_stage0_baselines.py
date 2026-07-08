#!/usr/bin/env python3
"""Evaluate Stage-0 local-reach baselines before PPO training."""
from __future__ import annotations
import argparse, sys, os
from pathlib import Path
import numpy as np, torch
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Stage-0 baseline evaluation.")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--num_episodes", type=int, default=200)
parser.add_argument("--ppo_stage", type=str, default="local_reach_stage0")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env_cfg import MulagUpperLevelEnvCfg
from kevin_integration.new_observations_pipeline.ppo1.horizon_idfd_env_wrapper import HorizonIDFDEnv
from kevin_integration.new_observations_pipeline.ppo1.rewards_adapter import compute_tracking_error
from kevin_integration.utils.sim_memory import apply_kevin_sim_memory_optimizations

cfg = MulagUpperLevelEnvCfg()
cfg.scene.num_envs = int(args_cli.num_envs)
cfg.enable_scripted_vehicle_motion = False
cfg.enable_startup_summary = False
cfg.log_target_delta_xdot_clipping = False
cfg.horizon_pin_root_pose = False
cfg.horizon_reset_settle_steps = 0
if getattr(args_cli, "device", None):
    cfg.device = args_cli.device
    cfg.sim.device = args_cli.device
id_path = (
    Path(os.getcwd())
    / "outputs/cylinder_diagnostics/paper_matched_pipeline/model_finetune/id_physx_finetuned_best.pt"
).resolve()
cfg.id_checkpoint_override_path = str(id_path)
apply_kevin_sim_memory_optimizations(cfg.sim)

env = HorizonIDFDEnv(
    base_cfg=cfg,
    horizon_steps=10,
    action_scale_m=0.01,
    fd_gains=[2.0, 3.0, 3.0, 3.0],
    device=args_cli.device,
)
uw = env._uw
B = env.num_envs


def make_action_fn(mode, action_scale=0.01):
    """Return action function for given mode."""
    if mode == "zero":
        def fn():
            return torch.zeros((B, 4), device=env.device)
        return fn
    elif mode == "random":
        def fn():
            return 2.0 * torch.rand((B, 4), device=env.device) - 1.0
        return fn
    elif mode == "greedy":
        def fn():
            ee_pos_b = env._tcp_pos_base_frame()
            target_b = uw._command_pos_w[:, :3]
            err = target_b - ee_pos_b
            a = torch.clamp(err / action_scale, -1.0, 1.0)
            return torch.nan_to_num(a, nan=0.0)
        return fn
    elif mode == "random_scaled":
        def fn(scale=action_scale):
            a = 2.0 * torch.rand((B, 4), device=env.device) - 1.0
            return a
        return fn
    raise ValueError(f"Unknown mode: {mode}")


def run_episodes(name, action_fn, num_eps, action_scale=0.01):
    """Collect episode stats from the wrapper's step output."""
    results = []
    episodes_done = 0
    env.reset()

    # Re-initialize action function if it needs the current scale
    if callable(action_fn):
        fn = action_fn
    else:
        fn = make_action_fn(name, action_scale)

    while episodes_done < num_eps:
        action = fn() if not callable(fn) or name != "random_scaled" else make_action_fn("random", action_scale)()
        if name == "random_scaled":
            action = 2.0 * torch.rand((B, 4), device=env.device) - 1.0

        obs, reward, terminated, truncated, extras = env.step(action)
        done = terminated | truncated

        if done.any():
            done_indices = done.nonzero(as_tuple=False).squeeze(-1)
            if done_indices.ndim == 0:
                done_indices = done_indices.unsqueeze(0)
            for eid in done_indices:
                if episodes_done >= num_eps:
                    break
                ee = compute_tracking_error(uw)[eid].item()
                ws_v, cab_v = env._workspace_guard_components()
                guard_frac = float(env._last_guard_fraction[eid].item())
                act_norm = float(torch.norm(action[eid]).item())
                clip_frac = float(extras.get("reward_terms", {}).get("clipping_fraction", torch.zeros(1))[0].item()) if eid.item() < 1 else 0.0

                results.append({
                    "final_ee_error": ee,
                    "workspace_violation": bool(ws_v[eid].item()),
                    "cab_violation": bool(cab_v[eid].item()),
                    "guard_fraction": guard_frac,
                    "action_norm": act_norm,
                    "clipping_fraction": clip_frac,
                })
                episodes_done += 1

    if not results:
        return {}

    n = len(results)
    final_errs = [r["final_ee_error"] for r in results]
    ws_rate = sum(1 for r in results if r["workspace_violation"]) / n
    cab_rate = sum(1 for r in results if r["cab_violation"]) / n
    guard_frac = sum(r["guard_fraction"] for r in results) / n
    act_norm = sum(r["action_norm"] for r in results) / n
    clip_frac = sum(r["clipping_fraction"] for r in results) / n

    return {
        "name": name,
        "episodes": n,
        "mean_final_ee_error": sum(final_errs) / n,
        "min_ee_error": min(final_errs),
        "max_ee_error": max(final_errs),
        "workspace_violation_rate": ws_rate,
        "cab_violation_rate": cab_rate,
        "mean_guard_fraction": guard_frac,
        "mean_action_norm": act_norm,
        "mean_clip_fraction": clip_frac,
        "final_errs": final_errs,
    }


print("=" * 70, flush=True)
print("STAGE-0 BASELINE EVALUATION", flush=True)
print("=" * 70, flush=True)
num_eps = int(args_cli.num_episodes)

print("\n--- Zero Action ---", flush=True)
r_zero = run_episodes("zero", make_action_fn("zero"), num_eps)
for k, v in r_zero.items():
    if k != "final_errs":
        print(f"  {k}: {v}", flush=True)

print("\n--- Random Action ---", flush=True)
r_rand = run_episodes("random", make_action_fn("random"), num_eps)
for k, v in r_rand.items():
    if k != "final_errs":
        print(f"  {k}: {v}", flush=True)

print("\n--- Greedy Action ---", flush=True)
r_greedy = run_episodes("greedy", make_action_fn("greedy"), num_eps)
for k, v in r_greedy.items():
    if k != "final_errs":
        print(f"  {k}: {v}", flush=True)

print("\n" + "=" * 70, flush=True)
print("SUMMARY", flush=True)
print("=" * 70, flush=True)
for r in [r_zero, r_rand, r_greedy]:
    if not r:
        continue
    print(f"\n{r['name']}:", flush=True)
    print(f"  final_ee_error: {r['mean_final_ee_error']:.3f}m (min={r['min_ee_error']:.3f}, max={r['max_ee_error']:.3f})", flush=True)
    print(f"  workspace_violation_rate: {r['workspace_violation_rate']*100:.1f}%", flush=True)
    print(f"  cab_violation_rate: {r['cab_violation_rate']*100:.1f}%", flush=True)
    print(f"  mean_guard_fraction: {r['mean_guard_fraction']:.4f}", flush=True)
    print(f"  mean_action_norm: {r['mean_action_norm']:.4f}", flush=True)
    print(f"  mean_clip_fraction: {r['mean_clip_fraction']:.4f}", flush=True)

initial_ok = r_zero.get("mean_final_ee_error", 99) < 0.5
zero_ok = r_zero.get("workspace_violation_rate", 1) < 0.3
rand_ok = r_rand.get("workspace_violation_rate", 1) < 0.5
print(f"\nInitial EE < 0.5m: {initial_ok}", flush=True)
print(f"Zero action WS violation < 30%: {zero_ok}", flush=True)
print(f"Random action WS violation < 50%: {rand_ok}", flush=True)
print(f"Stage 0 valid: {initial_ok and zero_ok and rand_ok}", flush=True)

env.close()
simulation_app.close()
