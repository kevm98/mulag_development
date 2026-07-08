#!/usr/bin/env python3
"""Sweep action scale for Stage-0 random-action safety.

Runs random-action baseline at different action scales and reports
guard fraction, clipping, failure rate for each.
"""
from __future__ import annotations
import argparse, sys, os
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Action scale sweep for Stage-0.")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--num_episodes", type=int, default=100)
parser.add_argument("--scales", type=str, default="0.0025,0.005,0.0075,0.01")
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

scales = [float(x.strip()) for x in args_cli.scales.split(",")]
num_eps = int(args_cli.num_episodes)
results_all = {}

for scale in scales:
    print(f"\n{'='*60}", flush=True)
    print(f"ACTION SCALE: {scale}", flush=True)
    print(f"{'='*60}", flush=True)

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
        action_scale_m=scale,
        fd_gains=[2.0, 3.0, 3.0, 3.0],
        device=args_cli.device,
    )
    uw = env._uw
    B = env.num_envs

    results = []
    episodes_done = 0
    env.reset()

    while episodes_done < num_eps:
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
                results.append({
                    "final_ee_error": ee,
                    "workspace_violation": bool(ws_v[eid].item()),
                    "cab_violation": bool(cab_v[eid].item()),
                    "guard_fraction": guard_frac,
                    "action_norm": act_norm,
                })
                episodes_done += 1

    if results:
        n = len(results)
        final_errs = [r["final_ee_error"] for r in results]
        ws_rate = sum(1 for r in results if r["workspace_violation"]) / n
        guard_frac = sum(r["guard_fraction"] for r in results) / n
        act_norm = sum(r["action_norm"] for r in results) / n

        rpt = {
            "scale": scale,
            "episodes": n,
            "mean_final_ee_error": sum(final_errs) / n,
            "min_ee_error": min(final_errs),
            "max_ee_error": max(final_errs),
            "ws_violation_rate": ws_rate,
            "mean_guard_fraction": guard_frac,
            "mean_action_norm": act_norm,
        }
        results_all[scale] = rpt
        print(f"  final_ee_error: {rpt['mean_final_ee_error']:.3f}m (min={rpt['min_ee_error']:.3f}, max={rpt['max_ee_error']:.3f})", flush=True)
        print(f"  ws_violation_rate: {rpt['ws_violation_rate']*100:.1f}%", flush=True)
        print(f"  mean_guard_fraction: {rpt['mean_guard_fraction']:.4f}", flush=True)
        print(f"  mean_action_norm: {rpt['mean_action_norm']:.4f}", flush=True)

    env.close()

# Summary
print(f"\n{'='*60}", flush=True)
print("SWEEP SUMMARY", flush=True)
print(f"{'='*60}", flush=True)
print(f"{'Scale':>8s}  {'EE err':>7s}  {'WS rate':>7s}  {'Guard':>7s}  {'ActNorm':>7s}  {'Pass':>5s}", flush=True)
print("-" * 50, flush=True)

best_scale = None
for scale in sorted(results_all.keys()):
    r = results_all[scale]
    pass_fail = (r["ws_violation_rate"] < 0.20 and r["mean_guard_fraction"] < 0.15)
    marker = "PASS" if pass_fail else "FAIL"
    print(f"{r['scale']:8.4f}  {r['mean_final_ee_error']:7.3f}  {r['ws_violation_rate']:7.3f}  {r['mean_guard_fraction']:7.4f}  {r['mean_action_norm']:7.4f}  {marker:>5s}", flush=True)
    if pass_fail and best_scale is None:
        best_scale = scale

if best_scale:
    print(f"\nRecommended action scale: {best_scale} (largest passing scale)", flush=True)
else:
    print(f"\nNo scale passed. Use 0.0025.", flush=True)
    best_scale = 0.0025

simulation_app.close()
