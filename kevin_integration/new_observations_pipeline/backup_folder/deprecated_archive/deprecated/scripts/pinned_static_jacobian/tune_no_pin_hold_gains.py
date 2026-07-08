#!/usr/bin/env python3
"""Grid-search low-level hold gains without _pin_root.

Tests zero-action EE stability across gain configurations.
"""
from __future__ import annotations
import argparse, sys, itertools
import numpy as np, torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Grid-search no-pin hold gains.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--diagnostic_steps", type=int, default=20)
parser.add_argument("--settle_steps", type=int, default=10)
parser.add_argument("--action_scale_m", type=float, default=0.05)
parser.add_argument("--quick", action="store_true", help="Test only 3 promising configs")
parser.add_argument("--top_n", type=int, default=10, help="Show top N results")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env_cfg import MulagUpperLevelEnvCfg
from kevin_integration.new_observations_pipeline.ppo1.stage0_yz_env_wrapper import Stage0YZEnvWrapper
from kevin_integration.utils.sim_memory import apply_kevin_sim_memory_optimizations

device = args_cli.device
diag_steps = int(args_cli.diagnostic_steps)
settle_s = int(args_cli.settle_steps)
B = int(args_cli.num_envs)

# --- Grid definition ---
if args_cli.quick:
    horizon_grid = [10]
    kp_grid = [0.1, 0.2, 0.5]
    kd_grid = [0.5, 1.0]
    beta_grid = [0.8]
    fr_grid = [0.01]
else:
    horizon_grid = [1, 2, 5, 10]
    kp_grid = [0.0, 0.05, 0.1, 0.2, 0.5]
    kd_grid = [0.2, 0.5, 1.0, 2.0]
    beta_grid = [0.8, 0.9]
    fr_grid = [0.01, 0.02, 0.05]

all_configs = []
for H, kp, kd, beta, fr in itertools.product(horizon_grid, kp_grid, kd_grid, beta_grid, fr_grid):
    all_configs.append(dict(H=H, kp=kp, kd=kd, beta=beta, fr=fr))

total = len(all_configs)
print(f"[GRID] Testing {total} configs (quick={args_cli.quick})", flush=True)
print(f"[GRID] diagnostic_steps={diag_steps}, settle_steps={settle_s}", flush=True)

results = []
for idx, cfg_ov in enumerate(all_configs):
    H = cfg_ov["H"]
    kp = cfg_ov["kp"]
    kd = cfg_ov["kd"]
    beta = cfg_ov["beta"]
    fr = cfg_ov["fr"]

    # Build config
    cfg = MulagUpperLevelEnvCfg()
    cfg.scene.num_envs = B

    try:
        env = Stage0YZEnvWrapper(cfg, horizon_steps=H,
                                 action_scale_m=float(args_cli.action_scale_m),
                                 device=device,
                                 kp_scale=kp, kd_scale=kd,
                                 force_rate_limit=fr,
                                 velocity_filter_beta=beta)
    except Exception as e:
        print(f"[{idx+1}/{total}] H={H} kp={kp} kd={kd} beta={beta} fr={fr} -> ENV_ERROR: {e}", flush=True)
        results.append(dict(H=H, kp=kp, kd=kd, beta=beta, fr=fr,
                           mean_jump=float('inf'), mean_dist=float('inf'),
                           status='env_error'))
        continue

    # Inject _pin_root counter
    original_pin = env._pin_root
    pin_count = [0]
    env._pin_root = lambda: (pin_count.__setitem__(0, pin_count[0]+1), original_pin())[1]

    # Reset + settle
    env.reset()
    for _ in range(settle_s):
        env.step(torch.zeros((B, 2), device=device))
    env.resample_local_targets()

    jumps = []; dists = []; cyl_qerrs = []; cyl_qdots = []
    for _ in range(diag_steps):
        cqh = env._uw.robot.data.joint_pos[:, env._cyl_ids].clone()
        obs, reward, done, extras = env.step(torch.zeros((B, 2), device=device))
        cqa = env._uw.robot.data.joint_pos[:, env._cyl_ids].clone()
        cqv = env._uw.robot.data.joint_vel[:, env._cyl_ids].clone()
        jumps.append(extras["log"]["ee_jump"].item())
        dists.append(extras["log"]["distance"].item())
        cyl_qerrs.append(torch.abs(cqa - cqh).mean().item())
        cyl_qdots.append(torch.abs(cqv).mean().item())

    env.close()

    mean_j = float(np.mean(jumps))
    max_j = float(np.max(jumps))
    mean_d = float(np.mean(dists))
    max_d = float(np.max(dists))
    mean_qe = float(np.mean(cyl_qerrs))
    max_qe = float(np.max(cyl_qerrs))
    mean_qv = float(np.mean(cyl_qdots))
    max_qv = float(np.max(cyl_qdots))
    osc = float(np.std(jumps))

    results.append(dict(
        H=H, kp=kp, kd=kd, beta=beta, fr=fr,
        mean_jump=mean_j, max_jump=max_j,
        mean_dist=mean_d, max_dist=max_d,
        mean_cyl_qerr=mean_qe, max_cyl_qerr=max_qe,
        mean_cyl_qdot=mean_qv, max_cyl_qdot=max_qv,
        osc_index=osc, pin_calls=pin_count[0],
        status='ok'
    ))

    passed = "PASS" if mean_j < 0.05 and max_j < 0.10 else "FAIL"
    print(f"[{idx+1}/{total}] H={H} kp={kp:.2f} kd={kd:.1f} beta={beta} fr={fr:.3f}  "
          f"mean_jump={mean_j:.4f} max_jump={max_j:.4f}  mean_dist={mean_d:.4f}  "
          f"osc={osc:.4f}  {passed}  pin_calls={pin_count[0]}", flush=True)

# --- Sort and report ---
ok = [r for r in results if r['status'] == 'ok']
ok.sort(key=lambda r: (r['mean_jump'], r['max_jump'], r['osc_index']))

print(f"\n{'='*90}", flush=True)
print(f"TOP {min(args_cli.top_n, len(ok))} CONFIGS (sorted by mean_jump, max_jump, osc)", flush=True)
print(f"{'='*90}", flush=True)
print(f"{'H':>3s} {'kp':>5s} {'kd':>5s} {'beta':>5s} {'fr':>6s} {'mean_j':>8s} {'max_j':>8s} {'mean_d':>8s} {'max_d':>8s} {'osc':>8s} {'mean_qe':>9s} {'max_qe':>9s} {'pins':>5s}", flush=True)
for r in ok[:args_cli.top_n]:
    print(f"{r['H']:3d} {r['kp']:5.2f} {r['kd']:5.1f} {r['beta']:5.1f} {r['fr']:6.3f} "
          f"{r['mean_jump']:8.4f} {r['max_jump']:8.4f} {r['mean_dist']:8.4f} {r['max_dist']:8.4f} "
          f"{r['osc_index']:8.4f} {r['mean_cyl_qerr']:9.6f} {r['max_cyl_qerr']:9.6f} {r['pin_calls']:5d}",
          flush=True)

pass_count = sum(1 for r in ok if r['mean_jump'] < 0.05 and r['max_jump'] < 0.10)
strong_count = sum(1 for r in ok if r['mean_jump'] < 0.02 and r['mean_dist'] < 0.05)
print(f"\n  GATE pass (mean_jump<0.05 & max_jump<0.10): {pass_count}/{len(ok)}", flush=True)
print(f"  STRONG pass (mean_jump<0.02 & mean_dist<0.05): {strong_count}/{len(ok)}", flush=True)

simulation_app.close()
