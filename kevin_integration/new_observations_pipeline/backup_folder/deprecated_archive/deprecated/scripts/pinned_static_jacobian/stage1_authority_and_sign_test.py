#!/usr/bin/env python3
"""Stage-1 authority sweep: find config that passes +Z, action sign 4/4, greedy."""
from __future__ import annotations
import argparse, sys, os, math, itertools
from pathlib import Path
from datetime import datetime
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--kp_values", type=str, default="0.1,0.2,0.3,0.5")
parser.add_argument("--kd_scale", type=float, default=0.5)
parser.add_argument("--fr_values", type=str, default="0.01,0.02")
parser.add_argument("--max_cylinder_delta_m", type=float, default=0.005)
parser.add_argument("--target_y_range", type=float, default=0.03)
parser.add_argument("--target_z_range", type=float, default=0.01)
parser.add_argument("--target_hold_steps", type=int, default=999999)
parser.add_argument("--post_reset_settle_steps", type=int, default=5)
parser.add_argument("--action_scale", type=float, default=0.05)
parser.add_argument("--single_config", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env_cfg import MulagUpperLevelEnvCfg
from kevin_integration.new_observations_pipeline.ppo1.stage1_yz_tracking_env_wrapper import Stage1YZTrackingEnvWrapper

device = args_cli.device
kp_list = [float(x) for x in args_cli.kp_values.split(",")]
fr_list = [float(x) for x in args_cli.fr_values.split(",")]

def build_env(kp, fr):
    cfg = MulagUpperLevelEnvCfg(); cfg.scene.num_envs = 1
    return Stage1YZTrackingEnvWrapper(cfg, horizon_steps=10, action_scale_m=args_cli.action_scale,
                                       device=device, kp_scale=kp, kd_scale=args_cli.kd_scale,
                                       force_rate_limit=fr, velocity_filter_beta=0.8,
                                       trajectory_mode="sequence",
                                       target_hold_steps=args_cli.target_hold_steps,
                                       post_reset_settle_steps=args_cli.post_reset_settle_steps)

def settle(env):
    env.reset()
    for _ in range(10):
        env.step(torch.zeros((1, 2), device=device))

def zero_hold_test(env):
    settle(env)
    ee0 = env._tcp_yz()
    env._target_yz = ee0.clone() + torch.tensor([[0.03, 0.01]], device=device)
    env._target_vel_yz.zero_()
    env._prev_distance = torch.norm(env._tcp_yz() - env._target_yz, dim=-1)
    jj = []
    for _ in range(100):
        _, _, _, ex = env.step(torch.zeros((1, 2), device=device))
        jj.append(ex["log"]["ee_jump"].item())
    return float(np.mean(jj)), float(np.max(jj))

def z_authority_test(env):
    settle(env)
    ee = env._tcp_yz()
    z_target = ee.clone() + torch.tensor([[0.0, args_cli.target_z_range]], device=device)

    results = {}
    for act_label, act in [("plus_z", torch.tensor([[0., 1.]], device=device)),
                            ("minus_z", torch.tensor([[0., -1.]], device=device)),
                            ("zero", torch.zeros((1, 2), device=device))]:
        settle(env)
        env._target_yz = z_target.clone()
        env._target_vel_yz.zero_()
        env._prev_distance = torch.norm(env._tcp_yz() - env._target_yz, dim=-1)
        ee_before = env._tcp_yz()[0].cpu().numpy().copy()
        progs = []; ee_dz = []; jumps = []
        for _ in range(10):
            _, _, _, ex = env.step(act)
            progs.append(ex["log"]["progress"].item())
            jumps.append(ex["log"]["ee_jump"].item())
        ee_after = env._tcp_yz()[0].cpu().numpy().copy()
        dz = ee_after[1] - ee_before[1]
        results[act_label] = {"prog": float(np.mean(progs)), "dz": dz,
                              "max_jump": float(np.max(jumps)) if jumps else 0}
    return results

def action_sign_test(env):
    targets = [("+Y", [0.03, 0.0]), ("-Y", [-0.03, 0.0]), ("+Z", [0.0, 0.01]), ("-Z", [0.0, -0.01])]
    actions = {"+Y": torch.tensor([[1., 0.]], device=device), "-Y": torch.tensor([[-1., 0.]], device=device),
               "+Z": torch.tensor([[0., 1.]], device=device), "-Z": torch.tensor([[0., -1.]], device=device),
               "zero": torch.zeros((1, 2), device=device)}
    opposite = {"+Y": "-Y", "-Y": "+Y", "+Z": "-Z", "-Z": "+Z"}

    passed = 0
    for tlabel, toff in targets:
        progs = {}
        for alabel, act in actions.items():
            settle(env)
            ee = env._tcp_yz()
            env._target_yz = ee + torch.tensor([toff], device=device)
            env._target_vel_yz.zero_()
            env._prev_distance = torch.norm(ee - env._target_yz, dim=-1)
            pp = []
            for _ in range(10):
                _, _, _, ex = env.step(act)
                pp.append(ex["log"]["progress"].item())
            progs[alabel] = float(np.mean(pp))
        ok = progs[tlabel] > progs[opposite[tlabel]] and progs[tlabel] > progs["zero"]
        if ok: passed += 1
    return passed

def greedy_vs_zero(env, n_rollouts=20):
    gw_dist = 0; gw_prog = 0
    for r in range(n_rollouts):
        settle(env)
        zd = []
        for _ in range(100):
            _, _, _, ex = env.step(torch.zeros((1, 2), device=device))
            zd.append(ex["log"]["distance"].item())
        settle(env)
        gd = []; gp = []
        for _ in range(100):
            ee = env._tcp_yz()
            err = env._target_yz - ee
            act = torch.clamp(err / args_cli.action_scale, -1.0, 1.0)
            _, _, _, ex = env.step(act)
            gd.append(ex["log"]["distance"].item())
            gp.append(ex["log"]["progress"].item())
        if float(np.mean(gd)) < float(np.mean(zd)): gw_dist += 1
        if float(np.mean(gp)) >= float(np.mean(zp)): gw_prog += 1
    return gw_dist, gw_prog

# ====================================================================
# Main sweep
# ====================================================================
configs = list(itertools.product(kp_list, fr_list))
if args_cli.single_config:
    configs = [(0.1, 0.01)]  # baseline only

print(f"SWEEP: {len(configs)} configs", flush=True)
best = None
results = []

for kp, fr in configs:
    print(f"\n--- kp={kp:.2f} fr={fr:.3f} ---", flush=True)
    try:
        env = build_env(kp, fr)
    except Exception as e:
        print(f"  ENV_ERROR: {e}", flush=True)
        continue

    # Zero hold
    mj, mxj = zero_hold_test(env)
    print(f"  zero_hold: mean_j={mj:.4f} max_j={mxj:.4f}", flush=True)
    if mxj > 0.08 or mj > 0.03:
        print(f"  SKIP: hold fail", flush=True)
        env.close(); continue

    # +Z authority
    za = z_authority_test(env)
    pz_ok = za["plus_z"]["prog"] > za["zero"]["prog"] and za["plus_z"]["prog"] > za["minus_z"]["prog"]
    dz_ok = za["plus_z"]["dz"] > za["zero"]["dz"]
    print(f"  +Z auth: pz_prog={za['plus_z']['prog']:.4f} z_prog={za['zero']['prog']:.4f} mz_prog={za['minus_z']['prog']:.4f}  pass={pz_ok and dz_ok}", flush=True)

    # Action sign
    sign_p = action_sign_test(env)
    print(f"  action_sign: {sign_p}/4", flush=True)

    results.append({"kp": kp, "fr": fr, "mj": mj, "mxj": mxj, "+Z_pass": pz_ok and dz_ok, "sign": sign_p})

    if pz_ok and dz_ok and sign_p == 4:
        best = {"kp": kp, "fr": fr}
        print(f"  *** BEST CONFIG FOUND ***", flush=True)
        break

    env.close()

env.close()
simulation_app.close()

print(f"\n=== RESULTS ===", flush=True)
for r in results:
    s = "BEST" if best and r["kp"] == best["kp"] and r["fr"] == best["fr"] else ""
    print(f"  kp={r['kp']:.2f} fr={r['fr']:.3f}: hold={r['mj']:.4f}/{r['mxj']:.4f} +Z={'OK' if r['+Z_pass'] else 'NO'} sign={r['sign']}/4 {s}", flush=True)

if best:
    print(f"\nBEST: kp={best['kp']} fr={best['fr']}", flush=True)
else:
    print(f"\nNO CONFIG PASSES ALL GATES", flush=True)
    sys.exit(1)
