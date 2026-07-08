#!/usr/bin/env python3
"""Strict Stage-1 control stack debug — must pass all gates before PPO."""
from __future__ import annotations
import argparse, sys, os, math
from pathlib import Path
from datetime import datetime
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--kp_scale", type=float, default=0.1)
parser.add_argument("--kd_scale", type=float, default=0.5)
parser.add_argument("--max_cylinder_delta_m", type=float, default=0.0025)
parser.add_argument("--force_rate_limit_fraction", type=float, default=0.01)
parser.add_argument("--velocity_filter_beta", type=float, default=0.8)
parser.add_argument("--target_hold_steps", type=int, default=50)
parser.add_argument("--target_y_range", type=float, default=0.03)
parser.add_argument("--target_z_range", type=float, default=0.01)
parser.add_argument("--stage0_checkpoint", type=str, default=None)
parser.add_argument("--sweep_hold", action="store_true", default=False)
parser.add_argument("--skip_sign_test", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env_cfg import MulagUpperLevelEnvCfg
from kevin_integration.new_observations_pipeline.ppo1.stage1_yz_tracking_env_wrapper import Stage1YZTrackingEnvWrapper

device = args_cli.device
FAILED = []

def gate(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    line = f"  {name}: {status}  {detail}"
    print(line, flush=True)
    if not condition:
        FAILED.append((name, detail))
    return condition

def build_env(kp=None, kd=None, fr=None):
    cfg = MulagUpperLevelEnvCfg()
    cfg.scene.num_envs = 1
    e = Stage1YZTrackingEnvWrapper(cfg, horizon_steps=10, action_scale_m=0.05, device=device,
                                    kp_scale=kp or args_cli.kp_scale,
                                    kd_scale=kd or args_cli.kd_scale,
                                    force_rate_limit=fr or args_cli.force_rate_limit_fraction,
                                    velocity_filter_beta=args_cli.velocity_filter_beta,
                                    trajectory_mode="sequence",
                                    target_hold_steps=args_cli.target_hold_steps)
    e._target_y_range = args_cli.target_y_range
    e._target_z_range = args_cli.target_z_range
    return e

print(f"{'='*60}", flush=True)
print(f"STAGE-1 CONTROL STACK DEBUG (strict)", flush=True)
print(f"kp={args_cli.kp_scale} kd={args_cli.kd_scale} cyl_delta={args_cli.max_cylinder_delta_m}", flush=True)
print(f"range Y±{args_cli.target_y_range} Z±{args_cli.target_z_range} hold={args_cli.target_hold_steps}", flush=True)

env = build_env()
B = 1
zero_action = torch.zeros((B, 2), device=device)

# ====================================================================
# TEST 1: RESET SANITY
# ====================================================================
print(f"\n{'='*60}", flush=True)
print(f"TEST 1: Reset / Target / Obs Sanity", flush=True)
print(f"{'='*60}", flush=True)
env.reset()
ee = env._tcp_yz()[0]; target = env._target_yz[0]; terr = target - ee
terr_n = torch.norm(terr).item()
tvel = env._target_vel_yz[0]; pa = env._prev_action[0]; obs = env._get_obs()[0]
print(f"  ee=({ee[0]:.4f},{ee[1]:.4f}) target=({target[0]:.4f},{target[1]:.4f})", flush=True)
print(f"  err=({terr[0]:.4f},{terr[1]:.4f}) err_norm={terr_n:.4f}", flush=True)
print(f"  tvel=({tvel[0]:.4f},{tvel[1]:.4f}) prev_a=({pa[0]:.3f},{pa[1]:.3f})", flush=True)
print(f"  obs=[{obs[0]:.4f},{obs[1]:.4f},{obs[2]:.4f},{obs[3]:.4f},{obs[4]:.4f},{obs[5]:.4f}]", flush=True)
max_err = math.sqrt(args_cli.target_y_range**2 + args_cli.target_z_range**2) + 0.01
t1_pass = all([
    gate("1a: err_norm <= max", terr_n <= max_err, f"{terr_n:.4f} <= {max_err:.4f}"),
    gate("1b: tvel = [0,0]", torch.allclose(tvel, torch.zeros(2, device=device), atol=1e-6)),
    gate("1c: prev_a = [0,0]", torch.allclose(pa, torch.zeros(2, device=device), atol=1e-6)),
    gate("1d: obs[0:2] = err", torch.allclose(obs[:2], terr, atol=1e-4)),
    gate("1e: obs[2:4] = [0,0]", torch.allclose(obs[2:4], tvel, atol=1e-4)),
    gate("1f: obs[4:6] = prev_a", torch.allclose(obs[4:6], pa, atol=1e-4)),
])

# ====================================================================
# TEST 2: ZERO-ACTION HOLD WITH SPIKE INVESTIGATION
# ====================================================================
print(f"\n{'='*60}", flush=True)
print(f"TEST 2: Zero-Action Hold (100 steps, spike tracking)", flush=True)
print(f"{'='*60}", flush=True)
env.reset()
for _ in range(10):
    env.step(zero_action)

history = []
for s in range(100):
    ee_b = env._tcp_yz()[0].clone()
    target_b = env._target_yz[0].clone()
    cyl_b = env._uw.robot.data.joint_pos[0, env._cyl_ids].clone()
    cylv_b = env._uw.robot.data.joint_vel[0, env._cyl_ids].clone()

    obs, reward, done, extras = env.step(zero_action)

    ee_a = env._tcp_yz()[0]
    target_a = env._target_yz[0]
    cyl_a = env._uw.robot.data.joint_pos[0, env._cyl_ids].clone()
    cylv_a = env._uw.robot.data.joint_vel[0, env._cyl_ids].clone()

    jump = extras["log"]["ee_jump"].item()
    dist = extras["log"]["distance"].item()
    prog = extras["log"]["progress"].item()
    done_flag = bool(done.item()) if hasattr(done, 'item') else bool(done)
    tc = not torch.allclose(target_b, target_a, atol=1e-4)

    target_ch = not torch.allclose(target_b, target_a, atol=1e-4)
    entry = {
        'step': s, 'jump': jump, 'dist': dist, 'prog': prog, 'done': done_flag,
        'target_changed': target_ch,
        'ee_b': ee_b, 'ee_a': ee_a, 'target_b': target_b, 'target_a': target_a,
        'cyl_b': cyl_b, 'cyl_a': cyl_a, 'cylv_b': cylv_b, 'cylv_a': cylv_a,
        'target_changed': False,
    }
    history.append(entry)

    if jump > 0.06:
        print(f"\n[HOLD_SPIKE] step={s} jump={jump:.4f}m dist={dist:.4f}m done={done_flag}", flush=True)
        print(f"  ee_before=({ee_b[0]:.4f},{ee_b[1]:.4f}) ee_after=({ee_a[0]:.4f},{ee_a[1]:.4f})", flush=True)
        print(f"  target_before=({target_b[0]:.4f},{target_b[1]:.4f}) target_after=({target_a[0]:.4f},{target_a[1]:.4f})", flush=True)
        print(f"  cyl_before={cyl_b.tolist()} cyl_after={cyl_a.tolist()}", flush=True)
        print(f"  cylv_before={cylv_b.tolist()} cylv_after={cylv_a.tolist()}", flush=True)
        # Print previous 3 steps
        for prev in range(max(0, s-3), s):
            h = history[prev]
            print(f"  prev_step_{prev}: jump={h['jump']:.4f} dist={h['dist']:.4f} done={h['done']}", flush=True)
        # Print next step info (will fill in below)

jumps = [h['jump'] for h in history]
dists = [h['dist'] for h in history]
progs = [h['prog'] for h in history]
mean_j = float(np.mean(jumps)); max_j = float(np.max(jumps))
mean_d = float(np.mean(dists)); mean_p = float(np.mean(progs))
p95_j = float(np.percentile(jumps, 95)); p99_j = float(np.percentile(jumps, 99))
spikes = [h for h in history if h['jump'] > 0.06]

print(f"\n  Summary (200 steps):", flush=True)
print(f"  mean_jump={mean_j:.4f} max_jump={max_j:.4f} p95={p95_j:.4f} p99={p99_j:.4f}", flush=True)
print(f"  mean_dist={mean_d:.4f} mean_prog={mean_p:.4f}", flush=True)
print(f"  spikes > 0.06m: {len(spikes)}", flush=True)

# Classify spike sources
spike_sources = []
for h in spikes:
    s = h['step']
    target_changed = not torch.allclose(h['target_b'], h['target_a'], atol=1e-4)
    if h['done']:
        spike_sources.append(f"step_{s}: termination/reset (done=True)")
    elif h['target_changed']:
        spike_sources.append(f"step_{s}: target switch (target changed)")
    else:
        spike_sources.append(f"step_{s}: normal hold (no target switch, no done)")

for src in spike_sources:
    print(f"  spike: {src}", flush=True)

t2a = gate("2a: mean_jump < 0.03", mean_j < 0.03, f"{mean_j:.4f}")
t2b = gate("2b: max_jump < 0.06", max_j < 0.06, f"{max_j:.4f}")
t2c = gate("2c: no spikes > 0.06", len(spikes) == 0, f"{len(spikes)} spikes")
t2d = gate("2d: p99_jump < 0.05", p99_j < 0.05, f"{p99_j:.4f}")

env.close()

# ====================================================================
# SWEEP if hold failed during normal hold
# ====================================================================
if (not t2a or not t2b or not t2c) and args_cli.sweep_hold and len(spikes) > 0 and any('normal hold' in s for s in spike_sources):
    print(f"\n{'='*60}", flush=True)
    print(f"TEST 2B: Low-Level Hold Gain Sweep", flush=True)
    print(f"{'='*60}", flush=True)

    best = None
    for kp in [0.05, 0.1, 0.2]:
        for kd in [0.5, 0.75, 1.0]:
            for fr in [0.01, 0.02, 0.05]:
                try:
                    e = build_env(kp=kp, kd=kd, fr=fr)
                    e.reset()
                    for _ in range(5):
                        e.step(zero_action)
                    jj = []
                    for _ in range(100):
                        _, _, _, ex = e.step(zero_action)
                        jj.append(ex["log"]["ee_jump"].item())
                    e.close()
                    mj = float(np.mean(jj)); mxj = float(np.max(jj)); p99j = float(np.percentile(jj, 99))
                    line = f"  kp={kp:.2f} kd={kd:.2f} fr={fr:.3f}: mean_j={mj:.4f} max_j={mxj:.4f} p99={p99j:.4f}"
                    ok = mj < 0.03 and mxj < 0.06 and p99j < 0.05
                    line += " PASS" if ok else ""
                    print(line, flush=True)
                    if ok and (best is None or mj < best['mj']):
                        best = {'kp': kp, 'kd': kd, 'fr': fr, 'mj': mj, 'mxj': mxj}
                except Exception as ex:
                    print(f"  kp={kp:.2f} kd={kd:.2f} fr={fr:.3f}: ERROR {ex}", flush=True)

    if best:
        print(f"\n  BEST: kp={best['kp']} kd={best['kd']} fr={best['fr']} mj={best['mj']:.4f} mxj={best['mxj']:.4f}", flush=True)
        args_cli.kp_scale = best['kp']
        args_cli.kd_scale = best['kd']
        args_cli.force_rate_limit_fraction = best['fr']
    else:
        print(f"\n  NO CONFIG PASSES HOLD GATE", flush=True)

# ====================================================================
# TEST 3: ACTION SIGN (must pass 4/4)
# ====================================================================
if not args_cli.skip_sign_test:
    print(f"\n{'='*60}", flush=True)
    print(f"TEST 3: Action Sign Test (4/4 required)", flush=True)
    print(f"{'='*60}", flush=True)

    tests_sign = [
        ("+Y", torch.tensor([+0.03, 0.00], device=device), torch.tensor([+1.0, 0.0], device=device)),
        ("-Y", torch.tensor([-0.03, 0.00], device=device), torch.tensor([-1.0, 0.0], device=device)),
        ("+Z", torch.tensor([0.00, +0.01], device=device), torch.tensor([0.0, +1.0], device=device)),
        ("-Z", torch.tensor([0.00, -0.01], device=device), torch.tensor([0.0, -1.0], device=device)),
    ]

    sign_results = []
    for label, offset, correct_act in tests_sign:
        env2 = build_env()
        env2.reset()
        for _ in range(5):
            env2.step(zero_action)
        ee = env2._tcp_yz()
        env2._target_yz = ee + offset.unsqueeze(0)
        env2._target_vel_yz.zero_()
        env2._prev_distance = torch.norm(ee - env2._target_yz, dim=-1)
        dist_before = torch.norm(ee - env2._target_yz, dim=-1)[0].item()

        # Test correct, opposite, zero actions
        for act_label, act in [("correct", correct_act), ("opposite", -correct_act), ("zero", zero_action)]:
            env3 = build_env()
            env3.reset()
            for _ in range(5):
                env3.step(zero_action)
            ee3 = env3._tcp_yz()
            env3._target_yz = ee3 + offset.unsqueeze(0)
            env3._target_vel_yz.zero_()
            env3._prev_distance = torch.norm(ee3 - env3._target_yz, dim=-1)
            _, _, _, extras = env3.step(act.unsqueeze(0))
            da = extras["log"]["distance"].item()
            pa = extras["log"]["progress"].item()
            env3.close()
            sign_results.append((label, act_label, da, pa))

        env2.close()

    # Check 4/4
    passed_sign = 0
    for label, offset, correct_act in tests_sign:
        correct_entries = [(l, a, d, p) for l, a, d, p in sign_results if l == label]
        correct_d = [d for l, a, d, p in correct_entries if a == 'correct'][0]
        correct_p = [p for l, a, d, p in correct_entries if a == 'correct'][0]
        opposite_p = [p for l, a, d, p in correct_entries if a == 'opposite'][0]
        zero_p = [p for l, a, d, p in correct_entries if a == 'zero'][0]

        ok = correct_p > opposite_p and correct_p > zero_p
        print(f"  {label}: correct_prog={correct_p:.4f} opposite_prog={opposite_p:.4f} zero_prog={zero_p:.4f}  {'OK' if ok else 'WRONG SIGN'}", flush=True)
        if ok:
            passed_sign += 1

    gate("3: sign_test = 4/4", passed_sign == 4, f"{passed_sign}/4")

# ====================================================================
# TEST 5: GREEDY CONSISTENTLY BEATS ZERO
# ====================================================================
print(f"\n{'='*60}", flush=True)
print(f"TEST 5: Greedy vs Zero (10 rollouts, 100 steps each)", flush=True)
print(f"{'='*60}", flush=True)

greedy_wins_dist = 0; greedy_wins_prog = 0
for rollout in range(10):
    env5 = build_env()
    env5.reset()
    for _ in range(5):
        env5.step(zero_action)

    # Zero rollout
    z_dists = []; z_progs = []
    for _ in range(100):
        _, _, _, ex = env5.step(zero_action)
        z_dists.append(ex["log"]["distance"].item())
        z_progs.append(ex["log"]["progress"].item())
    z_md = float(np.mean(z_dists)); z_mp = float(np.mean(z_progs))

    # Reset and greedy rollout
    env5.reset()
    for _ in range(5):
        env5.step(zero_action)

    g_dists = []; g_progs = []; g_anorms = []
    for _ in range(100):
        ee = env5._tcp_yz(); err = env5._target_yz - ee
        action = torch.clamp(err / 0.05, -1.0, 1.0).unsqueeze(0)
        _, _, _, ex = env5.step(action)
        g_dists.append(ex["log"]["distance"].item())
        g_progs.append(ex["log"]["progress"].item())
        g_anorms.append(ex["log"]["action_norm"].item())
    g_md = float(np.mean(g_dists)); g_mp = float(np.mean(g_progs))
    g_an = float(np.mean(g_anorms))

    env5.close()

    if g_md < z_md: greedy_wins_dist += 1
    if g_mp >= z_mp: greedy_wins_prog += 1

gate("5a: greedy beats zero in dist (>=18/20)", greedy_wins_dist >= 9, f"{greedy_wins_dist}/20")
gate("5b: greedy beats zero in prog (>=18/20)", greedy_wins_prog >= 9, f"{greedy_wins_prog}/20")

# ====================================================================
# TEST 7: REWARD UNIT TEST
# ====================================================================
print(f"\n{'='*60}", flush=True)
print(f"TEST 7: Reward Unit Test", flush=True)
print(f"{'='*60}", flush=True)

def compute_reward_vals(pd, d, an, j):
    p = pd - d
    r = 10.0 * p - 1.0 * d - 0.001 * an - 0.05 * j
    return r, p

rA, _ = compute_reward_vals(0.08, 0.05, 0.5, 0.01)
rB, _ = compute_reward_vals(0.05, 0.08, 0.5, 0.01)
rC, _ = compute_reward_vals(0.05, 0.05, 0.0, 0.01)
rD, _ = compute_reward_vals(0.05, 0.05, 1.0, 0.01)
rE, _ = compute_reward_vals(0.05, 0.05, 0.5, 0.01)
rF, _ = compute_reward_vals(0.05, 0.05, 0.5, 0.10)

gate("7a: reward(progress>0) > reward(progress<0)", rA > rB, f"{rA:.3f} > {rB:.3f}")
gate("7b: reward(small action) > reward(large action)", rC > rD, f"{rC:.3f} > {rD:.3f}")
gate("7c: reward(small jump) > reward(large jump)", rE > rF, f"{rE:.3f} > {rF:.3f}")

# ====================================================================
# TEST 8: TRANSFER SANITY
# ====================================================================
if args_cli.stage0_checkpoint and Path(args_cli.stage0_checkpoint).exists():
    print(f"\n{'='*60}", flush=True)
    print(f"TEST 8: Stage-0 Transfer Sanity", flush=True)
    print(f"{'='*60}", flush=True)
    from rsl_rl.runners import OnPolicyRunner
    from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env_cfg import MulagUpperLevelPPORunnerCfg

    env8 = build_env()
    s0_ckpt = torch.load(args_cli.stage0_checkpoint, map_location='cpu', weights_only=False)
    s0_sd = s0_ckpt["actor_state_dict"]

    agent_cfg = MulagUpperLevelPPORunnerCfg()
    agent_cfg.device = device; agent_cfg.actor.hidden_dims = [64, 32]; agent_cfg.critic.hidden_dims = [64, 32]
    agent_cfg.actor.obs_normalization = False; agent_cfg.critic.obs_normalization = False
    runner = OnPolicyRunner(env8, agent_cfg.to_dict(), device=device)
    s1_actor = runner.alg.actor; s1_sd = s1_actor.state_dict()

    cr = min(s0_sd["mlp.0.weight"].shape[0], s1_sd["mlp.0.weight"].shape[0])
    w0 = s1_sd["mlp.0.weight"].clone()
    w0[:cr, 0] = s0_sd["mlp.0.weight"][:cr, 0]; w0[:cr, 1] = s0_sd["mlp.0.weight"][:cr, 1]
    w0[:cr, 4] = s0_sd["mlp.0.weight"][:cr, 2]; w0[:cr, 5] = s0_sd["mlp.0.weight"][:cr, 3]
    w0[:cr, 2] = 0.0; w0[:cr, 3] = 0.0
    s1_sd["mlp.0.weight"] = w0; s1_sd["mlp.0.bias"][:cr] = s0_sd["mlp.0.bias"][:cr]
    for k in ["mlp.2.weight", "mlp.2.bias", "mlp.4.weight", "mlp.4.bias"]:
        if s1_sd[k].shape == s0_sd[k].shape:
            s1_sd[k] = s0_sd[k]
    s1_actor.load_state_dict(s1_sd, strict=False)

    # Sample observations
    env8.reset()
    for _ in range(5):
        env8.step(zero_action)
    obs = env8._get_obs(); td = env8._to_tensordict(obs)
    with torch.no_grad():
        out = s1_actor(td)
    anorm = torch.norm(out, dim=-1).mean().item()
    print(f"  unscaled action_norm: {anorm:.4f}", flush=True)

    # Apply scale and check
    scale = 0.25
    with torch.no_grad():
        s1_actor.mlp[4].weight *= scale
        s1_actor.mlp[4].bias *= scale
    with torch.no_grad():
        out_s = s1_actor(td)
    anorm_s = torch.norm(out_s, dim=-1).mean().item()
    print(f"  scaled (0.25) action_norm: {anorm_s:.4f}", flush=True)
    gate("8: scaled action_norm 0.1-0.7", 0.1 <= anorm_s <= 0.7, f"{anorm_s:.4f}")
    env8.close()

# ====================================================================
# FINAL REPORT
# ====================================================================
print(f"\n{'='*60}", flush=True)
if FAILED:
    print(f"FAILED: {len(FAILED)} gate(s)", flush=True)
    for name, detail in FAILED:
        print(f"  - {name}: {detail}", flush=True)
    print(f"\nDo NOT run PPO until all gates pass.", flush=True)
    simulation_app.close()
    sys.exit(1)
else:
    print(f"ALL GATES PASSED", flush=True)
    print(f"Ready for easy sequence PPO.", flush=True)
    simulation_app.close()
    sys.exit(0)
