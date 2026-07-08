#!/usr/bin/env python3
"""Fast Stage-1 control stack tests — single env, no rebuilds."""
from __future__ import annotations
import argparse, sys, os, math
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--kp_scale", type=float, default=0.1)
parser.add_argument("--kd_scale", type=float, default=0.5)
parser.add_argument("--force_rate_limit_fraction", type=float, default=0.01)
parser.add_argument("--max_cylinder_delta_m", type=float, default=0.0025)
parser.add_argument("--target_y_range", type=float, default=0.03)
parser.add_argument("--target_z_range", type=float, default=0.01)
parser.add_argument("--target_hold_steps", type=int, default=50)
parser.add_argument("--post_reset_settle_steps", type=int, default=30)
parser.add_argument("--stage0_checkpoint", type=str, default=None)
parser.add_argument("--action_scale", type=float, default=0.05)
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
    s = "PASS" if condition else "FAIL"
    print(f"  {name}: {s}  {detail}", flush=True)
    if not condition: FAILED.append(name)
    return condition

# Build single env
cfg = MulagUpperLevelEnvCfg(); cfg.scene.num_envs = 1
env = Stage1YZTrackingEnvWrapper(cfg, horizon_steps=10, action_scale_m=args_cli.action_scale,
                                  device=device, kp_scale=args_cli.kp_scale, kd_scale=args_cli.kd_scale,
                                  force_rate_limit=args_cli.force_rate_limit_fraction,
                                  velocity_filter_beta=0.8, trajectory_mode="sequence",
                                  target_hold_steps=args_cli.target_hold_steps,
                                  post_reset_settle_steps=args_cli.post_reset_settle_steps)
env._target_y_range = args_cli.target_y_range; env._target_z_range = args_cli.target_z_range
B = 1; za = torch.zeros((B, 2), device=device)
print(f"kp={args_cli.kp_scale} kd={args_cli.kd_scale} cyl_d={args_cli.max_cylinder_delta_m} "
      f"range=Y{args_cli.target_y_range}/Z{args_cli.target_z_range} hold={args_cli.target_hold_steps} "
      f"post_settle={args_cli.post_reset_settle_steps}", flush=True)

# ====================================================================
# TEST 1: Reset sanity
# ====================================================================
print(f"\n{'='*50} TEST 1: Reset sanity", flush=True)
env.reset()
ee = env._tcp_yz()[0]; t = env._target_yz[0]; err = t - ee
err_n = torch.norm(err).item()
obs = env._get_obs()[0]; pa = env._prev_action[0]; tv = env._target_vel_yz[0]
max_e = math.sqrt(args_cli.target_y_range**2 + args_cli.target_z_range**2) + 0.01
gate("1a: err<={:.4f}".format(max_e), err_n <= max_e, f"{err_n:.4f}")
gate("1b: tvel=0", torch.allclose(tv, torch.zeros(2,device=device), atol=1e-6))
gate("1c: prev_a=0", torch.allclose(pa, torch.zeros(2,device=device), atol=1e-6))
gate("1d: obs[0:2]=err", torch.allclose(obs[:2], err, atol=1e-4))

# ====================================================================
# TEST 2: Zero-action hold
# ====================================================================
print(f"\n{'='*50} TEST 2: Zero hold (100 steps)", flush=True)
for _ in range(30): env.step(za)
# Re-init trajectory with huge hold to prevent resampling during diagnostic
ee0 = env._tcp_yz()
env._target_yz = ee0.clone() + torch.tensor([[0.03, 0.01]], device=device)
env._target_vel_yz.zero_()
env._prev_distance = torch.norm(env._tcp_yz() - env._target_yz, dim=-1)
for i in range(1):
    env._traj_params[i] = env._init_trajectory(ee0[i:i+1], "sequence")
    env._traj_params[i]["hold_left"] = 99999
jj = []; dd = []
for _ in range(100):
    _, _, _, ex = env.step(za)
    jj.append(ex["log"]["ee_jump"].item()); dd.append(ex["log"]["distance"].item())
mj = float(np.mean(jj)); mxj = float(np.max(jj)); md = float(np.mean(dd))
p99j = float(np.percentile(jj, 99))
spikes = sum(1 for j in jj if j > 0.06)
print(f"  mean_j={mj:.4f} max_j={mxj:.4f} p99={p99j:.4f} mean_d={md:.4f} spikes>{0.06}={spikes}", flush=True)
gate("2a: mean_j<0.03", mj < 0.03, f"{mj:.4f}")
gate("2b: max_j<0.06", mxj < 0.06, f"{mxj:.4f}")
gate("2c: spikes==0", spikes == 0, f"{spikes}")

# ====================================================================
# TEST 3: Action sign (must pass 4/4)
# ====================================================================
print(f"\n{'='*50} TEST 3: Action sign", flush=True)

def test_action_sign(target_rel, label):
    """Set target relative to settled EE, test actions."""
    env.reset()
    for _ in range(30): env.step(za)
    ee0 = env._tcp_yz()
    env._target_yz = ee0 + target_rel.unsqueeze(0)
    env._target_vel_yz.zero_()
    env._prev_distance = torch.norm(ee0 - env._target_yz, dim=-1)
    db = float(torch.norm(ee0 - env._target_yz, dim=-1)[0].item())

    results = {}
    for act_name, act in [("+Y", torch.tensor([[1.,0.]], device=device)),
                           ("-Y", torch.tensor([[-1.,0.]], device=device)),
                           ("+Z", torch.tensor([[0.,1.]], device=device)),
                           ("-Z", torch.tensor([[0.,-1.]], device=device)),
                           ("zero", za)]:
        env.reset()
        for _ in range(30): env.step(za)
        ee0b = env._tcp_yz()
        env._target_yz = ee0b + target_rel.unsqueeze(0)
        env._target_vel_yz.zero_()
        env._prev_distance = torch.norm(ee0b - env._target_yz, dim=-1)
        dists = []; progs = []
        for s in range(10):
            _, _, _, ex = env.step(act)
            dists.append(ex["log"]["distance"].item())
            progs.append(ex["log"]["progress"].item())
        results[act_name] = {"d1": dists[0], "d5": float(np.mean(dists[:5])),
                             "d10": float(np.mean(dists)), "p1": progs[0],
                             "p5": float(np.mean(progs[:5])), "p10": float(np.mean(progs))}
    return label, results

# Determine "correct" action for each target offset
targets_test = [
    ("+Y", torch.tensor([+0.03, 0.00], device=device), "+Y"),
    ("-Y", torch.tensor([-0.03, 0.00], device=device), "-Y"),
    ("+Z", torch.tensor([0.00, +0.01], device=device), "+Z"),
    ("-Z", torch.tensor([0.00, -0.01], device=device), "-Z"),
]

sign_pass = 0
for tlabel, toffset, correct_act in targets_test:
    label, results = test_action_sign(toffset, tlabel)
    cp = results[correct_act]["p10"]
    op = results["zero"]["p10"]
    # Find opposite direction
    opposite_map = {"+Y": "-Y", "-Y": "+Y", "+Z": "-Z", "-Z": "+Z"}
    ap = results[opposite_map[correct_act]]["p10"]
    ok = cp > op and cp > ap
    if ok: sign_pass += 1
    print(f"  {label}: correct={cp:.4f} zero={op:.4f} opposite={ap:.4f}  {'OK' if ok else 'WRONG SIGN'}", flush=True)
    if not ok:
        print(f"    ALL: +Y({results['+Y']['p10']:.4f}) -Y({results['-Y']['p10']:.4f}) "
              f"+Z({results['+Z']['p10']:.4f}) -Z({results['-Z']['p10']:.4f})", flush=True)

gate("3: sign=4/4", sign_pass == 4, f"{sign_pass}/4")

# ====================================================================
# TEST 5: Greedy vs zero
# ====================================================================
print(f"\n{'='*50} TEST 5: Greedy vs zero (10 rollouts)", flush=True)
gw_dist = 0; gw_prog = 0
for rollout in range(10):
    env.reset()
    for _ in range(30): env.step(za)
    z_d = []; z_p = []
    for _ in range(100):
        _, _, _, ex = env.step(za)
        z_d.append(ex["log"]["distance"].item()); z_p.append(ex["log"]["progress"].item())
    env.reset()
    for _ in range(30): env.step(za)
    g_d = []; g_p = []
    for _ in range(100):
        ee = env._tcp_yz(); err = env._target_yz - ee
        act = torch.clamp(err / args_cli.action_scale, -1.0, 1.0)
        _, _, _, ex = env.step(act)
        g_d.append(ex["log"]["distance"].item()); g_p.append(ex["log"]["progress"].item())
    if float(np.mean(g_d)) < float(np.mean(z_d)): gw_dist += 1
    if float(np.mean(g_p)) >= float(np.mean(z_p)): gw_prog += 1

gate("5a: greedy beats zero dist (>=9/10)", gw_dist >= 9, f"{gw_dist}/10")
gate("5b: greedy beats zero prog (>=9/10)", gw_prog >= 9, f"{gw_prog}/10")

# ====================================================================
# TEST 7: Reward unit test (no physics)
# ====================================================================
print(f"\n{'='*50} TEST 7: Reward unit test", flush=True)
def r(pd, d, an, j): return 10*(pd-d) - d - 0.001*an - 0.05*j
gate("7a: progress+ > progress-", r(0.08,0.05,0.5,0.01) > r(0.05,0.08,0.5,0.01))
gate("7b: small act > big act", r(0.05,0.05,0.0,0.01) > r(0.05,0.05,1.0,0.01))
gate("7c: small jump > big jump", r(0.05,0.05,0.5,0.01) > r(0.05,0.05,0.5,0.10))

# ====================================================================
# TEST 8: Transfer sanity
# ====================================================================
if args_cli.stage0_checkpoint and Path(args_cli.stage0_checkpoint).exists():
    print(f"\n{'='*50} TEST 8: Transfer sanity", flush=True)
    from rsl_rl.runners import OnPolicyRunner
    from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env_cfg import MulagUpperLevelPPORunnerCfg

    s0 = torch.load(args_cli.stage0_checkpoint, map_location='cpu', weights_only=False)["actor_state_dict"]
    agent_cfg = MulagUpperLevelPPORunnerCfg()
    agent_cfg.device = device; agent_cfg.actor.hidden_dims = [64, 32]; agent_cfg.critic.hidden_dims = [64, 32]
    agent_cfg.actor.obs_normalization = False; agent_cfg.critic.obs_normalization = False
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), device=device)
    s1a = runner.alg.actor; s1d = s1a.state_dict()

    cr = min(s0["mlp.0.weight"].shape[0], s1d["mlp.0.weight"].shape[0])
    w = s1d["mlp.0.weight"].clone()
    w[:cr,0]=s0["mlp.0.weight"][:cr,0]; w[:cr,1]=s0["mlp.0.weight"][:cr,1]
    w[:cr,4]=s0["mlp.0.weight"][:cr,2]; w[:cr,5]=s0["mlp.0.weight"][:cr,3]
    w[:cr,2]=0; w[:cr,3]=0; s1d["mlp.0.weight"]=w; s1d["mlp.0.bias"][:cr]=s0["mlp.0.bias"][:cr]
    for k in ["mlp.2.weight","mlp.2.bias","mlp.4.weight","mlp.4.bias"]:
        if s1d[k].shape==s0[k].shape: s1d[k]=s0[k]
    s1a.load_state_dict(s1d, strict=False)

    # Test transfer rollout
    for scale, label in [(1.0,"x1.0"),(0.5,"x0.5"),(0.25,"x0.25")]:
        with torch.no_grad():
            s1a.mlp[4].weight *= scale; s1a.mlp[4].bias *= scale
        env.reset()
        for _ in range(30): env.step(za)
        anorms = []; dd_t = []
        for _ in range(30):
            obs = env._get_obs(); td = env._to_tensordict(obs)
            with torch.no_grad(): act = s1a(td)
            _, _, _, ex = env.step(act)
            anorms.append(torch.norm(act, dim=-1).mean().item())
            dd_t.append(ex["log"]["distance"].item())
        print(f"  transfer_{label}: anorm={float(np.mean(anorms)):.4f} dist={float(np.mean(dd_t)):.4f}", flush=True)
    gate("8: transfer anorm 0.1-0.7 at x0.25", 0.1 <= float(np.mean(anorms)) <= 0.7,
         f"check output above")

# ====================================================================
print(f"\n{'='*50}", flush=True)
if FAILED:
    print(f"FAILED: {FAILED}", flush=True); sys.exit(1)
else:
    print(f"ALL TESTS PASSED", flush=True)

env.close(); simulation_app.close()
