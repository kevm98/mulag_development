"""Two decisive diagnostics.

A. Is the surrogate task controllable at all inside the frozen PPO's support?
   A simple proportional controller on the exact tracking error is the fairest
   possible non-learned reference.  If NOTHING beats the do-nothing null, the
   surrogate poses no control task and no controller robustness curve exists.

B. The support/task trade-off: as the PPO training-support gate is relaxed the
   reference starts to move (a real task appears) but the policy leaves its own
   input distribution.  We need a threshold where BOTH hold.
"""
import json
import numpy as np, torch

from frozen_ppo import load_frozen_actor
from robustness_env import RobustnessEnv, DATASET_ROOT, HISTORY_LAG, Episode
from stress import Stress
import protocol as P

actor = load_frozen_actor()


def episode_set(support_z, n=64, seed=P.EPISODE_SEED):
    mu = actor.obs_mean.flatten()[12:16].numpy(); sd = actor.obs_std.flatten()[12:16].numpy()
    cands = []
    for rid in range(118):
        p1 = DATASET_ROOT/"sd"/"dataset"/f"{rid:05d}"/"samples.npz"
        p2 = DATASET_ROOT/"aam"/"rom_rows"/f"{rid:05d}"/"rom_rows.npz"
        if not p1.is_file() or not p2.is_file(): continue
        with np.load(p1) as a, np.load(p2) as b:
            m = min(len(a["s_h"]), len(b["desired_cylinder_position"]))
            x = a["s_h"][:m,:4].astype(np.float32)
        ok = np.abs((x-mu)/sd).max(1) <= support_z
        for st in range(HISTORY_LAG-1, m-P.HORIZON-1):
            if ok[st:st+P.HORIZON+1].all(): cands.append(Episode(rid, st))
    if len(cands) < n: return None
    rng = np.random.default_rng(seed)
    by = {}
    for e in cands: by.setdefault(e.run_id, []).append(e)
    runs = sorted(by); rng.shuffle(runs)
    out=[]; i=0
    while len(out) < n:
        pool = by[runs[i % len(runs)]]; out.append(pool[int(rng.integers(0,len(pool)))]); i+=1
    return out


def run(env, policy, kp=0.0, stress=Stress()):
    obs = env.reset(stress)
    err, sat, ret = [], [], torch.zeros(env.num_envs)
    zs = []
    for t in range(P.HORIZON):
        if policy == "ppo":
            a = torch.clamp(actor.mean(obs), -1.0, 1.0)
        elif policy == "zero":
            a = torch.zeros(env.num_envs, 4)
        elif policy == "p":
            idx = torch.arange(env.num_envs)
            ref = env.ref_x[idx, torch.clamp(env.t, max=env.horizon)]
            # dx = action * 0.004 * (-1)  =>  action = -(ref - x)/0.004
            a = torch.clamp(-kp * (ref - env.s[:, :4]) / 0.004, -1.0, 1.0)
        zs.append(actor.z(obs).abs().amax())
        obs, r, d, info = env.step(a)
        ret += r; err.append(info["extension_rmse"]); sat.append(info["saturated"])
    e = torch.stack(err)
    return dict(median_mm=float(e.median())*1000, mean_mm=float(e.mean())*1000,
                ret=float(ret.mean()), sat=float(torch.stack(sat).mean()),
                obs_z=float(torch.stack(zs).max()))

print("=== A. controllability inside the PPO support (|z|<=3) ===")
eps = P.frozen_episode_set(actor.obs_mean, actor.obs_std)
env = RobustnessEnv(episodes=eps, horizon=P.HORIZON, obs_mean=actor.obs_mean)
print(f"{'controller':<16}{'median mm':>10}{'mean mm':>10}{'return':>10}{'sat':>7}{'obs|z|':>8}")
for name, kp in (("null (zero)",0.0), ("frozen PPO",0.0), ("P kp=0.1",0.1), ("P kp=0.25",0.25),
                 ("P kp=0.5",0.5), ("P kp=1.0",1.0), ("P kp=2.0",2.0)):
    pol = "zero" if name.startswith("null") else ("ppo" if name.startswith("frozen") else "p")
    r = run(env, pol, kp)
    print(f"{name:<16}{r['median_mm']:>10.3f}{r['mean_mm']:>10.3f}{r['ret']:>10.2f}{r['sat']:>7.3f}{r['obs_z']:>8.1f}")

print()
print("=== B. support / task trade-off ===")
print(f"{'|z|<=':>6}{'ref travel mm':>15}{'PPO obs|z|':>12}{'PPO mm':>9}{'null mm':>9}{'bestP mm':>10}{'PPO sat':>9}")
for zt in (3.0, 5.0, 10.0, 20.0, 50.0):
    s = episode_set(zt)
    if s is None:
        print(f"{zt:>6.0f}   (insufficient windows)"); continue
    e2 = RobustnessEnv(episodes=s, horizon=P.HORIZON, obs_mean=actor.obs_mean)
    trav = np.median([np.abs(e2.runs[x.run_id]["x_ref"][x.start+P.HORIZON] -
                            e2.runs[x.run_id]["x_ref"][x.start]).max() for x in s])*1000
    rp = run(e2, "ppo"); rn = run(e2, "zero")
    bp = min(run(e2, "p", k)["median_mm"] for k in (0.1,0.25,0.5,1.0))
    print(f"{zt:>6.0f}{trav:>15.3f}{rp['obs_z']:>12.1f}{rp['median_mm']:>9.3f}{rn['median_mm']:>9.3f}{bp:>10.3f}{rp['sat']:>9.3f}")
