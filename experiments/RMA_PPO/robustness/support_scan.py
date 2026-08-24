"""Scan the frozen dataset for windows inside the frozen PPO's training support."""
import json, sys
import numpy as np, torch
sys.path.insert(0, "/tmp/rrlab_scratch/work")
from cpu_surrogate_env import CpuSDTrackingEnv
from frozen_ppo import load_frozen_actor

H = 48
actor = load_frozen_actor()
mu = actor.obs_mean.flatten()[12:16].numpy()
sd = actor.obs_std.flatten()[12:16].numpy()
print("PPO training extension support (ch12:16):")
for i in range(4):
    print(f"  cyl{i}: mean={mu[i]:.5f} std={sd[i]:.5f}  +-3sd=[{mu[i]-3*sd[i]:.4f},{mu[i]+3*sd[i]:.4f}]")

env = CpuSDTrackingEnv(num_envs=2, run_ids=list(range(118)), horizon=H, seed=0)
print("runs loaded:", len(env.runs))
allx = np.concatenate([env.runs[r]["s"][:, :4] for r in sorted(env.runs)], axis=0)
print("dataset extension mean", allx.mean(0).round(4))
print("dataset extension std ", allx.std(0).round(4))
print("dataset extension min ", allx.min(0).round(4))
print("dataset extension max ", allx.max(0).round(4))
z = np.abs((allx - mu) / sd)
print("per-cyl |z| median", np.median(z, 0).round(1))
for thr in (3, 5, 10, 20):
    print(f"  frac rows max|z|<={thr:2d}: {float((z.max(1)<=thr).mean()):.4f}")

for thr in (3.0, 5.0, 10.0):
    good = tot = 0
    runs_ok = {}
    for r in sorted(env.runs):
        x = env.runs[r]["s"][:, :4]
        ok = (np.abs((x - mu) / sd).max(1) <= thr)
        n = len(ok) - H - 6
        if n <= 0:
            continue
        w = np.array([ok[i:i+H+1].all() for i in range(5, 5+n)])
        tot += n; good += int(w.sum())
        if w.any():
            runs_ok[r] = int(w.sum())
    print(f"|z|<={thr}: {good}/{tot} windows in support, across {len(runs_ok)} runs")
    if runs_ok:
        print("   runs:", sorted(runs_ok)[:25])
