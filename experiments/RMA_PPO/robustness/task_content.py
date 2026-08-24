"""How much control authority does the surrogate task actually require?"""
import numpy as np, torch
from frozen_ppo import load_frozen_actor
from robustness_env import RobustnessEnv, DATASET_ROOT
from stress import Stress
import protocol as P

actor = load_frozen_actor()
eps = P.frozen_episode_set(actor.obs_mean, actor.obs_std)
env = RobustnessEnv(episodes=eps, horizon=P.HORIZON, obs_mean=actor.obs_mean)

# 1. how far does the reference travel over one episode?
travel, expert_err, start_err = [], [], []
for i, e in enumerate(eps):
    run = env.runs[e.run_id]; st = e.start
    ref = run["x_ref"][st:st+P.HORIZON+1]
    act = run["s"][st:st+P.HORIZON+1, :4]          # recorded achieved extension
    travel.append(np.abs(ref[-1]-ref[0]).max())
    expert_err.append(np.sqrt(((act-ref)**2).mean(1)))
    start_err.append(np.sqrt(((act[0]-ref[0])**2).mean()))
travel = np.array(travel); expert = np.concatenate(expert_err)
print(f"reference travel over 48 ticks (max over cyl): median {np.median(travel)*1000:.2f} mm, p90 {np.percentile(travel,90)*1000:.2f} mm, max {travel.max()*1000:.2f} mm")
print(f"RECORDED (expert) extension RMSE vs its own reference: median {np.median(expert)*1000:.2f} mm, mean {expert.mean()*1000:.2f} mm")
print(f"episode-start error: median {np.median(start_err)*1000:.2f} mm")

# 2. reference velocity magnitude
vref = np.concatenate([env.runs[e.run_id]["v_ref"][e.start:e.start+P.HORIZON+1] for e in eps])
print(f"|v_ref| median {np.median(np.abs(vref))*1000:.3f} mm/s, p95 {np.percentile(np.abs(vref),95)*1000:.3f} mm/s")

# 3. how far does the plant drift open-loop (zero valve) vs the reference?
obs = env.reset(Stress())
drift = []
for _ in range(P.HORIZON):
    obs, r, d, info = env.step(torch.zeros(env.num_envs,4))
    drift.append(info["extension_rmse"])
d = torch.stack(drift)
print(f"open-loop (zero action) error: t=1 {float(d[0].median())*1000:.2f} mm, t=24 {float(d[23].median())*1000:.2f} mm, t=48 {float(d[-1].median())*1000:.2f} mm")

# 4. what does a full-authority constant command do? (is the plant even movable?)
for amp in (0.25, 1.0):
    obs = env.reset(Stress())
    x0 = env.s[:, :4].clone()
    for _ in range(P.HORIZON):
        obs, r, dn, info = env.step(torch.full((env.num_envs,4), amp))
    print(f"constant action {amp}: extension moved {float((env.s[:,:4]-x0).abs().max())*1000:.2f} mm (max), {float((env.s[:,:4]-x0).abs().median())*1000:.2f} mm (median)")
