#!/usr/bin/env python3
"""TRUE reach, measured through PHYSICS (constraints solved), not FK.

The MULAG arm is a CLOSED-LOOP linkage: besides the 4 actuated cylinder joints it
has passive constraint joints (Drehzapfen/Ausleger/Messerkopf_Schwenk
*_constraint_joint_*, all with +-inf limits) that PhysX must SOLVE.

Writing joint states directly while freezing those constraint joints breaks the
loop and gives a badly wrong (over-restricted) workspace -- that is what produced
the bogus "tool cannot reach the ground" result. Here we command the cylinder
position targets and let the solver do its job.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--num_envs", type=int, default=1)
p.add_argument("--seed", type=int, default=17)
p.add_argument("--corners", type=int, default=16)
p.add_argument("--hold", type=int, default=120)
p.add_argument("--widen", action="store_true", help="widen cylinder limits to the Simscape stroke")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
app = AppLauncher(a); sim_app = app.app

import gymnasium as gym, torch, itertools  # noqa: E402
torch.manual_seed(a.seed)
import ID_FD_pipline.tasks  # noqa: E402,F401
from plant_study.direct_drive import DirectCylinderDriver  # noqa: E402
from plant_study.widen_limits import widen_cylinder_limits  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

cfg = parse_env_cfg("Kevin-Mulag-IDFD-EETracking-PLAY-v0", device=a.device, num_envs=1)
cfg.enable_scripted_vehicle_motion = False
cfg.episode_length_s = 1e6
env = gym.make("Kevin-Mulag-IDFD-EETracking-PLAY-v0", cfg=cfg).unwrapped
env.reset(seed=int(a.seed))
if a.widen:
    widen_cylinder_limits(env)
drv = DirectCylinderDriver(env)   # picks up the (possibly widened) learned limits
drv.settle(80)
ee_body = env._ee_body_id
lo, hi = drv.lo, drv.hi
print(f"\n[PR] cylinder learned limits lo={np.round(lo[0].cpu().numpy(),3).tolist()} "
      f"hi={np.round(hi[0].cpu().numpy(),3).tolist()}", flush=True)


def tool_world():
    return env.robot.data.body_link_state_w[0, ee_body, :3].detach().cpu().numpy().copy()


print(f"[PR] rest tool world = {np.round(tool_world(),3).tolist()}", flush=True)
print("\n[PR] driving each cylinder-limit CORNER through physics (constraints solved)...", flush=True)

pts = []
corners = list(itertools.product([0.0, 1.0], repeat=4))  # 16 corners of the cylinder box
for k, c in enumerate(corners):
    env.reset(seed=int(a.seed)); drv.settle(60)
    t = torch.tensor(c, device=env.device, dtype=torch.float32).reshape(1, 4)
    goal = lo + (hi - lo) * t
    for _ in range(int(a.hold)):
        drv.step(goal)
    w = tool_world()
    pts.append(w)
    print(f"[PR]   corner {k:2d} {list(map(int,c))} -> tool world "
          f"x={w[0]:+.2f} y={w[1]:+.2f} z={w[2]:+.3f}", flush=True)

W = np.asarray(pts)
print(f"\n[PR] TOOL WORLD REACH (physics, constraints solved):", flush=True)
print(f"[PR]   x: [{W[:,0].min():+.3f}, {W[:,0].max():+.3f}]  span={W[:,0].ptp():.3f}", flush=True)
print(f"[PR]   y: [{W[:,1].min():+.3f}, {W[:,1].max():+.3f}]  span={W[:,1].ptp():.3f}", flush=True)
print(f"[PR]   z: [{W[:,2].min():+.3f}, {W[:,2].max():+.3f}]  span={W[:,2].ptp():.3f}   (ground = 0)", flush=True)
if W[:, 2].min() <= 0.10:
    print(f"[PR] *** TOOL REACHES THE GROUND (lowest z = {W[:,2].min():.3f} m) ***", flush=True)
else:
    print(f"[PR] *** lowest z = {W[:,2].min():.3f} m, still above ground ***", flush=True)
print("", flush=True)
env.close(); sim_app.close()
