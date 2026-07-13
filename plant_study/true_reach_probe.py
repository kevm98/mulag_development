#!/usr/bin/env python3
"""Can the tool ACTUALLY reach the ground? Sample ALL arm joints across their true
USD limits (not just the 4 cylinder joints) and report the tool's lowest world z.

The earlier yz_workspace_probe froze the 5th arm joint (Messerkopf_joint, coupled)
at its reset value, which sits BELOW the tool in the kinematic chain -- that can
artificially shrink the reachable set. If the tool reaches the ground here but not
with cylinders alone, the cylinder parameterization/limits are the blocker.
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
p.add_argument("--samples", type=int, default=6000)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
app = AppLauncher(a); sim_app = app.app

import gymnasium as gym, torch  # noqa: E402
torch.manual_seed(a.seed)
import ID_FD_pipline.tasks  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

TASK = "Kevin-Mulag-IDFD-EETracking-PLAY-v0"
cfg = parse_env_cfg(TASK, device=a.device, num_envs=1)
cfg.enable_scripted_vehicle_motion = False
env = gym.make(TASK, cfg=cfg).unwrapped
env.reset(seed=int(a.seed))
zero = torch.zeros((1, 4), device=env.device)
for _ in range(30):
    env.step(zero)

robot = env.robot
names = robot.joint_names
print(f"\n[R] ALL {len(names)} JOINTS:", flush=True)
_lim0 = robot.data.soft_joint_pos_limits[0]
for i, n in enumerate(names):
    print(f"[R]   {i:>2} {n:<34} {_lim0[i,0].item():+8.3f} .. {_lim0[i,1].item():+8.3f}", flush=True)
# every joint that is NOT a wheel/steering -> part of the arm chain
ARM = [n for n in names if ("Wheel" not in n and "Steering" not in n)]
arm_ids = [names.index(n) for n in ARM]
print(f"\n[R] sampling {len(ARM)} non-wheel joints: {ARM}", flush=True)
lim = robot.data.soft_joint_pos_limits[0]           # (num_joints, 2)
q0 = robot.data.joint_pos.detach().clone()
qd0 = robot.data.joint_vel.detach().clone()
ee = env._ee_body_id

print("\n########## TRUE ARM REACH (all 5 arm joints, full USD limits) ##########", flush=True)
print(f"{'joint':>26} {'min':>9} {'max':>9}  (rad)", flush=True)
for n, i in zip(ARM, arm_ids):
    print(f"{n:>26} {lim[i,0].item():>9.3f} {lim[i,1].item():>9.3f}", flush=True)

cyl_ids = env._cylinder_joint_ids
cyl_names = [names[i] for i in (cyl_ids.tolist() if hasattr(cyl_ids, "tolist") else cyl_ids)]
print(f"\n[R] cylinder-controlled joints = {cyl_names}", flush=True)
print(f"[R] arm joints NOT cylinder-controlled = {[n for n in ARM if n not in cyl_names]}", flush=True)


@torch.no_grad()
def tool_world_z(q_arm):
    q = q0.clone(); qd = qd0.clone()
    for k, i in enumerate(arm_ids):
        q[:, i] = q_arm[k]
    qd[:, arm_ids] = 0.0
    robot.write_joint_state_to_sim(q, qd)
    env.sim.forward()
    return robot.data.body_link_state_w[0, ee, :3].detach().cpu().numpy().copy()


lo = np.array([lim[i, 0].item() for i in arm_ids])
hi = np.array([lim[i, 1].item() for i in arm_ids])
pts = []
for _ in range(int(a.samples)):
    q_arm = lo + (hi - lo) * np.random.rand(len(ARM))
    pts.append(tool_world_z(q_arm))
robot.write_joint_state_to_sim(q0, qd0); env.sim.forward()
W = np.asarray(pts)

print(f"\n[R] ALL-5-JOINT tool world z: min={W[:,2].min():+.3f} max={W[:,2].max():+.3f} m", flush=True)
print(f"[R] tool world y: [{W[:,1].min():+.2f}, {W[:,1].max():+.2f}]  x: [{W[:,0].min():+.2f}, {W[:,0].max():+.2f}]", flush=True)
reaches = W[:, 2].min() <= 0.05
print(f"[R] => tool {'CAN reach the ground' if reaches else 'CANNOT reach ground (lowest %.3f m)' % W[:,2].min()}", flush=True)

# Now: with the 4 cylinder joints ONLY (5th frozen at reset) -- the current pipeline
cyl_arm_idx = [ARM.index(n) for n in ARM if n in cyl_names]
pts2 = []
for _ in range(int(a.samples)):
    q_arm = np.array([q0[0, i].item() for i in arm_ids])
    for k in cyl_arm_idx:
        q_arm[k] = lo[k] + (hi[k] - lo[k]) * np.random.rand()
    pts2.append(tool_world_z(q_arm))
robot.write_joint_state_to_sim(q0, qd0); env.sim.forward()
W2 = np.asarray(pts2)
print(f"\n[R] CYLINDERS-ONLY (5th joint frozen) tool world z: min={W2[:,2].min():+.3f} max={W2[:,2].max():+.3f} m", flush=True)
print(f"[R] => BLOCKER? cylinders-only lowest = {W2[:,2].min():.3f} m vs all-joints {W[:,2].min():.3f} m", flush=True)
if W[:, 2].min() < W2[:, 2].min() - 0.05:
    print("[R] *** The frozen 5th joint IS restricting reach. Free/couple it. ***", flush=True)
print("########################################################################\n", flush=True)
env.close(); sim_app.close()
