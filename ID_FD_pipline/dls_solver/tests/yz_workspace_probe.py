#!/usr/bin/env python3
"""Measure the arm's REACHABLE base-Y/Z EE envelope from the settled anchor pose.

Trajectories must be sized to this envelope. Sizing a path larger than the
reachable workspace makes any tracking gate unpassable (and any controller
tuning meaningless).

Method: sample cylinder configurations across their learned limits, write them
directly to the articulation (exact kinematics, no controller/settling), and
record the resulting EE Y/Z.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--num_envs", type=int, default=1)
p.add_argument("--seed", type=int, default=17)
p.add_argument("--samples", type=int, default=4000)
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

scaling = env._cylinder_adapter.scaling
lo = scaling.learned_min.to(env.device, torch.float32).reshape(1, 4)
hi = scaling.learned_max.to(env.device, torch.float32).reshape(1, 4)
q0 = env.robot.data.joint_pos.detach().clone()
qd0 = env.robot.data.joint_vel.detach().clone()
anchor_x, _ = env._refresh_cylinder_state()
anchor_ee = env._compute_ee_pos_local()[0].detach().cpu().numpy()


@torch.no_grad()
def ee_at(x: torch.Tensor) -> np.ndarray:
    q_goal = scaling.clamp_usd_q(scaling.learned_x_to_usd_q(x))
    q = q0.clone(); qd = qd0.clone()
    q[:, env._cylinder_joint_ids] = q_goal
    qd[:, env._cylinder_joint_ids] = 0.0
    env.robot.write_joint_state_to_sim(q, qd)
    env.sim.forward()
    return env._compute_ee_pos_local()[0].detach().cpu().numpy()


pts = []
wpts = []
for _ in range(int(a.samples)):
    x = lo + (hi - lo) * torch.rand((1, 4), device=env.device)
    pts.append(ee_at(x))
    wpts.append(env.robot.data.body_link_state_w[0, env._ee_body_id, :3].detach().cpu().numpy().copy())
env.robot.write_joint_state_to_sim(q0, qd0); env.sim.forward()
P = np.asarray(pts)

W = np.asarray(wpts)
print("\n########## TOOL WORLD-FRAME REACH (can it touch the ground?) ##########", flush=True)
print(f"[WORLD] tool world z: min={W[:,2].min():+.3f}  max={W[:,2].max():+.3f} m  (ground = 0.0)", flush=True)
print(f"[WORLD] tool world x: [{W[:,0].min():+.2f}, {W[:,0].max():+.2f}]  y: [{W[:,1].min():+.2f}, {W[:,1].max():+.2f}]", flush=True)
print(f"[WORLD] => tool {'CAN reach ground' if W[:,2].min() <= 0.05 else 'CANNOT reach ground; lowest = %.3f m ABOVE it' % W[:,2].min()}", flush=True)
print("######################################################################\n", flush=True)
print("\n========== REACHABLE EE BASE-Y/Z WORKSPACE ==========", flush=True)
print(f"[WS] anchor EE (x,y,z) = {np.round(anchor_ee, 4).tolist()}", flush=True)
for i, ax in enumerate("xyz"):
    print(f"[WS] {ax}: min={P[:, i].min():+.4f} max={P[:, i].max():+.4f} span={P[:, i].ptp():.4f} m", flush=True)
# reachable span RELATIVE to the anchor (what a trajectory anchored at reset can use)
dy = P[:, 1] - anchor_ee[1]
dz = P[:, 2] - anchor_ee[2]
print(f"\n[WS] relative to anchor: dy in [{dy.min():+.3f}, {dy.max():+.3f}]  "
      f"dz in [{dz.min():+.3f}, {dz.max():+.3f}]", flush=True)
# a symmetric, safely-reachable box around the anchor
sy = min(abs(dy.min()), abs(dy.max()))
sz = min(abs(dz.min()), abs(dz.max()))
print(f"[WS] symmetric reachable half-extent about anchor: dy=+-{sy:.3f} m  dz=+-{sz:.3f} m", flush=True)
print(f"\n[WS] RECOMMENDED trajectory sizing (70% of symmetric reach):", flush=True)
print(f"[WS]   line length      ~ {2*sy*0.7:.3f} m   (ping-pong along Y)", flush=True)
print(f"[WS]   sweep amp (Z)    ~ {sz*0.7:.3f} m", flush=True)
print(f"[WS]   curve radius     ~ {min(sy, sz)*0.7:.3f} m", flush=True)
print("====================================================\n", flush=True)
env.close(); sim_app.close()
