#!/usr/bin/env python3
"""The USD's arm joint limits clamp the cylinders to a tiny fraction of their real
stroke. Compute the limits implied by the Simscape/FD cylinder extension range,
widen them in PhysX, and re-measure whether the tool can now reach the ground.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, h5py
from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--num_envs", type=int, default=1)
p.add_argument("--samples", type=int, default=4000)
p.add_argument("--apply", action="store_true", help="actually widen the PhysX limits")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
app = AppLauncher(a); sim_app = app.app

import gymnasium as gym, torch  # noqa: E402
import ID_FD_pipline.tasks  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

# --- per-cylinder extension range actually present in the FD training data ---
with h5py.File(ROOT / "ID_FD_pipline/models/fd/FD_cylinder_data.mat", "r") as f:
    X = np.array(f["X_train_fd"]).T
EXT = X[:, 4:8]
ext_lo, ext_hi = EXT.min(0), EXT.max(0)
print("\n[LIM] Simscape/FD cylinder extension range (what the models were trained on):", flush=True)
for i in range(4):
    print(f"[LIM]   cyl{i}: {ext_lo[i]:+.4f} .. {ext_hi[i]:+.4f}  (stroke {ext_hi[i]-ext_lo[i]:.4f} m)", flush=True)

cfg = parse_env_cfg("Kevin-Mulag-IDFD-EETracking-PLAY-v0", device=a.device, num_envs=1)
cfg.enable_scripted_vehicle_motion = False
env = gym.make("Kevin-Mulag-IDFD-EETracking-PLAY-v0", cfg=cfg).unwrapped
env.reset(seed=17)
robot = env.robot
sc = env._cylinder_adapter.scaling
cyl_ids = env._cylinder_joint_ids
scale = sc.scale.numpy().astype(np.float64)
offset = sc.offset.numpy().astype(np.float64)

print("\n[LIM] cylinder x -> USD q is affine:  q = scale*x + offset", flush=True)
print(f"[LIM]   scale  = {np.round(scale,5).tolist()}", flush=True)
print(f"[LIM]   offset = {np.round(offset,5).tolist()}", flush=True)

q_a = scale * ext_lo + offset
q_b = scale * ext_hi + offset
q_need_lo = np.minimum(q_a, q_b)
q_need_hi = np.maximum(q_a, q_b)

cur = robot.data.soft_joint_pos_limits[0].detach().cpu().numpy()
ids = cyl_ids.tolist() if hasattr(cyl_ids, "tolist") else list(cyl_ids)
print("\n[LIM] REQUIRED vs CURRENT USD joint limits (rad):", flush=True)
print(f"{'cyl':>4} {'need_lo':>9} {'need_hi':>9} | {'usd_lo':>9} {'usd_hi':>9} | {'range_ratio':>11}", flush=True)
for i, j in enumerate(ids):
    need = q_need_hi[i] - q_need_lo[i]
    have = cur[j, 1] - cur[j, 0]
    print(f"{i:>4} {q_need_lo[i]:>9.3f} {q_need_hi[i]:>9.3f} | {cur[j,0]:>9.3f} {cur[j,1]:>9.3f} | "
          f"{have/max(need,1e-9):>10.1%}", flush=True)

ee = env._ee_body_id
q0 = robot.data.joint_pos.detach().clone()
qd0 = robot.data.joint_vel.detach().clone()


def sample_reach(lo_q, hi_q, tag):
    zs = []
    for _ in range(int(a.samples)):
        q = q0.clone(); qd = qd0.clone()
        r = torch.rand(4, device=env.device, dtype=torch.float32)
        qv = torch.as_tensor(lo_q, device=env.device, dtype=torch.float32) + \
             (torch.as_tensor(hi_q, device=env.device, dtype=torch.float32) -
              torch.as_tensor(lo_q, device=env.device, dtype=torch.float32)) * r
        q[:, cyl_ids] = qv
        qd[:, cyl_ids] = 0.0
        robot.write_joint_state_to_sim(q, qd)
        env.sim.forward()
        zs.append(float(robot.data.body_link_state_w[0, ee, 2].item()))
    robot.write_joint_state_to_sim(q0, qd0); env.sim.forward()
    zs = np.asarray(zs)
    print(f"[LIM] {tag}: tool world z min={zs.min():+.3f} max={zs.max():+.3f} "
          f"-> {'REACHES GROUND' if zs.min() <= 0.05 else 'lowest %.3f m above ground' % zs.min()}", flush=True)
    return zs.min()


print("", flush=True)
cur_lo = np.array([cur[j, 0] for j in ids]); cur_hi = np.array([cur[j, 1] for j in ids])
z_now = sample_reach(cur_lo, cur_hi, "CURRENT usd limits ")
z_new = sample_reach(q_need_lo, q_need_hi, "SIMSCAPE-implied  ")

print(f"\n[LIM] => widening the joint limits to the Simscape stroke lowers the tool from "
      f"{z_now:.3f} m to {z_new:.3f} m", flush=True)
if z_new <= 0.05:
    print("[LIM] *** FIX CONFIRMED: the USD joint limits were the blocker. ***", flush=True)
else:
    print("[LIM] *** Still cannot reach ground -- the USD geometry itself differs from "
          "teacher_student's asset. ***", flush=True)
env.close(); sim_app.close()
