#!/usr/bin/env python3
"""WHICH BODY is the articulation root? We compute the EE in `root_pos_w/root_quat_w`,
but if that root is the arm's slew base (Drehzapfen) rather than the truck chassis,
then the 'base frame' ROTATES WITH THE ARM -- which would explain why EE-local x is
identically 0 while world-frame x reach is 4.36 m.

teacher_student commands the EE at base-frame x=6.5..7.5, y=0.7..2.2, z=0.5.
Ours reads (0.0, -3.39, -0.18). Those cannot both be truck-frame.
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
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
app = AppLauncher(a); sim_app = app.app

import gymnasium as gym, torch  # noqa: E402
import ID_FD_pipline.tasks  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab.utils.math import quat_apply_inverse  # noqa: E402

cfg = parse_env_cfg("Kevin-Mulag-IDFD-EETracking-PLAY-v0", device=a.device, num_envs=1)
cfg.enable_scripted_vehicle_motion = False
env = gym.make("Kevin-Mulag-IDFD-EETracking-PLAY-v0", cfg=cfg).unwrapped
env.reset(seed=17)
zero = torch.zeros((1, 4), device=env.device)
for _ in range(60):
    env.step(zero)

robot = env.robot
root_p = robot.data.root_pos_w[0].detach().cpu().numpy()
print(f"\n[F] root_pos_w = {np.round(root_p,4).tolist()}", flush=True)
print("[F] distance from root to each body origin:", flush=True)
best = None
for i, n in enumerate(robot.body_names):
    bp = robot.data.body_link_state_w[0, i, :3].detach().cpu().numpy()
    d = float(np.linalg.norm(bp - root_p))
    if best is None or d < best[1]:
        best = (n, d)
    if d < 0.30:
        print(f"[F]   {n:<32} d={d:.4f}  pos={np.round(bp,3).tolist()}", flush=True)
print(f"\n[F] *** ROOT COINCIDES WITH BODY: '{best[0]}' (d={best[1]:.4f} m) ***", flush=True)

ee = env._ee_body_id
ee_w = robot.data.body_link_state_w[0, ee, :3]
rq = robot.data.root_quat_w
rq = rq / rq.norm(dim=-1, keepdim=True).clamp(min=1e-8)
ee_in_root = quat_apply_inverse(rq, (ee_w - robot.data.root_pos_w[0]).reshape(1, 3))[0].cpu().numpy()
print(f"\n[F] EE in ROOT frame (what we use today) = {np.round(ee_in_root,3).tolist()}", flush=True)

# now in the TRUCK CHASSIS frame
for cand in ("base_link", "Frontrahmen"):
    if cand in robot.body_names:
        j = robot.body_names.index(cand)
        bp = robot.data.body_link_state_w[:, j, :3]
        bq = robot.data.body_link_state_w[:, j, 3:7]
        bq = bq / bq.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        ee_in_body = quat_apply_inverse(bq, (ee_w.reshape(1, 3) - bp))[0].cpu().numpy()
        print(f"[F] EE in '{cand}' frame                 = {np.round(ee_in_body,3).tolist()}", flush=True)

print("\n[F] teacher_student mows at base-frame x=6.5..7.5, y=0.7..2.2, z=0.5", flush=True)
print("[F] -> whichever frame gives EE ~ (7, 1.5, 0.5) is the CORRECT truck frame.\n", flush=True)
env.close(); sim_app.close()
