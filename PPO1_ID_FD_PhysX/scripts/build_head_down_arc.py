#!/usr/bin/env python3
"""Build the verified head-down mowing arc from direct PhysX poses.

The mowing arc is a Drehzapfen slew: hold cylinders 2/3/4 at a ground-contacting
configuration and sweep cylinder 1. Cylinder 4 is fixed at the measured
head-down setting while cylinders 2 and 3 hold the boom geometry.

This sweeps cylinder 1 finely at that triple, keeps the poses that are genuinely
executed + holdable + on the ground, and fits the resulting cutter arc.  The output
replaces the arc geometry that was circle-fitted to saturated sag poses.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--boom", type=float, nargs=3, default=[0.577, 0.463, 0.588991],
                    help="cylinders 2,3,4 held constant through the slew")
parser.add_argument("--settle_steps", type=int, default=300)
parser.add_argument("--ground_tol_m", type=float, default=0.12)
parser.add_argument("--reach_tol_m", type=float, default=0.005)
parser.add_argument("--out", default="PPO1_ID_FD_PhysX/models/mowing_arc_head_down.json")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
sim_app = app_launcher.app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
import PPO1_ID_FD_PhysX.tasks  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from PPO1_ID_FD_PhysX.dls_solver.direct_drive import DirectCylinderDriver  # noqa: E402

TASK = "Kevin-Mulag-IDFD-CylinderSpace-PLAY-v0"
cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
cfg.enable_scripted_vehicle_motion = False
cfg.episode_length_s = 1.0e4
env = gym.make(TASK, cfg=cfg).unwrapped
env.reset(seed=17)

IDS = env._cylinder_joint_ids
N = env.num_envs
scaling = env._cylinder_adapter.scaling
lo = float(scaling.learned_min[0])
hi = float(scaling.learned_max[0])
EFFORT = torch.tensor(
    [float(torch.as_tensor(env.robot.actuators[k].effort_limit_sim).flatten()[0])
     for k in ("drehzapfen_cylinder", "ausleger_i_cylinder",
               "ausleger_ii_cylinder", "messerkopf_schwenk_cylinder")],
    device=env.device, dtype=torch.float32).reshape(1, 4)

# Sweep cylinder 1 (the slew) across its full learned range, boom held fixed.
c1 = np.linspace(lo + 0.005, hi - 0.005, N, dtype=np.float32)
cmd = torch.zeros((N, 4), device=env.device)
cmd[:, 0] = torch.from_numpy(c1).to(env.device)
cmd[:, 1:] = torch.tensor(args.boom, device=env.device, dtype=torch.float32)

driver = DirectCylinderDriver(env)
for _ in range(args.settle_steps):
    driver.step(cmd, substeps=2)
x1 = driver.measured_x().clone()
sat1 = (env.robot.data.computed_torque[:, IDS].abs() >= EFFORT * 0.999).any(dim=1)
for _ in range(60):
    driver.step(cmd, substeps=2)
x2 = driver.measured_x().clone()

cutter = env._compute_ee_pos_local().cpu().numpy()
zw = env._compute_cutter_world_z().cpu().numpy()
reach = (x1 - cmd).abs().amax(dim=1).cpu().numpy()
drift = (x2 - x1).abs().amax(dim=1).cpu().numpy()
sat = sat1.cpu().numpy()

ok = (reach < args.reach_tol_m) & (drift < 0.002) & (~sat) & (np.abs(zw) < args.ground_tol_m)
print(f"[ARC] boom={args.boom}  {int(ok.sum())}/{N} slew samples are executed+holdable+ground",
      flush=True)
if ok.sum() < 3:
    print("[ARC] FAILED: not enough ground poses on this boom triple", flush=True)
    env.close(); sim_app.close(); sys.exit(1)

c = cutter[ok]
# Circle fit in the truck XY plane (the slew traces a circle about the Drehzapfen).
A = np.c_[2 * c[:, 0], 2 * c[:, 1], np.ones(len(c))]
b = c[:, 0] ** 2 + c[:, 1] ** 2
sol, *_ = np.linalg.lstsq(A, b, rcond=None)
cx, cy = float(sol[0]), float(sol[1])
r = float(np.sqrt(sol[2] + cx ** 2 + cy ** 2))
resid = np.abs(np.hypot(c[:, 0] - cx, c[:, 1] - cy) - r)
theta = np.degrees(np.arctan2(c[:, 1] - cy, c[:, 0] - cx))
ta, tb = float(theta.min()), float(theta.max())
arc_len = r * np.radians(tb - ta)

print(f"[ARC] circle centre=({cx:.4f}, {cy:.4f})  radius={r:.4f} m", flush=True)
print(f"[ARC] fit residual mean={resid.mean()*1000:.1f} mm  max={resid.max()*1000:.1f} mm", flush=True)
print(f"[ARC] theta [{ta:.2f}, {tb:.2f}] deg  ->  ARC LENGTH = {arc_len:.3f} m", flush=True)
print(f"[ARC] cutter world z on band: [{zw[ok].min():+.3f}, {zw[ok].max():+.3f}]", flush=True)
print(f"[ARC] endpoint A = ({c[theta.argmin(), 0]:.4f}, {c[theta.argmin(), 1]:.4f}, "
      f"{c[theta.argmin(), 2]:.4f})", flush=True)
print(f"[ARC] endpoint B = ({c[theta.argmax(), 0]:.4f}, {c[theta.argmax(), 1]:.4f}, "
      f"{c[theta.argmax(), 2]:.4f})", flush=True)

order = np.argsort(theta)
table = [dict(
    theta_deg=float(theta[i]),
    cylinder_cmd=cmd[np.where(ok)[0][i]].cpu().tolist(),
    cylinder_measured=x1[np.where(ok)[0][i]].cpu().tolist(),
    cutter_truck=c[i].tolist(),
    cutter_world_z=float(zw[ok][i]),
) for i in order]

(ROOT / args.out).write_text(json.dumps(dict(
    schema_version=1,
    note="Mowing arc fitted to EXECUTED + HOLDABLE + ground cutter poses on the repaired "
         "plant. Supersedes the arc that was circle-fitted to saturated sag poses.",
    boom_cylinders_234=args.boom,
    arc_center_xy=[cx, cy], arc_radius_m=r,
    theta_a_deg=ta, theta_b_deg=tb, arc_length_m=float(arc_len),
    fit_residual_mean_m=float(resid.mean()), fit_residual_max_m=float(resid.max()),
    ground_tol_m=args.ground_tol_m,
    num_samples=int(ok.sum()),
    cylinder_teacher=table,
), indent=1))
print(f"[ARC] wrote {ROOT / args.out}", flush=True)

env.close()
sim_app.close()
