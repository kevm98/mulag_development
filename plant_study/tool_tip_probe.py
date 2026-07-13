#!/usr/bin/env python3
"""Measure the CUTTING HEAD offset: where the tool geometry actually is, relative
to the Messerkopf body ORIGIN.

We have been commanding the body origin, which sits ~0.66 m above the ground. The
physical cutter hangs BELOW it -- that is what must sweep the grass. This prints
the tool-tip offset to put into `tool_tip_local_offset`.
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
from pxr import Usd, UsdGeom, Gf  # noqa: E402
import omni.usd  # noqa: E402

cfg = parse_env_cfg("Kevin-Mulag-IDFD-EETracking-PLAY-v0", device=a.device, num_envs=1)
cfg.enable_scripted_vehicle_motion = False
env = gym.make("Kevin-Mulag-IDFD-EETracking-PLAY-v0", cfg=cfg).unwrapped
env.reset(seed=17)
zero = torch.zeros((1, 4), device=env.device)
for _ in range(60):
    env.step(zero)

robot = env.robot
ee = env._ee_body_id
ee_name = robot.body_names[ee]
origin_w = robot.data.body_link_state_w[0, ee, :3].detach().cpu().numpy()
quat_w = robot.data.body_link_state_w[0, ee, 3:7]

stage = omni.usd.get_context().get_stage()
# find the prim for the EE body by name (path layout varies)
prim = None
prim_path = None
for pr in stage.Traverse():
    pth = str(pr.GetPath())
    if pth.startswith("/World/envs/env_0/Robot") and pr.GetName() == ee_name:
        prim, prim_path = pr, pth
        break
if prim is None:
    print("[TIP] candidate prims containing the body name:", flush=True)
    for pr in stage.Traverse():
        pth = str(pr.GetPath())
        if ee_name.lower() in pth.lower() and pth.startswith("/World/envs/env_0"):
            print(f"[TIP]   {pth}", flush=True)
    raise SystemExit("could not locate EE prim")
print(f"\n[TIP] EE body = '{ee_name}'  prim={prim_path}  valid={bool(prim)}", flush=True)
print(f"[TIP] body ORIGIN world = {np.round(origin_w,4).tolist()}", flush=True)

cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                          [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
# LOCAL bound: pose-independent, expressed in the body's own frame. (ComputeWorldBound
# returns the USD-authored pose, NOT PhysX's live transform, so it is useless here.)
local = cache.ComputeLocalBound(prim)
local.Transform(local.GetMatrix())          # collapse the bound's internal matrix
rng = local.ComputeAlignedRange()
lo = np.array([rng.GetMin()[0], rng.GetMin()[1], rng.GetMin()[2]])
hi = np.array([rng.GetMax()[0], rng.GetMax()[1], rng.GetMax()[2]])
print(f"[TIP] tool geometry LOCAL AABB (body frame):", flush=True)
print(f"[TIP]   min = {np.round(lo,4).tolist()}", flush=True)
print(f"[TIP]   max = {np.round(hi,4).tolist()}", flush=True)
print(f"[TIP]   size = {np.round(hi-lo,4).tolist()} m", flush=True)

# the cutting point: centre in X/Y of the head, BOTTOM in Z
tip_local = np.array([0.5*(lo[0]+hi[0]), 0.5*(lo[1]+hi[1]), lo[2]])
print(f"\n[TIP] ===> tool_tip_local_offset = ({tip_local[0]:+.4f}, {tip_local[1]:+.4f}, {tip_local[2]:+.4f})", flush=True)
print(f"[TIP] the cutter hangs {-tip_local[2]:.4f} m BELOW the Messerkopf origin", flush=True)
print(f"[TIP] body origin world z = {origin_w[2]:.4f}  ->  cutter world z ~= {origin_w[2] + tip_local[2]:.4f} m", flush=True)
print(f"[TIP] set use_tool_tip_offset=True + this offset, and command the CUTTER.\n", flush=True)

env.close(); sim_app.close()
