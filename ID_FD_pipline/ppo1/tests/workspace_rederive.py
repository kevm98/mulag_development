#!/usr/bin/env python3
"""Re-derive the commandable+holdable cutter workspace on the FIXED plant.

ground_band_map.json was built when the cylinder drives were saturated, so it
recorded SAG poses: where the arm fell to, not where it was told to go (its
target_x and cylinder_x columns differ by up to 0.25 m).  Every artifact derived
from it -- the mowing arc fit, the 0.88 m arc length, the ground Jacobian -- is
therefore invalid.

This sweeps cylinder commands on the repaired plant and keeps only poses that are
genuinely EXECUTED:

    reached  : |x_measured - x_commanded| < tol  (the arm actually got there)
    unsaturated : no cylinder pinned at its effort limit
    holdable : stays put when the command is held

For each valid pose it records the cutter position, so a mowing path can be fitted
to poses the arm can really reach.  Batched across envs.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--grid", type=int, default=5, help="samples per cylinder axis")
parser.add_argument("--settle_steps", type=int, default=240)
parser.add_argument("--reach_tol_m", type=float, default=0.005)
parser.add_argument("--ground_tol_m", type=float, default=0.05)
parser.add_argument("--out", default="ID_FD_pipline/models/commandable_workspace.json")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
sim_app = app_launcher.app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
import ID_FD_pipline.tasks  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from plant_study.direct_drive import DirectCylinderDriver  # noqa: E402

TASK = "Kevin-Mulag-IDFD-CylinderSpace-PLAY-v0"
cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
cfg.enable_scripted_vehicle_motion = False
cfg.episode_length_s = 1.0e4
env = gym.make(TASK, cfg=cfg).unwrapped
env.reset(seed=17)

IDS = env._cylinder_joint_ids
N = env.num_envs
scaling = env._cylinder_adapter.scaling
lo = scaling.learned_min.to(env.device, torch.float32)
hi = scaling.learned_max.to(env.device, torch.float32)
driver = DirectCylinderDriver(env)
print(f"[WS] learned range lo={lo.tolist()} hi={hi.tolist()}", flush=True)

# Grid over the learned cylinder range, inset from the limits so the drive is not
# fighting a joint limit.
g = args.grid
axes = [np.linspace(float(lo[c]) + 0.02, float(hi[c]) - 0.02, g) for c in range(4)]
grid = np.array(list(itertools.product(*axes)), dtype=np.float32)
print(f"[WS] {len(grid)} candidate poses, {N} envs -> {int(np.ceil(len(grid)/N))} batches", flush=True)


EFFORT_LIMIT = torch.tensor(
    [float(torch.as_tensor(env.robot.actuators[k].effort_limit_sim).flatten()[0])
     for k in ("drehzapfen_cylinder", "ausleger_i_cylinder",
               "ausleger_ii_cylinder", "messerkopf_schwenk_cylinder")],
    device=env.device, dtype=torch.float32,
).reshape(1, 4)
print(f"[WS] effort_limit_sim={EFFORT_LIMIT[0].tolist()}", flush=True)


def saturated():
    return env.robot.data.computed_torque[:, IDS].abs() >= EFFORT_LIMIT * 0.999


samples = []
for b0 in range(0, len(grid), N):
    chunk = grid[b0:b0 + N]
    cmd = torch.zeros((N, 4), device=env.device)
    cmd[: len(chunk)] = torch.from_numpy(chunk).to(env.device)
    cmd[len(chunk):] = torch.from_numpy(chunk[-1]).to(env.device)  # pad

    env.reset(seed=17)
    for _ in range(args.settle_steps):
        driver.step(cmd, substeps=2)
    x1 = driver.measured_x().clone()
    sat1 = saturated().clone()
    # holdability: keep commanding the same thing, must not drift
    for _ in range(60):
        driver.step(cmd, substeps=2)
    x2 = driver.measured_x().clone()
    cutter_truck = env._compute_ee_pos_local()
    cutter_z_w = env._compute_cutter_world_z()

    reach = (x1 - cmd).abs().amax(dim=1)
    drift = (x2 - x1).abs().amax(dim=1)
    sat = sat1.any(dim=1)
    for i in range(len(chunk)):
        samples.append(dict(
            cylinder_cmd=cmd[i].tolist(),
            cylinder_measured=x1[i].tolist(),
            reach_err_m=float(reach[i]),
            hold_drift_m=float(drift[i]),
            saturated=bool(sat[i]),
            cutter_truck=cutter_truck[i].tolist(),
            cutter_world_z=float(cutter_z_w[i]),
            valid=bool(reach[i] < args.reach_tol_m and drift[i] < 0.002 and not bool(sat[i])),
        ))
    done = min(b0 + N, len(grid))
    nv = sum(s["valid"] for s in samples)
    print(f"[WS] {done}/{len(grid)} poses  valid={nv}", flush=True)

valid = [s for s in samples if s["valid"]]
ground = [s for s in valid if abs(s["cutter_world_z"]) < args.ground_tol_m]
print(f"\n[WS] executed+holdable: {len(valid)}/{len(samples)}", flush=True)
print(f"[WS] of those, cutter within {args.ground_tol_m} m of ground: {len(ground)}", flush=True)

if valid:
    cut = np.array([s["cutter_truck"] for s in valid])
    print(f"[WS] cutter truck-frame extent  x[{cut[:,0].min():.2f},{cut[:,0].max():.2f}] "
          f"y[{cut[:,1].min():.2f},{cut[:,1].max():.2f}] z[{cut[:,2].min():.2f},{cut[:,2].max():.2f}]",
          flush=True)
    zw = np.array([s["cutter_world_z"] for s in valid])
    print(f"[WS] cutter world z range [{zw.min():.3f}, {zw.max():.3f}]", flush=True)
if ground:
    cg = np.array([s["cutter_truck"] for s in ground])
    print(f"[WS] GROUND band extent  x[{cg[:,0].min():.2f},{cg[:,0].max():.2f}] "
          f"y[{cg[:,1].min():.2f},{cg[:,1].max():.2f}]", flush=True)

out = ROOT / args.out
out.write_text(json.dumps(dict(
    schema_version=1,
    note="Commandable+holdable poses on the repaired plant (effort_limit_sim x16). "
         "Supersedes ground_band_map.json, which recorded saturated sag poses.",
    grid=args.grid, reach_tol_m=args.reach_tol_m, ground_tol_m=args.ground_tol_m,
    learned_min=lo.tolist(), learned_max=hi.tolist(),
    num_samples=len(samples), num_valid=len(valid), num_ground=len(ground),
    samples=samples,
), indent=1))
print(f"[WS] wrote {out}", flush=True)

env.close()
sim_app.close()
