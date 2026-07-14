#!/usr/bin/env python3
"""Collect excitation data from the REPAIRED commandable plant.

What this does and does NOT provide
-----------------------------------
The simulator's cylinders have NO valve input -- the only actuator interface is the PhysX
joint position target.  So this CANNOT produce valve->Delta_x labels, and fabricating them
from DirectCylinderDriver is forbidden (and circular: set_debug_valve_cmd routes valve->dx
through FD itself).

What it DOES provide, which is what the retraining needs:

  * the STATE distribution the arm actually visits on the repaired plant
    (Extension, ExtensionVelocity) over the commandable workspace
  * the plant's true one-step transition  (x, xdot, target_delta_x) -> achieved delta_x,
    sampled on the MODEL's 0.05 s tick

That is the teacher for the identity-composition loss FD(ID(s, dx)) ~= dx and for the
multi-step rollout loss.  FD's valve labels stay the Simscape hydraulic data.

Excitation (per cylinder and coupled), all in cylinder target space:
  steps, slow ramps, triangular reversals, multi-sine, chirp, random-walk holds
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--seconds", type=float, default=120.0, help="sim seconds PER env")
parser.add_argument("--seed", type=int, default=17)
parser.add_argument("--out", default="ID_FD_pipline/models/plant_excitation.npz")
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
MODEL_DT = 0.05  # the ID/FD sample time; every logged transition spans exactly this

cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
cfg.enable_scripted_vehicle_motion = False
cfg.episode_length_s = 1.0e4
cfg.cylinder_success_hold_steps = 10 ** 9
env = gym.make(TASK, cfg=cfg).unwrapped
env.reset(seed=args.seed)

N = env.num_envs
step_dt = float(env.step_dt)
SUB = max(int(round(MODEL_DT / step_dt)), 1)     # control steps per model tick (=3)
print(f"[EXC] step_dt={step_dt:.5f}s  model_dt={MODEL_DT}s  -> {SUB} control steps/tick",
      flush=True)

scaling = env._cylinder_adapter.scaling
lo = scaling.learned_min.to(env.device, torch.float32).reshape(1, 4)
hi = scaling.learned_max.to(env.device, torch.float32).reshape(1, 4)
mid = 0.5 * (lo + hi)
span = (hi - lo)
driver = DirectCylinderDriver(env)

rng = np.random.RandomState(args.seed)
ticks = int(args.seconds / MODEL_DT)

# Per-env excitation programme. Each env gets its own random operating point and mode so
# the pooled dataset covers the workspace, both directions, reversals and coupled motion.
base = (mid + (torch.rand((N, 4), device=env.device) - 0.5) * 0.5 * span).clamp(lo, hi)
for _ in range(300):
    driver.step(base, substeps=2)
base = driver.measured_x().clone()

MODES = ["step", "ramp", "triangle", "multisine", "chirp", "randomwalk", "coupled"]
mode_id = rng.randint(0, len(MODES), size=N)
# amplitudes stay inside the ID/FD training distribution for TargetDeltaExtension
# (data std = [0.0066, 0.0077, 0.0245, 0.0177] m per 0.05 s)
amp = torch.tensor(rng.uniform(0.01, 0.06, size=(N, 4)), device=env.device, dtype=torch.float32)
freq = torch.tensor(rng.uniform(0.05, 0.35, size=(N, 4)), device=env.device, dtype=torch.float32)
phase = torch.tensor(rng.uniform(0, 2 * math.pi, size=(N, 4)), device=env.device, dtype=torch.float32)
active = torch.tensor(rng.randint(0, 4, size=N), device=env.device, dtype=torch.long)
walk = base.clone()

X, XD, TDX, DX, DXD = [], [], [], [], []


def target_for(t_idx: int) -> torch.Tensor:
    t = t_idx * MODEL_DT
    cmd = base.clone()
    for e in range(N):
        m = MODES[mode_id[e]]
        c = int(active[e])
        if m == "step":
            s = 1.0 if (int(t / 2.0) % 2 == 0) else -1.0
            cmd[e, c] = base[e, c] + s * amp[e, c]
        elif m == "ramp":
            s = ((t / 4.0) % 2.0) - 1.0
            cmd[e, c] = base[e, c] + s * amp[e, c]
        elif m == "triangle":
            p = (t / 3.0) % 2.0
            s = p if p <= 1.0 else 2.0 - p
            cmd[e, c] = base[e, c] + (2 * s - 1) * amp[e, c]
        elif m == "multisine":
            v = 0.0
            for k in (1.0, 2.3, 3.7):
                v += math.sin(2 * math.pi * float(freq[e, c]) * k * t + float(phase[e, c]))
            cmd[e, c] = base[e, c] + amp[e, c] * v / 3.0
        elif m == "chirp":
            f = 0.03 + 0.30 * (t / max(args.seconds, 1e-6))
            cmd[e, c] = base[e, c] + amp[e, c] * math.sin(2 * math.pi * f * t)
        elif m == "randomwalk":
            if t_idx % 10 == 0:
                walk[e, c] = walk[e, c] + float(rng.uniform(-1, 1)) * amp[e, c] * 0.5
            cmd[e, c] = walk[e, c]
        elif m == "coupled":  # all four cylinders move together -- cross-axis coverage
            for cc in range(4):
                cmd[e, cc] = base[e, cc] + amp[e, cc] * math.sin(
                    2 * math.pi * float(freq[e, cc]) * t + float(phase[e, cc]))
    return torch.clamp(cmd, lo, hi)


for k in range(ticks):
    x0, xd0 = env._refresh_cylinder_state()
    x0 = x0.clone(); xd0 = xd0.clone()
    cmd = target_for(k)
    tdx = cmd - x0                              # the target delta over this model tick
    for _ in range(SUB):
        driver.step(cmd, substeps=2)
    x1, xd1 = env._refresh_cylinder_state()
    X.append(x0.cpu().numpy()); XD.append(xd0.cpu().numpy())
    TDX.append(tdx.cpu().numpy())
    DX.append((x1 - x0).cpu().numpy())          # ACHIEVED delta over 0.05 s
    DXD.append((xd1 - xd0).cpu().numpy())       # achieved velocity increment
    if k % 200 == 0:
        print(f"[EXC] tick {k}/{ticks}", flush=True)

X = np.concatenate(X); XD = np.concatenate(XD)
TDX = np.concatenate(TDX); DX = np.concatenate(DX); DXD = np.concatenate(DXD)
print(f"\n[EXC] {len(X)} transitions ({N} envs x {ticks} ticks)", flush=True)
print(f"[EXC] extension      range {np.round(X.min(0),3)} .. {np.round(X.max(0),3)}", flush=True)
print(f"[EXC] ext. velocity  std   {np.round(XD.std(0),4)}", flush=True)
print(f"[EXC] target delta   std   {np.round(TDX.std(0),4)}  (ID data std [0.0066,0.0077,0.0245,0.0177])", flush=True)
print(f"[EXC] achieved delta std   {np.round(DX.std(0),4)}", flush=True)
# how faithfully does the plant execute the commanded delta over one model tick?
for c in range(4):
    k = float(np.dot(TDX[:, c], DX[:, c]) / max(np.dot(TDX[:, c], TDX[:, c]), 1e-12))
    r = float(np.corrcoef(TDX[:, c], DX[:, c])[0, 1])
    print(f"[EXC] cyl{c+1}: achieved ~= {k:.3f} * commanded   (corr {r:+.3f})", flush=True)

out = ROOT / args.out
np.savez_compressed(
    out, extension=X, extension_velocity=XD,
    target_delta_x=TDX, achieved_delta_x=DX, achieved_delta_xdot=DXD,
    model_dt=MODEL_DT, seed=args.seed,
)
print(f"[EXC] wrote {out}", flush=True)
env.close()
sim_app.close()
