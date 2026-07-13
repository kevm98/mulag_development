#!/usr/bin/env python3
"""Find a cylinder actuator config that is a real force-limited POSITION SERVO.

The authored config (stiffness 3e7-7.5e7, effort_limit 5e4-1e5) clips to max force
beyond 0.7-10 mrad of error, so d(measured)/d(target) ~ 0 and no controller has
authority.  This sweeps (stiffness, damping, effort) candidates and scores each on
what a position servo must actually do:

    * hold a commanded pose with ~zero steady-state error
    * respond to a target step with unity gain
    * settle quickly, without overshoot or limit-cycling
    * not sit pinned at the effort limit

Plant only -- ID/FD/DLS are not involved.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="outputs/lower_level_tracker/actuator_tuning.json")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.num_envs = 1
app_launcher = AppLauncher(args)
sim_app = app_launcher.app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
import ID_FD_pipline.tasks  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from plant_study.direct_drive import DirectCylinderDriver  # noqa: E402

TASK = "Kevin-Mulag-IDFD-CylinderSpace-PLAY-v0"

cfg = parse_env_cfg(TASK, device=args.device, num_envs=1)
cfg.enable_scripted_vehicle_motion = False
cfg.episode_length_s = 1.0e4
env = gym.make(TASK, cfg=cfg).unwrapped
env.reset(seed=17)

IDS = env._cylinder_joint_ids
DT = float(env.sim.get_physics_dt())
K0 = env.robot.data.joint_stiffness[0, IDS].clone()
D0 = env.robot.data.joint_damping[0, IDS].clone()
E0 = env.robot.data.joint_effort_limits[0, IDS].clone()
print(f"[BASE] K={K0.tolist()}\n[BASE] D={D0.tolist()}\n[BASE] E={E0.tolist()}", flush=True)

# (name, stiffness_mult, damping_mult, effort_mult)
# Effort must exceed the true load (measured: cyl1 stalls below ~16x).  Stiffness must
# be low enough that the operating error band stays INSIDE the effort limit:
#   linear band = effort / stiffness.  We want that >= ~0.1 rad of joint travel.
CANDIDATES = [
    ("effort_x4",   1.0, 1.0,   4.0),
    ("effort_x8",   1.0, 1.0,   8.0),
    ("effort_x16",  1.0, 1.0,  16.0),
    ("effort_x32",  1.0, 1.0,  32.0),
    ("effort_x64",  1.0, 1.0,  64.0),
    ("effort_x128", 1.0, 1.0, 128.0),
]


def set_gains(km, dm, em):
    n = env.num_envs
    env.robot.write_joint_stiffness_to_sim((K0 * km).reshape(1, 4).expand(n, 4), joint_ids=IDS)
    env.robot.write_joint_damping_to_sim((D0 * dm).reshape(1, 4).expand(n, 4), joint_ids=IDS)
    env.robot.write_joint_effort_limit_to_sim((E0 * em).reshape(1, 4).expand(n, 4), joint_ids=IDS)


def sat_frac():
    lim = env.robot.data.joint_effort_limits[0, IDS]
    return (env.robot.data.computed_torque[0, IDS].abs() >= lim * 0.999).float()


def evaluate(name, km, dm, em):
    env.reset(seed=17)
    set_gains(km, dm, em)
    driver = DirectCylinderDriver(env)

    # --- hold the arm where it already is: steady-state error of a pure hold ---
    x0 = driver.measured_x().clone()
    sat_hold = []
    for _ in range(300):
        driver.step(x0, substeps=2)
        sat_hold.append(sat_frac())
    x_hold = driver.measured_x()
    hold_err = (x_hold - x0).abs()[0]
    sat_hold = torch.stack(sat_hold).mean(dim=0)

    base = driver.measured_x().clone()
    per_cyl = []
    for c in range(4):
        step = 0.020  # 20 mm target step, well inside the learned range
        cmd = base.clone()
        cmd[0, c] += step
        traj, sat = [], []
        for _ in range(300):  # 5 s
            driver.step(cmd, substeps=2)
            traj.append(float(driver.measured_x()[0, c]))
            sat.append(float(sat_frac()[c]))
        traj = np.array(traj) - float(base[0, c])
        final = traj[-1]
        gain = final / step
        peak = traj.max() if step > 0 else traj.min()
        overshoot = (peak - final) / abs(step) if abs(step) > 1e-9 else 0.0
        # settle time: first index after which |x - final| stays < 10% of step
        band = 0.1 * abs(step)
        idx = np.where(np.abs(traj - final) > band)[0]
        settle = (float(idx[-1] + 1) / 60.0) if len(idx) else 0.0
        # residual oscillation over the last second
        ripple = float(np.std(traj[-60:]))
        per_cyl.append(dict(cylinder=c + 1, gain=float(gain), overshoot=float(overshoot),
                            settle_s=settle, ripple_m=ripple, sat_frac=float(np.mean(sat))))
        for _ in range(240):  # return
            driver.step(base, substeps=2)

    gains = np.array([p["gain"] for p in per_cyl])
    result = dict(
        name=name, stiffness_mult=km, damping_mult=dm, effort_mult=em,
        stiffness=(K0 * km).tolist(), damping=(D0 * dm).tolist(), effort=(E0 * em).tolist(),
        linear_band_rad=(E0 * em / (K0 * km)).tolist(),
        hold_err_mm=(hold_err * 1000).tolist(),
        hold_sat_frac=sat_hold.tolist(),
        per_cylinder=per_cyl,
        gain_error=float(np.abs(gains - 1.0).max()),
        worst_overshoot=float(max(p["overshoot"] for p in per_cyl)),
        worst_settle_s=float(max(p["settle_s"] for p in per_cyl)),
        worst_ripple_mm=float(max(p["ripple_m"] for p in per_cyl) * 1000),
        mean_sat=float(np.mean([p["sat_frac"] for p in per_cyl])),
    )
    print(f"\n=== {name}  K x{km} D x{dm} E x{em} ===", flush=True)
    print(f"  linear band (rad) : {[f'{v:.3g}' for v in result['linear_band_rad']]}", flush=True)
    print(f"  hold err (mm)     : {[f'{v:.3f}' for v in result['hold_err_mm']]}  "
          f"sat={[f'{v:.2f}' for v in result['hold_sat_frac']]}", flush=True)
    for p in per_cyl:
        print(f"  cyl{p['cylinder']} +20mm: gain={p['gain']:+.3f} overshoot={p['overshoot']:+.2f} "
              f"settle={p['settle_s']:.2f}s ripple={p['ripple_m']*1000:.3f}mm sat={p['sat_frac']:.2f}",
              flush=True)
    return result


results = [evaluate(*c) for c in CANDIDATES]
out = ROOT / args.out
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(results, indent=2))

print("\n\n==================== RANKING (position-servo quality) ====================", flush=True)
print(f"{'candidate':26s} {'|gain-1|':>9s} {'overshoot':>10s} {'settle_s':>9s} {'ripple_mm':>10s} {'sat':>6s}", flush=True)
for r in sorted(results, key=lambda r: (r["gain_error"], r["worst_overshoot"])):
    print(f"{r['name']:26s} {r['gain_error']:9.3f} {r['worst_overshoot']:10.2f} "
          f"{r['worst_settle_s']:9.2f} {r['worst_ripple_mm']:10.3f} {r['mean_sat']:6.2f}", flush=True)

env.close()
sim_app.close()
