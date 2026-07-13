#!/usr/bin/env python3
"""One-shot physics-settled map of the cutter's ground-level workspace.

This deliberately bypasses ID/FD and commands cylinder POSITION TARGETS while
stepping PhysX, so the passive closed-loop linkage is solved at every sample.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--grid", type=int, default=6, help="Samples per cylinder; 6 gives 1296 poses.")
p.add_argument("--settle_steps", type=int, default=40, help="Direct-driver control steps per pose.")
p.add_argument("--num_envs", type=int, default=32, help="Parallel physics-settled poses.")
p.add_argument("--ground_tol_m", type=float, default=0.10)
p.add_argument("--seed", type=int, default=17)
p.add_argument("--output", default=str(ROOT / "ID_FD_pipline/models/ground_band_map.json"))
p.add_argument("--sub_box", default="", help="lo0,lo1,lo2,lo3:hi0,hi1,hi2,hi3 -- restrict the grid to this cylinder-space box (densify a known-flat region)")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
if a.grid < 2:
    raise ValueError("--grid must be at least 2")
if a.settle_steps < 1:
    raise ValueError("--settle_steps must be positive")
if a.num_envs < 1:
    raise ValueError("--num_envs must be positive")

launcher = AppLauncher(a)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import ID_FD_pipline.tasks  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from plant_study.direct_drive import DirectCylinderDriver  # noqa: E402

TASK = "Kevin-Mulag-IDFD-EETracking-PLAY-v0"
cfg = parse_env_cfg(TASK, device=a.device, num_envs=int(a.num_envs))
cfg.seed = int(a.seed)
cfg.enable_scripted_vehicle_motion = False
cfg.episode_length_s = 100000.0
env = gym.make(TASK, cfg=cfg).unwrapped


def _angular_path_estimate(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    center = points[:, :2].mean(axis=0)
    angle = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angle), :2]
    return float(np.linalg.norm(np.diff(ordered, axis=0), axis=1).sum())


try:
    env.reset(seed=int(a.seed))
    driver = DirectCylinderDriver(env)
    driver.settle(60)
    lo = driver.lo[0].detach().cpu().numpy()
    hi = driver.hi[0].detach().cpu().numpy()
    if a.sub_box:
        lo_s, hi_s = a.sub_box.split(":")
        lo = np.maximum(lo, np.array([float(v) for v in lo_s.split(",")], dtype=np.float32))
        hi = np.minimum(hi, np.array([float(v) for v in hi_s.split(",")], dtype=np.float32))
        print(f"[GROUND-MAP] sub-box lo={np.round(lo,3).tolist()} hi={np.round(hi,3).tolist()}", flush=True)
    axes = [np.linspace(lo[i], hi[i], int(a.grid), dtype=np.float32) for i in range(4)]
    targets = list(itertools.product(*axes))
    samples: list[dict[str, object]] = []
    print(
        f"[GROUND-MAP] grid={a.grid} poses={len(targets)} envs={a.num_envs} "
        f"settle_steps={a.settle_steps} "
        f"ground_tol={a.ground_tol_m:.3f}m",
        flush=True,
    )
    for start in range(0, len(targets), int(a.num_envs)):
        batch_values = targets[start : start + int(a.num_envs)]
        valid = len(batch_values)
        if valid < int(a.num_envs):
            batch_values = batch_values + [batch_values[-1]] * (int(a.num_envs) - valid)
        target = torch.tensor(batch_values, device=env.device, dtype=torch.float32).reshape(-1, 4)
        for _ in range(int(a.settle_steps)):
            driver.step(target)
        measured = driver.measured_x()[:valid].detach().cpu().tolist()
        cutter_truck = driver.ee_local()[:valid].detach().cpu().tolist()
        cutter_world_z = env._compute_cutter_world_z()[:valid].detach().cpu().tolist()
        for offset, values in enumerate(batch_values[:valid]):
            samples.append(
                {
                    "target_x": [float(v) for v in values],
                    "cylinder_x": [float(v) for v in measured[offset]],
                    "cutter_truck": [float(v) for v in cutter_truck[offset]],
                    "cutter_world_z": float(cutter_world_z[offset]),
                    "ground": abs(float(cutter_world_z[offset])) <= float(a.ground_tol_m),
                }
            )
        print(f"[GROUND-MAP] sampled={min(start + valid, len(targets))}/{len(targets)}", flush=True)

    ground = [sample for sample in samples if bool(sample["ground"])]
    pts = np.asarray([sample["cutter_truck"] for sample in ground], dtype=np.float64)
    extent = None if not len(ground) else {
        "x_min": float(pts[:, 0].min()), "x_max": float(pts[:, 0].max()),
        "y_min": float(pts[:, 1].min()), "y_max": float(pts[:, 1].max()),
        "z_min": float(pts[:, 2].min()), "z_max": float(pts[:, 2].max()),
    }
    output = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "frame": "truck base_link",
        "control_point": "cutter",
        "tool_tip_local_offset_z": float(getattr(env.cfg, "tool_tip_local_offset_z", -0.79)),
        "grid_per_cylinder": int(a.grid),
        "settle_steps": int(a.settle_steps),
        "ground_tolerance_m": float(a.ground_tol_m),
        "learned_min": lo.tolist(),
        "learned_max": hi.tolist(),
        "extent": extent,
        "angular_order_path_estimate_m": _angular_path_estimate(pts),
        "samples": samples,
        "ground_band": ground,
    }
    out = Path(a.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2))
    if len(ground) < 8:
        all_z = np.asarray([sample["cutter_world_z"] for sample in samples], dtype=np.float64)
        print(
            f"[GROUND-MAP][INSUFFICIENT] saved all samples to {out}; ground={len(ground)} "
            f"world_z=[{all_z.min():.3f},{all_z.max():.3f}]",
            flush=True,
        )
        raise RuntimeError(f"Only {len(ground)} ground-band samples; refine the grid")
    print(
        f"[GROUND-MAP][DONE] output={out} all={len(samples)} ground={len(ground)} "
        f"x=[{extent['x_min']:.3f},{extent['x_max']:.3f}] "
        f"y=[{extent['y_min']:.3f},{extent['y_max']:.3f}] "
        f"z=[{extent['z_min']:.3f},{extent['z_max']:.3f}] "
        f"angular_path_est={output['angular_order_path_estimate_m']:.3f}m",
        flush=True,
    )
finally:
    env.close()
    simulation_app.close()
