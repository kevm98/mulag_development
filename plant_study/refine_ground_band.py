#!/usr/bin/env python3
"""Refine coarse ground candidates sequentially in one PhysX environment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--map", default=str(ROOT / "ID_FD_pipline/models/ground_band_map.json"))
p.add_argument("--settle_steps", type=int, default=40)
p.add_argument("--initial_settle_steps", type=int, default=40)
p.add_argument("--num_envs", type=int, default=32)
p.add_argument("--ground_tol_m", type=float, default=0.10)
p.add_argument("--seed", type=int, default=17)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
launcher = AppLauncher(a)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import ID_FD_pipline.tasks  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from plant_study.direct_drive import DirectCylinderDriver  # noqa: E402

path = Path(a.map).expanduser().resolve()
payload = json.loads(path.read_text())
candidates = payload.get("refinement_samples") or payload.get("ground_band", [])
if not candidates:
    raise RuntimeError(f"No coarse ground candidates in {path}")

cfg = parse_env_cfg("Kevin-Mulag-IDFD-EETracking-PLAY-v0", device=a.device, num_envs=int(a.num_envs))
cfg.seed = int(a.seed)
cfg.enable_scripted_vehicle_motion = False
cfg.episode_length_s = 100000.0
env = gym.make("Kevin-Mulag-IDFD-EETracking-PLAY-v0", cfg=cfg).unwrapped

try:
    driver = DirectCylinderDriver(env)
    refined: list[dict[str, object]] = []
    for start in range(0, len(candidates), int(a.num_envs)):
        batch = candidates[start : start + int(a.num_envs)]
        valid = len(batch)
        if valid < int(a.num_envs):
            batch = batch + [batch[-1]] * (int(a.num_envs) - valid)
        # Every pose is measured from the same reset/settle state. This removes
        # passive-linkage branch history from the map contract.
        env.reset(seed=int(a.seed))
        driver.settle(int(a.initial_settle_steps))
        target_values = [sample["target_x"] for sample in batch]
        target = torch.tensor(target_values, device=env.device, dtype=torch.float32).reshape(-1, 4)
        for _ in range(int(a.settle_steps)):
            driver.step(target)
        measured = driver.measured_x()[:valid].detach().cpu().tolist()
        cutter = driver.ee_local()[:valid].detach().cpu().tolist()
        world_z = env._compute_cutter_world_z()[:valid].detach().cpu().tolist()
        for offset in range(valid):
            z_value = float(world_z[offset])
            refined.append(
                {
                    "target_x": [float(value) for value in target_values[offset]],
                    "cylinder_x": [float(value) for value in measured[offset]],
                    "cutter_truck": [float(value) for value in cutter[offset]],
                    "cutter_world_z": z_value,
                    "ground": abs(z_value) <= float(a.ground_tol_m),
                    "refinement_env_index": int(offset),
                }
            )
        print(f"[GROUND-REFINE] sampled={min(start + valid, len(candidates))}/{len(candidates)}", flush=True)

    ground = [sample for sample in refined if bool(sample["ground"])]
    if len(ground) < 8:
        payload["refinement_samples"] = refined
        payload["ground_band"] = ground
        path.write_text(json.dumps(payload, indent=2))
        raise RuntimeError(f"Only {len(ground)} single-environment ground poses survived")
    points = np.asarray([sample["cutter_truck"] for sample in ground], dtype=np.float64)
    extent = {
        "x_min": float(points[:, 0].min()), "x_max": float(points[:, 0].max()),
        "y_min": float(points[:, 1].min()), "y_max": float(points[:, 1].max()),
        "z_min": float(points[:, 2].min()), "z_max": float(points[:, 2].max()),
    }
    payload["coarse_ground_candidates"] = len(candidates)
    payload["refined_single_env"] = True
    payload["refinement_reset_each_batch"] = True
    payload["refinement_num_envs"] = int(a.num_envs)
    payload["refinement_settle_steps"] = int(a.settle_steps)
    payload["refinement_samples"] = refined
    payload["ground_band"] = ground
    payload["extent"] = extent
    path.write_text(json.dumps(payload, indent=2))
    print(
        f"[GROUND-REFINE][DONE] map={path} coarse={len(candidates)} ground={len(ground)} "
        f"x=[{extent['x_min']:.3f},{extent['x_max']:.3f}] "
        f"y=[{extent['y_min']:.3f},{extent['y_max']:.3f}]",
        flush=True,
    )
finally:
    env.close()
    simulation_app.close()
