#!/usr/bin/env python3
"""One-shot: measure the TRUE stepped-physics truck-frame 3x4 EE Jacobian and cache
it to ID_FD_pipline/models/ee_jacobian_truck3x4.json.

Runs standalone because Kit aborts silently if an env is reset after the stepped
snapshot/restore rollouts in the same process."""
import argparse, json, sys
from pathlib import Path
from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
p = argparse.ArgumentParser(); p.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(p); a = p.parse_args()
app = AppLauncher(a); sim_app = app.app

import gymnasium as gym, torch  # noqa: E402
import ID_FD_pipline.tasks  # noqa: E402,F401
from ID_FD_pipline.dls_solver.kinematics import ee_xyz_jacobian_stepped  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

cfg = parse_env_cfg("Kevin-Mulag-IDFD-EETracking-PLAY-v0", device=a.device, num_envs=1)
cfg.enable_scripted_vehicle_motion = False
env = gym.make("Kevin-Mulag-IDFD-EETracking-PLAY-v0", cfg=cfg).unwrapped
env.reset(seed=17)
zero = torch.zeros((1, 4), device=env.device)
for _ in range(60):
    env.step(zero)
J = ee_xyz_jacobian_stepped(env)[0].cpu()
sv = torch.linalg.svdvals(J)
out = ROOT / "ID_FD_pipline/models/ee_jacobian_truck3x4.json"
out.write_text(json.dumps({"jacobian": J.tolist(), "svs": sv.tolist(),
                           "frame": "truck base_link, cutter tip, learned cylinder x"}))
print(f"[MJ] saved {out}  rank={int((sv > 0.01 * sv[0]).sum())}/3 SVs={sv.tolist()}", flush=True)
env.close(); sim_app.close()
