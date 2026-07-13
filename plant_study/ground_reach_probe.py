#!/usr/bin/env python3
"""How FAR can the cutter travel laterally (truck-Y) while staying at ground level,
through the real ID/FD cascade? Sizes the long mowing trajectories honestly."""
import argparse, sys
from pathlib import Path
from isaaclab.app import AppLauncher
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
p = argparse.ArgumentParser(); p.add_argument("--num_envs", type=int, default=1)
p.add_argument("--steps", type=int, default=900)
AppLauncher.add_app_launcher_args(p); a = p.parse_args()
app = AppLauncher(a); sim_app = app.app

import gymnasium as gym, torch, json  # noqa: E402
import ID_FD_pipline.tasks  # noqa: E402,F401
from ID_FD_pipline.dls_solver.cylinder import CylinderDLSConfig, CylinderDLSController  # noqa: E402
from ID_FD_pipline.dls_solver.ee_cascade import EECascadeController  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

cfg = parse_env_cfg("Kevin-Mulag-IDFD-EETracking-PLAY-v0", device=a.device, num_envs=1)
cfg.enable_scripted_vehicle_motion = False
cfg.episode_length_s = 1e6
env = gym.make("Kevin-Mulag-IDFD-EETracking-PLAY-v0", cfg=cfg).unwrapped
env.reset(seed=17)
zero = torch.zeros((1, 4), device=env.device)
for _ in range(30):
    env.step(zero)
J0 = torch.tensor(json.loads((ROOT / "ID_FD_pipline/models/ee_jacobian_truck3x4.json").read_text())["jacobian"], device=env.device)
ctrl = CylinderDLSController(env, CylinderDLSConfig(invert_learned_pipeline=False, error_gain=0.10,
                                                    max_desired_delta_m=0.01, max_action=1.0, action_smoothing=0.35))
cas = EECascadeController(env, cylinder_controller=ctrl, jacobian=J0, max_cylinder_lead_m=0.01)
cas.reset()
anchor = env._compute_ee_pos_local().detach().clone()
cutter_z0 = float(env.robot.data.body_link_state_w[0, env._ee_body_id, 2].item()) - 0.79
ground_anchor = anchor.clone(); ground_anchor[:, 2] -= max(cutter_z0 - 0.05, 0.0)
for _ in range(250):
    ctrl.step(cas.cylinder_goal(ground_anchor))
print(f"[GR] at ground anchor: truck-frame y={float(env._compute_ee_pos_local()[0,1]):.3f}", flush=True)

for sgn, tag in ((+1.0, "+Y"), (-1.0, "-Y")):
    reach = None
    for _ in range(int(a.steps)):
        goal = ground_anchor.clone(); goal[:, 1] += sgn * 5.0   # far beyond reach -> saturate
        ctrl.step(cas.cylinder_goal(goal))
        ee = env._compute_ee_pos_local()
        cz = float(env.robot.data.body_link_state_w[0, env._ee_body_id, 2].item()) - 0.79
        if abs(cz) < 0.15:   # only count positions still at ground level
            dy = float(ee[0, 1] - ground_anchor[0, 1])
            reach = dy if reach is None else (max(reach, dy) if sgn > 0 else min(reach, dy))
    print(f"[GR] {tag}: max ground-level lateral reach from anchor = {reach:+.3f} m", flush=True)
    # return to anchor before the other direction
    for _ in range(400):
        ctrl.step(cas.cylinder_goal(ground_anchor))
print("[GR] done", flush=True)
env.close(); sim_app.close()
