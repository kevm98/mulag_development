#!/usr/bin/env python3
"""Monte Carlo EE Y-Z workspace sampling — no-Jacobian, cylinder-direct actions."""
from __future__ import annotations
import argparse, sys, os, json, time
from pathlib import Path
import numpy as np

print("[WS] launching app", flush=True)
ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--num_samples", type=int, default=5000)
parser.add_argument("--save_path", type=str, default="outputs/ppo1_nojac/workspace_yz.json")
parser.add_argument("--backend", type=str, default="explicit_effort_pid_bias")
parser.add_argument("--kp_scale", type=float, default=0.1)
parser.add_argument("--kd_scale", type=float, default=0.5)
parser.add_argument("--bias_calibration_time_s", type=float, default=5.0)
parser.add_argument("--post_reset_settle_steps", type=int, default=5)
parser.add_argument("--max_cylinder_delta_m", type=float, default=0.0025)
parser.add_argument("--action_smoothing_beta", type=float, default=0.8)
parser.add_argument("--action_rate_limit", type=float, default=0.1)
parser.add_argument("--horizon_steps", type=int, default=10)
parser.add_argument("--ground_z", type=float, default=0.0)
parser.add_argument("--z_clearance", type=float, default=0.05)
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

# --- Build minimal env without any wrapper ---
from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env_cfg import MulagUpperLevelEnvCfg
from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env import MulagUpperLevelEnv
from kevin_integration.utils.sim_memory import apply_kevin_sim_memory_optimizations

cfg = MulagUpperLevelEnvCfg()
cfg.scene.num_envs = int(args_cli.num_envs)
B = cfg.scene.num_envs
device = args_cli.device
cfg.lowlevel_actuator_backend = str(args_cli.backend)
cfg.explicit_pid_kp_scale = float(args_cli.kp_scale)
cfg.explicit_pid_kd_scale = float(args_cli.kd_scale)
cfg.explicit_force_rate_limit_fraction = 0.01
cfg.explicit_pd_velocity_filter_beta = 0.8
cfg.bias_calibration_time_s = float(args_cli.bias_calibration_time_s)
cfg.bias_average_window_s = 0.5
cfg.bias_calibration_ki_scale = 5.0
cfg.allow_debug_manual_cylinder_targets = True
cfg.bias_calibration_requires_static = False
cfg.enable_scripted_vehicle_motion = False
cfg.enable_startup_summary = False
cfg.device = device
cfg.sim.device = device

id_path = (Path(os.getcwd()) / "outputs/cylinder_diagnostics/paper_matched_pipeline/model_finetune/id_physx_finetuned_best.pt").resolve()
cfg.id_checkpoint_override_path = str(id_path)
apply_kevin_sim_memory_optimizations(cfg.sim)

print(f"[WS] building env num_envs={B}", flush=True)
uw = MulagUpperLevelEnv(cfg=cfg)
cyl_ids = uw._cylinder_joint_ids
tool_id = uw._tool_joint_id
ee_id = uw._ee_body_id
H = int(args_cli.horizon_steps)
max_delta = float(args_cli.max_cylinder_delta_m)
smooth_beta = float(args_cli.action_smoothing_beta)
rate_limit = float(args_cli.action_rate_limit)
post_settle = int(args_cli.post_reset_settle_steps)

def get_ee_yz():
    from isaaclab.utils.math import quat_apply_inverse
    ee_w = uw.robot.data.body_link_state_w[:, ee_id, :3]
    bp = uw.robot.data.root_pos_w
    bq = uw.robot.data.root_quat_w
    bq = bq / bq.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return quat_apply_inverse(bq, ee_w - bp)[:, 1:3]

print(f"[WS] reset + calibration", flush=True)
uw.reset()
# Settle
cqh = uw.robot.data.joint_pos[:, cyl_ids].clone()
tool_t = uw.robot.data.joint_pos[:, tool_id].unsqueeze(-1).clone()
for _ in range(post_settle):
    uw.set_debug_manual_cylinder_targets(cqh, tool_position_target=tool_t)
    uw.step(torch.zeros((B, 5), device=device))

# Record initial EE position
ee_init = get_ee_yz().clone()
print(f"[WS] ee_init_yz mean=({ee_init[:,0].mean():.3f}, {ee_init[:,1].mean():.3f})", flush=True)

# Get cylinder limits
cyl_lims = uw.robot.data.soft_joint_pos_limits[:, cyl_ids]
cyl_lo = cyl_lims[..., 0]
cyl_hi = cyl_lims[..., 1]

# Start sampling
processed = torch.zeros((B, 4), device=device)
num_samples = int(args_cli.num_samples)
needed = num_samples
accepted = []
rejected = 0

total_steps = 0
max_steps = num_samples * 5  # prevent infinite loop
ground_z = float(args_cli.ground_z)
z_clear = float(args_cli.z_clearance)

print(f"[WS] sampling {num_samples} points...", flush=True)
t0 = time.time()

while len(accepted) < needed and total_steps < max_steps:
    # Generate random smooth cylinder action
    raw = (torch.rand(B, 4, device=device) * 2.0 - 1.0) * 0.5  # smaller actions for safety

    # Rate limit
    if rate_limit > 0:
        delta = raw - processed
        delta = torch.clamp(delta, -rate_limit, rate_limit)
        raw = processed + delta

    # Smooth
    processed = smooth_beta * processed + (1.0 - smooth_beta) * raw

    # Scale to cylinder delta
    dc = processed * max_delta

    # Apply to current positions
    cqh = uw.robot.data.joint_pos[:, cyl_ids].clone()
    cqt = torch.clamp(cqh + dc, cyl_lo, cyl_hi)

    # Step
    ee0 = get_ee_yz().detach().clone()
    root0 = uw.robot.data.root_pos_w.clone()
    for _ in range(H):
        uw.set_debug_manual_cylinder_targets(cqt, tool_position_target=tool_t)
        uw.step(torch.zeros((B, 5), device=device))
    ee1 = get_ee_yz()
    root1 = uw.robot.data.root_pos_w

    jump = torch.norm(ee1 - ee0, dim=-1)
    root_motion = torch.norm(root1 - root0, dim=-1)
    cyl_pos = uw.robot.data.joint_pos[:, cyl_ids]
    ee_w_z = uw.robot.data.body_link_state_w[:, ee_id, 2]  # world Z

    # Evaluate per env
    for i in range(B):
        total_steps += 1
        if total_steps > max_steps:
            break

        # Reject conditions
        if jump[i] > 0.08:
            rejected += 1; continue
        if root_motion[i] > 0.02:
            rejected += 1; continue
        if ee_w_z[i] < ground_z + z_clear:
            rejected += 1; continue
        # Check cylinder limits
        near_limit = ((cyl_pos[i] - cyl_lo[i]) < 0.02 * (cyl_hi[i] - cyl_lo[i])).any()
        near_hi = ((cyl_hi[i] - cyl_pos[i]) < 0.02 * (cyl_hi[i] - cyl_lo[i])).any()
        if near_limit or near_hi:
            rejected += 1; continue

        accepted.append(ee1[i].cpu().tolist())

        if len(accepted) % 500 == 0:
            elapsed = time.time() - t0
            print(f"[WS] {len(accepted)}/{needed} accepted, {rejected} rejected, {elapsed:.0f}s", flush=True)

    if len(accepted) >= needed:
        break

elapsed = time.time() - t0
acc_rate = len(accepted) / max(len(accepted) + rejected, 1) * 100
print(f"[WS] done: {len(accepted)} accepted, {rejected} rejected, rate={acc_rate:.1f}%, {elapsed:.0f}s", flush=True)

accepted = np.array(accepted)
y_vals = accepted[:, 0]
z_vals = accepted[:, 1]

p5_y, p95_y = np.percentile(y_vals, [5, 95])
p5_z, p95_z = np.percentile(z_vals, [5, 95])
margin_y = (p95_y - p5_y) * 0.10
margin_z = (p95_z - p5_z) * 0.10
safe_y_min = float(p5_y + margin_y)
safe_y_max = float(p95_y - margin_y)
safe_z_min = float(max(p5_z + margin_z, ground_z + z_clear))
safe_z_max = float(p95_z - margin_z)

# Farthest pair for line
from scipy.spatial.distance import pdist, squareform
dists = squareform(pdist(accepted))
i, j = np.unravel_index(np.argmax(dists), dists.shape)
line_p1 = accepted[i].tolist()
line_p2 = accepted[j].tolist()
line_len = float(dists[i, j])
# Shrink by 10%
center_line = (accepted[i] + accepted[j]) / 2
line_p1_safe = (center_line + (accepted[i] - center_line) * 0.9).tolist()
line_p2_safe = (center_line + (accepted[j] - center_line) * 0.9).tolist()

center_yz = [float(np.median(y_vals)), float(np.median(z_vals))]
half_y = min(abs(safe_y_max - center_yz[0]), abs(center_yz[0] - safe_y_min))
half_z = min(abs(safe_z_max - center_yz[1]), abs(center_yz[1] - safe_z_min))
ellipse_ry = float(half_y * 0.8)
ellipse_rz = float(half_z * 0.8)

workspace = {
    "center_yz": center_yz,
    "safe_y_min": safe_y_min,
    "safe_y_max": safe_y_max,
    "safe_z_min": safe_z_min,
    "safe_z_max": safe_z_max,
    "ground_z": ground_z,
    "z_clearance": z_clear,
    "accepted_samples": len(accepted),
    "rejected_samples": rejected,
    "acceptance_rate": round(acc_rate, 1),
    "recommended_target_y_range": round((safe_y_max - safe_y_min) / 2, 4),
    "recommended_target_z_range": round((safe_z_max - safe_z_min) / 2, 4),
    "line_endpoints": [line_p1_safe, line_p2_safe],
    "line_length": round(line_len * 0.9, 4),
    "ellipse_radius_y": ellipse_ry,
    "ellipse_radius_z": ellipse_rz,
    "curve_amplitude_y": round(half_y * 0.7, 4),
    "curve_amplitude_z": round(half_z * 0.7, 4),
    "curve_length": round(line_len * 0.9, 4),
    "num_envs": B,
    "max_cylinder_delta_m": max_delta,
}

os.makedirs(os.path.dirname(args_cli.save_path), exist_ok=True)
with open(args_cli.save_path, 'w') as f:
    json.dump(workspace, f, indent=2)
print(f"[WS] saved: {args_cli.save_path}", flush=True)
print(f"[WS] workspace: Y=[{safe_y_min:.3f}, {safe_y_max:.3f}] Z=[{safe_z_min:.3f}, {safe_z_max:.3f}]", flush=True)

uw.close()
simulation_app.close()
print("[WS] done", flush=True)
