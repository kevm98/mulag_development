#!/usr/bin/env python3
"""Measure J_eff = d(EE_base)/d(delta_x_cylinder) with D-only backend + drift subtraction."""
from __future__ import annotations
import argparse, sys, os
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Measure J_eff with D-only + drift subtraction.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--dx_m", type=float, default=0.005)
parser.add_argument("--kd_scale", type=float, default=0.2)
parser.add_argument("--horizon_steps", type=int, default=10)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env_cfg import MulagUpperLevelEnvCfg
from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env import MulagUpperLevelEnv
from kevin_integration.utils.sim_memory import apply_kevin_sim_memory_optimizations
from isaaclab.utils.math import quat_apply_inverse

cfg = MulagUpperLevelEnvCfg()
cfg.scene.num_envs = int(args_cli.num_envs)
cfg.enable_scripted_vehicle_motion = False
cfg.enable_startup_summary = False
cfg.log_target_delta_xdot_clipping = False
cfg.allow_debug_manual_cylinder_targets = True
cfg.lowlevel_actuator_backend = "explicit_effort_pd"
cfg.explicit_pd_gain_scale = 0.0
cfg.explicit_pd_damping_scale = float(args_cli.kd_scale)
cfg.explicit_pd_effort_scale = 1.0
cfg.explicit_force_rate_limit_fraction = 0.01
cfg.explicit_pd_velocity_filter_beta = 0.8
if getattr(args_cli, "device", None): cfg.device = args_cli.device; cfg.sim.device = args_cli.device
id_path = (Path(os.getcwd()) / "outputs/cylinder_diagnostics/paper_matched_pipeline/model_finetune/id_physx_finetuned_best.pt").resolve()
cfg.id_checkpoint_override_path = str(id_path)
apply_kevin_sim_memory_optimizations(cfg.sim)

env = MulagUpperLevelEnv(cfg=cfg); uw = env; B = env.num_envs; dev = uw.device
cyl_ids = uw._cylinder_joint_ids; tool_id = uw._tool_joint_id; ee_id = uw._ee_body_id
dt = float(uw.step_dt); H = int(args_cli.horizon_steps)
jnames = uw.robot.data.joint_names
dx = float(args_cli.dx_m)

def tcp_b():
    ee = uw.robot.data.body_link_state_w[:, ee_id, :3]; bp = uw.robot.data.root_pos_w
    bq = uw.robot.data.root_quat_w; bq = bq / bq.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return quat_apply_inverse(bq, ee - bp)

def run_trial(cmd_cyl_c=None, cmd_sign=0):
    """Run one horizon with optional cylinder command. cmd_sign: 0=zero, ±1."""
    env.reset()
    root_pose = uw.robot.data.root_state_w[:, :7].clone()
    root_vel = torch.zeros((B, 6), device=dev)
    cq_hold = uw.robot.data.joint_pos[:, cyl_ids].clone()
    tool_target = uw.robot.data.joint_pos[:, tool_id].unsqueeze(-1).clone()

    # Settle with D-only
    for _ in range(60):
        uw.robot.write_root_pose_to_sim(root_pose)
        uw.robot.write_root_velocity_to_sim(root_vel)
        uw.set_debug_manual_cylinder_targets(cq_hold, tool_position_target=tool_target)
        env.step(torch.zeros((B, 5), device=dev))
    cq_hold = uw.robot.data.joint_pos[:, cyl_ids].clone()

    # Record start
    ee0 = tcp_b().detach().clone()
    q0 = uw.robot.data.joint_pos[:, cyl_ids].detach().clone()

    # Apply command and run horizon
    if cmd_cyl_c is not None and cmd_sign != 0:
        cq_target = cq_hold.clone()
        cq_target[:, cmd_cyl_c] += cmd_sign * dx
    else:
        cq_target = cq_hold

    for _ in range(H):
        uw.robot.write_root_pose_to_sim(root_pose)
        uw.robot.write_root_velocity_to_sim(root_vel)
        uw.set_debug_manual_cylinder_targets(cq_target, tool_position_target=tool_target)
        env.step(torch.zeros((B, 5), device=dev))

    ee1 = tcp_b().detach().clone()
    q1 = uw.robot.data.joint_pos[:, cyl_ids].detach().clone()

    return ee1 - ee0, q1 - q0, ee0, ee1

print("=" * 70, flush=True)
print(f"J_eff MEASUREMENT (D-only kd={args_cli.kd_scale}, dx={dx}m, H={H})", flush=True)
print("=" * 70, flush=True)

J = torch.zeros((1, 3, 4), device=dev)
all_results = []

# For each cylinder, run paired trials
for c in range(4):
    # Zero baseline (3 trials for better statistics)
    dee_zeros = []
    for trial in range(3):
        dee, dq, ee0, ee1 = run_trial(cmd_cyl_c=None, cmd_sign=0)
        dee_zeros.append(dee)

    # Positive
    dee_pos, dq_pos, ee0p, ee1p = run_trial(cmd_cyl_c=c, cmd_sign=+1)

    # Negative
    dee_neg, dq_neg, ee0n, ee1n = run_trial(cmd_cyl_c=c, cmd_sign=-1)

    # Use first zero trial as baseline
    dee_zero = dee_zeros[0]

    # Drift subtraction
    response_pos = (dee_pos - dee_zero)[0].detach().cpu().numpy()
    response_neg = (dee_neg - dee_zero)[0].detach().cpu().numpy()

    # Central difference
    J_col = (response_pos - response_neg) / (2.0 * dx)
    J[0, :, c] = torch.from_numpy(J_col).float().to(dev)

    # Metrics
    baseline_norm = float(torch.norm(dee_zero).item())
    resp_pos_norm = float(np.linalg.norm(response_pos))
    resp_neg_norm = float(np.linalg.norm(response_neg))
    signal_pos = resp_pos_norm / max(baseline_norm, 1e-9)
    signal_neg = resp_neg_norm / max(baseline_norm, 1e-9)
    opposite_check = float(np.dot(response_pos, response_neg))  # should be negative

    dq_c_pos = float(dq_pos[0, c].item()) * 1000  # mm
    dq_c_neg = float(dq_neg[0, c].item()) * 1000

    print(f"\n  cyl{c} ({jnames[cyl_ids[c]]}):", flush=True)
    print(f"    baseline ΔEE = [{dee_zero[0,0]:.4f}, {dee_zero[0,1]:.4f}, {dee_zero[0,2]:.4f}] norm={baseline_norm:.4f}m", flush=True)
    print(f"    +dx ΔEE_raw  = [{dee_pos[0,0]:.4f}, {dee_pos[0,1]:.4f}, {dee_pos[0,2]:.4f}]", flush=True)
    print(f"    -dx ΔEE_raw  = [{dee_neg[0,0]:.4f}, {dee_neg[0,1]:.4f}, {dee_neg[0,2]:.4f}]", flush=True)
    print(f"    response_plus  = [{response_pos[0]:.4f}, {response_pos[1]:.4f}, {response_pos[2]:.4f}] norm={resp_pos_norm:.4f}m s2d={signal_pos:.1f}", flush=True)
    print(f"    response_minus = [{response_neg[0]:.4f}, {response_neg[1]:.4f}, {response_neg[2]:.4f}] norm={resp_neg_norm:.4f}m s2d={signal_neg:.1f}", flush=True)
    print(f"    J_col = [{J_col[0]:.4f}, {J_col[1]:.4f}, {J_col[2]:.4f}]", flush=True)
    print(f"    Δq_c = +{dq_c_pos:+.1f} / {dq_c_neg:+.1f} mm  dot(+,-)={opposite_check:.4f}", flush=True)

    all_results.append({
        "cylinder": c,
        "baseline_norm": baseline_norm,
        "signal_to_drift_pos": signal_pos,
        "signal_to_drift_neg": signal_neg,
        "opposite_dot": opposite_check,
        "J_col": J_col,
        "dq_pos_mm": dq_c_pos,
        "dq_neg_mm": dq_c_neg,
    })

# ---- Quality report ----
print(f"\n{'='*70}", flush=True)
print("QUALITY REPORT", flush=True)
print(f"{'='*70}", flush=True)

J_np = J[0].detach().cpu().numpy()
S = np.linalg.svd(J_np, compute_uv=False)
cond = S[0] / S[-1] if S[-1] > 1e-12 else float("inf")
rank = int(np.sum(S > 0.01 * S[0]))

s2d_min = min(r["signal_to_drift_pos"] for r in all_results)
s2d_all_pass = all(r["signal_to_drift_pos"] > 3.0 and r["signal_to_drift_neg"] > 3.0 for r in all_results)
opposite_all = all(r["opposite_dot"] < 0 for r in all_results)
finite_all = np.all(np.isfinite(J_np))

print(f"  J_eff:", flush=True)
print(f"    {J_np}", flush=True)
print(f"  Singular values: {S}", flush=True)
print(f"  Condition number: {cond:.1f}", flush=True)
print(f"  Rank (tol=0.01): {rank}", flush=True)
print(f"  Finite: {finite_all}", flush=True)
print(f"  Min signal/drift ratio: {s2d_min:.1f}", flush=True)
print(f"  All s2d > 3.0: {s2d_all_pass}", flush=True)
print(f"  All ± opposite: {opposite_all}", flush=True)

pass_j = (finite_all and rank >= 2 and cond < 5000 and s2d_all_pass and opposite_all)
print(f"\n  J_eff VALID: {pass_j}", flush=True)
if not pass_j:
    reasons = []
    if not finite_all: reasons.append("non-finite")
    if rank < 2: reasons.append(f"rank={rank}")
    if cond >= 5000: reasons.append(f"cond={cond:.0f}")
    if not s2d_all_pass: reasons.append("s2d<3")
    if not opposite_all: reasons.append("±not opposite")
    print(f"  FAIL reasons: {', '.join(reasons)}", flush=True)

env.close()
simulation_app.close()
