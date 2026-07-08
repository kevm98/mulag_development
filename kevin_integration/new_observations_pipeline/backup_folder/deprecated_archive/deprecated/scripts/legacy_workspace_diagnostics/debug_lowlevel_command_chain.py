#!/usr/bin/env python3
"""Debug the low-level command chain: delta_x_des_H -> IDFD -> PhysX -> EE.

Tests zero-command drift, direct cylinder perturbations, and IDFD bypass.
Prints every intermediate value and enforces hard safety assertions.
"""
from __future__ import annotations
import argparse, sys, os
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Low-level command chain debug.")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env_cfg import MulagUpperLevelEnvCfg
from kevin_integration.new_observations_pipeline.ppo1.horizon_idfd_env_wrapper import HorizonIDFDEnv
from kevin_integration.utils.sim_memory import apply_kevin_sim_memory_optimizations

cfg = MulagUpperLevelEnvCfg()
cfg.scene.num_envs = int(args_cli.num_envs)
cfg.enable_scripted_vehicle_motion = False
cfg.enable_startup_summary = False
cfg.log_target_delta_xdot_clipping = False
cfg.horizon_pin_root_pose = False
cfg.horizon_reset_settle_steps = 0
cfg.ee_action_mode = "task_space_delta"
if getattr(args_cli, "device", None):
    cfg.device = args_cli.device
    cfg.sim.device = args_cli.device
id_path = (Path(os.getcwd()) / "outputs/cylinder_diagnostics/paper_matched_pipeline/model_finetune/id_physx_finetuned_best.pt").resolve()
cfg.id_checkpoint_override_path = str(id_path)
apply_kevin_sim_memory_optimizations(cfg.sim)

env = HorizonIDFDEnv(base_cfg=cfg, horizon_steps=10, action_scale_m=0.01,
                     fd_gains=[2.0, 3.0, 3.0, 3.0], device=args_cli.device)
uw = env._uw
B = env.num_envs
H = env._H

# Save/restore robot state
def save_state():
    return {
        "root": uw.robot.data.root_state_w.clone(),
        "joint_pos": uw.robot.data.joint_pos.clone(),
        "joint_vel": uw.robot.data.joint_vel.clone(),
        "hold": env._cylinder_hold_target.clone(),
        "tool": env._tool_hold_target.clone(),
        "prev_action": env._prev_action.clone(),
        "prev_dx": env._prev_delta_x_des.clone(),
        "prev_fd": env._prev_fd_delta.clone(),
        "prev_cmd": env._prev_cmd_delta.clone(),
        "prev_meas": env._prev_meas_delta.clone(),
        "prev_guard": env._prev_guard_flags.clone(),
    }

def restore_state(state):
    uw.robot.write_root_state_to_sim(state["root"])
    uw.robot.write_joint_state_to_sim(
        state["joint_pos"], state["joint_vel"], None,
        torch.arange(B, device=env.device))
    env._cylinder_hold_target = state["hold"].clone()
    env._tool_hold_target = state["tool"].clone()
    env._prev_action = state["prev_action"].clone()
    env._prev_delta_x_des = state["prev_dx"].clone()
    env._prev_fd_delta = state["prev_fd"].clone()
    env._prev_cmd_delta = state["prev_cmd"].clone()
    env._prev_meas_delta = state["prev_meas"].clone()
    env._prev_guard_flags = state["prev_guard"].clone()
    uw.set_debug_manual_cylinder_targets(state["hold"], tool_position_target=state["tool"])
    env._pin_root_pose_if_enabled()
    uw.step(torch.zeros((B, 5), device=env.device))
    env._pin_root_pose_if_enabled()

def get_state_snapshot():
    """Return key state values as a dict."""
    base_lin = torch.norm(uw.robot.data.root_lin_vel_w, dim=-1)
    base_ang = torch.norm(uw.robot.data.root_ang_vel_w, dim=-1)
    ee = env._tcp_pos_base_frame()
    cq = uw.robot.data.joint_pos[:, uw._cylinder_joint_ids]
    return {
        "base_vel_norm": base_lin,
        "base_ang_norm": base_ang,
        "ee_base": ee,
        "cyl_pos": cq,
    }

def run_one_horizon(dx_des_H, use_idfd=True, label=""):
    """Run one horizon and return all intermediate values.

    Args:
        dx_des_H: [B,4] delta_x_des_H
        use_idfd: if False, apply dx_des_H as direct cylinder target delta (diagnostic bypass)
    """
    cq0 = uw.robot.data.joint_pos[:, uw._cylinder_joint_ids].detach().clone()
    ee0 = env._tcp_pos_base_frame().detach().clone()

    # IDFD rollout (or skip)
    if use_idfd:
        x0, xd0 = uw._cylinder_adapter.current_from_cylinder_joints(
            cq0, uw.robot.data.joint_vel[:, uw._cylinder_joint_ids])
        # Clone sequence history for isolated test
        id_hist = uw._sequence_id._step_history.detach().clone()
        fd_hist = uw._sequence_fd._step_history.detach().clone()
        x_r = x0.detach().clone()
        xd_r = xd0.detach().clone()
        fd_sum = torch.zeros((B, 4), device=env.device)
        model_dt = float(getattr(uw, "_sequence_model_dt", 0.05))
        for _ in range(H):
            dx_step = dx_des_H / H
            v_step = dx_step / model_dt
            id_in = torch.cat([dx_step, v_step, x_r, xd_r], dim=-1)
            with torch.no_grad():
                valve = env._id_model((id_in - env._X_mean) / env._X_std)
            uw._sequence_fd.push_current_state(valve, x_r, xd_r)
            fd_d = uw._sequence_fd.predict_delta_extension().delta_extension_raw
            fd_sum += fd_d
            x_r += fd_d
            xd_r[:, :] = v_step
        # Restore sequence history (don't contaminate real state)
        uw._sequence_id._step_history = id_hist
        uw._sequence_fd._step_history = fd_hist
        delta_fd_H = fd_sum
        delta_cmd_raw = env._fd_gains * delta_fd_H
        delta_cmd_H = torch.clamp(delta_cmd_raw, -0.02, 0.02)
    else:
        # Diagnostic bypass: direct cylinder target delta
        delta_fd_H = dx_des_H
        delta_cmd_H = torch.clamp(dx_des_H, -0.0005, 0.0005)

    # Smooth ramp
    neutral = torch.zeros(B, dtype=torch.bool, device=env.device)
    dx_isaac = uw._cylinder_adapter.simscape_delta_to_isaac_delta(delta_cmd_H)
    raw_target = cq0 + dx_isaac
    raw_target = torch.where(neutral.unsqueeze(-1), env._cylinder_hold_target, raw_target)
    lo_g, hi_g = env._joint_limits()
    clamped = torch.clamp(raw_target, lo_g, hi_g)
    guard_flags = (clamped - raw_target).abs().gt(1e-7)
    guarded_rows = env._workspace_guard_rows()
    if guarded_rows.any():
        clamped = torch.where(guarded_rows.unsqueeze(-1), env._cylinder_hold_target, clamped)
        guard_flags = guard_flags | guarded_rows.unsqueeze(-1)
    final_target = torch.nan_to_num(clamped, nan=0.0, posinf=0.0, neginf=0.0)

    zero_action = torch.zeros((B, 5), device=env.device)
    for i in range(1, H + 1):
        frac = i / H
        target = cq0 + frac * (final_target - cq0)
        uw.set_debug_manual_cylinder_targets(target.detach().clone(),
                                              tool_position_target=env._tool_hold_target)
        env._pin_root_pose_if_enabled()
        uw.step(zero_action)
        env._pin_root_pose_if_enabled()

    env._cylinder_hold_target = final_target.detach().clone()
    uw.set_debug_manual_cylinder_targets(env._cylinder_hold_target,
                                          tool_position_target=env._tool_hold_target)

    cq1 = uw.robot.data.joint_pos[:, uw._cylinder_joint_ids].detach().clone()
    ee1 = env._tcp_pos_base_frame().detach().clone()

    amplification = (delta_fd_H.abs() / (dx_des_H.abs() + 1e-9))
    physx_target_inc = final_target - cq0
    measured_cyl_delta = cq1 - cq0
    measured_ee_delta = ee1 - ee0

    return {
        "dx_des_H": dx_des_H, "delta_fd_H": delta_fd_H,
        "delta_cmd_H": delta_cmd_H, "amplification": amplification,
        "physx_target_inc": physx_target_inc,
        "measured_cyl_delta": measured_cyl_delta,
        "measured_ee_delta": measured_ee_delta,
        "ee_delta_norm": torch.norm(measured_ee_delta, dim=-1),
        "guard_flags": guard_flags.float().mean(dim=-1),
        "label": label,
    }

# =========================================================================
# Safety assertion function
# =========================================================================
def assert_safe(r, label):
    violations = []
    ee_norm = r["ee_delta_norm"].max().item()
    dx_max = r["dx_des_H"].abs().max().item()
    fd_max = r["delta_fd_H"].abs().max().item()
    cmd_max = r["delta_cmd_H"].abs().max().item()
    target_max = r["physx_target_inc"].abs().max().item()
    cyl_max = r["measured_cyl_delta"].abs().max().item()

    if ee_norm > 0.20:
        violations.append(f"EE delta {ee_norm:.4f}m > 0.20m HARD LIMIT")
    if dx_max > 0.005:
        violations.append(f"dx_des_H max {dx_max:.6f} > 0.005m")
    if fd_max > 0.05:
        violations.append(f"delta_fd_H max {fd_max:.6f} > 0.05m")
    if cmd_max > 0.01:
        violations.append(f"delta_cmd_H max {cmd_max:.6f} > 0.01m")
    if target_max > 0.01:
        violations.append(f"PhysX target inc max {target_max:.6f} > 0.01m")
    if cyl_max > 0.01:
        violations.append(f"measured cyl delta max {cyl_max:.6f} > 0.01m")

    if violations:
        print(f"  *** SAFETY VIOLATIONS for {label}:", flush=True)
        for v in violations:
            print(f"      {v}", flush=True)
        return False
    return True

# =========================================================================
# MAIN
# =========================================================================
print("=" * 80, flush=True)
print("LOW-LEVEL COMMAND CHAIN DEBUG", flush=True)
print("=" * 80, flush=True)

# ---- Stabilization ----
print("\n--- Stabilizing ---", flush=True)
env.reset()
for _ in range(5):
    uw.set_debug_manual_cylinder_targets(env._cylinder_hold_target,
                                          tool_position_target=env._tool_hold_target)
    env._pin_root_pose_if_enabled()
    uw.step(torch.zeros((B, 5), device=env.device))
    env._pin_root_pose_if_enabled()
snap = get_state_snapshot()
print(f"  base_vel={snap['base_vel_norm'].max().item():.6f}  base_ang={snap['base_ang_norm'].max().item():.6f}", flush=True)
print(f"  ee_base={snap['ee_base'][0].detach().cpu().numpy()}", flush=True)
print(f"  cyl_pos={snap['cyl_pos'][0].detach().cpu().numpy()}", flush=True)

# ---- Save baseline state ----
baseline_state = save_state()

# ---- Test A: Zero command ----
print("\n" + "=" * 80, flush=True)
print("TEST A: ZERO COMMAND DRIFT (20 horizons)", flush=True)
print("=" * 80, flush=True)
for h in range(20):
    restore_state(baseline_state)
    r = run_one_horizon(torch.zeros((B, 4), device=env.device), use_idfd=True, label=f"zero_{h}")
    snap = get_state_snapshot()
    print(f"  horizon {h}: ee_delta={r['ee_delta_norm'].max().item():.4f}m  "
          f"cyl_delta={r['measured_cyl_delta'].abs().max().item():.6f}  "
          f"base_vel={snap['base_vel_norm'].max().item():.6f}", flush=True)
    assert_safe(r, f"zero_{h}")
    baseline_state = save_state()  # accumulate drift for fair test

# ---- Reset and re-stabilize ----
env.reset()
for _ in range(5):
    uw.set_debug_manual_cylinder_targets(env._cylinder_hold_target,
                                          tool_position_target=env._tool_hold_target)
    env._pin_root_pose_if_enabled()
    uw.step(torch.zeros((B, 5), device=env.device))
    env._pin_root_pose_if_enabled()
base_snap0 = get_state_snapshot()
ee0_drift = base_snap0["ee_base"].detach().clone()
baseline_state = save_state()

# ---- Test B: Direct cylinder perturbations (IDFD mode) ----
print("\n" + "=" * 80, flush=True)
print("TEST B: DIRECT CYLINDER PERTURBATIONS (IDFD mode, ±0.5mm)", flush=True)
print("=" * 80, flush=True)

cyl_test_cases = []
for c in range(4):
    for sign in [+1, -1]:
        cyl_test_cases.append((c, sign))

for c, sign in cyl_test_cases:
    restore_state(baseline_state)
    dx = torch.zeros((B, 4), device=env.device)
    dx[:, c] = sign * 0.0005
    r = run_one_horizon(dx, use_idfd=True, label=f"cyl{c}_{'+' if sign>0 else '-'}0.5mm_IDFD")

    amp_str = ", ".join([f"{r['amplification'][0, i].item():.1f}" for i in range(4)])
    print(f"  cyl{c} {sign:+d}*0.5mm IDFD:", flush=True)
    print(f"    dx_des_H={r['dx_des_H'][0].tolist()}", flush=True)
    print(f"    delta_fd_H={[f'{x:.6f}' for x in r['delta_fd_H'][0].tolist()]}", flush=True)
    print(f"    amplification=[{amp_str}]", flush=True)
    print(f"    delta_cmd_H={[f'{x:.6f}' for x in r['delta_cmd_H'][0].tolist()]}", flush=True)
    print(f"    PhysX target inc={[f'{x:.6f}' for x in r['physx_target_inc'][0].tolist()]}", flush=True)
    print(f"    measured cyl delta={[f'{x:.6f}' for x in r['measured_cyl_delta'][0].tolist()]}", flush=True)
    print(f"    measured EE delta={[f'{x:.4f}' for x in r['measured_ee_delta'][0].tolist()]}  norm={r['ee_delta_norm'].max().item():.4f}m", flush=True)
    assert_safe(r, f"cyl{c}_{sign}")

# ---- Test C: Direct cylinder perturbations (DIRECT bypass, no IDFD) ----
print("\n" + "=" * 80, flush=True)
print("TEST C: DIRECT CYLINDER PERTURBATIONS (DIRECT bypass, ±0.5mm)", flush=True)
print("=" * 80, flush=True)

for c, sign in cyl_test_cases:
    restore_state(baseline_state)
    dx = torch.zeros((B, 4), device=env.device)
    dx[:, c] = sign * 0.0005
    r = run_one_horizon(dx, use_idfd=False, label=f"cyl{c}_{'+' if sign>0 else '-'}0.5mm_DIRECT")

    print(f"  cyl{c} {sign:+d}*0.5mm DIRECT:", flush=True)
    print(f"    dx_des_H={r['dx_des_H'][0].tolist()}", flush=True)
    print(f"    delta_cmd_H (clamped)={[f'{x:.6f}' for x in r['delta_cmd_H'][0].tolist()]}", flush=True)
    print(f"    PhysX target inc={[f'{x:.6f}' for x in r['physx_target_inc'][0].tolist()]}", flush=True)
    print(f"    measured cyl delta={[f'{x:.6f}' for x in r['measured_cyl_delta'][0].tolist()]}", flush=True)
    print(f"    measured EE delta={[f'{x:.4f}' for x in r['measured_ee_delta'][0].tolist()]}  norm={r['ee_delta_norm'].max().item():.4f}m", flush=True)
    assert_safe(r, f"cyl{c}_{sign}_direct")

# ---- Test D: Drift subtraction baseline ----
print("\n" + "=" * 80, flush=True)
print("TEST D: BASELINE DRIFT SUBTRACTION", flush=True)
print("=" * 80, flush=True)

# Re-measure with careful drift subtraction
for c, sign in cyl_test_cases:
    # Zero baseline
    restore_state(baseline_state)
    r0 = run_one_horizon(torch.zeros((B, 4), device=env.device), use_idfd=True, label=f"drift_zero_c{c}")
    D0 = r0["measured_ee_delta"].detach().clone()

    # Positive/negative with subtraction
    restore_state(baseline_state)
    dx_pos = torch.zeros((B, 4), device=env.device)
    dx_pos[:, c] = 0.0005
    r_pos = run_one_horizon(dx_pos, use_idfd=True, label=f"drift_pos_c{c}")
    Dpos_raw = r_pos["measured_ee_delta"].detach().clone()
    Dpos = Dpos_raw - D0

    restore_state(baseline_state)
    dx_neg = torch.zeros((B, 4), device=env.device)
    dx_neg[:, c] = -0.0005
    r_neg = run_one_horizon(dx_neg, use_idfd=True, label=f"drift_neg_c{c}")
    Dneg_raw = r_neg["measured_ee_delta"].detach().clone()
    Dneg = Dneg_raw - D0

    J_col = (Dpos - Dneg) / (2.0 * 0.0005)
    print(f"  cyl{c}: D0={[f'{x:.4f}' for x in D0[0].tolist()]}  "
          f"Dpos_raw={[f'{x:.4f}' for x in Dpos_raw[0].tolist()]}  "
          f"Dneg_raw={[f'{x:.4f}' for x in Dneg_raw[0].tolist()]}", flush=True)
    print(f"         Dpos={[f'{x:.4f}' for x in Dpos[0].tolist()]}  "
          f"Dneg={[f'{x:.4f}' for x in Dneg[0].tolist()]}  "
          f"J_col={[f'{x:.4f}' for x in J_col[0].tolist()]}", flush=True)

print("\n" + "=" * 80, flush=True)
print("DEBUG COMPLETE", flush=True)
print("=" * 80, flush=True)

env.close()
simulation_app.close()
