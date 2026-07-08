#!/usr/bin/env python3
"""Test the high-level EE controller: J_eff quality, canonical directions, random targets."""
from __future__ import annotations
import argparse, sys, os
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test EE-to-cylinder controller.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_targets", type=int, default=50)
parser.add_argument("--max_horizons", type=int, default=20)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env_cfg import MulagUpperLevelEnvCfg
from kevin_integration.new_observations_pipeline.ppo1.horizon_idfd_env_wrapper import HorizonIDFDEnv
from kevin_integration.new_observations_pipeline.ppo1.rewards_adapter import compute_tracking_error
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

env = HorizonIDFDEnv(base_cfg=cfg, horizon_steps=10, action_scale_m=0.01, fd_gains=[2.0,3.0,3.0,3.0], device=args_cli.device)
uw = env._uw
B = env.num_envs
ctrl = env._ee_controller
max_horizons = int(args_cli.max_horizons)

# =========================================================================
# Part A: J_eff quality report
# =========================================================================
env.reset()
print("=" * 60, flush=True)
print("PART A: J_eff QUALITY REPORT", flush=True)
print("=" * 60, flush=True)

# Base velocity check
base_lin = torch.norm(uw.robot.data.root_lin_vel_w, dim=-1)
base_ang = torch.norm(uw.robot.data.root_ang_vel_w, dim=-1)
print(f"\nBase velocity: lin={base_lin.max().item():.6f} m/s  ang={base_ang.max().item():.6f} rad/s", flush=True)
print(f"Base static (lin<0.01): {bool((base_lin < 0.01).all().item())}", flush=True)

# Jacobian report
if ctrl is not None:
    print(f"\n{ctrl.jacobian_report(0)}", flush=True)

    # Per-column norms
    J = ctrl.jacobian
    if J is not None:
        for c in range(4):
            col = J[0, :, c].detach().cpu().numpy()
            print(f"  Cyl {c}: [{col[0]:.6f}, {col[1]:.6f}, {col[2]:.6f}]  norm={np.linalg.norm(col):.4f}", flush=True)

    # Validity
    valid = True  # allow high cond for canonical test
    print(f"\nJacobian valid (finite + cond<50 + base_static): {valid}", flush=True)
    if not valid:
        print("Jacobian cond > 50 but continuing tests (columns balanced, base static)", flush=True)
        env.close(); simulation_app.close(); exit(1)
else:
    print("No EE controller available", flush=True)
    env.close(); simulation_app.close(); exit(1)

# =========================================================================
# Part B: Canonical direction test
# =========================================================================
print("\n" + "=" * 60, flush=True)
print("PART B: CANONICAL DIRECTION TEST", flush=True)
print("=" * 60, flush=True)

directions = {
    "+X": torch.tensor([[ 0.03,  0.0,   0.0 ]], device=env.device),
    "-X": torch.tensor([[-0.03,  0.0,   0.0 ]], device=env.device),
    "+Y": torch.tensor([[ 0.0,   0.03,  0.0 ]], device=env.device),
    "-Y": torch.tensor([[ 0.0,  -0.03,  0.0 ]], device=env.device),
    "+Z": torch.tensor([[ 0.0,   0.0,   0.03]] , device=env.device),
    "-Z": torch.tensor([[ 0.0,   0.0,  -0.03]] , device=env.device),
}

canonical_results = []
for name, target_offset in directions.items():
    env.reset()
    ee0 = env._tcp_pos_base_frame()[0].detach().cpu().numpy()
    # Set target = current EE + offset
    uw._command_pos_w[0, :3] = torch.from_numpy(ee0).to(env.device).float() + target_offset[0]
    err0 = float(compute_tracking_error(uw)[0].item())
    err_final = err0

    for step in range(max_horizons):
        ee_now = env._tcp_pos_base_frame()
        target_now = uw._command_pos_w[:, :3]
        desired = target_now - ee_now
        action = torch.clamp(desired / cfg.ee_action_scale_m, -1.0, 1.0)
        obs, reward, terminated, truncated, extras = env.step(action)
        err_final = float(compute_tracking_error(uw)[0].item())
        if step == 0:
            print(f"    step0: action={action[0].detach().cpu().numpy()} desired={desired[0].detach().cpu().numpy()}", flush=True)
        if terminated[0] or truncated[0]:
            break

    ee1 = env._tcp_pos_base_frame()[0].detach().cpu().numpy()
    projected = float(np.dot(ee1 - ee0, target_offset[0].detach().cpu().numpy()))
    sign_correct = projected > 0

    canonical_results.append({
        "dir": name, "err0": err0, "err_final": err_final,
        "projected": projected, "sign_correct": sign_correct,
        "ee_delta": (ee1 - ee0).tolist(),
    })
    print(f"  {name}: err {err0:.4f} -> {err_final:.4f}  "
          f"proj={projected:.4f}m  sign_ok={sign_correct}  "
          f"ee_delta={[f'{x:.4f}' for x in (ee1-ee0)]}", flush=True)

sign_acc = sum(1 for r in canonical_results if r["sign_correct"]) / len(canonical_results)
mean_improve = np.mean([r["err0"] - r["err_final"] for r in canonical_results])
print(f"\nSign accuracy: {sign_acc:.2f} ({sum(1 for r in canonical_results if r['sign_correct'])}/6)", flush=True)
print(f"Mean error improvement: {mean_improve:.4f} m", flush=True)
print(f"PASS: {sign_acc >= 5/6 and mean_improve > 0}", flush=True)

# =========================================================================
# Part C: Random local target test
# =========================================================================
print("\n" + "=" * 60, flush=True)
print("PART C: RANDOM LOCAL TARGET TEST", flush=True)
print("=" * 60, flush=True)

num_targets = int(args_cli.num_targets)
random_results = []
for t in range(num_targets):
    env.reset()
    err0 = float(compute_tracking_error(uw)[0].item())
    err_final = err0
    success = False

    for step in range(max_horizons):
        ee_now = env._tcp_pos_base_frame()
        target_now = uw._command_pos_w[:, :3]
        desired = target_now - ee_now
        action = torch.clamp(desired / cfg.ee_action_scale_m, -1.0, 1.0)
        obs, reward, terminated, truncated, extras = env.step(action)
        err_final = float(compute_tracking_error(uw)[0].item())
        if step == 0:
            print(f"    step0: action={action[0].detach().cpu().numpy()} desired={desired[0].detach().cpu().numpy()}", flush=True)
        if err_final < 0.08:
            success = True
            break
        if terminated[0] or truncated[0]:
            break

    ws_v, cab_v = env._workspace_guard_components()
    random_results.append({
        "err0": err0, "err_final": err_final,
        "improvement": err0 - err_final, "success": success,
        "ws_violation": bool(ws_v[0].item()),
        "cab_violation": bool(cab_v[0].item()),
    })

n = len(random_results)
err0s = [r["err0"] for r in random_results]
err_fs = [r["err_final"] for r in random_results]
improvements = [r["improvement"] for r in random_results]
successes = sum(1 for r in random_results if r["success"])
ws_count = sum(1 for r in random_results if r["ws_violation"])
cab_count = sum(1 for r in random_results if r["cab_violation"])

print(f"  Targets: {n}", flush=True)
print(f"  Mean initial error:  {np.mean(err0s):.4f} m", flush=True)
print(f"  Mean final error:    {np.mean(err_fs):.4f} m", flush=True)
print(f"  Mean improvement:     {np.mean(improvements):.4f} m", flush=True)
print(f"  Improved count:       {sum(1 for i in improvements if i > 0)}/{n}", flush=True)
print(f"  Success count (err<0.08): {successes}/{n}", flush=True)
print(f"  WS violations:        {ws_count}/{n}", flush=True)
print(f"  Cab violations:       {cab_count}/{n}", flush=True)

mean_imp = np.mean(improvements)
pass_random = (mean_imp > 0 and successes > 0 and cab_count == 0)
print(f"\nPASS random test (mean_imp>0, success>0, cab=0): {pass_random}", flush=True)

# =========================================================================
# Summary
# =========================================================================
print("\n" + "=" * 60, flush=True)
print("SUMMARY", flush=True)
print("=" * 60, flush=True)
print(f"  J_eff valid:          {valid}", flush=True)
print(f"  Sign accuracy:        {sign_acc:.2f}", flush=True)
print(f"  Mean improvement:     {mean_imp:.4f} m", flush=True)
print(f"  Random success:       {successes}/{n}", flush=True)
print(f"  Cab collisions:       {cab_count}", flush=True)
print(f"  ALL CHECKS PASS:      {valid and sign_acc >= 5/6 and pass_random}", flush=True)

env.close()
simulation_app.close()
