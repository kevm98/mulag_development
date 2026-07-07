#!/usr/bin/env python3
"""Test whether explicit effort PD can produce nonzero torque and hold the arm."""
from __future__ import annotations
import argparse, sys, os
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test drive activation methods.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--gravity_off", action="store_true", default=False)
parser.add_argument("--method", type=str, default="explicit_effort_pd", choices=["position_drive","explicit_effort_pd","both"])
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env_cfg import MulagUpperLevelEnvCfg
from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env import MulagUpperLevelEnv
from kevin_integration.utils.sim_memory import apply_kevin_sim_memory_optimizations

cfg = MulagUpperLevelEnvCfg()
cfg.scene.num_envs = int(args_cli.num_envs)
cfg.enable_scripted_vehicle_motion = False
cfg.enable_startup_summary = False
cfg.log_target_delta_xdot_clipping = False
if getattr(args_cli, "device", None):
    cfg.device = args_cli.device; cfg.sim.device = args_cli.device

# Gravity control
if args_cli.gravity_off:
    cfg.sim.gravity = (0.0, 0.0, 0.0)

id_path = (Path(os.getcwd()) / "outputs/cylinder_diagnostics/paper_matched_pipeline/model_finetune/id_physx_finetuned_best.pt").resolve()
cfg.id_checkpoint_override_path = str(id_path)
apply_kevin_sim_memory_optimizations(cfg.sim)

env = MulagUpperLevelEnv(cfg=cfg)
uw = env; B = env.num_envs; dev = uw.device
cyl_ids = uw._cylinder_joint_ids
jnames = uw.robot.data.joint_names
dt = float(uw.step_dt)

# PD gains for explicit effort (matching implicit config)
kp = torch.tensor([15_000_000, 6_000_000, 1_000_000, 1_000_000], device=dev, dtype=torch.float32)
kd = torch.tensor([1_000_000, 600_000, 500_000, 500_000], device=dev, dtype=torch.float32)
effort_limit = torch.tensor([100_000_000, 10_000_000, 1_000_000, 100_000], device=dev, dtype=torch.float32)

print("=" * 60, flush=True)
print(f"DRIVE ACTIVATION TEST: method={args_cli.method} gravity={'OFF' if args_cli.gravity_off else 'ON'}", flush=True)
print("=" * 60, flush=True)

env.reset()

# Pin root
root_pose = uw.robot.data.root_state_w[:, :7].clone()
root_vel = torch.zeros((B, 6), device=dev)

# Target = current position
q_target = uw.robot.data.joint_pos[:, cyl_ids].clone()
print(f"Initial joint pos: {q_target[0].tolist()}", flush=True)

# --- Test A: Step target +0.05 rad on Ausleger_I (joint 1) ---
test_joint = 1  # Ausleger_I
q_target[0, test_joint] += 0.05
print(f"Target: {q_target[0].tolist()} (Ausleger_I +0.05 rad)", flush=True)

for step in range(int(2.0 / dt)):
    uw.robot.write_root_pose_to_sim(root_pose)
    uw.robot.write_root_velocity_to_sim(root_vel)

    q_now = uw.robot.data.joint_pos[:, cyl_ids]
    qd_now = uw.robot.data.joint_vel[:, cyl_ids]

    # Explicit PD torque
    tau = kp.unsqueeze(0) * (q_target - q_now) - kd.unsqueeze(0) * qd_now
    tau = torch.clamp(tau, -effort_limit.unsqueeze(0), effort_limit.unsqueeze(0))

    # Apply effort target
    uw.robot.set_joint_effort_target(tau, joint_ids=cyl_ids)
    env.step(torch.zeros((B, 5), device=dev))

    if step % 60 == 0 or step < 5:
        q = uw.robot.data.joint_pos[0, cyl_ids].detach().cpu().numpy()
        qd = uw.robot.data.joint_vel[0, cyl_ids].detach().cpu().numpy()
        tau_app = uw.robot.root_physx_view.get_dof_actuation_forces()
        if isinstance(tau_app, torch.Tensor):
            tau_app_arr = tau_app[0].detach().cpu().numpy()
        else:
            tau_app_arr = np.zeros(21)
        tau_computed = tau[0].detach().cpu().numpy()
        err = q_target[0].detach().cpu().numpy() - q

        print(f"step {step:4d}: pos={[f'{x:.4f}' for x in q]}", flush=True)
        for i, jid in enumerate(cyl_ids.tolist()):
            print(f"  {jnames[jid]:30s} err={err[i]:+.4f}rad tau_cmd={tau_computed[i]:+.0f} tau_app={tau_app_arr[jid]:+.0f}", flush=True)

print("Done", flush=True)
env.close()
simulation_app.close()
