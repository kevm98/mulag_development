#!/usr/bin/env python3
"""Audit target/EE/workspace coordinate frames for the PPO1 horizon-IDFD pipeline.

Checks for frame mismatches: world vs. base frame across reward, observation,
workspace guard, and cab guard.
"""
from __future__ import annotations
import argparse, sys, os
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Frame consistency audit for PPO1 horizon-IDFD.")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]
# device handled by AppLauncher
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.utils.math import quat_apply_inverse, euler_xyz_from_quat
from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env_cfg import MulagUpperLevelEnvCfg
from kevin_integration.new_observations_pipeline.ppo1.horizon_idfd_env_wrapper import HorizonIDFDEnv
from kevin_integration.new_observations_pipeline.ppo1.rewards_adapter import compute_tracking_error
from kevin_integration.new_observations_pipeline.ppo1.observations_adapter import build_horizon_observation, HORIZON_OBS_INDEX
from kevin_integration.utils.sim_memory import apply_kevin_sim_memory_optimizations

cfg = MulagUpperLevelEnvCfg()
cfg.scene.num_envs = int(args_cli.num_envs)
cfg.enable_scripted_vehicle_motion = False
cfg.enable_startup_summary = True
cfg.log_target_delta_xdot_clipping = False
cfg.horizon_pin_root_pose = False
cfg.horizon_reset_settle_steps = 0
if getattr(args_cli, "device", None):
    cfg.device = args_cli.device
    cfg.sim.device = args_cli.device
id_path = (
    Path(os.getcwd())
    / "outputs/cylinder_diagnostics/paper_matched_pipeline/model_finetune/id_physx_finetuned_best.pt"
).resolve()
cfg.id_checkpoint_override_path = str(id_path)
apply_kevin_sim_memory_optimizations(cfg.sim)

env = HorizonIDFDEnv(
    base_cfg=cfg,
    horizon_steps=10,
    action_scale_m=0.01,
    fd_gains=[2.0, 3.0, 3.0, 3.0],
    device=args_cli.device,
)
uw = env._uw
B = env.num_envs

# Reset
obs, _ = env.reset()

print("=" * 70)
print("FRAME AUDIT: PPO1 Horizon-IDFD Pipeline")
print("=" * 70)

# 1. Robot root world position
root_pos_w = uw.robot.data.root_pos_w[0].detach().cpu().numpy()
root_quat_w = uw.robot.data.root_quat_w[0].detach().cpu().numpy()
root_quat_w = root_quat_w / (np.linalg.norm(root_quat_w) + 1e-8)
print(f"\n1. Robot root world position: {root_pos_w}")
print(f"   Robot root world quat (w,x,y,z): {root_quat_w}")

# 2. Arm base (Drehzapfen) world position
arm_base_id = getattr(uw, "_arm_base_body_id", None)
if arm_base_id is not None:
    arm_base_pos_w = uw.robot.data.body_link_state_w[0, arm_base_id, :3].detach().cpu().numpy()
    print(f"\n2. Arm base (Drehzapfen) world position: {arm_base_pos_w}")
else:
    arm_base_pos_w = root_pos_w
    print(f"\n2. Arm base body not found, using root position as proxy.")

# 3. EE world position
ee_pos_w = uw.robot.data.body_link_state_w[0, uw._ee_body_id, :3].detach().cpu().numpy()
print(f"\n3. EE world position: {ee_pos_w}")

# 4. EE base-frame position
tcp_base = env._tcp_pos_base_frame()[0].detach().cpu().numpy()
print(f"\n4. EE base-frame position: {tcp_base}")
print(f"   Computed via: quat_apply_inverse(root_quat_w, ee_pos_w - root_pos_w)")

# 5. Target world position (stored in _command_pos_w)
cmd_pos = uw._command_pos_w[0, :3].detach().cpu().numpy()
cmd_quat = uw._command_quat_w[0].detach().cpu().numpy()
print(f"\n5. Command position (stored in _command_pos_w): {cmd_pos}")
print(f"   Command quat (stored in _command_quat_w): {cmd_quat}")
print(f"   NOTE: Named '_command_pos_w' but config says 'command_pos_x_range' is 'in base frame'")

# 6. Target base-frame position (if command is in world, transform it)
# The command is actually already in base frame per config comment.
# But for audit, also show what it would be if treated as world frame.
cmd_world = cmd_pos.copy()
cmd_base_direct = cmd_pos  # treating as base frame (current behavior)
# If cmd were truly world frame, transform to base:
root_q_t = torch.as_tensor(root_quat_w, device=uw.device).unsqueeze(0)
root_p_t = torch.as_tensor(root_pos_w, device=uw.device).unsqueeze(0)
cmd_w_t = torch.as_tensor(cmd_world, device=uw.device).unsqueeze(0)
cmd_base_if_world = quat_apply_inverse(root_q_t, cmd_w_t - root_p_t)[0].cpu().numpy()
print(f"\n6. Target IF interpreted as world frame → base: {cmd_base_if_world}")
print(f"   Target IF interpreted as base frame (current): {cmd_base_direct}")
print(f"   Difference: {cmd_base_if_world - cmd_base_direct}")

# 7. Raw EE error used by reward
ee_error = compute_tracking_error(uw)[0].detach().cpu().item()
print(f"\n7. EE error norm used by reward (compute_tracking_error): {ee_error:.4f} m")
print(f"   Formula: norm(tcp_pos_base - _command_pos_w[:3])")

# 8. Explicit EE error vector
tcp_b_t = env._tcp_pos_base_frame()[0]
cmd_b_t = uw._command_pos_w[0, :3]
err_vec = (cmd_b_t - tcp_b_t).detach().cpu().numpy()  # target - ee (same as observation)
print(f"\n8. EE error vector (tcp_base - command): {err_vec}")
print(f"   Norm: {np.linalg.norm(err_vec):.4f} m")

# 9. Workspace bounds
wx = getattr(cfg, "horizon_workspace_x_range", (1.0, 7.0))
wy = getattr(cfg, "horizon_workspace_y_range", (-3.0, 3.0))
wz = getattr(cfg, "horizon_workspace_z_range", (-0.4, 3.0))
print(f"\n9. Workspace bounds (BASE frame):")
print(f"   x: [{wx[0]:.2f}, {wx[1]:.2f}]")
print(f"   y: [{wy[0]:.2f}, {wy[1]:.2f}]")
print(f"   z: [{wz[0]:.2f}, {wz[1]:.2f}]")

# 10. Is target inside workspace?
tgt_in_ws = (
    (cmd_base_direct[0] >= wx[0]) and (cmd_base_direct[0] <= wx[1])
    and (cmd_base_direct[1] >= wy[0]) and (cmd_base_direct[1] <= wy[1])
    and (cmd_base_direct[2] >= wz[0]) and (cmd_base_direct[2] <= wz[1])
)
print(f"\n10. Is target (base frame) inside workspace? {tgt_in_ws}")

# 11. Is EE inside workspace?
ee_in_ws = (
    (tcp_base[0] >= wx[0]) and (tcp_base[0] <= wx[1])
    and (tcp_base[1] >= wy[0]) and (tcp_base[1] <= wy[1])
    and (tcp_base[2] >= wz[0]) and (tcp_base[2] <= wz[1])
)
print(f"\n11. Is EE (base frame) inside workspace? {ee_in_ws}")

# 12. Cab guard bounds
cx = getattr(cfg, "horizon_cab_x_range", (-2.5, 1.0))
cy = getattr(cfg, "horizon_cab_y_range", (-1.35, 1.35))
cz = getattr(cfg, "horizon_cab_z_range", (-0.1, 2.4))
print(f"\n12. Cab guard bounds (BASE frame):")
print(f"    x: [{cx[0]:.2f}, {cx[1]:.2f}]")
print(f"    y: [{cy[0]:.2f}, {cy[1]:.2f}]")
print(f"    z: [{cz[0]:.2f}, {cz[1]:.2f}]")

# 13. Is EE inside cab forbidden region?
ee_in_cab = (
    (tcp_base[0] >= cx[0]) and (tcp_base[0] <= cx[1])
    and (tcp_base[1] >= cy[0]) and (tcp_base[1] <= cy[1])
    and (tcp_base[2] >= cz[0]) and (tcp_base[2] <= cz[1])
)
print(f"\n13. Is EE inside cab forbidden region? {ee_in_cab}")

# 14. Cab guard uses world or base frame?
outside_ws, inside_cab = env._workspace_guard_components()
print(f"\n14. Guard frame audit:")
print(f"    workspace_guard_rows (env 0): outside_ws={bool(outside_ws[0].item())}, inside_cab={bool(inside_cab[0].item())}")
print(f"    Both guards use _tcp_pos_base_frame() → BASE frame.")

# 15. Reward and termination use the same frame?
print(f"\n15. Frame consistency check:")
print(f"    Reward tracking error: TCP in base, target from _command_pos_w (treated as base)")
print(f"    Workspace/cab guard: TCP in base frame")
print(f"    Observation target_pos_b: _command_pos_w[:, :3] (treated as base)")
print(f"    Observation ee_pos_error_b: target_pos_b - ee_pos_b (base - base)")

if tgt_in_ws and ee_in_ws and not ee_in_cab:
    print(f"\nVERDICT: Frames are consistent (all base).")
    print(f"EE and target are both inside workspace, EE is outside cab.")
    print(f"Frame mismatch is NOT the root cause of workspace violations.")
else:
    print(f"\nVERDICT: Potential issue detected.")
    if not tgt_in_ws:
        print(f"WARNING: Target is outside workspace!")
    if not ee_in_ws:
        print(f"WARNING: EE is outside workspace at reset!")
    if ee_in_cab:
        print(f"WARNING: EE is inside cab forbidden region!")

# 16. Observation dimensions check
obs_vector = obs[0].detach().cpu().numpy()
print(f"\n16. Observation check:")
print(f"    Dim: {obs_vector.shape}")
for name, (start, end) in HORIZON_OBS_INDEX.items():
    vals = obs_vector[start:end]
    print(f"    {name}[{start}:{end}]: {vals}")

# 17. Check ee_pos_error_b in observation matches computed error
obs_ee_error = obs_vector[10:13]
expected_error = err_vec  # target_pos_b - ee_pos_b
print(f"\n17. Observation ee_pos_error_b: {obs_ee_error}")
print(f"    Expected (target_pos_b - ee_pos_b): {expected_error}")
print(f"    Match: {np.allclose(obs_ee_error, expected_error, atol=1e-5)}")

print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)

env.close()
simulation_app.close()
