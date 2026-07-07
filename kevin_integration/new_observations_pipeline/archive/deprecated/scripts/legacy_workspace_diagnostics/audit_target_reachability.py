#!/usr/bin/env python3
"""Audit target reachability for the PPO1 horizon-IDFD pipeline.

Samples N targets using the current command sampler and checks reachability.
"""
from __future__ import annotations
import argparse, sys, os
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Target reachability audit for PPO1 horizon-IDFD.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_samples", type=int, default=200)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]
# device handled by AppLauncher
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

num_samples = int(args_cli.num_samples)
wx = getattr(cfg, "horizon_workspace_x_range", (-0.5, 7.0))
wy = getattr(cfg, "horizon_workspace_y_range", (-3.6, 3.0))
wz = getattr(cfg, "horizon_workspace_z_range", (-0.4, 3.0))
cx = getattr(cfg, "horizon_cab_x_range", (-2.5, 1.0))
cy = getattr(cfg, "horizon_cab_y_range", (-1.35, 1.35))
cz = getattr(cfg, "horizon_cab_z_range", (-0.1, 2.4))
rx = cfg.command_pos_x_range
ry = cfg.command_pos_y_range

results = []
for i in range(num_samples):
    env.reset()
    n = env.num_envs
    new_x = torch.rand(n, device=env.device) * (rx[1] - rx[0]) + rx[0]
    new_y = torch.rand(n, device=env.device) * (ry[1] - ry[0]) + ry[0]
    uw._command_pos_w[:, 0] = new_x
    uw._command_pos_w[:, 1] = new_y
    uw._command_pos_w[:, 2] = cfg.command_pos_z
    uw._command_quat_w[:, 0] = 1.0
    uw._command_quat_w[:, 1:] = 0.0

    target = uw._command_pos_w[0, :3].detach().cpu().numpy()
    ee_pos_base = env._tcp_pos_base_frame()[0].detach().cpu().numpy()
    ee_error = float(np.linalg.norm(ee_pos_base - target))

    in_workspace = (
        (target[0] >= wx[0]) and (target[0] <= wx[1])
        and (target[1] >= wy[0]) and (target[1] <= wy[1])
        and (target[2] >= wz[0]) and (target[2] <= wz[1])
    )
    in_cab = (
        (target[0] >= cx[0]) and (target[0] <= cx[1])
        and (target[1] >= cy[0]) and (target[1] <= cy[1])
        and (target[2] >= cz[0]) and (target[2] <= cz[1])
    )

    max_reach = 3.5
    if not in_workspace:
        reachable = "unreachable: outside workspace"
    elif in_cab:
        reachable = "unreachable: inside cab"
    elif ee_error > max_reach:
        reachable = "unreachable: too far (>3.5m)"
    elif ee_error > 1.0:
        reachable = "unknown: moderate distance"
    else:
        reachable = "reachable"

    results.append({
        "sample": i,
        "target_base": target.tolist(),
        "ee_base": ee_pos_base.tolist(),
        "ee_error": ee_error,
        "in_workspace": in_workspace,
        "in_cab": in_cab,
        "reachable": reachable,
    })

n_unreachable = sum(1 for r in results if "unreachable" in r["reachable"])
n_reachable = sum(1 for r in results if r["reachable"] == "reachable")
n_unknown = sum(1 for r in results if "unknown" in r["reachable"])

print("=" * 70)
print("TARGET REACHABILITY AUDIT")
print("=" * 70)
print(f"\nSamples: {num_samples}")
print(f"Command range X: {rx}, Y: {ry}, Z: {cfg.command_pos_z}")
print(f"Workspace X: {wx}, Y: {wy}, Z: {wz}")
print(f"Cab forbidden X: {cx}, Y: {cy}, Z: {cz}")
print(f"\nReachability summary:")
print(f"  Reachable: {n_reachable} ({n_reachable/num_samples*100:.1f}%)")
print(f"  Unknown: {n_unknown} ({n_unknown/num_samples*100:.1f}%)")
print(f"  Unreachable: {n_unreachable} ({n_unreachable/num_samples*100:.1f}%)")

unreach_causes = {}
for r in results:
    if "unreachable" in r["reachable"]:
        cause = r["reachable"].split(": ", 1)[1]
        unreach_causes[cause] = unreach_causes.get(cause, 0) + 1
for cause, cnt in sorted(unreach_causes.items(), key=lambda x: -x[1]):
    print(f"    - {cause}: {cnt} ({cnt/num_samples*100:.1f}%)")

errors = [r["ee_error"] for r in results]
errors.sort()
print(f"\nInitial EE error stats:")
print(f"  Min: {errors[0]:.3f} m")
print(f"  P10: {errors[len(errors)//10]:.3f} m")
print(f"  P50: {errors[len(errors)//2]:.3f} m")
print(f"  P90: {errors[9*len(errors)//10]:.3f} m")
print(f"  Max: {errors[-1]:.3f} m")
print(f"  Mean: {sum(errors)/len(errors):.3f} m")

print(f"\nFirst 5 samples:")
for r in results[:5]:
    print(f"  Sample {r['sample']}: target={[f'{x:.2f}' for x in r['target_base']]}, "
          f"ee={[f'{x:.2f}' for x in r['ee_base']]}, "
          f"error={r['ee_error']:.3f}m, ws={r['in_workspace']}, cab={r['in_cab']}, "
          f"class={r['reachable']}")

unreachable_pct = n_unreachable / num_samples * 100
if unreachable_pct > 20:
    print(f"\nVERDICT: {unreachable_pct:.1f}% targets unreachable (>20%). Sampling MUST be replaced.")
else:
    print(f"\nVERDICT: {unreachable_pct:.1f}% targets unreachable (<20%).")

print("=" * 70)
env.close()
simulation_app.close()
