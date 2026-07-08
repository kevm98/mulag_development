#!/usr/bin/env python3
"""Headless reset collision proxy audit for the horizon IDFD PPO env."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Audit reset pose against cab/workspace and finite cylinder limits.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--horizon_steps", type=int, default=10)
parser.add_argument("--horizon_action_scale_m", type=float, default=0.01)
parser.add_argument("--fd_horizon_gain_per_cylinder", type=str, default="2.0,3.0,3.0,3.0")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]
if getattr(args_cli, "device", None) in (None, ""):
    args_cli.device = "cuda:0"
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from isaaclab.utils.math import quat_apply_inverse

from kevin_integration.new_observations_pipeline.ppo1.horizon_idfd_env_wrapper import HorizonIDFDEnv
from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env_cfg import MulagUpperLevelEnvCfg


def tcp_pos_base(uw) -> torch.Tensor:
    ee_pos_w = uw.robot.data.body_link_state_w[:, uw._ee_body_id, :3]
    base_pos_w = uw.robot.data.root_pos_w
    base_q = uw.robot.data.root_quat_w
    base_q = base_q / base_q.norm(dim=-1, keepdim=True).clamp(min=1.0e-8)
    return quat_apply_inverse(base_q, ee_pos_w - base_pos_w)


def main() -> int:
    fd_gains = [float(x.strip()) for x in args_cli.fd_horizon_gain_per_cylinder.split(",")]
    env_cfg = MulagUpperLevelEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.device = args_cli.device
    env_cfg.sim.device = args_cli.device

    env = HorizonIDFDEnv(
        base_cfg=env_cfg,
        horizon_steps=args_cli.horizon_steps,
        action_scale_m=args_cli.horizon_action_scale_m,
        fd_gains=fd_gains,
        device=args_cli.device,
    )
    try:
        env.reset()
        zero = torch.zeros((env.num_envs, 4), device=env.device)
        env.step(zero)

        tcp = tcp_pos_base(env._uw).detach()
        cfg = env.cfg
        cx = getattr(cfg, "horizon_cab_x_range", (-2.5, 1.0))
        cy = getattr(cfg, "horizon_cab_y_range", (-1.35, 1.35))
        cz = getattr(cfg, "horizon_cab_z_range", (-0.1, 2.4))
        inside_cab = (
            (tcp[:, 0] >= float(cx[0]))
            & (tcp[:, 0] <= float(cx[1]))
            & (tcp[:, 1] >= float(cy[0]))
            & (tcp[:, 1] <= float(cy[1]))
            & (tcp[:, 2] >= float(cz[0]))
            & (tcp[:, 2] <= float(cz[1]))
        )

        lo, hi = env._joint_limits()
        hold = env._cylinder_hold_target.detach()
        finite_hold = torch.isfinite(hold).all(dim=-1)
        in_limits = ((hold >= lo) & (hold <= hi)).all(dim=-1)
        root_z = env._uw.robot.data.root_pos_w[:, 2].detach()
        root_z_ok = root_z > -0.05
        passed = bool(((~inside_cab) & finite_hold & in_limits & root_z_ok).all().item())

        report_dir = ROOT / "outputs/cylinder_diagnostics/paper_matched_pipeline"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / "collision_audit_report.md"
        report_path.write_text(
            "\n".join(
                [
                    "# Collision Audit Report",
                    "",
                    "- audit_type: headless reset geometric proxy",
                    f"- tcp_base_min: {[round(float(x), 6) for x in tcp.min(dim=0).values.cpu()]}",
                    f"- tcp_base_max: {[round(float(x), 6) for x in tcp.max(dim=0).values.cpu()]}",
                    f"- cab_proxy_clear: {not bool(inside_cab.any().item())}",
                    f"- cylinder_targets_finite: {bool(finite_hold.all().item())}",
                    f"- cylinder_targets_in_soft_limits: {bool(in_limits.all().item())}",
                    f"- root_z_min_m: {float(root_z.min().cpu()):.6f}",
                    f"- root_z_ok: {bool(root_z_ok.all().item())}",
                    f"- max_guard_fraction: {float(env._last_guard_fraction.max().cpu()):.6f}",
                    f"- pass: {passed}",
                    "",
                    "No PPO1 arm/truck pair contact sensor is currently configured, so this audit does",
                    "not claim pairwise contact-force separation. It verifies the reset TCP is outside",
                    "the configured cab exclusion box, the base is not below the ground proxy, and",
                    "persistent cylinder targets are finite and inside soft limits.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        print(f"collision_audit_report={report_path}")
        print(f"cab_proxy_clear={not bool(inside_cab.any().item())} pass={passed}")
        return 0 if passed else 2
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
