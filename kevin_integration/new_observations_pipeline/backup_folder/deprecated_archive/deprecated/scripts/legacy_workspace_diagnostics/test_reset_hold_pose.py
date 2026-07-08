#!/usr/bin/env python3
"""Headless reset/zero-action hold audit for the horizon IDFD PPO env."""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Verify reset pose and persistent cylinder hold under zero action.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--duration_s", type=float, default=1.0)
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


def ee_pos_world(uw) -> torch.Tensor:
    return uw.robot.data.body_link_state_w[:, uw._ee_body_id, :3]


def inside_cab_proxy(env: HorizonIDFDEnv) -> torch.Tensor:
    tcp = tcp_pos_base(env._uw)
    cfg = env.cfg
    cx = getattr(cfg, "horizon_cab_x_range", (-2.5, 1.0))
    cy = getattr(cfg, "horizon_cab_y_range", (-1.35, 1.35))
    cz = getattr(cfg, "horizon_cab_z_range", (-0.1, 2.4))
    return (
        (tcp[:, 0] >= float(cx[0]))
        & (tcp[:, 0] <= float(cx[1]))
        & (tcp[:, 1] >= float(cy[0]))
        & (tcp[:, 1] <= float(cy[1]))
        & (tcp[:, 2] >= float(cz[0]))
        & (tcp[:, 2] <= float(cz[1]))
    )


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
        start_ee_w = ee_pos_world(env._uw).detach().clone()
        start_tcp_b = tcp_pos_base(env._uw).detach().clone()
        start_hold = env._cylinder_hold_target.detach().clone()
        steps = max(1, int(math.ceil(args_cli.duration_s / max(env._H * env._dt, 1.0e-6))))
        zero = torch.zeros((env.num_envs, 4), device=env.device)
        terminated = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        truncated = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        for _ in range(steps):
            _, _, term_i, trunc_i, _ = env.step(zero)
            terminated |= term_i
            truncated |= trunc_i

        end_ee_w = ee_pos_world(env._uw).detach()
        end_tcp_b = tcp_pos_base(env._uw).detach()
        drift = torch.norm(end_ee_w - start_ee_w, dim=-1)
        tcp_base_drift = torch.norm(end_tcp_b - start_tcp_b, dim=-1)
        lo, hi = env._joint_limits()
        hold = env._cylinder_hold_target.detach()
        finite_hold = torch.isfinite(hold).all(dim=-1)
        in_limits = ((hold >= lo) & (hold <= hi)).all(dim=-1)
        hold_delta = torch.norm(hold - start_hold, dim=-1)
        cab_proxy = inside_cab_proxy(env)
        pass_mask = (drift < 0.05) & finite_hold & in_limits & (~cab_proxy)
        passed = bool(pass_mask.all().item())

        report_dir = ROOT / "outputs/cylinder_diagnostics/paper_matched_pipeline"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / "reset_pose_hold_report.md"
        report_path.write_text(
            "\n".join(
                [
                    "# Reset Pose Hold Report",
                    "",
                    f"- duration_s: {args_cli.duration_s}",
                    f"- horizon_steps: {env._H}",
                    f"- horizon_steps_executed: {steps}",
                    f"- max_ee_drift_m: {float(drift.max().cpu()):.6f}",
                    f"- max_tcp_base_frame_drift_m: {float(tcp_base_drift.max().cpu()):.6f}",
                    f"- start_tcp_base_min: {[round(float(x), 6) for x in start_tcp_b.min(dim=0).values.cpu()]}",
                    f"- start_tcp_base_max: {[round(float(x), 6) for x in start_tcp_b.max(dim=0).values.cpu()]}",
                    f"- end_tcp_base_min: {[round(float(x), 6) for x in end_tcp_b.min(dim=0).values.cpu()]}",
                    f"- end_tcp_base_max: {[round(float(x), 6) for x in end_tcp_b.max(dim=0).values.cpu()]}",
                    f"- max_hold_target_delta_rad_or_m: {float(hold_delta.max().cpu()):.6f}",
                    f"- max_guard_fraction: {float(env._last_guard_fraction.max().cpu()):.6f}",
                    f"- cylinder_targets_finite: {bool(finite_hold.all().item())}",
                    f"- cylinder_targets_in_soft_limits: {bool(in_limits.all().item())}",
                    f"- cab_proxy_clear: {not bool(cab_proxy.any().item())}",
                    f"- base_terminated_any: {bool(terminated.any().item())}",
                    f"- base_truncated_any: {bool(truncated.any().item())}",
                    f"- pass: {passed}",
                    "",
                    "Collision note: this headless script checks the reset TCP against the configured cab box",
                    "and verifies finite in-limit cylinder targets. No dedicated arm/truck pair contact sensor",
                    "is configured in PPO1, so pairwise collision is audited by the companion geometric script.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        print(f"reset_hold_report={report_path}")
        print(f"max_ee_drift_m={float(drift.max().cpu()):.6f} pass={passed}")
        return 0 if passed else 2
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
