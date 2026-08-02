#!/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
"""Is the cutter head being shoved off its commanded joint reference by contact?

The stage-5 collapses are head X-axis excursions, and the measured authority
Jacobian (reports/head_axis_authority.json) shows no cylinder can steer X: the
best column is c1 at 0.0060 against 0.159 on Y, i.e. 3.8 %. So X cannot be
recovered and the only remaining question is whether it can be *prevented*.

Prevention is a torque question. ``Messerkopf_joint`` and
``Messerkopf_Schwenk_joint`` are driven at effort_limit_sim 20000 while the
measured cutter contact force runs 66-83 kN p95. If contact is dragging those
joints off the reference the control pipeline writes, then raising the effort
ceiling (or locking the 0.76 coupling) removes the excursion at its source. If
instead both joints sit on their references, the tilt originates upstream and
the head-joint lever is dead -- worth knowing before anyone changes gains.

This probe therefore reports, per arm joint, the error between measured joint
position and the commanded ``_joint_ref``, split by whether the head is healthy
or already tilted, plus the applied torque against each joint's effort limit.
The split is the point: an error that appears only once the head has gone is a
consequence, while an error that leads the excursion is the cause.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isaaclab.app import AppLauncher  # noqa: E402

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--num_envs", type=int, default=64)
p.add_argument("--seed", type=int, default=17)
p.add_argument("--stage", type=int, default=5)
p.add_argument("--steps", type=int, default=600)
p.add_argument("--checkpoint", default="", help="Omit for zero action.")
p.add_argument(
    "--tilt_action",
    type=float,
    default=0.0,
    help=(
        "Constant c4 command used to drive the head off world-down on purpose. "
        "Zero measures the undisturbed plant; a non-zero value forces the "
        "tilted regime the collapses reach, without waiting for exploration."
    ),
)
p.add_argument("--eval_terrain", action="store_true")
p.add_argument("--path_family", default="mixed")
p.add_argument("--report_out", default="")
AppLauncher.add_app_launcher_args(p)
a, unknown = p.parse_known_args()
sys.argv = [sys.argv[0], *unknown]

if a.num_envs <= 0 or a.steps <= 0:
    p.error("--num_envs and --steps must be positive")
if not -1.0 <= a.tilt_action <= 1.0:
    p.error("--tilt_action must lie in [-1, 1]")

launcher = AppLauncher(a)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import PPO3_SOLE_PhysX  # noqa: F401,E402
from PPO3_SOLE_PhysX.terrain_env_cfg import (  # noqa: E402
    TerrainSolePolicyPPORunnerCfg,
    apply_terrain_curriculum_stage,
    configure_terrain_gpu_physx_memory,
)
from PPO3_SOLE_PhysX.terrain_observations import (  # noqa: E402
    observation_space_for_cfg,
)
from PPO3_SOLE_PhysX.terrain_convergence import GATE_SPECS  # noqa: E402

TILT_THRESHOLD = 0.10
"""|head_axis_x| above this counts as tilted; healthy runs measure 0.0185."""


def build_env():
    env_cfg = parse_env_cfg(
        PPO3_SOLE_PhysX.TERRAIN_TASK_ID, device=a.device, num_envs=a.num_envs
    )
    apply_terrain_curriculum_stage(env_cfg, int(a.stage))
    env_cfg.sole_ik_hint = False
    env_cfg.observation_space = observation_space_for_cfg(env_cfg)
    env_cfg.use_eval_terrain = bool(a.eval_terrain)
    env_cfg.terrain_base_seed = int(a.seed)
    env_cfg.enable_contact_tracking_demo = False
    env_cfg.tracking_path_family = str(a.path_family)
    env_cfg.enable_visual_grid_floor = False
    env_cfg.enable_road_scenery = False
    # A synchronized reset mid-excursion would erase the state being measured.
    env_cfg.defer_safety_terminations = True
    env_cfg.head_down_termination_alignment = float(
        GATE_SPECS["head_alignment_hard_min"].threshold
    )
    env_cfg.randomize_start_phase = False
    env_cfg.randomize_trajectory_shape = False
    env_cfg.trajectory_shape = str(env_cfg.terrain_deployment_trajectory_shape)
    configure_terrain_gpu_physx_memory(env_cfg, int(a.num_envs))
    env = gym.make(PPO3_SOLE_PhysX.TERRAIN_TASK_ID, cfg=env_cfg, render_mode=None)
    return env, env.unwrapped


def main() -> int:
    env, raw = build_env()
    policy = None
    if a.checkpoint:
        runner_cfg = TerrainSolePolicyPPORunnerCfg()
        runner_cfg.seed = int(a.seed)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)
        runner = OnPolicyRunner(
            wrapped, runner_cfg.to_dict(), log_dir=None, device=a.device
        )
        runner.load(
            str(Path(a.checkpoint).resolve()),
            load_cfg={
                "actor": True, "critic": True, "optimizer": False,
                "iteration": True, "rnd": False,
            },
        )
        policy = runner.get_inference_policy(device=a.device)

    obs, _ = raw.reset(seed=int(a.seed))
    arm_ids = raw._arm_joint_ids
    arm_names = list(raw.cfg.arm_joint_names)
    effort_limits = raw.robot.data.joint_effort_limits[:, arm_ids][0].detach().cpu()

    base_action = torch.zeros(
        (raw.num_envs, int(raw.cfg.action_space)),
        device=raw.device, dtype=torch.float32,
    )
    if a.tilt_action != 0.0:
        base_action[:, 3] = float(a.tilt_action)

    errors, torques, tilts = [], [], []
    # Raw positions too: if the head joint is physically coupled in the asset,
    # its angle is set by that constraint and the drive can only fight it, which
    # shows up as an error independent of torque. Fitting measured Messerkopf
    # against measured Schwenk recovers the asset's true ratio and tests whether
    # the 0.76 the pipeline writes matches the mechanism.
    positions = []
    for _ in range(int(a.steps)):
        action = base_action if policy is None else policy(obs)
        obs, _, _, _, _ = raw.step(action)
        measured = raw.robot.data.joint_pos[:, arm_ids].detach()
        commanded = raw._joint_ref.detach()
        errors.append((measured - commanded).abs().clone())
        positions.append(measured.clone())
        torques.append(
            raw.robot.data.applied_torque[:, arm_ids].detach().abs().clone()
        )
        tilts.append(raw._head_down_axis_world()[:, 0].detach().abs().clone())

    err = torch.stack(errors).double().cpu()
    tau = torch.stack(torques).double().cpu()
    tilt = torch.stack(tilts).double().cpu()
    healthy = tilt <= TILT_THRESHOLD
    tilted = ~healthy

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "mode": "policy" if policy is not None else f"zero_action+c4={a.tilt_action}",
        "checkpoint": str(a.checkpoint) or None,
        "stage": int(a.stage),
        "num_envs": int(raw.num_envs),
        "steps": int(a.steps),
        "tilt_threshold_abs_head_axis_x": TILT_THRESHOLD,
        "healthy_sample_fraction": float(healthy.double().mean()),
        "tilted_sample_fraction": float(tilted.double().mean()),
        "max_abs_head_axis_x": float(tilt.max()),
        "joints": {},
    }
    for i, name in enumerate(arm_names):
        limit = float(effort_limits[i])
        col_err, col_tau = err[..., i], tau[..., i]
        entry = {
            "effort_limit": limit,
            "mean_abs_tracking_error_rad": float(col_err.mean()),
            "p99_abs_tracking_error_rad": float(torch.quantile(col_err.flatten(), 0.99)),
            "max_abs_tracking_error_rad": float(col_err.max()),
            "mean_abs_applied_torque": float(col_tau.mean()),
            "max_abs_applied_torque": float(col_tau.max()),
            "torque_at_limit_fraction": float(
                (col_tau >= 0.999 * limit).double().mean()
            ) if limit > 0 else None,
        }
        for label, mask in (("healthy", healthy), ("tilted", tilted)):
            n = int(mask.sum())
            entry[label] = (
                {"samples": n, "note": "no samples in this regime"}
                if n < 64
                else {
                    "samples": n,
                    "mean_abs_tracking_error_rad": float(col_err[mask].mean()),
                    "p99_abs_tracking_error_rad": float(
                        torch.quantile(col_err[mask], 0.99)
                    ),
                    "mean_abs_applied_torque": float(col_tau[mask].mean()),
                    "torque_at_limit_fraction": float(
                        (col_tau[mask] >= 0.999 * limit).double().mean()
                    ) if limit > 0 else None,
                }
            )
        report["joints"][name] = entry

    # Least-squares fit of measured Messerkopf against measured Schwenk. The
    # pipeline writes ref_messerkopf = joint6_coupling_a * ref_schwenk + b, so a
    # fitted slope far from joint6_coupling_a means the command contradicts the
    # mechanism and the drive spends its whole effort budget losing that fight.
    pos = torch.stack(positions).double().cpu()
    src = arm_names.index("Messerkopf_Schwenk_joint")
    dst = arm_names.index("Messerkopf_joint")
    x = pos[..., src].flatten()
    y = pos[..., dst].flatten()
    design = torch.stack([x, torch.ones_like(x)], dim=1)
    slope, intercept = torch.linalg.lstsq(design, y.reshape(-1, 1)).solution.flatten()
    resid = y - (slope * x + intercept)
    total = ((y - y.mean()) ** 2).sum()
    report["measured_coupling"] = {
        "fitted_slope": float(slope),
        "fitted_intercept": float(intercept),
        "r2": float(1.0 - (resid**2).sum() / total) if float(total) > 0 else None,
        "residual_std_rad": float(resid.std()),
        "configured_joint6_coupling_a": float(
            getattr(raw.cfg, "joint6_coupling_a", 0.76)
        ),
        "configured_joint6_coupling_b": float(
            getattr(raw.cfg, "joint6_coupling_b", 0.0)
        ),
        "schwenk_range_rad": [float(x.min()), float(x.max())],
        "messerkopf_range_rad": [float(y.min()), float(y.max())],
    }

    out = (
        Path(a.report_out).expanduser().resolve()
        if a.report_out
        else ROOT / "PPO3_SOLE_PhysX" / "reports" / "head_joint_tracking.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print(f"[PPO3-HEAD-JOINT] wrote {out}", flush=True)

    env.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
