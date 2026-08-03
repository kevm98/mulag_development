#!/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
"""Is the head controller BLIND to the axis it fails on?

The hypothesis, from the stage-6 failure and the repository's own notes:

* the head levelling correction and the c4 tilt shield both measure tilt as
  ``atan2(head_axis_y, -head_axis_z)`` -- a projection into the WORLD Y-Z
  plane;
* the joint can actually rotate about its own axis, which travels with the arm
  as it slews;
* so when the arm slews, the correctable tilt rotates out of the world Y-Z
  plane and into world X, where the measure under-reads it and the correction
  fades -- exactly where it is needed;
* stage 6 widened the tracking path 2.2x, which demands much more slew, and the
  measured failure is ``max_abs_head_axis_x = 0.726`` with alignment 0.5866.

`terrain_head_tilt_arm_frame` already exists and computes the exact one-DOF
solve about the joint's own axis. Its note says the frame reasoning is correct
but that enabling it measured WORSE once -- on stage 5, at 1.0x path amplitude,
where the two frames agree ("At evaluation poses (near-level, near-zero slew)
the two agree to the digit"). That negative result therefore says nothing about
the high-slew regime stage 6 creates.

This script decides it by measurement instead of argument. It logs both tilt
measures, the arm slew, and the head axis every step, then reports how they
differ SPECIFICALLY on the samples where alignment collapses.

It changes nothing: no config is flipped, no correction is applied.
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

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--checkpoint", required=True)
p.add_argument("--actuator", default="rom", choices=("rom", "idfd"))
p.add_argument("--num_envs", type=int, default=32)
p.add_argument("--steps", type=int, default=960)
p.add_argument("--seed", type=int, default=17)
p.add_argument("--stage", type=int, default=6)
p.add_argument("--eval_terrain", action="store_true", default=True)
p.add_argument("--eval_terrain_seed", type=int, default=0)
p.add_argument("--path_family", default="mixed")
p.add_argument("--tracking_amplitude", type=float, default=None)
p.add_argument("--collapse_alignment", type=float, default=0.90)
p.add_argument("--report", default="")
AppLauncher.add_app_launcher_args(p)
a, unknown = p.parse_known_args()
sys.argv = [sys.argv[0], *unknown]

launcher = AppLauncher(a)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import PPO3_SOLE_PhysX  # noqa: F401,E402
from PPO3_SOLE_PhysX.actuators import resolve_actuator  # noqa: E402
from PPO3_SOLE_PhysX.distill import assert_teacher_disabled  # noqa: E402
from PPO3_SOLE_PhysX.terrain_env_cfg import (  # noqa: E402
    apply_terrain_curriculum_stage,
    configure_terrain_gpu_physx_memory,
)
from PPO3_SOLE_PhysX.terrain_observations import (  # noqa: E402
    observation_space_for_cfg,
    override_cfg,
)

TASK_ID, RUNNER_CFG_CLS, _ = resolve_actuator(a.actuator)


def build_env():
    cfg = parse_env_cfg(TASK_ID, device=a.device, num_envs=a.num_envs)
    apply_terrain_curriculum_stage(cfg, int(a.stage))
    cfg.action_space = 5 if cfg.sole_head_joint_action else 4
    if cfg.sole_head_joint_action:
        cfg.terrain_messerkopf_level_gain = 0.0
        cfg.terrain_messerkopf_level_quadratic_gain = 0.0
    cfg.sole_ik_hint = False
    cfg.terrain_contact_observation = True
    override_cfg(cfg, "terrain_model_error_history", 3)
    override_cfg(cfg, "action_scale", 0.0035)
    override_cfg(cfg, "cylinder_target_max_lead_m", 0.018)
    if a.tracking_amplitude is not None:
        override_cfg(
            cfg, "terrain_tracking_amplitude_scale", float(a.tracking_amplitude)
        )
    cfg.observation_space = observation_space_for_cfg(cfg)
    cfg.use_eval_terrain = bool(a.eval_terrain)
    cfg.terrain_eval_seed_override = int(a.eval_terrain_seed)
    cfg.terrain_base_seed = int(a.seed)
    cfg.tracking_path_family = str(a.path_family)
    cfg.enable_contact_tracking_demo = False
    cfg.enable_visual_grid_floor = False
    cfg.enable_road_scenery = False
    cfg.terrain_terminate_on_contact_loss = False
    cfg.terrain_terminate_on_hard_impact = False
    cfg.defer_safety_terminations = True
    # Do NOT let the head-down safety threshold censor the very excursion this
    # script exists to measure.
    cfg.head_down_termination_alignment = -1.0
    cfg.randomize_start_phase = False
    cfg.randomize_trajectory_shape = False
    cfg.trajectory_shape = str(cfg.terrain_deployment_trajectory_shape)
    configure_terrain_gpu_physx_memory(cfg, int(a.num_envs))
    assert_teacher_disabled(cfg, context="head-frame probe")
    env = gym.make(TASK_ID, cfg=cfg, render_mode=None)
    return env, env.unwrapped


def _q(t: torch.Tensor, level: float) -> float:
    return float(torch.quantile(t.flatten().float(), level)) if t.numel() else float("nan")


def main() -> None:
    env, raw = build_env()
    runner_cfg = RUNNER_CFG_CLS()
    runner_cfg.seed = int(a.seed)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)
    runner = OnPolicyRunner(wrapped, runner_cfg.to_dict(), log_dir=None, device=a.device)
    runner.load(
        a.checkpoint,
        load_cfg={"actor": True, "critic": True, "optimizer": False,
                  "iteration": True, "rnd": False},
    )
    policy = runner.get_inference_policy(device=a.device)

    slew_id = int(raw._arm_joint_ids[0])
    obs, _ = raw.reset(seed=int(a.seed))
    A, W, J, S, X, Y, Q5 = [], [], [], [], [], [], []
    for _ in range(int(a.steps)):
        action = policy(obs)
        obs, _, _, _, _ = raw.step(action)
        axis = raw._head_down_axis_world().detach()
        A.append(-axis[:, 2].cpu())
        X.append(axis[:, 0].cpu())
        Y.append(axis[:, 1].cpu())
        # What the shield and the levelling correction actually use.
        W.append(torch.atan2(axis[:, 1], -axis[:, 2]).cpu())
        # What the joint can actually correct.
        J.append(raw._head_joint_signed_tilt().detach().cpu())
        S.append(raw.robot.data.joint_pos[:, slew_id].detach().cpu())
        Q5.append(raw._sole_raw[:, 4].detach().cpu() if raw._sole_raw.shape[-1] > 4
                  else torch.zeros(raw.num_envs))

    align = torch.stack(A); world = torch.stack(W); joint = torch.stack(J)
    slew = torch.stack(S); ax = torch.stack(X); ay = torch.stack(Y); q5 = torch.stack(Q5)

    collapse = align < float(a.collapse_alignment)
    ok = ~collapse

    def block(mask, name):
        if not bool(mask.any()):
            return {"name": name, "samples": 0}
        w = world[mask].abs(); j = joint[mask].abs()
        return {
            "name": name,
            "samples": int(mask.sum()),
            "fraction": float(mask.float().mean()),
            "mean_alignment": float(align[mask].mean()),
            "mean_abs_slew_rad": float(slew[mask].abs().mean()),
            "p95_abs_slew_rad": _q(slew[mask].abs(), 0.95),
            "mean_abs_world_tilt_rad": float(w.mean()),
            "mean_abs_joint_tilt_rad": float(j.mean()),
            # THE number. >1 means the world measure under-reads what the joint
            # could correct, i.e. the controller is blind in this population.
            "joint_over_world_ratio": float(j.mean() / w.mean().clamp(min=1e-9)),
            "mean_abs_head_axis_x": float(ax[mask].abs().mean()),
            "mean_abs_head_axis_y": float(ay[mask].abs().mean()),
            "x_over_y_ratio": float(
                ax[mask].abs().mean() / ay[mask].abs().mean().clamp(min=1e-9)
            ),
            "mean_abs_q5_action": float(q5[mask].abs().mean()),
        }

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "checkpoint": str(a.checkpoint),
        "actuator": a.actuator,
        "stage": int(a.stage),
        "path_family": a.path_family,
        "tracking_amplitude": float(raw.cfg.terrain_tracking_amplitude_scale),
        "eval_terrain_seed": int(a.eval_terrain_seed),
        "steps": int(a.steps),
        "num_envs": int(raw.num_envs),
        "collapse_threshold": float(a.collapse_alignment),
        "min_alignment": float(align.min()),
        "p01_alignment": _q(align, 0.01),
        "overall": block(torch.ones_like(collapse), "all"),
        "healthy": block(ok, f"alignment >= {a.collapse_alignment}"),
        "collapsed": block(collapse, f"alignment < {a.collapse_alignment}"),
        "slew_correlation": {
            "corr_abs_slew_vs_abs_world_minus_joint": float(
                torch.corrcoef(
                    torch.stack([
                        slew.abs().flatten(),
                        (joint.abs() - world.abs()).flatten(),
                    ])
                )[0, 1]
            ),
            "corr_abs_slew_vs_alignment": float(
                torch.corrcoef(
                    torch.stack([slew.abs().flatten(), align.flatten()])
                )[0, 1]
            ),
        },
    }
    print(json.dumps(report, indent=2), flush=True)
    if a.report:
        path = Path(a.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2))
        print(f"[PPO3-SOLE] wrote {path}", flush=True)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
