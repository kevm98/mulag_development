#!/usr/bin/env python3
"""Measure which cylinders can steer the cutter head's world-down axis.

The stage-5 collapses are all X-axis head excursions (``max_abs_head_axis_x``
0.86-0.98 against 0.0185 healthy) that never recover -- measured directly:
``head_down_latch_released_fraction`` stayed at 0.00000 for a whole block. The
c4 tilt shield is explicitly blind to that axis (it fades on
``atan2(axis_y, -axis_z)``), and the repository already unit-tests the gap in
``test_directional_tilt_shield_cannot_act_on_a_pure_out_of_plane_excursion``,
whose docstring says holding the head against this axis needs "authority the
shield does not have (termination, or the DOF that produced the tilt)".

Nobody has measured whether that authority exists. This probe answers it
directly, by finite-differencing the head's world down-axis against each
cylinder in turn:

    J[i][a] = d(head_axis_world[a]) / d(action_i)

Read it as: if ``J[i][x]`` is materially non-zero for any cylinder, the X
excursion IS steerable and the fix is to give the shield or the reward that
channel. If every ``J[i][x]`` is ~0, X is genuinely uncontrollable by the four
cylinders and no amount of shaping or exploration tuning can recover it -- the
answer then lies in the geometry (the 0.76 c4->Messerkopf coupling, or locking
that joint) rather than in the policy.

The Jacobian is sampled at several path phases, not only at deployment, because
a slewed arm rotates the plane c4 acts in: authority measured at one pose does
not generalize. Zero action is held between perturbations so each column is a
clean derivative rather than a mixture.
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
p.add_argument("--num_envs", type=int, default=32)
p.add_argument("--seed", type=int, default=17)
p.add_argument("--stage", type=int, default=5)
p.add_argument(
    "--settle_steps",
    type=int,
    default=40,
    help="Zero-action steps between perturbations so each column is clean.",
)
p.add_argument(
    "--hold_steps",
    type=int,
    default=25,
    help="Steps a perturbation is held before the response is read.",
)
p.add_argument(
    "--phases",
    type=int,
    default=4,
    help="Path phases at which the Jacobian is re-measured.",
)
p.add_argument(
    "--phase_gap_steps",
    type=int,
    default=90,
    help="Zero-action steps advanced between phases.",
)
p.add_argument("--amplitude", type=float, default=0.6, help="Perturbation action.")
p.add_argument("--eval_terrain", action="store_true")
p.add_argument("--path_family", default="mixed")
p.add_argument("--report_out", default="")
AppLauncher.add_app_launcher_args(p)
a, unknown = p.parse_known_args()
sys.argv = [sys.argv[0], *unknown]

if min(a.num_envs, a.settle_steps, a.hold_steps, a.phases) <= 0:
    p.error("env count, settle/hold steps and phase count must be positive")
if not 0.0 < a.amplitude <= 1.0:
    p.error("--amplitude must lie in (0, 1]")

launcher = AppLauncher(a)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import verge_task  # noqa: F401,E402
from verge_task.terrain_env_cfg import (  # noqa: E402
    apply_terrain_curriculum_stage,
    configure_terrain_gpu_physx_memory,
)
from verge_task.terrain_observations import (  # noqa: E402
    observation_space_for_cfg,
)
from verge_task.terrain_convergence import GATE_SPECS  # noqa: E402

AXES = ("x", "y", "z")


def build_env():
    env_cfg = parse_env_cfg(
        verge_task.TERRAIN_TASK_ID, device=a.device, num_envs=a.num_envs
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
    # Never let a head excursion reset the scene mid-measurement: a synchronized
    # reset would silently replace the pose the derivative is taken about.
    env_cfg.defer_safety_terminations = True
    env_cfg.head_down_termination_alignment = float(
        GATE_SPECS["head_alignment_hard_min"].threshold
    )
    env_cfg.randomize_start_phase = False
    env_cfg.randomize_trajectory_shape = False
    env_cfg.trajectory_shape = str(env_cfg.terrain_deployment_trajectory_shape)
    configure_terrain_gpu_physx_memory(env_cfg, int(a.num_envs))
    env = gym.make(verge_task.TERRAIN_TASK_ID, cfg=env_cfg, render_mode=None)
    return env, env.unwrapped


def main() -> int:
    env, raw = build_env()
    raw.reset(seed=int(a.seed))
    zero = torch.zeros(
        (raw.num_envs, int(raw.cfg.action_space)),
        device=raw.device,
        dtype=torch.float32,
    )

    def step_n(action: torch.Tensor, count: int) -> None:
        for _ in range(int(count)):
            raw.step(action)

    def head_axis() -> torch.Tensor:
        return raw._head_down_axis_world().detach().double().cpu()

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "stage": int(a.stage),
        "terrain": "held_out" if a.eval_terrain else "training",
        "num_envs": int(raw.num_envs),
        "amplitude": float(a.amplitude),
        "hold_steps": int(a.hold_steps),
        "settle_steps": int(a.settle_steps),
        "note": (
            "J[cylinder][axis] = mean change in the head's world down-axis "
            "component per unit action, held for hold_steps. The desired axis "
            "is (0,0,-1); an X excursion is what the collapses show."
        ),
        "phases": [],
    }

    for phase in range(int(a.phases)):
        step_n(zero, a.settle_steps)
        baseline = head_axis()
        phase_rows = []
        for cylinder in range(4):
            for sign in (1.0, -1.0):
                action = zero.clone()
                action[:, cylinder] = float(sign) * float(a.amplitude)
                step_n(action, a.hold_steps)
                moved = head_axis()
                delta = (moved - baseline) / (float(sign) * float(a.amplitude))
                phase_rows.append(
                    {
                        "cylinder": cylinder + 1,
                        "sign": sign,
                        "d_axis_per_unit_action": {
                            AXES[i]: float(delta[:, i].mean()) for i in range(3)
                        },
                        "abs_d_axis_per_unit_action": {
                            AXES[i]: float(delta[:, i].abs().mean()) for i in range(3)
                        },
                    }
                )
                # Return toward the baseline pose before the next column.
                step_n(zero, a.settle_steps)
                baseline = head_axis()
        report["phases"].append(
            {
                "phase": phase,
                "baseline_axis": {
                    AXES[i]: float(baseline[:, i].mean()) for i in range(3)
                },
                "baseline_alignment": float((-baseline[:, 2]).mean()),
                "columns": phase_rows,
            }
        )
        step_n(zero, a.phase_gap_steps)

    # Authority summary: the largest |d axis / d action| any cylinder achieves on
    # each axis, and which cylinder achieves it.
    summary = {}
    for axis in AXES:
        best_cyl, best_val = None, 0.0
        per_cyl = {c: 0.0 for c in range(1, 5)}
        for ph in report["phases"]:
            for col in ph["columns"]:
                v = abs(col["d_axis_per_unit_action"][axis])
                per_cyl[col["cylinder"]] = max(per_cyl[col["cylinder"]], v)
                if v > best_val:
                    best_val, best_cyl = v, col["cylinder"]
        summary[axis] = {
            "max_abs_authority": best_val,
            "best_cylinder": best_cyl,
            "per_cylinder_max_abs": per_cyl,
        }
    report["authority_summary"] = summary
    x_auth = summary["x"]["max_abs_authority"]
    y_auth = summary["y"]["max_abs_authority"]
    report["x_vs_y_authority_ratio"] = (
        float(x_auth / y_auth) if y_auth > 0.0 else None
    )
    report["verdict"] = (
        "X is steerable; extend the shield/reward to the responsible cylinder"
        if x_auth >= 0.1 * max(y_auth, 1.0e-9)
        else "X is not steerable by the four cylinders; the fix is geometric "
        "(c4->Messerkopf coupling or locking that joint), not policy-side"
    )

    out = (
        Path(a.report_out).expanduser().resolve()
        if a.report_out
        else ROOT / "verge_task" / "reports" / "head_axis_authority.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["authority_summary"], indent=2), flush=True)
    print("x/y authority ratio:", report["x_vs_y_authority_ratio"], flush=True)
    print("VERDICT:", report["verdict"], flush=True)
    print(f"[PPO3-HEAD-AUTHORITY] wrote {out}", flush=True)

    env.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
