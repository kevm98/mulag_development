#!/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
"""Is cylinder 4's authority limited by the MODEL, or only by the command?

WHY MEASURE FIRST
=================
The head/lateral coupling is the last open PPO3 blocker: whenever the head is
protected, worst-family lateral lands at ~0.52; whenever it is free, lateral
reaches 0.37 and head alignment falls to ~0.71. c4 (``messerkopf_schwenk_cylinder``)
is the axis that could decouple it -- if it has the authority.

"More authority" can mean three different things, and only measurement says
which one is binding:

  1. COMMAND authority  -- the anti-windup lead cap
     (``cylinder_target_max_lead_per_axis_m``) bounds how far the reference may
     lead the plant. Swept to 0.20 only.
  2. ACTION authority   -- ``action_scale`` bounds how much command the policy
     adds per step.
  3. MODEL authority    -- ``effort_limit_sim`` bounds the force the drive can
     produce. All five cylinders currently share one global
     ``CYLINDER_EFFORT_SCALE = 16.0``.

Raising (3) when the drive never saturates achieves exactly nothing, and raising
it when it DOES saturate is the only way the other two become effective. This
probe reports, per cylinder, the applied torque against the enforced limit.

READING THE RESULT
==================
* ``saturated_fraction`` near 0 for c4  -> the model is NOT the limit. Do not
  touch effort; the constraint is the lead cap and/or action scale.
* ``saturated_fraction`` materially > 0 -> the drive is force-limited and c4
  physically cannot hold the head. Raising c4's effort limit is then justified,
  and is the smallest change that makes the command-side knobs meaningful.

Evaluation only. No weights are modified.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--num_envs", type=int, default=16)
p.add_argument("--steps", type=int, default=480)
p.add_argument("--seed", type=int, default=17)
p.add_argument("--stage", type=int, default=6)
p.add_argument("--eval_terrain_seed", type=int, default=0)
p.add_argument("--report", default="")
p.add_argument(
    "--saturation_margin",
    type=float,
    default=0.98,
    help="Fraction of the enforced limit above which a step counts as saturated.",
)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
a.headless = True

app = AppLauncher(a).app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import verge_task  # noqa: E402,F401  (registers the shared verge task)
from verge_task import TERRAIN_TASK_ID  # noqa: E402
from verge_task.terrain_env_cfg import apply_terrain_curriculum_stage  # noqa: E402
from verge_task.terrain_observations import override_cfg  # noqa: E402

CYLINDERS = (
    "drehzapfen_cylinder",
    "ausleger_i_cylinder",
    "ausleger_ii_cylinder",
    "messerkopf_schwenk_cylinder",
)


def main() -> None:
    env_cfg = parse_env_cfg(TERRAIN_TASK_ID, device=a.device, num_envs=a.num_envs)
    apply_terrain_curriculum_stage(env_cfg, int(a.stage))
    override_cfg(env_cfg, "use_eval_terrain", True)
    if a.eval_terrain_seed > 0:
        override_cfg(env_cfg, "terrain_eval_seed_override", int(a.eval_terrain_seed))

    env = gym.make(TERRAIN_TASK_ID, cfg=env_cfg, render_mode=None)
    raw = env.unwrapped
    raw.reset(seed=int(a.seed))

    # No checkpoint is loaded on purpose: this is a PLANT measurement under
    # zero action, so the observation contract is irrelevant to it.

    # Enforced limits, read from the live articulation rather than assumed.
    limits: dict[str, float] = {}
    for name in CYLINDERS:
        actuator = raw.robot.actuators.get(name)
        limit = getattr(actuator, "effort_limit_sim", None) if actuator else None
        # effort_limit_sim is a per-environment tensor, not a scalar. Every
        # environment shares the authored value, so take the first finite entry.
        if limit is None:
            limits[name] = float("nan")
        elif torch.is_tensor(limit):
            limits[name] = float(limit.detach().flatten()[0])
        else:
            limits[name] = float(limit)

    joint_ids = raw._cylinder_joint_ids
    peak = torch.zeros(len(joint_ids), device=raw.device)
    total = torch.zeros(len(joint_ids), device=raw.device)
    saturated = torch.zeros(len(joint_ids), device=raw.device)
    samples = 0

    limit_row = torch.tensor(
        [limits[name] for name in CYLINDERS][: len(joint_ids)],
        device=raw.device,
        dtype=torch.float32,
    )

    with torch.no_grad():
        for _ in range(int(a.steps)):
            # Zero action is deliberate here: this probe measures the PLANT's
            # force demand while holding the deployed pose, which is the load
            # c4 must overcome to keep the head level. A policy in the loop
            # would confound "the drive cannot" with "the policy did not ask".
            action = torch.zeros(
                (raw.num_envs, int(raw.cfg.action_space)), device=raw.device
            )
            raw.step(action)
            effort = raw.robot.data.applied_torque[:, joint_ids].abs()
            peak = torch.maximum(peak, effort.max(dim=0).values)
            total += effort.mean(dim=0)
            saturated += (
                effort >= float(a.saturation_margin) * limit_row
            ).float().mean(dim=0)
            samples += 1

    report = {
        "steps": samples,
        "environments": int(raw.num_envs),
        "eval_terrain_seed": int(a.eval_terrain_seed),
        "saturation_margin": float(a.saturation_margin),
        "cylinders": {},
    }
    for index, name in enumerate(CYLINDERS[: len(joint_ids)]):
        report["cylinders"][name] = {
            "effort_limit_sim_n": limits[name],
            "mean_abs_effort_n": float(total[index] / max(samples, 1)),
            "peak_abs_effort_n": float(peak[index]),
            "utilisation_peak": float(peak[index]) / max(limits[name], 1e-9),
            "saturated_fraction": float(saturated[index] / max(samples, 1)),
        }

    c4 = report["cylinders"].get("messerkopf_schwenk_cylinder", {})
    report["c4_model_limited"] = bool(c4.get("saturated_fraction", 0.0) > 0.01)
    report["verdict"] = (
        "c4 IS force-limited: raising its effort_limit_sim is justified"
        if report["c4_model_limited"]
        else "c4 is NOT force-limited: the constraint is the lead cap / action "
        "scale, not the model. Do not raise effort."
    )

    print("[PPO3-1-C4-PROBE] " + json.dumps(report, sort_keys=True), flush=True)
    if a.report:
        out = Path(a.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True))
    env.close()


main()
app.close()
