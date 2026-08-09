#!/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
"""Score a PPO3-1 checkpoint on the 8-gate multi-mesh acceptance contract.

This is the acceptance path the review brief flags as missing: the shared
``evaluate.py`` scores the old 17-gate contract, while the rebuilt 8-gate
contract in ``PPO_ROM/gates.py`` is exercised only by unit tests and
preflight. This script runs an actor-mean rollout on at least three held-out
meshes, collects exactly the eight metrics ``gates.check_gates`` reads, and
reports the worst-mesh aggregate plus explicit margins.

Usage:
    evaluate_gates.py logs/rsl_rl/ppo3_1_sole/student_distilled.pt
    evaluate_gates.py logs/rsl_rl/ppo3_1_sole/student_distilled.pt --report out.json
    evaluate_gates.py <ckpt> --meshes 1337,17,909 --steps 960 --num_envs 64
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isaaclab.app import AppLauncher  # noqa: E402

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("checkpoint", nargs="?", default=None)
p.add_argument("--num_envs", type=int, default=64)
p.add_argument("--seed", type=int, default=17)
p.add_argument("--steps", type=int, default=960)
p.add_argument("--stage", type=int, default=5)
p.add_argument("--meshes", default="1337,17,909")
p.add_argument("--report", default="")
p.add_argument(
    "--cylinder_target_max_lead_per_axis_m",
    default="",
    help=(
        "Per-cylinder anti-windup lead caps in metres, comma separated, one "
        "per cylinder (e.g. '0.05,0.05,0.05,0.018'). Measured in this repo: "
        "relaxing the c1-c3 cap halves lateral error on the sustained-offset "
        "path families (boom_sweep 0.3745 -> 0.1602), while the SAME "
        "relaxation on c4 collapses head alignment."
    ),
)
p.add_argument(
    "--teacher",
    action="store_true",
    help=(
        "Score the privileged DLS teacher instead of a checkpoint. The teacher "
        "is not a checkpoint -- it is a runtime env mode, so no policy is "
        "loaded and every environment is driven by the teacher. Use this to "
        "anchor gate thresholds in what the plant can actually achieve."
    ),
)
p.add_argument(
    "--null",
    action="store_true",
    help=(
        "Score the null (zero-action) policy. Gate thresholds must separate a "
        "working controller from doing nothing, and gates.py's own rule "
        "demands >=3 meshes -- this measures that baseline on the same "
        "contract instead of a single pinned mesh."
    ),
)
p.add_argument(
    "--tracking_amplitude",
    type=float,
    default=None,
    help=(
        "Override terrain_tracking_amplitude_scale (and collapse the "
        "randomization band to it). This is how the GENERALIZATION ENVELOPE is "
        "measured: the policy trains on 0.30-0.75, so scoring at 0.20 or 0.85 "
        "asks what it does OUTSIDE its training distribution. Measured here on "
        "2026-08-06, a student trained at 2.2 and scored at 0.7 degraded 5x -- "
        "so the envelope is a real limit, not a formality."
    ),
)
p.add_argument(
    "--task",
    default="",
    help=(
        "Gym task to score. Empty = PPO3_1 (ROM-A). Set "
        "RRLAB-Mulag-Verge-IDFD-v0 to score the ID/FD pipeline on the "
        "IDENTICAL terrain, reward and gates -- that shared path is what makes "
        "a PPO3_1 vs PPO4_1 number a statement about the actuator rather than "
        "about two diverged copies of a script."
    ),
)
AppLauncher.add_app_launcher_args(p)
a, unknown = p.parse_known_args()
sys.argv = [sys.argv[0], *unknown]

meshes = tuple(int(part) for part in a.meshes.split(",") if part.strip())
if not meshes:
    p.error("--meshes must name at least one terrain seed")
if a.checkpoint is None and not a.teacher and not a.null:
    p.error("evaluate_gates.py requires a checkpoint path (or --teacher / --null)")
if a.checkpoint is not None and (a.teacher or a.null):
    p.error("--teacher/--null score a runtime mode, not a checkpoint; pass one or the other")
if a.teacher and a.null:
    p.error("--teacher and --null are mutually exclusive")

launcher = AppLauncher(a)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import PPO_ROM  # noqa: F401,E402
from PPO_ROM.gates import GATES, check_gates  # noqa: E402

if a.task.startswith("RRLAB-Mulag-Verge-IDFD"):
    import importlib

    _pkg = importlib.import_module("PPO_ID_FD")
    TASK_ID = a.task
    EXPECTED_ACTUATOR = "idfd"
else:
    _pkg = PPO_ROM
    TASK_ID = a.task or PPO_ROM.TERRAIN_TASK_ID
    EXPECTED_ACTUATOR = "rom"
print(f"[PPO3-1-GATES] task={TASK_ID} actuator={EXPECTED_ACTUATOR}", flush=True)
from PPO_ROM.terrain_env_cfg import (  # noqa: E402
    TerrainC4AuthorityPPORunnerCfg,
    apply_c4_authority_stage,
)
from verge_task.actuators import assert_checkpoint_actuator  # noqa: E402
from verge_task.terrain_env_cfg import (  # noqa: E402
    configure_terrain_gpu_physx_memory,
)
from verge_task.terrain_observations import (  # noqa: E402
    observation_space_for_cfg,
    override_cfg,
)


@torch.no_grad()
def roll_mesh(raw, policy, steps: int, mesh: int, *, teacher: bool = False) -> dict:
    """Actor-mean rollout on one held-out mesh, collecting the 8 gate metrics.

    With ``teacher=True`` the DLS teacher drives every environment. The mask is
    set AFTER reset because ``_teacher_execute_mask`` is allocated at env
    construction and reset is the point the rollout actually starts from.
    """
    obs, _ = raw.reset(seed=int(a.seed) + mesh)
    if teacher:
        raw.set_teacher_execute_mask(
            torch.ones(raw.num_envs, dtype=torch.bool, device=raw.device)
        )
    errors, accelerations, alignments = [], [], []
    laterals, contacts = [], []
    headrooms = []
    safety_terminations = 0
    # Progress output every 10% of the rollout. Without it this loop is silent
    # for its whole duration, which makes a hang indistinguishable from normal
    # work -- a teacher evaluation sat at 108% CPU for 40 minutes on 2026-08-06
    # with nothing written since the env built, and the only way to tell was to
    # compare the log mtime against the wall clock. Silence is not progress.
    _report_every = max(int(steps) // 10, 1)
    _t0 = time.time()
    for _step in range(int(steps)):
        if _step and _step % _report_every == 0:
            _elapsed = time.time() - _t0
            print(
                f"[PPO3-1-GATES] mesh {mesh}: step {_step}/{int(steps)} "
                f"({100.0 * _step / int(steps):.0f}%) {_elapsed:.0f}s elapsed, "
                f"~{_elapsed / _step * (int(steps) - _step):.0f}s left",
                flush=True,
            )
        action = policy(obs)
        obs, _, _, _, _ = raw.step(action)
        errors.append(raw.cut_height_error().abs())
        accelerations.append(
            torch.linalg.norm(raw._measured_tool_acceleration_truck, dim=-1)
        )
        alignments.append(-raw._head_down_axis_world()[:, 2])
        laterals.append(raw.verge_lateral_error().abs())
        contacts.append(raw.cutter_ground_contact().float())
        measured_x, _ = raw._refresh_cylinder_state()
        headroom_low, headroom_high = raw._cylinder_headroom(measured_x)
        headrooms.append(
            torch.minimum(headroom_low, headroom_high).amin(dim=-1)
        )
        safety_terminations += int(raw.terrain_safety_violation().sum())

    error_steps = torch.stack(errors)
    lateral_steps = torch.stack(laterals)
    contact_steps = torch.stack(contacts)
    alignment_values = torch.cat(alignments)
    headroom_steps = torch.stack(headrooms)
    error = error_steps.reshape(-1)

    family_rows = []
    for family in sorted(set(raw._tracking_path_family_names)):
        family_mask = torch.as_tensor(
            [name == family for name in raw._tracking_path_family_names],
            dtype=torch.bool,
            device=raw.device,
        )
        family_rows.append(
            {
                "family": family,
                "mean_abs_cut_height_error_m": float(
                    error_steps[:, family_mask].mean()
                ),
                "mean_abs_lateral_error_m": float(
                    lateral_steps[:, family_mask].mean()
                ),
                "ground_contact_fraction": float(
                    contact_steps[:, family_mask].mean()
                ),
            }
        )

    return {
        "mesh": mesh,
        "steps": int(steps),
        "num_envs": int(raw.num_envs),
        # The eight keys the 8-gate contract reads, and nothing else.
        "mean_abs_cut_height_error_m": float(error.mean()),
        "worst_family_mean_abs_lateral_error_m": max(
            row["mean_abs_lateral_error_m"] for row in family_rows
        ),
        "worst_family_ground_contact_fraction": min(
            row["ground_contact_fraction"] for row in family_rows
        ),
        "p01_head_alignment": float(torch.quantile(alignment_values, 0.01)),
        "hard_impact_step_fraction": float(raw._hard_impact_steps.sum())
        / max(steps * raw.num_envs, 1),
        "p95_tool_acceleration_mps2": float(
            torch.quantile(torch.cat(accelerations), 0.95)
        ),
        "min_cylinder_headroom_m": float(headroom_steps.min()),
        "safety_termination_rate": safety_terminations
        / max(steps * raw.num_envs, 1),
        "path_family_results": family_rows,
    }


def main() -> None:
    # The teacher is a runtime env mode, not a checkpoint: there is nothing to
    # load and nothing to validate. `assert_teacher_disabled` is deliberately
    # NOT imported here -- it guards the policy evaluator, whose numbers would
    # be meaningless if the teacher were secretly driving. Here the teacher IS
    # the subject, and the report records that explicitly.
    if a.teacher or a.null:
        checkpoint_path = None
    else:
        checkpoint_path = Path(a.checkpoint).resolve()
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        infos = payload.get("infos") or {}
        source = str(infos.get("source", ""))
        schema = str(infos.get("schema", ""))
        if schema not in {"terrain_teacher_student_v1", "terrain_shared_v2"} or source not in {
            "PPO3 terrain teacher distillation v1",
            "PPO3 terrain PPO",
            "PPO3-1 terrain PPO",
        }:
            raise RuntimeError(
                "evaluate_gates accepts only supported terrain policy checkpoints; "
                f"got source={source!r}, schema={schema!r}"
            )
        assert_checkpoint_actuator(
            infos, EXPECTED_ACTUATOR, context="PPO3-1 gate evaluation"
        )

    per_mesh: dict[str, dict] = {}
    for mesh in meshes:
        env_cfg = parse_env_cfg(
            TASK_ID,
            device=a.device,
            num_envs=a.num_envs,
        )
        apply_c4_authority_stage(env_cfg, int(a.stage))
        if a.cylinder_target_max_lead_per_axis_m:
            override_cfg(
                env_cfg,
                "cylinder_target_max_lead_per_axis_m",
                tuple(
                    float(part)
                    for part in a.cylinder_target_max_lead_per_axis_m.split(",")
                    if part.strip()
                ),
            )
        env_cfg.observation_space = observation_space_for_cfg(env_cfg)
        if a.tracking_amplitude is not None:
            # Collapse the band too: leaving the randomization tuple in place
            # would keep sampling the TRAINED amplitudes and the override would
            # measure nothing.
            override_cfg(
                env_cfg, "terrain_tracking_amplitude_scale",
                float(a.tracking_amplitude),
            )
            override_cfg(
                env_cfg, "terrain_tracking_amplitude_scales",
                (float(a.tracking_amplitude),),
            )
        env_cfg.use_eval_terrain = True
        env_cfg.terrain_base_seed = int(a.seed)
        env_cfg.terrain_eval_seed_override = mesh
        env_cfg.enable_contact_tracking_demo = False
        env_cfg.enable_visual_grid_floor = False
        env_cfg.enable_road_scenery = False
        env_cfg.randomize_start_phase = False
        env_cfg.randomize_trajectory_shape = False
        env_cfg.trajectory_shape = str(env_cfg.terrain_deployment_trajectory_shape)
        if a.teacher:
            # Must be set before construction: set_teacher_execute_mask raises
            # while the mode is still "off".
            env_cfg.sole_teacher_mode = "execute"
        configure_terrain_gpu_physx_memory(env_cfg, int(a.num_envs))

        env = gym.make(TASK_ID, cfg=env_cfg, render_mode=None)
        raw = env.unwrapped
        if a.teacher or a.null:
            # No policy exists to load. Under --teacher the teacher overwrites
            # the command after the plant action is final, so the zeros are
            # ignored; under --null the zeros ARE the policy under test.
            zeros = torch.zeros(
                (raw.num_envs, int(env_cfg.action_space)), device=a.device
            )

            def policy(_obs, _zeros=zeros):
                return _zeros
        else:
            runner_cfg = TerrainC4AuthorityPPORunnerCfg()
            runner_cfg.seed = int(a.seed)
            wrapped = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)
            runner = OnPolicyRunner(
                wrapped, runner_cfg.to_dict(), log_dir=None, device=a.device
            )
            runner.load(
                str(checkpoint_path),
                load_cfg={
                    "actor": True,
                    "critic": True,
                    "optimizer": False,
                    "iteration": True,
                    "rnd": False,
                },
            )
            policy = runner.get_inference_policy(device=a.device)
        per_mesh[str(mesh)] = roll_mesh(
            raw, policy, int(a.steps), mesh, teacher=bool(a.teacher)
        )
        env.close()

    if len(per_mesh) >= 3:
        result = check_gates(per_mesh)
    else:
        result = {
            "meshes": len(per_mesh),
            "metrics": {},
            "gates": {},
            "margins": {},
            "passed": False,
            "failing": [],
            "thin_margins": [],
            "note": "fewer than 3 meshes; per-mesh metrics saved, gates not evaluated",
        }
    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        # Provenance, so teacher numbers can never be mistaken for policy
        # numbers by a later reader.
        "subject": (
            "dls_teacher"
            if a.teacher
            else "null_zero_action" if a.null else "policy_checkpoint"
        ),
        "sole_teacher_mode": "execute" if a.teacher else "off",
        # The lead cap changes what the plant can physically do, so a report
        # that does not state its own caps cannot be compared with another.
        "cylinder_target_max_lead_per_axis_m": list(
            getattr(env_cfg, "cylinder_target_max_lead_per_axis_m", ()) or ()
        ),
        "lead_cap_overridden": bool(a.cylinder_target_max_lead_per_axis_m),
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "stage": int(a.stage),
        "actuator": "rom",
        "action_width": int(env_cfg.action_space),
        "meshes": sorted(per_mesh),
        # Keep the per-family breakdown: worst-family gates are only
        # interpretable when the driving family is visible (difficulty vs
        # infeasibility).
        "per_mesh": per_mesh,
        "gates": result,
    }
    printable = {
        "checkpoint": str(checkpoint_path),
        "gates": result,
        "per_mesh_metrics": report["per_mesh"],
    }
    print(json.dumps(printable, indent=2), flush=True)

    if a.report:
        path = Path(a.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2))
        print(f"[PPO3-1-GATES] wrote {path}", flush=True)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
