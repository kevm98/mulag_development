#!/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
"""Can the privileged teacher still mow when ROM-A is replaced by ID/FD?

This is the decision gate before any PPO4 training. The teacher is the adaptive
terrain/contact controller PPO3 distills from; it reads privileged state and
commands cylinders directly. If it cannot hold the cutter on the verge through
the ID/FD actuator, then no student distilled from it can either, and training
would be measuring the actuator's inability rather than the policy's.

Three things are measured, in the order they can fail:

1.  **Authority.** Do commanded cylinder increments actually move the cylinders?
    A +/- basis sweep per cylinder, reporting realized displacement per unit
    command. The FD model was once rank ~2 and under-fit the valve, so this is
    checked first and explicitly -- a rank-deficient actuator produces plausible
    tracking numbers while two axes quietly do nothing.
2.  **Tracking.** With the teacher driving, cut-height and lateral error against
    the same thresholds PPO3's gates use.
3.  **Contact.** Ground-contact fraction and the ejection statistics that
    diagnosed the PPO3 bounce, so a regression in the compliant plant shows up
    here rather than after a training run.

Every statistic reports its sample count. Run it against PPO3 as well
(--task ppo3) to get the ROM-A reference on the identical terrain and seed --
the comparison is the whole point, and an absolute PPO4 number means little on
its own.
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
p.add_argument(
    "--task",
    default="ppo4",
    choices=("ppo4", "ppo3"),
    help="ppo4 = ID/FD actuator; ppo3 = ROM-A reference on identical terrain.",
)
p.add_argument("--num_envs", type=int, default=32)
p.add_argument("--seed", type=int, default=17)
p.add_argument("--steps", type=int, default=480)
p.add_argument("--stage", type=int, default=5)
p.add_argument("--eval_terrain", action="store_true")
p.add_argument("--path_family", default="mixed")
p.add_argument(
    "--authority_steps",
    type=int,
    default=40,
    help="Steps per direction in the per-cylinder authority sweep.",
)
p.add_argument(
    "--authority_magnitude",
    type=float,
    default=0.5,
    help="Commanded action magnitude for the authority sweep.",
)
p.add_argument("--settle_steps", type=int, default=20)
p.add_argument(
    "--reference_report",
    default="",
    help=(
        "A probe report from the OTHER actuator on identical terrain/seed. "
        "Supplying it turns the verdict into a relative comparison, which is "
        "the only thing an actuator swap needs to prove."
    ),
)
p.add_argument("--report", default="")
AppLauncher.add_app_launcher_args(p)
a, unknown = p.parse_known_args()
sys.argv = [sys.argv[0], *unknown]

launcher = AppLauncher(a)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

if a.task == "ppo4":
    import PPO4_ID_FD_PhysX as PKG  # noqa: E402

    TASK_ID = PKG.TERRAIN_TASK_ID
else:
    import PPO3_SOLE_PhysX as PKG  # noqa: E402

    TASK_ID = PKG.TERRAIN_TASK_ID

from PPO3_SOLE_PhysX.terrain_env_cfg import (  # noqa: E402
    apply_terrain_curriculum_stage,
    configure_terrain_gpu_physx_memory,
)
from PPO3_SOLE_PhysX.terrain_observations import (  # noqa: E402
    observation_space_for_cfg,
)


def build_env():
    env_cfg = parse_env_cfg(TASK_ID, device=a.device, num_envs=a.num_envs)
    apply_terrain_curriculum_stage(env_cfg, int(a.stage))
    env_cfg.sole_ik_hint = False
    env_cfg.observation_space = observation_space_for_cfg(env_cfg)
    env_cfg.use_eval_terrain = bool(a.eval_terrain)
    env_cfg.terrain_base_seed = int(a.seed)
    env_cfg.tracking_path_family = str(a.path_family)
    env_cfg.enable_visual_grid_floor = False
    env_cfg.enable_road_scenery = False
    env_cfg.defer_safety_terminations = True
    env_cfg.randomize_start_phase = False
    env_cfg.randomize_trajectory_shape = False
    env_cfg.trajectory_shape = str(env_cfg.terrain_deployment_trajectory_shape)
    configure_terrain_gpu_physx_memory(env_cfg, int(a.num_envs))
    env = gym.make(TASK_ID, cfg=env_cfg, render_mode=None)
    return env, env.unwrapped


def stats(t) -> dict:
    if t is None or t.numel() == 0:
        return {}
    t = t.float()
    return {
        "n": int(t.numel()),
        "mean": round(float(t.mean()), 5),
        "p50": round(float(torch.quantile(t, 0.5)), 5),
        "p95": round(float(torch.quantile(t, 0.95)), 5),
        "min": round(float(t.min()), 5),
        "max": round(float(t.max()), 5),
    }


def measure_authority(raw) -> dict:
    """Per-cylinder realized displacement under a held +/- command.

    Reported per cylinder and per direction. A cylinder that moves in only one
    direction, or not at all, is a rank-deficiency signature -- the FD model has
    had exactly that failure before, and it is invisible in tracking metrics.
    """
    rows = []
    steps = int(a.authority_steps)
    magnitude = float(a.authority_magnitude)
    for cylinder in range(4):
        entry = {"cylinder": cylinder + 1}
        for label, sign in (("plus", 1.0), ("minus", -1.0)):
            raw.reset(seed=int(a.seed))
            for _ in range(int(a.settle_steps)):
                raw.step(torch.zeros((raw.num_envs, 4), device=raw.device))
            start = raw._refresh_cylinder_state()[0].detach().clone()
            action = torch.zeros((raw.num_envs, 4), device=raw.device)
            action[:, cylinder] = sign * magnitude
            for _ in range(steps):
                raw.step(action)
            end = raw._refresh_cylinder_state()[0].detach().clone()
            delta = (end - start)[:, cylinder]
            entry[f"{label}_mean_delta_m"] = round(float(delta.mean()), 6)
            # Sign agreement: did it move the way it was told, per environment?
            entry[f"{label}_correct_sign_fraction"] = round(
                float(((delta * sign) > 0).float().mean()), 4
            )
        moved = (
            abs(entry["plus_mean_delta_m"]) > 1.0e-4
            and abs(entry["minus_mean_delta_m"]) > 1.0e-4
        )
        # NOTE on sign: a +action produces NEGATIVE extension on this plant, so
        # correct_sign_fraction reads ~0.00 for every cylinder. Measured
        # identically through ROM-A (PPO3), so it is the shared action
        # convention, not an ID/FD inversion. What matters for rank is that the
        # two directions produce OPPOSITE motion, which is what is asserted.
        entry["bidirectional_authority"] = bool(
            moved
            and entry["plus_mean_delta_m"] * entry["minus_mean_delta_m"] < 0.0
        )
        rows.append(entry)
    return {
        "per_cylinder": rows,
        "all_cylinders_bidirectional": all(r["bidirectional_authority"] for r in rows),
    }


def main() -> None:
    env, raw = build_env()

    authority = measure_authority(raw)

    # --- teacher rollout -------------------------------------------------
    raw.reset(seed=int(a.seed))
    cut, lateral, contact, force, align, accel, vz = [], [], [], [], [], [], []
    warm = []
    for step in range(int(a.steps)):
        # The privileged teacher is queried through the env's own contract, so
        # this measures the same controller PPO3 distils from.
        action = raw._compute_teacher_action()
        raw.step(action[:, :4])
        cut.append(raw.cut_height_error().abs().clone())
        lateral.append(raw.verge_lateral_error().abs().clone())
        contact.append(raw.cutter_ground_contact().float().clone())
        force.append(raw.cutter_terrain_contact_force_n().clone())
        align.append(raw._measure_head_down_alignment().clone())
        accel.append(
            torch.linalg.norm(raw._measured_tool_acceleration_truck, dim=-1).clone()
        )
        vz.append(raw._measured_tool_velocity_truck[:, 2].clone())
        warm.append(
            ((raw._preposition_counter > 0) | (raw._settle_counter > 0)).clone()
        )

    start = int(a.settle_steps)
    valid = ~torch.stack(warm)[start:]
    cut_t = torch.stack(cut)[start:][valid]
    lat_t = torch.stack(lateral)[start:][valid]
    con_t = torch.stack(contact)[start:][valid]
    frc_t = torch.stack(force)[start:][valid]
    ali_t = torch.stack(align)[start:][valid]
    acc_t = torch.stack(accel)[start:][valid]

    contact_frac = float(con_t.mean()) if con_t.numel() else 0.0
    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "task": a.task,
        "task_id": TASK_ID,
        "actuator": "ID->FD" if a.task == "ppo4" else "command ROM (ROM-A)",
        "cylinder_dynamics_model": str(raw.cfg.cylinder_dynamics_model),
        "arm_stiffness_scale": float(raw.cfg.terrain_arm_stiffness_scale),
        "stage": int(a.stage),
        "terrain": "held_out" if a.eval_terrain else "training",
        "num_envs": int(raw.num_envs),
        "steps": int(a.steps),
        "scored_samples": int(cut_t.numel()),
        "authority": authority,
        "teacher": {
            "abs_cut_height_error_m": stats(cut_t),
            "abs_lateral_error_m": stats(lat_t),
            "ground_contact_fraction": round(contact_frac, 5),
            "cutter_contact_force_n": stats(frc_t),
            "head_down_alignment": stats(ali_t),
            "tool_acceleration_mps2": stats(acc_t),
        },
    }

    # The question this probe answers is NOT "is the teacher good enough to
    # ship" -- it is "does swapping ROM-A for ID/FD cost anything". Those need
    # different thresholds, and conflating them is how a gate ends up rejecting
    # a healthy plant.
    #
    # Measured 2026-07-31 on identical terrain and seed: the teacher reaches
    # ground_contact 0.8707 through ID/FD and 0.8713 through ROM-A. An absolute
    # 0.90 bar (the STUDENT's gate) therefore fails BOTH actuators and says
    # nothing about either. The teacher is a privileged search controller, not a
    # trained policy, and 0.87 is simply what it does on the rough verge.
    #
    # So the actuator verdict is relative: pass --reference_report with the
    # ROM-A run and the swap is judged on the difference. The absolute checks
    # below are kept only as sanity floors, set from what the teacher actually
    # achieves rather than from the student's gate.
    reference = None
    if a.reference_report:
        reference = json.loads(Path(a.reference_report).read_text())

    verdict = {
        "authority_ok": authority["all_cylinders_bidirectional"],
        "mean_cut_error_ok": bool(report["teacher"]["abs_cut_height_error_m"].get("mean", 9e9) <= 0.040),
        "mean_lateral_ok": bool(report["teacher"]["abs_lateral_error_m"].get("mean", 9e9) <= 0.30),
        # Sanity floor, not the student's 0.90 gate. Well below the measured
        # 0.87 so a genuinely broken actuator still trips it.
        "contact_floor_ok": bool(contact_frac >= 0.80),
    }
    if reference is not None:
        ref = reference["teacher"]
        report["reference"] = {
            "task": reference.get("task"),
            "actuator": reference.get("actuator"),
            "ground_contact_fraction": ref["ground_contact_fraction"],
            "abs_cut_height_error_m_mean": ref["abs_cut_height_error_m"]["mean"],
            "abs_lateral_error_m_mean": ref["abs_lateral_error_m"]["mean"],
        }
        # Tolerances are "not materially worse than the actuator we are
        # replacing", which is the only claim this swap needs to support.
        verdict["contact_not_worse_than_reference"] = bool(
            contact_frac >= ref["ground_contact_fraction"] - 0.03
        )
        verdict["cut_error_not_worse_than_reference"] = bool(
            report["teacher"]["abs_cut_height_error_m"]["mean"]
            <= ref["abs_cut_height_error_m"]["mean"] + 0.010
        )
        verdict["lateral_not_worse_than_reference"] = bool(
            report["teacher"]["abs_lateral_error_m"]["mean"]
            <= ref["abs_lateral_error_m"]["mean"] + 0.030
        )

    report["verdict"] = verdict
    report["passed"] = all(verdict.values())
    report["verdict_is_relative"] = reference is not None

    print("[PPO4-TEACHER] " + json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        print(
            "[PPO4-TEACHER] TEACHER GATE FAILED. Do not start PPO4 training on "
            "this: a student cannot distil a behaviour its teacher cannot "
            "produce, and the resulting numbers would describe the actuator, "
            "not the policy.",
            flush=True,
        )
    if a.report:
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(json.dumps(report, indent=2))

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
