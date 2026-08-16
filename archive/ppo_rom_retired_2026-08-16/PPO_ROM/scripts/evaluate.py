#!/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
"""Deterministic held-out-terrain evaluation for the PPO3-1 policy.

Builds the PPO3-1 task (inherited PPO3 verge task + c4 authority), loads a
checkpoint, runs an actor-mean rollout and scores the same shared 17 gates as
PPO3/PPO4 (``summarize_convergence``). Optional live tracking plot.
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
p.add_argument("checkpoint", nargs="?", default=None)
p.add_argument("--num_envs", type=int, default=64)
p.add_argument("--seed", type=int, default=17)
p.add_argument("--steps", type=int, default=960)
p.add_argument("--stage", type=int, default=5)
p.add_argument("--eval_terrain", action="store_true")
p.add_argument(
    "--eval_terrain_seed",
    type=int,
    default=0,
    help="Move the held-out mesh off its pinned seed 1337. 0 keeps the gate mesh.",
)
p.add_argument(
    "--path_family",
    default="mixed",
    choices=(
        "mixed",
        "straight",
        "lateral_sine",
        "s_curve",
        "random_spline",
        "offset_return",
        "boom_sweep",
        "lateral_chirp",
        "double_s_curve",
        "lane_change_hold",
        "gaussian_slalom",
        "fourier_meander",
    ),
)
p.add_argument("--terrain_height_scale", type=float, default=0.0)
p.add_argument("--plot_tracking", action="store_true")
p.add_argument("--plot_update_interval", type=int, default=15)
p.add_argument("--plot_out", default="")
p.add_argument("--zero_action", action="store_true")
p.add_argument("--report", default="")
p.add_argument(
    "--drive_substep_interpolation",
    default="",
    choices=("", "true", "false"),
)
p.add_argument(
    "--head_termination_alignment",
    type=float,
    default=None,
    help="Override the head-down safety threshold (e.g. -1 to measure uncensored).",
)
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

import PPO_ROM  # noqa: F401,E402
from PPO_ROM.terrain_env_cfg import (  # noqa: E402
    TerrainC4AuthorityPPORunnerCfg,
    apply_c4_authority_stage,
)
from verge_task.terrain_env_cfg import (  # noqa: E402
    configure_terrain_gpu_physx_memory,
)
from verge_task.terrain_observations import (  # noqa: E402
    observation_space_for_cfg,
)
from verge_task.terrain_convergence import (  # noqa: E402
    GATE_SPECS,
    summarize_convergence,
)
from verge_task.actuators import assert_checkpoint_actuator  # noqa: E402


class LiveTrackingPlot:
    """Plan-view XY plot of environment zero's desired and actual paths."""

    def __init__(self, *, family: str, update_interval: int, output: Path):
        import matplotlib

        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt

        self.plt = plt
        self.family = family
        self.update_interval = int(update_interval)
        self.output = output
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.target_x_m: list[float] = []
        self.target_y_m: list[float] = []
        self.cutter_x_m: list[float] = []
        self.cutter_y_m: list[float] = []

        plt.ion()
        self.figure, self.axis = plt.subplots(figsize=(9, 6))
        self.figure.canvas.manager.set_window_title(
            f"PPO3-1 tracking — {family}"
        )
        (self.target_line,) = self.axis.plot(
            [], [], "#d62728", linewidth=2.0, label="desired XY path"
        )
        (self.actual_line,) = self.axis.plot(
            [], [], "#1f77b4", linewidth=1.5, label="actual cutter XY trajectory"
        )
        self.axis.set_xlabel("x [m]")
        self.axis.set_ylabel("y [m]")
        self.axis.set_aspect("equal", adjustable="datalim")
        self.axis.grid(True, alpha=0.3)
        self.axis.legend(loc="best")
        self.axis.set_title(f"PPO3-1 path tracking: {family}")
        self.figure.tight_layout()

    @torch.no_grad()
    def sample(self, raw, step: int) -> None:
        cutter_world = raw._compute_cutter_pos_w()[0]
        cutter_x = float(cutter_world[0] - raw.scene.env_origins[0, 0])
        cutter_y = float(cutter_world[1] - raw.scene.env_origins[0, 1])
        self.target_x_m.append(cutter_x)
        self.target_y_m.append(float(raw.tracking_line_y_local()[0]))
        self.cutter_x_m.append(cutter_x)
        self.cutter_y_m.append(cutter_y)
        if step % self.update_interval == 0:
            self.refresh()

    def refresh(self) -> None:
        self.target_line.set_data(self.target_x_m, self.target_y_m)
        self.actual_line.set_data(self.cutter_x_m, self.cutter_y_m)
        self.axis.relim()
        self.axis.autoscale_view()
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()

    def finalize(self) -> None:
        self.refresh()
        self.figure.savefig(self.output, dpi=180, bbox_inches="tight")
        print(f"[PPO3-1-PLOT] wrote {self.output}", flush=True)


@torch.no_grad()
def evaluate_verge(raw, policy, steps: int, tracking_plot=None) -> dict:
    """Actor-mean rollout collecting every metric the shared gates read."""
    obs, _ = raw.reset(seed=int(a.seed))
    errors, accelerations, alignments = [], [], []
    laterals, scan_validity, contacts = [], [], []
    wheel_forces = []
    truck_heading_errors = []
    signed_laterals = []
    chassis_lateral_reactions = []
    chassis_attitude_reactions = []
    actions, applied_actions, shield_gains = [], [], []
    safety_terminations = 0
    for step in range(steps):
        action = policy(obs)
        if a.zero_action:
            action = torch.zeros_like(action)
        obs, _, terminated, _, _ = raw.step(action)
        errors.append(raw.cut_height_error().abs())
        accelerations.append(
            torch.linalg.norm(raw._measured_tool_acceleration_truck, dim=-1)
        )
        alignments.append(-raw._head_down_axis_world()[:, 2])
        signed_lateral = raw.verge_lateral_error()
        signed_laterals.append(signed_lateral)
        laterals.append(signed_lateral.abs())
        scan_validity.append(raw.scan_valid_fraction())
        contacts.append(raw.cutter_ground_contact().float())
        wheel_forces.append(raw.wheel_road_contact_forces_n())
        truck_heading_errors.append(raw._truck_heading_error_rad)
        chassis_lateral_reactions.append(
            raw._chassis_lateral_reaction_m.detach().clone()
        )
        chassis_attitude_reactions.append(
            raw._chassis_attitude_reaction_rad.detach().clone()
        )
        actions.append(action)
        applied_actions.append(raw._sole_applied.detach().clone())
        shield_gains.append(raw._sole_shield_gain.detach().clone())
        safety_terminations += int(raw.terrain_safety_violation().sum())
        if tracking_plot is not None:
            tracking_plot.sample(raw, step)

    error_steps = torch.stack(errors)
    lateral_steps = torch.stack(laterals)
    contact_steps = torch.stack(contacts)
    error = error_steps.reshape(-1)
    row = {
        "terrain": "held_out" if a.eval_terrain else "training",
        "steps": int(steps),
        "num_envs": int(raw.num_envs),
        "mean_abs_cut_height_error_m": float(error.mean()),
        "p95_abs_cut_height_error_m": float(torch.quantile(error, 0.95)),
        "p95_tool_acceleration_mps2": float(
            torch.quantile(torch.cat(accelerations), 0.95)
        ),
        "min_head_alignment": float(torch.cat(alignments).min()),
        "p01_head_alignment": float(torch.quantile(torch.cat(alignments), 0.01)),
        "mean_abs_lateral_error_m": float(lateral_steps.mean()),
        "mean_signed_lateral_error_m": float(torch.stack(signed_laterals).mean()),
        "scan_valid_fraction": float(torch.cat(scan_validity).mean()),
        "ground_contact_fraction": float(contact_steps.mean()),
        "min_wheel_road_contact_duty_cycle": float(
            (
                torch.stack(wheel_forces)
                > float(getattr(raw.cfg, "road_contact_force_threshold_n", 10.0))
            ).float().mean(dim=(0, 1)).min()
        ),
        "max_truck_post_constraint_drift_m": float(
            raw._max_truck_post_constraint_drift.max()
        ),
        "max_truck_heading_error_deg": float(
            torch.rad2deg(torch.cat(truck_heading_errors).max())
        ),
        "p95_chassis_lateral_reaction_mm": 1000.0
        * float(torch.quantile(torch.stack(chassis_lateral_reactions), 0.95)),
        "p95_chassis_attitude_reaction_deg": float(
            torch.rad2deg(
                torch.quantile(torch.stack(chassis_attitude_reactions), 0.95)
            )
        ),
        "safety_termination_rate": safety_terminations / max(steps * raw.num_envs, 1),
        "mean_requested_policy_action": torch.stack(actions)
        .mean(dim=(0, 1))
        .tolist(),
        "mean_abs_applied_action": torch.stack(applied_actions)
        .abs()
        .mean(dim=(0, 1))
        .tolist(),
        "shield_active_fraction": float(
            (torch.stack(shield_gains) < 0.999).float().mean()
        ),
        "teacher_mode": str(raw.cfg.sole_teacher_mode),
        "direct_target_contact_demo": bool(
            raw.cfg.enable_contact_tracking_demo
        ),
    }
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
                "num_envs": int(family_mask.sum()),
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
    row["path_family_results"] = family_rows
    row["worst_family_mean_abs_cut_height_error_m"] = max(
        item["mean_abs_cut_height_error_m"] for item in family_rows
    )
    row["worst_family_mean_abs_lateral_error_m"] = max(
        item["mean_abs_lateral_error_m"] for item in family_rows
    )
    row["worst_family_ground_contact_fraction"] = min(
        item["ground_contact_fraction"] for item in family_rows
    )
    if tracking_plot is not None:
        tracking_plot.finalize()
    return row


def main() -> None:
    if a.checkpoint is None:
        raise SystemExit("evaluate.py requires a checkpoint path")
    env_cfg = parse_env_cfg(
        PPO_ROM.TERRAIN_TASK_ID,
        device=a.device,
        num_envs=a.num_envs,
    )
    apply_c4_authority_stage(env_cfg, int(a.stage))
    env_cfg.observation_space = observation_space_for_cfg(env_cfg)
    env_cfg.use_eval_terrain = bool(a.eval_terrain)
    env_cfg.terrain_base_seed = int(a.seed)
    env_cfg.tracking_path_family = str(a.path_family)
    env_cfg.enable_contact_tracking_demo = False
    env_cfg.enable_visual_grid_floor = False
    env_cfg.enable_road_scenery = False
    if a.eval_terrain_seed != 0:
        env_cfg.terrain_eval_seed_override = int(a.eval_terrain_seed)
    if a.terrain_height_scale > 0.0:
        env_cfg.terrain_height_scales = (float(a.terrain_height_scale),)
    if a.head_termination_alignment is not None:
        env_cfg.head_down_termination_alignment = float(
            a.head_termination_alignment
        )
    else:
        env_cfg.head_down_termination_alignment = float(
            GATE_SPECS["head_alignment_hard_min"].threshold
        )
    if a.drive_substep_interpolation:
        env_cfg.terrain_kinematic_drive_substep_interpolation = (
            a.drive_substep_interpolation == "true"
        )
    env_cfg.randomize_start_phase = False
    env_cfg.randomize_trajectory_shape = False
    env_cfg.trajectory_shape = str(env_cfg.terrain_deployment_trajectory_shape)
    configure_terrain_gpu_physx_memory(env_cfg, int(a.num_envs))
    env = gym.make(
        PPO_ROM.TERRAIN_TASK_ID, cfg=env_cfg, render_mode=None
    )
    raw = env.unwrapped

    runner_cfg = TerrainC4AuthorityPPORunnerCfg()
    runner_cfg.seed = int(a.seed)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)
    runner = OnPolicyRunner(
        wrapped, runner_cfg.to_dict(), log_dir=None, device=a.device
    )
    checkpoint_path = Path(a.checkpoint).resolve()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    infos = payload.get("infos") or {}
    source = str(infos.get("source", ""))
    schema = str(infos.get("schema", ""))
    checkpoint_width = int(infos.get("observation_width", -1))
    checkpoint_action_width = int(infos.get("action_width", 4))
    if schema not in {"terrain_teacher_student_v1", "terrain_shared_v2"} or source not in {
        "PPO3 terrain teacher distillation v1",
        "PPO3 terrain PPO",
        "PPO3-1 terrain PPO",
    }:
        raise RuntimeError(
            "evaluation accepts only supported terrain policy checkpoints; "
            f"got source={source!r}, schema={schema!r}"
        )
    if checkpoint_width != int(raw.cfg.observation_space):
        raise RuntimeError(
            f"checkpoint observation width {checkpoint_width} != "
            f"environment width {raw.cfg.observation_space}"
        )
    if checkpoint_action_width != int(raw.cfg.action_space):
        raise RuntimeError(
            f"checkpoint action width {checkpoint_action_width} != "
            f"environment width {raw.cfg.action_space}"
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
    assert_checkpoint_actuator(infos, "rom", context="PPO3-1 evaluation")
    policy = runner.get_inference_policy(device=a.device)

    tracking_plot = None
    if a.plot_tracking:
        plot_path = (
            Path(a.plot_out).expanduser().resolve()
            if a.plot_out
            else ROOT
            / "PPO_ROM"
            / "plots"
            / f"{a.path_family}_tracking.png"
        )
        tracking_plot = LiveTrackingPlot(
            family=str(a.path_family),
            update_interval=int(a.plot_update_interval),
            output=plot_path,
        )

    row = evaluate_verge(
        raw,
        policy,
        int(a.steps),
        tracking_plot=tracking_plot,
    )
    summary = summarize_convergence(row)
    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "checkpoint": str(checkpoint_path),
        "stage": int(a.stage),
        "actuator": "rom",
        "action_width": int(raw.cfg.action_space),
        "head_joint_action": bool(raw.cfg.sole_head_joint_action),
        "c4_authority": raw.c4_authority_report(),
        "drive_substep_interpolation": bool(
            raw.cfg.terrain_kinematic_drive_substep_interpolation
        ),
        "mow_strip_width_m": float(raw.cfg.mow_strip_width_m),
        "tracking_amplitude_scale": float(
            raw.cfg.terrain_tracking_amplitude_scale
        ),
        "terrain_provenance": {
            "split": "held_out_validation" if a.eval_terrain else "training",
            "base_seed": int(raw._mow_strip_cfg().seed),
            "path_family": str(a.path_family),
            "path_families_realized": sorted(
                set(raw._tracking_path_family_names)
            ),
        },
        "verge": row,
        "gates": summary["gates"],
        "normalized_gate_violations": summary["normalized_gate_violations"],
        "score": summary["score"],
        "passed": summary["passed"],
    }
    printable = {
        "verge": row,
        "gates": report["gates"],
        "score": report["score"],
        "passed": report["passed"],
    }
    print(json.dumps(printable, indent=2), flush=True)

    if a.report:
        path = Path(a.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2))
        print(f"[PPO3-1-SOLE] wrote {path}", flush=True)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
