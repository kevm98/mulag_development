#!/usr/bin/env python3
"""Decompose ``p95_tool_acceleration_mps2`` into what a policy can and cannot control.

``p95_tool_acceleration`` is the dominant PPO3 stage-5 gate: at the block-1
checkpoint of ``lawn_rough_stage5_v3`` it accounted for 0.6367 of a 0.6586 total
normalized violation, and it has risen above its starting value in every reward
configuration tried so far (148.7, 136.6 and 130.9 m/s^2 after one block, against
125.9 for the checkpoint each block resumed from). Strengthening the penalty
5x slowed that climb without reversing it, which is the signature of a metric the
policy does not fully author.

The gate metric is a second finite difference of the measured cutter position in
the truck frame, reduced with a bare ``torch.quantile`` over every (step, env)
sample. It therefore sums four contributions that a single number cannot separate:

1. commanded arm motion -- the only part PPO can shape;
2. cutter/verge contact impulses -- ``terrain_convergence.GATE_SPECS`` already
   relaxed this gate 40 -> 80 m/s^2 because of them;
3. the truck driving over rough ground, which moves the tool frame regardless of
   what the arm does;
4. finite-difference transients on the step after a reset, where the measured
   position jumps discontinuously.

Only (1) responds to training. This probe measures the split with four
independent tests, and reports the sample count behind every statistic so that
an empty or tiny subset can never be mistaken for a passing result -- the failure
mode this repository has hit repeatedly with vacuous gates.

Run it twice: once with ``--checkpoint`` for the policy, and once without for the
zero-action floor. The floor is the decisive number. It is the acceleration that
remains when the arm is commanded to do nothing, so no policy under this action
contract can go below it. If the floor already exceeds 80 m/s^2, the gate is
unreachable by training and the finding is a gate-definition problem.
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
p.add_argument("--num_envs", type=int, default=128)
p.add_argument("--seed", type=int, default=17)
p.add_argument("--steps", type=int, default=960)
p.add_argument("--stage", type=int, default=5)
p.add_argument(
    "--checkpoint",
    default="",
    help="Policy to probe. Omitted means zero action: the physical floor.",
)
p.add_argument("--eval_terrain", action="store_true")
p.add_argument("--path_family", default="mixed")
p.add_argument("--trajectory_library_path", default="")
p.add_argument("--command_rom_checkpoint", default="")
p.add_argument(
    "--reset_mask_steps",
    type=int,
    default=3,
    help=(
        "Samples this many steps after a reset are flagged as post-reset. They "
        "are reported both included and excluded; the gate itself includes them."
    ),
)
p.add_argument("--report_out", default="")
AppLauncher.add_app_launcher_args(p)
a, unknown = p.parse_known_args()
sys.argv = [sys.argv[0], *unknown]

if a.num_envs <= 0 or a.steps <= 0:
    p.error("--num_envs and --steps must be positive")
if not 1 <= a.stage <= 5:
    p.error("--stage must be between 1 and 5")
if a.reset_mask_steps < 0:
    p.error("--reset_mask_steps cannot be negative")

launcher = AppLauncher(a)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rrlab_compat.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import verge_task  # noqa: F401,E402
from verge_task.distill import assert_teacher_disabled  # noqa: E402
from verge_task.terrain_env_cfg import (  # noqa: E402
    TerrainSolePolicyPPORunnerCfg,
    apply_terrain_curriculum_stage,
    configure_terrain_gpu_physx_memory,
)
from verge_task.terrain_observations import (  # noqa: E402
    observation_space_for_cfg,
)
from verge_task.terrain_convergence import GATE_SPECS  # noqa: E402

GATE_MPS2 = float(GATE_SPECS["p95_tool_acceleration"].threshold)
# Below this many samples a quantile is not a measurement. Reported as null with
# the count attached rather than as a number that looks like evidence.
MIN_SUBSET = 64


def build_env():
    """Mirror ``evaluate.py``'s environment exactly.

    The probe is only trustworthy if its unconditional p95 reproduces the gate
    number, so every field that evaluation sets is set identically here.
    """
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
    env_cfg.defer_safety_terminations = True
    env_cfg.head_down_termination_alignment = float(
        GATE_SPECS["head_alignment_hard_min"].threshold
    )
    env_cfg.randomize_start_phase = False
    env_cfg.randomize_trajectory_shape = False
    env_cfg.trajectory_shape = str(env_cfg.terrain_deployment_trajectory_shape)
    if a.trajectory_library_path:
        env_cfg.trajectory_library_path = str(
            Path(a.trajectory_library_path).resolve()
        )
    if a.command_rom_checkpoint:
        env_cfg.command_rom_checkpoint = str(
            Path(a.command_rom_checkpoint).resolve()
        )
    configure_terrain_gpu_physx_memory(env_cfg, int(a.num_envs))
    assert_teacher_disabled(env_cfg, context="PPO3 tool-acceleration probe")
    env = gym.make(verge_task.TERRAIN_TASK_ID, cfg=env_cfg, render_mode=None)
    return env, env.unwrapped


def subset_stats(values: torch.Tensor, name: str) -> dict:
    """Quantiles plus the sample count that produced them."""
    count = int(values.numel())
    stats = {"name": name, "samples": count}
    if count < MIN_SUBSET:
        stats.update(
            {
                "p95_mps2": None,
                "mean_mps2": None,
                "max_mps2": None,
                "note": f"fewer than {MIN_SUBSET} samples; no quantile reported",
            }
        )
        return stats
    stats.update(
        {
            "p95_mps2": float(torch.quantile(values, 0.95)),
            "mean_mps2": float(values.mean()),
            "max_mps2": float(values.max()),
            "exceeds_gate_fraction": float((values > GATE_MPS2).float().mean()),
        }
    )
    return stats


def explained_variance(target: torch.Tensor, features: torch.Tensor) -> float:
    """R^2 of an ordinary least-squares fit, with an intercept column."""
    x = torch.cat(
        [torch.ones(features.shape[0], 1, dtype=torch.float64), features], dim=1
    )
    y = target.reshape(-1, 1)
    solution = torch.linalg.lstsq(x, y).solution
    residual = y - x @ solution
    total = ((y - y.mean()) ** 2).sum()
    if float(total) <= 0.0:
        return 0.0
    return float(1.0 - (residual**2).sum() / total)


def main() -> int:
    env, raw = build_env()
    policy = None
    checkpoint_contract = None
    if a.checkpoint:
        runner_cfg = TerrainSolePolicyPPORunnerCfg()
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
        width = int(infos.get("observation_width", -1))
        if schema != "terrain_teacher_student_v1" or source not in {
            "PPO3 terrain teacher distillation v1",
            "PPO3 terrain PPO",
        }:
            raise RuntimeError(
                "probe accepts only terrain_teacher_student_v1 checkpoints; "
                f"got source={source!r}, schema={schema!r}"
            )
        if width != int(raw.cfg.observation_space):
            raise RuntimeError(
                f"checkpoint observation width {width} != "
                f"environment width {raw.cfg.observation_space}"
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
        checkpoint_contract = {
            "path": str(checkpoint_path),
            "source": source,
            "schema": schema,
            "observation_width": width,
        }

    obs, _ = raw.reset(seed=int(a.seed))
    zero_action = torch.zeros(
        (raw.num_envs, int(raw.cfg.action_space)),
        device=raw.device,
        dtype=torch.float32,
    )

    accelerations, contact_forces, in_contacts = [], [], []
    applied_actions, post_reset_flags = [], []
    # Stroke telemetry. The stage-5 collapses coincide exactly with cylinder
    # headroom pinning at zero (lawn_rough_stage5_v5 block 1, iters 742-751 and
    # 769-781: min low headroom 0.0000 sustained, recovering only at reset),
    # while the head-down latch release never fires -- the head cannot recover
    # because there is no travel left. Recording headroom per cylinder here
    # answers the question a single aggregate cannot: whether the environment
    # alone exhausts the stroke, or whether the policy drives itself into it.
    headroom_lows, headroom_highs, alignments = [], [], []
    # Steps since this environment last reset. Starts inside the mask because
    # step 0 differences against the reset pose.
    steps_since_reset = torch.zeros(raw.num_envs, device=raw.device)

    for _ in range(int(a.steps)):
        action = zero_action if policy is None else policy(obs)
        obs, _, terminated, truncated, _ = raw.step(action)

        accelerations.append(
            torch.linalg.norm(
                raw._measured_tool_acceleration_truck.detach(), dim=-1
            ).clone()
        )
        contact_forces.append(raw.cutter_terrain_contact_force_n().detach().clone())
        in_contacts.append(raw.cutter_ground_contact().detach().float().clone())
        applied_actions.append(raw._sole_applied.detach().clone())
        post_reset_flags.append(
            (steps_since_reset < float(a.reset_mask_steps)).float().clone()
        )
        measured, _ = raw._refresh_cylinder_state()
        low, high = raw._cylinder_headroom(measured)
        headroom_lows.append(low.detach().clone())
        headroom_highs.append(high.detach().clone())
        alignments.append(raw._measure_head_down_alignment().detach().clone())

        done = (terminated | truncated).detach()
        steps_since_reset = torch.where(
            done, torch.zeros_like(steps_since_reset), steps_since_reset + 1.0
        )

    accel = torch.stack(accelerations).flatten().double().cpu()
    force = torch.stack(contact_forces).flatten().double().cpu()
    contact = torch.stack(in_contacts).flatten().double().cpu()
    post_reset = torch.stack(post_reset_flags).flatten().double().cpu()

    applied = torch.stack(applied_actions).double().cpu()
    rate = torch.zeros_like(applied)
    rate[1:] = applied[1:] - applied[:-1]
    applied_flat = applied.reshape(-1, applied.shape[-1])
    rate_flat = rate.reshape(-1, rate.shape[-1])

    force_steps = torch.stack(contact_forces).double().cpu()
    force_delta = torch.zeros_like(force_steps)
    force_delta[1:] = force_steps[1:] - force_steps[:-1]
    force_delta_flat = force_delta.flatten().abs()

    clean = post_reset <= 0.0
    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "mode": "policy" if policy is not None else "zero_action_floor",
        "checkpoint_contract": checkpoint_contract,
        "stage": int(a.stage),
        "terrain": "held_out" if a.eval_terrain else "training",
        "num_envs": int(raw.num_envs),
        "steps": int(a.steps),
        "gate_mps2": GATE_MPS2,
        "total_samples": int(accel.numel()),
    }

    # Test 0: reproduce the gate reduction exactly, then show what the unmasked
    # post-reset transients contribute to it.
    report["gate_reduction"] = {
        "all_samples": subset_stats(accel, "all samples (this is the gate)"),
        "excluding_post_reset": subset_stats(accel[clean], "post-reset excluded"),
        "post_reset_only": subset_stats(accel[~clean], "post-reset only"),
        "post_reset_sample_fraction": float((~clean).double().mean()),
        "reset_mask_steps": int(a.reset_mask_steps),
    }

    # Test 1: contact-conditioned split on the clean samples. A free-swing p95
    # already above the gate means commands alone breach it.
    free = clean & (contact <= 0.0)
    touching = clean & (contact > 0.0)
    report["contact_split"] = {
        "free_swing": subset_stats(accel[free], "cutter not in contact"),
        "in_contact": subset_stats(accel[touching], "cutter in contact"),
        "contact_sample_fraction": float(contact[clean].mean())
        if int(clean.sum()) > 0
        else None,
    }

    # Test 2: how much of the worst tail coincides with contact events.
    if int(clean.sum()) >= MIN_SUBSET:
        clean_accel = accel[clean]
        tail = clean_accel >= torch.quantile(clean_accel, 0.95)
        force_delta_clean = force_delta_flat[clean]
        spike = force_delta_clean >= torch.quantile(force_delta_clean, 0.95)
        report["tail_coincidence"] = {
            "tail_samples": int(tail.sum()),
            "tail_in_contact_fraction": float(contact[clean][tail].mean()),
            "baseline_in_contact_fraction": float(contact[clean].mean()),
            "tail_force_spike_fraction": float(spike[tail].double().mean()),
            "baseline_force_spike_fraction": float(spike.double().mean()),
        }
    else:
        report["tail_coincidence"] = {
            "note": f"fewer than {MIN_SUBSET} clean samples",
            "clean_samples": int(clean.sum()),
        }

    # Test 3: variance of acceleration explained by the command stream alone,
    # then the increment from adding the contact-force change. The command R^2
    # is an upper bound on what any amount of further training can address.
    if int(clean.sum()) >= MIN_SUBSET:
        command_features = torch.cat(
            [rate_flat[clean].abs(), applied_flat[clean].abs()], dim=1
        )
        contact_feature = force_delta_flat[clean].reshape(-1, 1)
        command_r2 = explained_variance(accel[clean], command_features)
        combined_r2 = explained_variance(
            accel[clean], torch.cat([command_features, contact_feature], dim=1)
        )
        report["explained_variance"] = {
            "command_only_r2": command_r2,
            "command_plus_contact_r2": combined_r2,
            "contact_increment_r2": combined_r2 - command_r2,
            "samples": int(clean.sum()),
            "note": (
                "command features are |applied action rate| and |applied action| "
                "per cylinder; contact feature is |delta cutter contact force|"
            ),
        }
    else:
        report["explained_variance"] = {
            "note": f"fewer than {MIN_SUBSET} clean samples",
            "clean_samples": int(clean.sum()),
        }

    # Stroke exhaustion. `pinned` is a per-(step, env, cylinder) fraction, not a
    # minimum, so one unlucky environment cannot manufacture the finding the way
    # `cylinder/min_low_headroom_m` can.
    low_steps = torch.stack(headroom_lows).double().cpu()
    high_steps = torch.stack(headroom_highs).double().cpu()
    align_steps = torch.stack(alignments).double().cpu().flatten()
    pin_m = 1.0e-4
    low_pinned = (low_steps <= pin_m).reshape(-1, low_steps.shape[-1])
    high_pinned = (high_steps <= pin_m).reshape(-1, high_steps.shape[-1])
    any_pinned = (low_pinned | high_pinned).any(dim=1)
    report["stroke"] = {
        "pin_threshold_m": pin_m,
        "any_cylinder_pinned_fraction": float(any_pinned.double().mean()),
        "low_pinned_fraction_per_cylinder": [
            float(low_pinned[:, i].double().mean()) for i in range(low_pinned.shape[1])
        ],
        "high_pinned_fraction_per_cylinder": [
            float(high_pinned[:, i].double().mean())
            for i in range(high_pinned.shape[1])
        ],
        "min_low_headroom_m_per_cylinder": [
            float(low_steps[..., i].min()) for i in range(low_steps.shape[-1])
        ],
        "p01_low_headroom_m_per_cylinder": [
            float(torch.quantile(low_steps[..., i].flatten(), 0.01))
            for i in range(low_steps.shape[-1])
        ],
        "median_low_headroom_m_per_cylinder": [
            float(low_steps[..., i].median()) for i in range(low_steps.shape[-1])
        ],
        "head_alignment_p01": float(torch.quantile(align_steps, 0.01)),
        "head_alignment_min": float(align_steps.min()),
        "below_latch_fraction": float((align_steps < 0.95).double().mean()),
        "pinned_and_below_latch_fraction": float(
            (any_pinned & (align_steps < 0.95)).double().mean()
        ),
        "note": (
            "zero_action_floor mode measures stroke the environment consumes on "
            "its own; a policy run's excess over that is self-inflicted"
        ),
    }

    out = (
        Path(a.report_out).expanduser().resolve()
        if a.report_out
        else ROOT
        / "verge_task"
        / "reports"
        / f"tool_acceleration_{report['mode']}_{a.stage}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print(f"[PPO3-ACCEL-PROBE] wrote {out}", flush=True)

    env.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
