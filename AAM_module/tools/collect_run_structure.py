#!/usr/bin/env python
"""Collect master raw runs driven by the Stage 13 run structure (DLS + PRBS).

This is the collection tool for the final manifest (Stage 15): each run is
the 11-segment Stage 13 plan (`aam_data.run_structure.build_run_plan`, by
default 11304 control steps / 188.4 s) executed on the deployed DLS teacher
chain over the same command ROM the Stage 8 probe collector uses.

For every selected manifest row the tool:

  1. resets the headless CylinderSpace env and applies the fixed factor draw
     (hydraulic via ROM `system_factors`, mechanical via PhysX mass/COM on
     the cutter body), exactly like `collect_master_dataset.py`;
  2. deploys to posture A with a cosine ramp (DirectCylinderDriver, no
     impulse) and syncs the env internals (cylinder target, ROM reset);
  3. steps the Stage 13 plan segment by segment:
       - DLS segments (settle, posture ramps, workspace / coupled /
         reach-and-return trajectories, stow) are driven by the verified
         cylinder DLS teacher (Stage 9/11): the controller closes the loop on
         the measured cylinder state toward the plan's seed reference and
         writes a normalized 4-D action onto the action port;
       - identification segments carry the bounded PRBS probe from the plan
         (Stage 12/13 amplitude) directly on the action port;
  4. records the full Stage 8 master schema signals (state T+1, command T,
     flag T) and writes `metadata.json` + `trajectory.npz`, validated against
     the manifest row with `aam_data.validate_run`.

The recorded dataset is byte-compatible with the Stage 8 schema, so Stage 16
freezing and all downstream tools consume it unchanged.

Usage:
    python collect_run_structure.py --limit 1 --id-s 1.0
    python collect_run_structure.py --run-ids 0,1,2
    python collect_run_structure.py --manifest results/final_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from isaaclab.app import AppLauncher


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
AAM_ROOT = Path(__file__).resolve().parents[1]
if str(AAM_ROOT) not in sys.path:
    sys.path.insert(0, str(AAM_ROOT))


p = argparse.ArgumentParser(description=__doc__)
p.add_argument(
    "--manifest",
    default=str(AAM_ROOT / "results/final_manifest.csv"),
    help="Stage 15 final manifest (28 runs) or any master-schema manifest",
)
p.add_argument("--out", default=str(AAM_ROOT / "datasets/master_v1"))
p.add_argument("--limit", type=int, default=None, help="collect only the first N runs")
p.add_argument("--run-ids", type=str, default=None, help="comma-separated run ids")
p.add_argument(
    "--num-envs",
    type=int,
    default=1,
    help="number of manifest rows to collect concurrently in one vectorized simulator",
)
p.add_argument("--seed-base", type=int, default=20260808)
p.add_argument(
    "--matched-reset-seed",
    action="store_true",
    help="use the same reset seed for every factor configuration (common "
    "random numbers for the Stage-22 matched-excitation recollection)",
)
p.add_argument(
    "--resume",
    action="store_true",
    help="skip already present runs only after schema/manifest validation",
)
p.add_argument(
    "--max-isolated-safety-attempts",
    type=int,
    default=3,
    help="maximum env-0 attempts for a run that violates a safety envelope "
    "in a vector slot; no failed attempt is written",
)
p.add_argument("--factor-profile", default="stage5")
# Stage 13 run-structure timing (seconds); keep the defaults for the final
# manifest, shrink for quick smoke tests.
p.add_argument("--settle-s", type=float, default=2.0)
p.add_argument("--posture-ramp-s", type=float, default=3.0)
p.add_argument("--id-s", type=float, default=30.0)
p.add_argument("--stow-s", type=float, default=4.0)
p.add_argument("--probe", choices=("prbs", "multisine"), default="prbs")
p.add_argument("--amplitude", type=float, default=0.015)
p.add_argument("--probe-seed", type=int, default=20260807)
p.add_argument("--probe-hold-steps", type=int, default=10)
p.add_argument(
    "--trajectory-stride",
    type=int,
    default=1,
    help="subsample DLS reference paths for bounded test flights only; the "
    "full collection must use 1",
)
p.add_argument("--preposition-steps", type=int, default=200)
p.add_argument("--pre-settle-steps", type=int, default=30)
# DLS teacher hyperparameters (the Stage 11 pilot's verified values).
p.add_argument("--error-gain", type=float, default=0.18)
p.add_argument("--velocity-feedforward", type=float, default=1.0)
p.add_argument("--velocity-damping", type=float, default=0.0)
p.add_argument("--action-smoothing", type=float, default=0.20)
p.add_argument("--max-desired-delta-m", type=float, default=0.004)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()

launcher = AppLauncher(a)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402

import mulag_plant.tasks  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from aam_data import (  # noqa: E402
    COMMAND_SIGNALS,
    FLAG_SIGNALS,
    SCHEMA_VERSION,
    STATE_SIGNALS,
    STEPS_PER_ROM_TICK,
    UNAVAILABLE_SIGNALS,
    build_run_plan,
    e_to_mechanical,
    e_to_system_factors,
    validate_run,
    write_run,
)
from mulag_plant.dls_solver.cylinder import (  # noqa: E402
    CylinderDLSConfig,
    CylinderDLSController,
)
from mulag_plant.dls_solver.direct_drive import DirectCylinderDriver  # noqa: E402


TASK = "RRLAB-Mulag-PPO2-ROM-PhysX-CylinderSpace-v0"
CUTTER_BODY = "Messerkopf"
COLLECTION_SAFETY_LIMITS = {
    "invalid": 0.0,
    "cylinder_limit": 0.06,
    "command_saturation": 0.11,
    "valve_saturation": 0.001,
}


def load_manifest(path: Path) -> list[dict]:
    rows = list(csv.DictReader(open(path, newline="")))
    for r in rows:
        r["run_id"] = int(r["run_id"])
        for key in ("flow", "compliance", "seal", "pressure", "mass_scale", "com_z"):
            r[key] = float(r[key])
    return rows


def select_rows(rows: list[dict]) -> list[dict]:
    if a.run_ids is not None:
        wanted = {int(x) for x in a.run_ids.split(",")}
        return [r for r in rows if r["run_id"] in wanted]
    if a.limit is not None:
        return rows[: a.limit]
    return rows


def snapshot_state(env) -> dict[str, np.ndarray]:
    """Snapshot every vector environment as arrays shaped ``(N, channels)``."""
    robot = env.robot
    cyl_q = robot.data.joint_pos[:, env._cylinder_joint_ids]
    cyl_qdot = robot.data.joint_vel[:, env._cylinder_joint_ids]
    x, xdot = env._cylinder_adapter.current_from_cylinder_joints(cyl_q, cyl_qdot)
    ee_pos_w = robot.data.body_link_state_w[:, env._ee_body_id, :3]
    ee_quat_w = robot.data.body_link_state_w[:, env._ee_body_id, 3:7]
    return {
        "cylinder_extension": x.detach().cpu().numpy().astype(np.float32),
        "cylinder_velocity": xdot.detach().cpu().numpy().astype(np.float32),
        "dP": env._dP.detach().cpu().numpy().astype(np.float32),
        "joint_position": robot.data.joint_pos[:, env._arm_joint_ids].detach().cpu().numpy().astype(np.float32),
        "joint_velocity": robot.data.joint_vel[:, env._arm_joint_ids].detach().cpu().numpy().astype(np.float32),
        "ee_pos_local": env._compute_ee_pos_local().detach().cpu().numpy().astype(np.float32),
        "ee_pos_w": ee_pos_w.detach().cpu().numpy().astype(np.float32),
        "ee_quat_w": ee_quat_w.detach().cpu().numpy().astype(np.float32),
        "ee_lin_vel_w": robot.data.body_link_lin_vel_w[:, env._ee_body_id].detach().cpu().numpy().astype(np.float32),
        "ee_ang_vel_w": robot.data.body_link_ang_vel_w[:, env._ee_body_id].detach().cpu().numpy().astype(np.float32),
    }


def snapshot_command(
    env, action: torch.Tensor, *, probe_action: torch.Tensor | None = None
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    def grab(name: str, tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy().astype(np.float32)

    valve = env._current_valve_cmd.detach().cpu().numpy().astype(np.float32)
    nan4 = np.full((env.num_envs, 4), np.nan, dtype=np.float32)
    nan1 = np.full((env.num_envs, 1), np.nan, dtype=np.float32)
    nan3 = np.full((env.num_envs, 3), np.nan, dtype=np.float32)
    command = {
        "probe_excitation": (
            probe_action if probe_action is not None else torch.zeros_like(action)
        ).detach().cpu().numpy().astype(np.float32),
        "controller_output": grab("controller", env._current_action),
        "desired_cylinder_position": grab("pos", env._cylinder_ref),
        "desired_cylinder_velocity": grab("vel", env._cylinder_vel_ref),
        "desired_delta_extension": grab("dx", env._action_target_delta_x_learned),
        "desired_delta_extension_velocity": grab("dxv", env._action_target_delta_xdot_learned),
        "command_before_limits": grab("pre", env._action_target_delta_x_pre_gain_learned),
        "command_after_limits": grab("post", env._action_target_delta_x_learned),
        "requested_valve_cmd": valve,
        "applied_valve_cmd": nan4,
        "rom_predicted_valve_cmd": valve,
        "rom_effective_actuation": valve,
        "delta_extension_per_step": grab("ext", env._predicted_delta_extension),
        "delta_isaac": grab("isaac", env._delta_isaac),
        "cylinder_pos_target": grab("target", env._cylinder_pos_target),
        "rom_tick": np.asarray(
            (env._command_cylinder_rom.counter == 1).detach().cpu().numpy(),
            dtype=np.float32,
        ).reshape(-1, 1),
    }
    quality = {
        "cylinder_limit_flag": _limit_flags(env),
        "command_saturation_flag": _saturation_flags(env),
        "valve_saturation_flag": (np.abs(valve) >= 1.0 - 1e-6).astype(np.float32),
        "invalid_physics_flag": _invalid_flags(env),
        "dls_condition_number": nan1,
        "dls_damping": nan1,
        "contact_state": nan1,
        "contact_force": nan3,
    }
    return command, quality


def _limit_flags(env) -> np.ndarray:
    q = env.robot.data.joint_pos[:, env._cylinder_joint_ids]
    lo = env.robot.data.soft_joint_pos_limits[:, env._cylinder_joint_ids, 0]
    hi = env.robot.data.soft_joint_pos_limits[:, env._cylinder_joint_ids, 1]
    eps = 1e-3
    near = ((q - lo).abs() < eps) | ((hi - q).abs() < eps)
    return near.detach().cpu().numpy().astype(np.float32)


def _saturation_flags(env) -> np.ndarray:
    pre = env._action_target_delta_x_pre_gain_clip_mask
    post = env._action_target_delta_x_post_gain_clip_mask
    dxdot = env._action_target_delta_xdot_clip_mask
    return (pre | post | dxdot).detach().cpu().numpy().astype(np.float32)


def _invalid_flags(env) -> np.ndarray:
    vals = torch.cat(
        (
            env._cylinder_x_current,
            env._cylinder_xdot_current,
            env._dP,
            env._delta_isaac,
            env._cylinder_pos_target,
        ),
        dim=-1,
    )
    finite = torch.isfinite(vals).all(dim=-1)
    return (~finite).detach().cpu().numpy().astype(np.float32).reshape(-1, 1)


def safety_rates(arrays: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        "invalid": float(np.mean(arrays["invalid_physics_flag"] > 0.5)),
        "cylinder_limit": float(np.mean(arrays["cylinder_limit_flag"] > 0.5)),
        "command_saturation": float(np.mean(arrays["command_saturation_flag"] > 0.5)),
        "valve_saturation": float(np.mean(arrays["valve_saturation_flag"] > 0.5)),
    }


def apply_factors(
    env,
    rows: list[dict],
    *,
    baseline_masses: torch.Tensor,
    baseline_coms: torch.Tensor,
    body_idx: int,
) -> list[tuple[list[float], float, float]]:
    """Apply run i's factor draw, always starting from the env's baseline mass/COM.

    `set_masses` / `set_coms` write into the PhysX view and are NOT reset by
    `env.reset()`, so applying factors onto the *current* values would
    accumulate mass scales (multiplicatively) and COM shifts (additively)
    across consecutive runs within one process. Baseline snapshots are taken
    once at env build and each run starts from them.
    """
    if len(rows) != env.num_envs:
        raise ValueError(f"need {env.num_envs} factor rows, got {len(rows)}")
    applied = []
    system_factor_rows = []
    view = env.robot.root_physx_view
    idx = torch.arange(env.num_envs, dtype=torch.long)
    masses = baseline_masses.clone()
    coms = baseline_coms.clone()
    for env_id, e in enumerate(rows):
        system_factors = e_to_system_factors(e)
        mass_scale, com_z = e_to_mechanical(e)
        system_factor_rows.append(system_factors)
        masses[env_id, body_idx] *= mass_scale
        coms[env_id, body_idx, 2] += com_z
        applied.append((system_factors, mass_scale, com_z))
    env._command_rom_system_factors = torch.tensor(
        system_factor_rows, device=env.device, dtype=torch.float32
    )
    view.set_masses(masses, idx)
    view.set_coms(coms, idx)
    return applied


def controller_gains(env) -> dict:
    cfg = env.cfg
    keys = (
        "action_scale",
        "cylinder_target_max_lead_m",
        "cylinder_target_leak",
        "cylinder_neutral_hold_deadband_m",
        "ppo_neutral_action_deadband",
        "command_rom_dp_abs_limit_pa",
        "cylinder_rom_delta_clip_m",
        "low_level_servo_mode",
        "cylinder_target_integrates_from_previous_target",
        "max_abs_valve",
    )
    return {k: getattr(cfg, k, None) for k in keys}


def deploy_to_posture(env, target_seed: np.ndarray, steps: int) -> None:
    """Cosine ramp from the reset pose to posture A via direct PhysX targets."""
    driver = DirectCylinderDriver(env)
    start_x = driver.measured_x().clone()
    target = torch.as_tensor(
        target_seed, device=env.device, dtype=torch.float32
    ).reshape(1, 4)
    for step in range(max(int(steps), 1)):
        u = float(step + 1) / max(int(steps), 1)
        blend = 0.5 * (1.0 - math.cos(math.pi * u))
        deploy_target = start_x + blend * (target - start_x)
        driver.step(deploy_target, substeps=2)
    # Sync the env's internal target/valve state to the deployed pose so the
    # first recorded ROM tick starts from the same condition as the pilot.
    scaling = env._cylinder_adapter.scaling
    q_a = scaling.clamp_usd_q(scaling.learned_x_to_usd_q(target))
    env._cylinder_pos_target[:] = q_a
    env._neutral_hold_cylinder_pos_target[:] = q_a
    env._prev_valve_cmd.zero_()
    env._current_valve_cmd.zero_()
    env._fd_delta_raw.zero_()
    env._fd_delta_corrected.zero_()
    env._prev_action.zero_()
    env._current_action.zero_()
    env._command_cylinder_rom.reset()


def main() -> None:
    rows = select_rows(load_manifest(Path(a.manifest)))
    if not rows:
        raise SystemExit("no runs selected")

    plan = build_run_plan(
        settle_s=a.settle_s,
        posture_ramp_s=a.posture_ramp_s,
        id_s=a.id_s,
        stow_s=a.stow_s,
        probe_name=a.probe,
        probe_amplitude=a.amplitude,
        probe_seed=a.probe_seed,
        probe_hold_steps=a.probe_hold_steps,
        trajectory_stride=a.trajectory_stride,
    )
    total_steps = plan.total_steps
    # The Stage 17 learning timebase requires the recorded ROM tick grid to be
    # exactly 0, 3, 6, ... and every transition t -> t+3 to lie inside the run.
    # That holds iff (a) the run length is a multiple of the ROM tick and
    # (b) the ROM counter phase is 0 when recording starts (see pre-settle
    # rounding below). The default Stage 13 lengths satisfy both.
    if total_steps % STEPS_PER_ROM_TICK != 0:
        raise SystemExit(
            f"run length {total_steps} is not a multiple of "
            f"{STEPS_PER_ROM_TICK} control steps per ROM tick; the last aligned "
            "50 ms transition would fall outside the run. Keep the segment "
            "timings (or adjust --stow-s) so the total is a multiple of 3."
        )
    print(
        f"run structure: {len(plan.segments)} segments, {total_steps} steps "
        f"= {total_steps * plan.control_dt:.1f} s (probe={a.probe}, amp={a.amplitude})",
        flush=True,
    )

    out = Path(a.out)
    runs_dir = out / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    manifest_src = Path(a.manifest)
    manifest_dst = out / "manifest.csv"
    if manifest_dst.exists() and manifest_dst.read_bytes() != manifest_src.read_bytes():
        raise SystemExit(
            f"existing output manifest differs from {manifest_src}; refusing to mix datasets"
        )
    if not manifest_dst.exists():
        shutil.copyfile(manifest_src, manifest_dst)

    if a.num_envs < 1:
        raise SystemExit("--num-envs must be at least 1")
    if a.max_isolated_safety_attempts < 1:
        raise SystemExit("--max-isolated-safety-attempts must be at least 1")
    if a.num_envs > 1 and not a.matched_reset_seed:
        raise SystemExit(
            "vectorized collection requires --matched-reset-seed so every "
            "concurrent run has the manifest-prescribed common reset"
        )

    pending_rows = []
    for row in rows:
        run_id = row["run_id"]
        run_dir = runs_dir / f"{run_id:05d}"
        if not run_dir.exists():
            pending_rows.append(row)
            continue
        if not a.resume:
            raise SystemExit(
                f"run output already exists: {run_dir} (use --resume to validate and skip)"
            )
        issues = validate_run(run_dir, manifest_row=row)
        if issues:
            raise SystemExit(f"cannot resume past invalid run {run_id:05d}: {issues}")
        print(f"run {run_id:05d} already valid -> SKIP", flush=True)
    if not pending_rows:
        print("all selected runs are already valid", flush=True)
        simulation_app.close()
        return

    cfg = parse_env_cfg(TASK, device=a.device, num_envs=a.num_envs)
    # This asset contains a nested CollisionGroup that PhysX cannot safely
    # replicate. With replication enabled, nominal clones diverge despite
    # identical reset/commands. Author independent physics schemas instead.
    if a.num_envs > 1:
        cfg.scene.replicate_physics = False
    cfg.enable_scripted_vehicle_motion = False
    cfg.episode_length_s = max(float(cfg.episode_length_s), total_steps / 60.0 + 10.0)
    cfg.sim.physx.enable_external_forces_every_iteration = True
    env = gym.make(TASK, cfg=cfg).unwrapped
    robot = env.robot

    env.reset(seed=a.seed_base)
    body_names = list(robot.body_names)
    if CUTTER_BODY not in body_names:
        raise RuntimeError(f"{CUTTER_BODY} not found; have {body_names}")
    body_idx = body_names.index(CUTTER_BODY)
    baseline_masses = robot.root_physx_view.get_masses().clone()
    baseline_coms = robot.root_physx_view.get_coms().clone()

    physics_dt = float(env.cfg.sim.dt)
    control_dt = float(getattr(env, "step_dt", physics_dt * int(env.cfg.decimation)))
    rom_dt = float(env._command_cylinder_rom.sample_time_s)
    rom_checkpoint = str(env._command_cylinder_rom.checkpoint_path)

    cylinder = CylinderDLSController(
        env,
        CylinderDLSConfig(
            invert_learned_pipeline=False,
            error_gain=float(a.error_gain),
            velocity_damping=float(a.velocity_damping),
            velocity_feedforward=float(a.velocity_feedforward),
            max_desired_delta_m=float(a.max_desired_delta_m),
            max_action=1.0,
            action_smoothing=float(a.action_smoothing),
        ),
    )

    dls_params = {
        "error_gain": float(a.error_gain),
        "velocity_damping": float(a.velocity_damping),
        "velocity_feedforward": float(a.velocity_feedforward),
        "max_desired_delta_m": float(a.max_desired_delta_m),
        "max_action": 1.0,
        "action_smoothing": float(a.action_smoothing),
        "invert_learned_pipeline": False,
    }
    plan_params = {
        "name": "run_structure",
        "probe": a.probe,
        "probe_amplitude": a.amplitude,
        "probe_seed": int(a.probe_seed),
        "probe_segment_seeds": [int(a.probe_seed + k) for k in range(3)],
        "probe_hold_steps": int(a.probe_hold_steps),
        "matched_reset_seed": bool(a.matched_reset_seed),
        "factor_profile": str(a.factor_profile),
        "trajectory_stride": int(a.trajectory_stride),
        "id_s": a.id_s,
        "settle_s": a.settle_s,
        "posture_ramp_s": a.posture_ramp_s,
        "stow_s": a.stow_s,
        "segments": len(plan.segments),
        "total_steps": total_steps,
    }

    work_batches = [
        (pending_rows[start : start + a.num_envs], 0)
        for start in range(0, len(pending_rows), a.num_envs)
    ]
    batch_number = 0
    while batch_number < len(work_batches):
        active_rows, retry_count = work_batches[batch_number]
        batch_number += 1
        # Rotate row-to-slot assignment between batches so any tiny solver-order
        # or clone-index effect is balanced rather than correlated with Sobol or
        # factor ordering in the manifest.
        rotation = (batch_number - 1) % len(active_rows) if retry_count == 0 else 0
        active_rows = active_rows[rotation:] + active_rows[:rotation]
        # IsaacLab has a fixed environment count. Pad only the final physical
        # batch; padded environments are stepped but never written.
        batch_rows = active_rows + [active_rows[-1]] * (a.num_envs - len(active_rows))
        batch_ids = [row["run_id"] for row in active_rows]
        seed = a.seed_base if a.matched_reset_seed else a.seed_base + batch_ids[0]
        t0 = time.time()
        start_iso = datetime.now(timezone.utc).isoformat()
        env.reset(seed=seed)
        applied = apply_factors(
            env,
            batch_rows,
            baseline_masses=baseline_masses,
            baseline_coms=baseline_coms,
            body_idx=body_idx,
        )
        deploy_to_posture(env, plan.postures["A"], a.preposition_steps)
        cylinder.reset()
        posture_a = torch.as_tensor(
            plan.postures["A"], device=env.device, dtype=torch.float32
        ).reshape(1, 4).expand(env.num_envs, -1)
        # Round the warm-up to a whole number of ROM ticks so the recorded
        # grid starts at control step 0 (the ROM counter phase is preserved).
        pre_settle = (
            (max(int(a.pre_settle_steps), 1) + STEPS_PER_ROM_TICK - 1)
            // STEPS_PER_ROM_TICK
        ) * STEPS_PER_ROM_TICK
        for _ in range(pre_settle):
            action, _sv = cylinder.action(posture_a)
            env.step(action)

        state = {k: [v] for k, v in snapshot_state(env).items()}
        cmd_buf = {k: [] for k in COMMAND_SIGNALS}
        flag_buf = {k: [] for k in FLAG_SIGNALS}
        time_s = [0.0]

        for seg in plan.segments:
            n = seg.steps
            if seg.probe_action is not None:
                probe = np.asarray(seg.probe_action, dtype=np.float32)
                for k in range(n):
                    action = torch.as_tensor(
                        probe[k : k + 1], device=env.device, dtype=torch.float32
                    ).expand(env.num_envs, -1)
                    env.step(action)
                    command, quality = snapshot_command(
                        env, action, probe_action=action
                    )
                    for name in COMMAND_SIGNALS:
                        cmd_buf[name].append(command[name])
                    for name in FLAG_SIGNALS:
                        flag_buf[name].append(quality[name])
                    time_s.append((len(time_s)) * control_dt)
                    for name, vec in snapshot_state(env).items():
                        state[name].append(vec)
                continue

            # DLS segment: close the loop on the measured cylinder state toward
            # the plan's seed reference path (verified teacher chain, Stage 9/11).
            cylinder.reset()
            seeds = np.asarray(seg.reference_seeds, dtype=np.float64)
            for k in range(n):
                goal = torch.as_tensor(
                    seeds[k], device=env.device, dtype=torch.float32
                ).reshape(1, 4).expand(env.num_envs, -1)
                if k + 1 < n:
                    vel = (seeds[k + 1] - seeds[k]) / control_dt
                else:
                    vel = np.zeros(4)
                goal_velocity = torch.as_tensor(
                    vel, device=env.device, dtype=torch.float32
                ).reshape(1, 4).expand(env.num_envs, -1)
                action, _sv = cylinder.action(goal, goal_velocity=goal_velocity)
                env.step(action)
                command, quality = snapshot_command(env, action)
                for name in COMMAND_SIGNALS:
                    cmd_buf[name].append(command[name])
                for name in FLAG_SIGNALS:
                    flag_buf[name].append(quality[name])
                time_s.append((len(time_s)) * control_dt)
                for name, vec in snapshot_state(env).items():
                    state[name].append(vec)

        batched_arrays = {}
        for name in STATE_SIGNALS:
            batched_arrays[name] = np.stack(state[name], axis=0)
        for name in COMMAND_SIGNALS:
            batched_arrays[name] = np.stack(cmd_buf[name], axis=0)
        for name in FLAG_SIGNALS:
            batched_arrays[name] = np.stack(flag_buf[name], axis=0)

        availability = {
            name: (name not in UNAVAILABLE_SIGNALS)
            for name in (list(STATE_SIGNALS) + list(COMMAND_SIGNALS) + list(FLAG_SIGNALS))
        }
        elapsed = time.time() - t0
        for env_id, row in enumerate(active_rows):
            run_id = row["run_id"]
            run_dir = runs_dir / f"{run_id:05d}"
            system_factors, mass_scale, com_z = applied[env_id]
            arrays = {"time_s": np.array(time_s, dtype=np.float32)}
            for name, values in batched_arrays.items():
                arrays[name] = values[:, env_id, ...]
            rates = safety_rates(arrays)
            violations = {
                name: (value, COLLECTION_SAFETY_LIMITS[name])
                for name, value in rates.items()
                if value > COLLECTION_SAFETY_LIMITS[name]
            }
            if violations:
                if len(active_rows) > 1:
                    work_batches.append(([row], 1))
                    print(
                        f"run {run_id:05d} slot {env_id} safety {violations} "
                        "-> RETRY ALONE IN ENV 0",
                        flush=True,
                    )
                    continue
                isolated_attempt = max(retry_count, 1)
                if isolated_attempt < a.max_isolated_safety_attempts:
                    next_attempt = isolated_attempt + 1
                    work_batches.append(([row], next_attempt))
                    print(
                        f"run {run_id:05d} isolated env-0 attempt "
                        f"{isolated_attempt}/{a.max_isolated_safety_attempts} "
                        f"safety {violations} -> RETRY",
                        flush=True,
                    )
                    continue
                raise SystemExit(
                    f"run {run_id:05d} failed safety after "
                    f"{a.max_isolated_safety_attempts} isolated env-0 attempts: {violations}"
                )
            metadata = {
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
                "domain": row["domain"],
                "split": row["split"],
                "random_seed": seed,
                "trajectory_schedule": plan_params,
                "segment_ids": [
                    {
                        "name": seg["name"],
                        "kind": seg["kind"],
                        "start_step": seg["start"],
                        "end_step": seg["end"],
                    }
                    for seg in plan.segment_table()
                ],
                "physics_dt": physics_dt,
                "control_dt": control_dt,
                "rom_dt": rom_dt,
                "controller_gains": controller_gains(env),
                "dls_parameters": dls_params,
                "excitation_config": plan_params,
                "factor_profile": str(a.factor_profile),
                "hydraulic_draw": {k: row[k] for k in ("flow", "compliance", "seal", "pressure")},
                "mechanical_draw": {"mass_scale": row["mass_scale"], "com_z": row["com_z"]},
                "flow": row["flow"],
                "compliance": row["compliance"],
                "seal": row["seal"],
                "pressure": row["pressure"],
                "mass_scale": row["mass_scale"],
                "com_z": row["com_z"],
                "applied_system_factors": system_factors,
                "applied_mass_scale": mass_scale,
                "applied_com_z": com_z,
                "start_time": start_iso,
                "end_time": datetime.now(timezone.utc).isoformat(),
                "task_name": TASK,
                "rom_checkpoint": rom_checkpoint,
                "num_control_steps": total_steps,
                "availability": availability,
                "vectorized_collection": {
                    "num_envs": int(a.num_envs),
                    "batch_env_id": int(env_id),
                    "active_batch_size": len(active_rows),
                    "isolated_safety_retry": bool(retry_count),
                    "safety_rates": rates,
                },
            }
            write_run(run_dir, metadata, arrays)
            issues = validate_run(run_dir, manifest_row=row)
            status = "OK" if not issues else f"ISSUES: {issues}"
            print(
                f"run {run_id:05d} [{row['domain']}/{row['split']}] "
                f"{total_steps} steps in batch {batch_ids} ({elapsed:.1f}s) -> {status}",
                flush=True,
            )
            if issues:
                raise SystemExit(f"run {run_id:05d} failed validation: {issues}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
