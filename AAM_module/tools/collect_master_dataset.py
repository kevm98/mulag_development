#!/usr/bin/env python
"""Collect the master raw Isaac Lab dataset (AAM Stage 8).

Reads the Stage 6/7 domain manifest, runs each selected run once in a headless
Isaac env, applies the fixed physical factor draw e_i (hydraulic via the ROM
system_factors channel, mechanical via PhysX mass/COM on the cutter body), and
records (tau_i, e_i) as metadata.json + trajectory.npz per run.

Everything downstream (SD transitions, AAM history windows) is derived from
this master set; no per-signal preprocessing is written here.

Usage:
    python collect_master_dataset.py --limit 2 --steps 180
    python collect_master_dataset.py --run-ids 0,1,2 --steps 600
"""

from __future__ import annotations

import argparse
import csv
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


p = argparse.ArgumentParser()
p.add_argument("--manifest", default=str(AAM_ROOT / "results/domain_manifest_sample.csv"))
p.add_argument("--out", default=str(AAM_ROOT / "datasets/master_v1"))
p.add_argument("--limit", type=int, default=None, help="collect only the first N runs")
p.add_argument("--run-ids", type=str, default=None, help="comma-separated run ids to collect")
p.add_argument("--steps", type=int, default=600, help="control steps per run (60 Hz)")
p.add_argument("--schedule", choices=("multisine", "trapezoid"), default="multisine")
p.add_argument("--num-envs", type=int, default=1)
p.add_argument("--seed-base", type=int, default=20260808)
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
    UNAVAILABLE_SIGNALS,
    MultisineSchedule,
    TrapezoidSchedule,
    e_to_mechanical,
    e_to_system_factors,
    validate_run,
    write_run,
)


TASK = "RRLAB-Mulag-PPO2-ROM-PhysX-CylinderSpace-v0"
CUTTER_BODY = "Messerkopf"


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


def build_schedule() -> tuple[np.ndarray, list[dict]]:
    if a.schedule == "trapezoid":
        sched = TrapezoidSchedule()
    else:
        sched = MultisineSchedule()
    action, segments = sched.build(total_steps=a.steps)
    return action, [seg.as_dict() for seg in segments]


def snapshot_state(env) -> dict[str, np.ndarray]:
    robot = env.robot
    cyl_q = robot.data.joint_pos[:, env._cylinder_joint_ids]
    cyl_qdot = robot.data.joint_vel[:, env._cylinder_joint_ids]
    x, xdot = env._cylinder_adapter.current_from_cylinder_joints(cyl_q, cyl_qdot)
    ee_pos_w = robot.data.body_link_state_w[:, env._ee_body_id, :3]
    ee_quat_w = robot.data.body_link_state_w[:, env._ee_body_id, 3:7]
    return {
        "cylinder_extension": x[0].detach().cpu().numpy().astype(np.float32),
        "cylinder_velocity": xdot[0].detach().cpu().numpy().astype(np.float32),
        "dP": env._dP[0].detach().cpu().numpy().astype(np.float32),
        "joint_position": robot.data.joint_pos[0, env._arm_joint_ids].detach().cpu().numpy().astype(np.float32),
        "joint_velocity": robot.data.joint_vel[0, env._arm_joint_ids].detach().cpu().numpy().astype(np.float32),
        "ee_pos_local": env._compute_ee_pos_local()[0].detach().cpu().numpy().astype(np.float32),
        "ee_pos_w": ee_pos_w[0].detach().cpu().numpy().astype(np.float32),
        "ee_quat_w": ee_quat_w[0].detach().cpu().numpy().astype(np.float32),
        "ee_lin_vel_w": robot.data.body_link_lin_vel_w[0, env._ee_body_id].detach().cpu().numpy().astype(np.float32),
        "ee_ang_vel_w": robot.data.body_link_ang_vel_w[0, env._ee_body_id].detach().cpu().numpy().astype(np.float32),
    }


def snapshot_command(env, probe_action: torch.Tensor) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    def grab(name: str, tensor) -> np.ndarray:
        return tensor[0].detach().cpu().numpy().astype(np.float32)

    valve = env._current_valve_cmd[0].detach().cpu().numpy().astype(np.float32)
    nan4 = np.full(4, np.nan, dtype=np.float32)
    nan1 = np.full(1, np.nan, dtype=np.float32)
    nan3 = np.full(3, np.nan, dtype=np.float32)
    command = {
        "probe_excitation": probe_action[0].detach().cpu().numpy().astype(np.float32),
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
        "rom_tick": np.array(
            [env._command_cylinder_rom.counter[0].item() == 1], dtype=np.float32
        ),
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
    return near[0].detach().cpu().numpy().astype(np.float32)


def _saturation_flags(env) -> np.ndarray:
    pre = env._action_target_delta_x_pre_gain_clip_mask
    post = env._action_target_delta_x_post_gain_clip_mask
    dxdot = env._action_target_delta_xdot_clip_mask
    return (pre | post | dxdot)[0].detach().cpu().numpy().astype(np.float32)


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
    return np.array([(~finite)[0].item()], dtype=np.float32)


def apply_factors(
    env,
    e: dict[str, float],
    *,
    baseline_masses: torch.Tensor,
    baseline_coms: torch.Tensor,
    body_idx: int,
) -> tuple[list[float], float, float]:
    """Apply run i's factor draw, always starting from the env's baseline mass/COM.

    ``set_masses`` / ``set_coms`` write into the PhysX view and are NOT reset by
    ``env.reset()``, so applying factors onto the *current* values would
    accumulate mass scales (multiplicatively) and COM shifts (additively)
    across consecutive runs within one process. Baseline snapshots are taken
    once at env build and each run starts from them.
    """
    system_factors = e_to_system_factors(e)
    env._command_rom_system_factors = torch.tensor(
        [system_factors] * env.num_envs, device=env.device, dtype=torch.float32
    )
    mass_scale, com_z = e_to_mechanical(e)
    view = env.robot.root_physx_view
    idx = torch.arange(env.num_envs, dtype=torch.long)
    masses = baseline_masses.clone()
    masses[:, body_idx] = masses[:, body_idx] * mass_scale
    view.set_masses(masses, idx)
    coms = baseline_coms.clone()
    coms[:, body_idx, 2] += com_z
    view.set_coms(coms, idx)
    return system_factors, mass_scale, com_z


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


def main() -> None:
    rows = select_rows(load_manifest(Path(a.manifest)))
    if not rows:
        raise SystemExit("no runs selected")

    out = Path(a.out)
    runs_dir = out / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(a.manifest), out / "manifest.csv")

    cfg = parse_env_cfg(TASK, device=a.device, num_envs=a.num_envs)
    cfg.enable_scripted_vehicle_motion = False
    cfg.episode_length_s = max(float(cfg.episode_length_s), a.steps / 60.0 + 5.0)
    env = gym.make(TASK, cfg=cfg).unwrapped
    robot = env.robot

    # Baseline physical properties, captured before any factor application.
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

    schedule, segments = build_schedule()
    # Keep the JSON small: store the schedule params, not the full array.
    excitation_cfg = {
        "name": a.schedule,
        "params": (
            MultisineSchedule().params
            if a.schedule == "multisine"
            else TrapezoidSchedule().params
        ),
        "steps": a.steps,
        "seed_base": a.seed_base,
    }

    print(
        f"collecting {len(rows)} runs -> {out} | physics_dt={physics_dt:.6g} "
        f"control_dt={control_dt:.6g} rom_dt={rom_dt:.6g}",
        flush=True,
    )
    for row in rows:
        run_id = row["run_id"]
        seed = a.seed_base + run_id
        t0 = time.time()
        env.reset(seed=seed)
        system_factors, mass_scale, com_z = apply_factors(
            env,
            row,
            baseline_masses=baseline_masses,
            baseline_coms=baseline_coms,
            body_idx=body_idx,
        )

        state = {k: [v] for k, v in snapshot_state(env).items()}
        cmd_buf = {k: [] for k in COMMAND_SIGNALS}
        flag_buf = {k: [] for k in FLAG_SIGNALS}
        time_s = [0.0]

        for k in range(a.steps):
            action = torch.tensor(schedule[k : k + 1], device=env.device)
            env.step(action)
            command, quality = snapshot_command(env, action)
            for name in COMMAND_SIGNALS:
                cmd_buf[name].append(command[name])
            for name in FLAG_SIGNALS:
                flag_buf[name].append(quality[name])
            time_s.append((k + 1) * control_dt)
            for name, vec in snapshot_state(env).items():
                state[name].append(vec)

        arrays = {"time_s": np.array(time_s, dtype=np.float32)}
        for name in STATE_SIGNALS:
            arrays[name] = np.stack(state[name], axis=0)
        for name in COMMAND_SIGNALS:
            arrays[name] = np.stack(cmd_buf[name], axis=0)
        for name in FLAG_SIGNALS:
            arrays[name] = np.stack(flag_buf[name], axis=0)

        availability = {
            name: (name not in UNAVAILABLE_SIGNALS)
            for name in (list(STATE_SIGNALS) + list(COMMAND_SIGNALS) + list(FLAG_SIGNALS))
        }
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "domain": row["domain"],
            "split": row["split"],
            "random_seed": seed,
            "trajectory_schedule": excitation_cfg,
            "segment_ids": segments,
            "physics_dt": physics_dt,
            "control_dt": control_dt,
            "rom_dt": rom_dt,
            "controller_gains": controller_gains(env),
            "dls_parameters": None,
            "excitation_config": excitation_cfg,
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
            "start_time": datetime.now(timezone.utc).isoformat(),
            "end_time": datetime.now(timezone.utc).isoformat(),
            "task_name": TASK,
            "rom_checkpoint": rom_checkpoint,
            "num_control_steps": a.steps,
            "availability": availability,
        }
        run_dir = runs_dir / f"{run_id:05d}"
        write_run(run_dir, metadata, arrays)
        issues = validate_run(run_dir, manifest_row=row)
        status = "OK" if not issues else f"ISSUES: {issues}"
        print(
            f"run {run_id:05d} [{row['domain']}/{row['split']}] "
            f"{a.steps} steps in {time.time()-t0:.1f}s -> {status}",
            flush=True,
        )
        if issues:
            raise SystemExit(f"run {run_id:05d} failed validation: {issues}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
