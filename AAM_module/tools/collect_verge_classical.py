#!/usr/bin/env python3
"""Collect final-task ROM data with a classical controller only.

The collector drives the final ROM deployment plant with a bounded DLS
controller plus deterministic PRBS/multisine excitation. PPO is deliberately
not loaded or queried. The resulting frozen ROM dataset is the single source
for both SD training and AAM training; ID-FD is a deployment/evaluation plant,
never a data-collection source. A separate ``privileged.npz`` sidecar stores
controller/probe traces and the recurrent ROM state for diagnostics only.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from isaaclab.app import AppLauncher


ROOT = Path(__file__).resolve().parents[2]
AAM_ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, AAM_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--actuator",
    choices=("rom",),
    default="rom",
    help="fixed to the final ROM deployment plant; ID-FD is never a collection source",
)
parser.add_argument(
    "--manifest",
    default=str(AAM_ROOT / "results/stage22_wide_manifest.csv"),
    help="factor manifest; rows are copied into the output dataset",
)
parser.add_argument(
    "--out",
    default=None,
    help="output dataset (default: datasets/verge_<actuator>_classical_v1)",
)
parser.add_argument("--limit", type=int, default=None)
parser.add_argument("--run-ids", default=None)
parser.add_argument("--seed-base", type=int, default=20260820)
parser.add_argument("--probe", choices=("prbs", "multisine"), default="prbs")
parser.add_argument("--probe-amplitude", type=float, default=0.02)
parser.add_argument("--probe-hold-steps", type=int, default=10)
parser.add_argument("--probe-seed", type=int, default=20260821)
parser.add_argument(
    "--step-factor",
    default=None,
    choices=("flow", "compliance", "seal", "pressure"),
    help=(
        "collect runs whose plant CHANGES mid-episode on this hydraulic "
        "factor. Required for any in-episode adaptation claim: with a constant "
        "plant there is nothing for the latent to track. Mechanical factors "
        "(mass_scale/com_z) are not supported here -- they are rigid-body "
        "properties, not ROM system_factors, and would need a different "
        "mechanism."
    ),
)
parser.add_argument(
    "--step-value", type=float, default=None,
    help="value the stepped factor takes after the change",
)
parser.add_argument(
    "--step-at-frac", type=float, default=0.5,
    help="when the step fires, as a fraction of the episode (0-1)",
)
parser.add_argument(
    "--step-every-n", type=int, default=2,
    help=(
        "step every Nth run, leaving the rest constant. A mixed dataset keeps "
        "between-plant separation learnable while adding in-episode tracking; "
        "1 = every run stepped, 0 = none"
    ),
)
parser.add_argument(
    "--max-steps",
    type=int,
    default=None,
    help="truncate the schedule for a short pilot; omit for the full campaign",
)
parser.add_argument(
    "--progress-steps",
    type=int,
    default=300,
    help="print progress every N controls (0 disables progress output)",
)
parser.add_argument(
    "--num-envs",
    type=int,
    default=1,
    help="parallel independent collection slots; use 6 for the full campaign",
)
parser.add_argument("--resume", action="store_true")
parser.add_argument(
    "--vehicle-motion",
    action="store_true",
    help="keep the final scripted truck motion enabled; default isolates arm dynamics",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import PPO_ID_FD  # noqa: E402,F401
import PPO_ROM  # noqa: E402,F401
from PPO_ID_FD.env_cfg import Ppo41IDFDEnvCfg  # noqa: E402
from PPO_ROM.env_cfg import Ppo31SoleEnvCfg  # noqa: E402
from aam_data import (  # noqa: E402
    COMMAND_SIGNALS,
    FLAG_SIGNALS,
    SCHEMA_VERSION,
    STATE_SIGNALS,
    UNAVAILABLE_SIGNALS,
    MultisineSchedule,
    PRBSSchedule,
    e_to_mechanical,
    e_to_system_factors,
    validate_run,
    write_run,
)
from mulag_plant.dls_solver.cylinder import (  # noqa: E402
    CylinderDLSConfig,
    CylinderDLSController,
)


TASK = "RRLAB-Mulag-Verge-ROM-v0"
OUT = Path(
    args.out
    or AAM_ROOT / "datasets" / f"verge_{args.actuator}_classical_v1"
).expanduser().resolve()
MANIFEST = Path(args.manifest).expanduser().resolve()
DT = 1.0 / 60.0
MODEL_DT = 0.05
STEPS_PER_MODEL_TICK = 3
CUTTER_BODY = "Messerkopf"


def _plant_signature(cfg) -> dict:
    return {
        "robot_usd": str(cfg.robot_cfg.spawn.usd_path),
        "arm_joint_names": tuple(cfg.arm_joint_names),
        "cylinder_joint_names": tuple(cfg.cylinder_joint_names),
        "sim_dt": float(cfg.sim.dt),
        "decimation": int(cfg.decimation),
        "episode_length_s": float(cfg.episode_length_s),
        "mow_strip_length_m": float(cfg.mow_strip_length_m),
        "tracking_active_length_m": float(cfg.terrain_tracking_active_length_m),
        "vehicle_speed": (
            float(cfg.vehicle_speed_min_kmh),
            float(cfg.vehicle_speed_max_kmh),
        ),
        "observation_space": int(cfg.observation_space),
        "action_space": int(cfg.action_space),
    }


def assert_shared_rom_plant(cfg) -> dict:
    """Reject collection if ROM and ID-FD no longer share the final plant."""
    canonical_rom = Ppo31SoleEnvCfg()
    canonical_idfd = Ppo41IDFDEnvCfg()
    rom_signature = _plant_signature(canonical_rom)
    idfd_signature = _plant_signature(canonical_idfd)
    if rom_signature != idfd_signature:
        mismatches = [
            key for key in rom_signature
            if rom_signature[key] != idfd_signature[key]
        ]
        raise RuntimeError(
            "ROM/ID-FD plant contract diverged before collection: "
            + ", ".join(mismatches)
        )
    current = _plant_signature(cfg)
    if current != rom_signature:
        mismatches = [key for key in rom_signature if current[key] != rom_signature[key]]
        raise RuntimeError(
            f"{TASK} is not the canonical PPO-ROM plant: {', '.join(mismatches)}"
        )
    expected_actuator = "command_rom"
    if str(getattr(cfg, "cylinder_dynamics_model", "")).lower() != expected_actuator:
        raise RuntimeError(
            f"{TASK} actuator is {getattr(cfg, 'cylinder_dynamics_model', None)!r}; "
            f"expected {expected_actuator!r}"
        )
    return {
        "rom_task": PPO_ROM.TERRAIN_TASK_ID,
        "idfd_task": PPO_ID_FD.TERRAIN_TASK_ID,
        "shared_signature": {key: (list(value) if isinstance(value, tuple) else value)
                              for key, value in rom_signature.items()},
        "current_task": TASK,
        "current_actuator": expected_actuator,
    }


def load_manifest(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "run_id", "domain", "split", "flow", "compliance", "seal",
        "pressure", "mass_scale", "com_z",
    }
    if not rows or not required.issubset(rows[0]):
        raise SystemExit(f"invalid factor manifest: {path}")
    for row in rows:
        row["run_id"] = int(row["run_id"])
        for key in ("flow", "compliance", "seal", "pressure", "mass_scale", "com_z"):
            row[key] = float(row[key])
    return rows


def select_rows(rows: list[dict]) -> list[dict]:
    if args.run_ids:
        wanted = {int(value) for value in args.run_ids.split(",")}
        rows = [row for row in rows if row["run_id"] in wanted]
    if args.limit is not None:
        rows = rows[: args.limit]
    return rows


def cosine_ramp(start: np.ndarray, end: np.ndarray, steps: int) -> np.ndarray:
    u = np.linspace(0.0, 1.0, max(int(steps), 2), dtype=np.float64)
    blend = 0.5 * (1.0 - np.cos(np.pi * u))
    return start[None, :] + blend[:, None] * (end - start)[None, :]


def resample_path(path: np.ndarray, steps: int) -> np.ndarray:
    path = np.asarray(path, dtype=np.float64)
    source = np.linspace(0.0, 1.0, path.shape[0])
    target = np.linspace(0.0, 1.0, max(int(steps), 2))
    return np.stack(
        [np.interp(target, source, path[:, channel]) for channel in range(path.shape[1])],
        axis=1,
    )


def build_plan(cfg) -> tuple[list[dict], dict]:
    """Build a DLS/probe plan from the current reachable library."""
    library_path = Path(str(cfg.trajectory_library_path)).expanduser().resolve()
    payload = json.loads(library_path.read_text())
    line = np.asarray(
        [sample["cylinder_cmd"] for sample in payload["shapes"]["line"]["samples"]],
        dtype=np.float64,
    )
    sine = np.asarray(
        [sample["cylinder_cmd"] for sample in payload["shapes"]["sine"]["samples"]],
        dtype=np.float64,
    )
    posture_a = np.asarray(cfg.terrain_deployment_cylinder_target_m, dtype=np.float64)
    posture_b = line[0].copy()
    posture_c = line[-1].copy()

    def probe(steps: int, seed: int) -> np.ndarray:
        if args.probe == "prbs":
            schedule = PRBSSchedule(
                amplitude=float(args.probe_amplitude),
                per_cylinder_steps=int(steps),
                seed=int(seed),
                hold_steps=int(args.probe_hold_steps),
            )
        else:
            schedule = MultisineSchedule(
                per_cylinder_steps=int(steps),
                amplitudes=tuple([float(args.probe_amplitude)] * 3),
                frequencies_hz=(0.2, 0.8, 2.0),
            )
        action, _ = schedule.build(total_steps=int(steps))
        return action.astype(np.float32)

    segments: list[dict] = []
    cursor = 0

    def add(name: str, kind: str, reference: np.ndarray, probe_action=None) -> None:
        nonlocal cursor
        reference = np.asarray(reference, dtype=np.float32)
        n = int(reference.shape[0])
        segments.append({
            "name": name,
            "kind": kind,
            "start": cursor,
            "end": cursor + n,
            "reference": reference,
            "probe": None if probe_action is None else np.asarray(probe_action, dtype=np.float32),
        })
        cursor += n

    settle = 120
    identification = 1800
    transition = 180
    workspace = 1800
    stow = 240
    add("settle_A", "dls", np.repeat(posture_a[None, :], settle, axis=0))
    add("id_A", "probe", np.repeat(posture_a[None, :], identification, axis=0), probe(identification, args.probe_seed))
    add("dls_to_B", "dls", cosine_ramp(posture_a, posture_b, transition))
    add("id_B", "probe", np.repeat(posture_b[None, :], identification, axis=0), probe(identification, args.probe_seed + 1))
    add("dls_line", "dls", resample_path(line, workspace))
    add("dls_to_C", "dls", cosine_ramp(line[-1], posture_c, transition))
    add("id_C", "probe", np.repeat(posture_c[None, :], identification, axis=0), probe(identification, args.probe_seed + 2))
    add("dls_sine", "dls", resample_path(sine, workspace))
    add("dls_reach_return", "dls", resample_path(np.concatenate((line, line[::-1]), axis=0), workspace))
    add("safe_stow", "dls", cosine_ramp(posture_c, posture_a, stow))
    if cursor % STEPS_PER_MODEL_TICK:
        raise RuntimeError(f"classical plan length {cursor} is not aligned to 50 ms")
    params = {
        "name": "verge_classical_dls_prbs",
        "task": TASK,
        "actuator": args.actuator,
        "control_dt": DT,
        "model_dt": MODEL_DT,
        "steps_per_model_tick": STEPS_PER_MODEL_TICK,
        "probe": args.probe,
        "probe_amplitude": float(args.probe_amplitude),
        "probe_hold_steps": int(args.probe_hold_steps),
        "probe_seed": int(args.probe_seed),
        "trajectory_library": str(library_path),
        "segments": [
            {k: v for k, v in segment.items() if k not in {"reference", "probe"}}
            for segment in segments
        ],
        "total_steps": cursor,
    }
    return segments, params


def limit_plan(segments: list[dict], params: dict, max_steps: int | None) -> tuple[list[dict], dict]:
    if max_steps is None:
        return segments, params
    max_steps = int(max_steps)
    if max_steps < STEPS_PER_MODEL_TICK or max_steps % STEPS_PER_MODEL_TICK:
        raise SystemExit("--max-steps must be a positive multiple of 3")
    if max_steps >= int(params["total_steps"]):
        return segments, params
    limited: list[dict] = []
    for segment in segments:
        if segment["start"] >= max_steps:
            break
        end = min(int(segment["end"]), max_steps)
        length = end - int(segment["start"])
        limited.append({
            **segment,
            "end": end,
            "reference": segment["reference"][:length],
            "probe": None if segment["probe"] is None else segment["probe"][:length],
        })
    limited_params = dict(params)
    limited_params["total_steps"] = max_steps
    limited_params["truncated_from_steps"] = int(params["total_steps"])
    limited_params["segments"] = [
        {key: value for key, value in segment.items() if key not in {"reference", "probe"}}
        for segment in limited
    ]
    return limited, limited_params


def snapshot_state(env) -> dict[str, np.ndarray]:
    robot = env.robot
    q = robot.data.joint_pos[:, env._cylinder_joint_ids]
    qdot = robot.data.joint_vel[:, env._cylinder_joint_ids]
    x, xdot = env._cylinder_adapter.current_from_cylinder_joints(q, qdot)
    ee_pos_w = robot.data.body_link_state_w[:, env._ee_body_id, :3]
    ee_quat_w = robot.data.body_link_state_w[:, env._ee_body_id, 3:7]

    def array(value) -> np.ndarray:
        return value[0].detach().cpu().numpy().astype(np.float32)

    return {
        "cylinder_extension": array(x),
        "cylinder_velocity": array(xdot),
        "dP": array(env._dP),
        "joint_position": array(robot.data.joint_pos[:, env._arm_joint_ids]),
        "joint_velocity": array(robot.data.joint_vel[:, env._arm_joint_ids]),
        "ee_pos_local": array(env._compute_ee_pos_local()),
        "ee_pos_w": array(ee_pos_w),
        "ee_quat_w": array(ee_quat_w),
        "ee_lin_vel_w": array(robot.data.body_link_lin_vel_w[:, env._ee_body_id]),
        "ee_ang_vel_w": array(robot.data.body_link_ang_vel_w[:, env._ee_body_id]),
    }


def snapshot_state_batch(env) -> dict[str, np.ndarray]:
    """Vectorized state snapshot with leading environment dimension."""
    robot = env.robot
    q = robot.data.joint_pos[:, env._cylinder_joint_ids]
    qdot = robot.data.joint_vel[:, env._cylinder_joint_ids]
    x, xdot = env._cylinder_adapter.current_from_cylinder_joints(q, qdot)
    ee_pos_w = robot.data.body_link_state_w[:, env._ee_body_id, :3]
    ee_quat_w = robot.data.body_link_state_w[:, env._ee_body_id, 3:7]

    def array(value) -> np.ndarray:
        return value.detach().cpu().numpy().astype(np.float32)

    return {
        "cylinder_extension": array(x),
        "cylinder_velocity": array(xdot),
        "dP": array(env._dP),
        "joint_position": array(robot.data.joint_pos[:, env._arm_joint_ids]),
        "joint_velocity": array(robot.data.joint_vel[:, env._arm_joint_ids]),
        "ee_pos_local": array(env._compute_ee_pos_local()),
        "ee_pos_w": array(ee_pos_w),
        "ee_quat_w": array(ee_quat_w),
        "ee_lin_vel_w": array(robot.data.body_link_lin_vel_w[:, env._ee_body_id]),
        "ee_ang_vel_w": array(robot.data.body_link_ang_vel_w[:, env._ee_body_id]),
    }


def snapshot_command(env, probe_action: np.ndarray, tick: bool) -> tuple[dict, dict]:
    def array(value) -> np.ndarray:
        return value[0].detach().cpu().numpy().astype(np.float32)

    current_valve = array(env._current_valve_cmd)
    predicted_delta = getattr(env, "_predicted_delta_extension", None)
    if predicted_delta is None:
        predicted_delta = getattr(env, "_fd_delta_corrected")
    command = {
        "probe_excitation": np.asarray(probe_action, dtype=np.float32),
        "controller_output": array(env._current_action),
        "desired_cylinder_position": array(env._cylinder_ref),
        "desired_cylinder_velocity": array(env._cylinder_vel_ref),
        "desired_delta_extension": array(env._action_target_delta_x_learned),
        "desired_delta_extension_velocity": array(env._action_target_delta_xdot_learned),
        "command_before_limits": array(env._action_target_delta_x_pre_gain_learned),
        "command_after_limits": array(env._action_target_delta_x_learned),
        "requested_valve_cmd": current_valve,
        "applied_valve_cmd": np.full(4, np.nan, dtype=np.float32),
        "rom_predicted_valve_cmd": current_valve,
        "rom_effective_actuation": current_valve,
        "delta_extension_per_step": array(predicted_delta),
        "delta_isaac": array(env._delta_isaac),
        "cylinder_pos_target": array(env._cylinder_pos_target),
        "rom_tick": np.asarray([float(tick)], dtype=np.float32),
    }
    q = env.robot.data.joint_pos[:, env._cylinder_joint_ids]
    limits = env.robot.data.soft_joint_pos_limits[:, env._cylinder_joint_ids]
    near_limit = ((q - limits[..., 0]).abs() < 1.0e-3) | ((limits[..., 1] - q).abs() < 1.0e-3)
    clipped = env._action_target_delta_x_pre_gain_clip_mask | env._action_target_delta_x_post_gain_clip_mask | env._action_target_delta_xdot_clip_mask
    finite_values = torch.cat((env._cylinder_x_current, env._cylinder_xdot_current, env._dP, env._delta_isaac), dim=-1)
    quality = {
        "cylinder_limit_flag": near_limit[0].detach().cpu().numpy().astype(np.float32),
        "command_saturation_flag": clipped[0].detach().cpu().numpy().astype(np.float32),
        "valve_saturation_flag": (np.abs(current_valve) >= 1.0 - 1.0e-6).astype(np.float32),
        "invalid_physics_flag": np.asarray([float(not bool(torch.isfinite(finite_values).all()))], dtype=np.float32),
        "dls_condition_number": np.asarray([np.nan], dtype=np.float32),
        "dls_damping": np.asarray([np.nan], dtype=np.float32),
        "contact_state": np.asarray([np.nan], dtype=np.float32),
        "contact_force": np.full(3, np.nan, dtype=np.float32),
    }
    return command, quality


def snapshot_command_batch(env, probe_action: np.ndarray, tick: bool) -> tuple[dict, dict]:
    """Vectorized command/quality snapshot with leading environment dimension."""
    def array(value) -> np.ndarray:
        return value.detach().cpu().numpy().astype(np.float32)

    current_valve = array(env._current_valve_cmd)
    predicted_delta = getattr(env, "_predicted_delta_extension", None)
    if predicted_delta is None:
        predicted_delta = getattr(env, "_fd_delta_corrected")
    command = {
        "probe_excitation": np.broadcast_to(
            np.asarray(probe_action, dtype=np.float32), (env.num_envs, 4)
        ).copy(),
        "controller_output": array(env._current_action),
        "desired_cylinder_position": array(env._cylinder_ref),
        "desired_cylinder_velocity": array(env._cylinder_vel_ref),
        "desired_delta_extension": array(env._action_target_delta_x_learned),
        "desired_delta_extension_velocity": array(env._action_target_delta_xdot_learned),
        "command_before_limits": array(env._action_target_delta_x_pre_gain_learned),
        "command_after_limits": array(env._action_target_delta_x_learned),
        "requested_valve_cmd": current_valve,
        "applied_valve_cmd": np.full((env.num_envs, 4), np.nan, dtype=np.float32),
        "rom_predicted_valve_cmd": current_valve,
        "rom_effective_actuation": current_valve,
        "delta_extension_per_step": array(predicted_delta),
        "delta_isaac": array(env._delta_isaac),
        "cylinder_pos_target": array(env._cylinder_pos_target),
        "rom_tick": np.full((env.num_envs, 1), float(tick), dtype=np.float32),
    }
    q = env.robot.data.joint_pos[:, env._cylinder_joint_ids]
    limits = env.robot.data.soft_joint_pos_limits[:, env._cylinder_joint_ids]
    near_limit = ((q - limits[..., 0]).abs() < 1.0e-3) | ((limits[..., 1] - q).abs() < 1.0e-3)
    clipped = (
        env._action_target_delta_x_pre_gain_clip_mask
        | env._action_target_delta_x_post_gain_clip_mask
        | env._action_target_delta_xdot_clip_mask
    )
    finite_values = torch.cat(
        (env._cylinder_x_current, env._cylinder_xdot_current, env._dP, env._delta_isaac),
        dim=-1,
    )
    quality = {
        "cylinder_limit_flag": near_limit.detach().cpu().numpy().astype(np.float32),
        "command_saturation_flag": clipped.detach().cpu().numpy().astype(np.float32),
        "valve_saturation_flag": (np.abs(current_valve) >= 1.0 - 1.0e-6).astype(np.float32),
        "invalid_physics_flag": (
            ~torch.isfinite(finite_values).all(dim=-1)
        ).detach().cpu().numpy().astype(np.float32).reshape(-1, 1),
        "dls_condition_number": np.full((env.num_envs, 1), np.nan, dtype=np.float32),
        "dls_damping": np.full((env.num_envs, 1), np.nan, dtype=np.float32),
        "contact_state": np.full((env.num_envs, 1), np.nan, dtype=np.float32),
        "contact_force": np.full((env.num_envs, 3), np.nan, dtype=np.float32),
    }
    return command, quality


def safety_rates(arrays: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        "invalid": float(np.mean(arrays["invalid_physics_flag"] > 0.5)),
        "cylinder_limit": float(np.mean(arrays["cylinder_limit_flag"] > 0.5)),
        "command_saturation": float(np.mean(arrays["command_saturation_flag"] > 0.5)),
        "valve_saturation": float(np.mean(arrays["valve_saturation_flag"] > 0.5)),
    }


def controller_checkpoint(env) -> str:
    return str(env._command_cylinder_rom.checkpoint_path)


def hidden_sidecar(env, tick: bool) -> np.ndarray | None:
    if not tick:
        return None
    hidden = getattr(env._command_cylinder_rom, "last_hidden", None)
    if hidden is None:
        return None
    return hidden[:, 0].detach().cpu().numpy().astype(np.float32)


def plan_factor_steps(rows: list[dict], *, factor: str | None, value: float | None,
                      at_frac: float, every_n: int, total_steps: int) -> dict:
    """Decide which runs step, when, and to what. Pure; unit-tested.

    Returns ``{run_id: {"from_tick": int, "factors": {name: value}}}`` for the
    stepped runs only. The mapping is written verbatim into each run's
    ``metadata["factor_schedule"]``, which is what
    ``build_stage31_aam_dataset.load_factor_track`` consumes to produce
    per-window labels. Runs absent from the mapping stay constant.
    """
    if not factor or every_n <= 0:
        return {}
    if value is None:
        raise SystemExit("--step-factor requires --step-value")
    if not 0.0 < at_frac < 1.0:
        raise SystemExit(f"--step-at-frac must be strictly inside (0, 1), got {at_frac}")
    from_tick = int(round(at_frac * total_steps))
    if from_tick <= 0 or from_tick >= total_steps:
        raise SystemExit(
            f"--step-at-frac {at_frac} gives tick {from_tick} outside "
            f"(0, {total_steps}); the step would never be observed"
        )
    plan = {}
    for index, row in enumerate(rows):
        if index % every_n:
            continue
        plan[int(row["run_id"])] = {
            "from_tick": from_tick,
            "factors": {factor: float(value)},
        }
    return plan


def apply_batch_factors(env, rows: list[dict], baseline_masses, baseline_coms, body_idx: int) -> dict:
    """Apply exact manifest rows to vector slots without factor accumulation."""
    if len(rows) != env.num_envs:
        raise ValueError(f"expected {env.num_envs} factor rows, got {len(rows)}")
    system = [e_to_system_factors(row) for row in rows]
    masses = baseline_masses.clone()
    coms = baseline_coms.clone()
    for env_id, row in enumerate(rows):
        mass_scale, com_z = e_to_mechanical(row)
        masses[env_id, body_idx] *= mass_scale
        coms[env_id, body_idx, 2] += com_z
    env.robot.root_physx_view.set_masses(
        masses, torch.arange(env.num_envs, dtype=torch.long)
    )
    env.robot.root_physx_view.set_coms(
        coms, torch.arange(env.num_envs, dtype=torch.long)
    )
    env._command_rom_system_factors = torch.as_tensor(
        system, dtype=torch.float32, device=env.device
    )
    env._deployment_factor_keys = [str(row["run_id"]) for row in rows]
    env._deployment_factor_run_ids[:] = torch.as_tensor(
        [int(row["run_id"]) for row in rows], dtype=torch.long, device=env.device
    )
    env._deployment_factor_values[:] = torch.as_tensor(
        [[row[name] for name in ("flow", "compliance", "seal", "pressure", "mass_scale", "com_z")] for row in rows],
        dtype=torch.float32,
        device=env.device,
    )
    return {
        "keys": [str(row["run_id"]) for row in rows],
        "run_ids": [int(row["run_id"]) for row in rows],
        "values": [[float(row[name]) for name in ("flow", "compliance", "seal", "pressure", "mass_scale", "com_z")] for row in rows],
        "dims": ["flow", "compliance", "seal", "pressure", "mass_scale", "com_z"],
        "enabled": True,
        "isolation": {
            "e_to_policy": False,
            "e_to_aam": False,
            "true_z_to_policy": False,
            "true_z_to_aam": False,
        },
    }


def make_metadata(
    *, row: dict, plan: dict, env, factor_meta: dict, started: str,
    ended: str, elapsed: float, dls_parameters: dict, plant_contract: dict,
    controller_name: str, factor_schedule: list | None = None,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": row["run_id"],
        "domain": row["domain"],
        "split": row["split"],
        "random_seed": int(args.seed_base),
        "trajectory_schedule": plan,
        "segment_ids": plan["segments"],
        "physics_dt": float(env.cfg.sim.dt),
        "control_dt": DT,
        "rom_dt": MODEL_DT,
        "controller_gains": {
            "type": controller_name,
            "error_gain": 0.18,
            "velocity_feedforward": 1.0,
            "velocity_damping": 0.0,
            "max_desired_delta_m": 0.004,
            "max_action": 0.5,
            "action_smoothing": 0.20,
        },
        "dls_parameters": dls_parameters,
        "excitation_config": {
            "name": args.probe,
            "amplitude": float(args.probe_amplitude),
            "hold_steps": int(args.probe_hold_steps),
            "seed": int(args.probe_seed),
        },
        "factor_profile": "stage22_wide_recollected_final_task",
        # Consumed by build_stage31_aam_dataset.load_factor_track to produce
        # per-window labels. Absent/empty means a constant plant.
        "factor_schedule": factor_schedule or [],
        "hydraulic_draw": {k: row[k] for k in ("flow", "compliance", "seal", "pressure")},
        "mechanical_draw": {"mass_scale": row["mass_scale"], "com_z": row["com_z"]},
        "flow": row["flow"], "compliance": row["compliance"],
        "seal": row["seal"], "pressure": row["pressure"],
        "mass_scale": row["mass_scale"], "com_z": row["com_z"],
        "applied_system_factors": e_to_system_factors(row),
        "applied_mass_scale": row["mass_scale"],
        "applied_com_z": row["com_z"],
        "start_time": started,
        "end_time": ended,
        "task_name": TASK,
        "actuator": args.actuator,
        "rom_checkpoint": controller_name,
        "num_control_steps": int(plan["total_steps"]),
        "availability": {
            name: name not in UNAVAILABLE_SIGNALS
            for name in list(STATE_SIGNALS) + list(COMMAND_SIGNALS) + list(FLAG_SIGNALS)
        },
        "collection": {
            "controller": "DLS",
            "ppo_used": False,
            "data_source": "rom_deployment_plant",
            "id_fd_dataset_collected": False,
            "rom_external_command": "desired_delta_extension_4 + desired_delta_velocity_4",
            "rom_external_valve_command": False,
            "rom_valve_signal": "rom_predicted_valve_cmd is internal ROM output; requested_valve_cmd is a compatibility alias",
            "sd_action_contract": "previous_valve_4 + current_valve_4",
            "aam_feature_contract": "joint_q_5 + joint_qdot_5 + dP_4 + previous_valve_4",
            "vehicle_motion": bool(args.vehicle_motion),
            "elapsed_wall_s": float(elapsed),
            "factor_metadata": factor_meta,
            "plant_contract": plant_contract,
        },
    }


def main_batched(rows: list[dict], segments: list[dict], plan: dict, cfg, plant_contract: dict) -> None:
    """Collect several independent manifest rows in one Isaac process."""
    if args.num_envs < 2:
        raise ValueError("main_batched requires --num-envs >= 2")
    cfg.scene.replicate_physics = False
    cfg.deployment_factor_enabled = False
    env = gym.make(TASK, cfg=cfg).unwrapped
    if env._command_cylinder_rom is None:
        raise RuntimeError("ROM collector built without command ROM")
    body_idx = next(
        (i for i, name in enumerate(env.robot.body_names) if name == CUTTER_BODY), None
    )
    if body_idx is None:
        raise RuntimeError(f"{CUTTER_BODY} not found in robot bodies")
    baseline_masses = env.robot.root_physx_view.get_masses().detach().clone()
    baseline_coms = env.robot.root_physx_view.get_coms().detach().clone()
    dls = CylinderDLSController(
        env,
        CylinderDLSConfig(
            invert_learned_pipeline=False,
            error_gain=0.18,
            velocity_feedforward=1.0,
            velocity_damping=0.0,
            max_desired_delta_m=0.004,
            max_action=0.5,
            action_smoothing=0.20,
        ),
    )
    controller_name = controller_checkpoint(env)
    pending = []
    for row in rows:
        run_dir = OUT / "runs" / f"{row['run_id']:05d}"
        if run_dir.exists():
            if not args.resume:
                raise SystemExit(f"run exists: {run_dir}; use --resume")
            issues = validate_run(run_dir, manifest_row=row)
            if issues:
                raise SystemExit(f"invalid existing run {row['run_id']}: {issues}")
            print(f"run {row['run_id']:05d} -> SKIP", flush=True)
        else:
            pending.append(row)
    for start in range(0, len(pending), args.num_envs):
        active = pending[start : start + args.num_envs]
        batch = active + [active[-1]] * (args.num_envs - len(active))
        env.reset(seed=args.seed_base)
        factor_meta = apply_batch_factors(env, batch, baseline_masses, baseline_coms, body_idx)
        # Which of these runs change plant mid-episode, and when.
        step_plan = plan_factor_steps(
            batch, factor=args.step_factor, value=args.step_value,
            at_frac=args.step_at_frac, every_n=args.step_every_n,
            total_steps=plan["total_steps"],
        )
        stepped_slots = [
            (slot, int(r["run_id"])) for slot, r in enumerate(batch)
            if int(r["run_id"]) in step_plan
        ]
        step_fired = False
        if stepped_slots:
            print(
                f"  factor step: {args.step_factor} -> {args.step_value} at tick "
                f"{step_plan[stepped_slots[0][1]]['from_tick']} for "
                f"{len(stepped_slots)}/{len(batch)} envs",
                flush=True,
            )
        dls.reset()
        state = {name: [value] for name, value in snapshot_state_batch(env).items()}
        commands = {name: [] for name in COMMAND_SIGNALS}
        flags = {name: [] for name in FLAG_SIGNALS}
        times = [0.0]
        controller_actions = []
        probe_actions = []
        hidden = []
        singular_values = []
        started = datetime.now(timezone.utc).isoformat()
        t0 = time.time()
        for step in range(plan["total_steps"]):
            segment = next(s for s in segments if s["start"] <= step < s["end"])
            local = step - segment["start"]
            goal = torch.as_tensor(
                segment["reference"][local], dtype=torch.float32, device=env.device
            ).reshape(1, 4).expand(args.num_envs, -1)
            base_action, singular = dls.action(goal)
            probe = (
                segment["probe"][local]
                if segment["probe"] is not None
                else np.zeros(4, dtype=np.float32)
            )
            action = torch.clamp(
                base_action + torch.as_tensor(probe, dtype=torch.float32, device=env.device).reshape(1, 4),
                -1.0, 1.0,
            )
            if stepped_slots and not step_fired:
                from_tick = step_plan[stepped_slots[0][1]]["from_tick"]
                if step >= from_tick:
                    # Mutate ONLY the ROM hydraulic channel, only for the slots
                    # whose run is scheduled to change. The controller is never
                    # told; it must infer the change from the response.
                    factors = env._command_rom_system_factors.clone()
                    for slot, run_id in stepped_slots:
                        updated = dict(batch[slot])
                        updated.update(step_plan[run_id]["factors"])
                        factors[slot] = torch.as_tensor(
                            e_to_system_factors(updated),
                            dtype=factors.dtype, device=factors.device,
                        )
                    env._command_rom_system_factors = factors
                    step_fired = True
                    print(f"  plant stepped at control step {step}", flush=True)
            _obs, _reward, terminated, truncated, _info = env.step(action)
            if bool(torch.as_tensor(terminated).any()) or bool(torch.as_tensor(truncated).any()):
                raise RuntimeError(f"environment ended during batch at step {step}")
            tick = step % STEPS_PER_MODEL_TICK == 0
            command, quality = snapshot_command_batch(env, probe, tick)
            for name in COMMAND_SIGNALS:
                commands[name].append(command[name])
            for name in FLAG_SIGNALS:
                flags[name].append(quality[name])
            for name, value in snapshot_state_batch(env).items():
                state[name].append(value)
            times.append((step + 1) * DT)
            controller_actions.append(base_action.detach().cpu().numpy().astype(np.float32))
            probe_actions.append(np.broadcast_to(probe, (args.num_envs, 4)).copy())
            if tick:
                singular_values.append(singular.detach().cpu().numpy().astype(np.float32))
                rom_hidden = getattr(env._command_cylinder_rom, "last_hidden", None)
                if rom_hidden is not None:
                    hidden.append(rom_hidden.detach().cpu().numpy().astype(np.float32))
            if args.progress_steps > 0 and (step + 1) % args.progress_steps == 0:
                print(
                    f"batch {start // args.num_envs + 1}: {step + 1}/{plan['total_steps']} controls",
                    flush=True,
                )
        elapsed = time.time() - t0
        batched_arrays = {
            name: np.stack(values, axis=0) for name, values in state.items()
        }
        batched_arrays.update({name: np.stack(values, axis=0) for name, values in commands.items()})
        batched_arrays.update({name: np.stack(values, axis=0) for name, values in flags.items()})
        batched_arrays["time_s"] = np.asarray(times, dtype=np.float32)
        for env_id, row in enumerate(active):
            run_dir = OUT / "runs" / f"{row['run_id']:05d}"
            arrays = {
                name: value[:, env_id, ...] for name, value in batched_arrays.items()
                if name != "time_s"
            }
            arrays["time_s"] = batched_arrays["time_s"]
            schedule = step_plan.get(int(row["run_id"]))
            meta = make_metadata(
                factor_schedule=[schedule] if schedule else [],
                row=row, plan=plan, env=env,
                factor_meta={**factor_meta, "slot": env_id},
                started=started, ended=datetime.now(timezone.utc).isoformat(),
                elapsed=elapsed, dls_parameters={
                    "error_gain": 0.18, "velocity_feedforward": 1.0,
                    "velocity_damping": 0.0, "max_desired_delta_m": 0.004,
                    "max_action": 0.5, "action_smoothing": 0.20,
                    "invert_learned_pipeline": False,
                }, plant_contract=plant_contract, controller_name=controller_name,
            )
            write_run(run_dir, meta, arrays)
            sidecar = {
                "factor_e": np.asarray(
                    [row[k] for k in ("flow", "compliance", "seal", "pressure", "mass_scale", "com_z")],
                    dtype=np.float32,
                ),
                "classical_controller_action": np.stack(controller_actions)[:, env_id],
                "probe_excitation": np.stack(probe_actions)[:, env_id],
                "rom_tick_control_step": np.flatnonzero(arrays["rom_tick"][:, 0] > 0.5),
                "dls_singular_values": np.stack(singular_values)[:, env_id],
            }
            if hidden:
                sidecar["rom_gru_hidden"] = np.stack(hidden)[:, :, env_id]
            np.savez_compressed(run_dir / "privileged.npz", **sidecar)
            issues = validate_run(run_dir, manifest_row=row)
            if issues:
                raise SystemExit(f"run {row['run_id']:05d} failed validation: {issues}")
            print(
                f"run {row['run_id']:05d} [{row['domain']}/{row['split']}] "
                f"{plan['total_steps']} steps in {elapsed:.1f}s -> OK",
                flush=True,
            )
    env.close()
    simulation_app.close()


def main() -> None:
    rows = select_rows(load_manifest(MANIFEST))
    if not rows:
        raise SystemExit("no manifest rows selected")
    if args.probe_amplitude <= 0.0 or args.probe_amplitude > 0.25:
        raise SystemExit("--probe-amplitude must be in (0, 0.25]")
    if args.probe_hold_steps < 1:
        raise SystemExit("--probe-hold-steps must be positive")

    cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
    cfg.sole_teacher_mode = "off"
    cfg.enable_scripted_vehicle_motion = bool(args.vehicle_motion)
    cfg.deployment_factor_enabled = True
    cfg.deployment_factor_manifest = str(MANIFEST)
    cfg.deployment_factor_log = False
    cfg.randomize_start_phase = False
    plant_contract = assert_shared_rom_plant(cfg)
    segments, plan = build_plan(cfg)
    segments, plan = limit_plan(segments, plan, args.max_steps)
    cfg.episode_length_s = max(
        float(cfg.episode_length_s), plan["total_steps"] * DT + 5.0
    )

    OUT.mkdir(parents=True, exist_ok=True)
    runs_dir = OUT / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    dst_manifest = OUT / "manifest.csv"
    # A bounded pilot must carry only the rows it actually records; otherwise
    # freeze_dataset correctly interprets the untouched manifest rows as
    # missing trajectories. Production runs still copy the complete manifest.
    selected_manifest = io.StringIO()
    writer = csv.DictWriter(selected_manifest, fieldnames=list(rows[0].keys()), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    selected_bytes = selected_manifest.getvalue().encode()
    if dst_manifest.exists() and dst_manifest.read_bytes() != selected_bytes:
        raise SystemExit(f"output manifest differs from selected rows: {dst_manifest}")
    if not dst_manifest.exists():
        dst_manifest.write_bytes(selected_bytes)

    if args.num_envs > 1:
        main_batched(rows, segments, plan, cfg, plant_contract)
        return

    env = gym.make(TASK, cfg=cfg).unwrapped
    if env._command_cylinder_rom is None:
        raise RuntimeError("ROM collector built without command ROM")

    initial = env.reset(seed=args.seed_base)
    del initial
    dls = CylinderDLSController(
        env,
        CylinderDLSConfig(
            invert_learned_pipeline=False,
            error_gain=0.18,
            velocity_feedforward=1.0,
            velocity_damping=0.0,
            max_desired_delta_m=0.004,
            max_action=0.5,
            action_smoothing=0.20,
        ),
    )
    print(
        f"[CLASSICAL-COLLECT] task={TASK} actuator={args.actuator} "
        f"runs={len(rows)} steps={plan['total_steps']} probe={args.probe} "
        f"amplitude={args.probe_amplitude}",
        flush=True,
    )

    for row in rows:
        run_dir = runs_dir / f"{row['run_id']:05d}"
        if run_dir.exists():
            if not args.resume:
                raise SystemExit(f"run exists: {run_dir}; use --resume")
            issues = validate_run(run_dir, manifest_row=row)
            if issues:
                raise SystemExit(f"invalid existing run {row['run_id']}: {issues}")
            print(f"run {row['run_id']:05d} -> SKIP", flush=True)
            continue

        cfg.deployment_factor_key = str(row["run_id"])
        cfg.deployment_factor_domain = str(row["domain"])
        seed = args.seed_base if not args.vehicle_motion else args.seed_base + row["run_id"]
        env.reset(seed=seed)
        factor_meta = env.deployment_factor_metadata()
        if factor_meta["run_ids"] != [row["run_id"]]:
            raise RuntimeError(
                f"factor mismatch for run {row['run_id']}: {factor_meta['run_ids']}"
            )
        dls.reset()

        state = {name: [value] for name, value in snapshot_state(env).items()}
        commands = {name: [] for name in COMMAND_SIGNALS}
        flags = {name: [] for name in FLAG_SIGNALS}
        times = [0.0]
        controller_actions: list[np.ndarray] = []
        probe_actions: list[np.ndarray] = []
        hidden: list[np.ndarray] = []
        dls_singular_values: list[np.ndarray] = []
        ticks: list[int] = []
        start_time = datetime.now(timezone.utc).isoformat()
        started = time.time()

        for step in range(plan["total_steps"]):
            segment = next(s for s in segments if s["start"] <= step < s["end"])
            local = step - segment["start"]
            goal = torch.as_tensor(
                segment["reference"][local], dtype=torch.float32, device=env.device
            ).reshape(1, 4)
            base_action, singular = dls.action(goal)
            probe = (
                segment["probe"][local]
                if segment["probe"] is not None
                else np.zeros(4, dtype=np.float32)
            )
            action = base_action + torch.as_tensor(
                probe, dtype=torch.float32, device=env.device
            ).reshape(1, 4)
            action = torch.clamp(action, -1.0, 1.0)
            _obs, _reward, terminated, truncated, _info = env.step(action)
            if bool(torch.as_tensor(terminated).any()) or bool(torch.as_tensor(truncated).any()):
                raise RuntimeError(f"environment ended during run {row['run_id']} at step {step}")
            tick = step % STEPS_PER_MODEL_TICK == 0
            command, quality = snapshot_command(env, probe, tick)
            for name in COMMAND_SIGNALS:
                commands[name].append(command[name])
            for name in FLAG_SIGNALS:
                flags[name].append(quality[name])
            for name, value in snapshot_state(env).items():
                state[name].append(value)
            times.append((step + 1) * DT)
            controller_actions.append(base_action[0].detach().cpu().numpy().astype(np.float32))
            probe_actions.append(np.asarray(probe, dtype=np.float32))
            if tick:
                tick_hidden = hidden_sidecar(env, True)
                if tick_hidden is not None:
                    hidden.append(tick_hidden)
                dls_singular_values.append(
                    singular[0].detach().cpu().numpy().astype(np.float32)
                )
                ticks.append(step)

            if not torch.isfinite(singular).all():
                raise RuntimeError(f"non-finite DLS diagnostic at step {step}")
            if args.progress_steps > 0 and (step + 1) % args.progress_steps == 0:
                print(
                    f"run {row['run_id']:05d}: {step + 1}/{plan['total_steps']} controls",
                    flush=True,
                )

        arrays = {name: np.stack(values, axis=0) for name, values in state.items()}
        arrays.update({name: np.stack(values, axis=0) for name, values in commands.items()})
        arrays.update({name: np.stack(values, axis=0) for name, values in flags.items()})
        arrays["time_s"] = np.asarray(times, dtype=np.float32)

        dls_parameters = {
            "error_gain": 0.18,
            "velocity_feedforward": 1.0,
            "velocity_damping": 0.0,
            "max_desired_delta_m": 0.004,
            "max_action": 0.5,
            "action_smoothing": 0.20,
            "invert_learned_pipeline": False,
        }
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "run_id": row["run_id"],
            "domain": row["domain"],
            "split": row["split"],
            "random_seed": seed,
            "trajectory_schedule": plan,
            "segment_ids": [
                {k: v for k, v in segment.items() if k not in {"reference", "probe"}}
                for segment in segments
            ],
            "physics_dt": float(env.cfg.sim.dt),
            "control_dt": DT,
            "rom_dt": MODEL_DT,
            "controller_gains": {
                "type": "dls",
                "error_gain": 0.18,
                "velocity_feedforward": 1.0,
                "velocity_damping": 0.0,
                "max_desired_delta_m": 0.004,
                "max_action": 0.5,
                "action_smoothing": 0.20,
            },
            "dls_parameters": dls_parameters,
            "excitation_config": {
                "name": args.probe,
                "amplitude": float(args.probe_amplitude),
                "hold_steps": int(args.probe_hold_steps),
                "seed": int(args.probe_seed),
            },
            "factor_profile": "stage22_wide_recollected_final_task",
            "hydraulic_draw": {k: row[k] for k in ("flow", "compliance", "seal", "pressure")},
            "mechanical_draw": {"mass_scale": row["mass_scale"], "com_z": row["com_z"]},
            "flow": row["flow"],
            "compliance": row["compliance"],
            "seal": row["seal"],
            "pressure": row["pressure"],
            "mass_scale": row["mass_scale"],
            "com_z": row["com_z"],
            "applied_system_factors": [
                row["compliance"], row["flow"], row["pressure"],
                row["seal"], row["seal"], row["seal"], row["seal"],
            ],
            "applied_mass_scale": row["mass_scale"],
            "applied_com_z": row["com_z"],
            "start_time": start_time,
            "end_time": datetime.now(timezone.utc).isoformat(),
            "task_name": TASK,
            "actuator": args.actuator,
            "rom_checkpoint": controller_checkpoint(env),
            "num_control_steps": plan["total_steps"],
            "availability": {
                name: name not in UNAVAILABLE_SIGNALS
                for name in list(STATE_SIGNALS) + list(COMMAND_SIGNALS) + list(FLAG_SIGNALS)
            },
            "collection": {
                "controller": "DLS",
                "ppo_used": False,
                "data_source": "rom_deployment_plant",
                "id_fd_dataset_collected": False,
                "rom_external_command": "desired_delta_extension_4 + desired_delta_velocity_4",
                "rom_external_valve_command": False,
                "rom_valve_signal": "rom_predicted_valve_cmd is internal ROM output; requested_valve_cmd is a compatibility alias",
                "sd_action_contract": "previous_valve_4 + current_valve_4",
                "aam_feature_contract": "joint_q_5 + joint_qdot_5 + dP_4 + previous_valve_4",
                "vehicle_motion": bool(args.vehicle_motion),
                "elapsed_wall_s": time.time() - started,
                "factor_metadata": factor_meta,
                "plant_contract": plant_contract,
            },
        }
        write_run(run_dir, metadata, arrays)
        sidecar = {
            "factor_e": np.asarray(
                [row[k] for k in ("flow", "compliance", "seal", "pressure", "mass_scale", "com_z")],
                dtype=np.float32,
            ),
            "classical_controller_action": np.stack(controller_actions),
            "probe_excitation": np.stack(probe_actions),
            "rom_tick_control_step": np.asarray(ticks, dtype=np.int64),
        }
        if hidden:
            sidecar["rom_gru_hidden"] = np.stack(hidden)
        if dls_singular_values:
            sidecar["dls_singular_values"] = np.stack(dls_singular_values)
        np.savez_compressed(run_dir / "privileged.npz", **sidecar)

        issues = validate_run(run_dir, manifest_row=row)
        if issues:
            raise SystemExit(f"run {row['run_id']} failed validation: {issues}")
        print(
            f"run {row['run_id']:05d} [{row['domain']}/{row['split']}] "
            f"{plan['total_steps']} steps in {time.time() - started:.1f}s -> OK",
            flush=True,
        )

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
