#!/usr/bin/env python3
"""Deterministic, isolated Jacobian gates for the cylinder ID/FD pipeline.

The old probe only differentiated the end-to-end action path, so a weak ID
Jacobian could be misreported as an FD failure.  This probe evaluates each model
in its own input coordinates at fixed states sampled across the original ID
validation distribution:

  adapter:  d(target_delta_x) / d(action)
  ID:       d(valve) / d(target_delta_x), holding target velocity fixed
  FD:       d(fd_delta) / d(valve)
  composed: d(fd_delta) / d(action), using the deployed adapter -> ID -> FD path

Acceptance criteria:
  * every isolated and composed Jacobian has numerical rank 4 (hard)
  * FD condition < 10, ID condition < 15, composed condition < 100 (soft)

The former 1%-of-largest-singular-value "rank" is still printed as an
effective-rank diagnostic, but it is not mathematical rank and is not a hard
gate; damped least squares is specifically intended for those weak directions.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_poses", type=int, default=16)
parser.add_argument("--seed", type=int, default=17)
parser.add_argument("--action_eps", type=float, default=0.005)
parser.add_argument("--target_eps_m", type=float, default=1.0e-4)
parser.add_argument("--valve_eps", type=float, default=0.01)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs != 1:
    raise ValueError("The deterministic pose probe requires --num_envs 1")
if not 10 <= args.num_poses <= 20:
    raise ValueError("--num_poses must be in [10, 20]")
if min(args.action_eps, args.target_eps_m, args.valve_eps) <= 0.0:
    raise ValueError("All finite-difference epsilons must be positive")

random.seed(args.seed)
np.random.seed(args.seed)

app = AppLauncher(args)
sim_app = app.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

import ID_FD_pipline.tasks  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from kevin_integration.new_observations_pipeline.ppo1.joint_to_cylinder_adapter import (  # noqa: E402
    CylinderReference,
)


TASK = "Kevin-Mulag-IDFD-CylinderSpace-PLAY-v0"
cfg = parse_env_cfg(TASK, device=args.device, num_envs=1)
cfg.seed = int(args.seed)
cfg.enable_scripted_vehicle_motion = False
env = gym.make(TASK, cfg=cfg).unwrapped


def _load_pose_bank() -> torch.Tensor:
    """Vary real arm extensions while holding every other state at fixed reset."""
    common = sys.modules.get("id_cylinder_common")
    if common is None:
        raise RuntimeError("ID common module was not loaded by CylinderIDRuntime")
    candidates = [
        Path(str(env._cylinder_id.checkpoint.get("data_path", ""))).expanduser(),
        Path(common.DEFAULT_DATA_PATH),
    ]
    data_path = next((path for path in candidates if path.is_file()), None)
    if data_path is None:
        raise FileNotFoundError(f"Could not find original ID data; checked {candidates}")
    data = common.load_id_data(data_path, load_train=False, load_val=True)
    raw = common.prepare_features_for_checkpoint(
        data, env._cylinder_id.checkpoint, split="val"
    )
    if raw.shape[1] != env._cylinder_id.input_dim:
        raise RuntimeError(f"Prepared pose width {raw.shape[1]} != ID width {env._cylinder_id.input_dim}")

    # Work on an evenly spaced candidate subset, then use deterministic farthest
    # point sampling in normalized extension space.  Every returned full row is a
    # real validation state, including dP, force, and previous valve command.
    pool_size = min(4096, len(raw))
    pool_indices = np.linspace(0, len(raw) - 1, pool_size, dtype=np.int64)
    pool = raw[pool_indices]
    mean = env._cylinder_id.normalizers["x_mean"].detach().cpu().numpy()
    std = env._cylinder_id.normalizers["x_std"].detach().cpu().numpy()
    extension_z = (pool[:, :4] - mean[:4]) / np.maximum(std[:4], 1.0e-8)
    selected = [int(np.argmin(np.sum(extension_z**2, axis=1)))]
    min_distance = np.sum((extension_z - extension_z[selected[0]]) ** 2, axis=1)
    for _ in range(1, args.num_poses):
        next_index = int(np.argmax(min_distance))
        selected.append(next_index)
        distance = np.sum((extension_z - extension_z[next_index]) ** 2, axis=1)
        min_distance = np.minimum(min_distance, distance)
    poses = torch.as_tensor(pool[selected], dtype=torch.float32, device=env.device)
    reset_x, reset_xdot = env._refresh_cylinder_state()
    poses[:, 4:8] = reset_xdot[0]
    poses[:, 8:12] = env._dP[0]
    poses[:, 12:16] = env._force[0]
    poses[:, 16:24] = 0.0
    if poses.shape[1] == 28:
        poses[:, 24:28] = 0.0
    return poses


def _reference(state: torch.Tensor, target_dx: torch.Tensor, target_dxdot: torch.Tensor) -> CylinderReference:
    x = state[:, 0:4]
    xdot = state[:, 4:8]
    return CylinderReference(
        x_current=x,
        xdot_current=xdot,
        x_ref=x + target_dx,
        xdot_ref=xdot + target_dxdot,
        target_delta_x=target_dx,
        target_delta_xdot=target_dxdot,
    )


def _adapter(action: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    converted = env._action_to_cylinder_target_adapter.convert(
        action=action,
        x_current=state[:, 0:4],
        xdot_current=state[:, 4:8],
    )
    return converted.reference.target_delta_x


def _id_input(target_dx: torch.Tensor, target_dxdot: torch.Tensor, state: torch.Tensor):
    reference = _reference(state, target_dx, target_dxdot)
    previous_valve = state[:, 24:28] if state.shape[1] == 28 else torch.zeros_like(target_dx)
    model_input = env._cylinder_adapter.build_id_input(
        reference,
        dP=state[:, 8:12],
        force=state[:, 12:16],
        prev_valve_cmd=previous_valve,
        input_dim=env._cylinder_id.input_dim,
    )
    return model_input, previous_valve


def _id(target_dx: torch.Tensor, target_dxdot: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    """Unclipped model output: the isolated learned ID stage."""
    model_input, previous_valve = _id_input(target_dx, target_dxdot, state)
    normalizers = env._cylinder_id.normalizers
    x_norm = (model_input - normalizers["x_mean"].reshape(1, -1)) / normalizers["x_std"].reshape(1, -1)
    valve = env._cylinder_id.model(x_norm) * normalizers["y_std"].reshape(1, -1) + normalizers["y_mean"].reshape(1, -1)
    if "delta" in env._cylinder_id.target_mode:
        valve = valve + previous_valve
    return valve


def _id_deployed(target_dx: torch.Tensor, target_dxdot: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    """Runtime output including command limits, used by the real composition."""
    model_input, previous_valve = _id_input(target_dx, target_dxdot, state)
    valve = env._cylinder_id.predict(model_input, prev_valve_cmd=previous_valve)
    if bool(getattr(env.cfg, "id_residual_over_zero_target", False)):
        zero_target_input = model_input.detach().clone()
        zero_target_input[:, 16:24] = 0.0
        zero_valve = env._cylinder_id.predict(
            zero_target_input, prev_valve_cmd=previous_valve
        )
        valve = valve - zero_valve
    return valve


def _fd(valve: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    delta = env._cylinder_fd.step(
        valve_cmd=valve,
        extension=state[:, 0:4],
        extension_velocity=state[:, 4:8],
        dP=state[:, 8:12],
        force=state[:, 12:16],
    ).delta_extension
    if bool(getattr(env.cfg, "fd_residual_over_zero_valve", False)):
        zero_delta = env._cylinder_fd.step(
            valve_cmd=torch.zeros_like(valve),
            extension=state[:, 0:4],
            extension_velocity=state[:, 4:8],
            dP=state[:, 8:12],
            force=state[:, 12:16],
        ).delta_extension
        delta = delta - zero_delta
    return delta


def _composed(action: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    converted = env._action_to_cylinder_target_adapter.convert(
        action=action,
        x_current=state[:, 0:4],
        xdot_current=state[:, 4:8],
    )
    valve = _id_deployed(
        converted.reference.target_delta_x,
        converted.reference.target_delta_xdot,
        state,
    )
    return _fd(valve, state)


@torch.no_grad()
def _central_jacobian(function, state: torch.Tensor, eps: float) -> torch.Tensor:
    columns = []
    for column in range(4):
        plus = torch.zeros((1, 4), dtype=torch.float32, device=env.device)
        minus = torch.zeros_like(plus)
        plus[:, column] = eps
        minus[:, column] = -eps
        columns.append(((function(plus, state) - function(minus, state)) / (2.0 * eps))[0])
    return torch.stack(columns, dim=1)


def _metrics(jacobian: torch.Tensor) -> tuple[int, int, float, list[float]]:
    singular_values = torch.linalg.svdvals(jacobian.float())
    rank = int(torch.linalg.matrix_rank(jacobian.float()).item())
    effective_rank = int((singular_values > 0.01 * singular_values[0]).sum().item()) if singular_values[0] > 0 else 0
    cond = float((singular_values[0] / singular_values[-1].clamp(min=1.0e-12)).item())
    return rank, effective_rank, cond, singular_values.detach().cpu().tolist()


try:
    env.reset(seed=int(args.seed))
    if env._cylinder_id is None or env._cylinder_fd is None:
        raise RuntimeError("Both cylinder ID and FD runtimes are required")
    pose_bank = _load_pose_bank()
    reset_x, _ = env._refresh_cylinder_state()
    print(
        f"[PROBE] seed={args.seed} poses={len(pose_bank)} fixed_reset_x={reset_x[0].cpu().tolist()}",
        flush=True,
    )
    print(f"[PROBE] ID={env._cylinder_id.checkpoint_path}", flush=True)
    print(f"[PROBE] FD={env._cylinder_fd.checkpoint_path}", flush=True)

    results: dict[str, list[tuple[int, float]]] = {
        "adapter": [], "ID": [], "FD": [], "composed": []
    }
    zero_target_velocity = torch.zeros((1, 4), dtype=torch.float32, device=env.device)
    stages = (
        ("adapter", lambda perturb, state: _adapter(perturb, state), args.action_eps),
        ("ID", lambda perturb, state: _id(perturb, zero_target_velocity, state), args.target_eps_m),
        ("FD", lambda perturb, state: _fd(perturb, state), args.valve_eps),
        ("composed", lambda perturb, state: _composed(perturb, state), args.action_eps),
    )
    for pose_index, state in enumerate(pose_bank):
        state = state.reshape(1, -1)
        extension = state[0, :4].detach().cpu().tolist()
        print(f"\n[POSE {pose_index:02d}] extension={extension}", flush=True)
        for stage_name, function, eps in stages:
            jacobian = _central_jacobian(function, state, float(eps))
            rank, effective_rank, cond, singular_values = _metrics(jacobian)
            results[stage_name].append((rank, cond))
            sv_text = ", ".join(f"{value:.2e}" for value in singular_values)
            print(
                f"[STAGE] pose={pose_index:02d} {stage_name:8s} rank={rank}/4 "
                f"effective_rank_1pct={effective_rank}/4 cond={cond:8.2f} SVs=[{sv_text}]",
                flush=True,
            )

    print("\n========== WORST-CASE STAGE SUMMARY ==========", flush=True)
    for stage_name, stage_results in results.items():
        minimum_rank = min(rank for rank, _ in stage_results)
        maximum_cond = max(cond for _, cond in stage_results)
        worst_pose = max(range(len(stage_results)), key=lambda index: stage_results[index][1])
        print(
            f"[WORST] {stage_name:8s} min_rank={minimum_rank}/4 "
            f"max_cond={maximum_cond:.2f} pose={worst_pose:02d}",
            flush=True,
        )

    adapter_rank_pass = all(rank == 4 for rank, _ in results["adapter"])
    fd_pass = all(rank == 4 for rank, _ in results["FD"])
    id_pass = all(rank == 4 for rank, _ in results["ID"])
    composed_rank_pass = all(rank == 4 for rank, _ in results["composed"])
    fd_cond_pass = all(cond < 10.0 for _, cond in results["FD"])
    id_cond_pass = all(cond < 15.0 for _, cond in results["ID"])
    composed_cond_pass = all(cond < 100.0 for _, cond in results["composed"])
    hard_pass = adapter_rank_pass and fd_pass and id_pass and composed_rank_pass
    print(f"[GATE][FD-HARD] {'PASS' if fd_pass else 'FAIL'} numerical rank=4 at every pose", flush=True)
    print(f"[GATE][FD-SOFT] {'PASS' if fd_cond_pass else 'WARN'} cond<10 at every pose", flush=True)
    print(f"[GATE][ID-HARD] {'PASS' if id_pass else 'FAIL'} numerical rank=4 at every pose", flush=True)
    print(f"[GATE][ID-SOFT] {'PASS' if id_cond_pass else 'WARN'} cond<15 at every pose", flush=True)
    print(
        f"[GATE][COMPOSED-HARD] {'PASS' if composed_rank_pass else 'FAIL'} rank=4 at every pose",
        flush=True,
    )
    print(
        f"[GATE][COMPOSED-SOFT] {'PASS' if composed_cond_pass else 'WARN'} cond<100 at every pose",
        flush=True,
    )
    print(f"[GATE] {'PASS' if hard_pass else 'FAIL'}", flush=True)
finally:
    env.close()
    sim_app.close()
