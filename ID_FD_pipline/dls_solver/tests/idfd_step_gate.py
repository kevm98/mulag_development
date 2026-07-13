#!/usr/bin/env python3
"""Four-axis large-step gate for calibrated ID -> residual FD actuation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--num_envs", type=int, default=4)
p.add_argument("--seed", type=int, default=17)
p.add_argument("--steps", type=int, default=600)
p.add_argument("--goal_delta_m", type=float, default=0.08)
p.add_argument("--residual_gains", default="20,10,2,2")
p.add_argument(
    "--model_mode",
    choices=("absolute", "paired", "legacy_residual"),
    default="absolute",
)
p.add_argument("--tolerance_m", type=float, default=0.01)
p.add_argument("--error_gain", type=float, default=0.04)
p.add_argument("--action_smoothing", type=float, default=0.5)
p.add_argument("--invert_pipeline", action="store_true")
p.add_argument("--id_checkpoint", default="")
p.add_argument("--fd_checkpoint", default="")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
if a.num_envs != 4:
    raise ValueError("ID/FD step gate uses exactly four environments, one per cylinder")

launcher = AppLauncher(a)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

torch.manual_seed(a.seed)
import ID_FD_pipline.tasks  # noqa: E402,F401
from ID_FD_pipline.dls_solver.cylinder import CylinderDLSConfig, CylinderDLSController  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

gains = [float(value) for value in a.residual_gains.split(",")]
if len(gains) != 4:
    raise ValueError("--residual_gains requires four comma-separated values")

TASK = "Kevin-Mulag-IDFD-CylinderSpace-v0"
cfg = parse_env_cfg(TASK, device=a.device, num_envs=4)
cfg.seed = int(a.seed)
cfg.enable_scripted_vehicle_motion = False
cfg.episode_length_s = 300.0
cfg.cylinder_success_hold_steps = 100000
cfg.id_residual_over_zero_target = a.model_mode != "absolute"
cfg.fd_residual_over_zero_valve = a.model_mode == "legacy_residual"
cfg.paired_id_fd_residual = a.model_mode == "paired"
cfg.cylinder_target_integrates_from_previous_target = a.model_mode != "absolute"
cfg.cylinder_fd_residual_output_gain = gains
if a.id_checkpoint:
    cfg.id_cylinder_checkpoint = str(Path(a.id_checkpoint).expanduser().resolve())
if a.fd_checkpoint:
    cfg.fd_cylinder_checkpoint = str(Path(a.fd_checkpoint).expanduser().resolve())
env = gym.make(TASK, cfg=cfg).unwrapped


try:
    env.reset(seed=int(a.seed))
    zero = torch.zeros((4, 4), device=env.device)
    for _ in range(30):
        env.step(zero)
    start, _ = env._refresh_cylinder_state()
    scaling = env._cylinder_adapter.scaling
    lo = scaling.learned_min.to(env.device).reshape(1, 4)
    hi = scaling.learned_max.to(env.device).reshape(1, 4)
    goal = start.clone()
    directions = []
    for axis in range(4):
        plus_room = float((hi[0, axis] - start[axis, axis]).item())
        minus_room = float((start[axis, axis] - lo[0, axis]).item())
        direction = 1.0 if plus_room >= float(a.goal_delta_m) else -1.0
        if max(plus_room, minus_room) < float(a.goal_delta_m):
            raise RuntimeError(f"Cylinder {axis} has insufficient room for the requested step")
        goal[axis, axis] += direction * float(a.goal_delta_m)
        directions.append(int(direction))

    controller = CylinderDLSController(
        env,
        CylinderDLSConfig(
            invert_learned_pipeline=bool(a.invert_pipeline),
            error_gain=float(a.error_gain),
            max_desired_delta_m=0.01,
            max_action=1.0,
            action_smoothing=float(a.action_smoothing),
        ),
    )
    print(
        f"[IDFD-STEP] mode={a.model_mode} gains={gains} directions={directions} "
        f"error_gain={a.error_gain} smoothing={a.action_smoothing} "
        f"invert={a.invert_pipeline}",
        flush=True,
    )
    for step in range(int(a.steps)):
        controller.step(goal)
        if step % 100 == 0 or step == int(a.steps) - 1:
            measured, _ = env._refresh_cylinder_state()
            target = controller.target_x()
            axis_error = torch.stack(
                [(goal[i, i] - measured[i, i]).abs() for i in range(4)]
            )
            cross_error = (goal - measured).abs()
            print(
                f"[IDFD-STEP] step={step:04d} axis_error={axis_error.cpu().tolist()} "
                f"max_error={cross_error.amax(dim=1).cpu().tolist()} "
                f"target_gap={(target-measured).abs().amax(dim=1).cpu().tolist()} "
                f"fd_absmax={env._fd_delta_corrected.abs().amax(dim=1).cpu().tolist()}",
                flush=True,
            )

    measured, _ = env._refresh_cylinder_state()
    target = controller.target_x()
    errors = (goal - measured).abs().amax(dim=1)
    gaps = (target - measured).abs().amax(dim=1)
    passed_axes = (errors < float(a.tolerance_m)) & (gaps < 0.02)
    passed = bool(passed_axes.all().item())
    print(
        f"[IDFD-STEP][GATE] {'PASS' if passed else 'FAIL'} "
        f"errors={errors.cpu().tolist()} gaps={gaps.cpu().tolist()} axes={passed_axes.cpu().tolist()}",
        flush=True,
    )
    if not passed:
        raise SystemExit(2)
finally:
    env.close()
    simulation_app.close()
