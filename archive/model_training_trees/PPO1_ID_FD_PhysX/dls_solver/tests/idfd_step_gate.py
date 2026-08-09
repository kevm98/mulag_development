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
p.add_argument("--return_steps", type=int, default=0,
               help="After the outbound gate, reverse to the initial pose and hold")
p.add_argument("--goal_delta_m", type=float, default=0.08)
p.add_argument("--direction", choices=("auto", "positive", "negative"), default="auto")
p.add_argument("--residual_gains", default="20,10,2,2")
p.add_argument(
    "--model_mode",
    choices=("absolute", "paired", "legacy_residual"),
    default="absolute",
)
p.add_argument("--tolerance_m", type=float, default=0.01)
p.add_argument("--error_gain", type=float, default=0.04)
p.add_argument("--jac_eps", type=float, default=0.005)
p.add_argument("--action_smoothing", type=float, default=0.5)
p.add_argument("--reference_blend", type=float, default=0.0)
p.add_argument("--target_lead_m", type=float, default=-1.0,
               help="Override cylinder_target_max_lead_m; negative keeps config")
p.add_argument("--backend", choices=("idfd", "direct"), default="idfd")
p.add_argument("--direct_ramp_step_m", type=float, default=0.0,
               help="Rate-limit DirectCylinderDriver target; zero applies the full jump")
p.add_argument("--invert_pipeline", action="store_true")
p.add_argument("--id_checkpoint", default="")
p.add_argument("--fd_checkpoint", default="")
p.add_argument("--base_rom_checkpoint", default="")
p.add_argument("--command_rom_checkpoint", default="")
p.add_argument("--dynamics_model", choices=("fd", "base_rom", "command_rom"), default="fd")
p.add_argument(
    "--hydraulic_mode",
    choices=("zero", "training_mean", "fd_dp_state"),
    default="training_mean",
)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
if a.num_envs != 4:
    raise ValueError("ID/FD step gate uses exactly four environments, one per cylinder")

launcher = AppLauncher(a)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

torch.manual_seed(a.seed)
import PPO1_ID_FD_PhysX.tasks  # noqa: E402,F401
from PPO1_ID_FD_PhysX.dls_solver.cylinder import CylinderDLSConfig, CylinderDLSController  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from PPO1_ID_FD_PhysX.dls_solver.direct_drive import DirectCylinderDriver  # noqa: E402

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
cfg.cylinder_idfd_reference_blend = float(a.reference_blend)
if a.target_lead_m >= 0.0:
    cfg.cylinder_target_max_lead_m = float(a.target_lead_m)
cfg.cylinder_hydraulic_feature_mode = a.hydraulic_mode
cfg.cylinder_dynamics_model = a.dynamics_model
if a.dynamics_model == "base_rom":
    cfg.low_level_servo_mode = "none"
if a.id_checkpoint:
    cfg.id_cylinder_checkpoint = str(Path(a.id_checkpoint).expanduser().resolve())
if a.fd_checkpoint:
    cfg.fd_cylinder_checkpoint = str(Path(a.fd_checkpoint).expanduser().resolve())
if a.base_rom_checkpoint:
    cfg.base_rom_checkpoint = str(Path(a.base_rom_checkpoint).expanduser().resolve())
if a.command_rom_checkpoint:
    cfg.command_rom_checkpoint = str(Path(a.command_rom_checkpoint).expanduser().resolve())
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
        if a.direction == "positive":
            direction = 1.0
        elif a.direction == "negative":
            direction = -1.0
        else:
            direction = 1.0 if plus_room >= float(a.goal_delta_m) else -1.0
        chosen_room = plus_room if direction > 0 else minus_room
        if chosen_room < float(a.goal_delta_m):
            raise RuntimeError(
                f"Cylinder {axis} has only {chosen_room:.4f} m room in requested {a.direction} direction"
            )
        if max(plus_room, minus_room) < float(a.goal_delta_m):
            raise RuntimeError(f"Cylinder {axis} has insufficient room for the requested step")
        goal[axis, axis] += direction * float(a.goal_delta_m)
        directions.append(int(direction))

    controller = CylinderDLSController(
        env,
        CylinderDLSConfig(
            jac_eps=float(a.jac_eps),
            invert_learned_pipeline=bool(a.invert_pipeline),
            error_gain=float(a.error_gain),
            max_desired_delta_m=0.01,
            max_action=1.0,
            action_smoothing=float(a.action_smoothing),
        ),
    )
    direct_driver = DirectCylinderDriver(env)
    print(
        f"[IDFD-STEP] mode={a.model_mode} gains={gains} directions={directions} "
        f"error_gain={a.error_gain} smoothing={a.action_smoothing} "
        f"invert={a.invert_pipeline} hydraulic_mode={a.hydraulic_mode} "
        f"reference_blend={a.reference_blend:g} target_lead_m={cfg.cylinder_target_max_lead_m:g} "
        f"backend={a.backend} dynamics={a.dynamics_model}",
        flush=True,
    )
    axis_index = torch.arange(4, device=env.device)
    direction_t = torch.tensor(directions, device=env.device, dtype=start.dtype)
    previous_measured = start.clone()
    previous_target = controller.target_x().clone()
    previous_progress = torch.zeros(4, device=env.device, dtype=start.dtype)
    direct_target = start.clone()
    max_progress = previous_progress.clone()
    max_measured_step = previous_progress.clone()
    max_target_step = previous_progress.clone()
    backward_steps = torch.zeros(4, device=env.device, dtype=torch.long)
    for step in range(int(a.steps)):
        if a.backend == "direct":
            if a.direct_ramp_step_m > 0.0:
                direct_target = direct_target + torch.clamp(
                    goal - direct_target,
                    -float(a.direct_ramp_step_m),
                    float(a.direct_ramp_step_m),
                )
            else:
                direct_target = goal
            direct_driver.step(direct_target, substeps=2)
        else:
            controller.step(goal)
        measured, _ = env._refresh_cylinder_state()
        target = direct_target if a.backend == "direct" else controller.target_x()
        progress = (measured[axis_index, axis_index] - start[axis_index, axis_index]) * direction_t
        progress_step = progress - previous_progress
        backward_steps += progress_step < -1.0e-4
        max_progress = torch.maximum(max_progress, progress)
        max_measured_step = torch.maximum(
            max_measured_step, (measured - previous_measured).abs().amax(dim=1)
        )
        max_target_step = torch.maximum(
            max_target_step, (target - previous_target).abs().amax(dim=1)
        )
        previous_progress = progress
        previous_measured = measured.clone()
        previous_target = target.clone()
        if step % 100 == 0 or step == int(a.steps) - 1:
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
    target = direct_target if a.backend == "direct" else controller.target_x()
    errors = (goal - measured).abs().amax(dim=1)
    gaps = (target - measured).abs().amax(dim=1)
    overshoot = torch.clamp(max_progress - float(a.goal_delta_m), min=0.0)
    backward_fraction = backward_steps.float() / max(int(a.steps), 1)
    smooth = (
        (overshoot < 0.005)
        & (backward_fraction < 0.10)
        & (max_measured_step < 0.01)
        & ((max_target_step < 0.01) if a.backend == "idfd" else torch.ones_like(overshoot, dtype=torch.bool))
    )
    # DirectCylinderDriver writes the full goal instantly and is the plant comparator, not the
    # smooth deployment path. Its target gap is identical to plant error by construction.
    passed_axes = (errors < float(a.tolerance_m)) & (gaps < 0.02) & smooth
    return_pass = True
    if int(a.return_steps) > 0:
        if a.backend != "idfd":
            raise ValueError("--return_steps currently exercises the ID/FD controller only")
        return_max_measured_step = torch.zeros(4, device=env.device)
        return_max_target_step = torch.zeros(4, device=env.device)
        prev_m = measured.clone()
        prev_t = target.clone()
        for _ in range(int(a.return_steps)):
            controller.step(start)
            measured, _ = env._refresh_cylinder_state()
            target = controller.target_x()
            return_max_measured_step = torch.maximum(
                return_max_measured_step, (measured - prev_m).abs().amax(dim=1)
            )
            return_max_target_step = torch.maximum(
                return_max_target_step, (target - prev_t).abs().amax(dim=1)
            )
            prev_m = measured.clone()
            prev_t = target.clone()
        return_error = (start - measured).abs().amax(dim=1)
        return_gap = (target - measured).abs().amax(dim=1)
        return_target_error = (start - target).abs().amax(dim=1)
        return_axes = (
            (return_error < float(a.tolerance_m))
            & (return_gap < 0.02)
            & (return_target_error < 0.001)
            & (return_max_measured_step < 0.01)
            & (return_max_target_step < 0.01)
        )
        return_pass = bool(return_axes.all().item())
        print(
            f"[IDFD-STEP][RETURN] errors={return_error.cpu().tolist()} "
            f"target_errors={return_target_error.cpu().tolist()} gaps={return_gap.cpu().tolist()} "
            f"max_measured_step={return_max_measured_step.cpu().tolist()} "
            f"max_target_step={return_max_target_step.cpu().tolist()} axes={return_axes.cpu().tolist()}",
            flush=True,
        )
    passed = bool(passed_axes.all().item()) and return_pass
    print(
        f"[IDFD-STEP][SMOOTH] overshoot={overshoot.cpu().tolist()} "
        f"backward_fraction={backward_fraction.cpu().tolist()} "
        f"max_measured_step={max_measured_step.cpu().tolist()} "
        f"max_target_step={max_target_step.cpu().tolist()} smooth={smooth.cpu().tolist()}",
        flush=True,
    )
    for env_id in range(4):
        print(
            f"[IDFD-STEP][FINAL env={env_id}] goal={goal[env_id].cpu().tolist()} "
            f"target={target[env_id].cpu().tolist()} measured={measured[env_id].cpu().tolist()} "
            f"error={(goal[env_id]-measured[env_id]).cpu().tolist()}",
            flush=True,
        )
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
