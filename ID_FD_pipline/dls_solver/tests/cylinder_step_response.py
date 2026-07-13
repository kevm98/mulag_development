#!/usr/bin/env python3
"""Trace a large cylinder step through DLS -> ID -> FD -> target -> PhysX."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=17)
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--settle_steps", type=int, default=30)
parser.add_argument("--goal_delta_m", type=float, default=0.10)
parser.add_argument("--axis", type=int, choices=range(4), default=2)
parser.add_argument("--direction", type=int, choices=(-1, 1), default=1)
parser.add_argument("--fd_mode", choices=("residual", "full", "both"), default="both")
parser.add_argument("--fd_gain", type=float, default=1.0)
parser.add_argument(
    "--residual_gains",
    type=str,
    default="1,1,1,1",
    help="Comma-separated per-cylinder gains applied to FD(valve)-FD(zero).",
)
parser.add_argument("--jac_eps", type=float, default=0.005)
parser.add_argument("--damping_ratio", type=float, default=0.10)
parser.add_argument("--max_desired_delta_m", type=float, default=0.0015)
parser.add_argument(
    "--control_mode",
    choices=("learned", "id_direct", "position_ramp", "direct_delta"),
    default="learned",
)
parser.add_argument("--log_every", type=int, default=25)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs != 1:
    raise ValueError("Step response requires --num_envs 1")

random.seed(args.seed)
np.random.seed(args.seed)
launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

import ID_FD_pipline.tasks  # noqa: E402,F401
from ID_FD_pipline.dls_solver import CylinderDLSController  # noqa: E402
from ID_FD_pipline.dls_solver.cylinder import CylinderDLSConfig  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

TASK = "Kevin-Mulag-IDFD-CylinderSpace-PLAY-v0"
cfg = parse_env_cfg(TASK, device=args.device, num_envs=1)
cfg.seed = int(args.seed)
cfg.enable_scripted_vehicle_motion = False
cfg.episode_length_s = 300.0
cfg.cylinder_success_hold_steps = 100000
cfg.strict_cylinder_usd_scaling = False  # deliberate response-gain diagnostic
cfg.cylinder_fd_delta_gain = float(args.fd_gain)
residual_gains = [float(value) for value in args.residual_gains.split(",")]
if len(residual_gains) != 4:
    raise ValueError("--residual_gains must contain four comma-separated values")
cfg.cylinder_fd_residual_output_gain = residual_gains
if args.control_mode == "position_ramp":
    cfg.low_level_servo_mode = "position_ramp_pi"
elif args.control_mode == "direct_delta":
    cfg.low_level_servo_mode = "direct_position_delta"
env = gym.make(TASK, cfg=cfg).unwrapped


@torch.no_grad()
def measured_x() -> torch.Tensor:
    return env._refresh_cylinder_state()[0].detach().clone()


@torch.no_grad()
def target_x() -> torch.Tensor:
    zeros = torch.zeros_like(env._cylinder_pos_target)
    return env._cylinder_adapter.current_from_cylinder_joints(
        env._cylinder_pos_target, zeros
    )[0].detach().clone()


def run(mode: str) -> dict[str, float]:
    env.cfg.fd_residual_over_zero_valve = mode == "residual"
    env.reset(seed=int(args.seed))
    zero = torch.zeros((1, 4), device=env.device)
    for _ in range(int(args.settle_steps)):
        env.step(zero)

    start_measured = measured_x()
    start_target = target_x()
    scaling = env._cylinder_adapter.scaling
    lower = scaling.learned_min.to(env.device).reshape(1, 4)
    upper = scaling.learned_max.to(env.device).reshape(1, 4)
    requested = start_measured.clone()
    requested[:, int(args.axis)] += int(args.direction) * float(args.goal_delta_m)
    goal = torch.clamp(requested, lower, upper)
    actual_step = float((goal - start_measured)[0, int(args.axis)].item())
    if abs(actual_step) < 0.8 * float(args.goal_delta_m):
        raise RuntimeError(
            f"Requested step is limit-clipped to {actual_step:.4f} m; choose the opposite --direction or another --axis"
        )

    controller = CylinderDLSController(
        env,
        CylinderDLSConfig(
            jac_eps=float(args.jac_eps),
            damping_smax_ratio=float(args.damping_ratio),
            invert_learned_pipeline=args.control_mode != "id_direct",
            max_desired_delta_m=float(args.max_desired_delta_m),
            max_action=1.0 if args.control_mode == "id_direct" else 0.5,
        ),
    )
    max_target_advance = 0.0
    max_measured_advance = 0.0
    fd_abs_sum = 0.0
    target_gap_samples: list[float] = []
    print(
        f"[STEP][{mode}] axis={args.axis} requested_step={actual_step:+.4f} "
        f"control={args.control_mode} fd_gain={args.fd_gain:g} "
        f"residual_gains={residual_gains} "
        f"start_measured={start_measured[0].cpu().tolist()} "
        f"goal={goal[0].cpu().tolist()}",
        flush=True,
    )
    for step in range(int(args.steps)):
        before_target = target_x()
        before_measured = measured_x()
        if step == 0:
            print(f"[STEP][{mode}] computing_dls_action", flush=True)
            if args.control_mode == "learned":
                probe_zero = controller.predict_fd_delta(torch.zeros((1, 4), device=env.device))
                print(f"[STEP][{mode}] zero_fd_probe={probe_zero[0].cpu().tolist()}", flush=True)
                _, probe_mid, probe_s = controller.local_model()
                print(
                    f"[STEP][{mode}] local_model midpoint={probe_mid[0].cpu().tolist()} "
                    f"singular={probe_s[0].cpu().tolist()}",
                    flush=True,
                )
        action, singular_values = controller.action(goal)
        if step == 0:
            print(f"[STEP][{mode}] advancing_physics action={action[0].cpu().tolist()}", flush=True)
        env._cylinder_goal_x[:] = goal
        env.step(action)
        if step == 0:
            print(f"[STEP][{mode}] physics_advanced", flush=True)
        after_target = target_x()
        after_measured = measured_x()
        fd = env._fd_delta_corrected.detach().clone()
        valve = env._current_valve_cmd.detach().clone()

        target_advance = float(abs((after_target - start_target)[0, args.axis].item()))
        measured_advance = float(abs((after_measured - start_measured)[0, args.axis].item()))
        max_target_advance = max(max_target_advance, target_advance)
        max_measured_advance = max(max_measured_advance, measured_advance)
        fd_abs_sum += float(fd.abs().max().item())
        target_gap = float((after_target - after_measured).abs().max().item())
        target_gap_samples.append(target_gap)

        if step % int(args.log_every) == 0 or step == int(args.steps) - 1:
            goal_error = (goal - after_measured).abs()
            cond = float(
                (singular_values[0, 0] / singular_values[0, -1].clamp(min=1.0e-12)).item()
            )
            print(
                f"[STEP][{mode}] step={step:04d} "
                f"goal_err_max={float(goal_error.max().item()):.5f} "
                f"ref={after_target[0].cpu().tolist()} measured={after_measured[0].cpu().tolist()} "
                f"fd={fd[0].cpu().tolist()} target_dx={(after_target-before_target)[0].cpu().tolist()} "
                f"measured_dx={(after_measured-before_measured)[0].cpu().tolist()} "
                f"action={action[0].cpu().tolist()} valve={valve[0].cpu().tolist()} cond={cond:.2f}",
                flush=True,
            )

    final_target = target_x()
    final_measured = measured_x()
    ref_error = float((goal - final_target).abs().max().item())
    plant_error = float((goal - final_measured).abs().max().item())
    target_measured_gap = float((final_target - final_measured).abs().max().item())
    mean_fd_max = fd_abs_sum / max(int(args.steps), 1)
    if ref_error > 0.02:
        diagnosis = "FD_OR_TARGET_INTEGRATOR"
    elif target_measured_gap > 0.02:
        diagnosis = "PHYSX_FOLLOWING"
    else:
        diagnosis = "CLOSED"
    print(
        f"[STEP][{mode}][RESULT] diagnosis={diagnosis} ref_error={ref_error:.5f} "
        f"plant_error={plant_error:.5f} target_measured_gap={target_measured_gap:.5f} "
        f"max_target_advance={max_target_advance:.5f} max_measured_advance={max_measured_advance:.5f} "
        f"mean_fd_absmax={mean_fd_max:.7f} peak_gap={max(target_gap_samples):.5f}",
        flush=True,
    )
    return {
        "ref_error": ref_error,
        "plant_error": plant_error,
        "target_measured_gap": target_measured_gap,
    }


try:
    modes = ("residual", "full") if args.fd_mode == "both" else (args.fd_mode,)
    results = {mode: run(mode) for mode in modes}
    print(f"[STEP][SUMMARY] {results}", flush=True)
finally:
    env.close()
    simulation_app.close()
