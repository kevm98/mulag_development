#!/usr/bin/env python3
"""Functional gate: track mixed-sign cylinder setpoints with online ID/FD DLS."""

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
parser.add_argument("--steps_per_goal", type=int, default=900)
parser.add_argument("--goal_delta_m", type=float, default=0.02)
parser.add_argument("--tolerance_m", type=float, default=0.01)
parser.add_argument("--jac_eps", type=float, default=0.005)
parser.add_argument("--error_gain", type=float, default=0.02)
parser.add_argument("--max_desired_delta_m", type=float, default=5.0e-4)
parser.add_argument("--damping_smax_ratio", type=float, default=0.10)
parser.add_argument("--max_action", type=float, default=0.2)
parser.add_argument("--action_smoothing", type=float, default=0.2)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs != 1:
    raise ValueError("DLS tracking gate requires --num_envs 1")

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


TASK = "Kevin-Mulag-IDFD-CylinderSpace-PLAY-v0"
cfg = parse_env_cfg(TASK, device=args.device, num_envs=1)
cfg.seed = int(args.seed)
cfg.enable_scripted_vehicle_motion = False
cfg.episode_length_s = 120.0
cfg.cylinder_success_hold_steps = 100000
env = gym.make(TASK, cfg=cfg).unwrapped


@torch.no_grad()
def predict_fd_delta(action: torch.Tensor) -> torch.Tensor:
    """Evaluate the deployed learned path without advancing physics or buffers."""
    x, xdot = env._refresh_cylinder_state()
    converted = env._action_to_cylinder_target_adapter.convert(
        action=action,
        x_current=x,
        xdot_current=xdot,
    )
    previous_valve = env._prev_valve_cmd.detach().clone()
    model_input = env._cylinder_adapter.build_id_input(
        converted.reference,
        dP=env._dP,
        force=env._force,
        prev_valve_cmd=previous_valve,
        input_dim=env._cylinder_id.input_dim,
    )
    valve = env._cylinder_id.predict(model_input, prev_valve_cmd=previous_valve)
    if bool(getattr(env.cfg, "id_residual_over_zero_target", False)):
        zero_target_input = model_input.detach().clone()
        zero_target_input[:, 16:24] = 0.0
        zero_target_valve = env._cylinder_id.predict(
            zero_target_input, prev_valve_cmd=previous_valve
        )
        valve = valve - zero_target_valve
    valve = torch.clamp(valve, -float(env.cfg.max_abs_valve), float(env.cfg.max_abs_valve))
    fd_out = env._cylinder_fd.step(
        valve_cmd=valve,
        extension=x,
        extension_velocity=xdot,
        dP=env._dP,
        force=env._force,
    )
    delta = fd_out.delta_extension
    if bool(getattr(env.cfg, "fd_residual_over_zero_valve", False)):
        fd_zero = env._cylinder_fd.step(
            valve_cmd=torch.zeros_like(valve),
            extension=x,
            extension_velocity=xdot,
            dP=env._dP,
            force=env._force,
        )
        delta = delta - fd_zero.delta_extension
    scale = float(getattr(env.cfg, "cylinder_fd_delta_gain", 1.0))
    if bool(getattr(env.cfg, "cylinder_fd_scale_delta_by_sample_time", False)):
        fd_dt = float(getattr(env.cfg, "cylinder_fd_sample_time_s", 0.0))
        if fd_dt <= 0.0:
            fd_dt = float(env._cylinder_fd.sample_time_s)
        scale *= float(env.step_dt) / max(fd_dt, 1.0e-6)
    return delta * scale


@torch.no_grad()
def local_model() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    columns = []
    midpoint = torch.zeros((1, 4), dtype=torch.float32, device=env.device)
    for column in range(4):
        plus = torch.zeros((1, 4), dtype=torch.float32, device=env.device)
        minus = torch.zeros_like(plus)
        plus[:, column] = args.jac_eps
        minus[:, column] = -args.jac_eps
        y_plus = predict_fd_delta(plus)
        y_minus = predict_fd_delta(minus)
        columns.append(((y_plus - y_minus) / (2.0 * args.jac_eps))[0])
        midpoint += (y_plus + y_minus) / 8.0
    jacobian = torch.stack(columns, dim=1)
    singular_values = torch.linalg.svdvals(jacobian)
    return jacobian, midpoint, singular_values


@torch.no_grad()
def dls_action(error: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    jacobian, midpoint, singular_values = local_model()
    desired = torch.clamp(
        float(args.error_gain) * error,
        -float(args.max_desired_delta_m),
        float(args.max_desired_delta_m),
    )
    desired = desired - midpoint
    u, s, vh = torch.linalg.svd(jacobian, full_matrices=False)
    damping = torch.clamp(float(args.damping_smax_ratio) * s[0], min=1.0e-7)
    gains = s / (s.square() + damping.square())
    action = vh.transpose(0, 1) @ (gains * (u.transpose(0, 1) @ desired[0]))
    action = torch.clamp(action, -float(args.max_action), float(args.max_action)).reshape(1, 4)
    return action, singular_values


try:
    env.reset(seed=int(args.seed))
    # Let contacts settle while the pipeline stays in neutral hold.
    zero = torch.zeros((1, 4), dtype=torch.float32, device=env.device)
    for _ in range(30):
        env.step(zero)
    start, _ = env._refresh_cylinder_state()
    start = start.detach().clone()
    patterns = torch.tensor(
        [[1.0, -1.0, 1.0, -1.0], [-1.0, 1.0, -1.0, 1.0]],
        dtype=torch.float32,
        device=env.device,
    )
    goals = start + float(args.goal_delta_m) * patterns
    passed = True
    print(f"[DLS] start={start[0].cpu().tolist()}", flush=True)
    for goal_index, goal in enumerate(goals):
        if goal_index > 0:
            env.reset(seed=int(args.seed))
            for _ in range(30):
                env.step(zero)
        goal = goal.reshape(1, 4)
        previous_action = torch.zeros((1, 4), dtype=torch.float32, device=env.device)
        within_tolerance_steps = 0
        minimum_error = float("inf")
        for step in range(int(args.steps_per_goal)):
            current, _ = env._refresh_cylinder_state()
            zero_velocity = torch.zeros_like(env._cylinder_pos_target)
            target_x, _ = env._cylinder_adapter.current_from_cylinder_joints(
                env._cylinder_pos_target, zero_velocity
            )
            # The learned FD delta integrates into _cylinder_pos_target.  Close
            # this inner loop on that reference; measured x is the slower PhysX
            # plant output and is used below for the functional acceptance test.
            error = goal - target_x
            action, singular_values = dls_action(error)
            alpha = float(args.action_smoothing)
            action = alpha * action + (1.0 - alpha) * previous_action
            previous_action = action.detach().clone()
            env._cylinder_goal_x[:] = goal
            before_x = current.detach().clone()
            before_target = env._cylinder_pos_target.detach().clone()
            predicted_delta = predict_fd_delta(action).detach().clone() if step == 0 else None
            env.step(action)
            current, _ = env._refresh_cylinder_state()
            if step == 0:
                print(
                    f"[DLS][STEP-CHECK] goal={goal_index} predicted_fd={predicted_delta[0].cpu().tolist()} "
                    f"runtime_fd={env._fd_delta_corrected[0].cpu().tolist()} "
                    f"measured_dx={(current - before_x)[0].cpu().tolist()} "
                    f"usd_target_delta={(env._cylinder_pos_target - before_target)[0].cpu().tolist()}",
                    flush=True,
                )
            max_error = float((goal - current).abs().max().item())
            minimum_error = min(minimum_error, max_error)
            within_tolerance_steps = within_tolerance_steps + 1 if max_error < args.tolerance_m else 0
            if step % 50 == 0 or within_tolerance_steps == 20:
                cond = float((singular_values[0] / singular_values[-1].clamp(min=1.0e-12)).item())
                print(
                    f"[DLS] goal={goal_index} step={step:03d} max_error={max_error:.5f} "
                    f"action={action[0].cpu().tolist()} model_cond={cond:.1f}",
                    flush=True,
                )
            if within_tolerance_steps >= 20:
                break
        final, _ = env._refresh_cylinder_state()
        final_error = (goal - final).abs()
        goal_pass = bool((final_error < args.tolerance_m).all().item())
        passed &= goal_pass
        print(
            f"[DLS][GOAL {goal_index}] {'PASS' if goal_pass else 'FAIL'} "
            f"goal={goal[0].cpu().tolist()} final={final[0].cpu().tolist()} "
            f"abs_error={final_error[0].cpu().tolist()} min_max_error={minimum_error:.5f}",
            flush=True,
        )
    print(f"[DLS][GATE] {'PASS' if passed else 'FAIL'} both mixed-sign setpoints < {args.tolerance_m:.3f} m", flush=True)
finally:
    env.close()
    sim_app.close()
