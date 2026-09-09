#!/usr/bin/env python3
"""Measure each cylinder's MAXIMUM, twice: through the pipeline and bypassing it.

The question this answers is "where does the arm's authority actually go".
Every previous authority probe measured one stage in isolation -- effort
utilisation, the anti-windup clamp
(`audit_plant.py`), the path's demand (`probe_path_feasibility.py`) -- so none
of them could say whether the learned actuator, the command chain or the
mechanism is the binding one. This runs the SAME excitation through both paths
and reports the ratio.

Two phases, same env, same start pose, same wall-clock per control step:

  PIPELINE   env.step() with a sustained +-1 action on one axis. This is the
             entire deployed chain: action clip -> EMA filter -> rate limit ->
             action_scale -> ROM (or ID/FD) -> direction guard -> anti-windup
             lead clamp -> PhysX drive. It is the most a trained policy can
             ever ask for.

  DIRECT     DirectCylinderDriver writes the stroke limit straight to the PhysX
             position target, bypassing the learned actuator and every clamp
             above it. This is the mechanism's own ceiling.

`direct / pipeline` per axis is the authority the command chain and the learned
model are costing you. `direct` against the cylinder's stroke is what the
mechanism can do at all.

One environment per (axis, direction): 4 cylinders x 2 directions = 8. Driving
one axis at a time is deliberate -- a combined command confounds the axes
through the linkage, and the per-axis number is the one that maps onto
`action_scale`, the lead cap and the stroke.

Nothing here writes to the plant outside the two drive paths themselves; the
env is built through the same `resolve_actuator` path the training scripts use,
so "the plant" means the deployed one and not a probe-only variant.
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
p.add_argument("--actuator", default="rom", choices=("rom", "idfd", "id_aam_fd"))
p.add_argument("--stage", type=int, default=6)
p.add_argument("--seed", type=int, default=17)
p.add_argument("--steps", type=int, default=240, help="Control steps per phase")
p.add_argument("--settle_steps", type=int, default=40, help="Zero-action settle before each phase")
p.add_argument(
    "--action_scale",
    type=float,
    default=None,
    help="Override the curriculum action_scale; default keeps the shipped value",
)
p.add_argument(
    "--lead_m",
    type=float,
    default=None,
    help="Override the scalar anti-windup lead; default keeps the shipped per-axis tuple",
)
p.add_argument(
    "--multi_axis",
    action="store_true",
    help="Command ALL FOUR cylinders at once instead of one per env. The "
    "single-axis default leaves three axes at zero command, and the direction "
    "guard forces a zero-command axis's learned response to exactly zero, so "
    "the linkage cannot redistribute -- which under-reads the driven axis. "
    "Envs 0-3 drive every axis toward its upper limit, envs 4-7 toward its "
    "lower, so each run reports all four cylinders in both directions with "
    "four replicates each.",
)
p.add_argument(
    "--c4_gain",
    type=float,
    default=None,
    help="Override c4_action_scale_gain. The shipped 3.0 gives c4 0.012 m of "
    "command per step against a 0.020 m anti-windup cap -- it saturates the "
    "clamp in under two steps, where c1-c3 take twelve, which is the asymmetry "
    "that makes c4 bang-bang.",
)
p.add_argument(
    "--lead_per_axis",
    default="",
    help="Four comma-separated anti-windup caps in metres, one per cylinder. Each "
    "env drives ONE axis, so a single run can sweep a different cap on every "
    "cylinder at once. Overrides --lead_m.",
)
p.add_argument(
    "--dump_traces",
    action="store_true",
    help="Store the per-step extension and cutter tracks for offline inspection",
)
p.add_argument(
    "--direction_guard",
    default="keep",
    choices=("keep", "off"),
    help="Disable command_rom_direction_guard, which clamps the ROM's realised "
    "displacement to a multiple of the command and zeros counter-direction motion",
)
p.add_argument(
    "--command_max_ratio",
    type=float,
    default=None,
    help="Override command_rom_command_max_ratio. The runtime divides it by the "
    "ROM's steps_per_tick, so the shipped 3.0 clamps the realised displacement "
    "to 1.0x the commanded delta per control step",
)
p.add_argument(
    "--shield_margin",
    type=float,
    default=None,
    help="Override sole_limit_shield_margin_m; 0 disables the near-limit authority fade",
)
p.add_argument("--report", default="reports/cylinder_maxima.json")
AppLauncher.add_app_launcher_args(p)
a, unknown = p.parse_known_args()
sys.argv = [sys.argv[0], *unknown]

launcher = AppLauncher(a)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import verge_task  # noqa: F401,E402
from verge_task.actuators import resolve_actuator  # noqa: E402
from verge_task.distill import assert_teacher_disabled  # noqa: E402
from verge_task.terrain_env_cfg import (  # noqa: E402
    apply_terrain_curriculum_stage,
    configure_terrain_gpu_physx_memory,
)
from verge_task.terrain_observations import (  # noqa: E402
    observation_space_for_cfg,
    override_cfg,
)
from mulag_plant.dls_solver.direct_drive import DirectCylinderDriver  # noqa: E402

NUM_ENVS = 8
CYLINDERS = ("c1_drehzapfen", "c2_ausleger_I", "c3_ausleger_II", "c4_messerkopf_schwenk")
#: env e drives axis e // 2 in direction +1 (e even) or -1 (e odd).
AXIS_OF_ENV = [e // 2 for e in range(NUM_ENVS)]
DIR_OF_ENV = [1.0 if e % 2 == 0 else -1.0 for e in range(NUM_ENVS)]

TASK_ID, _RUNNER, ACTUATOR_LABEL = resolve_actuator(a.actuator)


def build_env():
    """The deployed task, with only the episode plumbing that would cut a probe short."""
    cfg = parse_env_cfg(TASK_ID, device=a.device, num_envs=NUM_ENVS)
    apply_terrain_curriculum_stage(cfg, int(a.stage))
    # PPO_ROM re-derives its per-axis c4 authority from the scalar the curriculum
    # just assigned; without this the tuple is whatever the previous stage left.
    if a.actuator == "rom":
        from PPO_ROM.terrain_env_cfg import apply_c4_authority_stage

        apply_c4_authority_stage(cfg, int(a.stage))
    cfg.action_space = 5 if cfg.sole_head_joint_action else 4
    cfg.sole_ik_hint = False
    if a.c4_gain is not None:
        # Applied BEFORE apply_c4_authority_stage would normally run, so
        # re-derive the per-axis scale tuple the adapter actually reads.
        override_cfg(cfg, "c4_action_scale_gain", float(a.c4_gain))
        base = float(cfg.action_scale)
        cfg.cylinder_action_scale_per_axis_m = (
            base, base, base, base * float(a.c4_gain),
        )
    if a.action_scale is not None:
        override_cfg(cfg, "action_scale", float(a.action_scale))
    if a.lead_per_axis:
        row = tuple(float(v) for v in a.lead_per_axis.split(","))
        if len(row) != 4:
            raise ValueError(f"--lead_per_axis needs four values, got {row}")
        override_cfg(cfg, "cylinder_target_max_lead_m", float(row[0]))
        cfg.cylinder_target_max_lead_per_axis_m = row
    elif a.lead_m is not None:
        lead = float(a.lead_m)
        override_cfg(cfg, "cylinder_target_max_lead_m", lead)
        # The per-axis tuple overrides the scalar in the runtime clamp, so the
        # scalar alone changes nothing. It cannot be cleared to None either:
        # TerrainC4AuthorityEnv asserts four values with c4 strictly the
        # largest, which is the whole point of that subclass. Keep c4 above the
        # swept value so the assertion holds and c4 stays effectively unclamped.
        c4_lead = max(
            float(getattr(cfg, "cylinder_target_lead_c4_m", 0.30)), lead * 1.5
        )
        cfg.cylinder_target_max_lead_per_axis_m = (lead, lead, lead, c4_lead)
    if a.command_max_ratio is not None:
        override_cfg(cfg, "command_rom_command_max_ratio", float(a.command_max_ratio))
    if a.direction_guard == "off":
        override_cfg(cfg, "command_rom_direction_guard", False)
        override_cfg(cfg, "cylinder_fd_command_direction_guard", False)
    if a.shield_margin is not None:
        override_cfg(cfg, "sole_limit_shield_margin_m", float(a.shield_margin))
    cfg.observation_space = observation_space_for_cfg(cfg)
    cfg.terrain_base_seed = int(a.seed)
    # A capability probe must not be cut short by the task's safety machinery,
    # and must not have the truck driving the arm through the terrain while the
    # cylinder is being swept.
    cfg.enable_scripted_vehicle_motion = False
    cfg.episode_length_s = 1.0e6
    cfg.terrain_terminate_on_contact_loss = False
    cfg.terrain_terminate_on_hard_impact = False
    cfg.defer_safety_terminations = True
    cfg.head_down_termination_alignment = -1.0
    cfg.randomize_start_phase = False
    cfg.randomize_trajectory_shape = False
    cfg.enable_contact_tracking_demo = False
    cfg.enable_visual_grid_floor = False
    cfg.enable_road_scenery = False
    configure_terrain_gpu_physx_memory(cfg, NUM_ENVS)
    assert_teacher_disabled(cfg, context="cylinder maxima probe")
    env = gym.make(TASK_ID, cfg=cfg, render_mode=None)
    return env, env.unwrapped


@torch.no_grad()
def sample(raw) -> tuple[torch.Tensor, torch.Tensor]:
    """Learned-space extension and cutter position in the TRUCK frame."""
    x, _ = raw._refresh_cylinder_state()
    return x.detach().clone(), raw._compute_ee_pos_local().detach().clone()


@torch.no_grad()
def run_phase(env, raw, *, phase: str, control_dt: float) -> dict:
    """Sweep every axis to its limit and record what the arm actually did."""
    env.reset(seed=int(a.seed))
    zero = torch.zeros((NUM_ENVS, int(raw.cfg.action_space)), device=raw.device)
    for _ in range(int(a.settle_steps)):
        env.step(zero)

    scaling = raw._cylinder_adapter.scaling
    lo = scaling.learned_min.to(raw.device, torch.float32).reshape(1, 4)
    hi = scaling.learned_max.to(raw.device, torch.float32).reshape(1, 4)

    x0, tool0 = sample(raw)
    driver = DirectCylinderDriver(raw) if phase == "direct" else None

    # PIPELINE: a sustained saturated action on the driven axis, zero elsewhere.
    # DIRECT: the stroke limit itself, written as the position target every step.
    #
    # `direction` is always a LEARNED-space direction so both phases sweep the
    # same way. The action does not: the mapping is q_usd = scale * x_learned
    # with scale = -1 on every cylinder here, so a +1 action drives learned
    # extension DOWN. Multiplying by the sign is what makes the pipeline and
    # direct columns comparable -- without it the two phases drive opposite
    # ends of the stroke and the ratio between them is meaningless.
    sign = scaling.scale.to(raw.device, torch.float32).reshape(4)
    action = torch.zeros((NUM_ENVS, int(raw.cfg.action_space)), device=raw.device)
    direct_goal = x0.clone()
    # (env, axis) -> direction, or 0 for "not driven in this env".
    driven: list[list[float]] = [[0.0] * 4 for _ in range(NUM_ENVS)]
    for e in range(NUM_ENVS):
        if a.multi_axis:
            axes = range(4)
            env_direction = 1.0 if e < NUM_ENVS // 2 else -1.0
        else:
            axes = (AXIS_OF_ENV[e],)
            env_direction = DIR_OF_ENV[e]
        for axis in axes:
            action[e, axis] = env_direction * float(sign[axis])
            direct_goal[e, axis] = hi[0, axis] if env_direction > 0 else lo[0, axis]
            driven[e][axis] = env_direction

    xs, tools = [x0], [tool0]
    # How often the ROM direction guard vetoes the learned response. The guard
    # zeroes any predicted delta whose sign opposes the command, so a high veto
    # rate on one axis means the ROM and the command disagree about direction
    # on that axis -- a model fault, not a tuning value.
    guard_vetoes = torch.zeros(NUM_ENVS, 4, device=raw.device)
    guard_samples = 0
    for _ in range(int(a.steps)):
        if phase == "direct":
            driver.step(direct_goal, substeps=int(raw.cfg.decimation))
        else:
            env.step(action)
            raw_delta = getattr(raw, "_fd_delta_raw", None)
            corrected = getattr(raw, "_fd_delta_corrected", None)
            if raw_delta is not None and corrected is not None:
                vetoed = (raw_delta.abs() > 1.0e-9) & (corrected.abs() <= 1.0e-12)
                guard_vetoes += vetoed.float()
                guard_samples += 1
        x, tool = sample(raw)
        xs.append(x)
        tools.append(tool)

    X = torch.stack(xs)  # (T+1, N, 4)
    T = torch.stack(tools)  # (T+1, N, 3)
    dX = (X[1:] - X[:-1]).abs()
    dT = T[1:] - T[:-1]

    results = []
    for e in range(NUM_ENVS):
      for axis in range(4):
        direction = driven[e][axis]
        if direction == 0.0:
            continue
        start = float(X[0, e, axis])
        limit = float(hi[0, axis]) if direction > 0 else float(lo[0, axis])
        room = abs(limit - start)
        travel = float((X[-1, e, axis] - X[0, e, axis]) * direction)
        step_axis = dX[:, e, axis]
        tool_speed = torch.linalg.norm(dT[:, e, :], dim=-1) / control_dt
        lateral_speed = dT[:, e, 1].abs() / control_dt
        results.append(
            {
                "cylinder": CYLINDERS[axis],
                "axis": axis,
                "env": e,
                "direction": "extend" if direction > 0 else "retract",
                "stroke_total_m": float(hi[0, axis] - lo[0, axis]),
                "room_from_start_m": room,
                "travel_m": travel,
                "travel_fraction_of_room": travel / room if room > 1e-9 else 0.0,
                "peak_step_m": float(step_axis.max()),
                "p95_step_m": float(torch.quantile(step_axis, 0.95)),
                "peak_speed_mps": float(step_axis.max()) / control_dt,
                "p95_speed_mps": float(torch.quantile(step_axis, 0.95)) / control_dt,
                "peak_tool_speed_mps": float(tool_speed.max()),
                "peak_tool_lateral_speed_mps": float(lateral_speed.max()),
                "p95_tool_lateral_speed_mps": float(torch.quantile(lateral_speed, 0.95)),
                "guard_veto_fraction_driven_axis": (
                    float(guard_vetoes[e, axis]) / guard_samples
                    if guard_samples
                    else None
                ),
            }
        )
    out = {"phase": phase, "control_dt_s": control_dt, "axes": results}
    if a.dump_traces:
        # Per-env track of its OWN driven axis, plus the cutter. Net travel and
        # p95 speed cannot distinguish "slow but monotonic" from "fast but
        # oscillating about a standstill", and c4 needs exactly that call.
        out["traces"] = {
            "driven_axis_extension": [
                X[:, e, AXIS_OF_ENV[e]].cpu().tolist() for e in range(NUM_ENVS)
            ],
            "cutter_truck_y": [T[:, e, 1].cpu().tolist() for e in range(NUM_ENVS)],
        }
    return out


def main() -> None:
    env, raw = build_env()
    try:
        control_dt = float(raw.sim.get_physics_dt()) * int(raw.cfg.decimation)
        frozen = {
            "action_scale": float(raw.cfg.action_scale),
            "cylinder_action_scale_per_axis_m": list(
                getattr(raw.cfg, "cylinder_action_scale_per_axis_m", ()) or ()
            ),
            "cylinder_target_max_lead_m": float(
                getattr(raw.cfg, "cylinder_target_max_lead_m", 0.0)
            ),
            "cylinder_target_max_lead_per_axis_m": list(
                getattr(raw.cfg, "cylinder_target_max_lead_per_axis_m", None) or ()
            ),
            "sole_action_filter": float(getattr(raw.cfg, "sole_action_filter", 0.0)),
            "sole_max_action_rate": float(getattr(raw.cfg, "sole_max_action_rate", 0.0)),
            "sole_limit_shield_margin_m": float(
                getattr(raw.cfg, "sole_limit_shield_margin_m", 0.0)
            ),
            "hydraulic_target_filter_beta": float(
                getattr(raw.cfg, "hydraulic_target_filter_beta", 0.0)
            ),
            "command_rom_direction_guard": bool(
                getattr(raw.cfg, "command_rom_direction_guard", False)
            ),
            "control_dt_s": control_dt,
        }
        print(f"[MAXIMA] actuator={ACTUATOR_LABEL} stage={a.stage} cfg={frozen}", flush=True)

        pipeline = run_phase(env, raw, phase="pipeline", control_dt=control_dt)
        direct = run_phase(env, raw, phase="direct", control_dt=control_dt)

        ledger = []
        for pipe_row, direct_row in zip(pipeline["axes"], direct["axes"]):
            ratio = (
                pipe_row["peak_speed_mps"] / direct_row["peak_speed_mps"]
                if direct_row["peak_speed_mps"] > 1e-9
                else None
            )
            ledger.append(
                {
                    "cylinder": pipe_row["cylinder"],
                    "direction": pipe_row["direction"],
                    "stroke_total_m": pipe_row["stroke_total_m"],
                    "room_from_start_m": pipe_row["room_from_start_m"],
                    "pipeline_peak_speed_mps": pipe_row["peak_speed_mps"],
                    "direct_peak_speed_mps": direct_row["peak_speed_mps"],
                    "pipeline_fraction_of_direct": ratio,
                    "pipeline_p95_speed_mps": pipe_row["p95_speed_mps"],
                    "direct_p95_speed_mps": direct_row["p95_speed_mps"],
                    "pipeline_travel_m": pipe_row["travel_m"],
                    "direct_travel_m": direct_row["travel_m"],
                    "pipeline_travel_fraction_of_room": pipe_row["travel_fraction_of_room"],
                    "direct_travel_fraction_of_room": direct_row["travel_fraction_of_room"],
                    "pipeline_tool_lateral_mps": pipe_row["peak_tool_lateral_speed_mps"],
                    "direct_tool_lateral_mps": direct_row["peak_tool_lateral_speed_mps"],
                }
            )

        print("\n[MAXIMA] cylinder            dir      stroke  pipe m/s  direct m/s   pipe/direct")
        for row in ledger:
            ratio = row["pipeline_fraction_of_direct"]
            ratio_text = "   n/a" if ratio is None else f"{ratio:6.3f}"
            print(
                f"[MAXIMA] {row['cylinder']:<20s} {row['direction']:<8s}"
                f"{row['stroke_total_m']:7.3f}  {row['pipeline_peak_speed_mps']:8.4f}  "
                f"{row['direct_peak_speed_mps']:10.4f}  {ratio_text}",
                flush=True,
            )

        report = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "actuator": a.actuator,
            "actuator_label": ACTUATOR_LABEL,
            "stage": int(a.stage),
            "steps_per_phase": int(a.steps),
            "frozen_cfg": frozen,
            "pipeline": pipeline,
            "direct": direct,
            "ledger": ledger,
        }
        out = Path(a.report)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1))
        print(f"\n[MAXIMA] wrote {out}", flush=True)
    finally:
        env.close()


main()
simulation_app.close()
