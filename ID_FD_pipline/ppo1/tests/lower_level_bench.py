#!/usr/bin/env python3
"""Lower-level ID->FD tracker bench.  NO state machine, NO EE IK, NO truck.

Runs on the plain cylinder-space env (4-D action == cylinder DLS action), so the
only thing under test is:

    cylinder goal -> DLS action -> ID(valve) -> FD(delta) -> target integrator -> PhysX

Modes
    truck : 60 s zero-action truck-stationarity proof
    hold  : pre-position to endpoint A (DirectCylinderDriver), hand off, hold 10 s
    cyl   : per-cylinder step / ramp / sine reference tracking through ID->FD
    jac   : composite d(FD delta)/d(action) and d(FD delta)/d(target_delta_x) probe
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="hold", choices=["truck", "hold", "cyl", "jac", "plant"])
parser.add_argument("--seed", type=int, default=17)
parser.add_argument("--hold_seconds", type=float, default=10.0)
parser.add_argument("--truck_seconds", type=float, default=60.0)
parser.add_argument("--handoff", default="target", choices=["target", "measured"],
                    help="what the PhysX target integrator is synced to at handoff")
parser.add_argument("--out", default="outputs/lower_level_tracker")
parser.add_argument("--effort_mult", type=float, default=1.0,
                    help="DIAGNOSTIC: scale cylinder actuator effort limits")
parser.add_argument("--pose", default="band", choices=["band", "reset"],
                    help="operating point: band=ground-band endpoint A, reset=spawn pose")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.num_envs = 1
app_launcher = AppLauncher(args)
sim_app = app_launcher.app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
import ID_FD_pipline.tasks  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from ID_FD_pipline.dls_solver.cylinder import CylinderDLSConfig, CylinderDLSController  # noqa: E402
from ID_FD_pipline.stop_mow.parking import ParkingBrake  # noqa: E402
from plant_study.direct_drive import DirectCylinderDriver  # noqa: E402

TASK = "Kevin-Mulag-IDFD-CylinderSpace-PLAY-v0"
OUT = ROOT / args.out
OUT.mkdir(parents=True, exist_ok=True)

# Endpoint A of the accepted 0.88 m arc (truck frame), from ppo1/env_cfg.py.
CX, CY, R = 4.327, -3.224, 1.286
THETA_A_DEG = -26.7
EE_A = (CX + R * math.cos(math.radians(THETA_A_DEG)),
        CY + R * math.sin(math.radians(THETA_A_DEG)))


def build_env():
    cfg = parse_env_cfg(TASK, device=args.device, num_envs=1)
    cfg.seed = int(args.seed)
    cfg.enable_scripted_vehicle_motion = False   # hard-disable the truck
    cfg.episode_length_s = 10_000.0              # never time out mid-bench
    cfg.cylinder_success_hold_steps = 10**9      # never "succeed" -> never auto-reset
    env = gym.make(TASK, cfg=cfg).unwrapped
    env.reset(seed=int(args.seed))
    if args.effort_mult != 1.0:
        ids = env._cylinder_joint_ids
        lim = env.robot.data.joint_effort_limits[:, ids] * float(args.effort_mult)
        env.robot.write_joint_effort_limit_to_sim(lim, joint_ids=ids)
        print(f"[BENCH] DIAGNOSTIC effort_limit x{args.effort_mult} -> "
              f"{[f'{v:.3g}' for v in lim[0].tolist()]}", flush=True)
    return env


def band_endpoint_a():
    payload = json.loads((ROOT / "ID_FD_pipline/models/ground_band_map.json").read_text())
    band = payload["ground_band"]
    cutters = np.array([s["cutter_truck"] for s in band])
    dist = np.linalg.norm(cutters[:, :2] - np.array(EE_A), axis=1)
    idx = int(np.argmin(dist))
    return band[idx], float(dist[idx])


def truck_state(env):
    root = env.robot.data.root_state_w[0]
    wheels = env._wheel_joint_ids
    return dict(
        root_pos=root[:3].tolist(),
        root_quat=root[3:7].tolist(),
        lin_vel=root[7:10].tolist(),
        ang_vel=root[10:13].tolist(),
        wheel_pos=env.robot.data.joint_pos[0, wheels].tolist(),
        wheel_vel=env.robot.data.joint_vel[0, wheels].tolist(),
        wheel_vel_target=env.robot.data.joint_vel_target[0, wheels].tolist(),
    )


def snap(env, controller):
    """One row of lower-level state for env 0."""
    x, xdot = env._refresh_cylinder_state()
    root = env.robot.data.root_state_w[0]
    return dict(
        x=x[0].tolist(),
        xdot=xdot[0].tolist(),
        x_target=controller.target_x()[0].tolist(),
        valve=env._current_valve_cmd[0].tolist(),
        target_delta_x=env._action_target_delta_x_learned[0].tolist(),
        fd_raw=env._fd_delta_raw[0].tolist(),
        fd_corrected=env._fd_delta_corrected[0].tolist(),
        neutral_hold=bool(env._neutral_hold_mask[0]),
        xdot_clip=env._action_target_delta_xdot_clip_mask[0].tolist(),
        ee=env._compute_ee_pos_local()[0].tolist(),
        truck_pos=root[:3].tolist(),
        truck_lin_vel=float(torch.linalg.norm(root[7:10])),
        truck_ang_vel=float(torch.linalg.norm(root[10:13])),
    )


# --------------------------------------------------------------------------- #
def mode_truck(env):
    brake = ParkingBrake(env)
    brake.engage()
    steps = int(args.truck_seconds / float(env.step_dt))
    zero = torch.zeros((1, 4), device=env.device)
    start = truck_state(env)
    rows = []
    for i in range(steps):
        brake.engage()
        env.step(zero)
        if i % 60 == 0:
            rows.append(dict(step=i, **truck_state(env)))
    end = truck_state(env)
    disp = float(np.linalg.norm(np.array(end["root_pos"]) - np.array(start["root_pos"])))
    q0, q1 = np.array(start["root_quat"]), np.array(end["root_quat"])
    yaw = lambda q: math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
    dyaw = math.degrees(abs(yaw(q1) - yaw(q0)))
    result = dict(
        seconds=args.truck_seconds, steps=steps,
        root_translation_m=disp, yaw_drift_deg=dyaw,
        max_lin_speed_mps=max(float(np.linalg.norm(r["lin_vel"])) for r in rows),
        max_wheel_vel_target=max(max(abs(v) for v in r["wheel_vel_target"]) for r in rows),
        max_wheel_vel=max(max(abs(v) for v in r["wheel_vel"]) for r in rows),
        start=start, end=end,
    )
    (OUT / "truck_freeze.json").write_text(json.dumps(result, indent=2))
    print(f"[TRUCK] {args.truck_seconds}s  disp={disp*1000:.3f} mm  yaw={dyaw:.4f} deg  "
          f"max|v|={result['max_lin_speed_mps']:.5f} m/s  "
          f"max wheel target={result['max_wheel_vel_target']:.3g}", flush=True)


# --------------------------------------------------------------------------- #
def preposition(env):
    """DirectCylinderDriver to endpoint A. Returns (commanded_x, settled_x, settled_ee)."""
    driver = DirectCylinderDriver(env)
    if args.pose == "reset":
        # Operating point = the spawn pose itself: command the cylinder values the
        # arm is ALREADY at, so nothing depends on the (saturated) band map.
        cmd = driver.measured_x().clone()
        dist = 0.0
    else:
        sample, dist = band_endpoint_a()
        cmd = torch.tensor(sample["target_x"], device=env.device, dtype=torch.float32).reshape(1, 4)
    for _ in range(400):
        driver.step(cmd, substeps=2)
    x, _ = env._refresh_cylinder_state()
    ee = env._compute_ee_pos_local()
    print(f"[PREPOS] band sample dist to EE_A={dist:.4f} m", flush=True)
    print(f"[PREPOS] commanded x = {[f'{v:.4f}' for v in cmd[0].tolist()]}", flush=True)
    print(f"[PREPOS] settled   x = {[f'{v:.4f}' for v in x[0].tolist()]}", flush=True)
    print(f"[PREPOS] drive steady-state offset = "
          f"{[f'{v:.4f}' for v in (x[0] - cmd[0]).tolist()]} m", flush=True)
    print(f"[PREPOS] settled EE  = {[f'{v:.4f}' for v in ee[0].tolist()]}", flush=True)
    return cmd, x.clone(), ee.clone()


def handoff(env, controller, commanded_x, settled_x):
    """Sync every integrator/history to the endpoint-A state before ID/FD takes over."""
    scaling = env._cylinder_adapter.scaling
    if args.handoff == "target":
        # The plant is in equilibrium *under the commanded target*. Seed the PhysX
        # target integrator with that same command, not with the sagged measurement.
        q = scaling.clamp_usd_q(scaling.learned_x_to_usd_q(commanded_x))
    else:
        q = env.robot.data.joint_pos[:, env._cylinder_joint_ids].detach().clone()
    env._cylinder_pos_target[:] = q
    env._neutral_hold_cylinder_pos_target[:] = q
    env._prev_valve_cmd[:] = 0.0
    env._current_valve_cmd[:] = 0.0
    env._fd_delta_raw[:] = 0.0
    env._fd_delta_corrected[:] = 0.0
    env._prev_action[:] = 0.0
    env._current_action[:] = 0.0
    if env._cylinder_id is not None and hasattr(env._cylinder_id, "reset"):
        env._cylinder_id.reset(torch.arange(env.num_envs, device=env.device))
    controller.reset()


def run_tracking(env, controller, goal_fn, steps, tag):
    tag = f"{tag}_{args.pose}_e{args.effort_mult:g}"
    """Drive the ID/FD chain toward goal_fn(step) -> (1,4) cylinder goal."""
    rows = []
    prev_x = env._refresh_cylinder_state()[0][0].clone()
    for i in range(steps):
        goal = goal_fn(i)
        action, _ = controller.action(goal)
        env._cylinder_goal_x[:] = goal
        env.step(action)
        row = snap(env, controller)
        x_now = torch.tensor(row["x"], device=env.device)
        row.update(
            step=i, t=i * float(env.step_dt),
            goal=goal[0].tolist(),
            action=action[0].tolist(),
            actual_delta=(x_now - prev_x).tolist(),
            err=(goal[0] - x_now).tolist(),
        )
        prev_x = x_now
        rows.append(row)
    path = OUT / f"{tag}.json"
    path.write_text(json.dumps(rows, indent=1))
    return rows


def report(rows, tag, ee_goal=None):
    err = np.array([r["err"] for r in rows])              # (T,4)
    enorm = np.abs(err)
    valve = np.array([r["valve"] for r in rows])
    fd_raw = np.array([r["fd_raw"] for r in rows])
    fd_cor = np.array([r["fd_corrected"] for r in rows])
    tdx = np.array([r["target_delta_x"] for r in rows])
    act_d = np.array([r["actual_delta"] for r in rows])
    x = np.array([r["x"] for r in rows])
    xt = np.array([r["x_target"] for r in rows])
    ee = np.array([r["ee"] for r in rows])
    summary = dict(
        tag=tag, steps=len(rows),
        cyl_abs_err_mean=enorm.mean(axis=0).tolist(),
        cyl_abs_err_p95=np.percentile(enorm, 95, axis=0).tolist(),
        cyl_abs_err_max=enorm.max(axis=0).tolist(),
        cyl_err_final=err[-1].tolist(),
        drift_first_to_last=(x[-1] - x[0]).tolist(),
        target_minus_measured_final=(xt[-1] - x[-1]).tolist(),
        valve_abs_mean=np.abs(valve).mean(axis=0).tolist(),
        valve_sat_frac=(np.abs(valve) >= 0.999).mean(axis=0).tolist(),
        fd_raw_abs_mean=np.abs(fd_raw).mean(axis=0).tolist(),
        fd_corrected_abs_mean=np.abs(fd_cor).mean(axis=0).tolist(),
        guard_kill_frac=((np.abs(fd_raw) > 1e-9) & (np.abs(fd_cor) < 1e-9)).mean(axis=0).tolist(),
        guard_ratio_cap_frac=(np.abs(fd_cor) >= 1.5 * np.abs(tdx) - 1e-12).mean(axis=0).tolist(),
        target_delta_x_abs_mean=np.abs(tdx).mean(axis=0).tolist(),
        actual_delta_abs_mean=np.abs(act_d).mean(axis=0).tolist(),
        fd_vs_actual_gain=(np.abs(act_d).mean(axis=0) /
                           np.maximum(np.abs(fd_cor).mean(axis=0), 1e-12)).tolist(),
        neutral_hold_frac=float(np.mean([r["neutral_hold"] for r in rows])),
        truck_disp_m=float(np.linalg.norm(np.array(rows[-1]["truck_pos"]) -
                                          np.array(rows[0]["truck_pos"]))),
    )
    if ee_goal is not None:
        eerr = np.linalg.norm(ee - np.array(ee_goal), axis=1)
        summary.update(
            ee_err_mean=float(eerr.mean()),
            ee_err_p95=float(np.percentile(eerr, 95)),
            ee_err_max=float(eerr.max()),
            ee_drift_m=float(np.linalg.norm(ee[-1] - ee[0])),
        )
    (OUT / f"{tag}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[{tag.upper()}] " + json.dumps(summary, indent=2), flush=True)
    return summary


# --------------------------------------------------------------------------- #
def mode_hold(env):
    brake = ParkingBrake(env)
    brake.engage()
    commanded_x, settled_x, settled_ee = preposition(env)
    controller = CylinderDLSController(
        env, CylinderDLSConfig(invert_learned_pipeline=True, error_gain=0.10,
                               max_desired_delta_m=0.01, max_action=1.0,
                               action_smoothing=1.0),
    )
    handoff(env, controller, commanded_x, settled_x)
    goal = settled_x.clone()          # hold exactly where the plant settled
    steps = int(args.hold_seconds / float(env.step_dt))
    rows = run_tracking(env, controller, lambda i: goal, steps, f"hold_{args.handoff}")
    report(rows, f"hold_{args.handoff}", ee_goal=settled_ee[0].tolist())


# --------------------------------------------------------------------------- #
def mode_cyl(env):
    """Per-cylinder bidirectional tracking through the real ID->FD chain."""
    brake = ParkingBrake(env)
    brake.engage()
    commanded_x, settled_x, _ = preposition(env)
    controller = CylinderDLSController(
        env, CylinderDLSConfig(invert_learned_pipeline=True, error_gain=0.10,
                               max_desired_delta_m=0.01, max_action=1.0,
                               action_smoothing=1.0),
    )
    handoff(env, controller, commanded_x, settled_x)
    base = settled_x.clone()
    fps = 1.0 / float(env.step_dt)
    out = {}
    for c in range(4):
        for name, amp, kind, secs in [
            ("step_p5mm", 0.005, "step", 3.0),
            ("step_m5mm", -0.005, "step", 3.0),
            ("step_p10mm", 0.010, "step", 3.0),
            ("sine_010hz", 0.010, "sine", 10.0),
        ]:
            def goal_fn(i, c=c, amp=amp, kind=kind):
                g = base.clone()
                if kind == "step":
                    g[0, c] += amp
                else:
                    g[0, c] += amp * math.sin(2 * math.pi * 0.10 * i / fps)
                return g
            tag = f"cyl{c+1}_{name}"
            rows = run_tracking(env, controller, goal_fn, int(secs * fps), tag)
            out[f"{tag}_{args.pose}_e{args.effort_mult:g}"] = report(rows, tag)
            # settle back to base between probes
            run_tracking(env, controller, lambda i: base.clone(), int(2.0 * fps), f"{tag}_settle")
    (OUT / f"cylinder_matrix_{args.pose}_e{args.effort_mult:g}.json").write_text(json.dumps(out, indent=2))


# --------------------------------------------------------------------------- #
def mode_jac(env):
    """Composite ID->FD local map at endpoint A."""
    brake = ParkingBrake(env)
    brake.engage()
    commanded_x, settled_x, _ = preposition(env)
    controller = CylinderDLSController(env, CylinderDLSConfig(invert_learned_pipeline=True))
    handoff(env, controller, commanded_x, settled_x)
    J, mid, sv = controller.local_model()
    J0 = J[0].cpu().numpy()
    s = sv[0].cpu().numpy()
    # scale action->target_delta_x so the map is in cylinder units both sides
    scaling = env._cylinder_adapter.scaling
    result = dict(
        jacobian_dFD_daction=J0.tolist(),
        midpoint=mid[0].cpu().numpy().tolist(),
        singular_values=s.tolist(),
        condition=float(s[0] / max(s[-1], 1e-12)),
        rank=int((s > 1e-3 * s[0]).sum()),
        diag=np.diag(J0).tolist(),
        offdiag_max=float(np.abs(J0 - np.diag(np.diag(J0))).max()),
        learned_scale=scaling.scale.cpu().numpy().tolist(),
        action_scale=float(env.cfg.action_scale),
    )
    (OUT / "idfd_jacobian_A.json").write_text(json.dumps(result, indent=2))
    print("[JAC] " + json.dumps(result, indent=2), flush=True)


# --------------------------------------------------------------------------- #
def mode_plant(env):
    """Plant-only: d(measured x)/d(commanded target x) at endpoint A, via PhysX drives.

    Answers whether the actuator is effort-saturated (small target increments do
    nothing) or simply offset (increments move the plant ~1:1 with a bias).
    """
    brake = ParkingBrake(env)
    brake.engage()
    commanded_x, settled_x, _ = preposition(env)
    driver = DirectCylinderDriver(env)
    ids = env._cylinder_joint_ids
    lim = [float(a.effort_limit.flatten()[0]) for a in env.robot.actuators.values()
           if a.joint_names and a.joint_names[0] in
           ("Drehzapfen_joint", "Ausleger_I_joint", "Ausleger_II_joint",
            "Messerkopf_Schwenk_joint")]
    tau0 = env.robot.data.applied_torque[0, ids].tolist()
    print(f"[PLANT] effort_limit={[f'{v:.3g}' for v in lim]}", flush=True)
    print(f"[PLANT] applied_torque@A={[f'{v:.3g}' for v in tau0]}", flush=True)

    rows = []
    for c in range(4):
        for d in (-0.050, -0.010, -0.005, 0.005, 0.010, 0.050):
            cmd = commanded_x.clone()
            cmd[0, c] += d
            for _ in range(240):          # 4 s settle
                driver.step(cmd, substeps=2)
            x, _ = env._refresh_cylinder_state()
            tau = env.robot.data.applied_torque[0, ids]
            rows.append(dict(
                cylinder=c + 1, target_delta=d,
                measured_delta=(x[0] - settled_x[0]).tolist(),
                gain=float((x[0, c] - settled_x[0, c]) / d),
                torque=tau.tolist(),
            ))
            print(f"[PLANT] cyl{c+1} dtarget={d*1000:+7.1f} mm -> dmeas={float(x[0,c]-settled_x[0,c])*1000:+8.3f} mm "
                  f"gain={rows[-1]['gain']:+6.3f}  tau={tau[c]:.4g}", flush=True)
        for _ in range(240):               # return to A
            driver.step(commanded_x, substeps=2)
    (OUT / "plant_gain_A.json").write_text(json.dumps(
        dict(commanded_x=commanded_x[0].tolist(), settled_x=settled_x[0].tolist(),
             effort_limit=lim, rows=rows), indent=2))


try:
    env = build_env()
    print(f"[BENCH] step_dt={float(env.step_dt):.6f}s  servo_mode={env._low_level_servo_mode()}  "
          f"guard={env.cfg.cylinder_fd_command_direction_guard} "
          f"ratio={env.cfg.cylinder_fd_command_max_ratio} "
          f"action_scale={env.cfg.action_scale}", flush=True)
    {"truck": mode_truck, "hold": mode_hold, "cyl": mode_cyl, "jac": mode_jac, "plant": mode_plant}[args.mode](env)
finally:
    try:
        env.close()
    except Exception:
        pass
    sim_app.close()
