#!/usr/bin/env python3
"""Runtime half of the wiring audit: runtime-composite and direct-PhysX 4x4 matrices.

Uses the SAME perturbation size as the offline audit so the four matrices are directly
comparable:

  runtime-FD        : d(FD DeltaExtension) / d(ValveCmd)      -- must match the offline
                      learned-FD matrix, else the runtime feeds FD wrong inputs
  runtime-composite : d(applied FD delta) / d(target_delta_x) -- the map the DLS inverts
  direct-PhysX      : d(measured delta x) / d(target delta x) -- the plant, ID/FD bypassed

Also runs the direction-guard ablation (guard ON vs OFF) on the composite matrix.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser()
parser.add_argument("--valve_eps", type=float, default=0.05)
parser.add_argument("--target_eps", type=float, default=0.002, help="target_delta_x perturbation [m]")
parser.add_argument("--physx_eps", type=float, default=0.005, help="PhysX target perturbation [m]")
parser.add_argument("--out", default="outputs/lower_level_tracker/wiring_audit_runtime.json")
parser.add_argument("--skip_physx", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.num_envs = 1
app_launcher = AppLauncher(args)
sim_app = app_launcher.app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
import ID_FD_pipline.tasks  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from plant_study.direct_drive import DirectCylinderDriver  # noqa: E402

TASK = "Kevin-Mulag-IDFD-CylinderSpace-PLAY-v0"


def build(guard: bool):
    cfg = parse_env_cfg(TASK, device=args.device, num_envs=1)
    cfg.enable_scripted_vehicle_motion = False
    cfg.episode_length_s = 1.0e4
    cfg.cylinder_success_hold_steps = 10 ** 9
    cfg.cylinder_fd_command_direction_guard = bool(guard)
    env = gym.make(TASK, cfg=cfg).unwrapped
    env.reset(seed=17)
    return env


def show(name, M, extra=""):
    print(f"\n===== {name} {extra}", flush=True)
    for i in range(4):
        print("   " + "".join(f"{M[i, j]:+12.6f}" for j in range(4)), flush=True)
    sv = np.linalg.svd(M, compute_uv=False)
    print(f"   diag={np.round(np.diag(M), 6).tolist()}", flush=True)
    print(f"   sv={np.round(sv, 6).tolist()}  cond={sv[0]/max(sv[-1],1e-15):.1f}  "
          f"rank={int((sv > 1e-6*sv[0]).sum())}", flush=True)
    return dict(matrix=M.tolist(), diag=np.diag(M).tolist(),
                singular_values=sv.tolist(),
                condition=float(sv[0] / max(sv[-1], 1e-15)),
                rank=int((sv > 1e-6 * sv[0]).sum()))


results = {}

# --------------------------------------------------------------------------- #
# A. direct-PhysX matrix: d(measured x)/d(target x). ID/FD/DLS not involved.
# --------------------------------------------------------------------------- #
env = build(guard=True)
driver = DirectCylinderDriver(env)
base_cmd = driver.measured_x().clone()
for _ in range(300):
    driver.step(base_cmd, substeps=2)
base_cmd = driver.measured_x().clone()

J_physx = np.zeros((4, 4))
for c in ([] if args.skip_physx else range(4)):
    sides = []
    for sgn in (+1.0, -1.0):
        cmd = base_cmd.clone()
        cmd[0, c] += sgn * args.physx_eps
        for _ in range(300):
            driver.step(cmd, substeps=2)
        sides.append(driver.measured_x()[0].clone().cpu().numpy())
        for _ in range(300):
            driver.step(base_cmd, substeps=2)
    J_physx[:, c] = (sides[0] - sides[1]) / (2 * args.physx_eps)
results["direct_physx"] = show("DIRECT-PhysX  d(measured x)/d(target x)", J_physx,
                               f"(eps={args.physx_eps} m)")

# --------------------------------------------------------------------------- #
# B. runtime-FD matrix: d(FD delta)/d(valve), at the live runtime state.
#    Must match the offline learned-FD matrix; if not, the runtime feeds FD bad inputs.
# --------------------------------------------------------------------------- #
x, xdot = env._refresh_cylinder_state()


def fd_delta_for_valve(valve):
    out = env._cylinder_fd.step(valve_cmd=valve, extension=x, extension_velocity=xdot,
                                dP=env._dP, force=env._force)
    return out.delta_extension


v0 = torch.zeros((1, 4), device=env.device)
J_fd_rt = np.zeros((4, 4))
for c in range(4):
    p, m = v0.clone(), v0.clone()
    p[0, c] += args.valve_eps
    m[0, c] -= args.valve_eps
    J_fd_rt[:, c] = ((fd_delta_for_valve(p) - fd_delta_for_valve(m))
                     / (2 * args.valve_eps))[0].cpu().numpy()
results["runtime_fd"] = show("RUNTIME-FD  d(FD delta)/d(valve)", J_fd_rt,
                             f"(eps={args.valve_eps})")
results["runtime_hydraulic_features"] = dict(
    dP=env._dP[0].tolist(), force=env._force[0].tolist(),
    extension=x[0].tolist(), extension_velocity=xdot[0].tolist(),
)
print(f"\n[STATE] runtime dP={np.round(env._dP[0].cpu().numpy(), 3).tolist()}", flush=True)
print(f"[STATE] runtime force={np.round(env._force[0].cpu().numpy(), 3).tolist()}", flush=True)

# ID input z-scores at this state (feature-distribution coverage)
ref_dummy = env._action_to_cylinder_target_adapter.convert(
    action=torch.zeros((1, 4), device=env.device), x_current=x, xdot_current=xdot)
id_in = env._cylinder_adapter.build_id_input(
    ref_dummy.reference, dP=env._dP, force=env._force,
    prev_valve_cmd=env._prev_valve_cmd, input_dim=env._cylinder_id.input_dim)
results["id_input_z"] = env._cylinder_id.input_z_score_summary(id_in)
print(f"[STATE] ID input z-scores: {results['id_input_z']}", flush=True)


# --------------------------------------------------------------------------- #
# C. runtime-composite: d(applied FD delta)/d(target_delta_x), guard ON vs OFF.
# --------------------------------------------------------------------------- #
def composite(env, eps):
    """Perturb target_delta_x directly (not the action) so the map is in cylinder units."""
    from ID_FD_pipline.dls_solver.cylinder import CylinderDLSController, CylinderDLSConfig
    ctrl = CylinderDLSController(env, CylinderDLSConfig())
    scaling = env._cylinder_adapter.scaling
    action_scale = float(env.cfg.action_scale)
    # target_delta_x = usd_delta_to_learned_delta(action * action_scale)
    # so   action = learned_delta_to_usd_delta(target_delta_x) / action_scale
    J = np.zeros((4, 4))
    for c in range(4):
        cols = []
        for sgn in (+1.0, -1.0):
            tdx = torch.zeros((1, 4), device=env.device)
            tdx[0, c] = sgn * eps
            act = scaling.learned_delta_to_usd_delta(tdx) / action_scale
            cols.append(ctrl.predict_fd_delta(act)[0].cpu().numpy())
        J[:, c] = (cols[0] - cols[1]) / (2 * eps)
    return J


def guarded_composite(env, eps, ratio=None):
    """The delta the PLANT actually receives: FD output AFTER the direction guard.

    NOTE the inconsistency this exposes: CylinderDLSController.predict_fd_delta() -- the
    map the DLS builds and inverts -- does NOT apply the guard, but
    _run_cylinder_id_pipeline() does. The controller therefore inverts a different map
    from the one the plant experiences. This replicates the pipeline's guard verbatim.
    """
    from ID_FD_pipline.dls_solver.cylinder import CylinderDLSController, CylinderDLSConfig
    ctrl = CylinderDLSController(env, CylinderDLSConfig())
    scaling = env._cylinder_adapter.scaling
    action_scale = float(env.cfg.action_scale)
    if ratio is None:
        ratio = float(getattr(env.cfg, "cylinder_fd_command_max_ratio", 1.5))
    J = np.zeros((4, 4))
    for c in range(4):
        cols = []
        for sgn in (+1.0, -1.0):
            tdx = torch.zeros((1, 4), device=env.device)
            tdx[0, c] = sgn * eps
            act = scaling.learned_delta_to_usd_delta(tdx) / action_scale
            d = ctrl.predict_fd_delta(act)
            same = d * tdx >= 0.0                       # pipeline guard, verbatim
            limit = ratio * tdx.abs()
            d = torch.where(same, torch.sign(d) * torch.minimum(d.abs(), limit),
                            torch.zeros_like(d))
            cols.append(d[0].cpu().numpy())
        J[:, c] = (cols[0] - cols[1]) / (2 * eps)
    return J


J_unguarded = composite(env, args.target_eps)
results["runtime_composite_unguarded"] = show(
    "RUNTIME-COMPOSITE (UNGUARDED) -- the map the DLS inverts", J_unguarded,
    f"(eps={args.target_eps} m)")

J_guarded = guarded_composite(env, args.target_eps)
results["runtime_composite_guarded"] = show(
    "RUNTIME-COMPOSITE (GUARDED) -- the delta the PLANT receives", J_guarded,
    f"(eps={args.target_eps} m, ratio={env.cfg.cylinder_fd_command_max_ratio})")

for r in (2.0, 3.0):
    results[f"runtime_composite_guarded_ratio{r}"] = show(
        f"RUNTIME-COMPOSITE (GUARDED, ratio={r})", guarded_composite(env, args.target_eps, r), "")
env.close()

# --------------------------------------------------------------------------- #
print("\n\n==================== IDENTITY-COMPOSITION CHECK ====================", flush=True)
print("For the pipeline to be correct, FD(ID(s, dx)) must equal dx, i.e. the composite", flush=True)
print("matrix must be the IDENTITY. Deviation from I is the wiring/model error.", flush=True)
for name in ("runtime_composite_unguarded", "runtime_composite_guarded",
             "runtime_composite_guarded_ratio2.0", "runtime_composite_guarded_ratio3.0"):
    J = np.array(results[name]["matrix"])
    err = np.abs(J - np.eye(4))
    print(f"  {name:32s}: max|J - I| = {err.max():8.3f}   "
          f"diag = {np.round(np.diag(J), 3).tolist()}", flush=True)
    results[name]["identity_error_max"] = float(err.max())
    results[name]["identity_error_diag"] = np.abs(np.diag(J) - 1.0).tolist()

Path(ROOT / args.out).write_text(json.dumps(results, indent=2))
print(f"\n[OUT] {ROOT / args.out}", flush=True)
sim_app.close()
