#!/usr/bin/env python
"""PhysX joint-response sensitivity of payload mass / COM and joint damping.

Headless Isaac gate, 4 envs. Same position-target trajectory is run under
baseline, payload-mass-x1.5, COM-shift, and joint-damping-x3 configs. Reports
peak cylinder-joint and EE deviations from the baseline response.

Usage:
    python mech_sensitivity_gate.py [--steps 300] [--num_envs 4]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=300)
p.add_argument("--num_envs", type=int, default=4)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()

launcher = AppLauncher(a)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402

import mulag_plant.tasks  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


TASK = "RRLAB-Mulag-PPO2-ROM-PhysX-CylinderSpace-v0"


def main() -> None:
    cfg = parse_env_cfg(TASK, device=a.device, num_envs=a.num_envs)
    cfg.enable_scripted_vehicle_motion = False
    cfg.episode_length_s = 300.0
    cfg.allow_debug_manual_cylinder_targets = True
    env = gym.make(TASK, cfg=cfg).unwrapped
    robot = env.robot
    cyl_ids = env._cylinder_joint_ids
    device = env.device

    env.reset(seed=7)
    body_names = list(robot.body_names)
    if "Messerkopf" not in body_names:
        raise RuntimeError(f"Messerkopf body not found; have {body_names}")
    body_idx = body_names.index("Messerkopf")
    view = robot.root_physx_view
    base_masses = view.get_masses().clone()
    n_links = base_masses.shape[1]
    print(f"bodies={len(body_names)} physx_links={n_links} mass_idx={body_idx} "
          f"baseline_mass={float(base_masses[0, body_idx]):.4f}", flush=True)
    if n_links != len(body_names):
        raise RuntimeError("PhysX link order does not match body_names length")
    base_coms = view.get_coms().clone()
    base_damping = robot.data.joint_damping[:, cyl_ids].clone()

    def settle(steps: int = 40) -> None:
        zero = torch.zeros((a.num_envs, 4), device=device)
        for _ in range(steps):
            env.step(zero)

    def run_trajectory(perturb: str | None, label: str) -> dict[str, float]:
        env.reset(seed=7)
        # Restore the pristine baseline every run: set_masses / set_coms /
        # damping writes persist across env.reset(), so a previous run's
        # perturbation must not bleed into the next measurement.
        view.set_masses(base_masses.clone(), torch.arange(a.num_envs, dtype=torch.long))
        view.set_coms(base_coms.clone(), torch.arange(a.num_envs, dtype=torch.long))
        robot.write_joint_damping_to_sim(base_damping.clone(), joint_ids=cyl_ids)
        if perturb == "mass":
            m = base_masses.clone()
            m[:, body_idx] = m[:, body_idx] * 1.5
            view.set_masses(m, torch.arange(a.num_envs, dtype=torch.long))
            print(f"[{label}] set mass {float(base_masses[0, body_idx]):.4f} -> {float(m[0, body_idx]):.4f}", flush=True)
        elif perturb == "com":
            com = base_coms.clone()
            com[:, body_idx, 2] -= 0.10
            view.set_coms(com, torch.arange(a.num_envs, dtype=torch.long))
            print(f"[{label}] shifted cutter COM z by -0.10 m", flush=True)
        elif perturb == "damping":
            robot.write_joint_damping_to_sim(
                base_damping * 3.0, joint_ids=cyl_ids
            )
            print(f"[{label}] cylinder joint damping x3", flush=True)
        settle()

        q0 = robot.data.joint_pos[:, cyl_ids].clone()
        t0 = env._compute_ee_pos_local().clone()
        strokes = 4
        per = max(a.steps // (strokes * 3), 20)
        step_amp = 0.02
        base = q0.clone()
        peak_joint = 0.0
        peak_ee = 0.0
        q = q0.clone()
        for c in range(strokes):
            axis = c % 4
            target = base.clone()
            target[:, axis] = base[:, axis] + step_amp
            env.set_debug_manual_cylinder_targets(target)
            for _ in range(per):
                env.step(torch.zeros((a.num_envs, 4), device=device))
            q = robot.data.joint_pos[:, cyl_ids]
            peak_joint = max(peak_joint, float((q - q0).abs().max()))
            peak_ee = max(peak_ee, float((env._compute_ee_pos_local() - t0).abs().max()))
            env.set_debug_manual_cylinder_targets(base)
            for _ in range(per):
                env.step(torch.zeros((a.num_envs, 4), device=device))
            q = robot.data.joint_pos[:, cyl_ids]
            peak_joint = max(peak_joint, float((q - q0).abs().max()))
            peak_ee = max(peak_ee, float((env._compute_ee_pos_local() - t0).abs().max()))
        env.clear_debug_manual_cylinder_targets()
        return {"peak_joint_m": peak_joint, "peak_ee_m": peak_ee}

    base = run_trajectory(None, "baseline")
    print(f"[baseline] peak joint excursion {base['peak_joint_m']:.4f} m, "
          f"peak EE excursion {base['peak_ee_m']:.4f} m", flush=True)
    results = {}
    for name, perturb in (("mass_x1.5", "mass"), ("com_shift", "com"), ("damping_x3", "damping")):
        res = run_trajectory(perturb, name)
        d_joint = abs(res["peak_joint_m"] - base["peak_joint_m"])
        d_ee = abs(res["peak_ee_m"] - base["peak_ee_m"])
        rel_joint = d_joint / max(base["peak_joint_m"], 1e-9)
        rel_ee = d_ee / max(base["peak_ee_m"], 1e-9)
        results[name] = {"d_joint_m": d_joint, "d_ee_m": d_ee, "rel_joint": rel_joint, "rel_ee": rel_ee}
        print(f"[{name}] joint dev {d_joint:.4f} m ({rel_joint*100:.1f}%), "
              f"EE dev {d_ee:.4f} m ({rel_ee*100:.1f}%)", flush=True)

    print("RESULT " + str(results), flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
