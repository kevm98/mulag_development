#!/usr/bin/env python3
"""Plant identification: inject force pulses directly, measure cylinder response."""
from __future__ import annotations
import argparse, sys, os
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Identify cylinder effort plant.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--force_fractions", type=str, default="0.01,0.02,0.05")
parser.add_argument("--pulse_duration_s", type=float, default=0.3)
parser.add_argument("--settle_duration_s", type=float, default=1.5)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env_cfg import MulagUpperLevelEnvCfg
from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env import MulagUpperLevelEnv
from kevin_integration.utils.sim_memory import apply_kevin_sim_memory_optimizations
from isaaclab.utils.math import quat_apply_inverse

cfg = MulagUpperLevelEnvCfg()
cfg.scene.num_envs = int(args_cli.num_envs)
cfg.enable_scripted_vehicle_motion = False
cfg.enable_startup_summary = False
cfg.log_target_delta_xdot_clipping = False
cfg.allow_debug_manual_cylinder_targets = True
cfg.lowlevel_actuator_backend = 'explicit_effort_pd'
cfg.explicit_pd_gain_scale = 0.0
cfg.explicit_pd_damping_scale = 0.3
cfg.explicit_pd_effort_scale = 1.0
cfg.explicit_force_rate_limit_fraction = 0.01
cfg.explicit_pd_velocity_filter_beta = 0.8
if getattr(args_cli, "device", None): cfg.device = args_cli.device; cfg.sim.device = args_cli.device
id_path = (Path(os.getcwd()) / "outputs/cylinder_diagnostics/paper_matched_pipeline/model_finetune/id_physx_finetuned_best.pt").resolve()
cfg.id_checkpoint_override_path = str(id_path)
apply_kevin_sim_memory_optimizations(cfg.sim)

env = MulagUpperLevelEnv(cfg=cfg); uw = env; B = env.num_envs; dev = uw.device
cyl_ids = uw._cylinder_joint_ids; tool_id = uw._tool_joint_id; ee_id = uw._ee_body_id
dt = float(uw.step_dt); pv = uw.robot.root_physx_view
jnames = uw.robot.data.joint_names

eff_lims = torch.tensor([100e6, 10e6, 1e6, 100e3], device=dev)  # N
force_fractions = [float(x) for x in args_cli.force_fractions.split(",")]
pulse_s = float(args_cli.pulse_duration_s)
settle_s = float(args_cli.settle_duration_s)
pulse_steps = int(pulse_s / dt)
settle_steps = int(settle_s / dt)

def tcp_b():
    ee = uw.robot.data.body_link_state_w[:, ee_id, :3]; bp = uw.robot.data.root_pos_w
    bq = uw.robot.data.root_quat_w; bq = bq / bq.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return quat_apply_inverse(bq, ee - bp)

print("=" * 70, flush=True)
print(f"PLANT IDENTIFICATION (pulse={pulse_s}s settle={settle_s}s)", flush=True)
print("=" * 70, flush=True)

all_results = []

for c in range(4):
    for sign in [+1, -1]:
        for frac in force_fractions:
            F_test = frac * eff_lims[c].item()

            env.reset()
            root_pose = uw.robot.data.root_state_w[:, :7].clone()
            root_vel = torch.zeros((B, 6), device=dev)
            cq_hold = uw.robot.data.joint_pos[:, cyl_ids].clone()
            tool_target = uw.robot.data.joint_pos[:, tool_id].unsqueeze(-1).clone()

            # Stabilize with D-only
            for _ in range(60):
                uw.robot.write_root_pose_to_sim(root_pose)
                uw.robot.write_root_velocity_to_sim(root_vel)
                uw.set_debug_manual_cylinder_targets(cq_hold, tool_position_target=tool_target)
                env.step(torch.zeros((B, 5), device=dev))
            cq_hold = uw.robot.data.joint_pos[:, cyl_ids].clone()

            # Baseline
            q0 = uw.robot.data.joint_pos[0, cyl_ids].detach().cpu().numpy()
            qd0 = uw.robot.data.joint_vel[0, cyl_ids].detach().cpu().numpy()
            ee0 = tcp_b()[0].detach().cpu().numpy()

            # Record timeline
            q_hist = [q0.copy()]
            qd_hist = [qd0.copy()]
            ee_hist = [ee0.copy()]
            effort_hist = []
            force_cmd_hist = []

            # Run: baseline → pulse → settle
            total_steps = pulse_steps + settle_steps
            for s in range(total_steps):
                uw.robot.write_root_pose_to_sim(root_pose)
                uw.robot.write_root_velocity_to_sim(root_vel)
                uw.set_debug_manual_cylinder_targets(cq_hold, tool_position_target=tool_target)

                # Get current state
                q = uw.robot.data.joint_pos[:, cyl_ids]
                qd = uw.robot.data.joint_vel[:, cyl_ids]
                error = cq_hold - q

                # D-only baseline (applied by env's _explicit_effort_target)
                # But we need to inject EXTRA force on cylinder c
                # The env applies: force = kp*error - kd*qd_filt
                # We override with: force = kp*error - kd*qd_filt + pulse
                kp_vals = torch.tensor([15e6, 6e6, 1e6, 1e6], device=dev) * 0.0
                kd_vals = torch.tensor([3e6, 1.8e6, 1.5e6, 1.5e6], device=dev) * 0.3

                # Read filtered qd from the env
                qd_filt = uw._qd_filt if hasattr(uw, '_qd_filt') else qd
                force_cmd = kp_vals.unsqueeze(0) * error - kd_vals.unsqueeze(0) * qd_filt

                # Inject pulse
                if s < pulse_steps:
                    force_extra = torch.zeros((B, 4), device=dev)
                    force_extra[0, c] = sign * F_test
                    force_cmd = force_cmd + force_extra

                force_cmd = torch.clamp(force_cmd, -eff_lims.unsqueeze(0), eff_lims.unsqueeze(0))
                uw.robot.set_joint_effort_target(force_cmd, joint_ids=cyl_ids)
                env.step(torch.zeros((B, 5), device=dev))

                # Record every 3rd step
                if s % 3 == 0:
                    q_hist.append(uw.robot.data.joint_pos[0, cyl_ids].detach().cpu().numpy())
                    qd_hist.append(uw.robot.data.joint_vel[0, cyl_ids].detach().cpu().numpy())
                    ee_hist.append(tcp_b()[0].detach().cpu().numpy())
                    effort = pv.get_dof_actuation_forces()
                    effort_hist.append(effort[0].detach().cpu().numpy() if isinstance(effort, torch.Tensor) else np.zeros(21))

            q_hist = np.array(q_hist)
            qd_hist = np.array(qd_hist)
            ee_hist = np.array(ee_hist)

            # Compute response metrics
            dq_c = q_hist[:, c] - q_hist[0, c]  # cylinder c displacement
            max_dq = np.max(np.abs(dq_c))
            dq_final = q_hist[-1, c] - q_hist[0, c]
            sign_correct = (dq_final * sign) > 0

            # Gain: Δq / ΔF (mm/MN)
            gain = abs(dq_final) * 1000.0 / max(F_test, 1e-9) * 1e6  # mm per 1MN

            # Cross-coupling
            cross_coupling = {i: float(q_hist[-1, i] - q_hist[0, i]) * 1000 for i in range(4) if i != c}

            # EE response
            dee = np.linalg.norm(ee_hist[-1] - ee_hist[0])

            # Settling: time to return within 10% of peak
            peak_idx = np.argmax(np.abs(dq_c))
            peak_val = np.abs(dq_c[peak_idx])
            settled_idx = peak_idx
            for i in range(peak_idx, len(dq_c)):
                if np.abs(dq_c[i]) < 0.1 * peak_val:
                    settled_idx = i
                    break
            settle_time = (settled_idx - peak_idx) * 3 * dt  # 3 steps per sample

            print(f"  cyl{c} {'+' if sign>0 else '-'}{F_test/1e6:.1f}MN: "
                  f"dq={dq_final*1000:+.2f}mm sign_ok={sign_correct} "
                  f"gain={gain:.1f}mm/MN cross={cross_coupling} "
                  f"ee_d={dee:.3f}m settle≈{settle_time:.2f}s",
                  flush=True)

            all_results.append({
                "cylinder": c, "sign": sign, "force": F_test, "frac": frac,
                "dq_final_mm": dq_final * 1000, "sign_correct": sign_correct,
                "gain_mm_per_MN": gain, "max_dq_mm": max_dq * 1000,
                "ee_delta_m": dee, "settle_time_s": settle_time,
                "cross_coupling": cross_coupling,
            })

# ---- Summary ----
print(f"\n{'='*70}", flush=True)
print("PER-CYLINDER SUMMARY", flush=True)
print(f"{'='*70}", flush=True)
for c in range(4):
    cyl_results = [r for r in all_results if r["cylinder"] == c]
    signs_ok = sum(1 for r in cyl_results if r["sign_correct"])
    gains = [r["gain_mm_per_MN"] for r in cyl_results if r["gain_mm_per_MN"] < 1e9]  # filter infs
    mean_gain = np.mean(gains) if gains else 0
    ee_ds = [r["ee_delta_m"] for r in cyl_results]
    print(f"  {jnames[cyl_ids[c]]}: sign_accuracy={signs_ok}/{len(cyl_results)} "
          f"mean_gain={mean_gain:.1f}mm/MN ee_range=[{min(ee_ds):.3f},{max(ee_ds):.3f}]m",
          flush=True)

env.close()
simulation_app.close()
