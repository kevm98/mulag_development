#!/usr/bin/env python3
"""Compare current_horizon_idfd_ff (A) vs horizon_idfd_ff_pi (B) controllers."""
from __future__ import annotations
import argparse, csv, sys, os
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--controller", type=str, default="both")
parser.add_argument("--delta_cmds", type=str, default="-0.01,0.01")
parser.add_argument("--horizon_steps", type=str, default="10")
parser.add_argument("--fd_gain", type=float, default=1.0)
parser.add_argument("--tracking_gains", type=str, default="0.0,0.5")
parser.add_argument("--pi_kps", type=str, default="0.25")
parser.add_argument("--pi_kis", type=str, default="0.03")
parser.add_argument("--pi_leak", type=float, default=0.90)
parser.add_argument("--pi_ilim", type=float, default=0.02)
parser.add_argument("--pi_dclip", type=float, default=0.01)
parser.add_argument("--cylinders", type=str, default="0,1,2,3")
parser.add_argument("--settle_s", type=float, default=0.2)
parser.add_argument("--output_dir", type=str,
                    default="outputs/cylinder_diagnostics/paper_matched_pipeline/controller_comparison")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]
if getattr(args_cli, "device", None) in (None, ""): args_cli.device = "cuda:0"
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env import MulagUpperLevelEnv
from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env_cfg import MulagUpperLevelEnvCfg
from kevin_integration.utils.sim_memory import apply_kevin_sim_memory_optimizations
from finetune_id_physx_dynamic import VelocityToValveMLP

outdir = Path(args_cli.output_dir); outdir.mkdir(parents=True, exist_ok=True)
deltas = [float(x.strip()) for x in args_cli.delta_cmds.split(",") if x.strip()]
horizons = [int(x.strip()) for x in args_cli.horizon_steps.split(",") if x.strip()]
tr_gains = [float(x.strip()) for x in args_cli.tracking_gains.split(",") if x.strip()]
pi_kps = [float(x.strip()) for x in args_cli.pi_kps.split(",") if x.strip()]
pi_kis = [float(x.strip()) for x in args_cli.pi_kis.split(",") if x.strip()]
cids = [int(x.strip()) for x in args_cli.cylinders.split(",") if x.strip()]
fd_gain = args_cli.fd_gain
run_a = args_cli.controller in ("current_horizon_idfd_ff", "both")
run_b = args_cli.controller in ("horizon_idfd_ff_pi", "both")

cfg = MulagUpperLevelEnvCfg()
cfg.scene.num_envs = int(args_cli.num_envs)
id_path = (Path(os.getcwd()) / "outputs/cylinder_diagnostics/paper_matched_pipeline/model_finetune/id_physx_finetuned_best.pt").resolve()
cfg.id_checkpoint_override_path = str(id_path)
cfg.low_level_servo_mode = "none"
cfg.enable_scripted_vehicle_motion = False
cfg.enable_startup_summary = True; cfg.log_target_delta_xdot_clipping = False
if getattr(args_cli, "device", None): cfg.device = args_cli.device; cfg.sim.device = args_cli.device
apply_kevin_sim_memory_optimizations(cfg.sim)

env = MulagUpperLevelEnv(cfg=cfg)
uw = env.unwrapped; dt = float(uw.step_dt); dev = uw.device
settle = max(int(round(args_cli.settle_s / dt)), 0)
model_dt = float(getattr(uw, "_sequence_model_dt", 0.05))

id_ckpt = torch.load(str(id_path), map_location=dev, weights_only=False)
id_model = VelocityToValveMLP().to(dev)
id_model.load_state_dict(id_ckpt["model_state_dict"]); id_model.eval()
X_mean_t = torch.from_numpy(id_ckpt["X_mean"]).to(dev)
X_std_t = torch.from_numpy(id_ckpt["X_std"]).to(dev)

za = torch.zeros((uw.num_envs, uw.cfg.action_space), device=dev)
results = []
print("START", flush=True)

def run_trial(label, h, ci, dx_des, x0, xd0, cq0, ee0, extra):
    """Run one trial: ID/FD rollout + correction + ramp + measure."""
    id_hist = uw._sequence_id._step_history.detach().clone()
    fd_hist = uw._sequence_fd._step_history.detach().clone()
    x_r = x0.detach().clone(); xd_r = xd0.detach().clone()
    fd_sum = 0.0
    for _ in range(h):
        dx_step = dx_des / h; v_step = dx_step / model_dt
        dd = torch.zeros((1, 4), device=dev); dd[:, ci] = dx_step
        vd = torch.zeros((1, 4), device=dev); vd[:, ci] = v_step
        with torch.no_grad(): valve = id_model(torch.cat([dd, vd, x_r, xd_r], -1).sub(X_mean_t).div(X_std_t))
        uw._sequence_fd.push_current_state(valve, x_r, xd_r)
        fd_d = float(uw._sequence_fd.predict_delta_extension().delta_extension_raw[:, ci].mean().cpu())
        fd_sum += fd_d; x_r[:, ci] += fd_d; xd_r[:, ci] = v_step
    uw._sequence_id._step_history = id_hist
    uw._sequence_fd._step_history = fd_hist
    
    # Correction
    e_prev = extra.get("e_prev", 0.0)
    I_val = extra.get("I", 0.0)
    if label == "A_ff":
        tg_val = extra.get("tg", 0.0)
        delta_cmd_H = fd_gain * fd_sum + tg_val * e_prev
        delta_cmd_H = np.clip(delta_cmd_H, -0.02, 0.02)
        fd_c = abs(fd_gain * fd_sum); tr_c = abs(tg_val * e_prev)
    else:  # B_ff_pi
        kp_val = extra.get("kp", 0.25); ki_val = extra.get("ki", 0.03)
        I_new = np.clip(args_cli.pi_leak * I_val + e_prev, -args_cli.pi_ilim, args_cli.pi_ilim)
        extra["I"] = I_new
        delta_cmd_H = fd_gain * fd_sum + kp_val * e_prev + ki_val * I_new
        delta_cmd_H = np.clip(delta_cmd_H, -args_cli.pi_dclip, args_cli.pi_dclip)
        fd_c = abs(fd_gain * fd_sum); tr_c = abs(kp_val * e_prev + ki_val * I_new)
    
    # Ramp execution
    dx_learned = torch.zeros((1, 4), device=dev); dx_learned[:, ci] = float(delta_cmd_H)
    dx_isaac = uw._cylinder_adapter.simscape_delta_to_isaac_delta(dx_learned)
    for i in range(1, h + 1):
        uw._debug_manual_cylinder_position_target = (cq0 + (i/h) * dx_isaac).detach().clone()
        obs, _, _, _, _ = env.step(za)
    uw._debug_manual_cylinder_position_target = None
    
    cq1 = uw.robot.data.joint_pos[:, uw._cylinder_joint_ids].detach().clone()
    x1, _ = uw._cylinder_adapter.current_from_cylinder_joints(cq1, uw.robot.data.joint_vel[:, uw._cylinder_joint_ids])
    dx_meas = float((x1 - x0)[:, ci].mean().cpu())
    ee1 = uw.robot.data.body_state_w[:, uw._ee_body_id, 0:3].detach().clone()
    
    return dx_meas, fd_sum, delta_cmd_H, fd_c, tr_c, float((ee1-ee0).norm(dim=-1).mean().cpu())

# ---- Controller A ----
if run_a:
    print("CTRL_A", flush=True)
    for h in horizons:
        for tg in tr_gains:
            prev_meas = {ci: 0.0 for ci in cids}
            for ci in cids:
                for dx_des in deltas:
                    obs, _ = env.reset()
                    for _ in range(settle): obs, _, _, _, _ = env.step(za)
                    cq0 = uw.robot.data.joint_pos[:, uw._cylinder_joint_ids].detach().clone()
                    x0, xd0 = uw._cylinder_adapter.current_from_cylinder_joints(cq0, uw.robot.data.joint_vel[:, uw._cylinder_joint_ids])
                    ee0 = uw.robot.data.body_state_w[:, uw._ee_body_id, 0:3].detach().clone()
                    e_prev = dx_des - prev_meas[ci] if prev_meas[ci] != 0 else 0.0
                    extra = {"e_prev": e_prev, "tg": tg}
                    dx_meas, fd_sum, delta_cmd_H, fd_c, tr_c, ee_d = run_trial("A_ff", h, ci, dx_des, x0, xd0, cq0, ee0, extra)
                    prev_meas[ci] = dx_meas
                    sign_ok = 1 if np.sign(dx_meas) == np.sign(dx_des) else 0
                    results.append(dict(ctrl="A_ff", h=h, tg=tg, kp=0, ki=0, cyl=ci, dx_des=dx_des,
                                       dx_meas=dx_meas, fd_sum=fd_sum, delta_cmd=delta_cmd_H,
                                       sign_ok=sign_ok, disp_r=abs(dx_meas)/max(abs(dx_des),1e-9),
                                       fd_c=fd_c, tr_c=tr_c, ee_d=ee_d))
                    if len(results) % 5 == 0:
                        g = sum(1 for r in results if r["sign_ok"])
                        print(f"  A[{len(results)}] cyl={ci} dx={dx_des:+.3f} tg={tg:.1f}: "
                              f"meas={dx_meas:+.5f} sign={'OK' if sign_ok else '??'} good={g}/{len(results)}", flush=True)

# ---- Controller B ----
if run_b:
    print("CTRL_B", flush=True)
    for h in horizons:
        for kp_val in pi_kps:
            for ki_val in pi_kis:
                prev_meas = {ci: 0.0 for ci in cids}
                int_state = {ci: 0.0 for ci in cids}
                for ci in cids:
                    for dx_des in deltas:
                        obs, _ = env.reset()
                        for _ in range(settle): obs, _, _, _, _ = env.step(za)
                        cq0 = uw.robot.data.joint_pos[:, uw._cylinder_joint_ids].detach().clone()
                        x0, xd0 = uw._cylinder_adapter.current_from_cylinder_joints(cq0, uw.robot.data.joint_vel[:, uw._cylinder_joint_ids])
                        ee0 = uw.robot.data.body_state_w[:, uw._ee_body_id, 0:3].detach().clone()
                        e_prev = dx_des - prev_meas[ci] if prev_meas[ci] != 0 else 0.0
                        extra = {"e_prev": e_prev, "I": int_state[ci], "kp": kp_val, "ki": ki_val}
                        dx_meas, fd_sum, delta_cmd_H, fd_c, tr_c, ee_d = run_trial("B_ff_pi", h, ci, dx_des, x0, xd0, cq0, ee0, extra)
                        prev_meas[ci] = dx_meas; int_state[ci] = extra["I"]
                        sign_ok = 1 if np.sign(dx_meas) == np.sign(dx_des) else 0
                        results.append(dict(ctrl="B_ff_pi", h=h, tg=0, kp=kp_val, ki=ki_val, cyl=ci, dx_des=dx_des,
                                           dx_meas=dx_meas, fd_sum=fd_sum, delta_cmd=delta_cmd_H,
                                           sign_ok=sign_ok, disp_r=abs(dx_meas)/max(abs(dx_des),1e-9),
                                           fd_c=fd_c, tr_c=tr_c, ee_d=ee_d))
                        if len(results) % 5 == 0:
                            g = sum(1 for r in results if r["sign_ok"])
                            print(f"  B[{len(results)}] cyl={ci} dx={dx_des:+.3f}: "
                                  f"meas={dx_meas:+.5f} sign={'OK' if sign_ok else '??'} I={extra['I']:+.4f} good={g}/{len(results)}", flush=True)

# ---- Save results ----
if results:
    _csv = outdir / "horizon_controller_comparison.csv"
    with _csv.open("w", newline="") as _f:
        _w = csv.DictWriter(_f, fieldnames=list(results[0].keys()))
        _w.writeheader(); _w.writerows(results)
    print(f"CSV: {_csv}", flush=True)
    
    a_res = [r for r in results if r["ctrl"] == "A_ff"]
    b_res = [r for r in results if r["ctrl"] == "B_ff_pi"]
    sa_a = np.mean([r["sign_ok"] for r in a_res]) if a_res else 0
    sa_b = np.mean([r["sign_ok"] for r in b_res]) if b_res else 0
    dr_a = np.mean([r["disp_r"] for r in a_res]) if a_res else 0
    dr_b = np.mean([r["disp_r"] for r in b_res]) if b_res else 0
    fd_a = np.mean([r["fd_c"]/max(abs(r["delta_cmd"]),1e-9) for r in a_res]) if a_res else 0
    fd_b = np.mean([r["fd_c"]/max(abs(r["delta_cmd"]),1e-9) for r in b_res]) if b_res else 0
    
    lines = [f"# Controller Comparison",
             f"A_ff: n={len(a_res)} sign={sa_a:.3f} ratio={dr_a:.3f} fd_contrib={fd_a:.3f}",
             f"B_ff_pi: n={len(b_res)} sign={sa_b:.3f} ratio={dr_b:.3f} fd_contrib={fd_b:.3f}"]
    for ci in cids:
        ca = [r for r in a_res if r["cyl"]==ci]; cb = [r for r in b_res if r["cyl"]==ci]
        if ca and cb:
            lines.append(f"  cyl {ci}: A sign={np.mean([r['sign_ok'] for r in ca]):.2f} ratio={np.mean([r['disp_r'] for r in ca]):.2f} | B sign={np.mean([r['sign_ok'] for r in cb]):.2f} ratio={np.mean([r['disp_r'] for r in cb]):.2f}")
    
    if sa_b > sa_a + 0.01:
        rec = "horizon_idfd_ff_pi (PI) — improved sign accuracy"
    elif abs(1.0-dr_b) < abs(1.0-dr_a) and sa_b >= sa_a:
        rec = "horizon_idfd_ff_pi (PI) — better displacement ratio"
    else:
        rec = "current_horizon_idfd_ff — simpler, equal performance"
    lines.append(f"\nRecommendation: {rec}")
    (outdir / "horizon_controller_comparison_report.md").write_text("\n".join(lines))
    print(rec, flush=True)

env.close()
simulation_app.close()
