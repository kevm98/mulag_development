#!/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
"""Ground truth for the cutter's world Z, from the USD geometry itself.

The clearance gate and the tracking reward disagree about where the cutter is by
about 1.58 m (`rom_b_frame_decomposition.md`). Two computations exist:

    _compute_cutter_world_z  = ee_origin_z + tool_tip_local_offset_z     (no rotation)
    _compute_cutter_pos_w    = ee_pos_w + quat_apply(ee_q, [0, 0, off])  (rotated)

Neither is authoritative. This queries the actual world-space bounding box of the
Messerkopf prim and reports which computation it agrees with, so the repair is
made against measurement rather than inference.

Read-only: no checkpoint is loaded, nothing is written to the scene.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isaaclab.app import AppLauncher


p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--num_envs", type=int, default=4)
p.add_argument("--seed", type=int, default=17)
p.add_argument("--trajectory_shape", default="line")
p.add_argument("--path_speed_mps", type=float, default=0.18)
p.add_argument(
    "--settle_steps",
    type=int,
    default=460,
    help="Steps to run before measuring, so the arm is deployed and settled.",
)
p.add_argument(
    "--command_rom_checkpoint",
    default=str(
        ROOT / "ROM-cylinders_pipeline/checkpoints"
        / "plant_reachable_v1_finetune_blend25/best_rom.pt"
    ),
)
p.add_argument(
    "--trajectory_library_path",
    default=str(
        ROOT / "PPO2_ROM_PhysX/models"
        / "plant_reachable_trajectory_library_v3_circle_c7_retime.json"
    ),
)
p.add_argument("--output", default="")
AppLauncher.add_app_launcher_args(p)
a, unknown = p.parse_known_args()
sys.argv = [sys.argv[0], *unknown]

launcher = AppLauncher(a)
simulation_app = launcher.app

import gymnasium as gym
import torch
from isaaclab_tasks.utils import parse_env_cfg
from pxr import Usd, UsdGeom
import omni.usd

import PPO2_ROM_B_PhysX.tasks  # noqa: F401,E402
from PPO2_ROM_PhysX.policy.env_cfg import apply_residual_curriculum_stage


def _world_bound(prim: Usd.Prim, time) -> tuple[float, float, float, float, float, float]:
    """World-space axis-aligned bounds of a prim subtree, as (min/max per axis)."""

    cache = UsdGeom.BBoxCache(
        time,
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=False,
    )
    box = cache.ComputeWorldBound(prim)
    rng = box.ComputeAlignedRange()
    lo, hi = rng.GetMin(), rng.GetMax()
    return (lo[0], lo[1], lo[2], hi[0], hi[1], hi[2])


def main() -> None:
    task = "RRLAB-Mulag-PPO2-ROM-B-Robust-Policy-v0"
    cfg = parse_env_cfg(task, device=a.device, num_envs=int(a.num_envs))
    cfg.seed = int(a.seed)
    apply_residual_curriculum_stage(cfg, 4)
    cfg.action_space = 4
    cfg.observation_space = 42
    cfg.command_rom_checkpoint = str(Path(a.command_rom_checkpoint).resolve())
    cfg.trajectory_library_path = str(Path(a.trajectory_library_path).resolve())
    cfg.trajectory_shape = a.trajectory_shape
    cfg.randomize_trajectory_shape = False
    cfg.procedural_trajectory_probability = 0.0
    cfg.path_speed_mps = float(a.path_speed_mps)
    cfg.force_zero_policy = True
    cfg.rom_b_mode = "nominal"
    cfg.rom_b_randomization_scale = 0.0

    gym_env = gym.make(task, cfg=cfg)
    try:
        raw = gym_env.unwrapped
        raw.set_trajectory_shape(a.trajectory_shape)
        with torch.inference_mode():
            # The gym OrderEnforcing wrapper requires its own reset before step.
            gym_env.reset(seed=int(a.seed))
            zero = torch.zeros(
                (raw.num_envs, int(cfg.action_space)), device=raw.device
            )
            for _ in range(int(a.settle_steps)):
                gym_env.step(zero)

        stage = omni.usd.get_context().get_stage()
        time = Usd.TimeCode.Default()
        ee_name = str(cfg.task_space_ee_body_name)
        offset = float(getattr(cfg, "tool_tip_local_offset_z", -0.79))

        gate = raw._compute_cutter_world_z().detach().cpu()
        rotated = raw._compute_cutter_pos_w()[:, 2].detach().cpu()
        ee_origin = raw.robot.data.body_link_state_w[
            :, raw._ee_body_id, 2
        ].detach().cpu()
        ee_quat = raw.robot.data.body_link_state_w[
            :, raw._ee_body_id, 3:7
        ].detach().cpu()
        truck_z = raw.robot.data.body_link_state_w[
            :, raw._truck_base_body_id(), 2
        ].detach().cpu()
        cutter_truck = raw._compute_ee_pos_local().detach().cpu()
        ref_truck = raw._original_reference_tool.detach().cpu()
        ref_world_z = raw._original_reference_world_z.detach().cpu()
        alignment = raw._measure_head_down_alignment().detach().cpu()

        # Discover the body prim rather than assuming a layout. Articulation
        # bodies are often nested several levels below the robot root.
        robot_root = str(raw.robot.cfg.prim_path).replace(".*", "0")
        found: dict[int, str] = {}
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if prim.GetName() != ee_name or "/envs/env_" not in path:
                continue
            env_token = path.split("/envs/env_")[1].split("/")[0]
            if not env_token.isdigit():
                continue
            env_id = int(env_token)
            # Shallowest match wins: that is the body link, not a child mesh.
            if env_id not in found or path.count("/") < found[env_id].count("/"):
                found[env_id] = path
        print(f"\n{'=' * 78}\nGROUND TRUTH: {ee_name} world bounds from USD\n{'=' * 78}")
        print(f"  robot root prim : {robot_root}")
        print(f"  body prims found: {len(found)} -> {sorted(found.values())[:2]}", flush=True)
        if not found:
            names = sorted({
                str(p.GetPath()) for p in stage.Traverse()
                if ee_name.lower() in p.GetName().lower()
            })
            print(f"  NO EXACT MATCH. Prims containing {ee_name!r}: {names[:10]}", flush=True)

        # Vertical datum: where is the ground, and where does the vehicle rest on it?
        print(f"\n{'-' * 78}\nVERTICAL DATUM\n{'-' * 78}", flush=True)
        ground_candidates = sorted({
            str(prim.GetPath()) for prim in stage.Traverse()
            if any(
                token in prim.GetName().lower()
                for token in ("groundplane", "ground", "terrain", "defaultground")
            )
        })
        ground_z = None
        for path in ground_candidates[:6]:
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                continue
            try:
                _, _, zlo, _, _, zhi = _world_bound(prim, time)
            except Exception as exc:  # non-geometric prim
                print(f"  {path}: no bound ({type(exc).__name__})", flush=True)
                continue
            print(f"  {path}: world Z [{zlo:.4f}, {zhi:.4f}]", flush=True)
            if ground_z is None:
                ground_z = zhi
        if ground_z is None:
            print("  no ground prim found; assuming world Z = 0.0", flush=True)
            ground_z = 0.0

        robot_prim = stage.GetPrimAtPath(f"/World/envs/env_0/Robot")
        vehicle_low = None
        if robot_prim.IsValid():
            _, _, vehicle_low, _, _, vehicle_high = _world_bound(robot_prim, time)
            print(
                f"  whole robot world Z bound: [{vehicle_low:.4f}, {vehicle_high:.4f}]",
                flush=True,
            )
        root_z = float(raw.robot.data.root_state_w[0, 2])
        print(f"  articulation root world Z: {root_z:.4f}", flush=True)
        print(f"  truck base body world Z  : {float(truck_z[0]):.4f}", flush=True)
        print(f"  ground datum used        : {ground_z:.4f}", flush=True)
        if vehicle_low is not None:
            print(
                f"  lowest robot geometry is {vehicle_low - ground_z:+.4f} m "
                f"relative to the ground datum", flush=True,
            )

        records = []
        for env_id in range(raw.num_envs):
            prim_path = found.get(env_id)
            prim = stage.GetPrimAtPath(prim_path) if prim_path else None
            if prim is None or not prim.IsValid():
                print(f"  env {env_id}: no valid prim for {ee_name}", flush=True)
                continue
            xlo, ylo, zlo, xhi, yhi, zhi = _world_bound(prim, time)
            record = {
                "env_id": env_id,
                "prim_path": prim_path,
                "measured_cutter_bottom_world_z": zlo,
                "measured_body_top_world_z": zhi,
                "measured_body_height_m": zhi - zlo,
                "ee_origin_world_z": float(ee_origin[env_id]),
                "ee_quat_wxyz": [float(v) for v in ee_quat[env_id]],
                "head_alignment": float(alignment[env_id]),
                "gate_compute_cutter_world_z": float(gate[env_id]),
                "rotated_compute_cutter_pos_w_z": float(rotated[env_id]),
                "truck_base_world_z": float(truck_z[env_id]),
                "measured_cutter_truck": [float(v) for v in cutter_truck[env_id]],
                "reference_cutter_truck": [float(v) for v in ref_truck[env_id]],
                "reference_world_z": float(ref_world_z[env_id]),
                "tool_tip_local_offset_z": offset,
            }
            records.append(record)
            print(
                f"\n  env {env_id}   head_alignment={record['head_alignment']:.5f}"
                f"   ee_quat(wxyz)={[round(v, 4) for v in record['ee_quat_wxyz']]}"
            )
            print(f"    ee_origin world Z                 {record['ee_origin_world_z']:>9.4f}")
            print(f"    MEASURED cutter bottom world Z    {zlo:>9.4f}   <-- ground truth")
            print(f"    MEASURED body top world Z         {zhi:>9.4f}"
                  f"   (body height {zhi - zlo:.4f} m)")
            print(f"    gate  _compute_cutter_world_z     {record['gate_compute_cutter_world_z']:>9.4f}"
                  f"   error {record['gate_compute_cutter_world_z'] - zlo:>+8.4f}")
            print(f"    rot   _compute_cutter_pos_w[2]    {record['rotated_compute_cutter_pos_w_z']:>9.4f}"
                  f"   error {record['rotated_compute_cutter_pos_w_z'] - zlo:>+8.4f}")
            print(f"    truck base world Z                {record['truck_base_world_z']:>9.4f}")
            print(f"    measured cutter (truck frame)     {record['measured_cutter_truck'][2]:>9.4f}")
            print(f"    reference cutter (truck frame)    {record['reference_cutter_truck'][2]:>9.4f}")
            print(f"    reference world Z (library)       {record['reference_world_z']:>9.4f}"
                  f"   error vs truth {record['reference_world_z'] - zlo:>+8.4f}")

        if records:
            gate_err = sum(
                abs(r["gate_compute_cutter_world_z"] - r["measured_cutter_bottom_world_z"])
                for r in records
            ) / len(records)
            rot_err = sum(
                abs(r["rotated_compute_cutter_pos_w_z"] - r["measured_cutter_bottom_world_z"])
                for r in records
            ) / len(records)
            ref_err = sum(
                abs(r["reference_world_z"] - r["measured_cutter_bottom_world_z"])
                for r in records
            ) / len(records)
            print(f"\n{'=' * 78}\nVERDICT (mean |error| vs measured cutter bottom)\n{'=' * 78}")
            print(f"  _compute_cutter_world_z (gate, unrotated)   {gate_err:>8.4f} m")
            print(f"  _compute_cutter_pos_w[2] (rotated)          {rot_err:>8.4f} m")
            print(f"  library reference_world_z                   {ref_err:>8.4f} m")
            better = "gate (unrotated)" if gate_err < rot_err else "rotated"
            truth = records[0]["measured_cutter_bottom_world_z"]
            print(f"\n  ground datum                 {ground_z:>8.4f} m")
            print(f"  true cutter bottom world Z   {truth:>8.4f} m")
            print(f"  TRUE CUTTER GROUND CLEARANCE {truth - ground_z:>8.4f} m")
            print(f"\n  -> the CUTTER COMPUTATION that matches geometry: {better}")
            print(f"  -> the library reference is {ref_err:.4f} m from the true cutter bottom")
            print("     at the trajectory start pose; a correct reference should be a")
            print("     small mowing clearance above ground, not a metre-scale offset.")

        payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "trajectory_shape": a.trajectory_shape,
            "ee_body_name": ee_name,
            "tool_tip_local_offset_z": offset,
            "settle_steps": int(a.settle_steps),
            "records": records,
        }
        out = (
            Path(a.output).expanduser().resolve()
            if a.output
            else ROOT / "PPO2_ROM_B_PhysX/reports"
            / f"cutter_frame_truth_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\n  written: {out}")
    finally:
        gym_env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
