#!/usr/bin/env python3
"""Render clean, high-resolution thesis views of the verge-following task.

The output is captured from Isaac Sim's RTX viewport rather than from the UI,
so it contains the production scene but no menus, overlays, or annotations.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output_dir", type=Path, required=True)
parser.add_argument("--capture_width", type=int, default=2560)
parser.add_argument("--capture_height", type=int, default=1440)
parser.add_argument("--seed", type=int, default=17)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

launcher = AppLauncher(args)
simulation_app = launcher.app

import torch  # noqa: E402
from PIL import Image  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors.camera import CameraCfg  # noqa: E402

from verge_task.terrain_env import TerrainSolePolicyEnv  # noqa: E402
from verge_task.terrain_env_cfg import TerrainSolePolicyEnvCfg  # noqa: E402


def _capture(env: TerrainSolePolicyEnv, path: Path) -> None:
    """Render the stage through Isaac Lab's native RTX camera sensor."""
    camera = env.scene["thesis_camera"]
    print(f"[THESIS-RENDER] Accumulating RTX frame for {path.name}...", flush=True)
    env.sim.render()
    camera.update(float(env.sim.get_physics_dt()), force_recompute=True)
    rgb = camera.data.output["rgb"][0, ..., :3].detach().cpu().numpy()
    Image.fromarray(rgb).save(path, format="PNG", compress_level=2)


def main() -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = TerrainSolePolicyEnvCfg()
    cfg.scene.num_envs = 1
    cfg.scene.thesis_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/ThesisCamera",
        update_period=0.0,
        height=int(args.capture_height),
        width=int(args.capture_width),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
    )
    cfg.sim.device = args.device
    # Avoid paying for 4K RTX frames during the reset-only deployment motion.
    # The viewport is refreshed explicitly once the final pose is reached.
    cfg.sim.render_interval = 10_000
    cfg.terrain_base_seed = int(args.seed)
    cfg.terrain_eval_seed_override = int(args.seed)
    cfg.use_eval_terrain = True
    cfg.tracking_path_family = "double_s_curve"
    cfg.vehicle_speed_min_kmh = 4.0
    cfg.vehicle_speed_max_kmh = 4.0
    cfg.enable_visual_grid_floor = False
    cfg.enable_road_scenery = True
    cfg.enable_road_edge_markings = False
    cfg.mow_strip_use_vertex_palette = True
    cfg.enable_verge_grass_tufts = True
    cfg.dark_visual_background = False
    cfg.preposition_steps = 40
    cfg.settle_steps = 12
    cfg.rom_reset_warmup_steps = 8
    cfg.terrain_deployment_correction_steps = 12
    cfg.terrain_deployment_settle_steps = 4
    cfg.trajectory_library_path = str(
        ROOT / "mulag_plant/models/plant_reachable_trajectory_library_v3_large.json"
    )
    cfg.command_rom_checkpoint = str(
        ROOT / "learned_models/rom_model/plant_reachable_v1_finetune_blend25/best_rom.pt"
    )

    print("[THESIS-RENDER] Building the production verge scene...", flush=True)
    env = TerrainSolePolicyEnv(cfg)
    # A figure does not need a second, scored episode reset. Drive the real
    # articulation to the same validated deterministic cylinder state with a
    # short non-rendered cosine ramp so every passive linkage is resolved.
    cylinder_target = torch.as_tensor(
        cfg.terrain_deployment_cylinder_target_m,
        dtype=torch.float32,
        device=env.device,
    ).reshape(1, 4)
    start, _ = env._refresh_cylinder_state()
    scaling = env._cylinder_adapter.scaling
    dt = float(env.sim.get_physics_dt())
    coupling_target_id = getattr(env, "_joint6_coupling_target_joint_id", None)
    coupling_source_id = getattr(env, "_joint6_coupling_source_joint_id", None)
    for step in range(80):
        u = float(step + 1) / 80.0
        blend = 0.5 * (1.0 - math.cos(math.pi * u))
        learned_goal = start + blend * (cylinder_target - start)
        usd_goal = scaling.clamp_usd_q(
            scaling.learned_x_to_usd_q(learned_goal)
        )
        env.robot.set_joint_position_target(
            usd_goal, joint_ids=env._cylinder_joint_ids
        )
        if coupling_target_id is not None and coupling_source_id is not None:
            source_column = int(
                (env._cylinder_joint_ids == int(coupling_source_id))
                .nonzero(as_tuple=False)[0, 0]
                .item()
            )
            coupled_goal = (
                float(getattr(cfg, "joint6_coupling_a", 0.76))
                * usd_goal[:, source_column : source_column + 1]
                + float(getattr(cfg, "joint6_coupling_b", 0.0))
            )
            env.robot.set_joint_position_target(
                coupled_goal, joint_ids=[int(coupling_target_id)]
            )
        env._hold_wheels()
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt)
    for _ in range(20):
        env.robot.set_joint_position_target(
            usd_goal, joint_ids=env._cylinder_joint_ids
        )
        env._hold_wheels()
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt)
    print("[THESIS-RENDER] Canonical physical deployment completed.", flush=True)

    base_id = env._truck_base_body_id()
    base = env.robot.data.body_link_state_w[0, base_id, :3].detach().cpu()
    bx, by, _ = (float(value) for value in base)
    views = {
        "verge_task_hero.png": (
            (bx - 8.7, by - 10.8, 6.6),
            (bx + 1.4, by - 1.6, 0.75),
        ),
        "verge_task_wide.png": (
            (bx - 11.5, by - 13.8, 8.7),
            (bx + 3.0, by - 1.7, 0.55),
        ),
    }

    for name, (eye, target) in views.items():
        path = args.output_dir / name
        print(f"[THESIS-RENDER] Positioning camera for {name}...", flush=True)
        camera = env.scene["thesis_camera"]
        camera.set_world_poses_from_view(
            torch.tensor([eye], dtype=torch.float32, device=env.device),
            torch.tensor([target], dtype=torch.float32, device=env.device),
        )
        _capture(env, path)
        print(f"[THESIS-RENDER] Wrote {path}", flush=True)

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
