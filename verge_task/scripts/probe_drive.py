#!/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
"""Grounding probe: is the truck really driving on the terrain?

Answers the three physical questions that have to be true before any arm task
is layered on top, and that no tracking metric would ever reveal:

*   **Is it moving, and at the commanded speed?**  Measured forward travel is
    compared against ``vehicle_speed_*_kmh``.  A truck whose wheels spin while
    the chassis stays put reports a speed near zero.
*   **Is it on the road, or floating?**  Chassis height above the flat road is
    reported over the whole run.  A floating truck's value climbs; one sinking
    into the plate goes negative.
*   **Does the height scan actually see the verge?**  The fraction of rays that
    hit is reported.  A scan that misses silently falls back to the strip's base
    height, which looks like perfectly flat ground rather than like an error.
*   **How far is the cutter above the verge?**  The quantity the mowing task
    will regulate, and the check on the -0.79 m tool offset.
*   **Is it holding its line?**  Lateral deviation, now that the root-pose pin
    that used to enforce it has been removed.

The arm is commanded zero throughout, so anything observed here is the vehicle
and the terrain, not the policy.
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

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--num_envs", type=int, default=8)
p.add_argument("--seed", type=int, default=17)
p.add_argument("--steps", type=int, default=400)
p.add_argument("--speed_kmh", type=float, default=4.0)
p.add_argument("--ride_height_target_m", type=float, default=None)
p.add_argument("--eval_terrain", action="store_true")
p.add_argument(
    "--path_family",
    default="mixed",
    choices=(
        "mixed",
        "straight",
        "lateral_sine",
        "s_curve",
        "random_spline",
        "offset_return",
        "boom_sweep",
        "lateral_chirp",
        "double_s_curve",
        "lane_change_hold",
        "gaussian_slalom",
        "fourier_meander",
    ),
    help="Red tracking-path family; mixed cycles all families across environments.",
)
p.add_argument(
    "--terrain_height_scale",
    type=float,
    default=0.0,
    help="Positive relief-scale override; zero uses the balanced curriculum.",
)
p.add_argument("--report", default="")
p.add_argument("--full_preposition", action="store_true",
               help="Use the production deployment ramp instead of the short probe one.")
p.add_argument(
    "--deployment_correction_steps",
    type=int,
    default=None,
    help="Override reset-only terrain deployment correction steps.",
)
p.add_argument("--deployment_shape", default="")
p.add_argument(
    "--randomize_start_phase",
    action=argparse.BooleanOptionalAction,
    default=None,
)
p.add_argument(
    "--debug_scan",
    action="store_true",
    help="Print ray origins/directions/hits and verge bounds at selected steps.",
)
p.add_argument(
    "--debug_body_paths",
    action="store_true",
    help="Print articulation body names and resolved USD link paths.",
)
p.add_argument(
    "--contact_tracking_demo",
    action="store_true",
    help=(
        "Run the non-PPO terrain-height/contact-force controller used to prove "
        "the physical task before training."
    ),
)
p.add_argument("--contact_target_force_n", type=float, default=None)
p.add_argument("--contact_search_step_m", type=float, default=None)
p.add_argument("--contact_force_step_m", type=float, default=None)
p.add_argument("--contact_max_offset_m", type=float, default=None)
p.add_argument(
    "--screen_actions",
    action="store_true",
    help="Apply +/- basis actions across environment groups and report response.",
)
p.add_argument("--screen_action_magnitude", type=float, default=0.5)
p.add_argument(
    "--repeat_resets",
    type=int,
    default=0,
    help="Repeat the same seeded reset and report whether deployment/sensor geometry drifts.",
)
AppLauncher.add_app_launcher_args(p)
a, unknown = p.parse_known_args()
sys.argv = [sys.argv[0], *unknown]
if a.terrain_height_scale < 0.0:
    p.error("--terrain_height_scale cannot be negative")

launcher = AppLauncher(a)
simulation_app = launcher.app

import torch  # noqa: E402
from isaaclab.utils.math import quat_apply  # noqa: E402

from verge_task.terrain_env import TerrainSolePolicyEnv  # noqa: E402
from verge_task.terrain_env_cfg import TerrainSolePolicyEnvCfg  # noqa: E402


def main() -> None:
    cfg = TerrainSolePolicyEnvCfg()
    cfg.scene.num_envs = int(a.num_envs)
    cfg.sim.device = a.device
    cfg.vehicle_speed_min_kmh = float(a.speed_kmh)
    cfg.vehicle_speed_max_kmh = float(a.speed_kmh)
    cfg.use_eval_terrain = bool(a.eval_terrain)
    cfg.terrain_base_seed = int(a.seed)
    cfg.terrain_height_scale_override = float(a.terrain_height_scale)
    cfg.tracking_path_family = str(a.path_family)
    cfg.enable_visual_grid_floor = not bool(a.headless)
    cfg.enable_road_scenery = not bool(a.headless)
    if cfg.enable_road_scenery:
        cfg.dark_visual_background = False
        cfg.visual_floor_color = (0.82, 0.83, 0.84)
        cfg.visual_grid_color = (0.90, 0.90, 0.91)
    cfg.enable_contact_tracking_demo = bool(a.contact_tracking_demo)
    if a.ride_height_target_m is not None:
        cfg.truck_ride_height_target_m = float(a.ride_height_target_m)
    if a.contact_target_force_n is not None:
        cfg.contact_demo_target_force_n = float(a.contact_target_force_n)
    if a.contact_search_step_m is not None:
        cfg.contact_demo_search_step_m = float(a.contact_search_step_m)
    if a.contact_force_step_m is not None:
        cfg.contact_demo_force_step_m = float(a.contact_force_step_m)
    if a.contact_max_offset_m is not None:
        cfg.contact_demo_max_vertical_offset_m = float(a.contact_max_offset_m)
    cfg.trajectory_library_path = str(
        ROOT / "mulag_plant/models/plant_reachable_trajectory_library_v3_large.json"
    )
    cfg.command_rom_checkpoint = str(
        ROOT
        / "rom_model/plant_reachable_v1_finetune_blend25/best_rom.pt"
    )
    if a.deployment_correction_steps is not None:
        cfg.terrain_deployment_correction_steps = int(a.deployment_correction_steps)
    if a.deployment_shape:
        cfg.trajectory_shape = str(a.deployment_shape)
    if a.randomize_start_phase is not None:
        cfg.randomize_start_phase = bool(a.randomize_start_phase)

    # The inherited reset runs a 300-step cosine preposition, a 120-step settle
    # and a 120-step ROM warm-up before the first scored transition. None of that
    # tells us anything about the vehicle, and against a collision mesh it is the
    # dominant cost of this probe, so shorten it. The arm is commanded zero
    # throughout either way.
    if not a.full_preposition:
        cfg.preposition_steps = 40
        cfg.settle_steps = 20
        cfg.rom_reset_warmup_steps = 20
    # A visualization/probe must not silently reset before its requested step
    # count. A command that crosses the 960-step training episode boundary
    # boundary, making final displacement appear zero and falsely failing speed.
    requested_duration_s = (
        int(a.steps) * float(cfg.sim.dt) * int(getattr(cfg, "decimation", 2))
    )
    cfg.episode_length_s = max(float(cfg.episode_length_s), requested_duration_s + 1.0)

    print("[PPO3-TERRAIN] building scene (terrain generation takes ~1 min)...", flush=True)
    env = TerrainSolePolicyEnv(cfg)
    print("[PPO3-TERRAIN] scene built; resetting...", flush=True)
    env.reset(seed=int(a.seed))
    print("[PPO3-TERRAIN] reset done; driving...", flush=True)

    if a.debug_body_paths:
        link_paths = getattr(env.robot.root_physx_view, "link_paths", None)
        print(
            "[PPO3-BODY-PATHS] "
            + json.dumps(
                {
                    "body_names": list(env.robot.body_names),
                    "link_paths": link_paths,
                },
                default=str,
            ),
            flush=True,
        )

    base_id = env._truck_base_body_id()
    wheel_body_ids, wheel_body_names = env.robot.find_bodies("wheel_.*")
    start_xy = env.robot.data.body_link_state_w[:, base_id, :2].clone()
    zero = torch.zeros((env.num_envs, 4), device=env.device)
    if a.repeat_resets > 0:
        reset_rows = []
        for reset_index in range(int(a.repeat_resets)):
            env.reset(seed=int(a.seed))
            # Refresh ray hits once, matching the first scored convergence step.
            env.step(zero)
            base_local = (
                env.robot.data.body_link_state_w[:, base_id, :3]
                - env.scene.env_origins
            )
            ray_starts = env._height_scanner._ray_starts_w
            ray_local_xy = ray_starts[..., :2] - env.scene.env_origins[:, None, :2]
            reset_rows.append(
                {
                    "reset": reset_index,
                    "scan_valid_mean": float(env.scan_valid_fraction().mean()),
                    "scan_valid_min": float(env.scan_valid_fraction().min()),
                    "base_local_xyz_mean": base_local.mean(dim=0).tolist(),
                    "ray_local_x_mean": float(ray_local_xy[..., 0].mean()),
                    "ray_local_y_mean": float(ray_local_xy[..., 1].mean()),
                    "cut_error_mean": float(env.cut_height_error().abs().mean()),
                    "lateral_error_mean": float(env.verge_lateral_error().abs().mean()),
                }
            )
        print("[PPO3-RESET-REPEAT] " + json.dumps(reset_rows), flush=True)
        env.reset(seed=int(a.seed))
        start_xy = env.robot.data.body_link_state_w[:, base_id, :2].clone()
    screen_action = zero.clone()
    screen_labels = []
    if a.screen_actions:
        for env_id in range(env.num_envs):
            group = env_id % 8
            cylinder = group // 2
            sign = 1.0 if group % 2 == 0 else -1.0
            screen_action[env_id, cylinder] = sign * float(a.screen_action_magnitude)
            screen_labels.append(f"cyl{cylinder + 1}{'+' if sign > 0 else '-'}")
    screen_start_cut = env.cutter_height_above_ground().detach().clone()
    screen_start_lateral = env.verge_lateral_error().detach().clone()
    screen_start_head = env._measure_head_down_alignment().detach().clone()

    ride, lateral, forward, contact, cutter_h, scan_valid = [], [], [], [], [], []
    contact_force, wheel_support, wheel_force, ground_height = [], [], [], []
    cutter_origin_z, base_tilt = [], []
    policy_obs, rewards, terminated_steps, truncated_steps = [], [], [], []
    cutter_xy, base_xy_track, line_error = [], [], []
    dt = float(env.step_dt)
    for i in range(int(a.steps)):
        action = screen_action if a.screen_actions else zero
        obs, reward, terminated, truncated, _ = env.step(action)
        policy_obs.append(obs["policy"].detach().clone())
        rewards.append(reward.detach().clone())
        terminated_steps.append(terminated.detach().clone())
        truncated_steps.append(truncated.detach().clone())
        if a.debug_scan and i in (0, 1, 10, 50, 100, int(a.steps) - 1):
            starts = env._height_scanner._ray_starts_w[0].detach().cpu()
            directions = env._height_scanner._ray_directions_w[0].detach().cpu()
            hits = env._height_scanner.data.ray_hits_w[0].detach().cpu()
            valid = torch.isfinite(hits[:, 2])
            origin = env.scene.env_origins[0].detach().cpu()
            strip = env._mow_strip_cfg()
            print(
                "[PPO3-SCAN-DEBUG] "
                f"step={i} valid={int(valid.sum())}/{valid.numel()} "
                f"origin_x=[{float(starts[:, 0].min()):.3f}, {float(starts[:, 0].max()):.3f}] "
                f"origin_y=[{float(starts[:, 1].min()):.3f}, {float(starts[:, 1].max()):.3f}] "
                f"origin_z=[{float(starts[:, 2].min()):.3f}, {float(starts[:, 2].max()):.3f}] "
                f"direction={directions[0].tolist()} "
                f"strip_x=[{float(origin[0] + strip.start_x_m):.3f}, "
                f"{float(origin[0] + strip.start_x_m + strip.length_m):.3f}] "
                f"strip_y=[{float(origin[1] + strip.center_y_m - 0.5 * strip.width_m):.3f}, "
                f"{float(origin[1] + strip.center_y_m + 0.5 * strip.width_m):.3f}] "
                f"valid_hit_x=[{float(hits[valid, 0].min()) if valid.any() else float('nan'):.3f}, "
                f"{float(hits[valid, 0].max()) if valid.any() else float('nan'):.3f}] "
                f"valid_hit_y=[{float(hits[valid, 1].min()) if valid.any() else float('nan'):.3f}, "
                f"{float(hits[valid, 1].max()) if valid.any() else float('nan'):.3f}] "
                f"terminated={bool(terminated[0])} truncated={bool(truncated[0])} "
                f"preposition={int(env._preposition_counter[0])} "
                f"settle={int(env._settle_counter[0])}",
                flush=True,
            )
        if i % 50 == 0:
            contact_offset_text = (
                f"  contact_offset={float(env._contact_demo_vertical_offset.mean()):.3f} m"
                if a.contact_tracking_demo
                else ""
            )
            print(
                f"  step {i:4d}  ride_height={float(env.truck_ride_height().mean()):.3f} m"
                f"  cutter_above_verge={float(env.cutter_height_above_ground().mean()):.3f} m"
                f"  cutter_force={float(env.cutter_terrain_contact_force_n().mean()):.1f} N"
                f"  wheel_support={float(env.wheel_road_contact_fraction().mean()):.2f}"
                + contact_offset_text
                + f"  scan_valid={float(env.scan_valid_fraction().mean()):.2f}",
                flush=True,
            )
        xy = env.robot.data.body_link_state_w[:, base_id, :2]
        delta = xy - start_xy
        forward.append(torch.linalg.norm(delta, dim=-1).clone())
        lateral.append(torch.abs(delta[:, 1]).clone())
        ride.append(env.truck_ride_height().clone())
        contact.append(env.cutter_ground_contact().float().clone())
        contact_force.append(env.cutter_terrain_contact_force_n().clone())
        wheel_support.append(env.wheel_road_contact_fraction().clone())
        wheel_force.append(env.wheel_road_contact_forces_n().clone())
        ground_height.append(env.ground_height_under_cutter().clone())
        cutter_origin_z.append(
            env.robot.data.body_link_state_w[:, env._ee_body_id, 2].clone()
        )
        base_quat = env.robot.data.body_link_state_w[:, base_id, 3:7]
        base_up = torch.zeros((env.num_envs, 3), device=env.device)
        base_up[:, 2] = 1.0
        base_up = quat_apply(base_quat, base_up)
        base_tilt.append(
            torch.acos(torch.clamp(base_up[:, 2], min=-1.0, max=1.0))
        )
        cutter_h.append(env.cutter_height_above_ground().clone())
        scan_valid.append(env.scan_valid_fraction().clone())
        # Where the tool actually travels, in env-local coordinates. The verge
        # strip has to be laid under THIS, not under a config value copied from a
        # disabled visual-only shelf.
        cutter_xy.append(
            (env._compute_cutter_pos_w()[:, :2] - env.scene.env_origins[:, :2]).clone()
        )
        line_error.append(env.verge_lateral_error().clone())
        base_xy_track.append(
            (env.robot.data.body_link_state_w[:, base_id, :2]
             - env.scene.env_origins[:, :2]).clone()
        )

    travelled = torch.stack(forward)[-1]
    elapsed = int(a.steps) * dt
    measured_kmh = (travelled / max(elapsed, 1e-6)) * 3.6
    ride_t = torch.stack(ride)
    obs_t = torch.stack(policy_obs)
    reward_t = torch.stack(rewards)
    contact_t = torch.stack(contact)
    contact_force_t = torch.stack(contact_force)
    wheel_support_t = torch.stack(wheel_support)
    wheel_force_t = torch.stack(wheel_force)
    ground_height_t = torch.stack(ground_height)
    cutter_origin_z_t = torch.stack(cutter_origin_z)
    base_tilt_t = torch.stack(base_tilt)
    acquisition_start = min(max(int(a.steps) // 3, 1), max(int(a.steps) - 1, 0))
    wheel_quat = env.robot.data.body_link_state_w[0, wheel_body_ids, 3:7]
    local_axle = torch.zeros((len(wheel_body_ids), 3), device=env.device)
    local_axle[:, 1] = 1.0
    world_axle = quat_apply(wheel_quat, local_axle)
    wheel_axle_straight_error_deg = torch.rad2deg(
        torch.atan2(world_axle[:, 0].abs(), world_axle[:, 1].abs().clamp(min=1.0e-8))
    )

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "num_envs": int(a.num_envs),
        "steps": int(a.steps),
        "elapsed_s": elapsed,
        "commanded_speed_kmh": float(a.speed_kmh),
        "measured_speed_kmh_mean": float(measured_kmh.mean()),
        "measured_speed_kmh_min": float(measured_kmh.min()),
        "measured_speed_kmh_max": float(measured_kmh.max()),
        "distance_travelled_m_mean": float(travelled.mean()),
        "ride_height_m_mean": float(ride_t.mean()),
        "ride_height_m_min": float(ride_t.min()),
        "ride_height_m_max": float(ride_t.max()),
        "ride_height_drift_m": float(ride_t[-1].mean() - ride_t[0].mean()),
        "lateral_deviation_m_max": float(torch.stack(lateral).max()),
        "cutter_contact_fraction": float(contact_t.mean()),
        "cutter_contact_fraction_after_acquisition": float(
            contact_t[acquisition_start:].mean()
        ),
        "cutter_contact_fraction_after_acquisition_per_env": contact_t[
            acquisition_start:
        ].mean(dim=0).detach().cpu().tolist(),
        "cutter_contact_force_n_mean_after_acquisition": float(
            contact_force_t[acquisition_start:].mean()
        ),
        "cutter_contact_force_n_p95_after_acquisition": float(
            torch.quantile(contact_force_t[acquisition_start:], 0.95)
        ),
        "wheel_road_contact_fraction_mean": float(wheel_support_t.mean()),
        "wheel_road_contact_fraction_min": float(wheel_support_t.min()),
        "wheel_road_force_n_mean_per_wheel": wheel_force_t.mean(
            dim=(0, 1)
        ).detach().cpu().tolist(),
        "wheel_road_force_n_p95_per_wheel": torch.quantile(
            wheel_force_t.reshape(-1, 4), 0.95, dim=0
        ).detach().cpu().tolist(),
        "wheel_road_contact_duty_cycle_per_wheel": (
            wheel_force_t
            > float(getattr(env.cfg, "road_contact_force_threshold_n", 10.0))
        ).float().mean(dim=(0, 1)).detach().cpu().tolist(),
        "base_tilt_deg_max": float(torch.rad2deg(base_tilt_t).max()),
        # How far the tool sits above the ground beneath it -- the quantity the
        # mowing task will have to regulate to target_cut_height_m.
        "scan_valid_fraction_mean": float(torch.stack(scan_valid).mean()),
        "scan_valid_fraction_min": float(torch.stack(scan_valid).min()),
        "cutter_env_local_x_m_min": float(torch.stack(cutter_xy)[..., 0].min()),
        "cutter_env_local_x_m_max": float(torch.stack(cutter_xy)[..., 0].max()),
        "cutter_env_local_y_m_mean": float(torch.stack(cutter_xy)[..., 1].mean()),
        "cutter_env_local_y_m_min": float(torch.stack(cutter_xy)[..., 1].min()),
        "cutter_env_local_y_m_max": float(torch.stack(cutter_xy)[..., 1].max()),
        "cutter_max_abs_line_error_m_per_env": torch.stack(line_error).abs()
        .amax(dim=0)
        .detach()
        .cpu()
        .tolist(),
        "cutter_p95_abs_line_error_m": float(
            torch.quantile(torch.stack(line_error).abs(), 0.95)
        ),
        "cutter_mean_signed_line_error_m": float(torch.stack(line_error).mean()),
        "truck_env_local_y_m_mean": float(torch.stack(base_xy_track)[..., 1].mean()),
        "wheel_body_names": list(wheel_body_names),
        "tracking_path_families": list(env._tracking_path_family_names),
        "terrain_curriculum_version": str(env.cfg.terrain_curriculum_version),
        "terrain_height_scales": list(env._terrain_height_scales),
        "wheel_axle_straight_error_deg": wheel_axle_straight_error_deg.detach()
        .cpu()
        .tolist(),
        "front_steering_joint_position_rad": env.robot.data.joint_pos[
            0, env._steering_joint_ids
        ].detach().cpu().tolist(),
        "wheel_center_world_z_m": env.robot.data.body_link_state_w[
            0, wheel_body_ids, 2
        ].detach().cpu().tolist(),
        "wheel_nominal_bottom_world_z_m": (
            env.robot.data.body_link_state_w[0, wheel_body_ids, 2]
            - float(getattr(env.cfg, "vehicle_wheel_radius_m", 0.575))
        ).detach().cpu().tolist(),
        "cutter_world_z_m_mean": float(env._cutter_world_z().mean()),
        "terrain_height_range_m": float(
            ground_height_t.max() - ground_height_t.min()
        ),
        "cutter_origin_height_range_m": float(
            cutter_origin_z_t.max() - cutter_origin_z_t.min()
        ),
        "cutter_height_above_ground_m_mean": float(torch.stack(cutter_h).mean()),
        "cutter_height_above_ground_m_min": float(torch.stack(cutter_h).min()),
        "cutter_height_above_ground_m_max": float(torch.stack(cutter_h).max()),
        "contact_demo_acquired_per_env": (
            env._contact_demo_acquired.detach().cpu().tolist()
            if a.contact_tracking_demo
            else []
        ),
        "contact_demo_vertical_offset_m_per_env": (
            env._contact_demo_vertical_offset.detach().cpu().tolist()
            if a.contact_tracking_demo
            else []
        ),
        "final_cylinder_extensions_m": env._refresh_cylinder_state()[0][0]
        .detach()
        .cpu()
        .tolist(),
        "final_cylinder_soft_limits_m": env.robot.data.soft_joint_pos_limits[
            0, env._cylinder_joint_ids
        ].detach().cpu().tolist(),
        "policy_observation_dim": int(obs_t.shape[-1]),
        "policy_observation_all_finite": bool(torch.isfinite(obs_t).all()),
        "policy_observation_abs_max": float(obs_t.abs().max()),
        "initial_height_profile_inter_env_std_mean": (
            float(obs_t[0, :, :9].std(dim=0).mean())
            if env.num_envs > 1
            else 0.0
        ),
        "reward_all_finite": bool(torch.isfinite(reward_t).all()),
        "reward_mean": float(reward_t.mean()),
        "reward_min": float(reward_t.min()),
        "reward_max": float(reward_t.max()),
        "terminated_fraction": float(torch.stack(terminated_steps).float().mean()),
        "truncated_fraction": float(torch.stack(truncated_steps).float().mean()),
    }
    if a.screen_actions:
        end_cut = env.cutter_height_above_ground().detach()
        end_lateral = env.verge_lateral_error().detach()
        end_head = env._measure_head_down_alignment().detach()
        rows = []
        for group in range(8):
            ids = torch.arange(group, env.num_envs, 8, device=env.device)
            rows.append(
                {
                    "action": screen_labels[group],
                    "envs": int(ids.numel()),
                    "delta_cut_height_m": float(
                        (end_cut[ids] - screen_start_cut[ids]).mean()
                    ),
                    "delta_lateral_error_m": float(
                        (end_lateral[ids] - screen_start_lateral[ids]).mean()
                    ),
                    "delta_head_alignment": float(
                        (end_head[ids] - screen_start_head[ids]).mean()
                    ),
                }
            )
        report["action_screen"] = rows

    speed_error = abs(report["measured_speed_kmh_mean"] - float(a.speed_kmh))
    report["is_moving"] = report["measured_speed_kmh_mean"] > 0.5 * float(a.speed_kmh)
    report["speed_within_20_percent"] = speed_error <= 0.2 * float(a.speed_kmh)
    # A grounded truck must have measured tyre/road interaction. Chassis height
    # alone previously mislabeled a visibly floating fixture as grounded.
    report["is_grounded"] = (
        abs(report["ride_height_drift_m"]) < 0.01
        and max(abs(value) for value in report["wheel_nominal_bottom_world_z_m"])
        <= 0.01
        and min(report["wheel_road_contact_duty_cycle_per_wheel"]) >= 0.03
    )
    report["truck_stable_under_arm_motion"] = (
        report["lateral_deviation_m_max"] <= 0.001
        and report["base_tilt_deg_max"] <= 1.0
        and max(abs(value) for value in report["wheel_nominal_bottom_world_z_m"])
        <= 0.01
        and min(report["wheel_road_contact_duty_cycle_per_wheel"]) >= 0.03
    )
    front_axle_errors = [
        error
        for name, error in zip(
            report["wheel_body_names"], report["wheel_axle_straight_error_deg"]
        )
        if "front" in name
    ]
    report["front_wheels_straight"] = (
        len(front_axle_errors) == 2 and max(front_axle_errors) <= 1.0
    )
    line_tolerance = float(
        getattr(env.cfg, "contact_demo_line_tolerance_m", 0.30)
    )
    report["contact_demo_line_tolerance_m"] = line_tolerance
    report["path_family_results"] = [
        {
            "family": family,
            "contact_fraction": contact,
            "max_abs_line_error_m": error,
            "passed": contact >= 0.50 and error <= line_tolerance,
        }
        for family, contact, error in zip(
            report["tracking_path_families"],
            report["cutter_contact_fraction_after_acquisition_per_env"],
            report["cutter_max_abs_line_error_m_per_env"],
        )
    ]
    report["cutter_tracks_contact_line"] = (
        min(report["cutter_contact_fraction_after_acquisition_per_env"]) >= 0.50
        and max(report["cutter_max_abs_line_error_m_per_env"]) <= line_tolerance
        and report["terrain_height_range_m"] >= 0.01
        and report["cutter_origin_height_range_m"] >= 0.005
    )
    # A height scan that mostly misses makes every downstream ground-tracking
    # number a fallback constant rather than a measurement.
    report["scan_sees_verge"] = report["scan_valid_fraction_mean"] > 0.8
    report["task_signals_valid"] = (
        report["policy_observation_dim"] == int(env._expected_observation_dim())
        and report["policy_observation_all_finite"]
        and report["reward_all_finite"]
    )
    report["terrain_profiles_differ"] = (
        report["initial_height_profile_inter_env_std_mean"] > 1.0e-4
        if env.num_envs > 1
        else True
    )
    report["passed"] = (
        report["is_moving"]
        and report["speed_within_20_percent"]
        and report["is_grounded"]
        and report["truck_stable_under_arm_motion"]
        and report["front_wheels_straight"]
        and report["scan_sees_verge"]
        and report["task_signals_valid"]
        and report["terrain_profiles_differ"]
        and (
            report["cutter_tracks_contact_line"]
            if a.contact_tracking_demo
            else True
        )
    )
    if hasattr(env, "_last_deployment_template"):
        report["deployment_template_env"] = int(env._last_deployment_template)
        report["deployment_start_s"] = float(env._last_deployment_start_s)
        report["deployment_target_cylinders_m"] = (
            env._last_deployment_target_x.detach().cpu().tolist()
        )

    print(json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        print(
            "[PPO3-TERRAIN] GROUNDING PROBE FAILED. Do not build the arm task on "
            "this: a truck that is floating, stalled, or sinking makes every "
            "downstream ground-tracking number meaningless.",
            flush=True,
        )

    if a.report:
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(json.dumps(report, indent=2))

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
