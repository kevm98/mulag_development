#!/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
"""Visualize the factor-blind ROM-B policy in one Isaac Sim environment."""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from collections import deque
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isaaclab.app import AppLauncher


p = argparse.ArgumentParser(description=__doc__)
p.add_argument(
    "--checkpoint",
    default=str(
        ROOT / "logs/rsl_rl/ppo2_rom_b_robust/2026-07-24_12-48-14"
        / "model_53_ppo2_rom_b_final.pt"
    ),
)
p.add_argument(
    "--trajectory_library_path",
    default=str(
        ROOT / "PPO2_ROM_PhysX/models"
        / "plant_reachable_trajectory_library_v3_circle_c7_retime.json"
    ),
)
p.add_argument(
    "--trajectory_shape",
    choices=("line", "arc", "sine", "circle", "square"),
    default="circle",
)
p.add_argument("--path_speed_mps", type=float, default=0.18)
p.add_argument("--seed", type=int, default=17)
p.add_argument(
    "--episodes",
    type=int,
    default=1,
    help="Number of A->B->A episodes to show before closing (default: 1).",
)
p.add_argument(
    "--randomized_plant",
    action="store_true",
    help="Sample all five hidden ROM-B factors at every reset.",
)
p.add_argument(
    "--fixed_h",
    type=float,
    default=None,
    help="Use one correlated h plant in [0.4, 1.2] instead of random factors.",
)
p.add_argument("--realtime", action="store_true")
p.add_argument(
    "--plot_update_steps",
    type=int,
    default=15,
    help="Refresh the live arm plots every N control steps.",
)
p.add_argument(
    "--disable_live_plots",
    action="store_true",
    help="Disable the live tracking plots. Headless mode always disables them.",
)
p.add_argument(
    "--disable_viewport_trajectory",
    action="store_true",
    help="Disable the blue reference trajectory in the Isaac Sim viewport.",
)
AppLauncher.add_app_launcher_args(p)
a, unknown = p.parse_known_args()
sys.argv = [sys.argv[0], *unknown]
if a.fixed_h is not None and not 0.4 <= a.fixed_h <= 1.2:
    p.error("--fixed_h must be inside [0.4, 1.2]")
if a.fixed_h is not None and a.randomized_plant:
    p.error("choose --randomized_plant or --fixed_h, not both")
if a.plot_update_steps < 1:
    p.error("--plot_update_steps must be at least 1")
if a.episodes < 1:
    p.error("--episodes must be at least 1")

launcher = AppLauncher(a)
simulation_app = launcher.app

import gymnasium as gym
import torch
from isaaclab.utils.math import quat_apply
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg
from rsl_rl.runners import OnPolicyRunner

import PPO2_ROM_B_PhysX.tasks  # noqa: F401,E402
from PPO2_ROM_B_PhysX.env_cfg import PPO2ROMBPPORunnerCfg
from PPO2_ROM_B_PhysX.runtime import health_scale_to_factors
from PPO2_ROM_PhysX.policy.env_cfg import apply_residual_curriculum_stage

if not bool(a.headless):
    import carb
    import omni.debugdraw as omni_debug_draw
else:
    carb = None
    omni_debug_draw = None


class ViewportTrajectoryLine:
    """Keep the complete reference path visible in the Isaac viewport."""

    BLUE = 0xFF248BFF  # omni.debugdraw packed AARRGGBB

    def __init__(self, raw) -> None:
        if omni_debug_draw is None:
            raise RuntimeError("viewport trajectory requires the Isaac GUI")
        self._draw = omni_debug_draw.get_debug_draw_interface()
        base = raw.robot.data.body_link_state_w[:, raw._truck_base_body_id()]
        base_pos = base[0, :3]
        base_quat = base[0, 3:7]
        local = torch.as_tensor(raw._goals, dtype=torch.float32, device=raw.device)
        world = base_pos.reshape(1, 3) + quat_apply(
            base_quat.reshape(1, 4).expand(local.shape[0], -1), local
        )
        self._points = [tuple(map(float, row)) for row in world.cpu().tolist()]

    @staticmethod
    def _point(position):
        point = omni_debug_draw.SimplexPoint()
        point.position = carb.Float3(*position)
        point.color = ViewportTrajectoryLine.BLUE
        point.width = 4.0
        return point

    def redraw(self) -> None:
        segments = []
        for start, end in zip(self._points[:-1], self._points[1:], strict=True):
            segments.extend((self._point(start), self._point(end)))
        if segments:
            self._draw.draw_lines(segments)


class LiveArmPlots:
    """Live ROM-B tool tracking and dynamics, styled like the ROM-A plots."""

    def __init__(self, raw, *, update_steps: int) -> None:
        import matplotlib.pyplot as plt

        self._plt = plt
        self._update_steps = int(update_steps)
        self._step = 0
        self._enabled = True
        self._measured_xy: list[list[float]] = []
        self._times: list[float] = []
        self._errors_cm: list[float] = []
        self._accelerations: list[float] = []
        self._episode = 0

        plt.ion()
        self.figure, (self.track_axis, self.metric_axis) = plt.subplots(
            1, 2, figsize=(13.5, 6.2)
        )
        try:
            self.figure.canvas.manager.set_window_title("ROM-B live tracking")
        except AttributeError:
            pass

        goals = torch.as_tensor(raw._goals).cpu().numpy()
        self.track_axis.plot(
            goals[:, 0], goals[:, 1], "--", color="tab:blue",
            linewidth=2.0, label="reference",
        )
        (self.measured_line,) = self.track_axis.plot(
            [], [], color="tab:orange", linewidth=1.8, label="measured"
        )
        self.track_axis.set_xlabel("truck X [m]")
        self.track_axis.set_ylabel("truck Y [m]")
        self.track_axis.set_aspect("equal", adjustable="box")
        self.track_axis.set_title("Mower-head X-Y tracking")
        self.track_axis.legend(loc="best")
        self.track_axis.grid(True)

        (self.error_line,) = self.metric_axis.plot(
            [], [], color="tab:red", linewidth=1.8, label="tracking error [cm]"
        )
        self.accel_axis = self.metric_axis.twinx()
        (self.accel_line,) = self.accel_axis.plot(
            [], [], color="tab:green", linewidth=1.5,
            label="tool acceleration [m/s²]",
        )
        self.metric_axis.set_xlabel("time [s]")
        self.metric_axis.set_ylabel("tracking error [cm]", color="tab:red")
        self.accel_axis.set_ylabel("tool acceleration [m/s²]", color="tab:green")
        self.metric_axis.set_title("Tracking error and tool acceleration")
        self.metric_axis.grid(True)
        lines = [self.error_line, self.accel_line]
        self.metric_axis.legend(lines, [line.get_label() for line in lines], loc="best")
        self.figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
        plt.show(block=False)
        self._draw()

    def begin_episode(
        self,
        episode: int,
        total_episodes: int,
        plant: str,
        factors: torch.Tensor,
    ) -> None:
        """Clear old traces and label the plots with the active plant."""
        self._episode = int(episode)
        factor_text = str(factors.detach().cpu().numpy().round(4))
        self.figure.suptitle(
            f"ROM-B episode {episode}/{total_episodes} | "
            f"plant={plant} | factors={factor_text}",
            fontsize=10,
        )
        self.track_axis.set_title(
            f"Mower-head X-Y tracking — episode {episode}/{total_episodes}"
        )
        self.reset()

    def sample(self, raw, acceleration: float) -> None:
        if not self._enabled:
            return
        measured = raw._compute_ee_pos_local()[0]
        reference = raw._original_reference_tool[0]
        self._measured_xy.append([float(measured[0]), float(measured[1])])
        self._times.append(self._step * float(raw.step_dt))
        self._errors_cm.append(float(torch.linalg.norm(reference - measured)) * 100.0)
        self._accelerations.append(float(acceleration))
        self._step += 1
        if self._step % self._update_steps == 0:
            self._draw()

    def _draw(self) -> None:
        if not self._plt.fignum_exists(self.figure.number):
            self._enabled = False
            return
        if self._measured_xy:
            self.measured_line.set_data(
                [row[0] for row in self._measured_xy],
                [row[1] for row in self._measured_xy],
            )
            self.error_line.set_data(self._times, self._errors_cm)
            self.accel_line.set_data(self._times, self._accelerations)
            self.metric_axis.relim()
            self.metric_axis.autoscale_view()
            self.accel_axis.relim()
            self.accel_axis.autoscale_view()
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()

    def reset(self) -> None:
        self._measured_xy.clear()
        self._times.clear()
        self._errors_cm.clear()
        self._accelerations.clear()
        self._step = 0
        self.measured_line.set_data([], [])
        self.error_line.set_data([], [])
        self.accel_line.set_data([], [])
        self._draw()

    def close(self) -> None:
        if self._enabled and self._plt.fignum_exists(self.figure.number):
            self._plt.close(self.figure)
        self._enabled = False


def main() -> None:
    task = "RRLAB-Mulag-PPO2-ROM-B-Robust-Policy-PLAY-v0"
    cfg = parse_env_cfg(task, device=a.device, num_envs=1)
    cfg.seed = int(a.seed)
    apply_residual_curriculum_stage(cfg, 4)
    cfg.action_space = 4
    cfg.observation_space = 42
    cfg.trajectory_library_path = str(
        Path(a.trajectory_library_path).expanduser().resolve()
    )
    cfg.trajectory_shapes = (str(a.trajectory_shape),)
    cfg.trajectory_shape = str(a.trajectory_shape)
    cfg.randomize_trajectory_shape = False
    cfg.randomize_start_phase = False
    cfg.procedural_trajectory_probability = 0.0
    cfg.path_speed_mps = float(a.path_speed_mps)
    cfg.residual_scale_m = 0.002
    cfg.cylinder4_residual_scale_m = 0.00025
    cfg.command_rom_direction_guard = True
    cfg.clearance_frame_consistent = True
    cfg.truck_full_pose_fixture = True
    # This is replaced with the selected trajectory's exact A->B->A duration
    # after construction, once its timing profile is available.
    cfg.episode_length_s = 200.0
    use_variable_plant = bool(a.randomized_plant or a.fixed_h is not None)
    cfg.rom_b_mode = "random" if use_variable_plant else "nominal"
    cfg.rom_b_randomization_scale = 1.0 if use_variable_plant else 0.0

    gym_env = gym.make(task, cfg=cfg)
    raw = gym_env.unwrapped
    raw.cfg.episode_length_s = float(raw._complete_cycle_time)
    print(
        "[ROM-B PLAY] truck fixture=FIXED CHASSIS (actual chassis pose and "
        "wheel/steering positions fixed; arm joints remain free; visualization only)",
        flush=True,
    )
    print(
        f"[ROM-B PLAY] episode=A->B->A duration="
        f"{raw.cfg.episode_length_s:.2f}s "
        f"({raw.cfg.episode_length_s / 60.0:.2f}min); "
        "safety violations can terminate earlier",
        flush=True,
    )
    agent_cfg = PPO2ROMBPPORunnerCfg()
    agent_cfg.device = a.device
    env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=a.device)
    checkpoint = str(Path(a.checkpoint).expanduser().resolve())
    runner.load(
        checkpoint,
        load_cfg={
            "actor": True,
            "critic": True,
            "optimizer": False,
            "iteration": False,
            "rnd": False,
        },
    )
    policy = runner.get_inference_policy(device=raw.device)

    fixed_factors = None
    if a.fixed_h is not None:
        fixed_factors = health_scale_to_factors(
            float(a.fixed_h), device=raw.device
        )[0]
    plant_log = (
        f"correlated-health h={float(a.fixed_h):.3f}"
        if a.fixed_h is not None
        else ("independent-random h=n/a" if a.randomized_plant else "nominal h=1.000")
    )

    def apply_fixed_factors() -> None:
        if fixed_factors is not None:
            raw._command_cylinder_rom.true_factors.copy_(fixed_factors.reshape(1, -1))

    obs, _ = env.reset()
    apply_fixed_factors()
    plots = None
    if not bool(a.headless) and not bool(a.disable_live_plots):
        plots = LiveArmPlots(raw, update_steps=int(a.plot_update_steps))
        plots.begin_episode(
            1,
            int(a.episodes),
            plant_log,
            raw._command_cylinder_rom.true_factors[0],
        )
        print("[ROM-B PLAY] live arm plots enabled", flush=True)
    viewport_line = None
    if not bool(a.headless) and not bool(a.disable_viewport_trajectory):
        viewport_line = ViewportTrajectoryLine(raw)
        viewport_line.redraw()
        print("[ROM-B PLAY] viewport trajectory enabled: blue=reference", flush=True)
    print(
        f"[ROM-B PLAY] checkpoint={checkpoint} shape={a.trajectory_shape} "
        f"plant={'h=' + str(a.fixed_h) if a.fixed_h is not None else ('random' if a.randomized_plant else 'nominal')} "
        "| close the Isaac Sim window to stop",
        flush=True,
    )

    acceleration_window: deque[float] = deque(maxlen=120)
    episode_step = 0
    episode = 1
    try:
        while simulation_app.is_running():
            started = time.perf_counter()
            with torch.no_grad():
                obs, _, dones, _ = env.step(policy(obs))
            acceleration = float(torch.linalg.norm(
                raw._measured_tool_acceleration_truck[0]
            ).item())
            acceleration_window.append(acceleration)
            if plots is not None:
                plots.sample(raw, acceleration)
            if viewport_line is not None:
                viewport_line.redraw()
            if episode_step % 30 == 0:
                error = float(torch.linalg.norm(
                    raw._original_reference_tool[0] - raw._compute_ee_pos_local()[0]
                ).item())
                residual_mm = (
                    raw._residual_filtered[0] * raw._residual_scale_vector() * 1000.0
                )
                acc95 = float(torch.quantile(
                    torch.tensor(list(acceleration_window)), 0.95
                ).item())
                factors = raw._command_cylinder_rom.true_factors[0]
                truck_pose = raw.robot.data.body_link_state_w[
                    0, raw._truck_base_body_id(), :7
                ]
                truck_anchor = raw._rom_b_truck_anchor_body_pose[0]
                truck_position_error_mm = float(torch.linalg.norm(
                    truck_pose[:3] - truck_anchor[:3]
                ).item() * 1000.0)
                truck_q = truck_pose[3:7] / torch.linalg.norm(
                    truck_pose[3:7]
                ).clamp(min=1.0e-8)
                anchor_q = truck_anchor[3:7] / torch.linalg.norm(
                    truck_anchor[3:7]
                ).clamp(min=1.0e-8)
                quaternion_dot = torch.abs(torch.sum(truck_q * anchor_q)).clamp(max=1.0)
                truck_rotation_error_deg = float(
                    torch.rad2deg(2.0 * torch.acos(quaternion_dot)).item()
                )
                print(
                    f"episode={episode:03d}/{a.episodes:03d} "
                    f"step={episode_step:05d} "
                    f"time={episode_step * float(raw.step_dt):.2f}s "
                    f"error={error:.4f}m "
                    f"acc={acceleration:.2f} acc95_window={acc95:.2f}m/s2 "
                    f"head={float(raw._measure_head_down_alignment()[0]):.5f} "
                    f"clearance={float(raw._clearance_signal()[0]):+.4f}m "
                    f"residual_mm={residual_mm.detach().cpu().numpy().round(4)} "
                    f"truck_pose_error=({truck_position_error_mm:.3f}mm," 
                    f"{truck_rotation_error_deg:.4f}deg) "
                    f"plant={plant_log} "
                    f"factors={factors.detach().cpu().numpy().round(4)}",
                    flush=True,
                )
            if bool(dones.any()):
                acceleration_window.clear()
                apply_fixed_factors()
                reason = "timeout/round-trip" if bool(raw.reset_time_outs[0]) else "safety termination"
                print(
                    f"[ROM-B PLAY] episode {episode}/{a.episodes} finished: {reason}",
                    flush=True,
                )
                if episode >= int(a.episodes):
                    break
                episode += 1
                episode_step = 0
                if plots is not None:
                    plots.begin_episode(
                        episode,
                        int(a.episodes),
                        plant_log,
                        raw._command_cylinder_rom.true_factors[0],
                    )
            else:
                episode_step += 1
            if a.realtime:
                time.sleep(max(0.0, raw.step_dt - (time.perf_counter() - started)))
    finally:
        if plots is not None:
            plots.close()
        env.close()


try:
    main()
except BaseException:
    traceback.print_exc()
    raise
finally:
    simulation_app.close()
