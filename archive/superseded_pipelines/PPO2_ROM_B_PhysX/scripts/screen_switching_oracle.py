#!/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
"""Compare ROM-B and its true-factor oracle across one endpoint plant switch."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isaaclab.app import AppLauncher


BASELINE = (
    ROOT / "logs/rsl_rl/ppo2_rom_b_robust/2026-07-24_12-48-14"
    / "model_53_ppo2_rom_b_final.pt"
)
ORACLE = (
    ROOT / "logs/rsl_rl/ppo2_rom_b_privileged_oracle/2026-07-25_22-25-49"
    / "model_49_ppo2_rom_b_final.pt"
)

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--shape", choices=("line", "arc", "sine", "circle", "square"), default="circle")
p.add_argument("--target_mode", choices=("health", "independent"), default="health")
p.add_argument("--target_h", type=float, default=0.4)
p.add_argument("--seed", type=int, default=17)
p.add_argument("--path_speed_mps", type=float, default=0.18)
p.add_argument(
    "--post_switch_duration_s",
    type=float,
    default=0.0,
    help="0 scores the full return leg; positive values are for short smoke checks.",
)
p.add_argument("--baseline_checkpoint", default=str(BASELINE))
p.add_argument("--oracle_checkpoint", default=str(ORACLE))
p.add_argument("--output", default="")
p.add_argument("--worker", choices=("romb", "oracle"), default="")
p.add_argument("--worker_output", default="")
AppLauncher.add_app_launcher_args(p)
a, unknown = p.parse_known_args()
if a.target_mode == "health" and not 0.4 <= float(a.target_h) <= 1.2:
    p.error("--target_h must be in [0.4, 1.2]")
if float(a.post_switch_duration_s) < 0.0:
    p.error("--post_switch_duration_s must be non-negative")


def orchestrate() -> None:
    output = (
        Path(a.output).expanduser().resolve()
        if a.output
        else ROOT / "PPO2_ROM_B_PhysX/reports" / (
            "switching_oracle_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".json"
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    with tempfile.TemporaryDirectory(prefix="rom_b_switch_") as temporary:
        for controller, checkpoint in (
            ("romb", a.baseline_checkpoint),
            ("oracle", a.oracle_checkpoint),
        ):
            worker_output = Path(temporary) / f"{controller}.json"
            command = [
                sys.executable, str(Path(__file__).resolve()),
                "--worker", controller,
                "--worker_output", str(worker_output),
                "--shape", str(a.shape),
                "--target_mode", str(a.target_mode),
                "--target_h", str(a.target_h),
                "--seed", str(a.seed),
                "--path_speed_mps", str(a.path_speed_mps),
                "--post_switch_duration_s", str(a.post_switch_duration_s),
                "--device", str(a.device),
                "--baseline_checkpoint", str(a.baseline_checkpoint),
                "--oracle_checkpoint", str(a.oracle_checkpoint),
                "--headless",
            ]
            print(f"[SWITCH SCREEN] running {controller} in a fresh process", flush=True)
            completed = subprocess.run(
                command, cwd=ROOT, text=True, capture_output=True, check=False
            )
            if completed.returncode != 0:
                tail = "\n".join((completed.stdout + completed.stderr).splitlines()[-80:])
                raise RuntimeError(f"{controller} worker failed:\n{tail}")
            results[controller] = json.loads(worker_output.read_text())

    baseline = float(results["romb"]["post_switch_rmse_m"])
    oracle = float(results["oracle"]["post_switch_rmse_m"])
    if results["romb"]["target_factors"] != results["oracle"]["target_factors"]:
        raise RuntimeError("baseline and oracle did not receive identical target factors")
    improvement = 100.0 * (baseline - oracle) / max(baseline, 1.0e-9)
    payload = {
        "diagnostic_only": True,
        "claim": "switching-plant oracle upper-bound screen; never a ROM-B certification result",
        "protocol": {
            "shape": str(a.shape),
            "initial_h": 1.0,
            "target_mode": str(a.target_mode),
            "target_h": float(a.target_h) if a.target_mode == "health" else None,
            "target_factors": results["romb"]["target_factors"],
            "independent_factor_seed": (
                500000 + int(a.seed) if a.target_mode == "independent" else None
            ),
            "switch": "first endpoint reversal (approximately zero reference speed)",
            "truck": "fixed chassis; arm joints free",
            "process_isolation": True,
            "seed": int(a.seed),
            "post_switch_duration_s": (
                "full return leg"
                if float(a.post_switch_duration_s) == 0.0
                else float(a.post_switch_duration_s)
            ),
        },
        "results": results,
        "oracle_post_switch_rmse_improvement_percent": improvement,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")
    window = (
        "full return leg" if float(a.post_switch_duration_s) == 0.0
        else f"first {float(a.post_switch_duration_s):.2f}s after switch"
    )
    print(f"\nSwitching-plant screen (true RMSE over {window})")
    if a.target_mode == "health":
        print(f"  shape: {a.shape}; h: 1.0 -> {float(a.target_h):.1f}")
    else:
        print(f"  shape: {a.shape}; nominal -> independent seed {int(a.seed)}")
        print(f"  target factors: {results['romb']['target_factors']}")
    print(f"  ROM-B : {baseline * 1000.0:.2f} mm")
    print(f"  oracle: {oracle * 1000.0:.2f} mm")
    print(f"  oracle improvement: {improvement:+.2f}%")
    print(f"  artifact: {output}")


def worker() -> None:
    sys.argv = [sys.argv[0], *unknown]
    launcher = AppLauncher(a)
    simulation_app = launcher.app
    try:
        import gymnasium as gym
        import torch
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab_tasks.utils import parse_env_cfg
        from rsl_rl.runners import OnPolicyRunner

        import PPO2_ROM_B_PhysX.tasks  # noqa: F401
        from PPO2_ROM_B_PhysX.env_cfg import PPO2ROMBPPORunnerCfg
        from PPO2_ROM_B_PhysX.runtime import health_scale_to_factors
        from PPO2_ROM_PhysX.policy.env_cfg import apply_residual_curriculum_stage

        privileged = a.worker == "oracle"
        checkpoint = Path(
            a.oracle_checkpoint if privileged else a.baseline_checkpoint
        ).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")

        task = "RRLAB-Mulag-PPO2-ROM-B-Robust-Policy-v0"
        cfg = parse_env_cfg(task, device=a.device, num_envs=1)
        cfg.seed = int(a.seed)
        apply_residual_curriculum_stage(cfg, 4)
        cfg.action_space = 4
        cfg.observation_space = 47 if privileged else 42
        cfg.rom_b_privileged_factors = privileged
        cfg.rom_b_mode = "health"
        cfg.rom_b_health_scale = 1.0
        cfg.rom_b_randomization_scale = 1.0
        cfg.trajectory_shape = str(a.shape)
        cfg.trajectory_shapes = (str(a.shape),)
        cfg.randomize_trajectory_shape = False
        cfg.randomize_start_phase = False
        cfg.procedural_trajectory_probability = 0.0
        cfg.path_speed_mps = float(a.path_speed_mps)
        cfg.residual_scale_m = 0.002
        cfg.cylinder4_residual_scale_m = 0.00025
        cfg.command_rom_direction_guard = True
        cfg.clearance_frame_consistent = True
        cfg.defer_safety_terminations = True
        cfg.episode_length_s = 200.0
        cfg.truck_full_pose_fixture = True

        gym_env = gym.make(task, cfg=cfg)
        raw = gym_env.unwrapped
        agent_cfg = PPO2ROMBPPORunnerCfg()
        agent_cfg.device = a.device
        env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=a.device)
        runner.load(
            str(checkpoint),
            load_cfg={
                "actor": True, "critic": True, "optimizer": False,
                "iteration": False, "rnd": False,
            },
        )
        policy = runner.get_inference_policy(device=raw.device)
        obs, _ = env.reset()

        if a.target_mode == "health":
            target = health_scale_to_factors(float(a.target_h), device=raw.device)
        else:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(500000 + int(a.seed))
            unit = torch.rand((1, 5), generator=generator, dtype=torch.float32)
            target = torch.empty_like(unit)
            target[:, 0] = -0.125 + unit[:, 0] * (0.020 - (-0.125))
            target[:, 1:5] = unit[:, 1:5]
            target = target.to(raw.device)
        switched = False
        pre_squared = []
        post_squared = []
        post_errors = []
        post_accelerations = []
        switch_time = None
        while simulation_app.is_running():
            elapsed = float(raw._scored_elapsed[0].item())
            if not switched and elapsed >= float(raw._one_way_s):
                raw._command_cylinder_rom.true_factors.copy_(target)
                obs = env.get_observations()
                switched = True
                switch_time = elapsed
            with torch.no_grad():
                obs, _, dones, _ = env.step(policy(obs))
            error = float(torch.linalg.norm(
                raw._original_reference_tool[0] - raw._compute_ee_pos_local()[0]
            ).item())
            if switched:
                post_squared.append(error * error)
                post_errors.append(error)
                post_accelerations.append(float(torch.linalg.norm(
                    raw._measured_tool_acceleration_truck[0]
                ).item()))
            else:
                pre_squared.append(error * error)
            if bool(dones.any()):
                raise RuntimeError("unexpected reset during deferred-safety switch screen")
            stop_time = float(raw._complete_cycle_time)
            if float(a.post_switch_duration_s) > 0.0:
                stop_time = float(raw._one_way_s) + float(a.post_switch_duration_s)
            if float(raw._scored_elapsed[0].item()) >= stop_time:
                break

        if not switched or not post_squared:
            raise RuntimeError("plant switch was not reached")
        result = {
            "controller": str(a.worker),
            "checkpoint": str(checkpoint),
            "privileged_factors": privileged,
            "target_mode": str(a.target_mode),
            "target_factors": [float(value) for value in target[0].cpu().tolist()],
            "switch_time_s": switch_time,
            "samples_before": len(pre_squared),
            "samples_after": len(post_squared),
            "pre_switch_rmse_m": (sum(pre_squared) / len(pre_squared)) ** 0.5,
            "post_switch_rmse_m": (sum(post_squared) / len(post_squared)) ** 0.5,
            "post_switch_mean_error_m": sum(post_errors) / len(post_errors),
            "post_switch_max_error_m": max(post_errors),
            "post_switch_p95_acceleration_mps2": float(torch.quantile(
                torch.tensor(post_accelerations), 0.95
            ).item()),
        }
        worker_output = Path(a.worker_output).expanduser().resolve()
        worker_output.write_text(json.dumps(result, indent=2) + "\n")
        env.close()
    finally:
        simulation_app.close()


if a.worker:
    worker()
else:
    orchestrate()
