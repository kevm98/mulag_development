#!/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
"""Fine-tune locked ROM-B input adapters with validated online AAM estimates."""

from __future__ import annotations

import argparse
import hashlib
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
VALIDATED_MAX_NUM_ENVS = 128
p.add_argument("--num_envs", type=int, default=VALIDATED_MAX_NUM_ENVS)
p.add_argument("--seed", type=int, default=17)
p.add_argument("--balanced_shape_rounds", type=int, default=50)
p.add_argument("--iterations_per_shape", type=int, default=2)
p.add_argument("--trajectory_shapes", default="line,arc,sine,circle,square")
p.add_argument("--path_speed_mps", type=float, default=0.18)
p.add_argument("--save_interval", type=int, default=10)
p.add_argument("--learning_rate", type=float, default=5.0e-6)
p.add_argument(
    "--rom_b_checkpoint",
    default=str(
        ROOT / "logs/rsl_rl/ppo2_rom_b_robust/2026-07-24_12-48-14"
        / "model_53_ppo2_rom_b_final.pt"
    ),
)
p.add_argument("--ppo2_c_checkpoint", default="")
p.add_argument(
    "--aam_checkpoint",
    default=str(ROOT / "AAM_SystemFactor/checkpoints/aam_system_factor_runtime.pt"),
)
p.add_argument(
    "--aam_compatibility_report",
    default=str(ROOT / "PPO2_ROM_C_PhysX/reports/aam_rom_b_compatibility.json"),
    help="Passing five-factor ROM-B compatibility audit for the AAM checkpoint.",
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
p.add_argument("--transfer_only", action="store_true")
AppLauncher.add_app_launcher_args(p)
a, unknown = p.parse_known_args()
sys.argv = [sys.argv[0], *unknown]

shapes = tuple(part.strip() for part in a.trajectory_shapes.split(",") if part.strip())
if not shapes:
    p.error("at least one trajectory shape is required")
if a.balanced_shape_rounds <= 0 or a.iterations_per_shape <= 0:
    p.error("balanced rounds and iterations per shape must be positive")
if a.num_envs <= 0 or a.path_speed_mps <= 0.0 or a.learning_rate <= 0.0:
    p.error("num_envs, path speed, and learning rate must be positive")
if a.num_envs > VALIDATED_MAX_NUM_ENVS:
    p.error(f"--num_envs exceeds the validated {VALIDATED_MAX_NUM_ENVS}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Reject an invalid adaptive experiment before Isaac Sim is launched.  The
# hashes make the evidence specific to the exact AAM and ROM checkpoints.
compatibility_path = Path(a.aam_compatibility_report).expanduser().resolve()
compatibility = None
if not a.transfer_only:
    if not compatibility_path.is_file():
        p.error(
            "AAM/ROM-B compatibility report is missing; run "
            "scripts/audit_aam_rom_b_compatibility.py first"
        )
    compatibility = json.loads(compatibility_path.read_text())
    if not bool(compatibility.get("passed", False)):
        p.error(
            "AAM does not identify the five ROM-B factors better than the "
            "constant-mean baseline; adaptive PPO training is blocked"
        )
    evidence_files = {
        "aam": Path(a.aam_checkpoint).expanduser().resolve(),
        "command_rom": Path(a.command_rom_checkpoint).expanduser().resolve(),
    }
    for label, path in evidence_files.items():
        if not path.is_file():
            p.error(f"{label} checkpoint not found: {path}")
        if compatibility.get(f"{label}_sha256") != _sha256(path):
            p.error(
                f"compatibility report does not cover the selected {label} "
                "checkpoint; rerun the audit"
            )

launcher = AppLauncher(a)
simulation_app = launcher.app

import gymnasium as gym
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg
from rsl_rl.runners import OnPolicyRunner

import PPO2_ROM_C_PhysX.tasks  # noqa: F401,E402
from PPO2_ROM_C_PhysX.checkpoint_transfer import (
    configure_aam_adapter_finetune,
    transfer_rom_b_checkpoint_into_runner,
)
from PPO2_ROM_C_PhysX.env_cfg import PPO2CAAMPPORunnerCfg
from PPO2_ROM_PhysX.policy.env_cfg import apply_residual_curriculum_stage


def main() -> None:
    task = "RRLAB-Mulag-PPO2-ROM-C-AAM-Policy-v0"
    cfg = parse_env_cfg(task, device=a.device, num_envs=int(a.num_envs))
    cfg.seed = int(a.seed)
    apply_residual_curriculum_stage(cfg, 4)
    cfg.action_space = 4
    cfg.observation_space = 47
    cfg.command_rom_checkpoint = str(Path(a.command_rom_checkpoint).resolve())
    cfg.aam_checkpoint = str(Path(a.aam_checkpoint).resolve())
    cfg.trajectory_library_path = str(Path(a.trajectory_library_path).resolve())
    cfg.trajectory_shapes = shapes
    cfg.trajectory_shape = shapes[0]
    cfg.randomize_trajectory_shape = False
    cfg.procedural_trajectory_probability = 0.0
    cfg.path_speed_mps = float(a.path_speed_mps)
    cfg.cylinder4_residual_scale_m = 0.00025
    cfg.command_rom_direction_guard = True
    cfg.defer_safety_terminations = True
    cfg.rom_b_mode = "random"
    cfg.rom_b_randomization_scale = 1.0
    cfg.clearance_frame_consistent = True

    total_iterations = (
        int(a.balanced_shape_rounds) * int(a.iterations_per_shape) * len(shapes)
    )
    agent_cfg = PPO2CAAMPPORunnerCfg()
    agent_cfg.seed = int(a.seed)
    agent_cfg.device = a.device
    agent_cfg.max_iterations = total_iterations
    agent_cfg.save_interval = int(a.save_interval)
    agent_cfg.algorithm.learning_rate = float(a.learning_rate)
    log_dir = (
        ROOT / "logs/rsl_rl" / agent_cfg.experiment_name
        / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    cfg.log_dir = str(log_dir)
    run_contract = {
        "variant": "PPO2-ROM_C",
        "base_policy": str(Path(a.rom_b_checkpoint).resolve()),
        "base_policy_variant": "PPO2-ROM_B factor-blind robust policy",
        "resume_policy": str(Path(a.ppo2_c_checkpoint).resolve()) if a.ppo2_c_checkpoint else None,
        "command_rom": cfg.command_rom_checkpoint,
        "aam": cfg.aam_checkpoint,
        "observations": 47,
        "actions": 4,
        "hidden_factor_randomization": "ROM-B five-factor independent uniform",
        "supported_factors": [
            "e_comp", "d_seal_1", "d_seal_2", "d_seal_3", "d_seal_4"
        ],
        "aam_compatibility_report": str(compatibility_path),
        "aam_compatibility": compatibility,
        "privileged_factors_in_policy": False,
        "shapes": list(shapes),
        "iterations": total_iterations,
        "learning_rate": float(a.learning_rate),
        "seed": int(a.seed),
    }
    (log_dir / "ppo2_c_run_contract.json").write_text(
        json.dumps(run_contract, indent=2) + "\n"
    )
    print(
        f"[PPO2-ROM-C] iterations={total_iterations} envs={a.num_envs} "
        f"obs=47 actions=4 shapes={','.join(shapes)} log={log_dir}",
        flush=True,
    )

    gym_env = gym.make(task, cfg=cfg)
    try:
        env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(
            env, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device
        )
        if a.ppo2_c_checkpoint:
            runner.load(
                str(Path(a.ppo2_c_checkpoint).resolve()),
                load_cfg={
                    "actor": True, "critic": True, "optimizer": False,
                    "iteration": False, "rnd": False,
                },
            )
            transfer = {"resume": str(Path(a.ppo2_c_checkpoint).resolve())}
        else:
            transfer = transfer_rom_b_checkpoint_into_runner(
                runner, a.rom_b_checkpoint
            )
            if transfer["new_actor_columns_max_abs"] != 0.0:
                raise RuntimeError("ROM-B transfer changed the neutral policy")
        adapter = configure_aam_adapter_finetune(runner)
        print(f"[PPO2-ROM-C] transfer={transfer}", flush=True)
        print(f"[PPO2-ROM-C] adapter={adapter}", flush=True)
        if a.transfer_only:
            print("[PPO2-ROM-C] transfer-only complete; PPO was not started", flush=True)
            return

        completed = 0
        for round_index in range(int(a.balanced_shape_rounds)):
            ordered = shapes if round_index % 2 == 0 else tuple(reversed(shapes))
            for shape in ordered:
                gym_env.unwrapped.set_trajectory_shape(shape)
                with torch.inference_mode():
                    gym_env.unwrapped.reset(
                        seed=int(a.seed) + round_index * len(shapes) + completed
                    )
                if completed:
                    # This RSL-RL version records the last completed index,
                    # rather than the next one, after each learn() call.
                    runner.current_learning_iteration += 1
                print(
                    f"[PPO2-ROM-C] round={round_index + 1}/"
                    f"{a.balanced_shape_rounds} shape={shape} "
                    f"iterations={a.iterations_per_shape}",
                    flush=True,
                )
                runner.learn(
                    num_learning_iterations=int(a.iterations_per_shape),
                    init_at_random_ep_len=False,
                )
                completed += 1
        final_path = log_dir / (
            f"model_{runner.current_learning_iteration}_ppo2_rom_c_final.pt"
        )
        runner.save(str(final_path), infos={"run_contract": run_contract})
        print(f"[PPO2-ROM-C] complete checkpoint={final_path}", flush=True)
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
