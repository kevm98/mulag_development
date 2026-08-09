#!/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
"""Check the PPO4 checkpoint, ID/FD runtime, viewer path, and cutter motion."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isaaclab.app import AppLauncher  # noqa: E402

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("checkpoint", nargs="?", default="")
p.add_argument("--num_envs", type=int, default=2)
p.add_argument("--steps", type=int, default=120)
p.add_argument("--stage", type=int, default=5)
p.add_argument("--seed", type=int, default=17)
p.add_argument("--path_family", default="s_curve")
p.add_argument("--report", default="")
AppLauncher.add_app_launcher_args(p)
a, unknown = p.parse_known_args()
sys.argv = [sys.argv[0], *unknown]

DEFAULT_CKPT = (
    ROOT
    / "experiments/preflight_four_action_idfd/model_direct_q5_106x5.pt"
)
CKPT = Path(a.checkpoint).resolve() if a.checkpoint else DEFAULT_CKPT

launcher = AppLauncher(a)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import PPO4_ID_FD_PhysX as PKG  # noqa: E402,F401
from PPO3_SOLE_PhysX.actuators import assert_checkpoint_actuator  # noqa: E402
from PPO3_SOLE_PhysX.terrain_env_cfg import (  # noqa: E402
    apply_terrain_curriculum_stage,
    configure_terrain_gpu_physx_memory,
)
from PPO3_SOLE_PhysX.terrain_observations import (  # noqa: E402
    observation_space_for_cfg,
)
from PPO4_ID_FD_PhysX.terrain_env_cfg import TerrainIDFDPPORunnerCfg  # noqa: E402

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    CHECKS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""), flush=True)
    return bool(ok)


def main() -> int:
    print(f"[PPO4-READY] checkpoint: {CKPT}", flush=True)
    if not check("checkpoint exists", CKPT.is_file(), str(CKPT)):
        finish()
        return 1

    payload = torch.load(CKPT, map_location="cpu", weights_only=False)
    infos = payload.get("infos") or {}
    tag = str(infos.get("actuator", "") or "rom")
    check("checkpoint tagged actuator=idfd", tag == "idfd", f"tag={tag!r}")
    check(
        "checkpoint schema is terrain_shared_v2",
        str(infos.get("schema", "")) == "terrain_shared_v2",
        str(infos.get("schema", "")),
    )
    # 54 without the actuator-model-error channels, 66 with them. The
    # checkpoint must simply agree with the environment it will be run in.
    from PPO4_ID_FD_PhysX.terrain_env_cfg import TerrainIDFDEnvCfg

    contract_cfg = TerrainIDFDEnvCfg()
    contract_cfg.sole_head_joint_action = bool(infos.get("head_joint_action", False))
    contract_cfg.action_space = 5 if contract_cfg.sole_head_joint_action else 4
    expected_width = observation_space_for_cfg(contract_cfg)
    check(
        f"observation width {expected_width}",
        int(infos.get("observation_width", -1)) == expected_width,
        str(infos.get("observation_width")),
    )

    # PLAY config: exactly what the viewer builds (1 env, visual scene on).
    env_cfg = parse_env_cfg(PKG.TERRAIN_PLAY_TASK_ID, device=a.device, num_envs=a.num_envs)
    apply_terrain_curriculum_stage(env_cfg, int(a.stage))
    env_cfg.sole_head_joint_action = bool(infos.get("head_joint_action", False))
    env_cfg.action_space = 5 if env_cfg.sole_head_joint_action else 4
    env_cfg.sole_head_action_to_c4_gain = 0.0
    env_cfg.terrain_kinematic_drive_substep_interpolation = True
    env_cfg.sole_ik_hint = False
    env_cfg.observation_space = observation_space_for_cfg(env_cfg)
    env_cfg.use_eval_terrain = True
    env_cfg.terrain_base_seed = int(a.seed)
    env_cfg.tracking_path_family = str(a.path_family)
    env_cfg.defer_safety_terminations = True
    env_cfg.randomize_start_phase = False
    env_cfg.randomize_trajectory_shape = False
    env_cfg.trajectory_shape = str(env_cfg.terrain_deployment_trajectory_shape)
    # Visual scene ON: this is the path the viewer takes and training never does.
    env_cfg.enable_visual_grid_floor = True
    env_cfg.enable_road_scenery = True
    env_cfg.dark_visual_background = False
    configure_terrain_gpu_physx_memory(env_cfg, int(a.num_envs))

    env = gym.make(PKG.TERRAIN_PLAY_TASK_ID, cfg=env_cfg, render_mode=None)
    raw = env.unwrapped
    check("PLAY scene builds with visuals", True, PKG.TERRAIN_PLAY_TASK_ID)
    check(
        "ID/FD runtime is in the loop",
        getattr(raw, "_cylinder_id", None) is not None
        and str(raw.cfg.cylinder_dynamics_model).lower() == "fd",
        str(raw.cfg.cylinder_dynamics_model),
    )
    check(
        "no ROM runtime constructed",
        all(
            getattr(raw, n, None) is None
            for n in ("_command_cylinder_rom", "_base_cylinder_rom", "_cylinder_rom")
        ),
    )
    check(
        "compliant plant active",
        abs(float(raw.cfg.terrain_arm_stiffness_scale) - 0.35) < 1e-9,
        f"stiffness_scale={float(raw.cfg.terrain_arm_stiffness_scale)}",
    )

    runner_cfg = TerrainIDFDPPORunnerCfg()
    runner_cfg.seed = int(a.seed)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)
    runner = OnPolicyRunner(wrapped, runner_cfg.to_dict(), log_dir=None, device=a.device)
    assert_checkpoint_actuator(infos, "idfd", context="PPO4 readiness check")
    runner.load(
        str(CKPT),
        load_cfg={"actor": True, "critic": True, "optimizer": False, "iteration": True, "rnd": False},
    )
    policy = runner.get_inference_policy(device=a.device)
    check("policy loads and builds an inference graph", True)

    obs, _ = raw.reset(seed=int(a.seed))
    cutter0 = raw._compute_cutter_pos_w()[:, :2].clone()
    finite = True
    contacts, laterals, cuts = [], [], []
    for _ in range(int(a.steps)):
        with torch.inference_mode():
            action = policy(obs)
        finite = finite and bool(torch.isfinite(action).all())
        obs, reward, _, _, _ = raw.step(action)
        finite = finite and bool(torch.isfinite(reward).all())
        finite = finite and bool(torch.isfinite(obs["policy"]).all())
        contacts.append(raw.cutter_ground_contact().float().mean())
        laterals.append(raw.verge_lateral_error().abs().mean())
        cuts.append(raw.cut_height_error().abs().mean())
    travel = float(
        torch.linalg.norm(raw._compute_cutter_pos_w()[:, :2] - cutter0, dim=-1).mean()
    )
    contact = float(torch.stack(contacts).mean())
    lateral = float(torch.stack(laterals).mean())
    cut = float(torch.stack(cuts).mean())

    check("rollout stays finite", finite)
    check("cutter is moving (mowing, not frozen)", travel > 0.5, f"{travel:.3f} m travelled")
    check("cutter reaches the ground", contact > 0.5, f"contact fraction {contact:.4f}")
    check("cutter follows the path", lateral < 0.30, f"mean lateral {lateral:.4f} m")
    check("cut height is regulated", cut < 0.10, f"mean cut error {cut:.4f} m")

    env.close()
    return finish()


def finish() -> int:
    ready = all(ok for _, ok, _ in CHECKS)
    print("\n[PPO4-READY] " + ("READY FOR VISUALIZATION" if ready else "NOT READY"), flush=True)
    if ready:
        print(
            "\nGUI command:\n"
            "  /home/rrlab/miniconda3/envs/env_isaaclab/bin/python "
            "PPO3_SOLE_PhysX/scripts/evaluate.py \\\n"
            f"    {CKPT} --actuator idfd \\\n"
            "    --device cuda:0 --num_envs 1 --steps 960 --stage 5 --eval_terrain \\\n"
            "    --head_joint_action \\\n"
            "    --drive_substep_interpolation true --path_family s_curve --plot_tracking",
            flush=True,
        )
    if a.report:
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(
            json.dumps(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "checkpoint": str(CKPT),
                    "ready": ready,
                    "checks": [
                        {"name": n, "passed": ok, "detail": d} for n, ok, d in CHECKS
                    ],
                },
                indent=2,
            )
        )
    return 0 if ready else 1


if __name__ == "__main__":
    try:
        code = main()
    finally:
        simulation_app.close()
    raise SystemExit(code)
