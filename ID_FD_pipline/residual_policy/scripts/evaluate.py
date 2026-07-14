"""Evaluate a trained residual policy against a live zero-action baseline."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--checkpoint", required=True)
p.add_argument("--num_seeds", type=int, default=5)
p.add_argument("--seed", type=int, default=17)
p.add_argument("--out", default="outputs/residual_policy/evaluation.json")
AppLauncher.add_app_launcher_args(p)
a, unknown = p.parse_known_args()
sys.argv = [sys.argv[0], *unknown]

launcher = AppLauncher(a)
simulation_app = launcher.app

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import ID_FD_pipline.tasks  # noqa: F401,E402
from ID_FD_pipline.residual_policy.env_cfg import ResidualPolicyPPORunnerCfg
from isaaclab_tasks.utils import parse_env_cfg


def evaluate_run(*, zero_policy: bool, seed: int) -> dict[str, float | int | bool]:
    task = "Kevin-Mulag-IDFD-Residual-PLAY-v0"
    cfg = parse_env_cfg(task, device=a.device, num_envs=1)
    cfg.seed = seed
    cfg.force_zero_policy = zero_policy
    cfg.episode_length_s = 200.0
    gym_env = gym.make(task, cfg=cfg)
    raw = gym_env.unwrapped
    agent_cfg = ResidualPolicyPPORunnerCfg()
    agent_cfg.device = a.device
    env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
    policy = None
    if not zero_policy:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(str(Path(a.checkpoint).expanduser().resolve()))
        policy = runner.get_inference_policy(device=raw.device)

    obs, _ = env.reset()
    goals: list[np.ndarray] = []
    tools: list[np.ndarray] = []
    errors: list[float] = []
    heights: list[float] = []
    heads: list[float] = []
    residuals: list[float] = []
    early_termination = False
    try:
        while float(raw._scored_elapsed[0].item()) < raw._complete_cycle_time:
            with torch.inference_mode():
                action = (
                    torch.zeros((1, 3), device=raw.device)
                    if policy is None
                    else policy(obs)
                )
                obs, _, dones, _ = env.step(action)
            active = raw._preposition_counter[0] <= 0 and raw._settle_counter[0] <= 0
            if active:
                goal = raw._original_reference_tool[0].detach().cpu().numpy().copy()
                tool = raw._compute_ee_pos_local()[0].detach().cpu().numpy().copy()
                goals.append(goal)
                tools.append(tool)
                errors.append(float(np.linalg.norm(goal - tool)))
                heights.append(float(raw._compute_cutter_world_z()[0].item()))
                heads.append(float(raw._measure_head_down_alignment()[0].item()))
                residuals.append(float(raw._residual_filtered[0].norm().item()))
            if dones.any():
                early_termination = True
                break
    finally:
        env.close()

    if len(errors) < 20:
        return {"completed_cycle": False, "steps": len(errors), "mean_tool_error": float("inf")}
    error = np.asarray(errors)
    goal = np.asarray(goals)
    tool = np.asarray(tools)
    warmup = min(120, max(len(error) // 20, 1))
    scored = error[warmup:]
    goal_path = float(np.linalg.norm(np.diff(goal, axis=0), axis=1).sum())
    kernel = np.ones(15, dtype=np.float64) / 15.0
    filtered_tool = np.column_stack(
        [np.convolve(tool[:, axis], kernel, mode="valid") for axis in range(3)]
    )
    filtered_path = float(np.linalg.norm(np.diff(filtered_tool, axis=0), axis=1).sum())
    min_head = float(np.min(heads))
    return {
        "completed_cycle": not early_termination,
        "steps": len(errors),
        "mean_tool_error": float(scored.mean()),
        "p95_tool_error": float(np.percentile(scored, 95)),
        "max_tool_error": float(scored.max()),
        "final_tool_error": float(error[-1]),
        "ground_fraction": float(np.mean(np.abs(heights) <= 0.10)),
        "min_head_alignment": min_head,
        "max_head_tilt_deg": float(np.degrees(np.arccos(np.clip(min_head, -1.0, 1.0)))),
        "max_truck_drift": float(raw._max_truck_drift.max().item()),
        "filtered_path_ratio": filtered_path / max(goal_path, 1.0e-9),
        "max_residual": float(np.max(residuals)),
    }


def _aggregate(runs: list[dict]) -> dict[str, float | bool]:
    keys = (
        "mean_tool_error", "p95_tool_error", "max_tool_error", "final_tool_error",
        "ground_fraction", "min_head_alignment", "max_head_tilt_deg",
        "max_truck_drift", "filtered_path_ratio", "max_residual",
    )
    return {
        "completed_all_cycles": all(bool(r.get("completed_cycle")) for r in runs),
        **{key: float(np.mean([float(r.get(key, np.inf)) for r in runs])) for key in keys},
    }


def main() -> None:
    baseline_runs, policy_runs = [], []
    for seed in range(int(a.seed), int(a.seed) + int(a.num_seeds)):
        baseline = evaluate_run(zero_policy=True, seed=seed)
        policy = evaluate_run(zero_policy=False, seed=seed)
        baseline_runs.append(baseline)
        policy_runs.append(policy)
        print(
            f"[EVAL seed={seed}] baseline={baseline.get('mean_tool_error', np.inf):.5f}m "
            f"policy={policy.get('mean_tool_error', np.inf):.5f}m", flush=True
        )
    baseline = _aggregate(baseline_runs)
    policy = _aggregate(policy_runs)
    passed = bool(
        baseline["completed_all_cycles"]
        and policy["completed_all_cycles"]
        and policy["mean_tool_error"] < baseline["mean_tool_error"]
        and policy["p95_tool_error"] < baseline["p95_tool_error"]
        and policy["max_tool_error"] <= 1.05 * baseline["max_tool_error"]
        and policy["ground_fraction"] >= 0.99
        and policy["min_head_alignment"] >= 0.995
        and policy["max_truck_drift"] < 0.002
        and 0.95 <= policy["filtered_path_ratio"] <= 1.25
    )
    result = {
        "passed": passed,
        "checkpoint": str(Path(a.checkpoint).expanduser().resolve()),
        "baseline": baseline,
        "policy": policy,
        "baseline_runs": baseline_runs,
        "policy_runs": policy_runs,
    }
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"[EVAL] {'PASS' if passed else 'FAIL'} {json.dumps(result)}", flush=True)
    if not passed:
        raise SystemExit(2)


try:
    main()
except BaseException:
    traceback.print_exc()
    raise
finally:
    simulation_app.close()
