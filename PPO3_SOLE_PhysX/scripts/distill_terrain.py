#!/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
"""Distil the validated adaptive terrain controller into the PPO3 actor.

The privileged controller is used only to label/partly execute DAgger rollouts.
Its desired cylinder target is converted to the same four normalized commands,
EMA, slew limits, safety shields and ROM-A plant used by the student.  The
checkpoint produced here is a terrain-policy initializer, not a deployed
teacher: normal PPO and evaluation force ``sole_teacher_mode='off'``.
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
p.add_argument("--num_envs", type=int, default=128)
p.add_argument("--seed", type=int, default=17)
p.add_argument("--stage", type=int, default=4, choices=(1, 2, 3, 4))
p.add_argument("--iterations", type=int, default=120)
p.add_argument("--steps_per_iteration", type=int, default=32)
p.add_argument("--epochs_per_iteration", type=int, default=3)
p.add_argument("--minibatch_size", type=int, default=8192)
p.add_argument("--replay_capacity", type=int, default=262144)
p.add_argument("--learning_rate", type=float, default=3.0e-4)
p.add_argument("--anneal_fraction", type=float, default=0.70)
p.add_argument("--probe_steps", type=int, default=240)
p.add_argument("--probe_only", action="store_true")
p.add_argument(
    "--out",
    default="logs/rsl_rl/ppo3_sole_terrain/terrain_teacher_student_v1.pt",
)
p.add_argument(
    "--report",
    default="PPO3_SOLE_PhysX/reports/terrain_teacher_distillation_v1.json",
)
AppLauncher.add_app_launcher_args(p)
a, unknown = p.parse_known_args()
sys.argv = [sys.argv[0], *unknown]

if min(a.num_envs, a.iterations, a.steps_per_iteration, a.minibatch_size) <= 0:
    p.error("environment, iteration, step and minibatch counts must be positive")
if a.replay_capacity < a.num_envs * a.steps_per_iteration:
    p.error("--replay_capacity must hold at least one rollout iteration")
if not 0.0 < a.anneal_fraction <= 1.0:
    p.error("--anneal_fraction must be in (0, 1]")
if a.learning_rate <= 0.0 or a.probe_steps <= 1:
    p.error("learning rate must be positive and probe steps must exceed one")

launcher = AppLauncher(a)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import PPO3_SOLE_PhysX  # noqa: F401,E402
from PPO3_SOLE_PhysX.distill import behaviour_cloning_loss  # noqa: E402
from PPO3_SOLE_PhysX.terrain_env_cfg import (  # noqa: E402
    TerrainSolePolicyPPORunnerCfg,
    apply_terrain_curriculum_stage,
    configure_terrain_gpu_physx_memory,
)


class ReplayBuffer:
    """Bounded on-device ring buffer for aggregated DAgger states."""

    def __init__(self, capacity: int, obs_width: int, action_width: int, device):
        self.capacity = int(capacity)
        self.obs = torch.empty((capacity, obs_width), device=device)
        self.labels = torch.empty((capacity, action_width), device=device)
        self.size = 0
        self.cursor = 0

    def add(self, obs: torch.Tensor, labels: torch.Tensor) -> None:
        if obs.shape[0] >= self.capacity:
            obs, labels = obs[-self.capacity :], labels[-self.capacity :]
        count = int(obs.shape[0])
        first = min(count, self.capacity - self.cursor)
        self.obs[self.cursor : self.cursor + first].copy_(obs[:first])
        self.labels[self.cursor : self.cursor + first].copy_(labels[:first])
        remaining = count - first
        if remaining:
            self.obs[:remaining].copy_(obs[first:])
            self.labels[:remaining].copy_(labels[first:])
        self.cursor = (self.cursor + count) % self.capacity
        self.size = min(self.size + count, self.capacity)

    def minibatches(self, batch_size: int):
        order = torch.randperm(self.size, device=self.obs.device)
        for start in range(0, self.size, batch_size):
            idx = order[start : start + batch_size]
            yield self.obs[idx], self.labels[idx]


def build_env():
    cfg = parse_env_cfg(
        PPO3_SOLE_PhysX.TERRAIN_TASK_ID, device=a.device, num_envs=a.num_envs
    )
    apply_terrain_curriculum_stage(cfg, int(a.stage))
    cfg.sole_teacher_mode = "execute"
    cfg.enable_contact_tracking_demo = False
    cfg.trajectory_shape = str(cfg.terrain_deployment_trajectory_shape)
    cfg.randomize_trajectory_shape = False
    cfg.randomize_start_phase = False
    configure_terrain_gpu_physx_memory(cfg, int(a.num_envs))
    env = gym.make(PPO3_SOLE_PhysX.TERRAIN_TASK_ID, cfg=cfg, render_mode=None)
    return env, env.unwrapped


@torch.no_grad()
def teacher_preflight(raw, steps: int) -> dict:
    """Exercise only the ROM-compatible teacher and report hard invariants."""
    obs, _ = raw.reset(seed=int(a.seed))
    raw.set_teacher_execute_mask(
        torch.ones(raw.num_envs, dtype=torch.bool, device=raw.device)
    )
    zero = torch.zeros((raw.num_envs, 4), device=raw.device)
    raw_actions, applied, errors, laterals, contacts, scans = [], [], [], [], [], []
    alignments, accelerations, wheel_support, rewards = [], [], [], []
    nonfinite = 0
    for _ in range(int(steps)):
        obs, reward, terminated, truncated, _ = raw.step(zero)
        label = raw.terrain_teacher_raw_action
        raw_actions.append(label.detach().clone())
        applied.append(raw.teacher_action.detach().clone())
        errors.append(raw.cut_height_error().abs())
        laterals.append(raw.verge_lateral_error().abs())
        contacts.append(raw.cutter_ground_contact().float())
        scans.append(raw.scan_valid_fraction())
        alignments.append(raw._measure_head_down_alignment())
        accelerations.append(
            torch.linalg.norm(raw._measured_tool_acceleration_truck, dim=-1)
        )
        wheel_support.append(raw.wheel_road_contact_fraction())
        rewards.append(reward)
        nonfinite += int((~torch.isfinite(obs["policy"]).all(dim=-1)).sum())
        nonfinite += int((~torch.isfinite(reward)).sum())
        if bool((terminated & ~truncated).any()):
            nonfinite += int(terminated.sum())

    raw_stream = torch.stack(raw_actions)
    applied_stream = torch.stack(applied)
    deltas = (applied_stream[1:] - applied_stream[:-1]).abs()
    alignment_values = torch.cat(alignments)
    family_counts = {
        family: raw._tracking_path_family_names.count(family)
        for family in sorted(set(raw._tracking_path_family_names))
    }
    report = {
        "steps": int(steps),
        "environments": int(raw.num_envs),
        "observation_width": int(obs["policy"].shape[-1]),
        "path_families": sorted(family_counts),
        "path_family_counts": family_counts,
        "all_finite": nonfinite == 0,
        "nonfinite_or_unexpected_terminal_count": nonfinite,
        "max_abs_teacher_raw_action": float(raw_stream.abs().max()),
        "max_abs_teacher_applied_action": float(applied_stream.abs().max()),
        "max_teacher_applied_step_delta": float(deltas.max()),
        "configured_slew_limit": float(raw.cfg.sole_max_action_rate),
        "slew_limit_respected": bool(
            (deltas <= float(raw.cfg.sole_max_action_rate) + 1.0e-3).all()
        ),
        "mean_abs_cut_height_error_m": float(torch.cat(errors).mean()),
        "p95_abs_cut_height_error_m": float(torch.quantile(torch.cat(errors), 0.95)),
        "mean_abs_lateral_error_m": float(torch.cat(laterals).mean()),
        "ground_contact_fraction": float(torch.cat(contacts).mean()),
        "scan_valid_fraction": float(torch.cat(scans).mean()),
        "min_head_alignment": float(alignment_values.min()),
        "p01_head_alignment": float(torch.quantile(alignment_values, 0.01)),
        "fraction_head_alignment_below_0_975": float(
            (alignment_values < 0.975).float().mean()
        ),
        "p95_tool_acceleration_mps2": float(
            torch.quantile(torch.cat(accelerations), 0.95)
        ),
        "mean_wheel_road_contact_fraction": float(torch.cat(wheel_support).mean()),
        "mean_reward": float(torch.cat(rewards).mean()),
        "direct_target_bypass_enabled": bool(raw.cfg.enable_contact_tracking_demo),
    }
    report["passed"] = bool(
        report["all_finite"]
        and report["slew_limit_respected"]
        and not report["direct_target_bypass_enabled"]
        and report["scan_valid_fraction"] >= 0.80
        and report["mean_wheel_road_contact_fraction"] >= 0.20
        and report["ground_contact_fraction"] >= 0.90
        and report["p95_abs_cut_height_error_m"] <= 0.08
        and report["mean_abs_lateral_error_m"] <= 0.30
        and report["p01_head_alignment"] >= 0.975
        and report["min_head_alignment"] >= 0.950
        and report["p95_tool_acceleration_mps2"] <= 80.0
    )
    return report


def main() -> None:
    env, raw = build_env()
    report = {
        "schema": "terrain_teacher_student_v1",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "stage": int(a.stage),
        "seed": int(a.seed),
    }
    preflight = teacher_preflight(raw, int(a.probe_steps))
    report["preflight"] = preflight
    print("[PPO3-TEACHER-PREFLIGHT] " + json.dumps(preflight, sort_keys=True), flush=True)
    if not preflight["passed"]:
        raise RuntimeError(
            "terrain teacher preflight failed; refusing to create a PPO initializer"
        )
    if a.probe_only:
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(json.dumps(report, indent=2))
        env.close()
        simulation_app.close()
        return

    runner_cfg = TerrainSolePolicyPPORunnerCfg()
    wrapped = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)
    log_dir = ROOT / "logs/rsl_rl/ppo3_sole_terrain_teacher"
    runner = OnPolicyRunner(
        wrapped, runner_cfg.to_dict(), log_dir=str(log_dir), device=a.device
    )
    actor = runner.alg.get_policy()
    optimizer = torch.optim.Adam(actor.parameters(), lr=float(a.learning_rate))
    obs, _ = raw.reset(seed=int(a.seed) + 1)
    replay = ReplayBuffer(
        int(a.replay_capacity), int(obs["policy"].shape[-1]), 4, raw.device
    )
    history = []

    for iteration in range(int(a.iterations)):
        anneal_steps = max(int(a.iterations * a.anneal_fraction), 1)
        beta = max(1.0 - iteration / anneal_steps, 0.0)
        collected_obs, collected_labels = [], []
        teacher_mask_total = 0
        with torch.no_grad():
            for _ in range(int(a.steps_per_iteration)):
                mask = torch.rand(raw.num_envs, device=raw.device) < beta
                raw.set_teacher_execute_mask(mask)
                actor.update_normalization(obs)
                collected_obs.append(obs["policy"].detach().clone())
                student_action = actor(obs)
                obs, _, _, _, _ = raw.step(student_action)
                collected_labels.append(
                    raw.terrain_teacher_raw_action.detach().clone()
                )
                teacher_mask_total += int(mask.sum())
        replay.add(torch.cat(collected_obs), torch.cat(collected_labels))

        losses = []
        for _ in range(int(a.epochs_per_iteration)):
            for batch_obs, batch_label in replay.minibatches(int(a.minibatch_size)):
                loss = behaviour_cloning_loss(
                    actor({"policy": batch_obs}), batch_label
                )
                optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                optimizer.step()
                losses.append((float(loss), float(grad_norm)))
        row = {
            "iteration": iteration,
            "teacher_execute_probability": beta,
            "teacher_executed_fraction": teacher_mask_total
            / max(a.steps_per_iteration * raw.num_envs, 1),
            "replay_size": replay.size,
            "bc_loss": sum(v[0] for v in losses) / max(len(losses), 1),
            "mean_grad_norm": sum(v[1] for v in losses) / max(len(losses), 1),
            "student_mean_abs_action": float(student_action.abs().mean()),
            "teacher_mean_abs_raw_action": float(
                raw.terrain_teacher_raw_action.abs().mean()
            ),
            "mean_abs_cut_height_error_m": float(raw.cut_height_error().abs().mean()),
            "mean_abs_lateral_error_m": float(raw.verge_lateral_error().abs().mean()),
            "ground_contact_fraction": float(raw.cutter_ground_contact().float().mean()),
        }
        history.append(row)
        print("[PPO3-TERRAIN-DAGGER] " + json.dumps(row, sort_keys=True), flush=True)

    out = (ROOT / a.out).resolve() if not Path(a.out).is_absolute() else Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = runner.alg.save()
    payload["iter"] = 0
    payload["infos"] = {
        "source": "PPO3 terrain teacher distillation v1",
        "schema": "terrain_teacher_student_v1",
        "observation_width": int(raw.cfg.observation_space),
        "teacher_preflight": preflight,
        "final_bc_loss": history[-1]["bc_loss"],
    }
    torch.save(payload, out)
    report.update({"checkpoint": str(out), "history": history})
    report_path = (ROOT / a.report).resolve() if not Path(a.report).is_absolute() else Path(a.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(f"[PPO3-TERRAIN-DAGGER] wrote {out}", flush=True)
    print("[PPO3-TERRAIN-DAGGER] teacher deleted at hand-off; PPO/evaluation run teacher_mode=off", flush=True)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
