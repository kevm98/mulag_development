#!/usr/bin/env python3
"""Integration smoke test for the contact-safe contract, both actuators.

The fast pytest suite runs without Isaac Sim, so it can only check the state
machine against a stub and the reward against tensors. This runs the real
thing -- a short rollout on the real plant, once per actuator -- and asserts
the properties that only the assembled system can violate:

* the observation is exactly the width the config derives, and finite
* the contact state machine advances exactly once per control step
* rewards are finite and the contact terms are present in the log
* PPO is the sole controller: teacher off, contact demo off
* PPO3 constructs ROM-A and PPO4 constructs ID -> FD with no ROM runtime

Usage:

    python verge_task/scripts/smoke_contact_safe.py --actuator rom
    python verge_task/scripts/smoke_contact_safe.py --actuator idfd

Exits non-zero on the first failed assertion, so it is usable as a gate in a
driver script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--actuator", default="rom", choices=("rom", "idfd"))
p.add_argument("--num_envs", type=int, default=4)
p.add_argument("--steps", type=int, default=30)
p.add_argument("--stage", type=int, default=5)
p.add_argument("--seed", type=int, default=17)
p.add_argument("--report", default="")
AppLauncher.add_app_launcher_args(p)
a, unknown = p.parse_known_args()
sys.argv = [sys.argv[0], *unknown]

launcher = AppLauncher(a)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import verge_task  # noqa: F401,E402
from verge_task.actuators import resolve_actuator  # noqa: E402
from verge_task.distill import assert_teacher_disabled  # noqa: E402
from verge_task.terrain_env_cfg import (  # noqa: E402
    apply_terrain_curriculum_stage,
    configure_terrain_gpu_physx_memory,
)
from verge_task.terrain_observations import (  # noqa: E402
    CONTACT_STATE_CHANNELS,
    observation_space_for_cfg,
)

TASK_ID, _, ACTUATOR_LABEL = resolve_actuator(a.actuator)

failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        failures.append(message)
        print(f"[FAIL] {message}", flush=True)
    else:
        print(f"[ok]   {message}", flush=True)


def main() -> int:
    env_cfg = parse_env_cfg(TASK_ID, device=a.device, num_envs=a.num_envs)
    apply_terrain_curriculum_stage(env_cfg, int(a.stage))
    env_cfg.action_space = 5 if env_cfg.sole_head_joint_action else 4
    env_cfg.sole_ik_hint = False
    env_cfg.terrain_contact_observation = True
    env_cfg.terrain_contact_safety_reward = True
    env_cfg.terrain_model_error_history = 3
    # Terminate ON, with a tight grace, so the smoke test exercises the
    # termination path rather than merely constructing it.
    env_cfg.terrain_terminate_on_contact_loss = True
    env_cfg.terrain_terminate_on_hard_impact = True
    env_cfg.terrain_contact_loss_grace_steps = 1
    env_cfg.enable_contact_tracking_demo = False
    env_cfg.terrain_base_seed = int(a.seed)
    expected_obs = observation_space_for_cfg(env_cfg)
    env_cfg.observation_space = expected_obs
    configure_terrain_gpu_physx_memory(env_cfg, int(a.num_envs))
    assert_teacher_disabled(env_cfg, context="contact-safe smoke test")

    env = gym.make(TASK_ID, cfg=env_cfg, render_mode=None)
    raw = env.unwrapped

    check(
        expected_obs == 96,
        f"contact-safe observation width is 96 (got {expected_obs})",
    )

    obs, _ = raw.reset(seed=int(a.seed))
    policy_obs = obs["policy"] if isinstance(obs, dict) else obs
    check(
        tuple(policy_obs.shape) == (raw.num_envs, expected_obs),
        f"reset observation is {(raw.num_envs, expected_obs)} "
        f"(got {tuple(policy_obs.shape)})",
    )
    check(bool(torch.isfinite(policy_obs).all()), "reset observation is finite")

    # Contact state must be clear immediately after reset.
    check(
        not bool(raw._contact_acquired.any()),
        "no environment claims support immediately after reset",
    )
    check(
        int(raw._contact_loss_event_count.sum()) == 0,
        "contact-loss counters are zero after reset",
    )

    # --- the rollout ---
    action = torch.zeros(
        (raw.num_envs, int(raw.cfg.action_space)), device=raw.device
    )
    update_counts = []
    rewards_finite = True
    obs_finite = True
    for step in range(int(a.steps)):
        before = getattr(raw, "_contact_safety_step", None)
        obs, reward, terminated, truncated, extras = raw.step(action)
        after = getattr(raw, "_contact_safety_step", None)
        update_counts.append((before, after))
        policy_obs = obs["policy"] if isinstance(obs, dict) else obs
        obs_finite &= bool(torch.isfinite(policy_obs).all())
        rewards_finite &= bool(torch.isfinite(reward).all())

    check(obs_finite, f"observations finite over {a.steps} steps")
    check(rewards_finite, f"rewards finite over {a.steps} steps")

    # The marker must advance by exactly one per step. If dones, rewards and
    # observations each ran their own update it would advance by more, and if
    # the cache were broken it would not advance at all.
    deltas = {
        (after - before)
        for before, after in update_counts
        if before is not None and after is not None
    }
    check(
        deltas == {1},
        f"contact state advanced exactly once per control step (deltas {sorted(deltas)})",
    )

    log = (raw.extras or {}).get("log", {})
    for key in (
        "contact/acquired_fraction",
        "contact/loss_event_fraction",
        "contact/max_airborne_gap_m",
        "reward/contact_total",
        "reward/contact_loss",
        "reward/contact_airborne_gap",
    ):
        check(key in log, f"log carries {key}")

    # --- sole-controller and actuator contracts ---
    check(
        str(raw.cfg.sole_teacher_mode) == "off",
        "teacher mode is off",
    )
    check(
        not bool(raw.cfg.enable_contact_tracking_demo),
        "direct target-writing contact demo is off",
    )
    rom_live = [
        name
        for name in ("_command_cylinder_rom", "_base_cylinder_rom", "_cylinder_rom")
        if getattr(raw, name, None) is not None
    ]
    if a.actuator == "idfd":
        check(
            getattr(raw, "_cylinder_id", None) is not None,
            "PPO4 constructed the ID runtime",
        )
        check(not rom_live, f"PPO4 constructed no ROM runtime (found {rom_live})")
        check(
            str(raw.cfg.cylinder_dynamics_model).lower() == "fd",
            "PPO4 runs cylinder_dynamics_model='fd'",
        )
    else:
        check(
            str(raw.cfg.cylinder_dynamics_model).lower() == "command_rom",
            "PPO3 runs cylinder_dynamics_model='command_rom'",
        )
        check(
            getattr(raw, "_cylinder_id", None) is None,
            "PPO3 constructed no ID runtime",
        )

    summary = {
        "actuator": str(a.actuator),
        "actuator_label": ACTUATOR_LABEL,
        "observation_width": int(expected_obs),
        "contact_state_channels": int(CONTACT_STATE_CHANNELS),
        "steps": int(a.steps),
        "num_envs": int(raw.num_envs),
        "contact_acquired_fraction": float(raw._contact_acquired.float().mean()),
        "contact_loss_events": int(raw._contact_loss_event_count.sum()),
        "hard_impacts": int(raw._hard_impact_event_count.sum()),
        "visible_lifts": int(raw._visible_lift_event_count.sum()),
        "max_airborne_gap_m": float(raw._max_airborne_gap_m.max()),
        "teacher_mode": str(raw.cfg.sole_teacher_mode),
        "direct_target_contact_demo": bool(raw.cfg.enable_contact_tracking_demo),
        "failures": failures,
        "passed": not failures,
    }
    print(json.dumps(summary, indent=2), flush=True)
    if a.report:
        path = Path(a.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2))
        print(f"[PPO3-SOLE] wrote {path}", flush=True)

    env.close()
    simulation_app.close()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
