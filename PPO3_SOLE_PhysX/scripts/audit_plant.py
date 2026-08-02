#!/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
"""Freeze-and-audit the corrected plant, at RUNTIME, for one actuator.

Config inspection is not enough. Three separate things can make a config value
untrue of the running plant:

* another config path re-asserts the inherited value after ``__post_init__``
  (PPO4's ROM-disabling fields already need re-asserting for exactly this
  reason);
* a curriculum stage rewrites it after the CLI override (measured: stage 5
  rewrites ``action_scale`` and ``sole_max_action_rate``);
* the value is honoured in different UNITS from the one the observation
  reports, so "target lead 0.0196" can sit next to "bound 0.010" without
  either being wrong.

So this reads the CONSTRUCTED runtime, and instruments the anti-windup clamp
where it is actually applied.

The clamp lives in ``mulag_upper_level_env._run_cylinder_id_pipeline`` and
bounds the commanded PhysX target against the cylinder position READ AT THE
TOP OF THAT CALL, in USD units. ``_commanded_target_lead`` -- which is what the
observation and every report show -- measures against the joint position at
OBSERVATION time, in learned units, one full control step later. The plant
moves in between, so the observed lead legitimately exceeds the bound. This
script measures both and reports the clamp's own binding fraction, which is
the only number that answers "does the target lead still saturate".

No plant behaviour is modified: the wrapper calls the original pipeline and
then reads state.
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
p.add_argument("--actuator", default="rom", choices=("rom", "idfd"))
p.add_argument("--checkpoint", default="")
p.add_argument("--num_envs", type=int, default=16)
p.add_argument("--steps", type=int, default=480)
p.add_argument("--seed", type=int, default=17)
p.add_argument("--stage", type=int, default=5)
p.add_argument("--eval_terrain", action="store_true")
p.add_argument("--path_family", default="boom_sweep")
p.add_argument("--terrain_height_scale", type=float, default=1.0)
p.add_argument("--contact_observation", default="true", choices=("true", "false"))
p.add_argument("--model_error_history", type=int, default=3)
p.add_argument("--action_scale", type=float, default=0.0035)
p.add_argument("--cylinder_target_max_lead_m", type=float, default=0.010)
p.add_argument("--cylinder_target_leak", type=float, default=None)
p.add_argument("--arm_stiffness_scale", type=float, default=None)
p.add_argument("--report", default="")
AppLauncher.add_app_launcher_args(p)
a, unknown = p.parse_known_args()
sys.argv = [sys.argv[0], *unknown]

launcher = AppLauncher(a)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import PPO3_SOLE_PhysX  # noqa: F401,E402
from PPO3_SOLE_PhysX.distill import assert_teacher_disabled  # noqa: E402
from PPO3_SOLE_PhysX.actuators import resolve_actuator  # noqa: E402
from PPO3_SOLE_PhysX.terrain_env_cfg import (  # noqa: E402
    apply_terrain_curriculum_stage,
    configure_terrain_gpu_physx_memory,
)
from PPO3_SOLE_PhysX.terrain_observations import (  # noqa: E402
    observation_space_for_cfg,
    override_cfg,
)

TASK_ID, RUNNER_CFG_CLS, ACTUATOR_LABEL = resolve_actuator(a.actuator)

#: Every field this audit freezes. Reported from the CONSTRUCTED cfg.
FROZEN_FIELDS = (
    "cylinder_target_max_lead_m",
    "cylinder_target_leak",
    "cylinder_target_integrates_from_previous_target",
    "action_scale",
    "sole_max_action_rate",
    "sole_action_filter",
    "terrain_arm_stiffness_scale",
    "terrain_head_joint_stiffness_scale",
    "terrain_arm_damping_scale",
    "cylinder_dynamics_model",
    "use_cylinder_id_pipeline",
    "sole_teacher_mode",
    "enable_contact_tracking_demo",
    "terrain_contact_observation",
    "terrain_model_error_history",
    "terrain_contact_safety_reward",
    "terrain_terminate_on_contact_loss",
    "terrain_terminate_on_hard_impact",
    "terrain_contact_loss_grace_steps",
    "terrain_contact_acquire_steps",
    "terrain_max_airborne_gap_m",
    "sole_head_joint_action",
    "terrain_head_nullspace_restore_gain",
    "terrain_messerkopf_level_gain",
)


def build_env():
    env_cfg = parse_env_cfg(TASK_ID, device=a.device, num_envs=a.num_envs)
    apply_terrain_curriculum_stage(env_cfg, int(a.stage))
    env_cfg.action_space = 5 if env_cfg.sole_head_joint_action else 4
    if env_cfg.sole_head_joint_action:
        env_cfg.terrain_messerkopf_level_gain = 0.0
        env_cfg.terrain_messerkopf_level_quadratic_gain = 0.0
    env_cfg.sole_ik_hint = False
    env_cfg.terrain_contact_observation = a.contact_observation == "true"
    override_cfg(env_cfg, "terrain_model_error_history", int(a.model_error_history))
    # Applied AFTER the curriculum, which rewrites both of these per stage.
    override_cfg(env_cfg, "action_scale", float(a.action_scale))
    override_cfg(
        env_cfg, "cylinder_target_max_lead_m", float(a.cylinder_target_max_lead_m)
    )
    if a.cylinder_target_leak is not None:
        override_cfg(env_cfg, "cylinder_target_leak", float(a.cylinder_target_leak))
    if a.arm_stiffness_scale is not None:
        override_cfg(
            env_cfg, "terrain_arm_stiffness_scale", float(a.arm_stiffness_scale)
        )
    env_cfg.observation_space = observation_space_for_cfg(env_cfg)
    env_cfg.use_eval_terrain = bool(a.eval_terrain)
    env_cfg.terrain_base_seed = int(a.seed)
    env_cfg.terrain_height_scale_override = float(a.terrain_height_scale)
    env_cfg.tracking_path_family = str(a.path_family)
    env_cfg.enable_contact_tracking_demo = False
    env_cfg.enable_visual_grid_floor = False
    env_cfg.enable_road_scenery = False
    env_cfg.terrain_terminate_on_contact_loss = False
    env_cfg.terrain_terminate_on_hard_impact = False
    env_cfg.defer_safety_terminations = True
    env_cfg.head_down_termination_alignment = -1.0
    env_cfg.randomize_start_phase = False
    env_cfg.randomize_trajectory_shape = False
    env_cfg.trajectory_shape = str(env_cfg.terrain_deployment_trajectory_shape)
    configure_terrain_gpu_physx_memory(env_cfg, int(a.num_envs))
    assert_teacher_disabled(env_cfg, context="plant audit")
    env = gym.make(TASK_ID, cfg=env_cfg, render_mode=None)
    return env, env.unwrapped


def install_clamp_probe(raw) -> dict:
    """Wrap the cylinder pipeline to measure the anti-windup clamp itself.

    The wrapper calls the original and then reads state. It writes nothing, so
    the plant is unchanged -- verified by the caller comparing a probed rollout
    against an unprobed one is NOT necessary, because no branch here can reach
    a setter.
    """
    original = raw._run_cylinder_id_pipeline
    state = {
        "samples": 0,
        "bound_hits": 0,
        "lead_usd_abs": [],
        "bound_usd": None,
    }

    lead_m = float(getattr(raw.cfg, "cylinder_target_max_lead_m", 0.0))

    def wrapped(*args, **kwargs):
        q_before = raw.robot.data.joint_pos[:, raw._cylinder_joint_ids].detach().clone()
        result = original(*args, **kwargs)
        target = raw._cylinder_pos_target.detach()
        lead_usd = (target - q_before).abs()
        # Scored samples only. Two paths write the target WITHOUT passing
        # through the anti-windup clamp -- the neutral-hold branch at the end
        # of the pipeline, and `_write_direct_learned_targets` during the reset
        # deployment ramp -- so warm-up samples both inflate the binding
        # fraction and produce leads ABOVE the bound. Including them was how
        # the first run of this audit reported a maximum of 0.0190 against a
        # 0.0100 bound, which is impossible for a clamped sample.
        #
        # The neutral-hold branch is the third such path and the one that
        # actually mattered here: it assigns the target AFTER the clamp and
        # does not re-apply it, so a held cylinder whose plant has drifted
        # reports a lead above the bound. That is what produced a 0.0190
        # maximum against a 0.0100 bound in the first run of this audit, and
        # it is a property of the inherited pipeline, not of the fix.
        active = (raw._preposition_counter <= 0) & (raw._settle_counter <= 0)
        held = getattr(raw, "_neutral_hold_mask", None)
        if held is not None:
            state["neutral_hold_steps"] = state.get("neutral_hold_steps", 0) + int(
                (held & active).sum()
            )
            active = active & ~held
        if lead_m > 0.0 and bool(active.any()):
            scored = lead_usd[active]
            bound = raw._cylinder_adapter.simscape_delta_to_isaac_delta(
                torch.full_like(target, lead_m)
            ).abs()[active]
            state["bound_usd"] = bound[0].tolist()
            # "Binding" means the clamp actually moved the target this step:
            # within 1 um of the bound. A strict equality test would miss the
            # float error the clamp itself introduces.
            state["bound_hits"] += int((scored >= bound - 1.0e-6).sum())
            state["samples"] += int(scored.numel())
            state["lead_usd_abs"].append(scored.cpu())
        state["warmup_samples"] = state.get("warmup_samples", 0) + int(
            (~active).sum() * lead_usd.shape[-1]
        )
        return result

    raw._run_cylinder_id_pipeline = wrapped
    return state


def _q(values: torch.Tensor, level: float) -> float:
    return float(torch.quantile(values.flatten().float(), level))


def main() -> int:
    env, raw = build_env()
    clamp = install_clamp_probe(raw)

    policy = None
    checkpoint_info = None
    if a.checkpoint:
        runner_cfg = RUNNER_CFG_CLS()
        runner_cfg.seed = int(a.seed)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)
        runner = OnPolicyRunner(
            wrapped, runner_cfg.to_dict(), log_dir=None, device=a.device
        )
        payload = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
        checkpoint_info = dict(payload.get("infos") or {})
        runner.load(
            a.checkpoint,
            load_cfg={
                "actor": True,
                "critic": True,
                "optimizer": False,
                "iteration": True,
                "rnd": False,
            },
        )
        policy = runner.get_inference_policy(device=a.device)

    obs, _ = raw.reset(seed=int(a.seed))
    observed_lead, forces, gaps = [], [], []
    for _ in range(int(a.steps)):
        if policy is None:
            action = torch.zeros(
                (raw.num_envs, int(raw.cfg.action_space)), device=raw.device
            )
        else:
            action = policy(obs)
        obs, _, _, _, _ = raw.step(action)
        measured, _ = raw._refresh_cylinder_state()
        observed_lead.append(raw._commanded_target_lead(measured).abs().detach().cpu())
        forces.append(raw.cutter_terrain_contact_force_n().detach().cpu())
        gaps.append(raw._airborne_gap_m.detach().cpu())

    observed = torch.stack(observed_lead)
    clamp_lead = (
        torch.cat(clamp["lead_usd_abs"])
        if clamp["lead_usd_abs"]
        else torch.zeros(1)
    )
    force = torch.stack(forces)
    gap = torch.stack(gaps)

    # --- effective plant, read from the constructed runtime ---
    effective = {
        name: (
            float(getattr(raw.cfg, name))
            if isinstance(getattr(raw.cfg, name, None), (int, float))
            and not isinstance(getattr(raw.cfg, name), bool)
            else getattr(raw.cfg, name, None)
        )
        for name in FROZEN_FIELDS
    }

    # Joint gains actually written to PhysX during scored driving, not the
    # scale that was requested.
    held = raw._terrain_held_joint_ids
    stiffness = raw.robot.data.joint_stiffness[:, held][0]
    nominal = raw._terrain_nominal_arm_stiffness[0]
    realized_scale = (stiffness / nominal.clamp(min=1.0e-9)).tolist()

    rom_runtimes = {
        name: getattr(raw, name, None) is not None
        for name in ("_command_cylinder_rom", "_base_cylinder_rom", "_cylinder_rom")
    }

    saturating_fraction = (
        clamp["bound_hits"] / clamp["samples"] if clamp["samples"] else float("nan")
    )

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "actuator": str(a.actuator),
        "actuator_label": ACTUATOR_LABEL,
        "checkpoint": str(a.checkpoint) or None,
        "checkpoint_infos": checkpoint_info,
        "path_family": str(a.path_family),
        "terrain_height_scale": float(a.terrain_height_scale),
        "terrain": "held_out" if a.eval_terrain else "training",
        "steps": int(a.steps),
        "num_envs": int(raw.num_envs),
        "effective_config": effective,
        "realized_arm_stiffness_scale_per_held_joint": realized_scale,
        "actuator_runtime": {
            "cylinder_id_constructed": getattr(raw, "_cylinder_id", None) is not None,
            "rom_runtimes_constructed": rom_runtimes,
        },
        "anti_windup_clamp": {
            "configured_lead_m": float(a.cylinder_target_max_lead_m),
            "bound_usd_per_cylinder": clamp["bound_usd"],
            "clamp_binding_fraction": saturating_fraction,
            "clamp_samples": clamp["samples"],
            "lead_usd_at_write_p50": _q(clamp_lead, 0.50),
            "lead_usd_at_write_p95": _q(clamp_lead, 0.95),
            "lead_usd_at_write_max": float(clamp_lead.max()),
            "warmup_samples_excluded": clamp.get("warmup_samples", 0),
            "neutral_hold_env_steps_excluded": clamp.get("neutral_hold_steps", 0),
        },
        "observed_lead_learned_space": {
            "note": (
                "Measured one control step after the clamp, against the joint "
                "position at OBSERVATION time. Legitimately exceeds the bound "
                "by however far the plant moved in that step; it is not a "
                "second measurement of the clamp."
            ),
            "p50_m": _q(observed, 0.50),
            "p95_m": _q(observed, 0.95),
            "max_m": float(observed.max()),
        },
        "contact": {
            "mean_force_n": float(force.mean()),
            "p95_force_n": _q(force, 0.95),
            "max_force_n": float(force.max()),
            "max_airborne_gap_m": float(gap.max()),
            "contact_loss_events": int(raw._contact_loss_event_count.sum()),
            "visible_lift_events": int(raw._visible_lift_event_count.sum()),
            "impact_ejection_events": int(raw._impact_ejection_event_count.sum()),
            "max_consecutive_airborne_steps": int(
                raw._max_consecutive_airborne_steps.max()
            ),
        },
    }

    # --- the audit's own pass/fail ---
    failures = []
    if a.actuator == "rom":
        if str(effective["cylinder_dynamics_model"]).lower() != "command_rom":
            failures.append("PPO3 is not running ROM-A")
        if getattr(raw, "_cylinder_id", None) is not None:
            failures.append("PPO3 constructed an ID runtime")
    else:
        if str(effective["cylinder_dynamics_model"]).lower() != "fd":
            failures.append("PPO4 is not running ID -> FD")
        if any(rom_runtimes.values()):
            failures.append(f"PPO4 constructed ROM runtimes {rom_runtimes}")
    if abs(effective["cylinder_target_max_lead_m"] - float(a.cylinder_target_max_lead_m)) > 1e-9:
        failures.append(
            "cylinder_target_max_lead_m was re-asserted after the override "
            f"({effective['cylinder_target_max_lead_m']} != {a.cylinder_target_max_lead_m})"
        )
    if abs(effective["cylinder_target_max_lead_m"] - 0.025) < 1e-9:
        failures.append("the inherited 25 mm lead is still in effect")
    if abs(effective["action_scale"] - float(a.action_scale)) > 1e-9:
        failures.append("action_scale was re-asserted after the override")
    if str(effective["sole_teacher_mode"]).lower() != "off":
        failures.append("teacher mode is not off")
    if bool(effective["enable_contact_tracking_demo"]):
        failures.append("the direct target-writing contact demo is active")
    # The freeze condition the task asks for: the clamp must not be binding for
    # a large fraction of the rollout.
    if saturating_fraction == saturating_fraction and saturating_fraction > 0.25:
        failures.append(
            f"anti-windup clamp binds on {saturating_fraction:.1%} of samples; "
            "the target integrator is still saturating"
        )
    report["failures"] = failures
    report["passed"] = not failures

    printable = {
        k: v for k, v in report.items() if k not in {"checkpoint_infos"}
    }
    print(json.dumps(printable, indent=2), flush=True)
    if a.report:
        path = Path(a.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2))
        print(f"[PPO3-SOLE] wrote {path}", flush=True)

    env.close()
    simulation_app.close()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
