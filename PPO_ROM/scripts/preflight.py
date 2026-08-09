#!/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
"""PPO3_1 preflight + stress. Run this BEFORE committing GPU hours.

Every check here corresponds to a failure that actually happened in this
project, so a green preflight means those specific ways of wasting an
afternoon are closed:

  P1  the task builds the MERGED env (c4 authority + rebuilt reward), not a
      parent class -- `train.py` instantiated the parent runner cfg directly
      and silently trained at the OLD regime (lr 3e-4, entropy 0.001).
  P2  the runner cfg really carries the standard regime.
  P3  c4 per-axis authority is active, not a stale PPO3 default.
  P4  the curriculum stage is the tracking stage -- the stage fallback used to
      be dead code (`getattr(cfg, "curriculum_stage", 4)` never fired because
      the attribute existed as 0) and the guard rejected it.
  P5  standard Isaac Lab log namespaces are published. Overriding
      `_get_rewards` silences the parent's logging entirely.
  S1  every reward term stays inside its published bound under adversarial
      actions (saturated, alternating, zero).
  S2  no non-finite reward under any of those.
  S3  the reward is not constant -- a dead objective trains nothing.
  S4  the reward RESPONDS to lateral error in the right direction, measured on
      the real plant rather than in a unit test.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--num_envs", type=int, default=32)
p.add_argument("--steps", type=int, default=120)
p.add_argument("--seed", type=int, default=17)
p.add_argument("--report", default="")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
a.headless = True
app = AppLauncher(a).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import PPO_ROM  # noqa: E402,F401
from PPO_ROM import TERRAIN_TASK_ID as _PPO31_TASK  # noqa: E402

# PPO4_1 runs these SAME checks against the ID/FD plant by setting this env var,
# rather than keeping a second copy of the file that would drift away from this
# one. P4 (path feasibility) and S5 (shaping gradients) are the checks that
# encode real failures; they must be identical across both pipelines or a
# PPO3_1-vs-PPO4_1 comparison measures the divergence instead of the actuator.
TERRAIN_TASK_ID = os.environ.get("PPO31_PREFLIGHT_TASK", "") or _PPO31_TASK
if TERRAIN_TASK_ID.startswith("RRLAB-Mulag-Verge-IDFD"):
    import PPO_ID_FD  # noqa: F401,E402
print(f"[preflight] task={TERRAIN_TASK_ID}", flush=True)
from PPO_ROM.env import Ppo31SoleEnv  # noqa: E402
from PPO_ROM.env_cfg import Ppo31PPORunnerCfg  # noqa: E402
from PPO_ROM.rewards import RewardWeights  # noqa: E402

failures: list[str] = []
results: dict[str, object] = {}


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'ok  ' if ok else 'FAIL'}] {name}{(' :: ' + detail) if detail else ''}", flush=True)
    results[name] = {"passed": bool(ok), "detail": detail}
    if not ok:
        failures.append(name)


# --- P2: runner cfg, checked without booting a scene ------------------------
runner = Ppo31PPORunnerCfg()
check(
    "P2 standard regime in runner cfg",
    abs(runner.algorithm.learning_rate - 1.0e-3) < 1e-12
    and str(runner.algorithm.schedule) == "adaptive"
    and abs(runner.algorithm.entropy_coef - 0.005) < 1e-12,
    f"lr={runner.algorithm.learning_rate} schedule={runner.algorithm.schedule} "
    f"entropy={runner.algorithm.entropy_coef}",
)

env_cfg = parse_env_cfg(TERRAIN_TASK_ID, device=a.device, num_envs=a.num_envs)
env = gym.make(TERRAIN_TASK_ID, cfg=env_cfg, render_mode=None)
raw = env.unwrapped
raw.reset(seed=int(a.seed))

check("P1 merged env is registered", isinstance(raw, Ppo31SoleEnv), type(raw).__name__)

lead = tuple(getattr(raw.cfg, "cylinder_target_max_lead_per_axis_m", ()) or ())
scale = tuple(getattr(raw.cfg, "cylinder_action_scale_per_axis_m", ()) or ())
check(
    "P3 c4 authority active",
    len(lead) == 4 and len(scale) == 4 and lead[3] > lead[0] and scale[3] > scale[0],
    f"lead={lead} scale={scale}",
)
# `curriculum_stage` reads 5 on purpose: apply_sole_curriculum_stage remaps
# stage 6 -> 5 because "stage 6 shares stage 5's arm contract exactly -- same
# authority, filter, slew and clearance band. Only the GROUND and the tracking
# path differ." So the ARM contract is stage 5 and asserting 6 there was simply
# the wrong field. What stage 6 actually changes is the relief profile and the
# path amplitude, which is what this checks.
# The amplitude is no longer pinned to a literal. It is now derived from what
# the cylinders can deliver, so the check that matters is FEASIBILITY, not a
# magic number: assert every path family's peak demanded lateral speed sits
# under the actuator ceiling. This is the same computation as
# scripts/probe_path_feasibility.py, run on the live config, and it is the
# check that would have caught the shipped 2.2 (9 of 11 families over ceiling,
# worst 3.7x over) before it cost a training campaign.
_ceiling_mps = float(raw.cfg.cylinder_action_scale_per_axis_m[0]) / (
    float(raw.cfg.rom_dt) * int(raw.cfg.decimation)
)
_demands = {}
try:
    from dataclasses import replace as _replace

    from verge_task.terrain import (
        TRACKING_PATH_FAMILIES,
        apply_relief_profile,
        tracking_line_center_y_at,
    )

    # Use the env's OWN profiled strip. Two earlier attempts got this wrong:
    # `apply_relief_profile("flat_patches")` REGENERATES hills and hollows from
    # the seed, so any height scaling applied before it is silently discarded,
    # and `terrain_height_scales` never reaches the terms boom_sweep reads.
    # Measuring the config the env actually builds removes both traps.
    _base = raw._mow_strip_cfg()
    _truck_mps = float(raw.cfg.vehicle_speed_min_kmh) / 3.6
    for _family in sorted(TRACKING_PATH_FAMILIES):
        _cfg = _replace(_base, tracking_path_family=_family)
        _x = np.linspace(_cfg.start_x_m, _cfg.start_x_m + _cfg.length_m, 4000)
        _y = tracking_line_center_y_at(_cfg, _x)
        _slope = np.abs(np.gradient(_y, _x))
        _demands[_family] = float(np.nanmax(_slope) * _truck_mps)
except Exception as exc:  # noqa: BLE001
    _demands = {"probe_failed": float("inf")}
    print(f"[preflight] feasibility probe error: {exc}")

_over = {f: v for f, v in _demands.items() if v > _ceiling_mps}
check(
    "P4 every path family is inside the cylinder rate ceiling",
    str(getattr(raw.cfg, "terrain_relief_profile", "")) == "flat_patches"
    and not _over,
    f"relief={getattr(raw.cfg, 'terrain_relief_profile', None)} "
    f"amplitude={getattr(raw.cfg, 'terrain_tracking_amplitude_scale', None)} "
    f"sweep_gain={getattr(raw.cfg, 'terrain_tracking_height_sweep_gain', None)} "
    f"ceiling={_ceiling_mps:.3f} m/s worst={max(_demands.values()):.3f} "
    f"({max(_demands, key=_demands.get)}) over={sorted(_over)} "
    f"all={ {k: round(v, 3) for k, v in sorted(_demands.items(), key=lambda kv: -kv[1])} } "
    f"(arm contract stage={getattr(raw.cfg, 'curriculum_stage', None)}, 5 is correct)",
)

weights = RewardWeights()
width = int(raw.cfg.action_space)


def rollout(make_action, steps: int) -> dict:
    """Run `steps` and return bound/finite/spread statistics."""
    worst = {name: 0.0 for name in weights.__dataclass_fields__}
    nonfinite = 0
    totals: list[float] = []
    laterals: list[float] = []
    lateral: list[float] = []
    for index in range(steps):
        raw.step(make_action(index))
        terms = getattr(raw, "_ppo31_last_terms", None)
        if terms is None:
            continue
        for name, value in terms.as_dict().items():
            if not torch.isfinite(value).all():
                nonfinite += 1
            worst[name] = max(worst.get(name, 0.0), float(value.abs().max()))
        totals.append(float(terms.total.mean()))
        laterals.append(float(terms.lateral.mean()))
        lateral.append(float(raw.verge_lateral_error().abs().mean()))
    return {
        "worst": worst,
        "nonfinite": nonfinite,
        "totals": totals,
        "lateral": lateral,
        "lateral_term": laterals,
    }


def act_zero(_):
    return torch.zeros((raw.num_envs, width), device=raw.device)


def act_saturated(_):
    return torch.ones((raw.num_envs, width), device=raw.device)


def act_alternating(index):
    sign = 1.0 if index % 2 == 0 else -1.0
    return torch.full((raw.num_envs, width), sign, device=raw.device)


def act_random(_):
    return torch.clamp(
        torch.randn((raw.num_envs, width), device=raw.device) * 0.5, -1.0, 1.0
    )


torch.manual_seed(int(a.seed))
stress = {}
for label, fn, steps in (
    ("zero", act_zero, a.steps // 4),
    ("saturated", act_saturated, a.steps // 4),
    ("alternating", act_alternating, a.steps // 4),
    ("random", act_random, a.steps // 4),
):
    stress[label] = rollout(fn, max(steps, 8))

bound_breaches = []
nonfinite_total = 0
for label, data in stress.items():
    nonfinite_total += data["nonfinite"]
    for name, peak in data["worst"].items():
        limit = abs(getattr(weights, name, 0.0))
        if limit and peak > limit + 1e-4:
            bound_breaches.append(f"{label}:{name}={peak:.4f}>{limit}")

check("S1 all terms inside published bounds", not bound_breaches, "; ".join(bound_breaches[:4]))
check("S2 no non-finite reward under stress", nonfinite_total == 0, f"count={nonfinite_total}")

spread = max(
    (max(d["totals"]) - min(d["totals"])) for d in stress.values() if d["totals"]
)
check("S3 reward is not a dead objective", spread > 1e-6, f"max spread={spread:.4f}")

# S4: on the real plant, worse lateral ERROR must earn less LATERAL REWARD.
# Deliberately not the total: the total is dominated by contact and head, which
# vary independently of lateral error (under saturated actions the arm can leave
# the ground while lateral error is still small), so comparing the total would
# measure the confound rather than the term.
pairs = []
for data in stress.values():
    pairs.extend(zip(data["lateral"], data["lateral_term"]))
pairs.sort()
if len(pairs) >= 6:
    third = max(len(pairs) // 3, 1)
    lo = sum(t for _, t in pairs[:third]) / third
    hi = sum(t for _, t in pairs[-third:]) / third
    check(
        "S4 lower lateral error earns more lateral reward",
        lo >= hi,
        f"best-third={lo:.4f} worst-third={hi:.4f}",
    )
else:
    check("S4 lower lateral error earns more lateral reward", False, "insufficient samples")

# S5: the check that would have saved 3.6 GPU-hours.
#
# Run 2026-08-05_11-51-48 trained 888 iterations with `Episode_Reward/lateral`
# pinned at 0.0000. `lateral_scale_m` was 0.30 and the clamp cap was 1.0, so the
# term saturated at 0.30 m and its derivative was identically zero beyond that.
# The policy sat at 0.734 m -- 2.4x past the cliff -- receiving no lateral signal
# at all, and mean lateral error rose 0.217 -> 0.736 over the run.
#
# A shaping term whose gradient dies BEFORE its own acceptance threshold can
# never drive the policy to pass that threshold. So: every shaping term must
# have a live gradient at 2x its gate, checked against the real thresholds in
# gates.py rather than against a number written here.
from PPO_ROM.gates import GATES  # noqa: E402
from PPO_ROM.rewards import compute_reward  # noqa: E402


def _term_gradient(field: str, kwarg: str, probe: float) -> float:
    n = 1
    error = torch.tensor([probe], requires_grad=True)
    inputs = dict(
        cut_height_error=torch.zeros(n),
        lateral_error=torch.zeros(n),
        head_down_alignment=torch.ones(n),
        cutter_contact=torch.ones(n, dtype=torch.bool),
        applied_action=torch.zeros((n, width)),
        previous_applied_action=torch.zeros((n, width)),
        tool_acceleration_norm=torch.zeros(n),
        contact_force_n=torch.zeros(n),
        cylinder_headroom_low_m=torch.ones((n, 4)),
        cylinder_headroom_high_m=torch.ones((n, 4)),
    )
    # The headroom terms are (envs, 4) and the term takes a min over cylinders,
    # so the probe has to fill the row rather than replace the tensor.
    if kwarg == "cylinder_headroom_low_m":
        inputs["cylinder_headroom_low_m"] = error.expand(n, 4)
    else:
        inputs[kwarg] = error
    getattr(compute_reward(**inputs), field).sum().backward()
    return float(error.grad.abs())


# EVERY shaping term, not just the ones already known to be broken.
#
# The first version of this check probed only `lateral` and `cut_height` -- the
# two terms I had just fixed -- so it confirmed my own work instead of auditing
# the class. Running it against all terms found THREE more dead gradients,
# including `head_alignment`, which was the one actually doing damage: with
# head_scale 0.02 against threshold 0.98 it saturated at alignment 0.96, so the
# whole failing range below the 0.900 gate carried zero gradient. Measured on
# model_850, alignment p01 0.666 and min 0.237 both returned exactly 0.000, and
# collapsing the head was therefore a FREE way to buy ground contact.
#
# For a "max" gate the failing side is a LARGER value, so probe 2x the
# threshold. For a "min" gate the failing side is SMALLER, so probe a value
# well below it -- taken from what a real collapsed policy measured.
PROBES = (
    # (gate name, reward field, kwarg, probe value, why this probe)
    ("lateral", "lateral", "lateral_error", 2.0 * GATES["lateral"].threshold, "2x gate"),
    ("lateral", "lateral", "lateral_error", 0.734, "failed-run operating point"),
    ("cut_height", "cut_height", "cut_height_error", 2.0 * GATES["cut_height"].threshold, "2x gate"),
    ("head_alignment", "head_alignment", "head_down_alignment", GATES["head_alignment"].threshold, "at gate"),
    ("head_alignment", "head_alignment", "head_down_alignment", 0.237, "model_850 min alignment"),
    ("tool_acceleration", "tool_acceleration", "tool_acceleration_norm", 2.0 * GATES["tool_acceleration"].threshold, "2x gate"),
    ("cylinder_headroom", "cylinder_headroom", "cylinder_headroom_low_m", -0.010, "driven past the travel limit"),
    (None, "contact_force", "contact_force_n", 123_000.0, "measured rigid-cutter peak"),
)

dead_terms = []
gradient_detail = []
for _gate, field, kwarg, probe, why in PROBES:
    grad = _term_gradient(field, kwarg, probe)
    gradient_detail.append(f"{field}@{probe:g}={grad:.4g}")
    if grad <= 1e-9:
        dead_terms.append(f"{field} has ZERO gradient at {probe:g} ({why})")

check(
    "S5 shaping gradients alive past 2x their gate",
    not dead_terms,
    "; ".join(dead_terms) if dead_terms else " ".join(gradient_detail),
)

log = raw.extras.get("log", {})
episode_keys = [k for k in log if k.startswith("Episode_Reward/")]
metric_keys = [k for k in log if k.startswith("Metrics/")]
check(
    "P5 Isaac Lab log namespaces published",
    len(episode_keys) >= 9 and len(metric_keys) >= 4,
    f"{len(episode_keys)} Episode_Reward/, {len(metric_keys)} Metrics/",
)

env.close()
summary = {"passed": not failures, "failures": failures, "checks": results}
print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
if a.report:
    out = Path(a.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True))
print(("[PREFLIGHT PASSED]" if not failures else f"[PREFLIGHT FAILED] {failures}"), flush=True)
app.close()
sys.exit(1 if failures else 0)
