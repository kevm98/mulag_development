#!/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
"""Build and stress-check the final task before a long training run."""

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
p.add_argument(
    "--stage",
    type=int,
    default=6,
    help=(
        "Curriculum stage to certify. Defaults to the stage both pipelines "
        "train at; the checks are meaningless for any other one."
    ),
)
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

TERRAIN_TASK_ID = os.environ.get("PPO31_PREFLIGHT_TASK", "") or _PPO31_TASK
if TERRAIN_TASK_ID.startswith("RRLAB-Mulag-Verge-IDFD"):
    import PPO_ID_FD  # noqa: F401,E402
print(f"[preflight] task={TERRAIN_TASK_ID}", flush=True)
from PPO_ROM.env import Ppo31SoleEnv  # noqa: E402
from PPO_ROM.env_cfg import Ppo31PPORunnerCfg  # noqa: E402
from PPO_ROM.rewards import RewardWeights  # noqa: E402
from PPO_ROM.terrain_env_cfg import apply_c4_authority_stage  # noqa: E402

failures: list[str] = []
results: dict[str, object] = {}


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'ok  ' if ok else 'FAIL'}] {name}{(' :: ' + detail) if detail else ''}", flush=True)
    results[name] = {"passed": bool(ok), "detail": detail}
    if not ok:
        failures.append(name)


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
apply_c4_authority_stage(env_cfg, int(a.stage))
env_cfg.trajectory_shape = str(env_cfg.terrain_deployment_trajectory_shape)
env_cfg.randomize_trajectory_shape = False
env_cfg.randomize_start_phase = False
env = gym.make(TERRAIN_TASK_ID, cfg=env_cfg, render_mode=None)
raw = env.unwrapped

_strip_start = float(raw.cfg.mow_strip_start_x_m)
_strip_length = float(raw.cfg.mow_strip_length_m)
_path_start = float(raw.cfg.terrain_tracking_reference_x_m)
_path_length = float(raw.cfg.terrain_tracking_active_length_m)
_phase_anchor = float(raw.cfg.tracking_path_phase_anchor_x_m)
check(
    "P0b full green strip is active",
    abs(_strip_start - _path_start) <= 1.0e-6
    and abs(_strip_length - _path_length) <= 1.0e-6
    and abs(_phase_anchor - _path_start) <= 1.0e-6,
    (
        f"strip=[{_strip_start:.2f}, {_strip_start + _strip_length:.2f}] "
        f"path=[{_path_start:.2f}, {_path_start + _path_length:.2f}] "
        f"phase_anchor={_phase_anchor:.2f}"
    ),
)
raw.reset(seed=int(a.seed))

_cutter_start_x = float(
    (
        raw._compute_cutter_pos_w()[:, 0]
        - raw.scene.env_origins[:, 0]
    ).mean()
)
_strip_end = _strip_start + _strip_length
_cutter_end_x = _cutter_start_x + _path_length
_coverage_tolerance_m = max(float(raw.cfg.mow_strip_resolution_m), 0.05)
check(
    "P0c cutter's full travel is covered by the strip",
    _cutter_start_x >= _strip_start - _coverage_tolerance_m
    and _cutter_end_x <= _strip_end + _coverage_tolerance_m,
    (
        f"cutter=[{_cutter_start_x:.2f}, {_cutter_end_x:.2f}] "
        f"strip=[{_strip_start:.2f}, {_strip_end:.2f}] "
        f"tolerance={_coverage_tolerance_m:.2f}m"
    ),
)

check(
    "P0 the truck is actually driving",
    float(raw.cfg.vehicle_speed_min_kmh) > 0.1,
    f"vehicle_speed={raw.cfg.vehicle_speed_min_kmh:.2f}-"
    f"{raw.cfg.vehicle_speed_max_kmh:.2f} km/h "
    f"trajectory={raw.cfg.trajectory_shape} "
    "(0.00 makes P4 vacuous)",
)

check("P1 merged env is registered", isinstance(raw, Ppo31SoleEnv), type(raw).__name__)

lead = tuple(getattr(raw.cfg, "cylinder_target_max_lead_per_axis_m", ()) or ())
scale = tuple(getattr(raw.cfg, "cylinder_action_scale_per_axis_m", ()) or ())
_c4_stroke_m = 0.170
_c1_steps = (lead[0] / scale[0]) if len(lead) == 4 and len(scale) == 4 and scale[0] else 0.0
_c4_steps = (lead[3] / scale[3]) if len(lead) == 4 and len(scale) == 4 and scale[3] else 0.0
check(
    "P3 c4 anti-windup can bind",
    len(lead) == 4
    and len(scale) == 4
    and lead[3] < _c4_stroke_m
    and _c4_steps * 4.0 >= _c1_steps,
    f"lead={lead} scale={scale} c4_steps_to_cap={_c4_steps:.1f} c1={_c1_steps:.1f}",
)
_ceiling_mps = float(raw.cfg.cylinder_action_scale_per_axis_m[0]) / (
    float(raw.sim.get_physics_dt()) * int(raw.cfg.decimation)
)
_demands = {}
try:
    from dataclasses import replace as _replace

    from verge_task.terrain import (
        TRACKING_PATH_FAMILIES,
        apply_relief_profile,
        tracking_line_center_y_at,
    )

    _base = raw._mow_strip_cfg()
    _truck_mps = float(raw.cfg.vehicle_speed_min_kmh) / 3.6
    _ladder = tuple(getattr(raw.cfg, "terrain_tracking_amplitude_scales", ()) or ())
    _amplitudes = sorted(set(_ladder) | {float(_base.tracking_path_amplitude_scale)})
    for _family in sorted(TRACKING_PATH_FAMILIES):
        _worst = 0.0
        for _amplitude in _amplitudes:
            _cfg = _replace(
                _base,
                tracking_path_family=_family,
                tracking_path_amplitude_scale=float(_amplitude),
            )
            _x = np.linspace(_cfg.start_x_m, _cfg.start_x_m + _cfg.length_m, 4000)
            _y = tracking_line_center_y_at(_cfg, _x)
            _slope = np.abs(np.gradient(_y, _x))
            _worst = max(_worst, float(np.nanmax(_slope) * _truck_mps))
        _demands[_family] = _worst
except Exception as exc:  # noqa: BLE001
    _demands = {"probe_failed": float("inf")}
    print(f"[preflight] feasibility probe error: {exc}")

_over = {f: v for f, v in _demands.items() if v > _ceiling_mps}
check(
    "P4 every path family is inside the cylinder rate ceiling",
    str(getattr(raw.cfg, "terrain_relief_profile", "")) == "flat_patches"
    and not _over,
    f"relief={getattr(raw.cfg, 'terrain_relief_profile', None)} "
    f"amplitudes={_amplitudes} "
    f"sweep_gain={getattr(raw.cfg, 'terrain_tracking_height_sweep_gain', None)} "
    f"ceiling={_ceiling_mps:.3f} m/s worst={max(_demands.values()):.3f} "
    f"({max(_demands, key=_demands.get)}) over={sorted(_over)} "
    f"all={ {k: round(v, 3) for k, v in sorted(_demands.items(), key=lambda kv: -kv[1])} } "
    f"(arm contract stage={getattr(raw.cfg, 'curriculum_stage', None)}, 5 is correct)",
)

weights = raw._reward_weights()
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
    if kwarg == "cylinder_headroom_low_m":
        inputs["cylinder_headroom_low_m"] = error.expand(n, 4)
    else:
        inputs[kwarg] = error
    getattr(compute_reward(**inputs, weights=weights), field).sum().backward()
    return float(error.grad.abs())


PROBES = (
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
