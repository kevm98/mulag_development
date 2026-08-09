#!/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
"""Does the cylinder target re-anchor to the MEASURED position each step?

WHY
===
`mulag_upper_level_env.py:2444` chooses the base the commanded delta is added to:

    target_base = previous_cylinder_pos_target if integrates_from_prev else cylinder_q
    new_target  = target_base + delta_isaac

With the flag False the base is `cylinder_q`, the MEASURED joint position. Any
sag, servo lag or load-induced droop is therefore absorbed into the next
baseline and the command that should have corrected it is lost. With the flag
True the base is the previous TARGET, so a droop shows up as tracking error the
servo actively drives out.

`cylinder_target_integrates_from_previous_target = True` is set in
PPO1_ID_FD_PhysX, mulag_plant/tasks, PPO4_ID_FD_PhysX and the ROM pipeline.
It is set NOWHERE in the PPO3 / PPO3_1 chain, so PPO3 runs at the runtime
default, False. PPO3 is also the one pipeline that has never learned tracking.
Two prior diagnoses in this repo (`ppo1-tracking-ratchet-fix`,
`idfd-learning-and-trajectory`) recorded the same defect and fixed it by setting
this flag True.

WHAT THIS MEASURES
==================
Under a ZERO action the policy is asking to hold station. So:

  * flag True  -> the target stands still; the servo resists droop and the
                  measured extension should stay near its initial value.
  * flag False -> the target follows the arm wherever it sagged to, so
                  extension and lateral error should walk away monotonically.

A monotone walk-away under zero command is the ratchet, and no reward shaping
can fix it: the policy's correction is discarded by the integrator.

This modifies nothing permanently -- the flag is set on a locally constructed
cfg, one env build per arm, and both arms use the same seed and action.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--num_envs", type=int, default=32)
p.add_argument("--steps", type=int, default=400)
p.add_argument("--seed", type=int, default=17)
p.add_argument("--report", default="PPO_ROM/reports/target_anchoring.json")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
a.headless = True
app = AppLauncher(a).app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import PPO_ROM  # noqa: E402,F401
from PPO_ROM import TERRAIN_TASK_ID  # noqa: E402

FLAG = "cylinder_target_integrates_from_previous_target"


env_cfg = parse_env_cfg(TERRAIN_TASK_ID, device=a.device, num_envs=a.num_envs)
DECLARED = FLAG in type(env_cfg).__dict__ or hasattr(env_cfg, FLAG)
INHERITED = getattr(env_cfg, FLAG, "<absent>")
ENV = gym.make(TERRAIN_TASK_ID, cfg=env_cfg, render_mode=None)
RAW = ENV.unwrapped


def run(integrates: bool) -> dict:
    """One arm. The env is built ONCE and the flag toggled between arms.

    `mulag_upper_level_env.py:2443` re-reads the flag with `getattr` on every
    control step, so flipping it on the live cfg genuinely changes the anchoring
    -- and building once means both arms see bit-identical terrain, spawn state
    and path assignment. Two separate builds would have confounded the
    comparison with terrain-generation variance, which is the single largest
    source of spread in this repo's measurements.
    """
    raw = RAW
    declared, inherited = DECLARED, INHERITED
    setattr(raw.cfg, FLAG, integrates)
    assert bool(getattr(raw.cfg, FLAG, False)) is integrates, "flag toggle did not stick"
    raw.reset(seed=int(a.seed))
    torch.manual_seed(int(a.seed))

    zero = torch.zeros((raw.num_envs, int(raw.cfg.action_space)), device=raw.device)
    lateral, extension, headroom = [], [], []
    # Read the env's OWN published metrics rather than re-calling internals.
    # The first version of this probe called `_refresh_cylinder_state()` here,
    # but `_get_rewards` already calls it inside the same step, so the ROM
    # pipeline's internal filter state was advanced twice per control step. The
    # process died mid-arm with no Python traceback (a native crash, not an
    # exception), which is exactly what corrupting that state can produce.
    # `Ppo31SoleEnv._publish_log` puts everything needed into extras["log"].
    for index in range(int(a.steps)):
        raw.step(zero)
        log = raw.extras.get("log", {})
        lateral.append(float(log.get("Metrics/lateral_error_m", float("nan"))))
        headroom.append(float(log.get("Episode_Reward/cylinder_headroom", float("nan"))))
        extension.append(float(log.get("Metrics/cut_height_error_m", float("nan"))))
        if index % 25 == 0:
            print(f"    step {index:>4}/{a.steps}  lateral={lateral[-1]:.4f} "
                  f"headroom_pen={headroom[-1]:.3f}", flush=True)

    n = len(lateral)
    first, last = lateral[: n // 5], lateral[-n // 5 :]
    return {
        "integrates_from_previous_target": integrates,
        "flag_was_declared_on_cfg": bool(declared),
        "inherited_value_before_override": str(inherited),
        "lateral_first_fifth_m": statistics.mean(first),
        "lateral_last_fifth_m": statistics.mean(last),
        "lateral_drift_m": statistics.mean(last) - statistics.mean(first),
        "lateral_max_m": max(lateral),
        "cut_height_error_first_fifth_m": statistics.mean(extension[: n // 5]),
        "cut_height_error_last_fifth_m": statistics.mean(extension[-n // 5 :]),
        "headroom_first_fifth_m": statistics.mean(headroom[: n // 5]),
        "headroom_last_fifth_m": statistics.mean(headroom[-n // 5 :]),
        "lateral_trace": [round(x, 5) for x in lateral[::20]],
    }


results = {}
for integrates in (False, True):
    label = "integrating(True)" if integrates else "measured-anchored(False)"
    print(f"\n=== zero-action hold, {label} ===", flush=True)
    results[str(integrates)] = run(integrates)
    r = results[str(integrates)]
    print(f"  lateral  first1/5={r['lateral_first_fifth_m']:.4f}  "
          f"last1/5={r['lateral_last_fifth_m']:.4f}  drift={r['lateral_drift_m']:+.4f} m", flush=True)
    print(f"  headroom first1/5={r['headroom_first_fifth_m']:.4f}  "
          f"last1/5={r['headroom_last_fifth_m']:.4f} m", flush=True)

off, on = results["False"], results["True"]
verdict = {
    # MEASURED 2026-08-05: PPO3_1 already INHERITS True, via
    # PPO3 -> mulag_plant/policy/env_cfg.py:14 -> mulag_plant/tasks/env_cfg.py:101.
    # A repo-wide grep for the flag name shows no hit inside the PPO3 files, which
    # is why this probe was written on the premise that PPO3 ran at the False
    # default. That premise was WRONG: grepping which FILES mention a field does
    # not resolve an inherited configclass attribute. `audit_plant.py` says exactly
    # this in its own docstring -- "Config inspection is not enough" -- and the
    # correct move is to read the CONSTRUCTED cfg, which is what
    # `inherited_value_before_override` below records.
    "current_ppo3_setting": str(INHERITED),
    "drift_measured_anchored_m": off["lateral_drift_m"],
    "drift_integrating_m": on["lateral_drift_m"],
    "integrating_reduces_drift": on["lateral_drift_m"] < off["lateral_drift_m"],
    "drift_reduction_m": off["lateral_drift_m"] - on["lateral_drift_m"],
}
print("\n=== VERDICT ===", flush=True)
print(json.dumps(verdict, indent=2), flush=True)

out = Path(a.report)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"verdict": verdict, "arms": results}, indent=2, sort_keys=True))
print(f"\nwrote {out}", flush=True)
ENV.close()
app.close()
