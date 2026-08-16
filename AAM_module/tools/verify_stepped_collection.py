#!/usr/bin/env python3
"""Did a stepped collection actually produce a plant that changes mid-episode?

Collection succeeding is not evidence of this. If the step never fired, or
fired on the wrong runs, or the schedule never reached metadata, then training
converges, figures render, and the tracking result measures nothing.

Checks, per run:

  1. metadata carries a factor_schedule (or is a declared constant control)
  2. the schedule's tick lies strictly inside the episode
  3. the RECORDED plant response actually changes across that tick -- the
     schedule is a claim, this is the evidence
  4. the mix contains BOTH stepped and constant runs, since the Phase 3 gate
     needs between-plant separation (constant runs) and tracking (stepped runs)

Run after collection, before building the AAM dataset.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


# Measured on the pilot: under closed-loop DLS the tracked outputs (dP,
# cylinder_velocity) barely move across a plant step because the controller
# absorbs the change. The evidence shows up in CONTROL EFFORT instead, so that
# is what is scored here. A direct A/B on the ROM confirmed the model responds
# strongly to seal (max|diff| 0.449 against an output scale of 0.337), so a
# flat control-effort response means the step did not reach the plant, not that
# the plant is insensitive.
EFFORT_SIGNALS = ("controller_output", "command_after_limits", "requested_valve_cmd")


def episode_signal(run_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """(ticks, control-effort) for a run."""
    path = run_dir / "trajectory.npz"
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as d:
        keys = set(d.files)
        for name in EFFORT_SIGNALS:
            if name not in keys:
                continue
            arr = np.asarray(d[name], dtype=np.float64)
            arr = arr.reshape(arr.shape[0], -1)
            if not np.isfinite(arr).any():
                continue           # e.g. applied_valve_cmd is unavailable
            return np.arange(arr.shape[0]), arr
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()

    runs_root = a.dataset / "runs"
    if not runs_root.is_dir():
        raise SystemExit(f"no runs directory under {a.dataset}")

    report, stepped, constant, problems = {}, [], [], []
    for run_dir in sorted(d for d in runs_root.iterdir() if d.is_dir()):
        meta_path = run_dir / "metadata.json"
        if not meta_path.is_file():
            problems.append(f"{run_dir.name}: no metadata.json")
            continue
        meta = json.loads(meta_path.read_text())
        schedule = meta.get("factor_schedule") or []
        total = int(meta.get("num_control_steps", 0))
        entry = {"schedule": schedule, "num_control_steps": total}

        sig = episode_signal(run_dir)
        default_tick = int(total // 2)
        if not schedule:
            constant.append(run_dir.name)
            entry["kind"] = "constant"
            # score the control runs at the SAME nominal tick so the two
            # groups are compared on equal footing
            if sig is not None:
                ticks, arr = sig
                pre, post = arr[ticks < default_tick], arr[ticks >= default_tick]
                if len(pre) and len(post):
                    pm = float(np.abs(pre).mean())
                    entry["effort_ratio"] = (
                        float(np.abs(post).mean()) / pm if pm else float("nan")
                    )
        else:
            stepped.append(run_dir.name)
            entry["kind"] = "stepped"
            seg = schedule[0]
            tick = int(seg["from_tick"])
            entry["from_tick"] = tick
            entry["factors"] = seg.get("factors", {})
            if not (0 < tick < total):
                problems.append(
                    f"{run_dir.name}: step tick {tick} outside (0, {total}); "
                    "no observable change"
                )
            if sig is None:
                entry["response_change"] = None
            else:
                ticks, arr = sig
                pre = arr[ticks < tick]
                post = arr[ticks >= tick]
                if len(pre) and len(post):
                    # Control effort settles over an episode for every run, so
                    # the absolute ratio is not evidence on its own; it is only
                    # meaningful against the constant runs, compared below.
                    pre_m = float(np.abs(pre).mean())
                    post_m = float(np.abs(post).mean())
                    entry["effort_ratio"] = post_m / pre_m if pre_m else float("nan")
        report[run_dir.name] = entry

    print(f"runs: {len(report)}   stepped: {len(stepped)}   constant: {len(constant)}")
    for name in sorted(report):
        e = report[name]
        ratio = e.get("effort_ratio")
        ratio_s = f"{ratio:.3f}" if isinstance(ratio, float) else "n/a"
        if e["kind"] == "stepped":
            print(f"  {name}  STEPPED at {e['from_tick']} -> {e['factors']}  "
                  f"effort post/pre {ratio_s}")
        else:
            print(f"  {name}  constant                                    "
                  f"effort post/pre {ratio_s}")

    s_r = [report[n]["effort_ratio"] for n in stepped
           if isinstance(report[n].get("effort_ratio"), float)]
    c_r = [report[n]["effort_ratio"] for n in constant
           if isinstance(report[n].get("effort_ratio"), float)]
    if s_r and c_r:
        sa, ca = np.asarray(s_r, float), np.asarray(c_r, float)
        s_m, c_m = float(sa.mean()), float(ca.mean())
        # Perfect separation is the right bar only at tiny n, where it is also
        # cheap to hit by luck (p = 1/3 at 2-vs-2). With real group sizes a
        # rank/permutation test is the correct instrument: overlapping ranges
        # are entirely compatible with a large, unambiguous effect.
        auc = float(np.mean([[1.0 if a > b else 0.5 if a == b else 0.0
                              for b in ca] for a in sa]))
        obs = abs(s_m - c_m)
        pool = np.concatenate([sa, ca])
        rng = np.random.default_rng(0)
        n_s, N = len(sa), 20000
        hits = 0
        for _ in range(N):
            rng.shuffle(pool)
            if abs(pool[:n_s].mean() - pool[n_s:].mean()) >= obs:
                hits += 1
        pval = (hits + 1) / (N + 1)
        print(f"\ncontrol effort post/pre -- stepped {s_m:.3f} (n={len(sa)})  "
              f"constant {c_m:.3f} (n={len(ca)})")
        print(f"  permutation p = {pval:.5f}   AUC = {auc:.3f}   "
              f"difference {s_m - c_m:+.3f}")
        if len(sa) < 5 or len(ca) < 5:
            problems.append(
                f"only {min(len(sa), len(ca))} runs in the smaller group: too "
                "few to establish the step is observable, whatever the "
                "direction (collect >= 5 per group)"
            )
        elif not (pval < 0.01 and auc > 0.8):
            problems.append(
                f"step not demonstrably observable in control effort "
                f"(p={pval:.4f}, AUC={auc:.3f}); expected p<0.01 and AUC>0.8"
            )
        else:
            print("  -> the plant step IS observable in the recorded response")

    if not stepped:
        problems.append(
            "no stepped runs: this dataset cannot support an in-episode "
            "tracking claim (rerun with --step-factor)"
        )
    if not constant:
        problems.append(
            "no constant runs: between-plant separation is not cleanly "
            "measurable (use --step-every-n > 1 for a mixed dataset)"
        )

    print()
    if problems:
        print("PROBLEMS")
        for msg in problems:
            print(f"  - {msg}")
    else:
        print("OK: mixed stepped/constant collection with observable plant changes")

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(
            {"format": "stepped_collection_check_v1",
             "n_stepped": len(stepped), "n_constant": len(constant),
             "problems": problems, "runs": report}, indent=2))
        print(f"\nwrote {a.out}")
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
