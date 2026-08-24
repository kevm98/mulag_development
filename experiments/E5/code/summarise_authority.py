#!/usr/bin/env python3
"""Degradation curve over actuator authority, and the frozen stress levels.

Applies the thresholds declared in the E5 README BEFORE any curve was seen:

    degradation onset    median lateral error >= 1.5x its own alpha=1 value,
                         or ground contact >= 5 percentage points below it
    functional breakdown ground contact < 0.700, or safety termination > 0.080

Two jobs, kept apart on purpose:

``--stack if --write-levels`` reads the IF-only discovery scan, locates the
knee and writes the stress levels the final evaluation will use.  Selection
happens here, on the non-adaptive baseline, and nowhere else.

``--stack all`` reports the final curve for the three stacks at those frozen
levels.  It does not select anything, so it cannot move the levels toward a
result.

The E3 magnitude comes from the same place: the level whose post/pre error
ratio sits in [1.5, 2.5] is the one with recovery headroom -- weaker and there
is nothing to recover, stronger and no controller can.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

STACKS = {"ppo_rom": "R", "ppo_id_fd": "IF", "ppo_aam_id_fd": "AIF"}
ONSET_ERROR_RATIO = 1.5
ONSET_CONTACT_DROP = 0.05
BREAKDOWN_CONTACT = 0.700
BREAKDOWN_TERMINATION = 0.080
HEADROOM_BAND = (1.5, 2.5)


def read_point(path: Path, mesh: str) -> dict | None:
    if not path.is_file():
        return None
    block = json.loads(path.read_text()).get("per_mesh", {}).get(mesh)
    if not block:
        return None
    families = block.get("path_family_results", [])
    lateral = np.asarray([f["mean_abs_lateral_error_m"] for f in families], dtype=float) \
        if families else np.asarray([np.nan])
    contact = np.asarray([f["ground_contact_fraction"] for f in families], dtype=float) \
        if families else np.asarray([np.nan])
    return {
        "median_lateral_m": float(np.nanmedian(lateral)),
        "mean_lateral_m": float(np.nanmean(lateral)),
        "median_contact": float(np.nanmedian(contact)),
        "worst_contact": float(block.get("worst_family_ground_contact_fraction", np.nan)),
        "termination": float(block.get("safety_termination_rate", np.nan)),
        "families": int(len(families)),
    }


def collect(report_dir: Path, stack: str, mesh: str) -> dict[float, dict]:
    points: dict[float, dict] = {}
    for path in sorted(report_dir.glob(f"{stack}_a*_m{mesh}.json")):
        match = re.match(rf"{re.escape(stack)}_a(?P<alpha>[0-9.]+)_m", path.name)
        if not match:
            continue
        row = read_point(path, mesh)
        if row is not None:
            points[float(match.group("alpha"))] = row
    return points


def classify(points: dict[float, dict]) -> dict:
    """Locate onset and breakdown against the predeclared thresholds."""
    if not points:
        return {"onset_alpha": None, "breakdown_alpha": None, "reference_alpha": None}
    reference_alpha = max(points)          # the least-stressed point available
    reference = points[reference_alpha]
    onset = breakdown = None
    for alpha in sorted(points, reverse=True):
        row = points[alpha]
        degraded = (
            row["median_lateral_m"] >= ONSET_ERROR_RATIO * reference["median_lateral_m"]
            or row["median_contact"] <= reference["median_contact"] - ONSET_CONTACT_DROP
        )
        broken = (row["worst_contact"] < BREAKDOWN_CONTACT
                  or row["termination"] > BREAKDOWN_TERMINATION)
        if degraded and onset is None:
            onset = alpha
        if broken and breakdown is None:
            breakdown = alpha
    return {"onset_alpha": onset, "breakdown_alpha": breakdown,
            "reference_alpha": reference_alpha}


def print_curve(name: str, points: dict[float, dict], verdict: dict) -> None:
    print(f"\n{name}: authority degradation curve")
    head = (f"  {'alpha':>7} {'median lat (m)':>15} {'x nominal':>10} "
            f"{'median contact':>15} {'worst contact':>14} {'term.':>7}")
    print(head)
    print("  " + "-" * (len(head) - 2))
    reference = points[verdict["reference_alpha"]] if points else None
    for alpha in sorted(points, reverse=True):
        row = points[alpha]
        ratio = (row["median_lateral_m"] / reference["median_lateral_m"]
                 if reference and reference["median_lateral_m"] > 0 else float("nan"))
        marks = []
        if alpha == verdict["onset_alpha"]:
            marks.append("ONSET")
        if alpha == verdict["breakdown_alpha"]:
            marks.append("BREAKDOWN")
        print(f"  {alpha:>7.2f} {row['median_lateral_m']:>15.4f} {ratio:>10.2f} "
              f"{row['median_contact']:>15.4f} {row['worst_contact']:>14.4f} "
              f"{row['termination']:>7.3f}  {' '.join(marks)}")


def choose_levels(points: dict[float, dict], verdict: dict) -> dict:
    """Frozen evaluation levels: nominal, onset, headroom, near breakdown."""
    reference_alpha = verdict["reference_alpha"]
    reference = points[reference_alpha]
    ratios = {alpha: (row["median_lateral_m"] / reference["median_lateral_m"]
                      if reference["median_lateral_m"] > 0 else float("nan"))
              for alpha, row in points.items()}
    headroom = [alpha for alpha, r in ratios.items()
                if HEADROOM_BAND[0] <= r <= HEADROOM_BAND[1]
                and points[alpha]["worst_contact"] >= BREAKDOWN_CONTACT
                and points[alpha]["termination"] <= BREAKDOWN_TERMINATION]
    chosen = {reference_alpha}
    if verdict["onset_alpha"] is not None:
        chosen.add(verdict["onset_alpha"])
    if headroom:
        chosen.add(max(headroom))          # mildest level with real headroom
    if verdict["breakdown_alpha"] is not None:
        chosen.add(verdict["breakdown_alpha"])
    return {
        "alpha_levels": sorted(chosen, reverse=True),
        "selected_on": "ppo_id_fd",
        "selection_rule": (
            "onset and breakdown by the predeclared thresholds; the E3 headroom "
            "level is the mildest alpha whose median lateral error is 1.5-2.5x "
            "nominal while contact and termination stay inside the gates"
        ),
        "e3_headroom_alpha": max(headroom) if headroom else None,
        "thresholds": {
            "onset_error_ratio": ONSET_ERROR_RATIO,
            "onset_contact_drop": ONSET_CONTACT_DROP,
            "breakdown_contact": BREAKDOWN_CONTACT,
            "breakdown_termination": BREAKDOWN_TERMINATION,
            "headroom_band": list(HEADROOM_BAND),
        },
        "curve": {f"{alpha:.2f}": points[alpha] for alpha in sorted(points, reverse=True)},
        "onset_alpha": verdict["onset_alpha"],
        "breakdown_alpha": verdict["breakdown_alpha"],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--report-dir", required=True)
    p.add_argument("--mesh", required=True)
    p.add_argument("--stack", default="all", choices=("all", "if", "r", "aif"))
    p.add_argument("--write-levels", default=None,
                   help="freeze the stress levels to this path (IF discovery only)")
    a = p.parse_args()

    report_dir = Path(a.report_dir).expanduser().resolve()
    wanted = ({"ppo_id_fd": "if", "ppo_rom": "r", "ppo_aam_id_fd": "aif"}
              if a.stack == "all" else None)
    prefixes = ({"if": "ppo_id_fd", "r": "ppo_rom", "aif": "ppo_aam_id_fd"}[a.stack],) \
        if a.stack != "all" else tuple(STACKS)

    results = {}
    for stack in prefixes:
        points = collect(report_dir, stack, a.mesh)
        if not points:
            print(f"\n{STACKS[stack]}: no reports in {report_dir}")
            continue
        verdict = classify(points)
        results[stack] = (points, verdict)
        print_curve(STACKS[stack], points, verdict)
        onset = verdict["onset_alpha"]
        breakdown = verdict["breakdown_alpha"]
        print(f"  onset at alpha = {onset if onset is not None else 'not reached'}; "
              f"breakdown at alpha = "
              f"{breakdown if breakdown is not None else 'not reached'}")

    if a.write_levels:
        if a.stack != "if":
            raise SystemExit("--write-levels is only valid with --stack if: the "
                             "levels must be selected on the non-adaptive baseline")
        points, verdict = results.get("ppo_id_fd", ({}, {}))
        if not points:
            raise SystemExit("no IF points to select from")
        levels = choose_levels(points, verdict)
        out = Path(a.write_levels).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(levels, indent=2))
        print(f"\nfrozen levels: {levels['alpha_levels']}")
        print(f"E3 headroom level: {levels['e3_headroom_alpha']}")
        if levels["e3_headroom_alpha"] is None:
            print("  no level lands in the 1.5-2.5x band: extend the scan before"
                  "\n  running E3, rather than using a level outside the band")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
