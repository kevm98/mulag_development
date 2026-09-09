#!/usr/bin/env python3
"""Collapse a strict-gate matrix into one promotion verdict per checkpoint.

Reads the JSON reports written by ``run_contact_safe_gates.sh``. A checkpoint
is promotable only if EVERY cell passes: the strict gates are absolute counts,
so a family that lifts once fails, and an aggregate over easy families is
exactly what the old duty-cycle gate let hide.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

STRICT = (
    "visible_lift_events",
    "impact_ejection_events",
    "hard_impacts",
    "consecutive_airborne_steps",
    "airborne_gap",
    "post_acquisition_contact",
    "contact_acquired",
    "takeoff_events",
    "contact_force_peak",
)

METRICS = (
    ("ground_contact_fraction", "contact", "{:.4f}"),
    ("post_acquisition_contact_fraction", "post-acq", "{:.4f}"),
    ("contact_loss_event_count", "loss ev", "{:.0f}"),
    ("visible_lift_event_count", "lifts", "{:.0f}"),
    ("impact_ejection_event_count", "eject", "{:.0f}"),
    ("hard_impact_event_count", "hard", "{:.0f}"),
    ("takeoff_event_count", "takeoff", "{:.0f}"),
    ("maximum_consecutive_airborne_steps", "airb", "{:.0f}"),
    ("maximum_airborne_gap_m", "maxgap", "{:.4f}"),
    ("maximum_contact_force_n", "maxF", "{:.0f}"),
    ("target_lead_max_m", "maxlead", "{:.4f}"),
    ("min_head_alignment", "align", "{:.4f}"),
    ("p95_tool_acceleration_mps2", "accel", "{:.1f}"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    docs = []
    for pattern in args.reports:
        for path in sorted(Path().glob(pattern)) or [Path(pattern)]:
            if path.is_file() and path.suffix == ".json":
                docs.append((path.stem, json.loads(path.read_text())))
    if not docs:
        raise SystemExit("no reports matched")

    lines = []
    header = ["cell", "17-gate", "strict"] + [label for _, label, _ in METRICS]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))

    all_promotable = True
    blockers_seen: dict[str, int] = {}
    for name, doc in docs:
        verge = doc.get("verge", {})
        gates = doc.get("contact_safety_gates") or {}
        strict_pass = all(gates.get(key, False) for key in STRICT)
        promotable = bool(doc.get("promotable", False))
        all_promotable &= promotable
        for blocker in doc.get("contact_safety_promotion_blockers") or []:
            blockers_seen[blocker] = blockers_seen.get(blocker, 0) + 1
        row = [
            name,
            "PASS" if doc.get("passed") else "fail",
            "PASS" if strict_pass else "fail",
        ]
        for key, _, fmt in METRICS:
            value = verge.get(key)
            row.append(fmt.format(value) if value is not None else "--")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append(f"**Promotable across every cell: {all_promotable}**")
    if blockers_seen:
        lines.append("")
        lines.append("| blocker | cells |")
        lines.append("|---|---|")
        for blocker, count in sorted(
            blockers_seen.items(), key=lambda kv: -kv[1]
        ):
            lines.append(f"| {blocker} | {count} |")

    # Teacher / demo provenance, checked rather than assumed.
    teacher = {str(d.get("verge", {}).get("teacher_mode")) for _, d in docs}
    demo = {bool(d.get("verge", {}).get("direct_target_contact_demo")) for _, d in docs}
    lines.append("")
    lines.append(f"teacher_mode across cells: {sorted(teacher)}")
    lines.append(f"direct_target_contact_demo across cells: {sorted(demo)}")

    text = "\n".join(lines)
    print(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")
        print(f"\n[wrote] {out}")


if __name__ == "__main__":
    main()
