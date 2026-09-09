#!/usr/bin/env python3
"""Aggregate contact-loss probe or evaluation reports into one comparison table.

Pure JSON in, Markdown/JSON out -- no Isaac Sim, no GPU. Kept as a script
rather than done by hand because the numbers in
``reports/CONTACT_SAFE_V1_REPORT.md`` have to be re-derivable from the stored
artifacts, and a hand-copied table is not.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROBE_ROWS = (
    ("ground_contact_fraction", "ground contact (duty cycle)", "{:.4f}"),
    ("post_acquisition_contact_fraction", "post-acquisition contact", "{:.4f}"),
    ("contact_loss_event_count", "contact-loss events", "{:.0f}"),
    ("visible_lift_event_count", "visible lifts (>5 mm)", "{:.0f}"),
    ("impact_ejection_event_count", "impact ejections", "{:.0f}"),
    ("hard_impact_event_count", "hard impacts (>30 kN)", "{:.0f}"),
    ("takeoff_event_count", "takeoffs", "{:.0f}"),
    ("max_consecutive_airborne_steps", "max consecutive airborne steps", "{:.0f}"),
    ("upward_separation_fraction", "upward separations", "{:.3f}"),
)

NESTED_ROWS = (
    ("airborne_gap", "max_m", "max airborne gap (m)", "{:.4f}"),
    ("airborne_gap", "p95_m", "p95 airborne gap (m)", "{:.4f}"),
    ("cutter_vertical_velocity", "max_positive_mps", "max upward velocity (m/s)", "{:.3f}"),
    ("contact_force_in_contact", "mean_n", "mean in-contact force (N)", "{:.0f}"),
    ("contact_force_in_contact", "p95_n", "p95 in-contact force (N)", "{:.0f}"),
    ("contact_force_in_contact", "max_n", "max contact force (N)", "{:.0f}"),
    ("contact_force_stable", "mean_n", "mean STABLE force (N)", "{:.0f}"),
    ("contact_force_stable", "p05_n", "p05 STABLE force (N)", "{:.0f}"),
    ("force_rate", "p95_abs_nps", "p95 |dF/dt| (N/s)", "{:.0f}"),
    ("target_lead", "p95_m", "p95 target lead (m)", "{:.4f}"),
    ("target_lead", "max_m", "max target lead (m)", "{:.4f}"),
    ("min_scan_valid_fraction", None, "min scan validity", "{:.4f}"),
    ("shield_active_fraction", None, "shield active fraction", "{:.4f}"),
    ("min_cylinder_headroom_m", None, "min cylinder headroom (m)", "{:.4f}"),
)


def _get(summary: dict, key: str, sub: str | None):
    value = summary.get(key)
    if sub is not None and isinstance(value, dict):
        value = value.get(sub)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    loaded = []
    for pattern in args.reports:
        for path in sorted(Path().glob(pattern)) or [Path(pattern)]:
            if not path.is_file():
                continue
            loaded.append((path, json.loads(path.read_text())))
    if not loaded:
        raise SystemExit("no reports matched")

    labels = [
        f"{doc.get('actuator', '?')}/s{doc.get('seed', '?')}" for _, doc in loaded
    ]
    lines = ["| metric | " + " | ".join(labels) + " |"]
    lines.append("|---|" + "---|" * len(labels))

    def row(label: str, values: list[str]) -> None:
        lines.append(f"| {label} | " + " | ".join(values) + " |")

    for key, label, fmt in PROBE_ROWS:
        cells = []
        for _, doc in loaded:
            value = doc["summary"].get(key)
            cells.append(fmt.format(value) if value is not None else "--")
        row(label, cells)
    for key, sub, label, fmt in NESTED_ROWS:
        cells = []
        for _, doc in loaded:
            value = _get(doc["summary"], key, sub)
            cells.append(fmt.format(value) if value is not None else "--")
        row(label, cells)

    # Event mechanism mix, summed across the loaded reports per actuator.
    kinds: dict[str, dict[str, int]] = {}
    for _, doc in loaded:
        actuator = str(doc.get("actuator", "?"))
        for kind, count in (doc["summary"].get("event_kind_counts") or {}).items():
            kinds.setdefault(kind, {}).setdefault(actuator, 0)
            kinds[kind][actuator] += int(count)

    lines.append("")
    lines.append("Event mechanism mix (summed over the loaded reports):")
    lines.append("")
    actuators = sorted({str(doc.get("actuator", "?")) for _, doc in loaded})
    lines.append("| mechanism | " + " | ".join(actuators) + " |")
    lines.append("|---|" + "---|" * len(actuators))
    for kind in sorted(kinds):
        lines.append(
            f"| {kind} | "
            + " | ".join(str(kinds[kind].get(act, 0)) for act in actuators)
            + " |"
        )

    text = "\n".join(lines)
    print(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")
        print(f"\n[wrote] {out}")


if __name__ == "__main__":
    main()
