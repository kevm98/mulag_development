#!/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
"""Propose gate thresholds from the measured stage-6 achievable band.

REPORTS ONLY. It never edits ``gates.py``: a threshold is a decision about what
the product must do, and the point of this script is to put the numbers in front
of that decision, not to make it automatically.

THE RULE IT APPLIES
===================
A threshold has to sit between two measurements, or it carries no information:

* **above the privileged DLS teacher** (with margin) for a "max" gate, below it
  for a "min" gate -- a limit the ground-truth controller cannot meet is a
  mis-set threshold, not a policy failure. This repo has re-derived three gates
  on exactly that argument already (``ground_contact`` 0.850 -> 0.700,
  ``hard_impacts`` 0.020 -> 0.100, ``safety_terminations`` 0.050 -> 0.080).
* **strictly on the far side of the null (zero-action) policy** -- otherwise
  doing nothing passes, and the gate is measuring the terrain rather than the
  task.

When those two orderings conflict -- the null is *better* than the teacher on a
metric -- no threshold separates them and the script says so instead of
inventing one. That is a real result about the metric, not a failure here.

Usage:
    derive_thresholds.py PPO_ROM/reports/stage6 --report proposal.json
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

from PPO_ROM.gates import GATES  # noqa: E402

TEACHER_MARGIN = 0.20

ANCHORS = {
    "rom": ("teacher_rom", "null_rom", ["romstudent", "romppo"]),
    "idfd": ("teacher_idfd", "null_rom", ["idfdstudent", "idfdppo"]),
}


def worst_mesh(report: Path) -> dict[str, float] | None:
    """Worst-mesh aggregate for every gate metric in a merged gate report."""
    if not report.is_file():
        return None
    payload = json.loads(report.read_text())
    per_mesh = payload.get("per_mesh") or {}
    if not per_mesh:
        return None
    out: dict[str, float] = {}
    for name, gate in GATES.items():
        base = gate.metric.removeprefix("worst_mesh_")
        values = [float(m[base]) for m in per_mesh.values() if base in m]
        if not values:
            continue
        out[name] = max(values) if gate.direction == "max" else min(values)
    return out


def propose(gate_name: str, teacher: float, null: float) -> dict:
    """One threshold, or an explicit refusal to invent one."""
    gate = GATES[gate_name]
    if gate.direction == "max":
        limit = teacher * (1.0 + TEACHER_MARGIN)
        limit = max(limit, teacher + 0.2 * abs(gate.threshold))
        separates = limit < null
    else:
        limit = teacher * (1.0 - TEACHER_MARGIN)
        limit = min(limit, teacher - 0.2 * abs(gate.threshold))
        separates = limit > null
    return {
        "gate": gate_name,
        "direction": gate.direction,
        "current": gate.threshold,
        "teacher_worst_mesh": teacher,
        "null_worst_mesh": null,
        "proposed": round(float(limit), 4),
        "separates_null": bool(separates),
        "note": (
            ""
            if separates
            else "NO threshold separates the teacher from the null on this "
            "metric; the null is at least as good as the ground-truth "
            "controller here. Do not set a gate on it until that is explained."
        ),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", help="directory of *_gates.json reports")
    p.add_argument("--report", default="")
    a = p.parse_args()

    root = Path(a.directory)
    measured = {
        name: worst_mesh(root / f"{name}_gates.json")
        for name in {
            "teacher_rom", "teacher_idfd", "null_rom",
            "romstudent", "idfdstudent", "romppo", "idfdppo",
        }
    }

    proposals: dict[str, list[dict]] = {}
    for pipeline, (teacher_key, null_key, _) in ANCHORS.items():
        teacher, null = measured.get(teacher_key), measured.get(null_key)
        if not teacher or not null:
            print(
                f"[DERIVE] {pipeline}: missing {teacher_key if not teacher else null_key}"
                " -- cannot anchor thresholds",
                flush=True,
            )
            continue
        proposals[pipeline] = [
            propose(name, teacher[name], null[name])
            for name in GATES
            if name in teacher and name in null
        ]

    subjects = [k for k, v in measured.items() if v]
    print("\n" + "=" * 100)
    print("MEASURED, stage 6, worst mesh of 6 (the acceptance aggregation)")
    print("=" * 100)
    print(f"{'gate':22s}{'thresh':>9s}" + "".join(f"{s:>13s}" for s in subjects))
    for name, gate in GATES.items():
        row = f"{name:22s}{gate.threshold:>9.3f}"
        for s in subjects:
            value = measured[s].get(name)
            row += "         n/a " if value is None else (
                f"{value:>11.4f}{'ok' if gate.passes(value) else 'XX'}"
            )
        print(row)

    for pipeline, rows in proposals.items():
        print("\n" + "-" * 100)
        print(f"PROPOSED for {pipeline} (teacher +/-{int(TEACHER_MARGIN * 100)}%)")
        print("-" * 100)
        print(f"{'gate':22s}{'current':>10s}{'teacher':>10s}{'null':>10s}"
              f"{'proposed':>11s}  separates null?")
        for r in rows:
            print(
                f"{r['gate']:22s}{r['current']:>10.3f}{r['teacher_worst_mesh']:>10.4f}"
                f"{r['null_worst_mesh']:>10.4f}{r['proposed']:>11.4f}"
                f"  {'yes' if r['separates_null'] else 'NO'}"
            )
        for r in rows:
            if r["note"]:
                print(f"  ! {r['gate']}: {r['note']}")

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "stage": 6,
        "teacher_margin": TEACHER_MARGIN,
        "measured": measured,
        "proposals": proposals,
        "applied": False,
    }
    if a.report:
        Path(a.report).write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(f"\n[DERIVE] proposal written to {a.report} (NOT applied)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
