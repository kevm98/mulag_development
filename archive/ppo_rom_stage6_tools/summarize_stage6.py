#!/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
"""One markdown table for every arm measured on the corrected stage-6 task.

Reads the merged ``*_gates.json`` reports a rebuild run produced and writes a
single comparison. The per-family lateral breakdown is included because the
lateral gate reads the WORST family, and knowing which family that is decides
whether the answer is "train longer" or "this path is not kinematically
reachable".
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

ARMS = [
    ("teacher (DLS, ROM plant)", "teacher_rom"),
    ("teacher (DLS, ID/FD plant)", "teacher_idfd"),
    ("null (zero action)", "null_rom"),
    ("ROM   distilled", "romstudent"),
    ("ID/FD distilled", "idfdstudent"),
    ("ROM   PPO (mispriced reward)", "romppo"),
    ("ROM   distilled + PPO", "romppo_best"),
    ("ID/FD distilled + PPO", "idfdppo_best"),
]


def load(root: Path, name: str) -> dict | None:
    path = root / f"{name}_gates.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def family_table(payload: dict) -> list[tuple[str, float, float]]:
    """Worst-mesh lateral error and ground contact per path family."""
    rows: dict[str, tuple[float, float]] = {}
    for metrics in (payload.get("per_mesh") or {}).values():
        for family in metrics.get("path_family_results", []):
            name = family["family"]
            lateral = float(family["mean_abs_lateral_error_m"])
            contact = float(family["ground_contact_fraction"])
            previous = rows.get(name)
            rows[name] = (
                max(lateral, previous[0]) if previous else lateral,
                min(contact, previous[1]) if previous else contact,
            )
    return sorted((n, v[0], v[1]) for n, v in rows.items())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory")
    p.add_argument("--report", default="")
    a = p.parse_args()
    root = Path(a.directory)

    loaded = [(label, name, load(root, name)) for label, name in ARMS]
    lines = [
        "# ROM and ID/FD on the corrected stage-6 driving task",
        "",
        f"Generated {datetime.now().isoformat(timespec='seconds')}.",
        "",
        "Stage 6 (flat ground with raised chunks), path amplitude band",
        "(0.30, 0.45, 0.60, 0.75), truck driving at 4 km/h, 6 held-out meshes,",
        "32 envs, 480 steps, worst mesh reported.",
        "",
        "These are the first numbers measured on the task the policies are",
        "actually trained on. Previously PPO trained with a PARKED truck (so the",
        "verge path never advanced), acceptance scored stage 5, and the stage-6",
        "curriculum overwrote the task's feasible amplitude band with a ladder",
        "whose every rung demanded more lateral cutter speed than the cylinders",
        "can deliver. Nothing measured before 2026-08-11 is comparable.",
        "",
    ]

    header = f"| {'arm':<28} | pass |" + "".join(
        f" {n[:11]:>11} |" for n in GATES
    )
    lines += [header, "|" + "-" * (len(header) - 2) + "|"]
    for label, _, payload in loaded:
        if payload is None:
            lines.append(
                f"| {label:<28} | --   |" + f" {'n/a':>11} |" * len(GATES)
            )
            continue
        gates = payload["gates"]
        cells = "".join(
            f" {gates['metrics'][g.metric]:>11.4f} |" for g in GATES.values()
        )
        lines.append(f"| {label:<28} | {'YES' if gates['passed'] else 'no ':<4} |{cells}")

    lines += ["", "Thresholds: " + ", ".join(
        f"{n} {'<=' if g.direction == 'max' else '>='}{g.threshold:g}"
        for n, g in GATES.items()
    ), ""]

    for label, _, payload in loaded:
        if payload is None:
            continue
        gates = payload["gates"]
        if not gates["passed"]:
            lines.append(f"- **{label}** fails: {', '.join(gates['failing'])}")
    lines.append("")

    lines += ["## Worst-mesh lateral error by path family", "",
              "The lateral gate reads the worst family. A family that is bad for the",
              "privileged teacher as well is a reachability question, not a training one.",
              ""]
    families = sorted({
        f[0] for _, _, payload in loaded if payload for f in family_table(payload)
    })
    present = [(label, payload) for label, _, payload in loaded if payload]
    hdr = f"| {'family':<18} |" + "".join(f" {l[:13]:>13} |" for l, _ in present)
    lines += [hdr, "|" + "-" * (len(hdr) - 2) + "|"]
    for family in families:
        row = f"| {family:<18} |"
        for _, payload in present:
            value = dict((n, lat) for n, lat, _ in family_table(payload)).get(family)
            row += f" {value:>13.4f} |" if value is not None else "           n/a |"
        lines.append(row)
    lines.append("")

    text = "\n".join(lines) + "\n"
    print(text)
    if a.report:
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
