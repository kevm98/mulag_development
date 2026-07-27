#!/usr/bin/env python3
"""Merge cylinder-4 sweeps into one best-aligned verified 3-D workspace."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+", type=Path)
    p.add_argument("--out", type=Path, default=Path("PPO2_ROM_PhysX/models/head_down_workspace_3d.json"))
    args = p.parse_args()
    grouped: dict[tuple[float, float, float], list[dict]] = defaultdict(list)
    for path in args.inputs:
        payload = json.loads((ROOT / path).read_text() if not path.is_absolute() else path.read_text())
        for row in payload["samples"]:
            if row["valid"]:
                key = tuple(round(float(x), 6) for x in row["cylinder_cmd"][:3])
                grouped[key].append(row)
    selected = [max(rows, key=lambda row: row["head_alignment"]) for rows in grouped.values()]
    selected.sort(key=lambda row: tuple(row["cylinder_cmd"][:3]))
    tool = np.asarray([row["cutter_truck"] for row in selected], dtype=np.float64)
    cylinder = np.asarray([row["cylinder_cmd"] for row in selected], dtype=np.float64)
    payload = {
        "schema_version": 2,
        "note": "Best head-down pose per cylinder-1/2/3 grid point across measured cylinder-4 sweeps.",
        "num_valid": len(selected),
        "tool_span_m": np.ptp(tool, axis=0).tolist(),
        "cylinder_span_m": np.ptp(cylinder, axis=0).tolist(),
        "cylinder3_extension_span_m": float(np.ptp(cylinder[:, 2])),
        "cutter_world_z_range_m": [
            float(min(row["cutter_world_z"] for row in selected)),
            float(max(row["cutter_world_z"] for row in selected)),
        ],
        "minimum_head_alignment": float(min(row["head_alignment"] for row in selected)),
        "samples": selected,
    }
    out = ROOT / args.out if not args.out.is_absolute() else args.out
    out.write_text(json.dumps(payload, indent=2))
    print(
        f"[WORKSPACE-MERGE] poses={len(selected)} tool_span={np.round(np.ptp(tool, axis=0), 3).tolist()} "
        f"cylinder_span={np.round(np.ptp(cylinder, axis=0), 3).tolist()} wrote={out}"
    )
    if len(selected) < 24 or np.ptp(cylinder[:, 2]) < 0.05:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
