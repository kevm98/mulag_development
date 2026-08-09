#!/usr/bin/env python3
"""Merge per-mesh evaluate_gates.py reports and score the 8-gate contract once.

`evaluate_gates.py` refuses to score gates on fewer than three meshes, so a
split run (one process per mesh, for wall-clock parallelism) writes per-mesh
metrics without a verdict. This joins them and calls `check_gates` exactly once,
which is the only place the worst-mesh aggregation is allowed to happen.

Pure Python -- `gates.py` imports nothing from Isaac Lab, so this runs in any
interpreter without booting a simulation.

Usage:
    merge_gates.py reports/frontier/student_m*.json --report merged.json
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

from PPO_ROM.gates import GATES, check_gates  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reports", nargs="+", help="per-mesh evaluate_gates.py JSONs")
    p.add_argument("--report", default="", help="write the merged verdict here")
    a = p.parse_args()

    per_mesh: dict[str, dict] = {}
    checkpoints: set[str] = set()
    for name in a.reports:
        path = Path(name)
        if not path.exists():
            raise SystemExit(f"missing per-mesh report: {path}")
        payload = json.loads(path.read_text())
        checkpoints.add(str(payload.get("checkpoint", "")))
        for mesh, metrics in (payload.get("per_mesh") or {}).items():
            if mesh in per_mesh:
                raise SystemExit(f"mesh {mesh} appears in more than one report")
            per_mesh[mesh] = metrics

    if len(checkpoints) != 1:
        raise SystemExit(f"reports are from different checkpoints: {sorted(checkpoints)}")

    result = check_gates(per_mesh)

    merged = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "checkpoint": checkpoints.pop(),
        "meshes": sorted(per_mesh),
        "per_mesh": per_mesh,
        "gates": result,
    }
    print(json.dumps(merged, indent=2, sort_keys=True), flush=True)

    print("\n" + "=" * 78, flush=True)
    print(f"{'gate':22s} {'value':>12s} {'threshold':>12s} {'margin':>12s}  verdict")
    print("-" * 78)
    for name, gate in GATES.items():
        value = result["metrics"][gate.metric]
        margin = result["margins"][name]
        verdict = "PASS" if result["gates"][name] else "FAIL"
        thin = " (thin)" if name in result["thin_margins"] else ""
        arrow = "<=" if gate.direction == "max" else ">="
        print(
            f"{name:22s} {value:12.4f} {arrow}{gate.threshold:11.4f} "
            f"{margin:+12.4f}  {verdict}{thin}"
        )
    print("=" * 78)
    print(
        f"meshes={result['meshes']}  passed={result['passed']}  "
        f"failing={result['failing']}",
        flush=True,
    )

    if a.report:
        out = Path(a.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(merged, indent=2, sort_keys=True))
        print(f"[PPO3-1-GATES] wrote {out}", flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
