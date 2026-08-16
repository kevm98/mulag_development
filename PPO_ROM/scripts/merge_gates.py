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


# These fields determine whether per-mesh results are valid members of one
# paired promotion campaign.  Keep them in the merged report; dropping them
# makes a later statistical gate unable to tell an L20 estimate from a legacy
# run or an 8 mm reserve from a nominal plant.
CONTRACT_FIELDS = (
    "actuator",
    "action_width",
    "cylinder_target_max_lead_per_axis_m",
    "cylinder_target_lead_high_speed_per_axis_m",
    "cylinder_target_lead_schedule_speed_kmh",
    "cylinder_target_integrates_from_previous_target",
    "cylinder_target_leak",
    "cylinder_target_limit_inset_m",
    "dls_null_space_centering_gain",
    "c3_physical_extension_reserve_m",
    "online_aam_latent_mode",
    "online_aam_controller_latent_mode",
    "online_aam_residual_scale",
    "z_hat_is_live",
    "vehicle_speed_min_kmh",
    "vehicle_speed_max_kmh",
    "factor_domain",
    "factor_key",
    "deployment_factor_enabled",
    "deployment_factor_manifest",
    "online_aam_checkpoint",
    "online_aam_adapter_checkpoint",
    "online_aam_history_lag",
    "online_aam_smoother_window",
    "evaluation_seed",
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reports", nargs="+", help="per-mesh evaluate_gates.py JSONs")
    p.add_argument("--report", default="", help="write the merged verdict here")
    a = p.parse_args()

    per_mesh: dict[str, dict] = {}
    checkpoints: set[str] = set()
    stages: set[int] = set()
    contract_values: dict[str, set[str]] = {key: set() for key in CONTRACT_FIELDS}
    contract_payloads: dict[str, object] = {}
    z_hat_across_env_stds: list[float] = []
    z_hat_temporal_deltas: list[float] = []
    for name in a.reports:
        path = Path(name)
        if not path.exists():
            raise SystemExit(f"missing per-mesh report: {path}")
        payload = json.loads(path.read_text())
        z_hat_across_env_stds.append(
            float(payload.get("z_hat_across_env_std_observed", 0.0))
        )
        z_hat_temporal_deltas.append(
            float(payload.get("z_hat_temporal_delta_observed", 0.0))
        )
        checkpoints.add(str(payload.get("checkpoint", "")))
        stages.add(int(payload.get("stage", -1)))
        for key in CONTRACT_FIELDS:
            value = payload.get(key)
            encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
            contract_values[key].add(encoded)
            contract_payloads.setdefault(key, value)
        for mesh, metrics in (payload.get("per_mesh") or {}).items():
            if mesh in per_mesh:
                raise SystemExit(f"mesh {mesh} appears in more than one report")
            per_mesh[mesh] = metrics

    if len(checkpoints) != 1:
        raise SystemExit(f"reports are from different checkpoints: {sorted(checkpoints)}")
    if len(stages) != 1:
        raise SystemExit(f"reports are from different stages: {sorted(stages)}")
    mismatches = {
        key: sorted(values)
        for key, values in contract_values.items()
        if len(values) > 1
    }
    if mismatches:
        raise SystemExit(
            "reports are from different promotion contracts: "
            f"{mismatches}"
        )

    result = check_gates(per_mesh)

    merged = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "checkpoint": checkpoints.pop(),
        "stage": stages.pop(),
        **contract_payloads,
        "z_hat_across_env_std_observed": max(z_hat_across_env_stds, default=0.0),
        "z_hat_temporal_delta_observed": max(z_hat_temporal_deltas, default=0.0),
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
