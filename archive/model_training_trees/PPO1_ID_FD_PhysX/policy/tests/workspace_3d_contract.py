"""Contract for mixed-height/head-down workspace coverage."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    payload = json.loads((ROOT / "PPO1_ID_FD_PhysX/models/head_down_workspace_3d.json").read_text())
    assert payload["num_valid"] >= 24
    assert payload["cylinder_span_m"][1] >= 0.05, "cylinder 2 must exercise height authority"
    assert payload["cylinder3_extension_span_m"] >= 0.05, "cylinder 3 must extend and contract"
    assert payload["tool_span_m"][2] >= 0.20, "tool reference must contain visible height variation"
    assert payload["minimum_head_alignment"] >= 0.995
    print(
        "workspace_3d_contract: PASS "
        f"poses={payload['num_valid']} tool_z_span={payload['tool_span_m'][2]:.3f}m "
        f"cylinder3_span={payload['cylinder3_extension_span_m']:.3f}m"
    )


if __name__ == "__main__":
    main()
