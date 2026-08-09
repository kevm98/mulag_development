"""Build the verified multi-shape head-down trajectory library."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PPO1_ID_FD_PhysX.trajectories.workspace_shapes import build_trajectory_library


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        default=str(ROOT / "PPO1_ID_FD_PhysX/models/head_down_workspace_3d.json"),
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "PPO1_ID_FD_PhysX/models/head_down_trajectory_library.json"),
    )
    parser.add_argument("--num_points", type=int, default=241)
    args = parser.parse_args()
    payload = build_trajectory_library(args.workspace, num_points=args.num_points)
    output = Path(args.out).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    for name, shape in payload["shapes"].items():
        print(
            f"[{name}] length={shape['length_m']:.3f}m "
            f"tool_span={shape['tool_span_m']} "
            f"cylinder3_span={shape['cylinder3_extension_span_m']:.3f}m "
            f"head_min={shape['minimum_head_alignment']:.6f}"
        )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
