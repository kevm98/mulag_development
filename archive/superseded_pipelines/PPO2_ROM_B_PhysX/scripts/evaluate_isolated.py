#!/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
"""Process-isolation parent for the ROM-B evaluator.

This file deliberately imports no Isaac or GPU modules. Each condition owns one
fresh Isaac process, and report assembly happens only after all shards exit.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVALUATOR = ROOT / "PPO2_ROM_B_PhysX/scripts/evaluate.py"


def _without_option(tokens: list[str], name: str, *, takes_value: bool) -> list[str]:
    result: list[str] = []
    skip = False
    for token in tokens:
        if skip:
            skip = False
            continue
        if token == name:
            skip = takes_value
            continue
        if takes_value and token.startswith(name + "="):
            continue
        result.append(token)
    return result


def main() -> int:
    original = sys.argv[1:]
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--trajectory_shapes", default="line,arc,sine,circle,square")
    parser.add_argument("--h_values", default="0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2")
    parser.add_argument("--monte_carlo_rounds", type=int, default=5)
    parser.add_argument("--h_sweep_rounds", type=int, default=3)
    parser.add_argument("--output_dir", default="")
    known, _unknown = parser.parse_known_args(original)
    shapes = tuple(value.strip() for value in known.trajectory_shapes.split(",") if value.strip())
    h_values = tuple(value.strip() for value in known.h_values.split(",") if value.strip())
    if not shapes or not h_values:
        raise SystemExit("isolated evaluation requires shapes and h values")
    output_dir = (
        Path(known.output_dir).expanduser().resolve()
        if known.output_dir
        else ROOT / "PPO2_ROM_B_PhysX/reports/evaluation_isolated"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    passthrough = _without_option(original, "--isolate_conditions", takes_value=False)
    passthrough = _without_option(passthrough, "--output_dir", takes_value=True)
    # One process per rollout, not merely per (suite, h, shape) condition.
    # Successive seed groups in one process measurably inherit simulator state.
    mc_rounds = int(known.monte_carlo_rounds)
    h_rounds = int(known.h_sweep_rounds)
    specs = [
        f"mc:{shape_index}:{round_index}"
        for shape_index in range(len(shapes))
        for round_index in range(mc_rounds)
    ]
    specs.extend(
        f"h:{h_index}:{shape_index}:{round_index}"
        for h_index in range(len(h_values))
        for shape_index in range(len(shapes))
        for round_index in range(h_rounds)
    )
    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory() as scratch:
        for index, spec in enumerate(specs, start=1):
            shard = Path(scratch) / f"shard_{index}.json"
            print(f"[ROM-B ISOLATE] {index}/{len(specs)} {spec}", flush=True)
            result = subprocess.run(
                [
                    sys.executable, str(EVALUATOR), "--shard", spec,
                    "--shard_output", str(shard), *passthrough,
                ],
                check=False,
            )
            if result.returncode != 0 or not shard.is_file():
                raise RuntimeError(f"shard {spec} failed with code {result.returncode}")
            rows.extend(json.loads(shard.read_text()))

    rows_path = output_dir / "isolated_rows.json"
    rows_path.write_text(json.dumps(rows, indent=2) + "\n")
    result = subprocess.run(
        [
            sys.executable, str(EVALUATOR), "--assemble_rows", str(rows_path),
            "--output_dir", str(output_dir), *passthrough,
        ],
        check=False,
    )
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
