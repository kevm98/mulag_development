from __future__ import annotations

import argparse
import ast
import csv
import json
import statistics
from pathlib import Path


parser = argparse.ArgumentParser(description="Fit a simple Simscape-FD to Isaac-effort torque adapter from CSV.")
parser.add_argument("--csv", type=Path, required=True, help="Path to logs/torque_testing/<run>/torque_test.csv.")
parser.add_argument(
    "--target_field",
    choices=["auto", "computed_torque", "applied_torque", "joint_effort", "tau_isaac_clamped"],
    default="auto",
    help="Isaac torque/effort CSV column to fit against.",
)
parser.add_argument(
    "--source_field",
    choices=["fd_simscape_output", "tau_fd_adapted", "tau_applied_raw", "tau_isaac_raw_after_adapter"],
    default="fd_simscape_output",
    help="CSV column used as the source signal.",
)
parser.add_argument(
    "--output",
    type=Path,
    default=None,
    help="Optional JSON output path. Defaults to torque_adapter_fit.json beside the CSV.",
)


def _parse_vector(text: str | None) -> list[float] | None:
    if text is None:
        return None
    text = text.strip()
    if not text:
        return None
    try:
        values = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return None
    if not isinstance(values, list) or len(values) != 4:
        return None
    try:
        return [float(value) for value in values]
    except (TypeError, ValueError):
        return None


def _choose_target_field(rows: list[dict[str, str]], requested: str) -> str:
    if requested != "auto":
        return requested
    for field in ("computed_torque", "applied_torque", "joint_effort", "tau_isaac_clamped"):
        if any(_parse_vector(row.get(field)) is not None for row in rows):
            return field
    raise RuntimeError("No usable target torque column found in CSV.")


def _fit_scale_bias(xs: list[float], ys: list[float]) -> tuple[float, float]:
    if len(xs) < 2:
        return 0.0, ys[0] if ys else 0.0
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    var_x = sum((x - x_mean) ** 2 for x in xs)
    if var_x <= 1.0e-12:
        return 0.0, y_mean
    cov_xy = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    scale = cov_xy / var_x
    bias = y_mean - scale * x_mean
    return scale, bias


def _robust_median_scale(xs: list[float], ys: list[float]) -> float:
    abs_x = [abs(x) for x in xs if abs(x) > 1.0e-12]
    abs_y = [abs(y) for y in ys]
    if not abs_x or not abs_y:
        return 0.0
    return statistics.median(abs_y) / max(statistics.median(abs_x), 1.0e-12)


def main(args: argparse.Namespace) -> None:
    csv_path = args.csv.expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    target_field = _choose_target_field(rows, args.target_field)
    source_by_joint: list[list[float]] = [[] for _ in range(4)]
    target_by_joint: list[list[float]] = [[] for _ in range(4)]

    for row in rows:
        source = _parse_vector(row.get(args.source_field))
        target = _parse_vector(row.get(target_field))
        if source is None or target is None:
            continue
        for joint_id in range(4):
            source_by_joint[joint_id].append(source[joint_id])
            target_by_joint[joint_id].append(target[joint_id])

    scales: list[float] = []
    biases: list[float] = []
    robust_scales: list[float] = []
    counts: list[int] = []
    for joint_id in range(4):
        xs = source_by_joint[joint_id]
        ys = target_by_joint[joint_id]
        scale, bias = _fit_scale_bias(xs, ys)
        scales.append(scale)
        biases.append(bias)
        robust_scales.append(_robust_median_scale(xs, ys))
        counts.append(len(xs))

    result = {
        "csv": str(csv_path),
        "source_field": args.source_field,
        "target_field": target_field,
        "scale": scales,
        "bias": biases,
        "robust_median_scale": robust_scales,
        "samples_per_joint": counts,
    }

    output_path = args.output.expanduser().resolve() if args.output is not None else csv_path.parent / "torque_adapter_fit.json"
    with output_path.open("w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")

    print(f"[INFO] source_field={args.source_field} target_field={target_field}")
    for joint_id in range(4):
        print(
            f"joint_{joint_id}: scale={scales[joint_id]:.8e} "
            f"bias={biases[joint_id]:.8e} "
            f"robust_median_scale={robust_scales[joint_id]:.8e} "
            f"samples={counts[joint_id]}"
        )
    print(f"[INFO] Wrote {output_path}")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
