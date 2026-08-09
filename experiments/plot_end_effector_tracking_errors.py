#!/usr/bin/env python3
"""Compare PPO1-A and PPO2-A end-effector tracking errors by shape."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PPO1_FILES = (
    "outputs/id_fd_policy_v3/model_575_evaluation.json",
)
DEFAULT_PPO2_FILES = (
    "outputs/ppo2_rom_residual/plant_reachable_v1_blend25_five_shape_strict.json",
)

SHAPE_ORDER = ("line", "arc", "sine", "circle", "square", "mower")


@dataclass(frozen=True)
class TrackingRecord:
    policy: str
    shape: str
    source: Path
    mean_error_m: float | None = None
    p95_error_m: float | None = None
    max_error_m: float | None = None
    final_error_m: float | None = None
    passed: bool | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot end-effector tracking errors for PPO1-A against PPO2-A. "
            "Lower error is better."
        )
    )
    parser.add_argument(
        "--ppo1",
        action="append",
        type=Path,
        default=None,
        help=(
            "PPO1-A JSON report to include. Can be passed multiple times. "
            "Defaults to the PPO1-A model_575 all-shape evaluation."
        ),
    )
    parser.add_argument(
        "--ppo2",
        action="append",
        type=Path,
        default=None,
        help=(
            "PPO2-A JSON report to include. Can be passed multiple times. "
            "Defaults to the canonical plant_reachable_v1_blend25 five-shape report."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "experiments",
        help="Directory for the generated PNG and CSV outputs.",
    )
    parser.add_argument(
        "--plot-name",
        default="end_effector_tracking_comparison.png",
        help="Output plot filename.",
    )
    parser.add_argument(
        "--csv-name",
        default="end_effector_tracking_summary.csv",
        help="Output CSV filename.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save files without opening an interactive matplotlib window.",
    )
    return parser.parse_args()


def resolve_paths(paths: Iterable[Path | str]) -> list[Path]:
    resolved = []
    for path in paths:
        path = Path(path)
        if not path.is_absolute():
            path = REPO_ROOT / path
        resolved.append(path)
    return resolved


def normalize_shape(value: str | None, source: Path) -> str:
    text = (value or source.stem).lower()
    text = text.replace("ground_", "").replace("trajectory_", "")
    if "straight" in text or text == "line":
        return "line"
    if "semicircle" in text or "arc" in text:
        return "arc"
    if "sine" in text:
        return "sine"
    if "circle" in text:
        return "circle"
    if "square" in text:
        return "square"
    if "mower" in text or "mow" in text:
        return "mower"
    return text.replace("_", "-")


def as_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * q
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def record_from_summary(policy: str, path: Path, data: dict) -> list[TrackingRecord]:
    shape = normalize_shape(data.get("trajectory_mode"), path)
    return [
        TrackingRecord(
            policy=policy,
            shape=shape,
            source=path,
            mean_error_m=as_float(data.get("tracking_err_mean")),
            max_error_m=as_float(data.get("tracking_err_max")),
            final_error_m=as_float(data.get("tracking_err_mean_last50")),
            passed=data.get("passed") if isinstance(data.get("passed"), bool) else None,
        )
    ]


def records_from_results(policy: str, path: Path, results: dict) -> list[TrackingRecord]:
    records = []
    for shape_name, metrics in results.items():
        if not isinstance(metrics, dict):
            continue
        records.append(
            TrackingRecord(
                policy=policy,
                shape=normalize_shape(shape_name, path),
                source=path,
                mean_error_m=as_float(metrics.get("mean_tool_error_m")),
                p95_error_m=as_float(metrics.get("p95_tool_error_m")),
                max_error_m=as_float(metrics.get("max_tool_error_m")),
                final_error_m=as_float(metrics.get("tool_error_m")),
                passed=metrics.get("completed_cycle")
                if isinstance(metrics.get("completed_cycle"), bool)
                else None,
            )
        )
    return records


def records_from_per_shape(policy: str, path: Path, per_shape: dict) -> list[TrackingRecord]:
    records = []
    for shape_name, pair in per_shape.items():
        if not isinstance(pair, dict):
            continue
        metrics = pair.get("candidate")
        if not isinstance(metrics, dict):
            continue
        records.append(
            TrackingRecord(
                policy=policy,
                shape=normalize_shape(shape_name, path),
                source=path,
                mean_error_m=as_float(metrics.get("mean_error_m")),
                p95_error_m=as_float(metrics.get("p95_error_m")),
                max_error_m=as_float(metrics.get("max_error_m")),
                final_error_m=None,
                passed=metrics.get("valid") if isinstance(metrics.get("valid"), bool) else None,
            )
        )
    return records


def records_from_events(policy: str, path: Path, events: list) -> list[TrackingRecord]:
    completed_records = []
    sample_errors: dict[str, list[float]] = {}
    for event in events:
        if not isinstance(event, dict):
            continue
        shape = normalize_shape(event.get("shape"), path)
        if event.get("event") == "arm_sample":
            error = as_float(event.get("tool_error_m"))
            if error is not None:
                sample_errors.setdefault(shape, []).append(error)
        if event.get("event") != "arm_cycle_complete":
            continue
        completed_records.append(
            TrackingRecord(
                policy=policy,
                shape=shape,
                source=path,
                mean_error_m=as_float(event.get("mean_tool_error_m")),
                p95_error_m=as_float(event.get("p95_tool_error_m")),
                max_error_m=max(sample_errors.get(shape, []), default=None),
                final_error_m=as_float(event.get("tool_error_m")),
                passed=True,
            )
        )

    if completed_records:
        return completed_records

    records = []
    for shape, errors in sample_errors.items():
        records.append(
            TrackingRecord(
                policy=policy,
                shape=shape,
                source=path,
                mean_error_m=sum(errors) / len(errors),
                p95_error_m=percentile(errors, 0.95),
                max_error_m=max(errors),
                final_error_m=errors[-1],
            )
        )
    return records


def load_records(policy: str, paths: Iterable[Path]) -> list[TrackingRecord]:
    records: list[TrackingRecord] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing {policy} report: {path}")
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            raise ValueError(f"Expected a JSON object in {path}")
        if "events" in data and isinstance(data["events"], list):
            records.extend(records_from_events(policy, path, data["events"]))
        elif "per_shape" in data and isinstance(data["per_shape"], dict):
            records.extend(records_from_per_shape(policy, path, data["per_shape"]))
        elif "results" in data and isinstance(data["results"], dict):
            records.extend(records_from_results(policy, path, data["results"]))
        elif "tracking_err_mean" in data:
            records.extend(record_from_summary(policy, path, data))
        else:
            raise ValueError(
                f"Could not find supported tracking metrics in {path}. "
                "Expected events, results, or tracking_err_mean."
            )
    return records


def choose_record(records: list[TrackingRecord]) -> TrackingRecord:
    """Keep the best run when multiple files produce the same policy/shape."""
    return min(records, key=lambda record: record.mean_error_m or float("inf"))


def combine_records(records: list[TrackingRecord]) -> dict[tuple[str, str], TrackingRecord]:
    grouped: dict[tuple[str, str], list[TrackingRecord]] = {}
    for record in records:
        grouped.setdefault((record.policy, record.shape), []).append(record)
    return {key: choose_record(value) for key, value in grouped.items()}


def ordered_shapes(records: dict[tuple[str, str], TrackingRecord]) -> list[str]:
    present = {shape for _, shape in records}
    ordered = [shape for shape in SHAPE_ORDER if shape in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def metric_value(record: TrackingRecord | None, metric: str) -> float | None:
    if record is None:
        return None
    return getattr(record, metric)


def write_csv(path: Path, records: dict[tuple[str, str], TrackingRecord]) -> None:
    rows = sorted(records.values(), key=lambda r: (r.shape, r.policy))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "policy",
                "shape",
                "mean_error_m",
                "p95_error_m",
                "max_error_m",
                "final_error_m",
                "passed",
                "source",
            ],
        )
        writer.writeheader()
        for record in rows:
            writer.writerow(
                {
                    "policy": record.policy,
                    "shape": record.shape,
                    "mean_error_m": record.mean_error_m,
                    "p95_error_m": record.p95_error_m,
                    "max_error_m": record.max_error_m,
                    "final_error_m": record.final_error_m,
                    "passed": record.passed,
                    "source": record.source.relative_to(REPO_ROOT)
                    if record.source.is_relative_to(REPO_ROOT)
                    else record.source,
                }
            )


def print_winner_table(records: dict[tuple[str, str], TrackingRecord], shapes: list[str]) -> None:
    print("\nEnd-effector tracking comparison (lower mean error is better)")
    print("shape       ppo1_a_mean_m   ppo2_a_mean_m   better")
    print("----------  --------------  --------------  ----------")
    for shape in shapes:
        ppo1 = records.get(("PPO1-A", shape))
        ppo2 = records.get(("PPO2-A", shape))
        ppo1_mean = metric_value(ppo1, "mean_error_m")
        ppo2_mean = metric_value(ppo2, "mean_error_m")
        if ppo1_mean is None and ppo2_mean is None:
            better = "n/a"
        elif ppo1_mean is None:
            better = "only PPO2-A"
        elif ppo2_mean is None:
            better = "only PPO1-A"
        elif ppo1_mean < ppo2_mean:
            better = "PPO1-A"
        elif ppo2_mean < ppo1_mean:
            better = "PPO2-A"
        else:
            better = "tie"
        print(
            f"{shape:<10}  "
            f"{ppo1_mean if ppo1_mean is not None else float('nan'):>14.5f}  "
            f"{ppo2_mean if ppo2_mean is not None else float('nan'):>14.5f}  "
            f"{better:<10}"
        )


def plot(records: dict[tuple[str, str], TrackingRecord], shapes: list[str], output_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    policies = ("PPO1-A", "PPO2-A")
    metrics = (
        ("mean_error_m", "Mean Tracking Error"),
        ("p95_error_m", "P95 Tracking Error"),
        ("max_error_m", "Max Tracking Error"),
    )
    colors = {"PPO1-A": "#4C78A8", "PPO2-A": "#F58518"}

    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5), constrained_layout=True)
    x = np.arange(len(shapes))
    width = 0.36

    for axis, (metric, title) in zip(axes, metrics):
        for index, policy in enumerate(policies):
            offset = (index - 0.5) * width
            values = [metric_value(records.get((policy, shape)), metric) for shape in shapes]
            numeric_values = [0.0 if value is None else value for value in values]
            bars = axis.bar(
                x + offset,
                numeric_values,
                width,
                label=policy,
                color=colors[policy],
                alpha=0.88,
            )
            for bar, value in zip(bars, values):
                if value is None:
                    bar.set_alpha(0.14)
                    axis.text(
                        bar.get_x() + bar.get_width() / 2,
                        0.002,
                        "n/a",
                        ha="center",
                        va="bottom",
                        rotation=90,
                        fontsize=8,
                    )
                else:
                    axis.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        f"{value * 100:.1f} cm",
                        ha="center",
                        va="bottom",
                        rotation=90,
                        fontsize=8,
                    )
        axis.set_title(title)
        axis.set_xticks(x)
        axis.set_xticklabels(shapes, rotation=30, ha="right")
        axis.set_ylabel("error [m]")
        axis.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.55)
        axis.legend()

    fig.suptitle("PPO1-A vs PPO2-A End-effector Tracking Errors by Shape")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)


def main() -> None:
    args = parse_args()
    ppo1_paths = resolve_paths(args.ppo1 or DEFAULT_PPO1_FILES)
    ppo2_paths = resolve_paths(args.ppo2 or DEFAULT_PPO2_FILES)

    records = combine_records(
        load_records("PPO1-A", ppo1_paths) + load_records("PPO2-A", ppo2_paths)
    )
    shapes = ordered_shapes(records)
    if not shapes:
        raise RuntimeError("No tracking records were loaded.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / args.csv_name
    plot_path = args.output_dir / args.plot_name
    write_csv(csv_path, records)
    print_winner_table(records, shapes)
    plot(records, shapes, plot_path)

    print(f"\nWrote summary CSV: {csv_path}")
    print(f"Wrote plot: {plot_path}")

    if not args.no_show:
        import matplotlib.pyplot as plt

        plt.show()


if __name__ == "__main__":
    main()
