#!/usr/bin/env python3
"""Monitor PPO wide-task logs and explain tracking regressions.

This is intentionally read-only with respect to training.  The divergence
guard remains the safety stop; this monitor records the first useful evidence
around a drift instead of waiting for the guard's hard 3x threshold.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path


ITER_RE = re.compile(r"Learning iteration\s+(\d+)/")
FIELDS = {
    "lateral": "Metrics/lateral_error_m",
    "cut_error": "Metrics/cut_height_error_m",
    "contact_fraction": "Metrics/ground_contact_fraction",
    "head_alignment": "Metrics/head_alignment",
    "reward_lateral": "Episode_Reward/lateral",
    "reward_cut": "Episode_Reward/cut_height",
    "reward_contact": "Episode_Reward/ground_contact",
    "reward_contact_loss": "Episode_Reward/contact_loss",
    "reward_hard_impact": "Episode_Reward/hard_impact",
    "reward_total": "Episode_Reward/total",
    "explained_variance": "Diagnostics/explained_variance",
    "action_std": "Mean action std",
    "value_loss": "Mean value loss",
    "surrogate_loss": "Mean surrogate loss",
    "update_rejected": "Mean update_rejected loss",
    "kl_early_stop": "Mean kl_early_stop loss",
    "reference_kl": "Mean reference_kl loss",
    "reference_kl_before_update": "Mean reference_kl_before_update loss",
    "reference_kl_attempted": "Mean reference_kl_attempted loss",
    "termination_limit": "termination/limit",
    "termination_nonfinite": "termination/nonfinite",
    "termination_truck_drift": "termination/truck_drift",
    "termination_head_down": "termination/head_down",
    "termination_clearance": "termination/clearance",
    "termination_timeout": "termination/timeout",
    "termination_terrain_nonfinite": "termination/terrain_nonfinite",
    "termination_contact_safety": "termination/contact_safety",
}


def _number(block: str, label: str) -> float | None:
    match = re.search(
        re.escape(label) + r"\s*:\s*([-+0-9.eE]+)", block
    )
    return float(match.group(1)) if match else None


def read_rows(log_path: Path) -> list[dict[str, float]]:
    if not log_path.is_file():
        return []
    text = log_path.read_text(errors="ignore")
    parts = ITER_RE.split(text)
    rows: dict[int, dict[str, float]] = {}
    for index in range(1, len(parts), 2):
        iteration = int(parts[index])
        block = parts[index + 1]
        row: dict[str, float] = {"iteration": float(iteration)}
        for name, label in FIELDS.items():
            value = _number(block, label)
            if value is not None:
                row[name] = value
        rows[iteration] = row
    return [rows[key] for key in sorted(rows)]


def median(rows: list[dict[str, float]], key: str) -> float | None:
    values = [row[key] for row in rows if key in row]
    return statistics.median(values) if values else None


def mean(rows: list[dict[str, float]], key: str) -> float | None:
    values = [row[key] for row in rows if key in row]
    return sum(values) / len(values) if values else None


def explain(rows: list[dict[str, float]], warmup: int) -> dict[str, object]:
    if not rows:
        return {"state": "waiting", "reason": "no learning iterations logged"}

    latest = rows[-1]
    warm_rows = [row for row in rows if row["iteration"] <= warmup]
    baseline = median(warm_rows, "lateral")
    recent = rows[-5:]
    previous = rows[-10:-5]
    recent_lateral = mean(recent, "lateral")
    previous_lateral = mean(previous, "lateral")
    baseline_contact = median(warm_rows, "contact_fraction")
    baseline_cut = median(warm_rows, "cut_error")
    recent_contact = mean(recent, "contact_fraction")
    previous_contact = mean(previous, "contact_fraction")
    recent_cut = mean(recent, "cut_error")
    previous_cut = mean(previous, "cut_error")

    flags: list[str] = []
    causes: list[str] = []
    if baseline is not None and latest.get("iteration", 0) > warmup:
        if latest.get("lateral", 0.0) > baseline * 1.5:
            flags.append("lateral_drift_watch")
        if latest.get("lateral", 0.0) > baseline * 3.0:
            flags.append("divergence_guard_threshold")
    if (
        recent_lateral is not None
        and previous_lateral is not None
        and recent_lateral > previous_lateral * 1.15
        and recent_lateral - previous_lateral > 0.01
    ):
        flags.append("recent_regression")
    if baseline_contact is not None and latest["iteration"] > warmup:
        if (
            recent_contact is not None
            and recent_contact < baseline_contact - 0.15
            and (
                previous_contact is None
                or recent_contact < previous_contact - 0.10
            )
        ):
            flags.append("contact_regression")
    if baseline_cut is not None and latest["iteration"] > warmup:
        if (
            recent_cut is not None
            and recent_cut > baseline_cut + 0.05
            and (
                previous_cut is None
                or recent_cut > previous_cut + 0.03
            )
        ):
            flags.append("cut_quality_regression")

    if flags:
        reward_lateral = mean(recent, "reward_lateral")
        reward_contact = mean(recent, "reward_contact")
        reward_cut = mean(recent, "reward_cut")
        old_reward_lateral = mean(previous, "reward_lateral")
        old_reward_contact = mean(previous, "reward_contact")
        old_reward_cut = mean(previous, "reward_cut")
        recent_rejected = mean(recent, "update_rejected")
        recent_early_stop = mean(recent, "kl_early_stop")
        if (recent_rejected is not None and recent_rejected >= 0.5) or (
            recent_early_stop is not None and recent_early_stop >= 0.5
        ):
            causes.append(
                "PPO actor updates were mostly rejected or KL-early-stopped; "
                "this excursion is a stochastic rollout/plant-basin event, "
                "not evidence that the actor weights moved in that window"
            )
        if (
            (recent_rejected is None or recent_rejected < 0.5)
            and (recent_early_stop is None or recent_early_stop < 0.5)
            and
            reward_lateral is not None
            and old_reward_lateral is not None
            and reward_contact is not None
            and old_reward_contact is not None
            and reward_cut is not None
            and old_reward_cut is not None
            and reward_lateral < old_reward_lateral - 0.15
            and reward_contact > old_reward_contact + 0.15
            and reward_cut > old_reward_cut + 0.15
        ):
            causes.append(
                "lateral objective is being traded for contact/cut-height reward; "
                "this matches the known saturation drift pattern"
            )

        if (
            "contact_regression" in flags
            or "cut_quality_regression" in flags
        ) and "recent_regression" not in flags:
            causes.append(
                "contact/cut quality regressed while lateral tracking stayed "
                "near baseline; PPO is trading safety/task terms for the lateral "
                "reward, so the filtered demonstration anchor is indicated"
            )

        ev = latest.get("explained_variance")
        if ev is not None:
            if ev < 0.3:
                causes.append("critic health is poor (explained variance < 0.3)")
            elif ev > 0.7:
                causes.append(
                    "critic explains returns (explained variance > 0.7); "
                    "critic failure is unlikely to be the primary cause"
                )
        std = latest.get("action_std")
        if std is not None and std >= 0.149:
            causes.append(
                "action std is at the 0.15 clamp; exploration inflation is not "
                "the cause"
            )
        causes.append(
            "observation normalizer is configured frozen; input-stat drift is "
            "not expected"
        )

    return {
        "state": "alert" if flags else ("warmup" if latest["iteration"] <= warmup else "stable"),
        "iteration": int(latest["iteration"]),
        "baseline_lateral": baseline,
        "baseline_contact_fraction": baseline_contact,
        "baseline_cut_error": baseline_cut,
        "latest": latest,
        "recent_lateral": recent_lateral,
        "previous_lateral": previous_lateral,
        "recent_contact_fraction": recent_contact,
        "previous_contact_fraction": previous_contact,
        "recent_cut_error": recent_cut,
        "previous_cut_error": previous_cut,
        "flags": flags,
        "causes": causes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=DIR",
        help="run directory; repeat for ROM and ID/FD",
    )
    parser.add_argument("--warmup", type=int, default=150)
    parser.add_argument("--poll", type=int, default=60)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    runs: list[tuple[str, Path]] = []
    for item in args.run:
        label, separator, directory = item.partition("=")
        if not separator or not label or not directory:
            parser.error(f"--run must be LABEL=DIR, got {item!r}")
        runs.append((label, Path(directory)))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    seen_alerts: set[tuple[str, int, tuple[str, ...]]] = set()
    history = output.open("a", encoding="utf-8")
    while True:
        try:
            timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            for label, directory in runs:
                report = explain(read_rows(directory / "train.log"), args.warmup)
                report = {"timestamp": timestamp, "run": label, **report}
                line = json.dumps(report, sort_keys=True)
                print(line, flush=True)
                history.write(line + "\n")
                history.flush()
                key = (
                    label,
                    int(report.get("iteration", -1)),
                    tuple(report.get("flags", [])),
                )
                if report.get("flags") and key not in seen_alerts:
                    (directory / "drift_analysis.json").write_text(
                        json.dumps(report, indent=2, sort_keys=True) + "\n"
                    )
                    seen_alerts.add(key)
            time.sleep(max(args.poll, 5))
        except KeyboardInterrupt:
            break
    history.close()


if __name__ == "__main__":
    main()
