from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


ACTUATOR_NAMES = (
    "Drehzapfen_joint",
    "Ausleger_I_joint",
    "Ausleger_II_joint",
    "Messerkopf_Schwenk_joint",
)

parser = argparse.ArgumentParser(description="Analyze ROM position-target integration CSV without IsaacLab.")
parser.add_argument("--csv", type=str, required=True, help="Path to rom_position_target_integration.csv.")
parser.add_argument("--reference_body", type=str, default="Frontrahmen", help="Reference body used during logging.")
parser.add_argument("--motion_threshold", type=float, default=1.0e-5, help="Absolute motion floor.")
parser.add_argument("--min_dx_from_zero", type=float, default=1.0e-5, help="Minimum ROM dx response beyond zero drift.")
parser.add_argument(
    "--min_actuator_delta_from_zero",
    type=float,
    default=1.0e-4,
    help="Minimum actuator motion beyond zero drift.",
)
parser.add_argument(
    "--min_expected_mech_from_zero",
    type=float,
    default=1.0e-4,
    help="Minimum expected mechanical-family motion beyond zero drift.",
)
parser.add_argument(
    "--max_target_tracking_error",
    type=float,
    default=0.02,
    help="Maximum allowed actuator target tracking error.",
)
parser.add_argument(
    "--opposite_sign_required",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Require positive/negative responses to have opposite signs.",
)


def _to_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_name(name: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in name).strip("_")


def _field(prefix: str, name: str, suffix: str) -> str:
    return f"{prefix}_{_safe_name(name)}_{suffix}"


def _strip_field_name(field_name: str, prefix: str, suffix: str) -> str:
    return field_name[len(prefix) : -len(suffix)]


def expected_mechanical_keywords(actuator_name: str) -> tuple[str, ...]:
    if actuator_name == "Drehzapfen_joint":
        return ("Drehzapfen",)
    return (actuator_name.replace("_joint", ""),)


def _is_expected(name: str, actuator_name: str) -> bool:
    lowered = name.lower()
    return any(keyword.lower() in lowered for keyword in expected_mechanical_keywords(actuator_name))


def _last_rows(csv_path: Path) -> list[dict[str, str]]:
    latest: dict[tuple[float, str], dict[str, str]] = {}
    with csv_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            magnitude = _to_float(row.get("valve_magnitude"))
            if not math.isfinite(magnitude):
                magnitude = abs(_to_float(row.get("valve_command"), 0.0))
            key = (magnitude, row.get("test_name", ""))
            latest[key] = row
    return [latest[key] for key in sorted(latest.keys(), key=lambda item: (item[0], item[1]))]


def _group_by_magnitude(rows: list[dict[str, str]]) -> dict[float, dict[str, dict[str, str]]]:
    grouped: dict[float, dict[str, dict[str, str]]] = {}
    for row in rows:
        magnitude = _to_float(row.get("valve_magnitude"))
        if not math.isfinite(magnitude):
            magnitude = abs(_to_float(row.get("valve_command"), 0.0))
        grouped.setdefault(magnitude, {})[row.get("test_name", "")] = row
    return grouped


def _infer_actuator_names(rows: list[dict[str, str]]) -> list[str]:
    names = [name for name in ACTUATOR_NAMES if any(f"actuator_{_safe_name(name)}_pos" in row for row in rows)]
    if names:
        return names
    inferred: list[str] = []
    for row in rows:
        test_name = row.get("test_name", "")
        for suffix in ("_positive_valve", "_negative_valve"):
            if test_name.endswith(suffix):
                name = test_name[: -len(suffix)]
                if name != "all" and name not in inferred:
                    inferred.append(name)
    return inferred


def _mechanical_deltas(row: dict[str, str] | None) -> dict[str, float]:
    if row is None:
        return {}
    suffix = "_delta_from_start"
    values = {}
    for field_name, text in row.items():
        if field_name.startswith("mech_") and field_name.endswith(suffix):
            values[_strip_field_name(field_name, "mech_", suffix)] = _to_float(text)
    return values


def _max_relative_mech(
    row: dict[str, str] | None,
    zero: dict[str, str] | None,
    actuator_name: str,
    *,
    expected: bool,
) -> tuple[str, float]:
    row_deltas = _mechanical_deltas(row)
    zero_deltas = _mechanical_deltas(zero)
    best_name = "none"
    best_value = 0.0
    for name, value in row_deltas.items():
        matches = _is_expected(name, actuator_name)
        if expected != matches or not math.isfinite(value):
            continue
        relative = value - zero_deltas.get(name, 0.0)
        if abs(relative) > abs(best_value):
            best_name = name
            best_value = relative
    return best_name, best_value


def _signed_ok(pos_value: float, neg_value: float, required: bool) -> bool:
    if not required:
        return True
    return math.isfinite(pos_value) and math.isfinite(neg_value) and pos_value * neg_value < 0.0


def _meaningful(value: float, threshold: float, motion_threshold: float) -> bool:
    return math.isfinite(value) and abs(value) >= max(threshold, motion_threshold)


def _value(row: dict[str, str] | None, field_name: str, default: float = math.nan) -> float:
    if row is None:
        return default
    return _to_float(row.get(field_name), default)


def _classify_pair(
    *,
    magnitude: float,
    actuator_name: str,
    zero: dict[str, str] | None,
    pos: dict[str, str] | None,
    neg: dict[str, str] | None,
    args: argparse.Namespace,
) -> dict[str, str | float]:
    safe = _safe_name(actuator_name)
    if zero is None or pos is None or neg is None:
        return {
            "valve_magnitude": magnitude,
            "actuator_name": actuator_name,
            "classification": "FAIL",
            "conclusion": "FAIL missing zero/positive/negative rows",
        }

    dx_zero = _value(zero, f"rom_dx_pred_{safe}")
    dx_pos = _value(pos, f"rom_dx_pred_{safe}")
    dx_neg = _value(neg, f"rom_dx_pred_{safe}")
    dx_pos_minus_zero = dx_pos - dx_zero
    dx_neg_minus_zero = dx_neg - dx_zero

    target_delta_zero = _value(zero, f"actuator_{safe}_target_delta_from_start")
    target_delta_pos = _value(pos, f"actuator_{safe}_target_delta_from_start")
    target_delta_neg = _value(neg, f"actuator_{safe}_target_delta_from_start")
    target_delta_pos_minus_zero = target_delta_pos - target_delta_zero
    target_delta_neg_minus_zero = target_delta_neg - target_delta_zero

    actuator_zero = _value(zero, f"actuator_{safe}_delta_from_start")
    actuator_pos = _value(pos, f"actuator_{safe}_delta_from_start")
    actuator_neg = _value(neg, f"actuator_{safe}_delta_from_start")
    actuator_delta_pos_minus_zero = actuator_pos - actuator_zero
    actuator_delta_neg_minus_zero = actuator_neg - actuator_zero

    expected_pos_name, expected_mech_pos_minus_zero = _max_relative_mech(pos, zero, actuator_name, expected=True)
    expected_neg_name, expected_mech_neg_minus_zero = _max_relative_mech(neg, zero, actuator_name, expected=True)
    unexpected_pos_name, unexpected_pos = _max_relative_mech(pos, zero, actuator_name, expected=False)
    unexpected_neg_name, unexpected_neg = _max_relative_mech(neg, zero, actuator_name, expected=False)

    target_tracking_error_pos = abs(_value(pos, f"actuator_{safe}_target_error"))
    target_tracking_error_neg = abs(_value(neg, f"actuator_{safe}_target_error"))
    done_frac = max(_value(pos, "done_frac",), _value(neg, "done_frac"), _value(zero, "done_frac"))
    nan_guard = max(
        _value(pos, "rom_nan_guard_triggered", 0.0),
        _value(neg, "rom_nan_guard_triggered", 0.0),
        _value(zero, "rom_nan_guard_triggered", 0.0),
    )
    target_clamp = max(
        _value(pos, "rom_target_clamp_fraction", 0.0),
        _value(neg, "rom_target_clamp_fraction", 0.0),
        _value(zero, "rom_target_clamp_fraction", 0.0),
    )
    pressure_clamp = max(
        _value(pos, "rom_pressure_clamp_fraction", 0.0),
        _value(neg, "rom_pressure_clamp_fraction", 0.0),
        _value(zero, "rom_pressure_clamp_fraction", 0.0),
    )
    expected_abs = max(abs(expected_mech_pos_minus_zero), abs(expected_mech_neg_minus_zero))
    unexpected_abs = max(abs(unexpected_pos), abs(unexpected_neg))

    fail_reasons = []
    warn_reasons = []
    if done_frac > 0.0:
        fail_reasons.append("done_frac nonzero")
    if nan_guard > 0.0:
        fail_reasons.append("ROM NaN guard triggered")
    if not (
        _meaningful(dx_pos_minus_zero, args.min_dx_from_zero, args.motion_threshold)
        and _meaningful(dx_neg_minus_zero, args.min_dx_from_zero, args.motion_threshold)
    ):
        fail_reasons.append("ROM dx nearly zero")
    if not _signed_ok(dx_pos_minus_zero, dx_neg_minus_zero, args.opposite_sign_required):
        fail_reasons.append("ROM dx pos/neg same sign")
    if not (
        _meaningful(actuator_delta_pos_minus_zero, args.min_actuator_delta_from_zero, args.motion_threshold)
        and _meaningful(actuator_delta_neg_minus_zero, args.min_actuator_delta_from_zero, args.motion_threshold)
    ):
        fail_reasons.append("actuator motion not different from zero drift")
    if not _signed_ok(actuator_delta_pos_minus_zero, actuator_delta_neg_minus_zero, args.opposite_sign_required):
        fail_reasons.append("actuator pos/neg same sign")
    if not (
        _meaningful(expected_mech_pos_minus_zero, args.min_expected_mech_from_zero, args.motion_threshold)
        and _meaningful(expected_mech_neg_minus_zero, args.min_expected_mech_from_zero, args.motion_threshold)
    ):
        fail_reasons.append("expected mechanical family does not move beyond zero drift")

    if not _signed_ok(expected_mech_pos_minus_zero, expected_mech_neg_minus_zero, args.opposite_sign_required):
        warn_reasons.append("expected mechanical family sign is inconsistent")
    if target_tracking_error_pos > args.max_target_tracking_error or target_tracking_error_neg > args.max_target_tracking_error:
        warn_reasons.append("target tracking error high")
    if unexpected_abs > expected_abs:
        warn_reasons.append("unexpected mechanical family dominates")
    if target_clamp > 0.5 or pressure_clamp > 0.5:
        warn_reasons.append("clamping dominates")
    if abs(actuator_zero) > args.min_actuator_delta_from_zero or expected_abs == 0.0 and abs(dx_zero) > args.min_dx_from_zero:
        warn_reasons.append("zero-valve drift is large")

    if fail_reasons:
        classification = "FAIL"
        conclusion = "FAIL " + "; ".join(fail_reasons)
    elif warn_reasons:
        classification = "WARN"
        conclusion = "WARN " + "; ".join(warn_reasons)
    else:
        classification = "PASS"
        conclusion = "PASS signed zero-relative ROM position-target behavior verified"

    return {
        "valve_magnitude": magnitude,
        "actuator_name": actuator_name,
        "dx_pred_pos_minus_zero": dx_pos_minus_zero,
        "dx_pred_neg_minus_zero": dx_neg_minus_zero,
        "target_delta_pos_minus_zero": target_delta_pos_minus_zero,
        "target_delta_neg_minus_zero": target_delta_neg_minus_zero,
        "actuator_delta_pos_minus_zero": actuator_delta_pos_minus_zero,
        "actuator_delta_neg_minus_zero": actuator_delta_neg_minus_zero,
        "expected_mech_pos_minus_zero": expected_mech_pos_minus_zero,
        "expected_mech_neg_minus_zero": expected_mech_neg_minus_zero,
        "largest_expected_mech_pos": expected_pos_name,
        "largest_expected_mech_neg": expected_neg_name,
        "largest_unexpected_mech_pos": unexpected_pos_name,
        "largest_unexpected_mech_neg": unexpected_neg_name,
        "target_tracking_error_pos": target_tracking_error_pos,
        "target_tracking_error_neg": target_tracking_error_neg,
        "max_done_frac": done_frac,
        "max_nan_guard": nan_guard,
        "max_target_clamp_fraction": target_clamp,
        "max_pressure_clamp_fraction": pressure_clamp,
        "classification": classification,
        "conclusion": conclusion,
    }


def analyze_rows(rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, str | float]]:
    grouped = _group_by_magnitude(rows)
    actuator_names = _infer_actuator_names(rows)
    results: list[dict[str, str | float]] = []
    for magnitude, tests in grouped.items():
        if magnitude <= 0.0:
            continue
        zero = tests.get("zero")
        for actuator_name in actuator_names:
            results.append(
                _classify_pair(
                    magnitude=magnitude,
                    actuator_name=actuator_name,
                    zero=zero,
                    pos=tests.get(f"{actuator_name}_positive_valve"),
                    neg=tests.get(f"{actuator_name}_negative_valve"),
                    args=args,
                )
            )
    return results


def print_table(rows: list[dict[str, str | float]]) -> None:
    fields = [
        "valve_magnitude",
        "actuator_name",
        "dx_pred_pos_minus_zero",
        "dx_pred_neg_minus_zero",
        "actuator_delta_pos_minus_zero",
        "actuator_delta_neg_minus_zero",
        "expected_mech_pos_minus_zero",
        "expected_mech_neg_minus_zero",
        "target_tracking_error_pos",
        "target_tracking_error_neg",
        "classification",
        "conclusion",
    ]
    widths = {field: len(field) for field in fields}
    for row in rows:
        for field in fields:
            value = row.get(field, "")
            text = str(value) if isinstance(value, str) else f"{float(value):.6g}"
            widths[field] = max(widths[field], len(text))
    print(" | ".join(field.ljust(widths[field]) for field in fields))
    print("-+-".join("-" * widths[field] for field in fields))
    for row in rows:
        parts = []
        for field in fields:
            value = row.get(field, "")
            text = str(value) if isinstance(value, str) else f"{float(value):.6g}"
            parts.append(text.ljust(widths[field]))
        print(" | ".join(parts))


def print_global_conclusion(rows: list[dict[str, str | float]]) -> None:
    if rows and all(row.get("classification") == "PASS" for row in rows):
        print("[PASS] all actuator paths passed strict signed zero-relative ROM validation")
    elif any(row.get("classification") == "PASS" for row in rows):
        print("[WARN] some actuator paths passed, but others are inconsistent or failed")
    else:
        print("[FAIL] no actuator path has fully physically meaningful signed behavior")
    if any(row.get("classification") != "PASS" for row in rows):
        print("[INFO] Treat this as integration evidence only; ROM parameters likely need identification.")


def main() -> None:
    args = parser.parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    rows = _last_rows(csv_path)
    if not rows:
        raise RuntimeError(f"No data rows found in {csv_path}.")
    analyzed = analyze_rows(rows, args)
    print(f"[INFO] Analyzing CSV: {csv_path}")
    print(f"[INFO] reference_body={args.reference_body}")
    print_table(analyzed)
    print_global_conclusion(analyzed)


if __name__ == "__main__":
    main()
