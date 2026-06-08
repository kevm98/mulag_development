from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


parser = argparse.ArgumentParser(description="Analyze actuator position-target transmission CSV without IsaacLab.")
parser.add_argument("--csv", type=str, required=True, help="Path to actuator_position_target_transmission.csv.")
parser.add_argument("--reference_body", type=str, default="Frontrahmen", help="Reference body for relative body motion.")
parser.add_argument("--motion_threshold", type=float, default=1.0e-5, help="Absolute motion floor.")
parser.add_argument("--min_delta_from_zero", type=float, default=1.0e-4, help="Minimum test-minus-zero motion.")
parser.add_argument(
    "--opposite_sign_required",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Require positive/negative target offsets to have opposite-signed effects.",
)


def _to_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_name(name: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in name).strip("_")


def _strip_field_name(field_name: str, prefix: str, suffix: str) -> str:
    return field_name[len(prefix) : -len(suffix)]


def _last_rows_by_test(csv_path: Path) -> list[dict[str, str]]:
    latest: dict[tuple[float, str], dict[str, str]] = {}
    with csv_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            key = (_to_float(row.get("position_offset", "nan")), row.get("test_name", ""))
            latest[key] = row
    return [latest[key] for key in sorted(latest.keys(), key=lambda item: (item[0], item[1]))]


def _group_rows(rows: list[dict[str, str]]) -> dict[float, dict[str, dict[str, str]]]:
    grouped: dict[float, dict[str, dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(_to_float(row.get("position_offset", "nan")), {})[row.get("test_name", "")] = row
    return grouped


def _infer_actuator_names(rows: list[dict[str, str]]) -> list[str]:
    names: list[str] = []
    for row in rows:
        name = row.get("applied_actuator_name", "")
        if name and name != "none" and name not in names:
            names.append(name)
    for row in rows:
        test_name = row.get("test_name", "")
        for suffix in ("_pos_target", "_neg_target"):
            if test_name.endswith(suffix):
                name = test_name[: -len(suffix)]
                if name not in names:
                    names.append(name)
    return names


def _actuator_delta(row: dict[str, str] | None, actuator_name: str) -> float:
    if row is None:
        return math.nan
    return _to_float(row.get(f"actuator_{_safe_name(actuator_name)}_delta_from_start"))


def _mechanical_deltas(row: dict[str, str] | None) -> dict[str, float]:
    if row is None:
        return {}
    suffix = "_delta_from_start"
    values = {}
    for field_name, text in row.items():
        if field_name.startswith("mech_") and field_name.endswith(suffix):
            values[_strip_field_name(field_name, "mech_", suffix)] = _to_float(text)
    return values


def _body_delta_norms(row: dict[str, str] | None, reference_body: str | None = None) -> dict[str, float]:
    if row is None:
        return {}
    suffix = "_translation_delta_norm"
    values = {}
    for field_name, text in row.items():
        if field_name.startswith("body_") and field_name.endswith(suffix):
            body_name = _strip_field_name(field_name, "body_", suffix)
            values[body_name] = _to_float(text)
    if not reference_body:
        return values
    ref_safe = _safe_name(reference_body)
    ref_vector = (
        _to_float(row.get(f"body_{ref_safe}_dx_from_start")),
        _to_float(row.get(f"body_{ref_safe}_dy_from_start")),
        _to_float(row.get(f"body_{ref_safe}_dz_from_start")),
    )
    if not all(math.isfinite(value) for value in ref_vector):
        return values
    relative = {}
    for body_name in values:
        safe = _safe_name(body_name)
        vector = (
            _to_float(row.get(f"body_{safe}_dx_from_start")),
            _to_float(row.get(f"body_{safe}_dy_from_start")),
            _to_float(row.get(f"body_{safe}_dz_from_start")),
        )
        if all(math.isfinite(value) for value in vector):
            dx = vector[0] - ref_vector[0]
            dy = vector[1] - ref_vector[1]
            dz = vector[2] - ref_vector[2]
            relative[body_name] = math.sqrt(dx * dx + dy * dy + dz * dz)
        else:
            relative[body_name] = values[body_name]
    return relative


def _expected_mechanical_keywords(actuator_name: str) -> tuple[str, ...]:
    if actuator_name == "Drehzapfen_joint":
        return ("Drehzapfen",)
    return (actuator_name.replace("_joint", ""),)


def _is_expected(name: str, keywords: tuple[str, ...]) -> bool:
    lowered = name.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def _relative_delta(
    test_deltas: dict[str, float],
    zero_deltas: dict[str, float],
    keywords: tuple[str, ...] | None = None,
    *,
    invert: bool = False,
) -> tuple[str, float, float]:
    best_name = "none"
    best_relative = 0.0
    best_raw = 0.0
    for name, value in test_deltas.items():
        if not math.isfinite(value):
            continue
        matches = True if keywords is None else _is_expected(name, keywords)
        if invert:
            matches = not matches
        if not matches:
            continue
        relative = value - zero_deltas.get(name, 0.0)
        if abs(relative) > abs(best_relative):
            best_name = name
            best_relative = relative
            best_raw = value
    return best_name, best_relative, best_raw


def _separated(value: float, relative_value: float, args: argparse.Namespace) -> bool:
    if not math.isfinite(value) or not math.isfinite(relative_value):
        return False
    return abs(relative_value) >= max(args.motion_threshold, args.min_delta_from_zero)


def _sign_ok(pos_value: float, neg_value: float, *, required: bool) -> bool:
    if not required:
        return True
    return math.isfinite(pos_value) and math.isfinite(neg_value) and pos_value * neg_value < 0.0


def _max_delta(deltas: dict[str, float]) -> tuple[str, float]:
    best_name = "none"
    best_value = 0.0
    for name, value in deltas.items():
        if math.isfinite(value) and abs(value) > abs(best_value):
            best_name = name
            best_value = value
    return best_name, best_value


def _classify(
    *,
    offset: float,
    actuator_name: str,
    zero: dict[str, str],
    pos: dict[str, str] | None,
    neg: dict[str, str] | None,
    args: argparse.Namespace,
) -> dict[str, str | float]:
    if pos is None or neg is None:
        return {
            "position_offset": offset,
            "actuator_name": actuator_name,
            "classification": "WARN",
            "conclusion": "WARN missing pos/neg rows",
            "recommendation_key": "incomplete",
        }

    zero_act = _actuator_delta(zero, actuator_name)
    pos_act = _actuator_delta(pos, actuator_name)
    neg_act = _actuator_delta(neg, actuator_name)
    pos_act_minus_zero = pos_act - zero_act
    neg_act_minus_zero = neg_act - zero_act

    zero_mech = _mechanical_deltas(zero)
    pos_mech = _mechanical_deltas(pos)
    neg_mech = _mechanical_deltas(neg)
    expected = _expected_mechanical_keywords(actuator_name)
    pos_expected_name, pos_expected_minus_zero, pos_expected_raw = _relative_delta(pos_mech, zero_mech, expected)
    neg_expected_name, neg_expected_minus_zero, neg_expected_raw = _relative_delta(neg_mech, zero_mech, expected)
    pos_unexpected_name, pos_unexpected_minus_zero, _ = _relative_delta(pos_mech, zero_mech, expected, invert=True)
    neg_unexpected_name, neg_unexpected_minus_zero, _ = _relative_delta(neg_mech, zero_mech, expected, invert=True)

    zero_body = _body_delta_norms(zero, args.reference_body)
    pos_body = _body_delta_norms(pos, args.reference_body)
    neg_body = _body_delta_norms(neg, args.reference_body)
    pos_largest_body, pos_body_raw = _max_delta(pos_body)
    neg_largest_body, neg_body_raw = _max_delta(neg_body)
    pos_body_minus_zero = pos_body_raw - zero_body.get(pos_largest_body, 0.0)
    neg_body_minus_zero = neg_body_raw - zero_body.get(neg_largest_body, 0.0)

    actuator_ok = _separated(pos_act, pos_act_minus_zero, args) and _separated(neg_act, neg_act_minus_zero, args)
    actuator_sign_ok = _sign_ok(pos_act_minus_zero, neg_act_minus_zero, required=args.opposite_sign_required)
    mech_ok = _separated(pos_expected_raw, pos_expected_minus_zero, args) and _separated(
        neg_expected_raw, neg_expected_minus_zero, args
    )
    mech_sign_ok = _sign_ok(pos_expected_minus_zero, neg_expected_minus_zero, required=args.opposite_sign_required)
    expected_abs = max(abs(pos_expected_minus_zero), abs(neg_expected_minus_zero))
    unexpected_abs = max(abs(pos_unexpected_minus_zero), abs(neg_unexpected_minus_zero))
    expected_dominates = expected_abs > max(unexpected_abs, args.min_delta_from_zero)
    heavily_clamped = (
        abs(_to_float(pos.get("target_position")) - _to_float(pos.get("clamped_target_position"))) > 1.0e-8
        or abs(_to_float(neg.get("target_position")) - _to_float(neg.get("clamped_target_position"))) > 1.0e-8
    )
    done_frac = max(_to_float(pos.get("done_frac"), 0.0), _to_float(neg.get("done_frac"), 0.0))

    if actuator_ok and actuator_sign_ok and mech_ok and mech_sign_ok and expected_dominates and done_frac == 0.0:
        classification = "PASS"
        conclusion = "PASS position target validates USD actuator-to-arm linkage"
        recommendation_key = "pass"
    elif not actuator_ok or not mech_ok:
        classification = "FAIL"
        conclusion = "FAIL position target did not transmit actuator stroke to expected arm family"
        recommendation_key = "linkage"
    elif unexpected_abs > expected_abs:
        classification = "WARN"
        conclusion = "WARN unexpected mechanical family dominates"
        recommendation_key = "unexpected"
    elif heavily_clamped:
        classification = "WARN"
        conclusion = "WARN actuator target was clamped; rerun with smaller offsets"
        recommendation_key = "clamped"
    else:
        classification = "WARN"
        conclusion = "WARN position target motion observed but not cleanly verified"
        recommendation_key = "inconclusive"

    return {
        "position_offset": offset,
        "actuator_name": actuator_name,
        "actuator_pos_delta_minus_zero": pos_act_minus_zero,
        "actuator_neg_delta_minus_zero": neg_act_minus_zero,
        "expected_mech_pos_delta_minus_zero": pos_expected_minus_zero,
        "expected_mech_neg_delta_minus_zero": neg_expected_minus_zero,
        "largest_expected_mech_pos": pos_expected_name,
        "largest_expected_mech_neg": neg_expected_name,
        "largest_unexpected_mech_pos": pos_unexpected_name,
        "largest_unexpected_mech_neg": neg_unexpected_name,
        "body_pos_delta_minus_zero": pos_body_minus_zero,
        "body_neg_delta_minus_zero": neg_body_minus_zero,
        "largest_body_pos": pos_largest_body,
        "largest_body_neg": neg_largest_body,
        "target_clamped": "yes" if heavily_clamped else "no",
        "done_frac": done_frac,
        "classification": classification,
        "conclusion": conclusion,
        "recommendation_key": recommendation_key,
    }


def _print_table(rows: list[dict[str, str | float]]) -> None:
    fields = [
        "position_offset",
        "actuator_name",
        "actuator_pos_delta_minus_zero",
        "actuator_neg_delta_minus_zero",
        "expected_mech_pos_delta_minus_zero",
        "expected_mech_neg_delta_minus_zero",
        "largest_unexpected_mech_pos",
        "largest_unexpected_mech_neg",
        "body_pos_delta_minus_zero",
        "body_neg_delta_minus_zero",
        "target_clamped",
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


def _print_recommendations(rows: list[dict[str, str | float]]) -> None:
    keys = {str(row.get("recommendation_key", "")) for row in rows}
    print("[INFO] Recommended next action:")
    if "pass" in keys:
        print("- Position target passes: USD linkage can transmit cylinder stroke through authored drive interface.")
        print("- Force actuation failed because raw effort is not the right interface yet or needs deeper actuator setup.")
    if "linkage" in keys:
        print("- Position target fails: USD linkage/constraints are not transmitting actuator stroke to expected arm motion.")
        print("- Repair or inspect asset/linkage constraints before learned control or force-actuated config work.")
    if "unexpected" in keys:
        print("- Unexpected mechanical family dominates: inspect USD coupling/linkage and joint naming/families.")
    if "clamped" in keys:
        print("- Target clamped: rerun with smaller --position_offsets.")


def main() -> None:
    args = parser.parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    rows = _last_rows_by_test(csv_path)
    if not rows:
        raise RuntimeError(f"No data rows found in {csv_path}")
    grouped = _group_rows(rows)
    actuator_names = _infer_actuator_names(rows)
    analyzed: list[dict[str, str | float]] = []
    for offset, tests in grouped.items():
        zero = tests.get("zero")
        if zero is None:
            print(f"[WARN] offset={offset:g}: missing zero row; skipping.")
            continue
        for actuator_name in actuator_names:
            analyzed.append(
                _classify(
                    offset=offset,
                    actuator_name=actuator_name,
                    zero=zero,
                    pos=tests.get(f"{actuator_name}_pos_target"),
                    neg=tests.get(f"{actuator_name}_neg_target"),
                    args=args,
                )
            )
    print(f"[INFO] Analyzing CSV: {csv_path}")
    print(f"[INFO] reference_body={args.reference_body}")
    print("[INFO] Zero-drift-relative position-target summary:")
    _print_table(analyzed)
    if any(row.get("classification") == "PASS" for row in analyzed):
        print("[PASS] position target validates USD actuator-to-arm linkage")
    elif any(row.get("classification") == "WARN" for row in analyzed):
        print("[WARN] position target motion observed but not cleanly verified")
    else:
        print("[FAIL] position target did not transmit actuator stroke to expected arm family")
    _print_recommendations(analyzed)


if __name__ == "__main__":
    main()
