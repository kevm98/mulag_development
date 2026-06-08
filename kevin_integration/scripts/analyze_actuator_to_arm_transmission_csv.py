from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


parser = argparse.ArgumentParser(description="Analyze actuator-to-arm transmission CSV without IsaacLab.")
parser.add_argument("--csv", type=str, required=True, help="Path to actuator_to_arm_transmission.csv.")
parser.add_argument("--motion_threshold", type=float, default=1.0e-5, help="Absolute motion floor.")
parser.add_argument(
    "--drift_margin_ratio",
    type=float,
    default=3.0,
    help="Test motion must exceed zero-drift magnitude by this ratio to count as separated.",
)
parser.add_argument(
    "--min_delta_from_zero",
    type=float,
    default=1.0e-4,
    help="Minimum absolute test-minus-zero delta to count as separated.",
)
parser.add_argument(
    "--opposite_sign_required",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Require pos/neg effects to have opposite signs.",
)
parser.add_argument(
    "--reference_body",
    type=str,
    default="Frontrahmen",
    help="Optional body name used to subtract global/base translation from body motion.",
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
            key = (_to_float(row.get("effort_value", "nan")), row.get("test_name", ""))
            latest[key] = row
    return [latest[key] for key in sorted(latest.keys(), key=lambda item: (item[0], item[1]))]


def _group_rows(rows: list[dict[str, str]]) -> dict[float, dict[str, dict[str, str]]]:
    grouped: dict[float, dict[str, dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(_to_float(row.get("effort_value", "nan")), {})[row.get("test_name", "")] = row
    return grouped


def _infer_actuator_names(rows: list[dict[str, str]]) -> list[str]:
    names: list[str] = []
    for row in rows:
        name = row.get("applied_actuator_name", "")
        if name and name != "none" and name not in names:
            names.append(name)
    if names:
        return names
    for row in rows:
        test_name = row.get("test_name", "")
        for suffix in ("_pos", "_neg"):
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

    relative_values = {}
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
            relative_values[body_name] = math.sqrt(dx * dx + dy * dy + dz * dz)
        else:
            relative_values[body_name] = values[body_name]
    return relative_values


def _expected_mechanical_keywords(actuator_name: str) -> tuple[str, ...]:
    if actuator_name == "Drehzapfen_joint":
        return ("Drehzapfen",)
    return (actuator_name.replace("_joint", ""),)


def _expected_body_keywords(actuator_name: str) -> tuple[str, ...]:
    if actuator_name == "Drehzapfen_joint":
        return ("Drehzapfen", "Ausleger_Schwenkzylinder_Kolben", "Ausleger_Schwenkzylinder_Rohr")
    return (actuator_name.replace("_joint", ""),)


def _is_expected(name: str, keywords: tuple[str, ...]) -> bool:
    lowered = name.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def _max_delta(deltas: dict[str, float], *, keywords: tuple[str, ...] | None = None, invert: bool = False) -> tuple[str, float]:
    best_name = "none"
    best_value = 0.0
    for name, value in deltas.items():
        if not math.isfinite(value):
            continue
        matches = True if keywords is None else _is_expected(name, keywords)
        if invert:
            matches = not matches
        if matches and abs(value) > abs(best_value):
            best_name = name
            best_value = value
    return best_name, best_value


def _relative_delta(
    test_deltas: dict[str, float],
    zero_deltas: dict[str, float],
    *,
    keywords: tuple[str, ...] | None = None,
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


def _relative_body_delta(
    test_deltas: dict[str, float],
    zero_deltas: dict[str, float],
    *,
    keywords: tuple[str, ...] | None = None,
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


def _separated(value: float, zero_value: float, relative_value: float, args: argparse.Namespace) -> bool:
    if not all(math.isfinite(item) for item in (value, zero_value, relative_value)):
        return False
    if abs(relative_value) < max(args.motion_threshold, args.min_delta_from_zero):
        return False
    return abs(value) >= args.drift_margin_ratio * max(abs(zero_value), args.motion_threshold)


def _sign_ok(pos_value: float, neg_value: float, *, required: bool) -> bool:
    if not required:
        return True
    return math.isfinite(pos_value) and math.isfinite(neg_value) and pos_value * neg_value < 0.0


def _base_like_body(body_name: str) -> bool:
    lowered = body_name.lower()
    return "frontrahmen" in lowered or "base" in lowered


def _classify_effort_actuator(
    *,
    effort_value: float,
    actuator_name: str,
    zero: dict[str, str],
    pos: dict[str, str] | None,
    neg: dict[str, str] | None,
    args: argparse.Namespace,
) -> dict[str, str | float]:
    expected_mech_keywords = _expected_mechanical_keywords(actuator_name)
    expected_body_keywords = _expected_body_keywords(actuator_name)
    if pos is None or neg is None:
        return {
            "effort_value": effort_value,
            "actuator_name": actuator_name,
            "classification": "WARN",
            "conclusion": "WARN missing pos/neg rows for actuator",
            "recommendation_key": "incomplete",
        }

    zero_actuator_delta = _actuator_delta(zero, actuator_name)
    pos_actuator_delta = _actuator_delta(pos, actuator_name)
    neg_actuator_delta = _actuator_delta(neg, actuator_name)
    pos_actuator_minus_zero = pos_actuator_delta - zero_actuator_delta
    neg_actuator_minus_zero = neg_actuator_delta - zero_actuator_delta

    zero_mech = _mechanical_deltas(zero)
    pos_mech = _mechanical_deltas(pos)
    neg_mech = _mechanical_deltas(neg)
    pos_expected_name, pos_expected_mech_minus_zero, pos_expected_mech_raw = _relative_delta(
        pos_mech, zero_mech, keywords=expected_mech_keywords
    )
    neg_expected_name, neg_expected_mech_minus_zero, neg_expected_mech_raw = _relative_delta(
        neg_mech, zero_mech, keywords=expected_mech_keywords
    )
    pos_unexpected_name, pos_unexpected_mech_minus_zero, _ = _relative_delta(
        pos_mech, zero_mech, keywords=expected_mech_keywords, invert=True
    )
    neg_unexpected_name, neg_unexpected_mech_minus_zero, _ = _relative_delta(
        neg_mech, zero_mech, keywords=expected_mech_keywords, invert=True
    )
    pos_largest_mech_name, pos_largest_mech_delta = _max_delta(pos_mech)
    neg_largest_mech_name, neg_largest_mech_delta = _max_delta(neg_mech)

    zero_body = _body_delta_norms(zero, args.reference_body)
    pos_body = _body_delta_norms(pos, args.reference_body)
    neg_body = _body_delta_norms(neg, args.reference_body)
    pos_expected_body_name, pos_body_minus_zero, pos_body_raw = _relative_body_delta(
        pos_body, zero_body, keywords=expected_body_keywords
    )
    neg_expected_body_name, neg_body_minus_zero, neg_body_raw = _relative_body_delta(
        neg_body, zero_body, keywords=expected_body_keywords
    )
    pos_largest_body_name, pos_largest_body_delta = _max_delta(pos_body)
    neg_largest_body_name, neg_largest_body_delta = _max_delta(neg_body)
    pos_largest_body_minus_zero = pos_largest_body_delta - zero_body.get(pos_largest_body_name, 0.0)
    neg_largest_body_minus_zero = neg_largest_body_delta - zero_body.get(neg_largest_body_name, 0.0)

    actuator_separated = _separated(
        pos_actuator_delta, zero_actuator_delta, pos_actuator_minus_zero, args
    ) and _separated(neg_actuator_delta, zero_actuator_delta, neg_actuator_minus_zero, args)
    expected_mech_separated = _separated(
        pos_expected_mech_raw, zero_mech.get(pos_expected_name, 0.0), pos_expected_mech_minus_zero, args
    ) and _separated(
        neg_expected_mech_raw, zero_mech.get(neg_expected_name, 0.0), neg_expected_mech_minus_zero, args
    )
    actuator_sign_ok = _sign_ok(pos_actuator_minus_zero, neg_actuator_minus_zero, required=args.opposite_sign_required)
    expected_sign_ok = _sign_ok(
        pos_expected_mech_minus_zero, neg_expected_mech_minus_zero, required=args.opposite_sign_required
    )
    expected_mech_abs = max(abs(pos_expected_mech_minus_zero), abs(neg_expected_mech_minus_zero))
    unexpected_mech_abs = max(abs(pos_unexpected_mech_minus_zero), abs(neg_unexpected_mech_minus_zero))
    expected_dominates = expected_mech_abs > max(unexpected_mech_abs, args.min_delta_from_zero)
    body_abs = max(abs(pos_body_minus_zero), abs(neg_body_minus_zero))
    body_raw_abs = max(abs(pos_body_raw), abs(neg_body_raw))
    body_has_relative_motion = body_abs >= args.min_delta_from_zero
    largest_body_base_like = _base_like_body(pos_largest_body_name) or _base_like_body(neg_largest_body_name)
    body_similar_to_zero = body_raw_abs > args.motion_threshold and body_abs < args.min_delta_from_zero

    if expected_mech_separated and expected_sign_ok and expected_dominates:
        classification = "PASS"
        conclusion = "PASS actuator-to-expected-arm-family transmission verified"
        recommendation_key = "pass"
    elif not actuator_separated and not expected_mech_separated:
        classification = "FAIL"
        conclusion = "FAIL actuator effort did not produce clear transmission beyond zero drift"
        recommendation_key = "actuator_drift"
    elif unexpected_mech_abs > expected_mech_abs:
        classification = "WARN"
        conclusion = "WARN largest mechanical motion is from unexpected joint family"
        recommendation_key = "unexpected_family"
    elif not actuator_sign_ok or not expected_sign_ok:
        classification = "WARN"
        conclusion = "WARN pos/neg effects do not have the required opposite signs"
        recommendation_key = "sign"
    elif body_has_relative_motion and not expected_mech_separated:
        classification = "WARN"
        conclusion = "WARN body motion alone is insufficient to verify actuator-to-arm transmission"
        recommendation_key = "body_only"
    else:
        classification = "WARN"
        conclusion = "WARN actuator-to-arm motion observed but not separable from zero drift"
        recommendation_key = "drift"

    warnings = []
    if body_similar_to_zero:
        warnings.append("[WARN] body motion exists but is also present in zero-effort test")
    if largest_body_base_like:
        warnings.append("[WARN] largest body motion may be global settling/base motion")
    if body_has_relative_motion and not expected_mech_separated:
        warnings.append("[WARN] body motion alone is insufficient to verify actuator-to-arm transmission")
    if unexpected_mech_abs > expected_mech_abs:
        warnings.append("[WARN] largest mechanical motion is from unexpected joint family")

    return {
        "effort_value": effort_value,
        "actuator_name": actuator_name,
        "zero_actuator_delta": zero_actuator_delta,
        "actuator_pos_delta_minus_zero": pos_actuator_minus_zero,
        "actuator_neg_delta_minus_zero": neg_actuator_minus_zero,
        "expected_family_mech_pos_delta_minus_zero": pos_expected_mech_minus_zero,
        "expected_family_mech_neg_delta_minus_zero": neg_expected_mech_minus_zero,
        "mechanical_pos_delta_minus_zero": pos_largest_mech_delta - zero_mech.get(pos_largest_mech_name, 0.0),
        "mechanical_neg_delta_minus_zero": neg_largest_mech_delta - zero_mech.get(neg_largest_mech_name, 0.0),
        "largest_expected_mechanical_joint_pos": pos_expected_name,
        "largest_expected_mechanical_joint_neg": neg_expected_name,
        "largest_unexpected_mechanical_joint_pos": pos_unexpected_name,
        "largest_unexpected_mechanical_joint_neg": neg_unexpected_name,
        "largest_mechanical_joint_pos": pos_largest_mech_name,
        "largest_mechanical_joint_neg": neg_largest_mech_name,
        "body_pos_delta_minus_zero": pos_body_minus_zero,
        "body_neg_delta_minus_zero": neg_body_minus_zero,
        "largest_body_pos": pos_largest_body_name,
        "largest_body_neg": neg_largest_body_name,
        "expected_body_pos": pos_expected_body_name,
        "expected_body_neg": neg_expected_body_name,
        "Drehzapfen_constraint_joint_1_pos_delta_minus_zero": pos_mech.get("Drehzapfen_constraint_joint_1", math.nan)
        - zero_mech.get("Drehzapfen_constraint_joint_1", 0.0),
        "Drehzapfen_constraint_joint_3_pos_delta_minus_zero": pos_mech.get("Drehzapfen_constraint_joint_3", math.nan)
        - zero_mech.get("Drehzapfen_constraint_joint_3", 0.0),
        "classification": classification,
        "conclusion": conclusion,
        "recommendation_key": recommendation_key,
        "warnings": " | ".join(dict.fromkeys(warnings)),
    }


def _print_table(rows: list[dict[str, str | float]]) -> None:
    fields = [
        "effort_value",
        "actuator_name",
        "actuator_pos_delta_minus_zero",
        "actuator_neg_delta_minus_zero",
        "expected_family_mech_pos_delta_minus_zero",
        "expected_family_mech_neg_delta_minus_zero",
        "largest_mechanical_joint_pos",
        "largest_mechanical_joint_neg",
        "body_pos_delta_minus_zero",
        "body_neg_delta_minus_zero",
        "largest_body_pos",
        "largest_body_neg",
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
    print("[INFO] Recommended next diagnostics:")
    if "actuator_drift" in keys or "drift" in keys:
        print("- Increase --settle_steps to 300 or 500 and rerun a shorter post-settle test.")
        print("- Test actuator position target instead of effort target.")
        print("- Consider diagnostic-only drive disabling/reducing later, but do not apply it automatically.")
    if "body_only" in keys or "unexpected_family" in keys:
        print("- Inspect USD linkage/coupling and log relative body transforms instead of only world translation.")
        print("- Subtract base/Frontrahmen motion from body motion with --reference_body Frontrahmen.")
    if "unexpected_family" in keys:
        print("- Inspect why the largest mechanical motion is outside the expected actuator family.")
    if "actuator_drift" in keys or "sign" in keys:
        print("- Inspect whether high stiffness/damping drive is holding the actuator at its target.")
        print("- Inspect linkage constraints and test prismatic position target on the actuator joint.")
    if keys == {"pass"}:
        print("- Transmission is verified for this diagnostic; continue with mapping/control checks before training.")


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
    for effort_value, tests in grouped.items():
        zero = tests.get("zero")
        if zero is None:
            print(f"[WARN] effort={effort_value:g}: missing zero-effort row; skipping drift-relative classification.")
            continue
        for actuator_name in actuator_names:
            analyzed.append(
                _classify_effort_actuator(
                    effort_value=effort_value,
                    actuator_name=actuator_name,
                    zero=zero,
                    pos=tests.get(f"{actuator_name}_pos"),
                    neg=tests.get(f"{actuator_name}_neg"),
                    args=args,
                )
            )

    print(f"[INFO] Analyzing CSV: {csv_path}")
    print(f"[INFO] motion_threshold={args.motion_threshold:g}")
    print(f"[INFO] drift_margin_ratio={args.drift_margin_ratio:g}")
    print(f"[INFO] min_delta_from_zero={args.min_delta_from_zero:g}")
    print(f"[INFO] opposite_sign_required={args.opposite_sign_required}")
    print(f"[INFO] reference_body={args.reference_body}")
    print("[INFO] Zero-drift-relative summary:")
    _print_table(analyzed)
    for row in analyzed:
        warnings = str(row.get("warnings", ""))
        if warnings:
            print(f"[INFO] effort={float(row['effort_value']):g} actuator={row['actuator_name']}: {warnings}")

    if any(row.get("classification") == "PASS" for row in analyzed):
        print("[PASS] actuator-to-expected-arm-family transmission verified")
    elif any(row.get("classification") == "WARN" for row in analyzed):
        if any(str(row.get("recommendation_key")) == "unexpected_family" for row in analyzed):
            print("[WARN] largest mechanical motion is from unexpected joint family")
        else:
            print("[WARN] actuator-to-arm motion observed but not separable from zero drift")
    else:
        print("[FAIL] actuator effort did not produce clear transmission beyond zero drift")
    _print_recommendations(analyzed)


if __name__ == "__main__":
    main()
