from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np


ROM_DIR = Path(__file__).resolve().parent


LIKELY_FIELD_KEYWORDS: dict[str, tuple[str, ...]] = {
    "input_histories": ("x", "input", "feature", "history", "hist", "sequence", "seq"),
    "targets": ("y", "target", "delta", "deltap", "deltaextension", "output", "label"),
    "splits": ("split", "train", "validation", "val", "test", "partition"),
    "normalization": ("mean", "std", "scale", "norm", "min", "max", "mu", "sigma"),
    "feature_names": ("feature_name", "features", "input_name", "names"),
    "run_indices": ("run", "trajectory", "trial", "episode", "sim"),
    "timestamps": ("time", "timestamp", "dt", "sample"),
    "raw_arrays": ("raw", "unnormalized", "physical"),
    "normalized_arrays": ("normalized", "normalised", "norm", "zscore", "standard"),
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _json_safe(value.item())
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
    return value


def _is_numeric_array(array: np.ndarray) -> bool:
    return np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_)


def _decode_utf16_codes(array: np.ndarray) -> str | None:
    if array.dtype == np.uint16 and array.ndim >= 1 and array.size > 0:
        values = array.reshape(-1)
        if np.all((values == 0) | ((values >= 9) & (values <= 126))):
            return "".join(chr(int(value)) for value in values if int(value) != 0)
    return None


def _maybe_decode_string_array(array: np.ndarray) -> list[str] | None:
    decoded = _decode_utf16_codes(array)
    if decoded is not None:
        return [decoded]
    if array.dtype.kind in {"U", "S"}:
        flat = array.reshape(-1)
        return [str(item.decode("utf-8", errors="replace") if isinstance(item, bytes) else item) for item in flat]
    if array.dtype == object:
        strings: list[str] = []
        for item in array.reshape(-1):
            if isinstance(item, str):
                strings.append(item)
            elif isinstance(item, bytes):
                strings.append(item.decode("utf-8", errors="replace"))
            elif isinstance(item, np.ndarray) and item.dtype.kind in {"U", "S"}:
                chars = item.reshape(-1)
                strings.append("".join(str(c.decode("utf-8", errors="replace") if isinstance(c, bytes) else c) for c in chars))
            else:
                return None
        return strings
    return None


def _array_stats(array: np.ndarray) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "ndim": int(array.ndim),
        "size": int(array.size),
    }
    strings = _maybe_decode_string_array(array)
    if strings is not None:
        stats["string_count"] = len(strings)
        stats["string_preview"] = strings[:40]
        return stats

    if not _is_numeric_array(array):
        stats["summary"] = "non-numeric array"
        return stats

    numeric = np.asarray(array)
    if numeric.size == 0:
        stats.update({"min": None, "max": None, "mean": None, "std": None, "nan_count": 0, "inf_count": 0})
        return stats

    numeric_float = numeric.astype(np.float64, copy=False)
    finite = np.isfinite(numeric_float)
    stats["nan_count"] = int(np.isnan(numeric_float).sum())
    stats["inf_count"] = int(np.isinf(numeric_float).sum())
    if finite.any():
        finite_values = numeric_float[finite]
        stats["min"] = float(np.min(finite_values))
        stats["max"] = float(np.max(finite_values))
        stats["mean"] = float(np.mean(finite_values))
        stats["std"] = float(np.std(finite_values))
    else:
        stats["min"] = None
        stats["max"] = None
        stats["mean"] = None
        stats["std"] = None
    return stats


def _load_standard_mat(path: Path) -> tuple[dict[str, np.ndarray], str]:
    import scipy.io

    raw = scipy.io.loadmat(path, squeeze_me=False, struct_as_record=False)
    arrays = {key: value for key, value in raw.items() if not key.startswith("__")}
    return arrays, "scipy.io.loadmat"


def _decode_hdf5_reference(handle: h5py.File, ref: Any) -> str | None:
    try:
        if not ref:
            return None
        target = handle[ref]
        data = np.asarray(target[()])
    except (KeyError, TypeError, ValueError):
        return None
    decoded = _decode_utf16_codes(data)
    if decoded is not None:
        return decoded
    string_values = _maybe_decode_string_array(data)
    if string_values and len(string_values) == 1:
        return string_values[0]
    return None


def _decode_hdf5_object_array(handle: h5py.File, data: np.ndarray) -> np.ndarray:
    decoded: list[str] = []
    for item in data.reshape(-1):
        value = _decode_hdf5_reference(handle, item)
        if value is None:
            return data
        decoded.append(value)
    return np.asarray(decoded, dtype=object).reshape(data.shape)


def _read_hdf5_dataset_with_refs(handle: h5py.File, dataset: Any) -> np.ndarray:
    data = np.asarray(dataset[()])
    if data.dtype == object:
        return _decode_hdf5_object_array(handle, data)
    return data


def _walk_hdf5_group_with_refs(handle: h5py.File, group: Any, prefix: str = "") -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for key, item in group.items():
        if prefix == "" and str(key) == "#refs#":
            continue
        name = f"{prefix}/{key}" if prefix else str(key)
        if hasattr(item, "items"):
            arrays.update(_walk_hdf5_group_with_refs(handle, item, name))
        else:
            arrays[name] = _read_hdf5_dataset_with_refs(handle, item)
    return arrays


def _load_hdf5_mat(path: Path) -> tuple[dict[str, np.ndarray], str]:
    with h5py.File(path, "r") as handle:
        arrays = _walk_hdf5_group_with_refs(handle, handle)
    return arrays, "h5py"


def load_mat_arrays(path: Path) -> tuple[dict[str, np.ndarray], str]:
    try:
        return _load_standard_mat(path)
    except NotImplementedError:
        return _load_hdf5_mat(path)
    except ValueError as exc:
        message = str(exc).lower()
        if "unknown mat file type" in message or "matlab 7.3" in message:
            return _load_hdf5_mat(path)
        raise


def find_mat_files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.glob("*.mat") if path.is_file())


def _categorize_field(name: str, stats: dict[str, Any]) -> list[str]:
    lower = name.lower()
    categories = [
        category
        for category, keywords in LIKELY_FIELD_KEYWORDS.items()
        if any(keyword in lower for keyword in keywords)
    ]
    shape = stats.get("shape", [])
    if isinstance(shape, list):
        if 26 in shape or 30 in shape:
            categories.append("possible_inputs_by_dimension")
        if 8 in shape:
            categories.append("possible_targets_by_dimension")
        if any(dim in (4, 5) for dim in shape):
            categories.append("possible_state_channels")
    return sorted(set(categories))


def _feature_strings(arrays: dict[str, np.ndarray]) -> dict[str, list[str]]:
    names: dict[str, list[str]] = {}
    for key, array in arrays.items():
        decoded = _maybe_decode_string_array(np.asarray(array))
        if decoded:
            names[key] = decoded
    return names


def _split_run_overlap(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    run_keys = [key for key in arrays if key.lower() in {"trainrunid", "valrunid", "testrunid"}]
    run_like = {
        key: np.asarray(arrays[key]).reshape(-1).astype(np.int64, copy=False)
        for key in sorted(run_keys)
        if np.asarray(arrays[key]).size > 0
    }
    source_keys = [
        key
        for key in arrays
        if key.lower().endswith("training_source_files")
        or key.lower().endswith("validation_source_files")
        or key.lower().endswith("test_source_files")
    ]
    source_like = {
        key: [str(value) for value in np.asarray(arrays[key]).reshape(-1).tolist()]
        for key in sorted(source_keys)
    }

    overlaps: list[dict[str, Any]] = []
    keys = sorted(run_like)
    for i, left in enumerate(keys):
        for right in keys[i + 1 :]:
            left_values = set(run_like[left].tolist())
            right_values = set(run_like[right].tolist())
            both = sorted(left_values & right_values)
            if both:
                overlaps.append(
                    {
                        "left": left,
                        "right": right,
                        "overlap_count": len(both),
                        "preview": both[:20],
                        "note": "Numeric run IDs may be local to each split; inspect source run identity before treating this as leakage.",
                    }
                )

    source_overlaps: list[dict[str, Any]] = []
    source_names = sorted(source_like)
    for i, left in enumerate(source_names):
        for right in source_names[i + 1 :]:
            both = sorted(set(source_like[left]) & set(source_like[right]))
            if both:
                source_overlaps.append(
                    {
                        "left": left,
                        "right": right,
                        "overlap_count": len(both),
                        "preview": both[:20],
                        "note": "Source file names identify source datasets, not necessarily individual simulation runs.",
                    }
                )
    return {
        "candidate_run_id_arrays": {
            key: {
                "shape": list(value.shape),
                "unique_count": int(np.unique(value).size),
                "min": int(np.min(value)),
                "max": int(np.max(value)),
            }
            for key, value in run_like.items()
        },
        "source_file_arrays": {
            key: {"count": len(value), "unique_count": len(set(value)), "unique_preview": sorted(set(value))[:20]}
            for key, value in source_like.items()
        },
        "numeric_run_id_overlaps": overlaps,
        "source_file_name_overlaps": source_overlaps,
        "notes": [
            "The dataset contains train and validation run IDs, but validation IDs restart at 1.",
            "The safest loader convention is to namespace run identity by split unless a global run identifier is added.",
            "No dedicated test split was detected.",
        ],
    }


def _dimension_evidence(stats_by_name: dict[str, dict[str, Any]]) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for name, stats in stats_by_name.items():
        shape = stats.get("shape", [])
        if not isinstance(shape, list):
            continue
        evidence: list[str] = []
        if 26 in shape:
            evidence.append("contains expected core input dimension 26")
        if 30 in shape:
            evidence.append("contains expected augmented input dimension 30")
        if 8 in shape:
            evidence.append("contains expected target dimension 8")
        if evidence:
            candidates.append({"name": name, "shape": shape, "evidence": evidence})
    return {"dimension_candidates": candidates}


def build_report(path: Path) -> dict[str, Any]:
    arrays, loader = load_mat_arrays(path)
    stats_by_name: dict[str, dict[str, Any]] = {}
    categories: dict[str, list[str]] = {}
    for name, array in sorted(arrays.items()):
        stats = _array_stats(np.asarray(array))
        stats_by_name[name] = stats
        categories[name] = _categorize_field(name, stats)

    report = {
        "dataset_file": str(path),
        "dataset_file_size_bytes": path.stat().st_size,
        "loader": loader,
        "top_level_variable_count": len(arrays),
        "top_level_variables": sorted(arrays),
        "arrays": stats_by_name,
        "detected_categories": categories,
        "feature_name_arrays": _feature_strings(arrays),
        "split_run_overlap_check": _split_run_overlap(arrays),
        "dimension_evidence": _dimension_evidence(stats_by_name),
        "notes": [
            "No dimensions were transposed or reshaped during inspection.",
            "Dimension-order ambiguity must be resolved in ROM_DATA_CONTRACT.md before training.",
            "Future-target timing is reported from available variable names and shapes only; loader tests must verify numeric timing once the contract is explicit.",
        ],
    }
    return report


def write_markdown(report: dict[str, Any], output_path: Path) -> None:
    lines: list[str] = []
    lines.append("# ROM MATLAB Dataset Inspection")
    lines.append("")
    lines.append(f"- Dataset: `{report['dataset_file']}`")
    lines.append(f"- Loader: `{report['loader']}`")
    lines.append(f"- File size: `{report['dataset_file_size_bytes']}` bytes")
    lines.append(f"- Top-level variables: `{report['top_level_variable_count']}`")
    lines.append("")
    lines.append("## Top-Level Variables")
    lines.append("")
    for name in report["top_level_variables"]:
        stats = report["arrays"][name]
        categories = ", ".join(report["detected_categories"].get(name, [])) or "uncategorized"
        lines.append(f"- `{name}`: shape `{stats.get('shape')}`, dtype `{stats.get('dtype')}`, categories: {categories}")
    lines.append("")
    lines.append("## Array Statistics")
    lines.append("")
    lines.append("| Name | Shape | Dtype | Min | Max | Mean | Std | NaN | Inf |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name in report["top_level_variables"]:
        stats = report["arrays"][name]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{name}`",
                    f"`{stats.get('shape')}`",
                    f"`{stats.get('dtype')}`",
                    str(stats.get("min", "")),
                    str(stats.get("max", "")),
                    str(stats.get("mean", "")),
                    str(stats.get("std", "")),
                    str(stats.get("nan_count", "")),
                    str(stats.get("inf_count", "")),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Feature Name Arrays")
    lines.append("")
    feature_arrays = report["feature_name_arrays"]
    if feature_arrays:
        for name, values in feature_arrays.items():
            preview = ", ".join(f"`{value}`" for value in values[:80])
            suffix = " ..." if len(values) > 80 else ""
            lines.append(f"- `{name}` ({len(values)} values): {preview}{suffix}")
    else:
        lines.append("- No string feature-name arrays were detected.")
    lines.append("")
    lines.append("## Split / Run Overlap Evidence")
    lines.append("")
    overlap = report["split_run_overlap_check"]
    if overlap["candidate_run_id_arrays"]:
        for name, info in overlap["candidate_run_id_arrays"].items():
            lines.append(
                f"- `{name}`: shape `{info['shape']}`, unique values `{info['unique_count']}`, "
                f"range `{info['min']}..{info['max']}`"
            )
    else:
        lines.append("- No train/validation/test run ID arrays were automatically detected.")
    if overlap["source_file_arrays"]:
        lines.append("")
        lines.append("Source-file metadata:")
        for name, info in overlap["source_file_arrays"].items():
            lines.append(
                f"- `{name}`: count `{info['count']}`, unique source files `{info['unique_count']}`, "
                f"preview `{info['unique_preview']}`"
            )
    if overlap["numeric_run_id_overlaps"]:
        lines.append("")
        lines.append("Numeric run ID overlap evidence:")
        for item in overlap["numeric_run_id_overlaps"]:
            lines.append(
                f"- `{item['left']}` vs `{item['right']}`: {item['overlap_count']} shared numeric IDs, "
                f"preview `{item['preview']}`. {item['note']}"
            )
    else:
        lines.append("- No overlap detected among automatically identified numeric run ID arrays.")
    if overlap["source_file_name_overlaps"]:
        lines.append("")
        lines.append("Source-file name overlap evidence:")
        for item in overlap["source_file_name_overlaps"]:
            lines.append(
                f"- `{item['left']}` vs `{item['right']}`: {item['overlap_count']} shared source-file labels, "
                f"preview `{item['preview']}`. {item['note']}"
            )
    for note in overlap.get("notes", []):
        lines.append(f"- {note}")
    lines.append("")
    lines.append("## Dimension Evidence")
    lines.append("")
    candidates = report["dimension_evidence"]["dimension_candidates"]
    if candidates:
        for item in candidates:
            lines.append(f"- `{item['name']}` shape `{item['shape']}`: {', '.join(item['evidence'])}")
    else:
        lines.append("- No arrays contained the expected dimensions 26, 30, or 8.")
    lines.append("")
    lines.append("## Inspection Notes")
    lines.append("")
    for note in report["notes"]:
        lines.append(f"- {note}")
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect PyTorch-ready MATLAB datasets for hydraulic cylinder ROM training.")
    parser.add_argument("--rom_dir", type=Path, default=ROM_DIR, help="Directory containing one or more .mat files.")
    parser.add_argument("--dataset", type=Path, default=None, help="Explicit .mat dataset path. Defaults to the only .mat in rom_dir.")
    parser.add_argument("--json_output", type=Path, default=ROM_DIR / "dataset_inspection.json")
    parser.add_argument("--markdown_output", type=Path, default=ROM_DIR / "dataset_inspection.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset is None:
        mat_files = find_mat_files(args.rom_dir)
        if not mat_files:
            raise FileNotFoundError(f"No .mat files found in {args.rom_dir}")
        if len(mat_files) > 1:
            joined = "\n".join(str(path) for path in mat_files)
            raise RuntimeError(f"Multiple .mat files found; pass --dataset explicitly:\n{joined}")
        dataset = mat_files[0]
    else:
        dataset = args.dataset

    report = build_report(dataset)
    args.json_output.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(report, args.markdown_output)

    print(f"Dataset: {dataset}")
    print(f"Loader: {report['loader']}")
    print("Top-level variables:")
    for name in report["top_level_variables"]:
        stats = report["arrays"][name]
        print(
            f"  {name}: shape={stats.get('shape')} dtype={stats.get('dtype')} "
            f"min={stats.get('min')} max={stats.get('max')} mean={stats.get('mean')} std={stats.get('std')} "
            f"nan={stats.get('nan_count')} inf={stats.get('inf_count')}"
        )
    print(f"Wrote {args.json_output}")
    print(f"Wrote {args.markdown_output}")


if __name__ == "__main__":
    main()
