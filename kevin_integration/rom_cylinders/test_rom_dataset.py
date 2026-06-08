from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from rom_data_contract import DP_SLICE, EXTENSION_SLICE, TARGET_DELTA_DP_SLICE, TARGET_DELTA_EXTENSION_SLICE
    from rom_dataset import (
        DEFAULT_DATASET_PATH,
        ROMSequenceDataset,
        check_split_run_overlap,
        find_default_dataset,
        load_metadata,
        validate_sequence_boundaries,
    )
except ImportError:
    from kevin_integration.rom_cylinders.rom_data_contract import (
        DP_SLICE,
        EXTENSION_SLICE,
        TARGET_DELTA_DP_SLICE,
        TARGET_DELTA_EXTENSION_SLICE,
    )
    from kevin_integration.rom_cylinders.rom_dataset import (
        DEFAULT_DATASET_PATH,
        ROMSequenceDataset,
        check_split_run_overlap,
        find_default_dataset,
        load_metadata,
        validate_sequence_boundaries,
    )


def _scan_finite(handle: h5py.File, key: str, chunk_samples: int = 32768) -> tuple[bool, int, int]:
    data = handle[key]
    sample_dim = data.ndim - 1
    sample_count = data.shape[sample_dim]
    bad_nan = 0
    bad_inf = 0
    for start in range(0, sample_count, chunk_samples):
        stop = min(start + chunk_samples, sample_count)
        if data.ndim == 3:
            chunk = np.asarray(data[:, :, start:stop])
        elif data.ndim == 2:
            chunk = np.asarray(data[:, start:stop])
        else:
            chunk = np.asarray(data[start:stop])
        bad_nan += int(np.isnan(chunk).sum())
        bad_inf += int(np.isinf(chunk).sum())
    return bad_nan == 0 and bad_inf == 0, bad_nan, bad_inf


def verify_target_timing(dataset: ROMSequenceDataset, *, max_checks: int = 2000, atol: float = 1.0e-4) -> dict[str, float | int]:
    checks = 0
    ext_max_abs_error = 0.0
    dp_max_abs_error = 0.0
    run_ids = dataset.run_ids
    time_indices = dataset.time_indices
    for idx in range(len(dataset) - 1):
        if run_ids[idx] != run_ids[idx + 1]:
            continue
        if time_indices[idx + 1] != time_indices[idx] + 1:
            continue
        _, _, info = dataset[idx]
        _, _, next_info = dataset[idx + 1]
        current_raw = info["raw_input"]
        next_raw = next_info["raw_input"]
        raw_target = info["raw_target"]
        ext_next_pred = current_raw[-1, EXTENSION_SLICE] + raw_target[TARGET_DELTA_EXTENSION_SLICE]
        dp_next_pred = current_raw[-1, DP_SLICE] + raw_target[TARGET_DELTA_DP_SLICE]
        ext_err = torch.max(torch.abs(ext_next_pred - next_raw[-1, EXTENSION_SLICE])).item()
        dp_err = torch.max(torch.abs(dp_next_pred - next_raw[-1, DP_SLICE])).item()
        ext_max_abs_error = max(ext_max_abs_error, ext_err)
        dp_max_abs_error = max(dp_max_abs_error, dp_err)
        checks += 1
        if checks >= max_checks:
            break
    if checks == 0:
        raise RuntimeError("No consecutive same-run samples were available for target timing verification.")
    if ext_max_abs_error > atol:
        raise AssertionError(f"DeltaExtension timing check failed: max abs error {ext_max_abs_error} > {atol}.")
    pressure_atol = max(atol, 16.0)
    if dp_max_abs_error > pressure_atol:
        raise AssertionError(f"DeltaDP timing check failed: max abs error {dp_max_abs_error} > {pressure_atol}.")
    return {"checks": checks, "extension_max_abs_error": ext_max_abs_error, "dp_max_abs_error": dp_max_abs_error}


def test_split(
    dataset_path: Path,
    split: str,
    *,
    use_force_input: bool,
    split_scheme: str,
    locked_manifest: Path,
    target_mode: str,
) -> None:
    dataset = ROMSequenceDataset(
        dataset_path,
        split=split,
        use_force_input=use_force_input,
        split_scheme=split_scheme,
        locked_manifest=locked_manifest,
        target_mode=target_mode,
        normalize_inputs=True,
        normalize_targets=True,
        return_metadata=True,
    )
    raw_dataset = ROMSequenceDataset(
        dataset_path,
        split=split,
        use_force_input=use_force_input,
        split_scheme=split_scheme,
        locked_manifest=locked_manifest,
        target_mode=target_mode,
        normalize_inputs=False,
        normalize_targets=False,
        return_metadata=True,
    )
    timing_dataset = ROMSequenceDataset(
        dataset_path,
        split=split,
        use_force_input=use_force_input,
        split_scheme=split_scheme,
        locked_manifest=locked_manifest,
        target_mode="extension_and_dp",
        normalize_inputs=False,
        normalize_targets=False,
        return_metadata=True,
    )

    expected_input_dim = 30 if use_force_input else 26
    expected_output_dim = 4 if target_mode == "extension_only" else 8
    assert dataset.input_dim == expected_input_dim
    assert dataset.output_dim == expected_output_dim
    assert dataset.history_length == 10
    assert len(dataset) > 0
    assert dataset.number_of_unique_runs > 0

    x, y, info = dataset[0]
    x_raw, y_raw, raw_info = raw_dataset[0]
    assert tuple(x.shape) == (10, expected_input_dim)
    assert tuple(y.shape) == (expected_output_dim,)
    assert tuple(x_raw.shape) == (10, expected_input_dim)
    assert tuple(y_raw.shape) == (expected_output_dim,)
    assert torch.isfinite(x).all()
    assert torch.isfinite(y).all()
    assert torch.isfinite(x_raw).all()
    assert torch.isfinite(y_raw).all()
    assert raw_info["current_extension"].shape == (4,)
    assert raw_info["current_dP"].shape == (4,)

    boundary = validate_sequence_boundaries(dataset)
    assert boundary["violation_count"] == 0, boundary
    timing = verify_target_timing(timing_dataset)

    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    batch_x, batch_y, batch_info = next(iter(loader))
    assert tuple(batch_x.shape) == (16, 10, expected_input_dim)
    assert tuple(batch_y.shape) == (16, expected_output_dim)
    assert torch.isfinite(batch_x).all()
    assert torch.isfinite(batch_y).all()

    print(f"\nSplit={split} split_scheme={split_scheme} target_mode={target_mode} use_force_input={use_force_input}")
    print(f"  samples={len(dataset)} unique_runs={dataset.number_of_unique_runs}")
    print(f"  first split_run_id={info['split_run_id']} time_index={int(info['time_index'])}")
    print(f"  normalized X[0,-1,:8]={x[-1, :8].tolist()}")
    print(f"  normalized Y[0]={y.tolist()}")
    print(f"  raw extension final={raw_info['current_extension'].tolist()}")
    print(f"  raw dP final={raw_info['current_dP'].tolist()}")
    print(f"  raw target={raw_info['raw_target'].tolist()}")
    print(f"  target timing={timing}")
    print(f"  dataloader batch split_run_ids={batch_info['split_run_id'][:3]}")

    dataset.close()
    raw_dataset.close()
    timing_dataset.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the ROMSequenceDataset loader.")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--use_force_input", action="store_true")
    parser.add_argument("--also_test_force", action="store_true", help="Test both core and force-augmented loaders.")
    parser.add_argument("--split_scheme", choices=["original", "locked"], default="original")
    parser.add_argument("--locked_manifest", type=Path, default=Path(__file__).resolve().parent / "rom_split_manifest.json")
    parser.add_argument("--target_mode", choices=["extension_only", "extension_and_dp", "both"], default="both")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset or (DEFAULT_DATASET_PATH if DEFAULT_DATASET_PATH.exists() else find_default_dataset())
    metadata = load_metadata(dataset_path)
    print(metadata)

    with h5py.File(dataset_path, "r") as handle:
        for key in ("X_train_rom", "Y_train_rom", "X_val_rom", "Y_val_rom", "Force_train_hist", "Force_val_hist"):
            ok, nan_count, inf_count = _scan_finite(handle, key)
            if not ok:
                raise AssertionError(f"{key} contains NaN={nan_count} Inf={inf_count}.")
            print(f"{key}: finite")

    overlap = check_split_run_overlap(dataset_path)
    print(f"Split-qualified run overlap: {overlap}")
    if overlap["overlaps"]:
        raise AssertionError(f"Split-qualified run overlap detected: {overlap['overlaps']}")

    force_modes = [args.use_force_input]
    if args.also_test_force and not args.use_force_input:
        force_modes.append(True)
    target_modes = ["extension_only", "extension_and_dp"] if args.target_mode == "both" else [args.target_mode]
    splits = ["train", "val"] if args.split_scheme == "original" else ["train", "val", "test"]
    for use_force in force_modes:
        for target_mode in target_modes:
            for split in splits:
                test_split(
                    dataset_path,
                    split,
                    use_force_input=use_force,
                    split_scheme=args.split_scheme,
                    locked_manifest=args.locked_manifest,
                    target_mode=target_mode,
                )

    print("\nROM dataset checks passed.")


if __name__ == "__main__":
    main()
