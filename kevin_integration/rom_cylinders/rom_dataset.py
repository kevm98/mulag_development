from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from rom_data_contract import (
        AUGMENTED_INPUT_DIM,
        CORE_INPUT_DIM,
        CORE_INPUT_FEATURE_NAMES,
        DP_SLICE,
        EXTENSION_SLICE,
        FORCE_FEATURE_NAMES,
        HISTORY_LENGTH,
        SUPPORTED_SPLITS,
    TARGET_DELTA_EXTENSION_SLICE,
    TARGET_DIM,
        TARGET_FEATURE_NAMES,
        validate_contract,
    )
except ImportError:
    from kevin_integration.rom_cylinders.rom_data_contract import (
        AUGMENTED_INPUT_DIM,
        CORE_INPUT_DIM,
        CORE_INPUT_FEATURE_NAMES,
        DP_SLICE,
        EXTENSION_SLICE,
        FORCE_FEATURE_NAMES,
        HISTORY_LENGTH,
        SUPPORTED_SPLITS,
    TARGET_DELTA_EXTENSION_SLICE,
    TARGET_DIM,
        TARGET_FEATURE_NAMES,
        validate_contract,
    )


ROM_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = ROM_DIR / "pytorch_hydraulic_rom_dataset.mat"
LOCKED_SPLIT_MANIFEST_PATH = ROM_DIR / "rom_split_manifest.json"
SplitName = Literal["train", "val", "test"]
TargetMode = Literal["extension_only", "extension_and_dp"]


@dataclass(frozen=True)
class NormalizationStats:
    input_mean: torch.Tensor
    input_std: torch.Tensor
    target_mean: torch.Tensor
    target_std: torch.Tensor
    force_mean: torch.Tensor | None = None
    force_std: torch.Tensor | None = None
    target_mode: str = "extension_and_dp"


@dataclass(frozen=True)
class ROMDatasetMetadata:
    dataset_path: str
    dataset_sha256: str | None
    input_feature_names: tuple[str, ...]
    force_feature_names: tuple[str, ...]
    target_feature_names: tuple[str, ...]
    history_length: int
    dt_seconds: float
    train_run_count: int | None
    validation_run_count: int | None
    train_sample_count: int | None
    validation_sample_count: int | None
    has_test_split: bool


def find_default_dataset(rom_dir: Path = ROM_DIR) -> Path:
    mats = sorted(path for path in rom_dir.glob("*.mat") if path.is_file())
    if not mats:
        raise FileNotFoundError(f"No .mat files found in {rom_dir}")
    if len(mats) > 1:
        raise RuntimeError(f"Multiple .mat files found in {rom_dir}; pass --dataset explicitly.")
    return mats[0]


def file_sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _decode_utf16_codes(array: np.ndarray) -> str | None:
    if array.dtype == np.uint16 and array.size > 0:
        values = array.reshape(-1)
        if np.all((values == 0) | ((values >= 9) & (values <= 126))):
            return "".join(chr(int(value)) for value in values if int(value) != 0)
    return None


def _decode_hdf5_reference(handle: h5py.File, ref: Any) -> str | None:
    try:
        if not ref:
            return None
        target = handle[ref]
        data = np.asarray(target[()])
    except (KeyError, TypeError, ValueError):
        return None
    return _decode_utf16_codes(data)


def _read_string_array(handle: h5py.File, key: str) -> tuple[str, ...]:
    data = np.asarray(handle[key][()])
    if data.dtype == object:
        values: list[str] = []
        for item in data.reshape(-1):
            decoded = _decode_hdf5_reference(handle, item)
            if decoded is None:
                raise ValueError(f"Could not decode MATLAB string reference in {key}.")
            values.append(decoded)
        return tuple(values)
    decoded = _decode_utf16_codes(data)
    if decoded is not None:
        return (decoded,)
    raise ValueError(f"{key} is not a supported MATLAB string array.")


def _scalar(handle: h5py.File, key: str, default: float | None = None) -> float | None:
    if key not in handle:
        return default
    value = np.asarray(handle[key][()]).reshape(-1)
    if value.size == 0:
        return default
    return float(value[0])


def _split_prefix(split: str) -> str:
    normalized = split.lower()
    if normalized in {"train", "training"}:
        return "train"
    if normalized in {"val", "validation", "valid"}:
        return "val"
    if normalized == "test":
        return "test"
    raise ValueError(f"Unsupported split {split!r}. Supported splits: {SUPPORTED_SPLITS}")


def _target_dim_for_mode(target_mode: str) -> int:
    if target_mode == "extension_only":
        return 4
    if target_mode == "extension_and_dp":
        return TARGET_DIM
    raise ValueError("target_mode must be 'extension_only' or 'extension_and_dp'.")


def _target_slice_for_mode(target_mode: str) -> slice:
    if target_mode == "extension_only":
        return TARGET_DELTA_EXTENSION_SLICE
    if target_mode == "extension_and_dp":
        return slice(0, TARGET_DIM)
    raise ValueError("target_mode must be 'extension_only' or 'extension_and_dp'.")


def load_metadata(dataset_path: str | Path = DEFAULT_DATASET_PATH, *, compute_hash: bool = False) -> ROMDatasetMetadata:
    path = Path(dataset_path)
    with h5py.File(path, "r") as handle:
        input_names = _read_string_array(handle, "input_feature_names")
        target_names = _read_string_array(handle, "output_feature_names")
        force_names = _read_string_array(handle, "force_feature_names") if "force_feature_names" in handle else FORCE_FEATURE_NAMES
        history_length = int(_scalar(handle, "ROMMetadata/history_length", _scalar(handle, "T", HISTORY_LENGTH)) or HISTORY_LENGTH)
        dt_seconds = float(_scalar(handle, "ROMMetadata/global_median_sample_time", 0.05) or 0.05)
        train_run_count = _scalar(handle, "ROMMetadata/num_training_runs")
        validation_run_count = _scalar(handle, "ROMMetadata/num_validation_runs")
        train_sample_count = _scalar(handle, "ROMMetadata/num_training_samples")
        validation_sample_count = _scalar(handle, "ROMMetadata/num_validation_samples")
        has_test_split = "X_test_rom" in handle and "Y_test_rom" in handle

    validate_contract(
        input_feature_names=input_names,
        target_feature_names=target_names,
        history_length=history_length,
        input_dimension=len(input_names),
        output_dimension=len(target_names),
        use_force_input=False,
    )
    if force_names != FORCE_FEATURE_NAMES:
        raise ValueError(f"Force feature order mismatch. Expected {FORCE_FEATURE_NAMES}, got {force_names}.")

    return ROMDatasetMetadata(
        dataset_path=str(path),
        dataset_sha256=file_sha256(path) if compute_hash else None,
        input_feature_names=input_names,
        force_feature_names=force_names,
        target_feature_names=target_names,
        history_length=history_length,
        dt_seconds=dt_seconds,
        train_run_count=int(train_run_count) if train_run_count is not None else None,
        validation_run_count=int(validation_run_count) if validation_run_count is not None else None,
        train_sample_count=int(train_sample_count) if train_sample_count is not None else None,
        validation_sample_count=int(validation_sample_count) if validation_sample_count is not None else None,
        has_test_split=has_test_split,
    )


def _column_vector(handle: h5py.File, key: str, expected_len: int) -> np.ndarray:
    value = np.asarray(handle[key][()], dtype=np.float32).reshape(-1)
    if value.shape != (expected_len,):
        raise ValueError(f"{key} must have length {expected_len}, got shape {value.shape}.")
    return value


def load_normalization_stats(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    *,
    use_force_input: bool = False,
    target_mode: TargetMode = "extension_and_dp",
) -> NormalizationStats:
    path = Path(dataset_path)
    with h5py.File(path, "r") as handle:
        x_mean = _column_vector(handle, "x_mean_rom", CORE_INPUT_DIM)
        x_std = _column_vector(handle, "x_std_rom", CORE_INPUT_DIM)
        y_mean = _column_vector(handle, "y_mean_rom", TARGET_DIM)
        y_std = _column_vector(handle, "y_std_rom", TARGET_DIM)
        force_mean = _column_vector(handle, "force_mean", len(FORCE_FEATURE_NAMES)) if use_force_input else None
        force_std = _column_vector(handle, "force_std", len(FORCE_FEATURE_NAMES)) if use_force_input else None

    if use_force_input:
        assert force_mean is not None
        assert force_std is not None
        x_mean = np.concatenate([x_mean, force_mean], axis=0)
        x_std = np.concatenate([x_std, force_std], axis=0)

    x_std = np.maximum(x_std, 1.0e-12)
    y_std = np.maximum(y_std, 1.0e-12)
    if force_std is not None:
        force_std = np.maximum(force_std, 1.0e-12)

    target_slice = _target_slice_for_mode(target_mode)
    y_mean = y_mean[target_slice]
    y_std = y_std[target_slice]

    return NormalizationStats(
        input_mean=torch.from_numpy(x_mean.astype(np.float32)),
        input_std=torch.from_numpy(x_std.astype(np.float32)),
        target_mean=torch.from_numpy(y_mean.astype(np.float32)),
        target_std=torch.from_numpy(y_std.astype(np.float32)),
        force_mean=torch.from_numpy(force_mean.astype(np.float32)) if force_mean is not None else None,
        force_std=torch.from_numpy(force_std.astype(np.float32)) if force_std is not None else None,
        target_mode=target_mode,
    )


def _load_locked_split(manifest_path: Path, split: str) -> dict[str, Any]:
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Locked split manifest not found at {manifest_path}. Run create_locked_rom_splits.py first."
        )
    import json

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    splits = manifest.get("splits", {})
    if split not in splits:
        raise KeyError(f"Locked split {split!r} is missing from {manifest_path}.")
    split_file = manifest_path.parent / splits[split]["file"]
    if not split_file.exists():
        raise FileNotFoundError(f"Locked split file missing: {split_file}")
    payload = torch.load(split_file, map_location="cpu")
    payload["manifest"] = manifest
    payload["split_file"] = str(split_file)
    return payload


class ROMSequenceDataset(Dataset):
    """Lazy PyTorch dataset for the hydraulic cylinder ROM MATLAB file."""

    def __init__(
        self,
        dataset_path: str | Path = DEFAULT_DATASET_PATH,
        *,
        split: str = "train",
        use_force_input: bool = False,
        normalize_inputs: bool = True,
        normalize_targets: bool = True,
        return_metadata: bool = False,
        reject_nonfinite: bool = True,
        split_scheme: str = "original",
        locked_manifest: str | Path = LOCKED_SPLIT_MANIFEST_PATH,
        target_mode: TargetMode = "extension_and_dp",
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.split = _split_prefix(split)
        self.split_scheme = str(split_scheme)
        if self.split_scheme not in {"original", "locked"}:
            raise ValueError("split_scheme must be 'original' or 'locked'.")
        if self.split == "test" and self.split_scheme != "locked":
            raise ValueError("split='test' is only available with split_scheme='locked'.")
        self.use_force_input = bool(use_force_input)
        self.target_mode = target_mode
        self.normalize_inputs = bool(normalize_inputs)
        self.normalize_targets = bool(normalize_targets)
        self.return_metadata = bool(return_metadata)
        self.reject_nonfinite = bool(reject_nonfinite)
        self.locked_manifest = Path(locked_manifest)
        self._locked_payload: dict[str, Any] | None = None

        self._handle: h5py.File | None = None
        self._metadata = load_metadata(self.dataset_path, compute_hash=False)
        self._stats = load_normalization_stats(
            self.dataset_path,
            use_force_input=self.use_force_input,
            target_mode=self.target_mode,
        )

        with h5py.File(self.dataset_path, "r") as handle:
            if self.split_scheme == "locked":
                self._locked_payload = _load_locked_split(self.locked_manifest, self.split)
                self._source_split = str(self._locked_payload["source_split"])
                if self._source_split not in {"train", "val"}:
                    raise ValueError(f"Unsupported locked source_split {self._source_split!r}.")
                self._sample_indices = np.asarray(self._locked_payload["sample_indices"], dtype=np.int64).reshape(-1)
                source_prefix = "Train" if self._source_split == "train" else "Val"
                self._validate_split_arrays(handle, self._source_split)
                self._length = int(self._sample_indices.shape[0])
                all_run_ids = np.asarray(handle[f"{source_prefix}RunID"][()]).reshape(-1).astype(np.int64)
                all_time_indices = np.asarray(handle[f"{source_prefix}TimeIndex"][()]).reshape(-1).astype(np.int64)
                all_sample_times = np.asarray(handle[f"{source_prefix}SampleTime"][()]).reshape(-1).astype(np.float32)
                all_classes = np.asarray(handle[f"{source_prefix}Class"][()]).reshape(-1).astype(np.int64)
                self._run_ids = all_run_ids[self._sample_indices]
                self._time_indices = all_time_indices[self._sample_indices]
                self._sample_times = all_sample_times[self._sample_indices]
                self._classes = all_classes[self._sample_indices]
                force_key = f"ForceAvailable_{self._source_split}"
                self._force_available = (
                    np.asarray(handle[force_key][()]).reshape(-1).astype(bool)[self._sample_indices]
                    if force_key in handle
                    else None
                )
            else:
                self._source_split = self.split
                self._sample_indices = None
                self._validate_split_arrays(handle, self.split)
                prefix = "Train" if self.split == "train" else "Val"
                self._length = int(handle[f"X_{self.split}_rom"].shape[2])
                self._run_ids = np.asarray(handle[f"{prefix}RunID"][()]).reshape(-1).astype(np.int64)
                self._time_indices = np.asarray(handle[f"{prefix}TimeIndex"][()]).reshape(-1).astype(np.int64)
                self._sample_times = np.asarray(handle[f"{prefix}SampleTime"][()]).reshape(-1).astype(np.float32)
                self._classes = np.asarray(handle[f"{prefix}Class"][()]).reshape(-1).astype(np.int64)
                self._force_available = (
                    np.asarray(handle[f"ForceAvailable_{self.split}"][()]).reshape(-1).astype(bool)
                    if f"ForceAvailable_{self.split}" in handle
                    else None
                )

        if self._run_ids.shape[0] != self._length:
            raise ValueError(f"Run ID length {self._run_ids.shape[0]} does not match sample count {self._length}.")
        if self._time_indices.shape[0] != self._length:
            raise ValueError(f"Time-index length {self._time_indices.shape[0]} does not match sample count {self._length}.")
        if self.use_force_input and self._force_available is not None and not np.all(self._force_available):
            missing = int((~self._force_available).sum())
            raise ValueError(f"Force input requested but {missing} samples are marked unavailable.")

        self.input_dim = AUGMENTED_INPUT_DIM if self.use_force_input else CORE_INPUT_DIM
        self.output_dim = _target_dim_for_mode(self.target_mode)
        self.history_length = HISTORY_LENGTH

    def _validate_split_arrays(self, handle: h5py.File, split_name: str) -> None:
        x_key = f"X_{split_name}_rom"
        y_key = f"Y_{split_name}_rom"
        if x_key not in handle or y_key not in handle:
            raise KeyError(f"Dataset split {self.split!r} is missing {x_key} or {y_key}.")
        x_shape = tuple(handle[x_key].shape)
        y_shape = tuple(handle[y_key].shape)
        if len(x_shape) != 3 or x_shape[0] != CORE_INPUT_DIM or x_shape[1] != HISTORY_LENGTH:
            raise ValueError(f"{x_key} must be [{CORE_INPUT_DIM}, {HISTORY_LENGTH}, sample], got {x_shape}.")
        if len(y_shape) != 2 or y_shape[0] != TARGET_DIM or y_shape[1] != x_shape[2]:
            raise ValueError(f"{y_key} must be [{TARGET_DIM}, sample] matching {x_key}, got {y_shape}.")
        if self.use_force_input:
            force_key = f"Force_{split_name}_hist"
            if force_key not in handle:
                raise KeyError(f"Force input requested but {force_key} is missing.")
            force_shape = tuple(handle[force_key].shape)
            if force_shape != (len(FORCE_FEATURE_NAMES), HISTORY_LENGTH, x_shape[2]):
                raise ValueError(
                    f"{force_key} must be [{len(FORCE_FEATURE_NAMES)}, {HISTORY_LENGTH}, sample], got {force_shape}."
                )

    @property
    def metadata(self) -> ROMDatasetMetadata:
        return self._metadata

    @property
    def normalization_stats(self) -> NormalizationStats:
        return self._stats

    @property
    def number_of_unique_runs(self) -> int:
        return int(np.unique(self._run_ids).size)

    @property
    def run_ids(self) -> np.ndarray:
        return self._run_ids.copy()

    @property
    def time_indices(self) -> np.ndarray:
        return self._time_indices.copy()

    def __len__(self) -> int:
        return self._length

    def _h5(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(self.dataset_path, "r")
        return self._handle

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __del__(self) -> None:
        self.close()

    def _raw_sample(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        handle = self._h5()
        source_index = int(self._sample_indices[index]) if self._sample_indices is not None else int(index)
        source_split = self._source_split
        x = np.asarray(handle[f"X_{source_split}_rom"][:, :, source_index], dtype=np.float32).T
        y = np.asarray(handle[f"Y_{source_split}_rom"][:, source_index], dtype=np.float32)
        y = y[_target_slice_for_mode(self.target_mode)]
        if self.use_force_input:
            force = np.asarray(handle[f"Force_{source_split}_hist"][:, :, source_index], dtype=np.float32).T
            x = np.concatenate([x, force], axis=-1)
        return x, y

    def _normalize_x(self, x: torch.Tensor) -> torch.Tensor:
        mean = self._stats.input_mean.to(dtype=x.dtype)
        std = self._stats.input_std.to(dtype=x.dtype)
        return (x - mean) / std

    def _normalize_y(self, y: torch.Tensor) -> torch.Tensor:
        mean = self._stats.target_mean.to(dtype=y.dtype)
        std = self._stats.target_std.to(dtype=y.dtype)
        return (y - mean) / std

    def denormalize_y(self, y: torch.Tensor) -> torch.Tensor:
        mean = self._stats.target_mean.to(device=y.device, dtype=y.dtype)
        std = self._stats.target_std.to(device=y.device, dtype=y.dtype)
        return y * std + mean

    def normalize_y(self, y: torch.Tensor) -> torch.Tensor:
        mean = self._stats.target_mean.to(device=y.device, dtype=y.dtype)
        std = self._stats.target_std.to(device=y.device, dtype=y.dtype)
        return (y - mean) / std

    def denormalize_x(self, x: torch.Tensor) -> torch.Tensor:
        mean = self._stats.input_mean.to(device=x.device, dtype=x.dtype)
        std = self._stats.input_std.to(device=x.device, dtype=x.dtype)
        return x * std + mean

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError(index)

        x_raw_np, y_raw_np = self._raw_sample(index)
        if self.reject_nonfinite and (not np.isfinite(x_raw_np).all() or not np.isfinite(y_raw_np).all()):
            raise ValueError(f"Non-finite sample detected in split={self.split} index={index}.")

        x_raw = torch.from_numpy(x_raw_np)
        y_raw = torch.from_numpy(y_raw_np)
        x = self._normalize_x(x_raw) if self.normalize_inputs else x_raw
        y = self._normalize_y(y_raw) if self.normalize_targets else y_raw

        if not self.return_metadata:
            return x, y

        info: dict[str, Any] = {
            "split": self.split,
            "split_scheme": self.split_scheme,
            "source_split": self._source_split,
            "sample_index": torch.tensor(index, dtype=torch.long),
            "source_sample_index": torch.tensor(
                int(self._sample_indices[index]) if self._sample_indices is not None else index,
                dtype=torch.long,
            ),
            "run_id": torch.tensor(int(self._run_ids[index]), dtype=torch.long),
            "split_run_id": f"{self.split}:{int(self._run_ids[index])}",
            "source_split_run_id": f"{self._source_split}:{int(self._run_ids[index])}",
            "time_index": torch.tensor(int(self._time_indices[index]), dtype=torch.long),
            "sample_time": torch.tensor(float(self._sample_times[index]), dtype=torch.float32),
            "class_id": torch.tensor(int(self._classes[index]), dtype=torch.long),
            "current_extension": x_raw[-1, EXTENSION_SLICE].clone(),
            "current_dP": x_raw[-1, DP_SLICE].clone(),
            "raw_target": y_raw.clone(),
            "raw_input": x_raw.clone(),
            "normalized": {
                "inputs": self.normalize_inputs,
                "targets": self.normalize_targets,
            },
        }
        return x, y, info


def split_run_sets(dataset_path: str | Path = DEFAULT_DATASET_PATH) -> dict[str, set[str]]:
    path = Path(dataset_path)
    result: dict[str, set[str]] = {}
    with h5py.File(path, "r") as handle:
        if "TrainRunID" in handle:
            ids = np.asarray(handle["TrainRunID"][()]).reshape(-1).astype(np.int64)
            result["train"] = {f"train:{int(value)}" for value in np.unique(ids)}
        if "ValRunID" in handle:
            ids = np.asarray(handle["ValRunID"][()]).reshape(-1).astype(np.int64)
            result["val"] = {f"val:{int(value)}" for value in np.unique(ids)}
        if "TestRunID" in handle:
            ids = np.asarray(handle["TestRunID"][()]).reshape(-1).astype(np.int64)
            result["test"] = {f"test:{int(value)}" for value in np.unique(ids)}
    return result


def check_split_run_overlap(dataset_path: str | Path = DEFAULT_DATASET_PATH) -> dict[str, Any]:
    runs = split_run_sets(dataset_path)
    overlaps: list[dict[str, Any]] = []
    names = sorted(runs)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            both = sorted(runs[left] & runs[right])
            if both:
                overlaps.append({"left": left, "right": right, "count": len(both), "preview": both[:20]})
    return {"run_counts": {name: len(values) for name, values in runs.items()}, "overlaps": overlaps}


def validate_sequence_boundaries(dataset: ROMSequenceDataset) -> dict[str, Any]:
    run_ids = dataset.run_ids
    time_indices = dataset.time_indices
    violations: list[dict[str, int]] = []
    for idx in range(1, len(dataset)):
        same_run = run_ids[idx] == run_ids[idx - 1]
        if same_run and time_indices[idx] != time_indices[idx - 1] + 1:
            violations.append(
                {
                    "previous_index": idx - 1,
                    "index": idx,
                    "run_id": int(run_ids[idx]),
                    "previous_time_index": int(time_indices[idx - 1]),
                    "time_index": int(time_indices[idx]),
                }
            )
    return {"checked_samples": len(dataset), "violation_count": len(violations), "violations_preview": violations[:20]}
