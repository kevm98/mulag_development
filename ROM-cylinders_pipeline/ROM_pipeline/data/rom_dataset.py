from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .rom_schema import DEFAULT_SCHEMA, ROMSchema


REQUIRED_ARRAYS = ("history", "target", "valve_label", "run_id", "time_index", "intervention_id")


@dataclass(frozen=True)
class NormalizationStats:
    input_mean: np.ndarray
    input_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray

    def torch(self) -> dict[str, torch.Tensor]:
        return {
            "input_mean": torch.as_tensor(self.input_mean, dtype=torch.float32),
            "input_std": torch.as_tensor(self.input_std, dtype=torch.float32),
            "target_mean": torch.as_tensor(self.target_mean, dtype=torch.float32),
            "target_std": torch.as_tensor(self.target_std, dtype=torch.float32),
        }


class ROMDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, path: str | Path, *, schema: ROMSchema = DEFAULT_SCHEMA) -> None:
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False, mmap_mode="r") as data:
            missing = [name for name in REQUIRED_ARRAYS if name not in data]
            if missing:
                raise ValueError(f"{self.path} is missing arrays: {missing}")
            self.history = np.asarray(data["history"], dtype=np.float32)
            self.target = np.asarray(data["target"], dtype=np.float32)
            self.valve_label = np.asarray(data["valve_label"], dtype=np.float32)
            self.run_id = np.asarray(data["run_id"], dtype=np.int64)
            self.time_index = np.asarray(data["time_index"], dtype=np.int64)
            self.intervention_id = np.asarray(data["intervention_id"], dtype=np.int64)
            self.source_id = (
                np.asarray(data["source_id"], dtype=np.int64)
                if "source_id" in data else np.full(len(self.history), -1, dtype=np.int64)
            )
        expected_history = (schema.history_length, schema.input_dim)
        if self.history.ndim != 3 or tuple(self.history.shape[1:]) != expected_history:
            raise ValueError(f"history must have [N, {expected_history[0]}, {expected_history[1]}], got {self.history.shape}")
        if self.target.shape != (len(self.history), schema.target_dim):
            raise ValueError(f"target must have shape [N, {schema.target_dim}], got {self.target.shape}")
        if self.valve_label.shape != (len(self.history), 4):
            raise ValueError(f"valve_label must have shape [N, 4], got {self.valve_label.shape}")
        if not all(len(value) == len(self.history) for value in (
            self.run_id, self.time_index, self.intervention_id, self.source_id
        )):
            raise ValueError("All sample metadata arrays must have the same length as history.")
        self._lookup = {(int(run), int(time)): index for index, (run, time) in enumerate(zip(self.run_id, self.time_index))}
        self._intervention_lookup: dict[tuple[int, int], list[int]] = {}
        for index, (intervention, time) in enumerate(zip(self.intervention_id, self.time_index)):
            if int(intervention) >= 0:
                self._intervention_lookup.setdefault((int(intervention), int(time)), []).append(index)

    def __len__(self) -> int:
        return len(self.history)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "index": torch.tensor(index, dtype=torch.long),
            "history": torch.from_numpy(self.history[index]),
            "target": torch.from_numpy(self.target[index]),
            "valve_label": torch.from_numpy(self.valve_label[index]),
            "run_id": torch.tensor(self.run_id[index], dtype=torch.long),
            "time_index": torch.tensor(self.time_index[index], dtype=torch.long),
            "intervention_id": torch.tensor(self.intervention_id[index], dtype=torch.long),
            "source_id": torch.tensor(self.source_id[index], dtype=torch.long),
        }

    def sequence_indices(self, starts: np.ndarray, horizon: int) -> np.ndarray:
        """Return contiguous within-run indices, dropping starts without a full horizon."""
        sequences = []
        for raw_start in np.asarray(starts).reshape(-1):
            start = int(raw_start)
            run = int(self.run_id[start])
            time = int(self.time_index[start])
            sequence = [self._lookup.get((run, time + offset), -1) for offset in range(horizon)]
            if min(sequence) >= 0:
                sequences.append(sequence)
        return np.asarray(sequences, dtype=np.int64).reshape(-1, horizon)

    def command_pair_indices(self, starts: np.ndarray) -> np.ndarray:
        """Find matched-state intervention partners at the same aligned time index."""
        pairs = []
        for raw_start in np.asarray(starts).reshape(-1):
            start = int(raw_start)
            intervention = int(self.intervention_id[start])
            if intervention < 0:
                continue
            candidates = self._intervention_lookup.get((intervention, int(self.time_index[start])), [])
            partner = next((index for index in candidates if int(self.run_id[index]) != int(self.run_id[start])), None)
            if partner is not None:
                pairs.append((start, partner))
        return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


@dataclass(frozen=True)
class ROMDatasetBundle:
    root: Path
    train: ROMDataset
    validation: ROMDataset
    test: ROMDataset
    stats: NormalizationStats
    manifest: dict[str, Any]


def load_dataset_bundle(root: str | Path, *, schema: ROMSchema = DEFAULT_SCHEMA) -> ROMDatasetBundle:
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if tuple(manifest["input_feature_names"]) != schema.input_feature_names:
        raise ValueError("Dataset input feature order does not match the runtime schema.")
    if tuple(manifest["target_feature_names"]) != schema.target_feature_names:
        raise ValueError("Dataset target feature order does not match the runtime schema.")
    if int(manifest["history_length"]) != schema.history_length:
        raise ValueError("Dataset history length does not match the runtime schema.")
    with np.load(root / "normalization.npz", allow_pickle=False) as data:
        stats = NormalizationStats(*(np.asarray(data[name], dtype=np.float32) for name in (
            "input_mean", "input_std", "target_mean", "target_std"
        )))
    return ROMDatasetBundle(
        root=root,
        train=ROMDataset(root / "train.npz", schema=schema),
        validation=ROMDataset(root / "validation.npz", schema=schema),
        test=ROMDataset(root / "test.npz", schema=schema),
        stats=stats,
        manifest=manifest,
    )
