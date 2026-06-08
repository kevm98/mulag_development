from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import torch


REQUIRED_FIELDS = ("valve_cmd", "x_cyl", "xdot_cyl", "dP")
OPTIONAL_FIELDS = ("Fnet", "q_arm", "qdot_arm")

parser = argparse.ArgumentParser(description="Prepare reduced hydraulic ROM parameter fitting from a dataset.")
parser.add_argument("--dataset", type=str, required=True, help="Dataset path. Supports .pt/.pth dicts and CSV files.")
parser.add_argument(
    "--dt",
    type=float,
    default=None,
    help="Dataset sample time. Required later for pressure/velocity fitting if not stored in the dataset.",
)


def _load_dataset(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix in (".pt", ".pth"):
        data = torch.load(path, map_location="cpu")
        if not isinstance(data, dict):
            raise TypeError(f"{path} must contain a dict, got {type(data).__name__}.")
        return data
    if path.suffix == ".csv":
        return _load_csv_dataset(path)
    raise ValueError(f"Unsupported dataset suffix {path.suffix!r}. Use .pt/.pth or .csv.")


def _load_csv_dataset(path: Path) -> dict[str, torch.Tensor]:
    rows: list[dict[str, str]] = []
    with path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows.extend(reader)
    if not rows:
        raise RuntimeError(f"No rows found in {path}.")

    data: dict[str, torch.Tensor] = {}
    for field in REQUIRED_FIELDS + OPTIONAL_FIELDS:
        columns = [f"{field}_{index}" for index in range(4)]
        if all(column in rows[0] for column in columns):
            values = [[float(row[column]) for column in columns] for row in rows]
            data[field] = torch.tensor(values, dtype=torch.float32)
    if "dt" in rows[0]:
        data["dt"] = torch.tensor(float(rows[0]["dt"]), dtype=torch.float32)
    return data


def _as_tensor(data: dict[str, Any], field: str) -> torch.Tensor:
    value = data[field]
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tensor.ndim < 2 or tensor.shape[-1] != 4:
        raise ValueError(f"{field} must have shape [T, 4] or [N, T, 4], got {tuple(tensor.shape)}.")
    return tensor.float()


def _validate_fields(data: dict[str, Any]) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in data]
    if missing:
        required = ", ".join(REQUIRED_FIELDS)
        missing_text = ", ".join(missing)
        raise RuntimeError(
            f"Dataset is missing required fields: {missing_text}. "
            f"Required fields are: {required}. Optional fields are: {', '.join(OPTIONAL_FIELDS)}."
        )


def _candidate_targets(data: dict[str, Any]) -> dict[str, torch.Tensor]:
    x_cyl = _as_tensor(data, "x_cyl")
    xdot_cyl = _as_tensor(data, "xdot_cyl")
    dP = _as_tensor(data, "dP")
    if x_cyl.shape[-2] < 2:
        raise ValueError("Dataset must contain at least two time samples.")
    return {
        "dx_cyl": x_cyl[..., 1:, :] - x_cyl[..., :-1, :],
        "xdot_cyl_next": xdot_cyl[..., 1:, :],
        "dP_next": dP[..., 1:, :],
    }


def estimate_delay_placeholder(data: dict[str, Any]) -> None:
    _ = data
    print("[TODO] Delay estimation placeholder: cross-correlate valve_cmd with dx_cyl/xdot_cyl response per cylinder.")


def estimate_deadzone_placeholder(data: dict[str, Any]) -> None:
    _ = data
    print("[TODO] Dead-zone estimation placeholder: fit asymmetric valve thresholds from near-zero cylinder motion.")


def fit_Kq_placeholder(data: dict[str, Any], targets: dict[str, torch.Tensor]) -> None:
    _ = data, targets
    print("[TODO] Kq fitting placeholder: least-squares fit pressure update residual after delay/dead-zone processing.")


def fit_damping_friction_placeholder(data: dict[str, Any], targets: dict[str, torch.Tensor]) -> None:
    _ = data, targets
    print("[TODO] Damping/friction fitting placeholder: regress force_like terms against cylinder velocity response.")


def main() -> None:
    args = parser.parse_args()
    path = Path(args.dataset)
    data = _load_dataset(path)
    _validate_fields(data)
    targets = _candidate_targets(data)

    print(f"[INFO] Loaded dataset: {path}")
    for field in REQUIRED_FIELDS:
        print(f"[INFO] {field} shape={tuple(_as_tensor(data, field).shape)}")
    for field in OPTIONAL_FIELDS:
        if field in data:
            value = data[field] if isinstance(data[field], torch.Tensor) else torch.as_tensor(data[field])
            print(f"[INFO] optional {field} shape={tuple(value.shape)}")
    dataset_dt = args.dt if args.dt is not None else float(data["dt"]) if "dt" in data else None
    print(f"[INFO] dt={dataset_dt if dataset_dt is not None else 'missing'}")
    print(f"[INFO] candidate dx_cyl shape={tuple(targets['dx_cyl'].shape)}")
    print(f"[INFO] candidate xdot_cyl_next shape={tuple(targets['xdot_cyl_next'].shape)}")
    print(f"[INFO] candidate dP_next shape={tuple(targets['dP_next'].shape)}")
    print("[INFO] No final parameters are produced by this scaffold.")

    estimate_delay_placeholder(data)
    estimate_deadzone_placeholder(data)
    fit_Kq_placeholder(data, targets)
    fit_damping_friction_placeholder(data, targets)


if __name__ == "__main__":
    main()
