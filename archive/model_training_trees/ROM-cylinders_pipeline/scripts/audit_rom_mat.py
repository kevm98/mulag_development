#!/usr/bin/env python3
"""Audit a MATLAB v7.3 candidate without loading its full sample tensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def decode_references(handle: h5py.File, dataset: h5py.Dataset) -> list[str]:
    values = []
    for ref in np.asarray(dataset).reshape(-1):
        if isinstance(ref, h5py.Reference):
            raw = np.asarray(handle[ref]).reshape(-1)
            values.append("".join(chr(int(value)) for value in raw if int(value)))
        else:
            values.append(str(ref))
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mat_file", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with h5py.File(args.mat_file, "r") as handle:
        keys = {name: list(value.shape) if isinstance(value, h5py.Dataset) else "group" for name, value in handle.items() if name != "#refs#"}
        features = decode_references(handle, handle["input_feature_names"]) if "input_feature_names" in handle else []
        report = {
            "path": str(args.mat_file.resolve()),
            "size_bytes": args.mat_file.stat().st_size,
            "top_level": keys,
            "input_feature_names": features,
            "has_desired_extension_command": any(name.startswith(("DesiredDeltaExtension", "TargetDeltaExtension")) for name in features),
            "has_desired_velocity_command": any(name.startswith(("DesiredDeltaExtensionVelocity", "TargetDeltaExtensionVelocity")) for name in features),
            "requires_valve_at_runtime": any(name.startswith("ValveCmd") for name in features),
            "classification": "command_conditioned_candidate" if any("DeltaExtension" in name for name in features) else "valve_conditioned_fd_like",
        }
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
