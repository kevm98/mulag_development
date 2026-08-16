#!/usr/bin/env python3
"""Freeze the selected Stage 31 SD encoder/model and its gate evidence."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    src = Path(args.src).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    required = ("encoder.pt", "sd.pt", "belief.pt", "metrics.json")
    missing = [name for name in required if not (src / name).is_file()]
    if missing:
        raise SystemExit(f"selected Stage 31 SD is incomplete: {missing}")
    if out.exists() and any(out.iterdir()) and not args.force:
        raise SystemExit(f"target exists and is not empty: {out} (use --force)")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    for name in required:
        shutil.copyfile(src / name, out / name)
    for name in ("rollout_gate.json", "gate_verdict.json"):
        if (src / name).is_file():
            shutil.copyfile(src / name, out / name)
    metrics = json.loads((src / "metrics.json").read_text())
    protocol = {
        "format": "stage31_frozen_sd_v1",
        "dataset_root": str(Path(args.dataset_root).expanduser().resolve()),
        "latent_source": metrics.get("latent_source"),
        "latent_dim": metrics.get("latent_dim"),
        "action_contract": "previous_valve_4 + current_valve_4",
        "aam_contract": "joint_q + joint_qdot + dP + previous_valve",
        "source_model": str(src),
        "frozen_at": datetime.now(timezone.utc).isoformat(),
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(f"frozen Stage 31 SD -> {out}")


if __name__ == "__main__":
    main()

