#!/usr/bin/env python
"""Build aligned 50 ms learning samples on the ROM tick grid (item 46).

For every raw run in a frozen master dataset, this tool:

- determines the timebases from the run metadata (physics_dt, control_dt,
  rom_dt) and verifies they match 1/120 s, 1/60 s, 0.05 s;
- verifies the recorded ``rom_tick`` grid is exactly one tick every 3 control
  steps starting at step 0 (the pilot-verified alignment);
- builds one learning sample per ROM tick as the exact transition
  ``state row t -> state row t+3`` under command row ``t``, with the 24-D ROM
  input row reconstructed from exact raw rows (no interpolation);
- writes the samples under ``<root>/sd/transitions/`` and a report under
  ``<root>/analysis/``.

The tool refuses to write outside the frozen derived directories and never
touches ``raw/``. Reads run data through ``read_raw_run(verify=True)`` so a
mutated raw file aborts the build.

Usage:
    python build_timebase_samples.py [--root datasets/master_v1_frozen]
        [--force] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


AAM_ROOT = Path(__file__).resolve().parents[1]
if str(AAM_ROOT) not in sys.path:
    sys.path.insert(0, str(AAM_ROOT))

from aam_data import (  # noqa: E402
    iter_raw_runs,
    load_runs_info,
    raw_dir,
    read_raw_run,
    sha256_file,
)
from aam_data.frozen import CHECKSUM_FILE  # noqa: E402
from aam_data.timebase import (  # noqa: E402
    CONTROL_DT,
    PHYSICS_DT,
    ROM_DT,
    STEPS_PER_ROM_TICK,
    build_transitions,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        default=str(AAM_ROOT / "datasets/master_v1_frozen"),
        help="frozen master dataset root",
    )
    p.add_argument(
        "--out",
        default=None,
        help="derived sample dir (default: <root>/sd/transitions)",
    )
    p.add_argument(
        "--report",
        default=None,
        help="report json (default: <root>/analysis/timebase_report.json)",
    )
    p.add_argument("--force", action="store_true", help="overwrite existing output")
    p.add_argument("--limit", type=int, default=None, help="only process first N runs")
    args = p.parse_args()

    root = Path(args.root).expanduser().resolve()
    raw = raw_dir(root)
    if not (raw / "manifest.csv").is_file():
        raise SystemExit(f"not a frozen dataset: {root}")
    out = Path(args.out or str(root / "sd" / "transitions")).expanduser().resolve()
    report_path = Path(
        args.report or str(root / "analysis" / "timebase_report.json")
    ).expanduser().resolve()
    if out == raw or raw in out.parents:
        raise SystemExit(f"refusing to write samples into raw/: {out}")
    if out.exists() and any(out.iterdir()) and not args.force:
        raise SystemExit(f"output exists and is not empty: {out} (use --force)")
    if out.exists():
        shutil.rmtree(out)

    try:
        runs_info = load_runs_info(root)
    except FileNotFoundError:
        # Splits are not required to build timebase samples; frozen_split is
        # recorded when available.
        runs_info = {}
    checksum_digest = sha256_file(raw / CHECKSUM_FILE)
    per_run: list[dict] = []
    n_total = 0
    for run_id, run_path in iter_raw_runs(root):
        if args.limit is not None and run_id > args.limit - 1:
            break
        data = read_raw_run(root, run_id, verify=True)
        meta, arrays = data["metadata"], data["arrays"]
        # Timebases must match the deployed constants exactly.
        control_dt = float(meta["control_dt"])
        rom_dt = float(meta["rom_dt"])
        physics_dt = float(meta["physics_dt"])
        dt_issues = []
        if abs(control_dt - CONTROL_DT) > 1e-12:
            dt_issues.append(f"control_dt {control_dt} != {CONTROL_DT}")
        if abs(rom_dt - ROM_DT) > 1e-12:
            dt_issues.append(f"rom_dt {rom_dt} != {ROM_DT}")
        if abs(physics_dt - PHYSICS_DT) > 1e-12:
            dt_issues.append(f"physics_dt {physics_dt} != {PHYSICS_DT}")
        if dt_issues:
            raise SystemExit(f"run {run_id:05d} timebase mismatch: {dt_issues}")

        built = build_transitions(arrays, control_dt=control_dt, rom_dt=rom_dt)
        samples = built["samples"]
        grid = built["grid"]
        n = len(samples)
        n_total += n

        run_out = out / f"{run_id:05d}"
        run_out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            run_out / "samples.npz",
            tick_control_step=np.asarray([s["tick_control_step"] for s in samples], dtype=np.int64),
            input=np.stack([s["input"] for s in samples]),
            state_start=np.stack([s["state_start"] for s in samples]),
            state_end=np.stack([s["state_end"] for s in samples]),
            target=np.stack([s["target"] for s in samples]),
            command=np.stack([s["command"] for s in samples]),
            rom_delta_extension_sum=np.stack(
                [s["rom_delta_extension_sum"] for s in samples]
            ),
            cylinder_limit_flag=np.stack(
                [s["flags"]["cylinder_limit_flag"] for s in samples]
            ),
            command_saturation_flag=np.stack(
                [s["flags"]["command_saturation_flag"] for s in samples]
            ),
            valve_saturation_flag=np.stack(
                [s["flags"]["valve_saturation_flag"] for s in samples]
            ),
            invalid_physics_flag=np.stack(
                [s["flags"]["invalid_physics_flag"] for s in samples]
            ),
        )
        (run_out / "sample_meta.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "domain": meta["domain"],
                    "frozen_split": runs_info.get(f"{run_id:05d}", {}).get("frozen_split"),
                    "n_samples": n,
                    "grid": grid,
                    "timebase": {
                        "physics_dt": physics_dt,
                        "control_dt": control_dt,
                        "rom_dt": rom_dt,
                        "steps_per_rom_tick": STEPS_PER_ROM_TICK,
                    },
                    "no_interpolation": True,
                    "built_at": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            )
            + "\n"
        )
        per_run.append(
            {
                "run_id": run_id,
                "domain": meta["domain"],
                "frozen_split": runs_info.get(f"{run_id:05d}", {}).get("frozen_split"),
                "n_samples": n,
                "grid_uniform": grid["uniform"],
                "grid_issues": grid["issues"],
                "steps_per_rom_tick": STEPS_PER_ROM_TICK,
            }
        )
        print(
            f"run {run_id:05d} [{meta['domain']}]: {n} samples "
            f"(grid uniform={grid['uniform']})",
            flush=True,
        )

    (out / "index.json").write_text(
        json.dumps(
            {
                "frozen_root": str(root),
                "raw_checksums_digest": checksum_digest,
                "steps_per_rom_tick": STEPS_PER_ROM_TICK,
                "no_interpolation": True,
                "timebase": {"physics_dt": PHYSICS_DT, "control_dt": CONTROL_DT, "rom_dt": ROM_DT},
                "n_runs": len(per_run),
                "n_samples_total": n_total,
                "built_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
        + "\n"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "timebase": {
                    "physics_dt": PHYSICS_DT,
                    "control_dt": CONTROL_DT,
                    "rom_dt": ROM_DT,
                    "steps_per_rom_tick": STEPS_PER_ROM_TICK,
                    "transition": "state row t -> state row t+3 under command row t, "
                    "t on the rom_tick grid; exact raw rows, no interpolation",
                },
                "runs": per_run,
                "n_runs": len(per_run),
                "n_samples_total": n_total,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\n{len(per_run)} runs, {n_total} aligned 50 ms samples -> {out}")
    print(f"report -> {report_path}")


if __name__ == "__main__":
    main()
