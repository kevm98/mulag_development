#!/usr/bin/env python3
"""Derive Stage 31 SD transitions and causal AAM rows from frozen ROM data.

This is the only dataset builder used by the new ROM-source pipeline.  It
reads checksum-verified raw runs and writes two derived trees:

* ``sd/dataset``: hydraulic transitions with ``u=[previous_valve,current_valve]``;
* ``aam/rom_rows``: deployment-observable rows and the recorded reference and
  valve signals needed later by ID fine-tuning.

The AAM tree contains no factor or latent input.  Latent targets are added by
``build_stage31_aam_dataset.py`` only after the SD representation is frozen.
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
    FACTOR_KEYS,
    iter_raw_runs,
    load_runs_info,
    raw_dir,
    read_raw_run,
    sha256_file,
)
from aam_data.frozen import CHECKSUM_FILE  # noqa: E402
from aam_data.rom_stage31 import (  # noqa: E402
    AAM_FEATURE_DIM,
    AAM_FEATURE_GROUPS,
    AAM_HISTORY_LAGS,
    ROM_DT,
    build_transitions,
    causal_stack,
)


def reset_output(path: Path, force: bool) -> None:
    if path.exists() and any(path.iterdir()) and not force:
        raise SystemExit(f"output exists and is not empty: {path} (use --force)")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="frozen ROM collection")
    parser.add_argument("--sd-out", default=None)
    parser.add_argument("--aam-out", default=None)
    parser.add_argument("--lags", type=int, nargs="+", default=list(AAM_HISTORY_LAGS))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not (raw_dir(root) / "manifest.csv").is_file():
        raise SystemExit(f"not a frozen dataset: {root}")
    sd_out = Path(args.sd_out or root / "sd" / "dataset").expanduser().resolve()
    aam_out = Path(args.aam_out or root / "aam" / "rom_rows").expanduser().resolve()
    if sd_out == raw_dir(root) or raw_dir(root) in sd_out.parents:
        raise SystemExit(f"refusing to write SD data into raw/: {sd_out}")
    if aam_out == raw_dir(root) or raw_dir(root) in aam_out.parents:
        raise SystemExit(f"refusing to write AAM data into raw/: {aam_out}")
    reset_output(sd_out, args.force)
    reset_output(aam_out, args.force)
    lags = tuple(sorted(set(int(lag) for lag in args.lags)))
    if not lags or any(lag < 1 for lag in lags):
        raise SystemExit("--lags must contain positive integers")

    info = load_runs_info(root)
    source_digest = sha256_file(raw_dir(root) / CHECKSUM_FILE)
    per_run = []
    total = 0
    for run_id, _ in iter_raw_runs(root):
        run_id = int(run_id)
        if args.limit is not None and run_id >= args.limit:
            break
        loaded = read_raw_run(root, run_id, verify=True)
        metadata, arrays = loaded["metadata"], loaded["arrays"]
        transitions = build_transitions(arrays)
        if not transitions:
            raise SystemExit(f"run {run_id:05d}: no complete ROM transitions")
        s = np.stack([row["state_start"] for row in transitions])
        y = np.stack([row["target"] for row in transitions])
        q = np.stack([row["q"] for row in transitions])
        u = np.stack([row["u"] for row in transitions])
        features = np.stack([row["aam_feature"] for row in transitions])
        ticks = np.asarray([row["tick_control_step"] for row in transitions], dtype=np.int64)
        desired_x = np.stack([arrays["desired_cylinder_position"][tick] for tick in ticks])
        desired_v = np.stack([arrays["desired_cylinder_velocity"][tick] for tick in ticks])
        current_valve = np.stack([row["current_valve"] for row in transitions])
        previous_valve = np.stack([row["previous_valve"] for row in transitions])
        if features.shape[1] != AAM_FEATURE_DIM or u.shape[1] != 8:
            raise AssertionError((features.shape, u.shape))

        sd_run = sd_out / f"{run_id:05d}"
        sd_run.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            sd_run / "samples.npz",
            tick_control_step=ticks,
            s_h=s,
            target=y,
            q=q,
            u=u,
            input_instantaneous=np.concatenate((s, q, u), axis=1),
            cylinder_limit_flag=np.stack([arrays["cylinder_limit_flag"][tick] for tick in ticks]),
            command_saturation_flag=np.stack([arrays["command_saturation_flag"][tick] for tick in ticks]),
            valve_saturation_flag=np.stack([arrays["valve_saturation_flag"][tick] for tick in ticks]),
            invalid_physics_flag=np.stack([arrays["invalid_physics_flag"][tick] for tick in ticks]),
        )
        for lag in lags:
            history, valid = causal_stack(np.concatenate((s, u), axis=1), lag)
            np.savez_compressed(
                sd_run / f"history_L{lag}.npz",
                history_input=history,
                valid_mask=valid,
                window_end_tick=ticks[np.flatnonzero(valid)],
            )
        (sd_run / "sample_meta.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "domain": metadata["domain"],
                    "frozen_split": info[f"{run_id:05d}"]["frozen_split"],
                    "n_samples": int(len(transitions)),
                    "z": {key: float(metadata[key]) for key in FACTOR_KEYS},
                    "u_contract": "previous_valve_4 + current_valve_4",
                    "id_training_contract": "ID(state, desired_reference, previous_valve) -> recorded current_valve",
                    "target": "next hydraulic state minus current hydraulic state",
                    "history_lags": list(lags),
                },
                indent=2,
            )
            + "\n"
        )

        aam_run = aam_out / f"{run_id:05d}"
        aam_run.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            aam_run / "rom_rows.npz",
            features=features,
            tick_control_step=ticks,
            desired_cylinder_position=desired_x,
            desired_cylinder_velocity=desired_v,
            previous_valve=previous_valve,
            current_valve=current_valve,
            invalid_physics_flag=np.stack([arrays["invalid_physics_flag"][tick] for tick in ticks]),
        )
        (aam_run / "sample_meta.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "domain": metadata["domain"],
                    "frozen_split": info[f"{run_id:05d}"]["frozen_split"],
                    "n_ticks": int(len(ticks)),
                    "feature_dim": AAM_FEATURE_DIM,
                    "feature_groups": AAM_FEATURE_GROUPS,
                    "features": "joint_q, joint_qdot, dP, previous_valve",
                    "id_target": "rom_predicted_valve_cmd at this ROM tick",
                    "excluded_from_features": [
                        "physical factors", "true latent", "ROM hidden state",
                        "future observations", "desired cylinder reference",
                    ],
                },
                indent=2,
            )
            + "\n"
        )
        per_run.append(
            {
                "run_id": run_id,
                "domain": metadata["domain"],
                "frozen_split": info[f"{run_id:05d}"]["frozen_split"],
                "n_ticks": int(len(ticks)),
            }
        )
        total += len(ticks)
        print(f"run {run_id:05d}: {len(ticks)} exact ROM transitions", flush=True)

    if not per_run:
        raise SystemExit("no runs were derived")
    common = {
        "format": "stage31_rom_derived_v1",
        "frozen_root": str(root),
        "raw_checksums_digest": source_digest,
        "timebase": {"control_dt": 1.0 / 60.0, "rom_dt": ROM_DT, "stride": 3},
        "u_contract": "previous_valve_4 + current_valve_4",
        "id_training_contract": "ID(state, desired_reference, previous_valve) -> recorded current_valve",
        "aam_feature_dim": AAM_FEATURE_DIM,
        "aam_feature_groups": AAM_FEATURE_GROUPS,
        "aam_causal_input": "joint_q + joint_qdot + dP + previous_valve",
        "aam_lags": list(lags),
        "runs": per_run,
        "n_runs": len(per_run),
        "n_ticks_total": total,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
    }
    (sd_out / "index.json").write_text(json.dumps({**common, "kind": "sd"}, indent=2) + "\n")
    (aam_out / "index.json").write_text(json.dumps({**common, "kind": "aam_rows"}, indent=2) + "\n")
    print(f"derived {len(per_run)} runs/{total} ROM ticks from {root}")


if __name__ == "__main__":
    main()
