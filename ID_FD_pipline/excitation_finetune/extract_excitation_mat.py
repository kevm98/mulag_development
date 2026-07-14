#!/usr/bin/env python3
"""STRICT extractor: excitation MAT -> canonical 49-column CSV.

The MAT is authoritative. The existing 41-column CSV is an incomplete/outdated export: it
omits all four TargetDeltaExtension and all four TargetDeltaExtensionVelocity columns (its
last four columns are TargetDeltadP). It is preserved only for comparison.

Hard rules, enforced -- this script raises rather than guesses:
  * fields are addressed by EXPLICIT NAME only
  * NO largest_numeric_matrix heuristics
  * NO row-count voting to infer orientation or identity
  * NO differences taken across a run boundary, ever
  * every target column is validated against raw k+1 - k differences
  * exactly 49 named columns are written, in the canonical order below
  * the written CSV is round-tripped and compared back to the in-memory frame
  * any schema, shape, dtype or validation mismatch is a fatal error

Usage:
    python extract_excitation_mat.py --mat <path.mat> [--out Excitation_Finetune_49col.csv]
    python extract_excitation_mat.py --mat <path.mat> --list      # print MAT fields and exit
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

# ----------------------------------------------------------------- canonical schema
META = ["run_id", "draw_id", "mode", "exc_ch", "t", "valid"]
U4 = [f"u{i}" for i in range(1, 5)]
X4 = [f"x{i}" for i in range(1, 5)]
XD4 = [f"xd{i}" for i in range(1, 5)]
DP4 = [f"dP{i}" for i in range(1, 5)]
F4 = [f"F{i}" for i in range(1, 5)]
Q5 = [f"q{i}" for i in range(1, 6)]
QD5 = [f"qd{i}" for i in range(1, 6)]
U5 = ["u5"]
TDX4 = [f"TargetDeltaExtension{i}" for i in range(1, 5)]
TDXD4 = [f"TargetDeltaExtensionVelocity{i}" for i in range(1, 5)]
TDDP4 = [f"TargetDeltadP{i}" for i in range(1, 5)]

CANONICAL = META + U4 + X4 + XD4 + DP4 + F4 + Q5 + QD5 + U5 + TDX4 + TDXD4 + TDDP4
if len(CANONICAL) != 49:
    raise AssertionError(f"canonical schema must be 49 columns, got {len(CANONICAL)}")

# MAT field name -> canonical column(s). Explicit. Aliases are listed explicitly too;
# nothing is inferred from shape or size.
FIELD_MAP: dict[str, list[str]] = {
    "run_id": ["run_id"], "runID": ["run_id"], "RunID": ["run_id"],
    "draw_id": ["draw_id"], "drawID": ["draw_id"], "DrawID": ["draw_id"],
    "mode": ["mode"], "Mode": ["mode"],
    "exc_ch": ["exc_ch"], "excCh": ["exc_ch"], "ExcCh": ["exc_ch"],
    "t": ["t"], "time": ["t"], "Time": ["t"],
    "valid": ["valid"], "Valid": ["valid"],
    "u": U4, "ValveCmd": U4, "valve_cmd": U4,
    "x": X4, "Extension": X4, "extension": X4,
    "xd": XD4, "ExtensionVelocity": XD4, "extension_velocity": XD4,
    "dP": DP4, "dp": DP4,
    "F": F4, "Force": F4, "force": F4,
    "q": Q5, "JointPos": Q5, "joint_pos": Q5,
    "qd": QD5, "JointVel": QD5, "joint_vel": QD5,
    "u5": U5, "ValveCmd5": U5, "aux": U5,
    "TargetDeltaExtension": TDX4, "target_delta_extension": TDX4,
    "TargetDeltaExtensionVelocity": TDXD4, "target_delta_extension_velocity": TDXD4,
    "TargetDeltadP": TDDP4, "TargetDeltaDP": TDDP4, "target_delta_dp": TDDP4,
    "ddP": TDDP4,
}

TOL_ABS = 1e-6      # absolute tolerance for the target-vs-difference validation
TOL_REL = 1e-6


class SchemaError(RuntimeError):
    pass


def die(msg: str) -> None:
    raise SchemaError(msg)


# ----------------------------------------------------------------- MAT reading
def read_mat_fields(path: Path) -> dict[str, np.ndarray]:
    """Return {field_name: ndarray}. Explicit names only; no heuristics."""
    try:
        import h5py
        with h5py.File(path, "r") as f:
            out = {}
            for name, obj in f.items():
                if name.startswith("#"):
                    continue
                if isinstance(obj, h5py.Dataset):
                    out[name] = np.array(obj)
                else:  # a struct: read its members as "<struct>.<member>"
                    for m, o in obj.items():
                        if isinstance(o, h5py.Dataset):
                            out[f"{name}.{m}"] = np.array(o)
            return out
    except OSError:
        pass  # not HDF5 -> fall through to the v7 reader
    import scipy.io as sio
    raw = sio.loadmat(path, squeeze_me=False, struct_as_record=False)
    return {k: np.asarray(v) for k, v in raw.items() if not k.startswith("__")}


def orient(a: np.ndarray, n_rows: int, name: str) -> np.ndarray:
    """Return (n_rows, k). Orientation is decided by MATCHING n_rows EXACTLY -- and if both
    axes match we refuse to guess."""
    a = np.asarray(a)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if a.ndim != 2:
        die(f"field {name!r}: expected 1-D or 2-D, got shape {a.shape}")
    r, c = a.shape
    if r == n_rows and c == n_rows:
        die(f"field {name!r}: square {a.shape}; orientation ambiguous, refusing to guess")
    if r == n_rows:
        return a
    if c == n_rows:
        return a.T
    die(f"field {name!r}: neither axis equals the sample count {n_rows} (shape {a.shape})")


# ----------------------------------------------------------------- build frame
def build_frame(fields: dict[str, np.ndarray]) -> pd.DataFrame:
    present = {k: v for k, v in fields.items() if k in FIELD_MAP or k.split(".")[-1] in FIELD_MAP}
    if not present:
        die("no recognised field names in the MAT. Run with --list and extend FIELD_MAP "
            "explicitly; this script will not guess.")

    # sample count comes from the time vector, by name -- never from a row-count vote
    tkey = next((k for k in present if FIELD_MAP.get(k.split(".")[-1], FIELD_MAP.get(k)) == ["t"]),
                None)
    if tkey is None:
        die("no time field ('t'/'time'/'Time') found; cannot establish the sample count")
    n = int(np.asarray(present[tkey]).size)

    cols: dict[str, np.ndarray] = {}
    for key, arr in present.items():
        targets = FIELD_MAP.get(key) or FIELD_MAP[key.split(".")[-1]]
        a = orient(arr, n, key)
        if a.shape[1] != len(targets):
            die(f"field {key!r}: has {a.shape[1]} column(s) but schema expects "
                f"{len(targets)} ({targets})")
        for j, cname in enumerate(targets):
            if cname in cols:
                die(f"column {cname!r} supplied twice (duplicate/conflicting MAT fields)")
            cols[cname] = a[:, j].astype(np.float64)

    missing = [c for c in CANONICAL if c not in cols]
    if missing:
        die(f"MAT is missing {len(missing)} required column(s): {missing}\n"
            f"     got: {sorted(cols)}")
    extra = [c for c in cols if c not in CANONICAL]
    if extra:
        die(f"MAT produced unexpected column(s) not in the 49-column schema: {extra}")

    df = pd.DataFrame({c: cols[c] for c in CANONICAL})
    if df.shape != (n, 49):
        die(f"assembled frame is {df.shape}, expected ({n}, 49)")
    for c in ("run_id", "draw_id", "mode", "exc_ch", "valid"):
        df[c] = df[c].astype(np.int64)
    return df


# ----------------------------------------------------------------- validation
def validate(df: pd.DataFrame) -> dict:
    report: dict = {}
    if not np.isfinite(df.to_numpy(dtype=np.float64)).all():
        die("non-finite values (NaN/Inf) present in the extracted frame")

    # per-run structure -- indexing is per run_id, never a global reshape
    runs = df["run_id"].to_numpy()
    if (np.diff(runs) < 0).any():
        die("rows are not grouped/ordered by run_id")
    per_run = {}
    for rid, sub in df.groupby("run_id", sort=True):
        t = sub["t"].to_numpy()
        dt = np.diff(t)
        if len(dt) and not np.all(dt > 0):
            die(f"run {rid}: time is not strictly increasing")
        if len(np.unique(t)) != len(t):
            die(f"run {rid}: duplicate timestamps")
        per_run[int(rid)] = dict(n=int(len(sub)), dt=float(np.median(dt)) if len(dt) else None)
    report["runs"] = per_run

    # targets MUST equal the raw k+1 - k differences, computed WITHIN a run only.
    # The final sample of each run has no k+1 and is excluded from the comparison.
    checks = [(TDX4, X4, "TargetDeltaExtension"),
              (TDXD4, XD4, "TargetDeltaExtensionVelocity"),
              (TDDP4, DP4, "TargetDeltadP")]
    errs = {}
    for tgt_cols, src_cols, label in checks:
        worst = 0.0
        for rid, sub in df.groupby("run_id", sort=True):
            for tc, sc in zip(tgt_cols, src_cols):
                s = sub[sc].to_numpy()
                got = sub[tc].to_numpy()[:-1]
                want = s[1:] - s[:-1]
                tol = TOL_ABS + TOL_REL * np.abs(want)
                bad = np.abs(got - want) > tol
                if bad.any():
                    i = int(np.argmax(np.abs(got - want)))
                    die(f"{label}: run {rid} column {tc} does not match "
                        f"{sc}(k+1)-{sc}(k) at index {i}: got {got[i]!r}, want {want[i]!r}")
                worst = max(worst, float(np.abs(got - want).max()))
        errs[label] = worst
    report["target_validation_max_abs_err"] = errs
    return report


def round_trip(df: pd.DataFrame, path: Path) -> None:
    back = pd.read_csv(path)
    if list(back.columns) != CANONICAL:
        die(f"round-trip: header mismatch\n  wrote {CANONICAL}\n  read  {list(back.columns)}")
    if back.shape != df.shape:
        die(f"round-trip: shape mismatch, wrote {df.shape}, read {back.shape}")
    a = df.to_numpy(dtype=np.float64)
    b = back.to_numpy(dtype=np.float64)
    if not np.allclose(a, b, rtol=0, atol=1e-9, equal_nan=False):
        i, j = np.unravel_index(int(np.argmax(np.abs(a - b))), a.shape)
        die(f"round-trip: value mismatch at row {i}, column {CANONICAL[j]!r}: "
            f"{a[i, j]!r} != {b[i, j]!r} (max abs diff {np.abs(a - b).max():.3g})")


# ----------------------------------------------------------------- main
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mat", required=True, type=Path)
    p.add_argument("--out", type=Path, default=HERE / "Excitation_Finetune_49col.csv")
    p.add_argument("--list", action="store_true", help="print MAT fields and exit")
    p.add_argument("--compare_csv", type=Path, default=HERE / "Excitation_Finetune.csv",
                   help="the old 41-column export, preserved for comparison only")
    a = p.parse_args()

    if not a.mat.is_file():
        die(f"MAT not found: {a.mat}")
    fields = read_mat_fields(a.mat)

    if a.list:
        print(f"fields in {a.mat.name}:")
        for k, v in sorted(fields.items()):
            known = "  <- mapped" if (k in FIELD_MAP or k.split('.')[-1] in FIELD_MAP) else ""
            print(f"  {k:44s} shape={np.asarray(v).shape}{known}")
        return 0

    df = build_frame(fields)
    print(f"[OK] assembled {df.shape[0]} rows x {df.shape[1]} columns "
          f"({df['run_id'].nunique()} runs)", flush=True)

    rep = validate(df)
    print(f"[OK] targets validated against raw k+1-k differences (within-run only): "
          f"{rep['target_validation_max_abs_err']}", flush=True)

    df.to_csv(a.out, index=False)
    round_trip(df, a.out)
    print(f"[OK] round-tripped {a.out}", flush=True)

    # comparison against the old 41-column export -- reported, never merged
    if a.compare_csv.is_file():
        old = pd.read_csv(a.compare_csv)
        cmp = dict(old_shape=list(old.shape), new_shape=list(df.shape),
                   old_is_41col=bool(old.shape[1] == 41),
                   columns_added=[c for c in CANONICAL if c in TDX4 + TDXD4])
        # do the state blocks differ per run in the NEW export? (the old one had only 12
        # distinct state trajectories across 24 runs -- duplicated SISO blocks)
        import hashlib
        state = X4 + XD4 + DP4 + F4 + Q5 + QD5
        h = {int(r): hashlib.md5(np.ascontiguousarray(
            s[state].to_numpy())).hexdigest() for r, s in df.groupby("run_id")}
        cmp["distinct_state_trajectories"] = len(set(h.values()))
        cmp["n_runs"] = len(h)
        if cmp["distinct_state_trajectories"] != cmp["n_runs"]:
            dup: dict[str, list[int]] = {}
            for r, v in h.items():
                dup.setdefault(v, []).append(r)
            die("EXPORT DEFECT REPRODUCED: only "
                f"{cmp['distinct_state_trajectories']} distinct state trajectories across "
                f"{cmp['n_runs']} runs. Runs sharing a state block: "
                f"{[v for v in dup.values() if len(v) > 1]}. Different valve excitations "
                "cannot produce bit-identical states; the MAT/export is still wrong.")
        print(f"[OK] {cmp['distinct_state_trajectories']}/{cmp['n_runs']} distinct state "
              f"trajectories -- no duplicated state blocks", flush=True)
        (HERE / "excitation_49col_vs_old41col.json").write_text(json.dumps(cmp, indent=2))

    (HERE / "excitation_49col_schema.json").write_text(json.dumps(dict(
        source_mat=str(a.mat), columns=CANONICAL, n_columns=49,
        rows=int(df.shape[0]), runs=int(df["run_id"].nunique()),
        validation=rep,
        note="MAT is authoritative. The 41-column CSV omitted TargetDeltaExtension(1:4) and "
             "TargetDeltaExtensionVelocity(1:4); its last four columns are TargetDeltadP.",
    ), indent=2, default=str))
    print(f"[OK] wrote excitation_49col_schema.json", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SchemaError as e:
        print(f"\n[FATAL] {e}", file=sys.stderr)
        sys.exit(2)
