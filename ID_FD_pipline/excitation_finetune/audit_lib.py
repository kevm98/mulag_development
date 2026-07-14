"""Shared loader / schema / masking for the open-loop Simscape excitation CSV.

The CSV is exported with generic DATA_1..DATA_41 headers. The column mapping below is the
verified one supplied with the dataset. The raw CSV is never modified.

Key contract: dx / dxd are NOT in the CSV and must be differenced WITHIN a run. Nothing in
this module ever differences, shifts or windows across a run boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CSV = HERE / "Excitation_Finetune.csv"

# verified mapping, 1-indexed as given -> 0-indexed DATA_n
COLUMNS = (
    ["run_id", "draw_id", "mode", "exc_ch", "t", "valid"]
    + [f"u{i}" for i in range(1, 5)]
    + [f"x{i}" for i in range(1, 5)]
    + [f"xd{i}" for i in range(1, 5)]
    + [f"dP{i}" for i in range(1, 5)]
    + [f"F{i}" for i in range(1, 5)]
    + [f"q{i}" for i in range(1, 6)]
    + [f"qd{i}" for i in range(1, 6)]
    + ["u5"]
    + [f"ddP{i}" for i in range(1, 5)]
)
assert len(COLUMNS) == 41, len(COLUMNS)

U = [f"u{i}" for i in range(1, 5)]
X = [f"x{i}" for i in range(1, 5)]
XD = [f"xd{i}" for i in range(1, 5)]
DP = [f"dP{i}" for i in range(1, 5)]
FF = [f"F{i}" for i in range(1, 5)]
Q = [f"q{i}" for i in range(1, 6)]
QD = [f"qd{i}" for i in range(1, 6)]
DDP = [f"ddP{i}" for i in range(1, 5)]

DT = 0.01

# reason codes for the sample mask
REASONS = [
    "VALID_ACTIVE",
    "ZERO_INPUT_BASELINE",
    "INITIALIZATION_TRANSIENT",
    "ENDSTOP_EVENT",
    "PRESSURE_SPIKE",
    "VELOCITY_SPIKE",
    "NONPHYSICAL_DISCONTINUITY",
    "OUTSIDE_DEPLOYMENT_WORKSPACE",
]


def load() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    if list(df.columns)[:3] == ["DATA_1", "DATA_2", "DATA_3"]:
        df.columns = COLUMNS
    df["run_id"] = df["run_id"].astype(int)
    df["mode"] = df["mode"].astype(int)
    df["exc_ch"] = df["exc_ch"].astype(int)
    return df


def add_derivatives(df: pd.DataFrame) -> pd.DataFrame:
    """dx_i(k) = x_i(k+1) - x_i(k), dxd likewise. Last sample of each run -> NaN."""
    df = df.copy()
    g = df.groupby("run_id", sort=False)
    for i in range(1, 5):
        df[f"dx{i}"] = g[f"x{i}"].shift(-1) - df[f"x{i}"]
        df[f"dxd{i}"] = g[f"xd{i}"].shift(-1) - df[f"xd{i}"]
    df["last_of_run"] = g.cumcount(ascending=False) == 0
    return df


def run_type(sub: pd.DataFrame) -> str:
    """SISO_k / CH5 / MIMO, decided from the actual u channels, not from a name."""
    act = [c for c in U if float(sub[c].abs().max()) > 1e-9]
    if not act:
        return "CH5" if float(sub["u5"].abs().max()) > 1e-9 else "ZERO"
    if len(act) == 1:
        return f"SISO_{act[0]}"
    return "MIMO"


@dataclass
class Thresholds:
    """Descriptive, data-derived. Nothing is dropped by these -- they only label."""
    init_seconds: float = 0.10          # first 10 samples of a run
    dP_spike_Pa: float = 1.0e8          # |dP| > 100 MPa
    xd_spike_mps: float = 1.0           # |xd| > 1 m/s
    xd_jump_mps: float = 0.5            # |d(xd)| between consecutive samples
    x_endstop_margin: float = 1e-4      # stuck at an observed per-cylinder extreme


def build_masks(df: pd.DataFrame, th: Thresholds | None = None) -> pd.DataFrame:
    """One boolean column per reason code. A sample may carry several reasons."""
    th = th or Thresholds()
    m = pd.DataFrame(index=df.index)

    u_active = df[U].abs().max(axis=1) > 1e-9
    m["ZERO_INPUT_BASELINE"] = ~u_active

    t0 = df.groupby("run_id")["t"].transform("min")
    m["INITIALIZATION_TRANSIENT"] = (df["t"] - t0) < th.init_seconds

    m["PRESSURE_SPIKE"] = (df[DP].abs() > th.dP_spike_Pa).any(axis=1)
    m["VELOCITY_SPIKE"] = (df[XD].abs() > th.xd_spike_mps).any(axis=1)

    # non-physical discontinuity: velocity teleports between consecutive in-run samples
    g = df.groupby("run_id", sort=False)
    jump = pd.concat([(g[c].diff().abs() > th.xd_jump_mps) for c in XD], axis=1).any(axis=1)
    m["NONPHYSICAL_DISCONTINUITY"] = jump.fillna(False)

    # end-stop-like: extension pinned at a per-cylinder observed extreme while a valve pushes
    endstop = pd.Series(False, index=df.index)
    for i in range(1, 5):
        lo, hi = df[f"x{i}"].min(), df[f"x{i}"].max()
        pinned = (df[f"x{i}"] <= lo + th.x_endstop_margin) | (df[f"x{i}"] >= hi - th.x_endstop_margin)
        endstop |= pinned & (df[f"u{i}"].abs() > 0.05) & (df[f"xd{i}"].abs() < 1e-4)
    m["ENDSTOP_EVENT"] = endstop

    m["OUTSIDE_DEPLOYMENT_WORKSPACE"] = pd.Series(False, index=df.index)  # set by caller if known

    bad = (m["INITIALIZATION_TRANSIENT"] | m["PRESSURE_SPIKE"] | m["VELOCITY_SPIKE"]
           | m["NONPHYSICAL_DISCONTINUITY"] | m["ENDSTOP_EVENT"])
    m["VALID_ACTIVE"] = u_active & ~bad
    return m
