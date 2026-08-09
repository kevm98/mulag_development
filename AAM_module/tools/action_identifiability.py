#!/usr/bin/env python
"""Input-rank and actuator-authority identifiability diagnostics (Stage 10).

Implements items 28-31 of the collection design spec:

- 28/29: build the plant action matrix ``U`` (T x 4) from the Stage 10
  trajectory set, then report rank, effective numerical rank, singular
  values, condition number, and cross-channel correlation.
- 30: temporal persistent excitation -- stack lagged actions into
  ``U_L = [u_t, u_{t-1}, ..., u_{t-L+1}]`` and report the same spectrum.
- 31: actuator-to-response authority -- drive the deployed command ROM
  closed-loop (pure torch, no Isaac) with the reference actions and measure
  ``u -> dP``, ``u -> xdot``, ``u -> x`` via lagged correlations, local
  gains, regression sensitivities, and input-output singular values.

The authority path reuses the ROM rollout pattern from
``tools/rom_sensitivity.py`` (history-6 GRU, 20 Hz ticks, states integrated
exactly as the runtime does).

Usage:
    python action_identifiability.py [--checkpoint PATH] [--lags 1 3 6 12]
        [--out results/action_identifiability.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
AAM_ROOT = Path(__file__).resolve().parents[1]
if str(AAM_ROOT) not in sys.path:
    sys.path.insert(0, str(AAM_ROOT))
sys.path.insert(0, str(ROOT / "rom_model"))

from ROM_pipeline.models import load_checkpoint  # noqa: E402

from aam_data import (  # noqa: E402
    ACTION_SCALE,
    ROM_DT,
    annotate_coverage,
    compact_set,
)


def input_diagnostics(U: np.ndarray, label: str, lags: tuple[int, ...]) -> dict:
    """Rank/SVD/condition number/correlation for U and its lagged matrices."""
    U = np.asarray(U, dtype=np.float64)
    report = {"label": label, "shape": list(U.shape)}
    sv = np.linalg.svd(U, compute_uv=False)
    sv = sv[sv > 1.0e-12]
    kappa = float(sv[0] / sv[-1]) if sv.size else float("inf")
    tol = max(U.shape) * np.finfo(np.float64).eps
    num_rank = int(np.count_nonzero(sv > sv[0] * max(tol, 1.0e-9)))
    report["rank"] = int(np.linalg.matrix_rank(U))
    report["effective_numerical_rank"] = num_rank
    report["singular_values"] = [float(v) for v in sv]
    report["condition_number"] = kappa
    if U.shape[1] > 1:
        corr = np.full((U.shape[1], U.shape[1]), np.nan)
        for i in range(U.shape[1]):
            for j in range(U.shape[1]):
                si, sj = np.std(U[:, i]), np.std(U[:, j])
                if si > 1.0e-12 and sj > 1.0e-12:
                    corr[i, j] = float(np.corrcoef(U[:, i], U[:, j])[0, 1])
        report["cross_channel_correlation"] = [
            [float(v) for v in row] for row in corr
        ]
    for L in lags:
        if U.shape[0] <= L:
            continue
        UL = np.stack(
            [U[L - 1 - k : U.shape[0] - k] for k in range(L)], axis=-1
        ).reshape(U.shape[0] - L + 1, U.shape[1] * L)
        svL = np.linalg.svd(UL, compute_uv=False)
        svL = svL[svL > 1.0e-12]
        kappaL = float(svL[0] / svL[-1]) if svL.size else float("inf")
        tolL = max(UL.shape) * np.finfo(np.float64).eps
        num_rankL = int(np.count_nonzero(svL > svL[0] * max(tolL, 1.0e-9)))
        report[f"lagged_L{L}"] = {
            "shape": list(UL.shape),
            "rank": int(np.linalg.matrix_rank(UL)),
            "effective_numerical_rank": num_rankL,
            "singular_values": [float(v) for v in svL],
            "condition_number": kappaL,
        }
    return report


def command_from_action(action: np.ndarray) -> np.ndarray:
    """Map a normalized 4-D action to the ROM 8-D command, per ROM tick.

    Follows the deployed adapter: ``target_delta_x = action * action_scale``
    (learned metres per 50 ms tick) and ``target_delta_xdot = delta_x / dt``
    (the cmd-only feedforward; the runtime subtracts measured xdot, which the
    closed-loop rollout provides as its own state).
    """
    action = np.asarray(action, dtype=np.float64)
    delta_x = action * ACTION_SCALE
    delta_xdot = delta_x / ROM_DT
    return np.concatenate([delta_x, delta_xdot], axis=-1)


def rom_rollout(model, command: np.ndarray, device: torch.device) -> dict[str, np.ndarray]:
    """Closed-loop ROM rollout: feed back state + predicted valve each tick."""
    ticks = command.shape[0]
    x = torch.zeros(1, 4, device=device) - 0.5
    xdot = torch.zeros(1, 4, device=device)
    dP = torch.zeros(1, 4, device=device)
    prev_valve = torch.zeros(1, 4, device=device)
    history = torch.zeros(1, 6, model.config.input_dim, device=device)
    traj = {"dP": [], "velocity": [], "extension": [], "valve": []}
    with torch.no_grad():
        for t in range(ticks):
            cmd = torch.as_tensor(command[t], device=device, dtype=torch.float32).reshape(1, -1)
            row = torch.cat((x, xdot, dP, prev_valve, cmd), dim=-1)
            history = torch.cat((history[:, 1:], row.unsqueeze(1)), dim=1)
            out = model(history, cmd)
            dP = out.next_pressure.clone()
            xdot = out.next_velocity.clone()
            x = out.next_extension.clone()
            prev_valve = out.predicted_valve.clone()
            traj["dP"].append(dP[0].detach().cpu().numpy())
            traj["velocity"].append(xdot[0].detach().cpu().numpy())
            traj["extension"].append(x[0].detach().cpu().numpy())
            traj["valve"].append(prev_valve[0].detach().cpu().numpy())
    return {k: np.stack(v) for k, v in traj.items()}


def authority_diagnostics(
    U: np.ndarray, responses: dict[str, np.ndarray], lags: tuple[int, ...]
) -> dict:
    """Measure u -> dP / xdot / x: correlations, gains, sensitivities, IO SVD."""
    U = np.asarray(U, dtype=np.float64)
    out: dict = {}
    n_u = U.shape[1]
    L_max = max(lags)
    for sig_name, Y in responses.items():
        Y = np.asarray(Y, dtype=np.float64)
        n_y = Y.shape[1]
        sig = {"lagged_correlation": [], "local_gain": [], "regression_sensitivity": []}
        for u_idx in range(n_u):
            u = U[:, u_idx]
            row = {}
            for y_idx in range(n_y):
                y = Y[:, y_idx]
                vals = []
                for lag in range(min(L_max, U.shape[0] - 1)):
                    cu = u[: U.shape[0] - lag]
                    cy = y[lag:]
                    denom = np.std(cu) * np.std(cy)
                    vals.append(float(np.corrcoef(cu, cy)[0, 1]) if denom > 1.0e-12 else 0.0)
                row[y_idx] = {
                    "max_abs": float(max(map(abs, vals))),
                    "argmax_lag": int(np.argmax(np.abs(vals))),
                    "series": [float(v) for v in vals],
                }
            sig["lagged_correlation"].append(row)
            # local gain: median slope of y on u over a sliding window
            w = 60
            gains = []
            for y_idx in range(n_y):
                y = Y[:, y_idx]
                slopes = []
                for k in range(0, U.shape[0] - w, w // 2):
                    du = np.diff(u[k : k + w])
                    dy = np.diff(y[k : k + w])
                    d = du.reshape(-1, 1)
                    if d.shape[0] >= 8 and np.linalg.norm(d) > 1.0e-12:
                        slopes.append(float(np.linalg.lstsq(d, dy, rcond=None)[0][0]))
                gains.append(
                    {
                        "median": float(np.median(slopes)) if slopes else 0.0,
                        "n_windows": len(slopes),
                    }
                )
            sig["local_gain"].append(gains)
            # regression sensitivity: OLS y ~ [u_lagged] coefficient norms
            L = L_max
            n_rows = U.shape[0] - L + 1
            if n_rows > L * n_u:
                X = np.stack(
                    [U[L - 1 - k : U.shape[0] - k] for k in range(L)], axis=-1
                ).reshape(n_rows, n_u * L)
                Xc = np.column_stack([X, np.ones(n_rows)])
                coefs = np.linalg.lstsq(Xc, Y[:n_rows], rcond=None)[0][:-1].reshape(n_u, L, n_y)
                sens = np.linalg.norm(coefs, axis=(1, 2))
                sig["regression_sensitivity"].append([float(v) for v in sens])
        out[sig_name] = sig
    # input-output singular values: SVD of the stacked gain matrix. Output
    # columns are normalized to unit variance so Pa-scale dP does not dwarf
    # velocity/extension; the spectrum is then a per-direction authority ratio.
    Y_all = np.concatenate(
        [responses[k] for k in ("dP", "velocity", "extension")], axis=-1
    )
    Y_all = Y_all / np.std(Y_all, axis=0).clip(min=1.0e-12)
    L = L_max
    n_rows = U.shape[0] - L + 1
    if n_rows > L * n_u:
        X = np.stack([U[L - 1 - k : U.shape[0] - k] for k in range(L)], axis=-1).reshape(
            n_rows, n_u * L
        )
        gain = np.linalg.lstsq(X, Y_all[:n_rows], rcond=None)[0]  # (4L, 12)
        sv = np.linalg.svd(gain, compute_uv=False)
        out["input_output_singular_values"] = [float(v) for v in sv]
        out["io_condition_number"] = float(sv[0] / sv[-1])
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint",
        default=str(
            ROOT / "rom_model/plant_reachable_v1_finetune_blend25/best_rom.pt"
        ),
    )
    p.add_argument("--lags", type=int, nargs="+", default=(1, 3, 6, 12))
    p.add_argument("--out", default=str(ROOT / "AAM_module/results/action_identifiability.json"))
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    model, _ = load_checkpoint(args.checkpoint, device=device)
    model.eval()

    trajectories = [annotate_coverage(t) for t in compact_set()]
    all_actions = np.concatenate([t.action for t in trajectories], axis=0)

    report = {
        "checkpoint": str(args.checkpoint),
        "lags": list(args.lags),
        "trajectories": [],
        "full_set": input_diagnostics(all_actions, "full_compact_set", tuple(args.lags)),
    }
    for t in trajectories:
        action_stats = {
            "rms": [float(v) for v in np.std(t.action, axis=0)],
            "max_abs": [float(v) for v in np.abs(t.action).max(axis=0)],
        }
        entry = {
            "name": t.name,
            "source": t.source,
            "num_steps": t.num_steps,
            "arc_length_m": round(t.arc_length_m, 4),
            "action_stats": action_stats,
            "coverage": t.coverage,
            "input": input_diagnostics(t.action, t.name, tuple(args.lags)),
        }
        command = command_from_action(t.action)
        responses = rom_rollout(model, command, device)
        entry["authority"] = authority_diagnostics(t.action, responses, tuple(args.lags))
        report["trajectories"].append(entry)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")

    print(f"checkpoint: {report['checkpoint']}")
    fs = report["full_set"]
    print(
        f"full compact set U: shape={fs['shape']} rank={fs['rank']} "
        f"numerical_rank={fs['effective_numerical_rank']} kappa={fs['condition_number']:.3g}"
    )
    print("singular values:", [f"{v:.4g}" for v in fs["singular_values"]])
    for L in args.lags:
        lag = fs.get(f"lagged_L{L}")
        if lag:
            print(
                f"  U_L L={L}: shape={lag['shape']} rank={lag['rank']} "
                f"numerical_rank={lag['effective_numerical_rank']} kappa={lag['condition_number']:.3g}"
            )
    print("\nper-trajectory input rank:")
    for t in report["trajectories"]:
        inp = t["input"]
        cov = t["coverage"]
        ax = t["action_stats"]
        print(
            f"  {t['name']:<22} rank={inp['rank']} num_rank={inp['effective_numerical_rank']} "
            f"kappa={inp['condition_number']:.3g} "
            f"dominant_cyl={cov['dominant_cylinder']} frac={cov['dominant_span_frac']} "
            f"reversals={cov['direction_reversals']} "
            f"max|u|={max(ax['max_abs']):.3f}"
        )
    print("\nauthority (max abs lagged correlation, argmax lag):")
    for t in report["trajectories"]:
        auth = t["authority"]
        line = [f"  {t['name']:<22}"]
        for sig in ("dP", "velocity", "extension"):
            best = 0.0
            for u_row in auth[sig]["lagged_correlation"]:
                for y_row in u_row.values():
                    best = max(best, y_row["max_abs"])
            line.append(f"{sig}={best:.2f}")
        io = auth.get("input_output_singular_values")
        if io:
            line.append("io_sv=[{}]".format(", ".join(f"{v:.3g}" for v in io[:6])))
        print(" ".join(line))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
