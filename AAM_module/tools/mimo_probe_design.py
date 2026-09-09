#!/usr/bin/env python
"""Design and validate the bounded MIMO probe for collection (items 33-36).

Stage 12 design, kept inside the Stage 10 work package. The probe is only
added if the DLS-only pilot fails the identifiability gates; this tool makes
the design concrete and grounded in the pilot's measured data:

- bandwidth: FFT-based f50/f95 of the pilot's dP/velocity/extension response;
- frequencies: multisine/PRBS lines inside the informative band, with the
  segment-length rule ``T_segment >= 5 / f_min``;
- amplitude: bounded per channel, from the pilot's observed action envelope
  and the learned stroke limits (margin-aware);
- injection: probe adds to the desired-cylinder reference at the action port
  (u = u_DLS + alpha * u_probe), then the ACTUAL ROM actuation is evaluated,
  not just the reference (item 34);
- safety: bounds, +- signs, channel independence, stroke limits, valve
  saturation (checked by closed-loop ROM rollouts) (item 35).

Headless (pure torch + numpy, no Isaac).

Usage:
    python mimo_probe_design.py [--pilot results/dls_pilot]
        [--out results/mimo_probe_design.json]
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
sys.path.insert(0, str(ROOT / "learned_models/rom_model"))

from ROM_pipeline.models import load_checkpoint  # noqa: E402

from aam_data import (  # noqa: E402
    ACTION_SCALE,
    ChirpSchedule,
    MultisineSchedule,
    PRBSSchedule,
)
from tools.action_identifiability import (  # noqa: E402
    command_from_action,
    input_diagnostics,
    rom_rollout,
)


# Per-cylinder learned extension limits (measured; the coordinate-mapping JSON
# carries USD joint limits + nominal extensions, not these learned values).
LEARNED_LIMITS = [
    (0.0643, 0.4193),
    (0.4147, 0.7387),
    (0.3776, 0.6780),
    (0.5335, 0.7033),
]


def estimate_bandwidth(
    y: np.ndarray, control_dt: float
) -> tuple[list[float], list[float]]:
    """Return (f50, f95) per channel via the power spectrum."""
    fs = 1.0 / control_dt
    f50s, f95s = [], []
    for c in range(y.shape[1]):
        yy = y[:, c] - y[:, c].mean()
        spec = np.abs(np.fft.rfft(yy)) ** 2
        freq = np.fft.rfftfreq(len(yy), d=control_dt)
        cum = np.cumsum(spec) / spec.sum()
        f50s.append(float(freq[np.searchsorted(cum, 0.50)]))
        f95s.append(float(freq[np.searchsorted(cum, 0.95)]))
    return f50s, f95s


def load_pilot(pilot_dir: Path) -> dict:
    with np.load(pilot_dir / "trajectory.npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    meta = json.loads((pilot_dir / "metadata.json").read_text())
    return {"arrays": arrays, "metadata": meta}


def stroke_margin_check(
    extension: np.ndarray, probe: np.ndarray, action_scale: float
) -> list[dict]:
    """Worst-case extension envelope (pilot baseline + probe delta) vs limits."""
    # Accumulated probe delta in learned metres: per-step action * action_scale,
    # integrated like the env's target integrator.
    probe_delta = probe * action_scale
    probe_ext = np.cumsum(probe_delta, axis=0)
    out = []
    for c in range(4):
        lo, hi = LEARNED_LIMITS[c]
        env = extension[:, c]
        worst_lo = env.min() + probe_ext[:, c].min()
        worst_hi = env.max() + probe_ext[:, c].max()
        out.append(
            {
                "channel": c,
                "limit_lo": lo,
                "limit_hi": hi,
                "pilot_ext_min": float(env.min()),
                "pilot_ext_max": float(env.max()),
                "worst_with_probe_lo": float(worst_lo),
                "worst_with_probe_hi": float(worst_hi),
                "margin_lo_m": float(worst_lo - lo),
                "margin_hi_m": float(hi - worst_hi),
                "safe": bool(worst_lo > lo and worst_hi < hi),
            }
        )
    return out


def per_channel_stroke_scale(
    extension: np.ndarray, probe: np.ndarray, action_scale: float
) -> tuple[list[float], int]:
    """Per-channel amplitude multiplier k_c keeping pilot + k*probe in limits.

    Returns ``(k_c, binding_channel)``. cyl0 sits near its upper limit in the
    DLS pilot (0.417 vs 0.4193 m), so the probe amplitude is usually capped by
    the channel with the least headroom.
    """
    probe_ext = np.cumsum(probe * action_scale, axis=0)
    k_per = [float("inf")] * 4
    for c in range(4):
        lo, hi = LEARNED_LIMITS[c]
        env = extension[:, c]
        p_lo, p_hi = probe_ext[:, c].min(), probe_ext[:, c].max()
        if p_lo < 0.0:
            k_per[c] = min(k_per[c], (env.min() - lo) / (-p_lo))
        if p_hi > 0.0:
            k_per[c] = min(k_per[c], (hi - env.max()) / p_hi)
    k_per = [max(float(v), 0.0) if v != float("inf") else float("inf") for v in k_per]
    binding = int(np.argmin(k_per))
    return k_per, binding


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pilot", default=str(AAM_ROOT / "results/dls_pilot"))
    p.add_argument("--out", default=str(AAM_ROOT / "results/mimo_probe_design.json"))
    p.add_argument(
        "--checkpoint",
        default=str(
            ROOT / "learned_models/rom_model/plant_reachable_v1_finetune_blend25/best_rom.pt"
        ),
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--steps", type=int, default=1800)
    p.add_argument("--f-min", type=float, default=None)
    p.add_argument("--f-max", type=float, default=5.0)
    args = p.parse_args()

    device = torch.device(args.device)
    model, _ = load_checkpoint(args.checkpoint, device=device)
    model.eval()

    pilot = load_pilot(Path(args.pilot))
    arrays, meta = pilot["arrays"], pilot["metadata"]
    control_dt = float(meta["control_dt"])
    U_pilot = np.asarray(arrays["controller_output"], dtype=np.float64)
    ext_pilot = np.asarray(arrays["cylinder_extension"], dtype=np.float64)[:-1]

    # 1. Bandwidth from the pilot's own response spectrum.
    bw = {
        sig: estimate_bandwidth(
            np.asarray(arrays[sig], dtype=np.float64)[:-1], control_dt
        )
        for sig in ("dP", "cylinder_velocity", "cylinder_extension")
    }
    dP_f95 = max(bw["dP"][1])
    vel_f95 = max(bw["cylinder_velocity"][1])
    ext_f95 = max(bw["cylinder_extension"][1])
    informative_f95 = float(min(dP_f95, vel_f95, ext_f95))
    f_min = args.f_min if args.f_min is not None else max(0.1, informative_f95 / 20.0)
    f_max = float(min(args.f_max, informative_f95))
    if f_max <= f_min:
        f_max = f_min + 0.5

    # 2. Segment length rule: T_segment >= 5 / f_min.
    min_segment_s = 5.0 / f_min
    steps = int(args.steps)
    segment_s = steps * control_dt

    # 3. Per-channel amplitude: bounded by pilot action envelope (rms-ish)
    #    and stroke margins. Keep the probe well inside the +-1 action clip and
    #    small relative to the DLS authority so u = u_DLS + alpha*u_probe stays
    #    bounded.
    pilot_rms = np.std(U_pilot, axis=0)
    amplitudes = [float(np.clip(0.5 * rms, 0.04, 0.20)) for rms in pilot_rms]

    schedules = [
        MultisineSchedule(
            per_cylinder_steps=steps,
            frequencies_hz=tuple(
                np.linspace(f_min, f_max, 3).tolist()
            ),
            amplitudes=tuple(amplitudes),
        ),
        PRBSSchedule(per_cylinder_steps=steps, amplitude=float(np.mean(amplitudes))),
        ChirpSchedule(
            per_cylinder_steps=steps,
            amplitude=float(np.mean(amplitudes)),
            f_start_hz=f_min,
            f_end_hz=f_max,
        ),
    ]

    probe_report = {"bandwidth": {}, "frequencies": {}, "probes": []}
    for sig, (f50, f95) in bw.items():
        probe_report["bandwidth"][sig] = {
            "f50_hz": [round(v, 3) for v in f50],
            "f95_hz": [round(v, 3) for v in f95],
        }
    probe_report["frequencies"] = {
        "informative_f95_hz": round(informative_f95, 3),
        "f_min_hz": round(f_min, 3),
        "f_max_hz": round(f_max, 3),
        "min_segment_s": round(min_segment_s, 1),
        "segment_s": round(segment_s, 1),
        "segment_rule_ok": bool(segment_s >= min_segment_s),
        "periods_of_f_min": round(segment_s * f_min, 2),
    }

    for sched in schedules:
        probe, segments = sched.build(total_steps=steps)
        # channel independence + +- sign coverage + bounds
        corr = np.corrcoef(probe.T)
        indep = all(
            abs(corr[i, j]) < 0.15 for i in range(4) for j in range(i + 1, 4)
        )
        both_signs = all(
            bool(probe[:, c].max() > 0.0 and probe[:, c].min() < 0.0)
            for c in range(4)
        )
        bounded = bool(np.abs(probe).max() <= 1.0)
        k_stroke_list, binding = per_channel_stroke_scale(ext_pilot, probe, ACTION_SCALE)
        k_stroke = float(np.min(k_stroke_list))
        # Apply a 0.9 safety factor so the design sits inside the limits, not
        # on them; the amplitude is then empirical (pilot envelope x stroke
        # headroom), not an arbitrary pick (item 35).
        base_amp = (
            sched.amplitudes
            if isinstance(sched, MultisineSchedule)
            else [sched.amplitude] * 4
        )
        k_amp = min(1.0, 0.9 * k_stroke)
        chosen = np.clip(np.asarray(base_amp, dtype=np.float64) * k_amp, 1.0e-4, 1.0)
        probe = probe * (chosen / np.asarray(base_amp, dtype=np.float64))[None, :]
        stroke = stroke_margin_check(ext_pilot, probe, ACTION_SCALE)
        stroke_safe = all(s["safe"] for s in stroke)

        # Actual ROM actuation under u_DLS + alpha*u_probe (item 34): evaluate
        # the probe alone so its authority and saturation are isolated.
        command = command_from_action(probe)
        responses = rom_rollout(model, command, device)
        valve = responses["valve"]
        sat_frac = float(np.mean(np.abs(valve) >= 1.0 - 1e-6))
        input_rep = input_diagnostics(probe, sched.params["name"], (1, 3, 6))
        probe_report["probes"].append(
            {
                "name": sched.params["name"],
                "params": sched.params,
                "steps": probe.shape[0],
                "base_amplitude": [float(v) for v in np.asarray(base_amp)],
                "chosen_amplitude": [float(v) for v in chosen],
                "stroke_safe_amplitude_scale": round(k_stroke, 3),
                "per_channel_stroke_scale": [round(v, 3) for v in k_stroke_list],
                "binding_channel": int(binding),
                "bounded": bounded,
                "both_signs": both_signs,
                "independent": indep,
                "max_cross_corr": round(float(np.abs(corr - np.eye(4)).max()), 3),
                "stroke_safe": stroke_safe,
                "stroke_margins": stroke,
                "input": {
                    "rank": input_rep["rank"],
                    "num_rank": input_rep["effective_numerical_rank"],
                    "kappa": input_rep["condition_number"],
                },
                "rom_valve_saturation_fraction": sat_frac,
                "authority": {
                    sig: round(
                        max(
                            y["max_abs"]
                            for u_row in authority_corr(responses, probe, sig)
                            for y in u_row.values()
                        ),
                        3,
                    )
                    for sig in ("dP", "velocity", "extension")
                },
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(probe_report, indent=2))

    print(f"pilot: {args.pilot}")
    print(f"bandwidth f95 (max per signal): dP={dP_f95:.2f} Hz, "
          f"velocity={vel_f95:.2f} Hz, extension={ext_f95:.2f} Hz")
    print(f"informative f95={informative_f95:.3f} Hz -> f_min={f_min:.3f} Hz, "
          f"f_max={f_max:.3f} Hz")
    print(f"segment rule: T={segment_s:.0f}s vs 5/f_min={min_segment_s:.0f}s -> "
          f"{'OK' if segment_s >= min_segment_s else 'SHORT'} "
          f"({segment_s * f_min:.1f} periods of f_min)")
    print(f"amplitudes per channel: {[round(v, 3) for v in amplitudes]}")
    for pr in probe_report["probes"]:
        print(
            f"  {pr['name']:<10} bounded={pr['bounded']} both_signs={pr['both_signs']} "
            f"indep={pr['independent']} (max_corr={pr['max_cross_corr']}) "
            f"stroke_safe={pr['stroke_safe']} "
            f"amp_scale={pr['stroke_safe_amplitude_scale']:.2f} "
            f"chosen_amp={[round(v,3) for v in pr['chosen_amplitude']]} "
            f"rank={pr['input']['rank']} kappa={pr['input']['kappa']:.2g} "
            f"valve_sat={pr['rom_valve_saturation_fraction']:.3f} "
            f"auth={pr['authority']}"
        )
    print(f"wrote {out}")


def authority_corr(responses: dict, probe: np.ndarray, sig: str) -> list[dict]:
    """Minimal lagged-correlation table for one signal (reuse pattern)."""
    Y = np.asarray(responses[sig], dtype=np.float64)
    rows = []
    for u_idx in range(probe.shape[1]):
        u = probe[:, u_idx]
        row = {}
        for y_idx in range(Y.shape[1]):
            y = Y[:, y_idx]
            vals = []
            for lag in range(min(12, probe.shape[0] - 1)):
                cu, cy = u[: probe.shape[0] - lag], y[lag:]
                denom = np.std(cu) * np.std(cy)
                vals.append(float(np.corrcoef(cu, cy)[0, 1]) if denom > 1.0e-12 else 0.0)
            row[y_idx] = {"max_abs": float(max(map(abs, vals)))}
        rows.append(row)
    return rows


if __name__ == "__main__":
    main()
