#!/usr/bin/env python
"""ROM-level sensitivity of the deployed command ROM to candidate hydraulic factors.

Pure PyTorch (no Isaac). Runs the same deterministic command trajectory under
nominal, +delta and -delta perturbations of one factor at a time, rolling the
ROM's own pressure/velocity/extension states forward (exactly how the runtime
integrates them). Reports peak and mean |delta y| and a normalized sensitivity
S_j = mean_t |y(+) - y(-)| / (2*delta) per signal.

Usage:
    python rom_sensitivity.py [--checkpoint PATH] [--ticks 40]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rom_model"))

from ROM_pipeline.models import load_checkpoint  # noqa: E402


def safe_trajectory(ticks: int, device: torch.device) -> torch.Tensor:
    """Deterministic staggered trapezoid command, one cylinder at a time.

    Returns [ticks, 1, 8] = cat(target_delta_x, target_delta_xdot).
    Each cylinder steps +1 mm/tick for 8 ticks, holds 4, returns, then the next
    cylinder starts. Amplitude stays inside ROM training authority.
    """
    command = torch.zeros(ticks, 1, 8, device=device)
    per_cyl = max(ticks // 5, 1)
    for c in range(4):
        start = c * per_cyl
        delta = 1.0e-3  # 1 mm per 50 ms tick
        xdot = delta / 0.05
        for t in range(start, min(start + per_cyl, ticks)):
            phase = t - start
            if phase < 8:
                value = delta
            elif phase < 12:
                value = 0.0
            elif phase < 20:
                value = -delta
            else:
                value = 0.0
            command[t, 0, c] = value
            command[t, 0, 4 + c] = xdot if value != 0.0 else 0.0
    return command


def rollout(model, command: torch.Tensor, *, system_factors=None, param_pert=None):
    """Closed-loop ROM rollout: feed back next state and predicted valve."""
    ticks = command.shape[0]
    device = next(model.parameters()).device
    x = torch.zeros(1, 4, device=device) - 0.5
    xdot = torch.zeros(1, 4, device=device)
    dP = torch.zeros(1, 4, device=device)
    prev_valve = torch.zeros(1, 4, device=device)
    history = torch.zeros(1, 6, model.config.input_dim, device=device)
    traj = {"dP": [], "velocity": [], "extension": []}
    with torch.no_grad():
        for t in range(ticks):
            row = torch.cat((x, xdot, dP, prev_valve, command[t]), dim=-1)
            history = torch.cat((history[:, 1:], row.unsqueeze(1)), dim=1)
            if param_pert is not None:
                state = model.state_dict()
                for key, factor in param_pert.items():
                    state[key] = state[key] * factor
                model.load_state_dict(state)
            out = model(history, command[t], system_factors=system_factors)
            dP = out.next_pressure.clone()
            xdot = out.next_velocity.clone()
            x = out.next_extension.clone()
            prev_valve = out.predicted_valve.clone()
            traj["dP"].append(dP)
            traj["velocity"].append(xdot)
            traj["extension"].append(x)
    return {k: torch.stack(v) for k, v in traj.items()}


def peak_rel(sig_nom, sig_pert):
    """Peak absolute deviation of perturbed vs nominal, relative to nominal range."""
    base = (sig_nom - sig_nom[0]).abs().amax().clamp_min(1e-12)
    peak = (sig_pert - sig_nom).abs().amax()
    return peak, (peak / base).item()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=str(ROOT / "rom_model/plant_reachable_v1_finetune_blend25/best_rom.pt"))
    p.add_argument("--ticks", type=int, default=80)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    model, _ = load_checkpoint(args.checkpoint, device=device)
    model.eval()
    command = safe_trajectory(args.ticks, device)

    def system_factors(flow=0.0, compliance=0.0, pressure=0.0, seal=0.0) -> torch.Tensor:
        return torch.tensor([[compliance, flow, pressure, seal, seal, seal, seal]], device=device)

    # factor -> (label, callable(plus/minus) -> rollout, delta)
    factors = {
        "flow_gain": ("system flow_factor +-0.10", lambda d: system_factors(flow=d), 0.10),
        "bulk_modulus": ("system compliance +-log(1.1)", lambda d: system_factors(compliance=d), 0.0953),
        "leakage_seal": ("system seal[4] +-0.20", lambda d: system_factors(seal=d), 0.20),
        "supply_pressure": ("system pressure_factor +-0.10", lambda d: system_factors(pressure=d), 0.10),
    }

    def run_friction(scale: float):
        sd = model.state_dict()
        sd["log_viscous_friction"] = sd["log_viscous_friction"] + torch.log(torch.tensor(scale, device=device))
        model.load_state_dict(sd)
        out = rollout(model, command)
        sd = model.state_dict()
        sd["log_viscous_friction"] = sd["log_viscous_friction"] - torch.log(torch.tensor(scale, device=device))
        model.load_state_dict(sd)
        return out

    def run_deadzone(shift: float):
        sd = model.state_dict()
        sd["deadzone.raw_width"] = sd["deadzone.raw_width"] + shift
        model.load_state_dict(sd)
        out = rollout(model, command)
        sd = model.state_dict()
        sd["deadzone.raw_width"] = sd["deadzone.raw_width"] - shift
        model.load_state_dict(sd)
        return out

    factors["viscous_friction"] = ("state_dict x1.10 / x0.90", None, 0.10)
    factors["deadzone_width"] = ("state_dict raw_width +-0.30", None, 0.30)

    nominal = rollout(model, command)
    rows = []
    for name, (mechanism, _, delta) in factors.items():
        if mechanism.startswith("system"):
            plus = rollout(model, command, system_factors=factors[name][1](delta))
            minus = rollout(model, command, system_factors=factors[name][1](-delta))
        elif name == "viscous_friction":
            plus = run_friction(1.0 + delta)
            minus = run_friction(1.0 - delta)
        else:
            plus = run_deadzone(delta)
            minus = run_deadzone(-delta)

        S, rel = {}, {}
        for signal in ("dP", "velocity", "extension"):
            peak, relval = peak_rel(nominal[signal], plus[signal])
            peak_minus, _ = peak_rel(nominal[signal], minus[signal])
            denom = 2 * delta
            S[signal] = ((plus[signal] - minus[signal]).abs().mean() / denom).item()
            rel[signal] = max(relval, (peak_minus / max((nominal[signal] - nominal[signal][0]).abs().amax(), 1e-12)).item())
        rows.append((name, mechanism, S, rel))

    print(f"ROM sensitivity over {args.ticks} ticks (trajectory: staggered +-1 mm steps per cylinder)")
    print(f"{'factor':<16}{'mechanism':<26}{'S_dP':>10}{'S_vel':>10}{'S_ext':>10}   rel_dP rel_vel rel_ext")
    for name, mechanism, S, rel in rows:
        print(f"{name:<16}{mechanism:<26}{S['dP']:>10.3e}{S['velocity']:>10.3e}{S['extension']:>10.3e}   "
              f"{rel['dP']*100:6.1f}% {rel['velocity']*100:6.1f}% {rel['extension']*100:6.1f}%")
    print("\nrelative = peak |y_pert - y_nom| / nominal signal range")


if __name__ == "__main__":
    main()
