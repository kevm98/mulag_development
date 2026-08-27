#!/usr/bin/env python
"""E5 Stage-0b: is effective bulk modulus a usable ROM degradation axis?

CPU-only, no Isaac. Drives the deployed command ROM through its real deployment
bridge (`CommandCylinderROMRuntime`, the same class `mulag_upper_level_env` uses
in its `command_rom` branch) on one deterministic command trajectory, while the
effective bulk modulus beta is moved off nominal.  Reports how far the ROM's
dP / velocity / realised extension move, and whether the requested stress
actually arrived.

Why this is not the `system_factors` compliance channel
-------------------------------------------------------
`HydraulicROM.forward` pins the compliance factor e_comp to [-0.25, 0.10] and
applies beta <- beta * exp(e_comp).  That path can only deliver 0.779x..1.105x
of nominal beta -- it cannot express the 0x / 0.5x / 2.0x grid at all.  E5's
withdrawn sweep is exactly the failure this causes.  So this sweep writes
`log_bulk_modulus` directly and then reports the *delivered* post-clamp beta,
so a flat curve can never be confused with an undelivered one.

Two grids:
  * scale    -- multiples of the checkpoint's own learned per-cylinder beta
                (E5's 0/0.5/1/1.5/2 convention, nominal = 1.0)
  * absolute -- beta in GPa, the same numbers read as physical stiffness

Plant surrogate
---------------
No PhysX here, so the plant is a perfect follower: the measured extension
integrates the ROM's realised delta_extension and is clamped to the checkpoint's
cylinder limits.  Crucially the measured *velocity* is differentiated from that
measured extension, not carried forward from the ROM's own integrator -- the
deployed env passes `reference.xdot_current` (a plant measurement), and feeding
the ROM its own unanchored velocity instead lets the rollout run away.

Two guard modes, reported separately:
  * guard on  -- `command_rom_direction_guard=True`, what the deployment task
                 configs actually set (`mulag_plant/policy/env_cfg.py`,
                 `verge_task/env_cfg.py`).  Clamps realised motion to
                 3x|command| and zeros any wrong-direction response.
  * guard off -- the raw ROM response.
This separates "beta does not move the ROM" from "beta moves the ROM but the
guard removes it before the plant sees it", which is the distinction E5 says a
null result must make.

Usage:
    python3.11 experiments/E5/code/rom_bulk_modulus_sweep.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rom_model"))

from mulag_plant.runtime.command_rom_runtime import CommandCylinderROMRuntime  # noqa: E402

BETA_FLOOR_PA = 1.0e7  # HydraulicROM.forward clamps beta to [1e7, 3e9]
BETA_CEIL_PA = 3.0e9
DP_ABS_LIMIT_PA = 1.5e8  # command_rom_dp_abs_limit_pa
COMMAND_MAX_RATIO = 3.0  # command_rom_command_max_ratio
DEFAULT_SCALES = (0.0, 0.5, 1.0, 1.5, 2.0)
DEFAULT_ABSOLUTE_GPA = (0.0, 0.5, 1.0, 1.5, 2.0)


def safe_trajectory(ticks: int, device: torch.device) -> torch.Tensor:
    """Deterministic staggered trapezoid command, one cylinder at a time.

    Returns [ticks, 1, 8] = cat(target_delta_x, target_delta_xdot).  Amplitude
    stays inside ROM training authority.  Same shape as the trajectory used by
    `AAM_module/tools/rom_sensitivity.py`, so the two screens are comparable.
    """
    command = torch.zeros(ticks, 1, 8, device=device)
    per_cyl = max(ticks // 4, 1)
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


class OutputTap:
    """Capture each `HydraulicROM.forward` result.

    `CommandROMStepOutput` does not carry the residual heads, but the flat
    extension curve can only be attributed if we can see how much of the
    increment is physics and how much is the learned residual running in
    parallel with it.  A forward hook is the least invasive way to get it.
    """

    def __init__(self, model) -> None:
        self.rows: list = []
        self._handle = model.register_forward_hook(lambda _m, _i, out: self.rows.append(out))

    def clear(self) -> None:
        self.rows.clear()

    def close(self) -> None:
        self._handle.remove()

    def shares(self, dt: float, velocities: torch.Tensor) -> tuple[float, float]:
        """Residual share of the dP increment and of the extension increment."""
        if not self.rows:
            return float("nan"), float("nan")
        dp = torch.stack([r.delta_pressure for r in self.rows])
        dp_res = torch.stack([r.pressure_residual for r in self.rows])
        dx = torch.stack([r.delta_extension for r in self.rows])
        dv = torch.stack([r.delta_velocity for r in self.rows])
        # physics half of delta_extension is 0.5*(v + v_next)*dt
        dx_phys = (velocities + 0.5 * dv) * dt
        share_dp = (dp_res.abs().mean() / dp.abs().mean().clamp_min(1e-12)).item()
        share_dx = ((dx - dx_phys).abs().mean() / dx.abs().mean().clamp_min(1e-18)).item()
        return share_dp, share_dx


def rollout(runtime, command: torch.Tensor, limits: torch.Tensor, *, guard: bool, tap=None) -> dict[str, torch.Tensor]:
    """Deployment-faithful rollout: ROM -> guard -> perfect-follower plant."""
    device = runtime.device
    dt = runtime.sample_time_s
    lower, upper = limits[:, 0].unsqueeze(0), limits[:, 1].unsqueeze(0)
    ratio = COMMAND_MAX_RATIO / float(max(runtime.steps_per_tick, 1))

    runtime.reset()
    if tap is not None:
        tap.clear()
    x = (0.5 * (lower + upper)).to(device)  # start mid-stroke, inside every limit
    x0 = x.clone()
    velocity = torch.zeros(1, 4, device=device)
    dP = torch.zeros(1, 4, device=device)

    traj: dict[str, list[torch.Tensor]] = {
        "dP": [], "velocity": [], "extension": [], "velocity_in": [],
        "delta_raw": [], "delta_applied": [], "limit_clipped": [], "guard_vetoed": [],
    }
    for t in range(command.shape[0]):
        traj["velocity_in"].append(velocity.clone())
        out = runtime.step(
            desired_cylinder_command=command[t],
            extension=x,
            extension_velocity=velocity,
            dP=dP,
        )
        delta_raw = out.delta_extension.clone()
        delta = delta_raw
        vetoed = torch.zeros_like(delta, dtype=torch.bool)
        if guard:
            command_delta = command[t][:, :4]
            same_direction = delta * command_delta > 0.0
            limit = ratio * command_delta.abs()
            delta = torch.where(
                same_direction,
                torch.sign(command_delta) * torch.minimum(delta.abs(), limit),
                torch.zeros_like(delta),
            )
            vetoed = ~same_direction

        dP = torch.clamp(dP + out.delta_dP, -DP_ABS_LIMIT_PA, DP_ABS_LIMIT_PA)
        proposed = x + delta
        x_next = torch.maximum(torch.minimum(proposed, upper), lower)
        traj["limit_clipped"].append(proposed.ne(x_next).float())
        # The plant measures velocity; it is not the ROM's own integrator state.
        velocity = (x_next - x) / dt
        x = x_next

        traj["dP"].append(dP.clone())
        traj["velocity"].append(velocity.clone())
        traj["extension"].append(x.clone())
        traj["delta_raw"].append(delta_raw)
        traj["delta_applied"].append(delta.clone())
        traj["guard_vetoed"].append(vetoed.float())
    stacked = {k: torch.stack(v) for k, v in traj.items()}
    stacked["extension_start"] = x0
    if tap is not None:
        share_dp, share_dx = tap.shares(dt, stacked["velocity_in"])
        stacked["residual_share_dP"] = torch.tensor(share_dp)
        stacked["residual_share_extension"] = torch.tensor(share_dx)
    return stacked


def set_beta(model, beta_pa: torch.Tensor) -> None:
    """Write the requested per-cylinder beta into the model (pre-clamp)."""
    state = model.state_dict()
    state["log_bulk_modulus"] = torch.log(beta_pa.clamp_min(1.0e-12))
    model.load_state_dict(state)


def delivered_beta(beta_pa: torch.Tensor) -> torch.Tensor:
    """What `forward` actually uses after its own clamp."""
    return beta_pa.clamp(BETA_FLOOR_PA, BETA_CEIL_PA)


def metrics(nominal: dict[str, torch.Tensor], run: dict[str, torch.Tensor]) -> dict:
    out: dict[str, float] = {}
    for signal in ("dP", "velocity", "extension"):
        nom, per = nominal[signal], run[signal]
        span = (nom - nom[0]).abs().amax().clamp_min(1e-12)
        diff = (per - nom).abs()
        out[f"peak_abs_{signal}"] = diff.amax().item()
        out[f"rms_{signal}"] = diff.pow(2).mean().sqrt().item()
        out[f"rel_peak_{signal}"] = (diff.amax() / span).item()
    out["travel_mm"] = ((run["extension"] - run["extension_start"]).abs().amax() * 1e3).item()
    out["peak_extension_err_mm"] = ((run["extension"] - nominal["extension"]).abs().amax() * 1e3).item()
    out["final_extension_err_mm"] = ((run["extension"][-1] - nominal["extension"][-1]).abs().amax() * 1e3).item()
    # Authority actually delivered to the plant, relative to the raw ROM response.
    raw = run["delta_raw"].abs().sum().clamp_min(1e-18)
    out["authority_kept"] = (run["delta_applied"].abs().sum() / raw).item()
    out["guard_veto_rate"] = run["guard_vetoed"].mean().item()
    out["limit_clip_rate"] = run["limit_clipped"].mean().item()
    out["dP_saturation"] = (run["dP"].abs().amax() / DP_ABS_LIMIT_PA).item()
    for key in ("residual_share_dP", "residual_share_extension"):
        if key in run:
            out[key] = run[key].item()
    out["finite"] = bool(torch.isfinite(run["extension"]).all())
    return out


def report(title: str, axis_name: str, rows: list[dict]) -> None:
    print(f"\n{title}")
    print(
        f"{axis_name:>8}{'beta_del':>10}{'deliv':>7}"
        f"{'rel dP':>9}{'rel vel':>9}{'rel ext':>9}"
        f"{'travel mm':>11}{'ext err mm':>12}{'auth':>8}{'veto':>8}{'res dx':>9}"
    )
    for r in rows:
        print(
            f"{r['axis']:>8}{r['beta_delivered_gpa']:>10.4f}{r['delivered_fraction']:>7.3f}"
            f"{r['rel_peak_dP']*100:>8.1f}%{r['rel_peak_velocity']*100:>8.1f}%"
            f"{r['rel_peak_extension']*100:>8.1f}%"
            f"{r['travel_mm']:>11.3f}{r['peak_extension_err_mm']:>12.4f}"
            f"{r['authority_kept']*100:>7.1f}%{r['guard_veto_rate']*100:>7.1f}%"
            f"{r.get('residual_share_extension', float('nan'))*100:>8.1f}%"
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=str(ROOT / "rom_model/plant_reachable_v1_finetune_blend25/best_rom.pt"))
    p.add_argument("--ticks", type=int, default=80)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=str(ROOT / "experiments/E5/results/rom_bulk_modulus_sweep.json"))
    args = p.parse_args()

    torch.manual_seed(17)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    # control_dt = the ROM's own 50 ms interval, so steps_per_tick = 1 and the
    # direction guard's net authority (3x|command| per ROM tick) is unchanged.
    runtime = CommandCylinderROMRuntime(
        checkpoint=args.checkpoint, num_envs=1, device=device, control_dt_s=0.05
    )
    model = runtime.model
    model.eval()
    limits = runtime.checkpoint["cylinder_limits"].to(device)
    command = safe_trajectory(args.ticks, device)

    tap = OutputTap(model)
    beta_nom = model.state_dict()["log_bulk_modulus"].exp().clone()
    print(f"checkpoint : {args.checkpoint}")
    print(f"device     : {device}   ticks: {args.ticks}   dt: {runtime.sample_time_s} s")
    print(f"nominal beta per cylinder (GPa): {[round(v, 4) for v in (beta_nom / 1e9).tolist()]}")
    print(
        "compliance-factor reach: "
        f"{float(torch.exp(torch.tensor(-0.25))):.3f}x .. {float(torch.exp(torch.tensor(0.10))):.3f}x "
        "of nominal -- the requested grid is unreachable through system_factors, "
        "so log_bulk_modulus is written directly."
    )

    payload: dict = {
        "checkpoint": args.checkpoint,
        "ticks": args.ticks,
        "device": str(device),
        "beta_nominal_gpa": (beta_nom / 1e9).tolist(),
        "beta_clamp_gpa": [BETA_FLOOR_PA / 1e9, BETA_CEIL_PA / 1e9],
        "cylinder_limits_m": limits.tolist(),
        "trajectory": "staggered +-1 mm trapezoid, one cylinder at a time",
        "plant_surrogate": "perfect follower; measured velocity differentiated from measured extension",
        "guard_modes": {},
    }

    for guard in (True, False):
        guard_key = "guard_on" if guard else "guard_off"
        set_beta(model, beta_nom)
        nominal_run = rollout(runtime, command, limits, guard=guard, tap=tap)
        grids = {}
        for grid_name, axis_values, beta_of in (
            ("scale", DEFAULT_SCALES, lambda s: beta_nom * s),
            ("absolute_gpa", DEFAULT_ABSOLUTE_GPA, lambda g: torch.full_like(beta_nom, g * 1e9)),
        ):
            rows = []
            for value in axis_values:
                beta_req = beta_of(value)
                set_beta(model, beta_req)
                run = rollout(runtime, command, limits, guard=guard, tap=tap)
                beta_del = delivered_beta(beta_req)
                rows.append({
                    "axis": f"{value:g}",
                    "beta_requested_gpa": (beta_req.mean() / 1e9).item(),
                    "beta_delivered_gpa": (beta_del.mean() / 1e9).item(),
                    "delivered_fraction": (beta_del.mean() / beta_req.mean().clamp_min(1e-12)).clamp(max=1.0).item(),
                    "beta_delivered_per_cylinder_gpa": (beta_del / 1e9).tolist(),
                    **metrics(nominal_run, run),
                })
            grids[grid_name] = rows
            label = "multiples of the learned beta" if grid_name == "scale" else "absolute beta in GPa"
            report(
                f"[{guard_key}] {label} (E5 0/0.5/1/1.5/2 grid)",
                "scale" if grid_name == "scale" else "GPa",
                rows,
            )
        nom_metrics = metrics(nominal_run, nominal_run)
        grids["nominal"] = nom_metrics
        payload["guard_modes"][guard_key] = grids
        print(
            f"  nominal @ {guard_key}: travel {nom_metrics['travel_mm']:.3f} mm, "
            f"authority kept {nom_metrics['authority_kept']*100:.1f}%, "
            f"guard veto {nom_metrics['guard_veto_rate']*100:.1f}%, "
            f"limit clip {nom_metrics['limit_clip_rate']*100:.1f}%, "
            f"dP/limit {nom_metrics['dP_saturation']*100:.1f}%, "
            f"residual share dP {nom_metrics['residual_share_dP']*100:.1f}% / "
            f"ext {nom_metrics['residual_share_extension']*100:.1f}%"
        )

    print("\nrel  = peak |y(beta) - y(nominal)| / nominal signal range")
    print("deliv= delivered beta / requested beta after the model's [0.01, 3.0] GPa clamp")
    print("auth = sum|delta applied| / sum|delta raw|; veto = guard zeroed the response")
    print("res dx = share of the extension increment coming from the learned residual, not physics")

    tap.close()
    set_beta(model, beta_nom)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
