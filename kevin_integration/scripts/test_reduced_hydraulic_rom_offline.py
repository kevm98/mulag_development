from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kevin_integration.controllers.reduced_hydraulic_actuator import (  # noqa: E402
    ReducedHydraulicActuatorROM,
    ReducedHydraulicActuatorROMCfg,
)


def _fmt(values: torch.Tensor) -> str:
    return "[" + ", ".join(f"{value:.6g}" for value in values.detach().flatten().cpu().tolist()) + "]"


def _fmt_tuple(values: tuple[float, ...] | tuple[int, ...]) -> str:
    return "[" + ", ".join(str(value) for value in values) + "]"


def _print_rom_parameters(cfg: ReducedHydraulicActuatorROMCfg) -> None:
    print("[INFO] ROM placeholder parameters:")
    print(f"  Abar={_fmt_tuple(cfg.Abar)}")
    print(f"  Vt={_fmt_tuple(cfg.Vt)}")
    print(f"  E={_fmt_tuple(cfg.E)}")
    print(f"  Kq={_fmt_tuple(cfg.Kq)}")
    print(f"  Pmax={_fmt_tuple(cfg.Pmax)}")
    print(f"  velocity_gain={_fmt_tuple(cfg.velocity_gain)}")
    print(f"  force_scale={_fmt_tuple(cfg.force_scale)}")
    print(f"  xdot_max={_fmt_tuple(cfg.xdot_max)}")
    print(f"  dx_max={_fmt_tuple(cfg.dx_max)}")
    print(f"  delay_steps={_fmt_tuple(cfg.delay_steps)}")
    print(f"  udz_pos={_fmt_tuple(cfg.udz_pos)}")
    print(f"  udz_neg={_fmt_tuple(cfg.udz_neg)}")


def _run_case(rom: ReducedHydraulicActuatorROM, valve_cmd: torch.Tensor, pressure: torch.Tensor, label: str):
    zeros = torch.zeros_like(valve_cmd)
    out = rom.step(
        valve_cmd=valve_cmd,
        x_cyl=zeros,
        xdot_cyl=zeros,
        P_L=pressure,
        dt=0.01,
    )
    print(
        f"{label:<14} valve_cmd={_fmt(valve_cmd[0])} "
        f"P_L={_fmt(out.P_L_next[0])} "
        f"xdot_pred={_fmt(out.xdot_cyl_pred[0])} "
        f"dx_pred={_fmt(out.dx_cyl_pred[0])} "
        f"pressure_clamp={_fmt(out.debug['rom_pressure_clamp_fraction'])} "
        f"nan_guard={_fmt(out.debug['rom_nan_guard_triggered'])}"
    )
    return out


def main() -> None:
    cfg = ReducedHydraulicActuatorROMCfg(
        Kq=(1.0e-8, 1.0e-8, 1.0e-8, 1.0e-8),
        xdot_max=(0.02, 0.02, 0.02, 0.02),
        dx_max=(5.0e-4, 5.0e-4, 5.0e-4, 5.0e-4),
        Pmax=(100.0, 100.0, 100.0, 100.0),
    )
    rom = ReducedHydraulicActuatorROM(num_envs=2, cfg=cfg, device="cpu")
    pressure = torch.zeros((2, 4), dtype=torch.float32)
    _print_rom_parameters(cfg)

    positive = _run_case(rom, 0.5 * torch.ones((2, 4), dtype=torch.float32), pressure, "positive")
    if not torch.isfinite(positive.dx_cyl_pred).all() or not torch.isfinite(positive.P_L_next).all():
        raise RuntimeError("Positive valve ROM output contains NaN/Inf.")
    if not torch.any(torch.abs(positive.P_L_next) > 0.0):
        raise RuntimeError("Positive valve command did not change pressure.")
    if not torch.any(torch.abs(positive.dx_cyl_pred) > 0.0):
        raise RuntimeError("Positive valve command did not produce a finite stroke increment.")
    if torch.max(torch.abs(positive.dx_cyl_pred)) < 1.0e-6:
        print(
            "[WARN] ROM dx_pred magnitude is extremely small. "
            "Placeholder parameters likely need identification from Simscape/full-order data."
        )

    rom.reset()
    negative = _run_case(rom, -0.5 * torch.ones((2, 4), dtype=torch.float32), pressure, "negative")
    if not torch.isfinite(negative.dx_cyl_pred).all() or not torch.isfinite(negative.P_L_next).all():
        raise RuntimeError("Negative valve ROM output contains NaN/Inf.")
    if not torch.all(torch.sign(positive.dx_cyl_pred[0]) == -torch.sign(negative.dx_cyl_pred[0])):
        raise RuntimeError("Positive and negative valve commands did not produce opposite stroke increment signs.")

    rom.reset()
    clamp = _run_case(rom, 100.0 * torch.ones((2, 4), dtype=torch.float32), pressure, "clamp")
    if torch.max(clamp.debug["rom_pressure_clamp_fraction"]) <= 0.0:
        raise RuntimeError("Clamp case did not trigger the pressure clamp.")

    rom.reset()
    invalid_valve = torch.full((2, 4), float("nan"), dtype=torch.float32)
    invalid = _run_case(rom, invalid_valve, pressure, "nan_guard")
    if torch.max(invalid.debug["rom_nan_guard_triggered"]) <= 0.0:
        raise RuntimeError("NaN guard case did not report sanitization.")
    if not torch.isfinite(invalid.dx_cyl_pred).all() or not torch.isfinite(invalid.P_L_next).all():
        raise RuntimeError("NaN guard did not sanitize ROM outputs.")

    print("[PASS] ReducedHydraulicActuatorROM offline sanity test passed.")


if __name__ == "__main__":
    main()
