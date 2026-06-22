#!/usr/bin/env python3
"""ROM stress probe: identify key pipeline problems without changing code."""
import sys, json, torch
from pathlib import Path

_FILE = Path(__file__).resolve()
ROOT = _FILE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(_FILE.parents[1] / "rom_cylinders"))

try:
    from valve_conditioned_causal_rom import ValveConditionedROMInference, EXPORT_DIR
    from rom_data_contract import (
        EXTENSION_SLICE, EXTENSION_VELOCITY_SLICE,
        VALVE_SLICE, DP_SLICE, JOINT_POS_SLICE, JOINT_VEL_SLICE,
        HISTORY_LENGTH, CORE_INPUT_FEATURE_NAMES,
    )
except ImportError:
    from kevin_integration.rom_cylinders.valve_conditioned_causal_rom import (
        ValveConditionedROMInference, EXPORT_DIR,
    )
    from kevin_integration.rom_cylinders.rom_data_contract import (
        EXTENSION_SLICE, EXTENSION_VELOCITY_SLICE,
        VALVE_SLICE, DP_SLICE, JOINT_POS_SLICE, JOINT_VEL_SLICE,
        HISTORY_LENGTH, CORE_INPUT_FEATURE_NAMES,
    )

ROM_DIR = ROOT / "kevin_integration" / "rom_cylinders"


def _rom_at_state(rom_obj, valve=None, dP=None, velocity=None, extension_offset=None):
    """Create a batch-1 history at a specific state."""
    batch = 1
    history = torch.zeros(batch, HISTORY_LENGTH, len(CORE_INPUT_FEATURE_NAMES))
    for t in range(HISTORY_LENGTH):
        history[0, t, EXTENSION_SLICE] = rom_obj.extension_mean + (extension_offset or 0.0)
        history[0, t, DP_SLICE] = dP if dP is not None else rom_obj.training_mean_dp
        history[0, t, JOINT_POS_SLICE] = rom_obj.joint_pos_mean
    if velocity is not None:
        history[0, -1, EXTENSION_VELOCITY_SLICE] = velocity
    if valve is not None:
        history[0, -1, VALVE_SLICE] = valve
    return history


def fmt(vals):
    return [f"{v:.6f}" for v in vals]


def main():
    inf = ValveConditionedROMInference(str(EXPORT_DIR), device="cpu")
    rom = inf.model

    mapping_path = ROM_DIR / "cylinder_coordinate_mapping.json"
    mapping = json.loads(mapping_path.read_text())
    order = ["Drehzapfen_joint", "Ausleger_I_joint", "Ausleger_II_joint", "Messerkopf_Schwenk_joint"]
    signs = torch.tensor([float(mapping["cylinders"][n]["sign"]) for n in order])

    GAIN = 0.65
    POS_SCALE = 0.005
    RATE_LIMIT = torch.tensor([0.80, 0.80, 0.80, 0.80]) * 0.01
    PRESSURE_CLAMP = 2.5e8

    print("=" * 70)
    print("ROM STRESS PROBE")
    print("=" * 70)

    mean_dp = rom.training_mean_dp

    # 1. Training-mean baseline
    print("\n1. TRAINING-MEAN BASELINE")
    base_hist = _rom_at_state(rom, dP=mean_dp)
    base_out = rom(base_hist)
    native_delta = signs.reshape(1, 4) * base_out["final_delta_extension"]
    print(f"   Training mean dP (MPa): {[round(v/1e6, 1) for v in mean_dp.tolist()]}")
    print(f"   Native ROM delta 50ms (m):  {fmt(native_delta[0].tolist())}")
    print(f"   Native ROM delta 10ms (m):  {fmt((native_delta[0]/5).tolist())}")

    # 2. Pressure sweep
    print("\n2. PRESSURE SWEEP (valve=0)")
    pressures_pa = [
        ("training_mean", mean_dp.clone()),
        ("mean-1sigma", mean_dp - 5e6),
        ("mean+1sigma", mean_dp + 5e6),
        ("mean-3sigma", mean_dp - 15e6),
        ("mean+3sigma", mean_dp + 15e6),
        ("extreme_high", torch.full_like(mean_dp, PRESSURE_CLAMP)),
        ("zero", torch.zeros_like(mean_dp)),
    ]
    for label, dp in pressures_pa:
        hist = _rom_at_state(rom, dP=dp, valve=torch.zeros(4))
        out = rom(hist)
        delta = signs.reshape(1, 4) * out["final_delta_extension"]
        dp_mpa = [round(v/1e6, 1) for v in dp.tolist()]
        print(f"   {label:20s} dP={dp_mpa} MPa -> delta={fmt(delta[0].tolist())}")

    # 3. Valve authority sweep
    print("\n3. VALVE AUTHORITY SWEEP (training-mean dP)")
    valves = torch.linspace(-1.0, 1.0, 21)
    for c in range(4):
        deltas = []
        for v in valves:
            hist = _rom_at_state(rom, dP=mean_dp, valve=torch.zeros(4))
            hist[0, -1, VALVE_SLICE] = 0.0
            hist[0, -1, VALVE_SLICE][c] = v.item()
            out = rom(hist)
            native = signs[c].item() * out["final_delta_extension"][0, c].item()
            ff = GAIN * POS_SCALE * v.item()
            total = native + ff
            total = max(min(total, RATE_LIMIT[c].item()), -RATE_LIMIT[c].item())
            deltas.append(total)

        slope_pos = (deltas[-1] - deltas[len(deltas)//2]) / 1.0
        slope_neg = (deltas[len(deltas)//2] - deltas[0]) / 1.0
        monotonic = all(deltas[i] <= deltas[i+1] for i in range(len(deltas)-1))
        direct_frac = abs(GAIN * POS_SCALE) / RATE_LIMIT[c].item() * 100
        print(f"   Cyl {c+1}: slope(+)= {slope_pos:.6f}, slope(-)= {slope_neg:.6f}, "
              f"monotonic={monotonic}, direct_ff_frac={direct_frac:.1f}%")

    # 4. Post-feedforward rate limit check
    print("\n4. POST-FEEDFORWARD RATE LIMIT CHECK")
    violations = 0
    total_cases = 0
    for v_test in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        for label, dp in pressures_pa:
            hist = _rom_at_state(rom, dP=dp, valve=torch.full((4,), v_test))
            out = rom(hist)
            native = signs.reshape(1, 4) * out["final_delta_extension"]
            ff = GAIN * POS_SCALE * torch.full((4,), v_test)
            total = native.squeeze(0) + ff
            for c_idx in range(4):
                total_cases += 1
                if abs(total[c_idx].item()) > RATE_LIMIT[c_idx].item() + 1e-10:
                    violations += 1
    print(f"   Pre-clamp violations: {violations}/{total_cases} ({violations/total_cases*100:.1f}%)")

    # 5. Near-pose authority fraction
    print("\n5. AUTHORITY AT NEAR-POSE STATE (vel=0.02, valve delta +1)")
    for c in range(4):
        vel = torch.zeros(4)
        vel[c] = 0.02
        hist = _rom_at_state(rom, dP=mean_dp, velocity=vel, valve=torch.zeros(4))
        out0 = rom(hist)
        hist_v = hist.clone()
        hist_v[0, -1, VALVE_SLICE] = 0.0
        hist_v[0, -1, VALVE_SLICE][c] = 1.0
        out1 = rom(hist_v)
        native0 = signs[c].item() * out0["final_delta_extension"][0, c].item()
        native1 = signs[c].item() * out1["final_delta_extension"][0, c].item()
        native_span = abs(native1 - native0)
        ff_span = GAIN * POS_SCALE * 1.0
        total_span = min(native_span + ff_span, RATE_LIMIT[c].item())
        native_frac = native_span / max(total_span, 1e-10) * 100
        ff_frac = ff_span / RATE_LIMIT[c].item() * 100
        print(f"   Cyl {c+1}: native_span={native_span:.6f}, ff_span={ff_span:.6f}, "
              f"total={total_span:.6f}, native_frac={native_frac:.1f}%, ff_frac={ff_frac:.1f}%")

    # 6. ROM pressure sensitivity
    print("\n6. ROM PRESSURE SENSITIVITY (valve=0, delta range across pressures)")
    for c in range(4):
        deltas_across_pressure = []
        for label, dp in pressures_pa:
            hist = _rom_at_state(rom, dP=dp, valve=torch.zeros(4))
            out = rom(hist)
            delta = signs[c].item() * out["final_delta_extension"][0, c].item()
            deltas_across_pressure.append(delta)
        d_range = max(deltas_across_pressure) - min(deltas_across_pressure)
        print(f"   Cyl {c+1}: range={d_range:.6f} m/50ms, "
              f"min={min(deltas_across_pressure):.6f}, max={max(deltas_across_pressure):.6f}")

    # 7. ROM at saturated states
    print("\n7. ROM AT SATURATED STATES")
    extreme_dp = torch.full_like(mean_dp, PRESSURE_CLAMP)
    extreme_vel = torch.tensor([0.5, 0.5, 0.5, 0.5])
    ext_hist = _rom_at_state(rom, dP=extreme_dp, velocity=extreme_vel, valve=torch.ones(4))
    ext_out = rom(ext_hist)
    ext_delta = signs.reshape(1, 4) * ext_out["final_delta_extension"]
    ext_ff = GAIN * POS_SCALE * torch.ones(4)
    ext_total = torch.clamp(ext_delta.squeeze(0) + ext_ff, -RATE_LIMIT, RATE_LIMIT)
    print(f"   ROM delta:    {fmt(ext_delta[0].tolist())}")
    print(f"   Total clamped:{fmt(ext_total.tolist())}")
    print(f"   Rate limit:   {fmt(RATE_LIMIT.tolist())}")
    at_limit = [abs(v) >= RATE_LIMIT[i].item() - 1e-8 for i, v in enumerate(ext_total.tolist())]
    print(f"   At rate limit: {at_limit}")

    # 8. ROM delta sign change with pressure
    print("\n8. ROM DELTA SIGN CHANGE WITH PRESSURE (valve=0)")
    for c in range(4):
        signs_found = set()
        for label, dp in pressures_pa:
            hist = _rom_at_state(rom, dP=dp, valve=torch.zeros(4))
            out = rom(hist)
            delta = signs[c].item() * out["final_delta_extension"][0, c].item()
            tag = "pos" if delta > 1e-10 else ("neg" if delta < -1e-10 else "zero")
            signs_found.add(tag)
        print(f"   Cyl {c+1}: signs across pressures = {signs_found}")
        if len(signs_found) > 1:
            print(f"         ** NOTE: ROM delta changes sign with pressure!")

    # 9. Feedforward authority fraction summary
    print("\n9. FEEDFORWARD AUTHORITY FRACTION")
    for c in range(4):
        ff_per_step = GAIN * POS_SCALE
        total = ff_per_step / RATE_LIMIT[c].item() * 100
        bar = "#" * int(total / 5)
        print(f"   Cyl {c+1}: ff_authority = {total:.1f}% of rate limit |{bar}")
        if total < 50:
            print(f"         ** ISSUE: feedforward authority below 50% on cylinder {c+1}")

    # 10. ROM update cadence stress
    print("\n10. ROM UPDATE CADENCE ANALYSIS")
    print(f"    ROM native dt: 0.05s, Isaac dt: 0.01s, ratio: 5")
    print(f"    Per-10ms max delta: {[round(v, 4) for v in RATE_LIMIT.tolist()]} m")
    print(f"    Max 50ms cumulative: {[round(v*5, 4) for v in RATE_LIMIT.tolist()]} m")

    # 11. Can ROM produce 5cm motion?
    print("\n11. CURRICULUM-RELEVANT: CAN ROM PRODUCE 5cm MOTION?")
    for c in range(4):
        max_per_step = RATE_LIMIT[c].item()
        steps_for_5cm = 0.05 / max_per_step if max_per_step > 0 else float("inf")
        print(f"   Cyl {c+1}: max/10ms = {max_per_step:.4f}m, steps for 5cm = {steps_for_5cm:.0f}")
        if steps_for_5cm <= 50:
            print(f"         OK: 5cm reachable within hold window")
        else:
            print(f"         ** ISSUE: 5cm may need >50 steps")

    # 12. Check SD scaler
    print("\n12. SD OBSERVER PRESSURE CHECK")
    sd_path = ROM_DIR / "exported_valve_conditioned_causal_rom" / "best_valve_conditioned_causal_rom_scaler.pth"
    if sd_path.exists():
        sd_scaler = torch.load(sd_path, map_location="cpu", weights_only=True)
        dP_mean = sd_scaler.get("dP_mean", None)
        dP_std = sd_scaler.get("dP_std", None)
        if dP_mean is not None:
            print(f"    dP training mean (MPa): {[round(v/1e6, 1) for v in dP_mean.tolist()]}")
        if dP_std is not None:
            print(f"    dP training std  (MPa): {[round(v/1e6, 1) for v in dP_std.tolist()]}")
            for c in range(4):
                support = 3 * dP_std[c].item()
                print(f"    Cyl {c+1}: +/-3sigma = +/-{support/1e6:.1f} MPa, "
                      f"clamp = {PRESSURE_CLAMP/1e6:.0f} MPa")

    print("\n" + "=" * 70)
    print("STRESS PROBE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
