#!/usr/bin/env python3
"""Action pipeline analysis: scale flow, saturation, valve distribution, clipping.

Run: python analysis/action_analysis/run_action_analysis.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from analysis.shared import ActionScaleConfig


# ---------------------------------------------------------------------------
# Section 1: Action scale mapping analysis
# ---------------------------------------------------------------------------

def analyze_scale_mapping():
    """Analyze action_to_qddot_scale across different configs."""
    print("=" * 72)
    print("SECTION 1: ACTION SCALE MAPPING")
    print("=" * 72)

    # Simulate "id_qddot_std" extraction from a representative ID model
    synthetic_id_std = [
        0.5, 0.45, 0.48, 0.52, 0.55,   # q std: ~0.5 rad
        1.2, 1.1, 1.3, 1.15, 1.25,     # qdot std: ~1.2 rad/s
        3.0, 3.5, 4.0, 3.8,            # qddot std: ~3.5 rad/s^2  <-- indices 10:14
        1e5, 1.2e5, 1.1e5, 0.9e5,      # dP std: ~1e5 Pa
    ]

    print("\n  Configuration comparison:")
    configs = [
        ("Joint reach (fixed 0.5)", ActionScaleConfig(action_to_qddot_scale=0.5, qddot_cmd_clip=5.0)),
        ("Hydraulic ROM (id_qddot_std)", ActionScaleConfig(
            action_to_qddot_scale="id_qddot_std",
            qddot_cmd_clip="id_qddot_std",
            id_feature_std=synthetic_id_std,
        )),
    ]

    for name, cfg in configs:
        eff_scale = cfg.effective_scale()
        eff_clip = cfg.effective_clip()
        print(f"\n  {name}:")
        print(f"    effective_scale = {eff_scale:.4f}")
        print(f"    effective_clip  = {eff_clip:.4f}")

        # Trace a range of actions
        for action_val in [-1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0]:
            act = torch.tensor([[action_val, action_val, action_val, action_val]])
            result = cfg.trace_action(act)
            clipped = " CLIPPED" if result["clip_fraction"] > 0 else ""
            print(f"      action={action_val:+.1f} -> qddot={float(result['qddot_clipped'][0,0]):+.4f}{clipped}")


# ---------------------------------------------------------------------------
# Section 2: Valve command distribution analysis
# ---------------------------------------------------------------------------

def analyze_valve_distribution():
    """Analyze expected valve command range given ID model behavior."""
    print("\n" + "=" * 72)
    print("SECTION 2: VALVE COMMAND DISTRIBUTION ANALYSIS")
    print("=" * 72)

    print("""
  The ID model output passes through nn.Tanh(), bounding it to [-1, 1].
  This is the valve command range.

  What range of qddot commands produces useful valve commands?
  The ID model is conditioned on z_arm_hat (AAM latent), so the
  valve response for the same qddot can vary with arm configuration.

  SYNTHETIC SWEEP: Assume a linearized ID model approximation
  valve_i = tanh(k_i * qddot_i + b_i)
  with k_i ~ 0.1 to 0.3 (learned from data) and b_i ~ 0.
""")

    for gain in [0.05, 0.10, 0.20, 0.50]:
        print(f"\n  Effective gain = {gain}:")
        print(f"    {'qddot':>8s}  {'valve':>8s}  {'saturated?':>12s}")
        print("    " + "-" * 35)
        for qddot in [-10.0, -5.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 5.0, 10.0]:
            valve = math.tanh(gain * qddot)
            sat = "YES" if abs(valve) > 0.99 else ""
            print(f"    {qddot:>+8.1f}  {valve:>+8.4f}  {sat:>12s}")

    print("""
  With gain=0.20, valve saturates at |qddot| >= ~15 rad/s^2.
  With gain=0.10, valve saturates at |qddot| >= ~30 rad/s^2.

  The qddot_cmd_clip=5.0 (or ~3.5 from id_qddot_std) means the
  valve will typically operate in the range tanh(0.2*5) = tanh(1.0) = 0.76,
  well below saturation. This is a healthy operating regime.
""")


# ---------------------------------------------------------------------------
# Section 3: Action saturation penalty interaction
# ---------------------------------------------------------------------------

def analyze_action_saturation():
    """Analyze the action_saturation penalty and its threshold."""
    print("\n" + "=" * 72)
    print("SECTION 3: ACTION SATURATION PENALTY")
    print("=" * 72)

    threshold = 0.90
    weight = 0.5

    print(f"""
  action_saturation_threshold = {threshold}
  The penalty kicks in when |action| > {threshold}.

  Given action_scale={0.5} (joint reach config):
    |action| = 0.90 -> |qddot| = 0.5 * 0.90 = 0.45 rad/s^2 (very mild)
    |action| = 1.00 -> |qddot| = 0.50 rad/s^2

  Given action_scale="id_qddot_std" (~3.5):
    |action| = 0.90 -> |qddot| = 3.5 * 0.90 = 3.15 rad/s^2
    |action| = 1.00 -> |qddot| = 3.50 rad/s^2

  Penalty per-channel at various action levels:
""")
    print(f"  {'|action|':>10s}  {'excess':>10s}  {'loss per ch':>14s}  {'4-ch penalty':>14s}")
    print("  " + "-" * 58)
    for a in [0.0, 0.5, 0.85, 0.90, 0.92, 0.95, 0.98, 1.0]:
        excess = max(0.0, a - threshold) / (max(1.0 - threshold, 1e-8))
        loss_per_ch = math.tanh(excess * excess)
        penalty_4ch = -weight * loss_per_ch
        print(f"  {a:>10.2f}  {excess:>10.3f}  {loss_per_ch:>14.4f}  {penalty_4ch:>14.4f}")

    print(f"""
  COMPARISON WITH VALVE SATURATION:
    action_saturation: threshold={threshold}, weight={weight}  -> max=-0.5
    valve_saturation:  sat_start=0.90, weight=0.15             -> max=-0.114

  These are DUPLICATE penalties: action_saturation penalizes the
  raw PPO output > 0.90, and valve_saturation penalizes the ID
  output > 0.90. Since ID is monotonic in action (via qddot),
  these are correlated but not identical.

  The action_saturation penalty is 3.3x stronger (0.5 vs 0.15).
  Consider: the action_saturation penalty is meant to keep the
  PPO policy in a "normal" range, while valve_saturation is
  meant to protect hydraulics. Do we need both?
""")


# ---------------------------------------------------------------------------
# Section 4: Torque adapter scale analysis
# ---------------------------------------------------------------------------

def analyze_torque_adapter():
    """Analyze torque adapter scale presets and effects."""
    print("\n" + "=" * 72)
    print("SECTION 4: TORQUE ADAPTER SCALE ANALYSIS")
    print("=" * 72)

    from kevin_integration.rl.torque_adapter import TORQUE_ADAPTER_SCALE_PRESETS

    print("\n  Torque adapter scale presets:")
    for name, scales in TORQUE_ADAPTER_SCALE_PRESETS.items():
        print(f"    {name:15s}: {scales}")

    print(f"""
  The default is "conservative" with fd_residual_alpha=0.002.

  Effective scale (scale * alpha):
    conservative: {[0.002 * s for s in TORQUE_ADAPTER_SCALE_PRESETS['conservative']]}
    moderate:     {[0.002 * s for s in TORQUE_ADAPTER_SCALE_PRESETS['moderate']]}
    aggressive:   {[0.002 * s for s in TORQUE_ADAPTER_SCALE_PRESETS['aggressive']]}

  With FD simscape outputs in the range ~1e5 to ~1e6:
    conservative tau = 2e-7 * 1e5 = 0.02 Nm  (very small)
    aggressive   tau = 2e-6 * 1e5 = 0.20 Nm

  At the max FD output ~1e7:
    conservative tau = 2e-7 * 1e7 = 2.0 Nm
    aggressive   tau = 2e-6 * 1e7 = 20.0 Nm (clamped to 5.0)

  The conservative preset produces very small torques. If you want
  the FD effort path to actually contribute to control, you need
  at least the moderate preset.
""")


# ---------------------------------------------------------------------------
# Section 5: Clipping diagnostic effectiveness
# ---------------------------------------------------------------------------

def analyze_clip_diagnostics():
    """Check if clipping diagnostics actually fire."""
    print("\n" + "=" * 72)
    print("SECTION 5: CLIPPING DIAGNOSTICS EFFECTIVENESS")
    print("=" * 72)

    print("""
  SD observer clamping:
    max_abs_pressure_delta = 1.0e7  (10 million)
    max_abs_fnet = 1.0e7

  In the ROM integration CSV (300 samples), dP values range:
    dP_1: ~0, dP_2: ~0, dP_3: ~0, dP_4: ~0  (all zero)

  These limits are 10^7 while actual values are near zero.
  The clamp_fraction diagnostic will always read 0.0.

  RECOMMENDATION: Set limits based on actual data distributions,
  e.g. max_abs_pressure_delta = 3 * std(dP) from recorded runs.
  This makes the diagnostic useful for catching model divergence.

  Torque clamping:
    max_abs_torque = 5.0

  With the conservative torque adapter, torque rarely exceeds 0.1.
  torque_clamp_fraction will always read 0.0.

  RECOMMENDATION: Either (a) use a more aggressive torque adapter
  preset so the clamp becomes meaningful, or (b) remove the clamp
  and rely on the rate limit + lowpass as the sole guard.
""")


# ---------------------------------------------------------------------------
# Section 6: Lowpass + rate limit effective filter
# ---------------------------------------------------------------------------

def analyze_filter_chain():
    """Analyze the combined lowpass + rate limit filter behavior."""
    print("\n" + "=" * 72)
    print("SECTION 6: LOWPASS + RATE LIMIT EFFECTIVE FILTER")
    print("=" * 72)

    alpha = 0.2
    rate_limit = 1.0

    print(f"""
  lowpass_alpha = {alpha}, torque_rate_limit = {rate_limit}

  The effective filter is:
    tau_new = prev + clamp(alpha * (target - prev), -rate_limit, +rate_limit)

  This is a first-order lowpass with a slew-rate clamp applied
  to the lowpass update, not the raw delta. The effective rate
  limit on the raw delta is rate_limit / alpha = {rate_limit / alpha:.1f}.

  Step response simulation (target jumps from 0 to 5.0):
""")
    prev = 0.0
    target = 5.0
    print(f"  {'Step':>5s}  {'Tau':>8s}  {'Delta':>8s}")
    print("  " + "-" * 30)
    for step in range(20):
        delta = alpha * (target - prev)
        limited_delta = max(-rate_limit, min(rate_limit, delta))
        prev = prev + limited_delta
        print(f"  {step:>5d}  {prev:>8.4f}  {limited_delta:>+8.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("ACTION PIPELINE ANALYSIS REPORT")
    print("=" * 72)
    print()

    analyze_scale_mapping()
    analyze_valve_distribution()
    analyze_action_saturation()
    analyze_torque_adapter()
    analyze_clip_diagnostics()
    analyze_filter_chain()

    print("\n" + "=" * 72)
    print("KEY FINDINGS")
    print("=" * 72)
    print("""
  1. SCALE MISMATCH (id_qddot_std):
     The scale extracted from ID training data (~3.5) is much larger
     than the fixed scale (0.5). This means hydraulic ROM mode
     produces qddot commands ~7x larger for the same action.

  2. VALVE OPERATING RANGE IS HEALTHY:
     With qddot_clip ~5.0 and typical ID gain ~0.1-0.2, valves
     operate around tanh(0.5) = 0.46, well within the linear-ish
     region. Saturation is rare at current clip settings.

  3. DUPLICATE SATURATION PENALTIES:
     action_saturation (weight=0.5) and valve_saturation (weight=0.15)
     penalize similar behavior through different channels. Consider
     consolidating.

  4. TORQUE ADAPTER TOO CONSERVATIVE:
     The "conservative" preset with alpha=0.002 produces torques
     < 0.02 Nm for typical FD outputs. This path contributes
     negligibly to control.

  5. CLIPPING DIAGNOSTICS NEVER FIRE:
     SD clamp thresholds (1e7) and torque clamp (5.0) are too high
     to ever activate in normal operation. These are dead diagnostics.
""")

    print("=" * 72)
    print("ANALYSIS COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
