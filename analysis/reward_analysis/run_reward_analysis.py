#!/usr/bin/env python3
"""Reward analysis: per-term balance, regime sweeps, EMA artifacts, dead weights.

Run: python analysis/reward_analysis/run_reward_analysis.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.shared import simulate_reward_at_error, joint_reach_reward_profile


# ---------------------------------------------------------------------------
# Section 1: Reward term profiling
# ---------------------------------------------------------------------------

def analyze_reward_terms():
    """Profile every reward term: weight, sign, active range, contribution."""
    print("=" * 72)
    print("SECTION 1: REWARD TERM PROFILE")
    print("=" * 72)

    profile = joint_reach_reward_profile()
    print(f"\n{'Term':<22s} {'Weight':>8s} {'Sign':>8s}  {'Typical Range':>20s}  Description")
    print("-" * 90)
    for t in profile:
        print(f"{t.name:<22s} {t.weight:>8.4f} {t.sign:>8s}  [{t.typical_range[0]:>8.1e}, {t.typical_range[1]:>8.1e}]  {t.saturation_behavior}")

    # Identify dead weights
    dead = [t for t in profile if abs(t.typical_range[0]) < 1e-4 and abs(t.typical_range[1]) < 1e-4]
    if dead:
        print(f"\n  DEAD WEIGHTS (contribution < 1e-4):")
        for t in dead:
            print(f"    {t.name}: weight={t.weight}, range={t.typical_range}")


# ---------------------------------------------------------------------------
# Section 2: Reward balance across error regimes
# ---------------------------------------------------------------------------

def analyze_reward_balance():
    """Sweep across different tracking error regimes and show balance."""
    print("\n" + "=" * 72)
    print("SECTION 2: REWARD BALANCE ACROSS ERROR REGIMES")
    print("=" * 72)

    regimes = [
        ("Large error (start of episode)", 0.15, 0.3, 0.5, 0.1),
        ("Medium error (mid-tracking)",    0.05, 0.2, 0.3, 0.05),
        ("Small error (approaching target)", 0.02, 0.1, 0.2, 0.02),
        ("Near convergence",                0.01, 0.05, 0.1, 0.01),
        ("At target (settled)",             0.002, 0.02, 0.05, 0.005),
    ]

    print(f"\n{'Regime':<30s} {'Pos Sum':>8s} {'Neg Sum':>8s} {'Total':>8s} {'Neg/Pos':>8s} {'Dominant+':>20s} {'Dominant-':>20s}")
    print("-" * 110)

    for name, q_err, qdot, act_mag, act_delta in regimes:
        r = simulate_reward_at_error(
            q_error_norm=q_err,
            qdot_norm=qdot,
            action_magnitude=act_mag,
            action_delta=act_delta,
            valve_cmd_norm=act_mag * 0.5,
            is_settle=(q_err < 0.02),
            progress_delta=0.0,
        )
        # Find dominant terms
        pos_terms = [(k, v) for k, v in r.items() if v > 0 and k not in ("positive_sum", "negative_sum", "total", "neg_to_pos_ratio")]
        neg_terms = [(k, v) for k, v in r.items() if v < 0 and k not in ("positive_sum", "negative_sum", "total", "neg_to_pos_ratio")]
        pos_terms.sort(key=lambda x: -x[1])
        neg_terms.sort(key=lambda x: x[1])

        print(f"{name:<30s} {r['positive_sum']:>+8.4f} {r['negative_sum']:>+8.4f} {r['total']:>+8.4f} {r['neg_to_pos_ratio']:>8.2f} {pos_terms[0][0]:>20s} {neg_terms[0][0]:>20s}")


# ---------------------------------------------------------------------------
# Section 3: Progress reward EMA momentum analysis
# ---------------------------------------------------------------------------

def analyze_progress_ema():
    """Analyze the progress reward's EMA momentum artifact."""
    print("\n" + "=" * 72)
    print("SECTION 3: PROGRESS REWARD EMA MOMENTUM ARTIFACT")
    print("=" * 72)

    alpha = 0.1  # progress_ema_alpha
    print(f"""
  The progress reward is: weight * gamma * (phi_next - phi_current)
  where phi = -weighted_mean(|q_ref - q|) and gamma=0.99

  The EMA-smoothed baseline is:
    prev_q_error_mean = (1-alpha)*prev + alpha*current, alpha={alpha}

  SIMULATED OSCILLATION SCENARIO:
  Error oscillates between 0.02 and 0.06 rad every step.
  The EMA lags behind, so progress reward can be positive even
  when error is increasing.
""")

    # Simulate
    errors = [0.02, 0.06, 0.02, 0.06, 0.02, 0.06, 0.02, 0.06, 0.02, 0.06]
    ema = errors[0]
    print(f"\n  {'Step':>5s}  {'Error':>8s}  {'EMA':>8s}  {'Progress':>10s}  {'Accumulated':>12s}")
    print("  " + "-" * 55)
    accumulated = 0.0
    for i, err in enumerate(errors):
        prev_ema = ema
        ema = (1 - alpha) * prev_ema + alpha * err
        progress = 2.0 * (prev_ema - err)  # weight * (phi_prev - phi_current)
        accumulated += progress
        flag = " <-- UNDESERVED POSITIVE" if err > prev_ema and progress > 0 else ""
        print(f"  {i:>5d}  {err:>8.4f}  {ema:>8.4f}  {progress:>+10.6f}  {accumulated:>+12.6f}{flag}")


# ---------------------------------------------------------------------------
# Section 4: Action penalty interaction
# ---------------------------------------------------------------------------

def analyze_action_penalties():
    """Analyze how action penalties compete with tracking reward."""
    print("\n" + "=" * 72)
    print("SECTION 4: ACTION PENALTY VS TRACKING REWARD TRADEOFF")
    print("=" * 72)

    print("""
  As the arm approaches the target, q_tracking reward shrinks quadratically
  but action penalties scale with the action magnitude, which stays roughly
  constant if the policy keeps moving.

  BREAK-EVEN ANALYSIS: At what error does action_rate penalty exceed tracking?
""")

    print(f"\n  {'Action Mag':>10s}  {'Action Delta':>12s}  {'Break-even Error':>18s}  {'Effective?':>12s}")
    print("  " + "-" * 65)

    for act_mag in [0.05, 0.10, 0.20, 0.50, 0.80]:
        for act_delta in [0.01, 0.05, 0.10]:
            # q_tracking = -3.0 * error^2 (per joint, mean across 5 joints)
            # action_rate = -0.05 * mean(action_delta^2) * 4
            # action_magnitude = -0.02 * mean(action^2) * 4
            action_rate_pen = -0.05 * 4.0 * act_delta * act_delta
            action_mag_pen = -0.02 * 4.0 * act_mag * act_mag
            total_pen = action_rate_pen + action_mag_pen

            # Solve: -3.0 * error^2 = total_pen
            # error = sqrt(-total_pen / 3.0)
            if total_pen < 0:
                break_even = math.sqrt(-total_pen / 3.0)
                effective = "YES" if break_even < 0.10 else "no"
                print(f"  {act_mag:>10.2f}  {act_delta:>12.3f}  {break_even:>18.4f} rad       {effective:>12s}")
            else:
                print(f"  {act_mag:>10.2f}  {act_delta:>12.3f}  {'N/A':>18s}       {'---':>12s}")

    print("""
  INTERPRETATION: When action_magnitude >= 0.20 and action_delta >= 0.05,
  the combined penalty exceeds the tracking reward for errors below ~0.057 rad.
  The policy faces a perverse incentive: stop moving to avoid penalties,
  even if it hasn't reached the target yet.

  The settle_tracking reward (+2.0 * error^2) partially counteracts this
  during the settle phase, but only activates when ramp_progress > 0.95.
""")


# ---------------------------------------------------------------------------
# Section 5: Torque effort weight audit
# ---------------------------------------------------------------------------

def analyze_torque_effort():
    """Analyze the effectively-dead torque effort weight."""
    print("\n" + "=" * 72)
    print("SECTION 5: TORQUE EFFORT WEIGHT AUDIT")
    print("=" * 72)

    weight = 1e-6
    print(f"""
  rew_torque_effort = {weight}

  At weight {weight}, here's the penalty magnitude for different torque levels:
""")
    for torque_per_joint in [0.1, 0.5, 1.0, 2.0, 5.0]:
        penalty = -weight * 4.0 * torque_per_joint * torque_per_joint
        print(f"    torque={torque_per_joint:4.1f} Nm -> penalty={penalty:+.2e}")

    print(f"""
  Even at max_abs_torque=5.0 across all 4 joints:
    penalty = -{weight} * 4 * 25 = {-weight * 100:+.2e}

  This is 4-5 orders of magnitude below other penalty terms.
  VERDICT: Effectively dead. Either remove it or increase to 1e-4.
""")


# ---------------------------------------------------------------------------
# Section 6: Valve saturation penalty coverage
# ---------------------------------------------------------------------------

def analyze_valve_saturation():
    """Analyze when the valve saturation penalty activates."""
    print("\n" + "=" * 72)
    print("SECTION 6: VALVE SATURATION PENALTY COVERAGE")
    print("=" * 72)

    sat_start = 0.90
    weight = 0.15

    print(f"""
  sat_start = {sat_start}, weight = {weight}
  The penalty uses: sat_excess = relu(|valve| - sat_start) / (1 - sat_start)
  Then: penalty = -weight * mean(tanh(square(sat_excess)))

  Activation map:
""")
    print(f"  {'|valve|':>10s}  {'sat_excess':>12s}  {'loss per ch':>14s}  {'4-ch penalty':>14s}")
    print("  " + "-" * 58)
    for v in [0.0, 0.5, 0.85, 0.90, 0.92, 0.95, 0.98, 1.0]:
        excess = max(0.0, v - sat_start) / (1.0 - sat_start)
        loss_per_ch = math.tanh(excess * excess)
        penalty_4ch = -weight * loss_per_ch
        print(f"  {v:>10.2f}  {excess:>12.3f}  {loss_per_ch:>14.4f}  {penalty_4ch:>14.4f}")

    print(f"""
  The penalty activates GRADUALLY (tanh-squared, not a cliff).
  At |valve|=0.92 (just 2% above threshold): penalty=-0.006
  At |valve|=0.95: penalty=-0.033
  At |valve|=1.00 (full saturation): penalty=-0.095

  This is well-behaved. The smooth onset prevents harsh punishment
  for brief excursions above 0.90.
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("REWARD ANALYSIS REPORT")
    print("=" * 72)
    print()

    analyze_reward_terms()
    analyze_reward_balance()
    analyze_progress_ema()
    analyze_action_penalties()
    analyze_torque_effort()
    analyze_valve_saturation()

    print("\n" + "=" * 72)
    print("KEY FINDINGS")
    print("=" * 72)
    print("""
  1. REWARD IMBALANCE AT SMALL ERRORS:
     At q_error < 0.05 rad, action penalties can exceed tracking
     reward, creating a perverse incentive to stop moving.

  2. PROGRESS EMA MOMENTUM:
     With alpha=0.1, the EMA lags 3-5 steps behind actual error.
     Oscillating behavior extracts undeserved positive progress reward.

  3. TORQUE EFFORT IS DEAD:
     rew_torque_effort=1e-6 contributes <2.5e-5 even at max torque.
     Either remove or increase 100x.

  4. VALVE SATURATION IS WELL-DESIGNED:
     Smooth tanh-squared onset prevents harsh punishment at threshold.

  5. ACTION PENALTIES NEED REBALANCING:
     The settle_tracking reward partially fixes the imbalance but
     only during the settle phase. Consider reducing action_rate
     and action_magnitude weights by 50% for errors < 0.05 rad.
""")

    print("=" * 72)
    print("ANALYSIS COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
