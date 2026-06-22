#!/usr/bin/env python3
"""Observation space analysis: redundancy, z_arm_hat diagnostics, signal contribution.

This script analyzes the observation space without requiring a running
simulation. It operates on:
  1. Static structural analysis of the observation schemas
  2. Synthetic/profiled data distributions
  3. Checkpoint metadata when available

Run: python analysis/obs_analysis/run_obs_analysis.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from analysis.shared import (
    OBS_SCHEMA_JOINT_SPACE,
    OBS_SCHEMA_TASK_SPACE,
    JOINT_OBS_DIM,
    TASK_SPACE_OBS_DIM,
    detect_redundant_groups,
    schema_to_slices,
    correlation_matrix,
    compute_feature_stats,
)


# ---------------------------------------------------------------------------
# Section 1: Structural redundancy analysis
# ---------------------------------------------------------------------------

def analyze_structural_redundancy():
    """Analyze which observation dimensions are algebraically redundant."""
    print("=" * 72)
    print("SECTION 1: STRUCTURAL REDUNDANCY ANALYSIS")
    print("=" * 72)

    for name, schema, dim in [
        (f"Joint-space ({JOINT_OBS_DIM}-dim)", OBS_SCHEMA_JOINT_SPACE, JOINT_OBS_DIM),
        (f"Task-space ({TASK_SPACE_OBS_DIM}-dim)", OBS_SCHEMA_TASK_SPACE, TASK_SPACE_OBS_DIM),
    ]:
        print(f"\n--- {name} ---")
        computed_dim = sum(d for _, d, _ in schema)
        assert computed_dim == dim, f"Schema dim {computed_dim} != declared {dim}"
        print(f"  Total dims: {dim}")

        groups = detect_redundant_groups(schema)
        total_redundant = sum(g["redundant_dims"] for g in groups)
        print(f"  Redundant groups found: {len(groups)}")
        print(f"  Total redundant dims: {total_redundant}")
        print(f"  Effective info dims: {dim - total_redundant}")
        print(f"  Compression ratio: {dim / max(dim - total_redundant, 1):.1f}x")

        for g in groups:
            print(f"\n  Group: {g['group']}")
            print(f"    Relationship: {g['relationship']}")
            print(f"    Wasted dims: {g['redundant_dims']}")
            print(f"    Recommended keep: {g['keep']}")

        # Show the recommended slim schema
        slices = schema_to_slices(schema)
        names = [n for n, _, _ in schema]
        print(f"\n  Feature names: {names}")

        # Compute recommended reduced feature set
        redundant_names = set()
        for g in groups:
            keep = g["keep"]
            if isinstance(keep, str):
                keep = [keep]
            redundant_names.update(n for n in g["group"] if n not in keep)

        reduced = [(n, d) for n, d, _ in schema if n not in redundant_names]
        reduced_dim = sum(d for _, d in reduced)
        print(f"  Reduced schema ({reduced_dim} dims): {[n for n, _ in reduced]}")


# ---------------------------------------------------------------------------
# Section 2: z_arm_hat (AAM latent) analysis
# ---------------------------------------------------------------------------

def analyze_aam_latent():
    """Analyze the AAM latent vector properties and risks."""
    print("\n" + "=" * 72)
    print("SECTION 2: AAM LATENT (z_arm_hat) ANALYSIS")
    print("=" * 72)

    print("""
  The AAM model's head architecture ends with: nn.Linear(hidden_dim, 24)
  No activation, no normalization, no output bounds.

  What this means for PPO observations:
  - z_arm_hat is concatenated raw into observations at indices 28:52 (joint)
    or 41:65 (task-space)
  - If the AAM checkpoint produces values in range [-100, 100], the PPO
    observation statistics will be dominated by these 24 dimensions
  - empirical_normalization in the PPO runner will eventually adapt,
    but only after collecting sufficient rollout data
  - Early training instability is likely if the AAM latent scale differs
    significantly from other observation components (q ~ [-3, 3], etc.)
""")

    # Demonstrate with synthetic AAM output at different scales
    print("  Demonstration: effect of unnormalized z_arm_hat on observation variance:")
    print()
    for latent_scale in [0.1, 1.0, 10.0, 100.0]:
        # Simulate a batch of observations with different latent scales
        q = torch.randn(4096, 5) * 0.5  # joint positions ~ rad
        qdot = torch.randn(4096, 5) * 0.3
        q_error = torch.randn(4096, 5) * 0.5
        qdot_error = torch.randn(4096, 5) * 0.3
        prev_action = torch.randn(4096, 4) * 0.5
        prev_valve = torch.randn(4096, 4) * 0.5
        z_arm = torch.randn(4096, 24) * latent_scale

        obs = torch.cat([q, qdot, q_error, qdot_error, prev_action, prev_valve, z_arm], dim=-1)

        z_var = z_arm.var().item()
        total_var = obs.var(dim=0).sum().item()
        z_var_fraction = (24 * z_var) / max(total_var, 1e-8)

        print(f"    latent_scale={latent_scale:6.1f}: "
              f"z_arm_hat per-dim std={math.sqrt(z_var):.2f}, "
              f"z_arm fraction of total variance={z_var_fraction:.1%}")


# ---------------------------------------------------------------------------
# Section 3: Interaction between AAM latent version and observation
# ---------------------------------------------------------------------------

def analyze_aam_version_tracking():
    """Analyze the AAM latent version-tracking mechanism."""
    print("\n" + "=" * 72)
    print("SECTION 3: AAM VERSION TRACKING HAZARD")
    print("=" * 72)
    print("""
  The mechanism enforcing "ID latent == PPO observation latent":
    - _update_aam_latent() increments _aam_latent_state_version
    - _get_observations() copies version to _observation_aam_state_version
    - _id_conditioning_latent() checks equality with _observation_aam_state_version

  HAZARD: If observations are sampled twice between control steps
  (e.g., during evaluation or logging), both calls see the same
  _aam_latent_state_version and pass the check. The ID would receive a
  stale latent that doesn't match the actual observation fed to the policy.

  This is likely safe in DirectRLEnv's normal step loop, but subtle
  in multi-call scenarios.

  RECOMMENDATION: Store the actual z_arm_hat tensor snapshot alongside
  the version, and compare tensor equality instead of version equality.
""")


# ---------------------------------------------------------------------------
# Section 4: Synthetic feature correlation analysis
# ---------------------------------------------------------------------------

def analyze_feature_correlation():
    """Correlation analysis on synthetic joint-space observation data."""
    print("\n" + "=" * 72)
    print("SECTION 4: FEATURE CORRELATION (SYNTHETIC)")
    print("=" * 72)

    # Generate synthetic observation data that reflects real kinematics
    N = 10000
    q = torch.rand(N, 5) * 3.0 - 1.5      # joint pos [-1.5, 1.5]
    qdot = torch.randn(N, 5) * 0.5          # joint vel
    q_target = q + torch.randn(N, 5) * 0.3   # target within ~0.3 rad
    qdot_target = torch.randn(N, 5) * 0.2
    q_error = q_target - q
    qdot_error = qdot_target - qdot
    prev_action = torch.randn(N, 4) * 0.5
    prev_valve = torch.randn(N, 4) * 0.5
    z_arm = torch.randn(N, 24) * 2.0

    obs = torch.cat([q, qdot, q_error, qdot_error, prev_action, prev_valve, z_arm], dim=-1)

    feature_names = [n for n, _, _ in OBS_SCHEMA_JOINT_SPACE]
    # Expand per-joint names
    per_joint_names = []
    for name, dim, _ in OBS_SCHEMA_JOINT_SPACE:
        if dim > 1 and name not in ("prev_action", "prev_valve_cmd", "z_arm_hat"):
            per_joint_names.extend(f"{name}_{i}" for i in range(dim))
        elif dim == 1:
            per_joint_names.append(name)
        else:
            per_joint_names.extend(f"{name}_{i}" for i in range(dim))

    print(f"  Computing correlation on {N} synthetic samples...")
    pairs = correlation_matrix(obs, per_joint_names, threshold=0.90)

    if pairs:
        print(f"  Found {len(pairs)} highly correlated feature pairs (|r| >= 0.90):")
        for p in pairs[:20]:
            print(f"    {p['feature_a']:20s} <-> {p['feature_b']:20s}  r={p['correlation']:+.4f}")
    else:
        print("  No highly correlated pairs found at r >= 0.90 (expected: noise dominates)")

    # Now do the targeted legacy redundancy check: q vs q_error vs q_ref
    print("\n  Historical redundancy among {q, q_ref, q_error}:")
    for j in range(5):
        q_col = q[:, j]
        qref_col = q_target[:, j]
        qerr_col = q_error[:, j]
        corr_q_ref = float(torch.corrcoef(torch.stack([q_col, qref_col]))[0, 1])
        corr_q_err = float(torch.corrcoef(torch.stack([q_col, qerr_col]))[0, 1])
        corr_ref_err = float(torch.corrcoef(torch.stack([qref_col, qerr_col]))[0, 1])
        print(f"    Joint {j}: corr(q,q_ref)={corr_q_ref:+.3f}  "
              f"corr(q,q_error)={corr_q_err:+.3f}  "
              f"corr(q_ref,q_error)={corr_ref_err:+.3f}")

    print("\n  Historical redundancy among {qdot, qdot_ref, qdot_error}:")
    for j in range(5):
        qd_col = qdot[:, j]
        qdref_col = qdot_target[:, j]
        qderr_col = qdot_error[:, j]
        corr_qd_qdref = float(torch.corrcoef(torch.stack([qd_col, qdref_col]))[0, 1])
        corr_qd_qderr = float(torch.corrcoef(torch.stack([qd_col, qderr_col]))[0, 1])
        print(f"    Joint {j}: corr(qdot,qdot_ref)={corr_qd_qdref:+.3f}  "
              f"corr(qdot,qdot_error)={corr_qd_qderr:+.3f}")


# ---------------------------------------------------------------------------
# Section 5: Task-space observation dimension contract check
# ---------------------------------------------------------------------------

def analyze_task_space_contract():
    """Validate the task-space observation dimension contract."""
    print("\n" + "=" * 72)
    print("SECTION 5: TASK-SPACE OBSERVATION CONTRACT CHECK")
    print("=" * 72)

    # Hard-coded from kevin_integration/tasks/task_space_reference.py
    # (avoids triggering full kevin_integration import chain)
    TASK_SPACE_OBSERVATION_DIM = 70
    TASK_SPACE_OBSERVATION_ORDER = (
        ("ee_orientation_command", 4),
        ("current_ee_pose", 7),
        ("ee_position_error", 3),
        ("ee_orientation_error", 1),
        ("mechanical_q", 5),
        ("mechanical_qdot", 5),
        ("cylinder_q", 4),
        ("cylinder_qdot", 4),
        ("previous_action", 4),
        ("previous_ValveCmd", 4),
        ("z_arm_hat", 24),
        ("fd_tau_hat", 4),
        ("vehicle_forward_speed_kmh", 1),
    )

    print(f"  Declared dim: {TASK_SPACE_OBSERVATION_DIM}")
    computed = sum(dim for _, dim in TASK_SPACE_OBSERVATION_ORDER)
    print(f"  Computed dim: {computed}")
    print(f"  Match: {computed == TASK_SPACE_OBSERVATION_DIM}")

    # Check that z_arm_hat occupies indices 41:65
    offset = 0
    for name, dim in TASK_SPACE_OBSERVATION_ORDER:
        if name == "z_arm_hat":
            print(f"\n  z_arm_hat occupies indices [{offset}:{offset + dim}] (24 dims)")
            assert offset == 41, f"Expected z_arm_hat start at 41, got {offset}"
            assert offset + dim == 65
        offset += dim

    # The hardcoded check in the env code
    print(f"\n  HARDCODED CHECK in _get_observations:")
    print(f"    obs[:, 41:65] must equal _z_arm_hat")
    print(f"  RISK: If TASK_SPACE_OBSERVATION_ORDER changes, this check silently")
    print(f"        validates the wrong slice. The check should use")
    print(f"        TASK_SPACE_OBSERVATION_ORDER to compute indices dynamically.")


# ---------------------------------------------------------------------------
# Section 6: Observation to reward gradient analysis
# ---------------------------------------------------------------------------

def analyze_obs_to_reward_gradient():
    """Analyze which observation components have nonzero gradient w.r.t. reward."""
    print("\n" + "=" * 72)
    print("SECTION 6: OBS->REWARD SIGNAL FLOW")
    print("=" * 72)

    print("""
  Joint-space reward terms and their observation dependencies:

  q_tracking      depends on: q_error = q_ref - q
                                -> needs: q, q_ref  OR  q_error alone
  qdot_penalty    depends on: qdot
                                -> needs: qdot
  progress        depends on: q_error (via EMA comparison)
                                -> needs: q_error
  settle_tracking depends on: q_error, is_settle_phase
                                -> needs: q_error, ramp_progress
  action_rate     depends on: action_delta (environment state, not observation)
                                -> NO observation dependency
  action_saturation depends on: action (environment state)
                                -> NO observation dependency
  valve_effort    depends on: valve_cmd (environment state)
                                -> NO observation dependency
  alive           depends on: terminated (environment state)
                                -> NO observation dependency

  OBSERVATION COMPONENTS WITH ZERO REWARD GRADIENT:
    - prev_action (4): no reward term uses previous action
    - prev_valve_cmd (4): valve_effort uses current valve_cmd, not prev
    - qdot_error (5): qdot_penalty uses actual qdot, not error
    - z_arm_hat (24): NO reward term depends on AAM latent!

  TOTAL DIMS WITH ZERO DIRECT REWARD SIGNAL: 37 out of 52 (71%)
  These dims only help the policy via state representation for the
  value function and action selection, but have no direct reward gradient.

  z_arm_hat is particularly notable: it's the LARGEST observation
  component (24 dims, 46% of total) with zero direct reward signal.
  The policy can only learn to use it through the advantage function.
""")


# ---------------------------------------------------------------------------
# Section 7: Suggested observation space improvements
# ---------------------------------------------------------------------------

def analyze_suggested_improvements():
    """Summarize evidence-based observation space improvements."""
    print("\n" + "=" * 72)
    print("SECTION 7: SUGGESTED OBSERVATION SPACE IMPROVEMENTS")
    print("=" * 72)

    print("""
  Based on the analysis above, here are evidence-backed improvements:

  1. REMOVE REDUNDANT FEATURES (applied conservatively):
     Joint-space: drop q_ref, keep q and q_error
                  drop qdot_ref, keep qdot and qdot_error
                  New size: 62 - 10 = 52 dims

     Task-space: drop only target position from ee_pose_command; keep target
                 orientation, current_ee_pose, and position/orientation errors
                 New size: 73 - 3 = 70 dims

  2. NORMALIZE z_arm_hat BEFORE CONCATENATION:
     Add a tanh or LayerNorm on the AAM output, or at minimum track
     per-dimension statistics and clip outliers.
     Even better: use running statistics of z_arm_hat from rollout data.

  3. CONSIDER DROPPING z_arm_hat FROM OBSERVATIONS:
     z_arm_hat has zero direct reward gradient and consumes 24 dims.
     The ID model already uses it internally. The policy might not
     need to see it. ABLATION TEST: train with and without z_arm_hat
     in observations; compare convergence speed and final performance.

  4. ADD prev_valve_cmd TO REWARD (MINOR):
     Currently valve_effort penalizes |valve_cmd|^2, which rewards
     the policy for keeping valves near zero. Adding a small penalty
     on valve_cmd_delta would encourage smoothness.

  5. REMOVE prev_action FROM OBSERVATIONS (ABLATION CANDIDATE):
     Like z_arm_hat, prev_action has zero direct reward gradient.
     The policy already sees q and qdot; the effect of previous
     actions is captured in the state. 4 dims saved.

  CONSERVATIVE FIRST STEP:
    - Drop q_ref, keep q and q_error (saves 5 dims)
    - Drop qdot_ref, keep qdot and qdot_error (saves 5 dims)
    - Drop task-space target position only; keep target orientation (saves 3 dims)
    - Add LayerNorm to z_arm_hat output
    - Keep everything else the same
    Expected: faster convergence with no loss of information
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("OBSERVATION SPACE ANALYSIS REPORT")
    print("=" * 72)
    print()

    analyze_structural_redundancy()
    analyze_aam_latent()
    analyze_aam_version_tracking()
    analyze_feature_correlation()
    analyze_task_space_contract()
    analyze_obs_to_reward_gradient()
    analyze_suggested_improvements()

    print("\n" + "=" * 72)
    print("ANALYSIS COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
