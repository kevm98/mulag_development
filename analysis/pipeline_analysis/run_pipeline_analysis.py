#!/usr/bin/env python3
"""Full control pipeline trace: signal flow, AAM properties, FD range, ROM coupling.

Run: python analysis/pipeline_analysis/run_pipeline_analysis.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch


# ---------------------------------------------------------------------------
# Section 1: End-to-end signal flow diagram
# ---------------------------------------------------------------------------

def analyze_signal_flow():
    """Trace the full signal path from PPO observation to joint effort."""
    print("=" * 72)
    print("SECTION 1: END-TO-END SIGNAL FLOW")
    print("=" * 72)

    print("""
  PPO POLICY
    |  observation (52-dim joint or 70-dim task-space)
    |  produces: action (4-dim, ~[-1, 1] from Gaussian head)
    v
  ACTION SCALING (build_qddot_cmd)
    |  qddot_4d = scale * action           (scale=0.5 or "id_qddot_std")
    |  qddot_clipped = clamp(qddot_4d, -clip, +clip)
    |  qddot_5d = [qddot_clipped, 0]       (tool joint always zero)
    v
  ID MODEL (InverseDynamicsModel)
    |  input:  [q(5), qdot(5), qddot_5d, dP(4)]  = 19-dim
    |  conditioned on: z_arm_hat (24-dim AAM latent)
    |  output: valve_cmd (4-dim, tanh-bounded to [-1,1])
    v
  FD OBSERVER (ForwardDynamicsModel, optional)
    |  input:  [q(5), qdot(5), dP(4), valve_cmd(4)]  = 18-dim
    |  output: fd_tau (4-dim Simscape torque estimate)
    v
  TORQUE ADAPTER (adapt_fd_to_isaac_torque)
    |  mode: "scale_bias" (default) or "tanh_squash" or "residual_pd"
    |  tau_isaac = scale * fd_tau + bias
    |  scaled by fd_residual_alpha=0.002
    v
  TORQUE FILTERS
    |  lowpass:  tau_filt = prev + alpha * (tau - prev), alpha=0.2
    |  rate_limit: tau = prev + clamp(delta, -1.0, +1.0)
    |  clamp:    tau_final = clamp(tau, -5.0, +5.0)
    v
  ISAAC LAB
    |  set_joint_effort_target(tau_final)
    v
  PHYSICS STEP  (100 Hz)
    |  q_new, qdot_new, dP_new
    v
  AAM UPDATE (_update_aam_latent)
    |  input:  [q(5), qdot(5), dP(4), prev_valve_cmd(4)] = 18-dim
    |  window: LSTM over history
    |  output: z_arm_hat (24-dim, UNNORMALIZED)
    v
  OBSERVATION ASSEMBLY (_get_observations)
    |  concat: q, qdot, q_error, qdot_error,
    |          prev_action, prev_valve_cmd, z_arm_hat
    v
  (back to PPO POLICY)
""")


# ---------------------------------------------------------------------------
# Section 2: Torque pipeline numerical trace
# ---------------------------------------------------------------------------

def analyze_torque_pipeline_trace():
    """Numerical trace of an action through the full torque pipeline."""
    print("\n" + "=" * 72)
    print("SECTION 2: TORQUE PIPELINE NUMERICAL TRACE")
    print("=" * 72)

    print("""
  SYNTHETIC TRACE (conservative torque adapter, alpha=0.002, max_torque=5.0):

  Step 1: RL action = [0.5, -0.3, 0.8, -0.1]
  Step 2: qddot = 0.5 * action = [0.25, -0.15, 0.4, -0.05]  (fixed scale)
  Step 3: ID model -> valve = tanh(gain * qddot) ~ [0.05, -0.03, 0.08, -0.01]
  Step 4: FD observer -> fd_tau (Simscape) ~ [1.2e5, -0.8e5, 2.0e5, -0.3e5]
  Step 5: Torque adapter (conservative):
           tau = scale * fd_tau + bias
           where scale = [1e-4, 1e-3, 5e-4, 1e-4] * 0.002
                       = [2e-7, 2e-6, 1e-6, 2e-7]
           tau   = [0.024, -0.16, 0.20, -0.006] Nm
  Step 6: Lowpass (alpha=0.2, prev=0):
           tau_filt = [0.0048, -0.032, 0.04, -0.0012] Nm
  Step 7: Rate limit (1.0): no effect at these magnitudes
  Step 8: Clamp (-5.0, 5.0): no effect

  FINAL TORQUE: ~[0.005, -0.03, 0.04, -0.001] Nm

  This is TINY. The MulaG arm weighs hundreds of kg.
  Joint 1 alone has a static load >> 1 Nm.

  VERDICT: The conservative torque adapter with fd_residual_alpha=0.002
  produces negligible torque. This path is effectively disabled.
""")


# ---------------------------------------------------------------------------
# Section 3: AAM model properties
# ---------------------------------------------------------------------------

def analyze_aam_properties():
    """Analyze AAM architecture stability properties."""
    print("\n" + "=" * 72)
    print("SECTION 3: AAM MODEL PROPERTIES")
    print("=" * 72)

    print("""
  AAM Architecture (from learned_models.py):
    temporal_encoder: LSTM(input=18, hidden=H, num_layers=L)
    head: Sequential(
      Linear(H -> head_dim),
      Activation (ReLU/ELU),
      Dropout,
      Linear(head_dim -> 24)
    )

  STABILITY CONCERNS:
  1. The final Linear layer has no activation. Output is unbounded.
     A single large input could produce latents in [-1000, 1000].
  2. The LSTM has no output normalization (no LayerNorm between
     encoder and head).
  3. Feature normalization is done at the INPUT level (q, qdot, dP,
     valve) but the latent SPACE has no regularization.

  INFERENCE SAFETY:
  - torch.inference_mode() used -> no gradients, dropout disabled
  - requires_grad_(False) + .eval() -> redundant but safe
  - Finiteness check after each forward pass -> catches NaN/Inf

  MISSING:
  - No output clipping or normalization
  - No running statistics tracking for latent distribution
  - No activation on final head layer

  RECOMMENDED FIX:
  Add nn.Tanh() after the final Linear in the AAM head during
  inference. This bounds z_arm_hat to [-1, 1] and prevents
  observation-space domination.

  If retraining the AAM with tanh is impossible, clip the output:
    z_arm_hat = torch.clamp(aam_output, -10.0, 10.0)
""")


# ---------------------------------------------------------------------------
# Section 4: ROM coupling analysis
# ---------------------------------------------------------------------------

def analyze_rom_coupling():
    """Analyze the final causal ROM integration coupling."""
    print("\n" + "=" * 72)
    print("SECTION 4: FINAL CAUSAL ROM COUPLING")
    print("=" * 72)

    print("""
  The final causal ROM path (use_final_causal_cylinder_rom=True):

  Control flow:
    _prepare_high_level_action  -> determines q_ref, action
    _pre_physics_step_final_causal_rom:
      1. ID model -> valve_cmd (if rom_use_id_valve_cmd)
      2. FD observer -> fd_tau_hat (for observation only)
      3. SD observer (if use_sd_observer)
      4. SD feedback -> dP, fnet (if rom_use_sd_feedback)
      5. Build ROM feature row from [mechanical_q, qdot, dP, fnet, valve]
      6. ROM predicts delta_extension at 50ms horizon
      7. Apply cylinder coordinate mapping: extension -> isaac position
      8. Rate limit the target, clamp to limits
    _apply_action:
      set_joint_position_target(rom_cylinder_pos_target)

  KEY OBSERVATION:
  In this path, the ID model produces ValveCmd that feeds the ROM,
  but the FD-derived torque is NEVER applied as joint effort.
  The FD is observer-only, used for:
    - fd_tau_hat in task-space observation (dim 65:69)
    - fd_load_excess_penalty in reward

  This means the torque adapter, lowpass, rate limit, and
  torque clamping are all DEAD CODE in ROM mode.

  FROM ROM INTEGRATION CSV (300 samples, 20260610_151725):
    - dP values: ALL ZERO (pressure source = training_mean)
    - ValveCmd: ALL ZERO (no ID model active in that test)
    - ROM delta at 50ms: ~1e-3 to ~2e-3 m extension
    - Joint motion: 0.02-0.05 rad per step for joints 2-4
    - Joint 1 (Drehzapfen): ~0.003 rad, barely moves

  JOINT 1 AUTHORITY GAP:
  The ROM integration CSV shows joint 1 (Drehzapfen, the base rotation)
  moves only ~0.003 rad while joints 2-4 move 0.02-0.05 rad.
  This is consistent with the q_tracking_reward_joint_weights:
    [1.5, 1.0, 1.0, 1.0, 0.0]

  Joint 1 has 1.5x weight but the ROM drive authority may be
  insufficient. The joint1_settle_tracking reward (3.0x weight
  with 0.005 rad deadband) tries to compensate.
""")


# ---------------------------------------------------------------------------
# Section 5: FD observer range analysis from checkpoint metadata
# ---------------------------------------------------------------------------

def analyze_fd_range():
    """Analyze FD observer output range from checkpoint metadata."""
    print("\n" + "=" * 72)
    print("SECTION 5: FD OBSERVER OUTPUT RANGE")
    print("=" * 72)

    print("""
  The FD model output is denormalized by the scaler checkpoint:
    y_denorm = y_norm * y_std + y_mean

  Without loading the actual scaler checkpoint (fd_scaler.pth),
  we can reason about ranges:

  If y_std ~ 1e6 and y_mean ~ 0:
    FD output range (3-sigma) ~ [-3e6, 3e6]
    After conservative adapter (alpha=0.002, scale~2e-7):
      tau_range ~ [-6e-1, 6e-1] Nm

  If y_std ~ 1e7:
    FD output range ~ [-3e7, 3e7]
    tau_range ~ [-6, 6] Nm -> clamped to [-5, 5]

  The fd_torque_scale parameter (1e6) is used only in tanh_squash mode,
  not in scale_bias mode. This means the parameter is effectively
  unused in the default configuration.

  RECOMMENDATION: Load fd_scaler.pth and print y_std to determine
  the actual operating range. Use that to tune torque_adapter_scale.
""")


# ---------------------------------------------------------------------------
# Section 6: Learnability assessment
# ---------------------------------------------------------------------------

def analyze_learnability():
    """Synthesize learnability assessment from all analyses."""
    print("\n" + "=" * 72)
    print("SECTION 6: LEARNABILITY ASSESSMENT")
    print("=" * 72)

    print("""
  WHAT HELPS LEARNING:
    (+) Tanh-bounded valve commands -> stable action space
    (+) LSTM temporal encoders -> capture dynamics history
    (+) Well-designed valve saturation penalty (smooth onset)
    (+) Deadband on joint1 settle tracking (avoids jitter at target)
    (+) qddot clipping prevents ID model from seeing crazy inputs
    (+) Alive bonus provides consistent positive signal per step

  WHAT HURTS LEARNING:
    (-) 37/52 joint observation dims have ZERO direct reward gradient (Section 6, obs)
    (-) z_arm_hat (24 dims) can dominate observation variance (Section 2, obs)
    (-) Action penalties exceed tracking reward for errors <0.035 rad (Section 4, reward)
    (-) Progress EMA creates oscillation exploit (Section 3, reward)
    (-) Torque effort reward is dead (Section 5, reward)
    (-) Double penalty: action_saturation (0.5) + valve_saturation (0.15) (Section 3, action)
    (-) Conservative torque adapter produces negligible torque (Section 4, action)
    (-) Clipping diagnostics never fire (Section 5, action)
    (-) legacy q_ref/qdot_ref redundancy was removed from joint-space observations

  ESTIMATED IMPACT OF FIXES (ordered by expected benefit):
    1. Drop lossless redundant obs dims (q_ref, qdot_ref, target position)
       -> 13 dims saved, faster convergence with no known information loss (HIGH IMPACT)
    2. Normalize z_arm_hat to [-1, 1] via tanh -> removes 99.7% variance
       domination risk (HIGH IMPACT)
    3. Reduce action_rate to 0.02, action_magnitude to 0.01 ->
       fixes perverse stop-moving incentive (MEDIUM IMPACT)
    4. Reduce progress_ema_alpha to 0.3 or higher -> reduces EMA lag
       from 5 steps to 2-3 steps (MEDIUM IMPACT)
    5. Consolidate action_saturation + valve_saturation -> one penalty
       at weight 0.2 -> simpler reward landscape (LOW-MEDIUM IMPACT)
    6. Increase torque_adapter to "moderate" preset -> makes FD path
       usable if needed (LOW IMPACT, only matters in effort mode)
    7. Set SD clamp limits from real data -> makes diagnostics useful
       for catching model divergence (LOW IMPACT, diagnostic only)
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("PIPELINE TRACE ANALYSIS REPORT")
    print("=" * 72)
    print()

    analyze_signal_flow()
    analyze_torque_pipeline_trace()
    analyze_aam_properties()
    analyze_rom_coupling()
    analyze_fd_range()
    analyze_learnability()

    print("=" * 72)
    print("ANALYSIS COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
