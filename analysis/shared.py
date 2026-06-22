"""Shared utilities for pipeline analysis scripts."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import math
import torch
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Observation dimension schemas (ground-truth reference)
# ---------------------------------------------------------------------------

OBS_SCHEMA_JOINT_SPACE: List[Tuple[str, int, int]] = [
    ("q",                  5,  0),
    ("qdot",               5,  5),
    ("q_error",            5,  10),
    ("qdot_error",         5,  15),
    ("prev_action",        4,  20),
    ("prev_valve_cmd",     4,  24),
    ("z_arm_hat",          24, 28),
]
JOINT_OBS_DIM = 52

OBS_SCHEMA_TASK_SPACE: List[Tuple[str, int, int]] = [
    ("ee_orientation_command", 4,  0),
    ("current_ee_pose",        7,  4),
    ("ee_position_error",      3,  11),
    ("ee_orientation_error",   1,  14),
    ("mechanical_q",           5,  15),
    ("mechanical_qdot",        5,  20),
    ("cylinder_q",             4,  25),
    ("cylinder_qdot",          4,  29),
    ("previous_action",        4,  33),
    ("previous_ValveCmd",      4,  37),
    ("z_arm_hat",              24, 41),
    ("fd_tau_hat",             4,  65),
    ("vehicle_speed_kmh",      1,  69),
]
TASK_SPACE_OBS_DIM = 70


def schema_to_slices(schema):
    return {name: slice(offset, offset + dim) for name, dim, offset in schema}


def detect_redundant_groups(schema):
    groups = []
    names = [n for n, _, _ in schema]

    if {"q", "q_ref", "q_error"}.issubset(set(names)):
        groups.append({
            "group": ["q", "q_ref", "q_error"],
            "relationship": "q_error = q_ref - q",
            "redundant_dims": 5,
            "keep": ["q", "q_error"],
            "reason": "q_ref is recoverable as q + q_error",
        })
    if {"qdot", "qdot_ref", "qdot_error"}.issubset(set(names)):
        groups.append({
            "group": ["qdot", "qdot_ref", "qdot_error"],
            "relationship": "qdot_error = qdot_ref - qdot",
            "redundant_dims": 5,
            "keep": ["qdot", "qdot_error"],
            "reason": "qdot_ref is recoverable as qdot + qdot_error",
        })
    if {"ee_pose_command", "current_ee_pose", "ee_position_error", "ee_orientation_error"}.issubset(set(names)):
        groups.append({
            "group": ["ee_pose_command", "current_ee_pose", "ee_position_error"],
            "relationship": "position_error = command_pos - current_pos",
            "redundant_dims": 3,
            "keep": ["ee_position_error", "current_ee_pose", "ee_orientation_command"],
            "reason": "target position is recoverable; target orientation must remain explicit",
        })
    return groups


# ---------------------------------------------------------------------------
# Reward term profiles
# ---------------------------------------------------------------------------

@dataclass
class RewardTermProfile:
    name: str
    weight: float
    sign: str
    typical_range: Tuple[float, float]
    saturation_behavior: str


def joint_reach_reward_profile():
    return [
        RewardTermProfile("q_tracking", 3.0, "positive", (-3.0, 0.0),
                          "quadratic decay: -3.0 * mean(error^2)"),
        RewardTermProfile("qdot_penalty", 0.25, "negative", (-0.25, 0.0),
                          "quadratic in qdot"),
        RewardTermProfile("progress", 2.0, "positive", (-0.5, 0.5),
                          "EMA-smoothed error delta"),
        RewardTermProfile("settle_tracking", 2.0, "positive", (-2.0, 0.0),
                          "active only when ramp_progress > 0.95"),
        RewardTermProfile("joint1_settle", 3.0, "positive", (-3.0, 0.0),
                          "deadband 0.005 rad on joint1"),
        RewardTermProfile("action_rate", 0.05, "negative", (-0.05, 0.0),
                          "quadratic in action_delta"),
        RewardTermProfile("action_magnitude", 0.02, "negative", (-0.02, 0.0),
                          "quadratic in |action|"),
        RewardTermProfile("action_saturation", 0.5, "negative", (-0.5, 0.0),
                          "threshold 0.90; tanh(square(excess))"),
        RewardTermProfile("valve_effort", 0.01, "negative", (-0.01, 0.0),
                          "quadratic in valve_cmd"),
        RewardTermProfile("torque_effort", 0.0, "negative", (0.0, 0.0),
                          "disabled by default; observer-only torque is not actuated in ROM mode"),
        RewardTermProfile("alive", 0.1, "positive", (0.0, 0.1),
                          "constant per step while alive"),
        RewardTermProfile("termination", 5.0, "negative", (-5.0, 0.0),
                          "one-shot on termination"),
    ]


def simulate_reward_at_error(
    q_error_norm=0.01,
    qdot_norm=0.1,
    action_magnitude=0.3,
    action_delta=0.05,
    valve_cmd_norm=0.2,
    is_settle=False,
    progress_delta=0.0,
    terminated=False,
):
    r = {}
    r["q_tracking"] = -3.0 * q_error_norm * q_error_norm
    r["qdot_penalty"] = -0.25 * 5.0 * qdot_norm * qdot_norm
    r["progress"] = 2.0 * progress_delta
    r["settle_tracking"] = -2.0 * q_error_norm * q_error_norm if is_settle else 0.0
    r["joint1_settle"] = -3.0 * max(0.0, q_error_norm - 0.005) ** 2 if is_settle else 0.0
    r["action_rate"] = -0.05 * 4.0 * action_delta * action_delta
    r["action_magnitude"] = -0.02 * 4.0 * action_magnitude * action_magnitude
    excess = max(0.0, action_magnitude - 0.90)
    r["action_saturation"] = -0.5 * 4.0 * math.tanh(excess * excess)
    r["valve_effort"] = -0.01 * 4.0 * valve_cmd_norm * valve_cmd_norm
    r["torque_effort"] = -1e-6 * 4.0 * 0.0
    r["alive"] = 0.1 if not terminated else 0.0
    r["termination"] = -5.0 if terminated else 0.0

    r["positive_sum"] = sum(v for v in r.values() if v > 0)
    r["negative_sum"] = sum(v for v in r.values() if v < 0)
    r["total"] = r["positive_sum"] + r["negative_sum"]
    r["neg_to_pos_ratio"] = abs(r["negative_sum"]) / max(abs(r["positive_sum"]), 1e-8)
    return r


# ---------------------------------------------------------------------------
# Action pipeline helpers
# ---------------------------------------------------------------------------

@dataclass
class ActionScaleConfig:
    action_to_qddot_scale: Any = 0.5
    qddot_cmd_clip: Any = 5.0
    id_feature_std: Optional[List[float]] = None

    def effective_scale(self):
        if isinstance(self.action_to_qddot_scale, str):
            if self.id_feature_std is None:
                return 1.0
            std = torch.as_tensor(self.id_feature_std, dtype=torch.float32)
            qddot_std = std[10:14].mean().item()
            return max(qddot_std, 1e-8)
        return float(self.action_to_qddot_scale)

    def effective_clip(self):
        if isinstance(self.qddot_cmd_clip, str):
            if self.id_feature_std is None:
                return 5.0
            std = torch.as_tensor(self.id_feature_std, dtype=torch.float32)
            qddot_std = std[10:14].mean().item()
            return max(qddot_std, 1e-8)
        return float(self.qddot_cmd_clip)

    def trace_action(self, action_4d):
        scale = self.effective_scale()
        clip = self.effective_clip()
        qddot_4d = scale * action_4d
        qddot_clipped = torch.clamp(qddot_4d, -clip, clip)
        clip_fraction = (qddot_4d != qddot_clipped).float().mean()
        return {
            "action": action_4d,
            "qddot_4d": qddot_4d,
            "qddot_clipped": qddot_clipped,
            "clip_fraction": clip_fraction,
            "scale": scale,
            "clip_limit": clip,
        }


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def compute_feature_stats(features, feature_names):
    if features.dim() == 2:
        dim = 0
    else:
        dim = None

    if dim is None:
        return {"all": {
            "mean": float(features.mean()),
            "std": float(features.std(unbiased=False)),
            "min": float(features.min()),
            "max": float(features.max()),
            "abs_max": float(features.abs().max()),
        }}

    stats = {}
    for i, name in enumerate(feature_names):
        col = features[:, i]
        stats[name] = {
            "mean": float(col.mean()),
            "std": float(col.std(unbiased=False)),
            "min": float(col.min()),
            "max": float(col.max()),
            "abs_max": float(col.abs().max()),
        }
    return stats


def correlation_matrix(features, feature_names, threshold=0.95):
    n = features.shape[-1]
    if n != len(feature_names):
        raise ValueError(f"Feature count mismatch: {n} vs {len(feature_names)}")
    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, keepdim=True).clamp(min=1e-8)
    z = (features - mean) / std
    corr = (z.T @ z) / (features.shape[0] - 1)

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            c = float(corr[i, j])
            if abs(c) >= threshold:
                pairs.append({
                    "feature_a": feature_names[i],
                    "feature_b": feature_names[j],
                    "correlation": round(c, 6),
                    "idx_a": i, "idx_b": j,
                })
    return sorted(pairs, key=lambda p: -abs(p["correlation"]))
