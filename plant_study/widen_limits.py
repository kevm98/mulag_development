"""Widen the USD cylinder joint limits to the stroke the ID/FD models were
trained on (Simscape), and keep the cylinder-space scaling consistent.

The USD clamps the cylinders to 40-75% of their real stroke, which is why the
tool bottoms out ~0.66 m above the ground. The ID/FD models were trained on the
FULL stroke, so widening is also the physically correct thing to do.

cylinder x -> USD q is affine with scale = -1:  q = -x + offset
so  q_lo = offset - x_hi ,  q_hi = offset - x_lo
"""
from __future__ import annotations

import numpy as np
import torch

# Per-cylinder extension range present in the FD training data (Simscape).
SIMSCAPE_EXT_LO = np.array([0.0182, 0.2753, 0.3092, 0.2691], dtype=np.float64)
SIMSCAPE_EXT_HI = np.array([0.4933, 0.7910, 0.8569, 0.6891], dtype=np.float64)


def widen_cylinder_limits(env, *, margin: float = 0.0, verbose: bool = True):
    """Widen the 4 cylinder joints to the Simscape stroke, in PhysX AND in the
    cylinder-space scaling (learned_min/max), so the controllers can use them.

    Returns the original (limits, learned_min, learned_max) so it can be reverted.
    """
    robot = env.robot
    sc = env._cylinder_adapter.scaling
    cyl_ids = env._cylinder_joint_ids
    ids = cyl_ids.tolist() if hasattr(cyl_ids, "tolist") else list(cyl_ids)

    scale = sc.scale.numpy().astype(np.float64)
    offset = sc.offset.numpy().astype(np.float64)

    ext_lo = SIMSCAPE_EXT_LO - margin
    ext_hi = SIMSCAPE_EXT_HI + margin
    q_a = scale * ext_lo + offset
    q_b = scale * ext_hi + offset
    q_lo = np.minimum(q_a, q_b)
    q_hi = np.maximum(q_a, q_b)

    original = robot.data.soft_joint_pos_limits.detach().clone()
    orig_learned_min = sc.learned_min.clone()
    orig_learned_max = sc.learned_max.clone()

    new_limits = torch.zeros((env.num_envs, len(ids), 2), device=env.device, dtype=torch.float32)
    new_limits[..., 0] = torch.as_tensor(q_lo, device=env.device, dtype=torch.float32)
    new_limits[..., 1] = torch.as_tensor(q_hi, device=env.device, dtype=torch.float32)
    robot.write_joint_limits_to_sim(new_limits, joint_ids=cyl_ids)

    # keep the cylinder-space scaling consistent with the new stroke
    # CylinderSpaceScaling is a frozen dataclass
    object.__setattr__(sc, "learned_min",
                       torch.as_tensor(np.minimum(ext_lo, ext_hi), dtype=sc.learned_min.dtype))
    object.__setattr__(sc, "learned_max",
                       torch.as_tensor(np.maximum(ext_lo, ext_hi), dtype=sc.learned_max.dtype))
    object.__setattr__(sc, "usd_min", torch.as_tensor(q_lo, dtype=sc.usd_min.dtype))
    object.__setattr__(sc, "usd_max", torch.as_tensor(q_hi, dtype=sc.usd_max.dtype))

    if verbose:
        cur = original[0].cpu().numpy()
        print("\n[WIDEN] cylinder joint limits (rad/m):", flush=True)
        for i, j in enumerate(ids):
            print(f"[WIDEN]   cyl{i}: {cur[j,0]:+.3f}..{cur[j,1]:+.3f}  ->  "
                  f"{q_lo[i]:+.3f}..{q_hi[i]:+.3f}", flush=True)
        print(f"[WIDEN] learned extension range -> lo={np.round(ext_lo,3).tolist()} "
              f"hi={np.round(ext_hi,3).tolist()}\n", flush=True)

    return original, orig_learned_min, orig_learned_max


def revert_cylinder_limits(env, saved, verbose: bool = True):
    """Restore the original USD limits and scaling."""
    original, orig_min, orig_max = saved
    robot = env.robot
    cyl_ids = env._cylinder_joint_ids
    sc = env._cylinder_adapter.scaling
    ids = cyl_ids.tolist() if hasattr(cyl_ids, "tolist") else list(cyl_ids)
    robot.write_joint_limits_to_sim(original[:, ids, :], joint_ids=cyl_ids)
    object.__setattr__(sc, "learned_min", orig_min)
    object.__setattr__(sc, "learned_max", orig_max)
    if verbose:
        print("[WIDEN] REVERTED to the original USD limits.", flush=True)
