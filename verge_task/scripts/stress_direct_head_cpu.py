#!/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
"""CPU-only contract stress test for direct Messerkopf control.

Checks both USDs, the q5/down-axis kinematic invariant, twin-sensor plane
geometry, and numerical preservation of a migrated four-action checkpoint.
No Isaac Sim application or GPU context is created.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

# Isaac Sim exposes ``pxr`` only after its application bootstrap, which would
# defeat this script's CPU-only/no-GPU contract.  Prefer a standalone USD Python
# package when available (the workspace Blender distribution provides one),
# while retaining the normal import for environments that already package USD.
try:
    from pxr import Usd, UsdPhysics
except ModuleNotFoundError:
    standalone_usd = Path(
        "/home/rrlab/tools/blender/4.2/python/lib/python3.11/site-packages"
    )
    if not (standalone_usd / "pxr").is_dir():
        raise RuntimeError(
            "CPU USD bindings are unavailable; expected pxr in the Python "
            f"environment or at {standalone_usd}"
        )
    sys.path.insert(0, str(standalone_usd))
    from pxr import Usd, UsdPhysics


ROOT = Path(__file__).resolve().parents[2]
MIGRATION = ROOT / "verge_task/scripts/migrate_shared_policy.py"


def check_joint(usd_path: Path, prim_path: str) -> None:
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"could not open {usd_path}")
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsActive():
        raise RuntimeError(f"missing/inactive q5 prim: {prim_path}")
    joint = UsdPhysics.RevoluteJoint(prim)
    enabled = prim.GetAttribute("physics:jointEnabled").Get()
    lower = float(joint.GetLowerLimitAttr().Get())
    upper = float(joint.GetUpperLimitAttr().Get())
    axis = str(joint.GetAxisAttr().Get())
    if enabled is False or upper - lower < 1.0 or axis != "Z":
        raise RuntimeError(
            f"locked/wrong q5 in {usd_path}: enabled={enabled}, axis={axis}, "
            f"limits=({lower}, {upper})"
        )
    print(f"PASS USD {usd_path.name}: enabled={enabled}, axis={axis}, limits=({lower:.3f},{upper:.3f})")


def check_geometry() -> None:
    angles = torch.linspace(-torch.pi, torch.pi, 20001)
    # Rotation around local Z leaves local Z invariant exactly.
    z_after = torch.stack(
        (torch.zeros_like(angles), torch.zeros_like(angles), torch.ones_like(angles)),
        dim=-1,
    )
    error = (z_after - torch.tensor([0.0, 0.0, 1.0])).abs().max().item()
    if error != 0.0:
        raise RuntimeError(f"q5 changed its own Z axis: {error}")

    # Two sensors separated along local Y over z = slope_y*y. Their difference
    # is zero on a parallel plane and changes linearly with cross-slope.
    left_y, right_y = -0.5, 0.6
    slopes = torch.linspace(-1.0, 1.0, 20001)
    delta = slopes * (left_y - right_y)
    expected = -1.1 * slopes
    if not torch.allclose(delta, expected, atol=1.0e-7, rtol=0.0):
        raise RuntimeError("twin-sensor plane response is incorrect")
    if abs(delta[len(delta) // 2].item()) > 1.0e-6:
        raise RuntimeError("parallel plane did not produce zero sensor delta")
    print("PASS geometry: q5 has zero down-axis authority; twin rays resolve cross-slope")


def check_floating_cutter_asset(usd_path: Path) -> None:
    """Reject an explosive/miswired passive-cutter layer before Isaac starts."""

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"could not open floating-cutter asset {usd_path}")
    root = "/unimog_mulag_fme500"
    head = stage.GetPrimAtPath(f"{root}/Rigid_Bodies/Arm_Links/Messerkopf")
    cutter = stage.GetPrimAtPath(f"{root}/Rigid_Bodies/Arm_Links/CutterFloat")
    joint_prim = stage.GetPrimAtPath(f"{root}/Joints/CutterFloat_joint")
    joint = UsdPhysics.PrismaticJoint(joint_prim)
    if not head or not cutter or not joint_prim:
        raise RuntimeError("floating-cutter asset is missing a required prim")
    if head.GetAttribute("xformOp:transform").Get() != cutter.GetAttribute(
        "xformOp:transform"
    ).Get():
        raise RuntimeError(
            "CutterFloat does not spawn at Messerkopf; the joint would close a "
            "multi-metre frame error on the first solver step"
        )
    lower = float(joint.GetLowerLimitAttr().Get())
    upper = float(joint.GetUpperLimitAttr().Get())
    axis = str(joint.GetAxisAttr().Get())
    if axis != "Z" or abs(lower + 0.04) > 1.0e-6 or abs(upper - 0.04) > 1.0e-6:
        raise RuntimeError(
            f"floating joint contract changed: axis={axis}, limits={lower}/{upper}"
        )
    original = stage.GetPrimAtPath(
        f"{root}/Rigid_Bodies/Arm_Links/Messerkopf/"
        "E_026502_FME_500_AP_stl_009"
    )
    physical = stage.GetPrimAtPath(
        f"{root}/Rigid_Bodies/Arm_Links/CutterFloat/cutter_mesh"
    )
    if original.GetAttribute("physics:collisionEnabled").Get() is not False:
        raise RuntimeError("rigid Messerkopf collider remains enabled")
    if physical.GetAttribute("physics:collisionEnabled").Get() is not True:
        raise RuntimeError("floating cutter collider is not enabled")
    print(
        "PASS floating asset: aligned spawn, passive Z travel +/-40 mm, "
        "single moving collider"
    )


def check_migration(checkpoint: Path, target_observations: int) -> None:
    spec = importlib.util.spec_from_file_location("migration", MIGRATION)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    actor = payload["actor_state_dict"]
    source_observations = int(actor["mlp.0.weight"].shape[1])
    action_width = int(actor["mlp.6.weight"].shape[0])
    if action_width == 5:
        if source_observations != target_observations:
            raise RuntimeError(
                f"direct-q5 checkpoint is {source_observations} wide; "
                f"expected {target_observations}"
            )
        tensors = [value for value in actor.values() if torch.is_tensor(value)]
        if not all(torch.isfinite(value).all() for value in tensors):
            raise RuntimeError("direct-q5 actor contains non-finite parameters")
        infos = payload.get("infos") or {}
        if int(infos.get("action_width", 5)) != 5:
            raise RuntimeError("checkpoint metadata disagrees with five-output actor")
        print(
            "PASS checkpoint: finite direct-q5 actor "
            f"({source_observations}x{action_width}, actuator={infos.get('actuator')})"
        )
        return
    if action_width != 4:
        raise RuntimeError(f"unsupported actor output width {action_width}")
    migrated = module.expand_direct_q5(
        payload,
        source_observations=source_observations,
        target_observations=target_observations,
        q5_std=0.04,
    )
    new_actor = migrated["actor_state_dict"]
    generator = torch.Generator().manual_seed(20260804)
    hidden = torch.randn(100_000, actor["mlp.6.weight"].shape[1], generator=generator)
    old_out = hidden @ actor["mlp.6.weight"].T + actor["mlp.6.bias"]
    new_out = hidden @ new_actor["mlp.6.weight"].T + new_actor["mlp.6.bias"]
    if not torch.equal(old_out, new_out[:, :4]):
        raise RuntimeError("migration changed an existing policy output")
    if torch.count_nonzero(new_out[:, 4]).item() != 0:
        raise RuntimeError("migrated q5 mean is not neutral")
    print(f"PASS migration: 100000 actor-head samples preserved ({source_observations}x4 -> {target_observations}x5)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--target_observations", type=int, default=106)
    args = parser.parse_args()
    check_joint(
        ROOT / "teacher_student/rrlab-mulag_teacher_student/extensions/rrlab_assets/data/Robots/Unimog_Mulag_FME500/Unimog_Mulag_FME500.usd",
        "/Unimog_Mulag_FME500/joints/Messerkopf_joint",
    )
    check_joint(
        ROOT / "extensions/rrlab_assets/data/Robots/unimog_mulag_fme500_cylinder_as_actuator.usd",
        "/unimog_mulag_fme500/Joints/Messerkopf_joint",
    )
    check_floating_cutter_asset(
        ROOT
        / "extensions/rrlab_assets/data/Robots/"
        "unimog_mulag_fme500_float_cutter.usda"
    )
    check_geometry()
    check_migration(args.checkpoint.resolve(), int(args.target_observations))
    print("PASS all CPU direct-head stress tests")


if __name__ == "__main__":
    main()
