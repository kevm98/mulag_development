from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser(description="Inspect the active Mulag USD asset used by Kevin adaptive RL.")
parser.add_argument("--task", type=str, default="Kevin-Mulag-AdaptiveRL-JointReach-v0", help="Gym task name.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg
from isaaclab_tasks.utils.hydra import hydra_task_config

import kevin_integration.tasks  # noqa: F401
from kevin_integration.tasks.mulag_adaptive_rl_env_cfg import MulagAdaptiveRLJointReachEnvCfg
import rrlab_assets.mulag as mulag_asset_cfg


JOINT_TYPE_NAMES = (
    "RevoluteJoint",
    "PrismaticJoint",
    "FixedJoint",
    "SphericalJoint",
    "DistanceJoint",
    "Joint",
)


def _attr_value(prim, name: str, default: Any = "unavailable") -> Any:
    attr = prim.GetAttribute(name)
    if attr and attr.HasAuthoredValueOpinion():
        return attr.Get()
    return default


def _rel_targets(prim, name: str) -> list[str]:
    rel = prim.GetRelationship(name)
    if not rel:
        return []
    return [str(target) for target in rel.GetTargets()]


def _api_applied(prim, api_name: str) -> bool:
    try:
        return prim.HasAPI(api_name)
    except Exception:
        return api_name in [str(item) for item in prim.GetAppliedSchemas()]


def _schema_type(prim, UsdPhysics) -> str | None:
    for type_name in JOINT_TYPE_NAMES:
        schema = getattr(UsdPhysics, type_name, None)
        if schema is not None and prim.IsA(schema):
            return type_name
    return None


def _drive_values(prim, UsdPhysics) -> list[tuple[str, dict[str, Any]]]:
    values: list[tuple[str, dict[str, Any]]] = []
    for drive_name in ("angular", "linear", "rotX", "rotY", "rotZ", "transX", "transY", "transZ"):
        try:
            drive = UsdPhysics.DriveAPI.Get(prim, drive_name)
        except Exception:
            drive = None
        if drive and drive.GetPrim().IsValid():
            values.append(
                (
                    drive_name,
                    {
                        "stiffness": drive.GetStiffnessAttr().Get(),
                        "damping": drive.GetDampingAttr().Get(),
                        "maxForce": drive.GetMaxForceAttr().Get(),
                        "targetPosition": drive.GetTargetPositionAttr().Get(),
                        "targetVelocity": drive.GetTargetVelocityAttr().Get(),
                    },
                )
            )
    return values


def _joint_info(prim, UsdPhysics, PhysxSchema) -> dict[str, Any] | None:
    type_name = _schema_type(prim, UsdPhysics)
    name = prim.GetName()
    path = str(prim.GetPath())
    if type_name is None and "joint" not in name.lower():
        return None

    applied_schemas = [str(item) for item in prim.GetAppliedSchemas()]
    physx_mimic = any("Mimic" in schema for schema in applied_schemas)
    if not physx_mimic and PhysxSchema is not None:
        try:
            mimic_schema = getattr(PhysxSchema, "PhysxMimicJointAPI", None)
            physx_mimic = bool(mimic_schema and prim.HasAPI(mimic_schema))
        except Exception:
            pass

    lower = _attr_value(prim, "physics:lowerLimit")
    upper = _attr_value(prim, "physics:upperLimit")
    locked = False
    try:
        locked = lower != "unavailable" and upper != "unavailable" and abs(float(upper) - float(lower)) < 1.0e-8
    except (TypeError, ValueError):
        locked = False

    return {
        "path": path,
        "name": name,
        "type": type_name or "unknown_joint_like_prim",
        "parent": _rel_targets(prim, "physics:body0"),
        "child": _rel_targets(prim, "physics:body1"),
        "axis": _attr_value(prim, "physics:axis"),
        "lower": lower,
        "upper": upper,
        "locked_by_limits": locked,
        "exclude_from_articulation": _attr_value(prim, "physics:excludeFromArticulation"),
        "break_force": _attr_value(prim, "physics:breakForce"),
        "break_torque": _attr_value(prim, "physics:breakTorque"),
        "drive_values": _drive_values(prim, UsdPhysics),
        "applied_schemas": applied_schemas,
        "mimic_or_coupling_schema": physx_mimic,
    }


def _print_joint(info: dict[str, Any]) -> None:
    print(f"[JOINT] {info['name']}", flush=True)
    print(f"  path: {info['path']}", flush=True)
    print(f"  type: {info['type']}", flush=True)
    print(f"  parent bodies: {info['parent']}", flush=True)
    print(f"  child bodies: {info['child']}", flush=True)
    print(f"  axis: {info['axis']}", flush=True)
    print(f"  lower/upper: {info['lower']} / {info['upper']}", flush=True)
    print(f"  locked_by_limits: {info['locked_by_limits']}", flush=True)
    print(f"  exclude_from_articulation: {info['exclude_from_articulation']}", flush=True)
    print(f"  breakForce/breakTorque: {info['break_force']} / {info['break_torque']}", flush=True)
    if info["drive_values"]:
        for drive_name, values in info["drive_values"]:
            print(f"  drive[{drive_name}]: {values}", flush=True)
    else:
        print("  drive: none authored", flush=True)
    print(f"  mimic_or_coupling_schema: {info['mimic_or_coupling_schema']}", flush=True)
    print(f"  applied_schemas: {info['applied_schemas']}", flush=True)


def _usd_path_from_cfg(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg) -> str:
    robot_cfg = getattr(env_cfg, "robot_cfg", None)
    if robot_cfg is None:
        robot_cfg = getattr(getattr(env_cfg, "scene", None), "robot", None)
    spawn_cfg = getattr(robot_cfg, "spawn", None)
    usd_path = getattr(spawn_cfg, "usd_path", None)
    if usd_path is None:
        raise RuntimeError("Could not resolve robot_cfg.spawn.usd_path from env config.")
    return str(usd_path)


def _print_cfg_info(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg) -> None:
    robot_cfg = getattr(env_cfg, "robot_cfg", None)
    if robot_cfg is None:
        robot_cfg = getattr(getattr(env_cfg, "scene", None), "robot", None)
    print(f"[INFO] Env cfg class: {type(env_cfg).__module__}.{type(env_cfg).__name__}", flush=True)
    print(f"[INFO] Kevin cfg source file: {inspect.getsourcefile(MulagAdaptiveRLJointReachEnvCfg)}", flush=True)
    print(f"[INFO] Robot asset cfg source file: {inspect.getsourcefile(mulag_asset_cfg)}", flush=True)
    print(f"[INFO] Robot cfg class: {type(robot_cfg).__module__}.{type(robot_cfg).__name__}", flush=True)
    print(f"[INFO] Robot cfg prim_path: {getattr(robot_cfg, 'prim_path', 'unavailable')}", flush=True)
    print(f"[INFO] Robot cfg spawn: {getattr(robot_cfg, 'spawn', 'unavailable')}", flush=True)
    print(f"[INFO] Robot cfg actuators: {getattr(robot_cfg, 'actuators', 'unavailable')}", flush=True)
    print(f"[INFO] Kevin state_joint_names: {getattr(env_cfg, 'state_joint_names', [])}", flush=True)
    print(f"[INFO] Kevin command_joint_names: {getattr(env_cfg, 'command_joint_names', [])}", flush=True)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    _ = agent_cfg
    _print_cfg_info(env_cfg)
    usd_path = _usd_path_from_cfg(env_cfg)
    print(f"[INFO] Active USD path: {usd_path}", flush=True)
    print(f"[INFO] USD path exists: {Path(usd_path).exists()}", flush=True)

    try:
        from pxr import PhysxSchema, Usd, UsdPhysics
    except Exception as exc:
        print(f"[ERROR] Could not import pxr USD modules: {exc}", flush=True)
        print("[INFO] Run this script through ./rrlab.sh so Isaac/PXR are on PYTHONPATH.", flush=True)
        simulation_app.close()
        return

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise RuntimeError(f"Could not open USD stage: {usd_path}")

    print(f"[INFO] Default prim: {stage.GetDefaultPrim().GetPath() if stage.GetDefaultPrim() else 'none'}", flush=True)
    print("[INFO] Physics/articulation API prims:", flush=True)
    for prim in stage.Traverse():
        applied = [str(item) for item in prim.GetAppliedSchemas()]
        is_articulation = "PhysicsArticulationRootAPI" in applied or "PhysxArticulationAPI" in applied
        is_rigid = prim.HasAPI(UsdPhysics.RigidBodyAPI) if hasattr(UsdPhysics, "RigidBodyAPI") else False
        if is_articulation or is_rigid:
            print(f"  {prim.GetPath()} applied={applied}", flush=True)

    joints = []
    for prim in stage.Traverse():
        info = _joint_info(prim, UsdPhysics, PhysxSchema)
        if info is not None:
            joints.append(info)

    print(f"[INFO] Found {len(joints)} joint/joint-like prims.", flush=True)
    similar = []
    for info in joints:
        _print_joint(info)
        if "drehzapfen" in str(info["name"]).lower() or "drehzapfen" in str(info["path"]).lower():
            similar.append(info)

    print("[INFO] Drehzapfen-like joint prims:", flush=True)
    if not similar:
        print("  none", flush=True)
    for info in similar:
        print(
            f"  name={info['name']} path={info['path']} type={info['type']} "
            f"axis={info['axis']} limits={info['lower']}/{info['upper']} "
            f"locked={info['locked_by_limits']} drives={info['drive_values']}",
            flush=True,
        )
    simulation_app.close()


if __name__ == "__main__":
    main()
