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

ACTUATOR_JOINT_NAMES = [
    "Drehzapfen_joint",
    "Ausleger_I_joint",
    "Ausleger_II_joint",
    "Messerkopf_Schwenk_joint",
]
MECHANICAL_FAMILIES = {
    "Drehzapfen_joint": ["Drehzapfen_constraint_joint_1", "Drehzapfen_constraint_joint_3"],
    "Ausleger_I_joint": ["Ausleger_I_constraint_joint_1", "Ausleger_I_constraint_joint_3"],
    "Ausleger_II_joint": ["Ausleger_II_constraint_joint_1", "Ausleger_II_constraint_joint_3"],
    "Messerkopf_Schwenk_joint": [
        "Messerkopf_Schwenk_constraint_joint_1",
        "Messerkopf_Schwenk_constraint_joint_3",
        "Messerkopf_Schwenk_constraint_joint_5",
        "Messerkopf_Schwenk_constraint_joint_6",
    ],
}
JOINT_TYPE_NAMES = (
    "RevoluteJoint",
    "PrismaticJoint",
    "FixedJoint",
    "SphericalJoint",
    "DistanceJoint",
    "Joint",
)

parser = argparse.ArgumentParser(description="Inspect USD/runtime drive values for Mulag cylinder actuator joints.")
parser.add_argument("--task", type=str, default="Kevin-Mulag-AdaptiveRL-JointReach-v0", help="Gym task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of Isaac Lab environments.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg
from isaaclab_tasks.utils.hydra import hydra_task_config

import kevin_integration.tasks  # noqa: F401
from kevin_integration.tasks.mulag_adaptive_rl_env_cfg import MulagAdaptiveRLJointReachEnvCfg
from kevin_integration.utils.sim_memory import apply_kevin_sim_memory_optimizations
import rrlab_assets.mulag as mulag_asset_cfg


def _usd_path_from_cfg(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg) -> str:
    robot_cfg = getattr(env_cfg, "robot_cfg", None)
    if robot_cfg is None:
        robot_cfg = getattr(getattr(env_cfg, "scene", None), "robot", None)
    spawn_cfg = getattr(robot_cfg, "spawn", None)
    usd_path = getattr(spawn_cfg, "usd_path", None)
    if usd_path is None:
        raise RuntimeError("Could not resolve robot_cfg.spawn.usd_path from env config.")
    return str(usd_path)


def _schema_type(prim, UsdPhysics) -> str | None:
    for type_name in JOINT_TYPE_NAMES:
        schema = getattr(UsdPhysics, type_name, None)
        if schema is not None and prim.IsA(schema):
            return type_name
    return None


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


def _drive_values(prim, UsdPhysics) -> list[tuple[str, dict[str, Any]]]:
    values: list[tuple[str, dict[str, Any]]] = []
    for drive_name in ("linear", "angular", "transX", "transY", "transZ", "rotX", "rotY", "rotZ"):
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


def _collect_usd_joints(usd_path: str) -> dict[str, dict[str, Any]]:
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise RuntimeError(f"Could not open USD stage: {usd_path}")
    by_name: dict[str, dict[str, Any]] = {}
    for prim in stage.Traverse():
        type_name = _schema_type(prim, UsdPhysics)
        if type_name is None and "joint" not in prim.GetName().lower():
            continue
        by_name[prim.GetName()] = {
            "path": str(prim.GetPath()),
            "type": type_name or "unknown_joint_like_prim",
            "parent": _rel_targets(prim, "physics:body0"),
            "child": _rel_targets(prim, "physics:body1"),
            "axis": _attr_value(prim, "physics:axis"),
            "lower": _attr_value(prim, "physics:lowerLimit"),
            "upper": _attr_value(prim, "physics:upperLimit"),
            "drives": _drive_values(prim, UsdPhysics),
        }
    return by_name


def _get_joint_names(robot) -> list[str]:
    names = getattr(robot, "joint_names", None)
    if names is None:
        names = getattr(robot.data, "joint_names", None)
    return list(names) if names is not None else []


def _resolve_joint(robot, joint_name: str) -> tuple[int | None, str | None]:
    try:
        joint_ids, resolved_names = robot.find_joints(joint_name)
    except Exception as exc:
        print(f"[WARN] Could not query joint {joint_name!r}: {exc}", flush=True)
        return None, None
    if len(joint_ids) == 0:
        return None, None
    joint_id = int(joint_ids[0])
    if resolved_names:
        return joint_id, resolved_names[0]
    joint_names = _get_joint_names(robot)
    return joint_id, joint_names[joint_id] if joint_id < len(joint_names) else joint_name


def _tensor_value(data: Any, field_names: tuple[str, ...], joint_id: int | None) -> tuple[str, float | str]:
    if joint_id is None:
        return "unavailable", "unavailable"
    for field_name in field_names:
        value = getattr(data, field_name, None)
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu()
            if tensor.ndim >= 2:
                tensor = tensor[0]
            flat = tensor.flatten()
            if 0 <= joint_id < flat.numel():
                return field_name, float(flat[joint_id])
    return "unavailable", "unavailable"


def _drive_is_high(usd_info: dict[str, Any] | None, stiffness: float | str, damping: float | str) -> bool:
    candidates = []
    if isinstance(stiffness, float):
        candidates.append(stiffness)
    if isinstance(damping, float):
        candidates.append(damping)
    if usd_info:
        for _, values in usd_info.get("drives", []):
            for key in ("stiffness", "damping"):
                value = values.get(key)
                if isinstance(value, (int, float)):
                    candidates.append(float(value))
    return any(value >= 1.0e5 for value in candidates)


def _print_usd_runtime_joint(
    *,
    joint_name: str,
    usd_info: dict[str, Any] | None,
    robot,
    command_joint_names: list[str],
) -> None:
    joint_id, runtime_name = _resolve_joint(robot, joint_name)
    effort_field, effort_limit = _tensor_value(
        robot.data, ("joint_effort_limits", "soft_joint_effort_limits", "effort_limits"), joint_id
    )
    velocity_field, velocity_limit = _tensor_value(
        robot.data, ("joint_velocity_limits", "soft_joint_velocity_limits", "joint_vel_limits", "soft_joint_vel_limits"), joint_id
    )
    stiffness_field, stiffness = _tensor_value(robot.data, ("joint_stiffness", "default_joint_stiffness", "stiffness"), joint_id)
    damping_field, damping = _tensor_value(robot.data, ("joint_damping", "default_joint_damping", "damping"), joint_id)
    print(f"[ACTUATOR] {joint_name}", flush=True)
    print(f"  included_in_kevin_command_joint_names: {joint_name in command_joint_names}", flush=True)
    print(f"  runtime_joint_id: {joint_id}", flush=True)
    print(f"  runtime_joint_name: {runtime_name}", flush=True)
    if usd_info is None:
        print("  USD: not found", flush=True)
    else:
        print(f"  usd_path: {usd_info['path']}", flush=True)
        print(f"  usd_joint_type: {usd_info['type']}", flush=True)
        print(f"  is_prismatic: {usd_info['type'] == 'PrismaticJoint'}", flush=True)
        print(f"  parent_body: {usd_info['parent']}", flush=True)
        print(f"  child_body: {usd_info['child']}", flush=True)
        print(f"  axis: {usd_info['axis']}", flush=True)
        print(f"  lower_upper_limits: {usd_info['lower']} / {usd_info['upper']}", flush=True)
        if usd_info["drives"]:
            for drive_name, values in usd_info["drives"]:
                print(f"  drive_type: {drive_name}", flush=True)
                print(f"    stiffness: {values.get('stiffness')}", flush=True)
                print(f"    damping: {values.get('damping')}", flush=True)
                print(f"    maxForce: {values.get('maxForce')}", flush=True)
                print(f"    targetPosition: {values.get('targetPosition')}", flush=True)
                print(f"    targetVelocity: {values.get('targetVelocity')}", flush=True)
        else:
            print("  drive: none authored", flush=True)
    print(f"  runtime_effort_limit[{effort_field}]: {effort_limit}", flush=True)
    print(f"  runtime_velocity_limit[{velocity_field}]: {velocity_limit}", flush=True)
    print(f"  runtime_stiffness[{stiffness_field}]: {stiffness}", flush=True)
    print(f"  runtime_damping[{damping_field}]: {damping}", flush=True)
    print(f"  likely_fights_effort_control: {_drive_is_high(usd_info, stiffness, damping)}", flush=True)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    print("[INFO] Inspecting cylinder actuator drives", flush=True)
    print(f"[INFO] Kevin cfg source file: {inspect.getsourcefile(MulagAdaptiveRLJointReachEnvCfg)}", flush=True)
    print(f"[INFO] Robot asset cfg source file: {inspect.getsourcefile(mulag_asset_cfg)}", flush=True)
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device
    apply_kevin_sim_memory_optimizations(env_cfg.sim, verbose=True)

    usd_path = _usd_path_from_cfg(env_cfg)
    print(f"[INFO] Active USD path: {usd_path}", flush=True)
    try:
        usd_joints = _collect_usd_joints(usd_path)
    except Exception as exc:
        print(f"[ERROR] Could not inspect USD with PXR: {exc}", flush=True)
        usd_joints = {}

    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)
        unwrapped = env.unwrapped
        robot = unwrapped.robot
        command_joint_names = list(getattr(unwrapped.cfg, "command_joint_names", []))
        print(f"[INFO] Kevin command_joint_names: {command_joint_names}", flush=True)
        print("[INFO] Full runtime Isaac joint order:", flush=True)
        for index, name in enumerate(_get_joint_names(robot)):
            print(f"  [{index}] {name}", flush=True)

        for joint_name in ACTUATOR_JOINT_NAMES:
            _print_usd_runtime_joint(
                joint_name=joint_name,
                usd_info=usd_joints.get(joint_name),
                robot=robot,
                command_joint_names=command_joint_names,
            )
            print("  candidate_mechanical_output_joints:", flush=True)
            for mech_name in MECHANICAL_FAMILIES[joint_name]:
                mech_id, mech_runtime_name = _resolve_joint(robot, mech_name)
                usd_info = usd_joints.get(mech_name)
                if mech_id is None and usd_info is None:
                    print(f"    [WARN] {mech_name}: missing in runtime/USD", flush=True)
                    continue
                print(
                    f"    {mech_name}: runtime_id={mech_id} runtime_name={mech_runtime_name} "
                    f"usd_type={usd_info.get('type') if usd_info else 'missing'}",
                    flush=True,
                )
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
