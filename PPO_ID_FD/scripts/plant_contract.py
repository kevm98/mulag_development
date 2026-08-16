#!/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
"""Check that ROM and ID-FD share the same physical task plant."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--report", default="")
from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
simulation_app = AppLauncher(args).app

import PPO_ID_FD  # noqa: E402,F401
import PPO_ROM  # noqa: E402,F401
from PPO_ID_FD.env_cfg import Ppo41IDFDEnvCfg  # noqa: E402
from PPO_ROM.env_cfg import Ppo31SoleEnvCfg  # noqa: E402


def actuator_signature(cfg) -> dict:
    result = {}
    for name, actuator in sorted(cfg.robot_cfg.actuators.items()):
        result[name] = {
            "joint_names_expr": list(getattr(actuator, "joint_names_expr", ())),
            "stiffness": getattr(actuator, "stiffness", None),
            "damping": getattr(actuator, "damping", None),
            "effort_limit_sim": getattr(actuator, "effort_limit_sim", None),
            "velocity_limit_sim": getattr(actuator, "velocity_limit_sim", None),
        }
    return result


def same(name: str, left, right, failures: list[str]) -> None:
    if left != right:
        failures.append(name)


rom = Ppo31SoleEnvCfg()
idfd = Ppo41IDFDEnvCfg()
failures: list[str] = []

rom_usd = str(rom.robot_cfg.spawn.usd_path)
idfd_usd = str(idfd.robot_cfg.spawn.usd_path)
same("robot USD", rom_usd, idfd_usd, failures)
same("arm joint map", tuple(rom.arm_joint_names), tuple(idfd.arm_joint_names), failures)
same("cylinder joint map", tuple(rom.cylinder_joint_names), tuple(idfd.cylinder_joint_names), failures)
same("PhysX actuator signature", actuator_signature(rom), actuator_signature(idfd), failures)
same("simulation dt", float(rom.sim.dt), float(idfd.sim.dt), failures)
same("decimation", int(rom.decimation), int(idfd.decimation), failures)
same("episode length", float(rom.episode_length_s), float(idfd.episode_length_s), failures)
same("mowing strip length", float(rom.mow_strip_length_m), float(idfd.mow_strip_length_m), failures)
same("tracking length", float(rom.terrain_tracking_active_length_m), float(idfd.terrain_tracking_active_length_m), failures)
same("vehicle speed", (float(rom.vehicle_speed_min_kmh), float(rom.vehicle_speed_max_kmh)), (float(idfd.vehicle_speed_min_kmh), float(idfd.vehicle_speed_max_kmh)), failures)
same("episode speed sampling", bool(rom.vehicle_speed_per_episode), bool(idfd.vehicle_speed_per_episode), failures)
same("path completion tolerance", float(rom.vehicle_path_completion_tolerance_m), float(idfd.vehicle_path_completion_tolerance_m), failures)
same("observation width", int(rom.observation_space), int(idfd.observation_space), failures)
same("action width", int(rom.action_space), int(idfd.action_space), failures)

assets = {
    "robot_usd": rom_usd,
    "rom_checkpoint": str(getattr(rom, "command_rom_checkpoint", "")),
    "id_checkpoint": str(getattr(idfd, "id_cylinder_checkpoint", "")),
    "fd_checkpoint": str(getattr(idfd, "fd_cylinder_checkpoint", "")),
}
for name, path in assets.items():
    if not Path(path).is_file():
        failures.append(f"missing {name}")

if str(getattr(rom, "cylinder_dynamics_model", "")).lower() != "command_rom":
    failures.append("ROM actuator branch")
if str(getattr(idfd, "cylinder_dynamics_model", "")).lower() != "fd":
    failures.append("ID-FD actuator branch")
if str(getattr(idfd, "command_rom_checkpoint", "")).lower() != "disabled":
    failures.append("ID-FD command ROM disabled")

report = {
    "passed": not failures,
    "failures": failures,
    "rom": {
        "task": PPO_ROM.TERRAIN_TASK_ID,
        "actuator": "command_rom",
        "command_rom_checkpoint": assets["rom_checkpoint"],
        "external_command": "desired_delta_extension_4 + desired_delta_velocity_4",
        "external_valve_command": False,
        "predicted_valve": "internal ROM diagnostic/history only",
    },
    "idfd": {
        "task": PPO_ID_FD.TERRAIN_TASK_ID,
        "actuator": "idfd",
        "id_checkpoint": assets["id_checkpoint"],
        "fd_checkpoint": assets["fd_checkpoint"],
        "external_command": "desired cylinder motion -> ID valve_4 -> FD",
        "external_valve_command": True,
    },
    "shared": {
        "robot_usd": rom_usd,
        "arm_joint_names": list(rom.arm_joint_names),
        "cylinder_joint_names": list(rom.cylinder_joint_names),
        "sim_dt": float(rom.sim.dt),
        "decimation": int(rom.decimation),
        "control_dt": float(rom.sim.dt) * int(rom.decimation),
        "episode_length_s": float(rom.episode_length_s),
        "mow_strip_length_m": float(rom.mow_strip_length_m),
        "tracking_active_length_m": float(rom.terrain_tracking_active_length_m),
        "vehicle_speed_kmh": [float(rom.vehicle_speed_min_kmh), float(rom.vehicle_speed_max_kmh)],
        "vehicle_speed_per_episode": bool(rom.vehicle_speed_per_episode),
        "path_completion_tolerance_m": float(rom.vehicle_path_completion_tolerance_m),
        "observation_width": int(rom.observation_space),
        "action_width": int(rom.action_space),
        "physx_actuators": actuator_signature(rom),
    },
}
print(json.dumps(report, indent=2, default=str))
if args.report:
    report_path = Path(args.report).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=str) + "\n")
simulation_app.close()
if failures:
    raise SystemExit(1)
