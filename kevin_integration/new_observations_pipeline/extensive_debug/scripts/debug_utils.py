#!/usr/bin/env python3
"""Shared utilities for extensive debug tests.

Provides helpers for launching the MULAG env, inspecting articulation state,
computing TCP poses, and saving results to the extensive_debug folder.

IMPORTANT: Importing this module bootstraps the Isaac Sim SimulationApp.
All test scripts must import from this module before importing isaaclab.
"""
from __future__ import annotations

import json, csv, os, sys, time, math, traceback
from pathlib import Path
from typing import Any

# ── Bootstrap Isaac Sim before anything else ───────────────────────────
import isaacsim  # noqa: F401 — bootstraps kit kernel
from isaacsim import SimulationApp

_simulation_app = None


def _ensure_sim_app(headless: bool = True):
    global _simulation_app
    if _simulation_app is None:
        _simulation_app = SimulationApp({"headless": headless})
    return _simulation_app


def _close_sim_app():
    global _simulation_app
    if _simulation_app is not None:
        _simulation_app.close()
        _simulation_app = None



import torch
import numpy as np

# Ensure repo root is on sys.path so imports work regardless of launch dir.
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEBUG_DIR = _REPO_ROOT / "kevin_integration/new_observations_pipeline/extensive_debug"
RESULTS_DIR = DEBUG_DIR / "results"
LOGS_DIR = DEBUG_DIR / "logs"
PLOTS_DIR = DEBUG_DIR / "plots"
REPORTS_DIR = DEBUG_DIR / "reports"

for _d in (RESULTS_DIR, LOGS_DIR, PLOTS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ── Known joint/body names from codebase inspection ──────────────────────────
ARM_JOINT_NAMES = (
    "Drehzapfen_joint",
    "Ausleger_I_joint",
    "Ausleger_II_joint",
    "Messerkopf_Schwenk_joint",
    "Messerkopf_joint",
)
CYLINDER_JOINT_NAMES = ARM_JOINT_NAMES[:4]
EE_BODY_NAME = "Messerkopf"
ARM_BASE_BODY_NAME = "Drehzapfen"

CYLINDER_SIGNS = [-1, -1, -1, -1]
CYLINDER_NOMINAL_SIMSCAPE = [0.335226446390152, 0.5414418578147888, 0.4934495687484741, 0.5867190957069397]
CYLINDER_NOMINAL_ISAAC    = [-0.0029414286836981773, 0.045830193907022476, 0.024504413828253746, 0.04233485460281372]


# ── Env construction ────────────────────────────────────────────────────────
def build_mulag_env(num_envs: int = 1, device: str = "cuda:0", headless: bool = True):
    from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env_cfg import MulagUpperLevelEnvCfg
    from kevin_integration.new_observations_pipeline.ppo1.mulag_upper_level_env import MulagUpperLevelEnv

    cfg = MulagUpperLevelEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.enable_startup_summary = False
    cfg.enable_verbose_startup = False
    cfg.enable_scripted_vehicle_motion = False
    cfg.enable_console_debug = False
    cfg.enable_pipeline_debug = False
    cfg.use_cylinder_id_pipeline = False

    from isaaclab.actuators import ImplicitActuatorCfg
    cfg.robot_cfg.actuators = {
        "drehzapfen_cylinder": ImplicitActuatorCfg(
            joint_names_expr=["Drehzapfen_joint"], stiffness=75000000.0, damping=2240000.0,
            effort_limit_sim=50000.0, velocity_limit_sim=1.0,
        ),
        "ausleger_i_cylinder": ImplicitActuatorCfg(
            joint_names_expr=["Ausleger_I_joint"], stiffness=30000000.0, damping=1340000.0,
            effort_limit_sim=100000.0, velocity_limit_sim=1.0,
        ),
        "ausleger_ii_cylinder": ImplicitActuatorCfg(
            joint_names_expr=["Ausleger_II_joint"], stiffness=5000000.0, damping=1120000.0,
            effort_limit_sim=50000.0, velocity_limit_sim=1.0,
        ),
        "messerkopf_schwenk_cylinder": ImplicitActuatorCfg(
            joint_names_expr=["Messerkopf_Schwenk_joint"], stiffness=5000000.0, damping=1120000.0,
            effort_limit_sim=20000.0, velocity_limit_sim=1.0,
        ),
        "messerkopf_passive": ImplicitActuatorCfg(
            joint_names_expr=["Messerkopf_joint"], stiffness=5000000.0, damping=1120000.0,
            effort_limit_sim=20000.0, velocity_limit_sim=1.0,
        ),
        "mulag_wheels": ImplicitActuatorCfg(
            joint_names_expr=["Wheel_Front_Left_joint", "Wheel_Front_Right_joint",
                              "Wheel_Rear_Left_joint", "Wheel_Rear_Right_joint"],
            stiffness=600000000.0, damping=2000000.0, effort_limit_sim=100000.0,
        ),
        "mulag_steering": ImplicitActuatorCfg(
            joint_names_expr=["Wheel_Front_Left_Steering_joint", "Wheel_Front_Right_Steering_joint"],
            stiffness=10.0, damping=1000000000.0, effort_limit_sim=100000.0,
        ),
    }

    env = MulagUpperLevelEnv(cfg=cfg)
    env.reset()
    for _ in range(30):
        env.step(torch.zeros((num_envs, 5), device=device))
    return env, cfg


# ── Articulation inspection helpers ─────────────────────────────────────────
def inspect_articulation(env) -> dict[str, Any]:
    robot = env.robot
    info: dict[str, Any] = {}
    info["num_bodies"] = robot.num_bodies
    info["num_joints"] = robot.num_joints
    info["num_joints_total"] = robot.num_joints
    info["body_names"] = list(robot.body_names)
    info["joint_names"] = list(robot.joint_names)
    try:
        info["dof_names"] = list(robot.dof_names)
    except (AttributeError, Exception):
        info["dof_names"] = []
    try:
        info["joint_types"] = [str(t) for t in robot.joint_types]
    except (AttributeError, Exception):
        info["joint_types"] = []
    try:
        actuator_info = {}
        for gn, gc in robot.actuator_groups.items():
            actuator_info[gn] = {"joint_names_expr": gc.joint_names_expr,
                                 "stiffness": gc.stiffness, "damping": gc.damping,
                                 "effort_limit": gc.effort_limit, "velocity_limit": gc.velocity_limit}
        info["actuator_groups"] = actuator_info
    except (AttributeError, Exception):
        info["actuator_groups"] = {}
    try:
        limits = robot.data.soft_joint_pos_limits[0].detach().cpu().tolist()
        effort_limits = robot.data.joint_effort_limit_sim[0].detach().cpu().tolist()
        vel_limits = robot.data.velocity_limit[0].detach().cpu().tolist()
        info["joint_limits_detail"] = []
        for i, name in enumerate(robot.joint_names):
            info["joint_limits_detail"].append({
                "index": i, "name": name,
                "pos_lower": limits[i][0] if i < len(limits) else None,
                "pos_upper": limits[i][1] if i < len(limits) else None,
                "effort_limit": effort_limits[i] if i < len(effort_limits) else None,
                "velocity_limit": vel_limits[i] if i < len(vel_limits) else None,
            })
    except (AttributeError, Exception):
        info["joint_limits_detail"] = []
    try:
        default_pos = robot.data.default_joint_pos[0].detach().cpu().tolist()
        info["default_joint_positions"] = dict(zip(robot.joint_names, default_pos))
    except (AttributeError, Exception):
        info["default_joint_positions"] = {}
    try:
        current_pos = robot.data.joint_pos[0].detach().cpu().tolist()
        info["current_joint_positions"] = dict(zip(robot.joint_names, current_pos))
    except (AttributeError, Exception):
        info["current_joint_positions"] = {}
    try:
        body_states = robot.data.body_link_state_w[0].detach().cpu()
        body_poses = {}
        for i, name in enumerate(robot.body_names):
            body_poses[name] = {"pos": body_states[i, :3].tolist(), "quat": body_states[i, 3:7].tolist()}
        info["body_poses"] = body_poses
    except (AttributeError, Exception):
        info["body_poses"] = {}
    for i, name in enumerate(robot.body_names):
        if name == EE_BODY_NAME:
            info["ee_body_index"] = i
            info["ee_body_name"] = name
            break
    arm_indices = {}
    for name in ARM_JOINT_NAMES:
        for i, jname in enumerate(robot.joint_names):
            if jname == name:
                arm_indices[name] = i
                break
    info["arm_joint_indices"] = arm_indices
    cylinder_indices = {}
    for name in CYLINDER_JOINT_NAMES:
        for i, jname in enumerate(robot.joint_names):
            if jname == name:
                cylinder_indices[name] = i
                break
    info["cylinder_joint_indices"] = cylinder_indices
    return info


def get_tcp_pose(env, env_id: int = 0) -> dict[str, Any]:
    robot = env.robot
    ee_idx = None
    for i, name in enumerate(robot.body_names):
        if name == EE_BODY_NAME:
            ee_idx = i
            break
    if ee_idx is None:
        raise RuntimeError(f"EE body '{EE_BODY_NAME}' not found.")
    from isaaclab.utils.math import quat_apply_inverse
    ee_world_pos = robot.data.body_link_state_w[env_id, ee_idx, :3].detach().cpu().tolist()
    ee_world_quat = robot.data.body_link_state_w[env_id, ee_idx, 3:7].detach().cpu().tolist()
    root_pos = robot.data.root_pos_w[env_id].detach().cpu().tolist()
    root_quat_t = robot.data.root_quat_w[env_id].detach().clone()
    root_quat_t = root_quat_t / root_quat_t.norm()
    ee_base_pos = quat_apply_inverse(
        root_quat_t.unsqueeze(0),
        robot.data.body_link_state_w[env_id, ee_idx, :3].unsqueeze(0) - robot.data.root_pos_w[env_id].unsqueeze(0),
    ).squeeze(0).detach().cpu().tolist()
    return {"world_pos": ee_world_pos, "world_quat": ee_world_quat,
            "base_pos": ee_base_pos, "root_world_pos": root_pos}


def get_cylinder_state(env, env_id: int = 0) -> dict[str, Any]:
    robot = env.robot
    state = {}
    for name in CYLINDER_JOINT_NAMES:
        for i, jname in enumerate(robot.joint_names):
            if jname == name:
                state[name] = {"pos": float(robot.data.joint_pos[env_id, i].detach().cpu().item()),
                               "vel": float(robot.data.joint_vel[env_id, i].detach().cpu().item())}
                break
    return state


def get_all_joint_state(env, env_id: int = 0) -> dict[str, Any]:
    robot = env.robot
    state = {}
    for name in ARM_JOINT_NAMES:
        for i, jname in enumerate(robot.joint_names):
            if jname == name:
                state[name] = {"pos": float(robot.data.joint_pos[env_id, i].detach().cpu().item()),
                               "vel": float(robot.data.joint_vel[env_id, i].detach().cpu().item())}
                break
    return state


# ── Save helpers ─────────────────────────────────────────────────────────────
def save_json(data: Any, filename: str):
    path = RESULTS_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved: {path}")


def save_csv(rows: list[dict], filename: str):
    path = RESULTS_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {path}")


def save_log(text: str, filename: str):
    path = LOGS_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  Saved: {path}")


def save_report(markdown: str, filename: str):
    path = REPORTS_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(markdown)
    print(f"  Saved: {path}")


def isaac_to_simscape(q_isaac: list[float]) -> list[float]:
    return [CYLINDER_NOMINAL_SIMSCAPE[i] + CYLINDER_SIGNS[i] * (q_isaac[i] - CYLINDER_NOMINAL_ISAAC[i])
            for i in range(4)]


def simscape_to_isaac(x_simscape: list[float]) -> list[float]:
    return [CYLINDER_NOMINAL_ISAAC[i] + CYLINDER_SIGNS[i] * (x_simscape[i] - CYLINDER_NOMINAL_SIMSCAPE[i])
            for i in range(4)]
