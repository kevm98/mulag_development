from __future__ import annotations

import argparse
import csv
import math
import sys
from datetime import datetime
from pathlib import Path

import torch
from isaaclab.app import AppLauncher


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kevin_integration.rl.torque_adapter import (  # noqa: E402
    DEFAULT_TORQUE_ADAPTER_PRESET,
    DEFAULT_TORQUE_ADAPTER_BIAS,
    TORQUE_ADAPTER_MODES,
    TORQUE_ADAPTER_PRESETS,
    adapt_fd_to_isaac_torque,
    apply_fd_residual_authority,
    effective_torque_adapter_scale,
    lowpass_torque,
    rate_limit_torque,
    resolve_torque_adapter_scale,
)


STATE_JOINT_NAMES = [
    "Drehzapfen_joint",
    "Ausleger_I_joint",
    "Ausleger_II_joint",
    "Messerkopf_Schwenk_joint",
    "Messerkopf_joint",
]
COMMAND_JOINT_NAMES = [
    "Drehzapfen_joint",
    "Ausleger_I_joint",
    "Ausleger_II_joint",
    "Messerkopf_Schwenk_joint",
]
EFFORT_SWEEP_VALUES = [
    0.0,
    0.1,
    -0.1,
    0.5,
    -0.5,
    1.0,
    -1.0,
    2.0,
    -2.0,
    5.0,
    -5.0,
    10.0,
    -10.0,
    20.0,
    -20.0,
    50.0,
    -50.0,
]


parser = argparse.ArgumentParser(description="Standalone Mulag torque and learned-FD debugging tool.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument(
    "--mode",
    choices=["sweep_effort", "hold_position", "fd_sanity", "compare_fd_vs_applied"],
    default="sweep_effort",
    help="Torque/debug test mode.",
)
parser.add_argument("--duration", type=float, default=10.0, help="Maximum test duration in simulated seconds.")
parser.add_argument("--hold_steps", type=int, default=100, help="Steps to hold each sweep effort value.")
parser.add_argument("--max_abs_torque", type=float, default=5.0, help="Clamp applied effort to this absolute value.")
parser.add_argument("--print_interval", type=int, default=50, help="Print diagnostics every N sim steps.")
parser.add_argument(
    "--torque_adapter_mode",
    choices=TORQUE_ADAPTER_MODES,
    default="scale_bias",
    help="Map FD Simscape output to Isaac effort with this adapter.",
)
parser.add_argument(
    "--torque_adapter_preset",
    choices=TORQUE_ADAPTER_PRESETS,
    default=DEFAULT_TORQUE_ADAPTER_PRESET,
    help="Named scale preset for the scale_bias/residual_pd adapter.",
)
parser.add_argument(
    "--torque_adapter_scale",
    type=str,
    default=None,
    help='Optional comma-separated per-joint scale, e.g. "0.0001,0.001,0.0005,0.0001".',
)
parser.add_argument(
    "--torque_adapter_bias",
    type=float,
    nargs=4,
    default=DEFAULT_TORQUE_ADAPTER_BIAS,
    metavar=("B0", "B1", "B2", "B3"),
    help="Per-joint bias for scale_bias/residual_pd adapter modes.",
)
parser.add_argument("--fd_torque_scale", type=float, default=1.0e6, help="FD scale for tanh_squash mode.")
parser.add_argument(
    "--use_fd_residual_alpha",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Enable deterministic residual authority scaling after the torque adapter.",
)
parser.add_argument("--fd_residual_alpha", type=float, default=0.002, help="FD residual authority gain.")
parser.add_argument("--torque_rate_limit", type=float, default=1.0, help="Per-step torque rate limit.")
parser.add_argument("--torque_lowpass_alpha", type=float, default=0.2, help="Torque low-pass alpha.")
parser.add_argument(
    "--models_dir",
    type=str,
    default=str(ROOT / "kevin_integration" / "models"),
    help="Directory with fd.pth and fd_scaler.pth.",
)
parser.add_argument("--no_csv", action="store_true", help="Disable CSV logging under logs/torque_testing.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.torque_adapter_scale = resolve_torque_adapter_scale(
    args_cli.torque_adapter_preset,
    args_cli.torque_adapter_scale,
)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

from kevin_integration.controllers.learned_models import ForwardDynamicsModel
from kevin_integration.rl.action_adapter import clamp_and_sanitize
from kevin_integration.rl.observation_builder import build_fd_input
from kevin_integration.utils.sim_memory import apply_kevin_sim_memory_optimizations
from rrlab_assets import MULAG_CFG


@configclass
class PlaneScene(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )
    robot = MULAG_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )


class SequenceBuffer:
    def __init__(self, *, num_envs: int, window_size: int, feature_dim: int, device: torch.device):
        self.data = torch.zeros((num_envs, window_size, feature_dim), dtype=torch.float32, device=device)

    def append(self, value: torch.Tensor) -> torch.Tensor:
        self.data = torch.roll(self.data, shifts=-1, dims=1)
        self.data[:, -1, :] = value.detach()
        return self.data


class CsvLogger:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.file = None
        self.writer = None
        self.path = None
        if not enabled:
            return
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = ROOT / "logs" / "torque_testing" / timestamp
        log_dir.mkdir(parents=True, exist_ok=True)
        self.path = log_dir / "torque_test.csv"
        self.file = self.path.open("w", newline="")
        self.writer = csv.DictWriter(
            self.file,
            fieldnames=[
                "mode",
                "step",
                "sim_time",
                "target_joint",
                "command_effort",
                "q",
                "qdot",
                "applied_fields",
                "torque_raw_mean",
                "torque_raw_max",
                "torque_clamped_mean",
                "torque_clamped_max",
                "torque_clamp_fraction",
                "torque_adapter_mode",
                "torque_adapter_preset",
                "torque_adapter_scale",
                "use_fd_residual_alpha",
                "fd_residual_alpha",
                "effective_torque_adapter_scale",
                "fd_simscape_output",
                "tau_fd_adapted",
                "tau_applied_raw",
                "tau_isaac_filtered",
                "tau_isaac_clamped",
                "applied_torque",
                "computed_torque",
                "joint_effort",
                "fd_case",
                "fd_input_raw_mean",
                "fd_input_raw_min",
                "fd_input_raw_max",
                "fd_input_norm_mean",
                "fd_input_norm_min",
                "fd_input_norm_max",
                "fd_output_model_mean",
                "fd_output_model_min",
                "fd_output_model_max",
                "fd_output_denorm_mean",
                "fd_output_denorm_min",
                "fd_output_denorm_max",
                "fd_output_denorm_values",
            ],
        )
        self.writer.writeheader()
        print(f"[INFO] Writing CSV diagnostics to: {self.path}")

    def write(self, row: dict) -> None:
        if self.writer is not None:
            self.writer.writerow(row)

    def close(self) -> None:
        if self.file is not None:
            self.file.close()


def find_joint_ids(robot, joint_names: list[str]) -> list[int]:
    ids: list[int] = []
    for joint_name in joint_names:
        joint_ids, _ = robot.find_joints(joint_name)
        if len(joint_ids) == 0:
            raise RuntimeError(f"Could not find joint: {joint_name}")
        ids.append(joint_ids[0])
    return ids


def print_joint_table(robot) -> None:
    print("[INFO] Robot joint names and IDs:")
    for joint_id, joint_name in enumerate(robot.joint_names):
        print(f"  {joint_id:02d}: {joint_name}")


def tensor_list(value: torch.Tensor, precision: int = 4) -> str:
    data = value.detach().flatten().cpu().tolist()
    return "[" + ", ".join(f"{item:.{precision}f}" for item in data) + "]"


def stats_text(name: str, value: torch.Tensor) -> str:
    value = value.detach()
    return (
        f"{name}: mean={value.mean().item():.4e} "
        f"min={value.min().item():.4e} max={value.max().item():.4e}"
    )


def stats_dict(prefix: str, value: torch.Tensor) -> dict[str, str]:
    value = value.detach()
    return {
        f"{prefix}_mean": f"{value.mean().item():.6e}",
        f"{prefix}_min": f"{value.min().item():.6e}",
        f"{prefix}_max": f"{value.max().item():.6e}",
    }


def safe_tensor(value: torch.Tensor, max_abs: float | None = None) -> torch.Tensor:
    return clamp_and_sanitize(value, max_abs)


def maybe_joint_field(robot, field_name: str, joint_ids: list[int]) -> torch.Tensor | None:
    if not hasattr(robot.data, field_name):
        return None
    try:
        value = getattr(robot.data, field_name)
    except Exception:
        return None
    if not isinstance(value, torch.Tensor):
        return None
    if value.ndim < 2 or value.shape[1] <= max(joint_ids):
        return None
    return value[:, joint_ids].detach()


def available_effort_fields(robot) -> list[str]:
    fields: list[str] = []
    for name in dir(robot.data):
        if name.startswith("_") or ("torque" not in name.lower() and "effort" not in name.lower()):
            continue
        try:
            value = getattr(robot.data, name)
        except Exception:
            continue
        if isinstance(value, torch.Tensor):
            fields.append(f"{name}{tuple(value.shape)}")
        else:
            fields.append(f"{name}:{type(value).__name__}")
    return fields


def read_effort_fields(robot, joint_ids: list[int]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name in ("applied_torque", "computed_torque", "joint_effort", "joint_torque", "joint_effort_target"):
        value = maybe_joint_field(robot, name, joint_ids)
        if value is not None:
            out[name] = value
    return out


def print_effort_fields_once(robot, joint_ids: list[int]) -> None:
    fields = read_effort_fields(robot, joint_ids)
    if fields:
        print("[INFO] Available selected effort/torque tensors:")
        for name, value in fields.items():
            print(f"  {name}: {tuple(value.shape)}")
        return
    print("[WARN] No recognized applied effort/torque tensor found for selected joints.")
    print("[INFO] robot.data fields containing effort/torque:")
    for item in available_effort_fields(robot):
        print(f"  {item}")


def zero_efforts(robot, command_joint_ids: list[int]) -> None:
    zeros = torch.zeros(
        (robot.data.joint_pos.shape[0], len(command_joint_ids)),
        dtype=torch.float32,
        device=robot.data.joint_pos.device,
    )
    robot.set_joint_effort_target(zeros, joint_ids=command_joint_ids)


def write_sim_step(sim: sim_utils.SimulationContext, scene: InteractiveScene) -> None:
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim.get_physics_dt())


def log_state(
    *,
    logger: CsvLogger,
    mode: str,
    step: int,
    sim_time: float,
    robot,
    state_joint_ids: list[int],
    command_joint_ids: list[int],
    target_joint: str = "",
    command_effort: float = 0.0,
    fd_simscape_output: torch.Tensor | None = None,
    tau_fd_adapted: torch.Tensor | None = None,
    tau_applied_raw: torch.Tensor | None = None,
    tau_isaac_filtered: torch.Tensor | None = None,
    tau_isaac_clamped: torch.Tensor | None = None,
    torque_clamp_fraction: torch.Tensor | None = None,
    torque_adapter_mode: str = "",
) -> None:
    q = robot.data.joint_pos[:, state_joint_ids].detach()
    qdot = robot.data.joint_vel[:, state_joint_ids].detach()
    effort_fields = read_effort_fields(robot, command_joint_ids)
    applied_summary = "; ".join(f"{name}={tensor_list(value[0])}" for name, value in effort_fields.items())
    fd_mean = fd_simscape_output.abs().mean().item() if fd_simscape_output is not None else 0.0
    fd_max = fd_simscape_output.abs().max().item() if fd_simscape_output is not None else 0.0
    tau_adapted_mean = tau_fd_adapted.abs().mean().item() if tau_fd_adapted is not None else 0.0
    tau_adapted_max = tau_fd_adapted.abs().max().item() if tau_fd_adapted is not None else 0.0
    tau_applied_mean = tau_applied_raw.abs().mean().item() if tau_applied_raw is not None else 0.0
    tau_applied_max = tau_applied_raw.abs().max().item() if tau_applied_raw is not None else 0.0
    tau_filtered_mean = tau_isaac_filtered.abs().mean().item() if tau_isaac_filtered is not None else 0.0
    tau_filtered_max = tau_isaac_filtered.abs().max().item() if tau_isaac_filtered is not None else 0.0
    tau_clamped_mean = tau_isaac_clamped.abs().mean().item() if tau_isaac_clamped is not None else 0.0
    tau_clamped_max = tau_isaac_clamped.abs().max().item() if tau_isaac_clamped is not None else 0.0
    clamp_fraction = torque_clamp_fraction.mean().item() if torque_clamp_fraction is not None else 0.0

    print(
        f"[{step:06d}] t={sim_time:.2f}s mode={mode} target={target_joint or '-'} effort={command_effort:.3f} "
        f"q={tensor_list(q[0])} qdot={tensor_list(qdot[0])}"
    )
    if applied_summary:
        print(f"           {applied_summary}")
    if fd_simscape_output is not None and tau_fd_adapted is not None and tau_isaac_clamped is not None:
        print(
            f"           torque_adapter_mode={torque_adapter_mode or args_cli.torque_adapter_mode} "
            f"torque_adapter_preset={args_cli.torque_adapter_preset} "
            f"use_fd_residual_alpha={args_cli.use_fd_residual_alpha} "
            f"fd_residual_alpha={args_cli.fd_residual_alpha:.4e} "
            f"effective_torque_adapter_scale={effective_torque_adapter_scale(args_cli.torque_adapter_scale, args_cli.fd_residual_alpha, args_cli.use_fd_residual_alpha)} "
            f"fd_simscape_mean={fd_mean:.4e} fd_simscape_max={fd_max:.4e} "
            f"tau_fd_adapted_mean={tau_adapted_mean:.4e} tau_fd_adapted_max={tau_adapted_max:.4e} "
            f"tau_applied_raw_mean={tau_applied_mean:.4e} tau_applied_raw_max={tau_applied_max:.4e} "
            f"tau_isaac_filtered_mean={tau_filtered_mean:.4e} tau_isaac_filtered_max={tau_filtered_max:.4e} "
            f"tau_isaac_clamped_mean={tau_clamped_mean:.4e} tau_isaac_clamped_max={tau_clamped_max:.4e} "
            f"torque_clamp_fraction={clamp_fraction:.3f}"
        )
        print(
            f"           fd_simscape_output={tensor_list(fd_simscape_output[0])} "
            f"tau_fd_adapted={tensor_list(tau_fd_adapted[0])} "
            f"tau_applied_raw={tensor_list(tau_applied_raw[0]) if tau_applied_raw is not None else '[]'} "
            f"tau_isaac_filtered={tensor_list(tau_isaac_filtered[0]) if tau_isaac_filtered is not None else '[]'} "
            f"tau_isaac_clamped={tensor_list(tau_isaac_clamped[0])}"
        )

    logger.write(
        {
            "mode": mode,
            "step": step,
            "sim_time": f"{sim_time:.6f}",
            "target_joint": target_joint,
            "command_effort": f"{command_effort:.6f}",
            "q": tensor_list(q[0], precision=6),
            "qdot": tensor_list(qdot[0], precision=6),
            "applied_fields": applied_summary,
            "torque_raw_mean": f"{fd_mean:.6e}",
            "torque_raw_max": f"{fd_max:.6e}",
            "torque_clamped_mean": f"{tau_clamped_mean:.6e}",
            "torque_clamped_max": f"{tau_clamped_max:.6e}",
            "torque_clamp_fraction": f"{clamp_fraction:.6f}",
            "torque_adapter_mode": torque_adapter_mode or args_cli.torque_adapter_mode,
            "torque_adapter_preset": args_cli.torque_adapter_preset,
            "torque_adapter_scale": tensor_list(torch.as_tensor(args_cli.torque_adapter_scale), precision=8),
            "use_fd_residual_alpha": str(args_cli.use_fd_residual_alpha),
            "fd_residual_alpha": f"{args_cli.fd_residual_alpha:.6e}",
            "effective_torque_adapter_scale": tensor_list(
                torch.as_tensor(
                    effective_torque_adapter_scale(
                        args_cli.torque_adapter_scale,
                        args_cli.fd_residual_alpha,
                        args_cli.use_fd_residual_alpha,
                    )
                ),
                precision=10,
            ),
            "fd_simscape_output": tensor_list(fd_simscape_output[0], precision=8)
            if fd_simscape_output is not None
            else "",
            "tau_fd_adapted": tensor_list(tau_fd_adapted[0], precision=8)
            if tau_fd_adapted is not None
            else "",
            "tau_applied_raw": tensor_list(tau_applied_raw[0], precision=8)
            if tau_applied_raw is not None
            else "",
            "tau_isaac_filtered": tensor_list(tau_isaac_filtered[0], precision=8)
            if tau_isaac_filtered is not None
            else "",
            "tau_isaac_clamped": tensor_list(tau_isaac_clamped[0], precision=8)
            if tau_isaac_clamped is not None
            else "",
            "applied_torque": tensor_list(effort_fields["applied_torque"][0], precision=8)
            if "applied_torque" in effort_fields
            else "",
            "computed_torque": tensor_list(effort_fields["computed_torque"][0], precision=8)
            if "computed_torque" in effort_fields
            else "",
            "joint_effort": tensor_list(effort_fields["joint_effort"][0], precision=8)
            if "joint_effort" in effort_fields
            else "",
        }
    )


def load_fd_model(device: str) -> ForwardDynamicsModel:
    models_dir = Path(args_cli.models_dir)
    return ForwardDynamicsModel(
        checkpoint_path=models_dir / "fd.pth",
        scaler_path=models_dir / "fd_scaler.pth",
        activation="relu",
        dropout=0.0,
        map_location="cpu",
        device=device,
    )


def fd_forward_debug(fd: ForwardDynamicsModel, fd_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        fd_input_norm = fd.normalize_x(fd_seq)
        out, _ = fd.model.lstm(fd_input_norm)
        h_last = out[:, -1, :]
        skip = fd.model.delta_skip(fd_input_norm[:, -1, :])
        hf = fd.model.hf_conv(fd_input_norm.transpose(1, 2)).squeeze(-1)
        fused = torch.cat([h_last, skip, hf], dim=-1)
        fd_output_model = fd.model.fc(fused)
        fd_output_denorm = fd.denormalize_y(fd_output_model)
    return fd_input_norm, fd_output_model, fd_output_denorm


def build_repeated_fd_seq(fd: ForwardDynamicsModel, feature: torch.Tensor) -> torch.Tensor:
    window_size = fd.window_size or 10
    return feature.unsqueeze(1).repeat(1, window_size, 1)


def run_fd_sanity(robot, state_joint_ids: list[int], logger: CsvLogger) -> None:
    device = robot.data.joint_pos.device
    fd = load_fd_model(str(device))
    q = robot.data.joint_pos[:, state_joint_ids]
    qdot = robot.data.joint_vel[:, state_joint_ids]
    zeros_qdot = torch.zeros_like(qdot)
    zeros_dP = torch.zeros((q.shape[0], 4), dtype=torch.float32, device=device)
    zeros_valve = torch.zeros((q.shape[0], 4), dtype=torch.float32, device=device)
    valve_01 = torch.full((q.shape[0], 4), 0.1, dtype=torch.float32, device=device)

    cases = [
        ("q_current_qdot0_dP0_valve0", q, zeros_qdot, zeros_dP, zeros_valve),
        ("q_current_qdot0_dP0_valve0.1", q, zeros_qdot, zeros_dP, valve_01),
        ("q_current_qdot_current_dP0_valve0", q, qdot, zeros_dP, zeros_valve),
    ]
    print("[INFO] FD output convention:")
    if fd.y_mean is None or fd.y_std is None:
        print("  fd.model.fc output appears to be physical torque directly; no y scaler was found.")
    else:
        print("  fd.model.fc output is treated as normalized/scaled torque and denormalized with fd_scaler.pth.")

    for name, q_case, qdot_case, dP_case, valve_case in cases:
        fd_input_raw = safe_tensor(build_fd_input(q_case, qdot_case, dP_case, valve_case), None)
        fd_seq = build_repeated_fd_seq(fd, fd_input_raw)
        fd_input_norm, fd_output_model, fd_output_denorm = fd_forward_debug(fd, fd_seq)
        fd_input_norm = safe_tensor(fd_input_norm, None)
        fd_output_model = safe_tensor(fd_output_model, None)
        fd_output_denorm = safe_tensor(fd_output_denorm, None)
        print(f"\n[FD_SANITY] {name}")
        print("  " + stats_text("fd_input_raw", fd_input_raw))
        print("  " + stats_text("fd_input_norm", fd_input_norm))
        print("  " + stats_text("fd_output_model", fd_output_model))
        print("  " + stats_text("fd_output_denorm_torque", fd_output_denorm))
        print(f"  fd_output_denorm_torque={tensor_list(fd_output_denorm[0])}")
        row = {
            "mode": "fd_sanity",
            "step": 0,
            "sim_time": "0.000000",
            "target_joint": "fd_sanity",
            "q": tensor_list(q_case[0], precision=6),
            "qdot": tensor_list(qdot_case[0], precision=6),
            "fd_case": name,
            "fd_output_denorm_values": tensor_list(fd_output_denorm[0], precision=6),
        }
        row.update(stats_dict("fd_input_raw", fd_input_raw))
        row.update(stats_dict("fd_input_norm", fd_input_norm))
        row.update(stats_dict("fd_output_model", fd_output_model))
        row.update(stats_dict("fd_output_denorm", fd_output_denorm))
        logger.write(row)


def run_sweep_effort(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    robot,
    state_joint_ids: list[int],
    command_joint_ids: list[int],
    logger: CsvLogger,
) -> None:
    print_effort_fields_once(robot, command_joint_ids)
    print(f"[INFO] Sweep effort values: {EFFORT_SWEEP_VALUES}")
    print(f"[INFO] Applied effort is clamped to +/-{args_cli.max_abs_torque:.3f}.")
    sim_dt = sim.get_physics_dt()
    max_steps = int(args_cli.duration / sim_dt) if args_cli.duration > 0 else math.inf
    step = 0
    try:
        for joint_name, joint_id in zip(COMMAND_JOINT_NAMES, command_joint_ids):
            for effort in EFFORT_SWEEP_VALUES:
                effort_cmd = torch.zeros(
                    (args_cli.num_envs, len(command_joint_ids)),
                    dtype=torch.float32,
                    device=robot.data.joint_pos.device,
                )
                local_joint_index = command_joint_ids.index(joint_id)
                effort_cmd[:, local_joint_index] = float(effort)
                effort_cmd = safe_tensor(effort_cmd, args_cli.max_abs_torque)
                for _ in range(args_cli.hold_steps):
                    if step >= max_steps or not simulation_app.is_running():
                        print("[INFO] sweep_effort stopped by duration/app limit.")
                        return
                    robot.set_joint_effort_target(effort_cmd, joint_ids=command_joint_ids)
                    if args_cli.print_interval > 0 and step % args_cli.print_interval == 0:
                        log_state(
                            logger=logger,
                            mode="sweep_effort",
                            step=step,
                            sim_time=step * sim_dt,
                            robot=robot,
                            state_joint_ids=state_joint_ids,
                            command_joint_ids=command_joint_ids,
                            target_joint=joint_name,
                            command_effort=float(effort_cmd[0, local_joint_index].item()),
                        )
                    write_sim_step(sim, scene)
                    step += 1
    finally:
        zero_efforts(robot, command_joint_ids)
        scene.write_data_to_sim()


def run_hold_position(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    robot,
    state_joint_ids: list[int],
    command_joint_ids: list[int],
    logger: CsvLogger,
) -> None:
    print_effort_fields_once(robot, state_joint_ids)
    sim_dt = sim.get_physics_dt()
    max_steps = int(args_cli.duration / sim_dt) if args_cli.duration > 0 else math.inf
    target = robot.data.default_joint_pos[:, state_joint_ids].clone()
    step = 0
    while simulation_app.is_running() and step < max_steps:
        robot.set_joint_position_target(target, joint_ids=state_joint_ids)
        if args_cli.print_interval > 0 and step % args_cli.print_interval == 0:
            log_state(
                logger=logger,
                mode="hold_position",
                step=step,
                sim_time=step * sim_dt,
                robot=robot,
                state_joint_ids=state_joint_ids,
                command_joint_ids=state_joint_ids,
            )
        write_sim_step(sim, scene)
        step += 1


def run_compare_fd_vs_applied(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    robot,
    state_joint_ids: list[int],
    command_joint_ids: list[int],
    logger: CsvLogger,
) -> None:
    print_effort_fields_once(robot, command_joint_ids)
    device = robot.data.joint_pos.device
    fd = load_fd_model(str(device))
    print(
        "[INFO] compare_fd_vs_applied uses adapter "
        f"mode={args_cli.torque_adapter_mode} preset={args_cli.torque_adapter_preset} "
        f"scale={args_cli.torque_adapter_scale} "
        f"bias={args_cli.torque_adapter_bias} "
        f"use_fd_residual_alpha={args_cli.use_fd_residual_alpha} "
        f"fd_residual_alpha={args_cli.fd_residual_alpha}; raw FD is never applied directly."
    )
    fd_buffer = SequenceBuffer(
        num_envs=args_cli.num_envs,
        window_size=fd.window_size or 10,
        feature_dim=18,
        device=device,
    )
    sim_dt = sim.get_physics_dt()
    max_steps = int(args_cli.duration / sim_dt) if args_cli.duration > 0 else math.inf
    step = 0
    prev_tau_isaac = torch.zeros((args_cli.num_envs, len(command_joint_ids)), dtype=torch.float32, device=device)
    phases = torch.linspace(0.0, math.pi, len(command_joint_ids), dtype=torch.float32, device=device).reshape(1, -1)
    try:
        while simulation_app.is_running() and step < max_steps:
            q = robot.data.joint_pos[:, state_joint_ids]
            qdot = robot.data.joint_vel[:, state_joint_ids]
            dP = torch.zeros((args_cli.num_envs, 4), dtype=torch.float32, device=device)
            valve_cmd = 0.1 * torch.sin(2.0 * math.pi * 0.2 * (step * sim_dt) + phases)
            fd_input_raw = safe_tensor(build_fd_input(q, qdot, dP, valve_cmd), None)
            fd_seq = fd_buffer.append(fd_input_raw)
            fd_simscape_output = safe_tensor(fd(fd_seq), None)
            tau_fd_adapted = adapt_fd_to_isaac_torque(
                fd_simscape_output,
                mode=args_cli.torque_adapter_mode,
                scale=args_cli.torque_adapter_scale,
                bias=args_cli.torque_adapter_bias,
                fd_torque_scale=args_cli.fd_torque_scale,
                fd_residual_alpha=args_cli.fd_residual_alpha,
                max_abs_torque=args_cli.max_abs_torque,
                q=q,
                qdot=qdot,
            )
            tau_applied_raw = apply_fd_residual_authority(
                tau_fd_adapted,
                fd_residual_alpha=args_cli.fd_residual_alpha,
                use_fd_residual_alpha=args_cli.use_fd_residual_alpha,
            )
            tau_isaac_filtered = lowpass_torque(tau_applied_raw, prev_tau_isaac, args_cli.torque_lowpass_alpha)
            tau_isaac_filtered, torque_rate_limited_fraction = rate_limit_torque(
                tau_isaac_filtered,
                prev_tau_isaac,
                args_cli.torque_rate_limit,
            )
            tau_isaac_filtered = safe_tensor(tau_isaac_filtered, None)
            torque_clamped = safe_tensor(tau_isaac_filtered, args_cli.max_abs_torque)
            prev_tau_isaac = torque_clamped.detach().clone()
            torque_limit = max(float(args_cli.max_abs_torque), 1.0e-8)
            torque_clamp_fraction = torch.mean((torch.abs(tau_isaac_filtered) > torque_limit).float(), dim=-1)
            robot.set_joint_effort_target(torque_clamped, joint_ids=command_joint_ids)

            if args_cli.print_interval > 0 and step % args_cli.print_interval == 0:
                log_state(
                    logger=logger,
                    mode="compare_fd_vs_applied",
                    step=step,
                    sim_time=step * sim_dt,
                    robot=robot,
                    state_joint_ids=state_joint_ids,
                    command_joint_ids=command_joint_ids,
                    target_joint="fd_online",
                    command_effort=float(torque_clamped.abs().max().item()),
                    fd_simscape_output=fd_simscape_output,
                    tau_fd_adapted=tau_fd_adapted,
                    tau_applied_raw=tau_applied_raw,
                    tau_isaac_filtered=tau_isaac_filtered,
                    tau_isaac_clamped=torque_clamped,
                    torque_clamp_fraction=torque_clamp_fraction,
                    torque_adapter_mode=args_cli.torque_adapter_mode,
                )
                if float(torque_clamp_fraction.mean().item()) > 0.5:
                    print("[WARN] final torque still too clamped")
                if float(torque_rate_limited_fraction.mean().item()) > 0.8:
                    print("[WARN] torque still rate-limited")
                tau_applied_raw_max = float(tau_applied_raw.abs().max().item())
                if tau_applied_raw_max > 2.0 * float(args_cli.max_abs_torque):
                    print("[WARN] residual alpha still too high")
                if tau_applied_raw_max < 0.1:
                    print("[WARN] residual alpha may be too low")
            write_sim_step(sim, scene)
            step += 1
    finally:
        zero_efforts(robot, command_joint_ids)
        scene.write_data_to_sim()


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene) -> None:
    robot = scene["robot"]
    print_joint_table(robot)
    state_joint_ids = find_joint_ids(robot, STATE_JOINT_NAMES)
    command_joint_ids = find_joint_ids(robot, COMMAND_JOINT_NAMES)
    print("[INFO] State joints:")
    for joint_name, joint_id in zip(STATE_JOINT_NAMES, state_joint_ids):
        print(f"  {joint_name}: {joint_id}")
    print("[INFO] Command/actuated joints:")
    for joint_name, joint_id in zip(COMMAND_JOINT_NAMES, command_joint_ids):
        print(f"  {joint_name}: {joint_id}")

    logger = CsvLogger(enabled=not args_cli.no_csv)
    if logger.path is not None:
        metadata_path = logger.path.parent / "joint_names.txt"
        with metadata_path.open("w") as f:
            f.write("All robot joints:\n")
            for joint_id, joint_name in enumerate(robot.joint_names):
                f.write(f"{joint_id}: {joint_name}\n")
            f.write("\nState joints:\n")
            for joint_name, joint_id in zip(STATE_JOINT_NAMES, state_joint_ids):
                f.write(f"{joint_id}: {joint_name}\n")
            f.write("\nCommand joints:\n")
            for joint_name, joint_id in zip(COMMAND_JOINT_NAMES, command_joint_ids):
                f.write(f"{joint_id}: {joint_name}\n")
        print(f"[INFO] Writing joint metadata to: {metadata_path}")
    try:
        if args_cli.mode == "sweep_effort":
            run_sweep_effort(sim, scene, robot, state_joint_ids, command_joint_ids, logger)
        elif args_cli.mode == "hold_position":
            run_hold_position(sim, scene, robot, state_joint_ids, command_joint_ids, logger)
        elif args_cli.mode == "fd_sanity":
            run_fd_sanity(robot, state_joint_ids, logger)
        elif args_cli.mode == "compare_fd_vs_applied":
            run_compare_fd_vs_applied(sim, scene, robot, state_joint_ids, command_joint_ids, logger)
        else:
            raise ValueError(f"Unsupported mode: {args_cli.mode}")
    finally:
        zero_efforts(robot, command_joint_ids)
        logger.close()


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device, render_interval=2)
    apply_kevin_sim_memory_optimizations(sim_cfg, verbose=True)

    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([11, 8, 1.3], [0.0, 0.0, 0.0])
    scene_cfg = PlaneScene(num_envs=args_cli.num_envs, env_spacing=20.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[INFO] Setup complete.")
    run_simulator(sim, scene)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
