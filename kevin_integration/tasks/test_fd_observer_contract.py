from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_OBS = _load_module("observation_builder", ROOT / "kevin_integration" / "rl" / "observation_builder.py")
_LEARNED = _load_module("learned_models_for_fd_test", ROOT / "kevin_integration" / "controllers" / "learned_models.py")
_TASK = _load_module("task_space_reference_for_fd_test", ROOT / "kevin_integration" / "tasks" / "task_space_reference.py")


def test_task_space_observation_builder_includes_fd_tau_hat() -> None:
    n = 2
    obs = _OBS.build_task_space_rl_obs(
        ee_pose_command=torch.zeros(n, 7),
        current_ee_pose=torch.zeros(n, 7),
        ee_position_error=torch.zeros(n, 3),
        ee_orientation_error=torch.zeros(n, 1),
        mechanical_q=torch.zeros(n, 5),
        mechanical_qdot=torch.zeros(n, 5),
        cylinder_q=torch.zeros(n, 4),
        cylinder_qdot=torch.zeros(n, 4),
        prev_action=torch.zeros(n, 4),
        prev_valve_cmd=torch.zeros(n, 4),
        z_arm_hat=torch.zeros(n, 24),
        fd_tau_hat=torch.ones(n, 4),
        vehicle_speed_kmh=torch.zeros(n, 1),
    )
    if obs.shape != (n, _TASK.TASK_SPACE_OBSERVATION_DIM):
        raise AssertionError(obs.shape)
    fd_start = 7 + 7 + 3 + 1 + 5 + 5 + 4 + 4 + 4 + 4 + 24
    if not torch.equal(obs[:, fd_start : fd_start + 4], torch.ones(n, 4)):
        raise AssertionError("fd_tau_hat is not at the documented observation offset")


def test_action4_expands_to_id_qddot5() -> None:
    id_input = _OBS.build_id_input(
        q=torch.zeros(1, 5),
        qdot=torch.zeros(1, 5),
        qddot_cmd=torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
        dP=torch.zeros(1, 4),
    )
    if id_input.shape != (1, 19):
        raise AssertionError(id_input.shape)
    if not torch.equal(id_input[:, 10:15], torch.tensor([[1.0, 2.0, 3.0, 4.0, 0.0]])):
        raise AssertionError(id_input[:, 10:15])


def test_fd_observer_only_source_contract() -> None:
    cfg_source = (ROOT / "kevin_integration" / "tasks" / "mulag_adaptive_rl_env_cfg.py").read_text()
    env_source = (ROOT / "kevin_integration" / "tasks" / "mulag_adaptive_rl_env.py").read_text()
    controller_source = (ROOT / "kevin_integration" / "rl" / "adaptive_rl_controller.py").read_text()
    for needle in (
        "use_fd_model = True",
        "use_fd_as_observer = True",
        "use_fd_in_policy_observation = True",
        "use_fd_safety_limits = True",
        "use_fd_auxiliary_reward = True",
        "use_fd_effort_actuation = False",
        "fd_torque_soft_limit = \"fd_output_std\"",
        "fd_torque_hard_limit = \"fd_output_3std\"",
    ):
        if needle not in cfg_source:
            raise AssertionError(f"missing FD config: {needle}")
    for needle in (
        "def _assert_single_actuator_writer",
        "ROM position-target actuation and FD effort actuation",
        "def _run_fd_observer_only",
        "self._controller.run_fd_observer(q, qdot, dP, valve_cmd)",
        "self._fd_tau_hat_normalized",
        "self._fd_soft_limit_fraction",
        "set_joint_position_target(self._rom_cylinder_pos_target",
    ):
        if needle not in env_source:
            raise AssertionError(f"missing env FD/writer evidence: {needle}")
    for needle in (
        "def run_fd_observer",
        "def fd_output_normalizer",
        "\"fd_input\": \"[q5, qdot5, dP4, ValveCmd4]\"",
        "\"fd_effort_actuation\": \"disabled in ROM position-target mode\"",
    ):
        if needle not in controller_source:
            raise AssertionError(f"missing controller FD evidence: {needle}")


def test_fd_checkpoint_scaler_and_inference_are_finite() -> None:
    fd = _LEARNED.ForwardDynamicsModel(
        checkpoint_path=ROOT / "kevin_integration" / "models" / "fd.pth",
        scaler_path=ROOT / "kevin_integration" / "models" / "fd_scaler.pth",
        map_location="cpu",
        device="cpu",
    )
    window_size = int(fd.window_size or 10)
    if window_size <= 0:
        raise AssertionError(window_size)
    x_mean = torch.as_tensor(fd.x_mean, dtype=torch.float32).flatten()
    x_std = torch.as_tensor(fd.x_std, dtype=torch.float32).flatten()
    y_mean = torch.as_tensor(fd.y_mean, dtype=torch.float32).flatten()
    y_std = torch.as_tensor(fd.y_std, dtype=torch.float32).flatten()
    if x_mean.numel() != 18 or x_std.numel() != 18:
        raise AssertionError((x_mean.numel(), x_std.numel()))
    if y_mean.numel() != 4 or y_std.numel() != 4:
        raise AssertionError((y_mean.numel(), y_std.numel()))
    if not torch.isfinite(torch.cat([x_mean, x_std, y_mean, y_std])).all():
        raise AssertionError("FD scaler contains non-finite values")
    if not torch.all(x_std > 0) or not torch.all(y_std > 0):
        raise AssertionError("FD scaler standard deviations must be positive")
    x_hist = x_mean.reshape(1, 1, 18).repeat(1, window_size, 1)
    y = fd(x_hist)
    if y.shape != (1, 4):
        raise AssertionError(y.shape)
    if not torch.isfinite(y).all():
        raise AssertionError(y)


def test_preflight_decision_vocabulary() -> None:
    expected = {
        "BLOCKED_FD_INPUT_CONTRACT",
        "BLOCKED_FD_OUTPUT_NONFINITE",
        "BLOCKED_FD_ACTUATOR_CONFLICT",
        "BLOCKED_TASKSPACE_OBSERVATION",
        "BLOCKED_TASKSPACE_REWARD",
        "BLOCKED_VEHICLE_COMMAND",
        "READY_FOR_PPO",
    }
    env_source = (ROOT / "kevin_integration" / "tasks" / "mulag_adaptive_rl_env.py").read_text()
    missing = [name for name in expected if name not in env_source and not hasattr(_TASK, name)]
    if missing:
        raise AssertionError(missing)


def main() -> None:
    test_task_space_observation_builder_includes_fd_tau_hat()
    test_action4_expands_to_id_qddot5()
    test_fd_observer_only_source_contract()
    test_fd_checkpoint_scaler_and_inference_are_finite()
    test_preflight_decision_vocabulary()
    print("PASS: FD observer and task-space policy contract tests passed.")


if __name__ == "__main__":
    main()
