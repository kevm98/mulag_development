"""Offline PPO2_C contracts; these tests do not launch Isaac Sim."""

from pathlib import Path

import torch

from PPO2_ROM_C_PhysX.checkpoint_transfer import _expanded_state
from PPO2_ROM_C_PhysX.runtime.aam_runtime import (
    AdaptiveCommandCylinderROMRuntime,
    OnlineAAMRuntime,
)
from PPO2_ROM_PhysX.runtime.command_rom_runtime import CommandCylinderROMRuntime


ROOT = Path(__file__).resolve().parents[2]
AAM = ROOT / "AAM_SystemFactor/checkpoints/aam_system_factor_runtime.pt"
ROM = (
    ROOT / "ROM-cylinders_pipeline/checkpoints"
    / "plant_reachable_v1_finetune_blend25/best_rom.pt"
)
ROM_B = (
    ROOT / "logs/rsl_rl/ppo2_rom_b_robust/2026-07-24_12-48-14"
    / "model_53_ppo2_rom_b_final.pt"
)


def test_aam_waits_for_ten_real_50ms_samples():
    runtime = OnlineAAMRuntime(
        checkpoint=AAM, num_envs=2, device="cpu", control_dt_s=1.0 / 60.0
    )
    zero = torch.zeros(2, 4)
    for _ in range(27):
        physical, policy, ready = runtime.observe(
            valve=zero, extension=zero, velocity=zero, dP=zero, force=zero
        )
        assert not ready.any()
        assert torch.count_nonzero(physical) == 0
        assert torch.count_nonzero(policy) == 0
    physical, policy, ready = runtime.observe(
        valve=zero, extension=zero, velocity=zero, dP=zero, force=zero
    )
    assert ready.all()
    assert physical.shape == policy.shape == (2, 5)
    assert torch.isfinite(policy).all()
    assert torch.all(policy.abs() <= 1.0)


def test_zero_factors_are_exactly_backward_compatible():
    nominal = CommandCylinderROMRuntime(
        checkpoint=ROM, num_envs=2, device="cpu", control_dt_s=1.0 / 60.0
    )
    adaptive = AdaptiveCommandCylinderROMRuntime(
        checkpoint=ROM, num_envs=2, device="cpu", control_dt_s=1.0 / 60.0,
        randomize_factors=False,
    )
    zero = torch.zeros(2, 4)
    values = dict(
        command=torch.zeros(2, 8), extension=zero,
        extension_velocity=zero, dP=zero,
    )
    expected = nominal.step(**values)
    actual = adaptive.step(**values)
    for field in (
        "predicted_valve", "delta_dP", "extension_velocity_next",
        "delta_extension", "update_mask",
    ):
        assert torch.equal(getattr(expected, field), getattr(actual, field))


def test_hidden_factors_change_the_displacement_seen_by_physx():
    runtime = AdaptiveCommandCylinderROMRuntime(
        checkpoint=ROM, num_envs=2, device="cpu", control_dt_s=1.0 / 60.0,
        randomize_factors=True,
    )
    zero = torch.zeros(2, 4)
    runtime.true_factors[0] = 0.0
    runtime.true_factors[1] = torch.tensor([-0.10, 1.0, 1.0, 1.0, 1.0])
    extension = runtime.model.input_mean[:4].expand(2, -1).clone()
    velocity = zero.clone()
    pressure = zero.clone()
    for step in range(120):
        command = torch.zeros(2, 8)
        command[:, :4] = 0.03 * torch.sin(torch.tensor(step / 25.0))
        output = runtime.step(
            command=command,
            extension=extension,
            extension_velocity=velocity,
            dP=pressure,
        )
        extension = extension + output.delta_extension
        velocity = output.extension_velocity_next
        pressure = pressure + output.delta_dP
    assert torch.mean(torch.abs(extension[0] - extension[1])) > 1.0e-4


def test_rom_b_transfer_keeps_old_columns_and_zeros_aam_columns():
    payload = torch.load(ROM_B, map_location="cpu", weights_only=False)
    actor_source = payload["actor_state_dict"]
    actor_template = {key: value.clone() for key, value in actor_source.items()}
    for key in ("obs_normalizer._mean", "obs_normalizer._var", "obs_normalizer._std"):
        old = actor_template[key]
        fill = 0.0 if key.endswith("_mean") else 1.0
        actor_template[key] = torch.full((*old.shape[:-1], 47), fill)
    old_teacher = actor_source["mlp.teacher_mlp.0.weight"]
    actor_template["mlp.teacher_mlp.0.weight"] = torch.empty(
        old_teacher.shape[0], 47
    )
    old_pitch = actor_source["mlp.pitch_mlp.0.weight"]
    actor_template["mlp.pitch_mlp.0.weight"] = torch.empty(
        old_pitch.shape[0], old_pitch.shape[1] + 5
    )
    actor_template["mlp.pitch_obs_indices"] = torch.cat((
        actor_source["mlp.pitch_obs_indices"], torch.arange(42, 47)
    ))
    expanded = _expanded_state(actor_source, actor_template, actor=True)
    assert torch.equal(
        expanded["mlp.teacher_mlp.0.weight"][:, :42], old_teacher
    )
    assert torch.count_nonzero(
        expanded["mlp.teacher_mlp.0.weight"][:, 42:]
    ) == 0
    assert torch.equal(
        expanded["mlp.pitch_mlp.0.weight"][:, :old_pitch.shape[1]], old_pitch
    )
    assert torch.count_nonzero(
        expanded["mlp.pitch_mlp.0.weight"][:, old_pitch.shape[1]:]
    ) == 0
