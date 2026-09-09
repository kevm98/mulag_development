from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[2]
ROM_ROOT = ROOT / "learned_models/rom_model"
sys.path.insert(0, str(ROM_ROOT))

from ROM_pipeline.models import HydraulicROM, save_checkpoint  # noqa: E402
from mulag_plant.runtime.command_rom_runtime import (  # noqa: E402
    CommandCylinderROMRuntime,
    pack_desired_cylinder_command,
)


def test_command_rom_runtime_is_multirate_and_prediction_is_nonmutating(tmp_path):
    checkpoint = tmp_path / "rom.pt"
    save_checkpoint(
        checkpoint,
        HydraulicROM().eval(),
        target_mean=torch.zeros(16),
        target_std=torch.ones(16),
        cylinder_limits=torch.tensor([[0.0, 1.0]] * 4),
        training_metadata={"test": True},
    )
    runtime = CommandCylinderROMRuntime(
        checkpoint=checkpoint, num_envs=2, device="cpu", control_dt_s=1.0 / 60.0
    )
    values = {
        "desired_cylinder_command": torch.zeros(2, 8),
        "extension": torch.full((2, 4), 0.4),
        "extension_velocity": torch.zeros(2, 4),
        "dP": torch.zeros(2, 4),
    }
    prediction = runtime.predict_delta(**values)
    assert prediction.shape == (2, 4)
    assert runtime.history_count.sum() == 0
    first = runtime.step(**values)
    assert first.update_mask.tolist() == [True, True]
    assert torch.isfinite(first.predicted_valve).all()
    assert torch.allclose(first.delta_extension, prediction / 3.0)
    applied = first.delta_extension.clone()
    for _ in range(2):
        held = runtime.step(**values)
        assert not held.update_mask.any()
        assert torch.allclose(held.delta_extension, first.delta_extension)
        applied += held.delta_extension
    assert torch.allclose(applied, prediction)
    assert runtime.step(**values).update_mask.all()


def test_command_rom_runtime_partial_reset(tmp_path):
    checkpoint = tmp_path / "rom.pt"
    save_checkpoint(
        checkpoint, HydraulicROM().eval(), target_mean=torch.zeros(16), target_std=torch.ones(16),
        cylinder_limits=torch.tensor([[0.0, 1.0]] * 4), training_metadata={"test": True},
    )
    runtime = CommandCylinderROMRuntime(
        checkpoint=checkpoint, num_envs=2, device="cpu", control_dt_s=1.0 / 60.0
    )
    values = dict(desired_cylinder_command=torch.zeros(2, 8), extension=torch.full((2, 4), 0.4),
                  extension_velocity=torch.zeros(2, 4), dP=torch.zeros(2, 4))
    runtime.step(**values)
    runtime.step(**values)
    runtime.reset(torch.tensor([0]))
    assert runtime.step(**values).update_mask.tolist() == [True, False]


def test_command_rom_rejects_valve_width_and_packs_motion_command():
    extension = torch.zeros(2, 4)
    velocity = torch.ones(2, 4)
    command = pack_desired_cylinder_command(extension, velocity)
    assert command.shape == (2, 8)
    assert torch.equal(command[:, :4], extension)
    assert torch.equal(command[:, 4:], velocity)

    try:
        pack_desired_cylinder_command(extension, torch.zeros(2, 4, 1))
    except ValueError:
        pass
    else:
        raise AssertionError("a non-4D motion command must be rejected")


def test_command_rom_public_keyword_is_motion_not_valve(tmp_path):
    checkpoint = tmp_path / "rom.pt"
    save_checkpoint(
        checkpoint,
        HydraulicROM().eval(),
        target_mean=torch.zeros(16),
        target_std=torch.ones(16),
        cylinder_limits=torch.tensor([[0.0, 1.0]] * 4),
        training_metadata={"test": True},
    )
    runtime = CommandCylinderROMRuntime(
        checkpoint=checkpoint, num_envs=1, device="cpu", control_dt_s=1.0 / 60.0
    )
    values = dict(
        desired_cylinder_command=torch.zeros(1, 8),
        extension=torch.full((1, 4), 0.4),
        extension_velocity=torch.zeros(1, 4),
        dP=torch.zeros(1, 4),
    )
    runtime.step(**values)
    try:
        runtime.step(
            command=torch.zeros(1, 8),
            **{key: value for key, value in values.items() if key != "desired_cylinder_command"},
        )
    except TypeError:
        pass
    else:
        raise AssertionError(
            "the command-ROM API must reject the ambiguous command keyword"
        )
