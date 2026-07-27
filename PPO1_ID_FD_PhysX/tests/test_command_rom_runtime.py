from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[2]
ROM_ROOT = ROOT / "ROM-cylinders_pipeline"
sys.path.insert(0, str(ROM_ROOT))

from ROM_pipeline.models import HydraulicROM, save_checkpoint  # noqa: E402
from PPO1_ID_FD_PhysX.runtime.command_rom_runtime import CommandCylinderROMRuntime  # noqa: E402


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
        "command": torch.zeros(2, 8),
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
    for _ in range(2):
        held = runtime.step(**values)
        assert not held.update_mask.any()
        assert torch.count_nonzero(held.delta_extension) == 0
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
    values = dict(command=torch.zeros(2, 8), extension=torch.full((2, 4), 0.4),
                  extension_velocity=torch.zeros(2, 4), dP=torch.zeros(2, 4))
    runtime.step(**values)
    runtime.step(**values)
    runtime.reset(torch.tensor([0]))
    assert runtime.step(**values).update_mask.tolist() == [True, False]
