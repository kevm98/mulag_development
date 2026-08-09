from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[2]
ROM_ROOT = ROOT / "ROM-cylinders_pipeline"
if str(ROM_ROOT) not in sys.path:
    sys.path.insert(0, str(ROM_ROOT))

from ROM_pipeline.models import BaseHydraulicROM, save_base_checkpoint  # noqa: E402
from PPO1_ID_FD_PhysX.runtime.base_rom_runtime import BaseCylinderROMRuntime  # noqa: E402


def _checkpoint(path: Path) -> None:
    save_base_checkpoint(
        path,
        BaseHydraulicROM().eval(),
        target_mean=torch.zeros(16),
        target_std=torch.ones(16),
        cylinder_limits=torch.tensor([[0.0, 1.0]] * 4),
        training_metadata={"test": True},
    )


def _inputs(num_envs: int) -> dict[str, torch.Tensor]:
    return {
        "valve_cmd": torch.full((num_envs, 4), 0.25),
        "extension": torch.full((num_envs, 4), 0.4),
        "extension_velocity": torch.zeros(num_envs, 4),
        "dP": torch.zeros(num_envs, 4),
        "force": torch.zeros(num_envs, 4),
    }


def test_base_rom_runtime_applies_native_delta_once_per_tick(tmp_path):
    checkpoint = tmp_path / "base.pt"
    _checkpoint(checkpoint)
    runtime = BaseCylinderROMRuntime(
        checkpoint=checkpoint, num_envs=2, device="cpu", control_dt_s=1.0 / 60.0
    )
    inputs = _inputs(2)

    first = runtime.step(**inputs)
    assert first.update_mask.tolist() == [True, True]
    assert torch.isfinite(first.delta_extension).all()
    history_before = runtime.history.clone()
    counter_before = runtime.tick_counter.clone()
    predicted = runtime.predict_delta(**inputs)
    assert predicted.shape == (2, 4)
    assert torch.equal(runtime.history, history_before)
    assert torch.equal(runtime.tick_counter, counter_before)
    for _ in range(2):
        held = runtime.step(**inputs)
        assert not held.update_mask.any()
        assert torch.count_nonzero(held.delta_extension) == 0
    fourth = runtime.step(**inputs)
    assert fourth.update_mask.tolist() == [True, True]


def test_base_rom_runtime_partial_reset_restarts_only_selected_clock(tmp_path):
    checkpoint = tmp_path / "base.pt"
    _checkpoint(checkpoint)
    runtime = BaseCylinderROMRuntime(
        checkpoint=checkpoint, num_envs=2, device="cpu", control_dt_s=1.0 / 60.0
    )
    inputs = _inputs(2)
    runtime.step(**inputs)
    runtime.step(**inputs)
    runtime.reset(torch.tensor([0]))
    output = runtime.step(**inputs)
    assert output.update_mask.tolist() == [True, False]
    assert runtime.history_count[0] == runtime.schema.history_length
    assert runtime.history_count[1] == runtime.schema.history_length
