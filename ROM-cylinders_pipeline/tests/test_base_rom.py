import torch

from ROM_pipeline.data import BASE_ROM_SCHEMA
from ROM_pipeline.models import BaseHydraulicROM, load_base_checkpoint, save_base_checkpoint
from ROM_pipeline.runtime import BaseMultirateConfig, BaseROMActuator, BaseROMMultirate


def test_base_rom_contract_and_valve_authority():
    torch.manual_seed(5)
    model = BaseHydraulicROM().eval()
    history = torch.zeros(2, BASE_ROM_SCHEMA.history_length, BASE_ROM_SCHEMA.input_dim)
    history[:, :, BASE_ROM_SCHEMA.extension_slice] = 0.4
    negative = model(history, -torch.ones(2, 4))
    positive = model(history, torch.ones(2, 4))
    assert negative.delta_extension.shape == (2, 4)
    assert torch.isfinite(positive.delta_pressure).all()
    assert not torch.allclose(negative.effective_actuation, positive.effective_actuation)


def test_base_checkpoint_and_runtime(tmp_path):
    model = BaseHydraulicROM()
    path = tmp_path / "base.pt"
    save_base_checkpoint(
        path, model, target_mean=torch.zeros(16), target_std=torch.ones(16),
        cylinder_limits=torch.tensor([[0.0, 1.0]] * 4), training_metadata={"test": True},
    )
    restored, _ = load_base_checkpoint(path)
    actuator = BaseROMActuator(
        restored, num_envs=1, cylinder_lower=torch.zeros(1, 4), cylinder_upper=torch.ones(1, 4)
    )
    measured = torch.full((1, 4), 0.4)
    state = torch.zeros(1, 26)
    state[:, :4] = measured
    actuator.reset(measured, torch.cat((torch.zeros(1, 4), state), -1))
    output = actuator.step(state_row=state, valve_command=torch.zeros(1, 4), measured_extension=measured)
    assert output.position_target.shape == (1, 4)
    assert torch.isfinite(output.position_target).all()


def test_multirate_applies_full_rom_delta_once_and_holds():
    torch.manual_seed(8)
    model = BaseHydraulicROM().eval()
    actuator = BaseROMActuator(
        model, num_envs=1, cylinder_lower=torch.zeros(1, 4), cylinder_upper=torch.ones(1, 4)
    )
    wrapper = BaseROMMultirate(
        actuator, num_envs=1, config=BaseMultirateConfig(control_dt_s=0.01, rom_dt_s=0.05)
    )
    measured = torch.full((1, 4), 0.4)
    state = torch.zeros(1, 26)
    state[:, :4] = measured
    row = torch.cat((torch.zeros(1, 4), state), -1)
    wrapper.reset(measured, row)
    first, updated, output = wrapper.step(
        state_row=state, valve_command=torch.ones(1, 4) * 0.5, measured_extension=measured
    )
    assert updated and output is not None
    for _ in range(4):
        held, updated, _ = wrapper.step(
            state_row=state, valve_command=torch.ones(1, 4) * 0.5, measured_extension=measured
        )
        assert not updated
        assert torch.allclose(held, first)
    _, updated, _ = wrapper.step(
        state_row=state, valve_command=torch.ones(1, 4) * 0.5, measured_extension=measured
    )
    assert updated
