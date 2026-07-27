import torch

from ROM_pipeline.data import DEFAULT_SCHEMA
from ROM_pipeline.models import HydraulicROM


def test_model_is_finite_and_integrates_position():
    torch.manual_seed(3)
    model = HydraulicROM().eval()
    history = torch.zeros(3, DEFAULT_SCHEMA.history_length, DEFAULT_SCHEMA.input_dim)
    history[:, :, DEFAULT_SCHEMA.extension_slice] = 0.4
    command = torch.zeros(3, 8)
    command[:, 0] = torch.tensor([-0.005, 0.0, 0.005])
    output = model(history, command)
    assert output.delta_extension.shape == (3, 4)
    assert torch.isfinite(output_tensor(output)).all()
    assert torch.allclose(output.next_extension, history[:, -1, :4] + output.delta_extension)


def output_tensor(output):
    return torch.cat((output.predicted_valve, output.delta_pressure, output.delta_velocity, output.delta_extension), -1)


def test_command_reaches_effective_actuation_head():
    torch.manual_seed(4)
    model = HydraulicROM().eval()
    history = torch.zeros(2, DEFAULT_SCHEMA.history_length, DEFAULT_SCHEMA.input_dim)
    negative = torch.zeros(2, 8)
    positive = torch.zeros(2, 8)
    negative[:, 0] = -0.01
    positive[:, 0] = 0.01
    assert not torch.allclose(model(history, negative).effective_actuation, model(history, positive).effective_actuation)
