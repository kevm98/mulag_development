import torch

from ROM_pipeline.models import HydraulicROM, load_checkpoint, save_checkpoint


def test_checkpoint_round_trip(tmp_path):
    model = HydraulicROM()
    path = tmp_path / "rom.pt"
    save_checkpoint(
        path,
        model,
        target_mean=torch.zeros(16),
        target_std=torch.ones(16),
        cylinder_limits=torch.tensor([[0.0, 1.0]] * 4),
        training_metadata={"test": True},
    )
    restored, payload = load_checkpoint(path)
    assert payload["training_metadata"]["test"] is True
    assert set(restored.state_dict()) == set(model.state_dict())
