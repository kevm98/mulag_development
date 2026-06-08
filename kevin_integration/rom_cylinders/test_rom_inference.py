from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

try:
    from models import ModelConfig, build_model
    from rom_dataset import DEFAULT_DATASET_PATH, ROMSequenceDataset, find_default_dataset
    from rom_inference import DEFAULT_EXPORT_DIR, LearnedCylinderROM
except ImportError:
    from kevin_integration.rom_cylinders.models import ModelConfig, build_model
    from kevin_integration.rom_cylinders.rom_dataset import DEFAULT_DATASET_PATH, ROMSequenceDataset, find_default_dataset
    from kevin_integration.rom_cylinders.rom_inference import DEFAULT_EXPORT_DIR, LearnedCylinderROM


def load_exported_model(export_dir: Path, device: torch.device) -> tuple[torch.nn.Module, dict, dict]:
    model_payload = torch.load(export_dir / "best_rom.pth", map_location=device)
    scaler = torch.load(export_dir / "best_rom_scaler.pth", map_location=device)
    metadata = json.loads((export_dir / "best_rom_metadata.json").read_text(encoding="utf-8"))
    config = json.loads((export_dir / "best_rom_config.json").read_text(encoding="utf-8"))
    stats = SimpleNamespace(
        input_mean=scaler["input_mean"].detach().cpu(),
        input_std=scaler["input_std"].detach().cpu(),
        target_mean=scaler["target_mean"].detach().cpu(),
        target_std=scaler["target_std"].detach().cpu(),
    )
    model = build_model(
        ModelConfig(
            model_type=config["model_type"],
            input_dim=int(metadata["input_dimension"]),
            output_dim=int(metadata["target_dimension"]),
            target_mode=str(metadata.get("target_mode", config.get("target_mode", "extension_and_dp"))),
            hidden_dim=int(config.get("hidden_dim", 64) or 64),
            num_layers=int(config.get("num_layers", 2) or 2),
            dropout=float(config.get("dropout", 0.0) or 0.0),
            bounded_outputs=bool(config.get("bounded_outputs", False)),
            output_scale=tuple(config["output_scale"]) if config.get("output_scale") is not None else None,
        ),
        normalization_stats=stats,
    )
    model.load_state_dict(model_payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model, scaler, metadata


def compare_device(export_dir: Path, dataset_path: Path, device: torch.device, *, batch_size: int, atol: float) -> torch.Tensor:
    metadata = json.loads((export_dir / "best_rom_metadata.json").read_text(encoding="utf-8"))
    use_force = bool(metadata["use_force_input"])
    target_dim = int(metadata["target_dimension"])
    dataset = ROMSequenceDataset(
        dataset_path,
        split=str(metadata.get("export_split", "val")),
        use_force_input=use_force,
        split_scheme=str(metadata.get("split_scheme", "original")),
        locked_manifest=metadata.get("locked_manifest") or Path(export_dir) / "rom_split_manifest.json",
        target_mode=str(metadata.get("target_mode", "extension_and_dp")),
        normalize_inputs=False,
        normalize_targets=False,
        return_metadata=True,
    )
    raw_sequences = []
    for idx in range(batch_size):
        _, _, info = dataset[idx]
        raw_sequences.append(info["raw_input"])
    raw = torch.stack(raw_sequences, dim=0).to(device)

    direct, scaler, _ = load_exported_model(export_dir, device)
    input_mean = scaler["input_mean"].to(device).float().reshape(1, 1, -1)
    input_std = torch.clamp(scaler["input_std"].to(device).float().reshape(1, 1, -1), min=1.0e-12)
    target_mean = scaler["target_mean"].to(device).float().reshape(1, -1)
    target_std = torch.clamp(scaler["target_std"].to(device).float().reshape(1, -1), min=1.0e-12)
    with torch.no_grad():
        direct_norm = direct((raw - input_mean) / input_std)
        direct_raw = direct_norm * target_std + target_mean

    wrapper = LearnedCylinderROM(export_dir, num_envs=batch_size, device=device)
    delta_ext, delta_dp, debug = wrapper.predict_from_feature_sequence(raw)
    wrapper_raw = torch.cat([delta_ext, delta_dp], dim=-1)
    if not torch.allclose(direct_raw, wrapper_raw[:, :target_dim], atol=atol, rtol=1.0e-5):
        diff = torch.max(torch.abs(direct_raw - wrapper_raw[:, :target_dim])).item()
        raise AssertionError(f"Wrapper output mismatch on {device}: max abs diff {diff}")
    assert torch.isfinite(wrapper_raw).all()
    assert torch.isfinite(debug["pred_raw"]).all()
    assert tuple(debug["pred_raw"].shape) == (batch_size, target_dim)
    assert tuple(delta_ext.shape) == (batch_size, 4)
    assert tuple(delta_dp.shape) == (batch_size, 4)
    if target_dim == 4:
        assert not bool(debug["has_delta_dp_output"])
        assert torch.allclose(delta_dp, torch.zeros_like(delta_dp))
    wrapper.reset()
    assert torch.equal(wrapper.history_filled, torch.zeros_like(wrapper.history_filled))
    dataset.close()
    return wrapper_raw.detach().cpu()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify exported ROM inference parity.")
    parser.add_argument("--export_dir", type=Path, default=DEFAULT_EXPORT_DIR)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--atol", type=float, default=1.0e-5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    required = [
        args.export_dir / "best_rom.pth",
        args.export_dir / "best_rom_scaler.pth",
        args.export_dir / "best_rom_metadata.json",
        args.export_dir / "best_rom_config.json",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing exported ROM files: {missing}. Run export_rom.py first.")
    dataset_path = args.dataset or (DEFAULT_DATASET_PATH if DEFAULT_DATASET_PATH.exists() else find_default_dataset())
    cpu_out = compare_device(args.export_dir, dataset_path, torch.device("cpu"), batch_size=args.batch_size, atol=args.atol)
    print("CPU inference parity passed.")
    if torch.cuda.is_available():
        cuda_out = compare_device(args.export_dir, dataset_path, torch.device("cuda"), batch_size=args.batch_size, atol=max(args.atol, 1.0e-4))
        if not torch.allclose(cpu_out, cuda_out, atol=1.0e-4, rtol=1.0e-4):
            diff = torch.max(torch.abs(cpu_out - cuda_out)).item()
            raise AssertionError(f"CPU/CUDA parity failed: max abs diff {diff}")
        print("CPU/CUDA inference parity passed.")
    print("ROM inference checks passed.")


if __name__ == "__main__":
    main()
