from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

try:
    from evaluate_rom import evaluate_model, load_checkpoint_model
    from rom_data_contract import (
        AUGMENTED_INPUT_FEATURE_NAMES,
        CORE_INPUT_FEATURE_NAMES,
        FORCE_FEATURE_NAMES,
        TARGET_FEATURE_NAMES,
    )
    from rom_dataset import DEFAULT_DATASET_PATH, LOCKED_SPLIT_MANIFEST_PATH, ROMSequenceDataset, file_sha256, find_default_dataset
    from rollout_rom import compute_training_state_limits, run_rollouts
except ImportError:
    from kevin_integration.rom_cylinders.evaluate_rom import evaluate_model, load_checkpoint_model
    from kevin_integration.rom_cylinders.rom_data_contract import (
        AUGMENTED_INPUT_FEATURE_NAMES,
        CORE_INPUT_FEATURE_NAMES,
        FORCE_FEATURE_NAMES,
        TARGET_FEATURE_NAMES,
    )
    from kevin_integration.rom_cylinders.rom_dataset import (
        DEFAULT_DATASET_PATH,
        LOCKED_SPLIT_MANIFEST_PATH,
        ROMSequenceDataset,
        file_sha256,
        find_default_dataset,
    )
    from kevin_integration.rom_cylinders.rollout_rom import compute_training_state_limits, run_rollouts


ROM_DIR = Path(__file__).resolve().parent
DEFAULT_EXPORT_DIR = ROM_DIR / "exported"


def checkpoint_sha256(path: Path) -> str:
    return file_sha256(path)


def tensor_to_list(value: torch.Tensor) -> list[float]:
    return [float(item) for item in value.detach().cpu().reshape(-1).tolist()]


def target_names_for_mode(target_mode: str) -> tuple[str, ...]:
    if target_mode == "extension_only":
        return TARGET_FEATURE_NAMES[:4]
    if target_mode == "extension_and_dp":
        return TARGET_FEATURE_NAMES
    raise ValueError("target_mode must be 'extension_only' or 'extension_and_dp'.")


def export_model(
    *,
    checkpoint: Path,
    dataset_path: Path,
    output_dir: Path,
    split: str,
    split_scheme: str | None,
    locked_manifest: Path | None,
    device: torch.device,
    include_rollout: bool,
    max_rollout_runs: int,
    rollout_mode: str,
    state_limit_source: str,
    state_limit_margin_fraction: float,
    trace_torchscript: bool,
) -> dict[str, Path]:
    payload = torch.load(checkpoint, map_location="cpu")
    config = payload["config"]
    use_force = bool(config.get("use_force_input", False))
    target_mode = str(config.get("target_mode", "extension_and_dp"))
    resolved_split_scheme = split_scheme or str(config.get("split_scheme", "original"))
    resolved_locked_manifest = locked_manifest or Path(config.get("locked_manifest", LOCKED_SPLIT_MANIFEST_PATH))
    dataset = ROMSequenceDataset(
        dataset_path,
        split=split,
        use_force_input=use_force,
        split_scheme=resolved_split_scheme,
        locked_manifest=resolved_locked_manifest,
        target_mode=target_mode,
    )
    model, _ = load_checkpoint_model(checkpoint, dataset, device)
    metrics = evaluate_model(model, dataset, batch_size=512, device=device)
    rollout_metrics = None
    if include_rollout:
        state_limits = None
        extension_limit_abs = 10.0
        dp_limit_abs = 2.0e8
        if state_limit_source == "training":
            state_limits = compute_training_state_limits(
                dataset_path=dataset_path,
                split_scheme=resolved_split_scheme,
                locked_manifest=resolved_locked_manifest,
                target_mode=target_mode,
                use_force_input=use_force,
                margin_fraction=state_limit_margin_fraction,
            )
        elif state_limit_source == "none":
            extension_limit_abs = float("inf")
            dp_limit_abs = float("inf")
        rollout_dataset = ROMSequenceDataset(
            dataset_path,
            split=split,
            use_force_input=use_force,
            split_scheme=resolved_split_scheme,
            locked_manifest=resolved_locked_manifest,
            target_mode=target_mode,
            normalize_inputs=False,
            normalize_targets=False,
            return_metadata=True,
        )
        rollout_metrics = run_rollouts(
            model,
            rollout_dataset,
            device=device,
            max_runs=max_rollout_runs,
            extension_limit_abs=extension_limit_abs,
            dp_limit_abs=dp_limit_abs,
            rollout_mode=rollout_mode,
            state_limits=state_limits,
        )
        rollout_dataset.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "best_rom.pth"
    scaler_path = output_dir / "best_rom_scaler.pth"
    metadata_path = output_dir / "best_rom_metadata.json"
    config_path = output_dir / "best_rom_config.json"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "source_checkpoint": str(checkpoint),
        },
        model_path,
    )
    stats = dataset.normalization_stats
    scaler = {
        "input_mean": stats.input_mean,
        "input_std": stats.input_std,
        "target_mean": stats.target_mean,
        "target_std": stats.target_std,
        "force_mean": stats.force_mean,
        "force_std": stats.force_std,
    }
    torch.save(scaler, scaler_path)
    input_feature_names = AUGMENTED_INPUT_FEATURE_NAMES if use_force else CORE_INPUT_FEATURE_NAMES
    target_feature_names = target_names_for_mode(target_mode)
    current_run_ids = sorted({int(value) for value in dataset.run_ids.tolist()})
    metadata: dict[str, Any] = {
        "model_type": config.get("model_type"),
        "architecture": {
            "input_dim": config.get("input_dim"),
            "output_dim": config.get("output_dim", dataset.output_dim),
            "history_length": config.get("history_length", 10),
            "hidden_dim": config.get("hidden_dim"),
            "num_layers": config.get("num_layers"),
            "dropout": config.get("dropout"),
            "bounded_outputs": bool(config.get("bounded_outputs", False)),
            "output_scale": config.get("output_scale"),
        },
        "input_feature_order": input_feature_names,
        "force_feature_order": FORCE_FEATURE_NAMES,
        "target_order": target_feature_names,
        "input_dimension": config.get("input_dim"),
        "target_dimension": config.get("output_dim", dataset.output_dim),
        "target_mode": target_mode,
        "history_length": config.get("history_length", 10),
        "export_split": split,
        "timestep_statistics": {
            "dt_seconds": dataset.metadata.dt_seconds,
        },
        "normalization_statistics": {
            "input_mean": tensor_to_list(stats.input_mean),
            "input_std": tensor_to_list(stats.input_std),
            "target_mean": tensor_to_list(stats.target_mean),
            "target_std": tensor_to_list(stats.target_std),
            "force_mean": tensor_to_list(stats.force_mean) if stats.force_mean is not None else None,
            "force_std": tensor_to_list(stats.force_std) if stats.force_std is not None else None,
        },
        "split_run_ids": {
            split: [f"{split}:{run_id}" for run_id in current_run_ids],
        },
        "split_scheme": resolved_split_scheme,
        "locked_manifest": str(resolved_locked_manifest) if resolved_split_scheme == "locked" else None,
        "use_force_input": use_force,
        "per_cylinder_metrics": {
            "delta_extension": metrics["delta_extension"],
            "delta_dp": metrics.get("delta_dp", {}),
        },
        "rollout_metrics": rollout_metrics["horizons"] if rollout_metrics is not None else None,
        "rollout_mode": rollout_metrics["rollout_mode"] if rollout_metrics is not None else None,
        "state_limit_source": state_limit_source if rollout_metrics is not None else None,
        "checkpoint_hash": checkpoint_sha256(checkpoint),
        "dataset_file": str(dataset_path),
        "dataset_sha256": file_sha256(dataset_path),
        "export_note": (
            "Exported from offline validation tooling. This does not by itself mean the model is ready for Isaac "
            "integration; require held-out complete-run performance, rollout stability, baseline superiority, "
            "and inference parity."
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")

    if trace_torchscript:
        try:
            example = torch.zeros(
                (1, int(config.get("history_length", 10)), int(config["input_dim"])),
                dtype=torch.float32,
                device=device,
            )
            scripted = torch.jit.trace(model.eval(), example)
            scripted.save(str(output_dir / "best_rom_traced.pt"))
        except Exception as exc:
            metadata["torchscript_export_error"] = str(exc)
            metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
            print(f"TorchScript trace failed: {exc}")

    dataset.close()
    print(f"Wrote {model_path}")
    print(f"Wrote {scaler_path}")
    print(f"Wrote {metadata_path}")
    print(f"Wrote {config_path}")
    return {
        "model": model_path,
        "scaler": scaler_path,
        "metadata": metadata_path,
        "config": config_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the selected cylinder ROM with scaler and metadata.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_EXPORT_DIR)
    parser.add_argument("--split", choices=["val", "train", "test"], default="val")
    parser.add_argument("--split_scheme", choices=["original", "locked"], default=None)
    parser.add_argument("--locked_manifest", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--include_rollout", action="store_true")
    parser.add_argument("--max_rollout_runs", type=int, default=16)
    parser.add_argument("--rollout_mode", choices=["teacher_state", "teacher_pressure", "predicted_pressure"], default="teacher_pressure")
    parser.add_argument("--state_limit_source", choices=["training", "absolute", "none"], default="training")
    parser.add_argument("--state_limit_margin_fraction", type=float, default=0.05)
    parser.add_argument("--trace_torchscript", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset or (DEFAULT_DATASET_PATH if DEFAULT_DATASET_PATH.exists() else find_default_dataset())
    export_model(
        checkpoint=args.checkpoint,
        dataset_path=dataset_path,
        output_dir=args.output_dir,
        split=args.split,
        split_scheme=args.split_scheme,
        locked_manifest=args.locked_manifest,
        device=torch.device(args.device),
        include_rollout=args.include_rollout,
        max_rollout_runs=args.max_rollout_runs,
        rollout_mode=args.rollout_mode,
        state_limit_source=args.state_limit_source,
        state_limit_margin_fraction=args.state_limit_margin_fraction,
        trace_torchscript=args.trace_torchscript,
    )


if __name__ == "__main__":
    main()
