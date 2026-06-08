from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import torch

try:
    from evaluate_rom import evaluate_model, load_checkpoint_model
    from models import ModelConfig, build_model
    from rom_dataset import DEFAULT_DATASET_PATH, LOCKED_SPLIT_MANIFEST_PATH, ROMSequenceDataset, find_default_dataset
    from rollout_rom import compute_training_state_limits, run_rollouts
except ImportError:
    from kevin_integration.rom_cylinders.evaluate_rom import evaluate_model, load_checkpoint_model
    from kevin_integration.rom_cylinders.models import ModelConfig, build_model
    from kevin_integration.rom_cylinders.rom_dataset import (
        DEFAULT_DATASET_PATH,
        LOCKED_SPLIT_MANIFEST_PATH,
        ROMSequenceDataset,
        find_default_dataset,
    )
    from kevin_integration.rom_cylinders.rollout_rom import compute_training_state_limits, run_rollouts


ROM_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT_DIR = ROM_DIR / "checkpoints"
RESULTS_DIR = ROM_DIR / "results"


def discover_checkpoints(checkpoint_dir: Path) -> list[Path]:
    return sorted(checkpoint_dir.glob("*/best_checkpoint.pth"))


def row_from_metrics(name: str, model_type: str, use_force: bool, metrics: dict[str, Any], rollout: dict[str, Any] | None) -> dict[str, Any]:
    ext = metrics["delta_extension"]["aggregate"]
    dp = metrics.get("delta_dp", {}).get("aggregate", {})
    row = {
        "name": name,
        "model_type": model_type,
        "use_force_input": use_force,
        "split": metrics["split"],
        "split_scheme": metrics.get("split_scheme"),
        "target_mode": metrics.get("target_mode"),
        "sample_count": metrics["sample_count"],
        "run_count": metrics["run_count"],
        "delta_extension_rmse": ext["rmse"],
        "delta_extension_mae": ext["mae"],
        "delta_extension_nrmse": ext["normalized_rmse"],
        "delta_extension_r2": ext["r_squared"],
        "delta_extension_sign_accuracy": ext.get("sign_accuracy"),
        "delta_dp_rmse": dp.get("rmse"),
        "delta_dp_mae": dp.get("mae"),
        "delta_dp_nrmse": dp.get("normalized_rmse"),
        "delta_dp_r2": dp.get("r_squared"),
    }
    if rollout is not None:
        full = rollout["horizons"]["full_run"]
        one_second = rollout["horizons"]["1.0s"]
        half_second = rollout["horizons"]["0.5s"]
        row.update(
            {
                "rollout_mode": rollout["rollout_mode"],
                "rollout_0p5s_extension_rmse": half_second["extension_trajectory_rmse_mean"],
                "rollout_1s_extension_rmse": one_second["extension_trajectory_rmse_mean"],
                "rollout_full_extension_rmse": full["extension_trajectory_rmse_mean"],
                "rollout_unstable_fraction": full["fraction_unstable_rollouts"],
                "rollout_limit_violation_count": full["limit_violation_count"],
            }
        )
    return row


def write_comparison(rows: list[dict[str, Any]], output_dir: Path, *, stem: str = "model_comparison") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{stem}.csv"
    md_path = output_dir / f"{stem}.md"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# ROM Model Comparison", ""]
    if not rows:
        lines.append("No models were evaluated.")
    else:
        lines.append("| Name | Type | Target Mode | Force | Ext RMSE | Ext Sign Acc | dP RMSE | Teacher/Rollout 0.5s Ext RMSE | 1s Ext RMSE | Unstable Fraction |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for row in rows:
            lines.append(
                f"| {row['name']} | {row['model_type']} | {row.get('target_mode')} | {row['use_force_input']} | "
                f"{row.get('delta_extension_rmse')} | {row.get('delta_extension_sign_accuracy')} | "
                f"{row.get('delta_dp_rmse')} | {row.get('rollout_0p5s_extension_rmse', '')} | "
                f"{row.get('rollout_1s_extension_rmse', '')} | "
                f"{row.get('rollout_unstable_fraction', '')} |"
            )
        lines.append("")
        lines.append("Selection gates: beat zero and persistence on locked-test DeltaExtension RMSE, beat persistence on teacher-pressure rollout at 0.5 s and 1.0 s, preserve sign behavior for every cylinder, avoid NaN/Inf and state-limit violations, pass inference parity, avoid unavailable Force dependencies, and remain stable across runs. Do not select an extension+pressure model over extension-only unless pressure rollout is demonstrably stable and useful.")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare ROM baselines and trained checkpoints.")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--checkpoint_dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--split", choices=["val", "train", "test"], default="val")
    parser.add_argument("--split_scheme", choices=["original", "locked"], default=None)
    parser.add_argument("--locked_manifest", type=Path, default=LOCKED_SPLIT_MANIFEST_PATH)
    parser.add_argument("--target_mode", choices=["extension_only", "extension_and_dp"], default="extension_only")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--include_rollout", action="store_true")
    parser.add_argument("--rollout_mode", choices=["teacher_state", "teacher_pressure", "predicted_pressure"], default="teacher_pressure")
    parser.add_argument("--state_limit_source", choices=["training", "absolute", "none"], default="training")
    parser.add_argument("--state_limit_margin_fraction", type=float, default=0.05)
    parser.add_argument("--max_rollout_runs", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset or (DEFAULT_DATASET_PATH if DEFAULT_DATASET_PATH.exists() else find_default_dataset())
    device = torch.device(args.device)
    rows: list[dict[str, Any]] = []

    baseline_split_scheme = args.split_scheme or "original"
    core_dataset = ROMSequenceDataset(
        dataset_path,
        split=args.split,
        use_force_input=False,
        split_scheme=baseline_split_scheme,
        locked_manifest=args.locked_manifest,
        target_mode=args.target_mode,
    )
    for baseline in ("zero", "persistence"):
        model = build_model(
            ModelConfig(
                model_type=baseline,
                input_dim=core_dataset.input_dim,
                output_dim=core_dataset.output_dim,
                target_mode=args.target_mode,
            ),
            normalization_stats=core_dataset.normalization_stats,
        )
        metrics = evaluate_model(model, core_dataset, batch_size=args.batch_size, device=device)
        rollout = None
        if args.include_rollout:
            state_limits = None
            extension_limit_abs = 10.0
            dp_limit_abs = 2.0e8
            if args.state_limit_source == "training":
                state_limits = compute_training_state_limits(
                    dataset_path=dataset_path,
                    split_scheme=baseline_split_scheme,
                    locked_manifest=args.locked_manifest,
                    target_mode=args.target_mode,
                    use_force_input=False,
                    margin_fraction=args.state_limit_margin_fraction,
                )
            elif args.state_limit_source == "none":
                extension_limit_abs = float("inf")
                dp_limit_abs = float("inf")
            rollout_dataset = ROMSequenceDataset(
                dataset_path,
                split=args.split,
                use_force_input=False,
                split_scheme=baseline_split_scheme,
                locked_manifest=args.locked_manifest,
                target_mode=args.target_mode,
                normalize_inputs=False,
                normalize_targets=False,
                return_metadata=True,
            )
            rollout = run_rollouts(
                model,
                rollout_dataset,
                device=device,
                max_runs=args.max_rollout_runs,
                extension_limit_abs=extension_limit_abs,
                dp_limit_abs=dp_limit_abs,
                rollout_mode=args.rollout_mode,
                state_limits=state_limits,
            )
            rollout_dataset.close()
        rows.append(row_from_metrics(baseline, baseline, False, metrics, rollout))
    core_dataset.close()

    checkpoints = list(args.checkpoint) + discover_checkpoints(args.checkpoint_dir)
    seen: set[Path] = set()
    for checkpoint in checkpoints:
        checkpoint = checkpoint.resolve()
        if checkpoint in seen or not checkpoint.exists():
            continue
        seen.add(checkpoint)
        payload = torch.load(checkpoint, map_location="cpu")
        config = payload["config"]
        use_force = bool(config.get("use_force_input", False))
        target_mode = str(config.get("target_mode", "extension_and_dp"))
        split_scheme = args.split_scheme or str(config.get("split_scheme", "original"))
        locked_manifest = args.locked_manifest or Path(config.get("locked_manifest", LOCKED_SPLIT_MANIFEST_PATH))
        dataset = ROMSequenceDataset(
            dataset_path,
            split=args.split,
            use_force_input=use_force,
            split_scheme=split_scheme,
            locked_manifest=locked_manifest,
            target_mode=target_mode,
        )
        model, _ = load_checkpoint_model(checkpoint, dataset, device)
        metrics = evaluate_model(model, dataset, batch_size=args.batch_size, device=device)
        rollout = None
        if args.include_rollout:
            state_limits = None
            extension_limit_abs = 10.0
            dp_limit_abs = 2.0e8
            if args.state_limit_source == "training":
                state_limits = compute_training_state_limits(
                    dataset_path=dataset_path,
                    split_scheme=split_scheme,
                    locked_manifest=locked_manifest,
                    target_mode=target_mode,
                    use_force_input=use_force,
                    margin_fraction=args.state_limit_margin_fraction,
                )
            elif args.state_limit_source == "none":
                extension_limit_abs = float("inf")
                dp_limit_abs = float("inf")
            rollout_dataset = ROMSequenceDataset(
                dataset_path,
                split=args.split,
                use_force_input=use_force,
                split_scheme=split_scheme,
                locked_manifest=locked_manifest,
                target_mode=target_mode,
                normalize_inputs=False,
                normalize_targets=False,
                return_metadata=True,
            )
            rollout = run_rollouts(
                model,
                rollout_dataset,
                device=device,
                max_runs=args.max_rollout_runs,
                extension_limit_abs=extension_limit_abs,
                dp_limit_abs=dp_limit_abs,
                rollout_mode=args.rollout_mode,
                state_limits=state_limits,
            )
            rollout_dataset.close()
        rows.append(row_from_metrics(checkpoint.parent.name, str(config.get("model_type")), use_force, metrics, rollout))
        dataset.close()

    write_comparison(rows, args.output_dir, stem="model_comparison")
    if args.include_rollout and args.rollout_mode == "teacher_pressure":
        write_comparison(rows, args.output_dir, stem="teacher_pressure_comparison")


if __name__ == "__main__":
    main()
