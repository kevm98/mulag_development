from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from evaluate_rom import load_checkpoint_model
    from rom_data_contract import DP_SLICE, EXTENSION_SLICE, EXTENSION_VELOCITY_SLICE, TARGET_DELTA_DP_SLICE, TARGET_DELTA_EXTENSION_SLICE
    from rom_dataset import DEFAULT_DATASET_PATH, ROMSequenceDataset, find_default_dataset
    from rollout_rom import _normalize_x, run_rollouts
except ImportError:
    from kevin_integration.rom_cylinders.evaluate_rom import load_checkpoint_model
    from kevin_integration.rom_cylinders.rom_data_contract import (
        DP_SLICE,
        EXTENSION_SLICE,
        EXTENSION_VELOCITY_SLICE,
        TARGET_DELTA_DP_SLICE,
        TARGET_DELTA_EXTENSION_SLICE,
    )
    from kevin_integration.rom_cylinders.rom_dataset import DEFAULT_DATASET_PATH, ROMSequenceDataset, find_default_dataset
    from kevin_integration.rom_cylinders.rollout_rom import _normalize_x, run_rollouts


ROM_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ROM_DIR / "results"


def _fmt_tensor(tensor: torch.Tensor) -> str:
    return "[" + ", ".join(f"{float(value): .6g}" for value in tensor.detach().cpu().reshape(-1).tolist()) + "]"


def audit_sequence(
    *,
    model: torch.nn.Module,
    dataset: ROMSequenceDataset,
    indices: np.ndarray,
    device: torch.device,
    steps: int,
    rollout_mode: str,
    extension_limit_abs: float,
    dp_limit_abs: float,
) -> tuple[list[str], dict[str, Any]]:
    assert EXTENSION_SLICE == slice(8, 12), f"Extension feature indices changed: {EXTENSION_SLICE}"
    assert DP_SLICE == slice(4, 8), f"dP feature indices changed: {DP_SLICE}"
    model.eval()
    _, _, first_info = dataset[int(indices[0])]
    history_raw = first_info["raw_input"].clone().float()
    dt = float(first_info["sample_time"])
    lines: list[str] = []
    stats = {"nan_or_inf": 0, "extension_limit_violations": 0, "dp_limit_violations": 0}

    with torch.no_grad():
        for local_step, next_index in enumerate(indices[1 : steps + 1], start=1):
            _, _, next_info = dataset[int(next_index)]
            next_raw_true = next_info["raw_input"].clone().float()
            x_norm = _normalize_x(dataset, history_raw)
            pred_norm = model(x_norm.unsqueeze(0).to(device)).detach().cpu().squeeze(0)
            pred_raw = dataset.denormalize_y(pred_norm).detach().cpu()
            delta_ext = pred_raw[TARGET_DELTA_EXTENSION_SLICE]
            has_dp_output = pred_raw.shape[-1] >= 8
            delta_dp = pred_raw[TARGET_DELTA_DP_SLICE] if has_dp_output else torch.zeros(4, dtype=pred_raw.dtype)
            current_ext = history_raw[-1, EXTENSION_SLICE]
            current_dp = history_raw[-1, DP_SLICE]
            updated_ext = current_ext + delta_ext
            updated_dp = current_dp + delta_dp
            finite = bool(torch.isfinite(pred_norm).all() and torch.isfinite(pred_raw).all() and torch.isfinite(updated_ext).all() and torch.isfinite(updated_dp).all())
            if not finite:
                stats["nan_or_inf"] += 1
            ext_limit = bool(torch.max(torch.abs(updated_ext)).item() > extension_limit_abs)
            dp_limit = bool(torch.max(torch.abs(updated_dp)).item() > dp_limit_abs)
            stats["extension_limit_violations"] += int(ext_limit)
            stats["dp_limit_violations"] += int(dp_limit)

            lines.append(f"## Step {local_step}")
            lines.append("")
            lines.append(f"- run ID: `{int(first_info['run_id'])}`")
            lines.append(f"- source sample index: `{int(next_info['source_sample_index'])}`")
            lines.append(f"- current raw Extension_1..4: `{_fmt_tensor(current_ext)}`")
            lines.append(f"- predicted raw DeltaExtension_1..4: `{_fmt_tensor(delta_ext)}`")
            lines.append(f"- updated raw Extension_1..4: `{_fmt_tensor(updated_ext)}`")
            lines.append(f"- current raw dP_1..4: `{_fmt_tensor(current_dp)}`")
            lines.append(f"- predicted raw DeltaDP_1..4: `{_fmt_tensor(delta_dp)}`")
            lines.append(f"- updated raw dP_1..4: `{_fmt_tensor(updated_dp)}`")
            lines.append(f"- normalized Extension values: `{_fmt_tensor(x_norm[-1, EXTENSION_SLICE])}`")
            lines.append(f"- normalized dP values: `{_fmt_tensor(x_norm[-1, DP_SLICE])}`")
            lines.append(f"- model normalized outputs: `{_fmt_tensor(pred_norm)}`")
            lines.append(f"- denormalized outputs: `{_fmt_tensor(pred_raw)}`")
            lines.append(f"- finite-value flags: `{finite}`")
            lines.append(f"- extension limit flag: `{ext_limit}`")
            lines.append(f"- dP limit flag: `{dp_limit}`")
            lines.append("")

            if rollout_mode == "teacher_state":
                history_raw = next_raw_true
            elif rollout_mode == "teacher_pressure":
                next_history = next_raw_true.clone()
                next_history[:-1, EXTENSION_SLICE] = history_raw[1:, EXTENSION_SLICE]
                next_history[:-1, EXTENSION_VELOCITY_SLICE] = history_raw[1:, EXTENSION_VELOCITY_SLICE]
                next_history[-1, EXTENSION_SLICE] = updated_ext
                next_history[-1, EXTENSION_VELOCITY_SLICE] = delta_ext / max(dt, 1.0e-12)
                history_raw = next_history
            elif rollout_mode == "predicted_pressure":
                if not has_dp_output:
                    raise ValueError("predicted_pressure audit requires extension_and_dp checkpoint.")
                next_history = next_raw_true.clone()
                next_history[:-1, EXTENSION_SLICE] = history_raw[1:, EXTENSION_SLICE]
                next_history[:-1, DP_SLICE] = history_raw[1:, DP_SLICE]
                next_history[:-1, EXTENSION_VELOCITY_SLICE] = history_raw[1:, EXTENSION_VELOCITY_SLICE]
                next_history[-1, EXTENSION_SLICE] = updated_ext
                next_history[-1, DP_SLICE] = updated_dp
                next_history[-1, EXTENSION_VELOCITY_SLICE] = delta_ext / max(dt, 1.0e-12)
                history_raw = next_history
            else:
                raise ValueError(f"Unsupported rollout_mode {rollout_mode!r}.")
    return lines, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit raw-unit autoregressive ROM rollout transitions.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--split", choices=["val", "train", "test"], default="val")
    parser.add_argument("--split_scheme", choices=["original", "locked"], default=None)
    parser.add_argument("--locked_manifest", type=Path, default=None)
    parser.add_argument("--num_sequences", type=int, default=3)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--rollout_mode", choices=["teacher_state", "teacher_pressure", "predicted_pressure"], default="predicted_pressure")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "rollout_audit.md")
    parser.add_argument("--extension_limit_abs", type=float, default=10.0)
    parser.add_argument("--dp_limit_abs", type=float, default=2.0e8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu")
    config = payload["config"]
    dataset_path = args.dataset or (DEFAULT_DATASET_PATH if DEFAULT_DATASET_PATH.exists() else find_default_dataset())
    split_scheme = args.split_scheme or str(config.get("split_scheme", "original"))
    locked_manifest = args.locked_manifest or Path(config.get("locked_manifest", ROM_DIR / "rom_split_manifest.json"))
    dataset = ROMSequenceDataset(
        dataset_path,
        split=args.split,
        split_scheme=split_scheme,
        locked_manifest=locked_manifest,
        use_force_input=bool(config.get("use_force_input", False)),
        target_mode=str(config.get("target_mode", "extension_and_dp")),
        normalize_inputs=False,
        normalize_targets=False,
        return_metadata=True,
    )
    device = torch.device(args.device)
    model, _ = load_checkpoint_model(args.checkpoint, dataset, device)
    run_ids = dataset.run_ids
    all_lines = ["# ROM Rollout Audit", ""]
    all_lines.append(f"- Checkpoint: `{args.checkpoint}`")
    all_lines.append(f"- Split: `{args.split}`")
    all_lines.append(f"- Split scheme: `{split_scheme}`")
    all_lines.append(f"- Rollout mode: `{args.rollout_mode}`")
    all_lines.append("- Verified feature indices: `Extension=8:12`, `dP=4:8`")
    all_lines.append("- Update rule: raw `next = current + denormalized_delta`; normalization is applied only before model inference.")
    all_lines.append("")
    aggregate_stats = {"nan_or_inf": 0, "extension_limit_violations": 0, "dp_limit_violations": 0}
    for run_id in np.unique(run_ids)[: args.num_sequences]:
        indices = np.flatnonzero(run_ids == run_id)
        indices = indices[np.argsort(dataset.time_indices[indices])]
        lines, stats = audit_sequence(
            model=model,
            dataset=dataset,
            indices=indices,
            device=device,
            steps=args.steps,
            rollout_mode=args.rollout_mode,
            extension_limit_abs=args.extension_limit_abs,
            dp_limit_abs=args.dp_limit_abs,
        )
        all_lines.append(f"# Sequence run_id={int(run_id)}")
        all_lines.append("")
        all_lines.extend(lines)
        for key in aggregate_stats:
            aggregate_stats[key] += int(stats[key])

    summary = run_rollouts(
        model,
        dataset,
        device=device,
        max_runs=args.num_sequences,
        extension_limit_abs=args.extension_limit_abs,
        dp_limit_abs=args.dp_limit_abs,
        rollout_mode=args.rollout_mode,
    )
    full = summary["horizons"]["full_run"]
    if aggregate_stats["nan_or_inf"] > 0:
        verdict = "still unresolved: non-finite values appeared during audit"
    elif full["limit_violation_count"] > 0 or full["extension_trajectory_rmse_mean"] > 1.0:
        verdict = "verified model instability or distribution-shift sensitivity under this rollout mode"
    else:
        verdict = "no rollout implementation bug found in audited transitions"

    all_lines.insert(7, f"- Verdict: **{verdict}**")
    all_lines.insert(8, "")
    all_lines.append("# Summary")
    all_lines.append("")
    all_lines.append(f"- Aggregate audit stats: `{aggregate_stats}`")
    all_lines.append(f"- Rollout full-run summary over audited sequences: `{full}`")
    all_lines.append(f"- Catastrophic rollout classification: **{verdict}**")
    all_lines.append("")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(all_lines), encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Verdict: {verdict}")
    dataset.close()


if __name__ == "__main__":
    main()
