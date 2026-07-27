#!/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
"""Falsifiable five-factor AAM compatibility audit on ROM-B histories.

This is an offline screening test, not a policy-performance result. It asks the
cheapest question that must pass before adaptive PPO training: do the AAM
outputs identify individual ROM-B plants better than a constant-mean predictor?
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PPO2_ROM_B_PhysX.runtime import (  # noqa: E402
    FACTOR_NAMES,
    ROMBCommandCylinderROMRuntime,
    ROMBFactorConfig,
)
from PPO2_ROM_C_PhysX.runtime.aam_runtime import OnlineAAMRuntime  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_envs", type=int, default=512)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument(
        "--command_rom_checkpoint",
        default=str(
            ROOT / "ROM-cylinders_pipeline/checkpoints"
            / "plant_reachable_v1_finetune_blend25/best_rom.pt"
        ),
    )
    parser.add_argument(
        "--aam_checkpoint",
        default=str(ROOT / "AAM_SystemFactor/checkpoints/aam_system_factor_runtime.pt"),
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "PPO2_ROM_C_PhysX/reports/aam_rom_b_compatibility.json"),
    )
    args = parser.parse_args()
    if args.num_envs < 64 or args.steps < 120:
        parser.error("audit requires at least 64 plants and 120 control steps")

    torch.manual_seed(int(args.seed))
    num_envs = int(args.num_envs)
    control_dt = 1.0 / 60.0
    rom_path = Path(args.command_rom_checkpoint).expanduser().resolve()
    aam_path = Path(args.aam_checkpoint).expanduser().resolve()
    runtime = ROMBCommandCylinderROMRuntime(
        checkpoint=rom_path,
        num_envs=num_envs,
        device="cpu",
        control_dt_s=control_dt,
        factor_cfg=ROMBFactorConfig(mode="random", randomization_scale=1.0),
    )
    aam = OnlineAAMRuntime(
        checkpoint=aam_path,
        num_envs=num_envs,
        device="cpu",
        control_dt_s=control_dt,
    )

    generator = torch.Generator().manual_seed(int(args.seed) + 1)
    extension = runtime.model.input_mean[:4].reshape(1, 4).expand(
        num_envs, -1
    ).clone()
    extension += 0.02 * (torch.rand(num_envs, 4, generator=generator) - 0.5)
    velocity = torch.zeros(num_envs, 4)
    pressure = runtime.model.input_mean[8:12].reshape(1, 4).expand(
        num_envs, -1
    ).clone()
    piston_area = torch.tensor([0.2513, 0.2513, 0.1256, 0.1256]).reshape(1, 4)
    predictions: list[torch.Tensor] = []

    for step in range(int(args.steps)):
        phase = 2.0 * torch.pi * (torch.arange(num_envs) / num_envs + step / 90.0)
        target_delta = 0.004 * torch.stack((
            torch.sin(phase),
            torch.cos(phase),
            torch.sin(1.3 * phase),
            torch.cos(0.7 * phase),
        ), dim=1)
        target_velocity = 0.02 * torch.stack((
            torch.cos(phase),
            -torch.sin(phase),
            torch.cos(1.3 * phase),
            -torch.sin(0.7 * phase),
        ), dim=1)
        result = runtime.step(
            command=torch.cat((target_delta, target_velocity), dim=1),
            extension=extension,
            extension_velocity=velocity,
            dP=pressure,
        )
        extension = extension + result.delta_extension
        velocity = result.extension_velocity_next
        pressure = pressure + result.delta_dP
        physical, _policy, ready = aam.observe(
            valve=result.predicted_valve,
            extension=extension,
            velocity=velocity,
            dP=pressure,
            force=pressure * piston_area,
        )
        if bool(ready.all()) and step >= int(args.steps) // 2:
            predictions.append(physical.clone())

    if not predictions:
        raise RuntimeError("AAM never became ready during the compatibility audit")
    predicted = torch.stack(predictions).mean(dim=0)
    target = runtime.true_factors
    factor_rows: list[dict[str, object]] = []
    for index, name in enumerate(FACTOR_NAMES):
        residual = predicted[:, index] - target[:, index]
        rmse = residual.square().mean().sqrt()
        centered = target[:, index] - target[:, index].mean()
        mean_baseline_rmse = centered.square().mean().sqrt()
        r2 = 1.0 - residual.square().sum() / centered.square().sum().clamp_min(1.0e-12)
        correlation = torch.corrcoef(
            torch.stack((predicted[:, index], target[:, index]))
        )[0, 1]
        checks = {
            "rmse_below_constant_mean": bool(rmse < mean_baseline_rmse),
            "positive_r2": bool(r2 > 0.0),
            "positive_correlation_above_0p1": bool(correlation > 0.1),
        }
        factor_rows.append({
            "factor": name,
            "rmse": float(rmse),
            "constant_mean_rmse": float(mean_baseline_rmse),
            "correlation": float(correlation),
            "r2": float(r2),
            "prediction_mean": float(predicted[:, index].mean()),
            "prediction_std": float(predicted[:, index].std()),
            "target_mean": float(target[:, index].mean()),
            "target_std": float(target[:, index].std()),
            "checks": checks,
            "passed": all(checks.values()),
        })

    normalized_history = (aam.history - aam.input_mean) / aam.input_std
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "pre-PPO screening; not a policy performance result",
        "command_rom": str(rom_path),
        "command_rom_sha256": _sha256(rom_path),
        "aam": str(aam_path),
        "aam_sha256": _sha256(aam_path),
        "factor_order": list(FACTOR_NAMES),
        "num_envs": num_envs,
        "steps": int(args.steps),
        "seed": int(args.seed),
        "history_samples_averaged": len(predictions),
        "input_fraction_beyond_5_training_std": float(
            (normalized_history.abs() > 5.0).float().mean()
        ),
        "factors": factor_rows,
        "passed": all(bool(row["passed"]) for row in factor_rows),
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    for row in factor_rows:
        print(
            f"{row['factor']:10s} corr={row['correlation']:+.3f} "
            f"R2={row['r2']:+.3f} rmse={row['rmse']:.4f} "
            f"mean_baseline={row['constant_mean_rmse']:.4f} "
            f"{'PASS' if row['passed'] else 'FAIL'}"
        )
    print(f"overall={'PASS' if report['passed'] else 'FAIL'} output={output}")


if __name__ == "__main__":
    main()
