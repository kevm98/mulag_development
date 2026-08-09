#!/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
"""Observer ablation harness: does the plant-response observer do anything?

Presence of the observer block is not evidence of adaptation. This runs the six
checks required before any adaptation claim:

  1. observer on vs off   - "off" is the zeroed intervention, which for a widened
                            policy reproduces observer-off exactly (features and
                            their input weights both start at zero).
  2. zeroed and shuffled  - shuffling preserves the marginal feature distribution
                            but destroys the environment-to-plant correspondence,
                            separating "the policy reads these inputs" from "the
                            policy reads them informatively".
  3. input-weight norms   - per-column norm of the appended inputs against the
                            original 42, read straight from the checkpoint.
  4. factor predictability- ridge probe from observer features to the true hidden
                            factors, scored on a held-out split. This is the
                            direct test of whether the features carry plant
                            information at all.
  5. cross-trajectory     - a probe fitted on one shape scored on the others, so
                            "predicts the plant" is not confused with "memorised
                            this trajectory".
  6. severity-conditioned - performance binned by factor severity over the five
                            supported ROM-B factors.

Nothing here trains. The checkpoint is opened read-only.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isaaclab.app import AppLauncher


p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--num_envs", type=int, default=128)
p.add_argument("--seed", type=int, default=17)
p.add_argument("--eval_steps", type=int, default=120)
p.add_argument("--trajectory_shapes", default="line,arc,sine,circle,square")
p.add_argument("--rounds", type=int, default=2, help="Seed groups per shape.")
p.add_argument("--path_speed_mps", type=float, default=0.18)
p.add_argument("--residual_scale_m", type=float, default=0.002)
p.add_argument("--cylinder4_residual_scale_m", type=float, default=0.00025)
p.add_argument("--checkpoint", required=True)
p.add_argument(
    "--command_rom_checkpoint",
    default=str(
        ROOT / "ROM-cylinders_pipeline/checkpoints"
        / "plant_reachable_v1_finetune_blend25/best_rom.pt"
    ),
)
p.add_argument(
    "--trajectory_library_path",
    default=str(
        ROOT / "PPO2_ROM_PhysX/models"
        / "plant_reachable_trajectory_library_v3_circle_c7_retime.json"
    ),
)
p.add_argument("--output_dir", default="")
AppLauncher.add_app_launcher_args(p)
a, unknown = p.parse_known_args()
sys.argv = [sys.argv[0], *unknown]

CANONICAL_SHAPES = ("line", "arc", "sine", "circle", "square")
shapes = tuple(v.strip() for v in a.trajectory_shapes.split(",") if v.strip())
if any(s not in CANONICAL_SHAPES for s in shapes):
    p.error(f"shapes must be drawn from {CANONICAL_SHAPES}")
if a.rounds < 1:
    p.error("rounds must be positive")

launcher = AppLauncher(a)
simulation_app = launcher.app

import gymnasium as gym
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg
from rsl_rl.runners import OnPolicyRunner

import PPO2_ROM_B_PhysX.tasks  # noqa: F401,E402
from PPO2_ROM_B_PhysX.env_cfg import PPO2ROMBPPORunnerCfg
from PPO2_ROM_B_PhysX.runtime import (
    FACTOR_NAMES,
    PLANT_LAG_OBS_DIM,
    SUPPORTED_FACTOR_NAMES,
    run_deterministic_probe,
)
from PPO2_ROM_PhysX.policy.env_cfg import apply_residual_curriculum_stage


def ridge_r2(
    features: torch.Tensor, targets: torch.Tensor, *, train_frac: float = 0.7,
    ridge: float = 1e-3,
) -> list[float]:
    """Held-out R^2 per target column for a ridge probe features -> targets.

    R^2 <= 0 means the probe does no better than predicting the training mean,
    i.e. the features carry no usable information about that target.
    """

    n = features.shape[0]
    split = max(int(n * train_frac), 2)
    if n - split < 2:
        return [float("nan")] * targets.shape[1]
    x_tr, x_te = features[:split], features[split:]
    y_tr, y_te = targets[:split], targets[split:]
    x_mean, x_std = x_tr.mean(0, keepdim=True), x_tr.std(0, keepdim=True).clamp(min=1e-8)
    x_tr = torch.cat(((x_tr - x_mean) / x_std, torch.ones(split, 1)), dim=-1)
    x_te = torch.cat(((x_te - x_mean) / x_std, torch.ones(n - split, 1)), dim=-1)
    eye = torch.eye(x_tr.shape[1])
    weights = torch.linalg.solve(x_tr.T @ x_tr + ridge * eye, x_tr.T @ y_tr)
    pred = x_te @ weights
    ss_res = ((y_te - pred) ** 2).sum(0)
    ss_tot = ((y_te - y_tr.mean(0, keepdim=True)) ** 2).sum(0).clamp(min=1e-12)
    return (1.0 - ss_res / ss_tot).tolist()


def check_3_input_weight_norms(checkpoint: Path) -> dict[str, object]:
    """Per-column norms of the appended inputs against the original 42."""

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    out: dict[str, object] = {}
    for head in ("actor_state_dict", "critic_state_dict"):
        state = payload.get(head)
        if not state:
            continue
        for key, tensor in state.items():
            if not (key.endswith("weight") and tensor.dim() == 2):
                continue
            if tensor.shape[1] < 42 + PLANT_LAG_OBS_DIM:
                continue
            original = tensor[:, :42]
            appended = tensor[:, 42:]
            out[f"{head}.{key}"] = {
                "original_mean_column_norm": float(original.norm(dim=0).mean()),
                "appended_mean_column_norm": float(appended.norm(dim=0).mean()),
                "appended_over_original": float(
                    appended.norm(dim=0).mean() / original.norm(dim=0).mean().clamp(min=1e-12)
                ),
                "appended_is_still_zero": bool(appended.abs().max() == 0.0),
                "per_appended_column_norm": appended.norm(dim=0).tolist(),
            }
            break
    return out


def main() -> None:
    checkpoint = Path(a.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = (
        Path(a.output_dir).expanduser().resolve()
        if a.output_dir
        else ROOT / "PPO2_ROM_B_PhysX/reports" / f"observer_ablation_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    task = "RRLAB-Mulag-PPO2-ROM-B-Robust-Policy-v0"
    cfg = parse_env_cfg(task, device=a.device, num_envs=int(a.num_envs))
    cfg.seed = int(a.seed)
    apply_residual_curriculum_stage(cfg, 4)
    cfg.action_space = 4
    cfg.rom_b_plant_observer = True
    cfg.observation_space = 42 + PLANT_LAG_OBS_DIM
    cfg.command_rom_checkpoint = str(Path(a.command_rom_checkpoint).resolve())
    cfg.trajectory_library_path = str(Path(a.trajectory_library_path).resolve())
    cfg.trajectory_shapes = CANONICAL_SHAPES
    cfg.trajectory_shape = shapes[0]
    cfg.randomize_trajectory_shape = False
    cfg.procedural_trajectory_probability = 0.0
    cfg.path_speed_mps = float(a.path_speed_mps)
    cfg.residual_scale_m = float(a.residual_scale_m)
    cfg.cylinder4_residual_scale_m = float(a.cylinder4_residual_scale_m)
    cfg.command_rom_direction_guard = True
    cfg.defer_safety_terminations = True
    cfg.rom_b_mode = "random"
    cfg.rom_b_randomization_scale = 1.0
    cfg.log_dir = str(output_dir)

    agent_cfg = PPO2ROMBPPORunnerCfg()
    agent_cfg.seed = int(a.seed)
    agent_cfg.device = a.device

    report: dict[str, object] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "checkpoint": str(checkpoint),
        "supported_factors": list(SUPPORTED_FACTOR_NAMES),
        "note": (
            "Observer presence is not evidence of adaptation. Read check 1/2 "
            "together with check 4: a policy can be insensitive to the features "
            "(1 and 2 flat) simply because they carry no information (4 near 0)."
        ),
    }

    print("\n[3] INPUT-WEIGHT NORMS", flush=True)
    weights = check_3_input_weight_norms(checkpoint)
    report["check_3_input_weight_norms"] = weights
    for key, value in weights.items():
        print(
            f"  {key}: appended/original = {value['appended_over_original']:.4f}"
            f"  still_zero={value['appended_is_still_zero']}", flush=True,
        )
    if not weights:
        print("  no widened layer found - is this a 42-dim checkpoint?", flush=True)

    gym_env = gym.make(task, cfg=cfg)
    rows: list[dict[str, object]] = []
    probe_records: list[dict[str, object]] = []
    try:
        raw = gym_env.unwrapped
        env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(
            env, agent_cfg.to_dict(), log_dir=str(output_dir), device=agent_cfg.device
        )
        runner.load(
            str(checkpoint),
            load_cfg={"actor": True, "critic": True, "optimizer": False,
                      "iteration": False, "rnd": False},
        )
        policy = runner.get_inference_policy(device=raw.device)

        print("\n[1,2,5,6] INTERVENTIONS ACROSS SHAPES", flush=True)
        for mode in ("on", "zero", "shuffle"):
            raw.cfg.rom_b_plant_observer_mode = mode
            for round_index in range(int(a.rounds)):
                for shape in shapes:
                    seed = (
                        int(a.seed) + 700000
                        + round_index * len(CANONICAL_SHAPES)
                        + CANONICAL_SHAPES.index(shape)
                    )
                    row = run_deterministic_probe(
                        raw, env, policy, shape=shape, seed=seed,
                        scored_steps=int(a.eval_steps),
                    )
                    row.update({"observer_mode": mode, "round": round_index + 1})
                    rows.append(row)
                    print(
                        f"  mode={mode:7s} {shape:7s} r{round_index + 1} "
                        f"mean={float(row['mean_error_m']):.5f} "
                        f"p95={float(row['p95_error_m']):.5f} "
                        f"term={float(row['max_termination_rate']):.5f}",
                        flush=True,
                    )
                    if mode == "on":
                        features = getattr(raw, "_last_plant_features", None)
                        runtime = getattr(raw, "_command_cylinder_rom", None)
                        factors = getattr(runtime, "true_factors", None)
                        if features is not None and factors is not None:
                            probe_records.append({
                                "shape": shape,
                                "features": features.detach().cpu().tolist(),
                                "factors": factors.detach().cpu().tolist(),
                            })
    finally:
        gym_env.close()

    report["check_1_2_5_6_rows"] = rows

    def summarize(mode: str, shape: str | None = None) -> dict[str, float]:
        picked = [
            r for r in rows
            if r["observer_mode"] == mode and (shape is None or r["shape"] == shape)
        ]
        return {
            "mean_error_m": sum(float(r["mean_error_m"]) for r in picked) / len(picked),
            "p95_error_m": sum(float(r["p95_error_m"]) for r in picked) / len(picked),
            "termination_rate": sum(
                float(r["max_termination_rate"]) for r in picked
            ) / len(picked),
        }

    print("\n[1,2] OBSERVER ON vs ZEROED vs SHUFFLED (pooled)", flush=True)
    pooled = {mode: summarize(mode) for mode in ("on", "zero", "shuffle")}
    report["check_1_2_pooled"] = pooled
    for mode, stats in pooled.items():
        print(
            f"  {mode:7s} mean={stats['mean_error_m']:.5f} "
            f"p95={stats['p95_error_m']:.5f} term={stats['termination_rate']:.5f}",
            flush=True,
        )
    delta = pooled["zero"]["mean_error_m"] - pooled["on"]["mean_error_m"]
    report["check_1_observer_benefit_m"] = delta
    print(
        f"\n  observer benefit (zeroed minus on): {delta:+.6f} m"
        f"\n  -> {'the policy uses the features' if abs(delta) > 1e-4 else 'NO measurable use of the features'}",
        flush=True,
    )

    print("\n[5] PER-SHAPE (cross-trajectory)", flush=True)
    per_shape = {
        shape: {mode: summarize(mode, shape) for mode in ("on", "zero", "shuffle")}
        for shape in shapes
    }
    report["check_5_per_shape"] = per_shape
    for shape, modes in per_shape.items():
        print(
            f"  {shape:7s} on={modes['on']['mean_error_m']:.5f} "
            f"zero={modes['zero']['mean_error_m']:.5f} "
            f"shuffle={modes['shuffle']['mean_error_m']:.5f}", flush=True,
        )

    print("\n[4] FACTOR PREDICTABILITY FROM OBSERVER FEATURES (held-out R^2)", flush=True)
    supported_idx = list(range(len(FACTOR_NAMES)))
    check4: dict[str, object] = {}
    if probe_records:
        allf = torch.tensor(
            [row for rec in probe_records for row in rec["features"]],
            dtype=torch.float32,
        )
        allt = torch.tensor(
            [row for rec in probe_records for row in rec["factors"]],
            dtype=torch.float32,
        )[:, supported_idx]
        perm = torch.randperm(allf.shape[0], generator=torch.Generator().manual_seed(0))
        scores = ridge_r2(allf[perm], allt[perm])
        check4["pooled"] = dict(zip(SUPPORTED_FACTOR_NAMES, scores))
        for name, score in zip(SUPPORTED_FACTOR_NAMES, scores):
            verdict = "informative" if score > 0.1 else "no usable information"
            print(f"  {name:10s} R^2 = {score:+.4f}   {verdict}", flush=True)

        print("\n[5b] CROSS-SHAPE PROBE TRANSFER", flush=True)
        transfer: dict[str, object] = {}
        for shape in shapes:
            fit = [r for r in probe_records if r["shape"] == shape]
            other = [r for r in probe_records if r["shape"] != shape]
            if not fit or not other:
                continue
            xf = torch.tensor([r for rec in fit for r in rec["features"]], dtype=torch.float32)
            yf = torch.tensor([r for rec in fit for r in rec["factors"]], dtype=torch.float32)[:, supported_idx]
            xo = torch.tensor([r for rec in other for r in rec["features"]], dtype=torch.float32)
            yo = torch.tensor([r for rec in other for r in rec["factors"]], dtype=torch.float32)[:, supported_idx]
            k = xf.shape[0]
            merged_x = torch.cat((xf, xo)); merged_y = torch.cat((yf, yo))
            scores = ridge_r2(merged_x, merged_y, train_frac=k / merged_x.shape[0])
            transfer[shape] = dict(zip(SUPPORTED_FACTOR_NAMES, scores))
            print(
                f"  fit on {shape:7s} -> others: "
                + " ".join(f"{n}={v:+.3f}" for n, v in transfer[shape].items()),
                flush=True,
            )
        check4["cross_shape_transfer"] = transfer
    else:
        print("  no observer features captured", flush=True)
    report["check_4_factor_predictability"] = check4

    print("\n[6] SEVERITY-CONDITIONED PERFORMANCE", flush=True)
    severity_note = (
        "Severity uses the five supported factors only. Per-environment binning "
        "needs per-replica metrics, which the summary probe does not retain; run "
        "diagnose.py with --collect_diagnostics for replica-level severity."
    )
    report["check_6_note"] = severity_note
    print(f"  {severity_note}", flush=True)

    (output_dir / "observer_ablation.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(f"\n  written: {output_dir / 'observer_ablation.json'}", flush=True)
    print(
        "\nINTERPRETATION: claim adaptation only if check 4 shows R^2 clearly "
        "above 0 for at least one supported factor AND check 1/2 show 'on' better "
        "than both 'zero' and 'shuffle'. Either alone is insufficient.",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
