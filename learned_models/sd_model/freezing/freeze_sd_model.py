#!/usr/bin/env python
"""Freeze the selected privileged SD_z representation (AAM Stage 23).

Stage 23 spec: freeze the selected dz2 ``encoder.pt``, ``sd.pt`` and
``belief.pt`` together with the factor/state/input normalization, the input
mask and the rollout protocol. Record hashes and provenance. The frozen copy
must NOT be fine-tuned while the first AAM is trained, and it must reproduce
the Stage 22 rollout metrics (verified by ``verify_frozen_sd_model.py``).

The frozen directory is a self-contained copy of the model files plus an
explicit ``protocol.json`` (normalization stats, input mask, rollout
contract), a SHA-256 ``checksums.sha256`` manifest, the Stage 22 gate
evidence under ``evidence/`` and full provenance under ``metadata/``. The
source model directory is left untouched.

Usage:
    python freeze_sd_model.py \\
        --src outputs/sd_z_stage22_v6_wide/dz2_nodp_holdq \\
        --out ../learned_models/sd_model/stage23_frozen/dz2_nodp_holdq

With ``--readonly`` the frozen files are chmod'ed read-only as a best-effort
immutability marker (same convention as ``freeze_dataset.py``).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
AAM_ROOT = REPO / "AAM_module"  # SD tools live in learned_models/sd_model/; aam_data and datasets stay under AAM_module
if str(AAM_ROOT) not in sys.path:
    sys.path.insert(0, str(AAM_ROOT))

from aam_data import sha256_file  # noqa: E402


FROZEN_VERSION = "stage23_v1"
MODEL_FILES = ("encoder.pt", "sd.pt", "belief.pt", "metrics.json")
EVIDENCE_FILES = (
    "rollout_gate.json",
    "gate_verdict.json",
    "true_vs_none_h20.json",
)
NONE_ROLLOUT_NAME = "none_rollout_hold-start.json"
CHECKSUM_FILE = "checksums.sha256"
PROTOCOL_FILE = "protocol.json"
PROVENANCE_FILE = "provenance.json"
FACTOR_KEYS = ("flow", "compliance", "seal", "pressure", "mass_scale", "com_z")


def git_info() -> dict:
    out = {"commit": None, "branch": None, "describe": None, "dirty": None}
    try:
        res = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if res.returncode == 0:
            out["commit"] = res.stdout.strip()
        res = subprocess.run(
            ["git", "-C", str(REPO), "branch", "--show-current"],
            capture_output=True, text=True, timeout=5,
        )
        if res.returncode == 0:
            out["branch"] = res.stdout.strip() or None
        res = subprocess.run(
            ["git", "-C", str(REPO), "describe", "--always", "--dirty"],
            capture_output=True, text=True, timeout=5,
        )
        if res.returncode == 0:
            out["describe"] = res.stdout.strip()
    except Exception:
        pass
    return out


def environment() -> dict:
    import platform

    import numpy
    import torch

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
    }


def protocol_from_metrics(metrics: dict, *, seed: int | None) -> dict:
    """Explicit, self-describing freeze contract built from the trainer report.

    ``metrics.json`` is written by ``train_sd_z.py`` and already contains the
    input mask, belief/q protocol, loss weights and the train-only
    normalization statistics; this copies the fields the gate and later AAM
    stages need into a stable contract that does not depend on trainer
    internals.
    """
    return {
        "frozen_version": FROZEN_VERSION,
        "stage": metrics.get("stage"),
        "latent_source": metrics.get("latent_source"),
        "latent_dim": metrics.get("latent_dim"),
        "latent_in_dim": metrics.get("latent_in_dim"),
        "factor_keys": list(FACTOR_KEYS),
        "obs_drop": metrics.get("obs_drop"),
        "state_keep": metrics.get("state_keep"),
        "state_dim": metrics.get("state_dim"),
        "use_q": metrics.get("use_q"),
        "use_belief": metrics.get("use_belief"),
        "belief_dim": metrics.get("belief_dim"),
        "history_lag": metrics.get("history_lag"),
        "state_loss_weight": metrics.get("state_loss_weight"),
        "rollout_q_mode": metrics.get("rollout_q_mode"),
        "unroll": metrics.get("unroll"),
        "l1_weight": metrics.get("l1_weight"),
        "rollout_weight": metrics.get("rollout_weight"),
        "rollout_ramp_epochs": metrics.get("rollout_ramp_epochs"),
        "val_unroll": metrics.get("val_unroll"),
        "tf_floor": metrics.get("tf_floor"),
        "noise_std": metrics.get("noise_std"),
        "seed": seed,
        "best_epoch": metrics.get("best_epoch"),
        "e_normalization": metrics.get("e_normalization"),
        "s_stats": metrics.get("s_stats"),
        "u_stats": metrics.get("u_stats"),
        "q_stats": metrics.get("q_stats"),
        "y_stats": metrics.get("y_stats"),
        "test": metrics.get("test"),
        "val": {k: v for k, v in (metrics.get("val") or {}).items()
                if k in ("rmse", "loss")},
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--src",
        default=str(REPO / "outputs/sd_z_stage22_v6_wide/dz2_nodp_holdq"),
        help="selected Stage 22 dz2 model dir (source of truth)",
    )
    p.add_argument(
        "--out",
        default=str(REPO / "learned_models/sd_model/stage23_frozen/dz2_nodp_holdq"),
        help="frozen model directory to create",
    )
    p.add_argument(
        "--dataset-root",
        default=str(AAM_ROOT / "datasets/master_v3_stage22_wide_frozen"),
        help="frozen dataset the model was trained/evaluated on",
    )
    p.add_argument(
        "--none-rollout",
        default=str(REPO / "outputs/sd_z_stage22_v6_wide/none_nodp_holdq/rollout_hold-start.json"),
        help="protocol-matched no-latent control rollout (Stage 22 evidence)",
    )
    p.add_argument("--seed", type=int, default=20260813,
                   help="training seed used by the v6 pipeline")
    p.add_argument("--readonly", action="store_true",
                   help="chmod frozen files read-only (best-effort freeze)")
    p.add_argument("--force", action="store_true",
                   help="overwrite an existing --out")
    args = p.parse_args()

    src = Path(args.src).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    if out == src or src in out.parents:
        raise SystemExit(f"refusing to freeze into the source tree: {out}")
    missing = [f for f in MODEL_FILES if not (src / f).is_file()]
    if missing:
        raise SystemExit(f"source model dir incomplete {src}: missing {missing}")
    evidence_src = [src / f for f in EVIDENCE_FILES]
    missing_ev = [f for f in evidence_src if not f.is_file()]
    if missing_ev:
        raise SystemExit(
            f"source evidence incomplete {src}: missing {missing_ev}"
        )
    none_rollout = Path(args.none_rollout).expanduser().resolve()
    if not none_rollout.is_file():
        raise SystemExit(f"no-latent control rollout not found: {none_rollout}")
    if out.exists() and any(out.iterdir()) and not args.force:
        raise SystemExit(
            f"target exists and is not empty: {out} (use --force to overwrite)"
        )

    metrics = json.loads((src / "metrics.json").read_text())
    protocol = protocol_from_metrics(metrics, seed=args.seed)

    if out.exists():
        shutil.rmtree(out)
    (out / "evidence").mkdir(parents=True)
    (out / "metadata").mkdir(parents=True)

    for name in MODEL_FILES:
        shutil.copyfile(src / name, out / name)
    for name in EVIDENCE_FILES:
        shutil.copyfile(src / name, out / "evidence" / name)
    shutil.copyfile(none_rollout, out / "evidence" / NONE_ROLLOUT_NAME)
    (out / PROTOCOL_FILE).write_text(json.dumps(protocol, indent=2) + "\n")

    source_hashes = {
        name: sha256_file(src / name) for name in MODEL_FILES
    }
    source_hashes.update(
        {f"evidence/{name}": sha256_file(src / name) for name in EVIDENCE_FILES}
    )
    source_hashes[f"evidence/{NONE_ROLLOUT_NAME}"] = sha256_file(none_rollout)

    dataset_provenance = None
    dp = Path(args.dataset_root).expanduser() / "metadata" / PROVENANCE_FILE
    if dp.is_file():
        dataset_provenance = json.loads(dp.read_text())

    provenance = {
        "frozen_version": FROZEN_VERSION,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "do_not_finetune": True,
        "source": str(src),
        "source_hashes_sha256": source_hashes,
        "dataset_root": str(Path(args.dataset_root).expanduser().resolve()),
        "dataset_provenance": dataset_provenance,
        "command": " ".join(sys.argv),
        "readonly": bool(args.readonly),
        "git": git_info(),
        "environment": environment(),
    }
    (out / "metadata" / PROVENANCE_FILE).write_text(
        json.dumps(provenance, indent=2) + "\n"
    )

    # SHA-256 over every frozen file, including the provenance record itself
    # (the manifest is the only file not covered by its own checksums).
    frozen_files = (
        [out / name for name in MODEL_FILES]
        + [out / "evidence" / name for name in EVIDENCE_FILES]
        + [out / "evidence" / NONE_ROLLOUT_NAME]
        + [out / PROTOCOL_FILE]
        + [out / "metadata" / PROVENANCE_FILE]
    )
    checksum_lines = sorted(
        f"{sha256_file(path)}  {path.relative_to(out)}" for path in frozen_files
    )
    (out / CHECKSUM_FILE).write_text("\n".join(checksum_lines) + "\n")

    if args.readonly:
        for path in sorted(out.rglob("*")):
            if path.is_file():
                path.chmod(0o444)
            else:
                path.chmod(0o555)

    print(f"frozen {len(frozen_files)} files -> {out}")
    print(f"  latent: {protocol['latent_source']} dz={protocol['latent_dim']} "
          f"mask={protocol['obs_drop']} q={protocol['use_q']} "
          f"belief={protocol['use_belief']} lag={protocol['history_lag']} "
          f"q_mode={protocol['rollout_q_mode']}")
    print(f"  checksums: {out / CHECKSUM_FILE}")
    print(f"  protocol: {out / PROTOCOL_FILE}")
    print(f"  provenance: {out / 'metadata' / PROVENANCE_FILE}")
    print(f"  readonly: {args.readonly}")


if __name__ == "__main__":
    main()
