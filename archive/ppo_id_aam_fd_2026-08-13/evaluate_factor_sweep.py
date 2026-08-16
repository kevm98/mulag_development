#!/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
"""Run the paired Gate-1 hydraulic factor sweep and keep domain metrics separate.

This orchestration layer intentionally delegates each simulator process to the
shared gate evaluator.  Every arm/key gets its own JSON; no pooled result is
used for the Gate-1 decision.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVALUATOR = ROOT / "PPO_ROM/scripts/evaluate_gates.py"
ROM_TASK = "RRLAB-Mulag-Verge-ROM-v0"
IDFD_TASK = "RRLAB-Mulag-Verge-IDFD-v0"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rom_checkpoint", required=True)
    parser.add_argument("--idfd_checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--factor_keys", default="flow_min,flow_nominal,flow_max")
    parser.add_argument("--meshes", default="1337")
    parser.add_argument("--steps", type=int, default=960)
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--stage", type=int, default=6)
    parser.add_argument("--factor_manifest", default="")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    keys = tuple(part.strip() for part in args.factor_keys.split(",") if part.strip())
    if len(keys) < 3:
        raise SystemExit("Gate 1 requires at least three factor keys")
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    arms = (
        ("ppo_rom", ROM_TASK, Path(args.rom_checkpoint).expanduser().resolve()),
        ("ppo_id_fd", IDFD_TASK, Path(args.idfd_checkpoint).expanduser().resolve()),
    )
    records: dict[str, dict[str, dict]] = {}
    for arm, task, checkpoint in arms:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"{arm} checkpoint not found: {checkpoint}")
        records[arm] = {}
        for key in keys:
            report_path = out / arm / f"{key}.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                str(EVALUATOR),
                str(checkpoint),
                "--headless",
                "--device",
                args.device,
                "--task",
                task,
                "--meshes",
                args.meshes,
                "--steps",
                str(args.steps),
                "--num_envs",
                str(args.num_envs),
                "--seed",
                str(args.seed),
                "--stage",
                str(args.stage),
                "--factor_key",
                key,
                "--report",
                str(report_path),
            ]
            if args.factor_manifest:
                command.extend(["--factor_manifest", str(Path(args.factor_manifest).resolve())])
            print(f"[FACTOR-SWEEP] arm={arm} key={key}", flush=True)
            subprocess.run(command, cwd=str(ROOT), check=True)
            records[arm][key] = json.loads(report_path.read_text())

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "same_seed": int(args.seed),
            "same_meshes": args.meshes,
            "factor_keys": list(keys),
            "arms": [arm for arm, _, _ in arms],
            "pooled_metrics_used_for_gate": False,
        },
        "records": {
            arm: {key: {"report": str(out / arm / f"{key}.json"), "actuator": payload.get("actuator")}
                  for key, payload in by_key.items()}
            for arm, by_key in records.items()
        },
        "gate1": {
            "status": "requires_review",
            "rule": "ID-FD must degrade monotonically over the same hydraulic sweep; ROM must not",
            "pooled": False,
        },
    }
    (out / "factor_sweep_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[FACTOR-SWEEP] wrote separate arm/domain reports under {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
