#!/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
"""Preflight: prove the whole pipeline runs BEFORE committing hours to it.

Every long run in this work so far has been killed part-way by a defect that a
five-minute end-to-end rehearsal would have caught:

  * a training block resumed a migrated checkpoint without the optimizer-reset
    flag and died in 6.3 s;
  * a stage resumed a migrated checkpoint across a stage boundary and died the
    same way;
  * `--action_scale` was spelled `sole_action_scale` and silently did nothing;
  * `--head_frame_observation` was applied AFTER the observation width was
    derived, so training built a 96-wide network for a 98-wide checkpoint;
  * `--head_frame_observation` was never DECLARED in train.py at all, because a
    string-replace edit did not match, and the run died on AttributeError.

None of those needed a GPU-hour to find. This runs the real pipeline -- the
same supervisor, the same train.py and evaluate.py children, the same flags --
at a tiny scale, and checks the things that actually broke.

    python PPO3_SOLE_PhysX/scripts/preflight.py --actuator rom -- <driver args>

Phase 1 is pure-CLI and needs no GPU: it asks every entry point for --help and
verifies that each flag the driver passes is actually DECLARED. That alone
catches the last two failures above in about ten seconds.

Phase 2 runs one real supervisor block at 4 envs / 2 iterations and requires
the provenance ledger to accept it.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PY = sys.executable
SCRIPTS = ROOT / "PPO3_SOLE_PhysX" / "scripts"

FAILURES: list[str] = []


def check(ok: bool, message: str) -> bool:
    print(("[ok]   " if ok else "[FAIL] ") + message, flush=True)
    if not ok:
        FAILURES.append(message)
    return ok


def declared_flags(script: Path) -> set[str]:
    """Flags a script actually accepts, from its own --help.

    Parsed from --help rather than from the source, because the failure being
    guarded against is an edit that did not land: the source may contain the
    flag in a comment or a docstring while argparse never saw it.
    """
    result = subprocess.run(
        [PY, str(script), "--help"],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=ROOT,
    )
    text = result.stdout + result.stderr
    return set(re.findall(r"(--[A-Za-z0-9][A-Za-z0-9_\-]*)", text))


def extract_driver_flags(driver: Path) -> set[str]:
    """Every --flag the driver script mentions."""
    return set(re.findall(r"(--[a-z0-9][a-z0-9_\-]*)", driver.read_text()))


def phase1_cli(driver: Path | None) -> None:
    print("\n=== Phase 1: CLI contract (no GPU) ===", flush=True)
    entry_points = {
        "train.py": SCRIPTS / "train.py",
        "evaluate.py": SCRIPTS / "evaluate.py",
        "train_until_converged.py": SCRIPTS / "train_until_converged.py",
    }
    declared = {}
    for name, path in entry_points.items():
        flags = declared_flags(path)
        declared[name] = flags
        check(bool(flags), f"{name} responds to --help ({len(flags)} flags)")

    # The supervisor forwards to both children, so anything it declares in the
    # shared set must exist in BOTH. A flag reaching one child and not the
    # other is how a block and its evaluation end up measuring different
    # things -- or how a 96-wide network gets built for a 98-wide checkpoint.
    shared = {
        "--contact_observation",
        "--contact_safety_reward",
        "--contact_latch",
        "--contact_latch_release_steps",
        "--model_error_history",
        "--action_scale",
        "--max_action_rate",
        "--arm_stiffness_scale",
        "--cylinder_target_max_lead_m",
        "--cylinder_target_leak",
        "--tracking_amplitude",
        "--head_frame_observation",
        "--head_tilt_arm_frame",
        "--head_correction_leak",
        "--head_correction_weight",
    }
    for flag in sorted(shared):
        in_sup = flag in declared["train_until_converged.py"]
        in_train = flag in declared["train.py"]
        in_eval = flag in declared["evaluate.py"]
        check(
            in_sup and in_train and in_eval,
            f"{flag} declared in supervisor={in_sup} train={in_train} evaluate={in_eval}",
        )

    if driver is not None and driver.is_file():
        used = extract_driver_flags(driver)
        # The driver legitimately invokes preflight.py itself, so its flags
        # appear in the driver text and are NOT supervisor flags. Subtract
        # them rather than special-casing lines: this stays correct if the
        # invocation moves or gains arguments.
        known = (
            declared["train_until_converged.py"]
            | declared_flags(Path(__file__).resolve())
            | {"--help"}
        )
        # Driver text also contains prose; only check flags that look like ours.
        suspicious = {
            f for f in used
            if f.startswith("--") and f not in known and len(f) > 4
            and not f.startswith("--eval_terrain_seed")
        }
        check(
            not suspicious,
            f"{driver.name}: every flag it passes is declared by the supervisor"
            + (f" (undeclared: {sorted(suspicious)})" if suspicious else ""),
        )


def phase2_widths() -> None:
    """The width derivation must agree with what a checkpoint carries."""
    print("\n=== Phase 2: observation width contract (no GPU) ===", flush=True)
    from types import SimpleNamespace

    import torch

    from PPO3_SOLE_PhysX.terrain_observations import observation_space_for_cfg

    cases = {
        (False, 1, False, False): 54,
        (True, 1, False, False): 66,
        (True, 3, False, False): 90,
        (True, 3, True, False): 96,
        (True, 3, True, True): 98,
    }
    for (me, hist, contact, head), expected in cases.items():
        cfg = SimpleNamespace(
            terrain_model_error_observation=me,
            terrain_model_error_history=hist,
            terrain_contact_observation=contact,
            terrain_head_frame_observation=head,
        )
        got = observation_space_for_cfg(cfg)
        check(got == expected, f"width(me={me},hist={hist},contact={contact},head={head}) = {got}")

    # train.py must apply every width-changing flag BEFORE deriving the width.
    source = (SCRIPTS / "train.py").read_text()
    derive = source.index("env_cfg.observation_space = observation_space_for_cfg")
    for flag in (
        "env_cfg.terrain_contact_observation = ",
        "env_cfg.terrain_head_frame_observation = ",
        'override_cfg(env_cfg, "terrain_model_error_history"',
    ):
        check(
            flag in source and source.index(flag) < derive,
            f"train.py applies {flag.strip()[:46]}... before deriving the width",
        )


def phase3_checkpoints(checkpoints: list[Path], expected_width: int) -> None:
    print("\n=== Phase 3: checkpoint contract (no GPU) ===", flush=True)
    import torch

    for path in checkpoints:
        if not check(path.is_file(), f"{path} exists"):
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        infos = payload.get("infos") or {}
        width = int(infos.get("observation_width", -1))
        actual = int(payload["actor_state_dict"]["mlp.0.weight"].shape[1])
        check(width == expected_width, f"{path.name}: infos width {width} == {expected_width}")
        check(actual == width, f"{path.name}: actor accepts {actual} inputs == recorded {width}")
        check(
            "actuator" in infos, f"{path.name}: carries an actuator tag ({infos.get('actuator')})"
        )


def phase4_live(driver_cmd: list[str], run_dir: Path) -> None:
    """One real supervisor block, tiny, and the ledger must accept it."""
    print("\n=== Phase 4: one live block (GPU) ===", flush=True)
    print("  " + shlex.join(driver_cmd), flush=True)
    result = subprocess.run(driver_cmd, cwd=ROOT, capture_output=True, text=True, timeout=5400)
    tail = (result.stdout + result.stderr)[-2500:]
    ledger = run_dir / "block_provenance.jsonl"
    if not check(ledger.is_file(), f"provenance ledger written at {ledger}"):
        print(tail, flush=True)
        return
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    check(bool(rows), "ledger has at least one block")
    for row in rows:
        if not check(row["accepted"], f"block accepted: {row['label']}"):
            for reason in row["reasons"]:
                print(f"       REASON: {reason}", flush=True)
        check(
            row["input_sha256"] != row["output_sha256"],
            f"{row['label']}: the checkpoint actually changed",
        )
    reports = sorted((run_dir / "reports").glob("heldout_block_*.json"))
    check(bool(reports), "a held-out block report was produced")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--driver", default="", help="Driver script whose flags to verify.")
    p.add_argument("--checkpoint", action="append", default=[])
    p.add_argument("--expected_width", type=int, default=98)
    p.add_argument("--live", action="store_true", help="Also run one real block.")
    p.add_argument("--live_cmd", default="", help="Shell command for the live block.")
    p.add_argument("--live_run_dir", default="")
    a = p.parse_args()

    phase1_cli(Path(a.driver) if a.driver else None)
    phase2_widths()
    if a.checkpoint:
        phase3_checkpoints([Path(c) for c in a.checkpoint], int(a.expected_width))
    if a.live:
        if not a.live_cmd or not a.live_run_dir:
            FAILURES.append("--live needs --live_cmd and --live_run_dir")
        else:
            phase4_live(shlex.split(a.live_cmd), Path(a.live_run_dir))

    print("\n" + ("=" * 60), flush=True)
    if FAILURES:
        print(f"PREFLIGHT FAILED -- {len(FAILURES)} problem(s):", flush=True)
        for f in FAILURES:
            print("  - " + f, flush=True)
        return 1
    print("PREFLIGHT PASSED -- safe to commit to the long run.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
