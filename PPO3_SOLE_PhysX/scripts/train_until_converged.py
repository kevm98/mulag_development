#!/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
"""Persistently train PPO3 with 512 envs and gate on held-out terrain.

Training and held-out validation use separate Isaac Sim processes because the
terrain mesh is static after scene construction.  This supervisor alternates
one bounded stage-4 PPO block with one deterministic evaluation on the rougher
seed-1337 terrain family.  Only one GPU scene exists at a time, every block has
an exact resumable checkpoint, and training stops only when all shared gates
pass.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PPO3_SOLE_PhysX.block_provenance import (
    accept_training_block,
    append_ledger,
)
from PPO3_SOLE_PhysX.terrain_convergence import summarize_convergence


DEFAULT_PYTHON = Path("/home/rrlab/miniconda3/envs/env_isaaclab/bin/python")
DEFAULT_RUN_DIR = ROOT / "logs/rsl_rl/ppo3_sole_terrain/heldout_convergence"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resume",
        default="",
        help="Initial terrain PPO checkpoint. Omit only when resuming saved state.",
    )
    parser.add_argument("--run_dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--python", default=str(DEFAULT_PYTHON))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_envs", type=int, default=512)
    parser.add_argument("--eval_envs", type=int, default=128)
    parser.add_argument("--block_iterations", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=960)
    parser.add_argument(
        "--head_joint_action",
        action="store_true",
        default=True,
        help="Forward the fifth hydraulic head-residual action to training and evaluation.",
    )
    parser.add_argument(
        "--legacy_four_action",
        dest="head_joint_action",
        action="store_false",
        help="Compatibility only: supervise the retired 54x4 interface.",
    )
    parser.add_argument(
        "--head_action_to_c4_gain",
        type=float,
        default=None,
        help="Forward the q5-to-cylinder-4 mixer gain to training and evaluation.",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--learning_rate", type=float, default=1.0e-5)
    parser.add_argument("--entropy_coef", type=float, default=0.0)
    parser.add_argument(
        "--reset_optimizer_first_block",
        action="store_true",
        help="Discard optimizer moments only for the first block of a restarted run.",
    )
    parser.add_argument(
        "--reset_critic_head_first_block",
        action="store_true",
        help="Zero the critic output head only for the first block of a restarted run.",
    )
    parser.add_argument(
        "--first_block_c4_std",
        type=float,
        default=None,
        help="Optional c4 exploration std applied only to the first resumed block.",
    )
    parser.add_argument(
        "--exploration_std",
        type=float,
        default=None,
        help=(
            "Exploration std forced on every action at the start of EVERY block, "
            "not just the first. Head X-axis excursions are unrecoverable (c4 "
            "acts only in the Y-Z plane), so a std that drifts back up between "
            "blocks re-opens the failure."
        ),
    )
    parser.add_argument(
        "--c4_exploration_std",
        type=float,
        default=None,
        help=(
            "Exploration std forced on the HEAD-PITCH axis only, at the start of "
            "every block. Measured 2026-07-31: the v5 checkpoint's converged std "
            "is [0.128, 0.106, 0.146, 0.119]. Forcing all four axes to 0.04-0.05 "
            "(--exploration_std) amplifies PPO's gradient on the action mean by "
            "1/sigma and made every block regress; leaving all four at their "
            "natural 0.12 finally cut chassis sway 54.77 -> 39.22 mm and tool "
            "acceleration 111.6 -> 92.7, but flipped the head to alignment "
            "-0.8539. c4 acts only in the Y-Z plane, so a head X excursion is "
            "unrecoverable -- cap that axis alone and let c1-c3 keep the "
            "exploration that is doing the useful work."
        ),
    )
    parser.add_argument(
        "--q5_exploration_std",
        type=float,
        default=None,
        help="Exploration std forced on q5 at the start of every training block.",
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=4,
        help=(
            "Curriculum stage for both training blocks and the held-out "
            "evaluation. 4 is the gentle production verge; 5 is the rough-lawn "
            "verge. A stage-5 score is NOT comparable with a stage-4 one."
        ),
    )
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--train_timeout_s", type=int, default=7200)
    parser.add_argument("--eval_timeout_s", type=int, default=1800)
    parser.add_argument(
        "--min_blocks",
        type=int,
        default=0,
        help=(
            "Minimum resumed PPO blocks before convergence may stop training. "
            "Use this to force extensive fine-tuning of an already-passing checkpoint."
        ),
    )
    parser.add_argument(
        "--max_blocks",
        type=int,
        default=0,
        help="Optional safety cap; zero keeps training until convergence.",
    )
    parser.add_argument(
        "--actuator",
        default="rom",
        choices=("rom", "idfd"),
        help=(
            "Actuator for BOTH the training blocks and the held-out evaluation. "
            "rom = PPO3 command ROM; idfd = PPO4 ID->FD. Forwarded verbatim to "
            "train.py and evaluate.py so a block and its evaluation can never "
            "disagree about which plant they measured."
        ),
    )
    parser.add_argument("--trajectory_library_path", default="")
    parser.add_argument("--command_rom_checkpoint", default="")
    parser.add_argument(
        "--tail_focus",
        action="store_true",
        help="Oversample the measured late head-alignment failure cases in training.",
    )
    parser.add_argument(
        "--head_tail_risk",
        action="store_true",
        help="Enable the training-only CVaR head objective and failure memory.",
    )
    parser.add_argument(
        "--drive_substep_interpolation",
        default="",
        choices=("", "true", "false"),
        help="Use the same corrected truck/terrain substep dynamics in training and evaluation.",
    )
    # --- contact-safe experiment ------------------------------------------
    # Every one of these is forwarded to BOTH train.py and evaluate.py. The
    # observation flags in particular MUST reach the evaluator: they change the
    # network's input width, and a block evaluated at a different width would
    # either crash or -- with the shapes forced -- score a permuted policy.
    parser.add_argument(
        "--contact_focus",
        action="store_true",
        help="Oversample the measured contact-loss region (s_curve, late path).",
    )
    parser.add_argument(
        "--boom_focus",
        action="store_true",
        help="Stage B recovery distribution: half boom_sweep at late phases.",
    )
    parser.add_argument(
        "--contact_stage",
        default="",
        choices=("", "A", "B", "C"),
        help="Contact-loss grace curriculum: A=4 steps, B=2, C=1.",
    )
    parser.add_argument(
        "--contact_observation", default="", choices=("", "true", "false")
    )
    parser.add_argument(
        "--contact_safety_reward", default="", choices=("", "true", "false")
    )
    parser.add_argument(
        "--contact_termination",
        default="",
        choices=("", "true", "false"),
        help="Training-only. The evaluator never terminates on contact loss.",
    )
    parser.add_argument(
        "--contact_latch", default="", choices=("", "true", "false")
    )
    parser.add_argument("--contact_latch_release_steps", type=int, default=None)
    parser.add_argument("--model_error_history", type=int, default=None)
    parser.add_argument("--action_scale", type=float, default=None)
    parser.add_argument("--max_action_rate", type=float, default=None)
    parser.add_argument("--arm_stiffness_scale", type=float, default=None)
    parser.add_argument("--cylinder_target_max_lead_m", type=float, default=None)
    parser.add_argument("--cylinder_target_leak", type=float, default=None)
    parser.add_argument("--height_schedule", default="")
    parser.add_argument(
        "--strict_contact",
        action="store_true",
        help="Score the contact-safety promotion gates in every block evaluation.",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Ignore an existing supervisor state file and start from --resume.",
    )
    parser.add_argument(
        "--keep_regressed_checkpoint",
        action="store_true",
        help=(
            "Seed the next block from the block just trained even when its "
            "held-out result regressed. The default rolls back to model_best.pt "
            "so a bad block cannot compound."
        ),
    )
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.num_envs <= 0 or args.eval_envs <= 0:
        parser.error("training and evaluation environment counts must be positive")
    # Stage 4 is 960 environment steps and RSL collects 32 steps per update.
    # Fewer than 30 updates would restart before a complete production episode.
    if args.block_iterations < 30:
        parser.error("--block_iterations must be at least 30 for a full stage-4 episode")
    if args.eval_steps <= 0:
        parser.error("--eval_steps must be positive")
    if args.learning_rate <= 0.0:
        parser.error("--learning_rate must be positive")
    if args.first_block_c4_std is not None and args.first_block_c4_std <= 0.0:
        parser.error("--first_block_c4_std must be positive")
    if args.exploration_std is not None and args.exploration_std <= 0.0:
        parser.error("--exploration_std must be positive")
    if args.c4_exploration_std is not None and args.c4_exploration_std <= 0.0:
        parser.error("--c4_exploration_std must be positive")
    if args.q5_exploration_std is not None and args.q5_exploration_std <= 0.0:
        parser.error("--q5_exploration_std must be positive")
    if args.q5_exploration_std is not None and not args.head_joint_action:
        parser.error("--q5_exploration_std requires --head_joint_action")
    if args.head_action_to_c4_gain is not None and args.head_action_to_c4_gain <= 0.0:
        parser.error("--head_action_to_c4_gain must be positive")
    if args.save_interval <= 0:
        parser.error("--save_interval must be positive")
    if not 1 <= args.stage <= 5:
        parser.error("--stage must be between 1 and 5")
    if args.min_blocks < 0 or args.max_blocks < 0:
        parser.error("--min_blocks and --max_blocks cannot be negative")
    if args.max_blocks and args.max_blocks < args.min_blocks:
        parser.error("--max_blocks cannot be smaller than --min_blocks")


def training_command(
    args: argparse.Namespace,
    *,
    resume: Path,
    checkpoint_out: Path,
    block_index: int,
) -> list[str]:
    # Checkpoints do not contain RNG state. Vary the runner seed per process so
    # resuming the same optimizer state cannot replay an identical PPO block.
    block_seed = int(args.seed) + 10_000 + int(block_index)
    command = [
        str(Path(args.python).expanduser()),
        str(ROOT / "PPO3_SOLE_PhysX/scripts/train.py"),
        "--headless",
        "--actuator",
        str(args.actuator),
        "--device",
        str(args.device),
        "--num_envs",
        str(args.num_envs),
        "--seed",
        str(block_seed),
        "--stages",
        str(args.stage),
        "--stage_iterations",
        str(args.block_iterations),
        "--save_interval",
        str(args.save_interval),
        "--learning_rate",
        str(args.learning_rate),
        "--fixed_learning_rate",
        "--entropy_coef",
        str(args.entropy_coef),
        "--resume",
        str(resume),
        "--checkpoint_out",
        str(checkpoint_out),
        "--log_dir",
        str(Path(args.run_dir).expanduser() / "train_logs" / f"block_{block_index:04d}"),
    ]
    if args.trajectory_library_path:
        command.extend(["--trajectory_library_path", args.trajectory_library_path])
    if args.command_rom_checkpoint:
        command.extend(["--command_rom_checkpoint", args.command_rom_checkpoint])
    if args.drive_substep_interpolation:
        command.extend(
            ["--drive_substep_interpolation", args.drive_substep_interpolation]
        )
    if args.head_joint_action:
        command.append("--head_joint_action")
    if args.tail_focus:
        command.append("--tail_focus")
    if args.head_tail_risk:
        command.append("--head_tail_risk")
    if args.head_action_to_c4_gain is not None:
        command.extend(
            ["--head_action_to_c4_gain", str(args.head_action_to_c4_gain)]
        )
    if args.exploration_std is not None:
        command.extend(["--resume_std", str(args.exploration_std)])
    # Applied after --resume_std so the head axis wins, and on EVERY block: the
    # head can flip in any block, not just the first.
    if args.c4_exploration_std is not None:
        command.extend(["--resume_c4_std", str(args.c4_exploration_std)])
    if args.q5_exploration_std is not None:
        command.extend(["--resume_q5_std", str(args.q5_exploration_std)])
    if args.contact_focus:
        command.append("--contact_focus")
    if args.boom_focus:
        command.append("--boom_focus")
    if args.contact_stage:
        command.extend(["--contact_stage", args.contact_stage])
    if args.contact_termination:
        command.extend(["--contact_termination", args.contact_termination])
    if args.height_schedule:
        command.extend(["--height_schedule", args.height_schedule])
    command.extend(_shared_contact_options(args))
    if int(block_index) == 1:
        if args.reset_optimizer_first_block:
            command.append("--reset_optimizer_on_resume")
        if args.reset_critic_head_first_block:
            command.append("--reset_critic_head_on_resume")
        if args.first_block_c4_std is not None:
            command.extend(["--resume_c4_std", str(args.first_block_c4_std)])
    elif checkpoint_requires_optimizer_reset(resume):
        # Not only block 1. When a block regresses, the supervisor rolls back
        # to model_best.pt -- and if the best checkpoint is still the migrated
        # seed, the NEXT block resumes a migrated checkpoint without the flag
        # and train.py refuses it outright ("migrated shared-policy checkpoints
        # require --reset_optimizer_on_resume"). Measured: block 1 regressed,
        # block 2 rolled back to the seed and died in 6.3 seconds.
        #
        # The requirement is a property of the CHECKPOINT, so read it from the
        # checkpoint rather than from the block index.
        command.append("--reset_optimizer_on_resume")
        print(
            f"[PPO3-SUPERVISOR] resume checkpoint {resume} is a migration; "
            "adding --reset_optimizer_on_resume",
            flush=True,
        )
    return command


def checkpoint_requires_optimizer_reset(path: Path) -> bool:
    """Does this checkpoint carry the migration's optimizer-reset requirement?"""
    try:
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception:  # noqa: BLE001 - a bad checkpoint fails later, loudly
        return False
    return bool(
        (payload.get("infos") or {}).get("migration_requires_optimizer_reset", False)
    )


def _shared_contact_options(args: argparse.Namespace) -> list[str]:
    """Options that MUST be identical in training and in evaluation.

    Two of them (`--contact_observation`, `--model_error_history`) change the
    observation width, and the rest change the plant. Forwarding them to only
    one child is how a block gets evaluated against a network shape or a plant
    it was not trained on -- the same class of mistake the supervisor's
    existing `--actuator` forwarding test guards against.
    """
    options: list[str] = []
    if args.contact_observation:
        options.extend(["--contact_observation", args.contact_observation])
    if args.contact_safety_reward:
        options.extend(["--contact_safety_reward", args.contact_safety_reward])
    if args.contact_latch:
        options.extend(["--contact_latch", args.contact_latch])
    if args.contact_latch_release_steps is not None:
        options.extend(
            ["--contact_latch_release_steps", str(args.contact_latch_release_steps)]
        )
    if args.model_error_history is not None:
        options.extend(["--model_error_history", str(args.model_error_history)])
    if args.action_scale is not None:
        options.extend(["--action_scale", str(args.action_scale)])
    if args.max_action_rate is not None:
        options.extend(["--max_action_rate", str(args.max_action_rate)])
    if args.arm_stiffness_scale is not None:
        options.extend(["--arm_stiffness_scale", str(args.arm_stiffness_scale)])
    if args.cylinder_target_max_lead_m is not None:
        options.extend(
            ["--cylinder_target_max_lead_m", str(args.cylinder_target_max_lead_m)]
        )
    if args.cylinder_target_leak is not None:
        options.extend(["--cylinder_target_leak", str(args.cylinder_target_leak)])
    return options


def evaluation_command(
    args: argparse.Namespace, *, checkpoint: Path, report: Path
) -> list[str]:
    command = [
        str(Path(args.python).expanduser()),
        str(ROOT / "PPO3_SOLE_PhysX/scripts/evaluate.py"),
        str(checkpoint),
        "--headless",
        "--actuator",
        str(args.actuator),
        "--device",
        str(args.device),
        "--num_envs",
        str(args.eval_envs),
        "--seed",
        str(args.seed),
        "--steps",
        str(args.eval_steps),
        "--stage",
        str(args.stage),
        "--eval_terrain",
        "--report",
        str(report),
    ]
    if args.trajectory_library_path:
        command.extend(["--trajectory_library_path", args.trajectory_library_path])
    if args.command_rom_checkpoint:
        command.extend(["--command_rom_checkpoint", args.command_rom_checkpoint])
    if args.drive_substep_interpolation:
        command.extend(
            ["--drive_substep_interpolation", args.drive_substep_interpolation]
        )
    if args.head_joint_action:
        command.append("--head_joint_action")
    if args.head_action_to_c4_gain is not None:
        command.extend(
            ["--head_action_to_c4_gain", str(args.head_action_to_c4_gain)]
        )
    if args.strict_contact:
        command.append("--strict_contact")
    command.extend(_shared_contact_options(args))
    return command


def _display_command(command: Sequence[str]) -> str:
    """Readable command without shell evaluation or fragile quoting."""
    return json.dumps(list(command))


def run_child(
    command: Sequence[str],
    *,
    timeout_s: int,
    label: str,
    produces: Path | None = None,
    capture_to: Path | None = None,
) -> subprocess.CompletedProcess | None:
    """Run one child. ``capture_to`` tees its output so it can be parsed.

    Returns the completed process so a caller doing full block acceptance can
    read the return code without re-deriving it.
    """
    print(f"[PPO3-SUPERVISOR] starting {label}: {_display_command(command)}", flush=True)
    started = datetime.now()
    try:
        if capture_to is not None:
            # Isaac writes a great deal; stream it to disk and mirror it so the
            # supervisor's own log stays readable and the child's log can be
            # parsed for the acceptance criteria afterwards.
            capture_to.parent.mkdir(parents=True, exist_ok=True)
            with open(capture_to, "w", encoding="utf-8") as sink:
                completed = subprocess.run(
                    list(command),
                    cwd=ROOT,
                    timeout=int(timeout_s),
                    check=False,
                    stdout=sink,
                    stderr=subprocess.STDOUT,
                )
        else:
            completed = subprocess.run(
                list(command),
                cwd=ROOT,
                timeout=int(timeout_s),
                check=False,
            )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{label} exceeded {timeout_s} seconds") from exc
    elapsed = (datetime.now() - started).total_seconds()
    if completed.returncode != 0:
        raise RuntimeError(f"{label} exited with code {completed.returncode}")
    if produces is not None:
        # The return code is NOT sufficient here. Isaac's SimulationApp shuts
        # the process down through omni.kit, and a Python exception raised
        # inside main() can still exit 0 -- measured: training block 2 died on
        # a RuntimeError, printed the traceback, exited 0, and the supervisor
        # cheerfully evaluated the STALE model_latest.pt left by block 1,
        # reporting block 1's numbers as block 2's.
        #
        # So verify the artifact instead: the file the child was told to write
        # must exist and must have been written during this run.
        if not produces.is_file():
            raise RuntimeError(
                f"{label} exited 0 but did not write {produces}; the child "
                "almost certainly failed (Isaac can mask a non-zero exit)"
            )
        written = datetime.fromtimestamp(produces.stat().st_mtime)
        # Two seconds of slack. Filesystem mtime resolution is coarse and the
        # child may finish inside the same second it started, so a strict
        # comparison rejects a perfectly good fast run. The artifact this is
        # guarding against is a PREVIOUS block's, which is minutes or hours
        # old, so the slack costs nothing.
        if (started - written).total_seconds() > 2.0:
            raise RuntimeError(
                f"{label} exited 0 but {produces} was last written at "
                f"{written.isoformat(timespec='seconds')}, before this run "
                f"started at {started.isoformat(timespec='seconds')}. Refusing "
                "to continue from a stale artifact."
            )
    print(f"[PPO3-SUPERVISOR] finished {label} in {elapsed:.1f}s", flush=True)
    return completed


def atomic_write_json(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(row, indent=2, sort_keys=True))
    temporary.replace(path)


def load_validation_report(path: Path, *, expected_steps: int | None = None) -> dict:
    if not path.is_file():
        raise RuntimeError(f"held-out evaluator did not write {path}")
    report = json.loads(path.read_text())
    provenance = report.get("terrain_provenance") or {}
    if provenance.get("split") != "held_out_validation":
        raise RuntimeError(f"refusing non-held-out convergence report: {provenance}")
    if provenance.get("curriculum_version") != "path_height_v2":
        raise RuntimeError(f"refusing wrong terrain curriculum: {provenance}")
    if expected_steps is not None and int(provenance.get("steps", -1)) != int(
        expected_steps
    ):
        raise RuntimeError(
            f"held-out report has {provenance.get('steps')} steps; "
            f"expected {expected_steps}"
        )
    if "verge" not in report:
        raise RuntimeError(f"held-out report has no verge metrics: {path}")
    summary = summarize_convergence(report["verge"])
    # Recompute centrally even if the evaluator already supplied these fields.
    report["gates"] = summary["gates"]
    report["normalized_gate_violations"] = summary[
        "normalized_gate_violations"
    ]
    report["score"] = summary["score"]
    report["passed"] = summary["passed"]
    atomic_write_json(path, report)
    return report


def copy_checkpoint(source: Path, destination: Path) -> None:
    source = source.resolve()
    destination = destination.resolve()
    if source == destination:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def candidate_is_better(candidate: dict, *, best_score: float, best_gates: dict) -> bool:
    """Promote only monotone gate progress, then use score as the tie-breaker.

    The normalized score is dominated by the 1% ground-strike denominator. A
    checkpoint can therefore lower score while losing a gate that was already
    passed (block 3 did exactly this to safety terminations). Keep every passed
    gate sticky; among gate-preserving candidates prefer more passed gates,
    then the lower aggregate violation score.
    """
    gates = dict(candidate.get("gates") or {})
    if any(bool(passed) and not bool(gates.get(name, False)) for name, passed in best_gates.items()):
        return False
    passed_count = sum(bool(value) for value in gates.values())
    best_passed_count = sum(bool(value) for value in best_gates.values())
    return passed_count > best_passed_count or (
        passed_count == best_passed_count
        and float(candidate["score"]) < float(best_score)
    )


def evaluate_candidate(
    args: argparse.Namespace,
    *,
    checkpoint: Path,
    report_path: Path,
    label: str,
) -> dict:
    run_child(
        evaluation_command(args, checkpoint=checkpoint, report=report_path),
        timeout_s=args.eval_timeout_s,
        label=label,
        produces=report_path,
    )
    report = load_validation_report(
        report_path, expected_steps=int(args.eval_steps)
    )
    print(
        "[PPO3-HELDOUT] "
        + json.dumps(
            {
                "checkpoint": str(checkpoint),
                "score": report["score"],
                "passed": report["passed"],
                "metrics": report["verge"],
                "gates": report["gates"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)

    run_dir = Path(args.run_dir).expanduser().resolve()
    reports_dir = run_dir / "reports"
    state_path = run_dir / "state.json"
    latest_checkpoint = run_dir / "model_latest.pt"
    best_checkpoint = run_dir / "model_best.pt"
    converged_checkpoint = run_dir / "model_converged.pt"
    best_report_path = run_dir / "convergence_best.json"
    converged_report_path = run_dir / "convergence_heldout.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if state_path.is_file() and not args.restart:
        state = json.loads(state_path.read_text())
        # Prefer the recorded resume point so a restart honours a rollback.
        # States written before `resume_checkpoint` existed only have
        # `latest_checkpoint`, which was the resume point under the old rules.
        current_checkpoint = Path(
            state.get("resume_checkpoint") or state["latest_checkpoint"]
        ).resolve()
        completed_blocks = int(state.get("completed_blocks", 0))
        best_score = float(state.get("best_score", float("inf")))
        best_gates = dict(state.get("best_gates") or {})
        if not best_gates:
            saved_best_report = Path(state["best_report"]).resolve()
            if saved_best_report.is_file():
                best_gates = dict(
                    json.loads(saved_best_report.read_text()).get("gates") or {}
                )
        if not current_checkpoint.is_file():
            raise RuntimeError(
                f"saved supervisor checkpoint is missing: {current_checkpoint}"
            )
        print(
            f"[PPO3-SUPERVISOR] resuming state after {completed_blocks} blocks "
            f"from {current_checkpoint}",
            flush=True,
        )
    else:
        if not args.resume:
            parser.error("--resume is required when no saved supervisor state is used")
        current_checkpoint = Path(args.resume).expanduser().resolve()
        if not current_checkpoint.is_file():
            parser.error(f"resume checkpoint does not exist: {current_checkpoint}")
        completed_blocks = 0
        best_score = float("inf")

        initial_report_path = reports_dir / "heldout_initial.json"
        initial = evaluate_candidate(
            args,
            checkpoint=current_checkpoint,
            report_path=initial_report_path,
            label="initial held-out validation",
        )
        best_score = float(initial["score"])
        best_gates = dict(initial["gates"])
        copy_checkpoint(current_checkpoint, best_checkpoint)
        shutil.copy2(initial_report_path, best_report_path)
        if initial["passed"] and int(args.min_blocks) == 0:
            copy_checkpoint(current_checkpoint, converged_checkpoint)
            shutil.copy2(initial_report_path, converged_report_path)
            print(
                f"[PPO3-SUPERVISOR] initial checkpoint already PASSED; wrote "
                f"{converged_checkpoint}",
                flush=True,
            )
            return 0

        state = {
            "status": (
                "minimum_training" if initial["passed"] else "training"
            ),
            "latest_checkpoint": str(current_checkpoint),
            "best_checkpoint": str(best_checkpoint),
            "best_report": str(best_report_path),
            "best_score": best_score,
            "best_gates": best_gates,
            "completed_blocks": completed_blocks,
            "training_num_envs": int(args.num_envs),
            "heldout_num_envs": int(args.eval_envs),
            "heldout_steps": int(args.eval_steps),
            "block_iterations": int(args.block_iterations),
            "minimum_blocks": int(args.min_blocks),
        }
        atomic_write_json(state_path, state)

    while args.max_blocks == 0 or completed_blocks < args.max_blocks:
        block_index = completed_blocks + 1
        train_cmd = training_command(
            args,
            resume=current_checkpoint,
            checkpoint_out=latest_checkpoint,
            block_index=block_index,
        )
        label = f"{args.num_envs}-env training block {block_index}"
        child_log = run_dir / "child_logs" / f"train_block_{block_index:04d}.log"
        block_started = datetime.now()
        return_code = 0
        try:
            completed = run_child(
                train_cmd,
                timeout_s=args.train_timeout_s,
                label=label,
                capture_to=child_log,
            )
            return_code = int(completed.returncode) if completed else 0
        except RuntimeError as exc:
            # Do not abandon the run here. A rejected block is a RESULT, and
            # the ledger has to record it with a reason -- silently dying
            # leaves no evidence of what was rejected or why.
            print(f"[PPO3-SUPERVISOR] {label} raised: {exc}", flush=True)
            return_code = -1

        # Seven criteria, not one return code. Isaac can exit 0 after an
        # internal failure; the SHA256 comparison is what actually catches a
        # child that died before saving and left the previous block's file in
        # place. See PPO3_SOLE_PhysX/block_provenance.py.
        acceptance = accept_training_block(
            label=label,
            input_checkpoint=current_checkpoint,
            output_checkpoint=latest_checkpoint,
            started=block_started,
            finished=datetime.now(),
            return_code=return_code,
            iterations_requested=int(args.block_iterations),
            log_path=child_log,
        )
        append_ledger(run_dir / "block_provenance.jsonl", acceptance.as_row())
        print(
            "[PPO3-PROVENANCE] " + json.dumps(acceptance.as_row(), sort_keys=True),
            flush=True,
        )
        if not acceptance.accepted:
            raise RuntimeError(
                f"{label} REJECTED: {'; '.join(acceptance.reasons)}. Refusing "
                "to evaluate an unverified checkpoint."
            )
        current_checkpoint = latest_checkpoint

        report_path = reports_dir / f"heldout_block_{block_index:04d}.json"
        report = evaluate_candidate(
            args,
            checkpoint=current_checkpoint,
            report_path=report_path,
            label=f"held-out validation after block {block_index}",
        )
        completed_blocks = block_index
        score = float(report["score"])
        if candidate_is_better(
            report, best_score=best_score, best_gates=best_gates
        ):
            best_score = score
            best_gates = dict(report["gates"])
            copy_checkpoint(current_checkpoint, best_checkpoint)
            shutil.copy2(report_path, best_report_path)
            print(
                f"[PPO3-SUPERVISOR] new best score={best_score:.6f}; "
                f"wrote {best_checkpoint}",
                flush=True,
            )
        elif not args.keep_regressed_checkpoint:
            # A block that lost ground must not seed the next one. Every block
            # used to resume from model_latest.pt unconditionally, so the run
            # walked away from its own best policy with no path back: stage 5
            # block 1 regressed 0.673 -> 1.316 and block 2 would have continued
            # from the worse weights. Rolling back keeps the search anchored to
            # the best held-out checkpoint; a fresh seed and the reshuffled
            # terrain make the retry a genuinely different block, not a replay.
            current_checkpoint = best_checkpoint
            print(
                f"[PPO3-SUPERVISOR] block {block_index} regressed "
                f"(score={score:.6f} vs best={best_score:.6f}); "
                f"next block resumes from {best_checkpoint}",
                flush=True,
            )

        minimum_complete = completed_blocks >= int(args.min_blocks)
        state = {
            "status": (
                "converged"
                if report["passed"] and minimum_complete
                else "minimum_training"
                if report["passed"]
                else "training"
            ),
            # The checkpoint this block produced and `latest_score` scored, which
            # is not necessarily what the next block resumes from -- see the
            # rollback above and `resume_checkpoint`.
            "latest_checkpoint": str(latest_checkpoint),
            "latest_report": str(report_path),
            "latest_score": score,
            "resume_checkpoint": str(current_checkpoint),
            "best_checkpoint": str(best_checkpoint),
            "best_report": str(best_report_path),
            "best_score": best_score,
            "best_gates": best_gates,
            "completed_blocks": completed_blocks,
            "training_num_envs": int(args.num_envs),
            "heldout_num_envs": int(args.eval_envs),
            "heldout_steps": int(args.eval_steps),
            "block_iterations": int(args.block_iterations),
            "minimum_blocks": int(args.min_blocks),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        atomic_write_json(state_path, state)

        if report["passed"] and minimum_complete:
            copy_checkpoint(current_checkpoint, converged_checkpoint)
            shutil.copy2(report_path, converged_report_path)
            print(
                f"[PPO3-SUPERVISOR] HELD-OUT GATES PASSED after "
                f"{completed_blocks} blocks; wrote {converged_checkpoint}",
                flush=True,
            )
            return 0
        if report["passed"]:
            print(
                f"[PPO3-SUPERVISOR] gates pass, but continuing: completed "
                f"{completed_blocks}/{int(args.min_blocks)} required blocks",
                flush=True,
            )

    print(
        f"[PPO3-SUPERVISOR] stopped at --max_blocks={args.max_blocks} without "
        "convergence; state is resumable",
        flush=True,
    )
    state["status"] = "stopped_max_blocks"
    state["updated_at"] = datetime.now().isoformat(timespec="seconds")
    atomic_write_json(state_path, state)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
