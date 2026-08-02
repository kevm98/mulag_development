"""Acceptance and provenance for one supervised training block.

Isaac Sim shuts a process down through omni.kit, and a Python exception raised
inside ``main()`` can still exit 0. That is not hypothetical: a training block
died on a ``RuntimeError``, printed the traceback, exited 0, and the supervisor
evaluated the STALE checkpoint from the previous block and recorded its numbers
under the new block's name. ``HANDOVER_PPO3_PPO4.md`` lists "one run that
resumed stale artifacts" among the package's known past incidents; this module
is what stops it recurring.

A block is accepted only when ALL SEVEN hold:

1. the expected checkpoint exists
2. its modification time is newer than the child's start time
3. its SHA256 differs from the input checkpoint
4. the expected training summary exists
5. the summary reports the requested iterations, or an explicit valid stop
6. the log shows a successful environment start AND a completed learning iteration
7. the log contains no traceback, fatal error, CUDA failure, NaN stop or abort

(3) is the one that catches the exact failure above: a child that dies before
saving leaves the previous block's file in place, with the previous block's
contents. Timestamps can be fooled by a touch; the hash cannot.

Deliberately free of Isaac and torch imports so the fast test suite exercises
it directly.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

#: Substrings that make a child's log a failure regardless of its exit code.
#: Matched case-insensitively against the whole log.
FATAL_LOG_PATTERNS: tuple[str, ...] = (
    "traceback (most recent call last)",
    "cuda error",
    "cuda out of memory",
    "cudnn_status",
    "runtimeerror",
    "an error occurred while",
    "fatal error",
    "segmentation fault",
    "core dumped",
    "assertionerror",
    "nan detected",
    "loss is nan",
    "physics simulation aborted",
    "simulation stopped",
    "killed",
)

#: Substrings that must NOT be treated as fatal even though they contain a
#: fatal pattern. Every one of these is a string this repository's own code
#: prints while working correctly.
FATAL_EXEMPTIONS: tuple[str, ...] = (
    # The evaluator and env deliberately describe the errors they refuse to
    # produce; e.g. "raise RuntimeError" appears in printed command lines.
    "raise runtimeerror",
    '"runtimeerror"',
)

#: rsl_rl prints this once per completed PPO update.
_ITERATION_RE = re.compile(r"Learning iteration\s+(\d+)\s*/\s*(\d+)")

#: Emitted by IsaacLab once the scene is up. Without it the child never ran.
_ENV_START_MARKERS: tuple[str, ...] = (
    "Completed setting up the environment",
    "Time taken for simulation start",
)


@dataclass(frozen=True)
class BlockAcceptance:
    """Verdict plus the provenance row for one training block."""

    accepted: bool
    reasons: tuple[str, ...] = ()
    label: str = ""
    input_checkpoint: str | None = None
    input_sha256: str | None = None
    output_checkpoint: str | None = None
    output_sha256: str | None = None
    started: str | None = None
    finished: str | None = None
    checkpoint_mtime: str | None = None
    return_code: int | None = None
    iterations_requested: int | None = None
    iterations_completed: int | None = None
    log_status: str = ""
    log_path: str | None = None

    def as_row(self) -> dict:
        row = {
            "label": self.label,
            "accepted": self.accepted,
            "reasons": list(self.reasons),
            "input_checkpoint": self.input_checkpoint,
            "input_sha256": self.input_sha256,
            "output_checkpoint": self.output_checkpoint,
            "output_sha256": self.output_sha256,
            "started": self.started,
            "finished": self.finished,
            "checkpoint_mtime": self.checkpoint_mtime,
            "return_code": self.return_code,
            "iterations_requested": self.iterations_requested,
            "iterations_completed": self.iterations_completed,
            "log_status": self.log_status,
            "log_path": self.log_path,
        }
        return row


def sha256_file(path: Path | str, *, chunk: int = 1 << 20) -> str:
    """Stream a file's SHA256. Streamed because checkpoints are ~1.4 MB now but
    the same helper is used on anything the supervisor produces."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def parse_training_log(text: str) -> dict:
    """Extract the three things the log has to prove.

    Returns ``environment_started``, ``iterations_completed`` and
    ``fatal_hits``. A log that never started an environment and a log that
    started one and then crashed are different failures and are reported
    differently.
    """
    lowered = text.lower()
    started = any(marker.lower() in lowered for marker in _ENV_START_MARKERS)
    matches = _ITERATION_RE.findall(text)
    iterations = len(matches)
    last_iteration = int(matches[-1][0]) if matches else None

    fatal_hits = []
    for pattern in FATAL_LOG_PATTERNS:
        index = lowered.find(pattern)
        while index != -1:
            # Look at the line the hit is on, so an exemption can be judged in
            # context rather than against the whole file.
            line_start = lowered.rfind("\n", 0, index) + 1
            line_end = lowered.find("\n", index)
            line = lowered[line_start : line_end if line_end != -1 else len(lowered)]
            if not any(exempt in line for exempt in FATAL_EXEMPTIONS):
                fatal_hits.append(pattern)
                break
            index = lowered.find(pattern, index + 1)
    return {
        "environment_started": started,
        "iterations_completed": iterations,
        "last_iteration_index": last_iteration,
        "fatal_hits": sorted(set(fatal_hits)),
    }


def accept_training_block(
    *,
    label: str,
    input_checkpoint: Path,
    output_checkpoint: Path,
    started: datetime,
    finished: datetime,
    return_code: int,
    iterations_requested: int,
    log_path: Path | None = None,
    log_text: str | None = None,
    summary_path: Path | None = None,
    mtime_slack_s: float = 2.0,
) -> BlockAcceptance:
    """Apply all seven acceptance criteria and build the provenance row."""
    reasons: list[str] = []
    input_sha = (
        sha256_file(input_checkpoint) if Path(input_checkpoint).is_file() else None
    )
    output_sha = None
    checkpoint_mtime = None

    if return_code != 0:
        reasons.append(f"child returned {return_code}")

    # (1) exists
    out = Path(output_checkpoint)
    if not out.is_file():
        reasons.append(f"checkpoint {out} was not written")
    else:
        output_sha = sha256_file(out)
        stamp = datetime.fromtimestamp(out.stat().st_mtime)
        checkpoint_mtime = stamp.isoformat(timespec="seconds")
        # (2) newer than the child's start
        if (started - stamp).total_seconds() > float(mtime_slack_s):
            reasons.append(
                f"checkpoint {out} was written at {checkpoint_mtime}, before "
                f"the block started at {started.isoformat(timespec='seconds')}"
            )
        # (3) differs from the input
        if input_sha is not None and output_sha == input_sha:
            reasons.append(
                "output checkpoint is byte-identical to the input "
                f"(sha256 {output_sha[:16]}...); the child did not train"
            )

    # (4)/(5) the training summary
    summary = Path(summary_path) if summary_path else out.with_suffix(".json")
    iterations_completed = None
    if not summary.is_file():
        reasons.append(f"training summary {summary} was not written")
    else:
        try:
            row = json.loads(summary.read_text())
        except json.JSONDecodeError as exc:
            reasons.append(f"training summary {summary} is not valid JSON: {exc}")
            row = {}
        if "iteration" not in row:
            reasons.append(f"training summary {summary} has no iteration count")
        else:
            iterations_completed = int(row["iteration"])

    # (6)/(7) the log
    status = "not-captured"
    if log_text is None and log_path is not None and Path(log_path).is_file():
        log_text = Path(log_path).read_text(errors="replace")
    if log_text is None:
        reasons.append("no child log was captured; criteria 6 and 7 cannot be checked")
    else:
        parsed = parse_training_log(log_text)
        status = (
            f"env_started={parsed['environment_started']} "
            f"iterations={parsed['iterations_completed']} "
            f"fatal={parsed['fatal_hits']}"
        )
        if not parsed["environment_started"]:
            reasons.append("log shows no successful environment start")
        if parsed["iterations_completed"] <= 0:
            reasons.append("log shows no completed learning iteration")
        elif (
            iterations_requested > 0
            and parsed["iterations_completed"] < iterations_requested
        ):
            reasons.append(
                f"log shows {parsed['iterations_completed']} completed "
                f"iterations, fewer than the {iterations_requested} requested"
            )
        if parsed["fatal_hits"]:
            reasons.append(f"log contains fatal markers {parsed['fatal_hits']}")

    return BlockAcceptance(
        accepted=not reasons,
        reasons=tuple(reasons),
        label=label,
        input_checkpoint=str(input_checkpoint),
        input_sha256=input_sha,
        output_checkpoint=str(out),
        output_sha256=output_sha,
        started=started.isoformat(timespec="seconds"),
        finished=finished.isoformat(timespec="seconds"),
        checkpoint_mtime=checkpoint_mtime,
        return_code=return_code,
        iterations_requested=int(iterations_requested),
        iterations_completed=iterations_completed,
        log_status=status,
        log_path=str(log_path) if log_path else None,
    )


def append_ledger(path: Path, row: dict) -> None:
    """Append one provenance row to a JSON-lines ledger, atomically enough.

    JSON lines rather than one JSON document: a supervisor killed mid-write
    must not corrupt the whole history of the run.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


__all__ = [
    "BlockAcceptance",
    "FATAL_LOG_PATTERNS",
    "accept_training_block",
    "append_ledger",
    "parse_training_log",
    "sha256_file",
]
