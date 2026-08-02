"""Acceptance contracts for a supervised training block.

The failure these guard against actually happened: a training block died on a
RuntimeError, printed the traceback, exited 0 because Isaac shuts the process
down through omni.kit, and the supervisor evaluated the STALE checkpoint from
the previous block and recorded its numbers under the new block's name.

Every test below is written against that incident, not against a hypothetical.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from PPO3_SOLE_PhysX.block_provenance import (
    FATAL_LOG_PATTERNS,
    accept_training_block,
    append_ledger,
    parse_training_log,
    sha256_file,
)

GOOD_LOG = """
[INFO]: Time taken for simulation start : 1.03 seconds
[INFO]: Completed setting up the environment...
################################################################################
                           Learning iteration 692/732
                         Iteration time: 11.50s
################################################################################
                           Learning iteration 693/732
                         Iteration time: 11.74s
[PPO3-CHECKPOINT] {"iteration": 693}
"""


def _write(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _summary(path: Path, iteration: int = 693) -> Path:
    path.write_text(json.dumps({"iteration": iteration, "actuator": "rom"}))
    return path


def _accept(tmp_path: Path, **overrides):
    # NOTE: the defaults below are built lazily on purpose. `dict.pop(key,
    # default)` evaluates its default EAGERLY, so writing the files inline
    # rewrote them even when the test had supplied an override -- silently
    # undoing three tests' own setup and making them pass a healthy block.
    started = overrides.pop("started", datetime.now() - timedelta(seconds=30))
    inp = overrides.pop("input_checkpoint", None)
    if inp is None:
        inp = _write(tmp_path / "in.pt", b"INPUT-WEIGHTS")
    out = overrides.pop("output_checkpoint", None)
    if out is None:
        out = _write(tmp_path / "out.pt", b"TRAINED-WEIGHTS")
    if "summary_path" not in overrides and not out.with_suffix(".json").is_file():
        _summary(out.with_suffix(".json"))
    kwargs = dict(
        label="block 1",
        input_checkpoint=inp,
        output_checkpoint=out,
        started=started,
        finished=datetime.now(),
        return_code=0,
        iterations_requested=2,
        log_text=GOOD_LOG,
    )
    kwargs.update(overrides)
    return accept_training_block(**kwargs)


# ----------------------------------------------------------------------
# The seven criteria
# ----------------------------------------------------------------------


def test_a_healthy_block_is_accepted(tmp_path):
    result = _accept(tmp_path)
    assert result.accepted, result.reasons
    assert result.input_sha256 != result.output_sha256
    assert result.iterations_completed == 693
    assert result.return_code == 0


def test_1_missing_checkpoint_is_rejected(tmp_path):
    result = _accept(tmp_path, output_checkpoint=tmp_path / "never.pt")
    assert not result.accepted
    assert any("was not written" in r for r in result.reasons)


def test_2_stale_checkpoint_is_rejected(tmp_path):
    """The exact incident: the previous block's file left in place."""
    import os
    import time

    out = _write(tmp_path / "out.pt", b"PREVIOUS-BLOCK-WEIGHTS")
    _summary(out.with_suffix(".json"))
    old = time.time() - 3600
    os.utime(out, (old, old))
    result = _accept(tmp_path, output_checkpoint=out)
    assert not result.accepted
    assert any("before" in r and "started" in r for r in result.reasons)


def test_3_byte_identical_output_is_rejected(tmp_path):
    """Timestamps can be fooled by a touch. The hash cannot.

    A child that copied its input forward, or that saved before training,
    produces a fresh mtime and an unchanged file.
    """
    payload = b"IDENTICAL-WEIGHTS"
    inp = _write(tmp_path / "in.pt", payload)
    out = _write(tmp_path / "out.pt", payload)
    _summary(out.with_suffix(".json"))
    result = _accept(tmp_path, input_checkpoint=inp, output_checkpoint=out)
    assert not result.accepted
    assert any("byte-identical" in r for r in result.reasons)
    assert result.input_sha256 == result.output_sha256


def test_3_input_hash_must_be_captured_before_the_child_runs(tmp_path):
    """resume and checkpoint_out are normally the SAME file.

    The supervisor resumes model_latest.pt and writes model_latest.pt, so
    hashing the input inside the acceptance check hashes the child's own
    output and makes criterion 3 compare a file with itself. Measured: PPO4
    stage B1 block 2 was rejected as "byte-identical to the input" after
    training 40 iterations perfectly well.
    """
    same = _write(tmp_path / "model_latest.pt", b"AFTER-TRAINING")
    _summary(same.with_suffix(".json"))
    before = "0" * 64  # what the file hashed to BEFORE the child overwrote it

    naive = _accept(tmp_path, input_checkpoint=same, output_checkpoint=same)
    assert not naive.accepted, "hashing after the fact cannot detect training"

    correct = _accept(
        tmp_path,
        input_checkpoint=same,
        output_checkpoint=same,
        input_sha256=before,
    )
    assert correct.accepted, correct.reasons
    assert correct.input_sha256 == before


def test_supervisor_hashes_the_input_before_launching(tmp_path):
    source = (
        Path(__file__).resolve().parents[1] / "scripts" / "train_until_converged.py"
    ).read_text()
    body = source[source.index("while args.max_blocks") :]
    hashed = body.index("input_sha = (")
    launched = body.index("run_child(")
    assert hashed < launched, (
        "the input checkpoint is hashed after the child ran; with resume and "
        "checkpoint_out on the same path that hashes the child's own output"
    )


def test_4_missing_summary_is_rejected(tmp_path):
    out = _write(tmp_path / "out.pt", b"TRAINED")
    # deliberately no out.json
    result = _accept(tmp_path, output_checkpoint=out, summary_path=tmp_path / "nope.json")
    assert not result.accepted
    assert any("training summary" in r for r in result.reasons)


def test_5_summary_without_an_iteration_count_is_rejected(tmp_path):
    out = _write(tmp_path / "out.pt", b"TRAINED")
    out.with_suffix(".json").write_text(json.dumps({"actuator": "rom"}))
    result = _accept(tmp_path, output_checkpoint=out)
    assert not result.accepted
    assert any("no iteration count" in r for r in result.reasons)


def test_6_log_without_an_environment_start_is_rejected(tmp_path):
    result = _accept(tmp_path, log_text="Learning iteration 1/2\nIteration time: 1s\n")
    assert not result.accepted
    assert any("no successful environment start" in r for r in result.reasons)


def test_6_log_without_a_completed_iteration_is_rejected(tmp_path):
    result = _accept(
        tmp_path,
        log_text="[INFO]: Completed setting up the environment...\n",
    )
    assert not result.accepted
    assert any("no completed learning iteration" in r for r in result.reasons)


def test_6_fewer_iterations_than_requested_is_rejected(tmp_path):
    result = _accept(tmp_path, iterations_requested=40)
    assert not result.accepted
    assert any("fewer than the 40 requested" in r for r in result.reasons)


@pytest.mark.parametrize(
    "marker",
    [
        "Traceback (most recent call last):",
        "CUDA error: device-side assert triggered",
        "RuntimeError: migrated shared-policy checkpoints require reset",
        "Segmentation fault (core dumped)",
        "loss is nan",
    ],
)
def test_7_fatal_log_markers_are_rejected(tmp_path, marker):
    result = _accept(tmp_path, log_text=GOOD_LOG + "\n" + marker + "\n")
    assert not result.accepted
    assert any("fatal markers" in r for r in result.reasons)


def test_7_a_command_line_mentioning_runtimeerror_is_not_fatal(tmp_path):
    """The supervisor prints child command lines that contain the word.

    Rejecting on the substring alone would fail every healthy block.
    """
    noisy = GOOD_LOG + '\n[PPO3-SUPERVISOR] starting: ["--help", "raise RuntimeError"]\n'
    result = _accept(tmp_path, log_text=noisy)
    assert result.accepted, result.reasons


def test_a_nonzero_return_code_is_still_rejected(tmp_path):
    result = _accept(tmp_path, return_code=1)
    assert not result.accepted
    assert any("returned 1" in r for r in result.reasons)


# ----------------------------------------------------------------------
# Provenance
# ----------------------------------------------------------------------


def test_the_row_carries_full_provenance(tmp_path):
    result = _accept(tmp_path)
    row = result.as_row()
    for key in (
        "input_sha256",
        "output_sha256",
        "started",
        "finished",
        "checkpoint_mtime",
        "return_code",
        "iterations_requested",
        "iterations_completed",
        "log_status",
        "accepted",
        "reasons",
    ):
        assert key in row, f"provenance row is missing {key}"
    assert len(row["input_sha256"]) == 64
    assert len(row["output_sha256"]) == 64


def test_the_ledger_is_append_only_json_lines(tmp_path):
    """A supervisor killed mid-write must not corrupt the whole history."""
    ledger = tmp_path / "block_provenance.jsonl"
    append_ledger(ledger, {"label": "block 1", "accepted": True})
    append_ledger(ledger, {"label": "block 2", "accepted": False})
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert [r["label"] for r in rows] == ["block 1", "block 2"]


def test_sha256_matches_hashlib(tmp_path):
    import hashlib

    path = _write(tmp_path / "x.bin", b"abc" * 1000)
    assert sha256_file(path) == hashlib.sha256(b"abc" * 1000).hexdigest()


def test_log_parser_counts_iterations_and_finds_the_start(tmp_path):
    parsed = parse_training_log(GOOD_LOG)
    assert parsed["environment_started"] is True
    assert parsed["iterations_completed"] == 2
    assert parsed["last_iteration_index"] == 693
    assert parsed["fatal_hits"] == []


def test_every_fatal_pattern_is_lowercase():
    """The parser lowercases the log once; a mixed-case pattern never matches."""
    assert all(p == p.lower() for p in FATAL_LOG_PATTERNS)


# ----------------------------------------------------------------------
# The supervisor actually uses it
# ----------------------------------------------------------------------


def test_supervisor_rejects_before_evaluating(tmp_path):
    source = (
        Path(__file__).resolve().parents[1] / "scripts" / "train_until_converged.py"
    ).read_text()
    assert "accept_training_block(" in source
    assert "append_ledger(" in source
    assert "block_provenance.jsonl" in source
    # The rejection must happen BEFORE evaluate_candidate is reached.
    reject = source.index("REJECTED")
    evaluate = source.index("heldout_block_", source.index("while args.max_blocks"))
    assert reject < evaluate, (
        "an unverified checkpoint reaches the evaluator before the acceptance "
        "check rejects it"
    )
    assert "capture_to=child_log" in source, (
        "the child log is not captured, so criteria 6 and 7 cannot be checked"
    )
