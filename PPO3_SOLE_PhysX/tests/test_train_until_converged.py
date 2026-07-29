"""Static command contracts for the held-out convergence supervisor."""

from __future__ import annotations

from pathlib import Path

import pytest

from PPO3_SOLE_PhysX.scripts.train_until_converged import (
    build_parser,
    candidate_is_better,
    evaluation_command,
    training_command,
    validate_args,
)


def _args():
    parser = build_parser()
    args = parser.parse_args(["--resume", "/tmp/model.pt"])
    validate_args(parser, args)
    return args


def test_default_training_block_uses_512_envs_and_fixed_lr():
    args = _args()
    command = training_command(
        args,
        resume=Path("/tmp/model.pt"),
        checkpoint_out=Path("/tmp/latest.pt"),
        block_index=1,
    )
    assert command[command.index("--num_envs") + 1] == "512"
    assert "--fixed_learning_rate" in command
    assert command[command.index("--stages") + 1] == "4"
    assert command[command.index("--checkpoint_out") + 1] == "/tmp/latest.pt"


def test_evaluation_is_forced_onto_held_out_terrain():
    args = _args()
    command = evaluation_command(
        args,
        checkpoint=Path("/tmp/model.pt"),
        report=Path("/tmp/report.json"),
    )
    assert "--eval_terrain" in command
    assert command[command.index("--steps") + 1] == "720"
    assert command[command.index("--num_envs") + 1] == "128"


def test_semantic_reset_flags_apply_only_to_first_training_block():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--resume",
            "/tmp/model.pt",
            "--reset_optimizer_first_block",
            "--reset_critic_head_first_block",
            "--first_block_c4_std",
            "0.1",
        ]
    )
    validate_args(parser, args)
    first = training_command(
        args,
        resume=Path("/tmp/model.pt"),
        checkpoint_out=Path("/tmp/latest.pt"),
        block_index=1,
    )
    later = training_command(
        args,
        resume=Path("/tmp/latest.pt"),
        checkpoint_out=Path("/tmp/latest.pt"),
        block_index=2,
    )
    assert "--reset_optimizer_on_resume" in first
    assert "--reset_critic_head_on_resume" in first
    assert first[first.index("--resume_c4_std") + 1] == "0.1"
    assert "--reset_optimizer_on_resume" not in later
    assert "--reset_critic_head_on_resume" not in later
    assert "--resume_c4_std" not in later


def test_block_cannot_restart_before_one_full_stage4_episode():
    parser = build_parser()
    args = parser.parse_args(
        ["--resume", "/tmp/model.pt", "--block_iterations", "22"]
    )
    with pytest.raises(SystemExit):
        validate_args(parser, args)


def test_best_checkpoint_never_loses_an_already_passed_gate():
    best_gates = {"safety": True, "strikes": False, "head": False}
    lower_score_but_regressed = {
        "score": 1.0,
        "gates": {"safety": False, "strikes": False, "head": False},
    }
    assert not candidate_is_better(
        lower_score_but_regressed, best_score=2.0, best_gates=best_gates
    )


def test_new_gate_beats_score_but_score_breaks_equal_gate_ties():
    best_gates = {"safety": True, "strikes": False, "head": False}
    new_gate = {
        "score": 99.0,
        "gates": {"safety": True, "strikes": True, "head": False},
    }
    lower_tie = {
        "score": 1.0,
        "gates": {"safety": True, "strikes": False, "head": False},
    }
    assert candidate_is_better(new_gate, best_score=2.0, best_gates=best_gates)
    assert candidate_is_better(lower_tie, best_score=2.0, best_gates=best_gates)
