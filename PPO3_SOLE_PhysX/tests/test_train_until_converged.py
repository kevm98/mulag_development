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
    assert command[command.index("--steps") + 1] == "960"
    assert command[command.index("--num_envs") + 1] == "128"


def test_head_joint_action_is_forwarded_to_training_and_evaluation():
    parser = build_parser()
    args = parser.parse_args(
        ["--resume", "/tmp/model.pt", "--head_joint_action"]
    )
    validate_args(parser, args)
    train = training_command(
        args,
        resume=Path("/tmp/model.pt"),
        checkpoint_out=Path("/tmp/latest.pt"),
        block_index=1,
    )
    evaluate = evaluation_command(
        args,
        checkpoint=Path("/tmp/model.pt"),
        report=Path("/tmp/report.json"),
    )
    assert "--head_joint_action" in train
    assert "--head_joint_action" in evaluate


def test_head_action_mixer_gain_is_forwarded_to_training_and_evaluation():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--resume",
            "/tmp/model.pt",
            "--head_joint_action",
            "--head_action_to_c4_gain",
            "0.6",
        ]
    )
    validate_args(parser, args)
    train = training_command(
        args,
        resume=Path("/tmp/model.pt"),
        checkpoint_out=Path("/tmp/latest.pt"),
        block_index=1,
    )
    evaluate = evaluation_command(
        args,
        checkpoint=Path("/tmp/model.pt"),
        report=Path("/tmp/report.json"),
    )
    assert train[train.index("--head_action_to_c4_gain") + 1] == "0.6"
    assert evaluate[evaluate.index("--head_action_to_c4_gain") + 1] == "0.6"


def test_q5_exploration_cap_is_forwarded_to_training_only():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--resume",
            "/tmp/model.pt",
            "--head_joint_action",
            "--q5_exploration_std",
            "0.04",
        ]
    )
    validate_args(parser, args)
    train = training_command(
        args,
        resume=Path("/tmp/model.pt"),
        checkpoint_out=Path("/tmp/latest.pt"),
        block_index=2,
    )
    evaluate = evaluation_command(
        args,
        checkpoint=Path("/tmp/model.pt"),
        report=Path("/tmp/report.json"),
    )
    assert train[train.index("--resume_q5_std") + 1] == "0.04"
    assert "--resume_q5_std" not in evaluate


def test_tail_focus_is_forwarded_to_training_only():
    parser = build_parser()
    args = parser.parse_args(["--resume", "/tmp/model.pt", "--tail_focus"])
    validate_args(parser, args)
    train = training_command(
        args,
        resume=Path("/tmp/model.pt"),
        checkpoint_out=Path("/tmp/latest.pt"),
        block_index=1,
    )
    evaluate = evaluation_command(
        args,
        checkpoint=Path("/tmp/model.pt"),
        report=Path("/tmp/report.json"),
    )
    assert "--tail_focus" in train
    assert "--tail_focus" not in evaluate


def test_head_tail_risk_is_forwarded_to_training_only():
    parser = build_parser()
    args = parser.parse_args(["--resume", "/tmp/model.pt", "--head_tail_risk"])
    validate_args(parser, args)
    train = training_command(
        args,
        resume=Path("/tmp/model.pt"),
        checkpoint_out=Path("/tmp/latest.pt"),
        block_index=1,
    )
    evaluate = evaluation_command(
        args,
        checkpoint=Path("/tmp/model.pt"),
        report=Path("/tmp/report.json"),
    )
    assert "--head_tail_risk" in train
    assert "--head_tail_risk" not in evaluate


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
        ["--resume", "/tmp/model.pt", "--block_iterations", "29"]
    )
    with pytest.raises(SystemExit):
        validate_args(parser, args)


def test_minimum_blocks_can_force_extensive_continuation():
    parser = build_parser()
    args = parser.parse_args(
        ["--resume", "/tmp/model.pt", "--min_blocks", "20"]
    )
    validate_args(parser, args)
    assert args.min_blocks == 20
    assert args.max_blocks == 0


def test_maximum_blocks_cannot_precede_minimum_blocks():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--resume",
            "/tmp/model.pt",
            "--min_blocks",
            "20",
            "--max_blocks",
            "10",
        ]
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
