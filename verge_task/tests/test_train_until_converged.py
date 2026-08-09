"""Static command contracts for the held-out convergence supervisor."""

from __future__ import annotations

from pathlib import Path

import pytest

from verge_task.scripts.train_until_converged import (
    build_parser,
    candidate_is_better,
    evaluation_command,
    load_validation_report,
    supervisor_gates,
    supervisor_passed,
    supervisor_score,
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


def test_strict_contact_controls_supervisor_stop_and_ranking(tmp_path, monkeypatch):
    """17/17 is not convergence when the strict promotion screen fails."""

    report_path = tmp_path / "heldout.json"
    report_path.write_text(
        __import__("json").dumps(
            {
                "terrain_provenance": {
                    "split": "held_out_validation",
                    "curriculum_version": "path_height_v2",
                    "steps": 960,
                },
                "verge": {"placeholder": 1.0},
            }
        )
    )
    standard = {
        "gates": {"functional": True},
        "normalized_gate_violations": {"functional": 0.0},
        "score": 0.0,
        "passed": True,
    }
    strict_gates = {"functional": True}
    strict_gates.update(
        {name: name == "contact_acquired" for name in __import__(
            "verge_task.terrain_convergence", fromlist=["CONTACT_SAFETY_GATE_SPECS"]
        ).CONTACT_SAFETY_GATE_SPECS}
    )
    strict = {
        "gates": strict_gates,
        "normalized_gate_violations": {},
        "score": 12.0,
        "passed": False,
        "contact_safety_promotion_blockers": ["visible_lift_events"],
        "promotable": False,
    }

    def fake_summary(_metrics, *, contact_safety=False):
        return strict if contact_safety else standard

    monkeypatch.setattr(
        "verge_task.scripts.train_until_converged.summarize_convergence",
        fake_summary,
    )
    loaded = load_validation_report(
        report_path, expected_steps=960, strict_contact=True
    )
    assert loaded["passed"] is True  # stable historical 17-gate meaning
    assert loaded["promotable"] is False
    assert supervisor_passed(loaded) is False
    assert supervisor_gates(loaded) == strict_gates
    assert supervisor_score(loaded) == 12.0


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


def test_lateral_focus_is_forwarded_to_training_only():
    parser = build_parser()
    args = parser.parse_args(["--resume", "/tmp/model.pt", "--lateral_focus"])
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
    assert "--lateral_focus" in train
    assert "--lateral_focus" not in evaluate


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


def test_adaptive_supervisor_requires_one_complete_long_episode():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--resume", "/tmp/model.pt",
            "--adaptive_trajectories",
            "--adaptive_episode_length", "40",
            "--eval_steps", "2399",
        ]
    )
    with pytest.raises(SystemExit):
        validate_args(parser, args)
    args.eval_steps = 2400
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


def test_lateral_reward_balance_is_forwarded_to_training_only():
    """Reward weights steer training; they must never reach the evaluator.

    The gates score every block against fixed metric definitions. If a reward
    knob leaked into `evaluation_command` the supervisor would be comparing
    blocks measured under different rulers, which is the same class of mistake
    the `--actuator` and observation-width forwarding tests guard against.
    """
    parser = build_parser()
    args = parser.parse_args(
        [
            "--resume", "/tmp/model.pt",
            "--lateral_tracking_std_m", "0.15",
            "--lateral_tracking_weight", "6.0",
            "--dense_lateral_error_weight", "6.0",
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
    assert train[train.index("--lateral_tracking_std_m") + 1] == "0.15"
    assert train[train.index("--lateral_tracking_weight") + 1] == "6.0"
    assert train[train.index("--dense_lateral_error_weight") + 1] == "6.0"
    for flag in (
        "--lateral_tracking_std_m",
        "--lateral_tracking_weight",
        "--dense_lateral_error_weight",
    ):
        assert flag not in evaluate


def test_lateral_reward_balance_defaults_to_absent():
    """Unset knobs must leave the shipped reward untouched."""
    args = _args()
    train = training_command(
        args,
        resume=Path("/tmp/model.pt"),
        checkpoint_out=Path("/tmp/latest.pt"),
        block_index=1,
    )
    for flag in (
        "--lateral_tracking_std_m",
        "--lateral_tracking_weight",
        "--dense_lateral_error_weight",
    ):
        assert flag not in train


def _report(gates, violations, score):
    return {"gates": dict(gates), "normalized_gate_violations": dict(violations),
            "score": score, "supervisor_score": score}


def test_sticky_gate_slack_keeps_a_checkpoint_lost_by_a_rounding_margin():
    """Block 3 of the exploration run must not be binned over 0.0005.

    Measured: p01_head_alignment 0.9745 against a 0.975 threshold -- a 0.05 %
    relative miss -- while worst-family ground contact improved 0.6556 ->
    0.7503 and the violation score nearly halved.
    """
    from verge_task.scripts.train_until_converged import candidate_is_better

    best_gates = {"head_p01": True, "contact": False}
    candidate = _report({"head_p01": False, "contact": True},
                        {"head_p01": 0.0005, "contact": 0.0}, 1.03)

    # Strict behaviour (default) still refuses it.
    assert not candidate_is_better(
        candidate, best_score=1.86, best_gates=best_gates, best_metrics={}
    )
    # With slack, the marginal loss is noise and the far better score wins.
    assert candidate_is_better(
        candidate, best_score=1.86, best_gates=best_gates, best_metrics={},
        sticky_gate_slack=0.01,
    )


def test_sticky_gate_slack_still_vetoes_a_real_regression():
    from verge_task.scripts.train_until_converged import candidate_is_better

    best_gates = {"head_p01": True}
    # Lost by 20 %, far outside the slack band: an absolute veto regardless of score.
    candidate = _report({"head_p01": False}, {"head_p01": 0.20}, 0.01)
    assert not candidate_is_better(
        candidate, best_score=1.86, best_gates=best_gates, best_metrics={},
        sticky_gate_slack=0.01,
    )


def test_sticky_gate_slack_requires_a_materially_better_score():
    """A marginal loss must not be bought with a coin-flip improvement."""
    from verge_task.scripts.train_until_converged import candidate_is_better

    best_gates = {"head_p01": True}
    candidate = _report({"head_p01": False}, {"head_p01": 0.0005}, 1.80)
    # 1.80 vs 1.86 is better but not by the required 10 %.
    assert not candidate_is_better(
        candidate, best_score=1.86, best_gates=best_gates, best_metrics={},
        sticky_gate_slack=0.01,
    )


def test_worst_case_tail_flags_are_forwarded_to_training_only():
    parser = build_parser()
    args = parser.parse_args(
        ["--resume", "/tmp/model.pt",
         "--contact_tail_fraction", "0.25", "--contact_tail_multiplier", "4.0",
         "--lateral_tail_fraction", "0.25", "--lateral_tail_multiplier", "3.0"]
    )
    validate_args(parser, args)
    train = training_command(args, resume=Path("/tmp/model.pt"),
                             checkpoint_out=Path("/tmp/latest.pt"), block_index=1)
    evaluate = evaluation_command(args, checkpoint=Path("/tmp/model.pt"),
                                  report=Path("/tmp/report.json"))
    assert train[train.index("--contact_tail_fraction") + 1] == "0.25"
    assert train[train.index("--lateral_tail_multiplier") + 1] == "3.0"
    for flag in ("--contact_tail_fraction", "--contact_tail_multiplier",
                 "--lateral_tail_fraction", "--lateral_tail_multiplier"):
        assert flag not in evaluate


def test_per_axis_lead_cap_is_forwarded_to_BOTH_children():
    """The lead cap changes the PLANT, so training and evaluation must agree.

    Measured 2026-08-04 on the 16/17 checkpoint, evaluation only: raising the
    cap 0.018 -> 0.050 halved lateral error on the sustained-offset families
    (lane_change_hold 0.4833 -> 0.2310, boom_sweep 0.3745 -> 0.1602). A block
    trained under one cap and scored under another would be measuring a
    different machine.
    """
    parser = build_parser()
    args = parser.parse_args(
        ["--resume", "/tmp/model.pt",
         "--cylinder_target_max_lead_per_axis_m", "0.05,0.05,0.05,0.20"]
    )
    validate_args(parser, args)
    train = training_command(args, resume=Path("/tmp/model.pt"),
                             checkpoint_out=Path("/tmp/latest.pt"), block_index=1)
    evaluate = evaluation_command(args, checkpoint=Path("/tmp/model.pt"),
                                  report=Path("/tmp/report.json"))
    for command in (train, evaluate):
        index = command.index("--cylinder_target_max_lead_per_axis_m")
        assert command[index + 1] == "0.05,0.05,0.05,0.20"


def test_per_axis_lead_cap_defaults_to_absent_so_the_scalar_still_rules():
    args = _args()
    train = training_command(args, resume=Path("/tmp/model.pt"),
                             checkpoint_out=Path("/tmp/latest.pt"), block_index=1)
    evaluate = evaluation_command(args, checkpoint=Path("/tmp/model.pt"),
                                  report=Path("/tmp/report.json"))
    assert "--cylinder_target_max_lead_per_axis_m" not in train
    assert "--cylinder_target_max_lead_per_axis_m" not in evaluate
