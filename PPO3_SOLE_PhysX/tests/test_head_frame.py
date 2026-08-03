"""The head controller must be able to SEE the axis it fails on.

Measured on stage 6 at 2.2x tracking amplitude, 30 720 samples
(reports/stage6/head_frame/ppo3_amp2.2.json), split on head alignment:

                        healthy (>=0.9)   collapsed (<0.9)
    mean |arm slew|        0.0124 rad        0.1775 rad   (14x)
    joint / world tilt     1.048             1.545
    head axis X / Y        0.214             1.305
    mean |q5 action|       0.0               0.0

with corr(|slew|, joint_tilt - world_tilt) = +0.814 and
corr(|slew|, alignment) = -0.755.

Two independent defects follow, and these tests pin both fixes:

1. the levelling correction and the c4 shield measured tilt in the WORLD Y-Z
   plane, which under-reads the correctable tilt by 54 % exactly where the head
   collapses, because the excursion rotates into world X as the arm slews;
2. the policy could not observe the actionable error at all -- it saw an
   unsigned alignment cosine and a world-frame axis.
"""

from __future__ import annotations

import ast
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
ENV = (ROOT / "PPO3_SOLE_PhysX" / "terrain_env.py").read_text()

from PPO3_SOLE_PhysX.terrain_observations import (  # noqa: E402
    HEAD_FRAME_CHANNELS,
    HEAD_FRAME_CHANNEL_NAMES,
    verge_observation_dim,
)


def test_head_frame_adds_exactly_two_channels():
    base = verge_observation_dim(model_error=True, model_error_history=3,
                                 contact_state=True)
    wide = verge_observation_dim(model_error=True, model_error_history=3,
                                 contact_state=True, head_frame=True)
    assert wide - base == HEAD_FRAME_CHANNELS == 2
    assert len(HEAD_FRAME_CHANNEL_NAMES) == 2
    assert wide + 1 == 98


def test_the_shield_can_use_the_joint_frame():
    """A shield computed on a signal that fades as the failure grows is not a
    shield. The world-frame projection under-reads by 54 % during collapse."""
    body = ENV[ENV.index("def _task_action_shield") :]
    body = body[: body.index("directional_tilt_shield(")]
    assert "terrain_head_tilt_arm_frame" in body, (
        "the shield cannot select the joint frame"
    )
    assert "_head_joint_signed_tilt()" in body
    assert "torch.atan2(head_axis[:, 1], -head_axis[:, 2])" in body, (
        "the world-frame branch must remain, so the change stays reversible "
        "and every stored score stays reproducible"
    )


def test_the_head_frame_observation_exposes_tilt_and_slew():
    body = ENV[ENV.index("def _head_frame_channels") :]
    body = body[: body.index("def _get_observations")]
    assert "_head_joint_signed_tilt()" in body, "the actionable error is missing"
    assert "_arm_joint_ids[0]" in body, "the slew angle is missing"
    assert "terrain_head_frame_observation" in body, "the flag is not honoured"
    # Bounded, so a bad pose cannot hand the actor an unbounded input.
    assert "torch.clamp" in body


def test_head_frame_is_off_by_default_so_stored_scores_reproduce():
    cfg = (ROOT / "PPO3_SOLE_PhysX" / "terrain_env_cfg.py").read_text()
    assert "terrain_head_frame_observation: bool = False" in cfg
    assert "terrain_head_tilt_arm_frame: bool = False" in cfg


def test_the_width_helper_follows_the_head_frame_flag():
    """Seven entry points share one derivation; a flag that reaches six of
    seven is how an entry point builds a differently shaped network."""
    from types import SimpleNamespace

    from PPO3_SOLE_PhysX.terrain_observations import observation_space_for_cfg

    base = SimpleNamespace(
        terrain_model_error_observation=True,
        terrain_model_error_history=3,
        terrain_contact_observation=True,
        terrain_head_frame_observation=False,
    )
    assert observation_space_for_cfg(base) == 96
    base.terrain_head_frame_observation = True
    assert observation_space_for_cfg(base) == 98


def test_supervisor_forwards_the_head_flags_to_both_children(tmp_path):
    import importlib.util

    script = ROOT / "PPO3_SOLE_PhysX" / "scripts" / "train_until_converged.py"
    spec = importlib.util.spec_from_file_location("tuc", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = module.build_parser().parse_args(
        ["--resume", "/tmp/x.pt",
         "--head_frame_observation", "true",
         "--head_tilt_arm_frame", "true",
         "--tracking_amplitude", "2.2"]
    )
    train = module.training_command(
        args, resume=Path("/tmp/x.pt"), checkpoint_out=tmp_path / "o.pt",
        block_index=1,
    )
    evaluate = module.evaluation_command(
        args, checkpoint=tmp_path / "o.pt", report=tmp_path / "r.json"
    )
    for flag in ("--head_frame_observation", "--head_tilt_arm_frame",
                 "--tracking_amplitude"):
        assert flag in train, f"training is missing {flag}"
        assert flag in evaluate, f"evaluation is missing {flag}"
        assert train[train.index(flag) + 1] == evaluate[evaluate.index(flag) + 1]


def test_q5_is_a_real_output_and_the_probe_watches_it():
    """The 5th policy output measured IDENTICALLY ZERO on PPO3 across the whole
    stage-6 rollout, healthy and collapsed alike. PPO4's is 0.398, so it is not
    structural -- the migration zero-initialised it and training never moved
    it. The probe reports it so the next run cannot hide the same thing."""
    probe = (
        ROOT / "PPO3_SOLE_PhysX" / "scripts" / "probe_head_frame.py"
    ).read_text()
    assert "mean_abs_q5_action" in probe
    cfg = (ROOT / "PPO3_SOLE_PhysX" / "terrain_env_cfg.py").read_text()
    assert "sole_head_joint_action: bool = True" in cfg


def test_every_width_changing_flag_is_applied_before_the_width_is_derived():
    """Ordering, not spelling, is how this keeps breaking.

    Three flags change the observation width. If any is applied AFTER
    `observation_space_for_cfg`, training builds a network of one width for a
    checkpoint of another and dies before the first iteration -- which is
    exactly what happened to both actuators on the head-fix run:
    "checkpoint observation width 98 does not match the frozen terrain
    contract 96".
    """
    source = (
        ROOT / "PPO3_SOLE_PhysX" / "scripts" / "train.py"
    ).read_text()
    derive = source.index("env_cfg.observation_space = observation_space_for_cfg")
    for flag in (
        "env_cfg.terrain_contact_observation = ",
        "env_cfg.terrain_head_frame_observation = ",
        'override_cfg(env_cfg, "terrain_model_error_history"',
    ):
        assert flag in source, f"{flag} is gone"
        assert source.index(flag) < derive, (
            f"{flag} is applied AFTER the observation width is derived; the "
            "environment will be built at the wrong width"
        )


# ----------------------------------------------------------------------
# Checkpoint selection: the gate score saturates and cannot rank
# ----------------------------------------------------------------------


def _supervisor_module():
    import importlib.util

    script = ROOT / "PPO3_SOLE_PhysX" / "scripts" / "train_until_converged.py"
    spec = importlib.util.spec_from_file_location("tuc_sel", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ALL_PASS = {f"g{i}": True for i in range(17)}


def _report(score, **metrics):
    base = {
        "min_head_alignment": 0.98,
        "p01_head_alignment": 0.99,
        "max_abs_head_axis_x": 0.03,
        "visible_lift_event_count": 100,
        "maximum_consecutive_airborne_steps": 10,
        "ground_contact_fraction": 0.94,
    }
    base.update(metrics)
    return {"gates": dict(ALL_PASS), "score": score, "metrics": base}


def test_a_strictly_better_passing_checkpoint_is_not_discarded():
    """MEASURED: stage S6a block 2 scored 0.000000 against a best of 0.000000,
    was logged "regressed", and was thrown away -- while holding
    min_head_alignment 0.9911 against the kept checkpoint's 0.9826 and
    max_abs_head_axis_x 0.009 against 0.033.

    The normalized score saturates at zero once every gate passes, so it cannot
    rank two passing checkpoints at all.
    """
    module = _supervisor_module()
    best = _report(0.0, min_head_alignment=0.9826, max_abs_head_axis_x=0.033)
    better = _report(0.0, min_head_alignment=0.9911, max_abs_head_axis_x=0.009)
    assert module.candidate_is_better(
        better, best_score=0.0, best_gates=ALL_PASS, best_metrics=best["metrics"]
    ), "a strictly better passing checkpoint was discarded on a score tie"


def test_a_worse_passing_checkpoint_is_still_rejected_on_a_tie():
    module = _supervisor_module()
    best = _report(0.0, min_head_alignment=0.9911)
    worse = _report(0.0, min_head_alignment=0.9826)
    assert not module.candidate_is_better(
        worse, best_score=0.0, best_gates=ALL_PASS, best_metrics=best["metrics"]
    )


def test_head_alignment_outranks_contact_in_the_tie_break():
    """Head first: it is the axis every campaign has died on."""
    module = _supervisor_module()
    best = _report(0.0, min_head_alignment=0.98, visible_lift_event_count=50)
    # Better head, worse contact -> still promoted.
    candidate = _report(0.0, min_head_alignment=0.99, visible_lift_event_count=200)
    assert module.candidate_is_better(
        candidate, best_score=0.0, best_gates=ALL_PASS, best_metrics=best["metrics"]
    )
    # Equal head, better contact -> promoted on the secondary key.
    equal_head = _report(0.0, visible_lift_event_count=10)
    base = _report(0.0, visible_lift_event_count=100)
    assert module.candidate_is_better(
        equal_head, best_score=0.0, best_gates=ALL_PASS, best_metrics=base["metrics"]
    )


def test_a_lost_gate_still_vetoes_regardless_of_metrics():
    module = _supervisor_module()
    lost = dict(ALL_PASS)
    lost["g0"] = False
    candidate = {"gates": lost, "score": 0.0, "metrics": _report(0.0, min_head_alignment=1.0)["metrics"]}
    assert not module.candidate_is_better(
        candidate, best_score=0.0, best_gates=ALL_PASS,
        best_metrics=_report(0.0)["metrics"],
    )


# ----------------------------------------------------------------------
# q5 is an INTEGRATOR and was never penalised
# ----------------------------------------------------------------------


def test_q5_integrator_has_a_leak_and_it_defaults_off():
    """MEASURED: PPO4's q5 carries a DC bias of 0.324 (mean 0.324 against
    mean-abs 0.389, i.e. almost pure offset). At scale 0.010 rad that is
    0.0032 rad/step, saturating the 0.35 rad bound in ~108 of 960 steps -- a
    permanent 20 degree head offset. PPO3's q5 sits at 0.0114 and never winds
    up. That is the entire difference between their head alignment (0.5303 vs
    0.9875).

    Same anti-windup, and same rationale, as `cylinder_target_leak`.
    """
    env = (ROOT / "PPO3_SOLE_PhysX" / "env.py").read_text()
    assert "sole_head_correction_leak" in env
    body = env[env.index("sole_head_correction_leak") - 1200 :]
    body = body[: body.index("_head_joint_correction = correction")]
    assert "correction * (1.0 - leak)" in body, "the leak is not applied"
    cfg = (ROOT / "PPO3_SOLE_PhysX" / "env_cfg.py").read_text()
    assert "sole_head_correction_leak: float = 0.0" in cfg, (
        "the leak must default off so every stored score reproduces"
    )


def test_the_standing_head_offset_is_penalised_not_the_action():
    """The action is a RATE; the accumulated offset is what tilts the head.

    Penalising the action would tax useful transient corrections. And the
    existing magnitude/rate penalties act on `_sole_applied`, which is FOUR
    wide -- the fifth output has never been penalised at all.
    """
    src = (ROOT / "PPO3_SOLE_PhysX" / "terrain_env.py").read_text()
    body = src[src.index("terrain_head_correction_weight") :]
    body = body[: body.index("if contact_terms is not None:")]
    assert "_head_joint_correction" in body, "penalises something other than the offset"
    assert "sole_head_correction_max_rad" in body, "the offset is not normalised"
    assert "head_offset_penalty" in src
    # It must reach the reward sum, in the SAFETY group (a latched head must
    # still be charged for holding a standing offset).
    total = src[src.index("safety_reward = (") :]
    total = total[: total.index(")", total.index("chassis_penalty"))]
    assert "head_offset_penalty" in total, "the penalty never reaches the reward"
    cfg = (ROOT / "PPO3_SOLE_PhysX" / "terrain_env_cfg.py").read_text()
    assert "terrain_head_correction_weight: float = 0.0" in cfg


def test_the_head_offset_is_logged_so_windup_is_visible():
    """The windup was invisible: nothing reported the integrator's state."""
    src = (ROOT / "PPO3_SOLE_PhysX" / "terrain_env.py").read_text()
    assert "action/head_correction_rad_mean_abs" in src
    assert "action/head_correction_rad_max" in src
