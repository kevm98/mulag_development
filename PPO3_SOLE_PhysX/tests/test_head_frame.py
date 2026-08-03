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
