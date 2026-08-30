from __future__ import annotations

import pytest

from PPO_AAM_FD.scripts.train import TASK, prepare_training_args


def test_default_training_domain_is_episode_constant_two_to_seven_kmh():
    args = prepare_training_args(["--headless"])
    assert args.count("--task") == 1
    assert args[args.index("--task") + 1] == TASK
    assert args[args.index("--vehicle_speed_min_kmh") + 1] == "2.0"
    assert args[args.index("--vehicle_speed_max_kmh") + 1] == "7.0"
    assert "--vehicle_speed_kmh" not in args
    assert args[args.index("--vehicle_speed_sampling") + 1] == "uniform"


def test_explicit_speed_range_is_preserved_without_duplicate_defaults():
    args = prepare_training_args(
        ["--vehicle_speed_min_kmh", "2.0", "--vehicle_speed_max_kmh", "7.0"]
    )
    assert args.count("--vehicle_speed_min_kmh") == 1
    assert args.count("--vehicle_speed_max_kmh") == 1


def test_half_range_is_rejected():
    with pytest.raises(SystemExit, match="must be given together"):
        prepare_training_args(["--vehicle_speed_min_kmh", "2.0"])
