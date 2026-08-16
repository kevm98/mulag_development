"""Phase 4 needs the evaluation harness to express the whole ablation grid.

Two of the four routing arms set the CONTROLLER latent to `zero`, and the
online-adaptation claim needs `frozen`. If the harness cannot express those,
the ablation silently collapses to whatever it can express -- which is how a
"policy-only" screen ends up standing in for a full-system one.

Checked at the source level because building the parser requires Isaac Lab.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
HARNESS = REPO / "PPO_ROM/scripts/evaluate_gates.py"


def _arg(name: str) -> ast.Call:
    tree = ast.parse(HARNESS.read_text())
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == name):
            return node
    raise AssertionError(f"harness has no {name} argument")


def _choices(name: str) -> set[str]:
    for kw in _arg(name).keywords:
        if kw.arg == "choices":
            return {e.value for e in kw.value.elts}
    raise AssertionError(f"{name} has no choices")


def test_policy_latent_mode_covers_the_content_controls():
    c = _choices("--online_aam_latent_mode")
    for mode in ("estimated", "zero", "shuffled", "frozen"):
        assert mode in c, f"policy arm {mode!r} not expressible"


def test_controller_latent_mode_covers_the_routing_grid():
    """`zero` is what makes A1 (baseline) and A3 (PPO-only) possible."""
    c = _choices("--online_aam_controller_latent_mode")
    for mode in ("estimated", "zero"):
        assert mode in c, (
            f"controller arm {mode!r} not expressible; without it the routing "
            "ablation cannot separate ID adaptation from policy adaptation"
        )


def test_frozen_has_a_freeze_tick():
    """frozen without a latch tick raises in the runtime; the harness must be
    able to supply one."""
    _arg("--online_aam_latent_freeze_tick")


@pytest.mark.parametrize("flag", [
    "--rom_factor_perturbation",     # ROM plant: system_factors
    "--rom_factor_step",
    "--plant_alpha_range",           # ID-FD plant: valve effectiveness
    "--plant_delay_steps_range",
    "--plant_step_time_s",
    "--adaptation_trace_path",       # per-timestep evidence
])
def test_perturbation_and_trace_flags_exist(flag):
    _arg(flag)


def test_overrides_are_actually_applied():
    """A flag that parses but never reaches the cfg is worse than a missing
    one: the run looks configured and is not."""
    src = HARNESS.read_text()
    for cfg_key in (
        "online_aam_latent_freeze_tick",
        "rom_factor_perturbation_enabled",
        "rom_factor_perturbation_ranges",
        "rom_factor_perturbation_step_values",
        "plant_perturbation_enabled",
        "plant_perturbation_alpha_range",
        "plant_perturbation_delay_steps_range",
        "adaptation_trace_path",
    ):
        assert f'"{cfg_key}"' in src, f"{cfg_key} is never written to env_cfg"


def test_step_time_drives_both_plants():
    """ROM and ID-FD perturb through different channels; one step time must
    configure whichever plant is active."""
    src = HARNESS.read_text()
    i = src.index("a.plant_step_time_s is not None")
    window = src[i:i + 500]
    assert "plant_perturbation_step_time_s" in window
    assert "rom_factor_perturbation_step_time_s" in window
