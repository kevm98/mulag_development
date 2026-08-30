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


# --------------------------------------------------------------------------
# The latent modes are declared in THREE places: the runtime's
# `_apply_latent_mode`, the env's `_policy_latent_observation`, and the two
# argparse choice lists. They drifted: the controller list omitted `mean`,
# which the runtime has always supported, so the in-distribution
# no-information control could not be run and the arm died at startup after
# the GPU work had already been scheduled.

import sys as _sys

_sys.path.insert(0, str(REPO))

ENV_SRC = (REPO / "mulag_plant/runtime/mulag_upper_level_env.py").read_text()
RUNTIME_SRC = (REPO / "PPO_AAM_FD/online_runtime.py").read_text()


def _runtime_modes() -> set[str]:
    """The tuple the runtime validates latent_mode against."""
    tree = ast.parse(RUNTIME_SRC)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign)
                and any(getattr(t, "id", "") == "_LATENT_MODES" for t in node.targets)):
            return {e.value for e in node.value.elts}
    raise AssertionError("runtime no longer declares _LATENT_MODES")


def test_controller_choices_do_not_exclude_a_supported_mode():
    """Anything the runtime accepts must be reachable from the harness."""
    missing = _runtime_modes() - _choices("--online_aam_controller_latent_mode")
    assert not missing, (
        f"the runtime supports {sorted(missing)} but the harness cannot ask "
        "for it; the arm dies at argparse after the run is already scheduled"
    )


def test_policy_choices_cover_the_modes_the_env_implements():
    """Same drift risk on the policy side."""
    fn = next(n for n in ast.walk(ast.parse(ENV_SRC))
              if isinstance(n, ast.FunctionDef)
              and n.name == "_policy_latent_observation")
    body = ast.unparse(fn)
    implemented = {m for m in ("estimated", "zero", "mean", "shuffled", "frozen")
                   if f"'{m}'" in body or f'"{m}"' in body}
    missing = implemented - _choices("--online_aam_latent_mode")
    assert not missing, f"policy modes {sorted(missing)} are unreachable"
