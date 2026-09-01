"""The perturbation must stay hidden from the controller.

The AAM identifies a plant change from the mismatch between what was COMMANDED
and how the system RESPONDED. That only works while:

  * the plant is driven by the perturbed (applied) valve, and
  * the AAM's ``previous_valve`` input carries the unperturbed (commanded) valve.

If someone later "tidies" the AAM input to use the applied valve, the mismatch
vanishes, the AAM has nothing to infer, and every adaptation result silently
becomes meaningless while all other tests still pass.

These are source-level invariants precisely because the failure is silent: it
produces no exception and no obviously wrong number.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ENV = REPO / "mulag_plant/runtime/mulag_upper_level_env.py"


def _source() -> str:
    return ENV.read_text()


def test_fd_is_driven_by_the_applied_valve():
    """The FD plant step must consume the perturbed command."""
    src = _source()
    m = re.search(r"fd_out\s*=\s*self\._cylinder_fd\.step\((.*?)\)", src, re.S)
    assert m, "could not locate the FD plant step"
    assert "valve_cmd=applied_valve_cmd" in m.group(1), (
        "the FD step must be driven by applied_valve_cmd; driving it with "
        "_current_valve_cmd means the plant is never actually perturbed"
    )


def test_aam_input_uses_the_commanded_valve():
    """The AAM feature row must carry the UNPERTURBED command."""
    src = _source()
    m = re.search(r"aam_row\s*=\s*torch\.cat\(\s*\((.*?)\),", src, re.S)
    assert m, "could not locate the AAM feature row construction"
    row = m.group(1)
    assert "self._prev_valve_cmd" in row, "AAM row must use _prev_valve_cmd"
    assert "applied" not in row, (
        "the AAM must NOT receive the applied/perturbed valve -- doing so hides "
        "the commanded/response mismatch it is supposed to identify"
    )


def test_prev_valve_cmd_is_cloned_from_the_command():
    """_prev_valve_cmd feeds the AAM, so it must track the command."""
    src = _source()
    assigns = re.findall(r"self\._prev_valve_cmd\s*=\s*([^\n]+)", src)
    assert assigns, "no assignment to _prev_valve_cmd found"
    meaningful = [a for a in assigns if "zeros" not in a]
    assert meaningful, "expected a non-initialisation assignment"
    for a in meaningful:
        assert "_current_valve_cmd" in a, (
            f"_prev_valve_cmd must be cloned from _current_valve_cmd, got {a!r}"
        )
        assert "applied" not in a, (
            f"_prev_valve_cmd must not come from the applied valve, got {a!r}"
        )


def test_perturbation_is_applied_after_the_controller():
    """The perturbation must sit downstream of ID and the AAM residual."""
    src = _source()
    aam_condition = src.index("self._online_aam.condition_valve(")
    apply_call = src.index("perturbation.apply(self._current_valve_cmd)")
    assert aam_condition < apply_call, (
        "the plant perturbation must be applied AFTER the AAM has conditioned "
        "the valve, otherwise the controller is compensating for a change it "
        "was allowed to observe"
    )


def test_perturbation_resets_with_the_episode():
    src = _source()
    assert "perturbation.reset(env_ids)" in src, (
        "the perturbation must reset per episode or the scheduled step change "
        "fires once and every later episode is silently unperturbed"
    )


def test_perturbation_is_off_by_default():
    cfg = (REPO / "mulag_plant/runtime/mulag_upper_level_env_cfg.py").read_text()
    assert "plant_perturbation_enabled = False" in cfg
    assert "plant_perturbation_alpha_range = (1.0, 1.0)" in cfg
    assert "plant_perturbation_delay_steps_range = (0, 0)" in cfg
