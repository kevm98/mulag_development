"""The routing ablation needs the CONTROLLER and POLICY latents to be separable.

The grid is a 2x2 over where the latent is routed:

    A1 baseline   policy zero       controller zero
    A2 ID only    policy zero       controller estimated
    A3 PPO only   policy estimated  controller zero
    A4 full       policy estimated  controller estimated

A3 is the arm that catches the failure this file guards. The runtime used to
keep one `z_hat` tensor and overwrite it in place with the ablated value, so
setting the controller arm to `zero` also zeroed what the policy could read.
A3 then produced exactly A1's observation and scored as the baseline -- a null
result that looks like "the policy does not use the latent" but is really the
harness measuring nothing. Nothing in the report distinguished the two.

Two properties, tested at the two levels where they can break:

  RUNTIME   the raw estimate survives a degraded controller mode
  ENV       the policy observation is built from the raw estimate

`estimated` remains bit-identical to the pre-split behaviour, so existing
promotion evidence (controller mode `estimated` in every arm) is unaffected.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from PPO_ID_AAM_FD.online_runtime import OnlineAAMIDFDController  # noqa: E402

ENV = REPO / "mulag_plant/runtime/mulag_upper_level_env.py"


class _Stub:
    """Carries only what `_apply_latent_mode` touches."""

    def __init__(self, num_envs: int, mode: str, freeze_tick: int = 0):
        self.latent_mode = mode
        self.latent_freeze_tick = freeze_tick
        self.control_ticks = torch.zeros(num_envs, dtype=torch.long)
        self.z_frozen = torch.zeros((num_envs, 2))
        self.z_frozen_valid = torch.zeros(num_envs, dtype=torch.bool)
        self.true_z = torch.zeros((num_envs, 2))

    apply = OnlineAAMIDFDController._apply_latent_mode


# --------------------------------------------------------------- runtime ---

@pytest.mark.parametrize("mode", ["zero", "shuffled", "mean"])
def test_degrading_the_controller_does_not_consume_the_estimate(mode):
    """`_apply_latent_mode` must be a pure function of its argument.

    If it mutated the estimate in place, or if the caller assigned the result
    back over `z_hat`, the raw value would be gone for the policy arm.
    """
    s = _Stub(num_envs=4, mode=mode)
    z = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    original = z.clone()
    s.apply(z)
    assert torch.equal(z, original), (
        f"latent_mode={mode!r} modified the estimate in place"
    )


def test_zero_arm_returns_zeros_but_leaves_the_input_intact():
    s = _Stub(num_envs=3, mode="zero")
    z = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    out = s.apply(z)
    assert torch.count_nonzero(out) == 0
    assert torch.count_nonzero(z) == 6, "the estimate itself must survive"


def test_estimated_is_the_identity():
    """Bit-identical, not merely close: every promotion arm runs through here."""
    s = _Stub(num_envs=3, mode="estimated")
    z = torch.randn(3, 2)
    assert s.apply(z) is z


def test_observe_publishes_and_raw_latent_reports_separately():
    """Source-level: the two must be distinct attributes on distinct paths.

    Constructing the controller needs an AAM checkpoint and an adapter, so the
    wiring is checked in the source instead of by instantiation.
    """
    src = Path(REPO / "PPO_ID_AAM_FD/online_runtime.py").read_text()
    assert "self.z_published = self._apply_latent_mode(self.z_hat)" in src, (
        "the ablated latent must be published to a SEPARATE tensor; assigning "
        "it back over z_hat is the defect this file exists for"
    )
    assert "def raw_latent" in src, "policy arm has no way to read the estimate"
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "raw_latent")
    returned = {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)}
    assert "z_hat" in returned and "z_published" not in returned


def test_the_controller_consumes_the_published_latent():
    """The adapter feature vector must carry the ABLATED latent, or the
    controller arm of the grid does nothing."""
    src = Path(REPO / "PPO_ID_AAM_FD/online_runtime.py").read_text()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "condition_valve")
    attrs = {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)}
    assert "z_published" in attrs, "controller is reading the un-ablated estimate"
    assert "z_hat" not in attrs


# ------------------------------------------------------------------- env ---

def _policy_latent_fn() -> ast.FunctionDef:
    tree = ast.parse(ENV.read_text())
    return next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef)
                and n.name == "_policy_latent_observation")


def test_policy_observation_is_built_from_the_raw_estimate():
    attrs = {n.attr for n in ast.walk(_policy_latent_fn())
             if isinstance(n, ast.Attribute)}
    assert "_aam_z_hat_raw" in attrs, (
        "the policy arm reads the controller's published latent, so A3 "
        "(policy estimated / controller zero) degrades to A1"
    )


def test_policy_frozen_mode_exists_and_latches_independently():
    """A shared latch would tie the two arms together; the grid needs them
    independent (policy frozen at controller estimated, and vice versa)."""
    fn = _policy_latent_fn()
    src = ast.unparse(fn)
    assert "frozen" in src, "policy arm cannot express the online-adaptation control"
    assert "_aam_policy_z_frozen" in src
    assert "self._online_aam.z_frozen" not in src, "policy reuses the controller latch"


def test_policy_frozen_requires_a_freeze_tick():
    """Default 0 would latch at tick 0 -- before the AAM is even valid -- and
    silently produce a zero latent for the whole episode."""
    src = ast.unparse(_policy_latent_fn())
    assert "online_aam_latent_freeze_tick" in src
    assert "tick < 1" in src or "tick <= 0" in src


def test_freeze_tick_is_a_declared_cfg_field():
    """override_cfg refuses to create fields, so an undeclared one makes the
    whole frozen arm unreachable from the evaluation harness."""
    cfg = (REPO / "PPO_ID_AAM_FD/env_cfg.py").read_text()
    assert "online_aam_latent_freeze_tick" in cfg


@pytest.mark.parametrize("field", [
    "plant_perturbation_enabled",
    "plant_perturbation_alpha_range",
    "plant_perturbation_delay_steps_range",
    "plant_perturbation_step_time_s",
    "plant_perturbation_step_alpha_scale",
    "rom_factor_perturbation_enabled",
    "rom_factor_perturbation_ranges",
    "rom_factor_perturbation_step_time_s",
    "rom_factor_perturbation_step_values",
    "adaptation_trace_path",
])
def test_phase4_cfg_fields_are_declared(field):
    """Same failure mode as the freeze tick: the harness parses the flag, the
    cfg rejects it, and the run either dies or scores as the baseline."""
    cfg = (REPO / "mulag_plant/runtime/mulag_upper_level_env_cfg.py").read_text()
    assert field in cfg, f"{field} is not a declared cfg field"
