"""`policy_latent_mode='lpf'` band-limits the latent on the policy path.

Any use of this arm rests on one property: it must carry the SAME latent content
as `estimated` and differ only in how fast that content is allowed to move. If
the filter also shifted or attenuated the signal's mean, a difference measured
against `estimated` would be a content change wearing a bandwidth label, and it
would answer nothing the mean/shuffled arms had not already confounded.

Three failure modes are worth a test more than the algebra is:

* double integration, if the state were advanced in the observation builder
  (which runs more than once per control step) instead of in the pipeline;
* a startup transient, if the first sample were filtered towards zero rather
  than seeding the state -- it grows as the cutoff falls, so it would masquerade
  as a bandwidth effect;
* a stale latch surviving a reset, which would leak one episode into the next.

The tests bind the real methods to a lightweight stub, so they need no AAM
checkpoint, no adapter and no simulator.
"""
from __future__ import annotations

import ast
import math
import sys
import textwrap
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

ENV = REPO / "mulag_plant/runtime/mulag_upper_level_env.py"


def _methods(*names):
    """Pull the named methods out of the env source and compile them alone.

    Importing the module would pull in Isaac Sim, which is not available under
    plain pytest. Extracting by AST keeps these tests bound to the real source
    rather than to a copy that could drift away from it.
    """
    tree = ast.parse(ENV.read_text())
    cls = next(
        n for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "MulagUpperLevelEnv"
    )
    wanted = {}
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            src = textwrap.dedent(ast.get_source_segment(ENV.read_text(), node))
            ns = {"torch": torch, "math": math}
            exec(compile(src, str(ENV), "exec"), ns)
            wanted[node.name] = ns[node.name]
    missing = set(names) - set(wanted)
    assert not missing, f"methods vanished from the env source: {sorted(missing)}"
    return wanted


_M = _methods(
    "_policy_latent_lpf_alpha",
    "_advance_policy_latent_lpf",
    "_policy_latent_observation",
)


class _Cfg:
    def __init__(self, mode: str, hz: float):
        self.online_aam_policy_latent_mode = mode
        self.online_aam_policy_latent_lpf_hz = hz


class _Stub:
    """Carries only what the lpf methods touch."""

    def __init__(self, num_envs: int = 4, hz: float = 1.0, dt: float = 1.0 / 60.0,
                 mode: str = "lpf"):
        self.cfg = _Cfg(mode, hz)
        self.num_envs = num_envs
        self._control_dt = dt
        self._aam_z_hat_raw = torch.zeros((num_envs, 2))
        self._aam_z_hat = torch.zeros((num_envs, 2))
        self._aam_privileged_z = torch.zeros((num_envs, 2))
        self._aam_policy_z_lpf = torch.zeros((num_envs, 2))
        self._aam_policy_z_lpf_valid = torch.zeros(num_envs, dtype=torch.bool)
        self._online_aam = None

    # Bound under their real names, because the extracted methods call each
    # other through self; the short aliases are only for readability below.
    _policy_latent_lpf_alpha = _M["_policy_latent_lpf_alpha"]
    _advance_policy_latent_lpf = _M["_advance_policy_latent_lpf"]
    _policy_latent_observation = _M["_policy_latent_observation"]

    alpha = _policy_latent_lpf_alpha
    advance = _advance_policy_latent_lpf
    observe = _policy_latent_observation


def test_alpha_matches_the_one_pole_definition():
    s = _Stub(hz=1.0, dt=1.0 / 60.0)
    tau = 1.0 / (2.0 * math.pi * 1.0)
    assert s.alpha() == pytest.approx((1 / 60) / (tau + 1 / 60))


def test_cutoff_must_be_declared():
    s = _Stub(hz=0.0)
    with pytest.raises(ValueError, match="lpf_hz > 0"):
        s.alpha()


def test_first_sample_seeds_the_state_rather_than_ramping_from_zero():
    s = _Stub(num_envs=2, hz=0.05)  # a very slow filter: a ramp would be obvious
    z = torch.tensor([[-1.19, -2.34], [-1.00, -2.00]])
    s._aam_z_hat_raw = z.clone()
    s.advance()
    assert torch.allclose(s.observe(), z), (
        "the first sample of an episode must seed the filter; filtering it "
        "towards zero injects a transient that grows as the cutoff falls"
    )


def test_observation_is_read_only_so_repeated_builds_cannot_double_integrate():
    s = _Stub(num_envs=1, hz=2.0)
    s._aam_z_hat_raw = torch.tensor([[0.0, 0.0]])
    s.advance()                       # seed at zero
    s._aam_z_hat_raw = torch.tensor([[1.0, 1.0]])
    s.advance()                       # exactly one control step of filtering
    once = s.observe().clone()
    for _ in range(5):                # the terrain task rebuilds the observation
        assert torch.allclose(s.observe(), once), (
            "building the observation must not advance the filter"
        )


def test_step_response_approaches_the_raw_value_at_the_expected_rate():
    """After one time constant a one-pole is within ~1/e of the step."""
    hz, dt = 1.0, 1.0 / 60.0
    s = _Stub(num_envs=1, hz=hz, dt=dt)
    s._aam_z_hat_raw = torch.tensor([[0.0, 0.0]])
    s.advance()
    s._aam_z_hat_raw = torch.tensor([[1.0, 1.0]])
    tau = 1.0 / (2.0 * math.pi * hz)
    for _ in range(int(round(tau / dt))):
        s.advance()
    reached = float(s.observe()[0, 0])
    assert reached == pytest.approx(1.0 - math.exp(-1.0), abs=0.02)


def test_a_low_cutoff_moves_less_than_a_high_one():
    """The arm is only interpretable if it is monotone in the cutoff."""
    z0 = torch.tensor([[-1.19, -2.34]])
    z1 = torch.tensor([[-0.19, -1.34]])
    travelled = []
    for hz in (0.05, 0.5, 5.0):
        s = _Stub(num_envs=1, hz=hz)
        s._aam_z_hat_raw = z0.clone()
        s.advance()
        s._aam_z_hat_raw = z1.clone()
        for _ in range(30):
            s.advance()
        travelled.append(float((s.observe() - z0).abs().sum()))
    assert travelled[0] < travelled[1] < travelled[2]


def test_unfiltered_arms_are_untouched_by_the_advance():
    """Every other arm must stay bit-identical to its pre-lpf behaviour."""
    for mode in ("estimated", "mean", "shuffled", "zero"):
        s = _Stub(num_envs=3, hz=0.0, mode=mode)
        s._aam_z_hat_raw = torch.randn(3, 2)
        s.advance()  # must be a no-op, and must not trip the cutoff check
        assert not bool(s._aam_policy_z_lpf_valid.any())


def test_estimated_is_returned_before_the_filter_has_a_sample():
    s = _Stub(num_envs=2, hz=1.0)
    s._aam_z_hat_raw = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    assert torch.equal(s.observe(), s._aam_z_hat_raw), (
        "with no seeded sample the arm must fall through to the raw estimate, "
        "not to zero"
    )
