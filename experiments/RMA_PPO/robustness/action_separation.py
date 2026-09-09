"""Analytic action separation: does knowing the plant change the required command?

This is the §8.2 oracle-gate question asked WITHOUT a plant model, so none of the
SD surrogate's positional drift enters (see results/STAGE0_FINDINGS.md).

The argument.  Every stress mechanism in stress.py sits between the commanded and
the applied valve; everything downstream of the applied valve is untouched.  So a
controller that knows the plant can hold the response invariant by commanding the
preimage of the nominal applied valve:

    nominal:   command u        -> applied u
    stressed:  command g_p^-1(u) -> applied u

The gap  du = g_p^-1(u) - u  is then the command change a context-conditioned
policy MUST produce to stay on the nominal response.  It is a lower bound on
action separation: a policy could also choose to accept a different response, but
it cannot achieve the nominal one with less.  Three things fall out of it:

  * SEPARATION -- du mapped through the frozen ID Jacobian into action units.
    If this is negligible the oracle arm cannot beat the mean arm on that axis,
    and the axis should not consume GPU hours (§8.3, §16-C).
  * FEASIBILITY -- the fraction of samples where |g_p^-1(u)| exceeds the ID's own
    command_limits.  That is actuator saturation derived analytically, which is
    the §12 confound and part of gate F3, with no surrogate involved.
  * INVERTIBILITY -- not every mechanism has a causal preimage.  A pure transport
    delay does not: the required command at t is the nominal command at t+k, which
    a causal controller cannot know.  Such an axis demands ANTICIPATION rather than
    a static remap, and that is precisely the regime where a learned latent has
    something to offer that reactive feedback cannot.

Inputs are recorded states and recorded valve commands from the frozen ROM
deployment dataset.  No policy is rolled out and no dynamics model is evaluated.

LIMITATIONS.  This measures the STATIC preimage only.  A mechanism that offers no
static separation may still reward anticipation -- a rate limit is the clear case:
you cannot command around it at time t, but a policy that knows about it can start
moving earlier.  Reading "NEGLIGIBLE" on rate_limit as "no adaptation opportunity
whatsoever" therefore overstates what was tested; what is established is that no
instantaneous command remap helps.  The delay axis is reported as ANTICIPATION
rather than NEGLIGIBLE because its preimage exists and is computable from recorded
data, just not causally available.

Every clipping figure is reported against the NOMINAL clipping floor, because the
recorded valve command already exceeds the ID's command limits on its own -- 5.6%
of rows overall and 21.2% on cylinder 1.  Read the +clip% and +C1% columns, not
the absolute ones, when attributing cost to the stress.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from robustness_env import DATASET_ROOT, DT_S, ID_CHECKPOINT, REPO

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

N_SAMPLES = 4096
SAMPLE_SEED = 20260825

# Pre-declared, frozen before any number below was looked at.
ACTION_SEPARATION_MIN = 0.02   # >= 2% of the full action span to count as nontrivial
SATURATION_MAX = 0.30          # mirrors the §6.3 headroom rule
ACTION_SPAN = 2.0              # the actor is bounded to [-1, 1]

SWEEPS = {
    "authority": [0.9, 0.8, 0.7, 0.6, 0.5, 0.3, 0.1],
    "deadband": [0.01, 0.02, 0.05, 0.10, 0.20],
    "lag": [0.05, 0.10, 0.20, 0.40, 1.00],
    "rate_limit": [1.0, 0.5, 0.2, 0.1, 0.05, 0.01],
    "delay": [1, 2, 4, 6, 8, 12, 20],
}


# ---------------------------------------------------------------------------
# Sampling: recorded states and recorded valve commands
# ---------------------------------------------------------------------------
def load_samples(n: int, seed: int) -> dict[str, torch.Tensor]:
    """Draw matched (state, prev_valve, valve, next_valves) rows from the dataset."""
    rng = np.random.default_rng(seed)
    root = DATASET_ROOT / "sd" / "dataset"
    runs = sorted(p for p in root.glob("*/samples.npz"))
    max_lead = max(SWEEPS["delay"])

    s, u_prev, u_now, u_lead = [], [], [], []
    per_run = max(1, n // max(len(runs), 1) + 1)
    for path in runs:
        with np.load(path) as npz:
            state = npz["s_h"].astype(np.float32)
            valve = npz["u"].astype(np.float32)
        usable = len(state) - max_lead - 1
        if usable <= 1:
            continue
        idx = rng.integers(1, usable, size=min(per_run, usable - 1))
        s.append(state[idx])
        u_prev.append(valve[idx, :4])
        u_now.append(valve[idx, 4:8])
        # applied valve k steps ahead, for the non-causal delay preimage
        u_lead.append(np.stack([valve[idx + k, 4:8] for k in range(max_lead + 1)], axis=1))

    out = {
        "s": torch.from_numpy(np.concatenate(s)[:n]),
        "u_prev": torch.from_numpy(np.concatenate(u_prev)[:n]),
        "u": torch.from_numpy(np.concatenate(u_now)[:n]),
        "u_lead": torch.from_numpy(np.concatenate(u_lead)[:n]),
    }
    return out


# ---------------------------------------------------------------------------
# Causal preimages of the stress chain
# ---------------------------------------------------------------------------
def required_command(axis: str, severity: float, sample: dict[str, torch.Tensor]
                     ) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Command needed to deliver the recorded applied valve under this stress.

    Returns (required command, per-element achievable mask, causally invertible).
    """
    u = sample["u"]
    u_prev = sample["u_prev"]
    ok = torch.ones_like(u, dtype=torch.bool)

    if axis == "authority":
        return u / severity, ok, True

    if axis == "deadband":
        # applied = sign(u) * max(|u| - d, 0);  a nonzero target needs |u| + d
        return torch.sign(u) * (u.abs() + severity), ok, True

    if axis == "lag":
        # applied_t = applied_{t-1} + beta * (command_t - applied_{t-1})
        beta = 1.0 - float(np.exp(-DT_S / severity))
        return u_prev + (u - u_prev) / beta, ok, True

    if axis == "rate_limit":
        # the applied valve cannot move further than r per tick from its last value
        step = (u - u_prev).abs()
        ok = step <= severity
        clipped = u_prev + torch.clamp(u - u_prev, -severity, severity)
        return clipped, ok, True

    if axis == "delay":
        # applied_t = command_{t-k}, so the required command at t is the nominal
        # applied valve at t+k -- information a causal controller does not have.
        return sample["u_lead"][:, int(severity), :], ok, False

    raise ValueError(axis)


# ---------------------------------------------------------------------------
# ID Jacobian: valve units -> action units
# ---------------------------------------------------------------------------
def id_jacobian(sample: dict[str, torch.Tensor], eps: float = 0.05) -> torch.Tensor:
    """d(valve command) / d(action), by central differences through the frozen ID.

    Evaluated at a = 0, i.e. the local sensitivity at the recorded state.  The ID
    is a stateless MLP over the deployed 28-D contract, so this is exact up to the
    finite-difference step.
    """
    from mulag_plant.runtime.cylinder_models_runtime import CylinderIDRuntime

    s, u_prev = sample["s"], sample["u_prev"]
    n = s.shape[0]
    runtime = CylinderIDRuntime(
        models_root=REPO / "learned_models/id_fd_model/vendor_models", checkpoint=ID_CHECKPOINT,
        num_envs=n, device=torch.device("cpu"),
    )

    def valve_of(action: torch.Tensor) -> torch.Tensor:
        dx = action * 0.004 * torch.tensor([-1.0, -1.0, -1.0, -1.0])
        dxv = dx / DT_S - s[:, 4:8]
        row = torch.cat((s[:, :4], s[:, 4:8], s[:, 8:12], torch.zeros_like(s[:, :4]),
                         dx, dxv, u_prev), dim=-1)
        return runtime.predict(row, prev_valve_cmd=u_prev)

    cols = []
    for j in range(4):
        e = torch.zeros(n, 4)
        e[:, j] = eps
        cols.append((valve_of(e) - valve_of(-e)) / (2.0 * eps))
    return torch.stack(cols, dim=-1)          # [n, 4 valve, 4 action]


def action_delta(jac: torch.Tensor, dv: torch.Tensor) -> torch.Tensor:
    """Least-squares action change realizing a commanded valve change."""
    sol = torch.linalg.lstsq(jac, dv.unsqueeze(-1)).solution
    return sol.squeeze(-1)


# ---------------------------------------------------------------------------
def main() -> int:
    t0 = time.time()
    torch.manual_seed(SAMPLE_SEED)
    sample = load_samples(N_SAMPLES, SAMPLE_SEED)
    n = sample["s"].shape[0]
    print(f"{n} recorded rows from the frozen ROM deployment dataset")

    jac = id_jacobian(sample)
    cond = torch.linalg.cond(jac)
    print(f"ID Jacobian |dValve/dAction|: median gain={jac.abs().median():.4f}  "
          f"median condition number={cond.median():.1f}")

    from mulag_plant.runtime.cylinder_models_runtime import CylinderIDRuntime
    limits = CylinderIDRuntime(
        models_root=REPO / "learned_models/id_fd_model/vendor_models", checkpoint=ID_CHECKPOINT,
        num_envs=1, device=torch.device("cpu"),
    ).command_limits.reshape(1, 4)
    print(f"ID command limits: {limits.flatten().tolist()}")

    # The recorded valve already exceeds the ID's command limits on its own, so
    # every clipping figure below must be read against this floor, not against 0.
    nominal_over = sample["u"].abs() > limits
    nominal_clip = float(nominal_over.float().mean())
    nominal_clip_per_cyl = nominal_over.float().mean(dim=0)
    print(f"NOMINAL clipping floor (no stress): {nominal_clip*100:.1f}% overall, "
          f"per cylinder {[round(float(v)*100, 1) for v in nominal_clip_per_cyl]}%")

    report = {
        "screen": "analytic_action_separation",
        "n_samples": n,
        "sample_seed": SAMPLE_SEED,
        "action_separation_min": ACTION_SEPARATION_MIN,
        "saturation_max": SATURATION_MAX,
        "command_limits": limits.flatten().tolist(),
        "median_jacobian_condition": float(cond.median()),
        "nominal_clipped_fraction": nominal_clip,
        "nominal_clipped_per_cylinder": nominal_clip_per_cyl.tolist(),
        "axes": {},
    }

    for axis, grid in SWEEPS.items():
        print(f"\n=== {axis} ===")
        print(f"{'severity':>10}{'|du| med':>10}{'|du| p90':>10}{'|da|/span':>11}"
              f"{'infeas%':>9}{'clip%':>8}{'+clip%':>8}{'C1%':>7}{'+C1%':>7}  verdict")
        rows = []
        for sev in grid:
            u_req, ok, causal = required_command(axis, float(sev), sample)
            dv = u_req - sample["u"]

            over = u_req.abs() > limits
            clip_frac = float(over.float().mean())
            clip_per_cyl = over.float().mean(dim=0)
            infeasible = float((~ok).float().mean())

            # separation is only meaningful where the command is actually deliverable
            deliverable = (~over.any(dim=-1)) & ok.all(dim=-1)
            if int(deliverable.sum()) >= 32:
                da = action_delta(jac[deliverable], dv[deliverable])
                sep = float(da.abs().amax(dim=-1).median()) / ACTION_SPAN
            else:
                sep = float("nan")

            if clip_frac > SATURATION_MAX or infeasible > SATURATION_MAX:
                verdict = "INFEASIBLE"
            elif not (sep >= ACTION_SEPARATION_MIN):
                verdict = "NEGLIGIBLE"
            elif not causal:
                verdict = "ANTICIPATION"
            else:
                verdict = "SEPARATION"

            row = {
                "severity": sev,
                "causally_invertible": causal,
                "valve_delta_median": float(dv.abs().amax(dim=-1).median()),
                "valve_delta_p90": float(dv.abs().amax(dim=-1).quantile(0.90)),
                "action_separation_fraction": sep,
                "infeasible_fraction": infeasible,
                "clipped_fraction": clip_frac,
                "clipped_fraction_excess": clip_frac - nominal_clip,
                "clipped_per_cylinder": clip_per_cyl.tolist(),
                "clipped_c1_excess": float(clip_per_cyl[1] - nominal_clip_per_cyl[1]),
                "deliverable_fraction": float(deliverable.float().mean()),
                "verdict": verdict,
            }
            rows.append(row)
            print(f"{sev:>10}{row['valve_delta_median']:>10.4f}{row['valve_delta_p90']:>10.4f}"
                  f"{sep:>11.4f}{infeasible*100:>9.1f}{clip_frac*100:>8.1f}"
                  f"{row['clipped_fraction_excess']*100:>8.1f}{float(clip_per_cyl[1])*100:>7.1f}"
                  f"{row['clipped_c1_excess']*100:>7.1f}  {verdict}")

        usable = [r for r in rows if r["verdict"] in ("SEPARATION", "ANTICIPATION")]
        report["axes"][axis] = {
            "points": rows,
            "usable_severities": [r["severity"] for r in usable],
            "usable": bool(usable),
        }

    (OUT / "action_separation.json").write_text(json.dumps(report, indent=2))
    print("\n=== summary: severities with real, deliverable action separation ===")
    for axis, a in report["axes"].items():
        state = "OK   " if a["usable"] else "NONE "
        print(f"  {axis:<12}{state}{a['usable_severities']}")
    print(f"\n-> {OUT/'action_separation.json'}  ({time.time()-t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
