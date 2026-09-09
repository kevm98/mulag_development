#!/usr/bin/env python
"""Privileged-context ceiling test on the factor-conditioned command ROM.

The question this bench answers
-------------------------------
On a plant whose dependence on the hydraulic factor vector ``e`` is analytic,
strong and known -- the ``system_factors`` branch of ``HydraulicROM.forward``
(``learned_models/rom_model/ROM_pipeline/models/hydraulic_rom.py``) -- does a controller that
is TOLD the true ``e`` track better than one that is not?

Why this is worth running before any Simscape collection
--------------------------------------------------------
``reports/CONTROL_RELEVANT_ADAPTATION_LATENT.md`` recorded a NO-GO on the
privileged-context gate: correct context beat neither mean nor shuffled
context.  That evaluation ran on the frozen ID/FD plant, where the factor
vector has NO path into the dynamics at all (``CylinderFDRuntime.step`` takes
no factor argument), so correct context had nothing to exploit and the result
is uninformative about whether the task is adaptation-limited.

This bench removes that confound.  Here the plant's factor dependence is
maximal, noise-free and known in closed form.  If privileged context cannot win
HERE, it will not win on Simscape data either, and the adaptation line should
stop before a collection campaign is funded.  If it does win, the campaign is
justified and the measured effect size tells the excitation design what it has
to resolve.

What this bench is NOT
----------------------
It is not evidence about the real machine.  The factor -> behaviour map in the
ROM is asserted by hand (``bulk*exp(compliance)``, ``flow_gain*(1+flow)``,
``supply*(1+pressure)``, ``leakage*(1+0.25*seal)**3``, plus a
``factor_response_gain`` whose own comment says it exists so that factors
affect displacement and not only dP).  A model fitted to ROM rollouts learns to
invert an equation we already possess.  The ONLY claim available from this
bench is a negative one: an upper bound on how much knowing ``e`` can help.

Two stages
----------
``probe``   Training-free.  Sweeps a scalar controller gain per factor draw and
            compares the per-factor-optimal gain against one global gain.  The
            difference is the achievable benefit of knowing ``e`` inside that
            controller class, with no optimiser, seed or capacity confound.

``ceiling`` Learned.  Trains a bounded residual corrector by backpropagation
            through the differentiable ROM rollout, once conditioned on the
            true normalised factor and once blind, then evaluates the
            conditioned controller under true / mean / shuffled / zero context.

Factor sampling follows ``AAM_module/aam_data/master_schema.RANGES``, whose
bounds are exactly the clamps in ``HydraulicROM.forward``; sampling inside them
cannot clip, which keeps the factor effect from being confounded with
saturation (the failure mode audited in ``reports/factor_clamping_audit.json``,
where scale >= 1.5 clamps 13 of 16 rows).

``influence`` Training-free.  Reports which factor coordinates actually move
            closed-loop tracking error, one at a time and jointly.

Usage:
    python rom_privileged_ceiling.py probe     [--draws 64] [--ticks 200]
    python rom_privileged_ceiling.py ceiling   [--epochs 300]
    python rom_privileged_ceiling.py influence [--gain 2.4]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "learned_models/rom_model"))
sys.path.insert(0, str(ROOT / "AAM_module"))

from ROM_pipeline.models import load_checkpoint  # noqa: E402
from aam_data.master_schema import HYDRAULIC_DIMS, RANGES  # noqa: E402


DEFAULT_CHECKPOINT = ROOT / "learned_models/rom_model/plant_reachable_v1_finetune_blend25/best_rom.pt"
DT = 0.05

# Authoritative operating box, from the checkpoint's own training metadata
# (`cylinder_limits_from_training_extension`).  Rolling outside it would be
# extrapolation and any factor effect measured there would be uninterpretable.
CYLINDER_LIMITS = torch.tensor(
    [
        [0.01818416453897953, 0.49331751465797424],
        [0.27529862523078920, 0.79102206230163570],
        [0.30918031930923460, 0.85690146684646610],
        [0.26913416385650635, 0.68908423185348510],
    ]
)

# Per-cylinder DesiredDeltaExtension std from the checkpoint normalisation.
# Commands are scaled by these so every cylinder is driven inside its own
# in-distribution command band rather than a single global amplitude.
COMMAND_STD = torch.tensor([6.643025e-03, 7.693479e-03, 2.452950e-02, 1.733856e-02])


# ---------------------------------------------------------------------------
# factors
# ---------------------------------------------------------------------------

def sample_factors(n: int, seed: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw n hydraulic factor tuples.

    Returns (raw [n,4] in DIM order HYDRAULIC_DIMS, z [n,4] normalised to [-1,1]).
    """
    generator = torch.Generator().manual_seed(int(seed))
    lo = torch.tensor([RANGES[d][0] for d in HYDRAULIC_DIMS])
    hi = torch.tensor([RANGES[d][1] for d in HYDRAULIC_DIMS])
    unit = torch.rand((n, 4), generator=generator)
    raw = lo + unit * (hi - lo)
    z = 2.0 * unit - 1.0
    return raw.to(device), z.to(device)


def to_system_factors(raw: torch.Tensor) -> torch.Tensor:
    """[n,4] in HYDRAULIC_DIMS order -> [n,7] ROM channel order.

    Channel order is [compliance, flow, pressure, seal, seal, seal, seal],
    matching `master_schema.e_to_system_factors`.
    """
    idx = {name: i for i, name in enumerate(HYDRAULIC_DIMS)}
    flow = raw[:, idx["flow"] : idx["flow"] + 1]
    compliance = raw[:, idx["compliance"] : idx["compliance"] + 1]
    seal = raw[:, idx["seal"] : idx["seal"] + 1]
    pressure = raw[:, idx["pressure"] : idx["pressure"] + 1]
    return torch.cat([compliance, flow, pressure, seal, seal, seal, seal], dim=-1)


# ---------------------------------------------------------------------------
# reference trajectories
# ---------------------------------------------------------------------------

def make_reference(n: int, ticks: int, seed: int, device: torch.device) -> torch.Tensor:
    """Smooth multi-sine references, built in velocity space so the per-tick
    displacement is bounded by construction and stays inside ROM authority.

    Returns [ticks+1, n, 4] of target extensions.
    """
    generator = torch.Generator().manual_seed(int(seed))
    time = torch.arange(ticks, dtype=torch.float32).reshape(ticks, 1, 1) * DT
    # three components per cylinder in the 0.05-0.35 Hz band: slow enough to be
    # trackable, fast enough that hydraulic gain/leakage differences bite.
    freqs = 0.05 + 0.30 * torch.rand((3, n, 4), generator=generator)
    phases = 2.0 * torch.pi * torch.rand((3, n, 4), generator=generator)
    amps = 0.5 + torch.rand((3, n, 4), generator=generator)
    wave = sum(
        amps[k] * torch.sin(2.0 * torch.pi * freqs[k] * time + phases[k]) for k in range(3)
    )
    peak = wave.abs().amax(dim=0, keepdim=True).clamp_min(1e-9)
    # peak per-tick displacement = 1.5 command std for that cylinder
    step = 1.5 * COMMAND_STD.reshape(1, 1, 4)
    delta = wave / peak * step

    centre = CYLINDER_LIMITS.mean(dim=-1).reshape(1, 1, 4)
    half = 0.5 * (CYLINDER_LIMITS[:, 1] - CYLINDER_LIMITS[:, 0]).reshape(1, 1, 4)
    reference = torch.zeros((ticks + 1, n, 4))
    reference[0] = centre
    for t in range(ticks):
        nxt = reference[t] + delta[t]
        # keep well inside the trained box; 0.8 leaves margin for overshoot
        reference[t + 1] = torch.max(
            torch.min(nxt, centre + 0.8 * half), centre - 0.8 * half
        )
    return reference.to(device)


# ---------------------------------------------------------------------------
# plant rollout
# ---------------------------------------------------------------------------

def initial_state(n: int, model, device: torch.device):
    """Start at the centre of the trained extension box, at rest."""
    extension = CYLINDER_LIMITS.mean(dim=-1).reshape(1, 4).repeat(n, 1).to(device)
    velocity = torch.zeros((n, 4), device=device)
    # dP at the ROM's own training mean, so the pressure state starts in
    # distribution rather than at a physically meaningless zero.
    pressure = model.input_mean[model.schema.pressure_slice].reshape(1, 4).repeat(n, 1).to(device)
    prev_valve = torch.zeros((n, 4), device=device)
    return extension, velocity, pressure, prev_valve


def build_history(model, extension, velocity, pressure, prev_valve, command):
    row = torch.cat((extension, velocity, pressure, prev_valve, command), dim=-1)
    return row.unsqueeze(1).repeat(1, model.schema.history_length, 1)


def rollout(
    model,
    reference: torch.Tensor,
    system_factors: torch.Tensor,
    *,
    gain: torch.Tensor,
    residual: nn.Module | None = None,
    context: torch.Tensor | None = None,
    truncate: int = 20,
    device: torch.device,
) -> torch.Tensor:
    """Closed-loop tracking rollout. Returns per-run per-cylinder RMSE [n,4].

    `gain` is the proportional command gain, broadcastable to [n,4].
    `residual`, if given, adds a bounded correction to the command.
    `truncate` detaches the plant state every that many ticks so the learned
    stage has bounded-memory BPTT; it does not affect the forward trajectory.
    """
    ticks = reference.shape[0] - 1
    n = reference.shape[1]
    extension, velocity, pressure, prev_valve = initial_state(n, model, device)
    cmd_limit = (3.0 * COMMAND_STD).reshape(1, 4).to(device)
    command = torch.zeros((n, 4 + 4), device=device)
    history = build_history(model, extension, velocity, pressure, prev_valve, command)

    squared = torch.zeros((n, 4), device=device)
    for t in range(ticks):
        error = reference[t + 1] - extension
        base = gain * error
        if residual is not None:
            features = torch.cat(
                (
                    error / cmd_limit,
                    velocity,
                    pressure / 1.0e7,
                    prev_valve,
                    (reference[t + 1] - reference[t]) / cmd_limit,
                ),
                dim=-1,
            )
            if context is not None:
                features = torch.cat((features, context), dim=-1)
            base = base + residual(features)
        delta_x = torch.clamp(base, -cmd_limit, cmd_limit)
        command = torch.cat((delta_x, delta_x / DT), dim=-1)

        history = torch.cat((history[:, 1:], history[:, -1:].clone()), dim=1)
        history[:, -1, model.schema.extension_slice] = extension
        history[:, -1, model.schema.velocity_slice] = velocity
        history[:, -1, model.schema.pressure_slice] = pressure
        history[:, -1, model.schema.previous_valve_slice] = prev_valve
        history[:, -1, model.schema.command_slice] = command

        out = model(history, command, system_factors=system_factors)
        extension = out.next_extension
        velocity = out.next_velocity
        pressure = out.next_pressure
        prev_valve = out.predicted_valve

        squared = squared + (reference[t + 1] - extension) ** 2

        if truncate and (t + 1) % truncate == 0:
            extension = extension.detach()
            velocity = velocity.detach()
            pressure = pressure.detach()
            prev_valve = prev_valve.detach()
            history = history.detach()

    return torch.sqrt(squared / ticks)


# ---------------------------------------------------------------------------
# stage A: training-free gain probe
# ---------------------------------------------------------------------------

def bootstrap_ci(values: torch.Tensor, seed: int = 12345, samples: int = 10000):
    """Percentile bootstrap 95% CI of the mean of a paired difference vector."""
    generator = torch.Generator().manual_seed(int(seed))
    n = values.numel()
    idx = torch.randint(0, n, (samples, n), generator=generator)
    means = values.reshape(-1)[idx].mean(dim=1)
    lo = torch.quantile(means, 0.025).item()
    hi = torch.quantile(means, 0.975).item()
    return lo, hi


def stage_probe(args, model, device) -> dict:
    raw, z = sample_factors(args.draws, args.seed, device)
    system_factors = to_system_factors(raw)
    reference = make_reference(args.draws, args.ticks, args.seed + 1, device)

    grid = torch.linspace(args.gain_lo, args.gain_hi, args.gain_steps)
    table = torch.zeros((grid.numel(), args.draws), device=device)
    with torch.no_grad():
        for i, gain in enumerate(grid):
            rmse = rollout(
                model,
                reference,
                system_factors,
                gain=torch.full((args.draws, 4), float(gain), device=device),
                truncate=0,
                device=device,
            )
            table[i] = rmse.mean(dim=-1)
            print(f"  gain {float(gain):6.3f}  mean RMSE {table[i].mean().item():.6e}", flush=True)

    per_factor_best, best_index = table.min(dim=0)
    global_index = int(table.mean(dim=1).argmin().item())
    global_rmse = table[global_index]

    benefit = global_rmse - per_factor_best  # >= 0 by construction
    rel = (benefit / global_rmse.clamp_min(1e-12))
    lo, hi = bootstrap_ci(benefit.cpu())

    chosen = grid[best_index.cpu()]
    correlations = {}
    for i, name in enumerate(HYDRAULIC_DIMS):
        column = raw[:, i].cpu()
        if chosen.std() > 0 and column.std() > 0:
            stacked = torch.stack((chosen, column))
            correlations[name] = float(torch.corrcoef(stacked)[0, 1])
        else:
            correlations[name] = 0.0

    result = {
        "stage": "probe",
        "draws": args.draws,
        "ticks": args.ticks,
        "gain_grid": [float(g) for g in grid],
        "global_best_gain": float(grid[global_index]),
        "global_best_rmse_mean_m": float(global_rmse.mean()),
        "per_factor_best_rmse_mean_m": float(per_factor_best.mean()),
        "benefit_of_knowing_e_m": float(benefit.mean()),
        "benefit_relative_pct": float(100.0 * rel.mean()),
        "benefit_bootstrap95_m": [lo, hi],
        "per_factor_gain_min": float(chosen.min()),
        "per_factor_gain_max": float(chosen.max()),
        "per_factor_gain_std": float(chosen.std()),
        "gain_vs_factor_correlation": correlations,
        "rmse_spread_across_draws_at_global_gain_m": [
            float(global_rmse.min()), float(global_rmse.max())
        ],
    }
    return result


# ---------------------------------------------------------------------------
# stage B: learned bounded residual
# ---------------------------------------------------------------------------

class Residual(nn.Module):
    """Bounded command corrector, optionally conditioned on the factor.

    The 0.15*tanh bound mirrors the adapter used in the earlier privileged
    study so the two are comparable; it is scaled here into command units.
    """

    def __init__(self, feature_dim: int, bound: torch.Tensor, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 4),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.register_buffer("bound", bound.reshape(1, 4))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.bound * torch.tanh(self.net(features))


def train_residual(model, args, device, *, conditioned: bool):
    torch.manual_seed(args.seed)
    raw, z = sample_factors(args.draws, args.seed, device)
    system_factors = to_system_factors(raw)
    reference = make_reference(args.draws, args.ticks, args.seed + 1, device)
    bound = (1.5 * COMMAND_STD).to(device)
    feature_dim = 20 + (4 if conditioned else 0)
    residual = Residual(feature_dim, bound).to(device)
    optimiser = torch.optim.Adam(residual.parameters(), lr=args.lr)
    gain = torch.full((args.draws, 4), args.gain, device=device)

    for epoch in range(args.epochs):
        optimiser.zero_grad()
        rmse = rollout(
            model, reference, system_factors,
            gain=gain, residual=residual, context=z if conditioned else None,
            truncate=args.truncate, device=device,
        )
        loss = rmse.mean()
        loss.backward()
        nn.utils.clip_grad_norm_(residual.parameters(), 1.0)
        optimiser.step()
        if epoch % 25 == 0 or epoch == args.epochs - 1:
            tag = "ctx " if conditioned else "blind"
            print(f"  [{tag}] epoch {epoch:4d}  train RMSE {loss.item():.6e}", flush=True)
    return residual


def stage_ceiling(args, model, device) -> dict:
    blind = train_residual(model, args, device, conditioned=False)
    ctx = train_residual(model, args, device, conditioned=True)

    # held-out factor draws AND held-out references
    raw, z = sample_factors(args.eval_draws, args.seed + 500, device)
    system_factors = to_system_factors(raw)
    reference = make_reference(args.eval_draws, args.ticks, args.seed + 501, device)
    gain = torch.full((args.eval_draws, 4), args.gain, device=device)

    train_raw, train_z = sample_factors(args.draws, args.seed, device)
    mean_z = train_z.mean(dim=0, keepdim=True).repeat(args.eval_draws, 1)
    perm = torch.randperm(args.eval_draws, generator=torch.Generator().manual_seed(7))
    shuffled_z = z[perm.to(device)]
    zero_z = torch.zeros_like(z)

    arms = {}
    with torch.no_grad():
        arms["open_loop"] = rollout(
            model, reference, system_factors, gain=gain, truncate=0, device=device
        ).mean(dim=-1)
        arms["blind"] = rollout(
            model, reference, system_factors, gain=gain, residual=blind,
            truncate=0, device=device,
        ).mean(dim=-1)
        for name, context in (
            ("ctx_true", z), ("ctx_mean", mean_z),
            ("ctx_shuffled", shuffled_z), ("ctx_zero", zero_z),
        ):
            arms[name] = rollout(
                model, reference, system_factors, gain=gain, residual=ctx,
                context=context, truncate=0, device=device,
            ).mean(dim=-1)

    summary = {name: float(v.mean()) for name, v in arms.items()}
    comparisons = {}
    for other in ("blind", "ctx_mean", "ctx_shuffled", "ctx_zero"):
        diff = (arms[other] - arms["ctx_true"]).cpu()  # >0 means true context wins
        lo, hi = bootstrap_ci(diff)
        comparisons[f"{other}_minus_ctx_true"] = {
            "mean_m": float(diff.mean()),
            "bootstrap95_m": [lo, hi],
            "true_context_wins": bool(lo > 0.0),
        }

    gate = all(c["true_context_wins"] for c in comparisons.values())
    return {
        "stage": "ceiling",
        "train_draws": args.draws,
        "eval_draws": args.eval_draws,
        "ticks": args.ticks,
        "epochs": args.epochs,
        "gain": args.gain,
        "rmse_mean_m": summary,
        "paired_comparisons": comparisons,
        "gate_pass": gate,
    }


# ---------------------------------------------------------------------------
# stage C: which factors actually drive closed-loop performance
# ---------------------------------------------------------------------------

def stage_influence(args, model, device) -> dict:
    """Two independent reads on per-factor control relevance.

    (a) One-at-a-time sweep: hold three factors at their range midpoint, walk
        the fourth across its full range, and report the closed-loop tracking
        RMSE swing.  This is the closed-loop analogue of `rom_sensitivity.py`,
        which measures open-loop signal deviation instead.

    (b) Least-squares fit of RMSE on the normalised factor vector over random
        draws, giving each coordinate's average marginal effect in the presence
        of the others.

    A factor that moves neither number is not worth a context coordinate: the
    estimator would have to identify it and the policy would have nothing to do
    with it.
    """
    gain = args.gain

    # (a) one-at-a-time
    midpoint = torch.tensor(
        [0.5 * (RANGES[d][0] + RANGES[d][1]) for d in HYDRAULIC_DIMS], device=device
    )
    sweeps = {}
    steps = 9
    for i, name in enumerate(HYDRAULIC_DIMS):
        lo, hi = RANGES[name]
        values = torch.linspace(lo, hi, steps, device=device)
        raw = midpoint.reshape(1, 4).repeat(steps, 1).clone()
        raw[:, i] = values
        reference = make_reference(steps, args.ticks, args.seed + 1, device)
        # identical reference for every point on the sweep, so the only thing
        # that changes is the factor
        reference = reference[:, :1].repeat(1, steps, 1)
        with torch.no_grad():
            rmse = rollout(
                model, reference, to_system_factors(raw),
                gain=torch.full((steps, 4), gain, device=device),
                truncate=0, device=device,
            ).mean(dim=-1)
        sweeps[name] = {
            "values": [float(v) for v in values],
            "rmse_m": [float(v) for v in rmse],
            "swing_m": float(rmse.max() - rmse.min()),
            "swing_relative_pct": float(100.0 * (rmse.max() - rmse.min()) / rmse.mean()),
        }
        print(f"  sweep {name:<11} RMSE {rmse.min().item()*1000:7.3f} .. {rmse.max().item()*1000:7.3f} mm", flush=True)

    # (b) marginal effects over random draws
    raw, z = sample_factors(args.draws, args.seed + 900, device)
    reference = make_reference(args.draws, args.ticks, args.seed + 901, device)
    with torch.no_grad():
        rmse = rollout(
            model, reference, to_system_factors(raw),
            gain=torch.full((args.draws, 4), gain, device=device),
            truncate=0, device=device,
        ).mean(dim=-1)
    design = torch.cat((torch.ones((args.draws, 1), device=device), z), dim=-1)
    solution = torch.linalg.lstsq(design, rmse.reshape(-1, 1)).solution.reshape(-1)
    predicted = (design @ solution.reshape(-1, 1)).reshape(-1)
    ss_res = ((rmse - predicted) ** 2).sum()
    ss_tot = ((rmse - rmse.mean()) ** 2).sum()

    return {
        "stage": "influence",
        "gain": gain,
        "ticks": args.ticks,
        "draws": args.draws,
        "one_at_a_time": sweeps,
        "marginal_effect_m_per_unit_z": {
            name: float(solution[i + 1]) for i, name in enumerate(HYDRAULIC_DIMS)
        },
        "linear_fit_r2": float(1.0 - ss_res / ss_tot.clamp_min(1e-12)),
        "rmse_mean_m": float(rmse.mean()),
    }


# ---------------------------------------------------------------------------
# stage D: gradient-free ceiling in a richer controller class
# ---------------------------------------------------------------------------
# Backpropagation through this ROM rollout is not a usable descent direction.
# Measured on a 100-tick rollout: with the plant state detached every 20 ticks
# the gradient norm is ~1e-2, without truncation it is ~1e9 (the stiff pressure
# path explodes), and the hard command clamp zeroes the gradient on 21% of
# ticks.  A learning-rate sweep from 3e-3 to 3e-5 moved the training loss by at
# most -1.3% in 40 epochs and increased it in three of ten settings.  So the
# learned stage is run by population search instead, which needs no gradient
# and cannot be blamed for a null result.

CONTROLLER_PARAM_NAMES = ("kp_1", "kp_2", "kp_3", "kp_4", "kv_1", "kv_2", "kv_3", "kv_4")
PARAM_LOW = torch.tensor([0.0] * 4 + [-2.0] * 4)
PARAM_HIGH = torch.tensor([16.0] * 4 + [2.0] * 4)


def controlled_rollout(model, reference, system_factors, params, device):
    """Rollout under a per-run controller parameter vector `params` [n,8].

    delta_x = kp * tracking_error + kv * reference_step, then clamped.
    """
    ticks = reference.shape[0] - 1
    n = reference.shape[1]
    extension, velocity, pressure, prev_valve = initial_state(n, model, device)
    cmd_limit = (3.0 * COMMAND_STD).reshape(1, 4).to(device)
    command = torch.zeros((n, 8), device=device)
    history = build_history(model, extension, velocity, pressure, prev_valve, command)
    kp, kv = params[:, :4], params[:, 4:]
    squared = torch.zeros((n, 4), device=device)
    for t in range(ticks):
        error = reference[t + 1] - extension
        step = reference[t + 1] - reference[t]
        delta_x = torch.clamp(kp * error + kv * step, -cmd_limit, cmd_limit)
        command = torch.cat((delta_x, delta_x / DT), dim=-1)
        history = torch.cat((history[:, 1:], history[:, -1:].clone()), dim=1)
        history[:, -1, model.schema.extension_slice] = extension
        history[:, -1, model.schema.velocity_slice] = velocity
        history[:, -1, model.schema.pressure_slice] = pressure
        history[:, -1, model.schema.previous_valve_slice] = prev_valve
        history[:, -1, model.schema.command_slice] = command
        out = model(history, command, system_factors=system_factors)
        extension, velocity = out.next_extension, out.next_velocity
        pressure, prev_valve = out.next_pressure, out.predicted_valve
        squared = squared + (reference[t + 1] - extension) ** 2
    return torch.sqrt(squared / ticks).mean(dim=-1)


def cem(model, reference, system_factors, device, *, shared: bool, iters, population, elite, seed):
    """Cross-entropy search over controller parameters.

    shared=True  -> one parameter vector for every factor draw (context-blind).
    shared=False -> an independent vector per draw (true-context oracle).
    Both are evaluated on identical references and factors, so the difference
    between them is exactly the value of knowing e in this controller class.
    """
    n = reference.shape[1]
    generator = torch.Generator().manual_seed(int(seed))
    low, high = PARAM_LOW.to(device), PARAM_HIGH.to(device)
    rows = 1 if shared else n
    mean = ((low + high) / 2).reshape(1, 8).repeat(rows, 1)
    std = ((high - low) / 4).reshape(1, 8).repeat(rows, 1)
    best = None
    for iteration in range(iters):
        noise = torch.randn((population, rows, 8), generator=generator).to(device)
        candidates = torch.clamp(mean.unsqueeze(0) + std.unsqueeze(0) * noise, low, high)
        scores = torch.zeros((population, n), device=device)
        with torch.no_grad():
            for p in range(population):
                params = candidates[p].repeat(n, 1) if shared else candidates[p]
                scores[p] = controlled_rollout(model, reference, system_factors, params, device)
        ranking = scores.mean(dim=1, keepdim=True) if shared else scores
        order = ranking.argsort(dim=0)[:elite]
        chosen = torch.stack([candidates[order[e, r], r] for e in range(elite) for r in range(rows)])
        chosen = chosen.reshape(elite, rows, 8)
        mean, std = chosen.mean(dim=0), chosen.std(dim=0).clamp_min(1e-3)
        best = scores.min(dim=0).values if not shared else scores[ranking.reshape(-1).argmin()]
        tag = "shared" if shared else "per-factor"
        print(f"  cem[{tag}] iter {iteration:2d}  mean RMSE {best.mean().item()*1000:7.4f} mm", flush=True)
    return mean, best


def stage_search(args, model, device) -> dict:
    raw, z = sample_factors(args.eval_draws, args.seed + 500, device)
    system_factors = to_system_factors(raw)
    reference = make_reference(args.eval_draws, args.ticks, args.seed + 501, device)

    _, shared_rmse = cem(
        model, reference, system_factors, device, shared=True,
        iters=args.iters, population=args.population, elite=args.elite, seed=args.seed,
    )
    per_factor_params, per_factor_rmse = cem(
        model, reference, system_factors, device, shared=False,
        iters=args.iters, population=args.population, elite=args.elite, seed=args.seed + 1,
    )

    benefit = (shared_rmse - per_factor_rmse).cpu()
    lo, hi = bootstrap_ci(benefit)
    correlations = {}
    for i, name in enumerate(HYDRAULIC_DIMS):
        column = raw[:, i].cpu()
        for j in range(4):
            pass
        gains = per_factor_params[:, :4].mean(dim=-1).cpu()
        stacked = torch.stack((gains, column))
        correlations[name] = float(torch.corrcoef(stacked)[0, 1])

    return {
        "stage": "search",
        "controller": "per-cylinder kp and kv, 8 parameters, CEM",
        "draws": args.eval_draws,
        "ticks": args.ticks,
        "iters": args.iters,
        "population": args.population,
        "shared_rmse_mean_m": float(shared_rmse.mean()),
        "per_factor_rmse_mean_m": float(per_factor_rmse.mean()),
        "benefit_of_knowing_e_m": float(benefit.mean()),
        "benefit_relative_pct": float(100.0 * (benefit / shared_rmse.cpu().clamp_min(1e-12)).mean()),
        "benefit_bootstrap95_m": [lo, hi],
        "true_context_wins": bool(lo > 0.0),
        "mean_gain_vs_factor_correlation": correlations,
    }


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("stage", choices=("probe", "ceiling", "influence", "search"))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--draws", type=int, default=64)
    parser.add_argument("--eval-draws", dest="eval_draws", type=int, default=64)
    parser.add_argument("--ticks", type=int, default=200)
    parser.add_argument("--gain-lo", dest="gain_lo", type=float, default=0.05)
    parser.add_argument("--gain-hi", dest="gain_hi", type=float, default=1.50)
    parser.add_argument("--gain-steps", dest="gain_steps", type=int, default=16)
    parser.add_argument("--gain", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=3.0e-3)
    parser.add_argument("--truncate", type=int, default=20)
    parser.add_argument("--iters", type=int, default=12)
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--elite", type=int, default=8)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    model, _ = load_checkpoint(args.checkpoint, device=device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    print(f"checkpoint: {args.checkpoint}")
    print(f"stage: {args.stage}  draws={args.draws} ticks={args.ticks} device={args.device}")

    stages = {"probe": stage_probe, "ceiling": stage_ceiling, "influence": stage_influence, "search": stage_search}
    result = stages[args.stage](args, model, device)
    result["checkpoint"] = str(args.checkpoint)
    result["seed"] = args.seed

    print()
    print(json.dumps(result, indent=2))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
