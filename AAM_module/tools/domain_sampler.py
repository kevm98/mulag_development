#!/usr/bin/env python
"""Draw per-run physical factor vectors for the Stage 6 deployment domains.

Space-filling draws (scrambled Sobol by default, Latin Hypercube as an
alternative) inside the Stage 5 verified ranges. One draw per run; factors are
constant within a run. D_TEST is rejection-sampled with a different seed so its
combinations never collide with the training domains.

Usage:
    python domain_sampler.py [--train 4000] [--test 600]
        [--method sobol|lhs] [--out results/domain_manifest_sample.csv]
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

import torch
from torch.quasirandom import SobolEngine


# Stage 5 verified ranges (dimension order: flow, compliance, seal, pressure,
# mass_scale, com_z). These are the audited ranges; do not invent new ones.
RANGES = {
    "flow": (-0.10, 0.10),
    "compliance": (-math.log(1.1), math.log(1.1)),
    "seal": (0.0, 0.30),
    "pressure": (-0.10, 0.10),
    "mass_scale": (0.8, 1.5),
    "com_z": (-0.10, 0.05),
}
STAGE22_WIDE_RANGES = {
    # Audited hard limits in HydraulicROM.forward.  These deliberately use
    # the available physical-factor authority after the v2 data showed that
    # the narrower Stage-5 box was not distinguishable in delta prediction.
    "flow": (-0.25, 0.25),
    "compliance": (-0.25, 0.10),
    "seal": (0.0, 1.0),
    "pressure": (-0.15, 0.15),
    # Mechanics are not widened until a separate extreme-configuration
    # Isaac flight establishes safety; they are not the Stage-22 bottleneck.
    "mass_scale": (0.8, 1.5),
    "com_z": (-0.10, 0.05),
}
DIM_ORDER = ("flow", "compliance", "seal", "pressure", "mass_scale", "com_z")

# Dimensions that vary per domain; every other dim keeps its nominal value.
DOMAIN_DIMS = {
    "nominal": (),
    "hydraulic": ("flow", "compliance", "seal", "pressure"),
    "mechanical": ("mass_scale", "com_z"),
    "combined": DIM_ORDER,
}
# Fixed seeds. Note: Python's hash() on strings is salted per process, so the
# old sampler was not reproducible across runs; explicit seeds fix that.
DOMAIN_SEEDS = {"nominal": 101, "hydraulic": 202, "mechanical": 303, "combined": 404}
TEST_SEED = 20260807


def nominal_e() -> dict[str, float]:
    e = {name: 0.0 for name in DIM_ORDER}
    e["mass_scale"] = 1.0
    return e


def round_key(e: dict[str, float]) -> tuple:
    return tuple(round(e[name], 4) for name in DIM_ORDER)


def unit_to_e(
    unit: tuple[float, ...], active: tuple[str, ...], ranges: dict[str, tuple[float, float]] = RANGES
) -> dict[str, float]:
    """Map unit-cube coordinates onto the Stage 5 ranges for the active dims."""
    e = nominal_e()
    for name, u in zip(active, unit):
        lo, hi = ranges[name]
        e[name] = lo + u * (hi - lo)
    return e


def unit_stream(active: tuple[str, ...], method: str, seed: int, block: int):
    """Infinite stream of unit-cube points with space-filling structure."""
    dim = len(active)
    if dim == 0:
        while True:
            yield ()
        return
    if method == "sobol":
        engine = SobolEngine(dim, scramble=True, seed=seed)
        while True:
            yield tuple(float(v) for v in engine.draw(1)[0])
    elif method == "lhs":
        rng = random.Random(seed)
        while True:
            # One LHS block: each column holds one point per stratum.
            perms = [rng.sample(range(block), block) for _ in range(dim)]
            for i in range(block):
                yield tuple((perms[d][i] + rng.random()) / block for d in range(dim))
    else:
        raise ValueError(f"unknown method: {method}")


def sample(domain: str, count: int, seed: int, method: str,
           exclude: set | None = None,
           ranges: dict[str, tuple[float, float]] = RANGES,
           predicate=None) -> list[dict]:
    """Draw `count` disjoint factor vectors for one domain."""
    if domain == "nominal":
        # Single configuration by definition; repeated runs are the baseline set.
        return [nominal_e() for _ in range(count)]
    active = DOMAIN_DIMS[domain]
    seen = set() if exclude is None else set(exclude)
    rows = []
    gen = unit_stream(active, method, seed, count)
    guard = 0
    while len(rows) < count and guard < count * 200:
        e = unit_to_e(next(gen), active, ranges)
        if predicate is not None and not predicate(e):
            guard += 1
            continue
        key = round_key(e)
        if key in seen:
            guard += 1
            continue
        seen.add(key)
        rows.append(e)
    if len(rows) < count:
        raise RuntimeError(f"could not draw {count} disjoint samples for {domain}")
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=int, default=4000)
    p.add_argument("--test", type=int, default=600)
    p.add_argument("--method", choices=("sobol", "lhs"), default="sobol")
    p.add_argument(
        "--final",
        action="store_true",
        help="Stage 15 small informative manifest (20-30 runs), not the 4600-run plan",
    )
    p.add_argument(
        "--stage22-wide",
        action="store_true",
        help="Stage-22 recollection design: 118 matched-excitation runs using "
        "the ROM's audited hydraulic-factor limits",
    )
    p.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results/domain_manifest_sample.csv"))
    args = p.parse_args()

    if args.final and args.stage22_wide:
        raise SystemExit("--final and --stage22-wide are mutually exclusive")
    ranges = STAGE22_WIDE_RANGES if args.stage22_wide else RANGES
    if args.stage22_wide:
        # 102 train-pool configurations plus 16 reserved combinations.  The
        # allocation concentrates coverage where the SD target lives while
        # retaining the Stage-6 mechanical and combined domains.
        allocations = {
            "nominal": 2,
            "hydraulic": 40,
            "mechanical": 12,
            "combined": 48,
        }
        test_count = 16
    elif args.final:
        # Stage 15 (items 42-43): distinct dynamics domains over overlapping
        # windows. 28 total = 4 nominal + 7 hydraulic + 6 mechanical +
        # 6 combined (train) + 5 held-out (test), inside the 20-30 target.
        allocations = {
            "nominal": 4,
            "hydraulic": 7,
            "mechanical": 6,
            "combined": 6,
        }
        test_count = 5
    else:
        allocations = {
            "nominal": max(int(args.train * 0.05), 1),
            "hydraulic": int(args.train * 0.30),
            "mechanical": int(args.train * 0.25),
            "combined": args.train - int(args.train * 0.05) - int(args.train * 0.30) - int(args.train * 0.25),
        }
        test_count = args.test
    rows = []
    for domain, count in allocations.items():
        domain_rows: list[dict] = []
        # Guarantee one-factor-at-a-time boundary anchors for the widened
        # hydraulic box. Sobol points alone approach but do not hit endpoints.
        if args.stage22_wide and domain == "hydraulic":
            for name in DOMAIN_DIMS[domain]:
                for value in ranges[name]:
                    e = nominal_e()
                    e[name] = value
                    domain_rows.append(e)
        elif args.stage22_wide and domain == "combined":
            # Explicit slow/fast response corners bound the joint factor box
            # and are flown during preflight before the long collection.
            slow = nominal_e()
            slow.update({
                "flow": ranges["flow"][0],
                "compliance": ranges["compliance"][0],
                "seal": ranges["seal"][1],
                "pressure": ranges["pressure"][0],
                "mass_scale": ranges["mass_scale"][1],
                "com_z": ranges["com_z"][0],
            })
            fast = nominal_e()
            fast.update({
                # Do not stack every gain-increasing extreme. The exact all-
                # high corner produced 14.7% command-clamp entries in the
                # full-speed flight (above the v2 maximum 10.7%). This anchor
                # sits just inside the empirical composite-gain ceiling.
                "flow": 0.08,
                "compliance": 0.02,
                "seal": ranges["seal"][0],
                "pressure": 0.00,
                "mass_scale": ranges["mass_scale"][0],
                "com_z": ranges["com_z"][1],
            })
            domain_rows.extend((slow, fast))
        remaining = count - len(domain_rows)
        if remaining < 0:
            raise RuntimeError(f"allocation for {domain} is smaller than its anchors")
        domain_rows.extend(
            sample(
                domain,
                remaining,
                DOMAIN_SEEDS[domain],
                args.method,
                exclude={round_key(e) for e in domain_rows},
                ranges=ranges,
                predicate=(
                    (lambda e: (
                        math.exp(0.5 * e["compliance"])
                        * (1.0 + e["flow"])
                        * math.sqrt(1.0 + e["pressure"])
                        / (1.0 + 0.625 * e["seal"])
                    ) <= 1.10)
                    if args.stage22_wide and domain == "combined"
                    else None
                ),
            )
        )
        for e in domain_rows:
            rows.append((domain, "train", e))

    train_keys = {round_key(e) for _, _, e in rows}
    test_rows = []
    for e in sample(
        "combined", test_count, TEST_SEED, args.method,
        exclude=train_keys, ranges=ranges,
        predicate=(
            (lambda e: (
                math.exp(0.5 * e["compliance"])
                * (1.0 + e["flow"])
                * math.sqrt(1.0 + e["pressure"])
                / (1.0 + 0.625 * e["seal"])
            ) <= 1.10)
            if args.stage22_wide else None
        ),
    ):
        test_rows.append(("combined", "test", e))
    overlap = train_keys & {round_key(e) for _, _, e in test_rows}
    if overlap:
        raise RuntimeError(f"D_TEST collides with training: {len(overlap)} samples")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["run_id", "domain", "split"] + list(DIM_ORDER))
        all_rows = [(d, s, e) for d, s, e in rows] + [(d, s, e) for d, s, e in test_rows]
        for run_id, (domain, split, e) in enumerate(all_rows):
            writer.writerow([run_id, domain, split] + [f"{e[n]:.6f}" for n in DIM_ORDER])

    profile = "stage22_wide" if args.stage22_wide else "stage5"
    print(f"method: {args.method} (space-filling inside {profile} ranges)")
    print("allocation (train):", allocations)
    print("test:", len(test_rows))
    print(f"wrote {len(all_rows)} runs -> {out}"
          f"{' [Stage 15 FINAL manifest]' if args.final else ''}")
    combined_example = next(e for d, _, e in rows if d == "combined")
    print("example combined draw:", combined_example)
    print("example test draw:    ", test_rows[0][2])


if __name__ == "__main__":
    main()
