#!/usr/bin/env python3
"""Read the Phase 4 arm reports and lay the ablation grid out side by side.

Reports the metrics the claim is about and, for each contrast, the difference
that the claim predicts. It does NOT decide promotion -- promote_system.py owns
that and deliberately refuses the degraded controller arms this grid needs.

Two contrasts carry the argument, and they are not interchangeable:

  A4 vs C_shuffled   the latent's CONTENT matters (not just its dimensions)
  A4 vs C_frozen     the latent's UPDATING matters (not just static identity)

C_frozen is the stronger control and the one to report: it holds a real
pre-change estimate, so a gap can only come from tracking the change.

Every arm is the same checkpoint, mesh set, speed, seed and step count, so the
comparison is paired; the script refuses to tabulate arms that disagree on any
of those, because a difference would then have two explanations.

CPU only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# lower is better for all of these
METRICS = [
    ("worst_family_mean_abs_lateral_error_m", "lateral err (m)"),
    ("mean_abs_cut_height_error_m", "cut-height err (m)"),
    ("hard_impact_step_fraction", "hard impacts"),
    ("p95_tool_acceleration_mps2", "p95 accel"),
    ("safety_violation_rate", "safety viol."),
]
# these are better when higher
HIGHER_IS_BETTER = {"worst_family_ground_contact_fraction", "p01_head_alignment"}

ARM_ORDER = ["A1_baseline", "A2_id_only", "A3_ppo_only", "A4_full",
             "C_shuffled", "C_frozen"]

PAIRING = ("checkpoint", "stage", "actuator")


def mesh_mean(report: dict, key: str) -> float | None:
    vals = [row[key] for row in report.get("per_mesh", {}).values() if key in row]
    return sum(vals) / len(vals) if vals else None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()

    arms: dict[str, dict] = {}
    for name in ARM_ORDER:
        f = a.dir / f"{name}_gates.json"
        if f.is_file():
            arms[name] = json.loads(f.read_text())
    if not arms:
        raise SystemExit(f"no arm reports under {a.dir}")

    # ---- pairing: a difference must have only one possible cause ----------
    problems = []
    ref_name, ref = next(iter(arms.items()))
    for name, rep in arms.items():
        for field in PAIRING:
            if rep.get(field) != ref.get(field):
                problems.append(
                    f"{name}.{field}={rep.get(field)!r} != {ref_name}.{field}="
                    f"{ref.get(field)!r}; arms are not paired"
                )
        if set(rep.get("per_mesh", {})) != set(ref.get("per_mesh", {})):
            problems.append(f"{name} scored a different mesh set than {ref_name}")

    # ---- the grid ---------------------------------------------------------
    table = {}
    for name, rep in arms.items():
        row = {label: mesh_mean(rep, key) for key, label in METRICS}
        row["z_hat std"] = rep.get("z_hat_std_observed")
        row["policy latent"] = rep.get("online_aam_latent_mode")
        row["ctrl latent"] = rep.get("online_aam_controller_latent_mode")
        table[name] = row

    width = max(len(n) for n in table)
    labels = [label for _, label in METRICS]
    print(f"{'arm':<{width}}  {'policy':<10} {'ctrl':<10} " +
          " ".join(f"{l:>18}" for l in labels))
    for name in ARM_ORDER:
        if name not in table:
            continue
        r = table[name]
        cells = []
        for l in labels:
            v = r[l]
            cells.append(f"{v:>18.5f}" if isinstance(v, float) else f"{'--':>18}")
        print(f"{name:<{width}}  {str(r['policy latent']):<10} "
              f"{str(r['ctrl latent']):<10} " + " ".join(cells))

    # ---- the contrasts the claim rests on ---------------------------------
    contrasts = {}
    def contrast(better: str, worse: str, why: str) -> None:
        if better not in table or worse not in table:
            return
        deltas = {}
        for _, l in METRICS:
            b, w = table[better][l], table[worse][l]
            if isinstance(b, float) and isinstance(w, float):
                # negative = `better` arm has lower (better) error
                deltas[l] = b - w
        contrasts[f"{better} - {worse}"] = {"why": why, "deltas": deltas}

    contrast("A4_full", "A1_baseline", "does the latent help at all")
    contrast("A4_full", "A2_id_only", "does routing it to the POLICY add anything")
    contrast("A4_full", "A3_ppo_only", "does routing it to the ID add anything")
    contrast("A4_full", "C_shuffled", "is it the latent's CONTENT")
    contrast("A4_full", "C_frozen", "is it the ONLINE UPDATING")

    print("\nCONTRASTS (negative = the first arm is better)")
    for name, c in contrasts.items():
        print(f"\n  {name}   -- {c['why']}")
        for l, d in c["deltas"].items():
            print(f"    {l:<20} {d:+.5f}")

    if problems:
        print("\nPAIRING PROBLEMS -- do not read the contrasts above:")
        for m in problems:
            print(f"  - {m}")

    missing = [n for n in ARM_ORDER if n not in arms]
    if missing:
        print(f"\nmissing arms: {', '.join(missing)}")

    print("\nThese are evaluation-time ablations on ONE checkpoint. They show "
          "what the trained system uses at deployment, not what a system "
          "retrained without the latent would achieve.")

    if a.out:
        a.out.write_text(json.dumps(
            {"arms": table, "contrasts": contrasts, "problems": problems,
             "missing": missing}, indent=2))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
