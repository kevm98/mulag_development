"""Per-sample usability mask for open-loop excitation collections.

Why this exists
---------------
The open-loop excitation collection drives the plant hard on purpose. On the
leakiest plant draws that pushes some samples into command saturation or onto a
cylinder travel limit: 10.4 % of samples overall, 7.9 % in the median run, 45.8 %
in the worst. Those samples are not corrupt -- they are recorded, finite, and
grid-aligned -- but they carry almost no information about *which plant this is*,
because a pinned valve produces the same response whatever the seal condition
underneath it. Training a plant-identity latent on them is at best wasted
capacity and at worst actively misleading.

The mask is applied to the training LOSS, never by deleting rows. Deleting rows
would break the causal history windows the AAM reads (a 20-tick window must be
20 consecutive ticks of one run), and it would silently change what "the
previous valve command" means. Rows therefore stay exactly where they are; only
their contribution to the objective is dropped.

The same mask must be applied to every model in a comparison, otherwise the
baseline and the candidate are scored on different sample sets.
"""
from __future__ import annotations

import numpy as np

FLAGS = ("command_saturation_flag", "cylinder_limit_flag", "invalid_physics_flag")


def usable_rows(samples: dict | np.lib.npyio.NpzFile,
                flags: tuple[str, ...] = FLAGS) -> np.ndarray:
    """(n,) bool: True where the transition is worth training on.

    A row is unusable if ANY cylinder is flagged on ANY of ``flags`` at that
    sample. Valve saturation is deliberately not included: it is measured at
    ~0.0002 in this collection, far below its own 0.005 gate, so masking on it
    would remove nothing and only add a failure mode.
    """
    bad = None
    for name in flags:
        if name not in getattr(samples, "files", samples):
            continue
        f = np.asarray(samples[name])
        f = f.any(axis=1) if f.ndim > 1 else f.astype(bool)
        bad = f if bad is None else (bad | f)
    if bad is None:
        raise KeyError(f"none of {flags} present; refusing to assume all rows usable")
    return ~bad
