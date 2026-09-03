"""Shared style and data access for every Chapter 7 figure.

One module so that a stack keeps one colour, one typeface and one axis idiom
across all fourteen figures. Paths are resolved from this file rather than
hard-coded to the VM the runs were produced on.

The matched pair is the only cross-stack comparison the chapter reports.
PPO-(AAM)-ID-FD's cross-stack numbers were withdrawn when its command path was
rebuilt (Section 7.1.3), so PAIR -- not STACKS -- is what the cross-stack
figures draw. STACKS is kept for the one figure that legitimately shows three
rows: the pressure channel of Section 7.6.3, which is measured on the current
build.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve()
THESIS = HERE.parents[1]
REPO = HERE.parents[4]
IMAGES = THESIS / "images"
OUT = IMAGES
EXP = REPO / "experiments"
OUTPUTS = REPO / "outputs"
OPENLOOP = OUTPUTS / "stage31_openloop"

# Stack colours: red / amber / green. Not a colour-vision-safe triple, so every
# figure also separates the stacks by position and by an explicit legend order,
# and no figure asks the reader to tell red from green by hue alone.
C_RED, C_AMBER, C_GRN = "#C1272D", "#C8860D", "#1B843F"
# Okabe-Ito subset for series that are not stacks (channels, arms, latents).
C_BLUE, C_ORANGE, C_GREEN, C_PINK = "#0072B2", "#E69F00", "#009E73", "#CC79A7"
C_GRAY = "#666666"
C_RULE = "#5A6B78"

# The chapter names the stacks R, IF and AF (macros.tex), so the figures use
# the same tokens the tables and captions do, expanded once in the legend so a
# figure still reads on its own.
# The inverse model is not deployed in AF, so the displayed name must describe
# the adaptive-forward architecture rather than the historical experiment key.
# The report JSONs predate the rename and remain keyed on the old string.
FULL = {"ppo_rom": "PPO-ROM", "ppo_id_fd": "PPO-ID-FD",
        "ppo_aam_id_fd": "Adaptive Forward"}
JSON_KEY = {"ppo_rom": "PPO-ROM", "ppo_id_fd": "PPO-ID-FD",
            "ppo_aam_id_fd": "PPO-(AAM)-ID-FD"}
SHORT = {"ppo_rom": "R", "ppo_id_fd": "IF", "ppo_aam_id_fd": "AF"}
LEGEND = {k: f"{SHORT[k]} \u2014 {FULL[k]}" for k in FULL}

PAIR = [("ppo_rom", LEGEND["ppo_rom"], C_RED),
        ("ppo_id_fd", LEGEND["ppo_id_fd"], C_AMBER)]
STACKS = PAIR + [("ppo_aam_id_fd", LEGEND["ppo_aam_id_fd"], C_GRN)]
MESHES = ["1337", "17", "909"]
FULLW = 6.3

mpl.rcParams.update({
    "font.size": 9, "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.spines.top": True, "axes.spines.right": True, "axes.linewidth": 0.6,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 3.0, "ytick.major.size": 3.0,
    "grid.color": "#c8c8c8", "grid.linewidth": 0.4, "grid.alpha": 0.7,
    "lines.linewidth": 1.2, "legend.frameon": False,
    "pdf.fonttype": 42, "figure.constrained_layout.use": False,
})


def load(path):
    return json.loads(Path(path).read_text())


def _chapter_dir(name):
    """Figures live under images/<chapter>/, keyed off the chN_ name prefix.

    Anything without a chapter prefix stays at the top of images/, which is
    where the shared geometry data sits.
    """
    head = name.split("_", 1)[0]
    if len(head) == 3 and head.startswith("ch") and head[2].isdigit():
        d = IMAGES / head
        d.mkdir(parents=True, exist_ok=True)
        return d, f"{head}/"
    return IMAGES, ""


def save(fig, name):
    out, rel = _chapter_dir(name)
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)
    print(f"  wrote images/{rel}{name}.pdf")


# --------------------------------------------------------------------------
# E1 closed-loop reports
# --------------------------------------------------------------------------
E1 = OUTPUTS / "e1_tracking"

# The eight-gate contract (PPO_ROM/gates.py), less the two that are not scored
# per mesh in these reports. Duplicated rather than imported because that module
# pulls in gymnasium and the whole IsaacLab stack.
GATES = [
    ("cut height",        "mean_abs_cut_height_error_m",             0.040, "max"),
    ("lateral",           "worst_family_mean_abs_lateral_error_m",   0.300, "max"),
    ("ground contact",    "worst_family_ground_contact_fraction",    0.700, "min"),
    ("head alignment",    "p01_head_alignment",                      0.900, "min"),
    ("hard impacts",      "hard_impact_step_fraction",               0.350, "max"),
    ("tool acceleration", "p95_tool_acceleration_mps2",            120.000, "max"),
]


def e1_cells(stack, root=None, seed="s17", meshes=None):
    """(mesh, family) -> that cell's metric block."""
    root = root or E1
    meshes = meshes or MESHES
    cells = {}
    for m in meshes:
        blk = load(root / f"{stack}_{seed}_m{m}.json")["per_mesh"][m]
        for fam in blk["path_family_results"]:
            cells[(m, fam["family"])] = fam
    return cells


def e1_family_means(stack, key="mean_abs_lateral_error_m"):
    """family -> mean over the three held-out meshes."""
    cells = e1_cells(stack)
    fams = sorted({f for _, f in cells})
    return {f: float(np.mean([cells[(m, f)][key] for m in MESHES])) for f in fams}


def e1_worst_mesh(stack):
    """Gate metrics taken at their worst over the three meshes."""
    vals = {}
    for m in MESHES:
        blk = load(E1 / f"{stack}_{'s17'}_m{m}.json")["per_mesh"][m]
        for _, key, _, direction in GATES:
            v = blk.get(key)
            if v is None:
                continue
            vals[key] = v if key not in vals else (
                max(vals[key], v) if direction == "max" else min(vals[key], v))
    return vals
