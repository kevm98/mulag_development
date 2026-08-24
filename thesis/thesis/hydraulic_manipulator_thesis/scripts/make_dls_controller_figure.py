#!/usr/bin/env python3
"""Generate the standard DLS feedback block diagram used in Chapter 6."""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "images" / "ch6_dls_controller.pdf"

INK = "#202020"
GRAY = "#656565"
# Restrained technical accents: hierarchy without presentation-style colour.
BLUE = "#315F85"
TEAL = "#3F7C73"
ORANGE = "#A96E1D"
MAGENTA = VIOLET = INK
PALE_BLUE = "#F2F2F2"
PALE_ORANGE = "#FAFAFA"
PALE_TEAL = "#EEEEEE"
PALE_MAGENTA = "#F7F7F7"
PALE_GRAY = "#F2F2F2"

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9.5,
    "pdf.fonttype": 42,
    "savefig.facecolor": "white",
})


def box(ax, x, y, w, h, face, edge, title, detail=None, title_size=9.5,
        detail_size=10.5):
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.0,rounding_size=1.0",
        facecolor=face, edgecolor=edge, linewidth=1.25, zorder=3)
    ax.add_patch(patch)
    if detail is None:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                color=INK, fontsize=title_size, fontweight="bold", zorder=4)
    else:
        ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center",
                color=INK, fontsize=title_size, fontweight="bold", zorder=4)
        ax.text(x + w / 2, y + h * 0.30, detail, ha="center", va="center",
                color=INK, fontsize=detail_size, fontweight="semibold",
                zorder=4)
    return patch


def arrow(ax, start, end, color=INK, lw=1.25, zorder=5):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=9.5,
        linewidth=lw, color=color, shrinkA=1.5, shrinkB=1.5,
        zorder=zorder))


def main():
    fig, ax = plt.subplots(figsize=(6.3, 2.72))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 48)
    ax.axis("off")

    y, h = 28, 11
    yc = y + h / 2

    # Quiet scope line: informative, but not another visual object.
    ax.text(98.5, 45.0, "TRAINING AND DATA-COLLECTION CONTROLLER",
            ha="right", va="center", color=GRAY, fontsize=8.5,
            fontweight="bold")
    ax.plot([66.0, 98.5], [43.3, 43.3], color="#B7B7B7", lw=0.65)

    # Reference and standard negative-feedback summing junction.
    ax.text(0.4, yc + 4.7, "task-space target", ha="left",
            va="bottom", color=GRAY, fontsize=9.0)
    ax.text(0.4, yc, r"$\mathbf{r}^{*}$", ha="left", va="center",
            color=INK, fontsize=12, fontweight="bold")
    arrow(ax, (4.5, yc), (7.0, yc), BLUE)

    summing = Circle((9.5, yc), 2.5, facecolor="white",
                     edgecolor=BLUE, linewidth=1.1, zorder=3)
    ax.add_patch(summing)
    ax.text(9.5, yc, r"$\Sigma$", ha="center", va="center",
            color=INK, fontsize=10, zorder=4)
    ax.text(8.05, yc + 1.1, "+", color=BLUE, fontsize=8.5,
            ha="center", va="center", fontweight="bold", zorder=6)
    ax.text(9.5, yc - 1.45, "−", color=INK, fontsize=9.5,
            ha="center", va="center", fontweight="bold", zorder=6)

    arrow(ax, (12.0, yc), (14.5, yc), BLUE)
    ax.text(13.3, yc + 2.2, r"$\mathbf{e}$", ha="center", color=INK,
            fontsize=11.5, fontweight="bold")

    # The gain and task weighting are one error-shaping operation in the
    # governing equation; combining them removes an unnecessary micro-block.
    box(ax, 14.8, y, 14.0, h, "white", INK,
        "Error shaping", r"$g\,\mathbf{W}\mathbf{e}$",
        title_size=9.3, detail_size=11.5)
    arrow(ax, (28.8, yc), (31.5, yc), INK)

    box(ax, 31.8, y, 21.0, h, "#EDF3F7", BLUE,
        "Damped least-squares",
        r"$\mathbf{J}_c^{\#}(\lambda)$",
        title_size=9.3, detail_size=12.0)
    arrow(ax, (52.8, yc), (55.5, yc), INK)

    box(ax, 55.8, y, 11.5, h, "white", INK,
        "Step limiter", r"$\pm\,\Delta x_{\max}$",
        title_size=9.1, detail_size=11.0)
    arrow(ax, (67.3, yc), (70.0, yc), INK)

    box(ax, 70.3, y, 12.3, h, "#F0F7F5", TEAL,
        "Target update", r"$\Delta\mathbf{x}^{*}_{1:3}$",
        title_size=9.0, detail_size=11.5)
    arrow(ax, (82.6, yc), (85.3, yc), INK)

    box(ax, 85.6, y, 12.8, h, "#F5F5F5", INK,
        "Actuator / arm", r"$\mathbf{x}\mapsto\mathbf{r}$",
        title_size=8.6, detail_size=11.0)
    arrow(ax, (98.4, yc), (99.8, yc), INK)
    ax.text(99.2, yc + 2.8, r"$\mathbf{r}$", ha="center",
            color=INK, fontsize=9.5)

    # Cylinder 4 is a clean external reference input to the target-update
    # block, rather than another line of prose squeezed inside it.
    arrow(ax, (76.45, 23.0), (76.45, 27.9), TEAL, lw=0.9)
    ax.text(76.45, 21.5, r"$x_{4,\mathrm{ref}}$", ha="center", va="center",
            color=TEAL, fontsize=11.0, fontweight="bold")

    # Standard measured-output feedback to the negative summing input.
    ax.plot([99.2, 99.2, 9.5], [yc, 4.5, 4.5], color=INK,
            linewidth=1.0, zorder=1)
    arrow(ax, (9.5, 4.5), (9.5, yc - 2.6), INK, lw=1.0)
    ax.text(54.0, 3.0, r"measured cutter position  $\mathbf{r}$",
            ha="center", va="top", color=GRAY, fontsize=8.8)

    # The measured cylinder state updates the Jacobian used by the inverse.
    box(ax, 34.7, 11.5, 15.0, 7.5, "white", GRAY,
        "Jacobian evaluation", r"$\mathbf{J}_c(\mathbf{x})$",
        title_size=8.2, detail_size=11.0)
    ax.plot([92.0, 92.0, 49.7], [28.0, 15.25, 15.25], color=GRAY,
            linewidth=0.9, linestyle="--", zorder=1)
    arrow(ax, (52.0, 15.25), (49.75, 15.25), GRAY, lw=0.9)
    arrow(ax, (42.2, 19.0), (42.2, 27.9), GRAY, lw=0.9)
    ax.text(70.5, 16.6, r"measured cylinder state  $\mathbf{x}$",
            ha="center", va="bottom", color=GRAY, fontsize=8.8)

    fig.subplots_adjust(left=0.01, right=0.995, top=0.98, bottom=0.02)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
