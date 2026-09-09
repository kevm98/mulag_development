"""Dependency-free SVG/PDF renderer for the two Chapter 7 summary figures."""
from __future__ import annotations

import html
import json
import math
import random
import re
import statistics as st
import subprocess
from pathlib import Path

INK, MUTED, GRID, SPINE = "#26343F", "#71808C", "#D5DADF", "#7C8791"
RED, AMBER, GREEN, BLUE = "#C1272D", "#C8860D", "#1B843F", "#4E79A7"
FONT = "DejaVu Serif, Times New Roman, serif"


def _text(x, y, value, size=20, color=INK, anchor="middle", weight="normal",
          rotate=None):
    transform = f' transform="rotate({rotate} {x} {y})"' if rotate else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" '
            f'font-size="{size}" fill="{color}" text-anchor="{anchor}" '
            f'font-weight="{weight}"{transform}>{html.escape(str(value))}</text>')


def _line(x1, y1, x2, y2, color=INK, width=2, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{color}" stroke-width="{width}"{d}/>')


def _nice_ticks(lo, hi, count=5):
    span = max(hi - lo, 1e-9)
    raw = span / max(count - 1, 1)
    power = 10 ** math.floor(math.log10(raw))
    step = min((1, 2, 2.5, 5, 10), key=lambda n: abs(n * power - raw)) * power
    start = math.ceil(lo / step) * step
    vals, value = [], start
    while value <= hi + step * .01:
        vals.append(value)
        value += step
    return vals


def _pdf(svg, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import cairosvg
        cairosvg.svg2pdf(bytestring=svg.encode(), write_to=str(path))
        return
    except ImportError:
        pass
    chrome = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    scratch = path.with_suffix(".print.html")
    width = re.search(r'<svg[^>]* width="([0-9.]+)"', svg).group(1)
    height = re.search(r'<svg[^>]* height="([0-9.]+)"', svg).group(1)
    scratch.write_text(
        f'<meta charset="utf-8"><style>@page{{size:{width}px {height}px;margin:0}}'
        'html,body{margin:0;padding:0}</style>' + svg)
    subprocess.run([str(chrome), "--headless", "--disable-gpu",
                    "--no-pdf-header-footer", f"--print-to-pdf={path}",
                    scratch.resolve().as_uri()], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    scratch.unlink()


def _load_kappa(repo, dirname):
    out = {}
    for path in sorted((repo / dirname).glob("*_m1337.json")):
        arm = path.name.split("_k")[0]
        k = float(path.name.split("_k")[1].split("_m")[0])
        block = list(json.loads(path.read_text())["per_mesh"].values())[0]
        out.setdefault(arm, {})[k] = (
            st.median(v * 1000 for v in block["per_env_lateral_m"]),
            block["worst_family_ground_contact_fraction"])
    return out


def render_kappa(repo, dest):
    nom = _load_kappa(repo, "outputs/kappa")
    loaded = _load_kappa(repo, "outputs/kappa_loaded")
    nominal_series = [("R", RED, "R (PPO-ROM)", "circle"),
                      ("IF", AMBER, "IF (PPO-ID-FD)", "square"),
                      ("AFf", BLUE, "AF frozen", "diamond")]
    loaded_series = [("AFfL", BLUE, "AF frozen", "diamond"),
                     ("AFaL", GREEN, "AF adaptive", "triangle")]
    panels = [
        ("(a) Nominal payload - lateral error", nom, nominal_series, 0,
         "median lateral error (mm)", "lower is better"),
        ("(b) 4.0x payload - lateral error", loaded, loaded_series, 0,
         "median lateral error (mm)", "lower is better"),
        ("(c) Nominal payload - ground contact", nom, nominal_series, 1,
         "worst-family ground contact", "higher is better"),
        ("(d) 4.0x payload - ground contact", loaded, loaded_series, 1,
         "worst-family ground contact", "higher is better"),
    ]
    W, H = 1260, 1480
    left, right, top, ph, step = 135, 35, 92, 235, 350
    pw = W - left - right
    xlo, xhi = math.log10(.008), math.log10(1.25)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
           f'viewBox="0 0 {W} {H}"><rect width="100%" height="100%" fill="white"/>']
    ranges = {}
    for idx in (0, 1):
        vals = [d[k][idx] for _, data, series, metric, _, _ in panels if metric == idx
                for arm, _, _, _ in series
                for d in [data.get(arm, {})] for k in d]
        lo, hi = min(vals), max(vals)
        pad = .07 * (hi - lo)
        ranges[idx] = (lo - pad, hi + pad)

    x0 = left
    for panel_i, (title, data, series, idx, ylabel, cue) in enumerate(panels):
        y0 = top + panel_i * step
        ylo, yhi = ranges[idx]
        xm = lambda k: x0 + (math.log10(k) - xlo) / (xhi - xlo) * pw
        ym = lambda v: y0 + ph - (v - ylo) / (yhi - ylo) * ph
        svg.append(_text(x0, y0 - 50, title, 27, INK, "start", "bold"))
        legend_width = 250 if len(series) == 3 else 270
        lx = x0 + pw - legend_width * len(series)
        for legend_i, (_, color, label, marker) in enumerate(series):
            xx = lx + legend_i * legend_width
            svg.append(_line(xx, y0 - 22, xx + 42, y0 - 22, color, 5.2))
            mx = xx + 21
            if marker == "circle":
                svg.append(f'<circle cx="{mx}" cy="{y0-22}" r="7" fill="{color}"/>')
            elif marker == "square":
                svg.append(f'<rect x="{mx-7}" y="{y0-29}" width="14" height="14" fill="{color}"/>')
            elif marker == "diamond":
                svg.append(f'<polygon points="{mx},{y0-31} {mx+9},{y0-22} {mx},{y0-13} {mx-9},{y0-22}" fill="{color}"/>')
            else:
                svg.append(f'<polygon points="{mx},{y0-31} {mx-9},{y0-14} {mx+9},{y0-14}" fill="{color}"/>')
            svg.append(_text(xx + 53, y0 - 15, label, 20, INK, "start"))
        svg.append(f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" '
                   f'fill="white" stroke="{SPINE}" stroke-width="1.4"/>')
        for tick in _nice_ticks(ylo, yhi, 6):
            yy = ym(tick)
            svg.append(_line(x0, yy, x0 + pw, yy, GRID, 1))
            label = f"{tick:.0f}" if idx == 0 else f"{tick:.1f}"
            svg.append(_text(x0 - 14, yy + 7, label, 22, INK, "end"))
        for tick, label in ((.01, "0.01"), (.03, "0.03"), (.1, "0.1"),
                            (.3, "0.3"), (1.0, "1.0")):
            xx = xm(tick)
            svg.append(_line(xx, y0, xx, y0 + ph, GRID, 1))
            svg.append(_text(xx, y0 + ph + 28, label, 21))
        if idx == 1:
            svg.append(_line(x0, ym(.70), x0 + pw, ym(.70), MUTED, 2, "8 5"))
            svg.append(_text(x0 + 10, ym(.70) - 7, "contact gate 0.70",
                             18, MUTED, "start"))
        svg.append(_text(x0 + 12, y0 + 27, cue, 19, MUTED, "start"))
        for arm, color, _, marker in series:
            d = data.get(arm, {})
            pts = " ".join(f"{xm(k):.1f},{ym(d[k][idx]):.1f}" for k in sorted(d))
            svg.append(f'<polyline points="{pts}" fill="none" stroke="{color}" '
                       'stroke-width="5.2" stroke-linejoin="round"/>')
            for k in sorted(d):
                xx, yy = xm(k), ym(d[k][idx])
                if marker == "circle":
                    shape = f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="7"'
                elif marker == "square":
                    shape = f'<rect x="{xx-7:.1f}" y="{yy-7:.1f}" width="14" height="14"'
                elif marker == "diamond":
                    shape = f'<polygon points="{xx:.1f},{yy-9:.1f} {xx+9:.1f},{yy:.1f} {xx:.1f},{yy+9:.1f} {xx-9:.1f},{yy:.1f}"'
                else:
                    shape = f'<polygon points="{xx:.1f},{yy-9:.1f} {xx-9:.1f},{yy+8:.1f} {xx+9:.1f},{yy+8:.1f}"'
                svg.append(shape + f' fill="{color}" stroke="white" stroke-width="1.5"/>')
        svg.append(_text(31, y0 + ph / 2, ylabel, 25, INK, rotate=-90))
    svg.append(_text(x0 + pw / 2, H - 15,
                     "actuator authority kappa (fraction of nominal)", 22))
    svg.append("</svg>")
    _pdf("".join(svg), dest)


def _row(src, name):
    block = json.loads((src / name).read_text())["per_mesh"]
    return list(block.values())[0]


def _paired(frozen, arm, key, unit, higher_better, B=20000):
    f = [v * unit for v in frozen[key]]
    a = [v * unit for v in arm[key]]
    n = min(len(f), len(a))
    d = [(a[i] - f[i]) if higher_better else (f[i] - a[i]) for i in range(n)]
    if B == 0:
        return st.median(d), None, None, d
    random.seed(0)
    boot = sorted(st.median(d[random.randrange(n)] for _ in range(n)) for _ in range(B))
    return st.median(d), boot[int(.025 * B)], boot[int(.975 * B)], d


def render_adaptation(repo, dest):
    src = repo / "outputs/final"
    seeds, masses = ("17", "23", "101"), ("3.0", "4.0", "5.0")
    channels = (
        ("per_env_cut_error_m", "Cut-height error", "error (mm)", 1000., False),
        ("per_env_lateral_m", "Lateral error", "error (mm)", 1000., False),
        ("per_env_contact_fraction", "Ground contact", "contact fraction", 1., True),
    )
    W, H = 1470, 790
    left, right, gap, top, bottom = 95, 25, 70, 155, 650
    pw = (W - left - right - 2 * gap) / 3
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
           f'viewBox="0 0 {W} {H}"><rect width="100%" height="100%" fill="white"/>']
    svg += [f'<rect x="525" y="29" width="22" height="22" fill="{BLUE}"/>',
            _text(560, 48, "Frozen", 24, INK, "start"),
            f'<rect x="735" y="29" width="22" height="22" fill="{AMBER}"/>',
            _text(770, 48, "Adaptive", 24, INK, "start")]

    for col, (key, title, ylabel, unit, higher_better) in enumerate(channels):
        rows = []
        for mass in masses:
            frozen = _row(src, f"frozen_p{mass}.json")
            frozen_median = st.median(v * unit for v in frozen[key])
            pool = []
            for seed in seeds:
                _, _, _, differences = _paired(
                    frozen, _row(src, f"adaptive_s{seed}_p{mass}.json"),
                    key, unit, higher_better, B=0)
                pool += differences
            random.seed(0)
            n = len(pool)
            boot = sorted(st.median(pool[random.randrange(n)] for _ in range(n))
                          for _ in range(20000))
            effect, effect_lo, effect_hi = st.median(pool), boot[500], boot[19500]
            adaptive = (frozen_median + effect if higher_better
                        else frozen_median - effect)
            absolute_lo, absolute_hi = ((frozen_median + effect_lo,
                                         frozen_median + effect_hi)
                                        if higher_better else
                                        (frozen_median - effect_hi,
                                         frozen_median - effect_lo))
            rows.append((frozen_median, adaptive, absolute_lo, absolute_hi))

        ymax = max(max(f, a, hi) for f, a, _, hi in rows) * 1.22
        if higher_better:
            ymax = max(1.0, ymax)
        x0 = left + col * (pw + gap)
        ym = lambda value: bottom - value / ymax * (bottom - top)
        svg.append(f'<rect x="{x0}" y="{top}" width="{pw}" height="{bottom-top}" '
                   f'fill="white" stroke="{SPINE}" stroke-width="1.4"/>')
        for tick in _nice_ticks(0, ymax, 6):
            yy = ym(tick)
            svg.append(_line(x0, yy, x0 + pw, yy, GRID, 1))
            label = f"{tick:.0f}" if unit > 1 else f"{tick:.1f}"
            svg.append(_text(x0 - 13, yy + 7, label, 21, INK, "end"))
        svg.append(_text(x0 + pw / 2, 91, title, 29, INK, weight="bold"))
        cue = "lower is better" if unit > 1 else "higher is better"
        svg.append(_text(x0 + pw / 2, 127, cue, 20, MUTED))
        band = pw / 3
        bar_w, offset = 52, 31
        for i, (frozen, adaptive, ci_lo, ci_hi) in enumerate(rows):
            centre = x0 + band * (i + .5)
            fx, ax = centre - offset - bar_w / 2, centre + offset - bar_w / 2
            fy, ay = ym(frozen), ym(adaptive)
            svg.append(f'<rect x="{fx}" y="{fy}" width="{bar_w}" height="{bottom-fy}" fill="{BLUE}"/>')
            svg.append(f'<rect x="{ax}" y="{ay}" width="{bar_w}" height="{bottom-ay}" fill="{AMBER}"/>')
            cx = ax + bar_w / 2
            svg += [_line(cx, ym(ci_hi), cx, ym(ci_lo), INK, 2),
                    _line(cx - 10, ym(ci_hi), cx + 10, ym(ci_hi), INK, 2),
                    _line(cx - 10, ym(ci_lo), cx + 10, ym(ci_lo), INK, 2)]
            fmt = (lambda v: f"{v:.0f}") if unit > 1 else (lambda v: f"{v:.2f}")
            svg.append(_text(fx + bar_w / 2, fy - 10, fmt(frozen), 19, BLUE,
                             weight="bold"))
            svg.append(_text(ax + bar_w / 2, min(ay, ym(ci_hi)) - 10,
                             fmt(adaptive), 19, AMBER, weight="bold"))
            svg.append(_text(centre, bottom + 34, f"{masses[i]}x", 22))
        svg.append(_text(x0 + pw / 2, 728, "payload", 22))
        svg.append(_text(x0 - 58, (top + bottom) / 2, ylabel, 23, INK,
                         rotate=-90))
    svg.append("</svg>")
    _pdf("".join(svg), dest)
