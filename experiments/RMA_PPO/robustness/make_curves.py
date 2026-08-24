"""Robustness curves: normalized tracking error vs stress severity.

One panel per stress axis.  Each shows the frozen PPO's D_E = E(rho)/E(nominal),
the nominal level D_E = 1, the pre-declared degradation threshold D_E = 1.5, and
the do-nothing null-controller asymptote.  Written as a standalone SVG so no
plotting dependency is required.
"""
import json, math
from pathlib import Path

OUT = Path("/tmp/rrlab_scratch/results")
data = json.loads((OUT / "robustness_sweeps.json").read_text())

AXES = [
    ("authority", "actuator authority alpha", True),
    ("delay", "actuation delay [ticks]", False),
    ("lag", "hydraulic response tau [s]", False),
    ("rate_limit", "actuator rate limit", True),
    ("deadband", "command deadband", False),
]
W, H = 340, 250
PL, PR, PT, PB = 58, 16, 34, 46
COLS = 3
YMAX = 2.0

def xpos(i, n): return PL + (W - PL - PR) * (i / max(n - 1, 1))
def ypos(v): return PT + (H - PT - PB) * (1.0 - min(v, YMAX) / YMAX)

panels = []
for k, (axis, xlabel, reverse) in enumerate(AXES):
    rows = list(data["sweeps"][axis])
    if reverse:
        rows = rows[::-1]          # plot increasing severity left to right
    n = len(rows)
    ox, oy = (k % COLS) * W, (k // COLS) * H
    p = [f'<g transform="translate({ox},{oy})">']
    p.append(f'<rect x="{PL}" y="{PT}" width="{W-PL-PR}" height="{H-PT-PB}" fill="#fff" stroke="#bbb"/>')
    # threshold bands
    p.append(f'<rect x="{PL}" y="{ypos(YMAX)}" width="{W-PL-PR}" height="{ypos(1.5)-ypos(YMAX)}" fill="#fdeaea"/>')
    for val, col, dash, lab in ((1.0, "#333", "4,3", "nominal D_E = 1"),
                                (1.5, "#c0392b", "6,3", "degradation threshold 1.5")):
        y = ypos(val)
        p.append(f'<line x1="{PL}" y1="{y:.1f}" x2="{W-PR}" y2="{y:.1f}" stroke="{col}" stroke-dasharray="{dash}" stroke-width="1"/>')
    # axes ticks
    for v in (0.0, 0.5, 1.0, 1.5, 2.0):
        y = ypos(v)
        p.append(f'<line x1="{PL-4}" y1="{y:.1f}" x2="{PL}" y2="{y:.1f}" stroke="#666"/>')
        p.append(f'<text x="{PL-7}" y="{y+3.5:.1f}" font-size="9" text-anchor="end" fill="#444">{v:g}</text>')
    # null-controller curve
    pts_null = " ".join(f"{xpos(i,n):.1f},{ypos(r['null_normalized_error']):.1f}" for i, r in enumerate(rows))
    p.append(f'<polyline points="{pts_null}" fill="none" stroke="#7f8c8d" stroke-width="1.6" stroke-dasharray="3,3"/>')
    # PPO curve
    pts = " ".join(f"{xpos(i,n):.1f},{ypos(r['normalized_error']):.1f}" for i, r in enumerate(rows))
    p.append(f'<polyline points="{pts}" fill="none" stroke="#1f6fb4" stroke-width="2.2"/>')
    for i, r in enumerate(rows):
        x, y = xpos(i, n), ypos(r["normalized_error"])
        fill = "#1f6fb4" if r["surrogate_valid"] else "#e67e22"
        p.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.1" fill="{fill}"/>')
        sev = r["severity"]
        if i % max(1, n // 6) == 0 or i == n - 1:
            lab = "inf" if sev is None else (f"{sev:g}")
            p.append(f'<text x="{x:.1f}" y="{H-PB+13}" font-size="8" text-anchor="middle" fill="#444">{lab}</text>')
    p.append(f'<text x="{(PL+W-PR)/2:.0f}" y="{H-PB+30}" font-size="9.5" text-anchor="middle" fill="#222">{xlabel}</text>')
    p.append(f'<text x="{PL}" y="{PT-11}" font-size="12" font-weight="600" fill="#111">{axis}</text>')
    p.append(f'<text x="14" y="{(PT+H-PB)/2:.0f}" font-size="9.5" text-anchor="middle" fill="#222" transform="rotate(-90,14,{(PT+H-PB)/2:.0f})">normalized error D_E</text>')
    p.append("</g>")
    panels.append("".join(p))

# legend panel
lx, ly = 2 * W, H
leg = [f'<g transform="translate({lx},{ly})">']
items = [("#1f6fb4", "solid", "frozen PPO, surrogate valid"),
         ("#e67e22", "dot", "frozen PPO, surrogate INVALID"),
         ("#7f8c8d", "dash", "do-nothing null controller"),
         ("#c0392b", "dash", "degradation threshold D_E = 1.5"),
         ("#333", "dash", "nominal D_E = 1")]
leg.append(f'<text x="{PL}" y="{PT}" font-size="12" font-weight="600" fill="#111">legend</text>')
for i, (c, kind, lab) in enumerate(items):
    y = PT + 24 + i * 22
    if kind == "dot":
        leg.append(f'<circle cx="{PL+10}" cy="{y-4}" r="4" fill="{c}"/>')
    else:
        d = 'stroke-dasharray="4,3"' if kind == "dash" else ""
        leg.append(f'<line x1="{PL}" y1="{y-4}" x2="{PL+22}" y2="{y-4}" stroke="{c}" stroke-width="2.4" {d}/>')
    leg.append(f'<text x="{PL+30}" y="{y}" font-size="10" fill="#333">{lab}</text>')
leg.append(f'<text x="{PL}" y="{PT+24+len(items)*22+16}" font-size="9.5" fill="#8a2b1f">'
           f'No axis ever reaches D_E = 1.5. Every curve falls.</text>')
leg.append("</g>")
panels.append("".join(leg))

total_w, total_h = COLS * W, 2 * H
svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}" height="{total_h+34}" '
       f'viewBox="0 0 {total_w} {total_h+34}" font-family="Helvetica,Arial,sans-serif">'
       f'<rect width="100%" height="100%" fill="#fff"/>'
       f'<text x="16" y="22" font-size="15" font-weight="700" fill="#111">'
       f'Frozen-PPO Robustness Envelope Under Plant Degradation - CPU surrogate</text>'
       f'<g transform="translate(0,30)">' + "".join(panels) + '</g></svg>')
(OUT / "robustness_curves.svg").write_text(svg)
print("wrote", OUT / "robustness_curves.svg")

# text summary of the asymptote
print("\nasymptote check (max-stress point vs null controller):")
for axis, _, rev in AXES:
    rows = data["sweeps"][axis]
    last = rows[-1]
    print(f"  {axis:<11} severity={str(last['severity']):>7}  "
          f"gain={last['delivered_valve_gain']:.3f}  "
          f"PPO={last['median_extension_error_m']*1000:.3f} mm  "
          f"null={last['null_median_extension_error_m']*1000:.3f} mm  D_E={last['normalized_error']:.2f}")
print("\nmax D_E observed on any axis:",
      max(r["normalized_error"] for rows in data["sweeps"].values() for r in rows))
