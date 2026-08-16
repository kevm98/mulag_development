#!/usr/bin/env python
"""Generate visualization-only 2-D teacher references.

This deliberately does not launch Isaac Sim and does not write actions to a
plant.  It produces an SVG with the reference centreline, tangent direction,
and normal direction so geometry mistakes are caught before PPO or PhysX is
involved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verge_task.trajectory_2d import make_trajectory


def _make_teacher_trajectory(name: str, *, samples: int):
    """Use the same plant-feasible dimensions as trajectory-v2."""
    kwargs = {
        "line": {"length_m": 2.5},
        "sine_chain": {"length_m": 2.8, "amplitude_m": 0.26, "cycles": 3.0},
        "circle": {"radius_m": 0.42},
        "rounded_square": {"side_m": 0.82, "corner_radius_m": 0.14},
    }[name]
    return make_trajectory(name, count=max(samples, 32), **kwargs)


def _svg_path(points: np.ndarray, *, sx: float, sy: float, ox: float, oy: float) -> str:
    coords = [f"{ox + sx * float(x):.2f},{oy - sy * float(y):.2f}" for x, y in points]
    return "M " + " L ".join(coords)


def _write_route_svg(
    paths: dict[str, object],
    output: Path,
    *,
    station_shapes: tuple[str, ...],
    station_spacing_m: float,
) -> None:
    """Draw the stop-and-mow route used by the visualization-only teacher."""
    placed = [
        (name, paths[name].xy + np.array([index * station_spacing_m, 0.0]))
        for index, name in enumerate(station_shapes)
    ]
    all_xy = np.concatenate([points for _, points in placed], axis=0)
    lo = all_xy.min(axis=0) - 0.25
    hi = all_xy.max(axis=0) + 0.25
    width, height = 1400.0, 500.0
    scale = min(
        (width - 100.0) / max(hi[0] - lo[0], 1.0e-6),
        (height - 100.0) / max(hi[1] - lo[1], 1.0e-6),
    )
    ox, oy = 50.0 - scale * lo[0], 50.0 + scale * hi[1]
    colours = {"line": "#2563eb", "sine_chain": "#059669", "circle": "#dc2626", "rounded_square": "#7c3aed"}
    fragments = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{int(width)}" height="{int(height)}" viewBox="0 0 {int(width)} {int(height)}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{ox + scale * lo[0]:.2f}" y1="{oy:.2f}" x2="{ox + scale * hi[0]:.2f}" y2="{oy:.2f}" stroke="#9ca3af" stroke-width="2" stroke-dasharray="10 8"/>',
    ]
    for station, (name, points) in enumerate(placed):
        colour = colours.get(name, "#111827")
        fragments.append(f'<path d="{_svg_path(points, sx=scale, sy=scale, ox=ox, oy=oy)}" fill="none" stroke="{colour}" stroke-width="4" stroke-linejoin="round"/>')
        x = ox + scale * station * station_spacing_m
        fragments.append(f'<text x="{x:.2f}" y="{height - 18:.2f}" fill="{colour}" font-family="sans-serif" font-size="16" text-anchor="middle">station {station}: {name}</text>')
    fragments.append("</svg>\n")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(fragments))


def write_svg(
    shapes: tuple[str, ...],
    output: Path,
    *,
    samples: int,
    stations: int = 4,
    station_spacing_m: float = 2.0,
    route_output: Path | None = None,
) -> dict:
    paths = {name: _make_teacher_trajectory(name, samples=samples) for name in shapes}
    all_xy = np.concatenate([path.xy for path in paths.values()], axis=0)
    lo = all_xy.min(axis=0) - 0.15
    hi = all_xy.max(axis=0) + 0.15
    width, height = 1200.0, 700.0
    scale = min((width - 100.0) / max(hi[0] - lo[0], 1e-6), (height - 100.0) / max(hi[1] - lo[1], 1e-6))
    ox, oy = 50.0 - scale * lo[0], 50.0 + scale * hi[1]
    colours = {"line": "#2563eb", "sine_chain": "#059669", "circle": "#dc2626", "rounded_square": "#7c3aed"}
    fragments = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{int(width)}" height="{int(height)}" viewBox="0 0 {int(width)} {int(height)}">', '<rect width="100%" height="100%" fill="white"/>']
    legend = []
    payload = {
        "mode": "visualization_only_teacher",
        "shapes": {},
        "station_route": {
            "station_count": int(stations),
            "station_spacing_m": float(station_spacing_m),
            "closed_shape_protocol": "park_track_stow_drive_park_repeat",
            "open_shape_protocol": "fixed_low_speed_chassis_creep",
        },
    }
    for row, (name, path) in enumerate(paths.items()):
        colour = colours.get(name, "#111827")
        fragments.append(f'<path d="{_svg_path(path.xy, sx=scale, sy=scale, ox=ox, oy=oy)}" fill="none" stroke="{colour}" stroke-width="4" stroke-linejoin="round"/>')
        for index in np.linspace(0, path.sample_count - 1, 8, dtype=int):
            x, y = path.xy[index]
            tx, ty = path.tangent_xy[index]
            fragments.append(f'<line x1="{ox + scale*x:.2f}" y1="{oy-scale*y:.2f}" x2="{ox + scale*(x+0.10*tx):.2f}" y2="{oy-scale*(y+0.10*ty):.2f}" stroke="{colour}" stroke-width="1.5"/>')
        legend.append(f'<text x="20" y="{30 + row*22}" fill="{colour}" font-family="sans-serif" font-size="18">{name}</text>')
        payload["shapes"][name] = {
            "closed": path.closed,
            "length_m": path.length_m,
            "execution": {
                "truck_motion_mode": "parked" if path.closed else "creep",
                "truck_speed_mps": 0.0 if path.closed else 0.03,
            },
            "samples": [{"s_m": float(s), "x_m": float(x), "y_m": float(y), "tangent_x": float(t[0]), "tangent_y": float(t[1]), "normal_x": float(n[0]), "normal_y": float(n[1])} for s, (x, y), t, n in zip(path.s_m, path.xy, path.tangent_xy, path.normal_xy)],
        }
    fragments.extend(legend)
    fragments.append("</svg>\n")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(fragments))
    output.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")
    if route_output is not None:
        route_shapes = tuple(
            shapes[index % len(shapes)] for index in range(max(1, int(stations)))
        )
        _write_route_svg(
            paths,
            route_output,
            station_shapes=route_shapes,
            station_spacing_m=float(station_spacing_m),
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", default="line,sine_chain,circle,rounded_square")
    parser.add_argument("--samples", type=int, default=241)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/trajectory_2d/teacher.svg")
    parser.add_argument("--stations", type=int, default=4)
    parser.add_argument("--station_spacing_m", type=float, default=2.0)
    parser.add_argument("--route_output", type=Path, default=None)
    args = parser.parse_args()
    shapes = tuple(part.strip() for part in args.shapes.split(",") if part.strip())
    if not shapes or not set(shapes).issubset({"line", "sine_chain", "circle", "rounded_square"}):
        parser.error("--shapes must use line,sine_chain,circle,rounded_square")
    if args.stations < 1 or args.station_spacing_m <= 0.0:
        parser.error("--stations must be positive and --station_spacing_m must be positive")
    route_output = args.route_output or args.output.with_name(args.output.stem + "_route.svg")
    payload = write_svg(
        shapes,
        args.output,
        samples=max(args.samples, 32),
        stations=args.stations,
        station_spacing_m=args.station_spacing_m,
        route_output=route_output,
    )
    print(json.dumps({"mode": payload["mode"], "output": str(args.output), "json": str(args.output.with_suffix('.json')), "route": str(route_output), "shapes": {name: {"closed": row["closed"], "length_m": row["length_m"], "execution": row["execution"]} for name, row in payload["shapes"].items()}}, indent=2))


if __name__ == "__main__":
    main()
