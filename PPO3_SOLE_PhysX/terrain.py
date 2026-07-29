"""The verge strip: undulating ground beside a flat road.

A MULAG FME 500 is a verge mower.  The truck drives on the road and the arm
reaches out over the embankment, so the *truck* never leaves a smooth level
surface and only the *cutter* has to follow varying ground.  That split is the
whole geometry of the task, and it removes the class of problems that came from
driving the truck over generated terrain: no burying the chassis in a hillside
at spawn, no impact transients through a stiff arm, no ambiguity about ride
height, and no multi-million-triangle collision mesh under the wheels.

So this module deliberately does not use ``TerrainGeneratorCfg``.  That builds a
carpet for a vehicle to drive on, which is what teacher_student needs and what
this task explicitly does not.  Instead it builds one long, narrow height field
laid alongside the driving line, at the arm's measured reach.

Difficulty ramps with distance along the strip, so the ground genuinely gets
harder as the truck keeps driving rather than being uniformly random.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh


TRACKING_PATH_FAMILIES = (
    "straight",
    "lateral_sine",
    "s_curve",
    "random_spline",
    "offset_return",
    "boom_sweep",
    "lateral_chirp",
)


@dataclass(frozen=True)
class MowStripCfg:
    """Geometry and roughness of the verge the cutter has to track."""

    # Placement, in env-local coordinates. center_y comes from the measured
    # arm-reach band in PPO2's side-shelf config (mowing_terrain_local_y) and is
    # confirmed by the cutter's measured track (y = 1.38-1.42).
    #
    # start_x must sit BEHIND where the cutter begins. Measured 2026-07-28 the
    # deployed cutter starts at x = -27.12 and drives +x; starting the verge at
    # -26.0 left it mowing empty space until it caught up, and the height scan
    # reported only 0.35 valid over the run. -32.0 gives ~5 m of lead-in.
    start_x_m: float = -32.0
    center_y_m: float = 1.38
    length_m: float = 60.0
    width_m: float = 1.8

    # Sampling. 0.1 m along the drive direction resolves the undulation the
    # cutter must follow, and the strip is small enough that this stays cheap:
    # 60 x 4 m at 0.1 m is ~48k triangles, against 3.2M for a drivable carpet.
    resolution_m: float = 0.1

    base_z_m: float = 0.0
    """Mean height of the verge. 0.0 puts it level with the road."""

    elevation_rise_end_m: float = 0.12
    """Small background rise; broad irregular hills provide the main height."""

    hill_centers: tuple[float, ...] = (0.27, 0.56, 0.82)
    hill_heights_m: tuple[float, ...] = (0.18, 0.48, 0.27)
    hill_widths: tuple[float, ...] = (0.11, 0.16, 0.10)
    """Smooth non-uniform hill sections as fractions of strip length.

    The middle section is intentionally high and broad. Gaussian shoulders keep
    it visually realistic without introducing steps or difficult sharp edges.
    """

    tracking_reference_x_m: float = -25.4
    tracking_height_sweep_gain: float = 1.5
    tracking_path_family: str = "boom_sweep"
    """Curve the mowing path with the boom's measured high-terrain sweep."""

    # Roughness. Amplitude ramps linearly from `start` to `end` along the strip,
    # so the first metres are nearly flat and the ground becomes progressively
    # harder as the truck drives.
    # Start with an easy but visibly non-flat verge.  The controller must prove
    # that it follows height changes before roughness is increased for PPO.
    amplitude_start_m: float = 0.015
    amplitude_end_m: float = 0.06
    wavelength_long_m: float = 9.0
    wavelength_short_m: float = 4.5
    lateral_wavelength_m: float = 4.0
    cross_slope_end_rad: float = 0.005
    """Sideways tilt of the verge, ramping in like the amplitude."""

    seed: int = 42


def surface_height_at(cfg: MowStripCfg, x_local, y_local) -> np.ndarray:
    """Analytic surface height matching :func:`build_mow_strip_mesh`."""

    x, y = np.broadcast_arrays(
        np.asarray(x_local, dtype=np.float64),
        np.asarray(y_local, dtype=np.float64),
    )
    s = x - cfg.start_x_m
    t = y - cfg.center_y_m
    progress = np.clip(s / max(cfg.length_m, 1.0e-9), 0.0, 1.0)
    amplitude = cfg.amplitude_start_m + progress * (
        cfg.amplitude_end_m - cfg.amplitude_start_m
    )
    rng = np.random.default_rng(cfg.seed)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=4)
    # Seeded centre/height variation makes parallel environments genuinely
    # different while preserving the same easy, broad-hill difficulty class.
    hills = np.zeros_like(progress)
    for center, peak, width in zip(
        cfg.hill_centers, cfg.hill_heights_m, cfg.hill_widths
    ):
        varied_center = center + rng.uniform(-0.025, 0.025)
        varied_peak = peak * rng.uniform(0.90, 1.10)
        varied_width = width * rng.uniform(0.90, 1.10)
        hills = hills + varied_peak * np.exp(
            -0.5 * ((progress - varied_center) / varied_width) ** 2
        )
    # Broad hills plus low-frequency surface variation: visibly non-uniform,
    # but differentiable and gentle enough for continuous cutter contact.
    height = cfg.base_z_m + progress * cfg.elevation_rise_end_m + hills + amplitude * (
        0.6 * np.sin(2.0 * np.pi * s / cfg.wavelength_long_m + phase[0])
        + 0.3 * np.sin(2.0 * np.pi * s / cfg.wavelength_short_m + phase[1])
        + 0.1 * np.sin(2.0 * np.pi * t / cfg.lateral_wavelength_m + phase[2])
    )
    return height + progress * cfg.cross_slope_end_rad * t


def build_mow_strip_mesh(cfg: MowStripCfg) -> trimesh.Trimesh:
    """Build the verge surface as a triangle mesh in env-local coordinates.

    Deterministic for a given ``cfg.seed``, so a training run and its evaluation
    can be placed on the same or deliberately different ground.
    """
    nx = max(int(round(cfg.length_m / cfg.resolution_m)) + 1, 2)
    ny = max(int(round(cfg.width_m / cfg.resolution_m)) + 1, 2)

    x = cfg.start_x_m + np.linspace(0.0, cfg.length_m, nx)
    y = cfg.center_y_m + np.linspace(-0.5 * cfg.width_m, 0.5 * cfg.width_m, ny)
    xx, yy = np.meshgrid(x, y, indexing="ij")

    z = surface_height_at(cfg, xx, yy)

    vertices = np.column_stack((xx.ravel(), yy.ravel(), z.ravel()))

    # Two triangles per grid cell.
    i, j = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), indexing="ij")
    v00 = (i * ny + j).ravel()
    v01 = v00 + 1
    v10 = v00 + ny
    v11 = v10 + 1
    faces = np.vstack(
        (
            np.column_stack((v00, v10, v11)),
            np.column_stack((v00, v11, v01)),
        )
    )
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def build_tracking_line_mesh(
    cfg: MowStripCfg,
    *,
    width_m: float = 0.06,
    z_offset_m: float = 0.012,
) -> trimesh.Trimesh:
    """Build a thin ribbon projected onto the terrain centre line.

    It is visual-only and sits slightly above the green mesh to avoid z-fighting.
    The cutter's lateral target is the same ``center_y_m``, so this is the exact
    path the arm must follow rather than a decorative approximation.
    """

    nx = max(int(round(cfg.length_m / cfg.resolution_m)) + 1, 2)
    x = cfg.start_x_m + np.linspace(0.0, cfg.length_m, nx)
    half_width = 0.5 * max(float(width_m), 1.0e-3)
    center_y = tracking_line_center_y_at(cfg, x)
    y = np.stack(
        (
            center_y - half_width,
            center_y + half_width,
        ),
        axis=1,
    )
    xx = np.repeat(x[:, None], 2, axis=1)
    zz = surface_height_at(cfg, xx, y) + float(z_offset_m)
    vertices = np.column_stack((xx.ravel(), y.ravel(), zz.ravel()))
    i = np.arange(nx - 1)
    v00 = 2 * i
    v01 = v00 + 1
    v10 = v00 + 2
    v11 = v00 + 3
    faces = np.vstack(
        (
            np.column_stack((v00, v10, v11)),
            np.column_stack((v00, v11, v01)),
        )
    )
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def tracking_line_center_y_at(cfg: MowStripCfg, x_local) -> np.ndarray:
    """Return one of seven deterministic, smooth, strip-bounded path families."""

    x = np.asarray(x_local, dtype=np.float64)
    family = str(cfg.tracking_path_family)
    if family not in TRACKING_PATH_FAMILIES:
        raise ValueError(
            f"unknown tracking path family {family!r}; expected one of "
            f"{TRACKING_PATH_FAMILIES}"
        )
    rng = np.random.default_rng(cfg.seed + 104_729)
    active_length = max(
        cfg.start_x_m + cfg.length_m - cfg.tracking_reference_x_m, 1.0e-6
    )
    q = np.clip((x - cfg.tracking_reference_x_m) / active_length, 0.0, 1.0)
    offset = np.zeros_like(x)

    if family == "lateral_sine":
        amplitude = rng.uniform(0.06, 0.09)
        cycles = rng.uniform(1.25, 1.75)
        offset = amplitude * np.sin(2.0 * np.pi * cycles * q)
    elif family == "s_curve":
        amplitude = rng.uniform(0.08, 0.12)
        offset = amplitude * np.sin(2.0 * np.pi * q)
    elif family == "random_spline":
        knots = np.linspace(0.0, 1.0, 7)
        values = rng.uniform(-0.10, 0.10, size=knots.size)
        values[0] = 0.0
        values[-1] *= 0.35
        segment = np.minimum(
            np.searchsorted(knots, q, side="right") - 1, knots.size - 2
        )
        segment = np.maximum(segment, 0)
        u = (q - knots[segment]) / (knots[segment + 1] - knots[segment])
        blend = 0.5 - 0.5 * np.cos(np.pi * u)
        offset = values[segment] + blend * (
            values[segment + 1] - values[segment]
        )
    elif family == "offset_return":
        amplitude = rng.uniform(0.10, 0.14) * rng.choice((-1.0, 1.0))
        offset = amplitude * np.sin(np.pi * q) ** 2
    elif family == "boom_sweep":
        reference_height = surface_height_at(
            cfg,
            np.asarray(cfg.tracking_reference_x_m),
            np.asarray(cfg.center_y_m),
        )
        height = surface_height_at(cfg, x, np.full_like(x, cfg.center_y_m))
        offset = cfg.tracking_height_sweep_gain * (height - reference_height)
    elif family == "lateral_chirp":
        amplitude = rng.uniform(0.06, 0.08)
        phase = 2.0 * np.pi * (0.75 * q + 1.25 * q**2)
        offset = amplitude * np.sin(phase)

    # Preserve a green margin even for the height-coupled boom path.
    margin = 0.16
    half_available = max(0.5 * cfg.width_m - margin, 0.0)
    return cfg.center_y_m + np.clip(offset, -half_available, half_available)


def strip_height_at(cfg: MowStripCfg, x_local) -> np.ndarray:
    """Analytic centre-line height of the strip. Diagnostics only.

    The environment reads the surface with a ray caster, as a real machine would
    sense it. This exists so a probe can check the ray caster against the
    geometry it was built from, rather than trusting the sensor blindly -- the
    previous ride-height metric looked entirely plausible while being wrong.
    """
    x = np.asarray(x_local, dtype=np.float64)
    return surface_height_at(cfg, x, np.full_like(x, cfg.center_y_m))


# Training verge, and a held-out one at a different seed and rougher end state,
# so an evaluation number measures generalization to unseen ground.
MOW_STRIP_CFG = MowStripCfg()
MOW_STRIP_EVAL_CFG = MowStripCfg(
    seed=1337,
    elevation_rise_end_m=0.15,
    amplitude_end_m=0.07,
    hill_heights_m=(0.20, 0.55, 0.31),
)
