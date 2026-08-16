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

from dataclasses import dataclass, replace

import math

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
    "double_s_curve",
    "lane_change_hold",
    "gaussian_slalom",
    "fourier_meander",
)

DEFAULT_TERRAIN_HEIGHT_SCALES = (0.50, 0.70, 0.85, 1.00)


def apply_natural_verge_vertex_colors(
    mesh: trimesh.Trimesh,
    palette: tuple,
    *,
    soil_threshold: float = 1.0,
    asphalt_edge_y: float | None = None,
    asphalt_color: tuple = (0.03, 0.035, 0.04),
) -> float:
    """Color an existing terrain mesh as grass/soil without adding geometry."""
    colors = np.asarray(palette, dtype=np.float64)
    if colors.ndim != 2 or colors.shape[0] < 5 or colors.shape[1] != 3:
        raise ValueError("mow_strip_visual_palette must contain four grass RGB colors and soil")
    if not np.isfinite(colors).all() or np.any(colors < 0.0) or np.any(colors > 1.0):
        raise ValueError("mow_strip_visual_palette RGB values must be finite and in [0, 1]")

    xy = np.asarray(mesh.vertices[:, :2], dtype=np.float64)
    x, y = xy[:, 0], xy[:, 1]
    # Several incommensurate bands make broken, interlocking patches instead
    # of the large round tan islands produced by one low-frequency sine.  This
    # remains deterministic and costs no geometry or texture memory.
    patch_field = (
        0.62 * np.sin(0.83 * x + 1.15 * y)
        + 0.43 * np.sin(2.21 * x - 0.77 * y + 0.8)
        + 0.29 * np.cos(3.47 * x + 2.37 * y)
        + 0.18 * np.sin(5.13 * x - 3.11 * y + 0.2)
    )
    grass_field = np.sin(0.67 * x - 0.53 * y) + 0.45 * np.cos(1.29 * x + 0.91 * y)
    dry_field = np.sin(0.39 * x + 1.37 * y + 0.4)

    def smoothstep(low: float, high: float, values):
        t = np.clip((values - low) / max(high - low, 1.0e-9), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    green_mix = smoothstep(-1.25, 1.25, grass_field)[:, None]
    rgb = colors[0] * (1.0 - green_mix) + colors[2] * green_mix
    dry_mix = (0.35 * smoothstep(0.25, 1.0, dry_field))[:, None]
    rgb = rgb * (1.0 - dry_mix) + colors[3] * dry_mix
    soil_mix = smoothstep(
        float(soil_threshold) - 0.25,
        float(soil_threshold) + 0.25,
        patch_field,
    )[:, None]
    rgb = rgb * (1.0 - soil_mix) + colors[4] * soil_mix

    asphalt = np.zeros(len(x), dtype=bool)
    if asphalt_edge_y is not None:
        road_rgb = np.asarray(asphalt_color, dtype=np.float64)
        if road_rgb.shape != (3,) or not np.isfinite(road_rgb).all():
            raise ValueError("asphalt_color must be a finite RGB tuple")
        asphalt = y >= float(asphalt_edge_y)
        rgb[asphalt] = np.clip(road_rgb, 0.0, 1.0)

    rgba = np.empty((len(rgb), 4), dtype=np.uint8)
    rgba[:, :3] = np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    rgba[:, 3] = 255
    mesh.visual.vertex_colors = rgba
    visible_soil = (soil_mix[:, 0] >= 0.5) & ~asphalt
    return float(np.mean(visible_soil)) if len(visible_soil) else 0.0


def build_visual_verge_skin(
    mesh: trimesh.Trimesh,
    *,
    road_edge_y: float,
    shoulder_transition_m: float = 0.30,
    road_height_m: float = 0.0,
) -> trimesh.Trimesh:
    """Return a road-clipped, visually flattened copy of a verge mesh.

    The input is the untouched collision surface.  Only this returned display
    skin is clipped at the road edge and eased down over the final shoulder, so
    the road/grass boundary is straight while ray casts and contacts continue
    to use the original terrain exactly.
    """
    if shoulder_transition_m <= 0.0:
        raise ValueError("shoulder_transition_m must be positive")

    vertices = np.asarray(mesh.vertices, dtype=np.float64).copy()
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("mesh vertices must have shape (N, 3)")

    # Ease the rendered verge to road level before clipping.  The smoothstep
    # has zero slope at both ends, avoiding a shiny-looking hard crease.
    blend = np.clip(
        (vertices[:, 1] - (float(road_edge_y) - shoulder_transition_m))
        / shoulder_transition_m,
        0.0,
        1.0,
    )
    blend = blend * blend * (3.0 - 2.0 * blend)
    vertices[:, 2] = (
        vertices[:, 2] * (1.0 - blend) + float(road_height_m) * blend
    )

    inside = vertices[:, 1] <= float(road_edge_y) + 1.0e-10
    kept_vertices = vertices[inside].tolist()
    old_to_new = np.full(len(vertices), -1, dtype=np.int64)
    old_to_new[np.flatnonzero(inside)] = np.arange(np.count_nonzero(inside))
    kept_faces: list[tuple[int, int, int]] = []

    def intersection_index(a: int, b: int) -> int:
        va, vb = vertices[a], vertices[b]
        dy = vb[1] - va[1]
        if abs(dy) < 1.0e-12:
            point = va.copy()
        else:
            t = (float(road_edge_y) - va[1]) / dy
            point = va + np.clip(t, 0.0, 1.0) * (vb - va)
        point[1] = float(road_edge_y)
        point[2] = float(road_height_m)
        kept_vertices.append(point.tolist())
        return len(kept_vertices) - 1

    for face in faces:
        face_inside = inside[face]
        if face_inside.all():
            kept_faces.append(tuple(int(old_to_new[index]) for index in face))
            continue
        if not face_inside.any():
            continue

        # Sutherland-Hodgman clipping against y <= road_edge_y.  A triangle
        # becomes either one triangle or a four-vertex polygon, triangulated as
        # a fan. Boundary intersections are intentionally local to this face;
        # the seam is tiny and avoiding a global edge map keeps this pure helper
        # simple and deterministic.
        polygon: list[int] = []
        previous = int(face[-1])
        previous_inside = bool(inside[previous])
        for raw_current in face:
            current = int(raw_current)
            current_inside = bool(inside[current])
            if current_inside:
                if not previous_inside:
                    polygon.append(intersection_index(previous, current))
                polygon.append(int(old_to_new[current]))
            elif previous_inside:
                polygon.append(intersection_index(previous, current))
            previous, previous_inside = current, current_inside
        for index in range(1, len(polygon) - 1):
            kept_faces.append((polygon[0], polygon[index], polygon[index + 1]))

    if not kept_faces:
        raise ValueError("road edge clips away the entire visual verge mesh")
    return trimesh.Trimesh(
        vertices=np.asarray(kept_vertices, dtype=np.float64),
        faces=np.asarray(kept_faces, dtype=np.int64),
        process=False,
    )


def build_sparse_grass_tuft_mesh(
    cfg: MowStripCfg,
    *,
    count: int,
    road_edge_y: float,
    seed: int,
    tracking_clearance_m: float = 0.20,
    edge_clearance_m: float = 0.18,
    height_range_m: tuple[float, float] = (0.10, 0.19),
    width_range_m: tuple[float, float] = (0.07, 0.14),
    palette: tuple = (
        (0.035, 0.13, 0.018),
        (0.055, 0.18, 0.025),
        (0.085, 0.22, 0.035),
    ),
) -> trimesh.Trimesh:
    """Build deterministic four-blade grass clumps over the real surface.

    Each clump is only four triangles.  They are presentation geometry and are
    kept clear of the tracked white line and road seam so those two functional
    visual references stay legible.
    """
    count = int(count)
    if count < 0:
        raise ValueError("grass tuft count must be non-negative")
    if count == 0:
        return trimesh.Trimesh(
            vertices=np.empty((0, 3)), faces=np.empty((0, 3), dtype=np.int64)
        )
    h0, h1 = (float(value) for value in height_range_m)
    w0, w1 = (float(value) for value in width_range_m)
    if not 0.0 < h0 <= h1 or not 0.0 < w0 <= w1:
        raise ValueError("grass tuft height/width ranges must be positive and ordered")
    colors = np.asarray(palette, dtype=np.float64)
    if colors.ndim != 2 or colors.shape[1] != 3 or len(colors) == 0:
        raise ValueError("grass tuft palette must contain RGB colors")

    rng = np.random.default_rng(int(seed) % (2**32))
    x_min = float(cfg.start_x_m) + 0.25
    x_max = float(cfg.start_x_m + cfg.length_m) - 0.25
    y_min = float(cfg.center_y_m - 0.5 * cfg.width_m) + edge_clearance_m
    y_max = min(
        float(cfg.center_y_m + 0.5 * cfg.width_m) - edge_clearance_m,
        float(road_edge_y) - edge_clearance_m,
    )
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("grass tuft placement region is empty")

    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    vertex_colors: list[tuple[int, int, int, int]] = []
    accepted = 0
    attempts = 0
    max_attempts = max(200, 30 * count)
    while accepted < count and attempts < max_attempts:
        attempts += 1
        x = float(rng.uniform(x_min, x_max))
        y = float(rng.uniform(y_min, y_max))
        line_y = float(tracking_line_center_y_at(cfg, np.asarray([x]))[0])
        # Reserve enough room for the blade tips as well as the clump centre;
        # otherwise a nominally clear clump can still cover the white line.
        blade_envelope = 0.5 * w1 + 0.34 * h1
        if abs(y - line_y) < float(tracking_clearance_m) + blade_envelope:
            continue
        z = float(surface_height_at(cfg, np.asarray([x]), np.asarray([y]))[0])
        height = float(rng.uniform(h0, h1))
        width = float(rng.uniform(w0, w1))
        rotation = float(rng.uniform(0.0, 0.5 * np.pi))
        lean = float(rng.uniform(0.18, 0.34)) * height
        color = tuple(
            int(value)
            for value in np.rint(
                np.clip(colors[int(rng.integers(0, len(colors)))], 0.0, 1.0)
                * 255.0
            )
        ) + (255,)
        for blade in range(4):
            angle = rotation + blade * 0.5 * np.pi
            radial = np.asarray((np.cos(angle), np.sin(angle)))
            tangent = np.asarray((-radial[1], radial[0]))
            base = len(vertices)
            left = np.asarray((x, y)) - 0.5 * width * tangent
            right = np.asarray((x, y)) + 0.5 * width * tangent
            tip = np.asarray((x, y)) + lean * radial
            vertices.extend(
                (
                    (float(left[0]), float(left[1]), z + 0.004),
                    (float(right[0]), float(right[1]), z + 0.004),
                    (float(tip[0]), float(tip[1]), z + height),
                )
            )
            faces.append((base, base + 1, base + 2))
            vertex_colors.extend((color, color, color))
        accepted += 1

    if accepted != count:
        raise RuntimeError(f"placed only {accepted}/{count} grass tufts")
    result = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    result.visual.vertex_colors = np.asarray(vertex_colors, dtype=np.uint8)
    return result


def enable_adaptive_trajectory_curriculum(
    cfg,
    *,
    episode_length_s: float = 40.0,
    amplitude_scale: float = 8.0,
    frequency_scale: float = 1.6,
    frequency_jitter: float = 0.20,
) -> None:
    """Enable the opt-in multi-manoeuvre path without changing legacy stages."""

    duration = float(episode_length_s)
    if duration <= 16.0:
        raise ValueError("adaptive trajectory episodes must be longer than 16 s")
    if amplitude_scale <= 0.0 or frequency_scale <= 0.0:
        raise ValueError("trajectory amplitude and frequency scales must be positive")
    if not 0.0 <= frequency_jitter < 1.0:
        raise ValueError("trajectory frequency jitter must be in [0, 1)")
    speed_mps = 0.5 * (
        float(cfg.vehicle_speed_min_kmh) + float(cfg.vehicle_speed_max_kmh)
    ) / 3.6
    # The deployed cutter begins 1.72 m before the authored path reference.
    # Subtract that lead-in so the last segment finishes before timeout.
    active_length = duration * speed_mps - 1.72
    cfg.episode_length_s = duration
    cfg.terrain_composite_trajectory = True
    cfg.terrain_tracking_amplitude_scale = float(amplitude_scale)
    cfg.terrain_tracking_frequency_scale = float(frequency_scale)
    cfg.terrain_tracking_frequency_jitter = float(frequency_jitter)
    cfg.terrain_tracking_active_length_m = active_length
    cfg.mow_strip_length_m = 6.6 + active_length + 3.0
    cfg.scene.env_spacing = cfg.mow_strip_length_m + 2.0


def terrain_height_scale_for_environment(
    scales: tuple[float, ...], env_index: int, override: float = 0.0
) -> float:
    """Select a non-negative, deterministic relief scale for one environment.

    0.0 is allowed and means PERFECTLY FLAT ground for that environment. Every
    consumer multiplies by this scale (elevation rise, hill heights, ripple
    amplitudes), so zero is well defined and simply removes the relief. It is
    how a "tracking only, no terrain disturbance" environment is requested:
    include 0.0 in `terrain_height_scales` and that fraction of the parallel
    environments run flat while the rest keep their chunks.

    Negative values remain rejected -- they would invert the terrain.
    """

    values = tuple(float(value) for value in scales)
    if not values or min(values) < 0.0:
        raise ValueError("terrain height scales must not be negative")
    override = float(override)
    if override < 0.0:
        raise ValueError("terrain height scale override cannot be negative")
    return override if override > 0.0 else values[int(env_index) % len(values)]


def terrain_parameter_for_environment(
    values: tuple[float, ...], env_index: int, stride: int = 1
) -> float:
    """Deterministically pick one terrain parameter value for one environment.

    The same cycling contract as ``terrain_height_scale_for_environment``, but
    reusable for any parameter we want to VARY ACROSS environments rather than
    pin. A policy generalises over the distribution it was trained on and
    essentially nothing else, so a parameter held at a single value during
    training is a parameter the policy will not tolerate moving at deployment.
    Measured here on 2026-08-06: a student distilled at
    ``tracking_path_amplitude_scale`` 2.2 delivered 0.303 m of lateral error on
    an amplitude-0.7 terrain that physically permits 0.058 m -- five times worse
    than the plant allows, purely from being asked a question it was not
    trained on.

    Prefer COPRIME TUPLE LENGTHS over a stride to decorrelate two parameters:
    4 amplitudes against 3 sweep gains covers all 12 pairs every 12
    environments with stride 1 for both. ``stride`` exists for the cases that
    cannot be arranged that way, and it must be coprime with ``len(values)``:
    a stride sharing a factor with the length collapses the cycle onto a subset
    (stride 3 on a length-3 tuple selects index 0 for EVERY environment), which
    pins the parameter to one value while still looking randomized. That is
    rejected here rather than left to be discovered in a trained policy.
    """
    entries = tuple(float(value) for value in values)
    if not entries:
        raise ValueError("terrain parameter values must not be empty")
    step = max(int(stride), 1)
    if math.gcd(step, len(entries)) != 1:
        raise ValueError(
            f"stride {step} shares a factor with {len(entries)} values, so the "
            "cycle would only reach "
            f"{len(entries) // math.gcd(step, len(entries))} of them; use a "
            "stride coprime with the tuple length (1 is always safe)"
        )
    return entries[(int(env_index) * step) % len(entries)]


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
    width_m: float = 3.0

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
    tracking_active_length_m: float = 16.0
    """Length over which one complete lateral path is executed."""

    tracking_height_sweep_gain: float = 1.5
    tracking_path_family: str = "boom_sweep"
    """Curve the mowing path with the boom's measured high-terrain sweep."""

    tracking_path_amplitude_scale: float = 1.0
    """Multiplier on every family's lateral excursion, applied before bounding.

    The families author amplitudes of 0.10-0.24 m while the strip offers
    ``0.5 * width_m - 0.16`` = 0.74 m of half-width, so the paths were using
    barely a third of the available room and the tanh bound never engaged --
    the arm was exercised over a much narrower lateral band than the verge
    supports. Scaling here rather than editing eleven literals keeps every
    family's SHAPE and their relative difficulty ordering intact, and 1.0
    reproduces every previously measured score bit-for-bit.
    """

    tracking_path_frequency_scale: float = 1.0
    """Frequency multiplier for cyclic families; 1.0 is legacy behavior."""
    tracking_path_frequency_jitter: float = 0.0
    """Seeded fractional frequency jitter per strip; 0.0 is legacy behavior."""
    tracking_composite_enabled: bool = False
    """Concatenate multiple centre-returning manoeuvres in one active path."""
    tracking_composite_families: tuple[str, ...] = (
        "straight",
        "s_curve",
        "gaussian_slalom",
        "lane_change_hold",
    )

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

    # --- Lawn relief (zero by default; the rough stage-5 profile sets these) ---
    # Hollows are the only way the verge goes BELOW road level: every other term
    # is non-negative, so the gentle profile never dips under the tyre plane.
    hollow_centers: tuple[float, ...] = ()
    hollow_depths_m: tuple[float, ...] = ()
    hollow_widths: tuple[float, ...] = ()
    """Gaussian depressions as fractions of length; depths are negative metres."""

    # Mid-band roughness. The gentle profile's shortest component is 4.5 m, i.e.
    # 0.25 Hz at 4 km/h -- quasi-static. These two add the 0.5-2 m band a real
    # lawn has, which is where the arm's own bandwidth is actually tested.
    wavelength_mid_m: float = 1.2
    mid_weight: float = 0.0
    wavelength_fine_m: float = 0.6
    fine_weight: float = 0.0
    lateral_weight: float = 0.1
    """Weight of the lateral ripple; the gentle profile leaves this at 0.1."""

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
    # Depressions carrying the surface below the tyre plane. Drawn from the same
    # generator as the mounds so a seed still reproduces the whole verge exactly.
    for center, depth, width in zip(
        cfg.hollow_centers, cfg.hollow_depths_m, cfg.hollow_widths
    ):
        varied_center = center + rng.uniform(-0.025, 0.025)
        varied_depth = depth * rng.uniform(0.90, 1.10)
        varied_width = width * rng.uniform(0.90, 1.10)
        hills = hills + varied_depth * np.exp(
            -0.5 * ((progress - varied_center) / varied_width) ** 2
        )
    # Broad hills plus low-frequency surface variation: visibly non-uniform,
    # but differentiable and gentle enough for continuous cutter contact.
    # Two extra phases for the mid/fine band. Drawn after the hill loop so the
    # gentle profile, which leaves both weights at zero, keeps the exact surface
    # it had before -- stage 1-4 runs and the fixed held-out set stay comparable.
    lawn_phase = rng.uniform(0.0, 2.0 * np.pi, size=2)
    height = cfg.base_z_m + progress * cfg.elevation_rise_end_m + hills + amplitude * (
        0.6 * np.sin(2.0 * np.pi * s / cfg.wavelength_long_m + phase[0])
        + 0.3 * np.sin(2.0 * np.pi * s / cfg.wavelength_short_m + phase[1])
        + cfg.lateral_weight
        * np.sin(2.0 * np.pi * t / cfg.lateral_wavelength_m + phase[2])
        + cfg.mid_weight
        * np.sin(2.0 * np.pi * s / max(cfg.wavelength_mid_m, 1.0e-6) + lawn_phase[0])
        + cfg.fine_weight
        * np.sin(2.0 * np.pi * s / max(cfg.wavelength_fine_m, 1.0e-6) + lawn_phase[1])
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
    """Return one deterministic, smooth, strip-bounded path family."""

    x = np.asarray(x_local, dtype=np.float64)
    if bool(getattr(cfg, "tracking_composite_enabled", False)):
        families = tuple(str(value) for value in cfg.tracking_composite_families)
        if len(families) < 2 or any(
            family not in TRACKING_PATH_FAMILIES for family in families
        ):
            raise ValueError(
                "tracking_composite_families must contain at least two known "
                f"families, got {families!r}"
            )
        active_length = max(float(cfg.tracking_active_length_m), 1.0e-6)
        segment_length = active_length / len(families)
        progress = (x - cfg.tracking_reference_x_m) / active_length
        result = np.full_like(x, cfg.center_y_m)
        order_rng = np.random.default_rng(cfg.seed + 2_000_033)
        ordered = tuple(
            families[index] for index in order_rng.permutation(len(families))
        )
        for index, family in enumerate(ordered):
            lower = index / len(ordered)
            upper = (index + 1) / len(ordered)
            mask = (progress >= lower) & (
                progress <= upper if index == len(ordered) - 1 else progress < upper
            )
            if not np.any(mask):
                continue
            segment_cfg = replace(
                cfg,
                seed=cfg.seed + (index + 1) * 1_000_003,
                tracking_path_family=family,
                tracking_reference_x_m=(
                    cfg.tracking_reference_x_m + index * segment_length
                ),
                tracking_active_length_m=segment_length,
                tracking_composite_enabled=False,
            )
            result[mask] = tracking_line_center_y_at(segment_cfg, x[mask])
        return result

    family = str(cfg.tracking_path_family)
    if family not in TRACKING_PATH_FAMILIES:
        raise ValueError(
            f"unknown tracking path family {family!r}; expected one of "
            f"{TRACKING_PATH_FAMILIES}"
        )
    rng = np.random.default_rng(cfg.seed + 104_729)
    frequency_scale = float(getattr(cfg, "tracking_path_frequency_scale", 1.0))
    frequency_jitter = float(getattr(cfg, "tracking_path_frequency_jitter", 0.0))
    if frequency_scale <= 0.0 or not 0.0 <= frequency_jitter < 1.0:
        raise ValueError(
            "tracking frequency scale must be positive and jitter must be in [0, 1)"
        )
    if frequency_jitter:
        frequency_scale *= rng.uniform(
            1.0 - frequency_jitter, 1.0 + frequency_jitter
        )
    active_length = max(float(cfg.tracking_active_length_m), 1.0e-6)
    q = np.clip((x - cfg.tracking_reference_x_m) / active_length, 0.0, 1.0)
    offset = np.zeros_like(x)

    def smootherstep01(value):
        value = np.clip(value, 0.0, 1.0)
        return value**3 * (value * (value * 6.0 - 15.0) + 10.0)

    # Returning curves join the straight lead-in/out with zero first and second
    # derivative. The boom sweep instead eases into a constant final offset.
    # Both remove the hidden steering impulse that the old sine/chirp
    # definitions introduced at clipped endpoints.
    endpoint_fraction = 0.10
    start_window = smootherstep01(q / endpoint_fraction)
    endpoint_window = start_window * smootherstep01(
        (1.0 - q) / endpoint_fraction
    )

    # CYCLE COUNTS HALVED ON THE PERIODIC FAMILIES, to pay for the 4.4x
    # amplitude. Peak lateral demand goes as amplitude x cycles, so widening
    # alone took lateral_chirp to 0.802 m/s against a 0.240 m/s ceiling --
    # `error ~ 0.465 * demand` (PATH_FEASIBILITY.md) predicts 0.37 m, over the
    # 0.300 gate. Halving the cycles restores the demand while KEEPING the width:
    # a long, broad wave instead of a fast one.
    #
    # This is done per family rather than through `frequency_scale` because that
    # multiplier is not safe below 1.0 for every shape: `offset_return` is
    # sin(pi*f*q)**2, which stops returning to zero at f<1 and gets slammed shut
    # by the endpoint window (measured: 0.083 -> 0.948 m/s, an 11x increase).
    # Only the families that are genuinely periodic in q are re-timed here.
    if family == "lateral_sine":
        # Cycle count must be a MULTIPLE OF 0.5 or sin(2*pi*c*q) does not land
        # on zero at q=1 and the endpoint window shuts it discontinuously. The
        # old 1.25-1.75 draw never landed on zero either; it got away with it
        # because the amplitude was small. At 4.4x it does not, so this is now
        # one full cycle exactly -- the long, broad wave, randomised in phase
        # sign rather than in frequency.
        amplitude = rng.uniform(0.10, 0.14) * rng.choice((-1.0, 1.0))
        offset = amplitude * np.sin(2.0 * np.pi * frequency_scale * q)
    elif family == "s_curve":
        amplitude = rng.uniform(0.12, 0.18)
        offset = amplitude * np.sin(np.pi * frequency_scale * q)
    elif family == "random_spline":
        knots = np.linspace(0.0, 1.0, 7)
        values = rng.uniform(-0.16, 0.16, size=knots.size)
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
        amplitude = rng.uniform(0.14, 0.20) * rng.choice((-1.0, 1.0))
        offset = amplitude * np.sin(np.pi * frequency_scale * q) ** 2
    elif family == "boom_sweep":
        # PURE XY, as of the widening. This family used to be the ONLY one whose
        # offset was a function of the ground:
        #     raw_offset = tracking_height_sweep_gain * (height - reference)
        # so its lateral demand was set by terrain roughness rather than by the
        # path, which is why it alone got harder as the mesh got rougher and why
        # it demanded 2.9-4.6x the cylinder rate ceiling while even the
        # privileged DLS teacher failed it (PATH_FEASIBILITY.md). A z-driven
        # trajectory also cannot be widened coherently: scaling its amplitude
        # scales a height reading, not a path.
        #
        # It is now a broad single-lobe sweep out and back in the XY plane --
        # the boom swinging wide across the verge and returning, which is what
        # the name describes. Amplitude is drawn like every other family, so the
        # shared amplitude/frequency scaling applies to it uniformly.
        amplitude = rng.uniform(0.16, 0.22) * rng.choice((-1.0, 1.0))
        offset = amplitude * np.sin(np.pi * q) ** 2
    elif family == "lateral_chirp":
        amplitude = rng.uniform(0.10, 0.14)
        phase = 2.0 * np.pi * frequency_scale * (0.375 * q + 0.625 * q**2)
        offset = amplitude * np.sin(phase)
    elif family == "double_s_curve":
        # TWO full cycles, which is what "double" means and what makes this
        # family distinct. It was `sin(2*pi*f*q)` -- byte-identical to
        # `lateral_sine` after that family was re-authored to one full cycle --
        # so the library advertised 11 families and generated 10. The signature
        # test caught it (10 != 11); the worst-family gate had been aggregating
        # over a redundant pair, and `path_family_counts` was reporting two
        # populations that were one.
        #
        # 2.0 keeps the cycle count a multiple of 0.5, so the path still lands
        # on zero at q=1 and the endpoint window does not shut it
        # discontinuously. It doubles the lateral rate demand: measured at the
        # shipped band this family sits at 0.055 m/s against the 0.240 m/s
        # cylinder ceiling, so twice that is 0.11 m/s and still inside.
        amplitude = rng.uniform(0.10, 0.14)
        offset = amplitude * np.sin(2.0 * np.pi * 2.0 * frequency_scale * q)
    elif family == "lane_change_hold":
        amplitude = rng.uniform(0.12, 0.18) * rng.choice((-1.0, 1.0))
        rise = smootherstep01((q - 0.12) / 0.18)
        fall = smootherstep01((q - 0.70) / 0.18)
        offset = amplitude * (rise - fall)
    elif family == "gaussian_slalom":
        amplitude = rng.uniform(0.11, 0.16)
        centers = np.asarray((0.23, 0.50, 0.76))
        widths = np.asarray((0.105, 0.14, 0.09))
        signs = np.asarray((1.0, -0.85, 0.75)) * rng.choice((-1.0, 1.0))
        offset = np.zeros_like(q)
        for center, width, sign in zip(centers, widths, signs):
            offset = offset + amplitude * sign * np.exp(
                -0.5 * ((q - center) / width) ** 2
            )
    elif family == "fourier_meander":
        amplitude = rng.uniform(0.10, 0.15)
        phases = rng.uniform(-0.35, 0.35, size=3)
        offset = amplitude * (
            0.62 * np.sin(1.0 * np.pi * frequency_scale * q + phases[0])
            + 0.26 * np.sin(2.0 * np.pi * frequency_scale * q + phases[1])
            + 0.12 * np.sin(3.0 * np.pi * frequency_scale * q + phases[2])
        )

    # Every family now gets the endpoint window. boom_sweep was excluded only
    # because its height-coupled form ended at a nonzero offset that had to be
    # eased separately; the XY form is a sin^2 lobe that is already zero at both
    # ends, so the window is a no-op there rather than a special case.
    offset = offset * endpoint_window

    # Widen every family uniformly. Applied AFTER the endpoint windows so the
    # zero-derivative joins at the lead-in/out are preserved -- scaling before
    # them would reintroduce the steering impulse those windows exist to remove.
    offset = offset * float(getattr(cfg, "tracking_path_amplitude_scale", 1.0))

    # Preserve a green margin even for the height-coupled boom path. Smooth
    # saturation avoids the flat, non-differentiable sections produced by clip.
    margin = 0.16
    half_available = max(0.5 * cfg.width_m - margin, 0.0)
    if half_available <= 0.0:
        return np.full_like(x, cfg.center_y_m)
    bounded_offset = half_available * np.tanh(offset / half_available)
    return cfg.center_y_m + bounded_offset


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

RELIEF_PROFILES = ("gentle", "lawn_rough", "flat_patches")

#: Bound on the along-track surface slope any generated patch may produce.
#: A Gaussian ``A*exp(-0.5*((p-c)/w)^2)`` over a strip of length ``L`` peaks in
#: slope at ``0.607*A/(w*L)``, so the patch WIDTH is derived from the drawn
#: height rather than drawn independently -- otherwise a tall narrow draw makes
#: a wall the arm cannot follow, and the failure looks like a policy problem.
FLAT_PATCH_MAX_SLOPE = 0.26


def generate_flat_patches(
    cfg: MowStripCfg, *, rng: np.random.Generator
) -> tuple[tuple, tuple, tuple, tuple, tuple, tuple]:
    """Draw a random set of raised chunks and shallow scoops for a flat verge.

    Returns ``(hill_centers, hill_heights, hill_widths, hollow_centers,
    hollow_depths, hollow_widths)``.

    The layout is drawn from ``rng``, which the caller seeds from ``cfg.seed``.
    Every environment carries its own seed (``terrain_seed_stride``), so each
    one gets a genuinely different number of patches in different places at
    different heights -- which is what makes a policy trained across them
    adaptive rather than fitted to one mesh.

    Centres are drawn on a shuffled grid rather than uniformly, so patches
    cannot pile up into one long ridge and leave the rest of the strip empty.
    The flat ground BETWEEN them is the point of this profile.
    """
    length = max(float(cfg.length_m), 1.0e-6)
    count = int(rng.integers(3, 7))

    # Shuffled slots across the usable span, then jittered inside each slot.
    # 0.08-0.94 keeps a patch off the lead-in, where the cutter is still
    # settling, and off the very end.
    slots = np.linspace(0.08, 0.94, count + 2)[1:-1]
    slot_width = float(slots[1] - slots[0]) if count > 1 else 0.2
    centers = slots + rng.uniform(-0.35, 0.35, size=count) * slot_width

    heights = rng.uniform(0.10, 0.38, size=count)
    # Width from height, so the slope bound holds by construction.
    min_widths = 0.607 * heights / (FLAT_PATCH_MAX_SLOPE * length)
    widths = min_widths * rng.uniform(1.05, 1.6, size=count)

    # A couple of shallow scoops so the flat sections are not glassy. Kept to a
    # few centimetres: the arm's measured vertical authority is the limit, and
    # the existing notes are explicit that deep pits ask for travel it has not
    # got.
    hollow_count = int(rng.integers(2, 5))
    hollow_slots = np.linspace(0.10, 0.92, hollow_count + 2)[1:-1]
    hollow_centers = hollow_slots + rng.uniform(-0.03, 0.03, size=hollow_count)
    hollow_depths = -rng.uniform(0.020, 0.045, size=hollow_count)
    hollow_widths = rng.uniform(0.020, 0.032, size=hollow_count)

    return (
        tuple(round(float(v), 4) for v in centers),
        tuple(round(float(v), 4) for v in heights),
        tuple(round(float(v), 4) for v in widths),
        tuple(round(float(v), 4) for v in hollow_centers),
        tuple(round(float(v), 4) for v in hollow_depths),
        tuple(round(float(v), 4) for v in hollow_widths),
    )


def decoration_seed(base_seed: int, offset: int) -> int:
    """Fold a signed offset into a seed numpy will accept.

    Scene decoration is keyed by geometry -- a bank's centre y, for instance --
    and those coordinates are routinely negative, which ``default_rng`` rejects
    outright. Mixing before the mask keeps nearby offsets on unrelated streams,
    so two banks a few centimetres apart do not get near-identical relief.
    """
    mixed = (int(base_seed) & 0xFFFFFFFF) * 1_000_003 + int(offset) * 2_654_435_761
    return int(mixed % (2**32))


def apply_relief_profile(cfg: MowStripCfg, profile: str) -> MowStripCfg:
    """Return ``cfg`` re-cut for a named relief class.

    ``gentle`` is the identity: stages 1-4 and the fixed held-out set keep the
    exact surface they were scored on, so existing numbers stay comparable.

    ``lawn_rough`` is the stage-5 verge. Three changes over ``gentle``:

    * Hollows carry parts of the surface *below* road level. Every other term is
      non-negative, so the gentle verge never dips under the tyre plane and the
      cutter never has to work down into a depression.
    * Mounds are lifted a little above the gentle profile's tallest.
    * Mid (1.2 m) and fine (0.6 m) bands fill the 0.5-2 m gap. The gentle verge
      has nothing shorter than 4.5 m, i.e. 0.25 Hz at 4 km/h, which is
      quasi-static; 1.2 m is 0.93 Hz and is where arm bandwidth actually bites.

    Cross-slope is raised only to 0.035 rad (2.0 deg) rather than the 5-10 deg a
    real verge is canted at. Cant is the most direct load on head attitude, and
    ``head_alignment_hard_min`` has never passed on the gentle verge -- so it is
    deliberately the least aggressive of these knobs. Raise it once head-down
    holds, not before.
    """
    profile = str(profile)
    if profile not in RELIEF_PROFILES:
        raise ValueError(
            f"terrain_relief_profile must be one of {RELIEF_PROFILES}, got {profile!r}"
        )
    if profile == "gentle":
        return cfg
    if profile == "flat_patches":
        # Flat ground at road level, with randomly placed raised chunks between
        # long flat runs. The tracking task, not the terrain-following task:
        # the arm holds a constant height over most of the strip and has to
        # negotiate discrete obstacles, instead of continuously chasing an
        # everywhere-undulating surface.
        #
        # Every shaping term that made the verge undulate EVERYWHERE is off:
        # no background elevation ramp, no cross-slope, no lateral ripple, and
        # only a whisper of long-wavelength amplitude so the flat is natural
        # rather than glassy. What is left is the patch set, drawn per seed.
        #
        # NOTE the interaction with `terrain_height_scales`: the caller scales
        # hill heights and amplitudes by the relief scale, so a 0.5 scale here
        # gives lower chunks on the same flat, which is the intended
        # difficulty axis.
        rng = np.random.default_rng(cfg.seed)
        (
            hill_centers,
            hill_heights,
            hill_widths,
            hollow_centers,
            hollow_depths,
            hollow_widths,
        ) = generate_flat_patches(cfg, rng=rng)
        return replace(
            cfg,
            hill_centers=hill_centers,
            hill_heights_m=hill_heights,
            hill_widths=hill_widths,
            hollow_centers=hollow_centers,
            hollow_depths_m=hollow_depths,
            hollow_widths=hollow_widths,
            elevation_rise_end_m=0.0,
            cross_slope_end_rad=0.0,
            lateral_weight=0.0,
            # A little fine texture so the flat reads as ground rather than a
            # table, at an amplitude far below the cutter's tracking band.
            wavelength_mid_m=1.6,
            mid_weight=0.05,
            wavelength_fine_m=0.7,
            fine_weight=0.03,
            amplitude_start_m=0.006,
            amplitude_end_m=0.010,
        )
    # Four mounds of deliberately unequal height, scaled to whichever base verge
    # this is so the training/held-out difference survives. The monotonic
    # elevation ramp is dropped entirely: it is what held the gentle verge
    # everywhere above the tyre plane, and without it the surface returns to road
    # level between mounds instead of climbing steadily.
    hill_scale = max(cfg.hill_heights_m) / 0.48 if cfg.hill_heights_m else 1.0
    return replace(
        cfg,
        hill_centers=(0.21, 0.47, 0.70, 0.89),
        hill_heights_m=tuple(
            round(value * hill_scale, 4) for value in (0.14, 0.42, 0.68, 0.24)
        ),
        hill_widths=(0.042, 0.052, 0.072, 0.038),
        # Shallow scoops, not pits: a few centimetres under the road, enough that
        # the cutter has to work down to them without asking for vertical travel
        # the arm has not got. The Q4 authority collapse on the gentle verge is
        # the reason these are 3-4 cm and not 15.
        hollow_centers=(0.10, 0.34, 0.58, 0.80, 0.96),
        hollow_depths_m=(-0.030, -0.040, -0.028, -0.035, -0.026),
        hollow_widths=(0.028, 0.026, 0.024, 0.026, 0.023),
        elevation_rise_end_m=0.0,
        wavelength_mid_m=1.2,
        mid_weight=0.26,
        wavelength_fine_m=0.6,
        fine_weight=0.09,
        lateral_weight=0.24,
        cross_slope_end_rad=0.030,
        amplitude_start_m=0.010,
        amplitude_end_m=max(cfg.amplitude_end_m * 0.60, 0.036),
    )
