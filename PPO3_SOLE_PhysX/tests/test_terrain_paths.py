"""Pure geometry contracts for the expanded PPO3 terrain curriculum."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from PPO3_SOLE_PhysX.terrain import (
    DEFAULT_TERRAIN_HEIGHT_SCALES,
    MOW_STRIP_CFG,
    MOW_STRIP_EVAL_CFG,
    TRACKING_PATH_FAMILIES,
    surface_height_at,
    terrain_height_scale_for_environment,
    tracking_line_center_y_at,
)


def _scaled(cfg, scale: float):
    return replace(
        cfg,
        length_m=26.0,
        elevation_rise_end_m=cfg.elevation_rise_end_m * scale,
        hill_heights_m=tuple(value * scale for value in cfg.hill_heights_m),
        amplitude_start_m=cfg.amplitude_start_m * scale,
        amplitude_end_m=cfg.amplitude_end_m * scale,
        cross_slope_end_rad=cfg.cross_slope_end_rad * scale,
    )


def test_expanded_path_library_is_finite_distinct_smooth_and_strip_bounded():
    assert len(TRACKING_PATH_FAMILIES) == 11
    x = np.linspace(-32.0, -6.0, 10_001)
    paths = {}
    for family in TRACKING_PATH_FAMILIES:
        cfg = replace(
            MOW_STRIP_EVAL_CFG,
            length_m=26.0,
            tracking_path_family=family,
        )
        y = tracking_line_center_y_at(cfg, x)
        paths[family] = y
        assert np.isfinite(y).all()
        assert np.max(np.abs(y - cfg.center_y_m)) <= 0.5 * cfg.width_m - 0.16
        assert np.max(np.abs(np.gradient(y, x))) <= 0.18

    assert np.ptp(paths["straight"]) == 0.0
    for family in set(TRACKING_PATH_FAMILIES) - {"straight"}:
        assert np.ptp(paths[family]) >= 0.10
    signatures = {np.round(values, 5).tobytes() for values in paths.values()}
    assert len(signatures) == len(TRACKING_PATH_FAMILIES)


def test_paths_are_deterministic_per_seed_and_change_across_seeds():
    x = np.linspace(-32.0, -6.0, 1001)
    cfg = replace(
        MOW_STRIP_CFG, length_m=26.0, tracking_path_family="fourier_meander"
    )
    first = tracking_line_center_y_at(cfg, x)
    second = tracking_line_center_y_at(cfg, x)
    other = tracking_line_center_y_at(replace(cfg, seed=cfg.seed + 1), x)
    np.testing.assert_array_equal(first, second)
    assert not np.allclose(first, other)


def test_every_family_is_crossed_with_every_height_scale():
    pairs = {
        (
            TRACKING_PATH_FAMILIES[index % len(TRACKING_PATH_FAMILIES)],
            terrain_height_scale_for_environment(
                DEFAULT_TERRAIN_HEIGHT_SCALES, index
            ),
        )
        for index in range(
            len(TRACKING_PATH_FAMILIES) * len(DEFAULT_TERRAIN_HEIGHT_SCALES)
        )
    }
    expected = {
        (family, scale)
        for family in TRACKING_PATH_FAMILIES
        for scale in DEFAULT_TERRAIN_HEIGHT_SCALES
    }
    assert pairs == expected


def test_height_curriculum_is_visible_but_remains_smooth():
    x = np.linspace(-32.0, -6.0, 2001)
    ranges = []
    for base in (MOW_STRIP_CFG, MOW_STRIP_EVAL_CFG):
        for scale in DEFAULT_TERRAIN_HEIGHT_SCALES:
            cfg = _scaled(base, scale)
            z = surface_height_at(cfg, x, np.full_like(x, cfg.center_y_m))
            assert np.isfinite(z).all()
            assert np.max(np.abs(np.gradient(z, x))) <= 0.15
            ranges.append(float(np.ptp(z)))
    assert min(ranges) >= 0.25
    assert max(ranges) >= 0.65


def test_height_scale_override_is_explicit_and_validated():
    assert terrain_height_scale_for_environment((0.5, 1.0), 1) == 1.0
    assert terrain_height_scale_for_environment((0.5, 1.0), 1, 0.8) == 0.8
    for bad_scales, bad_override in (((), 0.0), ((0.0,), 0.0), ((1.0,), -0.1)):
        try:
            terrain_height_scale_for_environment(bad_scales, 0, bad_override)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid height-scale configuration was accepted")
