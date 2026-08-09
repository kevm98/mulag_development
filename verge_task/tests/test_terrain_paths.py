"""Pure geometry contracts for the expanded PPO3 terrain curriculum."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from verge_task.terrain import (
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


def test_frequency_and_composition_are_opt_in_and_seeded():
    x = np.linspace(-32.0, 24.0, 20_001)
    base = replace(MOW_STRIP_CFG, length_m=56.0, tracking_path_family="s_curve")
    explicit_legacy = replace(
        base,
        tracking_path_frequency_scale=1.0,
        tracking_path_frequency_jitter=0.0,
        tracking_composite_enabled=False,
    )
    np.testing.assert_array_equal(
        tracking_line_center_y_at(base, x),
        tracking_line_center_y_at(explicit_legacy, x),
    )

    adaptive = replace(
        base,
        tracking_active_length_m=44.0,
        tracking_path_amplitude_scale=8.0,
        tracking_path_frequency_scale=1.6,
        tracking_path_frequency_jitter=0.2,
        tracking_composite_enabled=True,
    )
    first = tracking_line_center_y_at(adaptive, x)
    replay = tracking_line_center_y_at(adaptive, x)
    other = tracking_line_center_y_at(replace(adaptive, seed=43), x)
    np.testing.assert_array_equal(first, replay)
    assert not np.allclose(first, other)
    assert np.max(np.abs(first - adaptive.center_y_m)) <= 0.74


def test_composite_segment_joins_have_zero_velocity_and_acceleration():
    cfg = replace(
        MOW_STRIP_CFG,
        tracking_active_length_m=44.0,
        tracking_path_amplitude_scale=8.0,
        tracking_path_frequency_scale=1.6,
        tracking_composite_enabled=True,
    )
    count = len(cfg.tracking_composite_families)
    segment_length = cfg.tracking_active_length_m / count
    h = 1.0e-4
    for index in range(1, count):
        join = cfg.tracking_reference_x_m + index * segment_length
        samples = tracking_line_center_y_at(
            cfg, np.asarray((join - h, join, join + h))
        )
        velocity = (samples[2] - samples[0]) / (2.0 * h)
        acceleration = (samples[2] - 2.0 * samples[1] + samples[0]) / h**2
        assert abs(samples[1] - cfg.center_y_m) < 1.0e-12
        assert abs(velocity) < 1.0e-5
        assert abs(acceleration) < 1.0e-3


def test_frequency_scale_raises_s_curve_demand_by_an_order_of_magnitude():
    speed_mps = 4.0 / 3.6
    x = np.linspace(-32.0, 20.4, 100_001)
    base = replace(
        MOW_STRIP_CFG,
        tracking_path_family="s_curve",
        tracking_path_amplitude_scale=2.2,
    )
    hard = replace(
        base,
        tracking_path_amplitude_scale=8.0,
        tracking_path_frequency_scale=1.6,
        tracking_active_length_m=42.7244,
        tracking_composite_enabled=True,
    )

    def peaks(cfg):
        y = tracking_line_center_y_at(cfg, x)
        velocity = np.gradient(y, x) * speed_mps
        acceleration = np.gradient(velocity, x) * speed_mps
        return np.max(np.abs(velocity)), np.max(np.abs(acceleration))

    base_velocity, base_acceleration = peaks(base)
    hard_velocity, hard_acceleration = peaks(hard)
    # Compare with the measured present-day demand in the task brief. The
    # generator's endpoint window has a larger mathematical peak than the
    # scored interior, so a ratio against that peak would test the wrong datum.
    assert hard_velocity > 10.0 * 0.13
    assert hard_acceleration > 10.0 * 0.05


def test_adaptive_trajectory_helper_is_explicit_and_updates_geometry():
    from types import SimpleNamespace

    from verge_task.terrain import (
        enable_adaptive_trajectory_curriculum,
    )

    cfg = SimpleNamespace(
        vehicle_speed_min_kmh=4.0,
        vehicle_speed_max_kmh=4.0,
        scene=SimpleNamespace(env_spacing=28.0),
    )
    enable_adaptive_trajectory_curriculum(cfg)
    assert cfg.episode_length_s == 40.0
    assert cfg.terrain_composite_trajectory is True
    assert cfg.terrain_tracking_frequency_scale == 1.6
    assert 42.0 < cfg.terrain_tracking_active_length_m < 43.0
    assert cfg.scene.env_spacing > cfg.mow_strip_length_m


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
    # (0.0,) is deliberately NOT in this list any more: a zero scale is a valid
    # request for perfectly flat ground, so that a pure tracking environment
    # can be mixed in alongside the chunked ones. Empty schedules, negative
    # scales and negative overrides stay rejected.
    for bad_scales, bad_override in (((), 0.0), ((-0.1,), 0.0), ((1.0,), -0.1)):
        try:
            terrain_height_scale_for_environment(bad_scales, 0, bad_override)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid height-scale configuration was accepted")


def test_gentle_relief_profile_is_the_identity():
    """Stages 1-4 and the held-out set must keep the surface they were scored on.

    Every existing PPO3 number was measured on the gentle verge. If applying the
    profile ever perturbs it, those scores silently stop being comparable.
    """
    from dataclasses import replace

    from verge_task.terrain import (
        MOW_STRIP_CFG,
        MOW_STRIP_EVAL_CFG,
        apply_relief_profile,
        surface_height_at,
    )

    for base in (MOW_STRIP_CFG, MOW_STRIP_EVAL_CFG):
        cfg = replace(base, length_m=26.0)
        assert apply_relief_profile(cfg, "gentle") is cfg
        x = np.linspace(cfg.start_x_m, cfg.start_x_m + cfg.length_m, 400)
        y = np.full_like(x, cfg.center_y_m)
        # Every lawn term is inert on the gentle profile, so the surface is
        # non-negative: it never dips below the tyre plane.
        assert surface_height_at(cfg, x, y).min() >= 0.0


def test_lawn_rough_dips_below_road_rises_higher_and_is_rougher():
    """The stage-5 contract: below road level, taller mounds, real roughness."""
    from dataclasses import replace

    from verge_task.terrain import (
        MOW_STRIP_EVAL_CFG,
        apply_relief_profile,
        surface_height_at,
    )

    base = replace(MOW_STRIP_EVAL_CFG, length_m=26.0, resolution_m=0.15)
    rough = apply_relief_profile(base, "lawn_rough")
    x = np.arange(base.start_x_m, base.start_x_m + base.length_m, 0.01)
    y = np.full_like(x, base.center_y_m)
    z_gentle = surface_height_at(base, x, y)
    z_rough = surface_height_at(rough, x, y)

    # Below the tyre plane over a meaningful stretch, but only by centimetres --
    # deep pits would demand vertical travel the arm has not got.
    assert -0.09 < z_rough.min() < -0.02
    assert 0.10 < float((z_rough < 0.0).mean()) < 0.30
    # Sections that sit level with the road, between the mounds and the scoops.
    assert float((np.abs(z_rough) < 0.03).mean()) > 0.10
    # Mounds a little above the gentle peak, not a different league.
    assert 1.02 < z_rough.max() / z_gentle.max() < 1.20
    # Genuinely rougher: the gentle verge peaks near 8 deg.
    slope_rough = np.abs(np.gradient(z_rough, 0.01)).max()
    slope_gentle = np.abs(np.gradient(z_gentle, 0.01)).max()
    assert slope_rough > 2.0 * slope_gentle
    # Reach guard: the arm already saturates in Q4 on the gentle verge, so the
    # vertical span must not run away.
    assert (z_rough.max() - z_rough.min()) < 1.3 * (z_gentle.max() - z_gentle.min())


def test_lawn_rough_rejects_unknown_profile():
    from verge_task.terrain import MOW_STRIP_CFG, apply_relief_profile

    try:
        apply_relief_profile(MOW_STRIP_CFG, "bumpy")
    except ValueError as exc:
        assert "terrain_relief_profile" in str(exc)
    else:
        raise AssertionError("unknown relief profile must raise")


def test_decoration_seed_accepts_the_negative_offsets_scene_geometry_produces():
    """Bank centres are negative in env-local y, and numpy rejects negative seeds.

    The near grass bank sits at y = -1.12 for the production strip, so the offset
    reaching this helper is -1120. Passing that straight to default_rng raised
    "expected non-negative integer" and aborted scene setup.
    """
    from verge_task.terrain import decoration_seed

    for base in (0, 17, 42, 1337, 10018):
        for offset in (-1120, -3200, -1, 0, 1, 480, 12900):
            seed = decoration_seed(base, offset)
            assert isinstance(seed, int)
            assert 0 <= seed < 2**32
            np.random.default_rng(seed)  # must not raise

    # Deterministic, and nearby offsets must not share a stream.
    assert decoration_seed(17, -1120) == decoration_seed(17, -1120)
    assert decoration_seed(17, -1120) != decoration_seed(17, -1121)
    assert decoration_seed(17, -1120) != decoration_seed(18, -1120)


def test_tracking_path_amplitude_scale_widens_without_changing_shape():
    """1.0 must be bit-identical; the scale must widen every family uniformly.

    The families author 0.10-0.24 m of lateral swing against 0.74 m of available
    half-strip, so the arm was exercised over a third of the room the verge
    supports. Scaling must preserve each family's SHAPE (and therefore their
    relative difficulty ordering), not just its extent.
    """
    import numpy as np
    from dataclasses import replace
    from verge_task.terrain import (
        MOW_STRIP_CFG,
        TRACKING_PATH_FAMILIES,
        tracking_line_center_y_at,
    )

    x = np.linspace(-25.4, -25.4 + 16.0, 200)
    for family in TRACKING_PATH_FAMILIES:
        base = replace(MOW_STRIP_CFG, tracking_path_family=family)
        unit = replace(base, tracking_path_amplitude_scale=1.0)
        assert np.array_equal(
            tracking_line_center_y_at(base, x), tracking_line_center_y_at(unit, x)
        ), f"{family}: scale 1.0 must reproduce the unscaled path exactly"

        if family == "straight":
            continue
        wide = tracking_line_center_y_at(
            replace(base, tracking_path_amplitude_scale=1.8), x
        )
        narrow = tracking_line_center_y_at(base, x)
        assert np.ptp(wide) > np.ptp(narrow) * 1.4, f"{family} did not widen"
        # Shape preserved: the widened path stays on the same side of centre
        # wherever the original is clearly off-centre.
        off = np.abs(narrow - MOW_STRIP_CFG.center_y_m) > 0.02
        assert np.all(
            np.sign(wide[off] - MOW_STRIP_CFG.center_y_m)
            == np.sign(narrow[off] - MOW_STRIP_CFG.center_y_m)
        ), f"{family}: scaling changed the path shape, not just its extent"


def test_training_widens_the_path_but_the_gated_config_does_not():
    """Same split as terrain_height_scales: widen training, pin the gate.

    Measured on the shipped checkpoint, which was never trained for a wider
    path: 1.2x -> 14/17, 1.4x -> 13/17, 1.8x -> 9/17 held-out gates. Enabling
    this by default would silently downgrade the verified result.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    cfg_source = (root / "terrain_env_cfg.py").read_text()
    assert "terrain_tracking_amplitude_scale: float = 1.0" in cfg_source

    train_source = (root / "scripts" / "train.py").read_text()
    # Asserted as two halves rather than one literal line, because the set of
    # focus modes that must keep the path at 1.0 grows: `--tail_focus`
    # reproduces the measured head-alignment tail, `--contact_focus` the
    # measured contact-loss region, and both were measured at 1.0x. Pinning
    # the exact expression made adding the second one look like a regression.
    assert "env_cfg.terrain_tracking_amplitude_scale = (" in train_source, (
        "train.py no longer sets the training path amplitude"
    )
    assert "a.tail_focus or a.contact_focus or a.boom_focus" in train_source, (
        "a focus mode would train against a widened path it was not measured on"
    )
    assert "else 1.8" in train_source, (
        "unfocused training no longer widens the path beyond the gated 1.0"
    )


# ----------------------------------------------------------------------
# Stage 6: flat ground at road level with randomly placed raised chunks
# ----------------------------------------------------------------------


def _flat_patch_cfg(seed: int):
    from dataclasses import replace

    from verge_task.terrain import MOW_STRIP_EVAL_CFG, apply_relief_profile

    base = replace(MOW_STRIP_EVAL_CFG, length_m=26.0, resolution_m=0.15, seed=seed)
    return apply_relief_profile(base, "flat_patches")


def _flat_patch_surface(seed: int, step: float = 0.02):
    from verge_task.terrain import surface_height_at

    cfg = _flat_patch_cfg(seed)
    x = np.arange(cfg.start_x_m, cfg.start_x_m + cfg.length_m, step)
    y = np.full_like(x, cfg.center_y_m)
    return cfg, x, surface_height_at(cfg, x, y)


def test_flat_patches_is_mostly_flat_at_ground_level():
    """The defining property: long runs at road level, not everywhere-undulating.

    Stage 5's verge undulates everywhere, which makes cut height the dominant
    difficulty. Stage 6 inverts that -- the ground is flat and the chunks are
    discrete disturbances -- so 'mostly flat' is the contract, not a nicety.
    """
    for seed in (42, 909, 1337, 4242):
        cfg, _, z = _flat_patch_surface(seed)
        flat = float((np.abs(z) < 0.03).mean())
        assert flat > 0.45, f"seed {seed}: only {flat:.2f} of the strip is flat"
        # Flat means flat AT ROAD LEVEL, not at some raised plateau.
        assert abs(float(np.median(z))) < 0.03, f"seed {seed}: flat run is offset"


def test_flat_patches_actually_has_raised_chunks():
    for seed in (42, 909, 1337, 4242):
        cfg, _, z = _flat_patch_surface(seed)
        assert 3 <= len(cfg.hill_heights_m) <= 6
        assert z.max() > 0.09, f"seed {seed}: no chunk worth negotiating"
        # And it dips slightly below road level, so the arm works both ways.
        assert -0.06 < z.min() < -0.005, f"seed {seed}: z.min()={z.min():.3f}"


def test_flat_patch_layout_is_random_per_seed_and_deterministic_per_seed():
    """Robustness comes from every environment seeing a different layout.

    Each environment carries its own terrain seed (terrain_seed_stride), so a
    policy trained across them cannot memorise one mesh -- but the SAME seed
    must still reproduce exactly, or a run and its evaluation diverge.
    """
    layouts = {}
    for seed in (42, 909, 1337, 4242, 17):
        cfg = _flat_patch_cfg(seed)
        layouts[seed] = (cfg.hill_centers, cfg.hill_heights_m, cfg.hill_widths)
        assert _flat_patch_cfg(seed) == cfg, "same seed did not reproduce"
    assert len({repr(v) for v in layouts.values()}) == len(layouts), (
        "different seeds produced the same patch layout"
    )
    counts = {len(v[1]) for v in layouts.values()}
    assert len(counts) > 1, "patch COUNT never varies across seeds"


def test_flat_patch_slopes_stay_followable():
    """Width is derived from height so a tall narrow draw cannot make a wall.

    An unfollowable slope produces a failure that looks like a policy problem
    and is not one.
    """
    from verge_task.terrain import FLAT_PATCH_MAX_SLOPE

    for seed in (42, 909, 1337, 4242, 17, 5, 99):
        cfg, _, z = _flat_patch_surface(seed)
        slope = float(np.abs(np.gradient(z, 0.02)).max())
        # The patch bound plus the small fine-texture term on top of it.
        assert slope < FLAT_PATCH_MAX_SLOPE + 0.06, f"seed {seed}: slope {slope:.3f}"
        assert np.isfinite(z).all()


def test_flat_patches_drops_the_everywhere_undulation_terms():
    """The terms that made stage 5 undulate everywhere must be off."""
    cfg = _flat_patch_cfg(42)
    assert cfg.elevation_rise_end_m == 0.0
    assert cfg.cross_slope_end_rad == 0.0
    assert cfg.lateral_weight == 0.0
    assert cfg.amplitude_end_m <= 0.012


def test_stage_six_widens_the_tracking_path_and_selects_flat_patches():
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "terrain_env_cfg.py"
    ).read_text()
    stage6 = source[source.index("elif stage == 6:") :]
    stage6 = stage6[: stage6.index("@configclass")]
    assert 'cfg.terrain_relief_profile = "flat_patches"' in stage6
    assert "cfg.terrain_tracking_amplitude_scale = 2.2" in stage6
    # Stage 6 must not quietly change the arm contract; only ground and path.
    for forbidden in ("sole_max_action_rate", "action_scale", "arm_stiffness"):
        assert forbidden not in stage6, (
            f"stage 6 changes {forbidden}; it must differ from stage 5 only in "
            "the ground and the tracking path"
        )


def test_training_does_not_clobber_stage_sixs_wide_path():
    """Stage 6's amplitude IS the stage. train.py must not overwrite it.

    train.py sets terrain_tracking_amplitude_scale after applying the
    curriculum, so without this guard stage 6 silently becomes a narrow-path
    task and measures something other than what it was built for.
    """
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "scripts" / "train.py"
    ).read_text()
    assert "if 6 not in stages:" in source
    guard = source.index("if 6 not in stages:")
    assign = source.index("env_cfg.terrain_tracking_amplitude_scale = (", guard)
    assert assign - guard < 400, "the stage-6 guard does not cover the assignment"


def test_every_environment_gets_its_own_chunk_layout():
    """The randomisation must reach the PATCH LAYOUT, not just the phases.

    `_mow_strip_cfg` applies the relief profile using the base seed, which
    fixes hill_centers/heights/widths once. Re-seeding that RESULT per
    environment -- which the spawner used to do -- changes only the sinusoid
    phases and the +-10 % jitter, so all 256 parallel environments carried
    identical chunks.

    Measured consequence: training across nine blocks saw nine layouts, not
    9 x 256, and held-out adaptivity was a coin flip (9/18 cells, alignment
    0.663-0.995, bimodal rather than smoothly degrading).
    """
    from dataclasses import replace

    from verge_task.terrain import MOW_STRIP_CFG, apply_relief_profile

    base = replace(MOW_STRIP_CFG, length_m=26.0, resolution_m=0.15, seed=42)
    stride = 7919

    # WRONG (the old order): profile first, then re-seed.
    profiled = apply_relief_profile(base, "flat_patches")
    wrong = [
        replace(profiled, seed=profiled.seed + i * stride).hill_heights_m
        for i in range(6)
    ]
    assert len(set(map(repr, wrong))) == 1, (
        "test premise is stale: re-seeding after the profile now varies layout"
    )

    # RIGHT (the fixed order): re-seed first, then draw the profile.
    right = [
        apply_relief_profile(
            replace(base, seed=base.seed + i * stride), "flat_patches"
        ).hill_heights_m
        for i in range(6)
    ]
    assert len(set(map(repr, right))) == 6, (
        "per-environment seeds must produce distinct chunk layouts"
    )
    assert len({len(h) for h in right}) > 1, "patch COUNT never varies across envs"


def test_the_spawner_draws_the_profile_after_the_per_env_seed():
    from pathlib import Path

    src = (
        Path(__file__).resolve().parents[1] / "terrain_env.py"
    ).read_text()
    body = src[src.index("def _spawn_mow_strip") :]
    body = body[: body.index("def _add_ground_sensors")]
    seed_at = body.index("seed=unprofiled_cfg.seed + env_index * seed_stride")
    draw_at = body.index("apply_relief_profile(env_strip_cfg, relief_profile)")
    assert seed_at < draw_at, "the profile is drawn before the per-env seed"
    assert "_mow_strip_cfg_unprofiled()" in body, (
        "the spawner re-profiles an already-profiled cfg; lawn_rough's "
        "hill_scale reads max(hill_heights_m) and would compound"
    )


def test_zero_height_scale_gives_a_flat_environment():
    """0.0 means flat, not an error: it is how a pure tracking case is mixed in."""
    from verge_task.terrain import terrain_height_scale_for_environment

    scales = (0.0, 0.5, 1.0)
    picked = [terrain_height_scale_for_environment(scales, i) for i in range(6)]
    assert picked == [0.0, 0.5, 1.0, 0.0, 0.5, 1.0]
    # A zero override still means "no override", so the schedule keeps control.
    assert terrain_height_scale_for_environment(scales, 1, 0.0) == 0.5
    assert terrain_height_scale_for_environment(scales, 1, 0.75) == 0.75


def test_negative_height_scale_is_still_rejected():
    from verge_task.terrain import terrain_height_scale_for_environment

    import pytest

    with pytest.raises(ValueError):
        terrain_height_scale_for_environment((-0.1, 0.5), 0)


def test_strip_width_sets_the_available_half_swing():
    """Half-swing is 0.5*width - 0.16; widening is what forces arm extension."""
    from verge_task.terrain import MOW_STRIP_CFG, tracking_line_center_y_at

    import numpy as np
    from dataclasses import replace

    x = np.linspace(-32.0, -6.0, 4001)
    narrow = replace(
        MOW_STRIP_CFG, width_m=1.8, tracking_path_amplitude_scale=2.2,
        tracking_path_family="lateral_sine",
    )
    wide = replace(narrow, width_m=3.0)

    narrow_swing = float(np.ptp(tracking_line_center_y_at(narrow, x)))
    wide_swing = float(np.ptp(tracking_line_center_y_at(wide, x)))

    # Same family and same amplitude scale: the ONLY reason the wide track
    # swings further is that the tanh bound saturates later.
    assert wide_swing > narrow_swing
    for cfg_, limit in ((narrow, 0.5 * 1.8 - 0.16), (wide, 0.5 * 3.0 - 0.16)):
        offsets = tracking_line_center_y_at(cfg_, x) - cfg_.center_y_m
        assert np.max(np.abs(offsets)) <= limit + 1e-9


def test_strip_and_verge_share_one_colour():
    """The tracked strip must not read as a different green slab on the lawn."""
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "terrain_env_cfg.py"
    ).read_text()
    strip = source.split("mow_strip_color: tuple = ")[1].split("\n")[0]
    verge = source.split("verge_visual_color: tuple = ")[1].split("\n")[0]
    assert strip == verge, f"strip {strip} != verge {verge}"
