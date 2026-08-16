"""User-facing selection of the wide task's tracked-line shapes."""

from __future__ import annotations

from verge_task.terrain import TRACKING_PATH_FAMILIES


TRAJECTORY_SHAPES = tuple(TRACKING_PATH_FAMILIES)


def parse_trajectory_shapes(
    *, single: str = "", multiple: str = ""
) -> tuple[str, ...] | None:
    """Parse one shape, a comma-separated subset, or ``all``.

    ``None`` means leave the task's configured mixed schedule untouched.  The
    term *trajectory shape* here always means the white line tracked by the
    cutter; the truck itself continues along the fixed straight road.
    """
    single = str(single).strip()
    multiple = str(multiple).strip()
    if single and multiple:
        raise ValueError(
            "use either --trajectory-shape or --trajectory-shapes, not both"
        )
    if not single and not multiple:
        return None

    requested = single or multiple
    if requested in {"all", "mixed"}:
        return TRAJECTORY_SHAPES
    values = tuple(part.strip() for part in requested.split(",") if part.strip())
    if not values:
        raise ValueError("trajectory shape selection is empty")
    unknown = sorted(set(values) - set(TRAJECTORY_SHAPES))
    if unknown:
        raise ValueError(
            f"unknown trajectory shape(s) {unknown}; available: "
            + ", ".join(TRAJECTORY_SHAPES)
        )
    return tuple(dict.fromkeys(values))


def apply_trajectory_shapes(cfg, shapes: tuple[str, ...] | None) -> None:
    """Apply a validated tracked-line schedule to an environment config."""
    if shapes is None:
        return
    if len(shapes) == 1:
        cfg.tracking_path_family = shapes[0]
        cfg.tracking_path_family_schedule = ()
    else:
        cfg.tracking_path_family = "mixed"
        cfg.tracking_path_family_schedule = tuple(shapes)


def require_enough_environments(
    shapes: tuple[str, ...] | None, num_envs: int
) -> None:
    """Refuse a multi-shape viewer command that cannot display every shape."""
    if shapes is not None and len(shapes) > int(num_envs):
        raise ValueError(
            f"{len(shapes)} trajectory shapes requested but --num-envs is "
            f"{num_envs}; use at least {len(shapes)} environments"
        )


__all__ = [
    "TRAJECTORY_SHAPES",
    "apply_trajectory_shapes",
    "parse_trajectory_shapes",
    "require_enough_environments",
]
