"""Deployment-domain factor selection for the shared verge task.

The factor vector is privileged experiment metadata.  It is latched at reset,
used only to perturb the frozen deployment plant, and is never concatenated to
policy observations or the causal AAM row.
"""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path


DIM_ORDER = ("flow", "compliance", "seal", "pressure", "mass_scale", "com_z")
HYDRAULIC_DIMS = ("flow", "compliance", "seal", "pressure")
MECHANICAL_DIMS = ("mass_scale", "com_z")
RANGES = {
    "flow": (-0.25, 0.25),
    "compliance": (-0.25, 0.10),
    "seal": (0.0, 1.0),
    "pressure": (-0.15, 0.15),
    "mass_scale": (0.8, 1.5),
    "com_z": (-0.10, 0.05),
}
DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "AAM_module/datasets/rom_deployment_stage31_v1_frozen/raw/manifest.csv"
)


@dataclass(frozen=True)
class DeploymentFactor:
    """One immutable, episode-level factor configuration."""

    run_id: int
    domain: str
    split: str
    values: tuple[float, float, float, float, float, float]
    label: str = ""

    @property
    def key(self) -> str:
        if self.label:
            return self.label
        split = "combined_train" if self.domain == "combined" and self.split == "train" else (
            "combined_test" if self.domain == "combined" and self.split == "test" else self.domain
        )
        return f"{split}:{self.run_id}"

    def as_dict(self) -> dict[str, float]:
        return dict(zip(DIM_ORDER, self.values))

    def hydraulic_system_factors(self) -> tuple[float, ...]:
        e = self.as_dict()
        # HydraulicROM's fixed channel order is
        # [compliance, flow, pressure, seal, seal, seal, seal].
        return (
            e["compliance"], e["flow"], e["pressure"],
            e["seal"], e["seal"], e["seal"], e["seal"],
        )


def nominal_factor(run_id: int = 0) -> DeploymentFactor:
    return DeploymentFactor(
        run_id=int(run_id),
        domain="nominal",
        split="train",
        values=(0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    )


def _builtin_factor(key: str) -> DeploymentFactor | None:
    """Return explicit boundary anchors used by the Gate-1 sweep."""
    anchors = {
        "flow_min": ("hydraulic", "train", (-0.25, 0.0, 0.0, 0.0, 1.0, 0.0)),
        "flow_nominal": ("nominal", "train", (0.0, 0.0, 0.0, 0.0, 1.0, 0.0)),
        "flow_max": ("hydraulic", "train", (0.25, 0.0, 0.0, 0.0, 1.0, 0.0)),
        "compliance_min": ("hydraulic", "train", (0.0, -0.25, 0.0, 0.0, 1.0, 0.0)),
        "compliance_max": ("hydraulic", "train", (0.0, 0.10, 0.0, 0.0, 1.0, 0.0)),
        "seal_min": ("hydraulic", "train", (0.0, 0.0, 0.0, 0.0, 1.0, 0.0)),
        "seal_max": ("hydraulic", "train", (0.0, 0.0, 1.0, 0.0, 1.0, 0.0)),
        "pressure_min": ("hydraulic", "train", (0.0, 0.0, 0.0, -0.15, 1.0, 0.0)),
        "pressure_max": ("hydraulic", "train", (0.0, 0.0, 0.0, 0.15, 1.0, 0.0)),
    }
    row = anchors.get(str(key).strip().lower())
    if row is None:
        return None
    domain, split, values = row
    return DeploymentFactor(run_id=-1, domain=domain, split=split, values=values, label=key)


def _parse_row(row: dict[str, str]) -> DeploymentFactor:
    domain = str(row["domain"]).strip().lower()
    split = str(row["split"]).strip().lower()
    if domain not in {"nominal", "hydraulic", "mechanical", "combined"}:
        raise ValueError(f"unsupported deployment factor domain {domain!r}")
    values = tuple(float(row[name]) for name in DIM_ORDER)
    for name, value in zip(DIM_ORDER, values):
        lo, hi = RANGES[name]
        if not math.isfinite(value) or value < lo - 1.0e-6 or value > hi + 1.0e-6:
            raise ValueError(f"factor {name}={value} is outside audited range {lo}..{hi}")
    return DeploymentFactor(
        run_id=int(row["run_id"]), domain=domain, split=split, values=values
    )


def load_manifest(path: str | Path = DEFAULT_MANIFEST) -> list[DeploymentFactor]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"deployment factor manifest not found: {path}")
    with path.open(newline="") as handle:
        rows = [_parse_row(row) for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError(f"deployment factor manifest is empty: {path}")
    return rows


def _matches_domain(row: DeploymentFactor, domain: str) -> bool:
    domain = str(domain).strip().lower()
    if domain == "combined_train":
        return row.domain == "combined" and row.split == "train"
    if domain == "combined_test":
        return row.domain == "combined" and row.split == "test"
    return row.domain == domain and row.split == "train"


def select_factors(
    *,
    domain: str,
    key: str = "",
    num_envs: int,
    seed: int,
    manifest: str | Path = DEFAULT_MANIFEST,
    reset_index: int = 0,
) -> list[DeploymentFactor]:
    """Select one latched factor row for each environment.

    ``key`` accepts a built-in Gate-1 anchor, a manifest run id, or a manifest
    key such as ``combined_test:102``.  Without a key, rows in the selected
    domain are assigned deterministically per environment; no global random
    state is touched.
    """
    if int(num_envs) <= 0:
        raise ValueError("num_envs must be positive")
    explicit = _builtin_factor(key) if key else None
    rows = load_manifest(manifest)
    if explicit is None and key:
        key_text = str(key).strip().lower()
        run_id: int | None = None
        if key_text.isdigit():
            run_id = int(key_text)
        else:
            prefix, separator, suffix = key_text.rpartition(":")
            if separator and suffix.isdigit():
                run_id = int(suffix)
        candidates = [row for row in rows if row.run_id == run_id] if run_id is not None else []
        if not candidates:
            candidates = [row for row in rows if row.key.lower() == key_text]
        if len(candidates) != 1:
            raise KeyError(f"factor key {key!r} did not resolve to exactly one manifest row")
        explicit = candidates[0]
    if explicit is not None:
        return [explicit] * int(num_envs)

    candidates = [row for row in rows if _matches_domain(row, domain)]
    if not candidates:
        raise KeyError(f"factor domain {domain!r} has no rows in {manifest}")
    # The local generator is deterministic and keyed by the episode reset, so
    # every arm sees the same factor keys when seed and reset index match.
    rng = random.Random(int(seed) + 1000003 * int(reset_index) + 7919 * len(domain))
    offset = rng.randrange(len(candidates))
    return [candidates[(offset + env_id) % len(candidates)] for env_id in range(int(num_envs))]


def validate_factor_isolation(*, policy_observation_width: int, expected_width: int, aam_feature_dim: int | None) -> None:
    """Fail loudly if factor plumbing changes the deployable interfaces."""
    if int(policy_observation_width) != int(expected_width):
        raise RuntimeError(
            "deployment factors must not change policy observation width: "
            f"expected {expected_width}, got {policy_observation_width}"
        )
    if aam_feature_dim is not None and int(aam_feature_dim) != 18:
        raise RuntimeError(
            "deployment factors must not change the causal AAM feature contract: "
            f"expected 18, got {aam_feature_dim}"
        )


__all__ = [
    "DEFAULT_MANIFEST",
    "DIM_ORDER",
    "DeploymentFactor",
    "HYDRAULIC_DIMS",
    "MECHANICAL_DIMS",
    "RANGES",
    "load_manifest",
    "nominal_factor",
    "select_factors",
    "validate_factor_isolation",
]
