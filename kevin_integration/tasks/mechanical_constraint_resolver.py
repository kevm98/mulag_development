from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


MECHANICAL_CONSTRAINT_FAMILIES: Mapping[str, tuple[str, ...]] = {
    "Drehzapfen_joint": (
        "Drehzapfen_constraint_joint_1",
        "Drehzapfen_constraint_joint_3",
        "Drehzapfen_joint",
    ),
    "Ausleger_I_joint": (
        "Ausleger_I_constraint_joint_1",
        "Ausleger_I_constraint_joint_3",
        "Ausleger_I_joint",
    ),
    "Ausleger_II_joint": (
        "Ausleger_II_constraint_joint_1",
        "Ausleger_II_constraint_joint_3",
        "Ausleger_II_joint",
    ),
    "Messerkopf_Schwenk_joint": (
        "Messerkopf_Schwenk_constraint_joint_1",
        "Messerkopf_Schwenk_constraint_joint_3",
        "Messerkopf_Schwenk_constraint_joint_5",
        "Messerkopf_Schwenk_constraint_joint_6",
        "Messerkopf_Schwenk_joint",
    ),
    "Messerkopf_joint": ("Messerkopf_joint",),
}


REQUIRED_MECHANICAL_FAMILY_JOINTS = frozenset(MECHANICAL_CONSTRAINT_FAMILIES)


@dataclass(frozen=True)
class MechanicalConstraintResolution:
    family_ids: dict[str, list[int]]
    family_names: dict[str, list[str]]
    missing_optional: dict[str, list[str]]


def resolve_mechanical_constraint_family_ids(
    available_joint_names: Sequence[str],
    *,
    families: Mapping[str, Sequence[str]] = MECHANICAL_CONSTRAINT_FAMILIES,
    required_joint_names: frozenset[str] = REQUIRED_MECHANICAL_FAMILY_JOINTS,
) -> MechanicalConstraintResolution:
    """Resolve Kevin hydraulic linkage families from IsaacLab runtime joint names."""
    joint_ids_by_name = {name: index for index, name in enumerate(available_joint_names)}
    family_ids: dict[str, list[int]] = {}
    family_names: dict[str, list[str]] = {}
    missing_optional: dict[str, list[str]] = {}
    missing_required: list[str] = []

    for family, expected_names in families.items():
        ids: list[int] = []
        names: list[str] = []
        optional_missing_for_family: list[str] = []
        for name in expected_names:
            joint_id = joint_ids_by_name.get(name)
            if joint_id is None:
                if name in required_joint_names:
                    missing_required.append(name)
                else:
                    optional_missing_for_family.append(name)
                continue
            ids.append(joint_id)
            names.append(name)
        family_ids[family] = ids
        family_names[family] = names
        if optional_missing_for_family:
            missing_optional[family] = optional_missing_for_family

    if missing_required:
        raise RuntimeError(
            "Could not find required Kevin mechanical family command/state joints: "
            f"{', '.join(missing_required)}"
        )

    return MechanicalConstraintResolution(
        family_ids=family_ids,
        family_names=family_names,
        missing_optional=missing_optional,
    )
