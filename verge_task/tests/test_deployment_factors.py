from __future__ import annotations

from pathlib import Path

from verge_task.deployment_factors import (
    DEFAULT_MANIFEST,
    select_factors,
    validate_factor_isolation,
)


def test_audited_manifest_domains_and_reserved_test_are_readable():
    rows = select_factors(
        domain="combined_test",
        num_envs=16,
        seed=17,
        manifest=DEFAULT_MANIFEST,
    )
    assert len(rows) == 16
    assert all(row.domain == "combined" and row.split == "test" for row in rows)
    assert len({row.run_id for row in rows}) == 16


def test_gate1_anchors_are_explicit_and_do_not_touch_manifest():
    before = Path(DEFAULT_MANIFEST).read_bytes()
    low = select_factors(domain="hydraulic", key="flow_min", num_envs=2, seed=17)
    nominal = select_factors(domain="hydraulic", key="flow_nominal", num_envs=2, seed=17)
    high = select_factors(domain="hydraulic", key="flow_max", num_envs=2, seed=17)
    assert [row.key for row in low] == ["flow_min", "flow_min"]
    assert [row.values[0] for row in (low[0], nominal[0], high[0])] == [-0.25, 0.0, 0.25]
    assert Path(DEFAULT_MANIFEST).read_bytes() == before


def test_domain_assignment_is_deterministic_and_per_environment():
    first = select_factors(domain="hydraulic", num_envs=8, seed=123, reset_index=4)
    second = select_factors(domain="hydraulic", num_envs=8, seed=123, reset_index=4)
    assert [row.key for row in first] == [row.key for row in second]
    assert all(row.domain == "hydraulic" for row in first)


def test_factor_isolation_contract_rejects_observation_width_changes():
    validate_factor_isolation(policy_observation_width=66, expected_width=66, aam_feature_dim=None)
    validate_factor_isolation(policy_observation_width=68, expected_width=68, aam_feature_dim=18)
