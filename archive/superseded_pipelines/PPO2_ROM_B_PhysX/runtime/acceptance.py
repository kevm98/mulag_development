"""Predeclared acceptance statistic for ROM-B termination gates.

The acceptance bound for one condition is

    upper = max(
        one-sided Clopper-Pearson upper bound on the pooled replica count,
        one-sided t upper bound on the per-seed-group rates,
    )

Neither term alone is safe:

* Pooled Clopper-Pearson assumes the replicas within a condition are exchangeable
  Bernoulli draws. On `line` the between-seed-group SD is roughly half the mean,
  so that assumption is optimistic and the bound is too narrow.
* The seed-group t bound carries no within-group binomial uncertainty. When every
  group reports the same count its standard error is zero, so `0/640` would
  certify a true rate of exactly zero, which no finite test can establish.

Taking the maximum keeps whichever failure mode binds. It is an engineering rule,
not an exact interval: a predeclared beta-binomial or hierarchical logistic model
would be the formal treatment, and with fewer than ~10 seed groups its
heterogeneity estimate is itself poorly determined. Both terms are always reported
so the decision can be audited.

Aggregation rules, fixed by predeclaration:

* Monte Carlo: one bound per shape, pooling only across seed groups.
* h sweep: one bound per (h, shape), pooling only across seed groups.
* Never pool across shapes, and never pool across h.
* Worst shape controls acceptance within a condition.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from scipy.stats import beta, chi2, t


DEFAULT_CONFIDENCE = 0.95
DEFAULT_MAX_TERMINATION_RATE = 0.05


def clopper_pearson_upper(
    failures: int, replicas: int, *, confidence: float = DEFAULT_CONFIDENCE
) -> float:
    """One-sided upper confidence bound for a binomial rate."""

    if replicas <= 0:
        raise ValueError("replicas must be positive")
    if not 0 <= failures <= replicas:
        raise ValueError("failures must lie in [0, replicas]")
    if failures == replicas:
        return 1.0
    return float(beta.ppf(confidence, failures + 1, replicas - failures))


def seed_group_t_upper(
    rates: tuple[float, ...] | list[float],
    *,
    confidence: float = DEFAULT_CONFIDENCE,
) -> float | None:
    """One-sided upper bound on the mean of per-seed-group rates.

    Returns None for a single group, where between-group spread is unmeasurable.
    """

    groups = [float(value) for value in rates]
    count = len(groups)
    if count < 2:
        return None
    mean = sum(groups) / count
    variance = sum((value - mean) ** 2 for value in groups) / (count - 1)
    standard_error = (variance / count) ** 0.5
    return float(mean + t.ppf(confidence, count - 1) * standard_error)


def quasi_binomial_dispersion(
    failures_per_group: tuple[int, ...] | list[int], group_size: int
) -> float | None:
    """Tarone dispersion phi; ~1 is binomial, >1 is between-group heterogeneity.

    Reported for audit only. It never enters the acceptance decision: with a
    handful of groups it has too few degrees of freedom to gate on.
    """

    counts = [int(value) for value in failures_per_group]
    groups = len(counts)
    if groups < 2 or group_size <= 0:
        return None
    rate = sum(counts) / (groups * group_size)
    if rate <= 0.0 or rate >= 1.0:
        return None
    statistic = sum(
        (count - group_size * rate) ** 2 for count in counts
    ) / (group_size * rate * (1.0 - rate))
    return float(statistic / (groups - 1))


def dispersion_p_value(
    failures_per_group: tuple[int, ...] | list[int], group_size: int
) -> float | None:
    """Chi-square p-value for the dispersion statistic, for the audit table."""

    phi = quasi_binomial_dispersion(failures_per_group, group_size)
    if phi is None:
        return None
    groups = len(failures_per_group)
    return float(1.0 - chi2.cdf(phi * (groups - 1), groups - 1))


@dataclass(frozen=True)
class AcceptanceBound:
    """Acceptance evidence for one (shape) or (h, shape) condition."""

    shape: str
    h: float | None
    seed_groups: int
    replicas_per_group: int
    failures_per_group: tuple[int, ...]
    total_failures: int
    total_replicas: int
    rate: float
    clopper_pearson_upper: float
    seed_group_t_upper: float | None
    upper_bound: float
    binding_term: str
    dispersion_phi: float | None
    dispersion_p_value: float | None
    threshold: float
    passed: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def condition_bound(
    *,
    shape: str,
    failures_per_group: tuple[int, ...] | list[int],
    replicas_per_group: int,
    h: float | None = None,
    confidence: float = DEFAULT_CONFIDENCE,
    threshold: float = DEFAULT_MAX_TERMINATION_RATE,
) -> AcceptanceBound:
    """Acceptance bound for one condition, pooling only across seed groups."""

    counts = tuple(int(value) for value in failures_per_group)
    if not counts:
        raise ValueError("a condition needs at least one seed group")
    if replicas_per_group <= 0:
        raise ValueError("replicas_per_group must be positive")
    if any(value < 0 or value > replicas_per_group for value in counts):
        raise ValueError("each group's failures must lie in [0, replicas_per_group]")

    total_failures = sum(counts)
    total_replicas = len(counts) * replicas_per_group
    cp = clopper_pearson_upper(
        total_failures, total_replicas, confidence=confidence
    )
    group_upper = seed_group_t_upper(
        [count / replicas_per_group for count in counts], confidence=confidence
    )
    upper = cp if group_upper is None else max(cp, group_upper)
    binding = (
        "clopper_pearson"
        if group_upper is None or cp >= group_upper
        else "seed_group_t"
    )
    return AcceptanceBound(
        shape=str(shape),
        h=None if h is None else float(h),
        seed_groups=len(counts),
        replicas_per_group=int(replicas_per_group),
        failures_per_group=counts,
        total_failures=total_failures,
        total_replicas=total_replicas,
        rate=total_failures / total_replicas,
        clopper_pearson_upper=cp,
        seed_group_t_upper=group_upper,
        upper_bound=upper,
        binding_term=binding,
        dispersion_phi=quasi_binomial_dispersion(counts, replicas_per_group),
        dispersion_p_value=dispersion_p_value(counts, replicas_per_group),
        threshold=float(threshold),
        passed=upper <= float(threshold),
    )


def worst_shape(bounds: list[AcceptanceBound]) -> AcceptanceBound:
    """The condition that controls acceptance: the largest upper bound."""

    if not bounds:
        raise ValueError("no conditions to accept")
    return max(bounds, key=lambda bound: bound.upper_bound)


def summarize_acceptance(bounds: list[AcceptanceBound]) -> dict[str, object]:
    """Worst-shape verdict plus every condition, for the report."""

    worst = worst_shape(bounds)
    return {
        "statistic": (
            "max(pooled_clopper_pearson_upper, seed_group_t_upper), "
            "one-sided, pooled only across seed groups"
        ),
        "confidence": DEFAULT_CONFIDENCE,
        "threshold": worst.threshold,
        "passed": all(bound.passed for bound in bounds),
        "worst_condition": worst.to_dict(),
        "failed_conditions": [
            bound.to_dict() for bound in bounds if not bound.passed
        ],
        "conditions": [bound.to_dict() for bound in bounds],
    }
