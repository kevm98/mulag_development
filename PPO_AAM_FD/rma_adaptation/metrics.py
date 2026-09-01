"""Future evaluation metrics; no simulation is run here."""

from __future__ import annotations

import torch


def delta_adapt(robust: torch.Tensor, privileged: torch.Tensor) -> torch.Tensor:
    """Adaptation headroom: E_robust - E_privileged (lower error is better)."""

    robust = torch.as_tensor(robust, dtype=torch.float32)
    privileged = torch.as_tensor(privileged, dtype=robust.dtype, device=robust.device)
    return robust - privileged


def recovery_fraction(recovery_time_s: torch.Tensor, horizon_s: float, threshold_s: float) -> torch.Tensor:
    """Fraction of scored episodes recovered before a declared threshold."""

    values = torch.as_tensor(recovery_time_s, dtype=torch.float32)
    if horizon_s <= 0 or threshold_s < 0:
        raise ValueError("horizon_s must be positive and threshold_s non-negative")
    return (values <= min(float(horizon_s), float(threshold_s))).float().mean()


def stress_thresholds(*, authority_onset: float = 0.25, delay_onset_steps: int = 2, lag_onset_s: float = 0.12) -> dict[str, float]:
    if not 0.0 < authority_onset <= 1.0 or delay_onset_steps < 0 or lag_onset_s < 0.0:
        raise ValueError("invalid stress threshold")
    return {
        "authority_onset_alpha": float(authority_onset),
        "delay_onset_steps": float(delay_onset_steps),
        "lag_onset_tau_s": float(lag_onset_s),
    }


def opportunity_grid(
    *,
    authority=(1.0, 0.5, 0.25, 0.01, 0.0),
    tau_s=(0.0, 0.04, 0.08, 0.12),
    delay_steps=(0, 1, 2, 3),
    mass_scale=(1.0, 0.85, 1.15),
) -> list[dict[str, float]]:
    """Return a deterministic future sweep grid without launching a plant."""

    return [
        {
            "actuator_authority_alpha": float(alpha),
            "hydraulic_response_time_tau": float(tau),
            "actuator_delay_steps": float(delay),
            "mass_scale": float(mass),
        }
        for alpha in authority
        for tau in tau_s
        for delay in delay_steps
        for mass in mass_scale
    ]


def recovery_plot_data(recovery_time_s: torch.Tensor, thresholds_s: torch.Tensor) -> dict[str, torch.Tensor]:
    """Prepare empirical recovery CDF data for a future plot."""

    times = torch.as_tensor(recovery_time_s, dtype=torch.float32).reshape(-1)
    thresholds = torch.as_tensor(thresholds_s, dtype=torch.float32).reshape(-1)
    return {"threshold_s": thresholds, "fraction": (times[None, :] <= thresholds[:, None]).float().mean(dim=1)}


def plot_recovery_curve(recovery_time_s: torch.Tensor, *, ax=None):
    """Plot helper; matplotlib is imported lazily and never required by CPU tests."""

    if ax is None:
        import matplotlib.pyplot as plt
        _, ax = plt.subplots()
    thresholds = torch.linspace(0.0, float(torch.as_tensor(recovery_time_s).max()), 100)
    data = recovery_plot_data(recovery_time_s, thresholds)
    ax.plot(data["threshold_s"].numpy(), data["fraction"].numpy())
    ax.set(xlabel="recovery time threshold (s)", ylabel="recovery fraction", ylim=(0.0, 1.0))
    return ax
