"""Curriculum helpers for ROM-B randomization ranges."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ROMBCurriculumThresholds:
    """Safety-aware thresholds for widening ROM-B randomization.

    Reward alone is intentionally absent.  The curriculum advances only when
    tracking and plant-health metrics are inside their configured limits.
    """

    max_tracking_error_m: float = 0.035
    max_safety_termination_rate: float = 0.02
    max_invalid_pressure_rate: float = 0.0
    max_action_saturation_rate: float = 0.20
    max_unstable_state_rate: float = 0.0
    consecutive_passes: int = 5


class ROMBCurriculumController:
    """Manual stage holder with optional metric-gated automatic progression."""

    def __init__(
        self,
        *,
        stage: int = 4,
        enabled: bool = False,
        thresholds: ROMBCurriculumThresholds | None = None,
    ) -> None:
        self.stage = int(stage)
        if self.stage < 1 or self.stage > 4:
            raise ValueError(f"ROM-B curriculum stage must be in [1, 4], got {stage}.")
        self.enabled = bool(enabled)
        self.thresholds = thresholds or ROMBCurriculumThresholds()
        self._pass_count = 0

    def update(self, metrics: dict[str, float]) -> int:
        """Return the active stage after checking a metric batch."""

        if not self.enabled or self.stage >= 4:
            return self.stage
        th = self.thresholds
        passed = (
            float(metrics.get("tracking_error_m", float("inf"))) <= th.max_tracking_error_m
            and float(metrics.get("safety_termination_rate", float("inf"))) <= th.max_safety_termination_rate
            and float(metrics.get("invalid_pressure_rate", float("inf"))) <= th.max_invalid_pressure_rate
            and float(metrics.get("action_saturation_rate", float("inf"))) <= th.max_action_saturation_rate
            and float(metrics.get("unstable_state_rate", float("inf"))) <= th.max_unstable_state_rate
        )
        if passed:
            self._pass_count += 1
            if self._pass_count >= th.consecutive_passes:
                self.stage += 1
                self._pass_count = 0
        else:
            self._pass_count = 0
        return self.stage
