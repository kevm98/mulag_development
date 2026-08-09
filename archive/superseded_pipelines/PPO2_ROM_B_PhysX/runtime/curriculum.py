"""Round-level progression gates for the ROM-B uncertainty curriculum."""

from __future__ import annotations

from dataclasses import asdict, dataclass


# Nominal ROM-A, then mild, medium, and full ROM-B randomization.
STAGE_SCALES = (0.00, 0.40, 0.70, 1.00)


@dataclass(frozen=True)
class ROMBCurriculumThresholds:
    max_shape_mean_error_m: float = 0.040
    max_shape_p95_error_m: float = 0.080
    max_shape_p95_acceleration_mps2: float = 40.0
    min_head_alignment: float = 0.980
    max_clearance_error_m: float = 0.200
    max_post_constraint_truck_drift_m: float = 0.001
    max_termination_rate: float = 0.05
    consecutive_passing_rounds: int = 3
    minimum_rounds_per_stage: int = 3
    minimum_full_scale_rounds: int = 5


@dataclass(frozen=True)
class ROMBCurriculumDecision:
    stage: int
    scale: float
    passed: bool
    advanced: bool
    complete: bool
    failed_checks: tuple[str, ...]
    stage_rounds: int
    consecutive_passes: int


class ROMBCurriculumController:
    """Advance only after complete five-shape rounds satisfy every gate."""

    def __init__(
        self,
        *,
        thresholds: ROMBCurriculumThresholds | None = None,
        stage: int = 1,
    ) -> None:
        if stage < 1 or stage > len(STAGE_SCALES):
            raise ValueError(f"curriculum stage must be in [1, {len(STAGE_SCALES)}]")
        self.thresholds = thresholds or ROMBCurriculumThresholds()
        self.stage = int(stage)
        self.stage_rounds = 0
        self.consecutive_passes = 0

    @property
    def scale(self) -> float:
        return float(STAGE_SCALES[self.stage - 1])

    def update(self, metrics: dict[str, float]) -> ROMBCurriculumDecision:
        th = self.thresholds
        checks = {
            "shape_mean_error": metrics["max_shape_mean_error_m"]
            <= th.max_shape_mean_error_m,
            "shape_p95_error": metrics["max_shape_p95_error_m"]
            <= th.max_shape_p95_error_m,
            "shape_p95_acceleration": metrics["max_shape_p95_acceleration_mps2"]
            <= th.max_shape_p95_acceleration_mps2,
            "head_alignment": metrics["min_head_alignment"]
            >= th.min_head_alignment,
            "clearance": metrics["max_clearance_error_m"]
            <= th.max_clearance_error_m,
            "truck_drift": metrics["max_post_constraint_truck_drift_m"]
            <= th.max_post_constraint_truck_drift_m,
            "terminations": metrics["max_termination_rate"]
            <= th.max_termination_rate,
        }
        failed = tuple(name for name, passed in checks.items() if not passed)
        passed = not failed
        self.stage_rounds += 1
        self.consecutive_passes = self.consecutive_passes + 1 if passed else 0
        advanced = False
        complete = False
        enough_passes = self.consecutive_passes >= th.consecutive_passing_rounds
        if self.stage < len(STAGE_SCALES):
            if self.stage_rounds >= th.minimum_rounds_per_stage and enough_passes:
                self.stage += 1
                self.stage_rounds = 0
                self.consecutive_passes = 0
                advanced = True
        elif self.stage_rounds >= th.minimum_full_scale_rounds and enough_passes:
            complete = True
        return ROMBCurriculumDecision(
            stage=self.stage,
            scale=self.scale,
            passed=passed,
            advanced=advanced,
            complete=complete,
            failed_checks=failed,
            stage_rounds=self.stage_rounds,
            consecutive_passes=self.consecutive_passes,
        )

    def contract(self) -> dict[str, object]:
        return {
            "stage_scales": list(STAGE_SCALES),
            "thresholds": asdict(self.thresholds),
        }

    def restore_progress(
        self, *, stage_rounds: int, consecutive_passes: int
    ) -> None:
        """Restore audited counters when continuing an interrupted run."""

        stage_rounds = int(stage_rounds)
        consecutive_passes = int(consecutive_passes)
        if stage_rounds < 0 or consecutive_passes < 0:
            raise ValueError("curriculum counters must be non-negative")
        if consecutive_passes > stage_rounds:
            raise ValueError("consecutive passes cannot exceed stage rounds")
        self.stage_rounds = stage_rounds
        self.consecutive_passes = consecutive_passes
