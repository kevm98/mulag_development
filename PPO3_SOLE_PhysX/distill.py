"""Distillation phase: learn the DLS, then delete it.

The point of PPO3 is that the policy is the controller.  Learning that from
scratch against 0.040 m gates is a hard exploration problem, and the repository
already contains an expert that solves it -- the cylinder DLS controller under
the EE cascade.  So the teacher is used the only way that does not compromise
the claim: as a source of *labels during training*, never as a layer in the
deployed controller.

The teacher is confined by three mechanisms:

1.  ``sole_teacher_mode`` defaults to ``"off"``, and in that mode the teacher
    code in the environment is unreachable.
2.  Evaluation and gate runs raise if the mode is not ``"off"``
    (``assert_teacher_disabled``), so no reported number can have been produced
    with the DLS in the loop.
3.  The runtime control-path audit corrupts the cascade's output and requires
    bit-identical plant commands, which only holds with the teacher gone.

Labels are chain-inverted (see ``invert_sole_chain``): the actor is trained to
emit the raw action whose passage through the filter/slew chain reproduces the
teacher's command, not to emit the teacher's command directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from PPO3_SOLE_PhysX.actions import invert_sole_chain

TEACHER_MODES = ("off", "label", "execute")


@dataclass
class DistillConfig:
    """Schedule for the distillation phase."""

    iterations: int = 200
    steps_per_iteration: int = 96
    learning_rate: float = 1.0e-3
    minibatch_size: int = 4096
    epochs_per_iteration: int = 4

    execute_teacher_start: float = 1.0
    """Probability of executing the teacher's action at iteration 0.

    Pure behaviour cloning on expert-visited states is the classic covariate
    shift failure: the student never sees the states its own mistakes produce.
    Annealing this to zero turns the phase into DAgger -- the student
    increasingly drives, and the expert keeps labelling.
    """

    execute_teacher_end: float = 0.0
    execute_teacher_anneal_fraction: float = 0.6
    """Fraction of iterations over which the teacher hands over control."""

    def execute_probability(self, iteration: int) -> float:
        span = max(
            float(self.execute_teacher_anneal_fraction) * max(self.iterations, 1), 1.0
        )
        t = min(max(float(iteration) / span, 0.0), 1.0)
        return float(
            self.execute_teacher_start
            + t * (self.execute_teacher_end - self.execute_teacher_start)
        )


@dataclass
class DistillBatch:
    """One iteration of collected (observation, chain-inverted label) pairs."""

    observations: list[torch.Tensor] = field(default_factory=list)
    labels: list[torch.Tensor] = field(default_factory=list)
    unreachable: list[torch.Tensor] = field(default_factory=list)

    def add(
        self, obs: torch.Tensor, label: torch.Tensor, unreachable: torch.Tensor
    ) -> None:
        self.observations.append(obs.detach().clone())
        self.labels.append(label.detach().clone())
        self.unreachable.append(unreachable.detach().clone())

    def stacked(self) -> tuple[torch.Tensor, torch.Tensor, float]:
        obs = torch.cat(self.observations, dim=0)
        labels = torch.cat(self.labels, dim=0)
        unreachable = torch.cat(self.unreachable, dim=0)
        return obs, labels, float(unreachable.float().mean())

    def __len__(self) -> int:
        return len(self.observations)


def assert_teacher_disabled(cfg, *, context: str) -> None:
    """Refuse to report a number produced with the teacher in the loop."""
    mode = str(getattr(cfg, "sole_teacher_mode", "off")).lower()
    if mode != "off":
        raise RuntimeError(
            f"{context} requires sole_teacher_mode='off', got {mode!r}. A result "
            "produced with the DLS teacher active is a distillation diagnostic, "
            "not a sole-controller result."
        )


def teacher_labels(
    *,
    teacher_action: torch.Tensor,
    previous_applied_action: torch.Tensor,
    filter_beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chain-inverted regression targets for the actor."""
    return invert_sole_chain(
        target_applied=teacher_action,
        previous_applied_action=previous_applied_action,
        filter_beta=filter_beta,
    )


def behaviour_cloning_loss(
    actor_mean: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Mean squared error between the actor's mean and the teacher's label.

    Deliberately on the *mean*, not on a sampled action: the distribution's
    scale is PPO's exploration budget and is left for the PPO phase to tune.
    """
    return torch.nn.functional.mse_loss(actor_mean, labels)


@torch.no_grad()
def teacher_reachability_report(
    teacher_stream: torch.Tensor,
    *,
    max_action_rate: float,
    reset_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    """Is the teacher's own command stream inside the student's slew envelope?

    If it is not, behaviour cloning is training toward a trajectory the student
    is structurally forbidden from reproducing, and the distilled policy will
    have a floor on its imitation error that no amount of training removes.
    Run this before committing GPU hours, not after the loss plateaus.

    ``teacher_stream`` is [T, N, 4] of consecutive teacher commands.

    ``reset_mask`` is [T, N] and marks steps after which the environment reset.
    Those transitions MUST be excluded: ``CylinderDLSController.reset`` zeroes
    its ``previous_action``, so the first command of a new episode differs from
    the last command of the old one by an amount that has nothing to do with the
    teacher's dynamics. Counting them reports a violation the student is never
    actually asked to reproduce, since it resets too.
    """
    if teacher_stream.ndim != 3 or teacher_stream.shape[-1] != 4:
        raise ValueError(
            f"teacher_stream must be [T, N, 4], got {tuple(teacher_stream.shape)}"
        )
    deltas = (teacher_stream[1:] - teacher_stream[:-1]).abs()

    crossings = 0
    if reset_mask is not None:
        keep = ~reset_mask[:-1].to(dtype=torch.bool, device=deltas.device)
        crossings = int((~keep).sum())
        deltas = deltas[keep]

    if deltas.numel() == 0:
        return {
            "steps": float(teacher_stream.shape[0]),
            "max_abs_teacher_action": float(teacher_stream.abs().max()),
            "max_teacher_step_delta": 0.0,
            "p99_teacher_step_delta": 0.0,
            "slew_limit": float(max_action_rate),
            "fraction_exceeding_slew_limit": 0.0,
            "reset_transitions_excluded": crossings,
            "teacher_is_reachable": True,
        }

    exceed = deltas > float(max_action_rate)
    return {
        "steps": float(teacher_stream.shape[0]),
        "max_abs_teacher_action": float(teacher_stream.abs().max()),
        "max_teacher_step_delta": float(deltas.max()),
        "p99_teacher_step_delta": float(torch.quantile(deltas.flatten(), 0.99)),
        "slew_limit": float(max_action_rate),
        "fraction_exceeding_slew_limit": float(exceed.float().mean()),
        "reset_transitions_excluded": crossings,
        "teacher_is_reachable": not bool(exceed.any()),
    }
