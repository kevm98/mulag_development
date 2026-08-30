"""RMA Phase-1 environment-factor encoder, trained jointly with the policy.

RMA (arXiv 2107.04034) trains the base policy pi(x, a_-1, z) AND the encoder
mu together, end-to-end, under the RL objective, with z = mu(e). That joint
training is the whole point: it makes z control-relevant by construction.

This repo previously did the opposite -- a FROZEN encoder fitted to a
dynamics-prediction objective. Measured paired, that latent carried ~0.4% plant
identity and scored 7.1% WORSE than mean context (CI [-14.5%, -0.7%]): feeding
a policy near-noise is worse than feeding it a constant.

Feeding the RAW factor vector instead is UP-OSI (arXiv 1702.02453), not RMA. It
is a valid upper bound -- if a policy cannot exploit e at full fidelity, no
encoder of any width can -- but it has no bottleneck, so it cannot answer
whether compressing the context helps.

Implementation note. A literal 6->2 bottleneck would narrow the actor's first
layer from 72 to 68 and break the warm start, and the warm start is not
optional here: training this task from scratch produced a policy with 675 mm
median lateral error against the baseline's 34 mm. So mu is applied as
6 -> latent -> 6 on the context columns only. The policy still sees e ONLY
through a `latent`-dimensional code trained by PPO, which is RMA's essential
property, while the trunk keeps its 72-wide input and the warm start loads.

The decoder is zero-initialised, so at construction the context contributes
exactly nothing and the wrapped actor reproduces the warm-started policy
bit-for-bit -- the same property the migrated checkpoint has, and the reason
any later gain is attributable to learning e rather than to relearning the task.
"""

from __future__ import annotations

import torch
from torch import nn


class ContextBottleneck(nn.Module):
    """Wrap an actor/critic so its context columns pass through mu."""

    def __init__(self, inner: nn.Module, *, base_dim: int, context_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.inner = inner
        self.base_dim = int(base_dim)
        self.context_dim = int(context_dim)
        self.latent_dim = int(latent_dim)
        self.encoder = nn.Sequential(
            nn.Linear(self.context_dim, 32),
            nn.ELU(),
            nn.Linear(32, self.latent_dim),
        )
        self.decoder = nn.Linear(self.latent_dim, self.context_dim)
        # Zero-init: z contributes nothing at construction, so the wrapped
        # module reproduces the warm-started policy exactly. Gradients still
        # flow (dL/dW_dec is proportional to the encoder output), so mu can
        # learn from the first update.
        nn.init.zeros_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def latent(self, observations: torch.Tensor) -> torch.Tensor:
        return self.encoder(observations[..., self.base_dim:])

    def forward(self, observations: torch.Tensor, *args, **kwargs):
        if observations.shape[-1] != self.base_dim + self.context_dim:
            raise ValueError(
                f"ContextBottleneck expects width {self.base_dim + self.context_dim}, "
                f"got {observations.shape[-1]}"
            )
        base = observations[..., : self.base_dim]
        context = observations[..., self.base_dim:]
        recoded = self.decoder(self.encoder(context))
        return self.inner(torch.cat((base, recoded), dim=-1), *args, **kwargs)

    def __getattr__(self, name: str):
        # nn.Module.__getattr__ handles registered submodules/params; anything
        # else is delegated to the wrapped model so the runner's calls into
        # actor internals (normalizer updates, distribution helpers) still work.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.inner, name)


def wrap_actor_critic(runner, *, base_dim: int, context_dim: int, latent_dim: int) -> dict:
    """Install mu on actor and critic and rebuild the optimizer.

    The optimizer must be rebuilt: rsl_rl chains actor and critic parameters at
    construction, so the encoder's weights would otherwise never be updated and
    the run would silently be plain UP-OSI wearing an encoder.
    """
    from itertools import chain

    device = next(runner.alg.actor.parameters()).device
    with torch.no_grad():
        probe = torch.zeros((2, base_dim + context_dim), device=device)
        before = runner.alg.actor(probe).clone()

    runner.alg.actor = ContextBottleneck(
        runner.alg.actor, base_dim=base_dim, context_dim=context_dim, latent_dim=latent_dim
    ).to(device)
    runner.alg.critic = ContextBottleneck(
        runner.alg.critic, base_dim=base_dim, context_dim=context_dim, latent_dim=latent_dim
    ).to(device)

    learning_rate = float(getattr(runner.alg, "learning_rate", 1.0e-3))
    runner.alg.optimizer = torch.optim.Adam(
        chain(runner.alg.actor.parameters(), runner.alg.critic.parameters()),
        lr=learning_rate,
    )

    with torch.no_grad():
        after = runner.alg.actor(probe)
    drift = float((after - before).abs().max())
    encoder_params = sum(
        p.numel() for p in runner.alg.actor.encoder.parameters()
    ) + sum(p.numel() for p in runner.alg.actor.decoder.parameters())
    return {
        "rma_encoder": True,
        "latent_dim": int(latent_dim),
        "context_dim": int(context_dim),
        "base_dim": int(base_dim),
        "encoder_parameters_per_head": int(encoder_params),
        # Must be 0: the zero-initialised decoder means the wrapped actor has to
        # reproduce the warm start exactly. A non-zero value means the surgery
        # changed the policy and any later comparison is confounded.
        "init_output_drift": drift,
    }
