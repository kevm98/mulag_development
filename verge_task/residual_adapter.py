"""Frozen base policy + history-conditioned residual head.

    u = clip(u0 + alpha * g(o, phi(H)), -1, 1)

with ``u0 = pi_frozen(o)``. Only ``phi`` (history encoder) and ``g`` (residual
head) are trained; the base policy is frozen and never updated.

Why this shape, given what was measured on this stack on 2026-08-29:

* **Fine-tuning the base is a dead end here.** Without a KL anchor the fine-tune
  walks away from the distilled behaviour (lateral 0.05 -> 1.25 m while total
  reward ROSE -- PPO correctly optimising an objective that prefers planting the
  cutter over holding the line). With the anchor the policy is stable but cannot
  acquire a new input dependence: the anchor's reference is context-blind by
  construction, so any context-dependent behaviour raises KL-to-reference and is
  penalised. A 1000x gradient multiplier on the context columns moved action
  sensitivity from 0.001% to 0.004% of the action scale. Both horns are
  structural, not tuning.

* **A residual head has no anchor.** It is new capacity whose only job is to
  depend on the latent, so nothing opposes that dependence.

* **phi and g are trained TOGETHER under RL**, not fitted to a separate latent
  target. That makes the latent control-relevant by construction (the concurrent
  scheme of Ji et al., RA-L 2022) instead of hoping a latent fitted to some
  auxiliary objective happens to be useful. The repo's previous frozen encoder,
  fitted offline to a dynamics objective, carried ~0.4% plant identity and scored
  7.1% WORSE than mean context.

* **The head is zero-initialised**, so at construction the residual is exactly
  zero and the wrapped policy reproduces the frozen base bit-for-bit. Any later
  gain is therefore attributable to the residual rather than to relearning the
  task.

The history is (base observation, previous action) pairs, strictly causal, and
z_hat at step t is computed from history that does NOT include step t -- the same
convention Phase 2 uses, so the two cannot silently disagree.
"""

from __future__ import annotations

import torch
from torch import nn


class ResidualMLP(torch.nn.Module):
    """Replaces an rsl_rl model's ``mlp``: base mean + alpha * history residual.

    Spliced at ``mlp`` rather than around the actor because rsl_rl builds the
    action distribution FROM the mlp output and calls a wide interface on the
    actor (``get_output_log_prob``, ``output_entropy``, ``get_kl_divergence``,
    ``update_normalization``...). Wrapping the actor's output would break all of
    it; replacing the mlp leaves every one of those paths intact and changes only
    the mean.

    Input is the ALREADY-NORMALISED wide latent:
        [ normalised base columns (base_width) | flattened history ]
    The outer normaliser is seeded with the base checkpoint's statistics on the
    base columns and identity (mean 0, std 1) on the history columns, so the
    frozen base sees exactly the inputs it was trained on.
    """

    def __init__(self, base_mlp, *, base_dim, base_width, action_dim,
                 history=20, latent_dim=8, hidden=64, alpha=0.25,
                 history_feature_dim=None):
        super().__init__()
        self.base_mlp = base_mlp
        for parameter in self.base_mlp.parameters():
            parameter.requires_grad_(False)
        self.base_dim = int(base_dim)
        self.base_width = int(base_width)
        self.action_dim = int(action_dim)
        self.history = int(history)
        self.alpha = float(alpha)

        # PLANT-ONLY history is the default worth using. Feeding the whole
        # observation gives 66+4 features per step -- 1400 numbers over a 20-step
        # window, almost all terrain and task state. A 53k-parameter head has
        # ample room to fit that noise, which is why the benefit swung +1529% to
        # -80% across seeds in BOTH directions.
        #
        # What identifies the plant is far smaller: what was COMMANDED against
        # how the cylinders RESPONDED (extensions 12:16, velocities 16:20, plus
        # the action). That signature is invariant to where the machine is on the
        # terrain, which is exactly the property a plant latent needs.
        feature_dim = int(history_feature_dim) if history_feature_dim else self.base_dim + self.action_dim
        self.history_feature_dim = feature_dim
        self.encoder = torch.nn.Sequential(torch.nn.Linear(feature_dim, hidden), torch.nn.ELU())
        self.temporal = torch.nn.Sequential(
            torch.nn.Conv1d(hidden, hidden, kernel_size=5), torch.nn.ELU(),
            torch.nn.Conv1d(hidden, hidden, kernel_size=3), torch.nn.ELU(),
        )
        with torch.no_grad():
            flat = self.temporal(torch.zeros(1, hidden, self.history)).flatten(1).shape[1]
        self.to_latent = torch.nn.Linear(flat, int(latent_dim))
        self.head = torch.nn.Sequential(
            torch.nn.Linear(self.base_dim + int(latent_dim), hidden), torch.nn.ELU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ELU(),
            torch.nn.Linear(hidden, self.action_dim),
        )
        # Zero-init: the residual is exactly zero at construction, so the wrapped
        # policy reproduces the frozen base bit-for-bit and any later gain is
        # attributable to the residual rather than to relearning the task.
        torch.nn.init.zeros_(self.head[-1].weight)
        torch.nn.init.zeros_(self.head[-1].bias)

    def latent(self, flat_history):
        hist = flat_history.reshape(*flat_history.shape[:-1], self.history,
                                    self.history_feature_dim)
        z = self.encoder(hist)
        z = self.temporal(z.transpose(-1, -2))
        return self.to_latent(z.flatten(start_dim=-2))

    def forward(self, latent):
        base_part = latent[..., : self.base_width]
        flat_history = latent[..., self.base_width:]
        expected = self.history * self.history_feature_dim
        if flat_history.shape[-1] != expected:
            raise ValueError(
                f"expected {expected} history columns after {self.base_width} base "
                f"columns, got {flat_history.shape[-1]}"
            )
        base_out = self.base_mlp(base_part)
        z = self.latent(flat_history)
        delta = self.head(torch.cat((base_part[..., : self.base_dim], z), dim=-1))
        # BOUNDED residual. An unbounded head diverged at ~iteration 314 and
        # produced a NaN action, which was written into the history buffer and
        # surfaced as "observation group 'policy' contains NaN". tanh caps the
        # correction at +/- alpha no matter what the head does, so a bad update
        # degrades the policy instead of destroying the run.
        #
        # tanh(0) = 0, so the zero-initialised head still makes this an EXACT
        # identity at construction.
        return base_out + self.alpha * torch.tanh(delta)

    def trainable_parameters(self):
        from itertools import chain
        return chain(self.encoder.parameters(), self.temporal.parameters(),
                     self.to_latent.parameters(), self.head.parameters())


class HistoryResidualActor(nn.Module):
    """Wrap a frozen actor with a history-conditioned residual."""

    def __init__(self, base_actor: nn.Module, *, base_dim: int, action_dim: int,
                 history: int = 20, latent_dim: int = 8, hidden: int = 64,
                 alpha: float = 0.25, base_width: int | None = None) -> None:
        super().__init__()
        self.base = base_actor
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.base_dim = int(base_dim)
        self.action_dim = int(action_dim)
        self.history = int(history)
        self.alpha = float(alpha)
        # Width the FROZEN base consumes (may exceed base_dim if it has a
        # context slot of its own, which it ignores).
        self.base_width = int(base_width if base_width is not None else base_dim)

        feature_dim = self.base_dim + self.action_dim
        self.encoder = nn.Sequential(nn.Linear(feature_dim, hidden), nn.ELU())
        self.temporal = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=5), nn.ELU(),
            nn.Conv1d(hidden, hidden, kernel_size=3), nn.ELU(),
        )
        with torch.no_grad():
            flat = self.temporal(torch.zeros(1, hidden, self.history)).flatten(1).shape[1]
        self.to_latent = nn.Linear(flat, int(latent_dim))
        self.head = nn.Sequential(
            nn.Linear(self.base_dim + int(latent_dim), hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, self.action_dim),
        )
        # Zero-init the last layer so the residual starts at exactly zero.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    # -- forward ----------------------------------------------------------
    def forward(self, observations, *args, **kwargs):
        """observations = [ base_obs (base_width) | flattened history ].

        The history is carried IN THE OBSERVATION, not in a buffer on this
        module. PPO calls the actor many times per update epoch, so a buffer
        living here would be advanced by the update loop and desynchronised from
        the simulation; storing it in the observation means the rollout buffer
        holds the exact history each action was taken under.
        """
        obs = observations
        if not torch.is_tensor(obs):
            obs = obs[next(iter(obs.keys()))]
        width = self.base_width
        base_obs_full = obs[..., :width]
        flat_history = obs[..., width:]
        expected = self.history * (self.base_dim + self.action_dim)
        if flat_history.shape[-1] != expected:
            raise ValueError(
                f"expected {expected} history columns after {width} base columns, "
                f"got {flat_history.shape[-1]}"
            )
        hist = flat_history.reshape(*flat_history.shape[:-1], self.history,
                                    self.base_dim + self.action_dim)
        base_out = self.base(base_obs_full, *args, **kwargs)
        z = self.encoder(hist)
        z = self.temporal(z.transpose(-1, -2))
        z = self.to_latent(z.flatten(start_dim=-2))
        delta = self.head(torch.cat((base_obs_full[..., : self.base_dim], z), dim=-1))
        return torch.clamp(base_out + self.alpha * delta, -1.0, 1.0)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base, name)

    def trainable_parameters(self):
        """Only the encoder, latent projection and head -- never the base."""
        from itertools import chain
        return chain(
            self.encoder.parameters(),
            self.temporal.parameters(),
            self.to_latent.parameters(),
            self.head.parameters(),
        )
