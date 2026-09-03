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

    HISTORY_COLUMN_SETS = {
        # Every base observation column. 66 + 4 action = 70 features per step,
        # 1400 numbers over a 20-step window, almost all terrain and task state.
        "full": None,
        # Legacy narrow set: cylinder extensions and velocities only.
        "ext_vel": ((12, 20),),
        # Plant-relevant channels, the starved encoder. Tracking error 9:11,
        # cylinder extensions 12:16 and velocities 16:20, current valve command
        # 32:36, measured-minus-commanded cylinder state 54:58, actuator
        # prediction residual 58:62 and normalised cylinder effort 62:66.
        # 26 + 4 action = 30 features per step, 600 numbers over the window.
        # Terrain profile, headroom, target lead, alignment, truck speed, path
        # preview and cutter pose are all excluded: they let the head clone the
        # task policy rather than identify the plant.
        "plant": ((9, 11), (12, 20), (32, 36), (54, 58), (58, 62), (62, 66)),
    }

    @classmethod
    def expand_columns(cls, name, base_dim):
        """Column-index tensor for a named set, or all base columns for 'full'."""
        spans = cls.HISTORY_COLUMN_SETS[name]
        if spans is None:
            return torch.arange(int(base_dim), dtype=torch.long)
        idx = [c for lo, hi in spans for c in range(lo, hi)]
        if max(idx) >= int(base_dim):
            raise ValueError(
                f"history column set '{name}' reaches column {max(idx)} but the "
                f"base observation has only {base_dim} columns")
        return torch.tensor(idx, dtype=torch.long)

    def __init__(self, base_mlp, *, base_dim, base_width, action_dim,
                 history=20, latent_dim=8, hidden=64, alpha=0.25,
                 history_feature_dim=None, history_cols=None):
        super().__init__()
        self.base_mlp = base_mlp
        for parameter in self.base_mlp.parameters():
            parameter.requires_grad_(False)
        self.base_dim = int(base_dim)
        self.base_width = int(base_width)
        self.action_dim = int(action_dim)
        self.history = int(history)
        self.alpha = float(alpha)

        # Feeding the whole observation gives 70 features per step, 1400 numbers
        # over a 20-step window, almost all terrain and task state. A 53k head
        # has ample room to fit that noise, and the measured history benefit
        # swung between seeds in both directions. What identifies the plant is
        # far smaller: what was COMMANDED against how the cylinders RESPONDED,
        # a signature invariant to where the machine sits on the terrain, which
        # is the property a plant latent needs. HISTORY_COLUMN_SETS names the
        # candidate sets and 'plant' is the starved one.
        #
        # The column spec rides in the state_dict as a buffer, so an evaluation
        # cannot silently feed the head a different set from the one it was
        # trained on -- the failure mode that a plain CLI flag invites.
        if history_cols is None:
            history_cols = torch.arange(self.base_dim, dtype=torch.long)
        history_cols = torch.as_tensor(history_cols, dtype=torch.long)
        self.register_buffer("history_cols", history_cols)
        feature_dim = int(history_feature_dim) if history_feature_dim else (
            history_cols.numel() + self.action_dim)
        if feature_dim != history_cols.numel() + self.action_dim:
            raise ValueError(
                f"history_feature_dim={feature_dim} disagrees with "
                f"{history_cols.numel()} columns + {self.action_dim} action")
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
