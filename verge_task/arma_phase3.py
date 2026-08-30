"""A-RMA Phase 3: fine-tune the base policy on the ESTIMATED latent.

RMA Phase 2 leaves a mismatch. The base policy was trained in Phase 1 on the
true privileged context e, but at deployment it receives z_hat = phi(history),
which carries estimation error. A-RMA (Kumar et al., "Adapting Rapid Motor
Adaptation for Bipedal Robots", 2022) closes that gap by running RL again with
phi frozen and z_hat in the loop, so the policy learns to be robust to the
errors the estimator actually makes rather than assuming a perfect latent.

The substitution happens at the ENVIRONMENT observation, not inside the actor.
That matters: PPO calls the actor many times per rollout during its update
epochs, so a history buffer living in the actor would be advanced by the update
loop and corrupted. Substituting in the observation means the rollout storage
already holds z_hat, and PPO optimises exactly the quantity that will be present
at deployment.

The causal convention is copied verbatim from ``train_phase2_dagger.py``: z_hat
at step t is computed from history that does NOT include the current
observation, i.e. from x_{t-k:t-1}, a_{t-k:t-1}, matching RMA. Any deviation
here is a silent train/test mismatch between the two phases.
"""

from __future__ import annotations

import torch


class Phase3LatentAdapter:
    """Wrap an rsl_rl vec-env so the context columns are replaced by phi(history)."""

    def __init__(self, inner, phi, *, base_dim: int, context_dim: int,
                 history: int, target_mean: torch.Tensor, target_std: torch.Tensor,
                 num_envs: int, action_dim: int, device) -> None:
        self.inner = inner
        self.phi = phi.eval()
        for parameter in self.phi.parameters():
            parameter.requires_grad_(False)
        self.base_dim = int(base_dim)
        self.context_dim = int(context_dim)
        self.history = int(history)
        self.device = device
        self.target_mean = target_mean.to(device)
        self.target_std = target_std.to(device)
        self.action_dim = int(action_dim)
        self._hist = torch.zeros(
            (int(num_envs), self.history, self.base_dim + self.action_dim), device=device
        )
        self._prev_action = torch.zeros((int(num_envs), self.action_dim), device=device)

    def __getattr__(self, name):
        # Everything the runner asks for that is not overridden here belongs to
        # the wrapped env (num_envs, device, episode length, cfg, ...).
        return getattr(self.inner, name)

    @torch.no_grad()
    def _estimate(self, obs: torch.Tensor) -> torch.Tensor:
        """z_hat from history that does NOT yet include the current observation."""
        return self.phi(self._hist) * self.target_std + self.target_mean

    @torch.no_grad()
    def _advance(self, obs_td) -> None:
        """Append the current step to the history, ONCE per environment step.

        Kept separate from ``_estimate`` because several observation streams are
        rewritten per step; advancing inside the rewrite would push the history
        forward once per stream and desynchronise it from the simulation.
        """
        for key in obs_td.keys():
            stream = obs_td[key]
            if stream.ndim == 2 and stream.shape[-1] == self.base_dim + self.context_dim:
                base = stream[:, : self.base_dim]
                break
        else:
            return
        step_feature = torch.cat((base, self._prev_action), dim=-1)
        self._hist = torch.roll(self._hist, shifts=-1, dims=1)
        self._hist[:, -1, :] = step_feature

    def _substitute_td(self, obs_td):
        """Replace the context columns in every observation stream of a TensorDict.

        ``RslRlVecEnvWrapper`` hands the runner a TensorDict, not a plain tensor.
        The critic stream must be substituted as well when it carries the same
        width: leaving the true context in the critic while the actor sees z_hat
        would give the value function privileged information the policy does not
        have, which silently changes what PPO is optimising.
        """
        z_hat = None
        for key in list(obs_td.keys()):
            stream = obs_td[key]
            if stream.ndim != 2 or stream.shape[-1] != self.base_dim + self.context_dim:
                continue
            if z_hat is None:
                z_hat = self._estimate(stream)
            obs_td[key] = torch.cat((stream[:, : self.base_dim], z_hat), dim=-1)
        if z_hat is None:
            raise RuntimeError(
                "Phase3LatentAdapter found no observation stream of width "
                f"{self.base_dim + self.context_dim}; keys={list(obs_td.keys())}"
            )
        self._advance(obs_td)
        return obs_td

    def get_observations(self):
        return self._substitute_td(self.inner.get_observations())

    def step(self, actions):
        obs_td, rewards, dones, extras = self.inner.step(actions)
        self._prev_action = actions.detach().clone()
        obs_td = self._substitute_td(obs_td)
        if dones is not None and bool(dones.any()):
            # A history spanning a reset describes two episodes and, with
            # per-env plant draws, two different plants.
            mask = dones.bool()
            self._hist[mask] = 0.0
            self._prev_action[mask] = 0.0
        return obs_td, rewards, dones, extras

    def reset(self, *args, **kwargs):
        out = self.inner.reset(*args, **kwargs)
        self._hist.zero_()
        self._prev_action.zero_()
        return out


def load_phi(path, device):
    """Load a Phase 2 checkpoint and rebuild phi."""
    from verge_task.adaptation_module import AdaptationModule

    payload = torch.load(path, map_location=device)
    phi = AdaptationModule(
        int(payload["feature_dim"]), int(payload["history"]), int(payload["context_dim"])
    ).to(device)
    phi.load_state_dict(payload["state_dict"])
    return phi, payload
