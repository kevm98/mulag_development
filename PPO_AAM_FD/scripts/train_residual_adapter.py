#!/usr/bin/env python3
"""Train a history-conditioned residual on top of a FROZEN policy.

    u = clip(u0 + alpha * g(o, phi(H)), -1, 1),   u0 = pi_frozen(o)

Only phi and g are trained. This is the design that survives what was measured
on this stack on 2026-08-29:

* Fine-tuning the base policy is a dead end. Unanchored it walks away from the
  distilled behaviour (lateral 0.05 -> 1.25 m while total reward ROSE). Anchored
  by reference KL it is stable but cannot acquire a new input dependence,
  because the anchor's reference is context-blind by construction -- a 1000x
  gradient multiplier on the context columns moved action sensitivity from
  0.001% to 0.004%. Both horns are structural.
* A residual head is unanchored new capacity, so nothing opposes its dependence
  on the latent, and the frozen base cannot collapse.
* phi and g train TOGETHER under RL, so the latent is control-relevant by
  construction rather than fitted to an auxiliary objective and hoped useful.

**The plant must vary WITHIN an episode for this to be worth running.** E4
measured the per-episode-constant latent at +13.6% error versus the online one,
while knowing which plant you are on was worth 0.43% -- 32x smaller. A static
per-episode context is therefore measurably close to worthless, and a mid-episode
mass step (``deployment_mass_step_time_s`` / ``_scale``) is what makes an online
estimate necessary rather than merely available.

The history rides IN THE OBSERVATION. PPO calls the actor many times per update
epoch, so a history buffer inside the actor would be advanced by the update loop
and desynchronised from the simulation; carrying it in the observation means the
rollout buffer holds the exact history each action was taken under.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

p = argparse.ArgumentParser()
p.add_argument("base_checkpoint", type=Path)
p.add_argument("--num_envs", type=int, default=256)
p.add_argument("--iterations", type=int, default=400)
p.add_argument("--history", type=int, default=20)
p.add_argument("--latent_dim", type=int, default=8)
p.add_argument("--alpha", type=float, default=0.25)
p.add_argument("--lr", type=float, default=3.0e-4)
p.add_argument("--seed", type=int, default=17)
p.add_argument("--stage", type=int, default=4)
p.add_argument("--save_interval", type=int, default=25)
p.add_argument("--log_dir", type=str, default=None)
p.add_argument("--task", type=str, default="RRLAB-Mulag-Verge-ID-AAM-FD-v0")
p.add_argument("--no_history", action="store_true",
               help="Ablation: zero the history, so the residual cannot adapt. "
                    "Isolates 'a residual helps' from 'ONLINE ADAPTATION helps'.")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()

launcher = AppLauncher(a)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402

import mulag_plant.tasks  # noqa: F401,E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from PPO_AAM_FD.env_cfg import PpoIDAAMFDPPORunnerCfg  # noqa: E402
from PPO_ROM.terrain_env_cfg import apply_c4_authority_stage  # noqa: E402
from verge_task.residual_adapter import ResidualMLP  # noqa: E402


class HistoryObservation:
    """Append the flattened (base_obs, prev_action) history to the observation."""

    def __init__(self, inner, *, base_dim, base_width, action_dim, history,
                 num_envs, device, zero_history=False):
        self.inner = inner
        self.base_dim = int(base_dim)
        self.base_width = int(base_width)
        self.action_dim = int(action_dim)
        self.history = int(history)
        self.device = device
        self.zero_history = bool(zero_history)
        self._hist = torch.zeros(
            (num_envs, self.history, self.base_dim + self.action_dim), device=device)
        self._prev = torch.zeros((num_envs, self.action_dim), device=device)

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def _augment(self, td):
        for key in list(td.keys()):
            stream = td[key]
            if stream.ndim != 2 or stream.shape[-1] != self.base_width:
                continue
            flat = self._hist.flatten(1)
            if self.zero_history:
                flat = torch.zeros_like(flat)
            td[key] = torch.cat((stream, flat), dim=-1)
        return td

    def _advance(self, base_obs):
        # Scrub before storing. The environment scrubs its OWN observation, but
        # the previous ACTION is not scrubbed anywhere, so one NaN action would
        # poison the whole history window and surface as a NaN observation --
        # which is exactly how the first bimodal run died at iteration 314.
        step = torch.cat((base_obs[:, : self.base_dim], self._prev), dim=-1)
        step = torch.nan_to_num(step, nan=0.0, posinf=0.0, neginf=0.0)
        self._hist = torch.roll(self._hist, shifts=-1, dims=1)
        self._hist[:, -1, :] = step

    def get_observations(self):
        td = self.inner.get_observations()
        first = td[next(iter(td.keys()))]
        self._advance(first)
        return self._augment(td)

    def step(self, actions):
        td, rew, dones, extras = self.inner.step(actions)
        self._prev = torch.nan_to_num(
            actions.detach().clone(), nan=0.0, posinf=0.0, neginf=0.0)
        first = td[next(iter(td.keys()))]
        self._advance(first)
        if dones is not None and bool(dones.any()):
            # A window spanning a reset describes two episodes and, with per-env
            # plant draws, two different plants.
            mask = dones.bool()
            self._hist[mask] = 0.0
            self._prev[mask] = 0.0
        return self._augment(td), rew, dones, extras


def main() -> int:
    env_cfg = parse_env_cfg(a.task, device=a.device, num_envs=a.num_envs)
    apply_c4_authority_stage(env_cfg, int(a.stage))
    base_width = int(env_cfg.observation_space)
    action_dim = int(env_cfg.action_space)
    base_dim = base_width - 4 if base_width in (70, 72) else base_width
    hist_cols = int(a.history) * (base_dim + action_dim)
    wide = base_width + hist_cols
    # env_cfg.observation_space MUST stay at base_width: the environment derives
    # its own expected width (`_expected_observation_dim`) and raises if the cfg
    # disagrees. Only the RUNNER sees the wide observation, and it sees it from
    # the wrapper's output rather than from this config.
    print(f"[residual] base_width={base_width} base_dim={base_dim} action={action_dim} "
          f"history={a.history} -> runner observation width {wide}", flush=True)

    env = gym.make(a.task, cfg=env_cfg, render_mode=None)

    runner_cfg = PpoIDAAMFDPPORunnerCfg()
    runner_cfg.seed = int(a.seed)
    runner_cfg.max_iterations = int(a.iterations)
    runner_cfg.save_interval = int(a.save_interval)
    runner_cfg.algorithm.learning_rate = float(a.lr)

    wrapped = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)
    wrapped = HistoryObservation(
        wrapped, base_dim=base_dim, base_width=base_width, action_dim=action_dim,
        history=int(a.history), num_envs=int(a.num_envs), device=a.device,
        zero_history=bool(a.no_history))

    log_dir = a.log_dir or str(
        ROOT / "logs/rsl_rl/ppo_id_aam_fd" /
        ("residual_nohistory" if a.no_history else "residual"))
    runner = OnPolicyRunner(wrapped, runner_cfg.to_dict(), log_dir=log_dir, device=a.device)

    # Splice the residual at the model's `mlp`. rsl_rl builds the action
    # distribution FROM the mlp output and calls a wide interface on the actor,
    # so replacing the mlp keeps every one of those paths intact and changes only
    # the mean.
    payload = torch.load(a.base_checkpoint, map_location=a.device, weights_only=False)
    sd = payload["actor_state_dict"]
    got = sd["mlp.0.weight"].shape[1]
    if got != base_width:
        raise SystemExit(
            f"base checkpoint expects observation width {got}, environment "
            f"produces {base_width}: wrong checkpoint for this task"
        )

    # Frozen base mlp, weights straight from the checkpoint.
    base_mlp = torch.nn.Sequential(
        torch.nn.Linear(sd["mlp.0.weight"].shape[1], sd["mlp.0.weight"].shape[0]),
        torch.nn.ELU(),
        torch.nn.Linear(sd["mlp.2.weight"].shape[1], sd["mlp.2.weight"].shape[0]),
        torch.nn.ELU(),
        torch.nn.Linear(sd["mlp.4.weight"].shape[1], sd["mlp.4.weight"].shape[0]),
        torch.nn.ELU(),
        torch.nn.Linear(sd["mlp.6.weight"].shape[1], sd["mlp.6.weight"].shape[0]),
    ).to(a.device)
    with torch.no_grad():
        for i in (0, 2, 4, 6):
            base_mlp[i].weight.copy_(sd[f"mlp.{i}.weight"])
            base_mlp[i].bias.copy_(sd[f"mlp.{i}.bias"])

    # Seed the WIDE normaliser: checkpoint statistics on the base columns,
    # identity on the history columns, then freeze it. Without this the frozen
    # base would receive inputs normalised by a fresh, wrong normaliser.
    norm = runner.alg.actor.obs_normalizer
    with torch.no_grad():
        for attr, base_val, hist_val in (("_mean", sd["obs_normalizer._mean"], 0.0),
                                         ("_var", sd["obs_normalizer._var"], 1.0),
                                         ("_std", sd["obs_normalizer._std"], 1.0)):
            if not hasattr(norm, attr):
                continue
            tgt = getattr(norm, attr)
            tgt[..., :base_width] = base_val.to(tgt.device).reshape(-1)[:base_width]
            tgt[..., base_width:] = hist_val
    for parameter in norm.parameters() if hasattr(norm, "parameters") else []:
        parameter.requires_grad_(False)
    if hasattr(runner.alg.actor, "update_normalization"):
        runner.alg.actor.update_normalization = lambda *args, **kwargs: None
        print("[residual] observation normaliser seeded from the base checkpoint "
              "and FROZEN", flush=True)

    residual = ResidualMLP(
        base_mlp, base_dim=base_dim, base_width=base_width, action_dim=action_dim,
        history=int(a.history), latent_dim=int(a.latent_dim), alpha=float(a.alpha),
    ).to(a.device)
    runner.alg.actor.mlp = residual

    from itertools import chain
    runner.alg.optimizer = torch.optim.Adam(
        chain(residual.trainable_parameters(), runner.alg.critic.parameters()),
        lr=float(a.lr))
    n_train = sum(p.numel() for p in residual.trainable_parameters())
    n_frozen = sum(p.numel() for p in residual.base_mlp.parameters())
    print(f"[residual] trainable {n_train} | frozen base {n_frozen} | alpha={a.alpha} "
          f"| history={'ZEROED (ablation)' if a.no_history else 'live'}", flush=True)

    runner.learn(num_learning_iterations=int(a.iterations), init_at_random_ep_len=False)
    env.close()
    return 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    raise SystemExit(code)
