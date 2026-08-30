#!/usr/bin/env python3
"""RMA Phase 2: adaptation module phi(history) -> z_hat, trained ON-POLICY.

Canonical RMA (arXiv 2107.04034) trains phi by regressing the privileged latent
from a state-action history, with the base policy frozen. The usual
implementation collects that regression set under the TEACHER's actions, which
is behaviour cloning on expert-visited states -- the classic covariate shift
failure. At deployment phi sees the states its OWN estimation errors produce,
which it was never trained on.

Lee et al. (Science Robotics 2020) fix exactly this with DAgger: the student
increasingly drives while the privileged teacher keeps labelling. This repo
already uses that pattern for the DLS distillation (``verge_task/distill.py``,
``execute_teacher_start`` 1.0 -> 0.0), so Phase 2 uses it too rather than
inventing anything.

Why not the existing adapter: ``AAM_module/control_latent/train_privileged_adapter.py``
is an offline CPU fit on a frozen ROM dataset. Measured paired, that frozen
latent carried ~0.4% plant identity and scored 7.1% WORSE than mean context --
feeding a policy near-noise is worse than feeding it a constant.

What phi can and cannot learn here is itself a result. Of the six privileged
columns (flow, compliance, seal, pressure, mass_scale, com_z) only mass_scale and
com_z act on this stack -- ``CylinderFD.step()`` takes no factor argument, so the
four hydraulic columns never reach the plant. The response therefore carries NO
information about them and their R^2 must come out at ~0. Per-column R^2 is
reported for that reason: it separates "phi failed" from "the column is
unidentifiable by construction".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

p = argparse.ArgumentParser()
p.add_argument("checkpoint", type=Path, help="Phase 1 policy (privileged context route)")
p.add_argument("--out", type=Path, required=True)
p.add_argument("--num_envs", type=int, default=176)
p.add_argument("--iterations", type=int, default=60)
p.add_argument("--steps_per_iteration", type=int, default=96)
p.add_argument("--history", type=int, default=20, help="history window k, in control steps")
p.add_argument("--lr", type=float, default=1.0e-3)
p.add_argument("--epochs_per_iteration", type=int, default=4)
p.add_argument("--minibatch", type=int, default=4096)
p.add_argument("--context_dim", type=int, default=6)
p.add_argument("--seed", type=int, default=17)
p.add_argument("--stage", type=int, default=4)
p.add_argument("--task", type=str, default="RRLAB-Mulag-Verge-ID-AAM-FD-v0")
# DAgger schedule: probability of feeding the policy the TRUE context.
p.add_argument("--teacher_start", type=float, default=1.0)
p.add_argument("--teacher_end", type=float, default=0.0)
p.add_argument("--teacher_anneal_fraction", type=float, default=0.6)
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
# NOT from PPO_ROM/scripts/evaluate_gates: that module runs AppLauncher at
# import time, so importing it here would launch a second simulation app.
from PPO_ROM.terrain_env_cfg import apply_c4_authority_stage  # noqa: E402


from verge_task.adaptation_module import AdaptationModule  # noqa: E402


def teacher_probability(it: int) -> float:
    span = max(1.0, float(a.iterations) * float(a.teacher_anneal_fraction))
    frac = min(1.0, float(it) / span)
    return float(a.teacher_start + (a.teacher_end - a.teacher_start) * frac)


def main() -> int:
    env_cfg = parse_env_cfg(a.task, device=a.device, num_envs=a.num_envs)
    apply_c4_authority_stage(env_cfg, int(a.stage))
    env = gym.make(a.task, cfg=env_cfg, render_mode=None)
    raw = env.unwrapped

    runner_cfg = PpoIDAAMFDPPORunnerCfg()
    runner_cfg.seed = int(a.seed)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)
    runner = OnPolicyRunner(wrapped, runner_cfg.to_dict(), log_dir=None, device=a.device)
    runner.load(
        str(a.checkpoint),
        load_cfg={"actor": True, "critic": True, "optimizer": False,
                  "iteration": True, "rnd": False},
    )
    policy = runner.get_inference_policy(device=a.device)

    obs_dim = int(env_cfg.observation_space)
    ctx = int(a.context_dim)
    base_dim = obs_dim - ctx
    act_dim = int(env_cfg.action_space)
    feature_dim = base_dim + act_dim
    device = torch.device(a.device)
    print(f"[phase2] obs={obs_dim} base={base_dim} context={ctx} action={act_dim} "
          f"history={a.history} envs={a.num_envs}", flush=True)

    phi = AdaptationModule(feature_dim, int(a.history), ctx).to(device)
    opt = torch.optim.Adam(phi.parameters(), lr=float(a.lr))

    n = int(a.num_envs)
    hist = torch.zeros((n, int(a.history), feature_dim), device=device)
    # RslRlVecEnvWrapper returns a TensorDict, not a tuple, and the rsl_rl
    # actor indexes it BY OBSERVATION GROUP -- it cannot be handed a plain
    # tensor. So the TensorDict is carried through the loop and the context
    # columns are rewritten inside it.
    obs_td = wrapped.get_observations()

    def context_streams(td):
        """Keys whose stream carries the context columns."""
        return [k for k in td.keys()
                if td[k].ndim == 2 and td[k].shape[-1] == base_dim + ctx]

    streams = context_streams(obs_td)
    if not streams:
        raise RuntimeError(
            f"no observation stream of width {base_dim + ctx}; keys={list(obs_td.keys())}"
        )
    print(f"[phase2] context-carrying observation streams: {streams}", flush=True)
    prev_action = torch.zeros((n, act_dim), device=device)

    # Target normalisation is estimated once from the first iteration's labels
    # and then frozen; a running estimate would move the target under the
    # regression and make the loss curve uninterpretable.
    tgt_mean = None
    tgt_std = None
    log: list[dict] = []

    for it in range(int(a.iterations)):
        p_teacher = teacher_probability(it)
        H: list[torch.Tensor] = []
        Y: list[torch.Tensor] = []
        for _ in range(int(a.steps_per_iteration)):
            reference = obs_td[streams[0]]
            base_obs = reference[:, :base_dim]
            true_ctx = reference[:, base_dim:].detach().clone()
            with torch.no_grad():
                est_ctx = phi(hist)
            if tgt_mean is not None:
                est_ctx = est_ctx * tgt_std + tgt_mean
            # DAgger: the student increasingly drives while the privileged
            # context keeps labelling, so phi is trained on the state
            # distribution its OWN errors produce.
            use_teacher = (torch.rand(n, 1, device=device) < p_teacher).float()
            fed_ctx = use_teacher * true_ctx + (1.0 - use_teacher) * est_ctx
            # Rewrite EVERY context-carrying stream, critic included: leaving the
            # true context in the critic while the actor sees a mixture would
            # feed the value function privileged information the actor lacks.
            for key in streams:
                obs_td[key] = torch.cat(
                    (obs_td[key][:, :base_dim], fed_ctx), dim=-1
                )

            H.append(hist.detach().clone())
            Y.append(true_ctx)

            with torch.no_grad():
                action = policy(obs_td)
            step_feat = torch.cat((base_obs, prev_action), dim=-1)
            hist = torch.roll(hist, shifts=-1, dims=1)
            hist[:, -1, :] = step_feat

            obs_td, _, dones, _ = wrapped.step(action)
            prev_action = action.detach().clone()
            # A history that straddles a reset describes two different episodes
            # and, with per-env plant draws, two different plants.
            if dones is not None and bool(dones.any()):
                hist[dones.bool()] = 0.0

        Hs = torch.cat(H, dim=0)
        Ys = torch.cat(Y, dim=0)
        if tgt_mean is None:
            tgt_mean = Ys.mean(dim=0, keepdim=True)
            tgt_std = Ys.std(dim=0, keepdim=True).clamp_min(1e-6)
            print(f"[phase2] target mean={tgt_mean.flatten().tolist()}", flush=True)
            print(f"[phase2] target std ={tgt_std.flatten().tolist()}", flush=True)
        Yn = (Ys - tgt_mean) / tgt_std

        total = 0.0
        count = 0
        for _ in range(int(a.epochs_per_iteration)):
            perm = torch.randperm(Hs.shape[0], device=device)
            for s in range(0, Hs.shape[0], int(a.minibatch)):
                idx = perm[s: s + int(a.minibatch)]
                pred = phi(Hs[idx])
                loss = nn.functional.mse_loss(pred, Yn[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total += float(loss) * idx.numel()
                count += idx.numel()

        with torch.no_grad():
            pred = phi(Hs) * tgt_std + tgt_mean
            ss_res = ((pred - Ys) ** 2).sum(dim=0)
            ss_tot = ((Ys - Ys.mean(dim=0, keepdim=True)) ** 2).sum(dim=0).clamp_min(1e-12)
            r2 = (1.0 - ss_res / ss_tot).tolist()
        row = {"iteration": it, "p_teacher": p_teacher,
               "loss": total / max(count, 1), "r2_per_column": r2}
        log.append(row)
        print(f"[phase2] it={it:3d} p_teacher={p_teacher:.2f} loss={row['loss']:.5f} "
              f"R2={[round(v, 3) for v in r2]}", flush=True)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": phi.state_dict(), "feature_dim": feature_dim,
         "history": int(a.history), "context_dim": ctx,
         "target_mean": tgt_mean.cpu(), "target_std": tgt_std.cpu()},
        out / "phi.pt",
    )
    (out / "metrics.json").write_text(json.dumps(
        {"columns": ["flow", "compliance", "seal", "pressure", "mass_scale", "com_z"],
         "final_r2": log[-1]["r2_per_column"] if log else None,
         "log": log}, indent=2))
    print(f"[phase2] wrote {out/'phi.pt'}", flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    raise SystemExit(code)
