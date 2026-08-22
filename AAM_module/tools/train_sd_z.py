#!/usr/bin/env python
"""Stage 20/21: privileged System Factor Encoder ``z_i = mu_psi(e_i)``.

Stage 20 spec (items 51-53):

- ``e_i`` is the 6-D physical deployment factor vector
  (flow, compliance, seal, pressure, mass_scale, com_z).
- It is normalized with TRAINING domains only: ``e_tilde = (e_i - mu_e)/sigma_e``;
  test/val domains never enter the normalization statistics.
- ``z_i = mu_psi(e_tilde_i)`` is the System Factor Encoder, trained from
  scratch, JOINTLY with the dynamics model ``SD_z(s_t, u_t, z_i) -> Delta s``.
  Gradients from the dynamics loss flow through the SD into psi, so the
  latent learns which physical differences matter dynamically (item 52).
- Latent bottleneck (item 53): ``d_z`` swept over {2, 4, 8} (2 and 4 are true
  bottlenecks below d_e=6; 8 is the reference dimension, not a bottleneck).
  The chosen latent is the SMALLEST whose held-out ROLLOUT performance stays
  close to the best (evaluate with tools/evaluate_sd_rollout.py --z-latent).

Stage 21 combined loss (spec item 54): when ``--unroll > 1`` the trainer
minimises the weighted combination

    L = lambda_1 * L_1step + lambda_R * L_rollout

where ``L_1step`` is a teacher-forced one-step prediction loss over the
rollout window (true states every step, no noise, no sampling) and
``L_rollout`` is the unrolled multi-step loss with scheduled sampling
(teacher forcing decays to ``--tf-floor``). With ``--rollout-ramp-epochs N``
``lambda_R`` ramps linearly from 0 to ``--rollout-weight`` over the first
``N`` epochs, so training starts with pure one-step prediction and
progressively introduces the rollout objective. Set ``--l1-weight 0`` to
reproduce the Stage 20 unrolled-only loss.

Stage 22 v3 (causal-belief protocol, 2026-08-09): restores the empirically
justified inputs that Stage 19b showed matter and separates the dynamic
plant state from the static configuration latent, following the Stage 22
gate evidence:

- The SD input now includes the causal observation/action window (``hist6``-
  style, ``--history-lag`` blocks of ``[s^H_keep; u]``) compressed by a GRU
  ``BeliefEncoder`` into a dynamic belief ``h_t``, plus the joint context
  ``q`` (10-D) and the static latent ``z``. ``h`` carries the instantaneous
  plant state (e.g. pressure when dP is masked at deployment); ``z`` is
  responsible only for slowly varying configuration. They are never fused
  into a single vector.
- The rollout term is replaced by normalized accumulated STATE error
  ``||(s_hat_{j+1} - s_true_{j+1}) / sigma_s||^2`` over the unrolled
  horizon; the one-step delta term ``L_1step`` is retained as an auxiliary
  teacher-forced objective. The previous term scored predicted delta
  against recorded delta even after the input state had drifted, so it was
  not a genuine state-rollout objective.

The encoder is PRIVILEGED (uses the true factors at training/collection
time). The online AAM that infers z_hat from observable history is the
follow-up stage; this stage builds the supervised latent target for it.

Usage:
    python train_sd_z.py --root datasets/master_v1_frozen --latent-dim 4 \
        --out outputs/sd_z/dz4
    python train_sd_z.py --root datasets/master_v1_frozen --latent-dim 2 \
        --unroll 10 --l1-weight 1.0 --rollout-weight 1.0 \
        --rollout-ramp-epochs 150 --out outputs/sd_z_stage21/dz2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
AAM_ROOT = Path(__file__).resolve().parents[1]
if str(AAM_ROOT) not in sys.path:
    sys.path.insert(0, str(AAM_ROOT))

from aam_data import load_split  # noqa: E402


FACTOR_KEYS = ("flow", "compliance", "seal", "pressure", "mass_scale", "com_z")
D_E = 6
HYD_STATE_DIM = 12
MECH_CONTEXT_DIM = 10
CMD_DIM = 8
TARGET_DIM = 12


def run_factors(sd_root: Path, run_id: int) -> np.ndarray:
    meta = json.loads((sd_root / f"{int(run_id):05d}" / "sample_meta.json").read_text())
    return np.asarray([float(meta["z"][k]) for k in FACTOR_KEYS], dtype=np.float64)


def _usable(z, enabled: bool, n: int) -> np.ndarray:
    """Per-row training usability; all-ones when masking is off."""
    if not enabled:
        return np.ones(n, dtype=np.float64)
    from aam_data.sample_mask import usable_rows
    return usable_rows(z).astype(np.float64)


def load_run(sd_root: Path, run_id: int, sample_mask: bool = False) -> dict:
    with np.load(sd_root / f"{int(run_id):05d}" / "samples.npz", allow_pickle=False) as z:
        return {
            "s": z["s_h"].astype(np.float64),
            "u": z["u"].astype(np.float64),
            "q": z["q"].astype(np.float64),
            "target": z["target"].astype(np.float64),
            "m": _usable(z, sample_mask, int(z["s_h"].shape[0])),
            "e": run_factors(sd_root, run_id),
        }


class FactorEncoder(nn.Module):
    """Small privileged encoder e_tilde (6-D) -> z (d_z-D)."""

    def __init__(self, latent_dim: int, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D_E, hidden), nn.SiLU(), nn.Linear(hidden, latent_dim)
        )

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        return self.net(e)


class FactorDecoder(nn.Module):
    """Auxiliary reconstruction head z -> e_hat (normalized 6-D factors).

    A ``factor_recon_weight > 0`` adds ``MSE(decoder(z), e_tilde)`` to the
    joint loss so the bottleneck is forced to retain ALL six factors, not
    just the two that happen to help the dynamics term (the measured reason
    the pressure/compliance factors were dropped at d_z=2).
    """

    def __init__(self, latent_dim: int, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.SiLU(), nn.Linear(hidden, D_E)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class BeliefEncoder(nn.Module):
    """Dynamic belief ``h_t`` inferred from the causal observation/action window.

    The GRU runs chronologically over the ``--history-lag`` most recent
    blocks ``[s_keep; u]``; the final hidden state is the belief fed to
    SD_z together with the static latent ``z``. Dynamic plant state (e.g.
    pressure when dP is masked) is therefore carried by ``h`` while ``z``
    carries slowly varying configuration only (RMA/PlaNet-style separation).
    """

    def __init__(self, block_dim: int, belief_dim: int) -> None:
        super().__init__()
        self.gru = nn.GRU(block_dim, belief_dim, batch_first=True)

    def init_from_window(self, blocks: torch.Tensor) -> torch.Tensor:
        """``blocks``: (B, lag, block_dim) oldest->newest -> (B, belief_dim)."""
        _, h = self.gru(blocks)
        return h[-1]

    def step(self, block: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """``block``: (B, block_dim); ``h``: (B, belief_dim) -> next belief."""
        _, h_new = self.gru(block[:, None, :], h[None, :, :])
        return h_new[-1]


class SDz(nn.Module):
    """SD_z(s_t, u_t, z_i) -> Delta s_{t+1}: base MLP plus the latent input."""

    def __init__(self, latent_dim: int, state_dim: int = HYD_STATE_DIM,
                 hidden: int = 256, depth: int = 3, dropout: float = 0.05,
                 use_q: bool = False, use_belief: bool = False,
                 belief_dim: int = 0) -> None:
        super().__init__()
        in_dim = state_dim + CMD_DIM + latent_dim
        if use_q:
            in_dim += MECH_CONTEXT_DIM
        if use_belief:
            in_dim += belief_dim
        layers = [nn.Linear(in_dim, hidden), nn.SiLU(), nn.Dropout(dropout)]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(hidden, TARGET_DIM))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def causal_windows_np(s_keep: np.ndarray, u: np.ndarray, lag: int) -> np.ndarray:
    """Causal blocks for every valid tick ``t >= lag-1``, oldest first.

    Returns ``(n - lag + 1, lag, state_dim + CMD_DIM)`` where position ``p``
    in the lag axis is tick ``t - (lag-1-p)``. Rows whose window reaches
    before tick 0 are dropped (never padded), matching the ``history_L*``
    files of the Stage 18 SD dataset.
    """
    n = s_keep.shape[0]
    if n < lag:
        return np.zeros((0, lag, s_keep.shape[1] + u.shape[1]), dtype=np.float64)
    B = n - lag + 1
    blocks = np.empty((B, lag, s_keep.shape[1] + u.shape[1]), dtype=np.float64)
    for p in range(lag):
        blocks[:, p, :] = np.concatenate(
            [s_keep[p : n - lag + p + 1], u[p : n - lag + p + 1]], axis=-1
        )
    return blocks


def group_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    out = {}
    for name, (lo, hi) in {"extension": (0, 4), "velocity": (4, 8), "dP": (8, 12)}.items():
        out[name] = float(np.sqrt(np.mean((y_true[:, lo:hi] - y_pred[:, lo:hi]) ** 2)))
    out["all"] = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=str(AAM_ROOT / "datasets/master_v1_frozen"))
    p.add_argument("--latent-dim", type=int, default=4)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument(
        "--history-lag",
        type=int,
        default=6,
        help="causal blocks (50 ms ticks) of [s^H_keep; u] fed to the GRU "
        "belief encoder (Stage 19b: hist6 saturates observable history). "
        "0 disables the belief encoder entirely (legacy instantaneous SD_z).",
    )
    p.add_argument(
        "--belief-dim",
        type=int,
        default=16,
        help="GRU belief width; 0 disables the dynamic belief (legacy "
        "instantaneous SD_z without recurrent context).",
    )
    p.add_argument(
        "--no-q",
        action="store_true",
        help="exclude the 10-D joint context q from the SD input "
        "(Stage 18 item 48 keeps joints as context; enabled by default).",
    )
    p.add_argument("--sample-mask", action="store_true",
                   help="exclude command-saturated / cylinder-limited samples "
                        "from the LOSS (rows stay in place). Required for the "
                        "open-loop excitation collection, and must match "
                        "whatever the model it is compared against used.")
    p.add_argument("--unroll", type=int, default=1,
                   help=">1 trains with an unrolled (multi-step) loss so the "
                   "latent must explain config-dependent state divergence "
                   "(RMA-style); 1 is the plain one-step joint training")
    p.add_argument(
        "--obs-drop",
        default="",
        help="comma-separated 0-based state channels to HIDE from the SD "
        "input (partial-observation variant: the latent then becomes the "
        "only config channel). Default '' keeps all 12 hydraulic channels.",
    )
    p.add_argument("--l1-weight", type=float, default=1.0,
                   help="Stage 21 weight for the teacher-forced one-step term")
    p.add_argument("--rollout-weight", type=float, default=1.0,
                   help="Stage 21 target weight for the unrolled rollout term")
    p.add_argument(
        "--state-loss-weight",
        type=float,
        default=1.0,
        help="multiplier applied to the normalized accumulated STATE error "
        "term inside the unrolled objective (Stage 22 v3 replaces the "
        "delta-vs-delta rollout term with accumulated state error)",
    )
    p.add_argument(
        "--rollout-q-mode",
        choices=("recorded", "hold-start"),
        default="recorded",
        help="joint-context contract for the autoregressive state-loss term: "
        "'recorded' uses measured future q; 'hold-start' keeps q at the "
        "rollout-start value. The teacher-forced one-step term always uses "
        "the recorded q at its current tick.",
    )
    p.add_argument(
        "--latent-rank-weight",
        type=float,
        default=0.0,
        help="weight for a bounded counterfactual rollout-ranking loss that "
        "requires the correctly paired latent to beat shuffled/mean training "
        "latents. Only valid for latent-conditioned unrolled training.",
    )
    p.add_argument(
        "--latent-rank-margin",
        type=float,
        default=0.01,
        help="minimum per-sample normalized state-MSE advantage required by "
        "the counterfactual latent-ranking hinge loss",
    )
    p.add_argument(
        "--latent-rank-control",
        choices=("shuffled", "mean", "both"),
        default="both",
        help="negative latent controls used by --latent-rank-weight",
    )
    p.add_argument("--rollout-ramp-epochs", type=int, default=0,
                   help="ramp lambda_R from 0 to --rollout-weight over the "
                   "first N epochs; 0 keeps lambda_R constant (no curriculum)")
    p.add_argument(
        "--factor-recon-weight",
        type=float,
        default=0.0,
        help="auxiliary MSE weight for the factor-reconstruction head "
        "decoder(z) -> e_tilde, so z is forced to retain ALL six factors "
        "(0 disables the head)",
    )
    p.add_argument(
        "--latent-source",
        choices=("encoder", "direct", "none"),
        default="encoder",
        help="'encoder' learns z = mu_psi(e_tilde) (Stage 20 default); "
        "'direct' conditions SD_z on the normalized physical factors "
        "e_tilde themselves (6-D), i.e. the deployment plan's first "
        "interpretable system-factor representation without a neural "
        "bottleneck (less capacity to overfit train combos); 'none' trains "
        "the identical SD architecture WITHOUT any latent input (the "
        "protocol-matched no-latent control for Stage 22).",
    )
    p.add_argument("--val-unroll", type=int, default=10)
    p.add_argument("--tf-start", type=float, default=1.0)
    p.add_argument("--tf-floor", type=float, default=0.2)
    p.add_argument("--noise-std", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--seed", type=int, default=20260807)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=str(AAM_ROOT / "outputs/sd_z"))
    p.add_argument("--log", default=None)
    args = p.parse_args()

    obs_drop = sorted({int(x) for x in args.obs_drop.split(",") if x.strip()})
    if any(i < 0 or i >= HYD_STATE_DIM for i in obs_drop):
        raise SystemExit(f"--obs-drop indices must be in 0..{HYD_STATE_DIM-1}: {obs_drop}")
    state_keep = [i for i in range(HYD_STATE_DIM) if i not in set(obs_drop)]
    if not state_keep:
        raise SystemExit("--obs-drop cannot hide every state channel")
    if args.latent_rank_weight < 0:
        raise SystemExit("--latent-rank-weight must be non-negative")
    if args.latent_rank_margin < 0:
        raise SystemExit("--latent-rank-margin must be non-negative")
    if args.latent_rank_weight > 0 and (args.unroll <= 1 or args.latent_source == "none"):
        raise SystemExit(
            "--latent-rank-weight requires --unroll > 1 and a latent-conditioned model"
        )
    state_dim = len(state_keep)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    root = Path(args.root)
    sd_root = root / "sd" / "dataset"
    train_ids = load_split(root, "train")
    val_ids = load_split(root, "val")
    test_ids = load_split(root, "test")

    train_runs = {rid: load_run(sd_root, rid, args.sample_mask) for rid in train_ids}
    val_runs = {rid: load_run(sd_root, rid, args.sample_mask) for rid in val_ids}
    test_runs = {rid: load_run(sd_root, rid, args.sample_mask) for rid in test_ids}

    # Normalize e with TRAINING domains only (item 51).
    E_train = np.stack([r["e"] for r in train_runs.values()])
    e_mean = E_train.mean(axis=0)
    e_std = E_train.std(axis=0).clip(min=1e-6)

    # Input/target z-score stats from train samples.
    def stack(field):
        return np.concatenate([r[field] for r in train_runs.values()], axis=0)

    S = np.concatenate([r["s"][:, state_keep] for r in train_runs.values()], axis=0)
    U, Q, T = stack("u"), stack("q"), stack("target")
    s_mean, s_std = S.mean(0), S.std(0).clip(min=1e-6)
    u_mean, u_std = U.mean(0), U.std(0).clip(min=1e-6)
    q_mean, q_std = Q.mean(0), Q.std(0).clip(min=1e-6)
    y_mean, y_std = T.mean(0), T.std(0).clip(min=1e-6)

    # Per-run normalized e tensor for the encoder.
    e_map = {rid: (r["e"] - e_mean) / e_std for rid, r in train_runs.items()}
    e_map.update({rid: (r["e"] - e_mean) / e_std for rid, r in val_runs.items()})
    e_map.update({rid: (r["e"] - e_mean) / e_std for rid, r in test_runs.items()})
    e_t = {rid: torch.as_tensor(v, dtype=torch.float32, device=device) for rid, v in e_map.items()}

    if args.latent_source == "encoder":
        latent_dim = args.latent_dim
    elif args.latent_source == "direct":
        latent_dim = D_E
    else:
        latent_dim = 0
    use_q = not args.no_q
    use_belief = args.belief_dim > 0 and args.history_lag > 0
    history_lag = max(args.history_lag, 0)
    encoder = (
        FactorEncoder(args.latent_dim).to(device)
        if args.latent_source == "encoder"
        else None
    )
    decoder = (
        FactorDecoder(args.latent_dim).to(device)
        if args.latent_source == "encoder" and args.factor_recon_weight > 0
        else None
    )
    belief = (
        BeliefEncoder(state_dim + CMD_DIM, args.belief_dim).to(device)
        if use_belief
        else None
    )
    sd = SDz(latent_dim, state_dim=state_dim, hidden=args.hidden,
             depth=args.depth, dropout=args.dropout,
             use_q=use_q, use_belief=use_belief,
             belief_dim=args.belief_dim if use_belief else 0).to(device)
    params = (
        list(sd.parameters())
        + (list(encoder.parameters()) if encoder is not None else [])
        + (list(belief.parameters()) if belief is not None else [])
        + (list(decoder.parameters()) if decoder is not None else [])
    )
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 1e-2
    )
    loss_fn = nn.MSELoss()

    s_mean_t = torch.as_tensor(s_mean, dtype=torch.float32, device=device)
    s_std_t = torch.as_tensor(s_std, dtype=torch.float32, device=device)
    u_mean_t = torch.as_tensor(u_mean, dtype=torch.float32, device=device)
    u_std_t = torch.as_tensor(u_std, dtype=torch.float32, device=device)
    q_mean_t = torch.as_tensor(q_mean, dtype=torch.float32, device=device)
    q_std_t = torch.as_tensor(q_std, dtype=torch.float32, device=device)
    y_mean_t = torch.as_tensor(y_mean, dtype=torch.float32, device=device)
    y_std_t = torch.as_tensor(y_std, dtype=torch.float32, device=device)
    # Full 12-D state std (the integrated state is always full-dimensional
    # even when the SD input is masked) and the scheduled-sampling noise scale.
    s_full = np.concatenate([r["s"] for r in train_runs.values()], axis=0)
    s_std_full_t = torch.as_tensor(
        s_full.std(0).clip(min=1e-6).astype(np.float32), device=device
    )

    def forward_loss(X, W, Y, run_ids) -> torch.Tensor:
        if args.latent_source == "none":
            z = torch.zeros(X.shape[0], 0, device=device)
        else:
            e_batch = torch.stack([e_t[int(r)] for r in run_ids.tolist()])
            z = encoder(e_batch) if encoder is not None else e_batch
        if W is not None:
            wn = torch.cat(
                [
                    (W[..., :state_dim] - s_mean_t) / s_std_t,
                    (W[..., state_dim:] - u_mean_t) / u_std_t,
                ],
                dim=-1,
            )
            h = belief.init_from_window(wn)
            Xf = torch.cat([X, h], dim=-1)
        else:
            Xf = X
        pred = sd(torch.cat([Xf, z], dim=-1))
        return loss_fn(pred, Y), pred

    def build(runs: dict[int, dict]) -> dict:
        """One-step input rows, concatenated across runs with run ids.

        ``X`` rows (raw units): ``[s_keep; u; (q)]``. ``W`` holds each row's
        causal window blocks (raw units, ``(N, lag, state_dim + CMD_DIM)``,
        oldest first) so the GRU belief is computed inside the forward pass
        and receives gradients; it is None when the belief is disabled. The
        first ``history_lag - 1`` ticks of each run are dropped (never
        padded), mirroring the ``history_L*`` files.
        """
        Xs, Ys, rids = [], [], []
        Ws = []
        for rid, r in runs.items():
            s_keep = r["s"][:, state_keep]
            u, q, target = r["u"], r["q"], r["target"]
            n = s_keep.shape[0]
            offset = history_lag - 1 if use_belief else 0
            if n <= offset:
                continue
            cols = [s_keep[offset:], u[offset:]]
            if use_q:
                cols.append(q[offset:])
            if use_belief:
                Ws.append(causal_windows_np(s_keep, u, history_lag))
            Xs.append(np.concatenate(cols, axis=-1))
            Ys.append(target[offset:])
            rids += [rid] * (n - offset)
        if Ws:
            W = np.concatenate(Ws, axis=0)
        else:
            W = None
        return {"X": np.concatenate(Xs, axis=0), "W": W,
                "Y": np.concatenate(Ys, axis=0),
                "run_ids": np.asarray(rids, dtype=np.int64)}

    x_mean = np.concatenate([s_mean, u_mean] + ([q_mean] if use_q else []))
    x_std = np.concatenate([s_std, u_std] + ([q_std] if use_q else []))
    Xn = lambda X: (X - x_mean) / x_std
    Yn = lambda Y: (Y - y_mean) / y_std

    if args.unroll == 1:
        tr, va, te = build(train_runs), build(val_runs), build(test_runs)
        Xtr = torch.as_tensor(Xn(tr["X"]).astype(np.float32), device=device)
        Ytr = torch.as_tensor(Yn(tr["Y"]).astype(np.float32), device=device)
        Xva = torch.as_tensor(Xn(va["X"]).astype(np.float32), device=device)
        Yva = torch.as_tensor(Yn(va["Y"]).astype(np.float32), device=device)
        Xte = torch.as_tensor(Xn(te["X"]).astype(np.float32), device=device)
        Yte = torch.as_tensor(Yn(te["Y"]).astype(np.float32), device=device)
        Wtr = torch.as_tensor(tr["W"].astype(np.float32), device=device) if tr["W"] is not None else None
        Wva = torch.as_tensor(va["W"].astype(np.float32), device=device) if va["W"] is not None else None
        Wte = torch.as_tensor(te["W"].astype(np.float32), device=device) if te["W"] is not None else None
        te_raw = te["Y"].astype(np.float64)
    else:
        tr = va = te = None
        Xtr = Ytr = Xva = Yva = Xte = Yte = te_raw = None
        Wtr = Wva = Wte = None

    def factor_recon_loss(ids: np.ndarray) -> torch.Tensor:
        """Auxiliary MSE(decoder(z), e_tilde); 0 when the head is disabled."""
        if decoder is None:
            return torch.zeros((), device=device)
        ids_list = ids.tolist() if hasattr(ids, "tolist") else ids
        e_batch = torch.stack([e_t[int(r)] for r in ids_list])
        z = encoder(e_batch)
        return args.factor_recon_weight * loss_fn(decoder(z), e_batch)

    @torch.no_grad()
    def evaluate(X, W, Y, run_ids, raw) -> dict:
        if X is None:
            return {"loss": float("nan"), "rmse": None, "r2_all": float("nan")}
        if encoder is not None:
            encoder.eval()
        if belief is not None:
            belief.eval()
        sd.eval()
        preds = []
        for start in range(0, X.shape[0], args.batch_size):
            b = slice(start, start + args.batch_size)
            Wb = W[b] if W is not None else None
            _, pred = forward_loss(X[b], Wb, Y[b], run_ids[b])
            preds.append(pred.detach().cpu().numpy())
        pred_norm = np.concatenate(preds, axis=0)
        pred = pred_norm * y_std + y_mean
        rmse = group_rmse(raw, pred)
        return {
            "loss": float(np.mean((pred_norm - Yn(raw)) ** 2)),
            "rmse": rmse,
            "r2_all": float(1 - np.sum((pred - raw) ** 2) / np.sum((raw - raw.mean(0)) ** 2)),
        }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log) if args.log else out / "training.log"
    log_fh = log_path.open("a")

    def log(msg: str) -> None:
        line = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
        print(line, flush=True)
        log_fh.write(line + "\n")
        log_fh.flush()

    # ---- unrolled path (latent must explain config-dependent state divergence)
    if args.unroll > 1:
        # All runs have the same number of samples (3768 in the v2 frozen
        # set), so the per-element python gather can be replaced by stacked
        # tensors + advanced indexing. This is the hot path of the unrolled
        # loss and was the reason epochs were ~5x slower than the old
        # instantaneous trainer.
        run_list = list(train_runs)

        def stack_runs(runs: dict[int, dict]) -> dict:
            order = list(runs)
            rows = {rid: i for i, rid in enumerate(order)}
            S = torch.stack(
                [torch.as_tensor(runs[rid]["s"], dtype=torch.float32, device=device) for rid in order]
            )
            U = torch.stack(
                [torch.as_tensor(runs[rid]["u"], dtype=torch.float32, device=device) for rid in order]
            )
            Q = torch.stack(
                [torch.as_tensor(runs[rid]["q"], dtype=torch.float32, device=device) for rid in order]
            )
            T = torch.stack(
                [torch.as_tensor(runs[rid]["target"], dtype=torch.float32, device=device) for rid in order]
            )
            M = torch.stack(
                [torch.as_tensor(runs[rid]["m"], dtype=torch.float32, device=device) for rid in order]
            )
            return {"rows": rows, "S": S, "U": U, "Q": Q, "T": T, "M": M}

        t_tr = stack_runs(train_runs)
        t_val = stack_runs(val_runs)
        E_tr = torch.stack(
            [torch.as_tensor(e_map[rid], dtype=torch.float32, device=device) for rid in run_list]
        )
        E_val = torch.stack(
            [torch.as_tensor(e_map[rid], dtype=torch.float32, device=device) for rid in val_runs]
        )

        def belief_from_window(S, U, rows, starts):
            """Belief at tick ``start`` from the causal lag window (oldest first)."""
            lag = history_lag
            ticks = starts[:, None] - (lag - 1 - torch.arange(lag, device=device))[None, :]
            s_blk = S[rows[:, None], ticks][:, :, state_keep]  # (B, lag, state_dim)
            u_blk = U[rows[:, None], ticks]  # (B, lag, CMD)
            blocks = torch.cat(
                [(s_blk - s_mean_t) / s_std_t, (u_blk - u_mean_t) / u_std_t], dim=-1
            )
            return belief.init_from_window(blocks)

        def unrolled_loss(runs, E, rows, starts, tf_ratio, noise_on, horizon,
                          teacher_forced=False):
            """Returns ``(one-step delta loss, accumulated state loss)``.

            The rollout term is the normalized accumulated STATE error
            ``||(s_hat_{j+1} - s_true_{j+1}) / sigma_s||^2`` (the model
            integrates its own deltas and is scored on where it actually
            lands); the one-step delta term stays as the teacher-forced
            auxiliary objective.
            """
            B = starts.shape[0]
            S, U, Q, T = runs["S"], runs["U"], runs["Q"], runs["T"]
            Mm = runs["M"]

            def _mmse(pred, tgt, w):
                """MSE over usable rows only; identical semantics to the e2e
                trainer's masked_mse so the two are scored the same way."""
                ww = w[:, None]
                denom = ww.sum() * pred.shape[1]
                if float(denom) == 0.0:
                    return pred.sum() * 0.0
                return ((pred - tgt) ** 2 * ww).sum() / denom
            s_cur = S[rows, starts]
            if args.latent_source == "none":
                z = torch.zeros(B, 0, device=device)
            else:
                e_batch = E[rows]
                z = encoder(e_batch) if encoder is not None else e_batch
            if use_belief:
                h = belief_from_window(S, U, rows, starts)
            else:
                h = None
            l1_total = torch.zeros((), device=device)
            st_total = torch.zeros((), device=device)
            for j in range(horizon):
                ticks = starts + j
                u = U[rows, ticks]
                q = (
                    Q[rows, ticks]
                    if teacher_forced or args.rollout_q_mode == "recorded"
                    else Q[rows, starts]
                )
                target = T[rows, ticks]
                s_true_next = S[rows, ticks + 1]
                s_in = s_cur + (args.noise_std * s_std_full_t * torch.randn_like(s_cur) if noise_on else 0.0)
                s_keep_n = (s_in[:, state_keep] - s_mean_t) / s_std_t
                u_n = (u - u_mean_t) / u_std_t
                if use_belief:
                    h = belief.step(torch.cat([s_keep_n, u_n], dim=-1), h)
                cols = [s_keep_n, u_n]
                if use_q:
                    cols.append((q - q_mean_t) / q_std_t)
                if use_belief:
                    cols.append(h)
                X = torch.cat(cols + [z], dim=-1)
                pred_norm = sd(X)
                # Saturated steps propagate but contribute no gradient.
                mj = Mm[rows, ticks]
                l1_total = l1_total + _mmse(pred_norm, (target - y_mean_t) / y_std_t, mj)
                delta_raw = pred_norm * y_std_t + y_mean_t
                s_next = s_cur + delta_raw
                if not teacher_forced:
                    st_total = st_total + _mmse(
                        s_next / s_std_full_t, s_true_next / s_std_full_t, mj)
                if teacher_forced:
                    s_cur = s_true_next
                else:
                    tf = torch.rand(B, device=device) < tf_ratio
                    s_cur = torch.where(tf[:, None], s_true_next, s_next)
            return l1_total / horizon, st_total / horizon

        def counterfactual_rank_loss(runs, E, rows, starts, horizon):
            """Hinge-rank true latent below shuffled/mean rollout error.

            Negatives are built only from training-run factors. Their latent
            values are detached so the factor encoder is rewarded for making
            the correctly paired representation predictive rather than being
            trained to arbitrarily corrupt donor embeddings. The supervised
            one-step/state losses still anchor the true prediction.
            """
            if args.latent_rank_weight <= 0 or args.latent_source == "none":
                return torch.zeros((), device=device)

            B = starts.shape[0]
            S, U, Q = runs["S"], runs["U"], runs["Q"]
            e_true = E[rows]
            z_true = encoder(e_true) if encoder is not None else e_true

            controls = []
            if args.latent_rank_control in ("shuffled", "both"):
                # A non-zero cyclic shift guarantees a different training run
                # for every row while retaining empirical factor support.
                shift = int(torch.randint(1, E.shape[0], (), device=device).item())
                wrong_rows = (rows + shift) % E.shape[0]
                z_wrong = encoder(E[wrong_rows]) if encoder is not None else E[wrong_rows]
                controls.append(z_wrong.detach())
            if args.latent_rank_control in ("mean", "both"):
                z_all = encoder(E) if encoder is not None else E
                controls.append(z_all.mean(dim=0, keepdim=True).detach().expand(B, -1))

            def rollout_error(z):
                s_cur = S[rows, starts]
                h = belief_from_window(S, U, rows, starts) if use_belief else None
                total = torch.zeros(B, device=device)
                for j in range(horizon):
                    ticks = starts + j
                    u = U[rows, ticks]
                    q = (
                        Q[rows, ticks]
                        if args.rollout_q_mode == "recorded"
                        else Q[rows, starts]
                    )
                    s_keep_n = (s_cur[:, state_keep] - s_mean_t) / s_std_t
                    u_n = (u - u_mean_t) / u_std_t
                    if use_belief:
                        h = belief.step(torch.cat([s_keep_n, u_n], dim=-1), h)
                    cols = [s_keep_n, u_n]
                    if use_q:
                        cols.append((q - q_mean_t) / q_std_t)
                    if use_belief:
                        cols.append(h)
                    pred_norm = sd(torch.cat(cols + [z], dim=-1))
                    s_cur = s_cur + pred_norm * y_std_t + y_mean_t
                    s_true_next = S[rows, ticks + 1]
                    total = total + torch.mean(
                        ((s_cur - s_true_next) / s_std_full_t) ** 2, dim=-1
                    )
                return total / horizon

            true_error = rollout_error(z_true)
            rank_terms = [
                torch.relu(args.latent_rank_margin + true_error - rollout_error(z_control)).mean()
                for z_control in controls
            ]
            return torch.stack(rank_terms).mean()

        @torch.no_grad()
        def val_unrolled() -> float:
            if encoder is not None:
                encoder.eval()
            if belief is not None:
                belief.eval()
            sd.eval()
            start_lo = history_lag - 1 if use_belief else 0
            starts_all, rows_all = [], []
            for rid, r in val_runs.items():
                n = r["s"].shape[0] - args.val_unroll
                if n <= start_lo:
                    continue
                row = t_val["rows"][rid]
                rows_all += [row] * (n - start_lo)
                starts_all += list(range(start_lo, n))
            if not rows_all:
                return float("nan")
            rng = np.random.default_rng(0)
            pick = rng.choice(len(rows_all), size=min(1024, len(rows_all)), replace=False)
            rows = torch.tensor([rows_all[i] for i in pick], dtype=torch.long, device=device)
            starts = torch.tensor([starts_all[i] for i in pick], dtype=torch.long, device=device)
            _, v = unrolled_loss(t_val, E_val, rows, starts, 0.0, False, args.val_unroll)
            if encoder is not None:
                encoder.train()
            if belief is not None:
                belief.train()
            sd.train()
            return float(v.item())

        best_val, best_epoch, no_improve = float("inf"), -1, 0
        history, t0 = [], time.time()
        n_train = sum(r["s"].shape[0] for r in train_runs.values())
        run_list_to_row = t_tr["rows"]
        start_lo = history_lag - 1 if use_belief else 0
        for epoch in range(1, args.epochs + 1):
            progress = (epoch - 1) / max(args.epochs - 1, 1)
            tf_ratio = args.tf_start + (args.tf_floor - args.tf_start) * progress
            lambda_r = args.rollout_weight
            if args.rollout_ramp_epochs > 0:
                lambda_r = args.rollout_weight * min(1.0, epoch / args.rollout_ramp_epochs)
            if encoder is not None:
                encoder.train()
            if belief is not None:
                belief.train()
            sd.train()
            steps = max(int(n_train / args.batch_size), 8)
            tot = 0.0
            tot_l1 = 0.0
            tot_st = 0.0
            tot_rank = 0.0
            for _ in range(steps):
                ids = np.random.choice(run_list, size=args.batch_size, replace=True)
                rows = torch.as_tensor(
                    [run_list_to_row[int(r)] for r in ids], dtype=torch.long, device=device
                )
                starts = []
                for rid in ids.tolist():
                    n = train_runs[rid]["s"].shape[0]
                    hi = max(n - args.unroll - 1, start_lo + 1)
                    starts.append(int(np.random.randint(start_lo, hi)))
                starts_t = torch.tensor(starts, dtype=torch.long, device=device)
                optimizer.zero_grad(set_to_none=True)
                l1, _ = unrolled_loss(t_tr, E_tr, rows, starts_t, 0.0, False,
                                      args.unroll, teacher_forced=True)
                _, st = unrolled_loss(t_tr, E_tr, rows, starts_t, tf_ratio,
                                      args.noise_std > 0.0, args.unroll)
                rank = counterfactual_rank_loss(
                    t_tr, E_tr, rows, starts_t, args.unroll
                )
                loss = (args.l1_weight * l1
                        + lambda_r * args.state_loss_weight * st
                        + lambda_r * args.latent_rank_weight * rank
                        + factor_recon_loss(ids.tolist()))
                loss.backward()
                if args.grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
                optimizer.step()
                tot += float(loss)
                tot_l1 += float(l1)
                tot_st += float(st)
                tot_rank += float(rank)
            scheduler.step()
            train_loss = tot / max(steps, 1)
            train_l1 = tot_l1 / max(steps, 1)
            train_st = tot_st / max(steps, 1)
            train_rank = tot_rank / max(steps, 1)
            v = val_unrolled()
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_l1": train_l1,
                    "train_state": train_st,
                    "train_rank": train_rank,
                    "lambda_r": lambda_r,
                    "val_loss": v,
                    "tf": tf_ratio,
                }
            )
            if v < best_val:
                best_val, best_epoch, no_improve = v, epoch, 0
                if encoder is not None:
                    torch.save(encoder.state_dict(), out / "encoder.pt")
                    if decoder is not None:
                        torch.save(decoder.state_dict(), out / "decoder.pt")
                if belief is not None:
                    torch.save(belief.state_dict(), out / "belief.pt")
                torch.save(sd.state_dict(), out / "sd.pt")
            else:
                no_improve += 1
            if epoch % 5 == 0 or epoch == 1:
                log(
                    f"epoch {epoch:03d}/{args.epochs} train={train_loss:.5f} "
                    f"l1={train_l1:.5f} state={train_st:.5f} "
                    f"rank={train_rank:.5f} "
                    f"lambda_r={lambda_r:.3f} val(unroll{args.val_unroll})={v:.5f} "
                    f"tf={tf_ratio:.2f} elapsed={time.time()-t0:.0f}s"
                )
            if args.patience and no_improve >= args.patience:
                log(f"early stopping at epoch {epoch}")
                break
        # The unrolled trainer's early stopping uses the unrolled val loss;
        # once training is done, evaluate the BEST checkpoint one-step on the
        # held-out val/test runs so metrics.json carries the same test
        # contract as the non-unrolled path (RMSE + standard R2).
        if encoder is not None:
            encoder.load_state_dict(torch.load(out / "encoder.pt", map_location=device))
        if belief is not None:
            belief.load_state_dict(torch.load(out / "belief.pt", map_location=device))
        sd.load_state_dict(torch.load(out / "sd.pt", map_location=device))
        va, te = build(val_runs), build(test_runs)
        Wva = torch.as_tensor(va["W"].astype(np.float32), device=device) if va["W"] is not None else None
        Wte = torch.as_tensor(te["W"].astype(np.float32), device=device) if te["W"] is not None else None
        Xva = torch.as_tensor(Xn(va["X"]).astype(np.float32), device=device)
        Yva = torch.as_tensor(Yn(va["Y"]).astype(np.float32), device=device)
        Xte = torch.as_tensor(Xn(te["X"]).astype(np.float32), device=device)
        Yte = torch.as_tensor(Yn(te["Y"]).astype(np.float32), device=device)
        val = evaluate(Xva, Wva, Yva, va["run_ids"], va["Y"].astype(np.float64))
        test = evaluate(Xte, Wte, Yte, te["run_ids"], te["Y"].astype(np.float64))
    else:
        best_val, best_epoch, no_improve = float("inf"), -1, 0
        history, t0 = [], time.time()
        n = Xtr.shape[0]
        for epoch in range(1, args.epochs + 1):
            if encoder is not None:
                encoder.train()
            if belief is not None:
                belief.train()
            sd.train()
            perm = torch.randperm(n)
            tot, nb = 0.0, 0
            for start in range(0, n, args.batch_size):
                b = perm[start : start + args.batch_size]
                optimizer.zero_grad(set_to_none=True)
                Wb = Wtr[b] if Wtr is not None else None
                loss, _ = forward_loss(Xtr[b], Wb, Ytr[b], tr["run_ids"][b])
                loss = loss + factor_recon_loss(tr["run_ids"][b])
                loss.backward()
                optimizer.step()
                tot += float(loss) * b.shape[0]
                nb += b.shape[0]
            scheduler.step()
            train_loss = tot / max(nb, 1)
            val = evaluate(Xva, Wva, Yva, va["run_ids"], va["Y"].astype(np.float64))
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val["loss"]})
            if val["loss"] < best_val:
                best_val, best_epoch, no_improve = val["loss"], epoch, 0
                if encoder is not None:
                    torch.save(encoder.state_dict(), out / "encoder.pt")
                    if decoder is not None:
                        torch.save(decoder.state_dict(), out / "decoder.pt")
                if belief is not None:
                    torch.save(belief.state_dict(), out / "belief.pt")
                torch.save(sd.state_dict(), out / "sd.pt")
            else:
                no_improve += 1
            if epoch % 5 == 0 or epoch == 1:
                log(f"epoch {epoch:03d}/{args.epochs} train={train_loss:.5f} val={val['loss']:.5f} elapsed={time.time()-t0:.0f}s")
            if args.patience and no_improve >= args.patience:
                log(f"early stopping at epoch {epoch}")
                break
        if encoder is not None:
            encoder.load_state_dict(torch.load(out / "encoder.pt", map_location=device))
        if belief is not None:
            belief.load_state_dict(torch.load(out / "belief.pt", map_location=device))
        sd.load_state_dict(torch.load(out / "sd.pt", map_location=device))
        test = evaluate(Xte, Wte, Yte, te["run_ids"], te_raw)
        val = evaluate(Xva, Wva, Yva, va["run_ids"], va["Y"].astype(np.float64))

    # Per-factor reconstruction quality of the auxiliary head on held-out runs
    # (the "z provably encodes all six factors" check).
    recon_report = None
    if decoder is not None:
        if encoder is not None:
            encoder.load_state_dict(torch.load(out / "encoder.pt", map_location=device))
        decoder.load_state_dict(torch.load(out / "decoder.pt", map_location=device))
        if encoder is not None:
            encoder.eval()
        decoder.eval()
        e_hats, e_trues = [], []
        for rid in sorted(set(val_runs) | set(test_runs)):
            e_n = torch.as_tensor(e_map[rid][None, :], dtype=torch.float32, device=device)
            with torch.no_grad():
                e_hat = decoder(encoder(e_n)).detach().cpu().numpy()[0]
            e_hats.append(e_hat)
            e_trues.append(e_map[rid])
        e_hat = np.stack(e_hats)
        e_true = np.stack(e_trues)
        recon_report = {
            "weight": args.factor_recon_weight,
            "r2_per_factor": {
                k: float(1 - ((e_hat[:, j] - e_true[:, j]) ** 2).sum()
                         / ((e_true[:, j] - e_true[:, j].mean()) ** 2).sum())
                for j, k in enumerate(FACTOR_KEYS)
            },
            "rmse_per_factor": {
                k: float(np.sqrt(((e_hat[:, j] - e_true[:, j]) ** 2).mean()))
                for j, k in enumerate(FACTOR_KEYS)
            },
            "evaluated_on": "held_out_normalized",
        }

    stage_label = "21" if (args.unroll > 1 and (args.l1_weight > 0 or args.rollout_weight > 0)) else "20"
    report = {
        "stage": stage_label,
        "latent_dim": args.latent_dim,
        "latent_source": args.latent_source,
        "latent_in_dim": latent_dim,
        "factor_recon_weight": args.factor_recon_weight,
        "factor_recon": recon_report,
        "obs_drop": obs_drop,
        "state_keep": state_keep,
        "state_dim": state_dim,
        "use_q": use_q,
        "use_belief": use_belief,
        "belief_dim": args.belief_dim if use_belief else 0,
        "history_lag": history_lag,
        "state_loss_weight": args.state_loss_weight,
        "sample_mask_applied": bool(args.sample_mask),
        "rollout_q_mode": args.rollout_q_mode,
        "latent_rank_weight": args.latent_rank_weight,
        "latent_rank_margin": args.latent_rank_margin,
        "latent_rank_control": args.latent_rank_control,
        "unroll": args.unroll,
        "l1_weight": args.l1_weight,
        "rollout_weight": args.rollout_weight,
        "rollout_ramp_epochs": args.rollout_ramp_epochs,
        "val_unroll": args.val_unroll if args.unroll > 1 else None,
        "tf_floor": args.tf_floor if args.unroll > 1 else None,
        "noise_std": args.noise_std if args.unroll > 1 else None,
        "d_e": D_E,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "train_runs": len(train_runs),
        "e_normalization": {
            "mean": [float(v) for v in e_mean],
            "std": [float(v) for v in e_std],
            "computed_on": "train_domains_only",
        },
        "s_stats": {"mean": [float(v) for v in s_mean], "std": [float(v) for v in s_std]},
        "u_stats": {"mean": [float(v) for v in u_mean], "std": [float(v) for v in u_std]},
        "q_stats": {"mean": [float(v) for v in q_mean], "std": [float(v) for v in q_std]},
        "y_stats": {"mean": [float(v) for v in y_mean], "std": [float(v) for v in y_std]},
        "test": {"rmse": test["rmse"], "r2_all": test["r2_all"]},
        "val": {"rmse": val["rmse"], "loss": val["loss"]},
        "history": history,
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    if test["rmse"] is not None:
        log(
            f"latent_source={args.latent_source} latent_in={latent_dim} "
            f"best_epoch={best_epoch} "
            f"test r2_all={test['r2_all']:.4f} rmse_all={test['rmse']['all']:.4g}"
        )
    else:
        log(
            f"latent_source={args.latent_source} latent_in={latent_dim} "
            f"unroll={args.unroll} best_epoch={best_epoch} "
            f"best_val(unroll)={best_val:.5f}"
        )
    log_fh.close()


if __name__ == "__main__":
    main()
