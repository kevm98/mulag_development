"""Pre-registered read of the anchor runs: does the actor DEPEND on context?

Bar to clear: the naive anchor (KL against a context-blind reference on the SAME
observation) moved action sensitivity only to 0.004% of action scale, even with a
1000x gradient multiplier on the context columns.
"""
import torch

def fwd(sd, obs):
    x = (obs - sd["obs_normalizer._mean"]) / sd["obs_normalizer._std"].clamp_min(1e-6)
    for i in (0, 2, 4):
        x = torch.nn.functional.elu(torch.nn.functional.linear(x, sd[f"mlp.{i}.weight"], sd[f"mlp.{i}.bias"]))
    return torch.nn.functional.linear(x, sd["mlp.6.weight"], sd["mlp.6.bias"])

R = "/workspace/projects/rrlab_mulag/logs/rsl_rl/ppo_id_aam_fd"
paths = [
    ("phase1_init (distilled, context zeroed)", f"{R}/distilled_v1/phase1_init_70.pt"),
    ("naive anchor  2026-08-29_15-20-58/399",   f"{R}/2026-08-29_15-20-58/model_399.pt"),
    ("anchor_nominal/399",                      f"{R}/anchor_nominal/model_399.pt"),
    ("anchor_livectx/399  (the retry)",         f"{R}/anchor_livectx/model_399.pt"),
]
torch.manual_seed(0)
for tag, p in paths:
    try:
        sd = torch.load(p, map_location="cpu", weights_only=False)["actor_state_dict"]
    except Exception as e:
        print(f"\n{tag}: ERR {e}"); continue
    w = sd["mlp.0.weight"]
    nin = w.shape[1]
    base = torch.randn(512, nin - 4)
    ctxw, basew = float(w[:, -4:].abs().max()), float(w[:, :-4].abs().max())
    a0 = fwd(sd, torch.cat((base, torch.zeros(512, 4)), -1))
    act = float(a0.abs().mean())
    print(f"\n{tag}   obs_dim={nin}")
    print(f"   context weight abs-max {ctxw:.6f}   (base columns {basew:.4f})")
    print(f"   mean |action| at zero context: {act:.5f}")
    for s in (0.5, 1.0, 2.0):
        d = [float((fwd(sd, torch.cat((base, torch.randn(512, 4) * s), -1)) - a0).abs().mean()) for _ in range(8)]
        dd = sum(d) / len(d)
        print(f"   context ~N(0,{s}): mean |delta action| = {dd:.6f}  ({100.0*dd/act:.3f}% of action scale)")
