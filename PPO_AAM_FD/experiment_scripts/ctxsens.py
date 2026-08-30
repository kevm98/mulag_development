"""Does the Phase 1 actor actually DEPEND on its context columns?

The gate is a task-level test and can only see a difference if the policy's
ACTIONS change with the context. This is the direct, upstream check: perturb the
four context columns and measure how much the action moves. If this is ~0 the
gate could never have shown anything, and the ablation was measuring nothing.
"""
import torch, glob

def fwd(sd, obs):
    x = (obs - sd["obs_normalizer._mean"]) / sd["obs_normalizer._std"].clamp_min(1e-6)
    for i in (0, 2, 4):
        x = torch.nn.functional.elu(torch.nn.functional.linear(x, sd[f"mlp.{i}.weight"], sd[f"mlp.{i}.bias"]))
    return torch.nn.functional.linear(x, sd["mlp.6.weight"], sd["mlp.6.bias"])

paths = [
    ("phase1_init (distilled, zeroed)", "/w/logs/rsl_rl/ppo_id_aam_fd/distilled_v1/phase1_init_70.pt"),
    ("phase1 model_399 (trained)",      "/w/logs/rsl_rl/ppo_id_aam_fd/2026-08-29_15-20-58/model_399.pt"),
]
torch.manual_seed(0)
base = torch.randn(512, 66)
for tag, p in paths:
    try:
        sd = torch.load(p, map_location="cpu", weights_only=False)["actor_state_dict"]
    except Exception as e:
        print(f"{tag}: ERR {e}"); continue
    w = sd["mlp.0.weight"]
    ctxw = float(w[:, -4:].abs().max())
    basew = float(w[:, :-4].abs().max())
    a0 = fwd(sd, torch.cat((base, torch.zeros(512, 4)), -1))
    rows = []
    for scale in (0.5, 1.0, 2.0):
        d = []
        for _ in range(8):
            c = torch.randn(512, 4) * scale
            d.append(float((fwd(sd, torch.cat((base, c), -1)) - a0).abs().mean()))
        rows.append((scale, sum(d)/len(d)))
    act_scale = float(a0.abs().mean())
    print(f"\n{tag}")
    print(f"   context weight abs-max {ctxw:.6f}   (base columns {basew:.4f})")
    print(f"   mean |action| at zero context: {act_scale:.5f}")
    for s, dd in rows:
        pct = 100.0 * dd / act_scale if act_scale else float('nan')
        print(f"   context ~N(0,{s}): mean |delta action| = {dd:.6f}  ({pct:.2f}% of action scale)")
