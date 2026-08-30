"""Does the TRAINED residual actually respond to its history?

Necessary condition for the ablation to mean anything. The RMA context route
failed exactly here -- 0.004% of action scale -- so the gate that followed was
comparing identical inputs and could only return a null. Costs seconds.
"""
import torch, glob, os, sys
run = sys.argv[1] if len(sys.argv) > 1 else "residual_history"
cks = sorted(glob.glob(f"/w/logs/rsl_rl/ppo_id_aam_fd/{run}/model_*.pt"),
             key=lambda p: int(''.join(c for c in os.path.basename(p) if c.isdigit())))
if not cks:
    raise SystemExit(f"no checkpoints under {run}")
BW, BD, AD, H = 70, 66, 4, 20
torch.manual_seed(0)
base = torch.randn(256, BW)
print(f"{'checkpoint':>14} {'|dAction| vs history':>21} {'% of action':>12}")
for p in cks[::max(1, len(cks)//6)] + cks[-1:]:
    sd = torch.load(p, map_location="cpu", weights_only=False)["actor_state_dict"]
    keys = [k for k in sd if k.startswith("mlp.")]
    # ResidualMLP submodules are stored under mlp.<name>
    if not any("head" in k for k in keys):
        print(f"{os.path.basename(p):>14}   (no residual head in checkpoint)")
        continue
    sys.path.insert(0, "/w")
    from verge_task.residual_adapter import ResidualMLP
    import torch.nn as nn
    bm = nn.Sequential(
        nn.Linear(sd["mlp.base_mlp.0.weight"].shape[1], sd["mlp.base_mlp.0.weight"].shape[0]), nn.ELU(),
        nn.Linear(sd["mlp.base_mlp.2.weight"].shape[1], sd["mlp.base_mlp.2.weight"].shape[0]), nn.ELU(),
        nn.Linear(sd["mlp.base_mlp.4.weight"].shape[1], sd["mlp.base_mlp.4.weight"].shape[0]), nn.ELU(),
        nn.Linear(sd["mlp.base_mlp.6.weight"].shape[1], sd["mlp.base_mlp.6.weight"].shape[0]))
    r = ResidualMLP(bm, base_dim=BD, base_width=BW, action_dim=AD, history=H,
                    latent_dim=8, alpha=0.5)
    r.load_state_dict({k[len("mlp."):]: v for k, v in sd.items() if k.startswith("mlp.")}, strict=False)
    r.eval()
    with torch.no_grad():
        h1 = torch.randn(256, H*(BD+AD))
        h2 = torch.randn(256, H*(BD+AD))
        a1 = r(torch.cat((base, h1), -1))
        a2 = r(torch.cat((base, h2), -1))
        scale = float(a1.abs().mean())
        d = float((a1 - a2).abs().mean())
    print(f"{os.path.basename(p):>14} {d:21.6f} {100*d/scale:11.3f}%")
