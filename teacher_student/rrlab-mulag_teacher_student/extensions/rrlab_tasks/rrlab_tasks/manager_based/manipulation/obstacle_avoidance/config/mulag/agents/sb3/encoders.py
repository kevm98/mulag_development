from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


def _act(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "elu":
        return nn.ELU()
    if name == "tanh":
        return nn.Tanh()
    if name == "gelu":
        return nn.GELU()
    if name == "leakyrelu":
        return nn.LeakyReLU()
    raise ValueError(f"Unknown activation: {name}")


def _mlp(in_dim: int, layers: Sequence[int], activation: str) -> nn.Sequential:
    mods: List[nn.Module] = []
    prev = in_dim
    for h in layers:
        mods += [nn.Linear(prev, h), _act(activation)]
        prev = h
    return nn.Sequential(*mods)


class _SimpleCNN(nn.Module):
    """Configurable CNN -> flatten -> (optional MLP) -> linear(out_dim) -> optional final activation."""
    def __init__(
        self,
        in_channels: int,
        input_hw: Tuple[int, int],
        conv_channels: Sequence[int] = (32, 64, 64),
        kernels: Sequence[int] = (8, 4, 3),
        strides: Sequence[int] = (4, 2, 1),
        paddings: Optional[Sequence[int]] = None,
        activation: str = "relu",
        out_dim: int = 128,
        head_layers: Optional[Sequence[int]] = None,
        head_final_activation: Optional[str] = "tanh",
    ):
        super().__init__()
        assert len(conv_channels) == len(kernels) == len(strides)

        if paddings is None:
            paddings = [0] * len(conv_channels)
        assert len(paddings) == len(conv_channels)

        act = _act(activation)
        convs: List[nn.Module] = []
        c = in_channels
        for oc, k, s, p in zip(conv_channels, kernels, strides, paddings): 
            convs.append(nn.Conv2d(c, oc, kernel_size=k, stride=s, padding=p))
            convs.append(act.__class__())
            c = oc
        self.conv = nn.Sequential(*convs)

        # infer flatten dim
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_hw[0], input_hw[1])
            flat_dim = int(self.conv(dummy).view(1, -1).shape[1])

        head_layers = list(head_layers) if head_layers is not None else []
        # Optional MLP before final linear
        if len(head_layers) > 0:
            self.pre = nn.Sequential(nn.Flatten(), _mlp(flat_dim, head_layers, activation))
            last_dim = head_layers[-1]
        else:
            self.pre = nn.Flatten()
            last_dim = flat_dim

        self.fc = nn.Linear(last_dim, out_dim)
        self.final_act = _act(head_final_activation) if head_final_activation else nn.Identity() 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        y = self.pre(y)
        y = self.fc(y)
        y = self.final_act(y)
        return y

class MultiBranchExtractor(BaseFeaturesExtractor):
    """Config-driven extractor supporting MLP + CNN branches.

    Obs must be Dict. Each branch chooses keys and encodes them, then fuses.
    """
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        *,
        branches: Dict[str, Dict[str, Any]],
        fusion: str = "concat",
        post_fusion_layers: Optional[Sequence[int]] = None,
        post_fusion_activation: str = "elu",
    ):
        if not isinstance(observation_space, gym.spaces.Dict):
            raise TypeError(f"Expected Dict observation_space, got {type(observation_space)}")

        # ---------- PASS 1: validate + compute output dims (NO nn.Modules on self) ----------
        fusion = fusion.lower()
        branch_names = list(branches.keys())
        branch_keys: Dict[str, Tuple[str, ...]] = {}
        branch_out_dims: Dict[str, int] = {}

        # store per-branch meta we’ll need in pass 2
        branch_meta: Dict[str, Dict[str, Any]] = {}

        for bname, bcfg in branches.items():
            btype = bcfg.get("type", "mlp").lower()
            keys = tuple(bcfg["keys"])
            if len(keys) == 0:
                raise ValueError(f"Branch '{bname}' has empty keys")
            for k in keys:
                if k not in observation_space.spaces:
                    raise KeyError(
                        f"Branch '{bname}' missing key '{k}'. "
                        f"Available: {list(observation_space.spaces.keys())}"
                    )

            branch_keys[bname] = keys

            if btype == "mlp":
                in_dim = 0
                for k in keys:
                    sp = observation_space.spaces[k]
                    if not isinstance(sp, gym.spaces.Box) or len(sp.shape) != 1:
                        raise ValueError(
                            f"MLP branch expects 1D Box, key '{k}' has shape {getattr(sp, 'shape', None)}"
                        )
                    in_dim += int(sp.shape[0])

                layers = list(bcfg.get("layers", [128, 64]))
                out_dim = layers[-1] if len(layers) > 0 else in_dim
                branch_out_dims[bname] = out_dim
                branch_meta[bname] = {"type": "mlp", "in_dim": in_dim, "layers": layers,
                                      "activation": bcfg.get("activation", "elu")}

            elif btype == "cnn":
                activation = bcfg.get("activation", "relu")
                out_dim = int(bcfg.get("out_dim", 128))

                sp0 = observation_space.spaces[keys[0]]
                if not isinstance(sp0, gym.spaces.Box):
                    raise ValueError(f"CNN branch expects Box, key '{keys[0]}' got {type(sp0)}")

                channel_order = bcfg.get("channel_order", "chw").lower()
                if channel_order not in ("chw", "hwc"):
                    raise ValueError("channel_order must be 'chw' or 'hwc'")

                # NEW: allow flattened vectors
                if len(sp0.shape) == 1:
                    # flattened (D,)
                    hw = bcfg["img_height_width"]          # e.g. [21, 41]
                    h, w = int(hw[0]), int(hw[1])
                    D = int(sp0.shape[0])
                    denom = h * w
                    if D % denom != 0:
                        raise ValueError(f"Flattened image dim D={D} not divisible by H*W={denom}")
                    c0 = D // denom                        # <-- this becomes 9 when history=3 and xyz=3
                else:
                    # your old image case (3D)
                    if len(sp0.shape) != 3:
                        raise ValueError(
                            f"CNN branch expects 3D image or 1D flattened, key '{keys[0]}' has shape {sp0.shape}"
                        )
                    if channel_order == "chw":
                        c0, h, w = sp0.shape
                    else:
                        h, w, c0 = sp0.shape

                # then total_c accumulates like before
                total_c = 0
                for k in keys:
                    sp = observation_space.spaces[k]
                    if len(sp.shape) == 1:
                        Dk = int(sp.shape[0])
                        denom = h * w
                        if Dk % denom != 0:
                            raise ValueError(f"Key '{k}' D={Dk} not divisible by H*W={denom}")
                        ck = Dk // denom
                    else:
                        if channel_order == "chw":
                            ck, hk, wk = sp.shape
                        else:
                            hk, wk, ck = sp.shape
                        if hk != h or wk != w:
                            raise ValueError(f"All CNN keys must share same H,W. '{k}' has {(hk,wk)} vs {(h,w)}")
                    total_c += ck

                branch_out_dims[bname] = out_dim
                branch_meta[bname] = {
                    "type": "cnn",
                    "activation": activation,
                    "out_dim": out_dim,
                    "channel_order": channel_order,
                    "h": h,
                    "w": w,
                    "total_c": total_c,
                    "conv_channels": bcfg.get("conv_channels", [32, 64, 64]),
                    "kernels": bcfg.get("kernels", [8, 4, 3]),
                    "strides": bcfg.get("strides", [4, 2, 1]),
                    "paddings": bcfg.get("paddings", [2, 1, 0]),
                    "head_layers": bcfg.get("head_layers", []),
                    "head_final_activation": bcfg.get("head_final_activation", "tanh"),
                }
            else:
                raise ValueError(f"Unknown branch type '{btype}' for branch '{bname}'")

        # fusion dim
        if fusion == "concat":
            fused_dim = sum(branch_out_dims.values())
        elif fusion in ("sum", "mean"):
            dims = set(branch_out_dims.values())
            if len(dims) != 1:
                raise ValueError(f"fusion='{fusion}' requires equal branch dims, got {branch_out_dims}")
            fused_dim = next(iter(dims))
        else:
            raise ValueError(f"Unknown fusion '{fusion}'")

        final_dim = fused_dim
        if post_fusion_layers is not None and len(post_fusion_layers) > 0:
            final_dim = int(list(post_fusion_layers)[-1])

        # ---------- IMPORTANT: initialize nn.Module/BaseFeaturesExtractor NOW ----------
        super().__init__(observation_space, features_dim=final_dim)

        # ---------- PASS 2: now it’s safe to create submodules ----------
        self.obs_space = observation_space
        self.branch_names = branch_names
        self.fusion = fusion
        self.branch_keys = branch_keys
        self._branches_cfg = branches  # keep your original config if you want

        self.encoders = nn.ModuleDict()
        for bname in self.branch_names:
            meta = branch_meta[bname]
            if meta["type"] == "mlp":
                self.encoders[bname] = _mlp(meta["in_dim"], meta["layers"], meta["activation"])
            else:
                self.encoders[bname] = _SimpleCNN(
                    in_channels=meta["total_c"],
                    input_hw=(meta["h"], meta["w"]),
                    conv_channels=meta["conv_channels"],
                    kernels=meta["kernels"],
                    strides=meta["strides"],
                    paddings=meta["paddings"],
                    activation=meta["activation"],
                    out_dim=meta["out_dim"],
                    head_layers=meta["head_layers"],
                    head_final_activation=meta["head_final_activation"],
                )
                # forward() uses these
                branches[bname]["_channel_order"] = meta["channel_order"]

        self.post_fusion = None
        if post_fusion_layers is not None and len(post_fusion_layers) > 0:
            self.post_fusion = _mlp(fused_dim, post_fusion_layers, post_fusion_activation)

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        feats: List[torch.Tensor] = []

        # 0) Sanity-check raw observations
        for k, v in obs.items():
            if torch.isnan(v).any() or torch.isinf(v).any():
                raise RuntimeError(f"NaN/Inf in obs[{k}] with shape {v.shape}")

        for bname in self.branch_names:
            bcfg = self._branches_cfg[bname]
            btype = bcfg.get("type", "mlp").lower()
            keys = self.branch_keys[bname]

            if btype == "mlp":
                # --- build input with per-key checks ---
                xs = []
                for k in keys:
                    v = obs[k]

                    # (A) NaN/Inf check per key (more informative than only checking the concatenated tensor)
                    if torch.isnan(v).any() or torch.isinf(v).any():
                        raise RuntimeError(f"NaN/Inf in obs[{k}] (branch='{bname}'), shape={v.shape}")

                    # (B) Magnitude check (finite but exploding values can still create NaNs after Linear/ELU)
                    vmax = v.abs().max()
                    if vmax > 1e4:  # adjust threshold if needed
                        raise RuntimeError(
                            f"Exploding magnitude in obs[{k}] (branch='{bname}'): "
                            f"max|v|={vmax.item():.3e}, shape={v.shape}"
                        )

                    # (Optional) Quick safety clamp if you prefer to continue instead of crashing:
                    # v = torch.clamp(v, -1e3, 1e3)

                    xs.append(v)

                x = torch.cat(xs, dim=1)

                # Check concatenated input too
                if torch.isnan(x).any() or torch.isinf(x).any():
                    raise RuntimeError(f"NaN/Inf in MLP input after cat (branch='{bname}'), shape={x.shape}")

                if bname == "state":
                    for n, p in self.encoders[bname].named_parameters():
                        if torch.isnan(p).any() or torch.isinf(p).any():
                            raise RuntimeError(f"NaN/Inf in state MLP parameter: {n}")


                # --- encode ---
                out = self.encoders[bname](x)

                if torch.isnan(out).any() or torch.isinf(out).any():
                    raise RuntimeError(f"NaN/Inf in branch '{bname}' output (mlp), shape={out.shape}")

                feats.append(out)


            elif btype == "cnn":
                channel_order = bcfg["_channel_order"]

                # Needed when IsaacLab flattens history/image to (N, D)
                hw = bcfg.get("img_height_width", None)
                if hw is None:
                    raise ValueError(f"CNN branch '{bname}' needs img_height_width when input is flattened.")
                H, W = int(hw[0]), int(hw[1])

                imgs = []
                for k in keys:
                    x = obs[k]

                    # Case 1: flattened (N, D) -> reshape to (N, C, H, W)
                    if x.ndim == 2:
                        N, D = x.shape
                        denom = H * W
                        if D % denom != 0:
                            raise RuntimeError(
                                f"Cannot reshape '{k}': D={D} not divisible by H*W={denom} (H={H}, W={W})"
                            )
                        C = D // denom
                        x = x.view(N, C, H, W)

                    # Case 2: already image-like
                    elif x.ndim == 4:
                        if channel_order == "hwc":
                            x = x.permute(0, 3, 1, 2)

                    else:
                        raise RuntimeError(
                            f"CNN obs '{k}' must be (N,D) or (N,C,H,W)/(N,H,W,C), got {x.shape}"
                        )

                    if torch.isnan(x).any() or torch.isinf(x).any():
                        raise RuntimeError(f"NaN/Inf in CNN key '{k}' after reshape/permute, shape={x.shape}")

                    imgs.append(x)

                x_img = torch.cat(imgs, dim=1)  # concat channels
                if torch.isnan(x_img).any() or torch.isinf(x_img).any():
                    raise RuntimeError(f"NaN/Inf in CNN input (branch='{bname}'), shape={x_img.shape}")

                out = self.encoders[bname](x_img)
                if torch.isnan(out).any() or torch.isinf(out).any():
                    raise RuntimeError(f"NaN/Inf in branch '{bname}' output (cnn), shape={out.shape}")

                feats.append(out)

            else:
                raise RuntimeError(f"Unexpected btype {btype}")

        # 1) Fuse branch features
        if self.fusion == "concat":
            y = torch.cat(feats, dim=1)
        elif self.fusion == "sum":
            y = torch.stack(feats, dim=0).sum(dim=0)
        else:
            y = torch.stack(feats, dim=0).mean(dim=0)

        if torch.isnan(y).any() or torch.isinf(y).any():
            raise RuntimeError("NaN/Inf right after fusion (before post_fusion)")

        # 2) Optional post-fusion MLP
        if self.post_fusion is not None:
            y = self.post_fusion(y)
            if torch.isnan(y).any() or torch.isinf(y).any():
                raise RuntimeError("NaN/Inf after post_fusion")

        # 3) Final sanity check (keep if you want)
        if torch.isnan(y).any() or torch.isinf(y).any():
            raise RuntimeError("NaN/Inf in extracted features")

        return y

