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
    """Simple configurable CNN -> flatten -> MLP head to get fixed feature dim."""
    def __init__(
        self,
        in_channels: int,
        input_hw: Tuple[int, int],
        conv_channels: Sequence[int] = (32, 64, 64),
        kernels: Sequence[int] = (8, 4, 3),
        strides: Sequence[int] = (4, 2, 1),
        activation: str = "relu",
        out_dim: int = 128,
        head_layers: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        assert len(conv_channels) == len(kernels) == len(strides)

        act = _act(activation)
        convs: List[nn.Module] = []
        c = in_channels
        for oc, k, s in zip(conv_channels, kernels, strides):
            convs.append(nn.Conv2d(c, oc, kernel_size=k, stride=s))
            convs.append(act.__class__())
            c = oc
        self.conv = nn.Sequential(*convs)

        # infer flatten dim
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_hw[0], input_hw[1])
            flat_dim = int(self.conv(dummy).view(1, -1).shape[1])

        head_layers = list(head_layers) if head_layers is not None else []
        self.head = nn.Sequential(
            nn.Flatten(),
            _mlp(flat_dim, [*head_layers, out_dim], activation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(x))


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
                if not isinstance(sp0, gym.spaces.Box) or len(sp0.shape) != 3:
                    raise ValueError(
                        f"CNN branch expects 3D Box image, key '{keys[0]}' has shape {getattr(sp0, 'shape', None)}"
                    )

                channel_order = bcfg.get("channel_order", "chw").lower()
                if channel_order not in ("chw", "hwc"):
                    raise ValueError("channel_order must be 'chw' or 'hwc'")

                if channel_order == "chw":
                    c0, h, w = sp0.shape
                else:
                    h, w, c0 = sp0.shape

                total_c = 0
                for k in keys:
                    sp = observation_space.spaces[k]
                    if len(sp.shape) != 3:
                        raise ValueError(f"CNN branch key '{k}' must be image (3D), got {sp.shape}")
                    if channel_order == "chw":
                        ck, hk, wk = sp.shape
                    else:
                        hk, wk, ck = sp.shape
                    if hk != h or wk != w:
                        raise ValueError(
                            f"All CNN keys must share same H,W. '{k}' has {(hk, wk)} vs {(h, w)}"
                        )
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
                    "head_layers": bcfg.get("head_layers", []),
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
                    activation=meta["activation"],
                    out_dim=meta["out_dim"],
                    head_layers=meta["head_layers"],
                )
                # forward() uses these
                branches[bname]["_channel_order"] = meta["channel_order"]

        self.post_fusion = None
        if post_fusion_layers is not None and len(post_fusion_layers) > 0:
            self.post_fusion = _mlp(fused_dim, post_fusion_layers, post_fusion_activation)

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        feats: List[torch.Tensor] = []

        for bname in self.branch_names:
            bcfg = self._branches_cfg[bname]
            btype = bcfg.get("type", "mlp").lower()
            keys = self.branch_keys[bname]

            if btype == "mlp":
                x = torch.cat([obs[k] for k in keys], dim=1)
                feats.append(self.encoders[bname](x))

            elif btype == "cnn":
                channel_order = bcfg["_channel_order"]

                imgs = []
                for k in keys:
                    x = obs[k]
                    # convert HWC -> CHW
                    if channel_order == "hwc":
                        x = x.permute(0, 3, 1, 2)
                    imgs.append(x)

                x_img = torch.cat(imgs, dim=1)  # concat channels
                feats.append(self.encoders[bname](x_img))

            else:
                raise RuntimeError(f"Unexpected btype {btype}")

        if self.fusion == "concat":
            y = torch.cat(feats, dim=1)
        elif self.fusion == "sum":
            y = torch.stack(feats, dim=0).sum(dim=0)
        else:
            y = torch.stack(feats, dim=0).mean(dim=0)

        if self.post_fusion is not None:
            y = self.post_fusion(y)
        return y
