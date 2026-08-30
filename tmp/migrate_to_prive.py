"""Warm-start a 72-wide privileged-e policy from the validated 68-wide checkpoint.

The 68-wide policy is [66 base | 2 frozen-encoder latent]; the privileged-e
policy is [66 base | 6 raw factors]. Appending 4 columns is not enough: the two
EXISTING latent columns would keep weights trained to read a 2-D code and would
now receive flow/compliance instead. So all six context columns are zeroed,
which is the RLA protocol's Stage-1 warm start (§7.3) -- the migrated actor
reproduces the validated policy exactly for any context value, and learns to
use e from a working baseline instead of from scratch.

Training from scratch was tried first and produced a policy with 675 mm median
lateral error against the baseline's 34 mm; the oracle comparison on it was
meaningless because every arm failed equally.
"""
import copy, json, sys, torch
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
OLD, NEW, BASE = 68, 72, 66
payload = torch.load(src, map_location="cpu", weights_only=False)
out = copy.deepcopy(payload)
for group in ("actor_state_dict", "critic_state_dict"):
    st = {k: (v.clone() if torch.is_tensor(v) else copy.deepcopy(v))
          for k, v in payload[group].items()}
    w = st["mlp.0.weight"]
    assert w.shape[1] == OLD, f"{group} mlp.0.weight width {w.shape[1]} != {OLD}"
    exp = torch.zeros((w.shape[0], NEW), dtype=w.dtype)
    exp[:, :BASE] = w[:, :BASE]          # keep base observation weights
    # columns BASE..NEW stay ZERO: both the old 2-D latent weights and the new
    # channels. Context therefore contributes nothing at init.
    st["mlp.0.weight"] = exp
    for key, fill in (("obs_normalizer._mean", 0.0),
                      ("obs_normalizer._std", 1.0),
                      ("obs_normalizer._var", 1.0)):
        old = st[key]
        assert tuple(old.shape) == (1, OLD), f"{key} shape {old.shape}"
        st[key] = torch.cat((old[:, :BASE],
                             torch.full((1, NEW - BASE), fill, dtype=old.dtype)), dim=1)
    out[group] = st
out["optimizer_state_dict"] = {}
infos = dict(payload.get("infos") or {})
infos.update({
    "observation_width": NEW,
    "initialization_source": "validated 68-D ID-AAM-FD policy; all 6 context columns zeroed",
    "rma_context_names": ["flow", "compliance", "seal", "pressure", "mass_scale", "com_z"],
    "rma_context_normalized": True,
    "migration_requires_optimizer_reset": True,
    "optimizer_reset_on_resume": True,
})
out["infos"] = infos
dst.parent.mkdir(parents=True, exist_ok=True)
torch.save(out, dst)
print(json.dumps({"output": str(dst), "observation_width": NEW}, indent=2))
