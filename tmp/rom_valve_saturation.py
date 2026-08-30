"""Is the command-ROM's predicted valve saturated the way the frozen ID's is?

This decides whether the ROM can serve as the delta-extension -> valve map in
the PPO -> FD -> plant architecture.

The frozen Simscape ID emits a valve at its +-1 stop on 73-86% of ticks
(base_at_limit, measured across every policy/action_scale/lead-cap/reward
tested), and Stage 28 recorded ~89% preclip saturation. A bounded adaptation
residual added to a command already at its stop is clipped away, which is why
residual_dead sits at ~0.35 and why the privileged-context gain is only ~3%.

The ROM consumes a desired cylinder command and predicts the valve internally,
and its predicted valve carries all four hydraulic factors (52-122% swing,
measured). If it is ALSO saturated, it cannot open the adaptation channel and
the problem is the demanded motion, not the converter. If it is not, it is the
drop-in replacement for the ID.

Pure PyTorch, no Isaac.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rom_model"))
sys.path.insert(0, str(ROOT / "AAM_module"))
sys.path.insert(0, str(ROOT / "AAM_module/tools"))

import rom_privileged_ceiling as R  # noqa: E402
from ROM_pipeline.models import load_checkpoint  # noqa: E402

dev = torch.device("cpu")
model, _ = load_checkpoint(str(R.DEFAULT_CHECKPOINT), device=dev)
model.eval()
for p in model.parameters():
    p.requires_grad_(False)

TICKS = 200
N = 32
raw, _ = R.sample_factors(N, 20260829, dev)
sf = R.to_system_factors(raw)
ref = R.make_reference(N, TICKS, 20260830, dev)

ext, vel, pre, pv = R.initial_state(N, model, dev)
cmd_lim = (3.0 * R.COMMAND_STD).reshape(1, 4)
cmd = torch.zeros((N, 8))
hist = R.build_history(model, ext, vel, pre, pv, cmd)

at_limit = 0
elements = 0
absvals = []
with torch.no_grad():
    for t in range(TICKS):
        err = ref[t + 1] - ext
        dx = torch.clamp(2.4 * err, -cmd_lim, cmd_lim)
        cmd = torch.cat((dx, dx / R.DT), dim=-1)
        hist = torch.cat((hist[:, 1:], hist[:, -1:].clone()), dim=1)
        hist[:, -1, model.schema.extension_slice] = ext
        hist[:, -1, model.schema.velocity_slice] = vel
        hist[:, -1, model.schema.pressure_slice] = pre
        hist[:, -1, model.schema.previous_valve_slice] = pv
        hist[:, -1, model.schema.command_slice] = cmd
        out = model(hist, cmd, system_factors=sf)
        v = out.predicted_valve
        at_limit += (v.abs() >= 1.0 - 1e-6).sum().item()
        elements += v.numel()
        absvals.append(v.abs())
        ext, vel = out.next_extension, out.next_velocity
        pre, pv = out.next_pressure, out.predicted_valve

a = torch.cat(absvals)
q = torch.quantile(a, torch.tensor([0.5, 0.9, 0.99]))
print("ROM predicted valve, %d ticks x %d envs" % (TICKS, N))
print("  at |v|>=1 (saturated) : %.4f   <-- frozen ID is 0.73-0.86" % (at_limit / elements))
print("  mean |v|              : %.4f" % a.mean().item())
print("  median |v|            : %.4f" % q[0].item())
print("  p90 |v|               : %.4f" % q[1].item())
print("  p99 |v|               : %.4f" % q[2].item())
print("  frac |v|>0.95         : %.4f" % (a > 0.95).float().mean().item())
