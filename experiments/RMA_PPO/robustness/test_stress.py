"""Nominal-invariance and positive-control tests for every stress axis.

Section 7: a stress mechanism is only admissible if switching it off restores
the original plant NUMERICALLY.
Section 8: increasing severity must demonstrably change realized dynamics, and
requested stress must equal delivered stress.
"""
import math, sys
import torch

sys.path.insert(0, "/Volumes/DATA/rrlab_mulag")
from PPO_ID_AAM_FD.rma_adaptation.core import ActuationConfig, DeploymentActuation
from stress import Stress, StressedActuation

DT = 0.05
results = {}
def check(name, ok, detail):
    results[name] = {"pass": bool(ok), "detail": detail}
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

torch.manual_seed(0)
seq = [torch.randn(6, 4) for _ in range(60)]

# --- 7.1 nominal invariance: the whole chain is the identity ---------------
act = StressedActuation(dt_s=DT); act.reset(6)
worst = max(float((act.step(x, Stress()) - x).abs().max()) for x in seq)
check("nominal_identity", worst == 0.0, f"max|out-in| over 60 steps = {worst:.3e}")

# --- 7.2 each new axis, switched off, is the identity ----------------------
for label, st in (("deadband_off", Stress(deadband=0.0)),
                  ("rate_limit_off", Stress(rate_limit=float("inf")))):
    a = StressedActuation(dt_s=DT); a.reset(6)
    w = max(float((a.step(x, st) - x).abs().max()) for x in seq)
    check(f"nominal_invariance_{label}", w == 0.0, f"max|out-in| = {w:.3e}")

# --- 7.3 authority/lag/delay reproduce the ORIGINAL wrapper exactly ---------
for alpha, tau, delay in ((1.0,0.0,0),(0.6,0.0,0),(1.0,0.12,0),(1.0,0.0,3),(0.5,0.08,2)):
    new = StressedActuation(dt_s=DT, max_delay_steps=8); new.reset(6)
    old = DeploymentActuation(ActuationConfig(dt_s=DT, action_dim=4, max_delay_steps=8)); old.reset(6)
    ctx = torch.tensor([alpha, tau, float(delay), 1.0])
    st = Stress(authority=alpha, lag_tau_s=tau, delay_steps=delay)
    w = max(float((new.step(x, st) - old.step(x, ctx)).abs().max()) for x in seq)
    # Held to float32 epsilon rather than exact equality.  The ORIGINAL
    # wrapper writes the tau = 0 bypass as the incremental update with
    # beta = 1, which is not exact in float32 once the lag state is nonzero;
    # this chain assigns through instead.  See original_nominal_exactness.
    check(f"matches_original_a{alpha}_t{tau}_d{delay}", w <= 1.0e-6, f"max diff = {w:.3e} (tol 1e-06)")

# --- 7.4 the ORIGINAL wrapper is not exactly the identity at nominal -------
# Its published check used a single step from a zeroed lag state, where the
# beta = 1 update happens to be exact.  Over a sequence it is not.  This is
# recorded, not worked around: the new chain is exact (nominal_identity above).
old = DeploymentActuation(ActuationConfig(dt_s=DT, action_dim=4, max_delay_steps=8)); old.reset(6)
ctx = torch.tensor([1.0, 0.0, 0.0, 1.0])
w_old = max(float((old.step(x, ctx) - x).abs().max()) for x in seq)
check("original_nominal_exactness", w_old > 0.0,
      f"original wrapper drifts {w_old:.3e} from the identity over 60 steps (new chain: 0.0)")

# --- 8.1 delay: requested == delivered FIFO lag -----------------------------
for d in (0, 2, 6, 12, 24, 40):
    a = StressedActuation(dt_s=DT, max_delay_steps=40); a.reset(1)
    out = [float(a.step(torch.full((1,4), float(i+1)), Stress(delay_steps=d))[0,0]) for i in range(60)]
    # input at step i is i+1; a delay of d means out[i] == i+1-d (0 before that)
    delivered = all(out[i] == (0.0 if i < d else float(i+1-d)) for i in range(60))
    check(f"delay_delivered_{d}", delivered, f"requested {d}, first nonzero at index {next(i for i,v in enumerate(out) if v>0)}")

# --- 8.2 authority: ||applied|| / ||raw|| == alpha ---------------------------
for alpha in (1.0, 0.5, 0.1, 0.01):
    a = StressedActuation(dt_s=DT); a.reset(6)
    num = den = 0.0
    for x in seq:
        y = a.step(x, Stress(authority=alpha)); num += float(y.norm())**2; den += float(x.norm())**2
    ratio = math.sqrt(num/den)
    check(f"authority_ratio_{alpha}", abs(ratio-alpha) < 1e-6, f"delivered {ratio:.7f} vs requested {alpha}")

# --- 8.3 lag: measured step response equals 1-exp(-dt/tau) ------------------
for tau in (0.0, 0.05, 0.15, 0.40, 1.0):
    a = StressedActuation(dt_s=DT); a.reset(1)
    first = float(a.step(torch.ones(1,4), Stress(lag_tau_s=tau))[0,0])
    want = 1.0 if tau <= 0 else 1.0 - math.exp(-DT/tau)
    check(f"lag_step_tau_{tau}", abs(first-want) < 1e-6, f"first step {first:.6f} vs analytic {want:.6f}")

# --- 8.4 rate limit: realized |du| never exceeds the configured maximum -----
for r in (2.0, 0.5, 0.1, 0.02):
    a = StressedActuation(dt_s=DT); a.reset(6)
    prev = torch.zeros(6,4); worst = 0.0
    for x in seq:
        y = a.step(x, Stress(rate_limit=r)); worst = max(worst, float((y-prev).abs().max())); prev = y
    check(f"rate_limit_{r}", worst <= r + 1e-6, f"max realized |du| = {worst:.6f} vs limit {r}")

# --- 8.5 deadband: sub-threshold commands are suppressed --------------------
for d in (0.05, 0.2, 0.5):
    a = StressedActuation(dt_s=DT); a.reset(1)
    below = float(a.step(torch.full((1,4), d*0.9), Stress(deadband=d))[0,0])
    a.reset(1)
    above = float(a.step(torch.full((1,4), d+0.3), Stress(deadband=d))[0,0])
    check(f"deadband_{d}", below == 0.0 and abs(above-0.3) < 1e-6,
          f"below-threshold -> {below:.3f}; above-threshold {d+0.3:.3f} -> {above:.6f}")

failed = [k for k,v in results.items() if not v["pass"]]
print()
print(f"{len(results)-len(failed)}/{len(results)} checks passed")
if failed:
    print("FAILED:", failed); sys.exit(1)
