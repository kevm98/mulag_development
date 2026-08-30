from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob, sys
run = sys.argv[1]
ea = EventAccumulator(sorted(glob.glob(f"/w/logs/rsl_rl/ppo_id_aam_fd/{run}/events*"))[-1]); ea.Reload()
tot = [x.value for x in ea.Scalars("Episode_Reward/total")]
lat = [x.value for x in ea.Scalars("Metrics/lateral_error_m")]
gc  = [x.value for x in ea.Scalars("Metrics/ground_contact_fraction")]
n = len(tot)
# An episode spans ~89 iterations; split into cycles and take each cycle's BEST
# phase, which is the like-for-like comparison across cycles.
P = 89
print(f"n={n} iterations, cycle={P}")
print(f"{'cycle':>6} {'range':>12} {'peak total':>11} {'lat@peak':>9} {'gc@peak':>8} {'trough tot':>11}")
for c in range(0, n, P):
    seg = list(range(c, min(c+P, n)))
    if len(seg) < 10: continue
    best = max(seg, key=lambda i: tot[i])
    worst = min(seg, key=lambda i: tot[i])
    print(f"{c//P:>6} {str((seg[0],seg[-1])):>12} {tot[best]:11.3f} {lat[best]:9.4f} {gc[best]:8.3f} {tot[worst]:11.3f}")
