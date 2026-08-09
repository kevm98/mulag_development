#!/home/rrlab/miniconda3/envs/env_isaaclab/bin/python
"""How is the PPO3_1 run going? One command, no simulator, no GPU.

    PPO_ROM/scripts/training_status.py            # latest run
    PPO_ROM/scripts/training_status.py --plot     # + PNG charts
    PPO_ROM/scripts/training_status.py --run <dir>

Reads the TensorBoard event file the trainer already writes, so it costs
nothing and can run while training continues.

WHAT IT IS LOOKING FOR
======================
This run exists to answer one question: does a critic warm-up stop PPO from
descending its own reward? So the report leads with the three signals that
answer it, each with an explicit verdict rather than a number to interpret:

1. **Value loss falls during the frozen window.** The measured root cause of
   every previous descent was a cold critic: value loss stuck at 6k-26k leaves
   the advantage estimate noise-dominated, so the policy random-walks away from
   the distilled student. If value loss is still at its starting magnitude when
   the actor unfreezes, the warm-up did not do its job and the fine-tune that
   follows is running on noise.

2. **Exploration std stays near 0.10.** The entropy probe measured lateral
   error growing 0.06 -> 0.59 m as std inflated toward 0.31. The run clamps std
   to [0.03, 0.15]; this check confirms the clamp is holding and that the
   policy is not drifting to the ceiling.

3. **Tracking does not regress after unfreeze.** `Metrics/lateral_error_m` is
   the tracking objective. The descent signature is this metric rising steadily
   after the actor unfreezes. That is the thing to kill a run over.

Gate columns are the acceptance contract in PPO_ROM/gates.py, but
note the honest caveat: TensorBoard metrics are TRAINING-terrain means, while
acceptance is a WORST-FAMILY, WORST-MESH number on held-out meshes. A training
mean that clears a gate is necessary, not sufficient -- confirm with
scripts/evaluate_gates.py before believing any gate result.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_LOG_ROOT = ROOT / "logs/rsl_rl/ppo3_1_sole"

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--run", default="", help="Run directory (default: most recent).")
p.add_argument("--plot", action="store_true", help="Also write status.png charts.")
p.add_argument("--tail", type=int, default=12, help="Rows in the history table.")
a = p.parse_args()


# Acceptance thresholds, imported so this tool can never disagree with the
# contract. gates.py has no simulator dependency.
sys.path.insert(0, str(ROOT / "PPO_ROM"))
import importlib.util

_spec = importlib.util.spec_from_file_location("ppo31_gates", ROOT / "PPO_ROM/gates.py")
_gates_mod = importlib.util.module_from_spec(_spec)
sys.modules["ppo31_gates"] = _gates_mod
_spec.loader.exec_module(_gates_mod)
GATES = _gates_mod.GATES


def find_run(explicit: str) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_dir():
            raise SystemExit(f"not a directory: {path}")
        return path
    candidates = [
        Path(d) for d in glob.glob(str(DEFAULT_LOG_ROOT / "*")) if os.path.isdir(d)
    ]
    candidates = [c for c in candidates if glob.glob(str(c / "events.out.tfevents*"))]
    if not candidates:
        raise SystemExit(f"no runs with event files under {DEFAULT_LOG_ROOT}")
    return max(candidates, key=lambda c: c.stat().st_mtime)


def load_scalars(run_dir: Path) -> dict[str, list[tuple[int, float]]]:
    from tensorboard.backend.event_processing import event_accumulator

    series: dict[str, list[tuple[int, float]]] = {}
    for event_file in sorted(glob.glob(str(run_dir / "events.out.tfevents*"))):
        acc = event_accumulator.EventAccumulator(
            event_file, size_guidance={"scalars": 0}
        )
        acc.Reload()
        for tag in acc.Tags()["scalars"]:
            series.setdefault(tag, []).extend(
                (int(e.step), float(e.value)) for e in acc.Scalars(tag)
            )
    for tag in series:
        series[tag].sort(key=lambda kv: kv[0])
    return series


SPARK = "▁▂▃▄▅▆▇█"


def spark(values: list[float], width: int = 40) -> str:
    """Unicode sparkline. Works over SSH, needs no display."""
    if not values:
        return ""
    if len(values) > width:
        # Average within buckets rather than sampling, so a spike cannot be
        # skipped over by the stride.
        bucket = len(values) / width
        values = [
            sum(values[int(i * bucket) : max(int((i + 1) * bucket), int(i * bucket) + 1)])
            / max(len(values[int(i * bucket) : max(int((i + 1) * bucket), int(i * bucket) + 1)]), 1)
            for i in range(width)
        ]
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return SPARK[0] * len(values)
    return "".join(SPARK[min(int((v - lo) / (hi - lo) * (len(SPARK) - 1)), len(SPARK) - 1)] for v in values)


def trend(values: list[float], frac: float = 0.25) -> tuple[float, float]:
    """(mean of first frac, mean of last frac). The shape that matters."""
    if not values:
        return (float("nan"), float("nan"))
    n = max(int(len(values) * frac), 1)
    return (sum(values[:n]) / n, sum(values[-n:]) / n)


def fmt(value: float, places: int = 4) -> str:
    if value != value:
        return "   n/a"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:.{places}f}"


def main() -> None:
    run_dir = find_run(a.run)
    manifest = {}
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
    freeze = int(manifest.get("freeze_actor_iterations", 0))

    series = load_scalars(run_dir)
    if not series:
        raise SystemExit(f"no scalars logged yet in {run_dir}")

    def vals(tag: str) -> list[float]:
        return [v for _, v in series.get(tag, [])]

    def steps(tag: str) -> list[int]:
        return [s for s, _ in series.get(tag, [])]

    # rsl_rl also writes "<tag>/time" series indexed by WALL-CLOCK SECONDS, not
    # by iteration. Taking a max over every tag therefore reads the elapsed
    # second count as an iteration number -- at 83 iterations it reported 1041,
    # which made the report claim fine-tuning was 891 iterations underway while
    # the actor was still frozen. Derive the axis from an iteration-indexed tag.
    iteration_axis = next(
        (t for t in ("Loss/value", "Policy/mean_std", "Episode_Reward/total") if series.get(t)),
        None,
    )
    if iteration_axis is not None:
        last_step = steps(iteration_axis)[-1]
    else:
        last_step = max(
            (s[-1] for s in (steps(t) for t in series if not t.endswith("/time")) if s),
            default=0,
        )

    print("=" * 78)
    print(f"PPO3_1 TRAINING STATUS   {run_dir.name}")
    print("=" * 78)
    print(f"  run dir          {run_dir}")
    if manifest:
        print(
            f"  regime           lr={manifest.get('learning_rate')} "
            f"kl={manifest.get('desired_kl')} entropy={manifest.get('entropy_coef')} "
            f"std_clamp={manifest.get('std_clamp')}"
        )
        print(
            f"  warm-up          {freeze} frozen-actor iterations "
            f"| envs={manifest.get('num_envs')} | seed={manifest.get('seed')}"
        )
        print(f"  resumed from     {manifest.get('initialization_source')}")
    running = "yes" if os.system("pgrep -f 'PPO_ROM/scripts/train.py' >/dev/null 2>&1") == 0 else "no"
    print(f"  iterations       {last_step}   (process still running: {running})")

    # ---------------------------------------------------------------- verdicts
    print("\n" + "-" * 78)
    print("THE THREE SIGNALS THIS RUN EXISTS TO PRODUCE")
    print("-" * 78)

    # A verdict needs enough points to be a trend rather than two samples.
    # Below this the honest answer is "too early", not PASS or FAIL: an early
    # false FAIL on tracking would argue for killing a perfectly healthy run.
    MIN_SAMPLES = 20
    unfrozen = max(0, last_step - freeze)
    phase = (
        f"WARM-UP in progress ({last_step}/{freeze} frozen iterations)"
        if last_step < freeze
        else f"FINE-TUNING ({unfrozen} iterations since the actor unfroze)"
    )
    print(f"  phase: {phase}\n")

    # Explained variance answers what raw value loss cannot. Value loss scales
    # with the square of the return magnitude, and this task's returns are ~1170
    # (reward ~11.7/step, gamma 0.99), so a loss of 16,000 is an 11% residual --
    # the same relative accuracy that reads as ~260 on an Isaac Lab locomotion
    # task rewarding ~1.5/step. Only EV separates "predicts the mean, explains
    # nothing" from "healthy critic on a big-reward task"; both produce the same
    # value loss. Runs started before this was added will not have the tag.
    ev = vals("Diagnostics/explained_variance")
    if ev:
        first, last = trend(ev)
        if last >= 0.7:
            verdict, note = "PASS", "critic explains the returns; advantages carry signal"
        elif last >= 0.3:
            verdict, note = "WATCH", "critic explains some of the returns; advantages are noisy"
        else:
            verdict, note = "FAIL", (
                "critic explains ~nothing -- advantages ARE noise, which is the "
                "measured root cause of every previous descent"
            )
        print(f"0. critic signal    explained variance {fmt(first,3)} -> {fmt(last,3)}")
        print(f"   {verdict}  {spark(ev)}")
        print(f"         {note}")
        print()

    value = vals("Loss/value")
    if value:
        # Judge the warm-up on the frozen window ONLY. After unfreeze the actor
        # is moving the value target around, so later value loss is a different
        # quantity and mixing them in would hide whether the warm-up worked.
        warm = value[:freeze] if freeze else value
        if len(warm) < MIN_SAMPLES:
            print(
                f"1. critic warm-up   too early: {len(warm)} of {freeze} frozen "
                f"iterations logged (need >= {MIN_SAMPLES} to call a trend)"
            )
            print(f"   ....  {spark(warm)}")
        else:
            first, last = trend(warm)
            change = (last - first) / first * 100 if first else float("nan")
            ok = last < first * 0.7
            direction = "fell" if change < 0 else "ROSE"
            print(
                f"1. critic warm-up   value loss {direction} {fmt(first,1)} -> "
                f"{fmt(last,1)} ({abs(change):.0f}% {'down' if change < 0 else 'UP'})"
            )
            print(f"   {'PASS' if ok else 'WATCH'}  {spark(warm)}")
            if not ok:
                print(
                    "   Value loss has not fallen materially. The advantages the "
                    "actor will train on are still\n   noise-dominated -- consider "
                    "a longer --freeze_actor_iterations before trusting anything\n"
                    "   after unfreeze. (Early rises are normal: the critic head "
                    "starts zeroed and returns are large.)"
                )

    std = vals("Policy/mean_std")
    if std:
        first, last = trend(std)
        ok = last <= 0.16
        print(
            f"\n2. exploration std  {fmt(first)} -> {fmt(last)} "
            f"(clamped to [0.03, 0.15]; 0.31 was the runaway that cost tracking)"
        )
        print(f"   {'PASS' if ok else 'FAIL'}  {spark(std)}")

    lateral = vals("Metrics/lateral_error_m")
    if lateral:
        if last_step < freeze:
            # The actor is frozen: it CANNOT be descending. Movement here is
            # terrain draw and episode-boundary noise. Calling it a descent
            # would be a false alarm on the one signal we would kill a run for.
            recent = sum(lateral[-5:]) / len(lateral[-5:])
            print(
                f"\n3. tracking         not judged yet -- the actor is still "
                f"frozen, so it cannot have moved."
            )
            print(
                f"   ....  lateral error is {fmt(recent)} m "
                f"(student baseline 0.28-0.40 worst-family on held-out meshes)"
            )
            print(f"         {spark(lateral)}")
        else:
            post = lateral[freeze:]
            if len(post) < MIN_SAMPLES:
                print(
                    f"\n3. tracking         too early: {len(post)} iterations "
                    f"since unfreeze (need >= {MIN_SAMPLES})"
                )
                print(f"   ....  {spark(post)}")
            else:
                first, last = trend(post)
                ok = last <= first * 1.10
                print(
                    f"\n3. tracking         lateral error {fmt(first)} -> "
                    f"{fmt(last)} m since unfreeze  (descent signature = rising)"
                )
                print(f"   {'PASS' if ok else 'FAIL'}  {spark(post)}")
                if not ok:
                    print(
                        "   This is the descent signature. Kill the run rather "
                        "than waiting for a recovery --\n   every previous run "
                        "that showed this never recovered."
                    )

    # ------------------------------------------------------------- gate table
    print("\n" + "-" * 78)
    print("TRAINING-TERRAIN METRICS vs THE ACCEPTANCE GATES")
    print("(means on TRAINING terrain -- necessary, NOT sufficient. Acceptance is")
    print(" worst-family/worst-mesh on held-out meshes via evaluate_gates.py.)")
    print("-" * 78)
    metric_to_gate = [
        ("Metrics/lateral_error_m", "lateral", "lateral error (m)"),
        ("Metrics/cut_height_error_m", "cut_height", "cut height error (m)"),
        ("Metrics/ground_contact_fraction", "ground_contact", "ground contact"),
        ("Metrics/head_alignment", "head_alignment", "head alignment"),
    ]
    print(f"  {'metric':24s} {'now':>9s} {'gate':>9s} {'':3s} trend")
    for tag, gate_name, label in metric_to_gate:
        v = vals(tag)
        if not v:
            continue
        gate = GATES[gate_name]
        now = sum(v[-max(len(v) // 10, 1):]) / max(len(v) // 10, 1)
        mark = "ok " if gate.passes(now) else "OVER" if gate.direction == "max" else "UNDER"
        arrow = "<=" if gate.direction == "max" else ">="
        print(f"  {label:24s} {fmt(now):>9s} {arrow}{gate.threshold:<8g} {mark:4s} {spark(v, 28)}")

    # --------------------------------------------------------- reward history
    print("\n" + "-" * 78)
    print("REWARD TERMS  (first quarter -> last quarter)")
    print("-" * 78)
    reward_tags = sorted(t for t in series if t.startswith("Episode_Reward/"))
    print(f"  {'term':24s} {'start':>10s} {'now':>10s} {'change':>9s}")
    for tag in reward_tags:
        v = vals(tag)
        first, last = trend(v)
        delta = last - first
        name = tag.split("/", 1)[1]
        flag = ""
        if name == "total" and delta < -0.5:
            flag = "  <== DESCENDING"
        print(f"  {name:24s} {fmt(first,3):>10s} {fmt(last,3):>10s} {delta:+9.3f}{flag}")

    # ------------------------------------------------------------ checkpoints
    ckpts = sorted(
        glob.glob(str(run_dir / "model_*.pt")),
        key=lambda f: int(Path(f).stem.split("_")[1]),
    )
    print("\n" + "-" * 78)
    print(f"CHECKPOINTS  ({len(ckpts)} saved)")
    print("-" * 78)
    if ckpts:
        print("  " + ", ".join(Path(c).name for c in ckpts[-8:]))
        print(f"\n  Score the newest on the real contract (3 held-out meshes, ~25 min):")
        print(
            f"    PPO_ROM/scripts/evaluate_gates.py {ckpts[-1]} \\\n"
            f"        --meshes 1337,17,909 --steps 480 --num_envs 32 --stage 5 \\\n"
            f"        --report PPO_ROM/reports/frontier/$(basename {ckpts[-1]} .pt)_gates.json --headless"
        )
        print("\n  Compare against the measured baselines in "
              "PPO_ROM/reports/frontier/:")
        print("    student_gates.json (the warm start)  teacher_gates.json (the ceiling)")

    if a.plot:
        write_plot(run_dir, series, freeze)

    print()


def write_plot(run_dir: Path, series: dict, freeze: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("Loss/value", "Critic value loss", "must fall during warm-up", "log"),
        ("Metrics/lateral_error_m", "Lateral tracking error (m)", "the objective", "linear"),
        ("Policy/mean_std", "Exploration std", "clamped to [0.03, 0.15]", "linear"),
        ("Episode_Reward/total", "Total episode reward", "descent = falling", "linear"),
        ("Metrics/ground_contact_fraction", "Ground contact fraction", "gate >= 0.700 (worst-family)", "linear"),
        ("Metrics/head_alignment", "Head alignment", "gate >= 0.900 (p01)", "linear"),
    ]
    available = [pnl for pnl in panels if series.get(pnl[0])]
    if not available:
        print("  (nothing to plot yet)")
        return
    rows = (len(available) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(13, 3.1 * rows), squeeze=False)
    for ax, (tag, title, subtitle, scale) in zip(
        [ax for row in axes for ax in row], available
    ):
        pts = series[tag]
        xs = [s for s, _ in pts]
        ys = [v for _, v in pts]
        ax.plot(xs, ys, linewidth=1.4, color="#2b6cb0")
        if scale == "log" and min(ys) > 0:
            ax.set_yscale("log")
        if freeze and max(xs) > freeze:
            # The unfreeze boundary is the single most important x on every
            # panel: everything before it is the critic learning, everything
            # after is the policy being allowed to move.
            ax.axvline(freeze, color="#c53030", linestyle="--", linewidth=1.2)
            ax.text(
                freeze, ax.get_ylim()[1], " actor unfrozen",
                color="#c53030", va="top", fontsize=8,
            )
        for gate_name, gate in GATES.items():
            if gate_name == "lateral" and "lateral_error" in tag:
                ax.axhline(gate.threshold, color="#2f855a", linestyle=":", linewidth=1.2)
            if gate_name == "ground_contact" and "ground_contact" in tag:
                ax.axhline(gate.threshold, color="#2f855a", linestyle=":", linewidth=1.2)
            if gate_name == "head_alignment" and tag.endswith("head_alignment"):
                ax.axhline(gate.threshold, color="#2f855a", linestyle=":", linewidth=1.2)
        ax.set_title(f"{title}\n{subtitle}", fontsize=10, loc="left")
        ax.set_xlabel("iteration", fontsize=8)
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.tick_params(labelsize=8)
    for ax in [ax for row in axes for ax in row][len(available):]:
        ax.axis("off")
    fig.suptitle(
        f"PPO3_1 warm-up + fine-tune — {run_dir.name}   "
        f"(red = actor unfrozen, green dotted = acceptance gate)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = run_dir / "status.png"
    fig.savefig(out, dpi=130)
    print(f"\n  charts written: {out}")
    print(f"  open with:      xdg-open {out}")


if __name__ == "__main__":
    main()
