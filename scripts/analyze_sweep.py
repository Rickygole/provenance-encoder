"""Compare the four arms across seeds.

The decisive comparison is A2 vs A4. A2 is architecturally identical to A1 --
its only intervention is six literal vocabulary tokens. If A2 matches A4, the
architecture contributed nothing that a tokenizer change could not, and there is
no architectural contribution to report.

Reports paired differences by seed, because seeds share data order and the
paired contrast has far lower variance than comparing group means.
"""

from __future__ import annotations

import json
import statistics as stats
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "local_only" / "runs"
ARMS = ["A1", "A2", "A3", "A4"]
DESC = {
    "A1": "vanilla (no provenance)",
    "A2": "provenance as literal tokens",
    "A3": "row-centered KxK attention bias",
    "A4": "bias + role/auth embeddings + time bias",
}


def load(size: str, min_steps: int) -> dict[str, dict[int, dict]]:
    """Only runs matching the target size AND step budget are comparable.

    Smoke tests and timing probes write to the same run directories with tiny
    step counts; silently averaging them in would be exactly the kind of quiet
    contamination this project exists to avoid.
    """
    out: dict[str, dict[int, dict]] = {a: {} for a in ARMS}
    skipped = []
    for p in sorted(RUNS.glob("*/summary.json")):
        s = json.loads(p.read_text())
        if s["arm"] not in out:
            continue
        if s.get("size") != size or s.get("steps", 0) < min_steps:
            skipped.append(f"{p.parent.name}(size={s.get('size')},steps={s.get('steps')})")
            continue
        out[s["arm"]][s["seed"]] = s
    if skipped:
        print(f"skipped {len(skipped)} non-comparable run(s): {', '.join(skipped)}\n")
    return out


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="10M")
    ap.add_argument("--min-steps", type=int, default=1000)
    args = ap.parse_args()
    data = load(args.size, args.min_steps)
    have = {a: sorted(v) for a, v in data.items() if v}
    if not have:
        raise SystemExit(f"no completed runs under {RUNS}")
    print("completed runs:", {a: len(v) for a, v in data.items() if v}, "\n")

    print(f"{'arm':<5}{'description':<42}{'n':>3}{'val_loss':>20}{'ppl':>10}{'params':>12}")
    print("-" * 94)
    means = {}
    for a in ARMS:
        runs = list(data[a].values())
        if not runs:
            continue
        losses = [r["final_val_loss"] for r in runs]
        m = stats.mean(losses)
        sd = stats.stdev(losses) if len(losses) > 1 else 0.0
        means[a] = m
        import math
        print(f"{a:<5}{DESC[a]:<42}{len(losses):>3}"
              f"{m:>13.4f} ±{sd:<5.4f}{math.exp(min(m, 20)):>10.1f}"
              f"{runs[0]['non_emb_params']:>12,}")

    # Paired contrasts -- same seed, same data order.
    print("\nPaired differences in val_loss (negative = second arm better):")
    pairs = [("A1", "A2"), ("A1", "A3"), ("A1", "A4"), ("A2", "A4"), ("A3", "A4")]
    verdicts = {}
    for x, y in pairs:
        seeds = sorted(set(data[x]) & set(data[y]))
        if not seeds:
            continue
        d = [data[x][s]["final_val_loss"] - data[y][s]["final_val_loss"] for s in seeds]
        m = stats.mean(d)
        sd = stats.stdev(d) if len(d) > 1 else 0.0
        verdicts[(x, y)] = (m, sd, len(d))
        flag = "" if abs(m) > 2 * sd and sd > 0 else "   (within noise)"
        print(f"  {x} - {y}: {m:+.4f} ± {sd:.4f}  (n={len(d)}){flag}")

    print("\nProvenance parameter movement from zero init (the ACCORD guard):")
    for a in ("A3", "A4"):
        runs = list(data[a].values())
        if not runs:
            continue
        pp = runs[0].get("provenance_params", {})
        print(f"  {a}:")
        for k, v in pp.items():
            print(f"    {k:<12} moved {v['abs_movement']:.3e}  "
                  f"final_norm {v['final_norm']:.3e}  mean_grad {v['mean_grad_norm']:.2e}")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    if ("A2", "A4") in verdicts:
        m, sd, n = verdicts[("A2", "A4")]
        if abs(m) <= 2 * sd or abs(m) < 0.01:
            print("  A2 (six vocabulary tokens) and A4 (full architecture) are")
            print("  INDISTINGUISHABLE. The architecture contributes nothing a tokenizer")
            print("  change does not. There is no architectural contribution to report.")
        elif m > 0:
            print(f"  A4 beats A2 by {m:.4f} nats. Before claiming this: confirm it")
            print("  survives at convergence, not just at this fixed step budget.")
        else:
            print(f"  A2 beats A4 by {-m:.4f} nats. The architecture is actively harmful.")
    if ("A1", "A4") in verdicts:
        m, sd, _ = verdicts[("A1", "A4")]
        print(f"\n  A1 - A4 = {m:+.4f} ± {sd:.4f}")
    print("\n  Reminder: a fixed-step comparison measures convergence speed, not")
    print("  asymptotic quality. Any gain here must be re-checked at convergence.")


if __name__ == "__main__":
    main()
