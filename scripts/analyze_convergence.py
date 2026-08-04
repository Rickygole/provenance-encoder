"""Does the segmentation-bias advantage survive longer training?

A fixed-step comparison cannot distinguish a real advantage from a head start.
This reads the eval trajectories of vanilla and the segment-index bias and asks
whether the gap NARROWS (convergence speed -- the artifact), holds, or WIDENS
(a different solution, not the same solution sooner).

Also reports the step-equivalence: how many vanilla steps are needed to reach
what the biased model reaches at step N.
"""

from __future__ import annotations

import json
from pathlib import Path

RUNS = Path("/Users/rickygole/mimic_local_only/runs")


def load(pattern: str):
    for p in RUNS.glob(pattern + "/summary.json"):
        s = json.loads(p.read_text())
        return s
    return None


def main() -> None:
    a1 = load("A1_10M_st4000_s0_shuf")
    a5 = load("A4_10M_st4000_s0_segidx_shuf")
    if not a1 or not a5:
        have = sorted(p.parent.name for p in RUNS.glob("*/summary.json"))
        raise SystemExit(f"need both 4000-step runs. have: {have}")

    h1 = {h["step"]: h["val_loss"] for h in a1["history"]}
    h5 = {h["step"]: h["val_loss"] for h in a5["history"]}
    steps = sorted(set(h1) & set(h5))

    print(f"{'step':>7}{'A1 vanilla':>13}{'A5 seg-index':>15}{'gap':>10}")
    print("-" * 45)
    for s in steps:
        print(f"{s:>7}{h1[s]:>13.4f}{h5[s]:>15.4f}{h1[s]-h5[s]:>10.4f}")

    gaps = [h1[s] - h5[s] for s in steps]
    first_q = sum(gaps[: max(1, len(gaps)//4)]) / max(1, len(gaps)//4)
    last_q = sum(gaps[-max(1, len(gaps)//4):]) / max(1, len(gaps)//4)

    print(f"\nmean gap, first quarter of training : {first_q:.4f}")
    print(f"mean gap, last quarter of training  : {last_q:.4f}")
    print(f"change                              : {last_q - first_q:+.4f}")

    # Step-equivalence: how far does vanilla have to train to match A5?
    print("\nSTEP EQUIVALENCE (vanilla steps needed to match A5 at a given step)")
    for s in steps:
        target = h5[s]
        reached = [t for t in steps if h1[t] <= target]
        if reached:
            print(f"  A5 @ {s:>5} (loss {target:.4f})  <-  vanilla reaches it @ {min(reached):>5}")
        else:
            print(f"  A5 @ {s:>5} (loss {target:.4f})  <-  vanilla NEVER reaches it in "
                  f"{max(steps)} steps")

    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    if last_q < first_q - 0.05:
        print("  Gap NARROWS with training. This is a convergence-SPEED effect and")
        print("  will likely vanish at convergence. Efficiency claim only; the")
        print("  comparison class is ALiBi and T5-RPE, both published as")
        print("  quality-at-fixed-budget results.")
    elif last_q > first_q + 0.05:
        print("  Gap WIDENS with training. Not a head start -- the biased model is")
        print("  finding a different solution, not the same solution sooner. That is")
        print("  a stronger claim and worth testing at a second scale before trusting.")
    else:
        print("  Gap is STABLE. Not obviously a head start, but 4000 steps is still")
        print("  under 5% of a real pretraining run; do not call this convergence.")
    print("\n  Caveat that must appear in any writeup: max steps here is "
          f"{max(steps)} ~= {max(steps)*8*256/1e6:.1f}M tokens. A converged small")
    print("  clinical MLM is in the 1.5-2.5 nat range; these runs end near "
          f"{min(min(h1.values()), min(h5.values())):.1f}.")


if __name__ == "__main__":
    main()
