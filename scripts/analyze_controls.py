"""Read the control runs and say what the mechanism actually is.

The question: A4 gained ~0.84 nats over vanilla. Is that PROVENANCE, or is it
any per-segment partition delivered through a shared-across-layers additive
attention bias?

  A5 (segment-index roles) ~ A4  =>  segmentation. Provenance contributes nothing.
  A5 ~ A1                        =>  role identity does real work.
  A4-length ~ A4                 =>  any informative per-segment category suffices.

Also reports loss by position WITHIN a segment. At seq 256 with ~6 segments each
note contributes ~36 tokens, so training sits in note openings -- the most
template-heavy region. An advantage concentrated in the first few tokens is
template identification under truncation, not provenance.
"""

from __future__ import annotations

import json
from pathlib import Path

RUNS = Path("/Users/rickygole/mimic_local_only/runs")

LABELS = {
    ("A1", ""): "A1  vanilla",
    ("A4", ""): "A4  true role labels",
    ("A4", "segment_index_roles"): "A5  segment-index (no provenance meaning)",
    ("A4", "random_roles"): "A4  random role per segment",
    ("A4", "length_roles"): "A4  note-length bucket",
    ("A3", ""): "A3  attention bias only",
}


def ctrl_key(s: dict) -> str:
    c = s.get("controls") or {}
    for k in ("segment_index_roles", "random_roles", "length_roles", "permute_roles"):
        if c.get(k):
            return k
    return ""


def main() -> None:
    runs = []
    for p in sorted(RUNS.glob("*/summary.json")):
        s = json.loads(p.read_text())
        if s.get("steps", 0) < 1000 or s.get("size") != "10M":
            continue
        s["_ctrl"] = ctrl_key(s)
        s["_shuf"] = bool((s.get("controls") or {}).get("shuffle_segments"))
        s["_dir"] = p.parent.name
        runs.append(s)
    if not runs:
        raise SystemExit("no completed 1000-step 10M runs yet")

    shuf = [r for r in runs if r["_shuf"]]
    unshuf = [r for r in runs if not r["_shuf"]]

    def table(rows, title):
        if not rows:
            return
        print(f"\n{title}")
        print(f"{'condition':<46}{'val_loss':>10}{'vs A1':>9}")
        print("-" * 65)
        a1 = next((r["final_val_loss"] for r in rows
                   if r["arm"] == "A1" and not r["_ctrl"]), None)
        for r in sorted(rows, key=lambda x: x["final_val_loss"]):
            lab = LABELS.get((r["arm"], r["_ctrl"]), f"{r['arm']} {r['_ctrl']}")
            d = f"{r['final_val_loss'] - a1:+.4f}" if a1 is not None else "--"
            print(f"{lab:<46}{r['final_val_loss']:>10.4f}{d:>9}")
        return a1

    a1_shuf = table(shuf, "SEGMENT ORDER SHUFFLED (role decorrelated from position)")
    table(unshuf, "ORIGINAL CHRONOLOGICAL ORDER (role confounded with position)")

    # --- position-within-segment breakdown -------------------------------
    have_pos = [r for r in runs if r.get("loss_by_segment_position")]
    if have_pos:
        bins = [k for k in have_pos[0]["loss_by_segment_position"] if not k.endswith("_n")]
        print("\n\nLOSS BY POSITION WITHIN SEGMENT (boilerplate diagnostic)")
        print(f"{'condition':<46}" + "".join(f"{b.replace('pos_',''):>12}" for b in bins))
        print("-" * (46 + 12 * len(bins)))
        for r in sorted(have_pos, key=lambda x: x["final_val_loss"]):
            lab = LABELS.get((r["arm"], r["_ctrl"]), f"{r['arm']} {r['_ctrl']}")
            vals = r["loss_by_segment_position"]
            print(f"{lab:<46}" + "".join(
                f"{vals[b]:>12.4f}" if vals.get(b) is not None else f"{'--':>12}"
                for b in bins))
        base = next((r for r in have_pos if r["arm"] == "A1" and not r["_ctrl"]), None)
        best = next((r for r in have_pos if r["arm"] == "A4" and not r["_ctrl"]), None)
        if base and best:
            print(f"\n{'gap (A1 - A4) by position':<46}" + "".join(
                f"{base['loss_by_segment_position'][b] - best['loss_by_segment_position'][b]:>12.4f}"
                if base['loss_by_segment_position'].get(b) is not None
                and best['loss_by_segment_position'].get(b) is not None else f"{'--':>12}"
                for b in bins))
            g = [base['loss_by_segment_position'][b] - best['loss_by_segment_position'][b]
                 for b in bins if base['loss_by_segment_position'].get(b) is not None
                 and best['loss_by_segment_position'].get(b) is not None]
            if len(g) >= 2 and g[0] > 0:
                print(f"\n  first-bin gap {g[0]:+.4f} vs last-bin gap {g[-1]:+.4f}")
                if g[0] > 2 * abs(g[-1]):
                    print("  -> advantage CONCENTRATED in segment openings. Consistent with")
                    print("     template identification under truncation, not provenance.")
                else:
                    print("  -> advantage roughly FLAT across position. Not a boilerplate artifact.")

    # --- verdict ----------------------------------------------------------
    def get(arm, ctrl):
        return next((r["final_val_loss"] for r in shuf
                     if r["arm"] == arm and r["_ctrl"] == ctrl), None)

    a4, a5 = get("A4", ""), get("A4", "segment_index_roles")
    rnd, ln = get("A4", "random_roles"), get("A4", "length_roles")
    print("\n" + "=" * 65)
    print("VERDICT")
    print("=" * 65)
    if a4 is None or a5 is None or a1_shuf is None:
        print("  incomplete -- need A1, A4 and A5 under shuffling")
        return
    span = a1_shuf - a4
    frac = (a1_shuf - a5) / span if span else float("nan")
    print(f"  A1 {a1_shuf:.4f} | A4 {a4:.4f} | A5 {a5:.4f}")
    print(f"  A5 recovers {frac:.0%} of A4's gain over vanilla.")
    if frac > 0.8:
        print("\n  Segment-index roles recover nearly all of it. The mechanism is")
        print("  SEGMENTATION, not provenance: a shared additive logit bias inducing")
        print("  soft block-diagonal attention. Comparison class is T5 relative")
        print("  position bias and ALiBi -- not BERT. The provenance thesis is dead,")
        print("  with a named mechanism in its place.")
    elif frac < 0.3:
        print("\n  Segment-index roles recover little. Role IDENTITY is doing real work.")
        print("  Next: confirm at convergence, not at a fixed step budget.")
    else:
        print(f"\n  Split result: {frac:.0%} segmentation, {1-frac:.0%} attributable to")
        print("  role identity. Report both; neither story is clean.")
    for name, v in (("random roles", rnd), ("length buckets", ln)):
        if v is not None:
            print(f"  {name:<16} {v:.4f}  ({(a1_shuf - v)/span:.0%} of A4's gain)"
                  if span else f"  {name}: {v:.4f}")


if __name__ == "__main__":
    main()
