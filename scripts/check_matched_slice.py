"""Is the length-matched subsample a weird slice?

Red-team objection: matching across a 12x length spread necessarily pairs long
nursing notes against short discharge summaries, and short discharge summaries
are a peculiar subpopulation (deaths, transfers, AMA departures). If so, the
97.5% number sits on an unrepresentative slice and the honest headline is the
99.7% scrubbed-unmatched figure instead.

Reports slice size, length distributions, and what fraction of each role's
original pool survives matching. Aggregates only; no note text.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from role_probe import OUT_DIR, length_matched  # noqa: E402

CACHE = OUT_DIR / "probe_sample.pkl"


def pct(x: np.ndarray, q: float) -> float:
    return float(np.percentile(x, q))


def main() -> None:
    with CACHE.open("rb") as f:
        buckets = pickle.load(f)
    matched = length_matched(buckets, seed=0)
    if matched is None:
        raise SystemExit("matching failed")

    total_orig = sum(len(v) for v in buckets.values())
    total_match = sum(len(v) for v in matched.values())
    print(f"original pool : {total_orig:,} notes")
    print(f"matched slice : {total_match:,} notes  ({total_match / total_orig:.1%} retained)\n")

    print(f"{'role':<12}{'orig N':>9}{'match N':>9}{'kept':>8}"
          f"{'orig med':>11}{'match med':>11}{'match p10':>11}{'match p90':>11}")
    print("-" * 82)
    rows = {}
    for role in sorted(buckets):
        o = np.array([len(t) for _, t in buckets[role]])
        m = np.array([len(t) for _, t in matched[role]])
        rows[role] = {
            "orig_n": int(o.size), "match_n": int(m.size),
            "retained": float(m.size / o.size),
            "orig_median": pct(o, 50), "match_median": pct(m, 50),
            "match_p10": pct(m, 10), "match_p90": pct(m, 90),
        }
        print(f"{role:<12}{o.size:>9,}{m.size:>9,}{m.size / o.size:>7.1%}"
              f"{pct(o, 50):>11,.0f}{pct(m, 50):>11,.0f}{pct(m, 10):>11,.0f}{pct(m, 90):>11,.0f}")

    print("\nPercentile of each role's ORIGINAL length distribution that the")
    print("matched slice occupies (50 = representative, <50 = short tail, >50 = long tail):")
    for role in sorted(buckets):
        o = np.array([len(t) for _, t in buckets[role]])
        m = np.array([len(t) for _, t in matched[role]])
        centile = float((o < np.median(m)).mean() * 100)
        print(f"  {role:<12} matched median sits at the {centile:5.1f}th percentile of its own pool")

    out = OUT_DIR / "matched_slice_profile.json"
    out.write_text(json.dumps({"total_original": total_orig,
                               "total_matched": total_match, "per_role": rows}, indent=2))
    print(f"\nwrote {out}")

    worst = min(r["retained"] for r in rows.values())
    print("\n" + "=" * 72)
    if total_match / total_orig < 0.25 or worst < 0.10:
        print("The matched slice IS a narrow subpopulation. Report 99.7% (scrubbed,")
        print("full pool) as the headline and the matched number as a supporting")
        print("control, with this composition table alongside it.")
    else:
        print("The matched slice is broadly representative; it can carry the headline.")


if __name__ == "__main__":
    main()
