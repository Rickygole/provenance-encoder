"""Aggregate scan of NOTEEVENTS. Emits counts only — never note text.

MIMIC-III is PhysioNet credentialed data. This script runs locally and prints
only category-level aggregates so no restricted content leaves the machine.
"""

import argparse
import os
from collections import Counter

import pandas as pd

NOTEEVENTS = os.environ.get("MIMIC_NOTEEVENTS_PATH", "data/NOTEEVENTS.csv.gz")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=NOTEEVENTS)
    ap.add_argument("--chunk", type=int, default=50_000)
    ap.add_argument("--limit-chunks", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    cat_counts: Counter[str] = Counter()
    desc_counts: Counter[tuple[str, str]] = Counter()
    char_totals: Counter[str] = Counter()
    hadm_with_cat: dict[str, set[float]] = {}
    n_rows = 0

    reader = pd.read_csv(
        args.path,
        chunksize=args.chunk,
        usecols=["SUBJECT_ID", "HADM_ID", "CATEGORY", "DESCRIPTION", "TEXT", "ISERROR"],
        compression="gzip",
        low_memory=False,
    )

    for i, chunk in enumerate(reader):
        if args.limit_chunks and i >= args.limit_chunks:
            break
        chunk = chunk[chunk["ISERROR"].isna()]
        n_rows += len(chunk)
        cats = chunk["CATEGORY"].fillna("<NA>").str.strip()
        cat_counts.update(cats)
        for c, d in zip(cats, chunk["DESCRIPTION"].fillna("<NA>").str.strip()):
            desc_counts[(c, d)] += 1
        lens = chunk["TEXT"].fillna("").str.len()
        for c, L in zip(cats, lens):
            char_totals[c] += int(L)
        for c, h in zip(cats, chunk["HADM_ID"]):
            if pd.notna(h):
                hadm_with_cat.setdefault(c, set()).add(float(h))
        if i % 10 == 0:
            print(f"  ...chunk {i}, rows so far {n_rows:,}", flush=True)

    print(f"\nTotal non-error notes: {n_rows:,}\n")
    print(f"{'CATEGORY':<28}{'NOTES':>12}{'MEAN CHARS':>12}{'ADMISSIONS':>12}")
    print("-" * 64)
    for cat, n in cat_counts.most_common():
        mean_chars = char_totals[cat] / n if n else 0
        print(f"{cat:<28}{n:>12,}{mean_chars:>12,.0f}{len(hadm_with_cat.get(cat, ())):>12,}")

    print("\nTop DESCRIPTION per category (schema only, no note text):")
    by_cat: dict[str, list[tuple[str, int]]] = {}
    for (c, d), n in desc_counts.items():
        by_cat.setdefault(c, []).append((d, n))
    for cat, items in sorted(by_cat.items(), key=lambda kv: -cat_counts[kv[0]])[:8]:
        top = sorted(items, key=lambda t: -t[1])[:4]
        print(f"  {cat}: " + ", ".join(f"{d} ({n:,})" for d, n in top))

    # How many admissions carry BOTH a nursing-type and a discharge summary?
    nursing = set()
    for c, hs in hadm_with_cat.items():
        if c.lower().startswith("nursing"):
            nursing |= hs
    discharge = hadm_with_cat.get("Discharge summary", set())
    physician = hadm_with_cat.get("Physician ", set()) | hadm_with_cat.get("Physician", set())
    print("\nMulti-provenance admission counts (the bundle-eligible population):")
    print(f"  nursing-any            : {len(nursing):,}")
    print(f"  discharge summary      : {len(discharge):,}")
    print(f"  physician              : {len(physician):,}")
    print(f"  nursing AND discharge  : {len(nursing & discharge):,}")
    print(f"  all three              : {len(nursing & discharge & physician):,}")


if __name__ == "__main__":
    main()
