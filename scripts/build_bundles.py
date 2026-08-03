"""Build multi-provenance note bundles from MIMIC-III.

A bundle is one admission's notes from several roles, packed into ONE sequence.
This is the only input shape under which the authority bias is not a no-op:
tests/test_invariants.py proves the bias is exactly inert when every token
shares a role.

Splits are group-disjoint by SUBJECT_ID, not by row and not by HADM_ID -- a
patient with several admissions leaks across an HADM_ID split. ACCORD used a
row-level random split, which is the contamination the reviewers caught.

Local only. Writes to local_only/ (gitignored). Prints aggregates only.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.roles import ROLE_NAMES, role_of  # noqa: E402

NOTEEVENTS = os.environ.get("MIMIC_NOTEEVENTS_PATH", "data/NOTEEVENTS.csv.gz")
OUT = Path(__file__).resolve().parent.parent / "local_only" / "bundles"

csv.field_size_limit(1_000_000_000)  # TEXT contains embedded newlines and is large


def parse_ts(row: dict) -> datetime | None:
    for key, fmt in (("CHARTTIME", "%Y-%m-%d %H:%M:%S"),
                     ("STORETIME", "%Y-%m-%d %H:%M:%S"),
                     ("CHARTDATE", "%Y-%m-%d")):
        v = (row.get(key) or "").strip()
        if not v:
            continue
        try:
            return datetime.strptime(v, fmt)
        except ValueError:
            continue
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=NOTEEVENTS)
    ap.add_argument("--max-admissions", type=int, default=20_000)
    ap.add_argument("--min-roles", type=int, default=2)
    ap.add_argument("--max-segments", type=int, default=6)
    ap.add_argument("--chars-per-segment", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    by_adm: dict[str, list[dict]] = defaultdict(list)
    subj_of: dict[str, str] = {}
    n_read = n_kept = 0

    with gzip.open(args.path, "rt", newline="") as f:
        for row in csv.DictReader(f):
            n_read += 1
            if (row.get("ISERROR") or "").strip():
                continue
            hadm = (row.get("HADM_ID") or "").strip()
            if not hadm:
                continue
            r = role_of(row.get("CATEGORY"))
            ts = parse_ts(row)
            if ts is None:
                continue
            txt = (row.get("TEXT") or "").strip()
            if len(txt) < 150:
                continue
            if len(by_adm) >= args.max_admissions and hadm not in by_adm:
                continue
            by_adm[hadm].append({"role": r, "ts": ts, "text": txt[: args.chars_per_segment]})
            subj_of[hadm] = (row.get("SUBJECT_ID") or "").strip()
            n_kept += 1
            if n_read % 250_000 == 0:
                print(f"  ...read {n_read:,} rows, {len(by_adm):,} admissions", flush=True)

    print(f"\nread {n_read:,} rows, kept {n_kept:,} notes across {len(by_adm):,} admissions")

    bundles = []
    for hadm, notes in by_adm.items():
        roles = {n["role"] for n in notes}
        if len(roles) < args.min_roles:
            continue
        notes.sort(key=lambda n: n["ts"])
        t0 = notes[0]["ts"]
        # Take the most recent note of each distinct role first (maximizes
        # provenance diversity, which is the variable of interest), then fill
        # by recency. Selection is label-independent.
        chosen, seen = [], set()
        for n in reversed(notes):
            if n["role"] not in seen:
                seen.add(n["role"])
                chosen.append(n)
        for n in reversed(notes):
            if len(chosen) >= args.max_segments:
                break
            if n not in chosen:
                chosen.append(n)
        chosen = chosen[: args.max_segments]
        chosen.sort(key=lambda n: n["ts"])
        bundles.append({
            "hadm_id": hadm,
            "subject_id": subj_of[hadm],
            "segments": [{
                "role": n["role"],
                "delta_t_hours": max(0.0, (n["ts"] - t0).total_seconds() / 3600.0),
                "text": n["text"],
            } for n in chosen],
        })

    print(f"bundles with >={args.min_roles} distinct roles: {len(bundles):,}")

    # Group-disjoint split by SUBJECT_ID.
    subjects = sorted({b["subject_id"] for b in bundles})
    rng = random.Random(args.seed)
    rng.shuffle(subjects)
    n = len(subjects)
    tr = set(subjects[: int(0.7 * n)])
    va = set(subjects[int(0.7 * n): int(0.8 * n)])
    splits = {"train": [], "val": [], "test": []}
    for b in bundles:
        s = b["subject_id"]
        splits["train" if s in tr else "val" if s in va else "test"].append(b)

    for a in ("train", "val", "test"):
        for c in ("train", "val", "test"):
            if a < c:
                sa = {b["subject_id"] for b in splits[a]}
                sc = {b["subject_id"] for b in splits[c]}
                assert not (sa & sc), f"SUBJECT_ID leaked between {a} and {c}"

    role_hist: dict[str, int] = defaultdict(int)
    seg_counts = []
    for b in bundles:
        seg_counts.append(len(b["segments"]))
        for s in b["segments"]:
            role_hist[ROLE_NAMES[s["role"]]] += 1

    for name, items in splits.items():
        p = OUT / f"{name}.jsonl"
        with p.open("w") as f:
            for b in items:
                f.write(json.dumps(b) + "\n")
        print(f"  {name:<6} {len(items):>8,} bundles  {len({b['subject_id'] for b in items}):>7,} subjects  -> {p.name}")

    print(f"\nmean segments/bundle: {sum(seg_counts)/len(seg_counts):.2f}")
    print("segment role distribution:")
    for k, v in sorted(role_hist.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<14}{v:>9,}")

    meta = {"n_bundles": len(bundles), "mean_segments": sum(seg_counts) / len(seg_counts),
            "role_hist": dict(role_hist),
            "splits": {k: len(v) for k, v in splits.items()}}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
