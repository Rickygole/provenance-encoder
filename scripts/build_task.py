"""Labeled multi-note clinical task from MIMIC-III.

The task ACCORD never had: in-hospital mortality predicted from the notes of the
first 48 hours, where the input is a BUNDLE of notes from several authors rather
than a single document. 5,854 positives against ACCORD's 12.

LEAKAGE RULES, enforced here and asserted again at train time. Each one is a
specific way this task could silently become trivial:

  1. DISCHARGE SUMMARIES ARE EXCLUDED. They are authored at discharge and state
     the outcome verbatim. Including them is the ACCORD Rule-F1=0.953 failure.
  2. Only notes charted within [ADMITTIME, ADMITTIME + 48h] are used.
  3. ADMISSIONS THAT END (death or discharge) INSIDE THE WINDOW ARE DROPPED.
     This replaces an earlier posthumous-note filter that was LABEL-CONDITIONED
     and therefore a leak: DEATHTIME is non-null for exactly the 5,854 positive
     admissions and zero negatives (verified), so filtering on it truncated the
     observation window for positives only. The window became a function of the
     outcome. Requiring survival through the window makes the input
     label-independent by construction, and the label becomes "died later in
     this admission" -- the standard benchmark definition.
  4. Splits are group-disjoint by SUBJECT_ID -- a patient with several
     admissions must not appear on both sides.

Local only; writes to local_only/. Prints aggregates, never note text.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.roles import OUTCOME_EXCLUDED_ROLES, ROLE_NAMES, role_of  # noqa: E402

NOTEEVENTS = os.environ.get("MIMIC_NOTEEVENTS_PATH", "data/NOTEEVENTS.csv.gz")
ADMISSIONS = os.environ.get("MIMIC_ADMISSIONS_PATH", "data/ADMISSIONS.csv")
OUT = Path(__file__).resolve().parent.parent / "local_only" / "task"

csv.field_size_limit(1_000_000_000)
FMT = "%Y-%m-%d %H:%M:%S"


def parse(v: str | None, fmt: str = FMT):
    v = (v or "").strip()
    if not v:
        return None
    for f in (fmt, "%Y-%m-%d"):
        try:
            return datetime.strptime(v, f)
        except ValueError:
            continue
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--notes", default=NOTEEVENTS)
    ap.add_argument("--admissions", default=ADMISSIONS)
    ap.add_argument("--window-hours", type=int, default=48)
    ap.add_argument("--max-segments", type=int, default=8)
    ap.add_argument("--chars-per-segment", type=int, default=1500)
    ap.add_argument("--min-segments", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    adm: dict[str, dict] = {}
    with open(args.admissions, newline="") as f:
        for r in csv.DictReader(f):
            t0 = parse(r["ADMITTIME"])
            if t0 is None:
                continue
            adm[r["HADM_ID"].strip()] = {
                "subject_id": r["SUBJECT_ID"].strip(),
                "t0": t0,
                "death": parse(r.get("DEATHTIME")),
                "discharge": parse(r.get("DISCHTIME")),
                "label": int(r["HOSPITAL_EXPIRE_FLAG"]),
            }
    print(f"admissions with ADMITTIME: {len(adm):,}")

    cutoff = timedelta(hours=args.window_hours)
    # LABEL-INDEPENDENT GATE. Keep only admissions still ongoing at the end of
    # the observation window, so no positive has its notes truncated relative to
    # a negative. Applied to admissions BEFORE any note is read.
    n_before = len(adm)
    adm = {h: a for h, a in adm.items()
           if (a["discharge"] is None or a["discharge"] > a["t0"] + cutoff)
           and (a["death"] is None or a["death"] > a["t0"] + cutoff)}
    print(f"admissions still ongoing at {args.window_hours}h: {len(adm):,} "
          f"(dropped {n_before - len(adm):,} that ended inside the window)")
    by_adm: dict[str, list] = defaultdict(list)
    n_rows = kept = drop_late = drop_role = 0
    drop_death = 0

    with gzip.open(args.notes, "rt", newline="") as f:
        for r in csv.DictReader(f):
            n_rows += 1
            if (r.get("ISERROR") or "").strip():
                continue
            h = (r.get("HADM_ID") or "").strip()
            a = adm.get(h)
            if a is None:
                continue
            role = role_of(r.get("CATEGORY"))
            if role in OUTCOME_EXCLUDED_ROLES:      # rule 1
                drop_role += 1
                continue
            ts = parse(r.get("CHARTTIME")) or parse(r.get("CHARTDATE"), "%Y-%m-%d")
            if ts is None:
                continue
            if ts < a["t0"] or ts > a["t0"] + cutoff:   # rule 2
                drop_late += 1
                continue
            # rule 3 is now enforced at the admission level, above: any
            # admission ending inside the window is gone, so no note here can be
            # posthumous and no per-note label-conditioned filter is needed.
            txt = (r.get("TEXT") or "").strip()
            if len(txt) < 100:
                continue
            by_adm[h].append({"role": role, "ts": ts,
                              "dt": (ts - a["t0"]).total_seconds() / 3600.0,
                              "text": txt[: args.chars_per_segment]})
            kept += 1
            if n_rows % 500_000 == 0:
                print(f"  ...{n_rows:,} rows, {len(by_adm):,} admissions", flush=True)

    print(f"\nrows read {n_rows:,} | notes kept {kept:,}")
    print(f"dropped: discharge-summary {drop_role:,} | outside {args.window_hours}h "
          f"{drop_late:,} | posthumous {drop_death:,}")

    rows = []
    for h, notes in by_adm.items():
        if len(notes) < args.min_segments:
            continue
        if len({n["role"] for n in notes}) < 2:
            continue
        notes.sort(key=lambda n: n["ts"])
        chosen, seen = [], set()
        for n in reversed(notes):
            if n["role"] not in seen:
                seen.add(n["role"]); chosen.append(n)
        for n in reversed(notes):
            if len(chosen) >= args.max_segments:
                break
            if n not in chosen:
                chosen.append(n)
        chosen = chosen[: args.max_segments]
        chosen.sort(key=lambda n: n["ts"])
        a = adm[h]
        rows.append({"hadm_id": h, "subject_id": a["subject_id"], "label": a["label"],
                     "segments": [{"role": n["role"], "delta_t_hours": n["dt"],
                                   "text": n["text"]} for n in chosen]})

    pos = sum(r["label"] for r in rows)
    print(f"\nexamples: {len(rows):,} | positives: {pos:,} ({pos/len(rows):.2%})")

    subs = sorted({r["subject_id"] for r in rows})
    random.Random(args.seed).shuffle(subs)
    n = len(subs)
    tr, va = set(subs[: int(.7*n)]), set(subs[int(.7*n): int(.8*n)])
    split = {"train": [], "val": [], "test": []}
    for r in rows:
        s = r["subject_id"]
        split["train" if s in tr else "val" if s in va else "test"].append(r)

    for a_ in ("train", "val", "test"):          # rule 4
        for b_ in ("train", "val", "test"):
            if a_ < b_:
                assert not ({r["subject_id"] for r in split[a_]} &
                            {r["subject_id"] for r in split[b_]}), f"{a_}/{b_} leak"

    role_hist = Counter(ROLE_NAMES[s["role"]] for r in rows for s in r["segments"])
    for k, v in split.items():
        p = sum(x["label"] for x in v)
        path = OUT / f"{k}.jsonl"
        with path.open("w") as f:
            for x in v:
                f.write(json.dumps(x) + "\n")
        print(f"  {k:<6}{len(v):>7,} examples  {p:>6,} pos ({p/max(len(v),1):.2%})  "
              f"{len({x['subject_id'] for x in v}):>6,} subjects")

    print("\nsegment roles (discharge summaries absent by construction):")
    for k, v in role_hist.most_common():
        print(f"  {k:<14}{v:>9,}")
    (OUT / "meta.json").write_text(json.dumps({
        "n": len(rows), "positives": pos, "window_hours": args.window_hours,
        "excluded_roles": [ROLE_NAMES[r] for r in OUTCOME_EXCLUDED_ROLES],
        "role_hist": dict(role_hist),
        "splits": {k: {"n": len(v), "pos": sum(x["label"] for x in v)}
                   for k, v in split.items()}}, indent=2))


if __name__ == "__main__":
    main()
