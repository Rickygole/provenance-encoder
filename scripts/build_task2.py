"""Two more labeled tasks from the SAME first-48h MIMIC-III note bundles as
scripts/build_task.py's in-hospital-mortality task, so the project is not a
single-task anecdote.

  TASK B (--label readmit30): 30-day readmission. Positive if the SAME
    SUBJECT_ID has a later admission with ADMITTIME within 30 days of THIS
    admission's DISCHTIME. Admissions that end in death are excluded (they
    cannot be readmitted) -- note that this exclusion is a fact about the
    INDEX admission's own recorded outcome (HOSPITAL_EXPIRE_FLAG / DEATHTIME
    for the admission being classified), not about the readmission label
    being predicted, so it is label-independent for this task in the same
    sense DISCHTIME/DEATHTIME gating is label-independent for mortality: it
    would be computable before the future admission (if any) is known.

  TASK C (--label los7): prolonged length of stay, DISCHTIME - ADMITTIME > 7
    days, predicted from the first 48h.

LEAKAGE RULES, copied verbatim from build_task.py and re-verified for each
new task:

  1. DISCHARGE SUMMARIES ARE EXCLUDED. Authored at discharge, state the
     outcome verbatim.
  2. Only notes charted within [ADMITTIME, ADMITTIME + 48h] are used.
  3. ADMISSIONS THAT END (death or discharge) INSIDE THE 48h WINDOW ARE
     DROPPED, decided from ADMITTIME/DISCHTIME/DEATHTIME alone, BEFORE any
     note is read and BEFORE any label (mortality, readmission, or LOS) is
     computed. This is the same gate as build_task.py's rule 3, reused
     unchanged: it is a fact about whether the 48h observation window is
     intact, not a fact conditioned on any of the three labels. In
     particular, for LOS it does NOT filter on the LOS label itself (LOS>7d
     admissions are not treated any differently from LOS<=7d admissions that
     happen to survive past 48h) -- both survive the gate whenever
     DISCHTIME > ADMITTIME + 48h, i.e. whenever they are simply long enough
     to have a 48h window at all. For readmission, the future admission (if
     any) is never consulted by this gate.
  4. Splits are group-disjoint by SUBJECT_ID, asserted in code. For
     readmission this is the critical one: an index admission and every
     admission that could make it a readmission positive belong to the same
     subject, and the whole subject goes to one split, so the future event
     that defines the label can never straddle train/val/test.

Bundle construction (window, role filter, segment selection, max/min
segments) is copied verbatim from build_task.py's main loop so the note
content for a given HADM_ID is identical to what that script would produce.
It is rebuilt here (not imported and reused across a run) only because the
two scripts read different admission-level fields; the logic is unchanged.

Local only; writes to local_only/task_readmit30/ and local_only/task_los7/.
Prints aggregates, never note text.
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
OUT_ROOT = Path(__file__).resolve().parent.parent / "local_only"

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


def build_bundles(args, adm: dict) -> dict[str, list]:
    """Identical note-selection logic to build_task.py, parameterized only by
    the pre-filtered admission dict. Returns hadm_id -> chosen segment list."""
    by_adm: dict[str, list] = defaultdict(list)
    n_rows = kept = drop_late = drop_role = 0
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
            if ts < a["t0"] or ts > a["t0"] + timedelta(hours=args.window_hours):  # rule 2
                drop_late += 1
                continue
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
    print(f"dropped: discharge-summary {drop_role:,} | outside "
          f"{args.window_hours}h {drop_late:,}")

    chosen_by_adm: dict[str, list] = {}
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
        chosen_by_adm[h] = chosen
    return chosen_by_adm


def write_split(rows: list, out: Path, seed: int, label_name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    pos = sum(r["label"] for r in rows)
    print(f"\n[{label_name}] examples: {len(rows):,} | positives: {pos:,} "
          f"({pos/max(len(rows),1):.2%})")

    subs = sorted({r["subject_id"] for r in rows})
    random.Random(seed).shuffle(subs)
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
        path = out / f"{k}.jsonl"
        with path.open("w") as f:
            for x in v:
                f.write(json.dumps(x) + "\n")
        print(f"  {k:<6}{len(v):>7,} examples  {p:>6,} pos ({p/max(len(v),1):.2%})  "
              f"{len({x['subject_id'] for x in v}):>6,} subjects")

    print(f"segment roles ({label_name}, discharge summaries absent by construction):")
    for k, v in role_hist.most_common():
        print(f"  {k:<14}{v:>9,}")

    (out / "meta.json").write_text(json.dumps({
        "label": label_name,
        "n": len(rows), "positives": pos,
        "role_hist": dict(role_hist),
        "splits": {k: {"n": len(v), "pos": sum(x["label"] for x in v)}
                   for k, v in split.items()}}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--notes", default=NOTEEVENTS)
    ap.add_argument("--admissions", default=ADMISSIONS)
    ap.add_argument("--window-hours", type=int, default=48)
    ap.add_argument("--max-segments", type=int, default=8)
    ap.add_argument("--chars-per-segment", type=int, default=1500)
    ap.add_argument("--min-segments", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--readmit-days", type=int, default=30)
    ap.add_argument("--los-days", type=int, default=7)
    ap.add_argument("--label", choices=["readmit30", "los7", "both"], default="both")
    args = ap.parse_args()

    # ---- pass 1: read ALL admissions (unfiltered) to know, for every
    # subject, the full timeline of admissions. This is required to answer
    # "is there a later admission within 30 days of THIS discharge" for
    # admissions that themselves might not pass the 48h-bundle gate (a
    # subject's readmission can be short, or end in death, or whatever --
    # none of that matters, only that it happened). Using the unfiltered
    # table for this lookup, rather than only the 48h-eligible subset, keeps
    # the "does a later admission exist" check independent of any of the
    # eligibility gates applied to the INDEX admission below.
    all_adm: dict[str, dict] = {}
    by_subject_admits: dict[str, list] = defaultdict(list)
    with open(args.admissions, newline="") as f:
        for r in csv.DictReader(f):
            t0 = parse(r["ADMITTIME"])
            dc = parse(r["DISCHTIME"])
            if t0 is None:
                continue
            rec = {
                "subject_id": r["SUBJECT_ID"].strip(),
                "t0": t0,
                "discharge": dc,
                "death": parse(r.get("DEATHTIME")),
                "expire_flag": int(r["HOSPITAL_EXPIRE_FLAG"]),
            }
            all_adm[r["HADM_ID"].strip()] = rec
            if dc is not None:
                by_subject_admits[rec["subject_id"]].append((t0, dc))
    for s in by_subject_admits:
        by_subject_admits[s].sort()
    print(f"admissions with ADMITTIME: {len(all_adm):,}")

    # ---- shared eligibility gate, identical to build_task.py rule 3: keep
    # only admissions still ongoing (not discharged, not dead) at the end of
    # the 48h window. Computed from ADMITTIME/DISCHTIME/DEATHTIME alone, the
    # same three fields for every admission regardless of which of the two
    # labels below will be attached to it -- no label is consulted yet.
    cutoff = timedelta(hours=args.window_hours)
    n_before = len(all_adm)
    elig = {h: a for h, a in all_adm.items()
            if (a["discharge"] is None or a["discharge"] > a["t0"] + cutoff)
            and (a["death"] is None or a["death"] > a["t0"] + cutoff)}
    print(f"admissions still ongoing at {args.window_hours}h: {len(elig):,} "
          f"(dropped {n_before - len(elig):,} that ended inside the window)")

    # ---- build the SAME note bundles for every eligible admission, once.
    # Both tasks below read from this single dict; only the label (and, for
    # readmission, one extra label-INDEPENDENT exclusion) differs.
    chosen_by_adm = build_bundles(args, elig)
    print(f"admissions with a usable bundle (>= {args.min_segments} segments, "
          f">= 2 roles): {len(chosen_by_adm):,}")

    # ================= TASK B: 30-day readmission =================
    if args.label in ("readmit30", "both"):
        rows = []
        n_death_excluded = 0
        for h, chosen in chosen_by_adm.items():
            a = elig[h]
            # Label-independent exclusion: THIS admission's own recorded
            # outcome (not the future admission) says it ended in death, so
            # readmission is not defined. Computable from fields already
            # read for the eligibility gate above.
            if a["expire_flag"] == 1 or a["death"] is not None:
                n_death_excluded += 1
                continue
            dc = a["discharge"]
            future = [t for (t, _) in by_subject_admits[a["subject_id"]] if t > dc]
            label = int(any((t - dc) <= timedelta(days=args.readmit_days) for t in future))
            rows.append({"hadm_id": h, "subject_id": a["subject_id"], "label": label,
                         "segments": [{"role": n["role"], "delta_t_hours": n["dt"],
                                       "text": n["text"]} for n in chosen]})
        print(f"\n[readmit{args.readmit_days}] excluded {n_death_excluded:,} "
              f"admissions ending in death (in-admission or later in this stay)")
        write_split(rows, OUT_ROOT / "task_readmit30", args.seed,
                    f"readmit{args.readmit_days}")

    # ================= TASK C: prolonged length of stay =================
    if args.label in ("los7", "both"):
        rows = []
        n_no_discharge = 0
        for h, chosen in chosen_by_adm.items():
            a = elig[h]
            if a["discharge"] is None:
                n_no_discharge += 1
                continue
            los_days = (a["discharge"] - a["t0"]).total_seconds() / 86400.0
            label = int(los_days > args.los_days)
            rows.append({"hadm_id": h, "subject_id": a["subject_id"], "label": label,
                         "segments": [{"role": n["role"], "delta_t_hours": n["dt"],
                                       "text": n["text"]} for n in chosen]})
        if n_no_discharge:
            print(f"\n[los{args.los_days}] excluded {n_no_discharge:,} admissions "
                  f"with no DISCHTIME")
        write_split(rows, OUT_ROOT / "task_los7", args.seed, f"los{args.los_days}")


if __name__ == "__main__":
    main()
