"""Build multi-provenance note bundles for MLM PRETRAINING from MIMIC-III.

A bundle is one admission's notes from several roles, packed into ONE sequence.
This is the only input shape under which the authority bias is not a no-op:
tests/test_invariants.py proves the bias is exactly inert when every token
shares a role.

CONTAMINATION FIX (was: independent split, no role/window discipline).
------------------------------------------------------------------------
This corpus used to be built with its OWN independent SUBJECT_ID shuffle --
a different partition from scripts/build_task.py's downstream mortality task
-- and with no discharge-summary exclusion. Discharge summaries state the
outcome verbatim, so the pretraining corpus (and the 32k tokenizer fit on it)
contained outcome-revealing text belonging to patients in the task's val and
test sets.

The fix has two parts, both enforced here, not just documented:

  1. SUBJECT UNIVERSE = the task's TRAIN split, and ONLY the train split.
     local_only/task/{train,val,test}.jsonl is read directly (via
     src.data.lineage.hash_task_splits) to get the SUBJECT_ID -> split
     assignment; only rows whose SUBJECT_ID is in the TASK's train set are
     ever added to `by_adm`. Val/test subjects are never read into memory
     here, so the bundle-level train/val split below (used only to monitor
     MLM loss during pretraining) is nested entirely inside the task's train
     partition and cannot leak into the task's val or test.
  2. DISCHARGE SUMMARIES ARE EXCLUDED, same OUTCOME_EXCLUDED_ROLES rule as
     scripts/build_task.py -- notes are dropped by role before they are ever
     appended to `by_adm`.

The resulting corpus is hash-stamped: meta.json records the sha256 of the
task split it was derived from (`task_split_hash`). scripts/train_tokenizer.py
copies that hash forward into a lineage.json alongside the trained vocab, and
scripts/pretrain.py recomputes BOTH hashes fresh at startup and refuses to run
if either is stale -- see src/data/lineage.py's module docstring.

Local only. Writes to local_only/bundles/ (gitignored) by default. Prints
aggregates only, never note text.
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
from src.data.lineage import hash_task_splits  # noqa: E402
from src.data.roles import OUTCOME_EXCLUDED_ROLES, ROLE_NAMES, role_of  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
NOTEEVENTS = os.environ.get("MIMIC_NOTEEVENTS_PATH", "data/NOTEEVENTS.csv.gz")
DEFAULT_TASK_DIR = ROOT / "local_only" / "task"
DEFAULT_OUT = ROOT / "local_only" / "bundles"

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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", default=NOTEEVENTS)
    ap.add_argument("--task-dir", default=str(DEFAULT_TASK_DIR),
                    help="downstream task split to derive the pretraining subject "
                         "universe from (must already exist -- run build_task.py first)")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help="override for smoke-testing into a scratch dir without "
                         "touching the corpus a live pretrain/tokenizer job may be reading")
    ap.add_argument("--max-admissions", type=int, default=20_000)
    ap.add_argument("--min-roles", type=int, default=2,
                    help="minimum distinct NON-discharge roles per admission")
    ap.add_argument("--max-segments", type=int, default=6)
    ap.add_argument("--chars-per-segment", type=int, default=1200)
    ap.add_argument("--val-frac", type=float, default=0.1,
                    help="fraction of TASK-train subjects held out as the "
                         "pretraining corpus's own internal val split (for "
                         "monitoring MLM loss only -- never the task's val/test)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    task_dir = Path(args.task_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    task_hash, task_counts = hash_task_splits(task_dir)
    train_subjects_all = set()
    with (task_dir / "train.jsonl").open() as f:
        for line in f:
            train_subjects_all.add(json.loads(line)["subject_id"])
    print(f"task split hash {task_hash} | subjects: {task_counts} "
          f"(source: {task_dir})")
    print(f"pretraining subject universe = task TRAIN split only: "
          f"{len(train_subjects_all):,} subjects. val/test subjects "
          f"({task_counts['val']:,} + {task_counts['test']:,}) are never read.")

    by_adm: dict[str, list[dict]] = defaultdict(list)
    subj_of: dict[str, str] = {}
    n_read = n_kept = n_drop_subject = n_drop_role = 0

    with gzip.open(args.path, "rt", newline="") as f:
        for row in csv.DictReader(f):
            n_read += 1
            if (row.get("ISERROR") or "").strip():
                continue
            hadm = (row.get("HADM_ID") or "").strip()
            if not hadm:
                continue
            subj = (row.get("SUBJECT_ID") or "").strip()
            if subj not in train_subjects_all:            # THE SPLIT FIX
                n_drop_subject += 1
                continue
            r = role_of(row.get("CATEGORY"))
            if r in OUTCOME_EXCLUDED_ROLES:                # THE ROLE FIX
                n_drop_role += 1
                continue
            ts = parse_ts(row)
            if ts is None:
                continue
            txt = (row.get("TEXT") or "").strip()
            if len(txt) < 150:
                continue
            if len(by_adm) >= args.max_admissions and hadm not in by_adm:
                continue
            by_adm[hadm].append({"role": r, "ts": ts, "text": txt[: args.chars_per_segment]})
            subj_of[hadm] = subj
            n_kept += 1
            if n_read % 250_000 == 0:
                print(f"  ...read {n_read:,} rows, {len(by_adm):,} admissions", flush=True)

    print(f"\nread {n_read:,} rows, kept {n_kept:,} notes across {len(by_adm):,} admissions "
          f"(dropped {n_drop_subject:,} rows outside the task train split, "
          f"{n_drop_role:,} discharge-summary rows)")

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

    print(f"bundles with >={args.min_roles} distinct non-discharge roles: {len(bundles):,}")

    # Internal train/val split for monitoring MLM loss ONLY -- nested entirely
    # inside the task's train-subject set, so it cannot touch the task's val
    # or test. No "test" split is produced: this corpus is never itself
    # evaluated on a locked test set, and a bundles/test.jsonl that nobody
    # reads is just another way to accidentally leak later.
    subjects = sorted({b["subject_id"] for b in bundles})
    assert set(subjects) <= train_subjects_all, (
        "bundle subject leaked outside the task train split -- this should be "
        "impossible given the per-row filter above; refusing to write output")
    rng = random.Random(args.seed)
    rng.shuffle(subjects)
    n_val = max(1, int(round(args.val_frac * len(subjects)))) if subjects else 0
    val_subjects = set(subjects[:n_val])
    train_subjects = set(subjects[n_val:])
    splits = {"train": [], "val": []}
    for b in bundles:
        splits["train" if b["subject_id"] in train_subjects else "val"].append(b)

    assert not ({b["subject_id"] for b in splits["train"]} &
                {b["subject_id"] for b in splits["val"]}), \
        "SUBJECT_ID leaked between pretraining-internal train/val"

    role_hist: dict[str, int] = defaultdict(int)
    seg_counts = []
    for b in bundles:
        seg_counts.append(len(b["segments"]))
        for s in b["segments"]:
            role_hist[ROLE_NAMES[s["role"]]] += 1

    for name, items in splits.items():
        p = out / f"{name}.jsonl"
        with p.open("w") as f:
            for b in items:
                f.write(json.dumps(b) + "\n")
        print(f"  {name:<6} {len(items):>8,} bundles  "
              f"{len({b['subject_id'] for b in items}):>7,} subjects  -> {p.name}")

    mean_seg = sum(seg_counts) / len(seg_counts) if seg_counts else 0.0
    print(f"\nmean segments/bundle: {mean_seg:.2f}")
    print("segment role distribution (discharge absent by construction):")
    for k, v in sorted(role_hist.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<14}{v:>9,}")

    meta = {
        "n_bundles": len(bundles), "mean_segments": mean_seg,
        "role_hist": dict(role_hist),
        "excluded_roles": [ROLE_NAMES[r] for r in OUTCOME_EXCLUDED_ROLES],
        "splits": {k: len(v) for k, v in splits.items()},
        "derived_from_task_dir": str(task_dir),
        "task_split_hash": task_hash,
        "task_split_counts": task_counts,
        "pretrain_val_frac": args.val_frac,
        "seed": args.seed,
        "contamination_fix": (
            "subjects restricted to the task TRAIN split only "
            "(val/test subjects never read); discharge-summary notes "
            "excluded before bundling; see src/data/lineage.py"
        ),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {out}/{{train,val}}.jsonl + meta.json  (task_split_hash={task_hash})")


if __name__ == "__main__":
    main()
