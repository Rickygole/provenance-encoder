"""Hashes that enforce the build_task -> build_bundles -> train_tokenizer ->
pretrain dependency chain.

The pretraining MLM corpus used to be built by scripts/build_bundles.py with
its OWN independent SUBJECT_ID shuffle -- a different partition from
scripts/build_task.py's downstream-task split. That let discharge summaries
(which state the outcome verbatim) belonging to the task's val/test patients
into the pretraining corpus and the 32k tokenizer fit on it.

The fix: scripts/build_bundles.py now derives its subject universe directly
from local_only/task/{train,val,test}.jsonl (train-split subjects only) and
drops OUTCOME_EXCLUDED_ROLES notes, same as build_task.py. This module makes
that dependency checkable rather than assumed: every stage hashes the thing it
was built from, and the next stage recomputes that hash FRESH (never from a
cache) and refuses to run on a mismatch. Nothing here reads note text.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path


def hash_task_splits(task_dir: Path) -> tuple[str, dict[str, int]]:
    """Sha256 (16 hex chars) over the SUBJECT_ID -> split assignment implied by
    task_dir/{train,val,test}.jsonl, plus the per-split subject counts.

    Reads only the "subject_id" field of each row -- no note text. Recomputed
    fresh on every call, so it can never be stale relative to what is actually
    on disk right now.
    """
    task_dir = Path(task_dir)
    splits: dict[str, set[str]] = {}
    for name in ("train", "val", "test"):
        p = task_dir / f"{name}.jsonl"
        if not p.exists():
            raise FileNotFoundError(
                f"{p} is missing. The pretraining corpus is derived from the "
                f"downstream task's split assignment and cannot be built "
                f"without it -- run scripts/build_task.py first."
            )
        subs: set[str] = set()
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                subs.add(json.loads(line)["subject_id"])
        splits[name] = subs

    leaks = {
        "train/val": splits["train"] & splits["val"],
        "train/test": splits["train"] & splits["test"],
        "val/test": splits["val"] & splits["test"],
    }
    bad = {k: len(v) for k, v in leaks.items() if v}
    if bad:
        raise AssertionError(
            f"SUBJECT_ID leaks across task splits on disk: {bad}. Refusing to "
            f"derive a pretraining corpus from a contaminated task split."
        )

    canonical = {k: sorted(v) for k, v in splits.items()}
    digest = hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()[:16]
    counts = {k: len(v) for k, v in splits.items()}
    return digest, counts


def hash_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Sha256 (16 hex chars) of a file's bytes, streamed so it stays cheap on
    the multi-ten-MB bundle corpus files."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Downstream-task-split lineage (accord/CORRECTIONS.md C32).
#
# local_only/task/*.jsonl was silently rebuilt on 2026-08-04 16:54 -- same
# filenames, different content (25,527 train rows became 23,846). Nothing
# recorded what a finetuning run actually trained on, so concat_s0 (old split)
# and concat_s1 (rebuilt split) were compared as if they differed only by
# seed. The pretraining chain already had exactly this kind of check (see
# hash_task_splits above and scripts/pretrain.py's check_lineage); this
# extends the same fresh-hash-vs-pinned-reference pattern to the downstream
# task runs in scripts/finetune_ours.py and scripts/finetune_pubmedbert.py.
# ---------------------------------------------------------------------------


def task_dir_lineage(task_dir: Path) -> dict:
    """Full data-lineage stamp for a downstream-task split directory: the
    subject-assignment hash from hash_task_splits, plus per-split sha256
    content hash (via hash_file -- no second hasher), row count, positive
    count, and prevalence for train/val/test, plus the task directory's
    mtime. Every finetuning run should embed this in its output record: it
    is the field that would have caught the Aug 4 rebuild silently changing
    n_train under an unchanged filename. Recomputed FRESH on every call --
    never cached -- so it can never be stale relative to what is on disk.

    Reads only "subject_id" (via hash_task_splits) and "label" per row --
    never note text.
    """
    task_dir = Path(task_dir)
    split_hash, split_counts = hash_task_splits(task_dir)
    # The directory inode's own mtime is NOT what we want here: build_task.py
    # opens train/val/test.jsonl with "w" and overwrites their content in
    # place, which changes each file's mtime but NOT the directory entry
    # (same filenames, no add/remove), so task_dir.stat().st_mtime silently
    # keeps the timestamp of whenever the directory was first created. That
    # is exactly the kind of staleness this module exists to catch, so the
    # stamp is the newest mtime among the files actually in the split
    # instead -- the one that would have shown 2026-08-04 16:54, not 09:50.
    split_paths = [task_dir / f"{name}.jsonl" for name in ("train", "val", "test")]
    split_paths.append(task_dir / "meta.json")
    mtimes = [p.stat().st_mtime for p in split_paths if p.exists()]
    task_dir_mtime = max(mtimes) if mtimes else task_dir.stat().st_mtime
    out: dict = {
        "task_dir_mtime": datetime.fromtimestamp(task_dir_mtime).isoformat(),
        "task_split_hash": split_hash,
        "task_split_subject_counts": split_counts,
    }
    for name in ("train", "val", "test"):
        p = task_dir / f"{name}.jsonl"
        if not p.exists():
            out[f"{name}_content_hash"] = None
            out[f"{name}_n"] = None
            out[f"{name}_n_pos"] = None
            out[f"{name}_prevalence"] = None
            continue
        out[f"{name}_content_hash"] = hash_file(p)
        n = pos = 0
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                n += 1
                pos += int(json.loads(line)["label"])
        out[f"{name}_n"] = n
        out[f"{name}_n_pos"] = pos
        out[f"{name}_prevalence"] = (pos / n) if n else None
    return out


def check_task_lineage(lineage: dict, pin_path: Path, repin: bool = False) -> None:
    """Refuse to start (or, with --repin-task-lineage, explicitly re-pin) a
    training run whose task_split_hash disagrees with the pinned reference
    at pin_path. Mirrors scripts/pretrain.py's check_lineage: the hash
    passed in is always freshly computed by task_dir_lineage (never cached),
    and comparison is against a reference committed to version control
    (results/task_lineage_pin.json is aggregate scalars only, no note text).

    - No pin file yet: bootstraps one from `lineage` and warns loudly --
      there is nothing to compare against on a cold start.
    - Pin matches: prints a one-line confirmation and returns.
    - Pin disagrees and repin=False (the default): raises SystemExit. This
      is the guard that did not exist for C32.
    - Pin disagrees and repin=True: overwrites the pin with `lineage` and
      proceeds -- the explicit, auditable way to accept an intentional
      rebuild of local_only/task.
    """
    pin_path = Path(pin_path)
    fresh_hash = lineage["task_split_hash"]

    def _write_pin() -> None:
        pin_path.parent.mkdir(parents=True, exist_ok=True)
        pin_path.write_text(json.dumps(
            {"task_split_hash": fresh_hash,
             "task_split_subject_counts": lineage["task_split_subject_counts"]},
            indent=2, sort_keys=True) + "\n")

    if not pin_path.exists():
        _write_pin()
        print(f"NOTICE: no pinned task-split reference at {pin_path} yet -- "
              f"pinning the CURRENT local_only/task split "
              f"(task_split_hash={fresh_hash}, subjects="
              f"{lineage['task_split_subject_counts']}) as the reference. "
              f"Future runs on a different split will be refused unless run "
              f"with --repin-task-lineage.")
        return

    pinned = json.loads(pin_path.read_text())
    stored_hash = pinned.get("task_split_hash")
    if stored_hash == fresh_hash:
        print(f"task lineage OK: task_split_hash={fresh_hash} matches pinned "
              f"reference at {pin_path}.")
        return

    if repin:
        _write_pin()
        print(f"RE-PINNED {pin_path}: task_split_hash was {stored_hash}, is "
              f"now {fresh_hash} (subjects={lineage['task_split_subject_counts']}). "
              f"--repin-task-lineage was passed, so the new split is accepted "
              f"as the reference and THIS run trains on it.")
        return

    raise SystemExit(
        f"STALE TASK SPLIT (accord/CORRECTIONS.md C32): local_only/task on "
        f"disk hashes to task_split_hash={fresh_hash}, but the pinned "
        f"reference at {pin_path} is {stored_hash}. This is exactly the "
        f"defect that let concat_s0 (trained on the old split) get compared "
        f"to concat_s1 (trained on a silently rebuilt split) as if they were "
        f"seed variance. Refusing to train. If local_only/task was rebuilt "
        f"on purpose, rerun with --repin-task-lineage to accept the new "
        f"split as the reference; otherwise find out why it changed before "
        f"training on it."
    )
