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
