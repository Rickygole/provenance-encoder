#!/usr/bin/env python3
"""Retire register entries whose artifact is not in the repository.

The register maps every quantity this thesis may print to the file it was read
from. Some entries name files that no longer exist in the tree: strands that
were dropped, superseded runs, and one coding exercise whose tables were never
committed. A live entry the reader cannot open is worse than no entry, because
the register's whole claim is that a number can be traced to a file.

This moves those entries under `withdrawn.artifact_not_in_repository`, the same
treatment `withdraw_hosted.py` gives values that may not be cited. Nothing is
deleted: the key, the value and the path it named are kept, so the record still
says what was measured and where it lived. The entries simply stop being
citable, and `verify_thesis.py`'s cited-artifact check stops having to make an
exception for them.

Entries whose "artifact" field is a derivation rather than a path (they say how
a number follows from two other registered numbers) are left alone: there is no
file for them to be missing.

Run from the repository root. Idempotent: rerunning moves nothing new.
"""
import collections
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
FACTS = ROOT / "results" / "accord" / "VERIFIED_FACTS.json"
PREFIX = "withdrawn.artifact_not_in_repository"


def leading_path(artifact):
    """The path part of an artifact field, or None if it names no file.

    Artifact fields come in three shapes: a plain path, a path with a JSON
    pointer after "::", and a path followed by a comma and a note about which
    field or which derivation was used. Anything that does not look like a path
    (no separator, or an opening word like "derived") names no file.
    """
    if not isinstance(artifact, str):
        return None
    head = artifact.split("::")[0].split(", ")[0].strip()
    if not head or "/" not in head:
        return None
    if head.lower().startswith(("derived", "computed", "the ")):
        return None
    if head.startswith("[") or head.startswith("~"):
        # A placeholder for storage outside the repository; already disclosed
        # as unverifiable elsewhere in the register.
        return None
    return head


def main():
    facts = json.loads(FACTS.read_text(),
                       object_pairs_hook=collections.OrderedDict)
    moved = []
    for key in list(facts):
        if key.startswith("withdrawn"):
            continue
        entry = facts[key]
        if not isinstance(entry, dict) or "artifact" not in entry:
            continue
        path = leading_path(entry["artifact"])
        if path is None or (ROOT / path).exists():
            continue
        facts[f"{PREFIX}.{key}"] = facts.pop(key)
        moved.append((key, path))

    if moved:
        FACTS.write_text(json.dumps(facts, indent=1) + "\n")
    print(f"retired {len(moved)} register entries naming a file not in the tree")
    by_path = collections.Counter(path for _, path in moved)
    for path, n in by_path.most_common():
        print(f"   {n:4d}  {path}")


if __name__ == "__main__":
    main()
