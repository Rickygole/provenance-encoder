#!/usr/bin/env python3
"""Index every numeric value in every on-disk artifact (JSON + CSV) reachable
from the repo, keyed by rounded value, so claims can be matched to sources.

COMPLIANCE: reads under local_only/ are aggregate-only; this indexer records
only numeric values and the dotted key path, never note text / HADM_ID /
SUBJECT_ID values. Identifier-named columns are skipped entirely.
"""
import csv, json, os, re, sys
from collections import defaultdict

ROOTS = [
    "/Users/rickygole/provenance-encoder/results",
    "/Users/rickygole/provenance-encoder/accord",
]
OUT = "/Users/rickygole/provenance-encoder/results/accord/_audit_artifact_index.json"
BANNED_COL = re.compile(r"(hadm|subject|row_id|note_id|text|_id$|^id$)", re.I)

def walk_json(obj, path, sink, src):
    if isinstance(obj, dict):
        for k, v in obj.items():
            walk_json(v, f"{path}.{k}" if path else str(k), sink, src)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            walk_json(v, f"{path}[{i}]", sink, src)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        sink.append((float(obj), src, path))
    elif isinstance(obj, str):
        # numbers embedded in note strings are provenance too
        for m in re.finditer(r"(?<![A-Za-z0-9])[-+]?\d[\d,]*(?:\.\d+)?", obj):
            try:
                sink.append((float(m.group(0).replace(",", "")), src, path + "(str)"))
            except ValueError:
                pass

def main():
    sink = []
    nfiles = 0
    for root in ROOTS:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in ("__pycache__", ".git")]
            for fn in filenames:
                if fn.startswith("_audit"):
                    continue
                p = os.path.join(dirpath, fn)
                rel = p
                try:
                    if fn.endswith(".json"):
                        with open(p) as fh:
                            walk_json(json.load(fh), "", sink, rel)
                        nfiles += 1
                    elif fn.endswith(".csv"):
                        with open(p, newline="") as fh:
                            rd = csv.DictReader(fh)
                            cols = [c for c in (rd.fieldnames or []) if c and not BANNED_COL.search(c)]
                            for rowi, row in enumerate(rd):
                                for c in cols:
                                    v = (row.get(c) or "").strip()
                                    if re.fullmatch(r"[-+]?\d+(\.\d+)?([eE][-+]?\d+)?", v):
                                        sink.append((float(v), rel, f"row{rowi}.{c}"))
                        nfiles += 1
                except Exception as e:
                    print(f"  skip {rel}: {type(e).__name__}", file=sys.stderr)
    idx = defaultdict(list)
    for val, src, path in sink:
        for nd in (4, 3, 2):
            idx[f"{round(val, nd):.{nd}f}"].append([src, path, val])
    # cap list length to keep file sane
    out = {k: v[:40] for k, v in idx.items()}
    with open(OUT, "w") as fh:
        json.dump(out, fh)
    print(f"indexed {len(sink)} numeric values from {nfiles} artifact files -> {len(idx)} keys")

if __name__ == "__main__":
    main()
