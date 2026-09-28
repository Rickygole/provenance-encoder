#!/usr/bin/env python3
"""Keep Appendix B's provenance counts equal to what the tools compute.

Appendix B prints eight counts: the size of the record of reported quantities,
how many of its entries are citable, and what the claim trace found (scanned,
resolved, distinct, uniquely matched, matched among several). They move every
time a chapter is edited, and twice an audit found them stale because they had
been typed by hand. This script is the one place they come from.

    python3 accord/stats/sync_appendix_counts.py            rewrite the counts
    python3 accord/stats/sync_appendix_counts.py --check    exit 1 if stale

Run accord/stats/trace_claims.py first: the trace counts are read from its
output, not recomputed here.
"""
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
FACTS = ROOT / "results" / "accord" / "VERIFIED_FACTS.json"
TRACE = ROOT / "results" / "accord" / "CLAIM_TRACE.json"
APPENDIX = ROOT / "accord" / "thesis" / "msu_appB_pointer.tex"


def fmt(n):
    return f"{n:,}"


def counts():
    facts = json.loads(FACTS.read_text())
    trace = json.loads(TRACE.read_text())
    total = len(facts)
    withdrawn = sum(1 for k in facts if k.startswith("withdrawn"))
    traced = trace["n_traced"]
    untraced = trace["n_untraced"]
    return {
        "total": total,
        "citable": total - withdrawn,
        "scanned": traced + untraced,
        "resolved": traced,
        "distinct": trace["n_distinct"],
        "unique": trace["n_unique_rows"],
        "ambiguous": trace["n_ambiguous_rows"],
        "untraced": untraced,
        "quoted": trace["n_quoted"],
    }


# Each count is anchored by the words that follow or precede it in the
# appendix's one paragraph, so an edit to the prose that removes an anchor
# fails loudly here and cannot leave a stale number behind.
PATTERNS = [
    ("total", r"(the record of reported quantities: )([\d,]+)( entries)"),
    ("citable", r"(; )([\d,]+)( of those entries are citable)"),
    ("scanned", r"(scans the chapters for )([\d,]+)( numeric quantities)"),
    ("resolved", r"(and resolves )([\d,]+)( of\s+them)"),
    ("distinct", r"(of the )([\d,]+)( distinct registered quantities)"),
    ("unique", r"(the trace names, )([\d,]+)( are matched uniquely)"),
    ("ambiguous", r"(matched uniquely and )([\d,]+)( are matched among)"),
    ("untraced", r"(The trace lists )([\d,]+)( quantities as untraced)"),
    ("quoted", r"(A further )([\d,]+)( figures in Chapter 2)"),
]


def main():
    check = "--check" in sys.argv
    want = counts()
    text = APPENDIX.read_text()
    stale = []
    for name, pat in PATTERNS:
        m = re.search(pat, text)
        if not m:
            sys.exit(f"FAIL: Appendix B no longer contains the anchor for '{name}'; "
                     f"update PATTERNS in {pathlib.Path(__file__).name}")
        have = m.group(2)
        if have != fmt(want[name]):
            stale.append((name, have, fmt(want[name])))
            text = text[:m.start(2)] + fmt(want[name]) + text[m.end(2):]
    if check:
        if stale:
            for name, have, new in stale:
                print(f"   stale: {name} prints {have}, tools compute {new}")
            sys.exit(1)
        print("   pass: Appendix B's provenance counts match the tools")
        return
    if stale:
        APPENDIX.write_text(text)
    for name, have, new in stale:
        print(f"   updated {name}: {have} -> {new}")
    print(f"   {len(stale)} count(s) updated")


if __name__ == "__main__":
    main()
