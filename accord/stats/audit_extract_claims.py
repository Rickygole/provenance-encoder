#!/usr/bin/env python3
"""Extract every numeric literal from the thesis .tex chapters with file:line provenance.

Re-runnable. Writes a JSON list of {file,line,number,raw,context} records.
No number in the audit is typed by hand; all come from this extractor plus
recomputation scripts in the same directory.
"""
import json, os, re, sys

THESIS = "/Users/rickygole/provenance-encoder/accord/thesis"
OUT = "/Users/rickygole/provenance-encoder/results/accord/_audit_claims_raw.json"

# numbers like 1{,}978  0.5145  41.4\%  83.90  -0.007  20.08  8.9
NUM = re.compile(r"(?<![A-Za-z0-9])[-+]?\d[\d,]*(?:\{,\})?[\d,]*(?:\.\d+)?")

def norm(tok):
    t = tok.replace("{,}", "").replace(",", "")
    try:
        return float(t)
    except ValueError:
        return None

def main():
    recs = []
    for fn in sorted(os.listdir(THESIS)):
        if not fn.endswith(".tex"):
            continue
        path = os.path.join(THESIS, fn)
        with open(path) as fh:
            for i, line in enumerate(fh, 1):
                s = line.rstrip("\n")
                if s.lstrip().startswith("%"):
                    continue
                # strip latex label/ref/cite/include args which carry digits
                clean = re.sub(r"\\(label|ref|cite[a-z]*|includegraphics|input|usepackage)\s*(\[[^\]]*\])?\{[^}]*\}", " ", s)
                clean = re.sub(r"\\[a-zA-Z]+", " ", clean)
                for m in NUM.finditer(clean):
                    v = norm(m.group(0))
                    if v is None:
                        continue
                    recs.append({
                        "file": fn, "line": i, "raw": m.group(0), "value": v,
                        "context": s.strip()[:300],
                    })
    with open(OUT, "w") as fh:
        json.dump(recs, fh, indent=1)
    print(f"extracted {len(recs)} numeric literals from {THESIS}")
    from collections import Counter
    for k, c in sorted(Counter(r["file"] for r in recs).items()):
        print(f"  {k}: {c}")

if __name__ == "__main__":
    main()
