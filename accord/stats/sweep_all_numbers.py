"""Extract every numeric literal from the six MSU chapters and try to match it to a
registered value (at any rounding) OR to a raw value inside any results artifact.
Numbers that match nothing are printed for manual adjudication.
"""
import json, re, glob, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
THESIS = ROOT/"accord/thesis"
REG = json.load(open(ROOT/"results/accord/VERIFIED_FACTS.json"))

# ---- pool of every scalar in every registered artifact + the register itself ----
pool = {}          # value -> set of "file:path"
def add(v, where):
    if isinstance(v, bool): return
    if isinstance(v, (int, float)):
        f = float(v)
        if f != f or f in (float('inf'), float('-inf')): return  # NaN/inf break bisect
        pool.setdefault(round(f, 10), set()).add(where)
def walk(o, where, path=""):
    if isinstance(o, dict):
        for k, v in o.items(): walk(v, where, path+"."+str(k))
    elif isinstance(o, list):
        for i, v in enumerate(o): walk(v, where, path+f"[{i}]")
    else:
        add(o, where+path)

FILES = []
for base in [ROOT/"results", ROOT/"accord/prevalence_replication/results"]:
    FILES += glob.glob(str(base)+"/**/*.json", recursive=True)
for f in FILES:
    try: walk(json.load(open(f)), os.path.relpath(f, ROOT))
    except Exception: pass
for k, v in REG.items():
    add(v.get("value"), "REGISTER:"+k)

vals = sorted(pool)
import bisect
def match(x, dp):
    """is there any pooled value that rounds to x at dp decimals?"""
    tol = 0.5 * 10**(-dp)
    i = bisect.bisect_left(vals, x - tol)
    hits = []
    while i < len(vals) and vals[i] <= x + tol:
        hits.append(vals[i]); i += 1
    return hits

NUM = re.compile(r'(?<![\w\\])(-?\d[\d,]*(?:\.\d+)?)')
SKIP_CTX = re.compile(r'\\(cite|ref|label|includegraphics|input|section|subsection|paragraph|chapter|caption)')
unmatched = []
checked = 0
for f in sorted(glob.glob(str(THESIS/"msu_ch*.tex"))):
    for ln, line in enumerate(open(f), 1):
        s = line
        if s.lstrip().startswith('%'): continue
        s = re.sub(r'\\cite[a-z]*\{[^}]*\}', ' ', s)
        s = re.sub(r'\\(label|ref|includegraphics|input)\{[^}]*\}', ' ', s)
        s = re.sub(r'\\[a-zA-Z]+', ' ', s)
        for m in NUM.finditer(s):
            raw = m.group(1)
            x = float(raw.replace(',', ''))
            checked += 1
            # decimals printed
            dp = len(raw.split('.')[1]) if '.' in raw else 0
            hits = match(x, dp)
            # also try percent form (value stored as fraction)
            hits2 = match(x/100.0, dp+2) if dp >= 0 else []
            if not hits and not hits2:
                unmatched.append((os.path.basename(f), ln, raw, line.strip()[:150]))
print("numeric literals scanned:", checked)
print("unmatched:", len(unmatched))
for u in unmatched:
    print(f"  {u[0]} L{u[1]}  [{u[2]}]  :: {u[3]}")
