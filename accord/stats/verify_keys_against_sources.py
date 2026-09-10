"""Go behind the register: for every dotted key cited in the MSU thesis chapters,
(1) confirm the key is in results/accord/VERIFIED_FACTS.json,
(2) OPEN THE ARTIFACT the register names and confirm the value is actually there,
    under a path that matches the key, with that value.
Emits per-key resolution status. Nothing is typed by hand.
"""
import json, re, glob, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
THESIS = ROOT / "accord" / "thesis"
REG = json.load(open(ROOT / "results" / "accord" / "VERIFIED_FACTS.json"))

# ---- collect key citations with chapter+line ----
cites = []  # (file, line, key)
KEYRE = re.compile(r'\\texttt\{((?:[^{}\\]|\\_)+)\}')
for f in sorted(glob.glob(str(THESIS / "msu_*.tex"))):
    for i, line in enumerate(open(f), 1):
        for m in KEYRE.finditer(line):
            k = m.group(1).replace("\\_", "_")
            if "." in k and " " not in k and not re.search(r'\.(py|json|csv|tex|pdf|txt|md)$', k):
                cites.append((os.path.basename(f), i, k))

def load(p):
    p = str(p)
    if not os.path.isabs(p):
        p = str(ROOT / p)
    if not os.path.exists(p):
        return None
    if p.endswith(".json"):
        try: return json.load(open(p))
        except Exception as e: return {"__parse_error__": str(e)}
    return {"__not_json__": p}

def descend(obj, parts):
    """Try to walk parts through obj, allowing the leading namespace token(s) to be absent."""
    cur = obj
    for p in parts:
        if isinstance(cur, dict):
            if p in cur: cur = cur[p]; continue
            # numeric string index into a list
            return None
        if isinstance(cur, list):
            try: cur = cur[int(p)]; continue
            except Exception: return None
        return None
    return cur

def find_path(obj, parts):
    """Try dropping 0..n leading tokens, and also searching any subtree for the tail path."""
    for drop in range(len(parts)):
        got = descend(obj, parts[drop:])
        if got is not None and not isinstance(got, (dict, list)):
            return got, ".".join(parts[drop:]), "direct"
        if got is not None and isinstance(got, (dict, list)):
            return got, ".".join(parts[drop:]), "direct-container"
    # recursive: find any node whose path ends with the tail
    tail = parts[-1]
    hits = []
    def rec(o, path):
        if isinstance(o, dict):
            for k, v in o.items():
                rec(v, path + [str(k)])
        elif isinstance(o, list):
            for j, v in enumerate(o):
                rec(v, path + [str(j)])
        if path and path[-1] == tail:
            hits.append((".".join(path), o))
    rec(obj, [])
    hits = [h for h in hits if not isinstance(h[1], (dict, list))]
    if len(hits) == 1:
        return hits[0][1], hits[0][0], "recursive-unique"
    if len(hits) > 1:
        return [h[1] for h in hits], "|".join(h[0] for h in hits), "recursive-ambiguous"
    return None, None, "not-found"

def close(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if a == b: return True
        return abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))
    return a == b

records = []
seen = {}
for fn, ln, key in cites:
    rec = {"chapter": fn, "line": ln, "key": key}
    if key not in REG:
        rec["status"] = "KEY_NOT_IN_REGISTER"
        records.append(rec); continue
    entry = REG[key]
    art = entry.get("artifact", "")
    val = entry.get("value")
    rec["register_value"] = val
    rec["register_artifact"] = art
    if art.startswith("derived"):
        rec["status"] = "DERIVED_IN_REGISTER"
        records.append(rec); continue
    # artifact string may carry a trailing ", derived: ..." comment
    path = art.split(",")[0].strip()
    rec["artifact_path"] = path
    obj = load(path)
    if obj is None:
        rec["status"] = "ARTIFACT_MISSING"
        records.append(rec); continue
    if "__not_json__" in obj or "__parse_error__" in obj:
        rec["status"] = "ARTIFACT_UNPARSED"
        records.append(rec); continue
    got, gpath, how = find_path(obj, key.split("."))
    rec["source_path_in_file"] = gpath
    rec["source_value"] = got
    rec["resolution"] = how
    if how == "not-found":
        rec["status"] = "KEY_NOT_IN_ARTIFACT"
    elif how == "recursive-ambiguous":
        rec["status"] = "AMBIGUOUS_IN_ARTIFACT"
    else:
        if isinstance(got, (dict, list)):
            rec["status"] = "RESOLVES_TO_CONTAINER"
        elif close(got, val):
            rec["status"] = "SOURCE_CONFIRMS_REGISTER"
        else:
            # register may be a rounded copy
            try:
                if isinstance(got,(int,float)) and isinstance(val,(int,float)):
                    for nd in range(0, 8):
                        if round(got, nd) == val:
                            rec["status"] = "SOURCE_CONFIRMS_REGISTER_ROUNDED"
                            rec["register_rounding_dp"] = nd
                            break
                    else:
                        rec["status"] = "REGISTER_DISAGREES_WITH_SOURCE"
                else:
                    rec["status"] = "REGISTER_DISAGREES_WITH_SOURCE"
            except Exception:
                rec["status"] = "REGISTER_DISAGREES_WITH_SOURCE"
    records.append(rec)

from collections import Counter
c = Counter(r["status"] for r in records)
print(json.dumps(c, indent=1))
out = ROOT / "accord" / "stats" / "key_resolution.json"
out.write_text(json.dumps(records, indent=1))
print("wrote", out, len(records), "citations,", len(set(r['key'] for r in records)), "distinct keys")
for r in records:
    if r["status"] not in ("SOURCE_CONFIRMS_REGISTER", "SOURCE_CONFIRMS_REGISTER_ROUNDED"):
        print(r["status"], "|", r["chapter"], "L"+str(r["line"]), "|", r["key"], "|", r.get("register_value"), "|", str(r.get("source_value"))[:120], "|", r.get("artifact_path"))
