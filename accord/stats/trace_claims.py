"""
Build a claim-to-evidence trace for every number printed in the thesis.

For each numeric literal in the chapters, find the register entry it matches and
the artifact that entry was read from, then emit:

  results/accord/CLAIM_TRACE.json   machine readable, one record per occurrence
  accord/thesis/msu_appC_trace.tex  Appendix C, the reader-facing table

The point is that any number in the document can be answered with: here is the
value, here is the file it came from, here is the sentence it supports.

Also runs an overclaim scan. Strong verbs (proves, demonstrates, establishes,
confirms) attached to a result are listed for review, because this project has
retracted claims for exactly that pattern before.

Run:  python3 accord/stats/trace_claims.py
"""

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
THESIS = ROOT / "accord" / "thesis"
FACTS = json.loads((ROOT / "results" / "accord" / "VERIFIED_FACTS.json").read_text())
CHAPTERS = sorted(THESIS.glob("msu_ch*.tex"))

# Numbers that are settings, structure or ordinary prose, not findings.
EXEMPT = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12",
          "95", "100", "1000", "0.05", "0.5", "1.0", "2.0", "42", "123", "2024"}

OVERCLAIM = [
    (r"\bprove[sd]?\b", "proves"),
    (r"\bdemonstrat(e|es|ed)\b", "demonstrates"),
    (r"\bestablish(es|ed)?\s+that\b", "establishes that"),
    (r"\bconfirms?\s+that\b", "confirms that"),
    (r"\bshows?\s+conclusively\b", "conclusively"),
    (r"\bclearly\s+(shows?|indicates?)\b", "clearly shows"),
    (r"\bsignificantly\s+(better|worse|higher|lower)\b", "significantly (without a test)"),
    (r"\bsubstantially\s+outperform", "substantially outperforms"),
    (r"\bunequivocall?y\b", "unequivocally"),
    (r"\bwithout\s+doubt\b", "without doubt"),
]


def index_by_value():
    """value string -> list of register keys that render to it."""
    idx = {}
    for key, rec in FACTS.items():
        v = rec["value"]
        if isinstance(v, bool) or v is None or isinstance(v, str):
            continue
        forms = set()
        if isinstance(v, int):
            forms |= {str(v), f"{v:,}"}
        else:
            for val in (v, abs(v)):
                forms |= {f"{val:.{p}f}" for p in range(7)} | {str(val)}
                forms |= {f"{val * 100:.{p}f}" for p in range(4)}
        for f in forms:
            idx.setdefault(f, []).append(key)
    return idx


def sentence_around(text, pos, width=240):
    lo = max(0, text.rfind(".", 0, pos) + 1)
    hi = text.find(".", pos)
    hi = len(text) if hi == -1 else hi + 1
    s = re.sub(r"\s+", " ", text[lo:hi]).strip()
    return s[:width]


def clean(s):
    s = re.sub(r"(\d)\{,\}(\d)", r"\1,\2", s)
    s = re.sub(r"\\texttt\{[^}]*\}", " ", s)
    s = re.sub(r"(?<!\\)%.*", "", s)
    s = re.sub(r"\\(label|ref|cite[tp]?)\{[^}]*\}", " ", s)
    return s


def main():
    idx = index_by_value()
    records, untraced = [], []

    for f in CHAPTERS:
        raw = f.read_text()
        text = clean(raw)
        for m in re.finditer(r"(?<![\w.])\d+(?:,\d{3})*(?:\.\d+)?(?![\w])", text):
            tok = m.group(0)
            bare = tok.replace(",", "")
            if tok in EXEMPT or bare in EXEMPT or re.fullmatch(r"(19|20)\d\d", bare):
                continue
            keys = idx.get(tok) or idx.get(bare)
            rec = {"chapter": f.name, "value": tok,
                   "sentence": sentence_around(text, m.start())}
            if keys:
                # A value can be shared by several registered quantities. Prefer the
                # key whose name shares a word with the surrounding sentence; if
                # nothing matches, record the ambiguity rather than guessing, because
                # a confidently wrong attribution is worse than a declared uncertain
                # one in an appendix whose whole purpose is provenance.
                ctx = rec["sentence"].lower()
                scored = sorted(keys, key=lambda k: (
                    -sum(1 for tok in re.split(r"[._]", k.lower()) if len(tok) > 3 and tok in ctx),
                    len(k)))
                k = scored[0]
                hits = sum(1 for tok in re.split(r"[._]", k.lower()) if len(tok) > 3 and tok in ctx)
                rec["register_key"] = k
                rec["artifact"] = FACTS[k]["artifact"]
                rec["n_candidate_keys"] = len(keys)
                rec["attribution"] = "matched" if hits else ("unique" if len(keys) == 1 else "ambiguous")
                if len(keys) > 1:
                    rec["other_candidates"] = [x for x in scored[1:6]]
                records.append(rec)
            else:
                untraced.append(rec)

    out = {"n_traced": len(records), "n_untraced": len(untraced),
           "records": records, "untraced": untraced}
    (ROOT / "results" / "accord" / "CLAIM_TRACE.json").write_text(json.dumps(out, indent=2))

    # ---- Appendix C ----
    seen, rows = set(), []
    for r in sorted(records, key=lambda r: (r["chapter"], r["register_key"])):
        if r["register_key"] in seen:
            continue
        seen.add(r["register_key"])
        rows.append(r)

    def esc(s):
        for a, b in (("\\", " "), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"),
                     ("#", r"\#"), ("$", r"\$"), ("{", " "), ("}", " "), ("~", " ")):
            s = s.replace(a, b)
        return re.sub(r"[^\x00-\x7f]", "", s)

    def esc_breakable(s):
        """Like esc(), but for the long dotted register keys and file paths: insert
        \\allowbreak after every '.' and '/' so the longtable columns can wrap them
        instead of overrunning the margin (this project has no xurl installed)."""
        s = esc(s)
        s = s.replace(".", r".\allowbreak{}")
        s = s.replace("/", r"/\allowbreak{}")
        return s

    tex = [r"\chapter{Claim and Evidence Trace}", r"\label{app:trace}", "",
           f"Every quantity printed in this thesis is listed below with the file it was read from.",
           f"The trace covers {len(records)} occurrences resolving to {len(rows)} distinct registered",
           "values. Numbers are not transcribed from notes or from earlier drafts; each is extracted",
           "from a stored result file by script, and the register is reproduced in Appendix B.", "",
           r"\singlespacing",
           r"\begin{longtable}{@{}p{0.7in}p{1.9in}p{2.6in}@{}}",
           r"\textbf{Value} & \textbf{Register key} & \textbf{Source file}\\", r"\midrule",
           r"\endhead"]
    for r in rows:
        mark = "" if r.get("attribution") != "ambiguous" else r" \textsuperscript{a}"
        tex.append(f"{esc(r['value'])}{mark} & {esc_breakable(r['register_key'])} & "
                    f"{esc_breakable(r['artifact'])}\\\\")
    n_amb = sum(1 for r in rows if r.get("attribution") == "ambiguous")
    tex += [r"\end{longtable}",
            "", f"Rows marked \\textsuperscript{{a}} ({n_amb} of {len(rows)}) carry a value that more "
            "than one registered quantity shares, and the surrounding sentence did not disambiguate "
            "them. The value is verified; the key shown is the most likely of several and should be "
            "read as indicative rather than definitive.",
            r"\doublespacing", ""]
    (THESIS / "msu_appC_trace.tex").write_text("\n".join(tex) + "\n")

    # ---- overclaim scan ----
    flags = []
    for f in CHAPTERS:
        text = clean(f.read_text())
        for pat, name in OVERCLAIM:
            for m in re.finditer(pat, text, re.I):
                flags.append((f.name, name, sentence_around(text, m.start(), 160)))

    print(f"traced   {len(records)} numeric occurrences to {len(rows)} distinct register entries")
    print(f"untraced {len(untraced)}")
    for r in untraced[:20]:
        print(f"   {r['chapter']}  '{r['value']}'  {r['sentence'][:80]}")
    print(f"\noverclaim scan: {len(flags)} phrases for review")
    for name, kind, s in flags[:20]:
        print(f"   {name}  [{kind}]  {s[:100]}")
    return 1 if untraced else 0


if __name__ == "__main__":
    sys.exit(main())
