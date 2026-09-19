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
# The side-strand appendices are moving to a companion document built from files
# named msu_sup*.tex (see verify_thesis.py for the fuller account of why: a
# second document nothing checks is a recorded project failure). A numeric
# claim printed there needs the same trace to a register key and a source file
# as one printed in a chapter, or Appendix C's promise -- every number answered
# with a file it came from -- is false for half the thesis.
SUPPLEMENT = sorted(THESIS.glob("msu_sup*.tex"))
ALL_CHAPTERS = CHAPTERS + SUPPLEMENT

# esc() and esc_breakable() are defined once, in make_appendices.py, and imported
# from there rather than redefined here: two independent escaping paths is how a
# raw artifact path reached this appendix before (make_appendices.py's redaction
# lived only in its own copy of esc(), and this script never called it).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from make_appendices import esc, esc_breakable

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
    """Build two indexes, keyed by the string a register value would render as.

    idx is the plain form: the literal must equal the stored value at the
    printed precision, nothing scaled. idx_pct is the same value times 100,
    kept separate on purpose. The old code folded the times-100 form into the
    same set of forms for every value unconditionally, which meant a register
    entry of 0.04 rendered a candidate string "4.00" alongside its own honest
    "0.04", and a passing rate of 0.05 rendered "5.00" too. That is how a
    G_253 over-calling ratio of 4.00 or 5.00 (a real value at that precision)
    collided with an unrelated precision or rate whose only connection was
    the coincidence of one being the other divided by 100. The percent form
    is only ever consulted by the caller when the literal in the prose is
    actually written as a percentage (a trailing \\% or the word "percent"),
    so a value is never scaled just because scaling happens to land on
    something else in the register.
    """
    idx, idx_pct = {}, {}
    for key, rec in FACTS.items():
        # Withdrawn and superseded values are held for the record only. Resolving a
        # prose digit to one would print a withdrawn (including hosted-channel) value
        # into the generated trace appendix, which is tracked and public.
        if key.startswith("withdrawn."):
            continue
        v = rec["value"]
        # A register entry can hold a Wilson interval or another pair as a
        # two-element list, which is a legitimate value shape this function
        # never rendered a string for. The old guard excluded bool, None and
        # str and then assumed everything left was a single int or float, so
        # a list reached abs(v) below and crashed the whole trace on the first
        # such key, one \\input away from every occurrence in the chapters
        # going untraced instead of just that one register entry. Excluding by
        # what the rest of this function can actually render, rather than by
        # what it happens to have seen before, does not crash on the next
        # shape either.
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        if isinstance(v, int):
            for f in {str(v), f"{v:,}"}:
                idx.setdefault(f, []).append(key)
            continue
        forms, pct_forms = set(), set()
        for val in (v, abs(v)):
            forms |= {f"{val:.{p}f}" for p in range(7)} | {str(val)}
            pct_forms |= {f"{val * 100:.{p}f}" for p in range(4)}
        for f in forms:
            idx.setdefault(f, []).append(key)
        for f in pct_forms:
            idx_pct.setdefault(f, []).append(key)
    return idx, idx_pct


# A literal is read as a percentage only when a percent marker sits next to it
# in the source, not merely somewhere in the same sentence: table rows pack a
# count and two percentages on one line ("134 & 2.24\% & 97.01\%"), and without
# a boundary the count would go looking for a times-100 match too. The window
# is cut short at a table cell or row break, a sentence-ending period, so a
# percent sign two cells over cannot leak backward onto an unrelated count.
PCT_WINDOW = 24
PCT_BOUNDARY = re.compile(r"\\\\|&|\.\s")
PCT_MARKER = re.compile(r"\\%|\bpercent\b")


def is_percent_context(text, end):
    window = text[end:end + PCT_WINDOW]
    boundary = PCT_BOUNDARY.search(window)
    if boundary:
        window = window[:boundary.start()]
    return bool(PCT_MARKER.search(window))


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
    # Layout lengths are not claims. A column width such as p{0.18\linewidth}
    # was being traced as a quantity and matched to whatever register value
    # happened to round to it, which inflated the counts Appendix B prints.
    s = re.sub(r"[\d.]+\s*\\(linewidth|textwidth|baselineskip)", " ", s)
    s = re.sub(r"\\(setlength|addtolength|vspace\*?|hspace\*?)\{[^}]*\}(\{[^}]*\})?", " ", s)
    s = re.sub(r"\{[\d.]+(in|pt|em|ex|cm|mm)\}", " ", s)
    return s


def main():
    idx, idx_pct = index_by_value()
    records, untraced = [], []

    for f in ALL_CHAPTERS:
        raw = f.read_text()
        text = clean(raw)
        for m in re.finditer(r"(?<![\w.])\d+(?:,\d{3})*(?:\.\d+)?(?![\w])", text):
            tok = m.group(0)
            bare = tok.replace(",", "")
            if tok in EXEMPT or bare in EXEMPT or re.fullmatch(r"(19|20)\d\d", bare):
                continue
            keys = idx.get(tok) or idx.get(bare) or []
            # The percent-scaled index is only consulted when the literal itself
            # is written as a percentage next to this token; see is_percent_context.
            # Candidates found this way are appended, not substituted, so a value
            # that is genuinely stored both ways (rare, but ablation.*.pct_of_
            # segment_index keys are pre-scaled) is not lost either direction.
            if is_percent_context(text, m.end()):
                pct_keys = idx_pct.get(tok) or idx_pct.get(bare) or []
                keys = list(dict.fromkeys(keys + [k for k in pct_keys if k not in keys]))
            rec = {"chapter": f.name, "value": tok,
                   "sentence": sentence_around(text, m.start())}
            if keys:
                # A value can be shared by several registered quantities. Prefer the
                # key whose name shares a word with the surrounding sentence; if
                # nothing matches, record the ambiguity rather than guessing, because
                # a confidently wrong attribution is worse than a declared uncertain
                # one in an appendix whose whole purpose is provenance.
                ctx = rec["sentence"].lower()
                # The key name breaks ties: two candidates scoring alike used to come
                # out in set order, so a rerun could reorder other_candidates and
                # produce a diff with no change of meaning.
                scored = sorted(keys, key=lambda k: (
                    -sum(1 for tok in re.split(r"[._]", k.lower()) if len(tok) > 3 and tok in ctx),
                    len(k), k))
                k = scored[0]
                rec["register_key"] = k
                rec["artifact"] = FACTS[k]["artifact"]
                rec["n_candidate_keys"] = len(keys)
                # "unique" means exactly one registered value renders to this literal:
                # no guessing happened. Anything else is a guess among competing
                # candidates, even when the winning candidate's name happens to share
                # a word with the sentence, and that word overlap used to be treated
                # as confirmation and skip the indicative marker below. That is
                # precisely how the 4.00 row for decontam_gold.aggregate.precision.min
                # got printed with no marker: the sentence said "gold set", the key
                # was named "decontam_gold", and the shared word "gold" was read as a
                # match despite three other candidates and despite the two having
                # nothing to do with each other. A literal with more than one
                # candidate is not unique no matter how it was picked, so it is always
                # marked here.
                rec["attribution"] = "unique" if len(keys) == 1 else "ambiguous"
                if len(keys) > 1:
                    rec["other_candidates"] = [x for x in scored[1:6]]
                records.append(rec)
            else:
                untraced.append(rec)

    # ---- Appendix C ----
    seen, rows = set(), []
    for r in sorted(records, key=lambda r: (r["chapter"], r["register_key"])):
        if r["register_key"] in seen:
            continue
        seen.add(r["register_key"])
        rows.append(r)
    n_amb = sum(1 for r in rows if r.get("attribution") == "ambiguous")
    n_uniq = sum(1 for r in rows if r.get("attribution") == "unique")

    # n_distinct, n_unique_rows and n_ambiguous_rows are computed once, here,
    # over the same dedup this appendix's table is built from, and written into
    # the artifact rather than left for another script to recompute from
    # records. make_appendices.py reads them to write the pointer paragraph in
    # msu_appB_pointer.tex, which used to be hand-typed and drifted from this
    # file's actual numbers; a second implementation of this dedup in that
    # script would be exactly the kind of two-computations-of-one-quantity this
    # project's other scripts go out of their way to avoid.
    out = {"n_traced": len(records), "n_untraced": len(untraced),
           "n_distinct": len(rows), "n_unique_rows": n_uniq,
           "n_ambiguous_rows": n_amb,
           "records": records, "untraced": untraced}
    (ROOT / "results" / "accord" / "CLAIM_TRACE.json").write_text(json.dumps(out, indent=2))

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
    tex += [r"\end{longtable}",
            "", f"Rows marked \\textsuperscript{{a}} ({n_amb} of {len(rows)}) carry a value that more "
            "than one registered quantity shares. The value is verified; the key shown is the "
            "best-scored guess among the candidates and should be read as indicative rather than "
            "definitive, even where the surrounding sentence shares a word with that guess's name, "
            "because a shared word is a tie-breaker between candidates, not proof the guess is right.",
            r"\doublespacing", ""]
    (THESIS / "msu_appC_trace.tex").write_text("\n".join(tex) + "\n")

    # ---- overclaim scan ----
    flags = []
    for f in ALL_CHAPTERS:
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
