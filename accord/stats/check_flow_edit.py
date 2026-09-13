"""
Gate a readability pass over a thesis chapter against everything it is not
allowed to change.

Usage:  python3 accord/stats/check_flow_edit.py BEFORE.tex AFTER.tex

Both files are read and comment-stripped before any check runs, so a comment
added or removed by an editor cannot trip or hide a failure. The document is
split on \\section and \\subsection into ordered units (the text before the
first heading is the chapter preamble, its own unit). Checks 3 through 8
compare multisets WITHIN each unit, so a label, a citation, a number, or a
scope word cannot silently migrate from one subsection into another even
while the whole-chapter multiset stays balanced.

Checks (see class Result below for what each prints):
 1. float sequence (table/figure envs, captions, apanotes) byte-identical, per unit
 2. heading sequence (kind, title) identical, whole document
 3. \\label args multiset equal, per unit
 4. \\ref / \\eqref / \\autoref args multiset equal, per unit
 5. \\cite / \\citet / \\citep keys multiset equal, per unit
 6. $...$ math strings multiset equal, per unit
 7. numeric-literal multiset equal, per unit (tokenised exactly as verify_thesis.py
    pass 2 does, via the imported strip_latex and a regex kept identical to it)
 8. \\texttt{} contents multiset equal, per unit
 9. per-unit counts of scope/hedge markers unchanged
10. locked regions byte-identical: the sec:r-frontier subsection, the prose of
    sec:r-feasgate, the bag-of-words-baseline subsection, and the sentence
    beginning "One score-emitting model was measured"
11. prose word-count delta (informational only, does not affect exit code)

Exit code is nonzero if any of checks 1-10 fails.
"""

import pathlib
import re
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from accord.stats.verify_thesis import strip_latex  # noqa: E402  same tokenizer both gates use

# Kept byte-identical to the general-prose number regex verify_thesis.py's pass 2
# uses (accord/stats/verify_thesis.py, in the loop over `re.findall(r"(?<![\w.])...`
# under "pass 2: numbers"). That regex is not exposed there as a named constant,
# so it is duplicated here on purpose rather than re-derived: if it ever changes
# in verify_thesis.py, this line has to change with it or the two gates drift
# apart and stop tokenising numbers the same way.
NUMBER_RE = re.compile(r"(?<![\w.])\d+(?:,\d{3})*(?:\.\d+)?(?![\w])")

LABEL_RE = re.compile(r"\\label\{([^}]*)\}")
REF_RE = re.compile(r"\\(?:ref|eqref|autoref)\{([^}]*)\}")
CITE_RE = re.compile(r"\\cite[tp]?\*?(?:\[[^\]]*\])*\{([^}]*)\}")
MATH_RE = re.compile(r"\$([^$]*)\$")
TEXTT_RE = re.compile(r"\\texttt\{([^}]*)\}")
HEADING_RE = re.compile(r"^\\(section|subsection)\{([^}]*)\}", re.M)
FLOAT_RE = re.compile(r"\\begin\{(table|figure)\}.*?\\end\{\1\}", re.S)

# Scope/hedge markers whose per-subsection counts must not move. "post hoc" is
# the one two-word phrase; everything else is a single word.
SCOPE_MARKERS = [
    "not", "no", "never", "only", "cannot", "neither", "nor", "without",
    "unless", "withdrawn", "post hoc", "established", "license", "licenses",
    "licensed", "could", "may", "might",
]

WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*|\d[\d.,]*")


def strip_comments(text):
    """Remove unescaped LaTeX comments. Nothing else is touched."""
    return re.sub(r"(?<!\\)%.*", "", text)


def extract_floats(text):
    """Ordered list of table/figure environments, verbatim, including
    captions and \\apanote."""
    return [m.group(0) for m in FLOAT_RE.finditer(text)]


def strip_floats(text):
    return FLOAT_RE.sub("", text)


def split_headings(text):
    """[(kind, title, body), ...]. The first entry is the chapter preamble,
    kind=None, everything before the first \\section or \\subsection."""
    matches = list(HEADING_RE.finditer(text))
    if not matches:
        return [(None, None, text)]
    units = [(None, None, text[: matches[0].start()])]
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        units.append((m.group(1), m.group(2), text[start:end]))
    return units


def cite_keys(text):
    keys = []
    for m in CITE_RE.finditer(text):
        keys.extend(k.strip() for k in m.group(1).split(",") if k.strip())
    return keys


def marker_counts(text):
    counts = {}
    for marker in SCOPE_MARKERS:
        pattern = r"\b" + re.escape(marker) + r"\b"
        counts[marker] = len(re.findall(pattern, text, re.I))
    return counts


def numeric_literals(text):
    return NUMBER_RE.findall(strip_latex(text))


def mask_protected(text):
    """Same-length copy of `text` with periods inside $...$ and \\texttt{...}
    replaced, so a sentence-boundary search does not split inside either."""
    chars = list(text)
    for m in re.finditer(r"\$[^$]*\$|\\texttt\{[^}]*\}", text, re.S):
        s, e = m.span()
        for i in range(s, e):
            if chars[i] == ".":
                chars[i] = "\x00"
    return "".join(chars)


SENTENCE_END_RE = re.compile(r"\.(?=\s|$)")


def extract_sentence(text, anchor):
    """The full sentence in `text` that starts with `anchor`, sliced out of
    `text` itself so the result is exact, using a masked copy only to find
    where the sentence ends."""
    idx = text.find(anchor)
    if idx == -1:
        return None
    masked = mask_protected(text)
    m = SENTENCE_END_RE.search(masked, idx)
    end = m.end() if m else len(text)
    return text[idx:end]


def get_unit_body(units, predicate):
    for k, t, body in units:
        if predicate(k, t, body):
            return body
    return None


def prose_word_count(raw_text):
    total = 0
    for _k, _t, body in split_headings(raw_text):
        total += len(WORD_RE.findall(strip_latex(strip_floats(body))))
    return total


def check_locked_regions(before_raw, after_raw, before_units, after_units, failures):
    b = get_unit_body(before_units, lambda k, t, body: "\\label{sec:r-frontier}" in body)
    a = get_unit_body(after_units, lambda k, t, body: "\\label{sec:r-frontier}" in body)
    if b is None or a is None:
        failures.append("check10: sec:r-frontier subsection not found in one of the two files")
    elif b != a:
        failures.append("check10: sec:r-frontier subsection is not byte-identical")

    b = get_unit_body(before_units, lambda k, t, body: "\\label{sec:r-feasgate}" in body)
    a = get_unit_body(after_units, lambda k, t, body: "\\label{sec:r-feasgate}" in body)
    if b is None or a is None:
        failures.append("check10: sec:r-feasgate subsection not found in one of the two files")
    elif strip_floats(b) != strip_floats(a):
        failures.append("check10: sec:r-feasgate prose (floats excluded) is not byte-identical")

    def find_bow_unit(units):
        # Prefer the unit actually titled for the bag-of-words baseline; a
        # section intro elsewhere in the chapter mentions the phrase in
        # passing (e.g. "a bag-of-words baseline outranks all of them"),
        # which a body-substring search alone would match first since it
        # sits earlier in the document than the real subsection.
        by_title = get_unit_body(units, lambda k, t, body: t is not None and "bag-of-words" in t.lower())
        if by_title is not None:
            return by_title
        return get_unit_body(units, lambda k, t, body: "bag-of-words baseline" in body.lower())

    b = find_bow_unit(before_units)
    a = find_bow_unit(after_units)
    if b is None or a is None:
        failures.append("check10: bag-of-words-baseline subsection not found in one of the two files")
    elif b != a:
        failures.append("check10: bag-of-words-baseline subsection is not byte-identical")

    anchor = "One score-emitting model was measured"
    b = extract_sentence(before_raw, anchor)
    a = extract_sentence(after_raw, anchor)
    if b is None or a is None:
        failures.append(f"check10: locked sentence '{anchor}' not found in one of the two files")
    elif b != a:
        failures.append(f"check10: locked sentence '{anchor}...' is not byte-identical")


def main():
    if len(sys.argv) != 3:
        print("usage: python3 check_flow_edit.py BEFORE.tex AFTER.tex")
        return 2

    before_path = pathlib.Path(sys.argv[1])
    after_path = pathlib.Path(sys.argv[2])
    before_raw = strip_comments(before_path.read_text())
    after_raw = strip_comments(after_path.read_text())

    failures = []

    before_units = split_headings(before_raw)
    after_units = split_headings(after_raw)
    before_heads = [(k, t) for k, t, _ in before_units if k]
    after_heads = [(k, t) for k, t, _ in after_units if k]

    print(f"BEFORE: {before_path}\nAFTER:  {after_path}\n")

    # ---------------- check 2: heading sequence ----------------
    print("CHECK 2  heading sequence")
    if before_heads != after_heads:
        print(f"   FAIL: heading sequence differs "
              f"({len(before_heads)} before, {len(after_heads)} after)")
        for i, (b, a) in enumerate(zip(before_heads, after_heads)):
            if b != a:
                print(f"   position {i}: before={b!r} after={a!r}")
        if len(before_heads) != len(after_heads):
            print("   heading count itself differs; per-unit checks skipped")
        failures.append("check2: heading sequence differs")
    else:
        print(f"   pass: {len(before_heads)} headings, unchanged order and text\n")

    # ---------------- checks 1, 3-9: per unit ----------------
    if before_heads == after_heads:
        print("CHECKS 1, 3-9  per-subsection invariants")
        for (bk, bt, bbody), (ak, at, abody) in zip(before_units, after_units):
            label = at if at is not None else "(chapter preamble)"

            bfloats = extract_floats(bbody)
            afloats = extract_floats(abody)
            if bfloats != afloats:
                failures.append(f"check1: float sequence changed in unit '{label}'")
                print(f"   FAIL check1  '{label}': float sequence not byte-identical "
                      f"({len(bfloats)} before, {len(afloats)} after)")

            checks = [
                ("check3", "labels", Counter(LABEL_RE.findall(bbody)), Counter(LABEL_RE.findall(abody))),
                ("check4", "refs", Counter(REF_RE.findall(bbody)), Counter(REF_RE.findall(abody))),
                ("check5", "cite keys", Counter(cite_keys(bbody)), Counter(cite_keys(abody))),
                ("check6", "math strings", Counter(MATH_RE.findall(bbody)), Counter(MATH_RE.findall(abody))),
                ("check7", "numeric literals", Counter(numeric_literals(bbody)), Counter(numeric_literals(abody))),
                ("check8", "texttt contents", Counter(TEXTT_RE.findall(bbody)), Counter(TEXTT_RE.findall(abody))),
            ]
            for tag, name, bc, ac in checks:
                if bc != ac:
                    failures.append(f"{tag}: {name} multiset changed in unit '{label}'")
                    diff = Counter(bc); diff.subtract(ac)
                    diff = {k: v for k, v in diff.items() if v != 0}
                    print(f"   FAIL {tag}  '{label}': {name} multiset differs, delta={diff}")

            bm = marker_counts(bbody)
            am = marker_counts(abody)
            if bm != am:
                moved = {k: (bm[k], am[k]) for k in bm if bm[k] != am[k]}
                failures.append(f"check9: scope-marker counts changed in unit '{label}'")
                print(f"   FAIL check9  '{label}': scope-marker counts changed {moved}")

        if not any(f.startswith(("check1", "check3", "check4", "check5", "check6", "check7", "check8", "check9"))
                   for f in failures):
            print(f"   pass: {len(before_units)} units, floats/labels/refs/cites/math/numbers/"
                  f"texttt/scope-markers all unchanged within each unit\n")
        else:
            print()

    # ---------------- check 10: locked regions ----------------
    print("CHECK 10  locked regions")
    n_before = len(failures)
    check_locked_regions(before_raw, after_raw, before_units, after_units, failures)
    if len(failures) == n_before:
        print("   pass: sec:r-frontier, sec:r-feasgate prose, the bag-of-words-baseline "
              "subsection, and the 'One score-emitting model was measured' sentence "
              "are all byte-identical\n")
    else:
        for f in failures[n_before:]:
            print(f"   FAIL {f}")
        print()

    # ---------------- check 11: prose word count (informational) ----------------
    print("CHECK 11  prose word count (informational, does not gate)")
    before_words = prose_word_count(before_raw)
    after_words = prose_word_count(after_raw)
    print(f"   before: {before_words}")
    print(f"   after:  {after_words}")
    print(f"   delta:  {after_words - before_words}\n")

    print("RESULT:", "FAIL" if failures else "pass", f"({len(failures)} failing check(s))")
    if failures:
        for f in failures:
            print("  -", f)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
