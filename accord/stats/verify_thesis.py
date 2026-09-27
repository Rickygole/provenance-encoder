"""
Five independent verification passes over the written thesis chapters.

Pass 1  dashes        no em dash, en dash, or unicode minus anywhere
Pass 2  numbers       every numeric literal in prose traces to VERIFIED_FACTS.json
Pass 3  citations     every \\cite key exists in refs.bib
Pass 4  style         formulaic phrasing that reads as machine-written
Pass 5  structure     required labels, caption placement, no leftover TODO

Run:  python3 accord/stats/verify_thesis.py
Exit code is nonzero if any pass fails, so it can gate a build.
"""

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
THESIS = ROOT / "accord" / "thesis"
FACTS = json.loads((ROOT / "results" / "accord" / "VERIFIED_FACTS.json").read_text())
# The front matter was outside this glob, so the abstract, the title page and the
# declaration were never scanned by any pass. That is the highest-visibility 250
# words in the document and the one place a conflated denominator survived, and it
# is the same failure this project already recorded twice: a gate that reports zero
# by scanning nothing. Appendices are still excluded because they are generated and
# print register keys and artifact paths by design.
CHAPTERS = sorted(THESIS.glob("msu_ch*.tex"))
# The side-strand material that used to be bound in the thesis body is moving to
# a companion document, main_supplement.tex, built from files named msu_sup*.tex.
# The harness historically read main_msu.pdf only, so a second document going
# stale was a recorded failure (CLAUDE.md); the equivalent failure here is a
# supplement whose prose nothing scans. msu_sup*.tex files are scanned in passes
# 1 to 4 and, where applicable, pass 5, alongside msu_ch*.tex.
SUPPLEMENT = sorted(THESIS.glob("msu_sup*.tex"))

# The supplement is being assembled concurrently with this change, so msu_sup*.tex
# matching nothing is currently the honest state, not a bug: main_supplement.tex
# does not yet \input any file by that name. The moment it does, this glob has to
# find it, or this is the exact ch*.tex / msu_ch*.tex defect CLAUDE.md records
# ("a gate that cannot fail"), just against the supplement instead of the thesis.
# Asserted here, at import time, rather than left to be discovered by a pass
# quietly scanning zero files and reporting a clean 0.
_SUP_MAIN = THESIS / "main_supplement.tex"
if _SUP_MAIN.exists():
    _sup_declared = re.findall(r"\\input\{(msu_sup[^}]*)\}",
                                re.sub(r"(?<!\\)%.*", "", _SUP_MAIN.read_text()))
    _sup_found = {p.stem for p in SUPPLEMENT}
    _sup_missing = [name for name in _sup_declared if name not in _sup_found]
    if _sup_declared and not SUPPLEMENT:
        raise AssertionError(
            f"main_supplement.tex \\input{{}}s {_sup_declared} but the glob "
            "'accord/thesis/msu_sup*.tex' matched no files at all. Fix the "
            "pattern or write the missing file; do not let this scan zero files "
            "and call it clean.")
    if _sup_missing:
        raise AssertionError(
            f"main_supplement.tex \\input{{}}s {_sup_missing}, which the "
            "'msu_sup*.tex' glob did not find on disk. A bound file that a glob "
            "cannot see is the same defect, just for one file instead of all of "
            "them.")

# Passes that check prose rather than document structure scan the front matter too.
# It was outside every glob, so the abstract, title page and declaration were never
# scanned by anything, which is how a conflated denominator survived in the
# highest-visibility 250 words of the document. Pass 5 keeps to CHAPTERS (not
# SUPPLEMENT) for the \chapter/\label{ch:} requirement, because front matter and
# the supplement's own side strands correctly need neither.
PROSE = CHAPTERS + sorted((THESIS / "frontmatter").glob("*.tex"))
# PROSE above was defined and never used: every pass looped over the chapters
# only, so the abstract, the highest-visibility 250 words in the document, was
# never scanned by the dash, number-provenance or style passes, while a comment
# said it was. That is the gate-that-scans-nothing failure this project has
# already recorded twice. The front matter is now scanned explicitly.
#
# Pass 1 (dashes) scans every front-matter page. Passes 2 and 4 scan only the
# abstract: the title, approval and declaration pages carry template dates and
# degree text that are not findings. The acknowledgements are the author's own
# and are excluded from every pass, so no gate can ever require editing them.
FRONTMATTER = [f for f in sorted((THESIS / "frontmatter").glob("*.tex"))
               if f.name != "acknowledgements.tex"]
ABSTRACT = [THESIS / "frontmatter" / "abstract.tex"]
assert ABSTRACT[0].exists(), "abstract.tex missing: refusing to scan nothing"
# The set every prose pass (1, 2, 3, 4) scans: chapters plus whatever side-strand
# material is currently bound in the supplement by name. When SUPPLEMENT is
# empty (the current, transitional state) this is exactly CHAPTERS, so today's
# run is unchanged; the moment a msu_sup*.tex file exists, its dashes, numbers,
# citations and style are gated the same way a chapter's are.
# The two appendix texts the thesis binds were in no glob either, so a generator
# overwrote audited appendix prose and an untraced count stopped adding up with
# every pass reporting clear. They are scanned by the prose passes now.
APPENDICES = [THESIS / "msu_appA_corrections.tex", THESIS / "msu_appB_pointer.tex"]
for _a in APPENDICES:
    assert _a.exists(), f"{_a.name} missing: refusing to scan nothing"
ALL_CHAPTERS = CHAPTERS + SUPPLEMENT + APPENDICES

# Values that are allowed to appear without being a research finding: structural
# numbers, years, and the small integers that show up in ordinary prose.
STRUCTURAL = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12",
              "100", "1000", "0.05", "0.5", "1.0", "2.0", "95",
              # random seeds and pre-registered thresholds, which are settings not findings
              "42", "123", "2024", "20260806", "20260809",
              "2.5", "3.0", "3.5", "0.15", "0.20", "0.65", "1.5", "160", "5000",
              # standard normal critical value, a setting in an interval formula
              # rather than a finding, exactly as 0.05 above is
              "1.96"}


# Keys under this prefix hold values the thesis has retracted. They are kept so the
# correction record can quote what was withdrawn, and they are excluded from the value
# set here: a retracted number must not make a live number resolvable. That is exactly
# how a withdrawn control-cost total passed this gate as a chapter headline; see C84.
WITHDRAWN_PREFIX = "withdrawn."

# Values produced through a channel the methodology chapter records as a departure
# from the data use agreement. Chapter 4 states that the GPT-4o rows "have been
# withdrawn rather than qualified, and no number in this chapter now derives from
# it", and Section m-ethics states they are withdrawn from every table and figure.
# That was true, and it was enforced by prose alone: all 25 of these keys sat live
# in the register, so this gate would have resolved a GPT-4o digit anywhere in the
# thesis without complaint. A withdrawal a gate cannot see is an assertion, not a
# control. They are excluded here for the same reason withdrawn.* is, and the
# separate prefix is kept because the ground differs: withdrawn.* is retracted for
# being wrong, this is excluded for being unusable. See C89.
NON_CITABLE_SUBSTRINGS = ("gpt4o",)

# Values obtained by transmitting credentialed sentences to a third-party hosted
# LLM service. PhysioNet's 24 September 2025 guidance prohibits that transmission
# without a vendor qualifier and withdrew the April 2023 list of suggested
# services, so no hosted route has a safe harbour and every result downstream of
# one is withdrawn. The classification and the relocation live in
# accord/hosted_withdrawal.py and accord/stats/withdraw_hosted.py; this gate
# refuses to run while any of them is still live, because a register that holds a
# hosted value under a citable name will resolve a hosted digit in prose, which
# is the exact way the GPT-4o withdrawal was an assertion rather than a control
# for as long as it was enforced by a sentence (C89).
sys.path.insert(0, str(ROOT))
from accord.hosted_withdrawal import is_hosted  # noqa: E402


def _non_citable(key):
    return (key.startswith(WITHDRAWN_PREFIX)
            or any(sub in key.lower() for sub in NON_CITABLE_SUBSTRINGS))


def live_hosted_keys():
    """Hosted-channel values still sitting under a citable name."""
    return [k for k, rec in FACTS.items()
            if not k.startswith(WITHDRAWN_PREFIX) and is_hosted(k, rec)]


def fact_values():
    """Every verified value, rendered in the string forms a chapter might use."""
    out = set()
    for key, rec in FACTS.items():
        if _non_citable(key):
            continue
        v = rec["value"]
        if isinstance(v, bool) or v is None:
            continue
        if isinstance(v, int):
            out.add(str(v))
            out.add(f"{v:,}")
        elif isinstance(v, float):
            for val in (v, abs(v)):
                for places in range(0, 7):
                    out.add(f"{val:.{places}f}")
                out.add(str(val))
                # A times-100 form used to be added here unconditionally, so a
                # registered 0.04 made the literal 4.00 resolvable and a
                # registered 0.61 made 61 resolvable. That is how this gate came
                # to accept numbers whose attribution is off by two orders of
                # magnitude, and it is the same defect trace_claims.py carried
                # into Appendix C, where it printed 4.00 against a precision
                # floor of 0.04. A percent form is only offered when the value
                # actually lies in [0, 1], where reading it as a percentage is
                # meaningful, and only at the precision a percentage is written
                # with. It is still a digit match and still not a proof of
                # provenance; it is one fewer way to be wrong.
                if 0.0 <= val <= 1.0:
                    for places in range(0, 3):
                        out.add(f"{val * 100:.{places}f}")
    return out


def strip_latex(text):
    """Remove comments, labels, code spans and file paths before scanning prose.

    LaTeX writes thousands as 10{,}638{,}720; collapse that to a single token
    first, otherwise the scanner reports each group as an unverified number.
    """
    text = re.sub(r"(\d)\{,\}(\d)", r"\1,\2", text)
    text = re.sub(r"p\{[\d.]+\\?\w*\}", " ", text)        # tabular column widths
    text = re.sub(r"\{[\d.]+\\(textwidth|linewidth)\}", " ", text)
    # arithmetic displayed in full (6 + 65 = 71): the result is checkable by the reader
    text = re.sub(r"\$?\s*\d[\d,.]*\s*\+\s*\d[\d,.]*\s*=\s*(\d[\d,.]*)\s*\$?", " ", text)
    text = re.sub(r"\\texttt\{[^}]*\}", " ", text)   # code identifiers, file paths, ICD codes
    text = re.sub(r"(?<!\\)%.*", "", text)
    text = re.sub(r"\\label\{[^}]*\}", "", text)
    text = re.sub(r"\\ref\{[^}]*\}", "", text)
    text = re.sub(r"\\cite[tp]?\{[^}]*\}", "", text)
    text = re.sub(r"\\includegraphics(\[[^\]]*\])?\{[^}]*\}", "", text)
    text = re.sub(r"\\(begin|end)\{[^}]*\}", "", text)
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    return text


BANNED_PHRASES = [
    "delve", "leverage the", "it is important to note", "it is worth noting",
    "plays a crucial role", "plays a vital role", "underscores the",
    "highlights the importance", "paves the way", "in today's world",
    "a testament to", "navigate the complex", "shed light on",
    "it should be noted that", "the landscape of", "realm of",
    "furthermore, the", "moreover, the", "additionally, the",
]

PARA_OPENERS = ["Moreover", "Furthermore", "Additionally", "Notably", "Importantly",
                "Consequently", "Overall"]


# ---------------------------------------------------------------- pass 1 rules
# The no-dash rule is a rule about prose: it exists so no sentence reads as
# machine-written. It is not a rule about numerals, and the School of Graduate
# Studies handbook requires the opposite there: "An en dash is a symbol used to
# show numerical range... Examples: 1995-1998, 32-35" (with en dashes in the
# handbook's own examples, not the hyphens this comment is limited to spelling).
# Enforcing the prose
# rule over the whole document forced \bibrangedash to a hyphen and printed 68
# page ranges in violation of the handbook.
#
# So the rule is scoped, not weakened. Exactly one construction is permitted
# anywhere: an en dash with a digit on each side and no space. Everything
# else still fails, including
#   * an en dash anywhere in prose (spaced, or against a letter),
#   * an en dash spaced away from the digits it joins,
#   * an em dash or a unicode minus, in any position at all,
#   * a spaced hyphen used as a dash (word - word), which the old rule missed,
#   * TeX's -- and ---, which render as an en and an em dash.
# A digit range is narrower than "between digits": the check below requires the
# dash to be flanked by a digit directly, so an en dash against a letter fails.
DIGIT_RANGE_ENDASH = re.compile(r"(?<=\d)\u2013(?=\d)")
SPACED_DASH = re.compile(r"(?:(?<=\s)|^)(?:-{1,3}|[\u2013\u2014\u2212])(?=\s|$)")
# Inline math, and the environments that carry display math over several lines.
# A spaced hyphen inside math is a minus sign, not punctuation, so the spaced-dash
# rule does not apply there; every other rule still does, which is what keeps a
# real en dash or unicode minus from hiding inside $...$.
INLINE_MATH = re.compile(r"(?<!\\)\$.*?(?<!\\)\$", re.S)
MATH_ENV = r"equation\*?|align\*?|alignat\*?|gather\*?|multline\*?|displaymath|eqnarray\*?"
MATH_OPEN = re.compile(r"\\begin\{(?:" + MATH_ENV + r")\}|\\\[")
MATH_CLOSE = re.compile(r"\\end\{(?:" + MATH_ENV + r")\}|\\\]")


def dash_violations(line, in_math=False):
    """Return [(name, count)] for every dash in one source line that is not a
    handbook numeric range. in_math marks a line inside display math, where an
    ASCII hyphen between spaces is a minus sign. Pure function of its arguments,
    so it is negative-testable without a document; see self_test()."""
    out = []
    # A spaced dash is a dash used as punctuation whatever its glyph, so it is
    # checked before the en-dash exemption can reach it. Math is removed first,
    # inline spans here and whole lines by the caller's in_math flag.
    if not in_math:
        n = len(SPACED_DASH.findall(INLINE_MATH.sub(" ", line)))
        if n:
            out.append(("spaced dash", n))
    stripped = DIGIT_RANGE_ENDASH.sub("", line)
    for ch, name in (("\u2014", "em dash"), ("\u2212", "minus sign")):
        if ch in line:
            out.append((name, line.count(ch)))
    if "\u2013" in stripped:
        out.append(("en dash outside a numeric range", stripped.count("\u2013")))
    if not line.lstrip().startswith("%"):
        # TeX ligatures: -- is an en dash and --- an em dash on the page. The
        # printed page is what the handbook and the checklist are read against.
        for m in re.finditer(r"-{2,}", line):
            out.append(("latex em dash (---)" if len(m.group(0)) > 2
                        else "latex en dash (--)", 1))
    return out


def scan_dashes(text):
    """Yield (lineno, name, count) for one file's text, tracking display math."""
    in_math = False
    for i, line in enumerate(text.splitlines(), 1):
        opened = bool(MATH_OPEN.search(line))
        closed = bool(MATH_CLOSE.search(line))
        for name, hits in dash_violations(line, in_math=in_math or opened):
            yield i, name, hits
        if opened and not closed:
            in_math = True
        elif closed:
            in_math = False


def self_test():
    """Negative tests for dash_violations. Run: verify_thesis.py --self-test"""
    cases = [
        # (line, expect_fail, what it is)
        ("Pages 32\u201335 of the handbook.", False, "handbook numeric range"),
        ("The interval was 0.12\u20130.19 wide.", False, "decimal range"),
        ("A range from 1995\u20131998.", False, "year range"),
        ("The pipeline\u2013and its gate\u2013failed.", True, "prose en dash"),
        ("The pipeline \u2013 and its gate \u2013 failed.", True, "spaced en dash"),
        ("The pipeline \u2014 and its gate \u2014 failed.", True, "spaced em dash"),
        ("The pipeline\u2014and its gate\u2014failed.", True, "prose em dash"),
        ("A drop of \u22120.03 points.", True, "unicode minus"),
        ("The gate - which never fired - reported zero.", True, "spaced hyphen"),
        ("Pages 32 \u2013 35 of the handbook.", True, "spaced dash between digits"),
        ("Pages 32--35 of the handbook.", True, "TeX -- in prose"),
        ("The gate---which never fired---reported zero.", True, "TeX ---"),
        ("A 3\u2013fold increase.", True, "en dash against a letter"),
        ("Cross-note, stage-by-stage, GPT-4o.", False, "ordinary hyphens"),
        ("32\u201335 and 1995\u20131998 on one line.", False, "two ranges"),
        ("The quantity $a - 0.5$ is the effect.", False, "minus in inline math"),
        ("  \\frac{\\max - \\min}{\\max} > \\tau ,", False, "minus in display math",
         True),
        ("A dash \u2013 inside $x - y$ math on the line.", True,
         "en dash beside math"),
        ("  x = 1 \u2013 2 in display math", True, "en dash inside display math", True),
    ]
    fails = 0
    for case in cases:
        line, expect_fail, what = case[:3]
        got = dash_violations(line, in_math=case[3] if len(case) > 3 else False)
        ok = bool(got) == expect_fail
        fails += not ok
        print(f"   {'ok  ' if ok else 'BAD '} {what}: "
              f"{'flagged' if got else 'clean'} "
              f"{[n for n, _ in got]}")
    print(f"   {'FAIL' if fails else 'pass'}: {len(cases) - fails}/{len(cases)} "
          "dash cases behave as specified")
    return fails


def main():
    if not CHAPTERS:
        print("no msu_ch*.tex files found yet")
        return 0

    values = fact_values()
    failures = 0

    print(f"verifying {len(CHAPTERS)} chapters + {len(SUPPLEMENT)} supplement files "
          f"against {len(FACTS)} verified facts\n")
    if not SUPPLEMENT:
        print("supplement: 0 msu_sup*.tex files present yet "
              "(main_supplement.tex does not \\input any by that name); "
              "passes 1-4 below are scanning only the thesis chapters\n")

    # ---------------- pass 1: dashes ----------------
    print("PASS 1  dashes")
    bad = 0
    for f in ALL_CHAPTERS + FRONTMATTER:
        lines = f.read_text().splitlines()
        for i, name, hits in scan_dashes(f.read_text()):
            print(f"   {f.name}:{i}  {name}: {lines[i - 1].strip()[:90]}")
            bad += hits
    # Figures are drawn from scripts, and a dash inside a rendered PDF is
    # invisible to a scan of the .tex. Check the generating source too.
    figsrc = 0
    for f in sorted((THESIS / "figures").glob("*.py")):
        for i, line in enumerate(f.read_text(errors="ignore").splitlines(), 1):
            for ch, name in (("\u2014", "em dash"), ("\u2013", "en dash"), ("\u2212", "minus sign")):
                if ch in line:
                    print(f"   figures/{f.name}:{i}  {name} in figure source: {line.strip()[:80]}")
                    figsrc += 1
    bad += figsrc
    print(f"   {'FAIL' if bad else 'pass'}: {bad} dash violations "
          f"({figsrc} in figure source)\n")
    failures += bad > 0

    # ---------------- pass 2: numbers ----------------
    print("PASS 2  numeric provenance")
    # Precondition, checked before the digit match rather than beside it: the
    # citable set has to be the right set before asking whether a digit is in it.
    hosted_live = live_hosted_keys()
    if hosted_live:
        print(f"   FAIL: {len(hosted_live)} hosted-channel values are live in "
              f"the register and would make a withdrawn digit resolvable.")
        for k in hosted_live[:10]:
            print(f"      {k}")
        if len(hosted_live) > 10:
            print(f"      ... and {len(hosted_live) - 10} more")
        print("   run: python3 accord/stats/withdraw_hosted.py")
        failures += 1
    else:
        n_held = sum(1 for k in FACTS if k.startswith("withdrawn.hosted."))
        print(f"   hosted channels: none live, {n_held} held for the record")
    unverified = []
    # The appendices' numbers count the register and the trace themselves, so
    # they cannot be registered without circularity. make_appendices.py checks
    # Appendix A's counts and sync_appendix_counts.py --check (deep pass 11)
    # checks Appendix B's.
    for f in [x for x in ALL_CHAPTERS if x not in APPENDICES] + ABSTRACT:
        # The literature review reports figures from cited works, which by
        # definition are not in our fact sheet. It is checked separately:
        # every number there must sit in a sentence carrying a citation.
        if "ch2_literature" in f.name:
            prose = f.read_text()
            for i, line in enumerate(prose.splitlines(), 1):
                nums = re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?![\w])", strip_latex(line))
                # A number here is legitimate if it is attributed to a cited work
                # OR if it is one of our own registered results being previewed.
                for tok in nums:
                    if tok in values or tok in STRUCTURAL:
                        continue
                    ctx = "\n".join(prose.splitlines()[max(0, i - 8):i + 3])
                    if "\\cite" in ctx:
                        continue
                    unverified.append((f.name, i, tok, line.strip()[:70]))
            continue
        prose = strip_latex(f.read_text())
        for i, line in enumerate(prose.splitlines(), 1):
            for tok in re.findall(r"(?<![\w.])\d+(?:,\d{3})*(?:\.\d+)?(?![\w])", line):
                clean = tok.replace(",", "")
                if tok in STRUCTURAL or clean in STRUCTURAL:
                    continue
                if re.fullmatch(r"(19|20)\d\d", clean):     # years
                    continue
                if tok in values or clean in values:
                    continue
                unverified.append((f.name, i, tok, line.strip()[:70]))
    for name, i, tok, ctx in unverified[:40]:
        print(f"   {name}:{i}  '{tok}'  in: {ctx}")
    if len(unverified) > 40:
        print(f"   ... and {len(unverified) - 40} more")
    print(f"   {'FAIL' if unverified else 'pass'}: {len(unverified)} numbers not traced to VERIFIED_FACTS.json\n")
    failures += len(unverified) > 0

    # A register entry has two columns: a value, checked above by digit match
    # against prose, and an "artifact" naming where that value came from. The
    # artifact column was never opened by anything -- a key could cite a path
    # that does not exist, or never did, and still clear every check above,
    # because nothing had ever read that field for anything but its own
    # printed text. That is the same "gate that cannot fail" defect this
    # project has already shipped more than once (CLAUDE.md): a column that
    # looks like verification but nothing checks. This closes it for every
    # register key the claim trace (results/accord/CLAIM_TRACE.json,
    # produced by accord/stats/trace_claims.py) actually resolved a printed
    # number to: the leading, path-shaped part of that key's artifact field
    # must exist on disk. Many artifact fields are not a path at all -- a
    # "derived: ..." computation, or prose describing one -- and some carry a
    # "path::json.pointer" form or a trailing comma-separated note; only the
    # part before the first "," or "::" is ever treated as a path, and only
    # when it is shaped like one (bare relative-path characters, an optional
    # trailing "(...)" annotation stripped). A derivation is not a missing
    # file, so it is never reported as one, and nothing here re-derives or
    # re-runs trace_claims.py -- it only reads the trace already committed,
    # so this stays a fast file-existence check, not a second copy of the
    # matching logic in trace_claims.py.
    print("PASS 2 (cont.)  cited-artifact existence")
    trace_path = ROOT / "results" / "accord" / "CLAIM_TRACE.json"
    if not trace_path.exists():
        print("   skipped: results/accord/CLAIM_TRACE.json not present "
              "(run accord/stats/trace_claims.py)\n")
    else:
        trace = json.loads(trace_path.read_text())
        cited_keys = sorted({r["register_key"] for r in trace.get("records", [])
                              if r.get("register_key") in FACTS})
        missing_artifacts = []
        checked = 0
        for key in cited_keys:
            artifact = FACTS[key].get("artifact", "")
            lead = artifact.split(",", 1)[0].split("::", 1)[0].strip()
            lead = re.sub(r"\s*\([^()]*\)\s*$", "", lead).strip()
            if not lead or lead.startswith("derived:"):
                continue                 # a computation, not a file to open
            if not re.fullmatch(r"[A-Za-z0-9_./-]+", lead):
                continue                 # does not look like a path; do not guess
            checked += 1
            if not (ROOT / lead).exists():
                missing_artifacts.append((key, lead))
        for key, lead in missing_artifacts:
            print(f"   {key}: artifact '{lead}' does not exist")
        print(f"   {'FAIL' if missing_artifacts else 'pass'}: {checked} of "
              f"{len(cited_keys)} cited register keys named a path checked; "
              f"{len(missing_artifacts)} artifact(s) not found\n")
        failures += len(missing_artifacts) > 0

    # ---------------- pass 3: citations ----------------
    print("PASS 3  citation keys")
    bib = (THESIS / "refs.bib").read_text() if (THESIS / "refs.bib").exists() else ""
    keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    missing = set()
    for f in ALL_CHAPTERS:
        for m in re.finditer(r"\\cite[tp]?\*?(?:\[[^\]]*\])*\{([^}]*)\}", f.read_text()):
            for k in m.group(1).split(","):
                k = k.strip()
                if k and k not in keys:
                    missing.add((f.name, k))
    for name, k in sorted(missing):
        print(f"   {name}: missing bib key '{k}'")
    print(f"   {'FAIL' if missing else 'pass'}: {len(missing)} unresolved citation keys "
          f"({len(keys)} entries in refs.bib)\n")
    failures += len(missing) > 0

    # ---------------- pass 4: style ----------------
    print("PASS 4  machine-written tells")
    hits = 0
    for f in ALL_CHAPTERS + ABSTRACT:
        text = f.read_text()
        low = text.lower()
        for phrase in BANNED_PHRASES:
            n = low.count(phrase)
            if n:
                print(f"   {f.name}: '{phrase}' x{n}")
                hits += n
        opens = re.findall(r"(?m)^(" + "|".join(PARA_OPENERS) + r")\b", text)
        if len(opens) > 2:
            print(f"   {f.name}: {len(opens)} paragraphs opening with {sorted(set(opens))}")
            hits += len(opens)
    # This pass counted its hits and never added them to `failures`, so it could
    # report flags and still let the run pass: the gate-that-cannot-fail defect
    # CLAUDE.md records. It gates now.
    print(f"   {'FAIL' if hits else 'pass'}: {hits} style flags\n")
    failures += hits > 0

    # ---------------- pass 5: structure ----------------
    print("PASS 5  structure")
    problems = 0
    chapter_set = set(CHAPTERS)
    for f in ALL_CHAPTERS:
        text = f.read_text()
        # The \chapter/\label{ch:} requirement is chapter-specific: it does not
        # apply to a msu_sup*.tex file, which is bound into the supplement's own
        # \appendix block (lettered labels, not "ch:"), same as front matter
        # correctly carries neither. The leftover-TODO scan and the
        # caption-above-tabular check below are not chapter-specific, so they
        # run over every file in ALL_CHAPTERS, supplement included.
        if f in chapter_set:
            if not re.search(r"\\chapter\{", text):
                print(f"   {f.name}: no \\chapter")
                problems += 1
            if not re.search(r"\\label\{ch:", text):
                print(f"   {f.name}: no chapter label")
                problems += 1
        for todo in re.findall(r"(?i)\[?TODO[^\]\n]*", text):
            print(f"   {f.name}: leftover {todo[:60]}")
            problems += 1
        # caption must precede tabular inside a table environment
        for env in re.findall(r"\\begin\{table\}.*?\\end\{table\}", text, re.S):
            cap, tab = env.find("\\caption"), env.find("\\begin{tabular}")
            if cap > -1 and tab > -1 and cap > tab:
                print(f"   {f.name}: caption below tabular (APA requires above)")
                problems += 1
    print(f"   {'FAIL' if problems else 'pass'}: {problems} structural problems\n")
    failures += problems > 0

    total_lines = sum(len(f.read_text().splitlines()) for f in CHAPTERS)
    total_sup_lines = sum(len(f.read_text().splitlines()) for f in SUPPLEMENT)
    print(f"total chapter lines: {total_lines}"
          + (f" (+ {total_sup_lines} supplement lines)" if SUPPLEMENT else ""))
    print("RESULT:", "FAIL" if failures else "all gating passes clear")
    return 1 if failures else 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        print("PASS 1 dash rule, negative tests")
        sys.exit(1 if self_test() else 0)
    sys.exit(main())
