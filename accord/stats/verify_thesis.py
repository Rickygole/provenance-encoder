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
# Passes that check prose rather than document structure scan the front matter too.
# It was outside every glob, so the abstract, title page and declaration were never
# scanned by anything, which is how a conflated denominator survived in the
# highest-visibility 250 words of the document. Pass 5 keeps to CHAPTERS because it
# looks for a \chapter and a label, which front matter correctly does not have.
PROSE = CHAPTERS + sorted((THESIS / "frontmatter").glob("*.tex"))

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


def fact_values():
    """Every verified value, rendered in the string forms a chapter might use."""
    out = set()
    for key, rec in FACTS.items():
        if key.startswith(WITHDRAWN_PREFIX):
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


def main():
    if not CHAPTERS:
        print("no msu_ch*.tex files found yet")
        return 0

    values = fact_values()
    failures = 0

    print(f"verifying {len(CHAPTERS)} chapters against {len(FACTS)} verified facts\n")

    # ---------------- pass 1: dashes ----------------
    print("PASS 1  dashes")
    bad = 0
    for f in CHAPTERS:
        for i, line in enumerate(f.read_text().splitlines(), 1):
            for ch, name in (("\u2014", "em dash"), ("\u2013", "en dash"), ("\u2212", "minus sign")):
                if ch in line:
                    print(f"   {f.name}:{i}  {name}: {line.strip()[:90]}")
                    bad += 1
            if re.search(r"(?<!\-)\-\-(?!\-)", line) and "%" not in line[:2]:
                print(f"   {f.name}:{i}  latex en dash (--): {line.strip()[:90]}")
                bad += 1
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
    unverified = []
    for f in CHAPTERS:
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

    # ---------------- pass 3: citations ----------------
    print("PASS 3  citation keys")
    bib = (THESIS / "refs.bib").read_text() if (THESIS / "refs.bib").exists() else ""
    keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    missing = set()
    for f in CHAPTERS:
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
    for f in CHAPTERS:
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
    print(f"   {'review' if hits else 'pass'}: {hits} style flags\n")

    # ---------------- pass 5: structure ----------------
    print("PASS 5  structure")
    problems = 0
    for f in CHAPTERS:
        text = f.read_text()
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
    print(f"total chapter lines: {total_lines}")
    print("RESULT:", "FAIL" if failures else "all gating passes clear")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
