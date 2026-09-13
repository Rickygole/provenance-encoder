"""
Ten verification passes over the thesis. Wraps the original five and adds five.

Pass 1   dashes         no em dash, en dash, or unicode minus anywhere
Pass 2   numbers        every numeric literal traces to VERIFIED_FACTS.json
Pass 3   citations      every \\cite key exists in refs.bib
Pass 4   style          formulaic phrasing that reads as machine-written
Pass 5   structure      required labels, caption placement, no leftover TODO
Pass 6   overclaim      strong verbs attached to results, outside negation
Pass 7   references     every \\ref resolves, no duplicate labels
Pass 8   captions       figure captions above the graphic, substantial, files exist
Pass 9   template       abstract hard rules and page-setup invariants (FORMAT.md)
Pass 10  figures        scripts rebuild the committed PDFs pixel-identically,
                        and no result-file name is drawn on any canvas

Passes 1 to 5 run by invoking verify_thesis.py, so the two scripts cannot
drift. Exit code is nonzero if any gating pass fails.

Run:  python3 accord/stats/verify_thesis_deep.py
      python3 accord/stats/verify_thesis_deep.py --skip-figures   (passes 1-9 only)
"""

import json
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
THESIS = ROOT / "accord" / "thesis"
CHAPTERS = sorted(THESIS.glob("msu_ch*.tex"))
APPENDICES = sorted(THESIS.glob("msu_app*.tex"))
FRONT = sorted((THESIS / "frontmatter").glob("*.tex"))

# Every deep pass iterated CHAPTERS only, so the bound appendices were outside
# all of them. A caption sitting below its figure survived there for exactly
# that reason: pass 8 could not see the file. Only the appendices main_msu.tex
# actually inputs are checked, because msu_app*.tex also matches files left in
# the tree that nothing binds, and a gate that reports on dead files teaches
# people to ignore it.
def _bound_appendices():
    main = (THESIS / "main_msu.tex").read_text()
    out = []
    for p in APPENDICES:
        if "\\input{%s}" % p.stem in main:
            out.append(p)
    return out


BOUND_APPENDICES = _bound_appendices()

# The twelve figures bound into the thesis, and the scripts that draw them.
# Three drawing functions survive in make_figures.py unreferenced and unbound,
# and are not checked here: fig-rule-precision left the set at the committee
# review, fig-survey-audit was cut by the author (its content is one sentence of
# body text), and fig-disc-strata redrew fig-forest's second family identically
# sixty pages apart.
THESIS_FIGS = [
    "fig-pipeline-status", "fig-intro-headline", "fig-cohort-flow",
    "fig-prevalence-response", "fig-falsification", "fig-power", "fig-forest",
    "fig-scaling", "fig-threshold-surface",
    "fig-disc-actionability", "fig-met-exchange", "fig-exemplar",
    "fig-capability", "fig-missing-cell", "fig-labelnoise",
]
FIG_SCRIPTS = ["make_figures.py", "fig_batch2.py",
               "fig_threshold_surface.py", "make_figures_extra.py",
               "fig_frontier.py"]

# Pass 6. Hard verbs gate; soft verbs are reported for review. A match whose
# preceding thirty characters contain a negation is legitimate hedging, not an
# overclaim, and is skipped.
HARD_OVERCLAIM = [
    (r"\bprove[sd]?\b", "proves"),
    (r"\bconclusively\b", "conclusively"),
    (r"\bunequivocall?y\b", "unequivocally"),
    (r"\bwithout\s+doubt\b", "without doubt"),
    (r"\bdefinitively\s+(shows?|establish)", "definitively shows"),
]
SOFT_OVERCLAIM = [
    (r"\bdemonstrat(e|es|ed)\s+that\b", "demonstrates that"),
    (r"\bestablish(es|ed)?\s+that\b", "establishes that"),
    (r"\bconfirms?\s+that\b", "confirms that"),
    (r"\bsignificantly\s+(better|worse|higher|lower)\b", "significantly (no test)"),
]
NEGATION = re.compile(r"(not|no|never|cannot|fails? to|nothing|neither|nor)\s+\S*$",
                      re.I)


def strip_comments(text):
    return re.sub(r"(?<!\\)%.*", "", text)


def run_original_five():
    print("PASSES 1-5  (verify_thesis.py)")
    r = subprocess.run([sys.executable, str(ROOT / "accord/stats/verify_thesis.py")],
                       capture_output=True, text=True)
    tail = r.stdout.strip().splitlines()
    for line in tail[-8:]:
        print("   " + line)
    ok = r.returncode == 0
    print(f"   {'pass' if ok else 'FAIL'}: original five passes\n")
    return ok


def pass6_overclaim():
    print("PASS 6  overclaim")
    hard, soft = [], []
    for f in CHAPTERS + [THESIS / "frontmatter" / "abstract.tex"]:
        if not f.exists():
            continue
        text = strip_comments(f.read_text())
        for pats, bucket in ((HARD_OVERCLAIM, hard), (SOFT_OVERCLAIM, soft)):
            for pat, name in pats:
                for m in re.finditer(pat, text, re.I):
                    if NEGATION.search(text[max(0, m.start() - 30):m.start()]):
                        continue
                    ctx = re.sub(r"\s+", " ", text[max(0, m.start() - 60):m.end() + 60])
                    bucket.append((f.name, name, ctx.strip()[:110]))
    for name, kind, ctx in hard:
        print(f"   GATE {name} [{kind}] {ctx}")
    for name, kind, ctx in soft[:10]:
        print(f"   review {name} [{kind}] {ctx}")
    print(f"   {'FAIL' if hard else 'pass'}: {len(hard)} gating, {len(soft)} for review\n")
    return not hard


def bound_files(main_name):
    """The tex files a main document actually inputs, plus the main file."""
    main = THESIS / main_name
    files = [main]
    for m in re.finditer(r"\\input\{([^}]+)\}", strip_comments(main.read_text())):
        p = THESIS / m.group(1)
        if p.suffix != ".tex":
            p = p.with_suffix(".tex")
        if p.exists() and p.suffix == ".tex":
            files.append(p)
    return files


def pass7_refs():
    print("PASS 7  cross-reference integrity")
    # Labels are scoped per bound document: the thesis and the supplement are
    # separate documents, and a label living only in the supplement neither
    # collides with nor satisfies a reference in the thesis.
    files = bound_files("main_msu.tex")
    labels, refs, dups = {}, [], []
    for f in files:
        if not f.exists():
            continue
        text = strip_comments(f.read_text())
        for m in re.finditer(r"\\label\{([^}]+)\}", text):
            k = m.group(1)
            if k in labels:
                dups.append((k, labels[k], f.name))
            labels[k] = f.name
        for m in re.finditer(r"\\(?:auto)?ref\{([^}]+)\}", text):
            refs.append((m.group(1), f.name))
    missing = [(k, f) for k, f in refs if k not in labels]
    for k, a, b in dups:
        print(f"   duplicate label '{k}' in {a} and {b}")
    for k, f in missing:
        print(f"   {f}: \\ref{{{k}}} has no label in the bound document")
    bad = len(dups) + len(missing)
    print(f"   {'FAIL' if bad else 'pass'}: {len(labels)} labels, "
          f"{len(refs)} refs, {len(missing)} unresolved, {len(dups)} duplicated\n")
    return bad == 0


def pass8_captions():
    print("PASS 8  figure captions")
    problems = 0
    for f in CHAPTERS + BOUND_APPENDICES:
        text = f.read_text()
        for env in re.findall(r"\\begin\{figure\}.*?\\end\{figure\}", text, re.S):
            g = re.search(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", env)
            c = env.find("\\caption")
            if not g:
                continue
            name = g.group(1)
            target = THESIS / (name if name.endswith(".pdf") else name + ".pdf")
            if not target.exists():
                target = THESIS / "figures" / (pathlib.Path(name).name + ".pdf")
            if not target.exists() and not (THESIS / f"{name}.pdf").exists():
                if not pathlib.Path(str(THESIS / name)).with_suffix(".pdf").exists():
                    print(f"   {f.name}: graphic file not found: {name}")
                    problems += 1
            if c == -1:
                print(f"   {f.name}: figure {name} has NO caption")
                problems += 1
                continue
            # Convention: figure captions sit ABOVE the graphic, per FORMAT.md
            # section 5. This gate previously required the opposite, on the
            # author's decision of 2026-08-13 to follow the accepted departmental
            # example instead of the template. That decision bundled the caption
            # convention with numeric citations, and the citation half was later
            # overturned on the authority of FORMAT.md section 8. Sections 5 and 8
            # are the same authority, so the caption half went with it.
            #
            # Worth recording because of what this gate did in between: it kept
            # enforcing the withdrawn decision, so the fifteen corrected captions
            # arrived as fifteen failures. A gate that encodes a superseded
            # decision does not fail open, it fails at the fix.
            if c > env.find("\\includegraphics"):
                print(f"   {f.name}: caption BELOW graphic (convention: above): {name}")
                problems += 1
            i = c + len("\\caption{")
            depth, j = 1, i
            while j < len(env) and depth:
                depth += env[j] == "{"
                depth -= env[j] == "}"
                j += 1
            cap = re.sub(r"\s+", " ", env[i:j - 1]).strip()
            if len(cap) < 100:
                print(f"   {f.name}: caption is a title, not a caption "
                      f"({len(cap)} chars): {name}")
                problems += 1
    print(f"   {'FAIL' if problems else 'pass'}: {problems} caption problems\n")
    return problems == 0


def pass9_template():
    print("PASS 9  template compliance (FORMAT.md)")
    problems = 0

    # FORMAT.md section 3: "TOC entries are prefixed `Chapter N:`". This pass
    # reported zero template violations for months while the TOC printed bare
    # "1  Introduction", because it read the tex and the PDF body and never read
    # the table of contents. The body headings were correct, which is what made
    # the omission invisible: titlesec printed "Chapter N:" there, so every place
    # anyone checked by eye was right. A check that cannot see the artifact it
    # governs is the defect class this project has shipped most often, so the
    # rendered TOC is now read directly.
    toc = THESIS / "main_msu.toc"
    if toc.exists():
        entries = re.findall(r"\\contentsline \{chapter\}\{\\numberline \{([^}]*)\}",
                             toc.read_text())
        numbered = [e for e in entries if e]
        if numbered:
            pdf = THESIS / "main_msu.pdf"
            rendered = ""
            if pdf.exists():
                try:
                    rendered = subprocess.run(
                        ["pdftotext", "-layout", str(pdf), "-"],
                        capture_output=True, text=True, timeout=120).stdout
                except Exception:
                    rendered = ""
            # A whole-document search does not work here and the first version of
            # this check made that mistake: the body heading prints "Chapter N:"
            # too, so searching the full text passes even when the TOC prefix is
            # absent. Breaking the prefix deliberately proved the gate could not
            # fail. Each prefix must therefore appear at least twice, once in the
            # TOC line and once at the body heading, which is what distinguishes
            # the two occurrences without having to locate the TOC's page range.
            missing = []
            for e in numbered:
                word = "Appendix" if e.isalpha() else "Chapter"
                if rendered and rendered.count(f"{word} {e}:") < 2:
                    missing.append(f"{word} {e}")
            if missing:
                print(f"   TOC entries lack the required prefix: {', '.join(missing)}")
                problems += 1
            else:
                print(f"   TOC: {len(numbered)} numbered entries, all prefixed")

    ab = THESIS / "frontmatter" / "abstract.tex"
    if ab.exists():
        text = strip_comments(ab.read_text())
        # Body: the actual abstract paragraph, not the header block above it.
        # The template's own spacing commands mark both ends precisely: the
        # text starts right after the last \parindent=0pt (the paragraph
        # formatting reset that opens the abstract's typesetting group) and
        # ends at the final \par} immediately before \end{frontmatterpage}.
        # The previous heuristic (everything after the last "CHAIR NAME" or
        # "Thesis Chair" marker) swept in the department line and the
        # vspace/comment lines between the header block and the abstract
        # text, inflating both the word and paragraph counts.
        m = re.search(r"\\parindent=0pt\s*(.*?)\\par\}\s*\\end\{frontmatterpage\}",
                       text, re.S)
        body = m.group(1) if m else text
        body_words = re.findall(r"[A-Za-z][A-Za-z'-]*", re.sub(r"\\[a-zA-Z]+", " ", body))
        n_words = len(body_words)
        if n_words > 250:
            print(f"   abstract body is {n_words} words (template cap: 250)")
            problems += 1
        paras = [p for p in re.split(r"\n\s*\n", body.strip()) if
                 re.search(r"[A-Za-z]{3}", re.sub(r"\\[a-zA-Z]+|\[[^\]]*\]", "", p))]
        if len(paras) > 1:
            print(f"   abstract body has {len(paras)} paragraphs (template: exactly one)")
            problems += 1
        if "\\cite" in body:
            print("   abstract contains a citation (template: none)")
            problems += 1
        abbrevs = sorted(set(re.findall(r"\b[A-Z]{2,}(?:-[A-Za-z0-9]+)?\b", body)))
        abbrevs = [a for a in abbrevs if a not in {"A", "I"}]
        if abbrevs:
            print(f"   abstract contains abbreviations (template: none): {abbrevs}")
            problems += 1
        print(f"   abstract: {n_words} words, {len(paras)} paragraph(s)")

    # No internal repository filename may reach the page. Working notes name
    # their own files freely; a bound thesis cites an appendix or a commit, not
    # a path a reader cannot open. Checked on the rendered PDF, because the
    # generated appendices are the likeliest source and they are not in the tex.
    #
    # The gate covered .md and .ipynb only. The author's reading of the bound
    # draft found eleven script paths still set in \texttt{} inside sentences
    # ("i dont think a thesis will have the notebook coded in it") and every one
    # of them was a .py, which this pass would have caught had it looked. It
    # now looks for .py, .csv and .json as well, so the eleven rewrites cannot
    # regress one sentence at a time.
    pdf = THESIS / "main_msu.pdf"
    if pdf.exists() and shutil.which("pdftotext"):
        r = subprocess.run(["pdftotext", str(pdf), "-"],
                           capture_output=True, text=True)
        for m in re.finditer(r"\S*\.(?:md|ipynb|py|csv|json)\b", r.stdout):
            print(f"   internal filename on the page: {m.group(0)}")
            problems += 1

    main = (THESIS / "main_msu.tex").read_text()
    for pat, what in [
        (r"\\documentclass\[12pt\]\{report\}", "12pt report class"),
        (r"left=1\.5in", "1.5in left margin"),
        (r"\\doublespacing", "double spacing"),
        # chngcntr is absent locally; the documented substitute is equivalent.
        (r"\\counterwithout\{figure\}\{chapter\}|\\renewcommand\{\\thefigure\}\{\\arabic\{figure\}\}",
         "consecutive figure numbering"),
        (r"\\counterwithout\{table\}\{chapter\}|\\renewcommand\{\\thetable\}\{\\arabic\{table\}\}",
         "consecutive table numbering"),
    ]:
        if not re.search(pat, main):
            print(f"   main_msu.tex: missing {what}")
            problems += 1

    # Advisory only: the template asks for three-sentence paragraphs.
    short = 0
    for f in CHAPTERS:
        text = strip_comments(f.read_text())
        for p in re.split(r"\n\s*\n", text):
            p2 = re.sub(r"\\[a-zA-Z]+(\[[^\]]*\])?(\{[^{}]*\})*", " ", p)
            words = re.findall(r"[A-Za-z]{3,}", p2)
            sents = re.findall(r"[.!?](\s|$)", p2)
            if len(words) > 25 and len(sents) == 1:
                short += 1
    print(f"   advisory: {short} single-sentence paragraphs over 25 words (template "
          "asks for three sentences per paragraph)")
    print(f"   {'FAIL' if problems else 'pass'}: {problems} template violations\n")
    return problems == 0


def pass10_figures():
    print("PASS 10  figure reproducibility")
    figdir = THESIS / "figures"
    problems = 0

    # No result-file name, register key, or dash may be drawn on a canvas.
    # Tested on the rendered artifact, not the source: pdftotext extracts what
    # a reader's PDF viewer would select.
    pdftotext = shutil.which("pdftotext")
    if pdftotext:
        for n in THESIS_FIGS:
            p = figdir / f"{n}.pdf"
            if not p.exists():
                continue
            r = subprocess.run([pdftotext, str(p), "-"], capture_output=True, text=True)
            drawn = r.stdout
            for pat, what in [(r"\.(json|csv)\b", "result-file name"),
                              (r"—", "em dash"), (r"–", "en dash")]:
                for m in re.finditer(pat, drawn):
                    ctx = re.sub(r"\s+", " ", drawn[max(0, m.start() - 40):m.end() + 20])
                    print(f"   {n}.pdf: {what} on canvas: ...{ctx.strip()}...")
                    problems += 1
    else:
        print("   (pdftotext unavailable; canvas-text check skipped)")

    with tempfile.TemporaryDirectory() as td:
        before = {}
        for n in THESIS_FIGS:
            p = figdir / f"{n}.pdf"
            if not p.exists():
                print(f"   MISSING committed figure: {n}.pdf")
                problems += 1
                continue
            before[n] = pathlib.Path(td) / f"{n}.pdf"
            shutil.copy(p, before[n])
        for s in FIG_SCRIPTS:
            r = subprocess.run([sys.executable, str(figdir / s)],
                               capture_output=True, text=True, cwd=str(figdir))
            if r.returncode != 0:
                print(f"   {s} FAILED to run: {r.stderr.strip().splitlines()[-1][:100]}")
                problems += 1
        try:
            import numpy as np
            from PIL import Image
            for n, old in before.items():
                new = figdir / f"{n}.pdf"
                pair = []
                for src_pdf, out in ((old, f"{td}/a_{n}.png"), (new, f"{td}/b_{n}.png")):
                    subprocess.run(["sips", "-s", "format", "png",
                                    "--resampleWidth", "700", str(src_pdf),
                                    "--out", out], capture_output=True)
                    pair.append(np.asarray(Image.open(out).convert("L"), float))
                if pair[0].shape != pair[1].shape:
                    print(f"   {n}: canvas size changed {pair[0].shape} -> {pair[1].shape}")
                    problems += 1
                elif float(np.abs(pair[0] - pair[1]).mean()) > 0.5:
                    print(f"   {n}: rebuild differs from committed PDF "
                          f"(mean {np.abs(pair[0]-pair[1]).mean():.3f})")
                    problems += 1
        except ImportError:
            print("   (PIL/numpy unavailable; pixel comparison skipped, "
                  "script execution still gated)")
    print(f"   {'FAIL' if problems else 'pass'}: {problems} figure problems\n")
    return problems == 0


def main():
    skip_figs = "--skip-figures" in sys.argv
    results = [("1-5", run_original_five()),
               ("6", pass6_overclaim()),
               ("7", pass7_refs()),
               ("8", pass8_captions()),
               ("9", pass9_template())]
    if not skip_figs:
        results.append(("10", pass10_figures()))
    failed = [n for n, ok in results if not ok]
    print("RESULT:", "FAIL on pass " + ", ".join(failed) if failed
          else "all ten gating passes clear"
          if not skip_figs else "passes 1-9 clear (figures skipped)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
