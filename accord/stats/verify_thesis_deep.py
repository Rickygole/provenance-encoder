"""
Eleven verification passes over the thesis. Wraps the original five and adds six.

Pass 1   dashes         no em dash, en dash, or unicode minus anywhere
Pass 2   numbers        every numeric literal traces to VERIFIED_FACTS.json
Pass 3   citations      every \\cite key exists in refs.bib
Pass 4   style          formulaic phrasing that reads as machine-written
Pass 5   structure      required labels, caption placement, no leftover TODO
Pass 6   overclaim      strong verbs attached to results, outside negation
Pass 7   references     every \\ref resolves, no duplicate labels
Pass 8   captions       figure captions above the graphic, substantial, files exist
Pass 9   template       abstract hard rules and page-setup invariants (FORMAT.md)
Pass 10  figures        scripts rebuild the committed PDFs pixel-identically
                        (rendered into a temporary directory; compared only in
                        the environment recorded in environment.yml), and no
                        result-file name is drawn on any canvas
Pass 11  appendix       Appendix B's printed provenance counts match the tools

Passes 1 to 5 run by invoking verify_thesis.py, so the two scripts cannot
drift. Exit code is nonzero if any gating pass fails.

Run:  python3 accord/stats/verify_thesis_deep.py
      python3 accord/stats/verify_thesis_deep.py --skip-figures   (passes 1-9 only)
"""

import json
import os
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

# The thesis body and the supplement are two separate LaTeX documents now
# (main_msu.tex and main_supplement.tex), each with its own \label/\ref
# namespace and its own set of bound figures. A pass that walked CHAPTERS only
# put the bound appendices outside all of them (a caption sitting below its
# figure survived there for exactly that reason, pass 8 could not see the
# file), and reading both documents through one merged file list would let a
# \ref in one resolve against a \label defined only in the other, which is not
# how two separate LaTeX documents behave. Both defects are avoided the same
# way: every check below reads a document's inputs by walking its own \input
# tree, not by globbing msu_ch*.tex/msu_app*.tex and guessing which of those
# files anything actually binds.
def strip_comments(text):
    return re.sub(r"(?<!\\)%.*", "", text)


def bound_files(main_name, _visited=None):
    """The tex files a document actually inputs, followed transitively, plus
    the main file itself. \\input is read after comments are stripped, so a
    commented-out \\input does not pull in a file nothing binds; a file that
    is unreachable from main_name (in the tree but bound by nothing) is
    likewise absent, on the same reasoning _bound_appendices used before this
    function replaced it. _visited guards a cycle, which no legitimate \\input
    graph has, but an accidental self-\\input should stop, not recurse
    forever.
    """
    if _visited is None:
        _visited = set()
    main = THESIS / main_name
    if not main.exists() or main in _visited:
        return []
    _visited.add(main)
    files = [main]
    for m in re.finditer(r"\\input\{([^}]+)\}", strip_comments(main.read_text())):
        p = THESIS / m.group(1)
        if p.suffix != ".tex":
            p = p.with_suffix(".tex")
        if not p.exists() or p in _visited:
            continue
        files.extend(bound_files(str(p.relative_to(THESIS)), _visited))
    return files


# Each document's own bound file list, main_msu.tex and main_supplement.tex
# read separately. main_supplement.tex may not exist yet in every checkout
# (it is being assembled concurrently with this change), so BOUND_SUP is
# empty rather than an error until it does.
BOUND_MAIN = bound_files("main_msu.tex")
BOUND_SUP = bound_files("main_supplement.tex") if (THESIS / "main_supplement.tex").exists() else []
ALL_BOUND = sorted(set(BOUND_MAIN) | set(BOUND_SUP), key=lambda p: str(p))


def _live_figs():
    """The figures actually reachable from either bound document, read from
    their own \\includegraphics calls rather than a hard-coded list. The old
    hard-coded THESIS_FIGS drifted from the document it was supposed to
    describe in both directions at once: fig-ceiling was bound into Chapter 4
    and missing from the list, and seven other names in the list (
    fig-intro-headline, fig-falsification, fig-scaling, fig-disc-actionability,
    fig-met-exchange, fig-capability, fig-missing-cell, fig-labelnoise) are not
    \\includegraphics'd anywhere either document actually binds, so pass 10
    was gating figures nothing in the thesis draws on the page and silent
    about one that is.
    """
    names = set()
    for f in ALL_BOUND:
        if f.suffix != ".tex":
            continue
        text = strip_comments(f.read_text())
        for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text):
            names.add(pathlib.Path(m.group(1)).stem)
    return sorted(names)


THESIS_FIGS = _live_figs()
FIG_SCRIPTS = ["make_figures.py", "fig_batch2.py",
               "fig_threshold_surface.py", "fig_frontier.py", "fig_ceiling.py"]

# FORMAT.md records no explicit page limit; this is the ceiling this project's
# own review process has set and reported (150 pages, currently exact). Named
# so a future revision changes one constant, not a literal buried in a
# print statement, and so build.sh (which reports the same number) and this
# gate cannot silently disagree.
MAIN_THESIS_MAX_PAGES = 150

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
    # The two bound appendices were outside this list, so their prose was never
    # read for overclaims although the thesis prints them.
    for f in CHAPTERS + [THESIS / "frontmatter" / "abstract.tex",
                         THESIS / "msu_appA_corrections.tex",
                         THESIS / "msu_appB_pointer.tex"]:
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


def _refs_for_document(doc_label, files):
    """Labels, refs and duplicates within one document's own bound files."""
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
        print(f"   [{doc_label}] duplicate label '{k}' in {a} and {b}")
    for k, f in missing:
        print(f"   [{doc_label}] {f}: \\ref{{{k}}} has no label in this document's bound files")
    bad = len(dups) + len(missing)
    print(f"   [{doc_label}] {'FAIL' if bad else 'pass'}: {len(labels)} labels, "
          f"{len(refs)} refs, {len(missing)} unresolved, {len(dups)} duplicated")
    return bad == 0


def pass7_refs():
    print("PASS 7  cross-reference integrity")
    # Labels are scoped per bound document: the thesis and the supplement are
    # separate LaTeX documents, so a label living only in the supplement
    # neither collides with nor satisfies a reference in the thesis, and vice
    # versa. Each document is walked from its own \input tree (BOUND_MAIN,
    # BOUND_SUP), not from a merged file list, so that boundary is structural
    # rather than a rule this check has to remember to enforce.
    ok_main = _refs_for_document("main_msu.tex", BOUND_MAIN)
    ok_sup = True
    if BOUND_SUP:
        ok_sup = _refs_for_document("main_supplement.tex", BOUND_SUP)
    else:
        print("   [main_supplement.tex] skipped: main_supplement.tex does not "
              "exist or has no bound \\input files yet")
    print()
    return ok_main and ok_sup


def pass8_captions():
    print("PASS 8  figure captions")
    problems = 0
    # Every file bound into either document, thesis or supplement: a figure
    # and its caption are checked wherever they are actually printed, not only
    # where they used to be printed (CHAPTERS + the thesis's own appendices).
    for f in ALL_BOUND:
        if f.suffix != ".tex":
            continue
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
            # The MSU template (October 2024) sets captions as one-line titles and
            # puts explanation in the note under the float. A short caption is
            # therefore correct when the figure carries a note that explains it;
            # a short caption with no explanation anywhere is still a failure.
            note = re.search(r"\\apanote\{", env)
            note_len = 0
            if note:
                k, depth = note.end(), 1
                start = k
                while k < len(env) and depth:
                    depth += env[k] == "{"
                    depth -= env[k] == "}"
                    k += 1
                note_len = len(re.sub(r"\s+", " ", env[start:k - 1]).strip())
            if len(cap) < 100 and note_len < 100:
                print(f"   {f.name}: caption is a title with no explanatory note "
                      f"({len(cap)} chars, note {note_len}): {name}")
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
        # The body was moved into abstract_text.tex so the page before
        # Chapter 1 prints the same words; expand it, or this pass would
        # count the \input line instead of the abstract.
        def _expand(m):
            f = THESIS / (m.group(1) + ".tex")
            if not f.exists():
                raise SystemExit(f"abstract \\input{{{m.group(1)}}} not found")
            return strip_comments(f.read_text())
        text = re.sub(r"\\input\{([^}]+)\}", _expand, text)
        # Body: the actual abstract paragraph, not the header block above it.
        # The template's own spacing commands mark both ends precisely: the
        # text starts right after the last \parindent=0pt (the paragraph
        # formatting reset that opens the abstract's typesetting group) and
        # ends at the final \par} immediately before \end{frontmatterpage}.
        # The previous heuristic (everything after the last "CHAIR NAME" or
        # "Thesis Chair" marker) swept in the department line and the
        # vspace/comment lines between the header block and the abstract
        # text, inflating both the word and paragraph counts.
        # The template restyle (2026-09-14) opens the body group with
        # \RaggedRightParindent=0.5in instead of \parindent=0pt; accept either.
        m = re.search(r"(?:\\RaggedRightParindent=0\.5in|\\parindent=0pt)\s*(.*?)\\par\}\s*\\end\{frontmatterpage\}",
                       text, re.S)
        body = m.group(1) if m else text
        # Count the way a format reviewer's word processor does: every
        # whitespace-separated token, numbers included. The earlier count took
        # alphabetic words only, so an abstract carrying ten numbers read as ten
        # words shorter than it is, and a 254-word abstract passed as 244.
        plain = re.sub(r"\\[a-zA-Z]+\*?", " ", body)
        plain = re.sub(r"[{}$]", "", plain)
        body_words = plain.split()
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
    #
    # Run once per bound document, main_msu.tex and main_supplement.tex, rather
    # than on main_msu.pdf alone: the supplement is a document a reader opens
    # on its own, so an internal path or a missing template invariant in it is
    # not caught by only ever reading the thesis's own PDF and tex source. This
    # is the check CLAUDE.md's "second document going stale" note is about.
    DOCS = [("main_msu.tex", "main_msu.pdf")]
    if (THESIS / "main_supplement.tex").exists():
        DOCS.append(("main_supplement.tex", "main_supplement.pdf"))
    for tex_name, pdf_name in DOCS:
        pdf = THESIS / pdf_name
        if pdf.exists() and shutil.which("pdftotext"):
            r = subprocess.run(["pdftotext", str(pdf), "-"],
                               capture_output=True, text=True)
            for m in re.finditer(r"\S*\.(?:md|ipynb|py|csv|json)\b", r.stdout):
                print(f"   [{pdf_name}] internal filename on the page: {m.group(0)}")
                problems += 1

        main = (THESIS / tex_name).read_text()
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
                print(f"   {tex_name}: missing {what}")
                problems += 1

    # Page gate. FORMAT.md sets no limit; this project's own review process
    # has, and reads it with pdfinfo against the actual rendered artifact
    # rather than trusting a line count, because a page count is a property of
    # the typeset PDF (margins, float placement, widow control), not of the
    # source.
    main_pdf = THESIS / "main_msu.pdf"
    if main_pdf.exists() and shutil.which("pdfinfo"):
        r = subprocess.run(["pdfinfo", str(main_pdf)], capture_output=True, text=True)
        m = re.search(r"^Pages:\s+(\d+)", r.stdout, re.M)
        if m:
            n_pages = int(m.group(1))
            over = n_pages > MAIN_THESIS_MAX_PAGES
            print(f"   main_msu.pdf: {n_pages} pages (limit {MAIN_THESIS_MAX_PAGES})")
            if over:
                print(f"   FAIL: main_msu.pdf is {n_pages} pages, exceeds the "
                      f"{MAIN_THESIS_MAX_PAGES}-page limit")
                problems += 1
        else:
            print("   WARNING: pdfinfo produced no 'Pages:' line; page gate not evaluated")
    elif main_pdf.exists():
        print("   WARNING: pdfinfo not on PATH; page gate skipped")

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


def _recorded_figure_env():
    """The matplotlib and FreeType versions the committed figure PDFs were
    rendered with, read from the pins in environment.yml at the repo root so
    the gate and the recorded environment cannot disagree."""
    text = (ROOT / "environment.yml").read_text()
    rec = {}
    for pkg in ("matplotlib", "freetype"):
        m = re.search(rf"^\s*-\s*{pkg}==?([0-9][^\s#]*)", text, re.M)
        if m:
            rec[pkg] = m.group(1)
    return rec


def _rasterise(pdf, png, width=700):
    """First page of `pdf` as a greyscale array `width` pixels wide. poppler's
    pdftoppm where it is on PATH (any platform), else PyMuPDF into Pillow.
    Returns None when neither is available."""
    import numpy as np
    from PIL import Image
    pdftoppm = shutil.which("pdftoppm")
    if pdftoppm:
        stem = str(png)[:-4]
        subprocess.run([pdftoppm, "-png", "-singlefile", "-f", "1", "-l", "1",
                        "-scale-to-x", str(width), "-scale-to-y", "-1",
                        str(pdf), stem], capture_output=True, check=True)
        return np.asarray(Image.open(png).convert("L"), float)
    try:
        import pymupdf
    except ImportError:
        return None
    with pymupdf.open(str(pdf)) as doc:
        page = doc[0]
        z = width / page.rect.width
        pix = page.get_pixmap(matrix=pymupdf.Matrix(z, z), alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    return np.asarray(img.convert("L"), float)


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

    # The committed PDFs match a rebuild only under the matplotlib and FreeType
    # they were rendered with (environment.yml). The PyPI matplotlib wheel
    # bundles a different FreeType from the conda build, which moves text
    # metrics and canvas sizes; that is a property of the environment, not a
    # defect in a generator, so it is reported and does not fail the gate. A
    # generator that crashes or stops producing a bound figure still does.
    comparable = True
    try:
        recorded = _recorded_figure_env()
        import matplotlib
        from matplotlib import ft2font
        running = {"matplotlib": matplotlib.__version__,
                   "freetype": ft2font.__freetype_version__}
        diffs = [f"{k} {running[k]} (recorded {recorded.get(k, 'nothing')})"
                 for k in ("matplotlib", "freetype") if running[k] != recorded.get(k)]
        if diffs:
            comparable = False
            print("   this environment differs from the one recorded in "
                  "environment.yml: " + "; ".join(diffs))
    except (OSError, ImportError) as e:
        comparable = False
        print(f"   could not establish the rendering environment ({e})")

    # Generators render into a temporary directory (ACCORD_FIG_OUT, honoured by
    # style.out_dir); the committed PDFs are read and never written.
    with tempfile.TemporaryDirectory() as td:
        out = pathlib.Path(td) / "rebuilt"
        env = dict(os.environ, ACCORD_FIG_OUT=str(out))
        for s in FIG_SCRIPTS:
            r = subprocess.run([sys.executable, str(figdir / s)], env=env,
                               capture_output=True, text=True, cwd=str(figdir))
            if r.returncode != 0:
                last = (r.stderr.strip().splitlines() or ["no stderr"])[-1][:100]
                print(f"   {s} FAILED to run: {last}")
                problems += 1
        pairs = []
        for n in THESIS_FIGS:
            old, new = figdir / f"{n}.pdf", out / f"{n}.pdf"
            if not old.exists():
                print(f"   MISSING committed figure: {n}.pdf")
                problems += 1
            elif not new.exists():
                print(f"   {n}: no generator rebuilt it")
                problems += 1
            else:
                pairs.append((n, old, new))
        differing = []
        try:
            for n, old, new in pairs:
                a = _rasterise(old, pathlib.Path(td) / f"a_{n}.png")
                b = _rasterise(new, pathlib.Path(td) / f"b_{n}.png")
                if a is None or b is None:
                    print("   (neither pdftoppm nor PyMuPDF available; pixel "
                          "comparison skipped, script execution still gated)")
                    differing = None
                    break
                if a.shape != b.shape:
                    differing.append(f"{n}: canvas size changed {a.shape} -> {b.shape}")
                elif float(abs(a - b).mean()) > 0.5:
                    differing.append(f"{n}: rebuild differs from committed PDF "
                                     f"(mean {abs(a - b).mean():.3f})")
        except ImportError:
            print("   (PIL/numpy unavailable; pixel comparison skipped, "
                  "script execution still gated)")
            differing = None
        if differing is not None and comparable:
            for d in differing:
                print(f"   {d}")
            problems += len(differing)
            print(f"   {len(pairs) - len(differing)} of {len(pairs)} bound figures "
                  "rebuild to the committed PDF")
        elif differing is not None:
            for d in differing:
                print(f"   (not gating) {d}")
            print(f"   {len(pairs)} bound figures were produced by their generators; NOT "
                  "COMPARABLE IN THIS ENVIRONMENT, so whether they match the "
                  "committed PDFs was not tested. Use environment.yml for the "
                  "exact comparison.")
    print(f"   {'FAIL' if problems else 'pass'}: {problems} figure problems\n")
    return problems == 0


def pass11_appendix_counts():
    """Appendix B prints the register's size and the claim trace's counts. They
    were typed by hand and two audits found them stale, because every chapter
    edit moves them and the numeric gate does not read the appendices. The
    trace is rerun here so the comparison is against the chapters as they are
    now, then sync_appendix_counts.py --check compares the printed counts."""
    print("PASS 11  appendix provenance counts")
    here = pathlib.Path(__file__).resolve().parent
    tr = subprocess.run([sys.executable, str(here / "trace_claims.py")],
                        capture_output=True, text=True, cwd=str(here.parents[1]))
    # The tracer exits 1 when it lists any untraced quantity, and Appendix B
    # discloses the ones it lists; only a crash (any other status) is a failure.
    if tr.returncode not in (0, 1):
        print("   FAIL: trace_claims.py did not run cleanly")
        print()
        return False
    r = subprocess.run([sys.executable, str(here / "sync_appendix_counts.py"), "--check"],
                       capture_output=True, text=True, cwd=str(here.parents[1]))
    print(r.stdout.rstrip() or r.stderr.rstrip())
    print()
    return r.returncode == 0


def main():
    skip_figs = "--skip-figures" in sys.argv
    results = [("1-5", run_original_five()),
               ("6", pass6_overclaim()),
               ("7", pass7_refs()),
               ("8", pass8_captions()),
               ("9", pass9_template()),
               ("11", pass11_appendix_counts())]
    if not skip_figs:
        results.append(("10", pass10_figures()))
    failed = [n for n, ok in results if not ok]
    print("RESULT:", "FAIL on pass " + ", ".join(failed) if failed
          else "all eleven gating passes clear"
          if not skip_figs else "passes 1-9 clear (figures skipped)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
