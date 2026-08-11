"""
Generate the thesis appendices from the project records, so they cannot drift
from their sources.

Appendix A  one line per correction and per post-freeze revision
Appendix B  provenance of every reported quantity, grouped by experiment

Run:  python3 accord/stats/make_appendices.py
"""

import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]
THESIS = ROOT / "accord" / "thesis"

GREEK = {"κ": r"$\kappa$", "ρ": r"$\rho$", "α": r"$\alpha$",
         "β": r"$\beta$", "σ": r"$\sigma$", "μ": r"$\mu$",
         "χ": r"$\chi$", "γ": r"$\gamma$", "≥": r"$\geq$",
         "≤": r"$\leq$", "×": r"$\times$", "→": r"$\rightarrow$"}


def esc(s):
    """LaTeX-safe, and free of the dashes the thesis forbids."""
    for g, tex in GREEK.items():
        s = s.replace(g, tex)
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
    s = re.sub(r"`(.+?)`", r"TTSTART\1TTEND", s)
    for ch, rep in (("\\", " "), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"),
                    ("#", r"\#"), ("$", r"\$"), ("~", " "), ("^", " "),
                    ("{", " "), ("}", " ")):
        s = s.replace(ch, rep)
    s = s.replace("TTSTART", r"\texttt{").replace("TTEND", "}")
    s = s.replace("—", ", ").replace("–", " to ").replace("−", "-")
    return re.sub(r"[^\x00-\x7f]", "", s)


def esc_breakable(s):
    """Like esc(), but for source-file paths: insert \\allowbreak after every '.'
    and '/' so the longtable column can wrap them instead of overrunning the
    margin (this project has no xurl installed)."""
    s = esc(s)
    s = s.replace(".", r".\allowbreak{}")
    s = s.replace("/", r"/\allowbreak{}")
    return s


# The two absolute local filesystem paths long enough, and space-free enough,
# to risk overrunning the page in both outputs -- \allowbreak fixes pdflatex,
# but pandoc drops it, so a plain-text table cell in Word is one long unbroken
# run. Abbreviate the shared, uninformative directory prefix; the full path is
# spelled out once, in a note under the table, so nothing is actually lost.
PATH_ABBREV = [
    ("/Users/rickygole/Library/CloudStorage/GoogleDrive-ricky.gole@morgan.edu"
     "/My Drive/thesis/outputs/", "[GDRIVE]/"),
    ("/Users/rickygole/mimic_local_only/", "[MIMIC]/"),
]


def abbreviate_path(s):
    for full, short in PATH_ABBREV:
        if s.startswith(full):
            return short + s[len(full):]
    return s


def main():
    corr = (ROOT / "accord" / "CORRECTIONS.md").read_text()
    entries = re.findall(r"^## (C\d+)\.\s*(.+)$", corr, re.M)
    rev = (ROOT / "accord" / "REVISION_LOG.md").read_text()
    revs = re.findall(r"^### (R\d+)\s*\((\d{4}-\d{2}-\d{2})\)\s*[-—]*\s*(.+)$", rev, re.M)

    a = [r"\chapter{The Correction Record}", r"\label{app:corrections}", "",
         "This appendix lists every correction issued during the work, in the order it was issued.",
         "Each entry records a claim that was withdrawn, narrowed, or confirmed against evidence.",
         "The full text of each, with its artifact and reasoning, is retained in the project",
         r"repository. The record is included because the sequence is part of the result: it shows",
         "which conclusions moved, and what moved them.", "",
         r"\section{Corrections}", "", r"\singlespacing",
         r"\begin{longtable}{@{}p{0.6in}p{4.6in}@{}}",
         r"\textbf{ID} & \textbf{Correction}\\", r"\midrule", r"\endhead"]
    a += [f"{cid} & {esc(t.strip())}\\\\" for cid, t in entries]
    a += [r"\end{longtable}", r"\doublespacing", ""]

    if revs:
        a += [r"\section{Revisions After the Record Closed}", "",
              "The correction record was closed once the frame was frozen. Findings after that point",
              "change a number, a chapter, or a stated limitation, and by rule cannot change the",
              "thesis statement. They are listed separately for that reason.", "",
              r"\singlespacing", r"\begin{longtable}{@{}p{0.6in}p{0.9in}p{3.8in}@{}}",
              r"\textbf{ID} & \textbf{Date} & \textbf{Revision}\\", r"\midrule", r"\endhead"]
        a += [f"{rid} & {d} & {esc(t.strip())}\\\\" for rid, d, t in revs]
        a += [r"\end{longtable}", r"\doublespacing", ""]

    (THESIS / "msu_appA_corrections.tex").write_text("\n".join(a) + "\n")

    F = json.loads((ROOT / "results" / "accord" / "VERIFIED_FACTS.json").read_text())
    groups = {}
    for k, rec in sorted(F.items()):
        groups.setdefault(k.split(".")[0], []).append(rec)

    b = [r"\chapter{Provenance of Reported Quantities}", r"\label{app:provenance}", "",
         f"Every quantity reported in this thesis is drawn from a register of {len(F)} values, each",
         "extracted from a stored result file by script rather than transcribed. The register is",
         "grouped below by the experiment that produced it, with the file each group was read from.", "",
         r"\singlespacing", r"\begin{longtable}{@{}p{1.9in}p{0.7in}p{2.6in}@{}}",
         r"\textbf{Group} & \textbf{Values} & \textbf{Source file}\\", r"\midrule", r"\endhead"]
    abbreviated = False
    for g, items in sorted(groups.items()):
        arts = sorted({i["artifact"] for i in items})
        shown = abbreviate_path(arts[0])
        abbreviated = abbreviated or (shown != arts[0])
        extra = f" and {len(arts) - 1} more" if len(arts) > 1 else ""
        b.append(f"{esc(g)} & {len(items)} & {esc_breakable(shown)}{extra}\\\\")
    b += [r"\end{longtable}"]
    if abbreviated:
        prefixes = "; ".join(f"{short} is {esc_breakable(full)}"
                              for full, short in PATH_ABBREV)
        b.append(r"\apanote{Two source paths are absolute paths on the machine the analysis "
                  f"was run on and are abbreviated above so the column fits the page: {prefixes}.}}")
    b += [r"\doublespacing", ""]
    (THESIS / "msu_appB_provenance.tex").write_text("\n".join(b) + "\n")

    print(f"Appendix A: {len(entries)} corrections, {len(revs)} revisions")
    print(f"Appendix B: {len(groups)} groups covering {len(F)} values")


if __name__ == "__main__":
    main()
