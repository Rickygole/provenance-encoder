"""
Generate the thesis appendices from the project records, so they cannot drift
from their sources.

Appendix A  one line per correction and per post-freeze revision
Appendix B  provenance of every reported quantity, grouped by experiment
Appendix B pointer  the short section (msu_appB_pointer.tex) that tells the
    reader the register, the claim trace and the survey table are bound in
    the supplement, and states how big each one is

msu_appB_pointer.tex used to be typed by hand. Its counts (how many values the
register holds, how many numeric occurrences the trace resolves, how many are
unique against how many are ambiguous, how many are untraced) drifted from the
generated numbers during an audit, because a hand-typed number does not move
when the script behind it is corrected. It is generated here now, from the
register itself and from results/accord/CLAIM_TRACE.json, the artifact
accord/stats/trace_claims.py writes: run that script before this one, or
before reading this file's printed counts as current, since the pointer's
trace figures are only as fresh as the last time trace_claims.py ran.

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


# Internal repository filenames must not reach the page. The correction and
# revision records are working notes and name their own files freely; a bound
# thesis cites an appendix or a commit, not a path a reader cannot open. Applied
# in esc(), so any heading text lifted from those records is sanitised on the
# way in and no future entry can leak one by being written before anyone looks.
INTERNAL_FILES = {
    "CORRECTIONS.md": "the correction record",
    "REVISION_LOG.md": "the revision log",
    "DESIGN.md": "a design note",
    "FRAMING.md": "the framing note",
    "FORMAT.md": "the format specification",
    "CONTRIBUTION.md": "the contribution note",
    "THESIS_STATUS.md": "a status note",
    "PHD_PLAN.md": "a planning note",
    "C44_PREREGISTRATION.md": "the pre-registered adjudication rule",
}

# Absolute filesystem paths, same rule as above but for a live machine path
# rather than a repository filename: a path under this repository is a
# citation nobody outside this machine can open, a path under the credentialed
# local-only directory names where restricted data sits, and a path under the
# Google Drive mount carries the operator's institutional email address. The
# register (results/accord/VERIFIED_FACTS.json) is normalised at the source so
# these patterns should not occur in an artifact string reaching this module;
# these are matched here too, as the second line of defense, so a future
# entry written straight from a live filesystem path is still caught.
_DRIVE_RE = re.compile(r"GoogleDrive-[^/]+/My Drive/(.*)")
_LOCAL_ONLY_RE = re.compile(r"/mimic_local_only/(.*)")
_REPO_RE = re.compile(r"/provenance-encoder/(.*)")


def normalize_artifact_path(s):
    """Rewrite the leading absolute filesystem path in an artifact string, if
    any, to a form safe to publish. A path under this repository becomes
    repo-relative; a path under the credentialed local-only directory or the
    Google Drive mount becomes a bracketed token. Only the directory prefix
    changes: the filename and any trailing ", derived: ..." explanation
    survive unchanged."""
    if "/Users/" not in s and "/home/" not in s:
        return s
    head, sep, tail = s.partition(", ")
    m = _DRIVE_RE.search(head)
    if m:
        head = "[DRIVE]/" + m.group(1)
    else:
        m = _LOCAL_ONLY_RE.search(head)
        if m:
            head = "[LOCAL]/" + m.group(1)
        else:
            m = _REPO_RE.search(head)
            if m:
                head = m.group(1)
    return head + sep + tail


def strip_internal_paths(s):
    """Replace any internal .md filename, with or without its directory, by a
    description of what it is, and normalise any absolute filesystem path (see
    normalize_artifact_path) so no machine path or institutional email address
    reaches the page."""
    s = normalize_artifact_path(s)
    for name, phrase in INTERNAL_FILES.items():
        s = re.sub(r"(?:[\w./-]*/)?" + re.escape(name), phrase, s)
    # anything else ending .md: describe rather than name
    s = re.sub(r"(?:[\w./-]*/)?[\w.-]+\.md\b", "a project record", s)
    return s


def esc(s):
    """LaTeX-safe, free of the dashes the thesis forbids, and free of internal
    repository filenames."""
    s = strip_internal_paths(s)
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


# The ten corrections that changed a number this thesis still prints, hand-selected because they
# trace directly to a value in a current chapter. Each is given three columns instead of one: the
# claim as originally stated, the evidence that retired it, and where the corrected version now
# lives. The other fifty-two corrections, and all nine post-freeze revisions, changed the record but
# not a number surviving into these six chapters, and are named by identifier in one paragraph
# rather than repeated here. Edit this table, not the generated output.
NUMBER_CHANGING = [
    ("C1", "The authority parameters converge during training to reflect clinical note hierarchy.",
     "Initialised at the exact pair of values later reported as discovered (1.0, 0.5); across three "
     "seeds they move to 1.0038 and 0.5076, behind a normalisation layer removing most gradient.",
     "Ch. 4, Architecture: authority conditioning is an initialisation artifact."),
    ("C2", "Rule-type head macro F1 of 0.9603 shows the auxiliary head learned rule structure.",
     "The rule identifier is written into the input feature vector one layer from the head "
     "predicting it; out-of-domain transfer collapses to 0.0654 (sepsis) and 0.0595 (AKI).",
     "Ch. 4, Architecture: the rule-type head is label leakage."),
    ("C3", "Kappa of 1.000 between two independent pharmacist instruments established reliability.",
     "The two instruments were the binary label and the ordinal danger rating; a non-contradiction "
     "has no danger by definition, so agreement was guaranteed by construction, not measured.",
     "Ch. 3, Gold Standard: what this design can and cannot establish."),
    ("C5", "The detector's evaluation AUROC on the gold set is a clean measurement.",
     "Evaluated rows overlapped the training pool; the contaminated run reads 0.4595 on the same "
     "253 pairs the decontaminated ensemble reads 0.5145 on.",
     "Ch. 4, decontaminated-detector table."),
    ("C8", "The candidate pool was split into train, validation and test sets appropriately.",
     "Splits were row-level, placing candidates sharing a subject on both sides; every split is now "
     "subject-disjoint, asserted in code.",
     "Ch. 3, extraction and splitting."),
    ("C18", "Decontamination excluded every gold-set subject from the training pool.",
     "158 of 266 gold subjects were also in the training pool under the original row-level "
     "exclusion; exclusion is now subject-level, asserted in code.",
     "Ch. 3, extraction and splitting; Ch. 4, decontaminated tables."),
    ("C37", "Domain pretraining materially improves the detector over a from-scratch encoder.",
     "At three seeds each, the pretrained encoder reaches 0.7693 (SD 0.0102) AUROC on the mortality "
     "control task and a tenth-the-parameters from-scratch encoder reaches 0.7662 (SD 0.0073).",
     "Ch. 4, mortality control task table."),
    ("C44", "The AG News pretraining benefit at one training budget generalises without a check.",
     "A pre-registered 48-run, four-budget sweep shows the 4.53-point gap persists at 24 epochs "
     "and both arms are converged, closing the under-training objection.",
     "Ch. 4, positive control; Ch. 3, decision-rule table."),
    ("C49", "The survey's strict and permissive audit counts rest on a per-paper coding table.",
     "No per-paper coding table existed on disk when the counts were first reported; it and the "
     "coding behind it were rebuilt from the written instrument.",
     "Ch. 4, Practice; coding table in the supplement."),
    ("C54", "A wider, roughly 22.4-million-parameter configuration tested whether capacity, not "
            "architecture, was the constraint.",
     "The Stage 0 script freezes the backbone unconditionally with no unfreeze path, and part of "
     "the parameter total is a pooler off the forward path that carries no gradient.",
     "Ch. 3: a configuration that was never trained."),
]


def main():
    corr = (ROOT / "accord" / "CORRECTIONS.md").read_text()
    entries = re.findall(r"^## (C\d+)\.\s*(.+)$", corr, re.M)
    entries.sort(key=lambda e: int(e[0][1:]))   # C13 before C14
    rev = (ROOT / "accord" / "REVISION_LOG.md").read_text()
    revs = re.findall(r"^### (R\d+)\s*\((\d{4}-\d{2}-\d{2})\)\s*[-—]*\s*(.+)$", rev, re.M)
    revs.sort(key=lambda e: int(e[0][1:]))

    selected_ids = {cid for cid, *_ in NUMBER_CHANGING}
    remaining = [cid for cid, _ in entries if cid not in selected_ids]

    a = [r"\chapter{The Correction Record}", r"\label{app:corrections}", "",
         "This appendix lists the ten corrections issued during the work that changed a number this",
         "thesis still prints, each traced from the claim as originally stated to the evidence that",
         "retired it and to where the corrected version now lives. The remaining corrections changed",
         "the record, not a number surviving into these six chapters, and are named below rather than",
         "repeated in full; their complete text, with artifact and reasoning, is retained in",
         "the project's correction record, released with the analysis code.", "",
         r"\section{Corrections That Changed a Printed Number}", "", r"\singlespacing",
         r"\renewcommand{\arraystretch}{0.9}",
         r"\footnotesize",
         r"\begin{longtable}{@{}p{1.9in}p{1.9in}p{1.9in}@{}}",
         r"\textbf{Claim as stated} & \textbf{Evidence that retired it} & "
         r"\textbf{Where it lives now}\\", r"\midrule", r"\endhead"]
    for cid, claim, evidence, location in NUMBER_CHANGING:
        a.append(f"\\textbf{{{cid}}}: {esc(claim)} & {esc(evidence)} & {esc(location)}\\\\")
        a.append(r"\addlinespace")
    a += [r"\end{longtable}", r"\normalsize", r"\doublespacing", ""]

    def as_ranges(ids):
        """Compress a sorted list of C-numbers into runs, e.g. C9 to C17, for a short sentence."""
        nums = [int(c[1:]) for c in ids]
        runs, start, prev = [], nums[0], nums[0]
        for n in nums[1:]:
            if n == prev + 1:
                prev = n
                continue
            runs.append((start, prev))
            start = prev = n
        runs.append((start, prev))
        return ", ".join(f"C{a}" if a == b else f"C{a} to C{b}" for a, b in runs)

    a += [r"\section{The Remaining Corrections}", "",
          f"{len(remaining)} further corrections were issued during the work: "
          + esc(as_ranges(remaining))
          + ". Each withdrew, narrowed or confirmed a claim against evidence. They are named "
          "rather than reproduced because the table above traces the ten whose corrected value "
          "a reader is most likely to want to follow, not because the rest left every printed "
          "number untouched: several of them produced numbers this thesis still prints. C16 "
          "identified the contaminated evaluation run and produced the recall of 0.7778 and "
          "precision of 0.0424 reported in Chapters 1, 5 and 6. C39 corrected the mortality "
          "control task's dataset to 34,018 examples, 2,975 positives and 570 test positives, "
          "the last of which is printed seven times across five chapters. C19 produced the "
          "defect-free control model's "
          "scores, C36 the four Matthews correlation intervals, C42 the public-text over-calling "
          "ratio, C55 the survey's strict and permissive counts, C57 the drift-slope flip count, "
          "C59 the survey reliability coefficients and C62 the convergence sweep's adjudication. "
          "No claim is made here that this list is exhaustive. Full text for each is in the "
          "project's correction record, released with the analysis code.", ""]

    if revs:
        a += [r"\section{Revisions After the Record Closed}", "",
              "The correction record was closed once the frame was frozen. Findings after that point",
              "change a number, a chapter, or a stated limitation, and by rule cannot change the",
              "thesis statement. They are listed separately for that reason.", "",
              r"\singlespacing", r"\renewcommand{\arraystretch}{0.9}", r"\footnotesize",
              r"\begin{longtable}{@{}p{0.35in}p{0.7in}p{4.5in}@{}}",
              r"\textbf{ID} & \textbf{Date} & \textbf{Revision}\\", r"\midrule", r"\endhead"]
        a += [f"{rid} & {d} & {esc(t.strip())}\\\\" for rid, d, t in revs]
        a += [r"\end{longtable}", r"\normalsize", r"\doublespacing", ""]

    # The full record goes to the supplement, which is where the other four
    # reference appendices already live, and the bound thesis carries a pointer
    # to it. The tables are 21 rows of paragraph-length prose and ran to twenty
    # pages of a document nobody reads sequentially; the supplement is the right
    # home for that, and the generator owns both files so they cannot drift.
    (THESIS / "msu_appA_full.tex").write_text("\n".join(a) + "\n")

    changed = [e for e in entries if e[0] in HEADLINE] if "HEADLINE" in dir() else []
    ptr = [r"\chapter{The Correction Record}", r"\label{app:corrections}", "",
           f"This work issued {len(entries)} numbered corrections and {len(revs)} "
           "post-freeze revisions. Each withdrew, narrowed or confirmed a claim "
           "against evidence. They are summarised rather than reprinted here "
           "because the complete record runs to twenty pages of table that no "
           "reader reads in sequence, and the full text of every one of them is "
           "released with the analysis code rather than bound, which is the "
           "same treatment given to the register, the claim trace and the "
           "survey coding table. The earlier wording promised the complete "
           "record in a supplementary file that accompanied this document. That "
           "file is no longer bound, so the promise described a deposit that "
           "did not exist, which is the defect class this record is full of.", "",
           "Ten of them changed a number this thesis prints, and those ten are "
           "the ones a reader is most likely to want to follow: they are named "
           "by identifier in the released record and cross-referenced from the "
           "chapters that carry their corrected values. The most consequential "
           "is the last, and it is the largest single withdrawal in the record. "
           "C95 withdrew every result this work obtained by sending sentence "
           "pairs to a hosted model service, on the authority of PhysioNet's "
           "guidance of 24 September 2025, which states the prohibition without "
           "a vendor qualifier and withdrew the list of suggested services that "
           "those routes had been selected against. That removed the only "
           "variation in labeller capability the design contained, so the "
           "question of whether over-calling magnitude belongs to the pool or "
           "to the labeller is open again rather than answered, and this thesis "
           "says so rather than returning to the attribution an earlier "
           "correction had already retracted. The withdrawal is enforced by the "
           "numeric provenance gate rather than by this sentence: every "
           "withdrawn value is held under a namespace the gate excludes from "
           "the set a reported figure may resolve against, because the "
           "preceding withdrawal in this record was enforced by prose alone and "
           "would have resolved a withdrawn digit anywhere in the thesis "
           "without complaint.", "",
           "The record is kept because a thesis about controls that reported no "
           "failures of its own would be evidence of a project that had stopped "
           "checking, not of one that had nothing to find.", ""]
    (THESIS / "msu_appA_corrections.tex").write_text("\n".join(ptr) + "\n")

    F = json.loads((ROOT / "results" / "accord" / "VERIFIED_FACTS.json").read_text())
    groups = {}
    for k, rec in sorted(F.items()):
        groups.setdefault(k.split(".")[0], []).append(rec)

    b = [r"\chapter{Provenance of Reported Quantities}", r"\label{app:provenance}", "",
         f"Every quantity reported in this thesis is drawn from a register of {len(F)} values, each",
         "extracted from a stored result file by script rather than transcribed. The register is",
         "grouped below by the experiment that produced it, with the file each group was read from.", "",
         r"\singlespacing", r"\renewcommand{\arraystretch}{0.9}", r"\footnotesize", r"\begin{longtable}{@{}p{1.6in}p{0.6in}p{3.6in}@{}}",
         r"\textbf{Group} & \textbf{Values} & \textbf{Source file}\\", r"\midrule", r"\endhead"]
    for g, items in sorted(groups.items()):
        arts = sorted({i["artifact"] for i in items})
        extra = f" and {len(arts) - 1} more" if len(arts) > 1 else ""
        b.append(f"{esc(g)} & {len(items)} & {esc_breakable(arts[0])}{extra}\\\\")
    b += [r"\end{longtable}", r"\doublespacing", ""]
    (THESIS / "msu_appB_provenance.tex").write_text("\n".join(b) + "\n")

    # ---- Appendix B pointer ----
    # This short section used to be typed by hand, and its numbers (how many
    # values the register holds, how many occurrences the trace resolves, the
    # unique/ambiguous split, the untraced count) moved during an audit without
    # any of the hand-typed figures moving with them: none of them appeared
    # anywhere in the generated output any more. Every number below is read
    # from the register (already loaded as F, above) or from
    # results/accord/CLAIM_TRACE.json, the artifact trace_claims.py writes; see
    # that script's own n_distinct / n_unique_rows / n_ambiguous_rows fields,
    # computed once over the same dedup its Appendix C table uses, rather than
    # recomputed a second way here.
    trace_path = ROOT / "results" / "accord" / "CLAIM_TRACE.json"
    if not trace_path.exists():
        raise SystemExit(
            "results/accord/CLAIM_TRACE.json is missing; run "
            "accord/stats/trace_claims.py before this script so the pointer "
            "numbers below have an artifact to be read from, rather than "
            "typing a placeholder here that would drift the same way the old "
            "hand-typed pointer did.")
    trace = json.loads(trace_path.read_text())
    n_scanned = trace["n_traced"] + trace["n_untraced"]
    survey_n = int(F["survey.n_included"]["value"])

    pointer = [
        r"% Flows directly after the correction record instead of opening its own page.",
        r"% The label binds to Appendix A's counter, so chapter references print the",
        r"% appendix this text actually lives in.",
        r"% Numbered rather than starred: four chapters reference this section by number,",
        r"% and a \label on a starred section captures the previous section counter, which",
        r"% made every one of those references print the number of A.3 instead.",
        r"\section{Reference Tables Released with the Code}",
        r"\label{app:provenance}",
        "",
        "Three reference tables belong to the record behind this thesis, machine-generated "
        "by the appendix builder and the claim tracer in the analysis code released with "
        "this thesis. The first two are released with that code rather than bound, "
        "because together they run to more than thirty pages of table, none of which is "
        "read sequentially. The first is the "
        "register of "
        f"reported quantities: {len(F)} values, each extracted from a stored result file "
        "by script rather than transcribed by hand, listed with the file it was read from. "
        f"The second is the claim and evidence trace, which scans the chapters for "
        f"{n_scanned} numeric literals and resolves {trace['n_traced']} of them against "
        f"the register, at the precision printed: of the {trace['n_distinct']} distinct "
        f"registered quantities the trace names, {trace['n_unique_rows']} are matched "
        f"uniquely and {trace['n_ambiguous_rows']} are matched among more than one "
        "candidate and marked as indicative rather than definitive. The remaining "
        f"{trace['n_untraced']} literals are not resolved to any registered value and are "
        "listed as untraced rather than omitted. The third, the survey coding table "
        "behind the prevalence measurement in Chapter~\\ref{ch:results}, is bound in "
        f"the Supplement, Section S3: all {survey_n} coded papers, reproduced in "
        "full so the counts can be recomputed rather than taken on trust.", "",
    ]
    (THESIS / "msu_appB_pointer.tex").write_text("\n".join(pointer) + "\n")

    print(f"Appendix A: {len(entries)} corrections, {len(revs)} revisions")
    print(f"Appendix B: {len(groups)} groups covering {len(F)} values")
    print(f"Appendix B pointer: register {len(F)}, trace {n_scanned} scanned / "
          f"{trace['n_traced']} resolved / {trace['n_untraced']} untraced, "
          f"survey {survey_n}")


if __name__ == "__main__":
    main()
