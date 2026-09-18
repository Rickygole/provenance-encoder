# Where a Cross-Note Medication Contradiction Detection Pipeline Fails: A Controlled Failure Analysis

Code, aggregate results and LaTeX sources for the MS thesis of the same title (Morgan State University). The
repository contains what is needed to rebuild the thesis PDF, regenerate its figures, and trace every number it
prints to the analysis that produced it. It contains no MIMIC-III data.

## What is here

```
accord/thesis/          LaTeX sources (main_msu.tex, frontmatter/, msu_ch1..5, appendices), refs.bib,
                        figures/ (figure PDFs, generators, figure gates), build.sh
accord/stats/           verification gates, the claim trace, appendix generator, statistics scripts
accord/<experiment>/    each pre-registered check: its pre-registration (C*_PREREGISTRATION.md, committed
                        before the run), run script and register script
accord/ (other)         cohort, extraction and mining code, detector and control model, labeller runners,
                        public-data controls
accord/CORRECTIONS.md   the correction record the thesis cites (released with the code)
results/accord/         aggregate result files; VERIFIED_FACTS.json is the register every printed number
                        resolves to; CLAIM_TRACE.json maps printed numbers to register keys
```

## Python requirements

`pip install -r requirements.txt` covers everything the verification gates, the figure generators and the
statistics scripts import. It also lists the packages individual experiment runners import (`torch`,
`transformers`, `spacy`, `statsmodels`, `boto3`, `google-genai`, `peft`, ...); most of those need MIMIC-III
credentialed access, a GPU or a locally served model and are not needed to rebuild the PDF or run the gates.
`requirements.txt` says which is which.

## LaTeX toolchain

Built with TeX Live (`pdflatex`) plus `biber` for the bibliography. Everything below except `biber` installs
into the user tree, no root needed:

```bash
tlmgr --usermode init-usertree   # once, if ~/Library/texmf does not exist
tlmgr --usermode install newtx tocloft hyphenat csquotes biblatex
brew install biber
```

`newtx` gives the Times body/math faces, `tocloft` the "Chapter N:" table-of-contents prefixes, `hyphenat`
lets long monospace register keys break across a line, and `csquotes`/`biblatex` (backend `biber`, style
`numeric-comp`) drive the references. Everything else `main_msu.tex` loads (`geometry`, `courier`, `fontenc`,
`amsmath`, `amssymb`, `graphicx`, `booktabs`, `longtable`, `caption`, `titlesec`, `float`, `setspace`,
`enumitem`, `indentfirst`, `microtype`, `ragged2e`, `hyperref`, ...) ships with a standard TeX Live install.

## Rebuild and verify

```bash
cd accord/thesis && bash build.sh                  # builds main_msu.pdf (pdflatex, biber, pdflatex x2)
python3 accord/stats/verify_thesis.py              # dashes, numeric provenance, citations, style, structure
python3 accord/stats/verify_thesis_deep.py         # ten passes, including template and figure reproducibility
python3 accord/stats/trace_claims.py               # every printed number -> register key -> artifact, Appendix C
python3 accord/stats/withdraw_hosted.py --check    # no hosted-channel value is live
cd accord/thesis/figures && python3 tools/check_figure_invariants.py && python3 tools/measure_ink.py
```

## Regenerate figures

```bash
cd accord/thesis/figures && python3 make_figures.py   # every figure sourced from results/accord/; writes manifest.json
```

`make_figures.py` reads only from `results/accord/` (or computes closed-form from numbers read from there), so it
runs from this repository alone. `make_figures_extra.py` builds two additional, non-thesis-bound figures
(`fig-asymmetry`, `fig-falsification`) from data held in this author's own local, credentialed storage outside
`results/accord/`; it is not runnable from the repository alone. The individual `fig_*.py` modules under
`accord/thesis/figures/` can also be run standalone (see each file's own docstring).

## Reproducing results

Experiments on public data (SNLI, AG News, synthetic fixtures, PUMS) run from this repository alone. Experiments on
the clinical pairs need your own PhysioNet credentialed access to MIMIC-III. The cohort/extraction script
(`accord/phase1_cohort.py`) reads `NOTEEVENTS.csv.gz`, `ADMISSIONS.csv`, `PATIENTS.csv` and `DIAGNOSES_ICD.csv.gz`
from PhysioNet's `raw_mimiciii` release layout, at paths you set with the `MIMIC_NOTEEVENTS_PATH`,
`MIMIC_ADMISSIONS_PATH`, `MIMIC_PATIENTS_PATH` and `MIMIC_DIAGNOSES_PATH` environment variables; it writes note
text and aggregates only under `accord/local_only/` (gitignored), which the medication-extraction and mining
scripts (`phase2_medication.py`, `phase3_mining.py`) then read in turn. No note text, HADM_ID or SUBJECT_ID
leaves that directory.

Every other clinical script reads and writes row-level files under one credentialed directory, named by the
`MIMIC_LOCAL_ONLY_DIR` environment variable (default `~/mimic_local_only`). The four per-file variables above
default to `$MIMIC_LOCAL_ONLY_DIR/raw_mimiciii/`. Nothing under that directory is tracked, and only aggregates
are written into `results/`.

Some results cannot be regenerated from this repository because the code that produced them was never committed or
no longer exists: the deployed weak-labelling run and its prompt, the original Rules 1 and 2 mining notebook, the
training run behind the rule-type head result, and several exploratory diagnostic notebooks whose aggregate outputs
are committed under `results/accord/`. The thesis discloses these.

## What you can and cannot reproduce

Public-data experiments (public controls, prompt-sensitivity, TF-IDF and other checks that read only committed
files under `results/accord/`) run end to end from this repository. Anything downstream of the clinical cohort
needs your own PhysioNet-credentialed access to MIMIC-III and the local directory layout described above; this
repository ships no MIMIC-III data or MIMIC-derived note text.

The register (`results/accord/VERIFIED_FACTS.json`) also keeps entries from withdrawn or superseded work whose
artifacts are not shipped in this tree, under `withdrawn.*` (including `withdrawn.hosted.*`, the hosted-API runs
disclosed above, and `withdrawn.artifact_not_in_repository.*`, keys retired because the file they cited is not
in this repository). Nothing under `withdrawn.*` is citable, and the numeric-provenance gate excludes it from
anything the thesis may report. Every number the thesis prints traces to a register key whose cited artifact
exists in this tree, a claim `accord/stats/verify_thesis.py` checks mechanically, not just asserts.

## Data handling

MIMIC-III is PhysioNet credentialed data. Everything touching it runs locally and emits aggregates only; no
note text or admission or subject identifier is printed, logged or committed. Credentialed and note-derived
files live outside this tree (`local_only/` is a symlink to credentialed local storage and is gitignored, as
are the MIMIC table names), and `/Users/rickygole` is itself a git repository, which is why those rules are
aggressive.

Earlier in the project, sentence pairs were sent to hosted model APIs in several runs. Every result obtained
that way is withdrawn: the values are held under `withdrawn.hosted.*` in the register, which the numeric
provenance gate excludes from anything the thesis may report, and the per-pair outputs are held in
credentialed local storage, not in this tree. The thesis methodology chapter's data-use section
(`accord/thesis/msu_ch3_methodology.tex`) states this. Every model comparison the thesis reports was run on
models served locally, the route PhysioNet's guidance of 24 September 2025 recommends; the hosted runners now
refuse to start without an explicit opt-in.

