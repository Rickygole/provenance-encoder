# Where a Clinical NLP Pipeline Fails: A Stage-by-Stage Audit of Cross-Note Medication Contradiction Detection

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
results/accord/         result files: aggregates, or per-pair outputs keyed only to internal sequential
                        indices (PAIR_0001, ...), never note text or identifiers; VERIFIED_FACTS.json is the
                        register every printed number resolves to; CLAIM_TRACE.json maps printed numbers to
                        register keys
environment.yml         the exact environment the committed figure PDFs were rendered in
requirements.txt        the same versions pinned for pip, plus what the experiment runners import
```

## Python requirements

Python 3.12 (3.12.7 was used). `conda env create -f environment.yml` recreates the environment the thesis was
built in: numpy 1.26.4, pandas 2.2.2, scipy 1.13.1, scikit-learn 1.5.1, matplotlib 3.9.2 linked against FreeType
2.12.1, Pillow 10.4.0, pyarrow 16.1.0, PyMuPDF 1.28.2. `requirements.txt` pins the same versions for pip. The
figure gate also needs poppler (`pdftoppm`, `pdftotext`; `brew install poppler` or `apt install poppler-utils`).

The figure comparison is exact only in the recorded environment. The matplotlib wheel on PyPI bundles FreeType
2.6.1, not the 2.12.1 the conda build links, and text metrics and canvas sizes shift between the two. In any
environment whose matplotlib or FreeType differs from `environment.yml`, pass 10 of `verify_thesis_deep.py` says so,
still fails if a generator crashes, and reports the figures as not comparable in this environment; it does not
fail on the raster difference. It renders into a temporary directory and never rewrites the committed PDFs.

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

Every command runs from the repository root.

```bash
bash accord/thesis/build.sh                                      # builds main_msu.pdf (pdflatex, biber, pdflatex x2)
python3 accord/stats/verify_thesis.py                            # dashes, numeric provenance, citations, style, structure
python3 accord/stats/verify_thesis_deep.py                       # eleven passes, including template and figure reproducibility
python3 accord/stats/trace_claims.py                             # every printed number -> register key -> artifact
python3 accord/stats/withdraw_hosted.py --check                  # no hosted-channel value is live
python3 accord/stats/sync_appendix_counts.py --check             # the appendix's printed provenance counts match the tools
python3 accord/thesis/figures/tools/check_figure_invariants.py
python3 accord/thesis/figures/tools/measure_ink.py
```

`trace_claims.py` exits 1 by design whenever it lists any untraced quantity, and it currently lists some; that is
not a failure. The closing section of Appendix A ("Reference Tables Released with the Code") discloses how many
it lists and what they are, and `sync_appendix_counts.py --check` holds that section's counts to the tool's
output. The trace table itself is released with the code (`results/accord/CLAIM_TRACE.json`), not bound into the PDF.

## Regenerate figures

```bash
python3 accord/thesis/figures/make_figures.py   # sourced from results/accord/; writes the PDFs and manifest.json
python3 accord/thesis/figures/fig_batch2.py
python3 accord/thesis/figures/fig_ceiling.py
```

The generators read only from `results/accord/` (or compute closed-form from numbers read from there), so they
run from this repository alone. Between them these three build the seven figures the thesis binds. They write
into `accord/thesis/figures/`, over the committed PDFs; set `ACCORD_FIG_OUT=/some/dir` to write elsewhere, which
is what the figure gate does. Outside the recorded environment (see Python requirements) the PDFs they write
will not match the committed ones pixel for pixel.

## Reproducing results

Experiments on public data (SNLI, AG News, synthetic fixtures, PUMS) run from this repository alone. The scripts
that score with the public RoBERTa-large-MNLI weights (`accord/differential/nli_baseline.py`,
`accord/public_controls/sizectl/score_roberta.py`, `accord/public_controls/sizectl_c108/score_roberta.py`) load
them offline from `~/models_public/roberta-large-mnli/`; `python3 accord/differential/fetch_nli_model.py` downloads
them there at the pinned commit and records per-file hashes in `results/accord/differential/nli_model_manifest.json`.
Experiments on the clinical pairs need your own PhysioNet credentialed access to MIMIC-III. The cohort/extraction script
(`accord/phase1_cohort.py`) reads `NOTEEVENTS.csv.gz`, `ADMISSIONS.csv`, `PATIENTS.csv` and `DIAGNOSES_ICD.csv.gz`
from PhysioNet's `raw_mimiciii` release layout, at paths you set with the `MIMIC_NOTEEVENTS_PATH`,
`MIMIC_ADMISSIONS_PATH`, `MIMIC_PATIENTS_PATH` and `MIMIC_DIAGNOSES_PATH` environment variables; it writes note
text and aggregates only under `$MIMIC_LOCAL_ONLY_DIR/accord/`, which the medication-extraction and mining
scripts (`phase2_medication.py`, `phase3_mining.py`) then read in turn. No note text, HADM_ID or SUBJECT_ID
leaves that directory.

Every clinical script reads and writes row-level files under one credentialed directory, named by the
`MIMIC_LOCAL_ONLY_DIR` environment variable (default `~/mimic_local_only`). The four per-file variables above
default to `$MIMIC_LOCAL_ONLY_DIR/raw_mimiciii/`. Nothing under that directory is tracked, and what is written
into `results/` is aggregates, or per-pair outputs keyed only to internal sequential indices, never note text or
identifiers. The scripts expect these subdirectories under it (names as they appear in the code):

```
raw_mimiciii/                       the PhysioNet MIMIC-III tables
accord/                             cohort, medication-mention and mining outputs (phase1 to phase3)
accord_supp/  accord_stage0/  derived/
accord_control/  accord_detector/  accord_differential/  accord_multillm/
accord_prompt_sensitivity_cache/  accord_c130_reproducible_student/
from_drive_2026-08-08/  from_drive_2026-09-13/    the annotated pairs and earlier outputs, moved off cloud storage
logs/                               run logs
```

The pharmacist's gold labels are derived from credentialed data. They are not distributed with this repository,
and they are not distributed by PhysioNet. A third party with MIMIC-III access can therefore rerun the cohort,
extraction and mining code, but cannot regenerate the clinical aggregates that depend on the gold labels; those
can be checked against the committed aggregate files under `results/accord/`, not reproduced.

Some results cannot be regenerated from this repository because the code that produced them was never committed or
no longer exists: the deployed weak-labelling run and its prompt, the original Rules 1 and 2 mining notebook, the
training run behind the rule-type head result, and several exploratory diagnostic notebooks whose aggregate outputs
are committed under `results/accord/`. The thesis discloses these.

## What you can and cannot reproduce

From this repository alone: the thesis PDF, the gates, the figures, the public-data experiments (the public
controls and the SNLI prevalence sweep, given a locally served model where a script calls one), and the statistics
scripts that read only committed files under `results/accord/`. The prompt-sensitivity (`accord/prompt_sensitivity/`)
and TF-IDF permutation (`accord/tfidf_permutation/`) analyses are not among them: their run and analysis scripts
read credentialed files under `MIMIC_LOCAL_ONLY_DIR`; their register scripts and dry-run fixtures work from committed files.
Anything downstream of the clinical cohort needs your own PhysioNet-credentialed access to MIMIC-III and the
directory layout described above, and anything that scores against the gold labels cannot be regenerated by a
third party at all (see above); this repository ships no MIMIC-III data or MIMIC-derived note text.

The register (`results/accord/VERIFIED_FACTS.json`) also keeps entries from withdrawn or superseded work whose
artifacts are not shipped in this tree, under `withdrawn.*` (including `withdrawn.hosted.*`, the hosted-API runs
disclosed above, and `withdrawn.artifact_not_in_repository.*`, keys retired because the file they cited is not
in this repository). Nothing under `withdrawn.*` is citable, and the numeric-provenance gate excludes it from
anything the thesis may report. Every number the thesis prints traces to a register key whose cited artifact
exists in this tree, a claim `accord/stats/verify_thesis.py` checks mechanically, not just asserts.

## Data handling

MIMIC-III is PhysioNet credentialed data. Everything touching it runs locally and emits aggregates only; no
note text or admission or subject identifier is printed, logged or committed. Credentialed and note-derived
files live outside this tree (any `local_only` path is gitignored, as are the MIMIC table names), and the
author's home directory (`~`) is itself a git repository, which is why those rules are aggressive.

Earlier in the project, sentence pairs were sent to hosted model APIs in several runs. Every result obtained
that way is withdrawn: the values are held under `withdrawn.hosted.*` in the register, which the numeric
provenance gate excludes from anything the thesis may report, and the per-pair outputs are held in
credentialed local storage, not in this tree. The thesis methodology chapter's data-use section
(`accord/thesis/msu_ch3_methodology.tex`) states this. Every model comparison the thesis reports was run on
models served locally, the route PhysioNet's guidance of 24 September 2025 recommends; the hosted runners now
refuse to start without an explicit opt-in.

