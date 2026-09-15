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

## Rebuild and verify

```bash
cd accord/thesis && bash build.sh                  # builds main_msu.pdf
python3 accord/stats/verify_thesis.py              # dashes, numeric provenance, citations, style, structure
python3 accord/stats/verify_thesis_deep.py         # ten passes, including template and figure reproducibility
python3 accord/stats/withdraw_hosted.py --check    # no hosted-channel value is live
cd accord/thesis/figures && python3 tools/check_figure_invariants.py && python3 tools/measure_ink.py
```

TeX packages install into the user tree: `tlmgr --usermode install apacite tocloft newtx chngcntr needspace placeins xurl`.

## Reproducing results

Experiments on public data (SNLI, AG News, synthetic fixtures) run from this repository alone. Experiments on the
clinical pairs need your own PhysioNet credentialed access to MIMIC-III; the scripts read credentialed files from a
local directory outside the repository and write aggregates only.

Some results cannot be regenerated from this repository because the code that produced them was never committed or
no longer exists: the deployed weak-labelling run and its prompt, the original Rules 1 and 2 mining notebook, the
training run behind the rule-type head result, and several exploratory diagnostic notebooks whose aggregate outputs
are committed under `results/accord/`. The thesis discloses these.

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

