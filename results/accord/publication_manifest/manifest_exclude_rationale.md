# What is excluded from the publication manifest, and why

Computed, not estimated. The include set is the fixpoint of the six reproduction
commands (see `closure_evidence.json` for the method and for the per-path
evidence). Everything below is what is left over after that fixpoint closed.

Baseline: 678 files / 222.0 MB in the working tree (403 files tracked by git, 38.1 MB).
Include set: 168 files / 8.45 MB. So 510 working-tree files and 213.6 MB drop out.

NOTE: this repository was being edited by another process while the closure was
computed (`fig_batch3.py`, `make_figures.py`, `fig_batch2.py` and
`figures/manifest.json` all changed mtime mid-run, and two figure PDFs appeared).
The manifest is a snapshot. Re-run `accord/publication_manifest/build_manifest.py`
before publishing.

Nothing was deleted, moved or modified to produce this list.

---

## 1. Directories that drop out ENTIRELY (nothing in the thesis or the gates reads them)

36 of the 40 directories under `accord/` contribute zero files:

    accord/audit_c137/            accord/c111_robustness/       accord/c112_decontam_defectfree/
    accord/c114_committee_checks/ accord/c116_rank_paired/      accord/c120_exact_join/
    accord/c126_teacher_student/  accord/c127_scale_ladder/     accord/c128_prospective_feasibility/
    accord/c129_exemplar_prevalence/ accord/c130_reproducible_student/ accord/c131_scale_ladder_v2/
    accord/c132_cluster_bootstrap/ accord/c133_interval_tipping/ accord/c134_reanalysis/
    accord/c135_dissociation_cost/ accord/c136_gate_validation/
    accord/clinical_core/         accord/dag/                   accord/defectfree_rescore/
    accord/denominator/           accord/detector/              accord/differential/
    accord/exemplar_direct/       accord/framing_mechanism/     accord/generality/
    accord/mining/                accord/opsweep/               accord/prompt_sensitivity/
    accord/public_controls/       accord/pums_replication/      accord/stage0/
    accord/tfidf_permutation/     accord/theory/                accord/__pycache__/

Four survive, and only partly:

| directory | in / total | what survives |
|---|---|---|
| `accord/thesis/` | 57 / 87 | the bound sources, the frontmatter, the 25 figure PDFs, the 7 figure generators, the 2 figure tools, `refs.bib`, `build.sh` |
| `accord/stats/` | 5 / 31 | `verify_thesis.py`, `verify_thesis_deep.py`, `make_appendices.py`, `trace_claims.py`, `sync_appendix_counts.py` |
| `accord/control/` | 2 / many | `report.py`, `train.py` (both CITED BY THE REGISTER as the generators behind control-arm values) |
| `accord/scaling/` | 2 / many | `C44_PREREGISTRATION.md`, `C44_convergence_check.ipynb` (cited by the register) |
| `accord/prevalence_replication/` | 1 | `results/sweep_results.json` (cited by the register) |

Plus five loose files at the `accord/` root: `hosted_withdrawal.py`, `dictionaries.py`,
`CORRECTIONS.md`, `REVISION_LOG.md` (all four load-bearing, see below).

Outside `accord/`: `src/` (10 files), `tests/` (4), `scripts/` (7) and `configs/` (0)
contribute nothing. `src/` is the pretraining package behind `results/pretraining_runs.json`;
the register cites the JSON, not the package, so no gate reads it.

Under `results/accord/`, 93 of 184 files survive (plus 2 files under `results/` itself, `pretraining_runs.json` and `task_runs.json`). Whole subdirectories that drop out:
`audit_c137/` (11), `c136_gate_validation/` (4), `pums/`, `withdrawn_hosted/`.
`framing_mechanism/answer_polarity.json` IS required, but only by `fig_batch3.py`,
which no gate runs: see finding M5 in `audit_findings.md`. The heavily thinned ones are `public_controls/` (5 of 41),
`prompt_sensitivity/` (3 of 24), `clinical_core/` (1 of 5), `c134_reanalysis/` (1 of 5).

## 2. The explicit exclusions, each verified against the closure

| exclusion | verified? | evidence |
|---|---|---|
| `results/accord/withdrawn_hosted/**` | **NOT REQUIRED.** Safe to exclude. | The directory holds exactly one file, `README.md`. No gate opens it; no register key names it. The six hosted artifacts the register still cites by path (`frontier_anthropic_metrics.json`, `frontier_anthropic_metrics_claudehaiku4520251001.json`, two `frontier_bedrock_metrics_metallama3*`, `frontier_label_noise_claudehaiku4520251001.json`, `transmission_ledger.json`) are **already absent from disk** and the gates still pass, because `verify_thesis.py` excludes the `withdrawn.*` namespace from the citable set before it checks existence. |
| `results/accord/multillm/**` | **REQUIRED, and it is clean.** | `multillm/per_model_results.json` is the directory's only file and `verify_thesis.py` reads it. It contains **no hosted output**: the per-model block is `llama3:8b`, `granite3.3:8b`, `qwen3:8b`, `qwen3:14b` served locally through Ollama, plus pool/gold confusion counts and a language regression. The prompt template it records uses `{nursing_note}` / `{discharge_summary}` placeholders, not note text. Publish as is. |
| `accord/CORRECTIONS.md` | **REQUIRED. See audit_findings.md, finding 1.** | `accord/stats/make_appendices.py` reads it to generate Appendix A, and `msu_appA_corrections.tex` is bound by `main_msu.tex`. Removing it makes `make_appendices.py` raise `FileNotFoundError`, which is reproduction command 3. `accord/REVISION_LOG.md` is required by the same module for the same reason. |
| `CLAUDE.md` and planning / framing / compliance-draft / personal files | **NOT REQUIRED.** Safe to exclude. | `CLAUDE.md`, `DESIGN.md`, `FRAMING.md`, `FORMAT.md`, `PHD_PLAN.md`, `WRITING_PLAN.md`, `THESIS_STATUS.md`, `CONTRIBUTION.md`: no gate opens any of them. `verify_thesis_deep.py` pass 9 is named "template compliance (FORMAT.md)" but the invariants are hard-coded regexes in the gate; it never reads `FORMAT.md`. `make_appendices.py` knows these filenames only in order to redact them out of the appendix (`INTERNAL_FILES`). |
| `local_only/**`, `accord/local_only/**` | **NOT REQUIRED.** Safe to exclude, and already unreachable. | Both are symlinks out of the tree (`~/mimic_local_only` and `.../accord`), both are matched by four `.gitignore` rules, and no file in the closure names either path. |
| `__pycache__`, `.DS_Store`, checkpoints, `*.pt` | **NOT REQUIRED.** Safe to exclude. | Only `.pyc` byte-compiled copies of files that are themselves in the include set. No `*.pt` is read by any gate. |

## 3. Large things that drop out (why the byte count falls 96 percent)

    188.1 MB  accord/pums_replication/data/2018/1-Year/psam_p48.csv   (PUMS replication input; nothing in the thesis closure reads it; already untracked by *.csv in .gitignore)
     15.5 MB  accord/public_controls/nlifactorial/data/heuristics_evaluation_set.txt  (HANS evaluation set, redistributable but unused by the closure)
      4.6 MB  results/accord/public_controls/sizectl*/roberta_scores.json x2  (only sizectl_c108's copy is cited, and only its *_meta and power_curves are read; the 2.3 MB scores file IS in the include set, its sizectl/ twin is not)
      0.6 MB  accord/public_controls/nlifactorial/manifest.json
      2.2 MB  results/accord/public_controls/nlifactorial/raw/*.jsonl  (per-item raw model outputs; the closure reads only the aggregated analysis.json)

## 4. Things NOT in the computed closure that a published repository still needs

These are publication hygiene, not reproduction closure, so they are listed here
rather than in `manifest_include.txt`:

    LICENSE            NOTICE            README.md            requirements.txt
    .gitignore         (the restricted-data rules are themselves a disclosure control)

`environment.yml` IS in the include set: `verify_thesis_deep.py` pass 10 reads the
recorded matplotlib and FreeType versions out of it.

## 5. Two honest limits on what "reproduces the thesis" means here

1. **The closure does not contain the generators of the 93 cited result artifacts.**
   The register stores only `{artifact, value}`; it has no generator field, and the
   gates check that the artifact file exists, never that a script that could rebuild
   it exists. A clone of the include set can rebuild the PDF, re-derive every printed
   digit from the register, and rebuild all 7 bound figures from the artifacts, but it
   cannot recompute the artifacts themselves. If the goal is generator-level
   reproducibility, a second tier has to be added: 112 distinct scripts across the
   experiment directories. The mapping from each of the 93 cited artifacts to the
   scripts in this repository that name it is computed and written to
   `tier2_cited_artifact_generators.json` next to this file. 15 of the 93 cited artifacts have **no
   script anywhere in the repository that names them**, so for those the generator
   cannot be added because it does not exist in the tree:
   `annotation_sheet_span_profile.json`, `bonus_bootstrap_ci.json`,
   `control/metrics_seed42.json`, four `differential/frontier_ollama_*_few_shot.json` plus `differential/frontier_ollama_granite33_zero_shot.json`,
   `prompt_sensitivity/cells/_qwen3_8b_nothink__P_neutral.json`,
   `signed_gap_analysis.json`, three `source_summaries/*.json`,
   `step4_threshold_fix_test.json`, `results/pretraining_runs.json`.

2. **The raw data can never be published.** Every path back past those artifacts ends
   in MIMIC-III under PhysioNet credentialing. That is a property of the project, not
   of this manifest.
