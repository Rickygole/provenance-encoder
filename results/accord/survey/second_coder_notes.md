# Second-coder notes (blind re-coding for inter-rater reliability)

Coded independently, without opening `coding_table.csv`, `quotes.json`, `coding_summary.json`,
`second_coder_reliability.json`, `accord/CORRECTIONS.md`, `accord/WRITING_PLAN.md`, or
`accord/THESIS_STATUS.md`. Only sections 4, 6.1 and 6.2 of `protocol.md` were read (screening
criteria and the full field-definition / borderline-rule text). Reading stopped at the first
sentence of section 7; no outcome counts were seen (section 4.1's retrieval totals were visible
in passing but concern retrieval, not coding outcomes).

## 1. Subsample rule

```
rows        = every row of results/accord/survey/screening_log.csv
included    = rows where the `included` column is true          -> 75 paper_ids
ids         = sorted(included paper_ids)                        -> lexicographic sort
subsample   = ids[0::4][:20]                                    -> every 4th, starting at index 0
```

75 included papers yields 19 subsampled papers (indices 0, 4, 8, ... 72). Fewer than the maximum
of 20, so all 19 were coded.

Reproduce with:

```python
import csv
rows = list(csv.DictReader(open('results/accord/survey/screening_log.csv')))
ids  = sorted(r['id'] for r in rows if r['included'].strip().lower() in ('true','1','yes'))
sub  = ids[0::4][:20]
```

## 2. The exact paper list (19 papers, all retrieved, all coded)

| # | paper_id | source | year | short title |
|---|---|---|---|---|
| 1 | AC2020_clinicalnlp-1_33 | ACL | 2020 | Exploring Text Specific and Blackbox Fairness Algorithms in Multimodal Clinical NLP |
| 2 | AX2003.00827 | arXiv | 2020 | CheXclusion: Fairness gaps in deep chest X-ray classifiers |
| 3 | AX2304.05986 | arXiv | 2023 | Auditing ICU Readmission Rates in an Clinical Database |
| 4 | AX2410.01089 | arXiv | 2024 | FMBench: Benchmarking Fairness in Multimodal LLMs on Medical Tasks |
| 5 | AX2502.16051 | arXiv | 2025 | MENTAT: Clinician-Annotated Fairness Dataset in Mental Healthcare |
| 6 | AX2510.00055 | arXiv | 2025 | Adapting LLMs to Mitigate Skin Tone Biases in Clinical Dermatology |
| 7 | AX2603.14631 | arXiv | 2026 | Anterior's Approach to Fairness Evaluation of Automated Prior Authorization |
| 8 | AX2607.16253 | arXiv | 2026 | Comprehensive Evaluation of ML for Type 2 Diabetes Risk Prediction |
| 9 | PM34798329 | PubMed | 2022 | Obstetric comorbidity scores and disparities in severe maternal morbidity |
| 10 | PM36714611 | PubMed | 2022 | Fairness in the prediction of acute postoperative pain |
| 11 | PM38422347 | PubMed | 2024 | Evaluating Algorithmic Bias in 30-Day Hospital Readmission Models |
| 12 | PM39399154 | PubMed | 2024 | Predicting Prenatal Depression and Assessing Model Bias |
| 13 | PM39831110 | PubMed | 2024 | Bias in Prediction Models for Colorectal Cancer Readmission |
| 14 | PM40298901 | PubMed | 2025 | External validation of a proprietary 1-year mortality risk model |
| 15 | PM40824638 | PubMed | 2025 | Performance of 4 Methods to Assess Health-Related Social Needs |
| 16 | PM40905712 | PubMed | 2025 | Racial Biases in Health Care Datasets With Subgroup Learnability (AEquity) |
| 17 | PM41639760 | PubMed | 2026 | ML performance for a small dataset: random oversampling and fairness |
| 18 | PM41840801 | PubMed | 2026 | Explainable AI for Determinants of Lifetime HIV Testing (BRFSS 2023) |
| 19 | PM42243352 | PubMed | 2026 | Demographic-aware temporal graph attention for fair 12-lead ECG detection |

## 3. Retrieval

No local full-text corpus exists in the repository, so all 19 full texts were retrieved
independently for this coding pass:

* PubMed (11): PMC IDs resolved via NCBI ELink, full text fetched via EFetch `db=pmc&retmode=xml`.
* arXiv (7): HTML5 (`arxiv.org/html/<id>`) where available, otherwise ar5iv, otherwise PDF via
  `pdftotext -layout` (AX2304.05986 and AX2607.16253 came from PDF).
* ACL (1): `aclanthology.org/2020.clinicalnlp-1.33.pdf` via `pdftotext -layout`.

Every retrieved file was title-checked against the screening-log title before coding; all 19
matched. **Unretrievable papers: 0.**

Two texts (AX2304.05986, AX2607.16253, AC2020_clinicalnlp-1_33) are two-column PDF conversions in
which adjacent source lines interleave with the other column. Quotes for those papers are
therefore deliberately short, single-line fragments so that they verify by exact string match.

Every quote in `second_coder_quotes.json` was programmatically re-checked as an exact substring of
the retrieved full text after normalising whitespace, unicode quotes/dashes and ligatures.
**0 of 90 quote checks failed.** No `not_mentioned` value was assigned without first running the
protocol's absence search (`missing, unknown, not reported, declined, imputation, complete case`)
over the full text; where that search returned an irrelevant hit, the hit is described in the
`location` field.

## 4. Result summary (this coder only)

| field | distribution |
|---|---|
| `audits_model_on_pool` | yes 19 |
| `cohort_is_filtered` | yes 19 |
| `audits_selection` (strict) | **no 19, yes 0** |
| `audits_selection_literal` (permissive) | **yes 4, no 15** |
| `null_attribute_handling` | not_mentioned 6, imputes 5, separate_category 4, drops 3, unclear 1 |

Permissive-yes papers: `AX2510.00055`, `AX2603.14631`, `AX2607.16253`, `PM39399154`.

The superset invariant (`audits_selection == yes` implies `audits_selection_literal == yes`) holds
trivially, since no paper was coded strict-yes.

No Cohen's kappa was computed and no comparison to the first coder was made.

## 5. Hard cases

Nine papers were flagged. In each, the value I chose is stated first, then what made it
contestable.

### AX2510.00055 - skin tone / SkinGPT-4 (strict **no**, permissive **yes**)
The single most contestable paper in the subsample. The authors drew a *deliberately skin-tone
balanced* sample of 600 SCIN cases, and Appendix 1 gives the SCIN parent distribution numerically
(Type I 7.5%, II 21.7%, III 26.4%, IV 17.1%, V 8.5%, VI 5.7%) together with the statement that
darker tones are underrepresented in SCIN. Strict = no because the retained sample's composition is
only ever described qualitatively ("balanced"), no retention rate by group is given, and the two
sets of figures sit in different sections and are never juxtaposed or tested. Permissive = yes
because specific parent-database group figures are reported and are meaningfully set against the
authors' own balanced selection - the textbook case of "comparing cohort demographics to the full
parent database before the authors' filtering". A coder who treats "balanced across Fitzpatrick
1-6" as an implicit retained-composition statement could defensibly code strict = yes here.

### AX2603.14631 - Anterior prior authorization (strict **no**, permissive **yes**)
The paper reports, per protected attribute, the analysis cohort size against the pre-filter
reviewed-case denominator N = 7,166 (Sex 6,505; Age 6,786; Race/Ethnicity 920; SES 1,929), and
Table 3 gives each group's frequency as a percentage of that same N = 7,166 with an explicit
`Missing` row (Race Missing 6,226 = 86.9%). Formally that is an inclusion rate reported against a
pre-selection denominator, which is why I coded the permissive column yes. Strict = no because
missingness is never disaggregated *by group within an attribute*: nothing reported lets you ask
whether non-White patients were more likely to have race undocumented, so differential retention
across groups is neither tested nor testable here. A first coder reading Table 3 as "percent
retained by group relative to the pre-filter cohort" would code strict = yes. This is the most
likely single source of strict-column disagreement in the subsample.

### AX2607.16253 - Type 2 diabetes external validation (strict **no**, permissive **yes**)
Table I compares the NHANES development cohort's demographic composition against the BRFSS
national sample with p-values (Male 48.5% vs 46.4%, p < 0.001; Ever Smoker 40.2% vs 37.9%, etc.).
That is a comparison of cohort group composition to an external, nationally representative figure,
so the permissive reading is met on the letter of the rule - even though the authors' purpose is
distribution-shift assessment for external validation, not a representativeness check. Strict = no:
neither cohort is compared to its own pre-filter denominator, and no retention rate by group is
reported. Secondary difficulty on `null_attribute_handling`: the paper is internally contradictory,
restricting NHANES to participants with "complete information on predictors" (complete case) while
also stating that missing categorical values were mode-imputed. I coded `imputes` on the explicit
imputation sentence; `drops` is defensible.

### AX2304.05986 - ICU readmission audit (strict **no**, permissive **no**)
The word "selection" is a trap here. The paper says repeatedly that groups are "underrepresented in
the selected set", but "selected set" denotes the model's predicted-positive set (Equal Parity /
Proportional Parity from the Aequitas toolkit), not the analysis cohort - a model audit, not a
selection audit. For the permissive column, the Data section gives MIMIC-III parent demographics
(~69% White) and Table III gives final-cohort group size ratios (White 0.69), but the paper never
juxtaposes them, and its only representativeness statement ("a subset of data, which may lead to
results that are not entirely representative of the overall population") cites no figure, which the
rule explicitly excludes. Also coded `unclear` for `null_attribute_handling`: "other/unknown"
appears only as a listed MIMIC insurance value, the analysis insurance groups contain no unknown
level, and no handling is stated.

### PM41639760 - PCI random oversampling (strict **no**, permissive **no**)
Table 2 lists the 280 Medicare/VA patients excluded from the socioeconomic analysis (58.2%) beside
the retained Advantaged (21.8%) and Disadvantaged (20.0%) groups, all as percentages of the same
481-patient cohort - formally a retained-versus-excluded breakdown against a pre-filter denominator.
I coded both readings no because the excluded stratum is defined *by the socioeconomic attribute
itself*, so no differential-retention question across groups is posed; nothing analogous is reported
for race or marital status; and there is no external or parent-database referent anywhere. A coder
applying the permissive rule mechanically to Table 2 could reach yes. Note that I reached the
opposite permissive verdict here from AX2603.14631, on the grounds that Anterior explicitly frames
N = 7,166 as the pre-filter population and reports "final cohort sizes" against it, whereas this
paper frames Table 2 purely as a Table-1-style description of its own cohort.

### PM40905712 - AEquity subgroup learnability (strict **no**, permissive **no**)
Thematic near-miss: the entire paper is about detecting dataset-level "sampling bias", which sounds
like a selection audit. But the operationalisation (AEq learning-curve values compared between
groups for the same label) never invokes a pre-selection population denominator or a retention rate;
it compares subgroup learnability *inside* an already-assembled dataset. The one sentence that
gestures at a source comparison - curation "reflecting the underlying empirical group frequencies" -
supplies no figure, which the permissive rule excludes.

### AX2502.16051 - MENTAT (strict **no**, permissive **no**)
Vignette benchmark in which patient demographics are assigned at random, so there is no patient
selection step to audit. The nearest thing is an annotator-bias check (whether annotation behaviour
differs by the gender coding of the question), which tests the labellers rather than retention of
patients. `cohort_is_filtered` = yes rests on the analysis-set narrowing to three task categories
for the free-form evaluation plus the authors' own curation of the item pool; a coder could read a
purpose-built vignette set as unfiltered.

### PM40824638 - Health-Related Social Needs (permissive column only)
The Results state "The sample was reflective of an urban, adult primary care population", but the
claim attaches a citation rather than a numeric external figure, and the protocol explicitly
excludes representativeness statements with no figure. Coded literal = no on that basis. Note that
no enrolment, refusal or approach counts are reported at all, so nothing about differential
recruitment into this convenience sample can be assessed.

### AX2003.00827 - CheXclusion (`null_attribute_handling` only)
The protocol's absence search returns zero hits, which would point to `not_mentioned`, but the Data
section states that CXR "has race and insurance type data (excluding around 100,000 images)". I read
that as those images being dropped from the race/insurance analyses and coded `drops`;
`reports_only` is the plausible alternative, since the sentence describes availability rather than
an analysis decision.

### PM36714611 - postoperative pain (`null_attribute_handling` only, weakest null code)
Coded `imputes` on a limitations sentence that covers missing variables generically ("we used the
information extracted from training data when imputing missing variables in the test data"). The
protected attributes (race, language, health literacy, insurance, ADI) are among the model
variables, but the paper never states specifically what happens to a patient with missing race. The
detail is in a supplement I did not retrieve. `unclear` would also be defensible.

## 6. Contamination disclosures

* **No first-coder field value was seen.** `coding_table.csv`, `quotes.json`,
  `coding_summary.json` and `second_coder_reliability.json` were never opened; the last of these
  was noticed in a directory listing (name and file size only) and deliberately left unread.
* `screening_log.csv` was opened, as permitted, for identifiers, titles and URLs. Defining "the
  INCLUDED papers" necessarily required reading its `included` column. That column records the
  stage-1/stage-3 screening disposition, not any of the five fields coded here, so it does not
  contaminate the five reliability fields - but it is disclosed for completeness, since it is
  technically a first-pass decision column.
* While reading `protocol.md` for the field definitions, the first paragraph of section 7
  ("Known limitations", item 0) was read. It states that `audits_model_on_pool` is yes for every
  included paper *by construction*, because inclusion criterion (ii) requires a disaggregated
  metric. That is a statement about the sampling frame rather than a coded outcome, but it is
  disclosed because it could in principle have anchored my `audits_model_on_pool` column. In
  practice every one of the 19 papers carried an explicit, verified by-group metric quote, so the
  column was determined by the papers rather than by that sentence - though it should be treated
  as an uninformative column for reliability purposes, exactly as the protocol says.
* No other results, counts, or summary text was read.
