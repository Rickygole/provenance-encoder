# Survey protocol: model audits vs. selection audits in clinical ML/NLP fairness evaluations

Rebuilt from scratch. All primary data in this directory was produced by the runs documented here.

**Search executed:** 2026-08-09 (single day). **Publication window:** 2019-2026.


## 1. Research question the coding serves

Fairness evaluations in clinical ML/NLP are typically computed on a cohort that has already been filtered (inclusion criteria, note/data availability, completeness rules). Two distinct audits are possible:

- **(A) Model audit** - auditing a trained model for disparities across groups *within the resulting pool*.
- **(B) Selection audit** - auditing the *filtering step itself* for differential retention across groups, against a pre-selection population denominator.

This survey measures how often each is done and how often both appear in the same paper. It is a descriptive coding exercise, not a hypothesis test. No prior count was consulted while building this table.


## 2. Databases and interfaces

Queries were issued against machine-readable APIs rather than a web search box, so they are exactly reproducible by re-running the recorded strings:

| Source | Interface |
|---|---|
| PubMed (incl. JAMIA, JAMIA Open, J Biomed Inform, npj Digit Med, JMIR, PLOS Digit Health) | NCBI E-utilities `esearch`/`efetch`, `db=pubmed` |
| PubMed Central (full text) | NCBI E-utilities `efetch`, `db=pmc` (open-access subset) |
| arXiv (cs.CL, cs.LG, cs.CY) | arXiv Atom API `export.arxiv.org/api/query` |
| ACL Anthology (Clinical NLP Workshop, BioNLP) | volume index pages on aclanthology.org |

**Coverage limitation, stated plainly.** FAccT, CHIL, ML4H and AIES were *not* searched through their own proceedings indexes. Papers from those venues enter this corpus only where an arXiv preprint exists and the arXiv queries surfaced it. Paywalled journal content is reachable only where a PubMed Central open-access full text exists. Both gaps are visible in `screening_log.csv` and are discussed in section 7.


## 3. Exact search strings

### 3.1 PubMed (`db=pubmed`, retmax=250 per query)

**P1** - total hits 1460, records retrieved 250

```
(fairness[Title/Abstract] OR "algorithmic bias"[Title/Abstract] OR "algorithmic fairness"[Title/Abstract]) AND ("machine learning"[Title/Abstract] OR "deep learning"[Title/Abstract] OR "artificial intelligence"[Title/Abstract] OR "natural language processing"[Title/Abstract]) AND (clinical[Title/Abstract] OR "electronic health record"[Title/Abstract] OR "electronic health records"[Title/Abstract] OR hospital[Title/Abstract] OR patients[Title/Abstract]) AND ("2019"[Date - Publication] : "2026"[Date - Publication])
```

**P2** - total hits 783, records retrieved 250

```
("subgroup performance"[Title/Abstract] OR "subgroup analysis"[Title/Abstract] OR "performance disparities"[Title/Abstract] OR "disaggregated"[Title/Abstract]) AND ("machine learning"[Title/Abstract] OR "prediction model"[Title/Abstract] OR "deep learning"[Title/Abstract]) AND (clinical[Title/Abstract] OR health[Title/Abstract]) AND ("2019"[Date - Publication] : "2026"[Date - Publication])
```

**P3** - total hits 1664, records retrieved 250

```
(bias[Title/Abstract] OR equity[Title/Abstract] OR disparities[Title/Abstract]) AND ("clinical natural language processing"[Title/Abstract] OR "clinical notes"[Title/Abstract] OR "clinical text"[Title/Abstract] OR "large language model"[Title/Abstract] OR "large language models"[Title/Abstract]) AND ("2019"[Date - Publication] : "2026"[Date - Publication])
```

**P4** - total hits 184, records retrieved 184

```
("model fairness"[Title/Abstract] OR "fairness metrics"[Title/Abstract] OR "equalized odds"[Title/Abstract] OR "demographic parity"[Title/Abstract]) AND (health[Title/Abstract] OR clinical[Title/Abstract] OR medical[Title/Abstract]) AND ("2019"[Date - Publication] : "2026"[Date - Publication])
```

**P5** - total hits 197, records retrieved 197

```
("race"[Title/Abstract] OR "ethnicity"[Title/Abstract] OR "sex"[Title/Abstract] OR "insurance"[Title/Abstract]) AND ("AUROC"[Title/Abstract] OR "AUC"[Title/Abstract] OR "calibration"[Title/Abstract]) AND ("machine learning"[Title/Abstract] OR "deep learning"[Title/Abstract]) AND (bias[Title/Abstract] OR fairness[Title/Abstract] OR disparit*[Title/Abstract]) AND ("2019"[Date - Publication] : "2026"[Date - Publication])
```

### 3.2 arXiv (Atom API, sortBy=relevance, up to 200 records per query)

**A1** - records retrieved 200

```
(cat:cs.CL OR cat:cs.LG OR cat:cs.CY) AND abs:"fairness" AND (abs:"clinical" OR abs:"health" OR abs:"medical" OR abs:"patient")
```

**A2** - records retrieved 184

```
(cat:cs.CL OR cat:cs.LG) AND abs:"bias" AND (abs:"clinical notes" OR abs:"electronic health record" OR abs:"EHR" OR abs:"MIMIC")
```

**A3** - records retrieved 49

```
abs:"algorithmic fairness" AND (abs:"healthcare" OR abs:"clinical")
```

**A4** - records retrieved 71

```
(abs:"subgroup" OR abs:"disparities") AND abs:"machine learning" AND (abs:"clinical" OR abs:"health") AND abs:"evaluation"
```

**A5** - records retrieved 104

```
abs:"fairness" AND abs:"large language model" AND (abs:"clinical" OR abs:"medical" OR abs:"health")
```

### 3.3 ACL Anthology

Volume indexes enumerated and titles matched case-insensitively against `fair|bias|disparit|equit|demographic|racial|gender`. Volumes searched (13 existed of 16 attempted):

```
2020.clinicalnlp-1, 2020.bionlp-1, 2021.bionlp-1, 2022.clinicalnlp-1, 2022.bionlp-1, 2023.clinicalnlp-1, 2023.bionlp-1, 2024.clinicalnlp-1, 2024.bionlp-1, 2025.clinicalnlp-1, 2025.bionlp-1, 2026.clinicalnlp-1, 2026.bionlp-1
```
Title-level hits: 7


## 4. Screening

**Inclusion criteria** (all three required):

1. Health/clinical/biomedical data about people.
2. Reports at least one disaggregated performance or outcome metric by a demographic or socioeconomic attribute.
3. Enough methods detail to determine what cohort the metric was computed on.

**Exclusion criteria:** commentary/position/review with no empirical evaluation; no disaggregated metric; not health data; methods too thin to identify the cohort; full text not retrievable.

**Stage 1 - title/abstract screen (automated, rule-based).** Applied to every retrieved record. A record was excluded if it had no usable abstract, named no demographic/socioeconomic attribute, reported no performance or disparity metric, was outside the date window, was not health data, or was a review/position article. The exact rule that excluded each record is written verbatim into `screening_log.csv`.

Records entering stage 1: **1596**. Passed stage 1: **544**.

**Stage 2 - full-text retrieval quota.** Full-text retrieval and hand-coding is the binding cost. Records that passed stage 1 were ranked by a pre-specified keyword-density score (count of distinct protected attributes x2, plus distinct metric terms, plus capped empirical-method terms) and the top-ranked records were advanced until the retrieval quota was met: top 50 PubMed records possessing a PubMed Central identifier, top 28 arXiv records, and all ACL Anthology title hits. Records passing stage 1 but below the quota are logged as `not_advanced_to_fulltext` - they were screened, not silently dropped, and are recoverable for extension.

**Stage 3 - full-text screen.** Applied by reading the retrieved full text against the three inclusion criteria.


### 4.1 Retrieval outcomes

- PubMed Central: 45 full texts retrieved; 5 not in the open-access subset (**unretrievable**).
- arXiv: 23 retrieved as HTML; the remainder were recovered from PDF via `pdftotext`.
- ACL Anthology: all title hits retrieved as PDF via `pdftotext`.
- Papers with usable full text advanced to coding: **80** (45 PubMed/PMC, 28 arXiv, 7 ACL).


## 5. Corpus integrity incident (disclosed)

The first full-text download used the XPath `.//ArticleId` on PubMed XML. That expression also matches identifiers inside each record's **reference list**, so for many records the extracted PMC ID and DOI belonged to a cited reference rather than to the article itself, and the wrong full text was downloaded. Coders flagged the mismatch independently. The extraction was corrected to `./PubmedData/ArticleIdList/ArticleId`; **731 of 1052 PMC IDs and 635 DOIs changed**. All PubMed full texts were re-downloaded, every download is now gated on an automated check that the fetched article's title matches the PubMed title, and **all PubMed papers were re-coded from scratch**. The discarded first-round PubMed codings were quarantined, not reused. arXiv and ACL retrieval used a different code path, was verified title-by-title (0 mismatches), and was unaffected.


## 6. Coding rules

Every coder received the following rule text verbatim, and coded independently from local full text only (no web access during coding). Papers were split across coders; each paper was coded once.

**Quote verification.** Every field requires a verbatim quote plus a section location. After coding, `verify_quotes.py` re-checks each quote by exact string match against the source full text (normalising only whitespace, unicode quotes/dashes and ligatures). **Any quote that fails to match is rejected and its field is forced to `unclear`.** No coded value survives in the table without verified textual support. The one documented exception is `null_attribute_handling = not_mentioned`, which asserts an absence: there the `location` field records the keyword search performed (`missing, unknown, not reported, declined, imputation, complete case`) and its null result, and the row is marked `quote_verified: "n/a_absence_check"`.

### 6.1 The stated rule for borderline `audits_selection` cases

Borderline cases were resolved by a single rule, applied to every paper:

> A paper counts as auditing selection **only if it compares the group composition, or the retention/inclusion rate, of the retained cohort against a denominator that exists prior to filtering** - or otherwise directly tests the filter for differential retention. A Table 1 that merely describes the final cohort's demographics is **not** a selection audit, because it has no denominator. A CONSORT-style flow diagram giving only total counts excluded, with no demographic breakdown, is **not** a selection audit.

Because that line is contestable, a second column, **`audits_selection_literal`**, records the more permissive reading: any comparison of retained group composition to *any* external or source figure counts, including comparison to census/national statistics, to the parent database before filtering, or to a catchment population. `audits_selection_literal` is a strict superset of `audits_selection`; the invariant is enforced in code. Both readings are reported everywhere. Papers the coders found genuinely hard are prefixed `HARD CASE:` in the `notes` column and counted in `coding_summary.json`.

### 6.2 Full rule text as issued to coders

```markdown
# CODING RULES — apply these EXACTLY. Do not improvise.

You are coding papers for a systematic survey. The survey question:

Fairness evaluations in clinical ML/NLP are almost always computed on a cohort that has ALREADY been
filtered (inclusion criteria, note availability, data completeness, etc.). Two DIFFERENT audits exist:

  (A) MODEL AUDIT  — auditing a trained model for disparities across groups *within the resulting pool*.
  (B) SELECTION AUDIT — auditing the FILTERING STEP ITSELF for differential retention across groups,
      against a PRE-SELECTION population denominator.

We are measuring how often each is done. This is NOT a hypothesis to confirm. Code what the paper says.
A paper that does (B) is just as interesting as one that does not. Do not shade your coding either way.

## STEP 1 — FULL-TEXT SCREENING (include/exclude)

INCLUDE only if ALL THREE hold:
 (i)   The paper uses health/clinical/biomedical data about people (patients, respondents, clinical text,
       medical images, health survey data, clinical QA/vignettes about patients).
 (ii)  It reports AT LEAST ONE disaggregated performance or outcome metric broken out by a demographic
       or socioeconomic attribute (race, ethnicity, sex, gender, age, insurance, language, SES,
       education, income, disability, skin tone, geography-as-SES).
 (iii) It has enough methods detail to determine WHAT COHORT the metric was computed on.

EXCLUDE if: pure commentary/position/review with no empirical evaluation; no disaggregated metric at all
(e.g. only reports overall accuracy, or only measures embedding association with no performance metric
by group); not health data; or methods too thin to tell what cohort was used.

If you exclude, set "included": false and give a one-sentence "exclusion_reason". Still emit the record.

## STEP 2 — FIELDS

For EVERY field you must supply a VERBATIM QUOTE copied character-for-character from the paper file,
plus a "location" (the section heading it came from, or "ABSTRACT" / "TABLE").
**If you cannot find a supporting quote, the value MUST be "unclear". Never infer, never guess.**
Quotes will be automatically string-matched against the source file. A quote that does not match exactly
will be rejected and the field forced to "unclear". Copy exactly; do not fix typos, do not paraphrase,
do not add ellipses in the middle. Keep quotes under ~320 characters.

### audits_model_on_pool  -> yes | no | unclear
"yes" if the paper reports at least one performance or outcome metric computed separately by group
(e.g. AUROC by race, FNR by sex, accuracy by insurance, outcome rate by ethnicity) on its analysis cohort.
Quote the sentence/table line that reports the by-group metric.

### cohort_is_filtered -> yes | no | unclear
"yes" if the analysis cohort was produced by ANY narrowing of a larger source population: inclusion/
exclusion criteria, age restriction, minimum length of stay, requiring a note/image/lab to exist,
complete-case restriction, requiring non-missing demographics, dataset-level curation the authors describe.
Quote the criterion. "no" only if the paper analyses an entire source population with no narrowing.
NOTE: for benchmark/vignette datasets the authors themselves constructed or sampled, sampling/curation
counts as filtering — quote it.

### audits_selection -> yes | no | unclear   [STRICT READING — this is the primary coding]
"yes" ONLY if the paper compares the group composition (or retention/inclusion RATE) of the RETAINED
cohort against a PRE-SELECTION or SOURCE population denominator, or otherwise directly evaluates the
filter for DIFFERENTIAL RETENTION across groups. Examples that qualify:
  - "Black patients were more likely to be excluded for missing notes (18% vs 11%, p<.001)"
  - a table comparing demographics of included vs excluded patients
  - reporting % retained by group relative to the pre-filter cohort
  - a statistical test of whether exclusion depended on a protected attribute
Examples that DO NOT qualify (code "no"):
  - a Table 1 that merely describes the demographics of the final cohort (no denominator)
  - a CONSORT/flow diagram giving only total counts excluded, with no demographic breakdown
  - reporting group sizes / class imbalance within the final cohort
  - noting that a group is "underrepresented" without any external or pre-selection figure
  - auditing the MODEL for disparities (that is audits_model_on_pool, not this)

### audits_selection_literal -> yes | no | unclear   [PERMISSIVE READING — record separately]
"yes" if there is ANY comparison of the retained cohort's group composition to ANY external or source
figure. This is a SUPERSET of the strict reading. Additionally counts:
  - comparing cohort demographics to census / national / regional population statistics
  - comparing cohort demographics to the full parent database before the authors' filtering
  - comparing to the demographics of the hospital/health-system catchment population
  - an explicit representativeness statement that cites a specific external figure
Still does NOT count: Table 1 alone with no external referent; vague "our sample skews white" with no figure.
If audits_selection == yes then audits_selection_literal MUST also be yes.

### null_attribute_handling -> drops | imputes | separate_category | reports_only | not_mentioned | unclear
What the paper does with records MISSING the stratifying (protected) attribute.
  drops              = records missing the attribute are excluded from analysis
  imputes            = missing attribute values imputed/inferred
  separate_category  = kept and analysed as "Unknown"/"Missing"/"Other"/"Declined" group
  reports_only       = missingness is reported but no handling is stated
  not_mentioned      = the paper never mentions missingness in the protected attribute at all
  unclear            = mentioned but you cannot tell what was done
For "not_mentioned" ONLY, you may leave quote as "" and set location to:
  "ABSENT - searched: missing, unknown, not reported, declined, imputation, complete case"
You must actually run that search over the file before using not_mentioned.

### protected_attributes -> list of strings
The attributes the paper actually disaggregates by. Use short lowercase labels:
race, ethnicity, sex, gender, age, insurance, language, ses, education, income, disability, skin_tone,
marital_status, geography. Quote the sentence listing them.

### fairness_metrics -> list of strings
The metrics used to compare groups. Short labels, e.g.: auroc_by_group, auprc_by_group, accuracy_by_group,
f1_by_group, sensitivity_by_group, specificity_by_group, fnr_gap, fpr_gap, calibration_by_group,
equalized_odds, demographic_parity, equal_opportunity, statistical_parity_difference, disparate_impact,
outcome_rate_by_group, error_rate_gap, subgroup_auc_gap, counterfactual_consistency, other.
Quote the sentence naming the metrics.

### notes
Anything ambiguous; ANY judgement call you made; especially flag borderline audits_selection cases and say
which way you went and why. If the paper is a HARD CASE for audits_selection, start notes with "HARD CASE:".

## OUTPUT
Write STRICT JSON (no markdown fence, no commentary) to the output path given in your task, shaped:

{"papers":[{"paper_id":"...","included":true,"exclusion_reason":null,
 "audits_model_on_pool":{"value":"yes","quote":"...","location":"Results"},
 "cohort_is_filtered":{"value":"yes","quote":"...","location":"Methods"},
 "audits_selection":{"value":"no","quote":"...","location":"Methods"},
 "audits_selection_literal":{"value":"no","quote":"...","location":"Methods"},
 "null_attribute_handling":{"value":"drops","quote":"...","location":"Methods"},
 "protected_attributes":{"value":["race","sex"],"quote":"...","location":"Methods"},
 "fairness_metrics":{"value":["auroc_by_group"],"quote":"...","location":"Results"},
 "notes":"..."}]}

For an EXCLUDED paper emit {"paper_id":"...","included":false,"exclusion_reason":"...","notes":"..."} only.

## PROCESS REQUIREMENTS
- READ THE WHOLE FILE for each paper before coding it (use Read with offset/limit to page through long files;
  also use Grep/Bash grep on the file to locate terms like "exclud", "missing", "Table 1", "census",
  "eligible", "compared with the", "underrepresent").
- Code ONLY from the local file you are given. Do NOT fetch anything from the web. Do NOT read any other
  files in the user's repo.
- Work paper by paper. Do not batch-guess.
```


## 7. Known limitations

0. **`audits_model_on_pool` is `yes` for every included paper by construction, and this is the most important caveat in the whole survey.** Inclusion criterion (ii) *requires* at least one disaggregated metric, so a paper that audits no model on its pool cannot enter the corpus. Two consequences follow. First, the marginal distribution of `audits_model_on_pool` carries no information - it is an artefact of the sampling frame, not a finding. Second, because every row has `audits_model_on_pool = yes`, `audits_both` is *mechanically identical* to `audits_selection`, and the model x selection cross-tab degenerates to a single row. The only quantity this corpus actually estimates is: **among papers that already audit a model for disparities, how many also audit the selection step.** It cannot speak to how common model audits are in clinical ML overall.
0b. **Corpus size.** The brief targeted 30-40 included papers; retrieval and coding succeeded more often than budgeted and 75 papers were included. Nothing was padded and nothing was dropped to hit a number - the ranked quota in section 4 was set before coding and every retrieved paper that met the criteria was kept. Readers wanting the smaller corpus can take the top of the ranking, which is preserved in `screening_log.csv`.
1. **Open-access bias.** Journal papers enter only via the PubMed Central open-access subset. Paywalled JAMIA / JBI / npj Digital Medicine papers are systematically under-represented and are logged as `unretrievable`.
2. **Venue coverage.** FAccT, CHIL, ML4H and AIES were not indexed directly (see section 2).
3. **Single coding pass.** Each paper was coded once, by one coder, under a shared written rule set. No second coder and no inter-rater reliability statistic was computed, so no agreement figure is claimed.
4. **Ranked-quota selection.** The full-text set is the top of a keyword-density ranking, not a random sample of everything that passed stage 1. The ranking favours papers that foreground fairness vocabulary in title/abstract, which plausibly *over*-samples fairness-aware work relative to the clinical ML literature as a whole.
5. **Extraction fidelity.** PDF-derived texts (ACL, some arXiv) are two-column extractions; quotes from them can be short line fragments. PMC-derived texts include section paragraphs and table content but not figure images, so a selection audit presented *only* inside a figure could be missed.


## 8. Files

| File | Contents |
|---|---|
| `coding_table.csv` | One row per included paper, all coded fields, both readings of `audits_selection`. |
| `quotes.json` | Per paper, per field: the verbatim quote, its location, and its verification status. |
| `screening_log.csv` | Every record screened, with disposition and the reason for exclusion or non-retrieval. |
| `protocol.md` | This file. |
| `tabulate.py` | Computes every reported number from `coding_table.csv`. |
| `coding_summary.json` | Output of `tabulate.py`. No count in it is typed by hand. |
