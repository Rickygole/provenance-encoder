# Premise survey: findings

N = 37 clinical and biomedical weak-supervision papers, 2019 to 2026, coded from full text.
Sampling frame, queries, eligibility and coding limitations: `survey_method.md`.
Record-level codes with supporting quotes: `survey.json`.

## 1. Headline

**The claim as currently written is not supported, and must be rewritten.**

The requested headline sentence, filled in honestly, reads:

> Of 37 clinical weak-supervision papers sampled, **0** evaluated the labelling stage against a
> chance or majority-class baseline only, and **0** reported a minimum detectable effect.

M is zero not because papers do better than chance comparison, but because **papers almost never
make a chance comparison at all**. Only 2 of 37 report any random or chance baseline anywhere in
the paper, and only 1 of 37 reports a majority-class baseline. The modal certification of a
labelling stage in this literature is an **absolute** precision, recall, F1 or AUROC on a small
gold set, with no reference distribution of any kind: no chance baseline, no majority-class
baseline, no non-text baseline, no stated null, and no statement of what effect the gold set was
large enough to detect.

That is a different and, for the paper's purposes, stronger finding: the problem is not that
the bar is set at chance, it is that **no bar is set at all**.

## 2. Counts

### 2.1 Was the labeller's own quality evaluated against gold?

| Code | N | Share |
|---|---|---|
| yes, labeller scored against gold or manual annotation | 20 | 54% |
| partial (only a component scored; or scored on the set used to tune the labelling functions; or the labeller-as-model scored but not the labels it emitted) | 6 | 16% |
| no gold-referenced evaluation of the labelling stage | 11 | 30% |
| **total** | **37** | |

"Partial" is doing real work here. In Datta 2023 the labelling functions are scored on the set
used to develop them. In Das 2025, Agnikula Kshatriya 2020 and 2021, and Gao 2022 the prompted
or rule-based labeller is scored, but the weak labels actually emitted onto the training corpus
are not. Treating partial as negative gives 17 of 37 (46%) with no clean held-out evaluation of
the emitted labels.

### 2.2 Comparators (coded generously: present anywhere in the paper)

| Comparator class | N | Share | Papers |
|---|---|---|---|
| random or chance baseline | 2 | 5% | Gan 2025 (random pairs, for relation detection, not for the phenotype labeller); Agrawal 2022 (random guessing) |
| majority-class baseline | 1 | 3% | Agrawal 2022 ("Most Common") |
| structured data / metadata / diagnosis-code-only baseline | 9 | 24% | Jeffery 2024; Dong 2023; Bejan 2022; Nogues 2022; Gan 2025; Eyuboglu 2021 (mortality endpoint only); Sandlin 2026 (ablation, not a named baseline); Al-Garadi 2026; Guevara 2024 |
| the same, restricted to a named non-text comparator on the same task as the labeller | 7 | 19% | the above minus Eyuboglu and Sandlin |
| simpler text model or rule baseline | 25 | 68% | |
| human inter-annotator agreement statistic reported | 10 | 27% | Das, Banerjee, Azzouzi, Dong, Bejan, Shen, Cusick, Sanyal, Sandlin, Guevara |
| a different automatic or human labeller | 23 | 62% | |
| a model trained on manual gold labels | 14 | 38% | |
| no comparator at all for the labelling stage | 1 | 3% | Gori 2019 |

Two further qualifications, both of which cut against the papers:

- Two additional papers (Eyuboglu 2021, Sandlin 2026) **mention** chance in prose ("no better
  than random", "remained above chance") without reporting a chance baseline. So the rhetorical
  move the claim describes does occur; it is just not usually backed by a computed comparator.
- Of the 7 papers with a non-text comparator on the same task, in 5 the non-text information is
  itself part of the labelling stage's input or is the label source (Bejan: ICD probabilities
  enter the weak label; Nogues: codified surrogates drive the silver labeller; Al-Garadi: the ICD
  heuristic **is** the labeller; Jeffery: administrative codes are among the labelling functions;
  Gan: codified features are among ARCH's inputs). In those cases beating the metadata baseline
  is not a test against an independent rival explanation of the score. **In no paper in the
  sample could I confirm from the retrieved text that the labeller's own output was compared
  against an independent metadata-only baseline on the same gold evaluation set.** One record
  (Dong 2023) is coded uncertain rather than negative on this point, because the ICD-only
  baseline is named in a figure caption but the full results table was not readable in the
  retrieved full text.

### 2.3 Statistical inference

| Item | N | Share |
|---|---|---|
| any p-value or named statistical test reported anywhere | 16 | 43% |
| any confidence interval reported anywhere | 13 | 35% |
| a confidence interval on the labelling stage's own metric | 6 | 16% |
| a test whose comparison involves the labelling stage itself | 4 | 11% |
| the null hypothesis of that comparison stated explicitly | 0 | 0% |
| **a test against a chance or zero-skill null** | **0** | **0%** |

The four labeller-level tests are Fries 2021 (Wilcoxon over label-model variants), Cusick 2021
(McNemar against the rule-based labeller's predictions), Smit 2020 (paired bootstrap F1
difference against the CheXpert labeller) and Eyuboglu 2021 (paired permutation, weak versus
supervised). In every case the null is a null of **no difference from another system**, left
implicit. No paper in the sample asks whether its labeller's agreement with gold is
distinguishable from a trivial or non-semantic predictor.

Several papers use the word "significant" or "significantly outperformed" with no test behind it
(Azzouzi 2024 abstract; Al-Garadi 2026; Yoon 2023 abstract; Zhu 2021).

### 2.4 Reporting of prevalence and gold positives

| Item | reported | derivable from reported counts | not reported |
|---|---|---|---|
| positive prevalence of the gold or evaluation set | 13 | 9 | 15 |
| number of gold positives in the evaluation set | 12 | 8 | 17 |

Where gold positives are reported, they are often very few: 5 positives in 599 visits
(Jeffery 2024, prevalence 0.83%), 15 positives in 80 reports (Das 2025), 19 positive patients
(Cusick 2021), 20 hand-annotated notes for the COVID case study (Fries 2021), 52 and 69
positives in the held-out sets (Greco 2026), 0.7% and 0.56% sentence-level positives
(Agnikula Kshatriya 2020, 2021). None of these is accompanied by any statement of resolvable
effect.

### 2.5 Power and minimum detectable effect

| Item | N |
|---|---|
| power analysis, sample-size justification or MDE for the labelling evaluation | **0 of 37** |
| a power figure reported for some other comparison | 1 (Gori 2019, post-hoc power 0.84 for the EHR versus EPIC-26 concordance analysis) |
| the word "power" used in a non-design sense | 1 (Gan 2025, power as sensitivity at a target FDR) |

This is the cleanest result in the survey. Across 37 papers, including papers published in
Nature Communications, npj Digital Medicine, JAMIA, JBI, JMIR Medical Informatics, EMNLP and
AAAI, and including gold sets with as few as five positive cases, **no paper reported what
difference its labeller evaluation was able to detect**.

### 2.6 Sensitivity analysis

Dropping one of each companion-paper pair (removing Gori 2019 and Agnikula Kshatriya 2021),
N = 35: chance baseline 2 of 35; majority-class 1 of 35; non-text comparator on the same task
7 of 35; no gold-referenced labeller evaluation 10 of 35; power or MDE 0 of 35. No conclusion
changes.

Dropping the one borderline-scope record (Gan 2025, a knowledge-graph paper), N = 36: chance
baseline falls to 1 of 36, which strengthens the negative finding.

## 3. The exact sentence the evidence supports

Use this, or a shortening of it. Every number in it is in the table above and traceable to a
coded record in `survey.json`.

> In a sample of 37 clinical and biomedical weak-supervision papers published between 2019 and
> 2026 and coded from full text, the labelling stage is typically certified by an absolute
> agreement score on a small gold set rather than against any stated reference. Eleven of the 37
> report no gold-referenced evaluation of the labelling stage at all, and six more evaluate only
> a component of it or evaluate it on the data used to tune it. Of the 37, 2 report any random or
> chance baseline, 1 reports a majority-class baseline, and 7 report a non-text comparator built
> from structured data or diagnosis codes, in 5 of which that non-text information is itself part
> of the labelling stage's input or is the label source. Fifteen do not report the positive
> prevalence of the evaluation set and 17 do not report the number of gold positives, which in
> the papers that do report it falls as low as five. No paper tests its labeller's agreement with
> gold against a chance or zero-skill null, no paper states the null hypothesis of its
> labeller-level comparison, and no paper reports a power analysis or minimum detectable effect
> for the labelling evaluation.

A one-sentence version for an introduction:

> Across 37 clinical and biomedical weak-supervision papers (2019 to 2026) we coded from full
> text, no paper reported a minimum detectable effect for its labeller evaluation, only 7
> compared the labelling stage against any non-text baseline, and 11 reported no gold-referenced
> evaluation of the labelling stage at all.

## 4. The exact sentence the evidence does NOT support

Do not write any of these:

> In practice, weak-supervision labellers in clinical NLP are commonly certified as adequate by
> showing that their output beats chance.

Unsupported. Only 2 of 37 papers report a chance baseline at all, and in neither is the
labelling stage certified by that comparison. Two further papers gesture at chance in prose
without computing it. "Commonly certified by beating chance" describes 0 of 37 papers.

> Clinical weak-supervision papers compare their labellers against chance rather than against a
> metadata-only baseline.

Unsupported as a contrast, because the first half is false. The comparison that is actually
missing is missing against **both** alternatives.

> Clinical weak-supervision papers do not evaluate their labellers.

Too strong. 20 of 37 do score the labeller against gold, and a handful (Smit 2020, Nogues 2022,
Vidmar 2025, Eyuboglu 2021) do it carefully, Nogues 2022 going so far as to report a silver
labeller with an AUC of 0.52. The defect is not absence of evaluation; it is evaluation without
a reference distribution and without a resolvable-effect statement.

## 5. What this licenses the paper to argue

1. The gap is **not** a too-low bar. It is a **missing bar**: a labelling stage is reported as
   adequate on an absolute agreement number, with no null, no rival explanation, and no statement
   of what the gold set could have detected. This is a stronger and more defensible framing than
   the one the reviewer objected to, and it is what the data show.
2. The specific, countable absence to build on is the **minimum detectable effect**: 0 of 37.
   That is a clean, checkable claim, and it is exactly the methodological contribution an
   evaluation-methodology paper can make without needing new data.
3. The second countable absence is **prevalence and gold-positive reporting**: 15 of 37 and 17 of
   37 respectively. Without these, a reader cannot even reconstruct the chance level, which is
   why the chance comparison is absent rather than lenient. This is the mechanism behind finding 1
   and should be stated as such.
4. A metadata-only or structured-feature baseline is the right rival explanation to propose,
   and the survey supports proposing it as **new practice**, not as a correction of bad practice:
   7 of 37 papers include such a comparator at all, and in 5 of those it is confounded with the
   label source. The paper should say "rarely available, and when present usually not independent
   of the labeller's own input", not "preferred to chance comparison".

## 6. Honest weaknesses a reviewer will find in this survey

- Purposive sample, not pre-registered, not exhaustive. One large PubMed block (241 records) was
  run and deliberately not screened; 26 of 67 arXiv records were unreadable.
- Open-access and machine-readability bias in eligibility; 5 candidate studies were dropped
  because the full text could not be read, 3 of them paywalled or embargoed.
- Single coder, no second rater, no coder agreement statistic. The survey has the same defect it
  documents, which should be stated in the paper rather than hidden.
- Comparators coded generously (present anywhere in the paper), which inflates every comparator
  count and therefore makes the negative results conservative.
- Four records were added by known-item search rather than by query; three of those four are
  among the most careful in the sample, so the deviation does not favour the claim.
