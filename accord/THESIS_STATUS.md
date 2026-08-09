# ACCORD thesis — status of record

**As of 2026-08-07.** 63 commits, 27 documented corrections, 5,402 lines of draft across ten
chapters. Every number below is traceable to a committed script and was independently
recomputed rather than relayed.


> **SUPERSEDED FRAMING — 2026-08-09.** **Evaluation instruments in clinical NLP routinely report numbers determined by the measurement procedure rather than by the quantity claimed, and the failure is invisible without a positive control or a degeneracy precondition.** (See CORRECTIONS.md C46 — the earlier 'audit blindness' framing is refuted by this project's own C38/C40/C41.)

---

## 1. What the thesis is now

ACCORD was built to detect dangerous medication contradictions between nursing notes and
discharge summaries in MIMIC-III. **It does not work, and the reason it does not work is now
known precisely.** What emerged from diagnosing it is an evaluation-methodology finding, and that
is the contribution.

**The claim, stated so every clause survives a hostile reader:**

> A per-group fairness audit computed on a post-selection pool is invariant to the selection rate
> itself. No audit performed at the model stage can detect that a subgroup was differentially
> removed upstream — regardless of whether the removal is discriminatory, benign, or purely
> mechanical. In a survey of 31 clinical NLP/ML fairness evaluations, no paper audits both stages,
> and the field's standard practice of dropping records with a null stratifying attribute deletes
> precisely the population most likely to have been removed upstream.

**The phrasing matters.** This must never be written as *"the audit missed a fairness harm."*
There is no demonstrated harm — see §4. It is *"the audit cannot see upstream removal at all."*
That restatement is what survives C22, C24 and C25, because four-fifths is defined on raw
selection rates and deliberately does not ask about cause.

**Contribution type:** evaluation critique with a survey and a worked example. Not a system, not a
benchmark, not theory. "We built a system" is not available.

---

## 2. What survives, with numbers

| finding | evidence | strength |
|---|---|---|
| **Survey: audits are disjoint** | 31 papers coded at methods level; of 25 auditing a model on a filtered pool, **0** audit the filter; of 4 auditing selection against a population denominator, **0** audit a model | **strongest — survived a falsifier and a blind reliability check** |
| **Survey reliability** | blind second coder, 10 papers, all retrieved, 21 verbatim quotes; **Cohen's κ 0.737–1.000**; decisive cell 0/10 strict, 2/10 literal (both flagged weak by the coder unprompted) | high |
| **Null-attribute handling** | 13 of 31 papers report nothing about records with a missing stratifying attribute; those that do mostly **drop** them | cleanest novelty delta |
| **Rule-transfer collapse** | auxiliary rule-head F1 **0.9603** in domain → **0.0654** sepsis / **0.0595** AKI, because the rule id is both an input feature and the target | high; gold-independent |
| **LLM over-calling** | 3 local 8B models, 1,679 pairs, 0% unparseable: **15.8×–23.6×** vs a 4.0% expert base rate. GPT-4o: **7.0×**. Floor across everything tested: **≥7×** | moderate — see C23 |
| **Three configurations at chance** | contaminated **0.4595**, row-decontaminated **0.5145** [0.321, 0.703], subject-disjoint control **0.497 ± 0.035**; DeLong **p = 0.930** | moderate; underpowered |
| **Anisotropy control** | 20 unrelated non-clinical sentences give cosine **0.9113** vs **0.9176** matched clinical — refutes the "clinical homogeneity" explanation positively | high, but not novel |
| **MedNLI** | BioMedBERT full fine-tune **83.9%** macro F1 vs LoRA r=16 **72.7%** | correctly run |

---

## 3. Novelty assessment: ~2.5/10

Checked twice against arXiv, OpenReview, FAccT/AIES/NeurIPS, JAMIA/JBI/npj Digital Medicine.

| claim | verdict | why |
|---|---|---|
| Null-attribute handling | **partially novel** | Martínez-Plumed 2019 owns the mechanism; Bhatti 2025 reports listwise deletion *maximizes* measured fairness — must be met head-on |
| Audit dissociation | **partially novel** | theory is Pfohl NeurIPS 2025; method is Qi JAMIA Open 2021; filter harm is Acitores Cortina JMIR 2025; stage-level auditing is Biswas & Rajan 2021 at n=37 pipelines |
| Leakage insensitivity | unremarked framing | evidence does not support the inference |
| LLM miscalibration | published phenomenon | direction is prompt-set, not model-set |
| Transfer collapse | **not a claim** | textbook; retract |

**The only defensible construction is the fusion:** the audit runs on a pool built by a filter
that itself fails the audit, *and* the audit's convention of dropping null-attribute records
deletes exactly the population that filter removed. That loop is in none of Pfohl (theory +
census), Qi (selection only), Acitores Cortina (filter harm only), or Biswas & Rajan (tabular
transformers, no missing-attribute dimension).

**Two citable attacks that must be answered in the paper, not discovered by a reviewer:**
- Watkins et al., *"The four-fifths rule is not disparate impact"* (arXiv:2202.09519) — applying an
  EEOC construct to a data-completeness flag. Defend explicitly or change metric.
- Pfohl et al. Section 5 already runs the selection simulations. Our sweep is replication.

**Coverage gap that can still move this down:** PubMed and the ACL Anthology have not been
searched. A JAMIA or JBI reporting-practice review would kill the null-attribute finding and
would not have been visible. **Run that pass before committing.**

---

## 4. What was retracted, and why it matters

Twenty-seven corrections. Seven framings died in two days, each to evidence rather than opinion.
The pattern is diagnostic:

> **Every claim that died required an adjustment set. Every claim that lived requires arithmetic.**

- **C14 — compounding bias.** OR 1.69 was a sign flip from conditioning on the mining rule, an
  output of the stage it was contrasted against. Crude direction is the opposite (0.82).
- **C15 — external validity.** "Something above chance is extractable" rested on GPT-4o MCC
  +0.109, which is χ² = 3.00, **p = 0.083**, bootstrap CI **[−0.032, 0.249]**.
- **C16 — contamination.** The reported evaluation was the contaminated run: decontaminated seeds
  give TP 9/10/9 of 12, the reported result is 11/12. Corrected AUROC is **0.5145, not 0.460**.
- **C17 — volume specification.** The adjustment used total note chars (91.4% discharge summary)
  while the rule binds on the nursing note. Entering both separately: HF **0.894, p = 0.13**.
- **C22 — degenerate classifier.** The audited model calls positive on **87.3%** of pairs; parity
  was near-guaranteed.
- **C24 — four-fifths passes on protected attributes.** Named minority **1.053**, recorded
  non-English **0.804**. Only the two missingness buckets fail.
- **C25 — era.** `LANGUAGE` is missing for **60.8%** of CareVue admissions vs **0.5%** of
  MetaVision, and yield differs 2.7× by era. Adjusted for volume + era: **0.859, p = 0.066**.

**Consequence for the clinical finding:** it is not a defensible standalone result. What remains
reportable is the raw composition shift — 46.6% of the cohort to 29.8% of the candidate pool —
with the mechanism openly attributed to documentation volume and EHR era, not patient language.

---

## 5. Plan

**Target: ML4H 2026 Proceedings, deadline 2026-09-10 (34 days).** Archival, PMLR, 8 pages.
The only venue whose decision (~late October) lands before December application deadlines.

**Split into two, do not merge:**
- **Paper A** (Proceedings, 8pp) — instrument validity; the audit work is one demonstration inside it, not the claim. ACCORD appears only as "the
  pipeline we audited." Does not depend on the detector working.
- **Paper B** (Findings, 4pp, non-archival) — the clinical negative result. Three configurations
  at chance, DeLong p = 0.930, TF-IDF (**0.576**) beating both neural models.

Merging them would import Paper B's underpowering into a paper that does not have it.

**Not realistic:** NeurIPS / ICML / ICLR / ACL main tracks. After Pfohl the contribution is
measurement, not method, and there is no releasable benchmark — MIMIC-III cannot be redistributed.

**Chapter plan:** seven chapters, ~3,150 lines, ~1,450 new. The audit-dissociation chapter does
not exist yet and is where nearly all new writing goes. `CORRECTIONS.md` goes in verbatim as
Appendix A — it is the credibility instrument and should not be paraphrased.

---

## 6. Open items

| item | status |
|---|---|
| **PubMed + ACL Anthology novelty pass** | **not done — can still invalidate the framing** |
| Mortality-cohort selection experiment | designed, not run. 34,018 examples, **2,975 positives**. Selection on outcome-predictive features, deliberately violating the independence assumption the simulation assumed. 12–16h |
| Propagate C16 through Chapters 7–9 | not started. Every AUROC there derives from the contaminated CSV |
| ~30 untraceable numbers | clustered in the MedNLI and architecture tables, incl. a parameter count that exists nowhere on disk |
| 15 mismatches | determinable fixes; two TODOs already answered (300th annotation is **PAIR_0076**; restricted CI is **[0.222, 0.762]**) |
| Compliance | two note-text CSVs and raw MIMIC tables still on Drive; `ch3_data.tex:595` contradicts `:604` seven lines apart |
| HF-API Llama row | record-only, must never appear in a submitted table |

---

## 7. The honest summary

Two weeks ago this was a system that did not work. It is now a workshop-scale paper with one
fused claim, aimed at a real deadline, with every supporting number verified and every failed
framing documented.

The strongest artifact the project produced is not on the evidence list. It is `CORRECTIONS.md`
itself — 27 self-audited corrections, including C14, where the analysis claiming to have learned
from the project's errors committed the same error. That scores near zero on novelty and is worth
more than the paper to an advisor reading it.

Two claims survived falsifiers designed to kill them: the ≥7× LLM over-call floor, and the survey
cross-tab. **Structure the paper so survival-under-falsification is visible**, because the loudest
signal this body of work currently sends is destruction, and that is not the signal to send in
December.
