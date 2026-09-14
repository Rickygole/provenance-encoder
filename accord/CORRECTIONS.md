# Corrections

> **NOT CLOSED. This banner declared the record closed at C62 on 2026-08-10 and said the
> closure was enforced rather than intended. Twenty-three corrections have followed it, so
> the banner is wrong in exactly the way it was written to mock: C46 declared closure and
> sixteen entries followed. It is left in place rather than deleted because a record that
> quietly removed its own failed closure would be making the point worse. The rule it
> announced, that new findings go to the revision log and cannot change the thesis
> statement or the contribution list, did not hold either: C65 withdrew a set of results
> and C85 changed which tests clear correction.**
> C46 declared this record closed and sixteen corrections followed it. New findings now go to
> `REVISION_LOG.md`, which by written rule cannot change the thesis statement, the contribution
> list, or the chapter structure. This file is Appendix A and is not the thesis's spine — it has
> one coder, no coding protocol, and an author-controlled denominator, which is a lower standard
> than this project applied to 75 strangers' papers. It is evidence of process, not a finding.

# ACCORD — corrections to the record

Every claim below appeared in the NeurIPS 2026 submission, its supplemental, or the
project's own result files, and is **not supported by the code that produced it**. Each is
listed with the evidence and the file it came from. These corrections are prerequisites for
the thesis: no chapter can cite a number that is not traceable to a script.

Compiled 2026-08-05 from an audit of the pipeline notebooks, `accord_phase3_v2_ablations.py`,
the private supplemental repo, and `thesis/outputs/` on Drive.

---

## C1. The authority weights are an initialization artifact

**Claimed.** *"Without explicit authority labels, the model's weights converge to the correct
clinical hierarchy (α_N ≈ 0.5, α_D ≈ 1.0),"* presented as the paper's third contribution and
as evidence that institutional authority is learnable from weak supervision.

**Actual.** `accord_phase3_v2_ablations.py:76,84`:

```python
def __init__(self, hidden_size=768, num_heads=8, dropout=0.1, authority_dampening=0.5):
    self.authority_scale = nn.Parameter(torch.tensor([authority_dampening, 1.0]))
```

The parameter is **initialized at exactly `[0.5, 1.0]`** — the values reported as discovered.
Across three seeds it moved to `(0.509, 1.006)`: 1.8% and 0.6%. Lines 105–108 apply the scale
to a cross-attention branch that is immediately LayerNorm'd, which normalizes it away and
leaves a near-zero gradient. Third-decimal agreement across seeds is the signature of a
parameter that did not train, not of a recovered signal.

Reviewers JDYj and dP2W both independently suspected this.

**Correction.** Retract the claim. Report the artifact, the mechanism, and the diagnostic that
would have caught it: for any parameter reported as *learned*, publish
‖θ_final − θ_init‖ / ‖θ_init‖ and a gradient-norm trace.

**Aggravating.** The paper's own ablation already refuted the claim: **symmetric ACNE
(α_N = α_D) beats the full asymmetric model** on test contradiction F1, 0.7775 vs 0.7368, and
on accuracy, 0.9192 vs 0.8939.

---

## C2. Rule F1 = 0.9603 is label leakage

**Claimed.** Rule-type prediction F1 of 0.9603 (0.953 ± 0.006 in Table 2), presented as
evidence that the multi-task auxiliary head learns pharmacologically grounded structure.

**Actual.** `_prior()` writes the rule id into the model's input:

```python
pf[i, 1] = rule / 4.0          # rule comes from the batch
...
return self.gghe(edge_repr, prior)
rule_logits = self.rule_type_head(attended)   # predicts RULE_MAP[rule]
```

The auxiliary label is an input feature and the prediction target in the same forward pass,
separated by one linear layer.

**Independent confirmation, and the better number to report.** On out-of-domain transfer the
rule head collapses: **sepsis F1 0.0654, AKI F1 0.0595** (`transfer_results_correct.json`),
against 0.96 in domain. On the held-out pharmacist set, rule-prediction accuracy is **45.2%**.

**Correction.** Withdraw 0.9603. Report the transfer collapse instead — it is the *evidence*
of leakage rather than a defect to conceal, and it answers the generalization question
reviewers asked.

---

## C3. The κ = 1.000 internal-consistency claim is circular

**Claimed.** `pharmacist_internal_consistency.json`: *"internal_consistency_kappa: 1.0 … Two
independent pharmacist judgment instruments agree perfectly. Rules out random or inconsistent
annotation,"* with `"rebuttal_use": "Answers N=1 annotator concern without a second annotator."*

**Actual.** The two "instruments" are the binary contradiction label and the 1–5 danger
rating. A pair that is *not* a contradiction has no danger **by definition**, so
`label == contradiction ⟺ danger_rating > 0` is an identity. κ = 1.000 is guaranteed by
construction and measures nothing about annotator reliability.

**Correction.** Remove from any rebuttal. Reviewers JDYj and p4E8 asked for inter-rater
reliability; this would read as evasion. The honest statement is that the study has one
annotator with a two-pass washout, and that inter-rater reliability is not established.

---

## C4. Three architectural components are not backed by running code

| Claimed | Actual |
|---|---|
| **PPC** — Pharmacological Prior Conditioning, 2,496 params, injects drug class / interaction / contraindication priors | **Deleted in v2.** Replaced by `_prior()`: a 64-dim vector of zeros with `pf[i,0] = danger_multiplier` and `pf[i,1] = rule/4.0`. **Zero learned parameters.** |
| **GGHE** — retrieval-augmented harm scoring against ACC/AHA 2022 and ESC 2021 guidelines | `nn.Parameter(torch.randn(20, 256))` — **20 random vectors**, cross-attended from a single query. No corpus, no index, no guideline text. The code comment reads *"In full version this attends over FAISS-retrieved chunks."* |
| **Guideline-grounded harm estimation** produces ranked danger scores | `harm_head` and `explanation_head` **receive no loss term**. Total loss is `1.0*c_loss + 0.3*r_loss`. Confirmed empirically: harm scores across 299 gold pairs span **0.398–0.610**, the range of an untrained sigmoid, and `harm_high_risk_rate = 0.0` in all three cohorts. |
| **1.54M new parameters** | Describes v1. v2 adds `AsymmetricCrossNoteEncoder` (two 8-head 768-dim MHA blocks plus two 768×768 projections ≈ 5.9M) for **≈7.5M**, and PPC's 2,496 are gone. |

**Correction.** Describe the model that produced the results. RQ3 (guideline-grounded severity
ranking) was **not tested** — the component that would answer it was never trained. State that.

---

## C5. The contaminated set must be removed

Reviewer 1yJj: *"There is no value in reporting on the 'contaminated' set, this is training on
the test set."* Correct. The contaminated run trained on the full 1,978-pair pool, which
includes the 299 evaluation pairs.

**Correction.** Delete all contaminated-set reporting. The decontaminated retrain is the only
honest ACCORD row.

---

## C6. The prefilter finding is an instrument failure

**Claimed.** `prefilter_finding.json`: *"Only 2 of 1610 filterable candidates were resolved by
intermediate documentation. This validates the clinical need for automated contradiction
detection."*

**Actual.** `RESOLUTION_PATTERNS` is a 20-way alternation containing **both directions at
once** — `resume|restart|continue|…|discontinue|hold|stop|…` — so it matches almost any
medication sentence, and a match anywhere within ±2 lines counts as "resolved." Three earlier
attempts returned 0 resolved because `CHARTTIME` is null for every discharge summary.

2 of 1,610 is what a broken keyword filter returns. It is not a measurement of how often
clinicians document conflict resolution.

**Correction.** Remove the claim. Report the `CHARTTIME` absence as a data limitation, which it
genuinely is.

---

## C7. Reported "3 seeds" are 2 seeds plus a hardcoded row

`SEEDS = [1, 2]`. Seed 0's metrics are a pasted literal from an earlier run at a different code
state. `set_seed()` is never called inside `train_ablation()`, so each variant inherits whatever
RNG state the previous one left. The data split is `random_state=42` in all cases, so the
reported ± std reflects initialization noise only, not sampling variance. Separately,
`ablation_results['VD_Full_ACCORDv2']` is a hardcoded literal — **the full model is not run in
the ablation script**, so the ablation table compares three freshly-trained variants against a
pasted reference.

**Correction.** Regenerate the table, or report it as two seeds with the reference row marked.

---

## C8. Splits are row-level, not patient-level

```python
labels_df = labels[labels['weak_label'] != 'ambiguous'].sample(frac=1, random_state=42)
train_df  = labels_df.iloc[:int(n*0.8)]
```

1,978 pairs come from **1,203 admissions** (~1.64 pairs each), drawn from a patient population
with known repeat admissions. Pairs sharing a nursing note, a discharge summary, or a drug class
routinely land on both sides. Reviewer 1yJj asked precisely this: *"Did any of these notes appear
in both training and testing (regardless of their pairing)?"*

**Correction.** Re-split by `SUBJECT_ID` with disjointness asserted in code
(`accord/phase3_mining.py` does this) and re-report.

---

## C9. `weak_confidence` is degenerate

The field takes exactly two values across all 1,679 training rows and all 299 audit rows —
0.95 iff `weak_label == contradiction`, 1.00 iff `not_contradiction`. It is a deterministic
function of the label, not a confidence estimate. The training code filters
`weak_confidence < 0.9` and additionally multiplies it into the per-example loss, so it
**filters and reweights by label**.

**Correction.** Remove the filter and the reweighting, or report them as label-conditioned.

---

## C10. The multi-LLM "model-agnostic" conclusion rests on one model

All three copies of the safety notebook show *"Failed to load Llama-3-8B: gated repo … 401."*
Only Mistral-7B executed, giving clinical 6.2×, finance 0.8×, legal **0.0×** (it predicted
`not_risky` for all 34 examples), cyber 0.3×. The `final_results` dict then contains a hardcoded
Llama row (21.4×, 15.6×) and prints *"Clinical miscalibration is model-agnostic."*

Additionally, `llm_preds.append(pred if pred else pk)` scores **every unparseable generation and
every exception as a positive**, inflating the positive rate that forms the numerator of the
miscalibration ratio.

**Correction.** Either run Llama or drop the model-agnostic claim. Fix the failure default and
recompute. Figure 7 fits an exponential to four invented "expertise" coordinates and must be
removed.

---

## C11. The ethnicity disparity is a missingness artifact

**Claimed.** Earlier fairness analysis in this project reported that candidate generation
under-selects minority patients, offered as the reviewer-requested sociodemographic breakdown.

**Actual.** Once `UNKNOWN/OTHER/DECLINED` is separated from named minority groups
(`results/accord/differential/exp3_disaggregation.json`):

| group | n | candidate yield |
|---|---|---|
| WHITE | 7,155 | 15.6% |
| NAMED MINORITY | 1,427 | **16.5%** |
| UNKNOWN / OTHER | 1,201 | 11.3% |

χ² = 0.55, p = 0.46 between White and named minority — no difference, and named minorities are
if anything selected slightly *more* often. The entire earlier effect was the unknown bucket.

**Correction.** Retract the ethnicity disparity. Report that there is **no evidence of a
race/ethnicity effect** in candidate generation, and that the apparent one was administrative
missingness misread as demography. This is the same failure mode the language analysis had.

---

## C12. The language finding splits, and only half survives

**Claimed.** Rule-based mining under-selects non-English-speaking patients (43.3% → 60.4%
English in the candidate pool, χ² = 150.7).

**Actual.** Admission-level logistic regression on all 9,783 cohort admissions
(`exp2a_candidate_yield_regression.json`), before and after adjusting for documentation volume:

| group | unadjusted OR | + log(note chars), LOS |
|---|---|---|
| recorded non-English | 0.77 [0.63, 0.95], p = 0.012 | **0.94 [0.76, 1.15], p = 0.53** |
| language not recorded | 0.41 [0.36, 0.46], p < 0.001 | **0.64 [0.55, 0.73], p < 0.001** |

The recorded-non-English deficit is fully explained by note volume and does not survive. The
missing-language deficit attenuates by ~40% and remains significant.

**An earlier version of this correction claimed the deficit "holds within every note-length
decile, CMH common OR 0.63, p = 1.2 × 10⁻¹³." That claim was wrong and is withdrawn.** The
per-decile odds ratios in `exp2b_decile_stratified.json` are:

| decile | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| OR | 0.90 | **1.35** | 0.98 | **1.10** | 0.86 | 0.58 | 0.56 | 0.49 | 0.48 | 0.42 |

The effect **reverses** in deciles 1 and 3, is null in 0, 2 and 4, and appears only in the upper
half, increasing monotonically with note length. Breslow-Day on these strata gives
**χ² = 31.11, p = 2.8 × 10⁻⁴** — homogeneity is rejected, so a CMH "common odds ratio" has no
referent here. This is effect modification by documentation volume, and it is concentrated in
decile 9, the only unbounded stratum and therefore the one with the most residual within-stratum
volume confounding. The stratification was run to rule volume out and its own output identifies
volume as the modifier.

**What the analysis is entitled to claim.** The missing-language deficit is real *among heavily
documented admissions* and absent among lightly documented ones. Report the interaction, not a
pooled OR. `DBSOURCE` (CareVue 2001–2008 vs MetaVision 2008–2012) is not in the adjustment set
and is the standard era proxy in MIMIC-III — `ADMITTIME` is date-shifted per patient so calendar
era is otherwise unrecoverable. Both LANGUAGE recording practice and nursing-note format changed
across that boundary, so OR 0.64 has an unexcluded era explanation until `DBSOURCE` is added.

**Correction.** The finding is not about language. It is about **administrative completeness**:
admissions with an unrecorded `LANGUAGE` field are systematically under-selected by rule-based
mining, independent of how much was written about them. State it that way.

---

## C14. The "compounding bias" claim is withdrawn — the sign was produced by adjustment

**Claimed** (by me, earlier the same day, and promoted as the thesis's novel contribution).
The LLM labeller moves in the *opposite* direction to the miner on the same axis: adjusted for
note length, mention count and rule, missing-language admissions receive a positive weak label
at OR 1.69 [1.11, 2.57], p = 0.015. *"The subgroup the miner drops is the subgroup the labeller
over-flags."*

**Actual.** Unadjusted, the effect points the other way and is not significant.
`exp1b_positive_rate_by_group.json`, the same n = 1,679 rows as the adjusted model:

| group | n | positive-call rate |
|---|---|---|
| ENGLISH | 1,064 | 0.8618 |
| **MISSING** | 464 | **0.8362** |
| RECORDED NON-ENGLISH | 151 | 0.7947 |

χ² = 5.42, **p = 0.066**. Crude MISSING-vs-ENGLISH OR = **0.82** — the missing group is flagged
*less*, not more. Adjustment moves 0.82 → 1.69, a reversal across the null.

**The mechanism of the reversal.** `accord/differential/run_experiments.py:348` conditions on
`C(rule_cat)`:

```python
"weak_pos ~ C(language_group3, Treatment('ENGLISH')) + log_note_chars + n_mentions_row + C(rule_cat)"
```

`rule_cat` is the identity of the mining rule that produced the row — a deterministic **output of
Stage 1**, the very stage C12 says selects differentially on this exposure. The sample is
additionally conditioned on `has_candidate == 1`. So the Stage-2 model adjusts for a mediator of
the exposure inside a sample selected on that mediator's parent: collider-stratification bias,
with C12 as the proof that the selection is differential. The volume covariates cannot be doing
the work — `log_note_chars` and `n_mentions_row` both have p ≈ 0.33 — which leaves `rule_cat`
(rule 4: OR 0.045, p = 1.5 × 10⁻⁴⁹). Rule 4 is the **omission** rule, "held in nursing, absent
from discharge," which is exactly what sparse, administratively incomplete admissions trigger.
The model conditions away the mechanism and reports the residue as bias.

**Independent replication, in the wrong direction, on the only data with ground truth.**
`exp1a_gold_error_rates_by_group.json`, the 299 pharmacist pairs — a sample *disjoint* from the
1,679:

| group | n | positive-call rate | FPR |
|---|---|---|---|
| ENGLISH | 191 | 0.880 | 0.874 |
| MISSING | 84 | 0.845 | **0.850** |

Two independent samples, both crude, both opposite to the claim. On the one measurement where
truth exists, the missing group's error rate is *lower*.

**Correction.** Delete "compounding" and OR 1.69. The defensible statement is: *unadjusted, the
labeller's positive-call rate does not differ by language-recording status (χ² p = 0.066); on the
gold subset the false-positive rate is 0.874 (English) vs 0.850 (missing); with 12 gold positives
a differential error rate is not estimable. The observed pipeline-level disparity is located at
Stage 1 alone.* Any retained Stage-2 statement must be fit marginally, without `rule_cat`, with
cluster-robust standard errors on `HADM_ID`, and reported next to the rule × language crosstab so
the mediator is visible.

**Why this one matters most.** C11 retracted an ethnicity disparity as a missingness artifact and
C12 split the language finding so only half survived. C14 is a *new* fairness finding recovered
from the half that had just been shown null, by adding a covariate that flips its sign. It is the
same failure the thesis is about, committed in the analysis that claims to have learned from it.
The document that catches C1–C13 did not catch C14, and C14 was the lead.

---

---

## C13. "Clinical homogeneity" does not explain the Stage 0 collapse

**Claimed.** Stage 0 self-supervised admission-coherence pretraining converged to exactly
ln(2) = 0.6931 with dev accuracy 0.5000 on every epoch, in both the frozen and partially
unfrozen runs. The explanation given: *heart failure admissions are too similar to each other
to generate a useful self-supervised signal.* Nothing in the notebook measures similarity.

**Actual** (`results/accord/stage0/`, four matched configurations, split by `SUBJECT_ID` with
a fixed dev set — the original had neither):

| config | head | window | final dev loss | dev acc |
|---|---|---|---|---|
| baseline | bottleneck | first 256 tok | 0.6933 | **0.5000 every epoch** |
| H1 ablation | direct concat | first 256 tok | 0.6929 | 0.5000, best 0.531 |
| H2 ablation | bottleneck | last 256 tok | 0.6933 | **0.5000 every epoch** |
| both fixed | direct concat | last 256 tok | 0.6929 | 0.5154 |

The baseline reproduces the published number exactly on a leak-free pipeline, so the collapse
is real. But the *cause* is architectural, not linguistic:

- **H2 (header truncation) is not supported.** Head → tail truncation moves cosine
  separability AUROC 0.5162 → 0.5177. Nothing.
- **H3 (clinical homogeneity) is refuted, positively rather than by absence of evidence.**
  Twenty deliberately unrelated non-clinical sentences — cooking, astronomy, finance, no MIMIC
  data — embedded through the identical frozen BioMedBERT CLS extraction give mean pairwise
  cosine **0.9113 ± 0.0207**, against **0.9176** for matched clinical pairs and **0.9162** for
  mismatched ones (`anisotropy_control.json`). A recipe and a marathon route are as "similar"
  as a real nursing/discharge pair from the same admission. The ~0.92 is CLS-token anisotropy
  in a non-contrastively-fine-tuned masked LM, not a property of heart failure notes.
- **H1 (the 1-D bottleneck) is the demonstrated cause.** `conflict_scorer → Sigmoid → Linear(1,1)`
  routes the entire 256-dim pair representation through one scalar before the loss. Both
  bottleneck configs sit at *exactly* 0.5000 in every epoch; both direct-head configs leave
  chance. It is the only intervention that changes the signature.

**Correction.** Remove the clinical-homogeneity explanation from the thesis and name the
bottleneck. State the compounding limitation honestly: even with the bottleneck removed the
task only reaches ~51.5%, because a frozen non-fine-tuned CLS token carries little
admission-level signal to begin with — which is the anisotropy result again, and is a
limitation of the *encoder choice*, not of clinical text.

**Caveat on this correction.** The reference notebook initialized the backbone from a Stage-1
MedNLI checkpoint that is not on disk; these runs use vanilla BioMedBERT. The backbone is
frozen in every configuration, so this cannot affect the H1/H2 gradient argument, but the
absolute cosine values are specific to the vanilla checkpoint.

---

## C15. The thesis's external-validity claim rests on a non-significant result

**Claimed.** `ch1_introduction.tex:249-252`: *"the task is not impossible: something above chance
is extractable from the same evidence, so the failure is a property of this pipeline rather than
of the problem."* This is the only sentence in ten chapters that converts an engineering
post-mortem into a research contribution; `ch1:275-281` explicitly disclaims everything else.

**Actual.** It rests entirely on GPT-4o zero-shot in `results/accord/step3_baselines.json`:
`tp 6, fn 6, fp 65, tn 176`, n = 253, MCC = +0.1089. Running the test that was never run:

| test | statistic | p |
|---|---|---|
| χ² = n·MCC², df 1 | 3.003 | **0.083** |
| χ² with Yates correction | 1.970 | 0.160 |
| Fisher exact, two-sided | OR 2.71 | 0.102 |
| bootstrap MCC 95% CI (20k) | — | **[−0.032, 0.249]** |

Expected TP under independence is 3.37 against 6 observed. The interval includes zero. It is the
maximum of four systems compared, uncorrected. No CI for it appears in any results file or any
chapter, and `step3_baselines.json`'s own note concedes no AUROC, no AP and no second operating
point can ever be computed for it because only aggregates were released.

**Why this is self-inflicted.** `ch8_metrics.tex:329-331` writes: *"A statistic that produces an
ordering over four systems with MCCs of +0.109, +0.012, −0.007, and −0.044 has produced an
ordering over a set in which three members are noise."* Three. The thesis applies its own
standard to every system except the one it needs. `ch8:332-335` then prescribes that such a claim
requires a threshold-free ranking measure with a confidence interval, a significance test, an
average precision against the base rate, and a top-k count. The +0.109 has none of the four. All
four systems are noise.

**Correction.** Withdraw "something above chance is extractable." State that no system tested,
including GPT-4o zero-shot, is distinguishable from chance on this evaluation.

**What this leaves, and the experiment that fixes it.** Without that sentence the residual claim
is *an author-built pipeline containing thirteen documented author-introduced defects did not
work* — and there is nothing to rebut it with, because **no vanilla cross-encoder, no NLI
baseline and no majority-class baseline exists on the ACCORD split.** `results/accord/` contains
exactly one continuous score, ACCORD's own `contra_prob`; every AUROC in the document (0.460,
0.5078, 0.5044) is that same defective model against the same 12 positives. The "four independent
measurements" of no-signal at `ch9:217-219` are four views of one instrument. The majority-class
baseline is still an open `[TODO]` at `ch6_architecture.tex:219` and `:526`, inside the list of
practices the thesis prescribes to others. And `section6_auroc_power.json` gives
`minimum_detectable_auc = 0.7382` at 80% power with 12 positives, so the instrument cannot
distinguish a broken system from a mediocre-but-real one at all.

The missing control is one experiment: a vanilla BioMedBERT cross-encoder pair classifier on the
`SUBJECT_ID`-disjoint split from `phase3_mining.py`, corrected exemplars, no rule feature, no
authority parameter, no auxiliary heads, scored continuously on all 299 with a bootstrap CI. If
it also lands at chance with an interval excluding 0.6, the thesis has, for the first time,
evidence that the failure is not the bugs. If it clears chance, the honest title is *a post-mortem
of an implementation*. Either result is defensible; shipping without it is not.

---

## C16. The reported evaluation appears to be the contaminated run

**Claimed, four separate times.** *"its results are not reported anywhere in this thesis"*
(ch4:513); *"The only reported configuration is the one trained on the held-out complement"*
(ch6:465); *"The single configuration that trained on those pairs is not reported anywhere in
this thesis"* (ch7:43); *"results from a configuration trained on the evaluation set are
withdrawn"* (ch7:754). C5 records the same commitment.

**Actual.** Every number in Chapters 7 and 8 and the Chapter 9 defect-free analysis derives from
`gold_standard_accord_results.csv`. Recomputed directly from that file: **TP 11, FN 1 — recall
11/12 = 0.9167.**

`accord_supp/results/accord_decontamination_paper_table.csv` gives, for the two configurations:

| metric | contaminated | decontaminated (3 seeds) |
|---|---|---|
| **recall** | **11/12 = 0.917** | **0.7778 ± 0.0393** |
| precision | 0.0470 | 0.0424 ± 0.0023 |
| f1_pos | 0.0894 | 0.0804 ± 0.0043 |
| pr_auc | 0.0461 | 0.0579 ± 0.0034 |
| kappa | −0.001 | −0.0105 ± 0.0048 |
| alpha_n / alpha_d | 0.509 / 1.006 (single seed) | 0.5076 ± 0.0006 / 1.0038 ± 0.0010 |

The decontaminated retrain's recall never reaches 11/12 in any seed — its range is 9–10 of 12.
The reported file matches the **contaminated** column on the one metric that cannot be
coincidental, and `step3_baselines.json` matches that column digit-for-digit on the 253-pair core
(f1 0.0894, mcc −0.007, pr_auc 0.0461, TP 11 / FN 1 / FP 223 / TN 18).

**Scope.** This propagates to AUROC 0.460 and its bootstrap CI, the top-15 check, the Spearman
correlations, the threshold sweep, the 0.5078 defect-free stratum, and all four rows of the
system comparison — every one is computed from that CSV.

### RESOLVED — it is the contaminated model, and the corrected numbers are recoverable

`accord_v2_decontam_gold_aggregate.json` holds the three decontaminated seeds' confusion
matrices on the 253-pair core:

| seed | TP | FN | FP | TN | recall |
|---|---|---|---|---|---|
| 0 | 9 | 3 | 216 | 25 | 0.750 |
| 1 | 10 | 2 | 210 | 31 | 0.833 |
| 2 | 9 | 3 | 207 | 34 | 0.750 |

The best decontaminated seed reaches **10/12**. The reported result is **11/12**. No
decontaminated seed can produce it. **`gold_standard_accord_results.csv` is the contaminated
configuration**, and Chapters 7, 8 and 9 report it.

**The corrected numbers.** `thesis/outputs/gold_eval_decontam_predictions.csv` on Drive carries
the per-pair continuous probabilities for all three seeds on the same 253 pairs, so the headline
AUROC does not need the checkpoints and can be recomputed now:

| | AUROC | Mann-Whitney p | top-15 TP | AP |
|---|---|---|---|---|
| seed 0 | 0.5494 | 0.565 | 1 | 0.0613 |
| seed 1 | 0.5090 | 0.918 | 0 | 0.0592 |
| seed 2 | 0.5048 | 0.956 | 0 | 0.0533 |
| **mean ± std** | **0.5211 ± 0.0201** | — | — | — |
| **3-seed ensemble** | **0.5145**, 95% CI **[0.321, 0.703]** | **0.867** | 1 | — |
| *contaminated, same 253 pairs* | *0.4595* | *(0.64 as reported)* | *0* | *0.0461* |

Plus, from the aggregate file: recall **0.7778 ± 0.0393**, precision **0.0424 ± 0.0023**,
PR-AUC **0.0579 ± 0.0034**, κ **−0.0105 ± 0.0048**, ECE **0.8015 ± 0.0226**.

**Correction.** Replace every Chapter 7/8/9 number derived from `gold_standard_accord_results.csv`
with the decontaminated equivalents above, and state that the earlier figures were the
contaminated configuration. **The headline AUROC becomes 0.52, not 0.46.**

**This makes the negative result cleaner, not weaker.** An AUROC *below* chance was always the
odd part of the old number; 0.5145 with a CI of [0.321, 0.703] and Mann-Whitney p = 0.867 is
simply no signal, which is what the thesis argues. It also now agrees with the independent
vanilla control (C19: 0.497 ± 0.035, DeLong p = 0.930) — two unrelated models, one flawed and one
deliberately plain, both at chance on the same pairs.

**One caveat carried forward, now settled — and it is worse than "not certified."** The
decontaminated run trained on the 1,679-row decontaminated pool, which spans **1,025 subjects**.
The 299 gold-evaluated pairs span **266 subjects**. **158 of those 266 — 59.4% — are in both.**
The decontaminated configuration is *not* subject-disjoint from its own evaluation set; row-level
removal of `pharmacist_overlap` rows left the majority of evaluated patients in training under
different `(drug, rule)` combinations.

So the corrected numbers in this section are the right ones to *report in place of* the
contaminated ones, but neither configuration is clean. The only subject-disjoint measurement in
the project is the control baseline in C19, which excludes the 291 overlapping rows and asserts
disjointness in code — and it lands at **0.497 ± 0.035**, indistinguishable from both.

That three-way agreement is the useful outcome: contaminated 0.4595, decontaminated 0.5145,
subject-disjoint control 0.497. Leakage at either level did not manufacture signal, because there
was none to manufacture.

**What this does not change.** The decontaminated run is *worse* on every metric, so the negative
conclusion survives and strengthens. C1's numbers are wrong in the same direction and for the
same reason: the symmetric ablation is **F1 0.7603 / acc 0.9091**, not 0.7775 / 0.9192, and the
authority weights **(0.509, 1.006)** are a contaminated single seed — the 3-seed decontaminated
values are **(0.5076 ± 0.0006, 1.0038 ± 0.0010)**, a 1.5% / 0.4% move rather than 1.8% / 0.6%.
Both corrections make the original claims *more* clearly artifacts, not less.

---

## C17. The volume adjustment controls the wrong note

**Claimed.** C12, and the 3-cohort replication built on it: rule-based mining under-selects
admissions with an unrecorded `LANGUAGE` field, adjusted OR 0.64, *"independent of how much was
written about them."*

**Actual.** The adjustment uses `log1p(total_note_chars)` — the sum of both notes. But the
discharge summary is **91.4%** of that total (median), the two lengths are nearly uncorrelated
(**r = 0.174** on the log scale), and the medians are **970** chars nursing against **10,643**
discharge. The mining rules require the drug to appear in **both** notes, so the binding
constraint is the *nursing* note. The paper's single covariate is therefore ~91% a measure of the
note that does not bind.

Refitting the identical model with the two lengths entered separately (recomputed directly,
n = 9,783):

| volume adjustment | MISSING OR [95% CI] | p |
|---|---|---|
| none | 0.415 [0.367, 0.470] | 1.0e-43 |
| **log(total) — the spec used** | **0.639 [0.558, 0.732]** | 1.0e-10 |
| log(nursing) only | 0.787 [0.685, 0.905] | 0.00074 |
| log(discharge) only | 0.527 [0.462, 0.602] | 3.2e-21 |
| **log(nursing) + log(discharge)** | **0.894 [0.772, 1.034]** | **0.13** |

The one specification reporting a strong effect is the one under-controlling the covariate the
rule actually depends on. Under the correct adjustment the interval includes 1.

**The stratified check confirms it from the other direction.** Stratifying on the *paper's*
variable, Breslow-Day rejects homogeneity (χ² = 50.19, p = 9.9 × 10⁻⁸) so the pooled CMH OR has
no referent. Stratifying on **nursing**-note deciles it is well behaved:

| stratifier | CMH common OR | p | Breslow-Day |
|---|---|---|---|
| total note chars | 0.551 | ~0 | **χ² 50.19, p = 9.9e-8 — rejected** |
| **nursing note chars** | **0.746** | **4.4e-5** | χ² 3.59, **p = 0.936 — homogeneous** |

**Correction.** Withdraw *"independent of how much was written."* What survives is real but
smaller and volume-entangled: **stratified on nursing-note length, common OR 0.746 (p = 4.4e-5),
homogeneous across all ten strata.** Report the full specification curve above rather than one
row, and state plainly that the effect is not separable from documentation volume once volume is
measured on the axis the rule depends on.

### The 3/3 replication is withdrawn. Re-run under the corrected specification it is 1/3.

Sepsis and AKI reproduced the finding under the *same* mis-specification. Re-run with the two
note lengths entered separately, cluster-robust on `SUBJECT_ID` (the re-run reproduces the
original "none" and "total" rows to 4 dp in all three cohorts, so this is the same pipeline):

| specification | HF (9,783) | Sepsis (4,339) | AKI (7,579) |
|---|---|---|---|
| none | 0.409 [0.360, 0.465] | 0.408 [0.321, 0.518] | 0.370 [0.315, 0.434] |
| log(total) — **the spec used** | 0.637 [0.557, 0.729] | 0.582 [0.455, 0.745] | 0.535 [0.452, 0.632] |
| log(nursing) only | 0.798 [0.691, 0.922] | 0.922 [0.697, 1.220] | 0.714 [0.596, 0.854] |
| log(discharge) only | 0.528 [0.462, 0.603] | 0.491 [0.384, 0.627] | 0.452 [0.383, 0.533] |
| **log(nursing) + log(discharge)** | **0.902 [0.777, 1.046]** p=0.17 | **1.005 [0.756, 1.336]** p=0.97 | **0.778 [0.647, 0.935]** p=0.0075 |

**Under the corrected regression the deficit replicates in 1 of 3 cohorts — AKI only.** HF and
sepsis collapse to null, sepsis to dead flat. The originally reported 0.64 / 0.58 / 0.53 was
carried by discharge-summary length, which the mining rule does not gate on.

**The stratified analysis disagrees, and the disagreement is the finding.** On *nursing* deciles,
Breslow-Day fails to reject homogeneity in all three cohorts, so pooled ORs are valid there:

| cohort | Breslow-Day | pooled CMH OR |
|---|---|---|
| HF | χ² 3.83, p = 0.92 | **0.743 [0.645, 0.855]**, p = 3.3e-5 |
| Sepsis | χ² 11.70, p = 0.23 | 0.864 [0.659, 1.133], p = 0.29 |
| AKI | χ² 3.19, p = 0.96 | **0.651 [0.547, 0.775]**, p = 1.2e-6 |

So: **2/3 conditioning on nursing length alone, 1/3 conditioning on both.** The entire difference
is whether discharge-summary length belongs in the adjustment set — and that is a causal
question, not a statistical one. If administrative thinness causes both the blank `LANGUAGE`
field and sparse documentation, discharge length is a **mediator** and conditioning on it is
adjusting away the mechanism. If it is an independent determinant of whether the rule can fire,
it is a confounder and belongs.

**Correction.** Withdraw "replicates 3/3." Report the specification curve, both analyses, and the
DAG the choice depends on. The honest summary is that the effect is somewhere in **OR 0.65–1.00
depending on cohort and specification**, is strongest in AKI, and is absent in sepsis under every
correct specification.

**Interactions.** `language × log(nursing chars)` reaches significance in no cohort (HF p = 0.26,
sepsis p = 0.082, AKI p = 0.086), so the earlier "the gap grows with note length" claim is also
not supported once the right volume variable is used.

**Clustering is a non-issue and can be stated as a robustness result.** Robust/naive SE ratios
are 0.97–1.03 across all specifications and all three cohorts.

**`DBSOURCE` remains open.** `ICUSTAYS.csv` is still not on this machine and was not downloaded.
The era confound is unclosed in all three cohorts.

**And the direction of modification undercuts the original reading.** The deficit is concentrated
where documentation is *heaviest* — crude OR below vs above median note length runs HF 1.16→2.85,
sepsis 1.64→2.36, AKI 1.56→3.28. That is the opposite of a story about under-documented patients
being missed, and it needs an explanation the thesis does not currently have.

---

## C18. Decontamination was row-level; 158 of 266 gold subjects were also in the training pool

**Claimed.** The weak-label pool was decontaminated against the 299 evaluation pairs by removing
overlapping rows (`removal_reason == 'pharmacist_overlap'`), and the decontaminated retrain is
described as the clean configuration.

**Actual.** Removing overlapping *rows* is not subject-level disjointness. Building the control
baseline surfaced that **158 of the 266 gold subjects also had rows in the 1,679-row training
pool — 291 rows in total**, under different `(drug, rule)` combinations within the same or other
admissions of the same patient. Those rows share the patient, and frequently the note text, with
evaluation pairs.

This is C8 recurring at a different layer. C8 was about the train/test split *within* the pool,
and `accord/phase3_mining.py` fixes that. The pool-versus-gold boundary was never checked at
subject level.

**Correction.** Any configuration described as decontaminated must assert `SUBJECT_ID`
disjointness against the gold set, not just row removal. `accord/control/data.py` does this and
`assert`s it rather than filtering silently. Whether ACCORD's own decontaminated retrain carries
this leakage is not established here and should be checked — it bears on C16.

---

## C19. The defect-free control result, and what it does and does not license

**The experiment.** A deliberately plain BioMedBERT cross-encoder —
`[CLS] nursing [SEP] discharge [SEP]` → CLS → one `nn.Linear` → BCE. No rule feature, no
authority parameter, no ATCG/PPC/GGHE, no harm head, no confidence weighting. Same weak labels,
same 299 gold pairs, subject-disjoint split, `set_seed()` inside `main()` (C7), fresh process per
seed.

| stratum | seed 42 | seed 123 | seed 2024 | mean ± std |
|---|---|---|---|---|
| all 299 (12 pos) | 0.459 | 0.503 | 0.528 | **0.497 ± 0.035** |
| defect-free rules 3+4 (6 pos) | 0.542 | 0.535 | 0.512 | **0.530 ± 0.016** |

Ensemble against ACCORD on the same pairs:

| | AUROC | bootstrap 95% CI | AP (base 4.0%) | top-15 TP |
|---|---|---|---|---|
| vanilla control | 0.469 | [0.288, 0.655] | 0.051 | 1 |
| ACCORD `contra_prob` | 0.460 | [0.288, 0.633] | 0.039 | 0 |

**DeLong, paired:** all-299 diff +0.009, z = 0.088, **p = 0.930**; defect-free diff +0.039,
z = 0.788, **p = 0.431**. No difference on either stratum.

**Floors.** Majority-class gives AUROC exactly 0.5 and AP exactly the base rate — the metric code
is correct at the degenerate case. **TF-IDF + logistic regression gives AUROC 0.576 / 0.573 and
AP 0.157 / 0.277 — higher than both neural models.** Reported as found; whether that is
small-*n* noise or genuine surface-form signal is not established.

**What this licenses.** A defect-free architecture, trained identically on the same split,
produces a result statistically indistinguishable from the flawed one. That is evidence *against*
"ACCORD's thirteen defects caused the chance-level result."

**What it does not license.** It is not evidence *for* "the task is unlearnable." The control's
CI upper bound is 0.655 (0.828 on the defect-free stratum), and the minimum detectable AUROC at
80% power is **0.739** for the full set and **0.839** for the defect-free stratum, which has only
6 positives. This evaluation cannot separate "genuinely near chance" from "a real but
sub-threshold effect." The entitled claim is: *no system tested, including a clean baseline, is
distinguishable from chance on this evaluation, and this evaluation is underpowered to establish
that the task is near chance.* Anything stronger repeats the error C15 identifies.

---

## C20. C10 repaired: the miscalibration is model-agnostic, the differential is not

C10 recorded that *"clinical miscalibration is model-agnostic"* was printed from a results dict
containing a **hardcoded** Llama row, because Llama-3-8B never loaded (gated repo, 401) and only
Mistral-7B ran — and that `llm_preds.append(pred if pred else pk)` scored every unparseable
generation as a positive. The claim was retracted. It has now been run for real.

**Method.** The original prompt was recovered verbatim from `prompts_existing.csv`
(`config_id = llama_baseline`) and reused unchanged. All **1,679** pool rows, three local models
via Ollama, temperature 0. Unparseable generations are recorded as their own `UNPARSEABLE`
category and never defaulted to positive — the C10 bug class. Same regression specification as
the original, including `C(rule_cat)`, so the comparison is like-for-like.

| model | positive rate | vs 4.0% gold | unparseable | MISSING-language OR [95% CI] | p |
|---|---|---|---|---|---|
| Llama-3-8B (original, HF API — reference) | 0.8487 | **21.2×** | — | 1.686 [1.108, 2.567] | 0.015 |
| `llama3:8b` (local) | 0.9470 | **23.6×** | 0.0% | **0.741** [0.426, 1.287] | 0.29 |
| `granite3.3:8b` | 0.7219 | **18.0×** | 0.0% | 0.887 [0.667, 1.180] | 0.41 |
| `qwen3:8b` | 0.6349 | **15.8×** | 0.0% | 1.031 [0.785, 1.355] | 0.82 |

**(i) Gross miscalibration replicates 3/3.** Every model over-calls contradictions by 15.8× to
23.6× against the 4.0% pharmacist base rate. The unparseable rate is 0.0% in all three, so none
of this is the C10 default-to-positive artifact. **The model-agnostic claim is now true and
earned** — it can be reinstated, citing this run rather than the hardcoded dict.

**(ii) The demographic differential replicates 0/3.** No model reaches significance, and the
local `llama3:8b` points the *opposite* way (0.741) from the original 1.686. This holds under the
*same* specification that produced 1.69, so the effect is not even a specification artifact that
generalizes — it is specific to the original endpoint's outputs.

**Correction.** Reinstate the miscalibration claim as measured across three open-weights models.
Report the differential as model-specific and not replicated. This is the third independent line
against it, after C14 (the crude association runs the other way, 0.82) and the marginal refit
(1.42 [0.96, 2.10], p = 0.081).

**What survives is the useful half.** A 15.8–23.6× over-call rate reproduced across three
independently trained model families says the failure is a property of the *task* — asking an
instruction-tuned model to adjudicate a low-prevalence clinical judgment from two sentences —
rather than of any one model or endpoint.

---

## C21. The denominator framing survives its falsifier — and my earlier "already published" call was too broad

**What I said earlier.** After a novelty sweep I reported that the denominator argument was
*"published, direct hit"* — Pfohl et al., NeurIPS 2025 (arXiv:2506.04193), which states that under
selection bias, disaggregated evaluation *"may be invalid without explicit assumptions regarding
the bias mechanism."* That is correct about the **theory** and I over-generalized it to the whole
claim.

**The falsifier.** A systematic coding of published clinical NLP / clinical ML fairness
evaluations was run specifically to kill the framing: if the field routinely audits
candidate generation against a source-population denominator, this thesis is a case study, not a
critique. Of 31 papers read at methods level, 29 have a filtering or candidate-generation stage:

| | yes | partial | no |
|---|---|---|---|
| reports subgroup composition **before** the filter | 3 | 2 | **24** |
| audit denominator is the **source population** | 4 | — | **25** |
| states how **missing** stratifying attributes are handled | 12 | 5 | **13** |
| reports any **per-stage** decomposition | 3 | 1 | **27** |

**The decisive cross-tab:** of the 25 papers that audit a downstream model on a filtered pool,
**0 also audit the filtering stage**. Of the 4 that audit selection against a population
denominator, **0 audit a downstream model**. The two activities are disjoint literatures.

**Correction to the claim's wording.** Do *not* write "the field never audits selection." Write:
*the field audits selection and audits models, but in disjoint literatures — cohort
representativeness work never carries through to a model audit, and model-audit work never looks
upstream.* The unclaimed contribution is the **dissociation**: a selection stage failing the
four-fifths rule at 0.468 while the pool audit passes on demographic parity (0.034), FPR gap
(0.024) and four-fifths (0.961).

**Prior work that must be cited and positioned against, not discovered by a reviewer:**
- **Acitores Cortina et al., JMIR Med Inform 2025;13:e67591** — 19 data-quality filters over ~12M
  patients across 4 databases; filters differentially remove minoritized groups; recommends
  reporting group sizes before *and* after filtering. Evaluates no downstream model and shows no
  dissociation. Its future-work section asks for this experiment.
- **Qi et al., JAMIA Open 2021;4(3):ooab077** — reframes RCT enrollment as a classifier and
  applies fairness metrics to the *selection decision* against NHANES. This pre-empts the
  **method**. It must be cited explicitly; a reviewer who knows it will ask.
- **Haghighathoseini et al., AMIA Annu Symp Proc 2024:473–482** — 16 cohort variants on N3C,
  composition shifts by sex/race/ethnicity; does not compare fairness metrics across variants.

**The one high-risk unknown is closed.** Huang et al., arXiv:2309.01935, *"The impact of EHR data
continuity on prediction model fairness and racial-ethnic disparities"* — flagged as the paper
most likely to pre-empt this framing and available only as an abstract. Retrieved: it stratifies
cohorts by an EHR data-continuity measure (MPEC) and reports that fairness varies across them. It
does **not** compare cohort composition to a source population, does not use standard subgroup
fairness metrics in the audit sense, and makes no denominator argument. **It was withdrawn by its
authors on 2026-02-26 pending substantial revision**, so it should not be cited as prior work at
all.

**A second, independent failure mode worth reporting.** 13 of 31 papers say nothing about how
records with a missing stratifying attribute are handled, and those that do state it
overwhelmingly say they **dropped** them — deleting exactly the population most likely to have
been differentially filtered upstream.

**Method limitation, stated because it bounds the claim.** The sample is purposive, not random:
it oversamples pipelines that *can* exhibit the failure. The counts answer "among papers where
this failure is possible, how many check?" — not "what fraction of the field checks?" Coverage of
ML4H, CHIL and Lancet Digital Health is thin, and 6 papers were inaccessible plus 3 abstract-only,
all excluded from the counts.

---

## C22. The pool audit "passes" partly because the audited classifier is near-constant

**Claimed.** The dissociation's real-data leg: candidate generation fails the four-fifths rule at
0.468 while the standard subgroup audit on the resulting pool passes everything — demographic
parity gap 0.034, FPR gap 0.024, precision gap 0.006, four-fifths 0.961.

**Actual.** The audited classifier predicts positive on **261 of 299 pairs — 87.3%** — against an
expert base rate of 4.0%. Per group (`exp1a_gold_error_rates_by_group.json`):

| group | n | predicted positive | rate | FPR |
|---|---|---|---|---|
| ENGLISH | 191 | 168 | 0.880 | 0.874 |
| MISSING | 84 | 71 | 0.845 | 0.850 |
| RECORDED NON-ENGLISH | 24 | 22 | 0.917 | 0.917 |

A classifier that calls positive ~87% of the time is close to constant, and a near-constant
classifier satisfies demographic parity almost by construction — the maximum spread available
across these groups is 0.072. So the audit did not *fail to detect* a disparity it could have
detected; it was applied to a model with little room to exhibit one.

**Correction.** The real-data leg cannot be reported as *"the audit was blind to the bias."* The
defensible version is: *"the audit was uninformative here, for two compounding reasons — the
denominator had already been filtered, and the audited model was near-degenerate."* The
degeneracy must be stated on the same page as the 0.961.

**Which leg is load-bearing.** The synthetic sweep, not the cohort. In simulation the downstream
classifier has AUROC 0.895 — genuinely non-degenerate — and the audit *still* returns a parity
gap of 0.0013 while the selection ratio is 0.063. That is the clean demonstration; the MIMIC
cohort is a worked example whose classifier happens to be degenerate. The paper's honest
structure is therefore: an analytic point, a simulation that demonstrates it, a survey showing
nobody checks, and a clinical instance — not *"demonstrated on real data"* with the simulation as
support.

**What would repair the real-data leg.** Run the pool audit against a *non-degenerate* downstream
classifier on the same cohort — the subject-disjoint control from C19 emits continuous scores and
could be thresholded to a sensible operating point. If the dissociation survives with a
non-degenerate model on real data, that leg becomes a strength rather than a liability.

**Note this is independent of the detector's quality.** The dissociation is a property of Stage 1
versus the pool audit and would hold if the downstream detector were perfect. The degeneracy
problem is specifically about whether *this* audit had the power to reject, not about whether the
argument is right.

---

## C23. "Model-agnostic miscalibration" is refuted by a file already in the results directory

**Claimed, by me, in C20.** Three 8B open-weights models over-call at 15.8×–23.6× against the
4.0% base rate, so *"the model-agnostic claim is now true and earned."*

**Actual.** `results/accord/step3_baselines.json`, present throughout:

| system | positives | n | rate | × vs 4.0% |
|---|---|---|---|---|
| llama_baseline | 242 | 253 | 0.957 | **23.9×** |
| **gpt4o_zeroshot** | 71 | 253 | 0.281 | **7.0×** |
| gpt4o_fewshot | 141 | 253 | 0.557 | 13.9× |

The one frontier model in the data over-calls at **one third** the rate of the 8B models, and
few-shot prompting moves the same model from 7.0× to 13.9×. Three 8B instruction-tuned models
under one fixed prompt is a single point in a scale × prompt space, and **both axes move the
quantity claimed to be model-invariant.**

**Correction.** Withdraw "model-agnostic." The defensible claim is: *every model tested over-calls
by at least 7×, and the magnitude varies with scale (7.0× to 23.9×) and with prompting (7.0× to
13.9× within one model).* That is still a strong result — a 7× floor across everything tested —
and it is the one the data supports.

**This is C10's failure mode repeating in the correction that repaired C10.** C10 was a
model-agnostic claim asserted from one model plus a hardcoded row. C20 replaced it with a
model-agnostic claim from three models of one size under one prompt, while a contradicting
measurement sat in the same results directory. The lesson is the same both times: "agnostic"
requires varying the thing you claim not to depend on.

**Second, separate problem.** Positive rate at the argmax is a *decision rate*, not calibration. A
calibration claim needs the yes-token logprob, a reliability diagram, ECE, and AUROC against
gold. Chapter 8 prescribes exactly this and the claim violates it.

**Third — the reference row must not appear in any paper.** The C20 table retains
"Llama-3-8B-Instruct (original, HF Inference API)" for comparison. It belongs *here*, because this
document's job is to record what was originally claimed and why it was wrong. It must not appear
in a submitted table, and no number traceable to it may be reported as a result: that run sent
MIMIC-III note text to a third-party hosted inference endpoint. The C20 re-run is local Ollama and
is the only version that should be cited going forward.

---

## C24. Both empirical legs of the lead claim are compromised

**Leg 1 — the real-data dissociation rests on the one contrast that is not a protected attribute.**
Four-fifths ratios at the selection stage (`exp3_disaggregation.json`):

| contrast | ratio | verdict |
|---|---|---|
| NAMED_MINORITY vs WHITE | 0.1647 / 0.1564 = **1.053** | **passes** |
| RECORDED_NON_ENGLISH vs ENGLISH | 0.1673 / 0.2081 = **0.804** | **passes** |
| UNKNOWN_OTHER vs WHITE | 0.1132 / 0.1564 = 0.724 | fails |
| **MISSING vs ENGLISH** | 0.0973 / 0.2081 = **0.468** | fails |

**On every attribute that is actually a protected characteristic, candidate generation passes.**
The only two failures are the two administrative-missingness buckets. C11 is a formal retraction
of exactly this inference — administrative missingness read as demography — and the lead claim
commits it again, then applies a disparate-impact standard designed for protected classes to a
category covering 46.6% of the cohort. Combined with C22 (the audited classifier is near-constant)
this leg cannot carry the claim.

**Leg 2 — the synthetic sweep has no positive control.** From
`accord/denominator/results/sweep1_summary.csv`:

| or_select | demographic-parity gap |
|---|---|
| 0.05 (20× under-selection) | 0.001287 |
| 0.64 (the observed value) | 0.000755 |
| **1.00 (no selection bias at all)** | **0.001116** |
| 2.00 | 0.001068 |

The null condition and the most extreme condition produce indistinguishable audit output. The
sweep cannot separate *"the audit is blind to selection bias"* from *"the generative model
contains no disparity for the audit to find."* `simulate.py` draws group, features and outcome
mutually independently, so with the leak parameter at zero every audit metric is forced to zero
gap by construction. The sweep illustrates its own assumption.

**Correction.** Neither leg demonstrates the claim as stated. **The 31-paper survey is the only
leg still standing**, and two independent adversarial reviews reached that conclusion separately.
Note the survey's purposive sampling cuts the *right* way here: deliberately oversampling
pipelines most exposed to the failure and still finding 0 of 25 makes the disjointness conclusion
conservative, not inflated.

**What would repair each leg.**
- Leg 1: sweep the pool audit's operating point across the three available continuous scorers
  (ACCORD `contra_prob`, the C19 control, TF-IDF+LR) from a 4% positive rate up to 90%, and report
  the audit verdict as a function of that rate. If the audit passes across the whole sweep while
  selection sits at 0.468, the claim holds. If it passes only in the degenerate region, it does not.
- Leg 2: make the group associated with features and outcome — which C12 asserts is true of these
  admissions — and make selection depend on documentation volume, which C17 shows it does. Then
  the sweep tests something.
- Better than either: state the invariance as a proposition with a short proof, and demote the
  sweep to numerical illustration of a theorem rather than evidence for a conjecture.

---

## C25. The era confound is closed without `ICUSTAYS.csv`, and it largely explains the effect

**What the record said.** C17 and the three-cohort re-run both state that `DBSOURCE` — the
CareVue/MetaVision era marker, and the only usable era proxy since `ADMITTIME` is date-shifted —
is unrecoverable because `ICUSTAYS.csv` is not on this machine. That was wrong.

**The proxy that is on disk.** MIMIC-III splits nursing documentation across two `NOTEEVENTS`
categories on the same boundary: `Nursing/other` (CareVue, 822,497 notes) and `Nursing`
(MetaVision, 223,556). An admission-level era label is a groupby on a file already local.

**The confounding is near-total.** For the 9,783 HF note-pair admissions (7,485 CareVue,
2,271 MetaVision, 27 both):

| era | ENGLISH | **MISSING** | OTHER | candidate yield |
|---|---|---|---|---|
| CareVue | 0.295 | **0.608** | 0.097 | **0.1093** |
| MetaVision | 0.882 | **0.005** | 0.114 | **0.2924** |

`LANGUAGE` was recorded for almost no one in CareVue and almost everyone in MetaVision — 60.8%
missing versus 0.5%. And candidate yield differs by era by a factor of **2.7×**. So
"unrecorded `LANGUAGE`" is very close to a relabelling of "CareVue era," and the two cannot be
cleanly separated in this data.

**The specification curve with era added:**

| adjustment | MISSING OR [95% CI] | p |
|---|---|---|
| raw | 0.415 [0.367, 0.470] | 1.0e-43 |
| + era | 0.723 [0.620, 0.843] | 3.4e-05 |
| + nursing + discharge length | 0.894 [0.772, 1.034] | 0.13 |
| **+ nursing + discharge + era** | **0.859 [0.730, 1.010]** | **0.066** |

**Correction.** The documentation-completeness selection effect does not survive joint adjustment
for note volume and documentation era. Combined with C17 (null in HF and sepsis under the correct
volume specification) and C24 (four-fifths passes on both actual protected attributes), the
clinical selection finding is **not a defensible standalone result** and must not be reported as
one. What remains reportable is the raw composition shift — 46.6% of the cohort to 29.8% of the
candidate pool — stated as a fact about the pipeline with its mechanism openly attributed to
documentation volume and EHR-system era rather than to patient language.

**Why this does not damage the lead claim, if the lead claim is stated correctly.** The audit
argument is about *detectability*, not about injustice: a per-group audit computed on the
post-selection pool is invariant to the selection rate, **regardless of whether the removal is
discriminatory, benign, or purely mechanical**. Under that statement the four-fifths ratio of
0.468 is the right computation — four-fifths is defined on raw selection rates and deliberately
does not ask about cause — and this correction becomes supporting evidence rather than a threat:
the mechanism is mundane, and blindness to *mechanical* filtering is a far more general problem
than blindness to discriminatory filtering, because mechanical filtering is in essentially every
pipeline the survey covers.

The lead claim must therefore never be phrased as "the audit missed a fairness harm." There is no
demonstrated harm here. It is "the audit cannot see upstream removal at all."

- **299 two-pass pharmacist annotations** with confidence, justification and 1–5 danger ratings
- **Phase 1 cohort construction** — 13,597 HF admissions, the 3-role → 2-role retreat documented
  with coverage at each step, plus sepsis (4,339) and AKI (7,579)
- **MedNLI 83.9% / 83.9% macro F1** full fine-tune vs **72.7%** LoRA r=16 — correctly run, inside
  published SOTA, provided it is labelled as BioMedBERT and not as ACCORD
- The **negative result itself**, which is measured four independent ways and is the thesis

---

## C26. Novelty reassessed: the sweep is pre-empted, and four-fifths on a missingness flag is attackable

**C21 recorded that the denominator framing survived its falsifier.** The *survey* leg did. A
second novelty pass then found prior art that C21's survey was not built to catch, and it removes
two of the lead claim's four legs.

**1. The synthetic sweep is a denser version of an experiment Pfohl et al. already ran.** C21 and
my own summary both described Pfohl et al. (NeurIPS 2025, arXiv:2506.04193) as theory-only. From
the PDF, Section 5: *"We conduct a small experiment to verify a subset of the properties expected
under selection… three selection mechanisms corresponding to cases where the selection mechanism
depends on X, Y, or {Y,A},"* plus simulations over their Table 1 DAGs and real-data experiments on
ACS PUMS. Our sweep's headline sub-finding — that blindness *"breaks down only when selection
depends directly on the outcome"* — **is their S←Y and S←{Y,A} condition**, and before that it is
Bareinboim–Pearl recoverability. With C24 (the sweep has no positive control), the simulation leg
must be **demoted to replication, not presented as evidence**.

**2. Stage-level fairness measurement across a pipeline was published empirically in 2021 at
n=37 pipelines.** Biswas & Rajan, *"Fair Preprocessing: Towards Understanding Compositional
Fairness of Data Transformers in Machine Learning Pipeline"* (arXiv:2106.06054): *"we conducted a
detailed fairness evaluation of the preprocessing stages in 37 pipelines… we showed how the local
fairness of a preprocessing stage composes in the global fairness of the pipeline."* That is
"measure fairness at the stage, not just the model," five years earlier. And Black et al.,
*"Toward Operationalizing Pipeline-aware ML Fairness"* (arXiv:2309.17337), establishes
pipeline-aware fairness as a **named agenda with a published roadmap** — so no sentence of the
form "nobody audits the pipeline rather than the model" is available.

**3. The four-fifths choice is now a citable attack.** Watkins, McKenna & Chen, *"The four-fifths
rule is not disparate impact: a woeful tale of epistemic trespassing in algorithmic fairness"*
(arXiv:2202.09519). The lead number applies a US employment-law construct, stripped of its
protected-class and practical-significance scaffolding, to a **data-completeness indicator** —
and C24 already showed four-fifths passes on both actual protected attributes. Defend the
extension explicitly in one paragraph, or switch to a metric without legal baggage. Unanswered,
this is fatal to the headline.

**4. Adjacent demonstrations of the same shape exist.** Sariola et al., *"The Illusion of
Fairness"* (arXiv:2507.02152): an intervention *"appears to achieve parity using traditional
measures, but in fact has roughly 10% disparity when measured appropriately."* And in
recommendation retrieval the point is assumed background — FairSync (arXiv:2402.10628):
*"without a high-quality set of candidates, the stage-2 ranker cannot ensure the required exposure
of groups."*

**5. "Model-agnostic" fails on a second, independent ground.** Beyond C23 (GPT-4o 7.0× vs 8B
23.9×), the *direction* of LLM annotator bias is unstable across the literature: Kotte
(arXiv:2606.12426) and MultiSoc-4D (arXiv:2605.06940) find annotators **under**-calling and
collapsing to fallback labels; Zhang et al. (arXiv:2605.16386) find compression toward the scale
midpoint across three model families. Three models sharing one prompt cannot separate
model-invariance from prompt-specificity. The reframe that survives is *"the direction of LLM
weak-label bias is set by the prompt, not the model"* — a better paper, requiring prompt variation
not yet run.

**6. Claim 4 (transfer collapse as a leakage diagnostic) should be retracted outright.**
Cross-domain collapse as a shortcut/leakage tell is textbook (Kaufman/Rosset/Perlich on leakage;
Geirhos et al. on shortcut learning — verify both before citing). Diagnosing one's own
input-equals-target bug is a reproducibility note.

**7. The null-attribute finding is the cleanest surviving delta, with two citations to engage.**
Martínez-Plumed et al., *"Fairness and Missing Values"* (arXiv:1905.12728) already shows rows with
missing values have *different* fairness characteristics than complete rows — the mechanism that
makes routine deletion indefensible, seven years earlier. Cite it as the mechanism and position
the survey as documenting that the field ignored it. Bhatti et al. (arXiv:2503.07313) reports
*"listwise deletion gives the highest fairness on average"* — a direct counter that must be met
head-on.

**The only defensible construction.** No single claim carries a thesis. Fuse the dissociation with
the null-attribute finding and the survey: *the audit runs on a pool built by a filtering stage
that itself fails the audit, and the audit's standard practice of dropping null-attribute records
deletes precisely the population that filter differentially removed* — 0/25 papers audit both
stages, 13/31 do not report null handling. That loop is in neither Pfohl (theory + census), nor
Qi (selection only), nor Acitores Cortina (filter harm only), nor Biswas & Rajan (tabular
transformers, no missing-attribute dimension).

**Coverage limitation, and it matters.** This pass ran on arXiv and OpenReview only — Semantic
Scholar rate-limited, **PubMed and the ACL Anthology not searched**. Claims 1, 3 and 5 all live
partly in PubMed-indexed venues, and a JAMIA or JBI reporting-practice review would kill the
null-attribute finding outright and would not have been visible here. **Run that pass before
committing to the framing.**

---

## C27. The survey's reliability holds under blind second coding

**The concern.** The 31-paper survey was single-coder, purposive, with no reliability estimate,
and it is now the only leg of the lead claim still standing. C3 in this same record retracts a
κ = 1.000 claim for exactly this class of error, so the survey had to be held to the standard it
was failing to find in others.

**The test.** A second coder, blind to the first coder's answers and to the hypothesis, was given
only the field definitions and 10 of the 31 papers. It retrieved **all 10 in full** — zero
inaccessible — and returned 21 verbatim supporting quotes with section attributions.

**Cohen's κ on the decisive field** ("does the paper audit the filtering stage against a pre-filter
or external denominator"), bracketing the first coder's *Partial* codes both ways:

| mapping | agreement | κ | disagreement |
|---|---|---|---|
| Partial = YES vs literal reading | 9/10 | **0.783** | Wang |
| Partial = NO vs strict reading | **10/10** | **1.000** | none |
| Partial = YES vs strict reading | 9/10 | **0.737** | Banda |

**The decisive cell is empty.** Papers auditing *both* a downstream model and the filtering stage:
**0 of 10** under the strict reading, 2 of 10 under a literal one — and the second coder flagged
both of those (Banda, Wang) as weak on its own initiative, noting Wang's before/after comparison
serves confounder balance in a matched design rather than equity, and never reports what fraction
of each racial group survived any filter.

**Independent structural confirmation, which the second coder volunteered.** The two papers that
audit filtering as their primary object (Acitores Cortina, Qi) **neither audits a downstream
model** — Acitores explicitly defers it to future work, Qi has no model at all. And all six papers
that audit a model on a filtered pool apply substantive filters, none of the unambiguous ones
examining the filter's compositional effect. That is the disjointness claim reproduced by a coder
who was not told it was the hypothesis.

**Verdict.** κ of 0.74–1.00 clears the ≥0.75 bar under two of three mappings and is borderline on
the third. The survey is real. **The count in the decisive cell is fragile to two judgment calls
and must be adjudicated explicitly in writing rather than reported as a bare number** — state the
strict and literal readings side by side, publish the per-paper coding table with quotes, and let
the reader see the two ambiguous cases.

**This is the only leg of the lead claim that has strengthened under adversarial review.** C24
compromised the real-data leg, C26 demoted the simulation to replication. The survey survived a
falsifier designed to kill it (C21) and now a blind reliability check. The thesis should be built
on it.

---

## C28. The real-data leg is dead: the audit is not blind, it fails almost everywhere

**The test.** C22 showed the audited classifier is near-constant, so "the audit passes" might be
degeneracy rather than blindness. The way to settle it is to sweep the operating point: threshold
each available continuous scorer so the predicted-positive rate runs from the 4.0% base rate up to
89%, and recompute the full audit at every point with bootstrap CIs (2,000 resamples, seed 42,
tolerance 0.05).

**Result — the audit detects large disparities at nearly every operating point.**

| scorer | operating points passing **all** metrics |
|---|---|
| ACCORD `contra_prob` | **2 of 35** (5.7%) |
| subject-disjoint control (C19) | **9 of 35** (25.7%) |
| TF-IDF + LR | **0 of 35** (0%) |

Across the ACCORD sweep the demographic-parity gap ranges **0.006 to 0.298** and the four-fifths
ratio **0.321 to 0.884**. At the top of the sweep (achieved rate 0.890) the DP gap is **0.120**,
not the 0.034 that was reported.

**Correction.** *"The pool audit passes every well-powered metric"* is false as a general
statement. It was true at one operating point and is false at 33 of 35. The audit is **not blind
to disparity in the pool** — it reports substantial gaps almost everywhere. **The real-data leg of
the audit-dissociation claim is withdrawn.**

What this does *not* touch: the audit measures disparity in the *model's behaviour on the pool*,
which is a different quantity from the *selection rate into the pool*. The conceptual argument —
that a pool-level audit has no mechanism to report upstream removal — is unaffected. But it can no
longer be illustrated with this cohort, and **the survey is the only remaining empirical leg**.

**Two traceability problems surfaced by the same run.**

1. **`accord_pred` is not a threshold of `accord_contra_prob`.** The best single threshold (0.3563)
   reproduces the binary predictions at only **0.970** accuracy, and a top-*k* selection by score
   agrees at 0.967. The released probability and the released decision do not come from the same
   decision rule. This is consistent with C16 — the two columns appear to originate from different
   runs.
2. **The predicted-positive count does not reconcile.** The raw file has `accord_pred` =
   `contradiction` on **278 of 299** rows (93.0%), while `exp1a_gold_error_rates_by_group.json`
   implies **261** (87.3%) — a **17-row** discrepancy. The column has three values
   (`contradiction` 278, `ambiguous` 15, `not_contradiction` 6), so the handling of `ambiguous`
   differs between the released file and the analysis that used it. Resolve which is correct
   before either number appears anywhere.

Note both readings are degenerate — 87.3% or 93.0% against a 4.0% base rate — so C22's conclusion
is unaffected and is if anything strengthened.

**Consequence for the paper.** Paper A leads with the survey and the conceptual argument. The MIMIC
cohort appears as a worked example of *why the audit is uninformative here* — degenerate classifier,
underpowered cells, and an operating point that flatters the result — not as a demonstration that
audits are blind. That is a smaller claim and it is the one the data supports.

---

## C29. The invariance proposition is verified numerically, and the metric taxonomy is the finding

**Method.** `accord/theory/selection_invariance.py` — fully discrete, so every quantity is **exact
with no Monte Carlo error**. Critically, every metric is computed by one function applied to a
joint table over (A, X, Y): "population" is that function on the population table, "pool" is the
*same* function on the selection-reweighted table. Nothing about the pool is computed from the
closed form being tested, so the check is genuine rather than tautological.

**The identity holds.** The covariance decomposition
`FPR^pool − FPR^pop = Cov(f(X), π(X,a) | Y=0, A=a) / E[π(X,a) | Y=0, A=a]` verifies at
**5.9 × 10⁻¹⁶**; the odds form at 1.3 × 10⁻¹⁵; a 19-point sweep tracks the closed form with
**R² = 1.000**, max error 2.5 × 10⁻¹⁶; and a 20,000,000-draw Monte Carlo agrees at max z = 2.09.

**The taxonomy is the real result.** Under selection depending on A and Y but **not** X
(S ⊥ X | A):

| metric | pool vs population gap | invariant? |
|---|---|---|
| **FPR** | −0.09700 vs −0.09700 (diff 5.6e-17) | **yes** |
| **TPR** | −0.04035 vs −0.04035 (diff 0.0) | **yes** |
| **AUROC** | +0.01364 vs +0.01364 (diff 0.0) | **yes** |
| demographic parity | −0.0707 vs +0.1456 (diff 0.216) | no |
| PPV | +0.0765 vs +0.6489 (diff 0.572) | no |
| NPV | −0.0068 vs −0.5966 (diff 0.590) | no |
| prevalence | +0.0153 vs +0.6506 (diff 0.635) | no |
| calibration | — (maxdiff 0.628) | no |

Under selection that depends on the **score**, everything breaks except calibration
(maxdiff 1.1 × 10⁻¹⁶).

**What this means, stated carefully.** *Invariant* means the pool gap equals the population gap —
the audit reports the truth about the model but is **structurally unable to register that
selection occurred**. *Not invariant* means the pool gap differs from the population gap — the
audit reports a **corrupted** number, not a detection. So equalized-odds metrics and AUROC are
unbiased-but-blind; parity, predictive-value and calibration metrics are actively contaminated by
selection. **Neither class detects upstream removal**, and that two-way split is a sharper claim
than "audits are blind."

**One further result.** The bias is **not monotone** in the strength of X-dependence: it peaks at
λ = 0.95 and is smaller at λ = 1.0. So "more selection on features ⇒ more bias" is false, which
matters for anyone trying to bound the error.

**Kill condition still OPEN.** Whether this is a corollary of Bareinboim–Pearl recoverability or of
Pfohl et al. (NeurIPS 2025) has **not** been determined — the agent checking it died before
finishing, and its log mentions Pfohl only once. Until that check is done, this must be described
as a verified proposition, **not** as a novel theoretical contribution.

---

## C30. The rebuilt simulator can now fail, and its controls fire correctly

`accord/denominator/simulate_v2.py`, replacing the instrument C24 showed was degenerate.

| control | result |
|---|---|
| **positive** — plant a population gap, no selection | DP gap **0.1389** [0.1384, 0.1394], four-fifths **0.557** → **detected** |
| **negative** — no gap, no selection | DP gap **0.00031** [−0.0002, 0.0008] |
| **noise floor** | DP upper-95 **0.00165**, FPR upper-95 **0.00151** |

The original simulator could not distinguish its null condition from a 20× selection disparity
(0.00112 vs 0.00129). The rebuild detects a planted gap 84× above its own noise floor and stays
quiet when there is nothing to find. Reported as CIs over replicates, not "3 seeds" — C7's lesson.

---

## C31. The invariance proposition is a textbook consequence. Do not claim it as theory.

**The kill condition in C29 has fired.** I derived and checked this directly rather than delegating
it, and the honest answer is that the result is elementary.

**The one-line derivation.** In the discrete case,

```
FPR^pool = SUM_x f(x) P(x | Y=0, A) pi(x,a)  /  SUM_x P(x | Y=0, A) pi(x,a)
```

If `pi(x,a) = pi(a)`, it factors out of numerator and denominator and the expression **is**
`FPR^pop`. That is not a theorem about fairness auditing; it is the statement that conditioning on
S leaves `P(X | Y=0, A)` unchanged when **S ⊥ X | (A, Y)** — the definition of conditional
independence. Verified numerically to 10 decimal places at selection rates from 0.9 down to 0.01,
with the covariance form reproducing the observed pool-minus-population difference exactly
(+0.0000676432 both ways).

**The covariance form is also standard.** "Bias equals a covariance between the outcome functional
and the selection probability" is the ordinary first-order quantification used throughout
selection-bias sensitivity analysis. It is a useful way to write it. It is not new.

**The epidemiology precedent, verified.** Hernán, Hernández-Díaz & Robins, *"A structural approach
to selection bias,"* **Epidemiology 2004;15(5):615–25** (PMID 15308962). Abstract retrieved: the
paper argues that the causal structure underlying selection bias across case-control and cohort
settings is *"essentially the same: conditioning on a common effect of 2 variables, one of which is
either exposure or a cause of exposure and the other is either the outcome or a cause of the
outcome."* Our condition is the complement of that structure — when selection is **not** a common
effect of the score and the outcome-stratum, there is no bias. Epidemiology has had this framework
for two decades.

**What actually remains, and it is small but real:**
1. **The metric taxonomy.** Which fairness metrics are invariant and which are corrupted is
   mechanical once stated — anything that is a functional of `P(f(X) | Y, A)` alone is invariant
   (FPR, TPR, AUROC); anything involving prevalence or `P(Y | ·)` is not (DP, PPV, NPV,
   calibration), because selection changes the outcome marginal within group. Mechanical is not the
   same as *written down*, and I have not found it tabulated for fairness auditing.
2. **The non-monotonicity** (bias peaks at λ = 0.95, smaller at λ = 1.0) — the only piece I could
   not place in prior work, and the only one that is non-obvious.

**Correction.** **C3 is not a theoretical contribution and must not be presented as one.** State
the invariance as a known consequence, cite Hernán/Hernández-Díaz/Robins 2004 and the
Bareinboim–Pearl recoverability line, and use it to *justify the protocol* — which is exactly what
C29's own kill condition said to do if this outcome occurred. The dissertation drops from four
contributions to three: C1 (audit blindness, survey-led), C2 (the protocol), C4 (proxy-metric
decoupling). The taxonomy becomes a table inside C2, not a chapter.

**This is the right outcome to have found in an afternoon rather than in month four.** The
verification work is not wasted — it is what licenses the protocol and it is now checked to machine
precision.

---

## C32. The C4 headline comparison is on a dataset that no longer exists

**Discovered while adding seeds to the PubMedBERT comparator.** `local_only/task/*.jsonl` were
**rebuilt on Aug 4 at 16:54**. Every run in `results/task_runs.json` predates that rebuild:

| run | written | n_train | AUROC | dataset |
|---|---|---|---|---|
| concat_s0 (PubMedBERT) | Aug 4 11:02 | **25,527** | 0.8223 | **old** |
| ours_blockpre_s0/s1/s2 | Aug 4 12:40–15:23 | not recorded | 0.8014 / 0.7865 / 0.7988 | **old** |
| ours_vanillapre_s0/s1/s2 | Aug 4 12:56–15:34 | not recorded | 0.7937 / 0.7899 / 0.8020 | **old** |
| ours_randinit_s0/s1/s2 | Aug 4 13:13–15:45 | not recorded | 0.7998 / 0.7974 / 0.8037 | **old** |
| **concat_s1** | **Aug 7 23:21** | **23,846** | **0.7811** | **current** |

Current `local_only/task/meta.json` gives train n = 23,846, matching `concat_s1` and **not**
`concat_s0`.

**Two consequences.**

1. **The new seed is not comparable to the old one.** `concat_s0` = 0.8223 and `concat_s1` = 0.7811
   differ by 0.041, but they were trained on different datasets with different validation
   prevalences (0.1025 vs 0.0925). **That gap is a dataset change, not seed variance**, and it
   cannot be used to estimate either.
2. **The existing four-arm comparison is internally consistent but no longer reproducible.** All
   twelve pre-rebuild runs used the same old data, so *"10.6M random-init at 0.8003 ± 0.0031 versus
   PubMedBERT at 0.8223"* is a valid comparison — on a dataset that has since been overwritten.

**Why this was not caught earlier.** The `ours_*` runs record `n_train` as 0, so the dataset
fingerprint that exposed the mismatch is missing from exactly the arms that carry the claim. The
lineage machinery in `src/data/lineage.py` exists for the pretraining corpus but was never applied
to the task split.

**Correction and the cheap fix.** The C4 comparison must be re-run end to end on the current data
before it is reported. The good news is that it is not expensive: the `ours_*` arms are 10.6M
parameters and take roughly 11–17 minutes each, so three seeds is about an hour even under current
memory pressure. PubMedBERT is the slow arm and two current-data seeds (s1 done, s2 running) are
already in hand. **Required: `ours_randinit` × 3 seeds on current data.** Then the comparison is
like-for-like with seed variance on both sides.

**Also required going forward:** stamp the task-split content hash into every run record, the way
`train_tokenizer.py` already stamps `bundles_content_hash` and `task_split_hash`. A run that does
not record what it trained on cannot be compared to anything later.

---

## C33. Drive compliance remediated

**Action taken 2026-08-08**, with the user's explicit go-ahead. A scan of the Drive `thesis/`
tree for note-text markers and known credentialed filenames returned **33 files**, roughly 370 MB:

| location | files | what they were |
|---|---|---|
| `accord_neurips2026_supplemental_CORRECTED/` | 2 | `accord_weak_supervision_labels_decontam.csv` (7,184 rows) and `accord_decontamination_audit.csv` (1,321 rows) — `HADM_ID` joined to verbatim `nursing_sentence` / `discharge_sentence`, staged inside a folder prepared for publication |
| `data/` | 7 | raw PhysioNet distributions `ADMISSIONS`, `PATIENTS`, `DIAGNOSES_ICD`, `D_ICD_DIAGNOSES`, plus MedNLI `mli_{train,dev,test}_v1.jsonl` — separately credentialed |
| `outputs/` | 22 | note-pair and mention tables for all three cohorts (`hf_note_pairs.csv` alone is 124 MB), candidate pools, the pharmacist annotation sheets, and `annotation_pharmacist_complete.xlsx` |
| `email/` | 1 | `annotation_sheet_for_pharmacist_clean.xlsx` |

**Method.** Copied to `mimic_local_only/from_drive_2026-08-08/` preserving relative paths, verified
by **sha256 on every file** (33/33 matched), then removed from Drive only where a verified local
copy existed. A `MANIFEST.txt` records exactly what moved. A rescan returns clean.

**Left in place deliberately:** `acc_aha_2022.pdf` and `esc_2021.pdf` (published guidelines, not
MIMIC), and the finance/legal/cybersecurity expert-validation spreadsheets, which belong to a
different study and contain no credentialed data.

**Structural hardening.** The supplemental folder's `.gitignore` covered `.DS_Store` and `*.key`
but not `*.csv` — one `git add -A` would have published both note-text files. It now blocks
`*.csv`, `*.jsonl`, `*.xlsx` and `*.parquet` by extension, with explicit allowances only for the
three pair-level artifacts that carry neither note text nor admission identifiers and that the
already-published supplemental already contains. Extension blocking rather than content filtering,
for the reason `scripts/sync_results.sh` gives: a note sentence with no de-identification surrogate
passes a content scan.

**Thesis text corrected.** `ch3_data.tex` previously asserted that note-derived files *"were never
transmitted to a third-party service"* seven lines above a paragraph acknowledging a hosted-API
comparison. That self-contradiction is replaced with a four-paragraph statement that discloses the
GPT-4o transmission directly, states the disaggregation and small-cell policy, and adds a clinical-
status disclaimer. A reviewer who already raised a DUA concern would have found the contradiction
in one pass, and a self-refuting compliance claim reads worse than the underlying conduct.

**What the annotation sheet holds.** 95 of 600 sentence cells in the sheet still carry `[** **]`
MIMIC de-identification surrogates, which cannot survive a derivation step, so the sheet holds
verbatim note text. Rows are keyed `PAIR_0001` to `PAIR_0300`; no admission or subject identifier
column is present, and the pair-to-admission crosswalk left the machine at no point. The sheet was
held on Google Drive and downloaded to this machine 2026-03-27T19:22:36Z. The annotation window ran
2026-03-27 to 2026-04-28.

**One artifact remains unexamined:** the cover email in the Drive `email/` folder, and the Gmail
sent record around 2026-03-27. The reason given here was that both connectors returned expired
tokens. That was wrong, and C87 corrects it: the tokens are valid and the connectors are
authenticated as a personal account, while the drive and the mail in question belong to the
university account. The artifact is out of reach because the wrong mailbox was queried, which is a
different fact with a different remedy.

**Separately noted, not a compliance item:** the protocol specifies a nominal ten-week schedule
(two passes of four weeks with a two-week washout between). The actual window was 32 days. That is
tight but not impossible, and it does not contradict the methodology chapter's statement that the
passes were separated by a two-week washout. It is recorded because the protocol's stated purpose
for the washout was an intra-rater agreement figure the thesis now correctly reports as
unrecoverable.

---

## C34. The C4 comparison re-run on current data: the efficiency claim holds

C32 established that the headline comparison was on a dataset overwritten on 2026-08-04. Re-run on
the current split, with the prevalence check confirming both arms saw the same data (0.09246 on
every run):

| arm | params | pretraining | AUROC | AUPRC |
|---|---|---|---|---|
| `ours_randinit` seed 0 | 10,638,720 | **none** | 0.7578 | 0.2204 |
| `ours_randinit` seed 1 | " | " | 0.7698 | 0.2433 |
| `ours_randinit` seed 2 | " | " | 0.7709 | 0.2479 |
| **mean ± sd** | | | **0.7662 ± 0.0073** | |
| **PubMedBERT** (`concat_s1`) | 109,482,240 | 3.1B words | **0.7811** | 0.2678 |

**Gap +0.0149**, which is **2.1×** the small model's seed standard deviation. The from-scratch
10.6M-parameter encoder reaches **98.1% of PubMedBERT's AUROC at 9.7% of the parameters and with
no pretraining at all.**

**Against the old-data figures**, which C32 showed were not comparable to anything current:

| | old data | current data |
|---|---|---|
| `ours_randinit` | 0.8003 ± 0.0031 | 0.7662 ± 0.0073 |
| PubMedBERT | 0.8223 | 0.7811 |
| **gap** | **+0.0220** | **+0.0149** |

Both arms drop on the rebuilt split — the task is harder — but the **relative position is stable
and the gap narrows**. The efficiency claim was not an artifact of the stale dataset.

**The remaining weakness, and it is real.** PubMedBERT has **one** current-data seed. Its own seed
variance is unmeasured, so the +0.0149 gap is expressed against the small model's variance only.
If PubMedBERT's seed sd is comparable (~0.007), the gap is roughly 1.5 pooled standard deviations
— suggestive, not decisive. **Two more PubMedBERT seeds are required before the gap is
characterised**, and until then the claim should be stated as *"within roughly two seed standard
deviations of a model with ten times the parameters"* rather than as a point difference.

**Cost note.** The small arm runs in 16–18 minutes per seed; PubMedBERT took 252 minutes for one,
and 22 hours for an attempt that had to be killed under swap pressure. That asymmetry is itself
part of the efficiency result and should be reported: the comparator is ~15× the wall-clock at
~10× the parameters.

Runs recorded via the C32 lineage stamp, so this comparison can be verified against the exact
split hash it used.

---

## C35. The PubMed/ACL pass: Part A survives narrowed, Part B loses "disjoint"

The last kill condition on the surviving claim. Prior novelty checks reached only arXiv and
OpenReview; this pass reached PubMed, PubMed Central, and part of the ACL Anthology.

### Part A — null-attribute handling: **ALIVE, narrowed**

No reporting-practice review, systematic review, or reporting guideline documents null-attribute
handling across a comparable sample of clinical fairness studies. The most precise query for the
construct returned **zero PubMed records**.

But one guideline must now be engaged rather than discovered by a reviewer. **CARE-CPM** (Siddique
et al., *Health Equity* 2023, PMC10698767) asks *"Was differential missingness of predictor data in
racial and ethnic groups reported?"* and reasons that *"Exclusion of participants with missing data
provides a selective sample."* That is one inferential step from this thesis's argument — but the
step is not taken: CARE-CPM governs missingness **of predictors within known groups**, never
missingness **of the group label itself**. TRIPOD+AI (Item 14, Item 23a) and MI-CLAIM require
subgroup performance and have **no** null-attribute provision at all.

**Correction to the wording.** Do not write "nobody has considered missingness here." Write:
*existing equity-appraisal tools regulate differential predictor missingness within known groups
and cohort representativeness, while leaving the null-stratifier group itself ungoverned and
unreported.*

Adjacent work to cite rather than be caught by: Fong et al. (*Anaesth Crit Care Pain Med* 2023,
PMID 37211215) reviews whether **databases** carry the demographic fields — unit is databases, not
audit studies. Lu et al. (*JAMA Netw Open* 2022, PMID 35984654) is the nearest *genre* precedent,
documenting non-reporting of fairness items, but audits whether subgroup analysis happened at all.

### Part B — "disjoint" is too strong: **one counterexample**

**Haghighathoseini et al., AMIA Annu Symp Proc 2024;2024:473-482 (PMC12099332)** does both in one
study: a selection audit (*"The distribution of races significantly differs among the cohorts, as
indicated by the low p-values from the Chi-squared test"*) and a downstream subgroup audit (AUC by
gender, ethnicity and race in Tables 3–5).

Two qualifications preserve a narrower claim: it compares four filtered cohorts **against each
other**, not against a source-population denominator, so the "0 of 4 selection-audit papers use a
population denominator" sub-claim is untouched; and it reports subgroup AUC only, not disparity
metrics.

**Correction.** Replace *"the two audits occur in disjoint literatures"* with *"the two audits are
almost never co-located; we found a single exception, and it compares cohorts to one another rather
than to a source population."* Cite Haghighathoseini directly. **Pre-empting the reviewer who finds
it is worth more than the stronger claim.**

**Note a coding disagreement inside our own survey.** This paper is row #22 of the 31, coded
"Partial" on stage decomposition and NO on auditing both. An independent reader reached the
opposite conclusion. That is exactly the ambiguity the blind second-coder exercise (C27) was built
to surface, and it must be adjudicated explicitly in the paper with the quotes on the page — not
averaged away.

### Two open risks

1. **Colacci et al., *J Clin Epidemiol* 2025 (PMID 39532254)** — 91 studies, extraction table
   inaccessible (ScienceDirect 403). The one remaining paper whose extraction fields could still
   kill Part A. **Obtain through institutional access.**
2. **ACL Anthology coverage is incomplete and cannot be certified.** Only ClinicalNLP 2024 and 2025
   were reachable; **BioNLP, LOUHI and the main \*ACL venues were not searched.** Low residual risk
   — the kill condition is far likelier to live in PubMed venues — but the gap must be closed
   before the survey is pre-registered as a systematic review.

**Also flagged:** some abstracts returned paraphrased rather than verbatim through the fetch layer.
Bibliographic facts came from NCBI E-utilities and are solid, but **the CARE-CPM signaling questions
and the Haghighathoseini chi-squared sentence are load-bearing and must be re-verified against the
PDFs before appearing in quotation marks.**

---

## C36. E2–E5: the intervals the thesis reported point estimates for

Four computations run in sequence. Code: `accord/stats/intervals.py`. Results:
`results/accord/stats/{intervals,cohort_icd_counts}.json`. B = 10,000, seed 42.

### E2 — MCC confidence intervals: **all four systems include zero**

| system | n | MCC | 95% CI | excludes 0 |
|---|---|---|---|---|
| GPT-4o zero-shot | 253 | **+0.1089** | **[−0.0319, 0.2484]** | **no** |
| GPT-4o few-shot | 253 | +0.0117 | [−0.1120, 0.1319] | no |
| ACCORD @ 0.5 | 253 | −0.0070 | [−0.1626, 0.0753] | no |
| Llama baseline | 253 | −0.0436 | [−0.2337, 0.0563] | no |

`ch8:329-331` states that *"three members are noise."* **Four are.** This closes the open TODO at
`ch8:442` and independently confirms C15 by a second route — the interval, not just the p-value.
The ordering over these four systems is not identified and no sentence may rely on it.

### E3 — defect-free stratum: the degeneracy is real and small

n = 134, 6 positives, AUROC 0.5078.

| | interval |
|---|---|
| unstratified (as reported) | [0.2217, 0.7615] |
| **draws dropped for zero positives** | **20 of 10,000 (0.20%)** |
| stratified, preserving 6/128 | [0.2643, 0.7344] |

Report the unstratified interval to match the full-set method, **state the 20 dropped draws**, and
give the stratified interval as a robustness check. An interval that silently discards
undefined resamples is the same class of omission this thesis is about.

### E4 — sepsis/AKI ICD counts: **false alarm, the thesis is correct**

| cohort | prefix | pre-LOS | post-LOS | removed |
|---|---|---|---|---|
| heart failure | 428 | **13,608** | 13,597 | 11 |
| sepsis | 038 | **6,265** | 6,253 | 12 |
| AKI | 584 | **11,422** | 11,408 | 14 |

`ch3:115` quotes the pre-LOS counts and they reproduce exactly. The phase-1 artifacts record the
post-LOS counts. Both are right; they are different filters. **No numeric change** — add a clause
naming which filter each column reflects.

### E5 — exemplar-affected pairs: use the measurement, not the estimate

`ch7:572` applied a pooled 41.4% rate to reach "approximately 68 affected pairs." The direct
measurement on the evaluated pairs
(`results/accord/exemplar/exemplar_correctness_recomputed.json`) is **65 wrong of 137 resolved**
rule-1 and rule-2 pairs — rule 1: 60 correct of 108; rule 2: 12 of 29. Replaced in the text. An
estimate standing in front of an available measurement is exactly what this chapter criticises
elsewhere.

**Running tally of "untraceable/mismatched" items that turned out correct: E4 joins U5, U6, U7, M4,
M6 and M7.** The audit's false-positive rate is high because the reference artifacts were
unindexed until the Drive quarantine made them local.

---

## C37. E1 complete: the small encoder is not distinguishable from PubMedBERT

Three seeds per arm on the current split (prevalence 0.09246 on all six runs, so C32's
dataset confound is closed on both sides).

| arm | params | pretraining | AUROC | AUPRC | min/seed |
|---|---|---|---|---|---|
| PubMedBERT | 109,482,240 | 3.1B words | **0.7693 ± 0.0102** | 0.2554 | 125 |
| **ours, random init** | **10,638,720** | **none** | **0.7662 ± 0.0073** | 0.2372 | **17.4** |

**Gap +0.0031 — 0.35 pooled seed standard deviations. Welch t = 0.429, p = 0.690.**

The from-scratch encoder reaches **99.6%** of PubMedBERT's AUROC at **9.7%** of the parameters,
with **no pretraining at all**, in **1/7th** the wall clock.

**How this must be worded.** p = 0.690 with three seeds per arm is a *failure to reject*, not a
demonstration of equivalence — the design has low power against small differences. The entitled
claim is: **at three seeds per arm the two are not distinguishable, and the difference between them
is smaller than the variation between seeds of either one.** PubMedBERT's own seeds span 0.0183
(0.7628 to 0.7811), which is six times the gap between the models.

**The trajectory across corrections is worth recording**, because it shows what the dataset
confound was doing:

| | gap | basis |
|---|---|---|
| C32 (old data, 1 PubMedBERT seed) | +0.0220 | not comparable — different datasets |
| C34 (current data, 1 PubMedBERT seed) | +0.0149 | measured against one arm's variance only |
| **C37 (current data, 3 seeds each)** | **+0.0031** | **pooled variance, both arms** |

Each time the comparison was made more honest, the gap shrank. That is the signature of a
difference that was an artifact of measurement rather than of the models.

**Why this matters beyond the number.** The project's other well-powered result is that 1.320 nats
of pretraining-loss separation produced 0.002–0.004 downstream AUROC. C37 is the same finding from
the other direction: 3.1 billion words of domain pretraining and ten times the parameters produce
a difference smaller than seed noise. Both say the pretraining proxy does not predict downstream
utility on this task.

**Caveat carried forward.** This is one task (in-hospital mortality from 48-hour note bundles),
one architecture family, and one scale point. Whether the gap opens at 30M or 100M parameters is
untested and needs a GPU. Do not state this as a general claim about clinical encoders.

Runs carry the C32 lineage stamp, so the split they used is verifiable.

---

## C38. E9 REFUTES the invariance claim in the realistic regime

The experiment C24 and C28 both failed to run properly: a semi-synthetic test on real data, at
power, with selection acting through **outcome-predictive text features** rather than through the
group label. Pre-registered decision rule written before any audit number was computed
(`accord/selection_experiment/PREREGISTRATION.md`).

**The instrument passed its own checks first, which is what the prior attempts lacked.**

| check | result |
|---|---|
| lineage | `task_split_hash=9a9fe073f4ea9544`, matches the pin |
| classifier non-degeneracy | val AUROC **0.8775**, held-out test **0.8588**, predicted-positive rate **20.2%** against an 8.7–9.2% base rate, PPV lift **3.55×**. Nothing like C22's 87.3%-positive model |
| **positive control, run first** | planted gap recovered: DP gap **0.2797** [0.262, 0.298], four-fifths **0.415** [0.391, 0.441] |
| sweep contrast | **0.008 → 0.189** across β. C24's simulation could not separate its null from its extreme (0.00112 vs 0.00129) |

**The result.** The first adequately-powered, CI-confirmed breach occurs at **β = 2.5**, where the
selection-stage four-fifths ratio is still **0.587** — *well before* the 0.47 target the real cohort
sits at. DP gap **0.090** [0.055, 0.125], MDE 0.048. Every point from there to β = 8.0 breaches.
OLS drift of the DP gap on β: slope **0.0224 [0.0185, 0.0263]**, excluding zero — credible
monotonic drift.

**Verdict: REFUTES.** The audit is *not* blind when selection acts through features that predict
the outcome. Blindness holds in the pure group-dependent regime (S ⊥ X | A) — which C29 proved and
C31 showed is elementary — and **does not extend to the mechanism real pipelines actually use**.
Documentation volume, the real selection driver per C17, is exactly such a feature.

**What this does to C1.** The claim narrows to: *a pool-level audit is invariant to selection that
is independent of the features given group; under selection through outcome-predictive features it
is not invariant, and it drifts in a direction that partially reveals the selection.* That is a
smaller and more precise claim, and it is the one the evidence supports. Combined with C28 (the
audit fails at 33/35 operating points on the real cohort), **the "audits are blind" framing cannot
carry a paper.** What survives is the survey and the null-attribute finding.

**Secondary axes.** ETHNICITY-unknown shows the same pattern (0.006 → 0.12). **INSURANCE does not**
— DP gap stays ~0.02–0.025 with CIs touching zero and four-fifths ≥ 0.88 throughout. A real
boundary condition, reported as measured.

**Not yet tested, and it matters:** threshold robustness. One operating point was used
(val ROC at FPR = 0.15). C28 found the earlier real-data audit read flipped at 33 of 35 thresholds.
**This REFUTES verdict is not established as threshold-robust and must be before it is reported.**

---

## C39. Two errors this experiment surfaced in my own record

**1. `hash()` seeding was never reproducible.** `accord/denominator/simulate.py:288` seeds replicate
RNGs with `hash((tag, or_sel, gamma, r)) % 2**32`. Python salts `hash()` on strings per process:
verified, the same tuple gives 507250177 in one process and 2915194920 in the next. **Every number
from the original `simulate.py` is unreproducible run to run.** C24's specific values (0.00112 vs
0.00129) are therefore not recoverable — though its *conclusion* stands, because the no-contrast
finding follows structurally from the missing X term, not from the seeding. `simulate_v2.py`
already documents and avoids this.

**2. The mortality-task figures in my own summaries are stale.** `THESIS_STATUS.md` and
`PHD_PLAN.md` quote **36,504 examples, 3,730 positives, 734 test positives**. Current
`local_only/task/meta.json` records **34,018 / 2,975**, with splits train 23,846 (2,091 pos), val
3,396 (314), test 6,776 (**570** pos). The larger figures are pre-rebuild — the same Aug 4 rebuild
as C32. I briefed the E9 agent with the stale numbers and it correctly reported the measured ones
instead. Both documents must be corrected, and every claim of "734 test positives" becomes **570**.

---

## C40. E9 is threshold-robust in the region where it can be tested, and never inverts

C38 was established at a single operating point. C28 had shown that an earlier real-data audit read
**flipped** across thresholds, so a one-threshold verdict was not established. Re-run across
18 thresholds × the original 15-point β grid = **270 cells**, classifier fitted once and scores
frozen so only the decision threshold varies, with the pre-registered rule from
`accord/selection_experiment/PREREGISTRATION.md` applied verbatim at each threshold.

| predicted-positive rate | verdict | n thresholds |
|---|---|---|
| 0.090 – 0.210 | **REFUTES** — confirmed breaches | 5 |
| 0.240 – 0.360 | **REFUTES** — credible monotonic drift, no individual breach | 5 |
| 0.390 – 0.600 | **INCONCLUSIVE** — underpowered, MDE ≥ 0.05 throughout | 8 |
| **any** | **SUPPORTS** | **0 — never occurs** |

**REFUTES holds at 10 of 18 operating points (55.6%); breaches are confirmed at 36 of 270 cells
(13.3%), all at predicted-positive rates ≤ 21%.**

**The two things that matter more than the fraction.**

**There is no inversion.** Where the verdict is not REFUTES it degrades to INCONCLUSIVE — the audit
loses power as the predicted-positive rate approaches 0.5 — and never to a clean pass. This is the
opposite of C28's finding for the earlier claim, where the read genuinely flipped. Here the effect
does not disappear or reverse; it becomes unmeasurable.

**The original E9 point reproduces.** The threshold nearest E9's operating point (population
predicted-positive rate 19.94%) gives first breach at β = 2.5, DP gap **0.0965 [0.0615, 0.1326]**,
selection four-fifths 0.587 — matching C38's 0.090 [0.055, 0.125]. The sweep is a faithful
re-run, not a different setup.

**Correction to C38's wording.** It stands, narrowed: *the pool-level audit is not blind under
selection through outcome-predictive text features, for predicted-positive rates from roughly 9% to
36% — including the operating point at which the original experiment was run. Above ~39% the test
is underpowered rather than passing; the audit is never observed to be cleanly blind at any
threshold in the swept range.*

### The `simulate.py` seeding bug is fixed

`accord/denominator/simulate.py` now seeds replicates with a deterministic
`zlib.crc32` scheme matching `simulate_v2.py`'s `stable_seed`, with a comment recording that the
original `hash()`-based seeding was salted per process and that the old outputs are preserved
unchanged because C24 and C26 cite them. The script was **not** re-run and
`accord/denominator/results/` is untouched.

**One imprecision of mine corrected in passing:** I described `simulate_v2.py`'s scheme as
sha256-based. It uses `zlib.crc32`; sha256 is what `run_experiment.py`'s `det_seed` uses. The fix
matches the code that exists rather than the description I gave.

---

## C41. The PUMS replication confirms the clinical finding, and corrects my own taxonomy

> **PARTIALLY RETRACTED by C47 (2026-08-09).** The claim below that "the clinical finding
> replicates" is wrong. E9 and E10 drift in **opposite directions**. I read the drift table for
> whether each slope CI excluded zero and discarded the sign. The corrected reading is in C47 and
> is a stronger finding than the one retracted here. Everything else in C41 — Regime A's total
> invariance, the taxonomy correction — stands.

E10 ran both selection regimes on ACS PUMS (ACSIncome, Texas 2018) — public data, no
credentialing, the same source Pfohl et al. use. Positive control passed (planted DP gap 0.0746
[0.0662, 0.0828]); classifier non-degeneracy passed.

**The agent's own verdict was "does not replicate." Reading its drift table, that is wrong.**

| metric | Regime A: selection on **group only** | Regime B: selection through an **outcome-predictive feature** |
|---|---|---|
| FPR gap | −0.00048, **no drift** | −0.00262, **no drift** |
| TPR gap | +0.00032, **no drift** | **−0.00557, DRIFTS** |
| AUROC gap | −0.00045, **no drift** | +0.00001, **no drift** |
| demographic parity | +0.00078, **no drift** | **−0.01177, DRIFTS** (R² 0.94) |
| PPV gap | +0.00130, **no drift** | **−0.00342, DRIFTS** |
| calibration | +0.00100, **no drift** | −0.00032, **no drift** |
| equalized odds | — | **−0.00443, DRIFTS** |

**Regime A: nothing moves at all.** Under selection that depends only on the group label, every
audit metric is stable across a selection four-fifths ratio sweeping from 0.985 down to 0.38. The
audit is completely blind — which is the invariance claim, confirmed on non-clinical public data.

**Regime B: the audit partially detects.** Once selection acts through a feature that predicts the
outcome, demographic parity, TPR, PPV and equalized odds all drift with credible slopes, while FPR
and AUROC stay flat. **This is the same result as E9/C38 on MIMIC — the audit is not blind when
selection acts through outcome-predictive features. The clinical finding replicates.**

**Why the agent called it a refutation.** Its rule required, in Regime A, that DP/PPV/calibration
*move* — taken from my C29 taxonomy. They did not move, so it recorded REFUTES. But **my taxonomy
was stated for a different condition and I did not flag the difference when briefing it.** C29's
E3 measured selection depending on A *and Y*; PUMS Regime A is selection on A *alone*. Under
A-only selection the within-group distributions are untouched, so `P(ŷ=1 | A=a, S=1) = P(ŷ=1 | A=a)`
and demographic parity is preserved. Under A-and-Y selection the within-group outcome composition
shifts and DP moves. Both results are correct; the conditions differ.

**Correction to C29.** The taxonomy there — FPR/TPR/AUROC invariant, DP/PPV/NPV/prevalence/
calibration not — holds **under selection depending on A and Y**. It must not be stated as a
general property. Under selection on **A alone**, *everything* is preserved. Under selection
through **X**, the invariant set shrinks to FPR and AUROC only. Three regimes, three answers, and
C29 reported one of them as though it were the rule.

**Correction to the agent's verdict.** "Does not replicate" is withdrawn. The correct statement is
that the clinical finding **does** replicate on public non-clinical data, and the replication
additionally establishes a cleaner three-regime structure than the clinical experiment could see
on its own.

**What this buys the thesis.** One fully reproducible artifact on public data that any reader can
rerun — `accord/pums_replication/`, committed, no credentialing required. It answers the "n=1
dataset" objection directly.

---

## C42. The prevalence replication REFUTES the generality of the clinical over-calling result

E11 did what the clinical data structurally could not: swept **true** prevalence across 2%, 5%,
10% and 25% on public non-clinical text, three local Ollama models, two prompt framings, 0%
unparseable throughout.

**Predicted-positive rate, with over-call ratio against the true rate:**

| model | true 2% | true 5% | true 10% | true 25% |
|---|---|---|---|---|
| llama3:8b | 0.07 (**3.5×**) | 0.17 (3.4×) | 0.06 (0.6×) | 0.12 (**0.5×**) |
| granite3.3:8b | 0.05 (2.5×) | 0.07 (1.4×) | 0.10 (1.0×) | 0.16 (0.6×) |
| qwen3:8b | 0.02 (1.0×) | 0.03 (0.6×) | 0.05 (0.5×) | 0.09 (0.4×) |

**The clinical result does not generalize.** On MIMIC candidate pairs at a 4.0% base rate these
same three families over-called by **15.8× to 23.6×** (C20). On non-clinical text at a comparable
base rate the maximum is **3.5×**, and by 25% prevalence every model **under**-calls (ratios 0.4–0.6).

**The labeller partially tracks truth.** granite3.3 moves 0.05 → 0.07 → 0.10 → 0.16 as truth moves
0.02 → 0.05 → 0.10 → 0.25 — compressed but monotone. Predicted-rate spans are 0.027–0.139 against
a true span of 0.23, so there is regression toward a prior, but the prior is **not** the near-
saturated one the clinical setting produced.

**Correction.** The claim that LLM weak labellers over-call rare classes by roughly an order of
magnitude is **specific to the clinical adjudication task**, not a general property of
instruction-tuned models on low-prevalence text. C20 and C23 must be scoped accordingly. Something
about the clinical setup — plausibly that the prompt asserts the conclusion (`ch5:252` already
flags this as the likely dominant cause and it remains unrun), or that the candidate pairs are
genuinely ambiguous — produces a 21× shift that ordinary rare-class text classification does not.

**Prompting matters as much as the domain.** Permitting abstention and stating the base rate moved
llama3's abstain rate to **0.32** and qwen3's to 0.25, and drove qwen3's positive rate to zero at
both low-prevalence levels. Consistent with C23: prompt moves the quantity by as much as model
choice does.

**The distillation half — I got this wrong; see C43.** I reported the student's marginal as
"closer to true prevalence than to the teacher's" and called it the opposite of prevalence
inheritance. That reading is withdrawn: `student_marginal_oof` is **0.0 in every combination**,
i.e. the student predicts the negative class on 100% of out-of-fold rows, so "closer to a small
true prevalence" is guaranteed by the degeneracy rather than measured. Corrected in C43, where a
threshold-independent soft metric and a well-powered N=300 replication both show the student
tracking the **teacher's** marginal to within 0.1–0.5pp.

**What this means for the thesis.** The clinical over-calling result stands as measured — it is
real, it replicates across three model families *in that setting*, and C20's repair of C10 holds.
What is withdrawn is the extrapolation. This is the fourth claim in this project to survive as a
measurement and fail as a generalization, and the pattern is worth stating explicitly in the
discussion: nothing here transferred out of the setting it was measured in without being tested,
and every time it was tested, it shrank.

---

## C43. C42's "distillation half is also negative" is a threshold artifact — measured properly, the student inherits the teacher's marginal

**Claimed, in C42.** *"The distillation half is also negative. Across the granite3.3 pools the
student's marginal came out closer to the true prevalence than to the teacher's at 2% and 5%, and
tied at 10% — the opposite of the clinical prevalence-inheritance finding."*

**Actual.** Every distillation number in C42 comes from `student_marginal_oof` — the mean of the
student's **0.5-threshold hard predictions**, out-of-fold. At N=100 (5-fold CV, ~64-80 training
rows per fold, 3000-dim TF-IDF), the student's out-of-fold prediction is `not_contradiction` for
**every single row, in every one of the 22 fittable model×prompt×prevalence combos** —
`degenerate_hard_threshold: true` throughout `results/distillation_results.json`. A constant-0
classifier's marginal is trivially 0.0, so "closer to true than to teacher" is guaranteed whenever
true prevalence is smaller than half the teacher's marginal — which is exactly the 2%/5% regime
C42 reports on. It is not a measurement of what the student learned; it is an artifact of
insufficient training data producing a majority-class classifier, restated as if it were a finding
about inheritance. The accompanying claim — that OOF accuracy against true gold (0.98, 0.95, 0.90)
exceeds accuracy against the teacher label — has the identical explanation: a constant-negative
classifier scores higher against whichever target has fewer positives, and true prevalence (2%,
5%, 10%) has fewer positives than the teacher's marginal (5%, 7%, 10%) at every level tested. This
is mechanically implied by the degeneracy, not independent evidence against inheritance.

**The threshold-independent number tells the opposite story.** Mean out-of-fold **predicted
probability** of `contradiction` (`student_marginal_oof_soft` — same fold splits, same features,
same classifier, just not collapsed through a 0.5 cutoff) tracks the teacher's marginal closely at
every one of the 22 fittable N=100 combos, e.g. llama3:8b bare prompt: teacher
0.070/0.170/0.060/0.120 vs. student-soft 0.068/0.165/0.059/0.117 — within 0.002–0.005 at every
level, against true prevalence 0.020/0.050/0.100/0.250, which the soft marginal tracks much more
loosely.

**Re-run at N=300 to rule out "N=100 is just too small to trust either way."** A fresh, disjoint
pool set (`build_pools.py`'s `DISTILL_POOL_N=300`, seed 42) and `llama3:8b`/bare prompt —
`run_distillation_supplement.py`, `results/distillation_supplement_results.json`:

| true prevalence | teacher marginal | student soft marginal | true prevalence (decided) | \|soft − teacher\| | \|soft − true\| |
|---|---|---|---|---|---|
| 2% | 0.097 | 0.094 | 0.020 | **0.003** | 0.074 |
| 5% | 0.063 | 0.062 | 0.050 | **0.002** | 0.012 |
| 10% | 0.140 | 0.135 | 0.100 | **0.005** | 0.035 |
| 25% | 0.143 | 0.140 | 0.250 | **0.003** | 0.110 |

At N=300 — three times the training data C42 used, and the hard-threshold student is *still*
degenerate at 3 of 4 levels, confirming the hard/soft distinction rather than an N=100 fluke — the
student's soft marginal is within 0.5pp of the teacher at every level, and off the true prevalence
by 1.2–11.0pp. This is tighter agreement with the teacher than the clinical pipeline's own
headline distillation number (reproduced the teacher's per-rule rate to within 1.5pp).

**Correction.** Withdraw C42's distillation conclusion. The corrected statement: on non-clinical
text, at both N=100 and a well-powered N=300, a TF-IDF+LR student trained on an LLM's rare-class
labels reproduces the LLM's own marginal, not the ground truth, whenever the two diverge — the
prevalence-inheritance mechanism from the clinical pipeline **does** generalize. What C42 got right
and this correction does not touch: the over-calling *magnitude* (3.5× max here vs. 15.8–23.6× on
clinical pairs, C20) is still clinical-task-specific, and the calibration-slope analysis in
`accord/prevalence_replication/results/REPORT.md` (0.028–0.62 across all six model×prompt
combinations, all well below the ideal of 1) still shows every model under-responding to true
prevalence — that finding does not depend on the threshold used and is unaffected by this
correction.

**Lesson, again.** A derived summary statistic (a thresholded marginal) can point the opposite
direction from the estimator it was thresholded from, and nothing in the pipeline flags this unless
the un-thresholded quantity is computed and compared. This is the same class of error as C9's
degenerate confidence field and C3's circular kappa: a number that looks like a measurement but is
actually determined by the measurement procedure's own limitations, not by the thing it claims to
measure.

### C43 addendum — how I made this error, since the pattern matters

C42's distillation reading is exactly the failure mode **C22** documents, committed by me two days
after writing it. C22's finding was that an audit "passed" because the classifier predicted
positive on 87.3% of pairs — a near-constant predictor satisfies parity by construction rather
than by fairness. Here a student predicting the negative class on 100% of rows lands "closer to a
small true prevalence" by construction rather than by tracking truth.

Both are the same mistake: **reading a criterion as satisfied without first checking whether the
predictor had the range to fail it.** The signal was in plain sight — `student_marginal_oof: 0.0`,
repeated identically across every model, prompt and prevalence level. A value that never varies
across eight conditions is a property of the code, not of the data, and I quoted it as a finding.

The protocol this thesis prescribes already covers it: **the degeneracy precondition must run
before any metric is reported.** It exists in `accord/selection_experiment/` (C38) and fires
correctly there. It was not applied to the distillation analysis because that analysis was read
off a JSON file by hand rather than run through the protocol.

That is the argument for making the precondition an executable gate rather than a checklist item,
and it belongs in the C2 protocol chapter as a worked example — including the fact that its author
tripped the same wire twice.

---

## C44. The scaling study refutes C37's generalization — and reframes it into a better claim

E12, run on Colab A100. Public data only (AG News), three seeds per arm, twelve runs.
`results/accord/scaling/scaling_results.json`.

| arm | non-emb params | accuracy | gap vs pretrained | pooled sd | Welch p |
|---|---|---|---|---|---|
| scratch 10M | 10,649,092 | 0.8790 ± 0.0040 | +0.0606 | 20.9 | <0.0001 |
| scratch 30M | 31,913,860 | 0.8833 ± 0.0009 | +0.0563 | 58.9 | <0.0001 |
| scratch 100M | 99,234,820 | 0.8878 ± 0.0016 | **+0.0518** | 38.1 | <0.0001 |
| **pretrained BERT** | 85,648,132 | **0.9396 ± 0.0010** | — | — | — |

**On this task, pretraining is worth about five accuracy points and more parameters do not close
it.** The scratch model at **99.2M non-embedding parameters — more than BERT's 85.6M — is still
5.2 points behind at 38 pooled standard deviations.** The gap narrows with scale (0.0606 → 0.0563
→ 0.0518) but at that rate it would take orders of magnitude more capacity to vanish, if it ever
did.

**So C37 does not generalize, and its wording must change.** C37 found a 10.6M from-scratch encoder
statistically indistinguishable from PubMedBERT on the clinical mortality task (gap +0.0031,
Welch p = 0.690). That cannot be stated as *"small from-scratch encoders are competitive with large
pretrained ones."* Here, at ten times the parameters, from-scratch loses decisively.

**The correct claim is about the task, not the model.** On the clinical mortality task, 3.1 billion
words of domain pretraining and ten times the parameters buy **nothing measurable**. On AG News the
same comparison buys **five points at p < 0.0001**. The pretraining is not weak; the clinical task
is one where it does not pay.

**This makes the finding stronger, not weaker**, and it is the thesis's own argument again.
Pretraining is an instrument believed to measure "how much this model will help downstream." On
the clinical task it measures nothing — 1.320 nats of pretraining-loss separation produced
0.002–0.004 AUROC, and now 3.1B words produce no detectable gain. The AG News arm is the positive
control that makes that claim safe: **the instrument works elsewhere, so its failure on the
clinical task is a property of the task rather than of the measurement.** Without this control,
C37 was indistinguishable from an experiment too insensitive to see anything.

**Caveats.** One public task, one architecture family, 3 epochs, 40k training examples, no
hyperparameter search per arm. The scratch arms may be under-trained relative to their capacity —
the 100M arm reaching only +0.009 over the 10M arm hints at that, and a longer schedule could
change the slope. This establishes that the clinical null does not transfer; it does not establish
a scaling law.

**Also confirmed:** the `non_emb` fix works — BERT reports 85.6M non-embedding against its ~109M
total, so both arms are measured on the same basis. Had that not been fixed, the 100M scratch arm
would have looked smaller than BERT rather than larger.

---

## C45. The DAG tests do not support my confounder lean — both readings stay live

I argued the evidence leaned toward discharge-summary length being a **confounder**, which would
mean including it and reporting the effect as null (1/3 cohorts). Three diagnostics
(`accord/dag/dag_tests.py`, `results/accord/dag/dag_tests.json`) say otherwise.

### Test 1 — is there an A → D path at all?

Regressing log discharge-summary length on LANGUAGE-missingness:

| adjustment | β | 95% CI | p |
|---|---|---|---|
| unadjusted | −0.3224 | [−0.3411, −0.3037] | 1e-237 |
| + era | **−0.2564** | [−0.2780, −0.2348] | 1e-116 |
| + era + admission type | −0.2565 | [−0.2779, −0.2350] | 2e-118 |
| + era + type + LOS | −0.2725 | [−0.2933, −0.2518] | 6e-142 |

**Era does not explain the association.** After conditioning on era, admission type and LOS,
missing-LANGUAGE admissions still have discharge summaries roughly **24% shorter**. The mediator
story requires a mechanism linking the exposure to documentation volume, and one is present. My
timing argument — clerical field at admission, dictation days later — does not dispose of it,
because a common cause spanning the stay produces exactly this.

### Test 2 — does the yield effect survive within era?

| stratum | specification | OR | 95% CI | p |
|---|---|---|---|---|
| CareVue (n=7,485) | raw | 0.723 | [0.624, 0.838] | 1.6e-5 |
| CareVue | **+ nursing and discharge volume** | **0.847** | [0.724, 0.992] | **0.040** |
| MetaVision | — | skipped: only **11** missing-LANGUAGE of 2,271 | | |

**This is the most consequential number here.** Pooled across eras with volume adjusted, the effect
is null (OR 0.894, p = 0.13). **Within CareVue alone, the same volume-adjusted specification is
significant** (OR 0.847, p = 0.040). So the pooled null was partly era, and the effect does not
vanish when era is held fixed. C25's "era largely explains it" is too strong and is narrowed here.

MetaVision cannot serve as a replication: with 11 exposed admissions there is no test to run. That
is a real limit, not an omission.

### Test 3 — negative control

Missing-LANGUAGE admissions have **longer** stays: **+53.0 h** unadjusted, **+35.6 h** after era and
admission type (p = 2.7e-8). This complicates the "administratively thin encounter" story rather
than supporting it — those admissions stay *longer* while having *shorter* discharge summaries. It
also means missingness marks a genuinely different kind of encounter, so residual confounding by
whatever produces that difference cannot be excluded.

### What this changes

**My lean was wrong and is withdrawn.** The tests do not settle confounder versus mediator; they
remove the grounds I had for preferring confounder. Association after era adjustment is not
direction — an unmeasured common cause still produces Test 1's result — so the mediator reading is
not established either.

**What the thesis should do.** Report the specification curve as the result, designate the
volume-adjusted estimate as primary with the DAG stated, give the nursing-stratified estimate equal
prominence as a pre-specified sensitivity analysis, and add the within-era result, which is new and
belongs on the page: **the effect is null pooled and significant within CareVue at the same
specification.** Then say plainly that the data cannot distinguish the two causal structures, and
that the honest envelope is **OR 0.65–1.00**.

This is the fourth time in this project that a claim I expected to hold one way went the other
under a test I designed. That the tests were run before the writing rather than after is the only
reason it is a paragraph and not a retraction.

---

# THE RECORD CLOSES HERE — 2026-08-09

## C46. The framing was wrong, and the experimental record is now closed

**The error, and it is mine.** `THESIS_STATUS.md:113` and `PHD_PLAN.md:80` both lead the thesis
with **audit blindness**. C38 concludes: *"The audit is not blind when selection acts through
features that predict the outcome."* C40 confirms it across 270 cells with `SUPPORTS` occurring at
**zero of 18** operating points. C41 replicates the refutation on public data. C28 had already
killed the real-data leg at 33 of 35 thresholds, and C31 established the underlying theory is
elementary.

**I have been planning a chapter around a proposition this project disproved.** Written that way
the thesis argues something its own chapter refutes, and it fails in the first twenty minutes of a
defense.

### The framing that survives everything

> **Evaluation instruments in clinical NLP routinely report numbers determined by the measurement
> procedure rather than by the quantity claimed, and the failure is invisible without a positive
> control or a degeneracy precondition.**

This is not a new claim — it is C44's sentence generalized: *the instrument works elsewhere, so its
failure on the clinical task is a property of the task rather than of the measurement.* Instances
already measured:

| instrument | reported | actually determined by |
|---|---|---|
| pretraining loss | downstream utility | nothing — 1.320 nats → 0.002–0.004 AUROC |
| domain pretraining + 10× params | capability | the task — +0.0031 (p=0.69) clinical vs **+5.2 points (p<0.0001)** on AG News |
| auxiliary rule head F1 0.9603 | learned structure | leakage — 0.065/0.060 out of domain |
| κ = 1.000 | annotator reliability | an identity by construction (C3) |
| `weak_confidence` | confidence | the label (C9) |
| demographic parity gap 0.034 | fairness | an 87.3%-positive classifier (C22) |
| student "tracks truth" | distillation fidelity | a constant-negative predictor (C43) |
| four MCCs ordered | system ranking | nothing — all four intervals contain zero (C36) |
| pool audit reading | selection bias | which of three regimes holds (C41) |

**The audit work belongs inside this chapter as one of three demonstrations** — the one where the
instrument's reading depends on a regime assumption nobody states — not as the thesis.

### The correction record is nine lineages, not 45 items

Presented chronologically, 45 entries read as churn. They are **nine claim lineages**, each with a
terminal verdict; 29 of the 45 sit in a chain that supersedes earlier entries:

| lineage | chain |
|---|---|
| LLM over-calling | C10 → C20 → C23 → C42 |
| invariance theory | C29 → C31 → C41 |
| scaling / small encoder | C32 → C34 → C37 → C44 |
| language selection | C12 → C17 → C25 → C45 |
| contamination | C1 → C16 → C18 → C19 |
| audit dissociation | C22 → C24 → C28 → C38 → C40 |
| survey | C21 → C27 → C35 |
| distillation | C43 |
| transfer leakage | C2 |

The thesis should present these nine with their terminal verdicts, and demote the chronological log
to a sub-appendix.

### The stopping rule

Ten corrections landed in the final two days, including C43 (my own documented failure mode,
committed two days after documenting it) and C45 (my prediction refuted by tests I designed). A
reader watching errors surface at an accelerating rate will assume C46 exists. **It does — this is
it — and the record closes here.**

**What was audited:** every numeric claim in the ten thesis chapters against its producing script;
every architectural claim against the code that would implement it; every reported metric against
its confusion matrix; the data lineage of every training run; and every framing claim against a
falsifier chosen before the result was known.

**What was deliberately not audited:** the pre-2026-08-04 task-split artifacts, which no longer
exist on disk (C32); the original rule-3/4 generating code, unrecoverable (C4); the MedNLI run log,
the v1 component parameter counts, and the transfer-collapse figures, all marked unrecoverable on
the page rather than deleted (C36); and the BioNLP/LOUHI/main-ACL venues, unreachable by the
available search tooling (C35).

**Closed 2026-08-09.** Further corrections, if any, belong in a revision history and not in this
document.

### Two items that are arithmetic, not experiments, and may still be resolved

The 17-row discrepancy between `accord_pred` (278 of 299) and `exp1a`'s implied 261 (C28), and the
`22,446,855` parameter count that reconstructs exactly but appears in no artifact (C36). Both are
computations on existing files. Neither reopens the record.


---

## C47. E9 and E10 drift in opposite directions; I fused them by discarding the sign

> **MECHANISM RETRACTED by C50 (2026-08-09).** The observation below — that E9 and E10 drift in
> opposite directions — stands and is verified. The *explanation* offered for it, that baseline
> disparity sets the sign, was tested against a pre-registered kill condition and refuted. The sign
> is set by the selection channel, not by the baseline. See C50.

C41 states the PUMS Regime B result is "the same result as E9/C38 on MIMIC" and that "the clinical
finding replicates." Recomputed from the committed sweeps:

| β | E9 demographic-parity gap (MIMIC) | E10 Regime B demographic-parity gap (PUMS) |
|---|---|---|
| 0.0 | 0.0084 | 0.1181 |
| 0.5 | 0.0419 | 0.1203 |
| 1.0 | 0.0224 | 0.1135 |
| 1.5 | 0.0590 | 0.1139 |
| 2.0 | 0.0754 | 0.1061 |
| 2.5 | **0.0895** — increasing | **0.0988** — decreasing |

Both slope CIs exclude zero, both were coded `DRIFTS`, and I reported them as one result. They are
opposite. **The sign was in the table I was reading.**

**The mechanism, which the two datasets differ on:** PUMS's population demographic-parity gap
*before any selection* is **0.1181**. MIMIC's is **0.0084**. Selection does not have one effect on
an audit; it moves the measured gap toward the value induced by the selection itself, and whether
that raises or lowers the reading depends entirely on where the population started.

**The corrected claim, which is stronger than the retracted one:**

> The sign of a pool-level fairness audit's response to upstream selection is determined by the
> population's pre-existing disparity, not by the selection. On a population with no disparity the
> audit **invents** one; on a population that has a real one the audit **erases** it. The audit is
> not merely insensitive to upstream selection — it is misleading, in a direction that cannot be
> predicted from anything the audit reports.

The second half is the worse case for practice and is directly visible in the committed sweep: in
PUMS Regime B, `four_fifths_breach` flips from `breach` to **`pass`** at β = 2.5 and stays `pass`
through β = 8.0, with `adequately_powered: True` at every row. **The audit certifies the model as
fairer exactly as the upstream filter becomes more severe.** An analyst applying the four-fifths
rule at the model stage, on a real population with a real disparity, is rewarded for filtering
harder.

**Requires no new runs.** Both sweeps are already committed. The error was in reading them.

**Consequence for the write-up.** WRITING_PLAN §1 row 1 must be rewritten: the pool fairness audit
is not characterised by "insensitive where S ⊥ X | A." It is characterised by *sign determined by
baseline disparity*, with C29's 5.9e-16 invariance as the special case where baseline disparity is
zero and there is nothing to move. This makes Chapter 3 a stronger chapter and it does not cost a
single additional experiment.

**Third time.** C14, C42/C43, and now C47 are the same error: reading a significance flag and not
reading the quantity it flags.

---

## C48. My own figure code hardcoded the values it claimed to read, and drew a power curve from an invented formula

`accord/thesis/figures/fig_batch2.py` carried the docstring *"All from committed aggregates."* It
did not.

- **Lines 66–68, 81–82** typed the C37 and C44 numbers as literals — `0.7693`, `0.7662`, `0.0102`,
  `0.0073`, `0.0031`, `0.0518` and their interval bounds — rather than reading the run records.
- **Line 106** drew the statistical-power curve from `0.5 + 0.5*sqrt(1/n_pos)*2.8/sqrt(2)`, an
  ad-hoc expression that ignores the negative count entirely. At the evaluation's twelve positives
  it returns **0.7858**; the committed minimum detectable AUROC is **0.7382**. The marker sat
  **0.048 off its own curve**, in a figure whose subject is that underpowered evaluations are
  reported as if they were informative.

This is the defect C7 and C10 retract elsewhere in this project, committed by me, in the figures for
the chapter that documents it.

**Fixed.** Every value now reads from `local_only/task_runs/*.json` and
`results/accord/scaling/scaling_results.json`; intervals are computed from the seed spread rather
than typed. The power curve is now the Hanley–McNeil variance solved numerically over **both** class
counts at the evaluation's own 12:241 ratio, giving **0.7355** against the committed 0.7382 — the
residual is exact-vs-approximate variance, not a different quantity. All four figures re-rendered.

**The general lesson, which belongs in the thesis and not only here:** a docstring asserting
provenance is not provenance. The assertion was written in the same commit as the literals it
described.

---

## C49. The survey — the thesis's strongest claim — has no per-paper coding table on disk

`THESIS_STATUS.md` rates the survey **"strongest — survived a falsifier and a blind reliability
check."** WRITING_PLAN allocates it Chapter 3, ~600 lines and two weeks, the largest block of new
writing in the thesis. The claim is that of 31 coded papers, **0 of 25** auditing a model on a
filtered pool audit the filter, and **0 of 4** auditing selection audit a model.

**The complete evidence on disk is one 976-byte file**,
`results/accord/survey/second_coder_reliability.json`. It contains summary counts. It contains:

- no paper identifiers — the 31 papers are not named anywhere in the repository
- no per-paper field values — nothing that produces the 25/4 cross-tab
- none of the 21 verbatim quotes it reports counting
- no coding instrument, no field definitions, no adjudication log

The κ values 0.737–1.000 have no rows to compute from. **The cross-tab that is the thesis's headline
number cannot be checked, recomputed, or challenged by anyone including me.**

This is the exact defect the thesis exists to document, sitting under the thesis's own primary
claim. It is worse than the C48 hardcoding: there the underlying runs existed and the figure merely
bypassed them; here the primary data was never persisted.

**No number from the survey may appear in any chapter until the coding table exists as an artifact.**
Recoding produces a *new* survey, not a certification of the old one — and if the recode does not
reproduce 0/25 and 0/4, that result governs. That is the correct test and it must be run before
Chapter 3 is written, not after.

**Related but not the same, and much smaller:** `results/task_runs.json` indexed only 12 of the 17
runs in `local_only/task_runs/`, omitting `concat_s2`, `concat_s3` and all three `ours_current`
seeds — the C37 comparison. **The runs themselves are on disk and always were**; the published index
was stale. Regenerated from the per-run records; the red team's reading that the runs were absent
was incorrect, but the index it read was in fact incomplete.


---

## C50. The C47 mechanism is refuted by its own pre-registered kill condition; the sign is set by the selection channel, not the baseline

> **SUBSTANTIALLY CORRECTED by C56 (2026-08-09).** The refutation of C47 stands. But C50's own
> supporting comparison — E9 against E10 Regime B — was itself built on absolute values, and in
> signed terms those two sweeps drift in the SAME direction. The "opposite directions" that C41,
> C47 and C50 were each built on does not exist. What survives is narrower. See C56.

C47 explained the E9/E10 sign difference by baseline disparity: *"on a population with no disparity
the audit invents one; on a population that has a real one it erases it."* That rested on **two
points**, one per dataset, and MIMIC and PUMS differ in far more than baseline disparity.

`accord/denominator/c47_baseline_sweep.py` sweeps baseline disparity as a free parameter in the
committed simulator, over 7 baselines × 6 selection strengths × 40 replicates. The kill condition
was written and **committed at 966c87e before any number was produced**: *no sign change across
rho in [0, 0.9] means the mechanism is unsupported.*

**Regime 1 — selection through outcome leakage (`leak_gamma`). Kill condition met.**

| baseline population DP gap | pool gap, min → max selection | drift slope |
|---|---|---|
| −0.0009 | 0.0003 → 0.0002 | +0.00035 |
| 0.0400 | 0.0411 → 0.0536 | +0.00539 |
| 0.0859 | 0.0889 → 0.1178 | +0.01123 |
| 0.1380 | 0.1424 → 0.1801 | +0.01495 |
| 0.1928 | 0.1908 → 0.2463 | +0.02146 |
| 0.2509 | 0.2491 → 0.3113 | +0.02493 |
| 0.3109 | 0.3053 → 0.3748 | +0.02937 |

**Zero sign changes.** The slope is positive at every baseline, and its relationship to baseline is
**+0.0925 — increasing**, the exact opposite of C47's prediction. A population that already has a
large disparity does not have it eroded; it is amplified *harder*.

**Regime 2 — selection acting directly through the feature (`sx_coef`). Slope negative throughout.**

Same grid: slopes −0.0021, −0.0053, −0.0103, −0.0207, −0.0347, −0.0450 as baseline rises, with
slope-vs-baseline **−0.1495**. The one nominal sign change sits between baseline −0.0009 and
+0.0400, where the "positive" slope is +0.00010 — indistinguishable from zero. There is no
crossover in the sense C47 required; there is "nothing moves at zero baseline, and erosion at every
positive one."

**What actually determines the sign: the channel selection operates through.**

> Outcome-leakage selection **amplifies** the measured disparity gap; feature-driven selection
> **erodes** it. Baseline disparity scales the magnitude of both — strongly, by a factor of ~80
> across the swept range — but it does not set the direction. An audit reading that moved cannot be
> inverted back to a statement about the population without knowing the channel, and the channel is
> not something the audit reports or the analyst typically knows.

This is identified over 84 simulated conditions rather than inferred from two datasets, it makes a
directional prediction that can be checked against any new selection mechanism, and it is a
**worse** situation for practice than C47 described: C47 at least implied the analyst could reason
about direction from the population. This says they cannot reason about it from anything the audit
gives them.

**Confirmed on real data, same day.** The simulator result reproduces in the committed PUMS run,
on one population, with baseline held fixed at 0.118:

| PUMS regime | selection channel | DP drift slope |
|---|---|---|
| Regime A | selection on the group label only | **+0.007805** [0.00495, 0.01066] — rises |
| Regime B | selection through an outcome-predictive feature | **−0.011765** [−0.01359, −0.00994] — falls |

Same population, same baseline disparity, opposite signs, determined entirely by the channel. The
red team independently reached this from `full_run.log:86-111` while I reached it from the
simulator sweep. C50 is no longer a simulator result — it is the mechanism, and C47's is dead.

**And the two sweeps do NOT differ in mechanism.** I had assumed E9 and E10 might be running unlike
procedures. They are near-twins: both use `logit P(S=1) = b0(beta) + beta*z(feature)`, both bisect
`b0` to pin the marginal selection rate at 0.30, both use the same beta grid, the same `audit.py`
with `GAP_TOLERANCE=0.05` / `FOUR_FIFTHS=0.80` / `B=5000`, and one classifier fit on train only.
Design correlations are near-identical: corr(feature, group) −0.204 vs −0.209. So the E9/E10 sign
difference is real and interpretable, and the explanation is the channel, not the population.

**Fourth time.** C14, C42/C43, C47, and now the C47 mechanism. The pattern is no longer "I misread
a flag" — it is that I reach for a mechanism as soon as two numbers differ, and the mechanism
survives exactly until someone sweeps the parameter it was supposed to depend on. The
pre-registration is what caught it this time, one day after the claim was made rather than at a
viva. **Every future mechanism claim in this project gets a pre-registered parameter sweep before
it is written down, not after.**


---

## C51. C41's Regime A table does not match the run it reports, and the "correction to C29" it motivated was unnecessary

C41 reports PUMS Regime A as **"nothing moves at all"** — every audit metric stable across the
sweep. Checked against the committed `accord/pums_replication/full_run.log:86-92`:

| metric | C41 reports | the committed run |
|---|---|---|
| FPR gap | −0.00048, no drift | −0.000473, no drift, R² 0.010 — **correct** |
| TPR gap | +0.00032, no drift | +0.000084, no drift, R² 0.002 — **correct** |
| AUROC gap | −0.00045, no drift | −0.000395, no drift, R² 0.033 — **correct** |
| demographic parity | +0.00078, **no drift** | **+0.007805** [0.00495, 0.01066], **credible_drift TRUE**, R² 0.788 |
| PPV gap | +0.00130, **no drift** | **+0.007822** [0.00482, 0.01082], **credible_drift TRUE**, R² 0.771 |
| calibration | +0.00100, **no drift** | **+0.008147** [0.00576, 0.01053], **credible_drift TRUE**, R² 0.853 |

The three metrics C41 got right are exactly the three that genuinely do not drift. The three it got
wrong are exactly the three that do — each off by **a factor of ten with the boolean inverted**.
The run's own verdict line reads `DP/PPV/calibration move: True` and `regimeA_verdict: SUPPORTS`
**because** they move. C41's numbers appear to come from a superseded first attempt that
`run_experiment.py:81-96` documents and replaced.

**Three consequences.**

1. **"Regime A: nothing moves at all" is false.** The DP gap climbs 0.1356 → 0.1944 across the
   sweep. Every sentence built on Regime A invariance is withdrawn.
2. **C41's "correction to C29" was reverse-engineered from a misread log.** PUMS Regime A satisfies
   S ⊥ X | (A,Y) — *C29's own condition* — and reproduces C29's taxonomy. **C29 needed no
   correction.** The three-regime taxonomy C41 introduced to fix it is withdrawn.
3. It is the cleanest disconfirmation of C47 available, and it was in a committed log the whole
   time: Regime A and Regime B share a population and a baseline and drift in opposite directions.

---

## C52. Every gap in these experiments is absolute-valued, so no directional claim was ever licensed

`accord/selection_experiment/audit.py:47`:

```python
out["demographic_parity_gap"] = abs(g1["pred_pos_rate"] - g0["pred_pos_rate"])
```

`four_fifths_ratio` is `min/max`. **Both statistics destroy the sign.** E9's saved rows store no
per-group `pred_pos_rate` at all, so the artifact never records which group has the higher
predicted-positive rate at any β.

**What this means for everything written above C50.** What C47 called "opposite signs" is the sign
of an **OLS slope fitted to an absolute value**. That is not the sign of a disparity. A sweep whose
signed gap runs +0.12 → −0.19 — a complete reversal of which group is disadvantaged — is recorded
identically to one that widens in place. "The audit invents a disparity" cannot even name *against
whom*.

C29 already documented that this matters: its signed DP row runs population **+0.1456** → pool
**−0.0707**. The gap crossed zero and reversed. I had a documented sign inversion in a signed table
and then built a directional claim on two unsigned ones.

**Binding: both sweeps must be re-run storing signed per-group `pred_pos_rate` before any
directional statement is written.** Until then, the only licensed claim is about magnitude.

**The four-fifths `breach → pass` flip is withdrawn entirely.** Four independent problems:

- **There is no breach→pass edge.** `breach` last appears at β=0.5; β=1.0–2.0 are `inconclusive`;
  β=2.5 is `pass`. The transition C47 describes does not exist in the column.
- **It is 0.41 selection-draw SD from the boundary.** The whole flip is `ci_lo = 0.80443` clearing
  0.80 by 0.0044. `run_experiment.py:359-365` computes 30 replicate draws and stores **no replicate
  four-fifths and no replicate SD** — the measurement that would settle it was run and discarded.
- **`dp_gap_breach` reads `breach` in the identical row**, and never reads `pass` at any β. The
  audit's two disparate-impact statistics disagree and C47 quoted the one that fit.
- **`adequately_powered` is not a flag on the four-fifths call.** `run_experiment.py:398` sets it
  from `mde_dp < GAP_TOLERANCE` — a power statement about the DP gap. It is `True` at β=0.0 where
  there is no selection at all. Citing it to support the flip is, verbatim, reading a significance
  flag without reading the quantity behind it — **in the correction whose stated lesson was not to
  do that.**

The script that produced it left the warning in place at `run_experiment.py:562-571`: the
absolute-tolerance breach call is *"contaminated by intrinsic population disparity, not evidence of
selection corrupting the audit ... NOT used as the primary criterion."* β=0.0, with no selection at
all, already reads `breach`. C47 promoted to headline the one statistic its own script flags as
unusable.

---

## C53. C28's premise was wrong — but the investigation found a real error that flips a sign in six places

**C28 does not exist as a discrepancy.** 278 and 261 count **different systems**: 278 is
`accord_pred == "contradiction"` (ACCORD v2's released decision); 261 is
`weak_label == "contradiction"` (the Llama 3 8B weak labeller). `exp1a` is titled "Llama weak-label
error rates by group" and never reads `accord_pred`. The cross-tab is 243 both / 35 ACCORD-only /
18 labeller-only / 3 neither — a **net** of 17, not a filter. The `ambiguous` hypothesis is
arithmetically impossible: 278 − 15 ambiguous = 263, and `weak_label` has no ambiguous level.
**261 appears in no `.tex` chapter**, so C28's claim that both are in the draft is false of the
source. 278 is correct at ch1:239 and ch7:165/590/690/804.

**But the audit found something worse.** Every row and sentence labelled *"Llama 3 8B (weak
labeller)"* actually carries the cells of `llama_baseline` in `results/accord/step3_baselines.json`
— the Llama **evaluation-baseline prompt run**, not the weak labeller. On the same 253 pairs the
weak labeller's own labels give TP 11 / FN 1 / FP 212 / TN 29:

| | as written | the weak labeller's actual labels |
|---|---|---|
| predicted positive | 242 | **223** |
| precision | 4.55% | **4.93%** |
| MCC | **−0.044** | **+0.024** |

**The sign flips.** The sentence *"the weak labeller it was distilled from scores −0.044, which is
worse than chance"* — ch1:243, ch7:453, ch7:933, ch8:226/266/346/359 — is not supported by the weak
labeller's labels. The fix is relabelling the row, not deleting it: the ACCORD row in the same
tables reproduces exactly.

---

## C54. The 22,446,855-parameter configuration was never built, and 590,592 of it could never have received a gradient

The reconstruction is sound and **unique**: exhaustive search over 104 configurations returns
exactly one hit — 3 unfrozen BERT layers (3 × 7,087,872) + pooler (590,592) + `ATCGPretrainer` head
(592,647) = 22,446,855, embeddings excluded.

Arithmetic that reproduces is not evidence a run happened:

- The literal figure appears in **zero artifacts**, in the repo and in the quarantine.
- **No ACCORD or Stage 0 checkpoint exists on disk at all.** The three checkpoints that exist are
  the provenance encoder (35,313,024) and two copies of the stock BiomedBERT backbone.
- **No code can produce the configuration.** `accord/stage0/train.py` freezes the backbone
  unconditionally; nothing anywhere slices `encoder.layer[...]` or has an unfreeze path. The only
  evidenced Stage 0 trainable counts are 592,647 and 460,033.
- **590,592 of the total is dead.** `ATCGPretrainer.encode` reads `last_hidden_state[:, 0, :]`; the
  pooler is never on the forward path. Verified with a real forward+backward:
  `n_requires_grad = 590,592`, `n_with_nonnull_grad = 0`, while the top encoder layer and
  `node_projection` both get full gradient. **The reconstruction only closes because a module that
  can never be updated is counted as trainable** — the harm_head / PPC defect again.

Live-parameter ratio is **36.88×**, not the 37.9× claimed at ch6:427/635/681 and ch2:354.

**Action: delete the "partially unfrozen backbone" row**, or demote it to prose flagged **in the
caption, not a footnote**, as a specification with no surviving run record. The Stage 0 collapse
argument can be made from the four re-diagnosis configurations in `mimic_local_only/accord_stage0/`,
which are real runs with recorded losses and dev accuracies.


---

## C55. The survey recode does not reproduce 0/25 and 0/4. Per C49 it governs, and the headline claim changes

C49 required the survey to be re-coded with rows persisted, blinded to the prior counts, and
established that the recode governs whatever it returned. It returned something different.

**New corpus: 1,596 records screened, 75 included**, coded against written field definitions with
machine-verified quotes. All five artifacts are on disk (`results/accord/survey/`): 75-row
`coding_table.csv`, `quotes.json` (225 quoted fields, **502/502 verified by exact string match, 0
rejected**), `screening_log.csv`, `protocol.md`, and `coding_summary.json` produced by
`tabulate.py`. Counts regenerate independently from the CSV.

| | audits the selection step | does not |
|---|---|---|
| **strict reading** | **4 / 75 (5.3%)** | 71 |
| **permissive reading** | **14 / 75 (18.7%)** | 61 |

**The old claim is withdrawn.** "Of 25 papers auditing a model on a filtered pool, **0** audit the
filter" is false as stated: **four do**, under a rule I can show, with quotes. Ten more do under a
deliberately generous reading. The honest ambiguity lives in that 10-paper gap, and both readings
must be reported side by side rather than the one that sounds better — the failure C52 records.

The four strict cases: subgroup retention compared across six sepsis criteria against the source
cohort; an included-vs-excluded sensitivity analysis; chi-square tests that LLM refusals were
independent of race and sex; and a Table 1 profiling the excluded race-unknown group beside the
retained ones.

**A design limitation that kills the cross-tab framing entirely.** Inclusion criterion (ii)
*requires* a disaggregated metric, so `audits_model_on_pool = yes` holds for all 75 **by
construction**. The 2×2 degenerates to a single row. The original "25 vs 4" framing implied papers
distributed across both cells; **this design cannot produce that table, and neither could the
original one if it used the same inclusion rule.** The dissociation claim as previously written is
not recoverable from a corpus built this way.

**What replaces it, and it is cleaner:**

> **All 75 papers computed their fairness metrics on a cohort they had themselves filtered**, and
> 4 checked whether the filtering was differential. Auditing the selection step is rare — not
> absent, rare — and the rate is 5.3% strictly or 18.7% generously.

`null_attribute_handling` across 75: not_mentioned 23, drops 17, separate_category 13, unclear 9,
imputes 7, reports_only 6. **40 of 75 either drop the records or never mention them.**

**No inter-rater reliability is claimed.** Each paper was coded once by one coder under shared
written rules. The prior κ of 0.737–1.000 has no successor and must not be carried forward. If a
reliability figure is wanted in the thesis, a second coder has to code a subsample — that is real
remaining work, not a number already in hand.

**Three limitations that bound the claim, stated because the thesis is about instruments:** the
full-text set is the top of a keyword-density ranking, which over-samples fairness-forward work;
coverage is PMC open access plus arXiv and ACL Anthology, not the full literature; and 459 papers
passed title/abstract screening but fell below the full-text retrieval quota — **logged, not
silently dropped**. This is not a prevalence estimate for clinical ML as a field and must never be
written as one.

**The recode caught its own corpus bug, which is worth recording.** The first PubMed extraction used
the XPath `.//ArticleId`, which also matches identifiers inside each record's *reference list* — so
papers were assigned a cited reference's PMC ID and DOI and the wrong full text was downloaded.
**731 of 1,052 PMC IDs and 635 DOIs were wrong.** It was caught by an independent title-match check,
the selector was fixed, 35 affected codings were quarantined, and all 45 PubMed papers were re-coded
from scratch behind an automated title-match gate. arXiv and ACL used a different code path and
verified clean at 35/35. Documented in `protocol.md` §5.

**Consequence for the plan.** Chapter 3 is unblocked, but it is a different chapter: "rare, here are
the four, adjudicated on the page, under both readings" rather than "no paper does this." That is a
weaker headline and a much harder one to attack, and it comes with 75 rows and 502 verified quotes
that anyone can recompute — which the previous version did not have at any point.


---

## C56. The "opposite directions" that three corrections were built on is an artifact of `abs()`, and the simulator does not corroborate the real data

The C52 re-run is complete. Reproduction is exact — **max discrepancy 0.0** across every row of
every sweep (E9 15×6, E10 Regime A 12×6, E10 Regime B 15×7), with every pre-existing key
reproducing bit-for-bit. Only wall-clock timing fields differ. This is the same experiment with
more recorded, and the signed values can be read.

**The premise of C41, C47 and C50 is gone.**

| sweep | channel | signed DP gap | OLS slope | higher group |
|---|---|---|---|---|
| E9 (MIMIC) | selection on `doc_feature` | +0.0084 → **+0.1894** | **+0.02238** | g1 throughout |
| E10 Regime B (PUMS) | selection on `WKHP` | −0.1181 → **−0.0410** | **+0.01177** | g0 throughout |
| E10 Regime A (PUMS) | group + outcome, not covariates | −0.1356 → **−0.1944** | **−0.00780** | g0 throughout |

**E9 and E10 Regime B have the same signed slope.** Both positive. They were only ever "opposite"
because their baseline gaps have opposite signs — E9's favours g1, E10's favours g0 — and under
`abs()` a positive-baseline gap moving positive reads as *growing* while a negative-baseline gap
moving positive reads as *shrinking*. **It is the same movement recorded twice with a sign
discarded.** No demographic-parity gap crosses zero anywhere in either sweep.

That single fact retracts the observational core of three corrections:

- **C41's** "the clinical finding replicates" was accidentally *closer to right* than C47's
  retraction of it — the two sweeps do behave alike — but for none of the reasons C41 gave, and its
  Regime A table remains wrong per C51.
- **C47's** "opposite directions" observation, which I verified and reported as confirmed, was
  verified against absolute values I did not check the construction of.
- **C50's** real-data leg, the E9-vs-Regime-B contrast, is withdrawn.

**What survives, and it is one sentence.** On a single population, with everything else held fixed,
two different selection channels produce signed DP drifts of **opposite sign**: Regime A −0.00780
[CI excludes zero], Regime B +0.01177 [CI excludes zero]. That is a genuine channel effect on real
data and it is all that is left.

**The simulator does not corroborate it — it anti-correlates.** Re-running the C47 baseline sweep
capturing signed values (all gaps positive throughout, so `abs()` was not distorting it there):

| simulator channel | signed slope, baseline 0 → 0.31 |
|---|---|
| `leak_gamma` (outcome-leakage) | +0.00028 → +0.02820, **positive throughout** |
| `sx_coef` (feature-driven) | −0.00058 → −0.04117, **negative throughout** |

The real data has feature-driven selection **positive** (E9 +0.0224, Regime B +0.0118) and
group/outcome selection **negative** (Regime A −0.0078). The simulator has feature-driven
**negative** and outcome-driven **positive**. **The mapping is inverted.** So the simulator sweep
establishes that a channel effect exists in a controlled setting; it does **not** validate the
direction observed in the real experiments, and C50 should not have presented it as confirmation.
Why they disagree is unknown and I am not going to guess — that is how C47 happened.

**Fifth mechanism claim in two weeks, and the first one caught before it was written down.** C14,
C42/C43, C47, C50's real-data leg, and now this. The difference is that C52 was a *measurement*
fix, not a mechanism, and it is what exposed all of it. The lesson for the thesis is not "be more
careful" — it is that **the instrument recorded `abs()` for the entire life of the project, and
every directional reading taken from it, including four of my own corrections, was unlicensed.**
That is the thesis's own argument, demonstrated on the thesis's own record.

**Standing rule.** No directional claim from any experiment in this project unless the artifact
stores the signed quantity and the per-group values it was computed from. Retrofitted here for the
selection experiments; the same audit must be run on every other metric before it is written up.


---

## C57. The drift regressions were still fitted to unsigned columns after the C52 retrofit; seven slopes flip

The C52 retrofit added signed and per-group fields to the sweep rows. It did **not** touch the
drift-regression block — `accord/pums_replication/run_experiment.py:557-591` and
`accord/selection_experiment/run_experiment.py:373-386` still fit their OLS slope to the
**unsigned** gap column, one function away from the signed companions added in the same commit.
Confirmed from the diff of `def441c`: lines ~417–607 were never modified.

Refitting on the signed columns already on disk (`accord/stats/sign_provenance_drift_recheck.py`):

| sweep / metric | abs-fit slope | signed slope | |
|---|---|---|---|
| E9 MIMIC / demographic parity | +0.02238 | +0.02238 | — |
| E9 MIMIC / TPR | +0.00147 | +0.00852 | magnitude ×5.8 |
| E9 MIMIC / FPR | +0.01790 | +0.01986 | — |
| E9 MIMIC / PPV | −0.00593 | −0.00593 | — |
| E9 MIMIC / equalized odds | +0.00708 | +0.01413 | magnitude ×2 |
| PUMS A / demographic parity | +0.00780 | **−0.00780** | **FLIP** |
| PUMS A / PPV | +0.00782 | **−0.00782** | **FLIP** |
| PUMS A / calibration | +0.00815 | +0.00815 | — |
| PUMS B / demographic parity | −0.01177 | **+0.01177** | **FLIP** |
| PUMS B / TPR | −0.00557 | **+0.00569** | **FLIP** |
| PUMS B / FPR | −0.00262 | **+0.00703** | **FLIP + credibility call changes** |
| PUMS B / PPV | −0.00342 | **+0.00342** | **FLIP** |
| PUMS B / equalized odds | −0.00443 | **+0.00884** | **FLIP** |

**Seven sign flips. One metric changes whether its drift is credible at all.** E9 shows no flips —
but only because its signed gap happens to stay same-signed across the whole sweep, not because the
code checks. Two E9 magnitudes are still wrong by factors of 2 and 5.8.

**An ambiguity this creates inside this record, which must be fixed before either number is
quoted.** C51 reports PUMS Regime A demographic parity as **+0.007805** and calls it "the committed
run"; C56 reports the same quantity as **−0.00780**. Both are correct and they are not the same
statistic — C51 quotes the **abs-fit** slope printed in `full_run.log`, C56 quotes the **signed**
slope. C51 did not say which it was, because when C51 was written the distinction had not yet been
discovered. **Every slope in this project must from now on be labelled abs-fit or signed at the
point of use.** C51's substantive finding — that the run's own drift flags contradict C41's table —
is unaffected.

**Five of these flips are metrics C56 never checked.** C56 examined demographic parity because that
was the metric under dispute, and generalized from it. The correct move was to check every metric
the same artifact reports, which is what this sweep did.

**Still unpatched, and requiring a re-run:** `accord/selection_experiment/threshold_robustness_sweep.py`
(the C40 evidence) calls the patched `audit_pool()` but discards the signed keys before saving. No
signed variant exists anywhere on disk and it cannot be recovered by recomputation.

**Current thesis exposure: none.** No chapter cites PUMS, ACSIncome, Regime A/B, four-fifths or
demographic parity — verified by grep across ch1–ch10. This is entirely a pre-write correction,
which is the first time in this project that a defect of this class has been caught before it
reached the draft rather than after.

---

## C58. The weak labeller's MCC interval is now computed rather than asserted

C53's chapter edits stated that the corrected weak-labeller MCC of +0.024 "certainly includes zero"
without computing an interval — flagged honestly by the editor as asserted, not computed. In a
thesis about numbers that cannot be checked, that could not stand.

Computed from the weak labeller's own cells (TP 11, FN 1, FP 212, TN 29 on n = 253), 10,000
bootstrap resamples, **0 degenerate draws**:

> **MCC = +0.0243, 95% CI [−0.1040, +0.0966].**

The interval contains zero. The corrected reading is therefore *indistinguishable from chance*,
which is what the chapters now say — and the point estimate being positive rather than negative
changes nothing about that conclusion, only about the false claim that it was **worse** than
chance.


---

## C59. The survey now has a reliability figure, and it does not cover the decisive cell

A blind second coder re-coded a deterministic subsample — `sorted(ids)[0::4][:20]`, giving 19 of the
75 papers. All 19 retrieved independently, **90/90 quotes verified as exact substrings**, no first-
coder value seen. Computed across the 19-paper overlap:

| field | agreement | Cohen's κ | disagreements |
|---|---|---|---|
| `audits_selection` (strict) | 100% | **undefined — no variance** | 0 |
| `audits_selection_literal` | 94.7% | **0.855** | 1 |
| `null_attribute_handling` | 78.9% | **0.730** | 4 |
| `cohort_is_filtered` | 100% | undefined — no variance | 0 |

**The caveat is the important part and it must not be smoothed over.** The strict field shows
perfect agreement, but κ is **undefined, not 1.0**: both coders returned "no" on all 19 papers, so
there is no variance to compute against chance. The subsample happened to contain **none of the 4
strict-yes papers** — with 4 of 75, the expected count in 19 draws is about 1, so this is unlucky
rather than anomalous, but the consequence is exact:

> **Reliability is established for the permissive reading (κ 0.855) and for null-attribute handling
> (κ 0.730). It is NOT established for the strict decisive cell.** 100% agreement across 19
> unanimous negatives says almost nothing about whether the two coders would agree on a positive.

To establish it, a second coder must code a subsample **enriched for the 4 strict-yes papers** —
those are exactly the calls the survey's headline rests on. That is outstanding work and the
chapter must say so rather than quoting 100%.

The second coder flagged 9 hard cases and named `AX2603.14631` as the most contestable call in the
subsample: it reports per-attribute cohort sizes against a pre-filter N = 7,166 with a `Missing` row
(race 86.9% missing), which is formally an inclusion rate against a pre-selection denominator, but
missingness is never disaggregated *within* an attribute, so group-differential retention is
untestable. Coded strict-no / permissive-yes by both coders independently. **That paper belongs in
the chapter, adjudicated on the page.**

The one strict/permissive disagreement is `PM42243352` (coder 1 yes, coder 2 no). Both coders'
permissive counts on the overlap are close — 5/19 and 4/19.

Two contamination items disclosed unprompted by the second coder: identifying "the included papers"
required reading the screening log's `included` column, and while reading field definitions they
read one paragraph stating that `audits_model_on_pool` is yes for all included papers by
construction. Neither touches a coded value; both are recorded because the disclosure standard
matters more than the cleanliness of the claim.

---

## C60. The novelty check the frame never had: both candidate framings are prior art, and the uncovered claim is one the project has been treating as an appendix

Two previous framings died to novelty checks run *after* adoption (C26, C31). The current frame had
never been checked. It has now been, across arXiv, DBLP, OpenReview, PubMed and the ACL Anthology.

**Frame A — "five instruments are insensitive over an unidentified regime": 2/10. Four of the five
have decisive standalone prior art, and the fifth is dead by this project's own C56.**

| instrument | closest prior work |
|---|---|
| I — pool fairness audit | Biswas & Rajan, ESEC/FSE 2021 — fairness measured stage-by-stage across a pipeline *including preprocessing filters*. Plus Pfohl NeurIPS 2025, Fogliato AISTATS 2020 |
| II — pretraining loss as proxy | Lourie, Hu & Cho, EMNLP Findings 2025 — predictable downstream scaling holds ~39% of the time, at meta-analytic scale. Plus Christodoulou et al. J Clin Epidemiol 2019 |
| III — LLM weak labeller | Egami et al., NeurIPS 2023 — surrogate labels bias downstream inference **even at 80–90% accuracy**. C43's "student reproduces the teacher's marginal" is a special case |
| IV — architectural ablation | **Hewitt & Liang, EMNLP 2019, Best Paper Runner-Up** — pair a component with an *arbitrary* substitute; if the arbitrary version recovers the gain, the instrument was not measuring the claimed structure. This is A5's design and A5's conclusion |
| V — low-prevalence metric | Feinstein & Cicchetti 1990; Byrt, Bishop & Carlin 1993. **36 years old in this project's own literature** |

Instrument IV is the one `WRITING_PLAN` risk #1 and the architect both propose to **lead with**. It
is the most crowded of the five. Leading with it puts the project's strongest experiment directly
against the strongest prior art in the set.

**Frame B — "retrofit the omitted control across a pipeline": 3/10.** The proposition *"an
instrument's reading is only interpretable relative to a control that reporting routinely omits"*
is **Lipsitch, Tchetgen Tchetgen & Cohen, Epidemiology 2010** on negative controls, and
**Adebayo et al., NeurIPS 2018**, *Sanity Checks for Saliency Maps* — verified directly: it proposes
tests of whether an instrument depends on the model and data at all, and finds some saliency methods
are independent of both. That is this thesis's argument, eight years old, in its canonical form.

**The viva opener, named:** Liao, Taori, Raji & Schmidt, *"Are We Learning Yet? A Meta Review of
Evaluation Failures Across Machine Learning"*, NeurIPS 2021 Datasets & Benchmarks — 107 surveys, a
taxonomy of evaluation failures split by internal and external validity. *"Your five instruments map
onto their categories. What is your delta?"* Second question, sixteen years old and inside this
project's own methodological tradition: Lipsitch 2010.

**The one clause that searched clean.** Across four independent indexes, nothing was found where a
researcher measures the failure rate of **their own** error-detection procedure. Everything in the
genre audits others: Kapoor & Narayanan audit 329 papers; Liao et al. review 107 surveys; Musgrave
and Dacrema audit competitors; Chuang & Keiser and Freiesleben & Zezulka prescribe. **Not one
measures whether the prescriber can apply their own prescription.**

**What this project has, computed from its own record** (`results/accord/self_audit_statistics.json`):

- **20 of 58 corrections retract or refute an earlier correction.**
- **4 explicitly record the correcting analysis committing the error it was correcting** — C11, C15,
  C47, C52. C43 commits C22's exact failure mode two days after C22 documented it.
- **5 mechanism claims in two weeks; 0 of 5 survived** a parameter sweep or a signed re-run.
- **C56: one unlogged `abs()` invalidated four directional readings, three of which were themselves
  corrections of corrections.**
- **A controlled comparison already on disk:** the degeneracy precondition fired correctly where it
  was an *executable gate* and failed to fire where it was a *checklist item read by hand* — same
  author, same week, same defect class.

**Coverage gaps, stated because absence of evidence is not evidence of absence.** Semantic Scholar
was rate-limited for the entire session. FAccT and AIES proceedings were not enumerated directly —
the largest gap, since much of the closest prior art is FAccT-adjacent. CHIL and ML4H proceedings
were not enumerated. Web search was unavailable. **Psychometrics is unsearched** — Cronbach & Meehl
1955 and Messick are the primary sources behind every ML construct-validity paper cited above, and
an examiner from a measurement background will open with Messick rather than Liao. Two citations
are flagged unverified: Forman's quantification papers, and three 2026 arXiv pretraining-loss
preprints surfaced in listings but not fetched.

**This correction does not itself choose a frame.** It records that the two candidate frames are
prior art, that one uncovered claim exists, and that the evidence for it is already on disk and
required no experiment. The choice is the author's.


## C61. C40 survives the signed reading; and I swept an agent's in-flight edits into an unrelated commit for the second time

**The substantive result first.** The two remaining sign-provenance gaps from C57 are closed. The
drift regressions now fit signed columns alongside unsigned, every drift record carries an explicit
`fit_basis: "absolute" | "signed"`, and `threshold_robustness_sweep.py` now persists the signed and
per-group keys `audit_pool()` had been returning and it had been discarding.

**Reproduction: max discrepancy 0.0** across every pre-existing numeric field in all three re-runs,
including **9,177 fields** in the threshold sweep. The signed slopes agree with the independent
recomputation in `sign_provenance_drift_recomputation.json` to **0.00e+00 on all 14 metrics** — two
implementations, written separately, identical to full float precision. C57's seven flips reproduce
exactly.

**C40 holds under the signed reading.** The verdict logic reads unsigned columns, which are
bit-for-bit unchanged, so the per-threshold table is identical: **REFUTES at 10/18 operating points,
SUPPORTS at 0/18, no inversion.** Read on signed values, all **36 confirmed-breach cells are
credible-positive** and zero are credible-negative. `demographic_parity_gap_signed` does cross zero
across the 270-cell grid (246 positive, 24 negative), and three metrics have credible cells of both
signs somewhere — but the 15 credible-negative DP cells sit at β = 0.0–0.5 with target
predicted-positive rates ≥33%, the region C40's own table already calls INCONCLUSIVE or
REFUTES-by-drift-only, not REFUTES-by-breach. **No credibly-signed cell contradicts a REFUTES call
anywhere in the sweep.** C40 is the first major result to survive a signed re-reading intact.

**Now the error.** Committing C59/C60 I ran `git add -A`, which swept an agent's in-progress edits
to `accord/pums_replication/run_experiment.py` and `accord/selection_experiment/run_experiment.py`
into commit `bff5180`, whose message is entirely about survey reliability and the novelty check and
does not mention them. The agent flagged it unprompted, having made no commit itself. Verified:
`bff5180` touches `CORRECTIONS.md` plus 101 lines across those two scripts.

**This is the second occurrence.** `d137a9f` did the same thing to a different agent's lineage
edits, under a commit titled for Drive compliance. That was recorded as a note and no rule was put
in place, so it happened again. Corrected here by committing the remainder under an accurate message
that names `bff5180`; history is not rewritten, because a wrong message that is corrected in the
record is more honest than a clean history that conceals the mistake.

**Why this belongs in the record rather than in a stray note.** It is the same shape as the finding
in C60: **a defect class enforced by intention recurs; one enforced by a gate does not.** The
degeneracy precondition fired correctly where it was executable and failed where it was a checklist
item read by hand. `git add -A` was a checklist item — "be careful what you stage" — and being
careful worked once and failed twice.

**Standing rule, executable rather than intended:** never `git add -A` or `git add .` while any
agent is running. Stage explicit paths only. If this recurs a third time it is evidence that the
rule needs to be a pre-commit hook, and that fact is itself data for the thesis.


## C62. The C44 convergence sweep returns OUTCOME A under the pre-registered rule: the arms were converged and the control holds

The sweep ran to completion — 48 runs, 4 arms × 4 budgets × 3 seeds, 9.2 GPU-hours. Adjudicated by
the rule committed at `b048829` **before the sweep existed**, applied automatically by the notebook
rather than by eye.

| arm | E=3 | E=6 | E=12 | E=24 | gain for 8× compute |
|---|---|---|---|---|---|
| scratch_10M | 0.8784 | 0.8795 | 0.8822 | 0.8838 | **+0.54 pts** |
| scratch_30M | 0.8833 | 0.8856 | 0.8845 | 0.8886 | **+0.52 pts** |
| scratch_100M | 0.8885 | 0.8903 | 0.8888 | **0.8925 ± 0.0004** | **+0.41 pts** |
| pretrained_bert | 0.9392 | 0.9369 | 0.9375 | **0.9379 ± 0.0029** | **−0.13 pts** |

> **VERDICT: OUTCOME A.** Gap at 24 epochs **4.53 accuracy points**, ±1 SD bands disjoint.
> Threshold was ≥3.0 and seed-resolved.

**The under-training hypothesis is refuted, and not narrowly.** Eight times the training budget
bought the 100M scratch arm **0.41 accuracy points**. Every scratch arm shows the same pattern, and
BERT actually *declines* 0.13 points with more epochs — mild overfitting, which is the expected
behaviour of an already-converged model. The 3-epoch arms were not stopped early; they were done.

**Honest detail that must be reported, not buried:** the gap does narrow slightly with budget, 5.07
→ 4.67 → 4.86 → 4.53 points. It is non-monotone and the movement is small relative to the gap, but
the correct statement is *"the gap narrows by roughly half a point under an eightfold budget
increase and remains 4.53 points, seed-resolved"* — **not** "the gap is unchanged."

**What this licenses.** Instrument II now has a **verified positive control**: the instrument
detects a 4.53-point pretraining benefit on AG News where one exists, at 3 seeds with SD ≤ 0.003.
C37's clinical result — gap +0.0031, Welch p = 0.690 — is therefore reportable as a **genuine null**
rather than as an experiment too insensitive to see anything. C44 stands.

**Consequences for the plan.** Under the pre-registered rule this is the nine-chapter branch: the
pretraining-proxy chapter stands at ~320 lines rather than collapsing to ~140 and merging.
`WRITING_PLAN` risk #3 — *"the one remaining experiment with a real argument for running it"* — is
closed, and closed in the direction that keeps a chapter rather than losing one.

**The scaling picture at convergence, which is a separate and weaker claim.** At 24 epochs the
from-scratch curve reads 0.8838 → 0.8886 → 0.8925 for 10M → 30M → 100M: **+0.87 points for a 10×
parameter increase**, still gentle, still far short of closing a 4.53-point gap. Nothing here
supports a claim that scale alone would close it, and the thesis must not imply one.

**First OUTCOME-A result in the record.** Fifty-plus corrections have moved claims down. This one
was pre-registered, ran against its own kill condition, and the kill condition did not fire. That is
worth stating plainly in the thesis, because a record in which nothing ever survives is not a record
of rigour — it is a record of a project with no findings.

---

## C63. The replication's missing cell: over-calling magnitude was attributed to the pool, and a frontier labeller on the same pool refutes it

**How this came up.** Not from an audit of the claim. A PhysioNet compliance problem forced the
GPT-4o comparison to be re-run through a different channel, which required a different model. The
falsification arrived as a side effect of a remediation.

**The claim, and the sentence that pre-registered its own test.** `msu_ch4_results.tex` said the
over-calling magnitude "belongs to the base rate of the pool rather than to the model," repeated at
`msu_ch1_introduction.tex` (contribution 5) and `msu_ch6_conclusion.tex`. The methodology chapter
had already written the firing condition: *"If over-calling is a property of one model it will not
survive a change of model; if it is a property of the task presentation it will."*

**Why the control could not decide it.** The replication varied model family three times and domain
once, and held scale fixed at 8B across all three labellers. With capability constant, any
clinical-versus-public difference is attributable to the pool by construction. The design was a
two-by-two in domain and prevalence, and the cell it never contained was frontier scale on the
clinical pool.

**The new cell.** Claude Sonnet 5, same recovered prompt, same synthetic demonstrations, same pairs,
generated by `accord/differential/frontier_metrics.py` from the two committed prediction artifacts:

| basis | flagged | over-calling | recall | FPR | MCC | 95% CI |
|---|---|---|---|---|---|---|
| G_253 zero-shot | 48 | 4.00x | 0.7500 | 0.1618 | +0.3189 | +0.1473 to +0.4622 |
| G_253 few-shot | 39 | 3.25x | 0.5000 | 0.1369 | +0.2137 | +0.0314 to +0.3838 |

The 8B range on this pool was 15.8196x to 23.5959x; the public-text sweep's maximum was 3.5000x. A
frontier labeller on the clinical pool lands in the public-text band. **The magnitude tracks the
labeller. The attribution to the pool is withdrawn.**

**Second thing this breaks, and it was already broken.** `msu_ch4_results.tex` said no ordering could
be trusted "because at 12 positives none of the four systems is distinguishable from chance." The
causal clause was false before this run. Twelve positives against 241 clear negatives has ample
power; the four systems were indistinguishable because all four were bad. Both Claude intervals
exclude zero on the same 12 positives, at a hypergeometric one-sided p of 2.282e-05 zero-shot. The
sample-size shield was too broad: it covers comparisons living on the positives and never covered
over-calling, which is measured on the negatives.

**Also corrected:** `msu_ch5_discussion.tex` said GPT-4o's +0.1089 "clears zero" while the Chapter 4
table note said all four intervals contain it. Same document, opposite claims, and the table note was
right. Changed to "is positive."

**What this gains, which is more than it costs.** `msu_ch1_introduction.tex` conceded that "no
sensitivity check of the contradiction evaluation itself was run." There is one now. Claude zero-shot
reaches balanced accuracy 0.7941, above the 0.7355 minimum detectable AUROC this evaluation derived
for itself. RQ4's answer moves from "of the task as this pipeline poses it" to "of the pipeline, not
of the task as posed," and the null becomes a demonstrated null rather than a possibly-blind one.

**A new finding, not a correction.** Claude's false positives concentrate on the pairs the pharmacist
marked ambiguous: 16/46 (0.3478) against 39/241 (0.1618), OR 2.76, one-sided Fisher p 0.0046;
few-shot replicates independently at 15/46 against 33/241, OR 3.05, p 0.0029. Two runs, same
direction, 287 negatives. The ambiguous category measures evidential difficulty rather than annotator
inattention, which `sec:m-gold` assumed without testing.

**Deviations from the original protocol, recorded because they are real.** `max_tokens` went 50 to
1024: the recorded 50 was ample for GPT-4o's one-word reply, and a reasoning model spends it on a
thinking block and returns no text, which silently produced 8 unparseable few-shot rows before the
budget was raised. The model does not accept a temperature parameter, so its labels are single draws,
not fixed values; PAIR_0256 returned opposite labels on two runs of an identical prompt during
diagnosis. One flip on a positive moves recall by 8.33 points. The Claude rows therefore sit beside
the GPT-4o rows rather than replacing them, per the existing rule that a reference row must come from
the same script and protocol as the rows it is compared against.

**Two claims I made during this work that were wrong, and are corrected here rather than quietly.**
First, I identified `run_multillm_replication.py:263,278` as the code binding note sentences into the
GPT-4o prompt. That file is the local Ollama script; the GPT-4o code is not in the repository at all.
The narrower payload account in `sec:m-ethics` is therefore a reconstruction from the annotation
sheet's columns and the archived prompt template, and is now labelled as one. Second, I wrote a
cross-reference in `sec:m-ethics` promising that `sec:m-data` states which rows the thesis relies on;
`sec:m-data` points back to `sec:m-ethics`, so the promise was made twice and kept zero times. Both
fixed.

**Undisclosed second channel, found in the same audit.** `msu_ch3_methodology.tex` said "one
departure is recorded rather than concealed." `run_multillm_replication.py:394` names a Llama 3 8B
calibration run through the Hugging Face Inference API over 1,679 pool rows, and Chapter 4 reports
its number. The count is now two. `README.md` also claimed no MIMIC data is cloud-synced, which is
false: `annotation_sample_300.csv`, carrying note sentences joined to HADM_ID, is read from a Google
Drive mount. The claim is withdrawn rather than restated.

**`sync_results.sh` was not the control the thesis cited.** Its code gate ran *after* the rsync to
Drive and removed nothing on abort, under a comment asserting the opposite; its marker set included
two column names that appear in the project's own pipeline sources, so it could never pass; the
identifier pattern missed a trailing-column CSV header; and an unquoted `find` silently skipped any
path containing a space. All four fixed, and both gates now pass on the current tree.

**Addendum, the replicate study that was designed and then not run.** The
single-draw limitation above looked like it wanted an empirical fix: re-run every
pair five times, report the majority label and the disagreement rate. The harness
was built and then two audits killed the design, both correctly.

The arithmetic first. Residual probability that a majority-of-five label is still
wrong is 5.79 percent for a pair flipping at p = 0.20 and 31.74 percent at
p = 0.40. Against a recall interval on 12 positives already more than thirty
points wide, five replicates buys roughly three and a half points. The binding
constraint is twelve, not one, and no feasible replicate count changes that.

The design flaw second, and it is the one that matters. Majority voting is a
sharpening operator: it maps a pair's per-draw positive propensity away from 0.5.
For a better-than-chance system it therefore raises the expected Matthews
correlation and lowers the expected over-calling ratio, by construction and
without exception. Both movements are in the direction this thesis needs. Running
an aggregation change that lifts the author's own headline, in a thesis whose
contribution is a catalogue of numbers that were artifacts of their own procedure,
would have been the most quotable thing in the document. It would also have
reported a five-sample self-consistency ensemble at five times the inference budget
under the label "zero-shot", against an ACCORD row that is a single archived run.

Three implementation defects were found in the same pass and are recorded because
they nearly cost a paid run: `frontier_metrics.py` read the unsuffixed artifact
path, so it would have silently re-emitted the old numbers and reported success;
an unresolved label reaching it as NaN was dropped from all four confusion cells
while still counting in n, then crashed on `int(yh.sum())` after ten thousand
bootstrap iterations; and the even-split tie rule is unreachable at odd replicate
counts, so the category built to record genuine indecision would instead have
recorded the parity of transient API failures. All are fixed. `--replicates`
remains available and is not used for any reported number.

**What was run instead, for zero calls.** `frontier_label_noise.py` sweeps a single
per-draw flip probability p and carries its binomial variance into the Matthews
correlation through the observed matrix's per-flip sensitivity. At the worst case
any model can assert, p = 0.5 with every label an independent fair coin, the
zero-shot interval is +0.0929 to +0.5448 and still excludes zero. The above-chance
reading therefore does not depend on the decoding being pinned. The few-shot
interval admits zero at p = 0.20, so its exclusion of zero is withdrawn and no
claim rests on it. One honest limit: the point estimate is held fixed, so this
bounds the interval and is not a posterior.

**A claim softened as a consequence.** The few-shot direction difference, three
positives of twelve, is not separable from draw-to-draw variation and is now
reported as a failure to replicate rather than as a reversal.

---

## C64. Two wrong accounts of the annotation, in opposite directions, neither of them checked against what was done

**What happened.** Section 3.9's annotator paragraph described the pharmacist as "working remotely
from a spreadsheet of candidate pairs." A later rewrite of that paragraph built further on the same
description.

**That description was wrong.** The annotation was done with the author physically present, on the
author's machine. No file was sent. No copy was made, printed or photographed. Nothing was retained
afterwards. The annotator read each pair on the author's screen and gave a judgment the author
recorded.

**How it happened, which is the part worth recording.** The phrase "working remotely" entered the
draft as a description nobody sourced, and every subsequent reading built on it. The premise was
never tested against the one source that could settle it, which was the author.

**This is the project's own failure mode, applied to the project.** Every other entry in this record
is a number that survived because nobody ran the control that would have caught it. This one is a
sentence that survived for the same reason, in a thesis whose argument is that the cheap check is
the one nobody runs. It took a direct question to the author to resolve, and that question should
have been the first step rather than the last.

**What the section says now.** The arrangement is stated: supervised, on the author's equipment, no
copy, nothing retained. The sheet's status as restricted data is unchanged and still measured.

**What does not change.** The GPT-4o and Hugging Face transmissions remain departures and remain
disclosed.

---

## C65. Every result obtained through a non-compliant channel is withdrawn, and the argument for keeping them is withdrawn with them

**What goes.** Three runs, two channels. GPT-4o zero-shot and few-shot over roughly 300 annotated
pairs through OpenAI's public API, and a Llama 3 8B calibration run over 1,679 candidate-pool rows
through the Hugging Face Inference API. PhysioNet's guidance of 18 April 2023 treats transmission of
credentialed data to either as prohibited.

**What stays.** The disclosure. Section 3.9 records both transmissions, and records them more fully
than it did: the count was wrong at one departure before this work, and the second channel carried
more rows than the first.

**Where each row was, and what replaced it.** The GPT-4o rows appeared in the per-pair comparison
table, the Matthews table, the frontier table, the forest figure, the exchange-rate figure and about
twenty prose sites across Chapters 4, 5 and 6. They are replaced by the Claude Sonnet 5 runs, which
are a different model and therefore a new measurement rather than a reproduction. The Llama 3
calibration row appeared once, in the over-calling table, and is replaced by the local run of the
same model on the same pool, which is a genuine reproduction and was already committed.

**The subsection that had to be rebuilt rather than edited.** "Whether a stronger labeller removes
it" was built entirely on the GPT-4o figures. Its question is now answered from the frontier
comparison alone.

**Counts that moved, recorded because a silent recount is how this kind of edit goes wrong.** The
Matthews table note went from four of six intervals containing zero to two of four. The multiplicity
disclosure went from ten interval estimates with seven failing to exclude the null, to eight with
five; both withdrawn intervals contained zero, which is why the second number fell with the first.
The forest figure's computed headline went from seven of ten crossing zero to five of eight. The
ordering passage lost its worked example, which was the widest-separated pair, and now uses the two
systems this pipeline produced.

**The position this reverses.** Section 3.9 previously retained the rows and argued that deleting
them would misrepresent the comparison actually run. That argument was right about record-keeping
and wrong about the thesis. The withdrawn values survive in this record and in the project's
artifacts, where an auditor can find them; what has gone is the request that a reader rely on them.
Reporting a result is asking for reliance, and asking for reliance on data obtained through a
prohibited channel is a different act from recording that the channel was used.

**What this costs.** The capability comparison loses its middle point. Three 8B labellers sit at
15.4167 to 23.7500 on the gold set and the frontier labeller at 4.5000 and 5.3333, with nothing
between them. Two clusters support the claim that they differ. Whether they support a claim about a
trend with capability is a weaker question, and filling that gap needs a mid-capability run through
the compliant channel rather than an argument.

---

## C66. The withdrawal of the deployed weak labeller's row was wrong, and reverted

**What was done.** C65 withdrew three runs. Two were GPT-4o. The third was recorded as a Llama 3
calibration figure of 21.1473, withdrawn on the stated ground that it was "genuinely redundant"
because the same model had been re-run locally and the compliant value was 23.5959 on the same pool.

**That ground was false, and the check that falsifies it is a value_counts.** The weak supervision
file the detector trained on carries 1425 contradictions of 1679 rows, and the 299-row audit carries
261. Those are the withdrawn run's numbers, not the replication's. The replication produced 1590 and
285. The two runs are 165 labels apart on the pool.

**Why they differ, which the thesis also had wrong.** The table note explained the gap as a
confidence threshold. Section 3.5 states that the threshold lies beneath the field's minimum and
removes nothing, so it cannot move a positive rate by 165 rows. The difference is the prompt. The
replication reran the evaluation baseline's template verbatim, a bare pair with a one-word answer
demanded; the deployed labeller was asked for a judgement plus a confidence value and was given the
conflict type before being asked whether a conflict was present. Two configurations of one model.

**What that makes the withdrawn row.** Not a redundant calibration figure. The only published
measurement of the configuration that produced the detector's training supervision, and therefore
of the run behind the 261-of-299 marginal that Chapter 4 reports as this thesis's weak-supervision
finding. Deleting it while every ACCORD number in the document still rests on that run's per-row
outputs would have removed the visible summary of a dependency without removing the dependency,
which is the shape of failure this thesis exists to name.

**Reverted.** The row is restored under its correct name, the deployed weak labeller rather than an
originally deployed calibration figure, and the note now states the prompt difference and the 165
labels instead of the threshold explanation. The GPT-4o withdrawal is untouched by this entry and
stands.

**How it happened.** The redundancy claim came from matching a model name and a pool size across two
rows and not checking the numbers underneath them. The arithmetic that settles it takes one line and
was not run until an adversarial review asked for it. The lesson is the project's own: a row that
looks like a duplicate of another row is a hypothesis about provenance, and provenance is the thing
this thesis says must be checked rather than assumed.

**Open, and not resolvable from the record.** Section 3.5 says the production labeller was run
locally. Section 3.9 says the Hugging Face Inference API run covered the full candidate pool and is
the originally deployed figure. Both cannot be true of the same run. Which channel produced the
1425 labels determines whether the detector's training supervision came through a compliant route,
and it is a question for the author rather than the artifacts.

---

## C67. The detector source is recovered, and every architectural finding can now be checked

**What was missing.** Section 3.9 disclosed that the detector's model and training code was not in
the repository, and that the architectural findings of Chapter 4 therefore rested on reading a file
no reader could open. That was the only research question in this thesis whose evidence could not be
independently checked.

**Where it was.** In the author's Downloads folder, exported from Colab and never committed. A
filesystem search for the parameter names the correction record quotes found it in one pass. The
same search found the Colab notebooks behind the architecture, the ablations, the transfer
experiments and the stage-zero pretraining, all of which are further recovery targets.

**Committed unmodified**, under a header that records where it came from and what it establishes.
Nothing was tidied or renamed. Its value is as evidence, and every line number the correction record
cites refers to the file as the audit read it.

**What it confirms, each now checkable by opening the file.** The authority parameters reported as
learned are initialised at exactly the values they were reported as converging to:
`nn.Parameter(torch.tensor([authority_dampening, 1.0]))`, with the default at 0.5. The guideline
bank is one randomly initialised parameter matrix, `nn.Parameter(torch.randn(20, hidden_dim))`, with
no corpus and no retrieval step. The training objective is
`CONTRADICTION_WEIGHT * c_loss + RULE_TYPE_WEIGHT * r_loss`, and the harm head's output appears in
neither term. The weak labeller's confidence field is multiplied into the per-example loss,
`(contradiction_loss_fn(...) * confidences).mean()`, which is the label-conditioned reweighting
Section 3.5 describes.

**No finding changed.** Every architectural claim in Chapter 4 is confirmed by the recovered source.
This entry records a gap closing rather than a number moving, which is the rarer kind and worth the
same treatment.

**Verified before publishing:** no note text, no admission or subject identifier, no credential, no
absolute path carrying a username. The only path in the file is a Colab mount.

## C68. [PREMISE CORRECTED BY C85 AND C84: the zero-shot gradient cleared no threshold at any count, and the four frontier tests used the wrong instrument] The multiplicity count was understated by two, and the understatement was the difference between the headline gradient clearing Bonferroni and not clearing it

**What was printed.** The multiplicity subsection of Chapter 4 opened "This chapter reports
fourteen hypothesis tests" and then enumerated them: five Mann-Whitney, five DeLong, one Fisher
exact on the mechanism split, one Welch on the anisotropy control, two Spearman against the danger
rating. That enumeration sums to fourteen and is internally consistent, which is why it survived
several reads.

**What the chapter actually reports.** Sixteen. Section~4 on the ambiguity gradient reports two
further Fisher exact tests, one per prompt condition, at $p = 0.0046$ and $p = 0.0029$. They were
added to the chapter when the frontier comparison was added and the count was never revisited. The
enumeration was consistent with itself and not with the chapter.

**Why the direction matters.** The Bonferroni threshold is $\alpha$ divided by the count. At
fourteen it is $0.05/14 = 0.0036$ and the zero-shot ambiguity gradient, which is the reading the
section leads with, clears it at $0.0046$. At sixteen it is $0.05/16 = 0.0031$ and the zero-shot
gradient does not clear it. The understated count was the count under which the headline figure
survived correction. No one constructed that; it is what an unmaintained number does, and it landed
on the flattering side, which is the only reason this entry is written at this length.

**What is true after the recount.** Two of the sixteen tests clear $0.0031$. One is the Welch test
on the anisotropy control, which Section~4 already declines to read as support for the mechanism on
effect-size grounds. The other is the few-shot ambiguity gradient at $0.0029$, which is the
dependent replicate: the same model on the same items under a second prompt, whose $p$ value the
section already says carries much less than its face value. So the family's only surviving member is
the dependent replicate of a test that did not survive. That is weaker than what was printed, and
Section~4 now says so in those terms.

**Two further numbers moved with the count.** The familywise upper bound, $1 - 0.95^{k}$, goes from
$0.51$ to $0.5599$. The register keys `stats.multiplicity.bonferroni_threshold` and
`stats.multiplicity.familywise_upper_bound` were derived text reading "divided by the 14 hypothesis
tests reported in the Results chapter" and are now bound to an artifact.

**The fix is structural, not a retyped number.** `accord/stats/test_register.py` enumerates the
family, one entry per test, each bound to the artifact and key its $p$ value is read from, with the
dependence blocks named. It computes the count, the Bonferroni threshold, the familywise upper
bound, which tests clear, and a Holm ordering for reference, and writes
`results/accord/stats/test_register.json`. A test added to the chapter is added to the list and every
derived number moves with it. A count in prose cannot do that, and this one did not.

**What the register does not do.** It does not apply a correction. Chapter 4 applies none and says
so. It computes what the threshold would be so that the sentence quoting the threshold cannot drift
from the family it describes.

## C69. The label-noise sensitivity analysis carried the variance and dropped the mean, and the claim it supported is withdrawn

**What was printed.** Chapter 4 asked how unstable the frontier labeller's labels would have to be
before the conclusion changed, and answered that they could be arbitrarily unstable: at $p = 0.5$,
where every predicted label is an independent fair coin, it reported a Matthews interval of
$+0.0829$ to $+0.5353$ for the zero-shot run and concluded that the result "survives even the
fair-coin worst case". Figure 14's deck read "One result survives any instability; the other does
not".

**Why that cannot be true, before any recomputation.** If every predicted label is a fair coin, the
predictions carry no information about the labels, and the Matthews correlation is centred on zero
by construction. An interval that excludes zero at that point is not a finding about the run; it is
a statement that the analysis is not computing what it claims to. This is the rare case where the
result is refutable from the definition alone, which is how it was caught.

**What the construction actually did.** It treated the flip count on each class as binomial, took a
first-order central difference of the Matthews correlation at the observed confusion matrix, and
propagated the variance of the flip count through that derivative. So it added the variance the
flips induce and never moved the point estimate, which it held at the value the run produced. But
flipping labels does not only widen the interval, it destroys the association, and the shift of the
centre is the dominant term at every $p$ that matters. The analysis carried the smaller term and
dropped the larger one, in the direction that flatters the result.

**What replaces it.** Direct simulation of the stated mechanism, in
`accord/differential/frontier_label_noise.py`: resample the 253 pairs, flip each predicted label
independently with probability $p$, recompute the statistic, $B = 10{,}000$ draws per value of $p$.
No linearisation and no analytic variance. Both terms travel because both are produced by the
simulation rather than modelled.

**What the corrected analysis returns.** At $p = 0.5$ the zero-shot interval is $-0.1223$ to
$+0.1226$ with a mean of $+0.0001$ and the statistic at or below zero in $0.502$ of draws. That is
the fair coin behaving as a fair coin, and it is the direct check that the new construction is
computing the right thing. Neither run survives arbitrary instability. The zero-shot interval admits
zero at $p = 0.2467$, roughly one label in four; the few-shot interval at $p = 0.0941$, roughly one
in eleven. The surviving claim is the narrower one, that the zero-shot exclusion holds up to
$0.2467$ and not beyond, and Chapter 4 now states it that way.

**Three numbers in the chapter moved with it.** The few-shot crossing was printed as $0.1159$ in the
body, in the note to the Matthews table and in Appendix C; it is $0.0941$. The zero-shot crossing
did not exist before, because the old analysis reported no crossing at all. The zero-shot worst-case
lower bound, $+0.0829$, is withdrawn and is now quoted only as the withdrawn figure.

**Two structural fixes, because the failure was not only arithmetic.** First, every
`labelnoise.*` key in the fact register was entered by hand, so when the script was corrected all 96
of them stayed behind on the withdrawn artifact, including
`labelnoise.zero_shot.survives_any_label_noise = True`. The script now rewrites its own block of the
register from the artifact it just wrote and deletes keys the old construction produced that the new
one does not, so the two cannot diverge again. Second, the figure plotted one dot at $p = 0$ and no
centre line, which is what a figure looks like when the analysis behind it holds the centre fixed;
it now draws the mean as a dashed line in each panel, and the end of that line, on the chance rule
at a fair coin, is the correction made visible. The figure's ink floors are lowered with the reason
recorded: the band area the old figure filled was the overstatement, so losing it is the fix.

## C70. The within-vendor test returns Outcome A, and the replication attempts it carried take the ambiguity gradient below correction

**The pre-registered test.** C68, committed at `b249eeb` before any pair was sent, asked whether
over-calling attenuates with capability or whether the ordering was an artifact of vendor or serving
channel. Every low-capability point in the earlier comparison was an open-weights model served
locally and every high-capability point a hosted API from one vendor, so the three varied together.
The test held family and maker fixed and varied the channel: `meta.llama3-70b-instruct-v1:0` on AWS
Bedrock, in the author's own account, against the `llama3:8b` already measured locally.

**Outcome A.** The G_299 zero-shot over-calling ratio is 10.5000, strictly inside the pre-registered
window of 6.2500 to 15.4167. Both conditions ran once over the 300 pairs, no restart, no splice, 0
unparseable replies and 0 pairs without a usable draw. The within-family step is 2.2619 against the
2.4667 measured across families and vendors, so vendor was not carrying the earlier result and the
attenuation claim may stand in Chapters 1, 4, 5 and 6.

**The awkward number in the same run.** The 70B's few-shot ratio is 6.2500, exactly Haiku's
zero-shot value and above Sonnet's. A prompt change moves one model by 1.6800, a large share of the
capability span the run was built to probe. The pre-registration named prompt sensitivity as what it
could not settle and the result confirms that rather than escaping it. Both factors are now reported
together and the attenuation claim is scoped to a fixed prompt.

**What the run did to a different claim.** It also measured the ambiguity gradient, and so did the
Haiku run. Six Fisher tests now exist where two did. Haiku reproduces the gradient under few-shot
prompting at an odds ratio of 2.9046 and p = 0.0019, the smallest of the six, so the gradient is not
one model's quirk. But it falls monotonically as the labeller weakens, at both prompts, and on the
70B neither condition reaches significance: 1.6756 at p = 0.0996 few-shot, 1.2257 at p = 0.3190
zero-shot. That ordering cuts against the item-property reading. If the pharmacist's ambiguous
category tracked difficulty in the items, a weaker labeller should find those pairs at least as
hard; it finds them less distinguishable. The claim is now scoped to labellers rather than items.

**And what it did to the multiplicity arithmetic, which is the part worth recording at length.**
Reporting the four replication tests takes the family from sixteen to twenty and the Bonferroni
threshold from 0.0031 to 0.0025. At that threshold two tests clear: the anisotropy Welch test, which
Chapter 4 already declines to read as support for the mechanism, and Haiku's few-shot gradient.
Sonnet's few-shot gradient at 0.0029 cleared sixteen and does not clear twenty. Its zero-shot
gradient at 0.0046, which Chapter 4 leads with, cleared neither and cleared the fourteen the chapter
originally claimed. So the headline gradient has now failed correction under every count that has
been correctly computed, and the count grew precisely because the claim was tested again. That is
the register of C68 working as intended on the first new result to reach it, and it is the argument
for building it: a hand-typed count would have stayed at sixteen, or at fourteen, and the four new
tests would have been reported as four more pieces of support rather than as four more divisions of
alpha.

**Two generating scripts, so none of this is typed.**
`accord/differential/capability_ladder.py` builds the four-row ladder on one denominator and one
prompt condition and computes the three derived ratios the chapter quotes, each a quotient across
four artifacts and exactly the shape of number that gets typed once and never moves.
`accord/differential/frontier_metrics.py` now rewrites its own block of the fact register under a
per-provider, per-model prefix, so a second labeller cannot overwrite the first's numbers in the
register any more than it can overwrite its artifacts. The first version of that function wrote a
chosen subset of fields and deleted by prefix, which removed the Matthews interval bounds the
chapter cites and replaced them with nothing; the deletion cannot be selective, so the write is now
total.

**Verified before publishing:** the two Bedrock artifacts carry pair identifiers and integer labels
only. No note text, no admission or subject identifier, no free text of any kind, and the runner
echoes no exception body, because a client error can quote the request back and the request carries
note sentences.

## C71. The Haiku capability test returns Outcome A, and its outcome entry was pointed at the wrong correction for a day

**Why this entry exists twice over.** It records the outcome of the C67 pre-registration, which had
none, and it records a bookkeeping failure in doing so. The pre-registration's outcome block said
"Full result in CORRECTIONS.md C67", which is the detector-source recovery, an unrelated entry. That
was a stale pointer, and it was then made worse rather than better: it was edited to point at C69,
which by the time of the edit was the label-noise correction. So for one day the file asserted a
specific location for its own result and was wrong about it twice, in the second case confidently.
The experiment labels C67 and C68 were assigned before the correction record reached those numbers,
and nothing reconciled the two sequences. The rule text itself was never edited, which is the
property that mattered; only the pointer moved.

**The pre-registered question.** After C65 withdrew the GPT-4o rows, the capability axis held three
8B labellers between 15.4167 and 23.7500 and one frontier labeller at 4.5000 and 5.3333, with
nothing between. An adversarial review objected that adding a mid-capability point after seeing that
the gap weakens the claim, with no committed prediction, is the defect `tab:m-prereg` exists to
prevent one level up: a claim that cannot be embarrassed by its own next datum is not a finding. The
answer was to commit the prediction first, at `05d9cdd`, before any pair reached the model.

**Outcome A.** Haiku 4.5's G_299 zero-shot over-calling ratio is 6.2500, strictly inside the
required window of 5.3333 to 15.4167. Only `ANTHROPIC_MODEL` changed: same script, prompt,
exemplars, pairs, parser and evaluation as the Sonnet runs. Few-shot is 6.1667, so unlike the
Bedrock 70B of C70 this model is nearly prompt-invariant on this quantity, and the prompt effect
that dominates the 70B is not a property of the measurement itself.

**What it settles and what it does not.** It puts a fourth point on the ladder between the 8B
cluster and Sonnet, so "attenuates with capability" is no longer an interpolation across a gap the
design never sampled. It does not remove vendor or channel, because Haiku shares both with Sonnet;
that is what C70 was for, and the two tests answer different objections. The capability proxy was
fixed in the pre-registration rather than chosen afterwards, neither model's parameter count being
public.

**A second result came with it that was not part of the rule.** Haiku's Matthews correlation on
G_253 is +0.2253 zero-shot and +0.2431 few-shot, both intervals excluding zero, so the frontier
separation is not one model's result. Its label-noise crossings are lower than Sonnet's, 0.1684
zero-shot and 0.1893 few-shot against 0.2467 and 0.0941, so Haiku's zero-shot exclusion tolerates
less instability than Sonnet's and its few-shot exclusion tolerates more. And its ambiguity gradient
splits by prompt, 1.7587 at p = 0.0790 zero-shot against 2.9046 at p = 0.0019 few-shot, which is the
result that took the multiplicity family to twenty in C70 and the headline gradient below correction.

**Still outstanding, and named rather than hidden.** These two runs sent 600 further prompts through
a hosted API. Section 3.9 disclosed one transmission when it was written and now has three to
disclose, and the draft disclosure letter to the chair and data steward has the same gap. Both are
being corrected; neither was correct at the time this entry was written.

## C72. The channel confound is measured rather than disclosed, and it is worth about ten percent

**What C68 left open.** C68 removed the model maker from the over-calling comparison
and left the serving path in it. Its 8B arm was `llama3:8b` served locally and its
70B arm `meta.llama3-70b-instruct-v1:0` served through Bedrock, so capability and
channel moved together across the 2.2619 step it reported. Chapter 4 disclosed that
in a table note and Chapter 6 listed closing it as future work. A reviewer could
have answered the whole capability claim with one sentence: you changed two things.

**Why it was worth a run rather than a caveat.** The mechanisms were plausible and
specific. A hosted endpoint can differ from a local one in quantisation, in default
sampling, in a server-side chat template and in an unseen system prompt. The local
build is a 4-bit quantisation and Bedrock does not publish its serving precision.
Any of those could have produced the step that had been attributed to scale.

**Outcome A, pre-registered at `76b6d69` before any pair was sent.**
`meta.llama3-8b-instruct-v1:0` on Bedrock returns a G_299 zero-shot over-calling
ratio of 21.6667. The rule required at or above 15.4167 for Outcome A; the value
lands inside the local 8B cluster of 15.4167 to 23.7500 and close to its own local
twin at 23.7500. Channel does not account for the step.

**The decomposition, which is what this run actually buys.** Two confounded factors
become three separable ones, each read by holding the others fixed:

| comparison | held fixed | factor |
| --- | --- | --- |
| 8B to 70B, both on Bedrock | family, maker, channel | 2.0635 |
| local 8B to hosted 8B | family, maker, scale | 1.0962 |
| zero-shot to few-shot, 70B | everything but the prompt | 1.6800 |

Capability first, prompt second, channel third. The channel is worth about ten
percent, so C68's 2.2619 was almost all capability and a little serving path, and
the claim it supported stands as written rather than needing rescoping.

**The part that was not anticipated, and is the reason this entry is not just a
confirmation.** The prompt effect is larger than the channel effect, and larger than
some steps in capability. A pre-registration written to ask whether the channel was
confounding capability returned the answer no and, incidentally, that the thing
which should have been controlled first was the prompt. Chapter 4 now orders the
three factors explicitly instead of reporting capability with a prompt caveat
attached, which is what it did before.

**What moved in the thesis.** The capability table gains a fifth row and its note
changes from disclosing a confound to explaining which pair of rows isolates which
factor. The prompt-caveat paragraph is replaced by the three-way decomposition.
Chapter 6's fourth future-work item is struck, because the work is done; the
paragraph introducing that section now records that an item stood there and was
removed by being executed, rather than silently returning to three.

**What it does not license.** The 8B arm is a near-null on every axis but this one:
Matthews correlation $+0.0286$ zero-shot on G_299, interval containing zero. It
contributes a denominator to the capability question and no signal to the detection
question. Capability remains a proxy, ordered by parameter count within a family and
by reputation across families, exactly as C67 fixed it in advance. And quantisation
was named beforehand as the mechanism to suspect if the outcome had been B or C;
neither landed, so the question does not arise, and nothing here measures a
mechanism.

**Verified before publishing:** both artifacts carry pair identifiers and integer
labels only. No note text, no admission or subject identifier, and no exception body
echoed.

## C73. [REVERSED BY C84: the 12-run episode is an independent run set and the total is 1216.4 over 79 runs] The control cost total double-counted the convergence sweep's own budget cell, against this project's own audit

**What was printed.** Chapter 5's cost table and two sentences in Chapter 6 reported the reconciled
control compute as 1216.4 GPU minutes over 79 runs, with the arithmetic shown as
555 + 34.9 + 612.3 + 0.7 + 13.5 and the run count as 48 + 12 + 17 + 1 + 1.

**Why that is wrong.** The 34.9-minute, 12-run entry in the cost log is labelled, in the log itself,
"C44 AG News positive control (3 epochs)". The 555-minute, 48-run entry is the C44 convergence sweep.
That sweep's committed results hold 48 runs at history lengths of 3, 6, 12 and 24 epochs, which is
four budget cells of twelve runs each. So the 12 runs at 3 epochs are that sweep's own 3-epoch cell,
already inside the 555 minutes and the 48 runs, and adding them counts one cell of the sweep twice.
The correct figures are 1181.5 minutes over 67 runs.

**The part that matters more than the arithmetic.** This project's own consistency audit had already
ruled on exactly this question and recorded the answer: that the 34.9-minute entry is a three-epoch
subset of the sweep and the reconciled total is 1181.5 over 67. The register key was then changed to
the larger figure, with the reason recorded in the key as "a separate set rather than a subset", and
the audit that ruled the other way was left on disk unretracted. Two determinations of one question
sat in the repository at once and the thesis printed the one that made the number bigger.

**What that costs, stated plainly.** "Every number is generated, never typed" was satisfied
throughout. A committed script computed 1216.4 and the register carried its derivation. The claim is
still wrong, because the author controls which entries the generator sums, and no gate in this
project checks a generator's set membership against a prior ruling on that set. The same shape
appears in the multiplicity family, where the count is computed from an enumerated list whose
boundary is also chosen. Generation removes transcription error and does not remove judgement error,
and this entry is the clearest instance of that distinction in the project.

**Superseded.** This entry's fix was itself wrong; the register carries 1216.4 and 79. See C84. The text below records what was believed at the time, unaltered with the subset reasoning in the key, the withdrawn
pair is kept under `withdrawn.*` because Chapter 5 quotes the superseded figure to say what it was,
and the three prose sites read the corrected total.

## C74. Six defects in the prose written over the last two days, found by adversarial review

Reported together because they share one cause: material was added faster than the apparatus that
checks it, and every one of them is in text written after the last full review pass.

**The step that removes the vendor was quoted as the step that removes the channel.** Chapters 1, 5
and 6 all cited 2.2619 as the within-family step holding family and maker fixed. That figure is the
local 8B against the hosted 70B, so it varies scale AND serving path together. C72 measured the
clean scale-only step at 2.0635 and the channel-only step at 1.0962, and only Chapter 4 carried them.
Three chapters quoted the confounded number as the clean one. Corrected in all three.

**Two new tables had no caption.** `tab:capability` and `tab:m-transmissions` carried a label and a
note and no caption, which the template requires and which is what puts a table in the List of
Tables. One of the two is the compliance disclosure. Both now have one.

**The ambiguity subsection contradicted itself thirteen lines apart.** One paragraph said the
capability fade "cuts against" the evidential-difficulty reading; a later paragraph, stale text from
the pre-replication draft, said the same evidence was "consistent with" it and opened "Two prompts of
one model" immediately after four tests on two further labellers had been reported. The stale
paragraph is replaced.

**The quantisation confound was in the correction record and not in the chapter.** C72 states that
the local build is a four-bit quantisation and that Bedrock does not publish its serving precision,
so the local-to-hosted pair varies precision as well as channel and 1.0962 bounds the two together.
Chapter 4 described that step as "changing only the serving path" and Chapter 5 repeated the framing.
Both now carry the qualification, and Chapter 4 states that a small net factor is also consistent
with two larger effects that cancel.

**The saturation confound in the gradient was not applied.** The ambiguity odds ratio falls
monotonically as the labeller weakens, and the weaker labellers also flag far more of the set. An
odds ratio compresses toward 1 as a marginal saturates, so part of the fade is arithmetic. This is
the degeneracy precondition the thesis offers as one of its own transferable checks, unapplied to the
thesis's own newest analysis. The section now names it, says the gradient was not recomputed at
matched predicted-positive rates, and scopes the claim accordingly.

**Three counting errors and a completeness claim that had become false.** "Five runs, not one" was
followed by "lagged them by three runs". The opening sentence of the ambiguity subsection attributed
39 false positives to a set larger than the 241 that define them. And `tab:overcalling` asserted
three times, in its text, its caption and its note, that it carries every over-calling ratio in the
chapter, while `tab:capability` reports three it does not. All four are corrected, and the
completeness claim is now scoped to the two evaluation sets it actually covers.

## C75. Checking Drive settles the Hugging Face question, and both of my accounts of it were wrong

**The open question.** C66 recorded that Section 3.5 said the production weak labeller ran locally
while Section 3.9 said a Hugging Face Inference API run covered the full candidate pool and produced
the deployed labels, that both could not be true of the same run, and that the record could not
settle it. The author was asked to check the notebooks rather than the artifacts.

**What the notebooks show.** The Inference API client is imported by exactly one notebook in the
project, a cross-domain safety and hallucination screen. That notebook contains no occurrence of
either 1679 or 1425. Its clinical arm draws `sample(n=50, random_state=42)` and its recorded
execution log reads `Running meta-llama/Meta-Llama-3-8B-Instruct on clinical (50 examples)`, followed
by a result. The weak-label artifacts, by contrast, carry runtime-tagged model keys of the local
serving convention. So the 1,679-row weak labelling did not pass through a third-party API, and the
Inference API exposure is 50 sampled pairs.

**Both of my accounts were wrong, in opposite directions.** The first said 1,679 rows, overstating the
exposure by a factor of about thirty-three and, worse, asserting that the detector's entire training
supervision came through a prohibited channel and could not be withdrawn. The second, written after
the first was withdrawn, said the run returned HTTP 401 and produced no output. The execution log
shows it completed and reported a miscalibration figure, so the second account understated it to
nothing. The correct figure is 50 pairs, transmitted, with output.

**What this changes in the thesis, and it is not small.** Section 3.9 built an argument on the first
account: that one departure could be withdrawn and the other could not, that the asymmetry should be
uncomfortable, and that no principled distinction separated them. That dilemma does not exist. Only
the original GPT-4o comparison used a prohibited channel, and its results are withdrawn, so no
result this thesis reports is downstream of a prohibited channel. The safety screen is a real
transmission of 50 pairs, is disclosed as one, and supports no number in these chapters.

**Why the first account survived as long as it did.** The prefix `hf_` in this project's notebooks
denotes the heart-failure cohort. It was read as Hugging Face. A cohort label was read as a channel
label, and a compliance section was then written around the misreading. The lesson is the one this
thesis keeps relearning: a filename is a hypothesis about provenance.

## C76. The multiplicity family omitted a fourth labeller's two tests, which had already been computed and registered

**What was stated.** That the gradient was put to "four replication attempts on two further
labellers", that six Fisher tests existed, that the family held twenty tests, and that the Bonferroni
threshold was 0.05/20 = 0.0025.

**What exists.** Four labellers were asked, not three. The C72 run on the hosted 8B computed the same
ambiguity gradient in both prompt conditions, its sixteen gradient keys are in the register with the
other hundred and ten that run wrote, and neither test was added to the family. Eight Fisher tests
exist. The family is twenty-two, the threshold 0.05/22 = 0.0023, the independence value 0.6765, and
the dependence groups nine.

**The same shape as C68, one level further in.** C68 recorded a family understated by the two tests
added most recently. This is a family understated by the two tests added most recently, in the entry
written to prevent exactly that. The register removed the hand-typed count and did not remove the
hand-maintained membership, which is the distinction C73 makes about generated numbers: generation
removes transcription error and not judgement error.

**The omitted data argues for the chapter's own claim, which is why it belongs in it.** The hosted
8B's odds ratios are 0.8224 zero-shot and 1.0779 few-shot, the lowest of the eight and the only one
below 1. That corroborates the capability fade the section reports. It also sharpens the saturation
confound the section concedes: that labeller flags about seven eighths of the set, not the two fifths
the section previously named as the extreme, so the compression toward 1 is more severe than stated.
Both now appear in the chapter.

**What did not change.** Holm still rejects two, and the two tests that clear Bonferroni are the same
two. The conclusion is unchanged and every printed threshold moved.

## C77. The provenance gate resolved numbers across two orders of magnitude, in both places it is implemented

**What the gate did.** Pass 2 of the verification harness indexes every registered value and every
string form of it, and it added a times-100 form unconditionally. So a registered precision floor of
0.04 made the literal 4.00 resolvable, a registered 0.61 made 61 resolvable, and a registered
over-calling ratio of 2.0548 made a 205-row count resolvable. The same code shape was in
`trace_claims.py`, which generates the appendix that claims to trace every number in the thesis to
its source, and there it printed the collision on the page: a row reading 4.00 against
`decontam_gold.aggregate.precision.min`, whose value is 0.04, with no indicative marker.

**Why this is the worst class of defect in this project.** Every number in this thesis is generated
and every number is gated, and the gate has been reporting success while accepting attributions that
are off by a factor of a hundred. The appendix built to certify provenance failed at a provenance
question, and its own self-flagging did not fire because the row resolved to exactly one candidate.
A reader who trusted the appendix was misled by the instrument, which is worse than having no
instrument.

**Fixed in both places.** The percent form is now offered only for values that lie in the unit
interval, where reading one as a percentage is meaningful, and only at the precision a percentage is
written with. In the appendix generator the gate is stricter still: a percent form is consulted only
when the literal is actually written as a percentage in the source, detected within a window that
stops at a table cell boundary so a percent sign two cells away cannot leak onto an unrelated count.
The attribution rule changed with it. A row is now marked indicative whenever more than one
registered quantity shares its value, rather than when the key's name happens not to share a word
with the surrounding sentence, which is how a genuinely ambiguous row came to be printed as
definitive.

**What it exposed.** One number in the thesis was resting on the collision and is now registered from
its real source: the 205-row validation pool, which is 1,388 minus 1,183 and had been matching an
over-calling ratio. Several appendix rows that had been silently wrong now show as untraced, which is
the correct state for a number whose source the tooling cannot establish.

**What it does not fix.** Pass 2 remains a digit match. Two quantities that genuinely share a value
are still indistinguishable to it, 97 rows of the appendix are still marked indicative for that
reason, and no amount of tightening makes a value comparison into a provenance proof. The honest
description, which the thesis now gives, is that the gate is a floor.

## C78. The compliance section cited a 2023 document as naming 2026 models, and the exemplar "range" is two measurements that cannot both be true

Two findings from a committee round, recorded together because both are the same error in
different material: a claim stated as sourced when it is a reading.

**The guidance claim.** Section 3.9's transmission note read "All routes here are named in the
guidance of 18 April 2023 as acceptable for credentialed data." A document of April 2023 cannot name
Claude Sonnet 5 or Claude Haiku 4.5, both released in 2026, and the guidance it refers to was
written about a different vendor's service. So the sentence asserted as a citation what is in fact
this author's application of the guidance's criteria to routes that post-date it. That is the error
this thesis exists to describe, committed in the section whose subject is disclosure, and the
project's own disclosure letter was more careful than its thesis: the letter says the reading is
unconfirmed and the thesis did not.

Rewritten to say what it is. The guidance suggests alternatives, the routes were selected against it
before use, the Bedrock selection is pre-registered, and the reading is a reading with two named open
questions put to the chair and the data steward. The thesis asserts no answer.

**Why that sentence outranked everything else open.** The four Claude rows are the only evidence in
this thesis that its twelve-positive evaluation can resolve an effect at all. If the channel is ruled
out they are withdrawn on the principle that withdrew the original comparison, and RQ4's answer
reverts from demonstrated sensitivity to a calculation. That dependency runs through four chapters
and nothing said so. Section 5 now states it, names what would still stand, and says plainly that the
remaining position is weaker than the one the chapter argues from.

**The exemplar range.** Chapter 4 reported "the honest range is therefore 0 to 0.4745", from a
reconstruction over 137 rows finding 65 wrong and an exact key join over 23 rows finding none wrong.
It is not a range. If the reconstruction's rate were true and joinability were independent of
correctness, the probability of 23 joinable rows all being correct is 0.5255 to the 23rd, about four
in ten million. The two measurements are mutually exclusive unless joinability is confounded with
correctness, and that confound was the load-bearing assumption behind keeping both numbers without
ever being stated. It is plausible, because a row is joinable when the bookkeeping the selector
defect corrupts was intact, and it is untested. The section now states the arithmetic, names the
assumption, names the test that would settle it, and says that one of the two numbers is wrong and
this thesis does not know which. The conclusion stops quoting the maximum as the actionable figure.

**The comparison the conclusion was not making.** Two committee members disagreed on whether this
thesis needs reframing and agreed on this: the closing chapter asserted that a standard-shaped
pipeline does not detect these contradictions while omitting the number that complicates it. On the
253 non-ambiguous pairs the detector reaches -0.0070 and a term-frequency baseline -0.0436, both
containing chance, and two frontier labellers given one zero-shot prompt and no training reach
+0.3189 and +0.2253 with intervals excluding it. Those systems saw the same rows, exemplars, base
rate, twelve positives and single annotator, so none of rarity, prevalence, sample size or the
exemplar defect can be what made the pipeline fail. The conclusion now leads with the comparison,
prints the precision and the false-positive contrast, and states the recommendation that follows:
before building a pipeline of this shape for a rare clinical event, prompt a frontier model on the
evaluation set and make the pipeline beat it.

**One thing was rejected rather than fixed.** An adversarial round proposed retitling the thesis
around a pipeline "dominated by the zero-shot teacher it distils from". The premise is false. Section
3.5 records that the production labeller was Llama 3 8B run locally, and the detector was trained on
its 1,425 labels; the frontier models entered later as a replacement for withdrawn baseline rows and
were never in the supervision path. The retitle would have described an experiment that does not
exist. The observation it was built on is real and is now in the conclusion; the framing it proposed
is not adopted.

**And one overclaim beside it.** Chapter 4 said the evaluation's sensitivity was "demonstrated" by
one frontier row above the minimum detectable effect. One observation above a threshold set at eighty
percent power is consistent with that power, not a demonstration of it, and the word is now
supported. The same passage now states the identity that makes a balanced accuracy comparable to an
area under a curve, which a reader who did not supply it would file as a category error.

## C79. The family was short by the only rejections on the clinical task, and the correction was applied to tests but not to intervals

**What was outside the family.** Section 4's multiplicity subsection stated twenty-two hypothesis
tests and presented the register as exhaustive by construction. Four more existed and had been
computable from committed cells throughout: Mann-Whitney tests on the frontier labels over the same
253 rows that Table 8 runs five Mann-Whitney tests on. Sonnet 5 zero-shot returns
p = 4.2 times ten to the minus seven, Haiku 4.5 zero-shot p = 0.00035, and both few-shot arms clear
as well.

**Why the omission runs in the worst direction.** These are the only rejections in this thesis that
concern the clinical task directly. They say the frontier labels separate the pharmacist's confirmed
cases from the rest on the rows where the systems this pipeline built do not. Leaving them out kept
the strongest results in the chapter outside the thesis's own correction, in the subsection whose
stated purpose is to prevent exactly that. It is the third time a family presented as exhaustive has
been short by tests already computed, after C68 and C76.

**What it changes.** k goes from 22 to 26, the Bonferroni threshold from 0.0023 to 0.0019, the
independence value from 0.6765 to 0.7365, and the dependence groups from nine to ten. Six tests now
clear correction rather than two, and four of the six are the frontier tests. Every one of them
clears the new, stricter threshold, so nothing concluded here changed. That is luck rather than
rigour and the chapter says so: a family that grows by four tests which all clear is not evidence
that the accounting was sound.

**The asymmetry that mattered more than the count.** The correction is applied to p values and not to
interval estimates, and two load-bearing claims in this thesis live in intervals. Read at 0.05/26
rather than at 0.05, Sonnet's Matthews interval on the 253 rows still excludes zero and Haiku's does
not: its lower bound moves to -0.0233. So the claim that two independent labellers clear the
evaluation's stated floor, which is what converts RQ4's null from uninformative to informative, could
not rest on the two intervals. It now rests on the two Mann-Whitney tests, which do clear correction.
Both readings are reported. A family that corrects its tests and not its intervals is applying its
discipline where it is safe, which is the reporting defect the subsection exists to describe, and it
was doing so on the one claim that most needed the discipline.

**Three smaller things from the same round.** The abstract read the 8B over-calling range off the
299-pair set and the frontier range off the 253-pair set inside one clause governed by "on the
annotated set", which is the conflation the conclusion warns cost this project a central claim. Both
ranges are now read off the 299-pair set, five to six rather than four to five, and the abstract
states the comparison the conclusion now leads with.

That conflation survived because the verification harness scanned `msu_ch*.tex` only, so the
abstract, the title page and the declaration were outside every pass. The prose passes now scan the
front matter; the structural pass does not, because front matter correctly has no chapter heading.
This is the third gate in this project found to be reporting success while scanning nothing or
enforcing a withdrawn rule, after the TODO glob and the caption convention. Note what the extension
did not achieve: the abstract spells its numbers as words, because the template forbids
abbreviations, so a digit gate could never have caught this particular conflation. The gate is wider
and the defect was still found by reading.

## C80. Every table and figure re-derived against current artifacts, and the results chapter gains the operating point it never reported

**The audit.** All 28 tables and 14 figures were checked cell by cell against the register and the
artifacts behind it. 574 numeric cells in table bodies; every one resolves. No table lacks a caption,
no figure caption sits below its graphic, every included figure has a PDF. What follows is what the
sweep found wrong.

**The results chapter did not contain the result.** Chapter 1 calls the detector's operating
behaviour "the cleanest single expression of the failure" and Section 5 rests the claim that the
system should not be deployed on it. Its precision of 0.0424, recall of 0.7778, expected calibration
error of 0.8015, kappa of -0.0105 and confusion cells appeared in Chapters 1, 5 and 6 and **nowhere in
Chapter 4**, which reported the ranking metrics and stopped. The correction record had even
enumerated the omission, saying those figures are "reported in Chapters 1, 5 and 6", without treating
the absence of the results chapter as a defect. A reader auditing the deployment claim against the
chapter that should support it found nothing to audit. Table 9 now gives the operating point per seed
and as a mean, every cell from the committed aggregate.

**Two things that table caught on the way in.** Its standard deviations are computed at ddof = 0, not
the sample convention, and a first draft of the note claimed ddof = 1. That would have mixed
conventions inside one chapter, which this thesis treats as a defect and which Tables 8 and 12 are
careful to keep apart; the note now states the convention, gives what the other would produce, and
says the two are never compared. And a sentence describing the detector's behaviour on the clear
negatives said "between 216 and 225 of 241" when the false-positive counts are 216, 210 and 207. The
range is 207 to 216. That number was mine, derived while writing, and it is the kind of arithmetic
the register cannot check because it never reached the register.

**The frontier table was missing the column its argument is read against.** Balanced accuracy is what
Section 4 compares to the minimum detectable effect, and it appeared in no table at all; Haiku's
value was stated only in Chapter 1. Table 6 now carries a balanced-accuracy column and both Haiku
rows. Its detector row also printed an empty over-calling cell, which removed the row that makes the
column's point: at 234 flagged against 12 confirmed the detector over-calls by 19.50 times, more than
any labeller in this thesis.

**Figures.** All 15 live figures regenerate from their committed generators. Three of the seven
retired ones do not: two import palette names the style module no longer defines, and one is named in
its generator but no longer emitted. Their PDFs are artifacts of a palette and a figure set that no
longer exist, which is a reason to keep them out of the thesis rather than a reason to repair the
generators. The invariant checker now records which retired figures are unreproducible and fails only
if one of them is ever included in the document, because a live figure that cannot be rebuilt is a
defect and a retired one that cannot is a fact. Every figure clears its coverage and chroma floors
with none under.

**One dangling reference, introduced by this round.** A note cited a table label that does not exist
in the document, invented while writing the ddof comparison. Pass 7 caught it before the build
shipped, which is the one thing in this entry that worked as designed.

## C81. The primary sources arrive, and they overturn a correction rather than the original claim

The author supplied the PhysioNet documents themselves: the Credentialed Health Data Use Agreement
1.5.0, the Credentialed Health Data License 1.5.0, the credentialing record, and the news archive
containing the guidance of 18 April 2023. Three things in this thesis were wrong against them, and
two of the three were wrong because of a correction rather than in spite of one.

**The clause citation was right, and I broke it.** The Use Agreement has ten numbered undertakings:
3 is not sharing access, 9 is contributing code for disseminated results, 10 is termination. The
thesis originally cited Clause 9 for the code obligation, which is correct. An adversarial review
reported that Clause 8 was code contribution and Clause 9 termination, and that reading was applied.
It was a reading of the wrong document. The License 1.5.0 is a separate instrument with nine clauses,
where code contribution is 8 and termination is 9. The two number the same obligation one apart, so a
clause number means nothing unless the instrument is named. Reverted to Clause 9, and the section now
names which instrument it counts, which is the fix that stops this recurring.

**The guidance names Anthropic Claude, and my hedge was worse than the overclaim.** The original text
said all routes are "named in the guidance of 18 April 2023 as acceptable", which a committee round
called an unsourceable claim on the ground that a 2023 document cannot name 2026 models. I rewrote it
into two open questions. The guidance itself says: "If you are interested in using the GPT family of
models, we suggest using one of the following services", and its fourth entry is "Anthropic Claude.
Claude does not use your prompts or its responses as data to train its models by default, and routine
human review of data is not performed." Amazon Bedrock is its second entry. So both routes used here
are named, the first of my two open questions asked about a distinction the document does not draw,
and the section is now built on quotation rather than on characterisation in either direction.

Two details in the same text are worth more than the headline. The guidance requires an opt-out form
for Azure and asks for none for Anthropic, so the absence of an opt-out here is the arrangement the
document describes rather than a gap. And the ground it gives for Bedrock is that after a private copy
of a base model is made, data is not shared back to that base model for training; the thesis and the
runner both attributed to the guidance a different ground, that inference runs inside a tenancy the
licensee controls, which is the author's reasoning. Both now give the document's ground and say which
was the author's.

**What is still open is narrower than any previous version claimed.** The guidance says Claude does
not train on prompts "by default", and whether a default should have been supplemented by a
zero-retention arrangement is not something the document answers. And it names a vendor and a service
rather than model versions, so reading a 2026 Claude model as the same named service is the natural
reading and remains a reading. Both go to the chair and the data steward, and the letter now quotes
the guidance rather than asking about a distinction that does not exist in it.

**The lesson, which is the same one twice.** A correction is not self-justifying. An agent reading the
License produced a confident, specific, wrong clause map, and it was applied without checking the
instrument the thesis actually cites; a committee round produced a correct objection to an overclaim
and the repair overshot into asserting uncertainty the source does not support. Both were avoided the
moment the primary document was read, and the primary document had been available throughout. The
thesis's own standard, that a claim is checked against the artifact rather than against a description
of it, applies to its corrections as much as to its results.

**No date was affected.** The credentialing record shows an application, agreement and decision date
of 23 February 2026. The thesis states no signing date anywhere, so nothing printed depended on it.

## C82. Two prohibited channels, not one; and the storage claim was true and not responsive

A compliance audit against the primary documents, which are now in hand, found four things wrong in
the ethics section and one in the unsent disclosure letter. All four were errors of my own making in
the two rewrites of that section.

**The safety screen was prohibited too.** The section said "Only one channel in this project's history
was prohibited at the time it was used." The guidance prohibits "sending it through APIs provided by
companies like OpenAI", which names a class of service and not a single vendor. The Hugging Face
Inference API is such a service, it is not among the four the guidance suggests, the guidance predates
the run by years, and the run transmitted 50 pairs and completed. So there were two prohibited
channels. An earlier version called the screen a departure and stopped short of calling it prohibited,
which is a distinction the document does not support. What survives unchanged is the part that bears
on the results: the GPT-4o rows are withdrawn, the screen supports no number this thesis reports, and
no reported result is downstream of either channel.

**The storage claim was scoped to the git tree, which is not the claim a reader takes from it.** The
section said note-derived intermediate files stay outside the version-controlled tree. True, and
non-responsive: a file joining note sentences to weak labels was held on a Google Drive mount and read
from there by an analysis notebook running in Colab, and that notebook is committed and mounts the
drive on its second cell. The exposure is established by this repository, not disclosed against it.
Raw tables were also held in a local downloads folder. Both are now stated, with the remediation, and
with the open question of whether the institution's cloud agreement covers credentialed data put as
two questions, storage and compute, rather than one. The section now says explicitly what it does not
claim: that custody never left local storage.

**The code-release gap count was two and is three.** The notebook behind the safety screen is also
absent, so the 50-pair figure rests on the same footing as the two weak-supervision numbers: stated
and not reproducible from the release.

**The letter understated its own ledger by 600 prompts.** It listed four runs and 1,920 prompts while
the thesis table lists five and 2,520. A previous correction to this letter replaced the paragraph
about the guidance and left the table and total above it untouched, which is a partial fix presenting
as a complete one. Sending a letter that undercounts your own disclosure, to the audience that has
already raised a charge, is the worst available version of this document, and it is the third time
this count has been wrong in this file.

**What the audit confirmed rather than found.** Nothing in the repository is a breach at present. No
tracked file holds an identifier value, note text, or a credential, and none ever has in history. The
only committed tabular data is two public benchmark files and nine aggregate CSVs that are one-way
marginals of the form variable, category, count. The sixteen published cells of four observations or
fewer fall under the stated policy, and the published files match that policy exactly.

**One claim in this thesis cannot be checked from this repository, and that is now said.** The
statement that the 1,679-row weak labelling ran against a locally served model is supported by
artifacts that live outside version control, by the absence of any committed script that sends
anything to a non-local host, and by a correction entry. It is an attestation rather than a
verifiable fact for a reader holding only the release, and the clause 9 paragraph already says those
numbers are not reproducible.

## C83. No fabrication found, and eight over-attributions corrected

The author asked directly whether anything in this thesis is fabricated. Four independent read-only
audits ran against the artifacts and the primary literature. The answer is no, and the qualifications
are recorded here rather than summarised away.

**Citations: nothing fabricated.** Fifty-seven of the ninety-three bibliography entries were checked
against primary records, Crossref, arXiv, the ACL Anthology, PMLR and the NeurIPS proceedings. Every
entry checked resolves to a real work with the stated title. No entry was found not to exist, no
citation is attached to a claim its source contradicts, and there is no citation anywhere to a
personal communication, a manuscript in preparation, or any other unverifiable source. The two
entries most exposed to being inventions, a 2026 preprint and a 2025 Findings paper, were verified
and the quantities this thesis reports from them are exact rather than rounded favourably.

**Eight over-attributions, all corrected.** These are scope slips rather than fabrications, and each
put a position in a source's mouth that the source does not hold. Saito and Rehmsmeier were cited as
recommending that both curve summaries be reported; they argue the precision-recall curve should be
primary at high skew, and reporting both is this thesis's own choice, which is now said. Powers was
cited among those recommending the Matthews correlation; he advocates informedness and markedness and
the coefficient is their geometric mean, which the literature chapter already said correctly and the
results chapter now says too. Lipsitch and colleagues formalise negative controls only, which the
literature chapter disclosed and five other passages did not; all five now carry the disclaimer that
the positive direction is this thesis's adaptation. A cross-reference sent the reader to the survey of
Chapter 4 for an absence that survey does not measure, and is removed. Three absence-of-literature
claims stated as universals are now stated as absences noticed in reading, with the fact that no
systematic search was run for them. And the count of annotation tasks in one cited study was given as
many where the study reports four.

**A quoted primary source had no locator.** The guidance of 18 April 2023 is quoted verbatim three
times and the compliance argument turns on its wording, yet it carried no reference a reader could
follow, the bibliography being frozen at ninety-three entries by the author's instruction. Its title
and address are now given in the note.

**Numbers: nothing fabricated, and the provenance labels were wrong in eighty-five places.** Every
artifact the register names was opened and checked for the value it claims to hold. Of 1,966
checkable keys, 1,881 held their value in place. Of the remaining 85, none turned out to be invented.
The architecture ablation chain was verified end to end: each arm mean equals the mean of its own
per-seed losses, each per-seed loss equals the final validation loss of its named run in the upstream
artifact, twenty-one of twenty-one, and each reported delta equals arm mean minus vanilla mean, six of
six. The clinical-task figures are the mean and sample deviation over named runs, and the two arms do
not share a seed set, which the results chapter already discloses. The register cited an upstream
file where the aggregate lives one level down; the labels are corrected and the derivations now state
their operands.

**One class of claim cannot be checked from the release, and that is now said where it matters.**
Several quantities trace to artifacts held in credentialed-only storage, so a reader holding the
public release cannot verify them. The code-release gaps are disclosed, now as three rather than two.

## C84. C73 was wrong and is reversed; the premise of C68 was arithmetically false; and the gate that missed both is fixed

A fabrication hunt recomputed every derivation and every table cell against the artifacts. It found no
invented number. It found four numbers that were wrong, three of them introduced by corrections in
this record rather than surviving in spite of them, which makes this the most important entry in it.

**C73 is reversed. The control cost is 1,216.4 GPU minutes over 79 runs.** C73 withdrew that total and
substituted 1,181.5 over 67, on the premise that the 12-run, 34.9-minute public-task episode was the
convergence sweep's own 3-epoch cell counted twice. The artifacts refute it. The scaling run set holds
12 runs totalling 34.95 minutes; of the three that match a convergence entry on arm, size and seed,
all three differ in accuracy and all three differ in runtime. They are distinct runs. The episode is
an independent run set, the cost log's own caption said so, and C73 overruled the caption on a
reading the data does not support. The original figure stands and the withdrawal is withdrawn.

Worse than the arithmetic: the thesis then printed three different reconciliations of one quantity,
1,216.4 over 79 in Chapter 1, 1,181.5 over 67 in a table whose own rows sum to 79 and 1,216.4, and
1,181.5 over 79 twice more. All four sites now read 1,216.4 over 79.

**The premise of C68 was arithmetically false.** That entry says the understated count of fourteen was
the count under which the headline gradient cleared correction. At fourteen the threshold is
0.05/14 = 0.0036, and the zero-shot gradient is 0.0046, which does not clear it. The zero-shot
gradient has cleared no Bonferroni threshold at any count this thesis has ever stated. What the
understated count protected was the few-shot gradient at 0.0029, which cleared fourteen and sixteen
and clears neither twenty-two nor twenty-six. The direction of the original error is unchanged, the
count was still understated, and the claim it protected was the weaker of the two. Chapter 4 now says
this in place of the false version.

**Two further wrong labels in Chapter 4, both mine.** The four frontier Mann-Whitney tests were given
as running to 0.0019; their largest is 0.0007, and 0.0019 is the Bonferroni threshold and also the
fifth test's p value, quoted one sentence later. And the operating-point note said precision sits
0.0051 above the confirmed rate minus its binomial standard error, which is a garbled construction of
a simple fact: precision is 0.0051 below the confirmed rate, and that margin is smaller than the
standard error. Chapters 1 and 6 had it right and only the table note was wrong.

**The gate that let the cost figure through is fixed.** Values retracted by a correction were kept in
the register so the record could quote them, under a withdrawn prefix, and the provenance pass made no
distinction: a retracted number remained resolvable and therefore printable. That is precisely how a
withdrawn cost total passed as a chapter headline. The pass now excludes that prefix. Two values the
thesis legitimately quotes in order to state what was withdrawn, the bounds of the discarded
label-noise interval, are moved to a separate prefix for quoted historical figures, because quoting a
past number to describe its retraction is a live citation and using one as a current result is not.

**One register value was not the quantity it named.** The probability that 23 joinable rows are all
correct under the reconstruction's rate was stored as the integer 4, the leading digit of the prose
phrase "about four in ten million". It is 3.7426 times ten to the minus seven, and storing 4 made the
bare literal 4 resolvable anywhere in the thesis. Corrected.

**What the same audit confirmed.** Fifty-seven of fifty-nine stated derivations reproduce exactly,
including every ablation delta, every nats conversion, the Welch statistics to four decimals, the
minimum-detectable-effect search, and both reproducible bootstraps. Twenty-seven of twenty-eight
tables are arithmetically clean cell by cell, the exception being the cost table above. Every word-form
and percentage quantity in the abstract and the chapters checks out, including that the abstract reads
both of its over-calling ranges off the same denominator. The multiplicity arithmetic is correct at
every count.

## C85. The four frontier tests used the wrong instrument, and it was anti-conservative in every case

A committee reader found the defect this thesis exists to describe, inside the strongest result in it.

**What was wrong.** The four tests that establish the frontier labels separate the pharmacist's
confirmed cases from the rest were computed as Mann-Whitney U on the binary label vector. That is the
wrong instrument. The comparison is a two-by-two table with fixed margins; the vector is binary, so
roughly four fifths of the pairs a rank test forms are ties; and the normal approximation it then uses
is anti-conservative on such a table. It was anti-conservative in all four cases, by factors between
four and fifty-four, and every one of the four errors ran in the direction that favoured this thesis.
The exact conditional test for each of those tables had already been computed and stored in the same
artifact the register reads.

**What changes.** One verdict, and it is the one that mattered. Sonnet 5 under the few-shot prompt
returned 0.0007 under the rank test and clears the corrected threshold; it returns 0.0042 under the
exact test and does not. So five of twenty-six tests clear correction rather than six, and three of
the four frontier tests clear rather than four. The claim that two independent labellers separate the
classes at a significance surviving correction still stands, because both zero-shot rows clear, but it
now rests on the two exact tests rather than on four rank tests.

**Why this entry is the most damaging in the record.** The thesis's subject is procedures chosen where
they happen to be favourable, and this is one, in its own multiplicity subsection, in the tests it
added two rounds ago specifically to stop a family being understated. The generated-not-typed
discipline did not catch it, because a script computed the wrong statistic faithfully. Nor did the
register, which stores whatever it is handed. What caught it was a reader asking which test the data
called for.

**Also corrected in the same pass.** The fifth clearing test, Haiku's few-shot ambiguity gradient,
was printed at the same rounding as the threshold it clears, 0.0019 against 0.0019, so its margin of
three parts in a hundred thousand was invisible; it is now stated, with the fact that it would not
clear at twenty-seven tests.

## C86. Five sentences left false by the compression, and an asymmetry the ethics section did not run on itself

The deletion pass that cut chapters 3, 4 and 6 orphaned qualifiers. A harsh reading found five.

**A false confession survived.** The ethics section said one breached run's output was withdrawn
because the thesis could do without it while another's was retained because it could not, and named
retraining the detector as the remedy. Both sentences were written when the deployed weak labels were
believed to have come through the second channel. C75 showed they did not; the paragraph around these
sentences was deleted and these were kept. As they stood, a reader was told the detector's training
supervision came through a prohibited channel and was kept anyway, which is false and is the worst
available misreading of this project. Withdrawn, with the reason.

**A remediation claim the repository falsifies.** The section said the affected files were being moved
under credentialed-only local storage and that the analysis scripts now read from that location. Seven
committed artifacts still mount the drive, six notebooks and the detector's own training script, which
also writes its checkpoints there. The claim is now what is true: the move is incomplete, and the
files that make it incomplete are named rather than described.

**A completed act asserted for the second time.** C83 corrected two sentences saying open questions
had been put to the chair and the data steward, because the letter is an unsent draft. It corrected
the methodology instance and missed the discussion instance. So the correction record contained a
false completion claim about a fix for a false completion claim. Corrected.

**A uniqueness claim contradicted four lines later.** The section said the GPT-4o run was the only
departure of its kind, and then disclosed a second of exactly that kind. C82 changed the framing
everywhere except this sentence.

**The abstract miscounted the pipeline and the frontier range.** It said three stages where the
methodology and discussion chapters both say four, and gave the frontier over-calling range as five to
six without naming a prompt, when one of the four frontier runs sits at 4.5000. Both fixed, and the
abstract is back under the word cap at 247.

**One asymmetry is recorded rather than resolved.** The section escalates the safety screen to
prohibited on the ground that the guidance names a class of service and not a vendor. The same
argument is available against hosted notebook compute, and there the section stops and defers to the
institutional data steward. The standard that convicts one channel is not run on the other. That is
noted here because the deferral is honest but the asymmetry is real, and a reader is entitled to see
it named by the author rather than to find it.

---

## C87. The annotation-transfer question, audited to artifacts and closed by the author

The question C33 opened and C64 closed on recollection was reopened and audited a third time,
this time against file metadata rather than against either the correction record or memory. Four
things were established that neither earlier pass had.

**The mail search had been failing for the wrong reason.** C33 recorded expired connector tokens.
The tokens are valid. The connectors are authenticated as the author's personal account, and the
drive and mailbox that would hold a cover email belong to the university account. A query of the
personal account's sent mail for the whole window 2026-03-20 to 2026-05-05 returns four messages,
three of them to the author's own second university address and one to a faculty member, none to a
clinician and none carrying a spreadsheet. So nothing was sent from the personal account, and the
university account is not reachable from the audited environment at all. C33's stated reason is
corrected in place above.

**The returned sheet was edited in a browser, not written by a script.** `annotation_pharmacist_complete.xlsx`
presents as script-generated: creator `openpyxl`, created `2026-03-27T19:21:54Z`, no
`lastModifiedBy`. That reading is wrong. Its `xl/workbook.xml` carries a
`GoogleSheetsCustomDataVersion2` extension with a round-trip checksum, every one of its twelve
archive members is stamped `2026-04-28 12:21`, and the file carries a drive item identifier as an
extended attribute. Google preserves the original creation stamp and drops the modifying author,
which is why the openpyxl fingerprint survived and misled the earlier read. The labels were
entered into a drive-hosted spreadsheet through a web browser and saved back on the closing day of
the annotation window.

**The protocol does not say what C33 quoted it as saying.** A keyword scan finds no occurrence of
remote, unsupervised, in person, send, email, download, delete, retain or return. C33's
"remote and unsupervised per the protocol" was an inference presented as a quotation, and is
withdrawn. What the protocol does contain is an instruction in the second person not to share
"the CSV file or any pair contents", which reads as presupposing custody.

**Author's determination, 2026-09-12:** the annotation was done on the author's computer, not the
annotator's. That is consistent with every artifact above rather than in tension with any of them:
a spreadsheet hosted on the author's own drive and edited through a browser on the author's own
machine produces exactly the signature found. The browser round-trip establishes where the file
lived, not who typed into it, and no send record, sharing grant or named recipient exists in
anything reachable. The methodology chapter's account stands unchanged.

**Two sentences changed as a result, neither of them that account.** First, the compliance
paragraph said "nothing was retained afterwards" of an arrangement in which the sheet itself is
retained and only the annotator retained nothing. In context the following sentence disambiguates
it, but the bare clause invites an auditor who finds the file to read a contradiction where there
is none, so it now reads "the annotator retained nothing afterwards." Second, the storage
disclosure named a drive-held file joining note sentences to weak labels and did not name the
annotation sheet, which lived on the same drive and carries the same restricted content. It is now
named there, with the metadata that establishes it, on the same ground the rest of that paragraph
is written: a fact one search of the artifacts would surface should be disclosed by the author
rather than discovered against him.

**Recorded against recurrence.** Three passes over one question produced two wrong answers, and
both wrong answers came from reading a summary instead of an artifact. C33 inferred a transfer from
a session schedule. C64 reversed it on recollection and cited nothing. The disposition that held is
the one where a claim about what happened was checked against file metadata and then put to the one
person who was there. The order matters: audit first, then ask, because the audit is what made the
author's answer checkable instead of merely authoritative.

---

## C88. Three free computations run, and the test family grew to twenty-seven

Three measurements that cost nothing had been outstanding because no one had run them,
not because anything blocked them. Running them changed one threshold, cost one
rejection, withdrew one number that had no generator, and added one disclosure.

**The weak labeller's own agreement was never measured.** Every statement this thesis
makes about the deployed weak labeller is a ratio of marginals: how often it says yes,
divided by how often the pharmacist does. That says nothing about whether it says yes to
the right pairs. Its Matthews correlation with the pharmacist on $G_{253}$ is 0.0117. The
teacher is at chance. It recovers 11 of the 12 confirmed contradictions and brings 200
false positives with them.

The figure is a bound rather than a measurement, and the methodology chapter's own
join-feasibility finding is why: the weak-label file and the gold file share no
identifying column, and row order carries no alignment, matching on rule at 95 of 299
against 94.3 expected by chance. The join used is a reconstruction through a
nursing-side key and leaves 19 of 253 pairs holding two conflicting weak labels. 0.0117
is computed on the 234 the join determines; resolving all 19 as positive gives 0.0059 and
all 19 as negative gives 0.0496. The interval is [0.0059, 0.0496] and every value in it
is at chance, so the ambiguity limits the precision and not the direction. It is written
into Section~4 as a bound with the interval attached, not as a four-decimal measurement.

This is the most load-bearing of the three. The thesis argues the pipeline fails rather
than the task being impossible, and a teacher at chance is the strongest evidence for
that reading: a detector distilled from supervision uncorrelated with truth cannot
acquire a correlation, so the detector's chance-level ranking becomes the predicted
outcome instead of a surprise.

**The detector's threshold was an open objection.** Table tab:operating reports the
configured 0.5, and a reader is entitled to say that a threshold is a free parameter and
a better one might expose signal. The answer is the best Matthews correlation attainable
at any cutoff, chosen with the labels already in hand, which no prospective procedure
could reach. On $G_{253}$ it is 0.0554 at a cutoff of 0.4422 on the contradiction
probability, against -0.0706 at 0.5; on the harm score it is 0.0631 at 0.5563 against
0.0561. The best cutoff that exists moves the correlation from roughly zero to roughly
zero.

**A disagreement between two artifacts, disclosed rather than smoothed.** Those figures
come from the released score file, the same file behind the AUROC results, and not from
the three-seed run behind tab:operating. The two disagree about the direction of the
degeneracy at 0.5: on the released scores the contradiction probability puts 0 of 12
positives and only 23 of 241 clear negatives above the threshold, a near-constant
negative, while every seed behind tab:operating is a near-constant positive. Both are
degenerate and neither discriminates, so no argument changes, but a reader who recomputes
the operating point from the released file will not reproduce that table. It is now said
in the chapter. This thesis does not resolve which run to prefer.

**The untested assumption is tested, and the family grew.** Chapter 4 kept two
irreconcilable wrong rates for the exemplar defect, 0.4745 from a reconstruction over 137
rows and 0 from an exact key join over 23, and stated in the text that keeping both rested
on an assumption it did not test: that a row is joinable when the bookkeeping the defect
corrupts was intact. Fisher exact on joinability against correctness gives p = 5.4e-08 if
the 23 joinable rows are among the 137 resolved and p = 2.9e-06 if they are disjoint from
them. No artifact carries both attributes per row, so both readings are run and the
register entry takes the larger p, which is the only defensible choice when the overlap is
unknown. Joinability predicts correctness either way. That confirms the assumption and
resolves nothing: it establishes the two measurements are taken on differently behaved
subsets, which is why they are not the ends of an interval, and it identifies neither rate
as correct.

Its p value therefore joins the family, because a p value this chapter reports is a member
whether or not it bears on the headline. k moves from 26 to 27, the Bonferroni threshold
from 0.001923 to 0.001852, the familywise upper bound from 0.7365 to 0.7497, and the
dependence groups from ten to eleven.

**The cost, which is the point.** Claude Haiku 4.5's few-shot ambiguity gradient sat at
p = 0.0018909 and cleared 0.001923. It does not clear 0.001852. The clearing set stays at
five only because the new test clears; its membership changed. The subsection had already
printed the sentence "it clears at twenty-six tests and would not clear at twenty-seven",
so the prediction was in the thesis before the test that falsified it existed. Every
earlier movement in this count was the repair of an omission and every one of them was
found to have run in the direction that kept a rejection. This one was not an omission and
it cost a rejection. Adding a test to an honestly enumerated family costs rejections; a
count allowed to lag buys them. That asymmetry is the whole reason the count is generated
from a register and not typed.

**A register value with no generator behind it, withdrawn.** The subsection reported
Haiku's Matthews interval widened to the corrected level as having a lower bound of
-0.0233, and the register described that as a percentile bootstrap at alpha 0.05/26 with
B = 10,000 at seed 42. No committed script produced it, and it could not be reproduced
from the registered confusion cells by the method described: resampling those cells at
that seed and that B gives -0.0492 at 0.05/26. So it is withdrawn, and
accord/stats/mcc_ci_at_corrected_alpha.py now generates both frontier rows and reads its
significance level from the test register, so a test entering the family moves the interval
instead of leaving a stale bound in the prose. At 0.05/27 Haiku's bound is -0.0494 and
Sonnet's is 0.0298. The claim the figure was cited for is unchanged: Haiku's interval
admits zero under either number, Sonnet's does not. The old figure is quoted in the chapter
to say it is withdrawn, so it is registered under quoted_historical rather than withdrawn.

**Two defects in my own edits, caught before the build.** I wrote a cross-reference to a
section label, sec:r-exemplar, that did not exist; the exemplar subsection had never been
labelled. Pass 7 would have caught it. And I wrote that the released-file and three-seed
disagreement above "is treated in Section 5.6", which was false, since no section treated
it. Rather than add a pointer I stated the discrepancy where it arises. Writing a
forward reference to a treatment that does not exist is the same failure as asserting a
completed act, which this record has now logged four times.

---

## C89. The withdrawal of the breached rows was enforced by prose, and the disclosure was out of proportion to what it disclosed

**No number in this thesis derives from either prohibited channel, and that was checked rather
than assumed.** The register holds 25 GPT-4o values: the two Matthews correlations, the two
precisions and recalls, four confusion cells each, two bootstrap intervals and one taxonomy
count. Nine of them are decimals distinctive enough to search for. None appears anywhere in the
chapters or the front matter. The safety screen has no result key at all; the only figure it
contributes is the count of 50 pairs, which is the disclosure and not a finding.

Two apparent hits were a coincidence worth recording, because it shows what the numeric
provenance gate can and cannot do. GPT-4o's few-shot Matthews correlation is 0.0117 and its
few-shot precision is 0.0496, and those are also, to four decimals, the weak labeller's own
correlation and the upper end of its bound from C88. Both strings occur in Chapter 4 and neither
occurrence is a GPT-4o figure. A digit-match gate cannot tell the difference, which is the
standing caveat on Pass 2 and is the reason the check above was done by searching for each value
and reading its context rather than by trusting the gate.

**The defect: all 25 keys sat live.** Chapter 4 says the GPT-4o rows "have been withdrawn rather
than qualified, and no number in this chapter now derives from it", and this section said they
are withdrawn from every table and figure. Both were true. Neither was enforced: none of the 25
keys carried the `withdrawn.` prefix, so Pass 2 would have resolved any GPT-4o digit appearing
anywhere in the thesis without complaint. Injecting 0.1089, GPT-4o's zero-shot correlation, into
Chapter 4 confirmed it passed silently.

Renaming the keys was rejected as the fix: several audit scripts read them by name, and breaking
a historical audit to tidy a prefix trades one defect for another. Instead the gate excludes
breach-channel values from the citable set, under a separate name from `withdrawn.` because the
ground differs: `withdrawn.` is retracted for being wrong, this is excluded for being unusable.
With the exclusion in force the same injection fails. The withdrawal is now a control.

**The proportionality pass.** With nothing downstream of either channel reported, the section was
arguing at length about experiments the thesis does not use. Each channel was described twice,
seventy lines apart, with the second treatment restating the first. The GPT-4o paragraph ran a
forensic measurement of the annotation sheet's character spans, 544 cells, a 200 character cap,
medians of 93 and 113, 94 cells at the cap, and then concluded that none of it was a defence,
so it argued a case at length and discarded it. Three passages narrated the section's own earlier
wrong versions, which is what the correction record is for. Each channel is now stated once: what
was sent, that it was prohibited, what the guidance says, that nothing reported derives from it,
and what replaced it. The span forensics are reduced to the one clause that bears on the
judgement and the meta-narration is cut to a single sentence pointing here.

**What was not reduced, and why.** The storage disclosure stands at full length. It is not a
report of an experiment that can be declined: seven committed artifacts still mount the drive,
among them the detector's own training script, which writes its checkpoints there, and the
annotation sheet carrying 95 cells of de-identification surrogates was edited in a browser on
that same drive. That is the current storage state of the data behind results this thesis does
report, and the repository is the evidence for it. A shorter statement here would be quieter than
the artifacts it ships with. The enumerated table of compliant transmissions also stands, because
the runs in it are reported and the guidance reading that licenses them is load-bearing.

Chapter 3 goes from 1,151 lines to 1,136 and the thesis from 195 pages to 194.

---

## C90. Three claims from C88 were reversed after adversarial review, and the test family returned to twenty-six

C88 added three computations and drew a conclusion from each. Two of the three conclusions were
wrong and one of the three tests did not belong where it was put. All three were found by
adversarial review and every number below was re-derived independently before the reversal was
made.

**The weak labeller's agreement is not measurable, and C88 reported it as measured.** C88 gave
0.0117 as the labeller's Matthews correlation with the pharmacist on G_253 and called it "the
result that locates the failure". The join it rests on is a reconstruction through a
nursing-side key, which C88 disclosed, and bounded only the 19 conflicting rows. That bounded
the wrong thing. Under a random re-alignment of the same labels, which leaves the labeller's
marginal untouched and varies only which rows its positives land on, the expected true
positives are 10.82 and the observed count is 11; the observed correlation sits at the 0.6648
point of that null distribution. Two thirds of scrambled joins score higher. The quantity is
indistinguishable from its own null, so it cannot separate a labeller at chance from an
arbitrary labeller joined badly, and C88's sentence that the qualification "limits its
precision and not its direction" was false: it limits identifiability entirely. Withdrawn.

What survives is stronger and needs no join. The labeller emits hard labels at a positive rate
of 0.9017 against a confirmed rate of 0.0474, and at that marginal the largest Matthews
correlation any thresholded label can express is 0.0768. The ceiling follows from the two
marginals alone. It is a statement about the labelling policy rather than the labeller's
competence, and it is the mechanism behind the detector's degeneracy, since a model fit to
labels emitted at a marginal of 0.9 inherits the marginal. The section now says that instead.

**The inference from it was invalid, independently of the measurement.** C88 called a
chance-level teacher the strongest evidence that the pipeline fails rather than the task being
unlearnable. Both readings predict a chance-level 8B labeller equally, so the observation
separates them not at all; it was the weakest admissible class of evidence, offered as the
strongest. The separation is carried by the frontier results, where two labellers on the same
rows return intervals excluding zero, which an unlearnable task would not permit. The section
now points there and the claim is deleted.

**The oracle threshold was compared against the wrong null.** C88 reported the best Matthews
correlation over any cutoff as 0.0554 and read it against zero. A statistic maximised over a
tuning parameter on the sample it is evaluated on is not centred at zero: the maximum of a
noisy process is positive by construction. On random scores at this sample size and positive
count the same maximisation has a median of 0.0968, and the observed optima sit at the 0.8195
and 0.7642 points of it. The conclusion is unchanged and is stronger stated correctly, because
the oracle threshold performs worse than roughly four in five pure-noise runs. The null is now
generated and reported beside the figure.

**The artifact disagreement C88 disclosed was mis-specified twice over.** It presented the
released score file and the three-seed run as disagreeing about the direction of the detector's
degeneracy. They are different models, the contaminated release and the decontaminated
retrain, not two readings of one artifact; and within the released file the harm score at 0.5
is a near-constant positive like the table, so the direction is not consistent across decision
variables inside one file either. Restated as a note that a reader recomputing from the
released file will get a different number, and why. Disclosure is not a substitute for
resolution when both artifacts are on disk.

**The joinability test is removed from the hypothesis family, which returns a rejection.** Two
grounds, neither of them the arithmetic. A family is defined by inferential target, and that
test's target is this project's own bookkeeping, which C88's own prose said on one page while
counting it in the denominator on another. And it is not a test of association: correctness for
the 23 key-joined rows was assessed by the key join and for the 137 others by the
reconstruction, so no row is measured by both, and the null such a table rejects includes the
clause that the two instruments have equal error rates, which the same section denies where it
calibrates the reconstruction at 0.8934 and 0.9508 fidelity. Rejecting a null one clause of
which the author has already denied in print is arithmetic, not evidence. The table is retained
as the descriptive statement that the 23 rows are not a random subsample of the 165, with no p
value and no directional language, and the confound is recorded as unfalsifiable from the
retained artifacts.

So k returns to 26, the threshold to 0.001923, the familywise bound to 0.7365, the dependence
groups to ten, and Claude Haiku 4.5's few-shot ambiguity gradient at 0.0018909 returns to the
clearing set. The bound Haiku's Matthews interval takes at the corrected level is -0.0492 and
Sonnet's is 0.0308, regenerated at 0.05/26.

**The direction is why the ground is recorded at length.** Every earlier movement of this count
was the repair of an omission that had kept a rejection, and C88 recorded with some satisfaction
that adding a test had cost one. This correction removes a test and gets the rejection back.
That is the favourable direction, which is exactly when the reasoning has to be checkable rather
than asserted, and it is why both grounds above are stated in the chapter and not only here.

---

## C92. A second vendor had already failed to replicate the headline, and its four tests were not in the family

The family this chapter corrects against was assembled by what each run was commissioned for
rather than by what each test asks. Four discrimination tests on the Bedrock labellers were
sitting in the results directory, generated by the same script, on the same 253 rows, against
the same pharmacist labels, at the same key path as the four the chapter leads with. They were
not counted.

| labeller | strategy | two-sided p | MCC | interval excludes zero | flag rate |
|---|---|---|---|---|---|
| Llama 3 70B | zero-shot | 0.2444 | 0.0762 | no | 0.4150 |
| Llama 3 70B | few-shot | 0.0391 | 0.1379 | no | 0.2372 |
| Llama 3 8B | zero-shot | 1.0000 | 0.0290 | no | 0.8735 |
| Llama 3 8B | few-shot | 1.0000 | 0.0219 | no | 0.8854 |

C90 had already settled the membership rule: a family is defined by its inferential target.
These four have the identical target as the four already counted. The ground for their absence,
the purpose the run was commissioned for, is not an inferential target and is the reasoning C90
rejected when it removed the joinability test on the same principle. Under the project's own
rule they were always members.

**The direction is why this is the worst of the four omissions.** All four are null. A second
vendor had failed to replicate the result this chapter leads with, twice, in two prompt
conditions each, and the family that omitted them was the family under which the headline
cleared. Every earlier movement of the count was a repair that left the conclusions intact, and
the entry recording the last one called that luck rather than rigour. This one costs a
rejection: k goes 26 to 30, the threshold 0.0019231 to 0.0016667, and Claude Haiku 4.5's
few-shot ambiguity gradient at 0.0018909, which cleared by three parts in a hundred thousand,
does not clear at thirty.

**Found by adversarial review of a different question.** A pre-registration, C91, was written to
commission a third vendor through Google Vertex AI to settle whether the discrimination result
generalises beyond Anthropic. Its motivating sentence asserted in writing that the Bedrock runs
"produce no discrimination test on G_253". A red-team pass checked that sentence against the
artifacts and it was false. The experiment was commissioned to answer a question whose answer
was already on disk, uncounted, and the pre-registration that commissioned it was the document
that would have directed a committee to the omission. C91 is not run.

**A second defect, found in the same pass.** The register read these p values from a key named
`fisher_exact_p_two_sided`, and nothing in this repository wrote it. `frontier_metrics.py`
emits `hypergeom_p_one_sided` and only that; the two were byte-identical in all four Anthropic
rows, so no reported number was wrong, but four headline p values had no generator and carried
a name the pipeline never computed. That is this project's own rule broken on its strongest
claim: every number is generated, never typed.

They are now generated, for all eight rows, by `accord/stats/frontier_discrimination_tests.py`,
which computes the one-sided and the two-sided value and records whether they agree per row
rather than assuming it. They agree on the four Anthropic rows and on Llama 3 70B few-shot.
They do not agree on the other three: Llama 3 70B zero-shot is 0.1804 one-sided against 0.2444
two-sided, and both Llama 3 8B rows are roughly 0.55 one-sided against 1.0 two-sided. So the
identity that made the mislabelling harmless was an accident of the Anthropic rows' margins and
not a property of the test, and adding the Bedrock rows under the old assumption would have
compared a one-sided value against a threshold calibrated on two-sided ones. The family now
reads the two-sided value for every row, which leaves the four Anthropic p values unchanged.

**What the nulls do and do not establish, which needs the ceiling.** A labeller emitting hard
labels at a fixed positive rate cannot reach an arbitrary agreement: holding its flagged count
where it is and placing every confirmed case inside it gives the best table it could have
produced. `accord/stats/marginal_ceilings.py` computes that ceiling for every row. Llama 3 8B
flags roughly seven pairs in eight, and its ceiling is 0.0849 zero-shot and 0.0803 few-shot, so
its null is close to arithmetic and carries little information about the vendor. Llama 3 70B is
not ceiling-bound: at flag rates of 0.4150 and 0.2372 its ceilings are 0.2649 and 0.4002 and it
reached 0.0762 and 0.1379. It had the headroom and did not use it, so that failure to replicate
is genuine and is the one that bears on the vendor question.

**C92, addendum after audit.** A results audit re-derived every value above from the cells and
confirmed all of them, and found four defects in the correction itself.

*A stale artifact the correction missed.* `mcc_ci_at_corrected_alpha.json` names the test
register as the source of its significance level and was not regenerated, so it published bounds
computed at 0.05/26 while the register it cites read 0.05/30, and the chapter quoted them
twenty-five lines after stating the new threshold. Regenerated: Sonnet's corrected lower bound
moves from 0.0308 to 0.0246 and Haiku's from -0.0492 to -0.0527. Both conclusions survive, the
digits did not. A generator that reads its own parameter from another artifact still has to be
re-run when that artifact moves; reading it dynamically is not the same as being current.

*A contradiction introduced by this entry.* The prose said a second vendor had failed to
replicate "twice, in two prompt conditions each", counting all four Bedrock tests as vendor
evidence, while the ceiling analysis in the same chapter argued two of them were uninformative.
Both could not stand. The audit supplied the sharper instrument: not the effect-size ceiling but
the power floor. At 221 and 224 of 253 pairs flagged, the smallest p the Llama 3 8B rows could
return at any outcome, with every confirmed case inside the flagged set, is 0.190 and 0.224.
They cannot reject at any outcome, so they are silent rather than negative. Llama 3 70B's best
attainable values are 1.8e-05 and 1.3e-08, so its failure is real. The second-vendor failure is
one model in two conditions and now says so.

*An attribution to a withdrawn instrument.* Table tab:mcc's note credited the two-labeller claim
to "their Mann-Whitney tests". Those values exist in the artifacts, and C85 withdrew that
instrument for exactly these tables because the label vector is binary and the rank test was
anti-conservative in all four cases. The note rested the chapter's strongest claim on the
statistic the chapter retracted. It now cites the zero-shot discrimination tests, which is what
the claim actually rests on and which clear the corrected threshold at k = 30.

*A headline declined.* Dividing each correlation by its own ceiling ranks the four Anthropic
rows above the four Meta rows, and that was going to be reported as the finding. The audit
showed why it should not be: the raw correlations separate by a wider margin, so the
normalisation narrows the gap rather than creating it; the quantity is algebraically
(recall - flag rate) / (1 - flag rate) rather than a correlation; and with the 8B rows' flag
rate near 0.88 a single reassigned true positive moves them by more than 0.65 and would carry
one above every Anthropic row. The chapter reports instead the one comparison that needs no
normalisation: Haiku zero-shot and Llama 3 70B few-shot flagged the same 60 pairs of 253, so
their ceilings and power floors are identical, and against that common margin they return 0.2253
and 0.1379. The normalised ordering is stated and explicitly declined in the same paragraph.

*Also fixed.* The ceiling generator recorded "anthropic" as a model id because both top-level
model fields are null in every metrics file and it fell through to the filename; the real ids
were unused one level down, per strategy.

---

## C92, second addendum. A second adversarial round on the fix, and three defects the fix introduced

The subsection written to answer round one was itself round two's target, and three of its
findings are defects introduced by that fix rather than inherited.

**The remedy table used the tail C92 closed, and cited C85 as permission.** The generator
computed a one-sided exact test only, and the table note called it "the instrument C85
established for these tables". C85 established exactness against a rank test. C92 separately
settled the tail and put the two-sided value in the family for every row, precisely because the
agreement of the two tails is an accident of particular margins. The defect-free stratum is new
margins, 108 rows and 6 positives at flag counts from 13 to 87, and the guard was not carried
across. Computing both now shows the divergence is real: the four Anthropic rows and Llama 3 70B
few-shot agree, and the other three do not. Llama 3 70B zero-shot moves from 0.4714 to 0.6859 and
both Llama 3 8B rows from 0.6694 to 1.0. The printed column was mixing tails across exactly the
cross-vendor rows the section reads. Corrected to two-sided throughout, and the false attribution
replaced with what each correction actually established.

**A conditional was inverted in the sentence that prices the thesis's largest dependency.** The
limitations paragraph read "Whether that difference withdraws the Claude rows is an open question
... If the answer is no, those rows are withdrawn." As written a favourable ruling destroyed the
evidence. It now reads "If it does". The sentence was introduced in the same edit that
re-anchored the compliance section, which is to say the repair and the defect arrived together.

**The compliance fix left three sections asserting what it had just withdrawn.** A paragraph
heading still read "Every transmission through a compliant channel, enumerated" over a table in
which 1,320 of 2,520 prompts are, by the paragraph's own text, on a route whose status is open.
The results chapter introduced the headline table by saying the run went "through one of the
services Section m-ethics identifies as acceptable for credentialed data", which that section no
longer says. And a claim that no channel was unreviewed rested on guidance the same page had just
called superseded. All three corrected.

**The rule stated in the first addendum was applied for its arithmetic and not for its
consequence.** Eight tests were kept out of the family as a sensitivity analysis of a hypothesis
already counted. That is defensible, and it means the primary claim has failed its own
sensitivity analysis: on the stratum where the annotation defect cannot have operated, neither of
the two zero-shot tests named as the two-labeller warrant clears the corrected threshold and
neither interval excludes zero, and the only row that does exclude zero is a configuration
Chapter 1 already declined to rest the claim on. Both branches of the rule are adverse. Under the
other branch the count reaches thirty-eight and the same warrant loses a labeller to the tighter
threshold. The chapter now takes the consequence rather than reporting only the arithmetic.

**And an affirmation of a null, in the paragraph defending the fix.** "So the separation is not an
artifact of the corrupted stratum" is a claim of no difference between strata, drawn from four
point estimates on six positives whose intervals run from below zero to above one half. It is the
inference this thesis condemns throughout, deployed in its own defence. Replaced with the narrow
statement the evidence supports: the effect survives the control and the evidence for it does not.

The pattern across both addenda is worth naming. Every defect in this entry was introduced by a
correction, not by the original work, and three of the five were introduced today by the same
hands that found the originals. A repair is an edit like any other and inherits the same
obligation to be checked.

---

## C93. Custody moved off cloud storage

**All eight artifacts that mounted Google Drive now resolve to credentialed-only local storage.**
Six notebooks and two scripts. Each reads its output path from the environment with a local
default under the quarantine directory, matching the convention the frontier runner already uses.
A reusable guard at accord/cloud_paths.py resolves a path and refuses it if it lands inside any
cloud-synced location, which on current macOS is one directory covering Drive, Dropbox, OneDrive,
iCloud and Box. It was tested against the Colab path, the real mount, and a genuine local path.

**The detector's training script is guarded rather than rewritten, and that choice is deliberate.**
That file is a recovered source committed unmodified as a record of what actually ran. Quietly
changing its output path would be an edit to evidence, which is the thing the recovery header
exists to prevent. Instead it now calls the guard on its own output path and exits with a clear
message, so the script cannot run against cloud storage while the record of where it did run
stays intact.

**What this does not do.** It changes where code points. It does not move the files already on
the drive, and it does not delete the 23 model checkpoints there, which is the author's action to
take after verifying his own backups. PhysioNet's guidance on derived datasets and models says a
model trained on credentialed data should be treated as carrying sensitive information, so those
checkpoints are in scope whatever the code now does.

**Two smaller repairs in the same pass.** The appendix said the full record is reproduced in a
supplementary file accompanying the document, which was true when this thesis was bound as two
documents and false since they were merged; it now points at the released repository. And four
bibliography entries printed internal database keys as if they were APA descriptions, because the
keys sat in note fields rather than comments. Moved to comments, entry count unchanged at 93.

**One defect introduced and caught inside this entry.** The rewritten appendix named a repository
filename in prose, which Pass 9 forbids and duly caught. The gate worked; the writer did not.

---

## C94. The channel effect was almost entirely the prompt

The capability ladder disclosed, in its own generator and twice in its provenance note, that its
local 8B row was run under a different prompt from the four hosted rows, and that the channel step
computed from that pair was therefore an upper bound confounded with a prompt difference. The
prompt effect it was confounded with, 1.6800, is larger than the channel effect it contaminated.
A disclosure that names its own defect and leaves it in place is a disclosure, not a control.

The fix needed no new data, no hosted call, no money and no approval: the same model, already
installed, re-run locally under the same template the hosted rows use. PhysioNet's guidance of
24 September 2025 recommends locally deployed models, so this route raises no question under the
agreement at all.

**The channel effect falls from 1.0962 to 1.0038.** Local and hosted serving of one model family
at one scale differ by four parts in a thousand. Almost the whole of the apparent serving-channel
effect was the prompt. The three-factor decomposition is now capability 2.0635, prompt 1.6800,
channel 1.0038, and the third term is measured rather than bounded.

The pre-registration at 0140c77 was committed before the first call and named the reverse outcome
in advance: a clean channel term reaching 1.6800 would have made the ordering this chapter reports
wrong as written. It did not, and saying so beforehand is why the result can be read as a test
rather than as a figure chosen after the fact. The superseded local ratio of 23.75 and the
superseded channel effect of 1.0962 are withdrawn to keys placed outside the generator's own
prefix, because every generator here deletes its prefix wholesale before rewriting and a
withdrawal parked inside one is erased on the next run.

**Two departures, recorded rather than tidied.** The qwen3 run did not complete: the machine
serving these models holds sixteen gigabytes and was already deep in swap, and its memory manager
stopped that run twice, once inside the batch and once running alone. The ladder generator was
changed to record an absent rung in the artifact rather than silently omit it, and no partial run
was spliced, which the committed rule requires. Nothing in the decision rule depends on the
missing rung: the channel step is one model against itself across two serving paths and both of
those runs completed with zero unparseable pairs. Second, the raised token ceiling for that model
was pre-recorded in the pre-registration, because it returns an empty string at the archived
ceiling when a reasoning block consumes the budget, which was confirmed by one synthetic call
before the rule was written.

**What this does not do.** It does not replace the hosted rows and is not evidence about them.
The measured ceiling on this machine is roughly 14B; a 70B model needs about four times the
available memory and more than the free disk. And it does not un-send the 1,320 prompts already
transmitted to the Anthropic API. A local run changes what happens next and changes nothing that
already happened, and the disclosure obligation for those transmissions is unaffected by anything
measured here.

## C95. Every hosted-channel result withdrawn, and the withdrawal made enforceable

PhysioNet replaced its 18 April 2023 post *Responsible use of MIMIC data with
online services like GPT* on **24 September 2025** with *Use of MIMIC Data with
Large Language Models and Online Services*. The replacement withdrew the list of
four suggested services, on the stated ground that PhysioNet "cannot verify the
data practices of external services and does not endorse or recommend specific
platforms". It states the prohibition without a vendor qualifier, as sharing
access "including sending it through APIs or using it on online platforms", and
adds an operative test: "MIMIC data must not be stored or retained by
third-party LLM services", with locally deployed models strongly recommended.

Every run this thesis made through a hosted service therefore has no safe
harbour. The four-service reading the routes were selected against does not
survive its own source. All of it is withdrawn: the Anthropic API runs (Claude
Sonnet 5, Claude Haiku 4.5, and the replicated Haiku draws) and the AWS Bedrock
runs (Llama 3 70B, Llama 3 8B), together with every analysis derived from them,
which is the frontier metrics, the label-noise sweep, the discrimination tests,
the marginal ceilings, the defect-free re-run, the system-difference intervals
and the hosted rungs of the capability ladder.

**718 register values** were relocated, out of 2,556. 1,838 remain live.

What makes this a control rather than a claim. The GPT-4o withdrawal was
enforced by prose alone for as long as it stood: the results chapter said no
number derived from it while all 25 of its keys sat live in the register, so the
provenance gate would have resolved a withdrawn digit anywhere in the thesis
without complaint. That was C89's finding and it is the failure mode repeated
here at twenty-nine times the scale. So:

- `accord/hosted_withdrawal.py` classifies a key as hosted from the artifact it
  was read from, not from its name. The test deliberately does not pattern match
  over model families: `llama38b` names the local Ollama labeller and
  `metallama38binstructv1` names the Bedrock one, and a substring rule would have
  collapsed them. That is the same defect class as the model tag that once landed
  in the wrong filename position and scored one model's predictions against
  another's artifact.
- `accord/stats/withdraw_hosted.py` relocates them to `withdrawn.hosted.*`,
  idempotently and losslessly, and `--check` exits nonzero if any is live.
- Pass 2 of `verify_thesis.py` now runs that check as a precondition, before the
  digit match, on the reasoning that the citable set has to be the right set
  before asking whether a digit is in it.

The gate immediately earned itself. Re-running `capability_ladder.py` after the
relocation put ten hosted values straight back under citable names, because every
generator rewrites its own namespace from scratch. A relocation done once by hand
would have come silently undone on the next register build.

`quoted_historical.*` is deliberately not exempt. That prefix exists so prose can
cite a withdrawn figure as history while the gate still treats it as live, and a
hosted value reaching the page through it is a hosted digit printed in the thesis
by another route. There is no reading of the guidance under which a number becomes
usable because the sentence around it calls it old.

The 16 hosted artifacts are moved to `results/accord/withdrawn_hosted/` with a
README stating the ground, rather than deleted, so this record can say what was
withdrawn and what each withdrawal cost.

### C95a. A clause-numbered admission of breach inside a machine-readable artifact

Five of the Anthropic run artifacts carried, on line 5, a `compliance_note`
ending: "Supersedes the original run against OpenAI's public API, **which
breached Clause 3**." That is a written determination of breach, by clause
number, inside a file that clause 9 of the data use agreement pushes toward
publication. Whether the transmissions were consistent with the agreement is a
determination for PhysioNet and for the institution. The characterisation is
replaced with the fact, that the earlier run is withdrawn on the same ground,
and the artifacts moved to the withdrawn directory. The four `frontier_ollama_*`
notes were left alone: they cite the current guidance correctly and describe a
local run.

### C95b. The custody remediation was genuinely incomplete, and the chapter was right

`accord/stats/final_number_audit.py` read `mining_summary.json` and
`phase2_summary.json` from the institutional Google Drive mount **at runtime**,
on every run. The 2026-08-08 remediation moved the 33 note-text files and left
the derived summaries behind, so the methodology chapter's statement that the
move was incomplete was accurate and the repository's earlier assertion that no
MIMIC data was cloud-synced was not.

A count-only scan of the mount (529 files, 68.0 GB) found no note text and no
identifier values remaining: 8 files reference identifier column *names* in
metadata. 271 data-bearing files, 196 MB, were copied to
`~/mimic_local_only/from_drive_2026-09-13/` with a per-file sha256 manifest, and
the audit script now resolves its inputs through `assert_local_only`.

The committed path also embedded the author's institutional email address, which
is what a Drive mount path carries, so every report the script wrote shipped that
address. `make_appendices.py` already redacted the pattern to `[DRIVE]` when
printing provenance; a source file should not have needed the redaction.
`audit_build_report.py` used the same path as a provenance label and is redacted
the same way.

**Not resolved, and it is not mine to resolve.** 23 model checkpoints totalling
**67.8 GB**, trained on MIMIC-III-derived weak labels, remain on the mount. Local
disk has 27 GB free, so they cannot be moved, and deleting them is irreversible
and the author's decision rather than a remediation step to be taken quietly.
Recorded here, and raised.

`accord_phase3_v2_ablations.py` was checked and is not part of this: its
recovered Colab Drive path is left unmodified as evidence, and the guard
immediately after the assignment calls `assert_local_only` and exits before any
write. The guard fires.

### C95c. An audit that could not run

`final_number_audit.py` died on `KeyError: 'gpt4o_zeroshot'` and then on
`KeyError: 'claude_sonnet5_zeroshot'`. The interval artifact gained hosted rows
after the audit was written, and four separate loops walked every key in it while
reaching for a matching row in `step3_baselines.json`, which has none. So the
audit had stopped auditing anything at all, silently, and this project has already
recorded twice what a gate that cannot fail is worth. The set it walks is now the
citable intersection, computed once rather than filtered at four call sites, so
the next artifact that gains a row cannot reintroduce the same break in only
three of them.

## C96. The withdrawal classifier read names, and a hosted result has no obligation to be named after its channel

C95's classifier tested the artifact path and the key name against a list of
channel tokens. It was caught by the adversarial pass on Chapter 4, and the
finding is the same one CLAUDE.md already records from an earlier round.

`results/accord/differential/frontier_label_noise.json` holds Claude Sonnet 5's
per-pair cells. Its filename contains no channel token. None of its **100**
register keys contains one either, because they are named after the analysis
(`labelnoise.*`) rather than after the model. So `is_hosted()` returned False for
every one of them and the gate reported PASS with 100 hosted values sitting live
and citable. The gate was not weaker than prose here; it was prose wearing a
gate's clothes, which is worse, because it had my confidence.

The artifact says what it is. Its `provenance.input` field reads
`results/accord/differential/frontier_anthropic_metrics.json`.

Fixed structurally rather than by adding `labelnoise` to the token list, which
would have left the next unnamed derivative to be caught by whoever happened to
read the right file. `_derives_from_hosted()` now follows an artifact's declared
provenance inputs transitively, depth-limited because a provenance field is free
text and could cycle, and a derived value is withdrawn if anything upstream of it
is. A token list enumerates the channels somebody remembered; a provenance walk
asks the artifact.

Withdrawn total rises from 718 to **818**.

The earlier instance of exactly this, quoted from the project's own working
agreement as a thing that had already bitten: "A register key maintained by hand.
All 96 `labelnoise.*` keys survived the correction that withdrew the script
behind them." It was the same 100 keys, the same namespace, and the second time
it was not a hand-maintained key but a hand-maintained *test*. The lesson that
generalises is not about labelnoise. It is that any check keyed on what a thing
is called will eventually meet a thing that is not called that.

## C97. RQ4 was replaced after the results were known, under a sentence saying it was not

Committee round after C95. Two findings, independently reached, changed the
structure of the thesis.

**The integrity finding.** Chapter 1 states that each research question is given as
posed, with its control specified in Chapter 3 before any result was seen. C95
replaced RQ4 with the marginal feasibility gate after every labeller output already
existed, and no pre-registration for the gate exists. The sentence was therefore
false for RQ4, and it is the one place a committee could move from criticising the
framing to questioning the conduct. RQ4 returns to the question Chapter 4 answers
with local evidence alone, whether the ranking null is real. The gate is reported
under RQ2 as a post hoc precondition check, Chapter 3 now specifies it and says it
was added after the hosted results were withdrawn, and Chapter 1 names it as the
one post hoc exception.

**The counting finding.** "Eight labeller configurations, half of which could not
reject" was two models by two prompts by two bases, and G_253 is a subset of G_299,
so it was four configurations on two nested bases. The could-not-reject set and the
at-ceiling set overlapped on llama3 few-shot, so "a separate four" was false. The
gate had also been applied only to rows that happened to have per-pair prediction
files, while the thesis printed five further hard-label marginals it never gated.
The generator now gates every printed hard-label marginal from registered counts:
nine configurations, five of which could not have reached p below 0.05 at any
outcome, and all five are llama3:8b configurations, including the deployed weak
labeller (261 of 299) and the prompted-once evaluation baseline (242 of 253).
granite3.3:8b and qwen3:8b pass.

**Readings withdrawn.** The gate bounds a thresholded decision and says nothing about
an AUROC, so every sentence crediting the marginal with predicting the ranking null
is removed. A labeller at its ceiling flagged every confirmed positive; the text had
said its agreement reported nothing but its flag rate, which discarded the one
positive signal in the thesis. A limitations sentence offered llama3 few-shot's
bootstrap interval excluding zero as evidence that twelve positives can separate a
system from chance, when that row could not reject at any outcome and its interval
excludes zero only because every positive is flagged; percentile bootstrap intervals
are now stated to be invalid as a test at the ceiling, and granite3.3 zero-shot is
reported by its nominal, uncorrected hypergeometric p instead. "No local labeller
was put to this test" was false for the ambiguity gradient; the local Fisher tests
are reported and in the family.

**Not cited, and flagged rather than resolved.** The committee noted that the
minimum attainable p-value under fixed margins is Tarone (1990, *Biometrics*
46:515-522). refs.bib is frozen at the author-supplied 93 entries, so it is not
cited; instead the thesis makes no novelty claim about the floor and calls the gate
applied rather than its own. The author should decide whether to add the reference.

Also in this round: the 1,425 figure was attached to the training set (it is the
positive-label count on the 1,679-row pool; training used 1,183 rows); "no join" was
said of a ceiling computed on joined rows; "identically sized encoder" contradicted
the parameter table; "every stage failed its control" contradicted the synthesis
table; Appendix A said the full reference tables were bound in a supplement that no
longer exists.

## C98. Committee re-review: a double count reintroduced by C97, and a graded score measured

The re-review of C97 found that the fix repeated the error it removed. The
methodology chapter states the replication reused the evaluation baseline's prompt
unchanged, so the baseline on G_253 and the baseline-prompt llama3:8b row on G_299
are one configuration on nested bases. "Nine configurations, five could not reject"
is eight and four. The generator now counts configurations, not rows, and a
configuration is infeasible if it could not reject on any basis it was measured on.
All four remain llama3:8b. The evaluation baseline was also printed as "not scored"
although it is scored row by row (MCC -0.0436); it now carries that value. A pointer
saying the gate was not run on the baseline contradicted the table and the
conclusion that relied on it. Chapter 1 said each research question is "stated as
posed", which was false for RQ4 after C97; the rewording is now disclosed in place.

The thesis stated that a score-emitting labeller is not bounded by its marginal and
never measured one. A graded local score was pre-registered before the first datum
(commit 57d42a2, `accord/differential/C98_PREREGISTRATION.md`) and run locally:
granite3.3:8b first-token log-odds on G_253, 251 distinct values, 0 degenerate,
AUROC 0.5467, one-sided Mann-Whitney p 0.2934, below the registered minimum
detectable AUROC 0.7355. Outcome B. Sweeping the threshold does not rescue it.

## C113. Committee round after the restructure: two outcomes reversed, one control rescoped, one undisclosed deviation

Three adversarial reviewers read the restructured thesis. Every finding below was checked
against the artifacts before being acted on; findings that did not survive the check are
recorded as refuted.

**C105 was recorded as Outcome B and is Outcome C.** Its rule declares the run invalid when
a training pool's teacher marginal is below twice pi_hat. The recorded outcome said 0 of 10
pools failed while quoting ratios of 1.4 to 2.4, which are below 2; on the artifacts 8 of 10
failed. The analysis script never evaluated Outcome C and compared against a field named
teacher_marginal_on_pool that holds the pool's true label mean. The teacher over-called the
true rate only 1.36 to 1.59 times, so there was no prior shift to correct. The script now
evaluates C first; the delta-AP interval is descriptive only.

**C102 Stages 3 and 4 were run untrained.** The registration specifies trained weights. S3a,
S3b, S3c and S4 carry no verdict; S3d (gradient connectivity) is valid untrained and stands.

**C110 scored the contaminated detector.** Its score column is the superseded contaminated
configuration (C16). C112 re-ran its rule on the decontaminated ensemble the thesis reports:
Outcome A, with the stratum AUROC 0.0950 from the G_253 value against a 0.10 band.

**The bag-of-words signal is not a weak-label signal.** C111: TF-IDF's average precision
survives leave-one-positive-out and beats the detector in a paired test, and its training set
shares no pair or subject with gold; but the off-the-shelf NLI model, never trained on weak
labels, finds the same signal and is indistinguishable from TF-IDF. Under C111's
pre-registered interpretation it is a surface lexical signal the detector lacks.

**"Naming the conflict" did not describe the manipulation.** P_asserting names no conflict
type or rule. It differs from P_neutral in persona, the word "dangerous", answer vocabulary,
system prompt and field labels at once. The contrast is a clinical-pharmacist framing against
a plain yes/no question; the effect is real and cannot be attributed to naming a conflict.

**The positive control was used beyond its registration.** C44's registration licenses the
AG News comparison only as a control for the pretraining-gap instrument and states the two
must not be merged. The thesis used it as evidence about the 12-positive contradiction
evaluation. That evidence is C108 alone, stated per rule: about 0.75 for the two-sided
Mann-Whitney test, about 0.81 for the rule C98 applied.

**C44 departed from its registration, undisclosed until now.** The registration specifies
selection on a held-out validation split. accord/scaling/C44_convergence_check.ipynb reports
the test accuracy of the final-epoch model with no checkpoint selection. Chapter 3 describes
the executed method accurately; C62 recorded the outcome without noting the departure. It is
recorded here.

**Counts corrected at their generators.** The feasibility gate double counted one configuration
measured under three names: 15 distinct configurations, 5 unable to reject, all llama3:8b
(was 18 and 6). The gold tipping point was selected from a stale row set: over every printed
hard-label row the closest to the public ceiling is qwen3:8b under the neutral prompt, which
5 relabelled flagged pairs would bring below 3.5 (was granite3.3:8b few-shot, 39). The
agreement ceiling at 261 of 299 is 0.0780; 0.0768 belongs to the joined G_253 rows.

**Refuted:** that the qwen3:8b local Fisher tests were missing from the multiplicity family
(they were present); that Appendix A's counts were hard-coded (they were generated; one
hard-coded "the last correction" phrase was fixed).

## C115. Committee rounds after C113 (2026-09-14)

Round 2 (idea-stress-tester, strategic skeptic) and round 3 (paper-reviewer) found, and the text now reflects:
- The detector tipping point was computed from the contaminated released column; C114 Task B recomputes it from the
  decontaminated ensemble (5, 6 and 7 flips to 0.7355, 0.7466, 0.8084).
- "Two simple models find a signal the detector lacks" inferred NLI's superiority from a non-significant NLI vs TF-IDF
  test; C114 Task A (p 0.10035, Outcome A2) restricts the claim to TF-IDF.
- The defect-free stratum carried no interval (C114 Task C, 0.3029 to 0.9010) and a floor for the wrong subset
  (0.8058 is the 134-pair subset; the 108-pair stratum's is 0.8064).
- The resolution floor was stated as 0.7355; it is now stated per rule from C108 (0.7466, 0.8084), scoped to AUROC.
- Round 1 of this pass (C113 fix list item F8) wrongly made 0.9315 the primary exemplar defect rate; C106 is Outcome B
  and no instrument establishes the rate. Reversed.
- The llama3:8b replication run and the frontier zero-shot / C103 P_asserting run use byte-identical prompt text but
  different reply parsers, seed options and run dates, and flag 285 and 261 of 299. The llama3:8b framing direction is
  within that gap and is no longer interpreted.
- The deployed weak labeller and the frontier zero-shot run share a 261 of 299 marginal and TP 11 but disagree on 24
  pairs (accord/stats/deployed_vs_frontier_zeroshot.py); they remain separate configurations.
- "No threshold rescues this detector" rested on the contaminated file; scoped to it.
- The C102 S1b benchmark is a synthetic one-sided mirror of the Chapter 4 Fisher test, not a validation of it.
- Contributions re-centred on the stage-wise audit, the marginal-implied agreement ceiling and trigger-exemplar
  dissociation; the detector null is a bounded supporting result; C102 demoted to a limitation.
- Title change from "Why" to "Where" recommended; adopted by the author on 2026-09-14.

## C116. The paired AP permutation test mixed score scales (2026-09-14)

Final committee pass (Reviewer 2) asked why TF-IDF minus detector (difference about 0.105) gave p 0.00005
while NLI minus detector (0.1088) gave p 0.10035. Both used `c111_run.paired_swap_perm_ap`, which swaps raw
scores between systems whose scales differ (G_253 medians 0.8858, 0.2966, 0.9647). C116 (pre-registered)
reran the tests on within-system ranks, which leaves AP unchanged: TF-IDF minus detector p 0.17875, NLI minus
detector p 0.14520. The claim that TF-IDF finds a signal the detector lacks is withdrawn from the text; C111
task b's p and C114 task A's p are superseded. C111 task a (each system against its own label-permutation
null) is unaffected.

## C117. Final committee passes after C116 (2026-09-14)

Final pass 1 (three reviewers) and final pass 2 found, and the text now reflects: the ceiling check described as run before
training (it was run after); the stage-wise claim widened to all four stages; an untested rarity causal sentence; the
rule-type head called a copy; a false direction-of-corrections sentence; llama3:8b's framing direction still read in Ch1/Ch6;
the claim that only one graded score was tested (C103 reports nine; the strongest did not confirm under C109); two p values
for one TF-IDF test (C107's 0.0159 kept); "surface lexical signal" stated as a finding (attributed to C111's registered
interpretation); Ch3 "falsification" vs Ch5 "confounded" for the rule-type head; a Ch3 clause affirming a null; Ch5 What
survived out of step with its table; a comparator called conservative; C98's prompt mislabelled as the plain prompt (it has
its own one-word prompt).

## C118. Third final committee pass (2026-09-14)

Found: Ch1 prose and the pipeline caption still said all four stages reported an unsupported number (now three, plus a
bounded null); and C109's registered, descriptive granite3.3:8b cells were never reported. Those cells rank above chance
under every wording on G_253 (0.6788 to 0.7936), and the best wording, chosen after the fact, is strongest on the Rules 3/4
stratum (0.8775, 6 positives) against Rules 1/2 (0.7302); the stratum ordering changes with wording. The "signal sits in the
defect rows" reading is now scoped to llama3:8b, a descriptive granite table is in Supplement S5, and C109's primary
Outcome B is unchanged.

## C119. Fourth final committee pass (2026-09-14)

Found: granite3.3:8b meets C109's two registered criteria under two of three new wordings (0.7936, p 0.0002; 0.7362,
p 0.0026, the latter above 0.7355 by 0.0007), which the thesis did not state, and Ch6 named no model when saying the
reversed score reached the threshold under none. Ch6 now names llama3:8b; Ch4, Ch5, Ch6 and Supplement S5 state the
granite result descriptively. C109 Outcome B, whose rule covers llama3:8b only, is unchanged.

## C121. Stage 1 discovery round after the title change (2026-09-14)

Five discovery agents (brutal reviewer, results auditor, red team, data-agreement review, format check) and one
provenance investigation. Found and fixed:
- Chapter 4 said the weak-label and gold files could not be joined while Chapter 3 used that join. C120
  (pre-registered) confirmed the exact five-key join is 1:1; the deployed labeller's agreement is measured and at
  chance (MCC 0.0269 of a possible 0.0780 on 299 pairs; 0.0243 of 0.0818 on 253). C88's reconstruction superseded.
- The gold sample was drawn by the original pipeline from its 1,978-candidate pool before the rebuild existed; the
  original Rules 3/4 selectors are lost. "Defect-free" is withdrawn as a label: the Rules 3 and 4 stratum is not
  subject to the documented Rules 1/2 defect, but its exemplar correctness is assumed, not verified, and the rule-group
  contrast is confounded with rule content and base rate. Ch3's "Rules 3 and 4 exist only in the rebuild" was false.
- The Rule 1/2 vs 3/4 power (0.0576) had no generator; exact enumeration gives 0.0495 and a minimum detectable
  difference of 0.094 (accord/stats/exemplar_power_exact.py).
- 0.52075 was quoted as the power of C109's rule; it is the C98 rule's.
- Contribution 2 is the classical marginal-constrained bound (Cohen 1960); the contribution is its use as a check on
  an LLM labeller, which needs a small gold base-rate sample. Contribution 3's bug class is reachable; its prevalence
  here is not established.
- Chapter 2 credited negative, randomisation and control-task controls as run; none was. Table 2.1 and the gap
  statement rewritten around the checks actually run and the three contributions.
- Figures 1.1, exemplar, ceiling, power, forest and prevalence-response rebuilt to argue what the text claims.
- The generated trace appendix resolved prose digits to 24 withdrawn hosted values; trace_claims.py now skips withdrawn
  keys. Hosted per-pair outputs moved out of the tracked tree to local storage.
- PhysioNet guidance wording corrected (the 2025 guidance requires zero retention and recommends local models; it is not
  established that it withdrew the 2023 list).
- C44 pretraining control moved to Supplement S6 (its registration forbids reading it against the evaluation).
- Format: three tables overflowing the page, four smaller overflows, abbreviation list and first-use definitions.
- Redundancy cut: main thesis 149 to 137 pages.
Open for the author: front-matter placeholders and the declaration wording.

## C122. Committee round after C121 (2026-09-14)

Paper reviewer (pass with majors), Reviewer 2 and strategic skeptic (fail on wording) found, and the text now reflects:
the abstract dropped the "assumed, not verified" caveat on the Rules 3 and 4 pairs; "caught by a one-line assertion"
claimed a check that never ran (the assertion is now demonstrated on 1,000 seeded synthetic candidates, catching all 485
positional picks outside the trigger set with no false alarms, accord/mining/, and described as not measuring the
defect's rate); the measured MCC was offered as agreeing with a ceiling it cannot exceed (contribution 2 now rests on the
attainable floor p 0.1894, Tarone's minimum attainable significance, the 5 of 15 configurations that could not reject, and
agreement measured at chance); stage verdicts disagreed between chapters and figure (now one scheme: not established,
unsupported, unsupported by code audit, bounded null); "control" used for checks and audits that were not controls;
Chapter 4 justified the Rules 3/4 stratum with the rebuild's selector; Appendix A kept the superseded PhysioNet wording;
the power figure plotted the C44 mortality control beside the evaluation; ungenerated exemplar power keys remained citable.
Open for the author: whether to keep "Controlled" in the title, given the thesis ran checks and audits, not randomised
controls.

## C123. Committee pass after C122 (2026-09-14)

Found: the central sentence said no stage's reported number survived, but the numbers largely stand (0.0401, recall 11 of
12, AUROC 0.5145 is itself the bounded null); what failed is each number's reading, and the sentence now says so everywhere,
including Figure 1.1. "None of which needs statistical power" contradicted the bounded-null verdict, which rests on C108.
Chapters 3 and 4 still called checks controls ("The Controls", "anisotropy control"), and the Chapter 3 overview implied every
check was fixed in advance. "Per-rule floors" read as mining rules; now "per-test". The synthetic assertion fixture gave each
candidate a single trigger; a second fixture with trigger sets of one to three mentions is added (positional selector fails
372 of 1,000, trigger selector 0), and the no-alarm result is labelled a construction check.

## C124. Fabrication and data-agreement audits, and two committee passes after C123 (2026-09-14)

Data-agreement audit: no clinical text, identifier value or shifted date in any tracked file, notebook output, figure,
built PDF, or in any of 471 commits' diffs. Two register keys (the 108/6 stratum counts) named a hosted-run artifact as
their source although the counts are properties of the gold standard; retired under withdrawn.*, and the same counts
resolve through local artifacts.
Fabrication audit: no fabrication. 935 of 937 priority register keys resolve at their artifact; 64 of 64 printed numbers
carry the right meaning; 19 of 19 pre-registrations exist, precede their artifacts and match their outcomes; 5 of 5
figures read numbers from the register. Gaps closed: the decontaminated aggregate (interval 0.3210 to 0.7034, Mann-Whitney
p 0.8668) had no generator and is now recomputed exactly by accord/stats/recompute_decontam_aggregate.py; the
byte-identical prompt claim rested on an uncommitted table and is now checked by accord/stats/prompt_identity_check.py
(booleans only; all true); two loose Cohen (1960) citations removed. Caveat recorded: an earlier history rewrite (trailer
removal) leaves author and committer dates identical, so the ordering of pre-registration and run rests on those
timestamps.
Committee passes: Supplement S1 and S4 still called checks controls; the Chapter 6 closing was split; C109's llama3:8b
confirmation failed on AUROC under every new wording and on Bonferroni under two of three (Chapter 5 said otherwise);
Chapters 1 and 3 overstated what was fixed in advance (only the numbered pre-registrations were); 0.0495 was placed as
C102's power though it belongs to the Rules 1/2 versus 3/4 Fisher test; a deployment sentence contradicted the bounded null;
two supplement pointers still named Chapter 4 for content moved to S6; Chapter 1 mixed pair sets for the baseline p values
(now all on G_253 from C111); the audited component count and the rule-type head verdict word made consistent.
