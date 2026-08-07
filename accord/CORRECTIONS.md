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
