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

## What survives all of this

- **299 two-pass pharmacist annotations** with confidence, justification and 1–5 danger ratings
- **Phase 1 cohort construction** — 13,597 HF admissions, the 3-role → 2-role retreat documented
  with coverage at each step, plus sepsis (4,339) and AKI (7,579)
- **MedNLI 83.9% / 83.9% macro F1** full fine-tune vs **72.7%** LoRA r=16 — correctly run, inside
  published SOTA, provided it is labelled as BioMedBERT and not as ACCORD
- The **negative result itself**, which is measured four independent ways and is the thesis
