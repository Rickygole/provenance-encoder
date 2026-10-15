# NOTES: JBI manuscript

> **STATUS as of 2026-10-15, after the gate review and the C140 re-analysis.**
> This file was substantially superseded. Read this block before anything below
> it; where the two disagree, this block is current.
>
> **JBI requirements are VERIFIED, not second-hand.** The guide for authors was
> read directly in a browser on 2026-10-15 (ScienceDirect returns 403 to
> automated fetch, so this was done interactively). Confirmed verbatim: body
> <= 6000 words; mandatory section order Introduction, Related Work, Methods,
> Results, Discussion, Conclusion; structured abstract with headings Objective,
> Methods, Results, Conclusion, <= 300 words; <= 8 figures and tables combined;
> a Statement of Significance; a graphical abstract required at submission; and
> the submission declaration excluding "academic thesis" from prior publication.
> Any statement below that these are unverified is stale.
>
> **The rival result is WITHDRAWN.** C140 (pre-registered 2026-10-15 at commit
> fbadcac, before any datum, run at 26989cf) established Outcome B: the
> rule-identity prior is an in-sample artefact. Refit permutation p 0.2702, not
> the 0.0367 obtained with the fitted score held fixed; cross-fitted AUROC
> 0.4412, below chance, interval 0.2086 to 0.6680; in-sample optimism 0.2066;
> rule-label association absent, chi-square 3.8078 on 3 df, p 0.2830. The claim
> that no configuration separates from a rule-identity prior, and the mechanism
> claim of confounding by rule-correlated content, are both withdrawn.
>
> **The ceiling is imported apparatus.** MCC on a 2x2 table is Pearson's phi,
> verified arithmetically on this paper's own numbers; the margin-constrained
> maximum is phi_max, published by Cureton (1959) and Guilford (1965). The paper
> claims the transport and the class-level consequence, not the bound.
>
> **Premise replaced by evidence.** The claim that stages are "commonly certified
> by beating chance" described 0 of 37 sampled papers and is deleted. The
> supported finding: 0 of 37 reported a minimum detectable effect, 0 tested
> against a chance or zero-skill null, 17 did not report the gold positive count.
> See premise_survey/.
>
> **Build state, measured 2026-10-15 after the C141 repair.** 38 pages in review
> mode, 0 undefined references and citations, 91 bibliography entries, 3 tables
> and 1 figure, abstract 283 words. Body **6,105 words** by a strict count that
> strips comments, floats, equations and citation commands and excludes the
> Statement of Significance and the Data availability statement. That is about
> 1.8% over the 6,000 limit and the overage is stated rather than estimated away;
> an earlier claim of 6,020 here was produced by a looser counter and was wrong.
>
> **Corrections applied.** Three false hosted-service statements replaced with
> vendor-neutral disclosure matching the thesis record; the spurious seventh
> Bonferroni bullet removed from Table 3 (artifact lists 6); sidedness stated for
> every test; counts deduplicated for the two determinism re-runs to 14 of 15, 4
> and 10; Demonstration 1's pre-registration history disclosed; the unrepeatable
> labelling run disclosed; a data availability statement added; a no-clinical-use
> sentence added.
>
> **C141, 2026-10-15.** C140's Q3 was not a cross-fit: leave-one-out scored each
> pair as (k_r - y_i)/(n_r - 1), a function of its own label, forcing every
> positive below every negative within its stratum for any data. Its 0.4412 and
> its 0.2066 optimism are withdrawn. C140's Q4 held each cell's observed standard
> error fixed in a counterfactual where the variance shrinks; its count of 8
> unreachable cells is withdrawn. Replacements: cross-fitted AUROC 0.4939 at K=5
> (0.3679 to 0.5668) and 0.5026 at K=3, indistinguishable from chance; 0 cells
> unreachable, and a configuration would need AUROC 0.9414. The withdrawal of
> C139's reading stands on C140's Q2, which is unaffected.
>
> **Framing retired.** The welded class-level claim is removed throughout. Its
> second clause was definitional and its first was a type error: phi_max bounds
> the labeller's agreement, not a screen's discrimination, and Table 1's own
> ratio rule reaches Youden J 0.6807 where the labeller ceiling is 0.0818. The
> survey is demoted to a motivating sample, matching its own method file.
>
> **Still outstanding and NOT done here.** The PhysioNet compliance email,
> drafted 2026-08-29, is still unsent. No IRB or exemption determination has been
> obtained. No clinician has evaluated the checks, which JBI expects. A
> presubmission enquiry on article type has not been sent. A graphical abstract
> has not been produced.

---

# NOTES: provenance, open items, reviewer defences, submission record

Manuscript: `main.tex`. Bibliography: `refs.bib`. Figure generator: `make_figures.py`.
All artifact paths below are relative to the repository root
(`/Users/rickygole/provenance-encoder`).

---

## 0. UNVERIFIED AND NOT DONE. Read this first.

### 0.1 The build was not executed

No shell tool was available in the session that produced these files, so
`pdflatex`, `bibtex` and `python3` were never invoked. Consequences, all of which
the author or coordinator must close before submission:

- `fig1.pdf` and `fig2.pdf` **do not exist yet**. `main.tex` will not compile
  until `make_figures.py` has been run.
- `main.bbl` **does not exist yet**. Editorial Manager requires it in the
  bundle; without it EM renders every citation as a question mark.
- The undefined-reference and undefined-citation counts are **unverified**.
- The **page count is unknown**.
- The **word count is an estimate from the source**, not a measurement, and it
  is **borderline against the 6,000 limit**. Counting prose source lines
  (Statement of Significance through Conclusion, excluding the abstract, table
  bodies, figure captions, the equation and the references) gives roughly 590
  lines. Hand-counting a sample paragraph gives 9.5 to 10 words per prose line
  once `\cite`-heavy and paragraph-final short lines are included, so the body
  is approximately **5,900 to 6,300 words**. This must be measured, and if it
  comes in over 6,000, Section 0.2 below gives an ordered cut list.
- The abstract is estimated at approximately 296 words against the 300 limit.
  This is tight; measure it. If it is over, delete "None requires new
  inference." from the Methods block (saves 4) and change "Three pre-registered
  demonstrations follow" to "We report three demonstrations" (saves 2).

### 0.2 Ordered cut list if the body measures over 6,000 words

Apply in this order and stop as soon as the count is under 6,000. Each item is a
deletion of redundant framing, never of a number or a caveat. Nothing on this
list is load-bearing for any claim.

1. **Related work, paragraph 2, the metric half** (approximately 110 words).
   Delete from "Separately, Matthews correlation and Cohen's $\kappa$ are
   bounded" through "sits far above 0.5 \cite{hanley1982meaning}." Keep the final
   two sentences ("That literature establishes the bound exists. What it does not
   establish ..."), and move the citations
   `matthews1975mcc, cohen1960kappa, feinstein1990paradoxes, cicchetti1990resolving,
   byrt1993pabak, chicco2020mcc, chicco2021mccbalanced, saito2015prcurve,
   delgado2019kappa, hanley1982meaning` into a single bracket on the retained
   sentence so no citation is lost.
2. **Methods 3.6, the student paragraph** (approximately 90 words). Delete the
   secondary TF-IDF student's description and the sentence beginning "The
   pre-registration required the primary student's clean-label reference AUROC",
   keeping "MiniLM reached 0.8167 and passed, while TF-IDF reached 0.6037". The
   headroom logic is restated in Results 4.1 under "Two caveats on this reading".
3. **Discussion 5.2, second paragraph** (approximately 120 words). Delete from
   "None of the three components is new as a statistical idea" through "does not
   currently cover", but **retain** the sentence beginning "What is new is the
   composition into a method of methods", because that sentence carries the
   method-of-methods framing the submission depends on. Re-attach
   `lipsitch2010negative, poliak2018hypothesis, feinstein1990paradoxes,
   byrt1993pabak, collins2024tripodai` to the retained sentence.
4. **Introduction, "The gap" paragraph** (approximately 70 words). Compress the
   four formalise clauses to one sentence listing the four controls with their
   citations, keeping the final sentence ("None is assembled into a protocol ...")
   verbatim.
5. **Results 4.3, second paragraph** (approximately 60 words). Delete "One
   permutation draw is also not a null distribution, and a detector already at
   chance on this gold set has little room to fall." This caveat is repeated in
   Limitations, so deleting it here loses nothing.
6. **Conclusion** (approximately 50 words). Delete the middle sentence beginning
   "We specified three checks that make those comparisons", since the three
   checks are named in the abstract, the Statement of Significance and Methods.

Items 1 through 6 total approximately 500 words, which clears the limit from the
top of the estimated range. Do **not** cut: any value in any table, the
in-sample disclaimers on 3.5833 and on the rival's 0.6478, the
leakage-versus-confounding correction, the determinism-duplicate disclosure, the
single-annotator limitation, or the "uninformative by construction" statement.

### 0.3 Required build sequence

Bibtex, not biber; no `latexmk -pdflatex=... -shell-escape`:

```
cd /Users/rickygole/provenance-encoder/papers/jbi_weak_supervision
python3 make_figures.py      # writes fig1.pdf and fig2.pdf into THIS directory
pdflatex main
bibtex   main                # produces main.bbl; must be committed
pdflatex main
pdflatex main
```

Word count:

```
texcount -inc -sum -nosub main.tex
```

If `texcount` is absent, `detex main.tex | wc -w` is a usable floor; subtract the
abstract and the table and caption text by hand.

### 0.4 JBI guide-for-authors specifics are second-hand

The following were supplied by the coordinator and **I could not verify them
myself**: the 6,000 word body limit; the mandatory section order; the
Objective/Methods/Results/Conclusion abstract headings and 300 word limit; the
limit of 8 combined figures and tables; the requirement for a Statement of
Significance; the requirement for a graphical abstract at submission; the
definitions of the Research Paper and Special Communication article types; the
"method of methods" scope sentence; and the prior-publication sentence quoted in
Section 3.2 below.

What I did verify: `https://www.elsevier.com/journals/journal-of-biomedical-informatics/1532-0464/guide-for-authors`
returns HTTP 301 to
`https://www.sciencedirect.com/science/journal/15320464/publish/guide-for-authors`,
which confirms the canonical location of the guide. That destination returned
HTTP 403 to automated fetch, so no sentence on it was read. **Re-check every item
in the paragraph above against the live page before submitting.**

### 0.5 Numbers in the manuscript that are arithmetic over artifact values rather
### than artifact keys

Each is exact and trivially recomputable, but none was written by a committed
generator, so each violates the project's "every number is generated, never
typed" rule in the strict sense. Flagged rather than removed.

| Figure in prose | How obtained |
|---|---|
| prevalence `0.0474` | `12/253`; also stated as `0.0474` in `accord/thesis/msu_ch4_results.tex` |
| `301` evaluations of the hypergeometric survival function | flag counts `0..300` inclusive |
| `0.05/17 = 0.0029` | rounding of the artifact key `bonferroni_alpha_over_n` |
| "roughly fourfold" rule-yield spread | `0.1071 / 0.0256`; the C139 pre-registration says "roughly fourfold" |
| "0.6 to 14 billion parameter range" | read off the model tags |
| per-model runtime range `501.8` to `8720.9` seconds | min and max of the `seconds` field over the 7 real labellers |

A total runtime of `23859` seconds (the sum of those `seconds` fields) was in an
earlier draft and was **removed**, because a hand-computed sum is exactly the
kind of typed number the project forbids. If the author wants the total, a
generator should emit it.

### 0.6 A defect found and fixed during drafting: the "94.7 percent" figure

`permuted_label_control.json::pool_label_marginal_note` reads "the replication
label vector is 94.7 percent positive on the pool", and the pre-registration
outcome says the same. The same artifact's gates record
`n_positive_before_permutation` = 1132 out of `n_rows` = 1183. Those are not the
same number: **1132/1183 = 0.9569, not 0.947.** The note's "pool" is evidently a
different denominator from the 1183-row subject-disjoint training split that the
gates count, and no artifact key states which.

An earlier draft of this manuscript wrote "the weak label vector is 1132 of 1183
positive, or 94.7%", equating them. That is exactly the kind of unchecked
carry-over this paper argues against, so it was removed rather than
reconciled. Every occurrence now states only the two integers, "positive on 1132
of 1183 training rows", and the permutation effect is given as the artifact's own
fraction key, 0.0845. The percentage 94.7 does not appear in the manuscript.
Verify with `grep -n "94\.7" main.tex`, which must return nothing.

**Open item for the author:** if the 94.7 percent figure is wanted, a generator
should emit it with its denominator named, and the artifact note should be
corrected or its denominator documented.

### 0.7 One artifact-versus-pre-registration discrepancy, resolved in the
### artifact's favour

`accord/labeller_interval_closure/PERMUTED_LABEL_PREREGISTRATION.md` describes
the real-label bootstrap interval as "0.375 AUROC wide"; the artifact's own
`what_this_does_not_establish` field says "0.374 AUROC wide". The true width is
`0.7195712309820194 - 0.345434996612753 = 0.374136...`. The manuscript avoids
the width entirely and prints the two endpoints (`0.3454` to `0.7196`) instead.

### 0.8 Not written, by instruction

Acknowledgements, funding statement, CRediT author contributions, and the
declaration of generative AI use are left as commented placeholders in
`main.tex`. The author writes those.

---

## 1. Provenance table: every numeric claim to artifact and key

Rounding convention throughout the manuscript: 4 decimal places, truncating
nothing, no re-rounding of a value already rounded.

### 1.1 Demonstration 1, the collapse check (SNLI)

Artifact: `results/accord/c136_gate_validation/c136.json`
Pre-registration: `accord/c136_gate_validation/C136_PREREGISTRATION.md`

| Claim in manuscript | JSON key | Artifact value |
|---|---|---|
| clean-label reference AUROC 0.8167 (MiniLM) | `clean_label_reference_auroc.minilm` | 0.816696875 |
| clean-label reference AUROC 0.6037 (TF-IDF) | `clean_label_reference_auroc.tfidf` | 0.6037117187500001 |
| headroom floor 0.75, passed | `headroom_precondition.floor`, `.passed` | 0.75, true |
| pilot 300 with 12 positives | `frames.pilot` | [300, 12] |
| pool 5000 with 200 positives | `frames.pool` | [5000, 200] |
| test 10,000 with 400 positives | `frames.test` | [10000, 400] |
| pool and test disjoint | `frames.pool_test_disjoint` | true |
| seed 42 | `seed` | 42 |
| gate K=12, N=300, alpha=0.05 | `gate.K`, `gate.N`, `gate.alpha` | 12, 300, 0.05 |
| failure set is {0} plus upper tail | `gate.fail_region_is_zero_plus_upper_tail` | true |
| failure set starts at 235 | `gate.smallest_failing_positive_flag_count` | 235 |
| flag rate 0.7833 | `gate.smallest_failing_positive_flag_rate` | 0.7833333333333333 |
| primary sensitivity 0.2941 | `primary_reading.gate.sensitivity` | 0.29411764705882354 |
| primary specificity 1.0000 | `primary_reading.gate.specificity` | 1.0 |
| primary PPV 1.0000 | `primary_reading.gate.ppv` | 1.0 |
| primary NPV 0.3684 | `primary_reading.gate.npv` | 0.3684210526315789 |
| primary Youden J 0.2941 | `primary_reading.gate.youden_j` | 0.2941176470588236 |
| 2x2 cells 5/0/12/7 | `primary_reading.gate.cell_*` | 5, 0, 12, 7 |
| 5 gate-fail, 19 gate-pass | `primary_reading.gate.n_gate_fail`, `.n_gate_pass` | 5, 19 |
| 12 of 19 passed still failing, 0.6316 | `primary_reading.gate.frac_gatepass_still_failing` | 0.631578947368421 |
| gate-equivalent ratio threshold 19.5833 | `primary_reading.ratio_at_gate_equivalent_threshold.ratio_threshold` | 19.583333333333332 |
| **zero disagreements** | `primary_reading.ratio_at_gate_equivalent_threshold.disagreements` | 0 |
| **identical verdicts** | `...identical_to_gate_verdicts` | true |
| gate adds nothing over raw ratio | `primary_reading.gate_adds_nothing_over_raw_ratio` | true |
| tuned ratio Youden J 0.6807 | `primary_reading.ratio_best_over_all_thresholds.youden_j` | 0.6806722689075628 |
| tuned sensitivity 0.8235 | `...ratio_best_over_all_thresholds.sensitivity` | 0.8235294117647058 |
| tuned specificity 0.8571 | `...ratio_best_over_all_thresholds.specificity` | 0.8571428571428571 |
| tuned PPV 0.9333 | `...ratio_best_over_all_thresholds.ppv` | 0.9333333333333333 |
| tuned NPV 0.6667 | `...ratio_best_over_all_thresholds.npv` | 0.6666666666666666 |
| tuned threshold 3.5833 | `...ratio_best_over_all_thresholds.ratio_threshold` | 3.5833333343333336 |
| tuned 2x2 cells 14/1/3/6 | `...ratio_best_over_all_thresholds.cell_*` | 14, 1, 3, 6 |
| 0.70 arm sensitivity 0.2381, NPV 0.1579, cells 5/0/16/3 | `synthetic_minilm.screening["auroc_lt_0.70"].gate.*` | 0.23809523809523808, 0.15789473684210525, 5/0/16/3 |
| 0.70 arm 16 of 19 still failing (0.8421) | `...["auroc_lt_0.70"].gate.frac_gatepass_still_failing` | 0.8421052631578947 |
| pooled sensitivity 0.3636, NPV 0.3913, cells 8/0/14/9 | `pooled_minilm.screening["auroc_lt_0.60"].gate.*` | 0.36363636363636365, 0.391304347826087, 8/0/14/9 |
| pooled 14 of 23 still failing (0.6087) | `...frac_gatepass_still_failing` | 0.6086956521739131 |
| pooled 2 disagreements | `pooled_minilm...ratio_at_gate_equivalent_threshold.disagreements` | 2 |
| real-arm sensitivity 0.6000, NPV 0.5000, cells 3/0/2/2 | `real_minilm.screening["auroc_lt_0.60"].gate.*` | 0.6, 0.5, 3/0/2/2 |
| TF-IDF arm cells 5/0/19/0, sens 0.2083, NPV 0.0000, spec undefined | `synthetic_tfidf.screening["auroc_lt_0.60"].gate.*` | 5/0/19/0, 0.20833333333333334, 0.0, null |
| Spearman ceiling MCC vs AUROC rho 0.5310, p 0.0076 | `synthetic_minilm.spearman_ceilingmcc_vs_auroc` | 0.5309850092907832, 0.007591134755258963 |
| Table 2, all 7 real rows | `labellers[]` where `family == "real_ollama"` | see below |
| per-model runtime 501.8 to 8720.9 s | `labellers[].seconds` min and max | 501.8, 8720.9 |
| synthetic flag counts 5 to 291 | `labellers[].flag_count` over synthetic; also stated in the pre-registration outcome | 5, 291 |

Table 2 row-by-row source (`labellers[]`, keys `flag_count`,
`pilot_true_positives_flagged`, `pilot_unparsed`, `gate_fail`, `floor_p`,
`student_auroc_minilm`, `student_auroc_tfidf`, `student_degenerate_minilm`):

| Model | flags | TPs | unparsed | gate_fail | floor_p | MiniLM | TF-IDF |
|---|---|---|---|---|---|---|---|
| qwen3:0.6b | 272 | 6 | 6 | true | 0.30148364942613176 | 0.24094557291666668 | 0.4110484375 |
| qwen3:1.7b | 176 | 0 | 0 | false | 0.001415658565469855 | 0.202765625 | 0.45975130208333337 |
| qwen3:4b | 0 | 0 | 300 | true | 1.0 | 0.5 (degenerate true) | 0.5 (degenerate true) |
| llama3:8b | 17 | 2 | 16 | false | 6.96982307275377e-18 | 0.42276354166666663 | 0.4667856770833333 |
| granite3.3:8b | 18 | 10 | 0 | false | 2.0909469218261317e-17 | 0.7807614583333332 | 0.5852200520833333 |
| qwen3:8b | 0 | 0 | 0 | true | 1.0 | 0.4120138020833334 | 0.4915057291666666 |
| qwen3:14b | 8 | 5 | 0 | false | 3.34219579242274e-13 | 0.7579536458333332 | 0.5948713541666666 |

Note for the record, not in the manuscript: `qwen3:8b` flagged 0 of 300 pilot
items but 2 of 5000 pool items (`pool_flag_count` = 2), so its pool labels were
not single-class and its student was trainable. Only `qwen3:4b` is degenerate.
The manuscript states this correctly.

### 1.2 Demonstration 2, the rival check (clinical)

Artifacts: `results/accord/c139_rule_prior/c139.json` and
`results/accord/labeller_interval_closure/intervals_17.json`
Pre-registration: `accord/c139_rule_prior/C139_PREREGISTRATION.md`

| Claim | Artifact and key | Value |
|---|---|---|
| n = 253, 12 positives | `c139.json::rule_prior.n`, `.n_pos` | 253, 12 |
| 17 configurations | `c139.json::n_configurations` | 17 |
| corrected threshold 0.0029 | `c139.json::bonferroni_alpha_over_n` | 0.0029411764705882353 |
| 16 of 17 reject chance at 0.05 | `intervals_17.json::summary.n_configurations_rejecting_chance_at_0_05` | 16 |
| the one exception | `intervals_17.json::summary.configurations_not_rejecting_chance_at_0_05` | ["promptsens::qwen3:8b::P_asserting"] |
| 6 survive Bonferroni vs chance | `intervals_17.json::summary.n_surviving_bonferroni_at_registered_B_20000` | 6 |
| the 6 names (Table 3 column B) | `...surviving_bonferroni_at_registered_B_20000` | list of 6 |
| 12 with bootstrap lower bound above 0.5 | `intervals_17.json::summary.n_cells_with_bootstrap_lower_bound_above_0_5_on_g253` | 12 |
| the 12 names (Table 3 column L>0.5) | `...cells_with_bootstrap_lower_bound_above_0_5_on_g253` | list of 12 |
| bootstrap 10,000 draws, seed 20260806 | `intervals_17.json::method.bootstrap_draws`, `.bootstrap_seed` | 10000, 20260806 |
| permutation 20,000 draws, seed 109 | `intervals_17.json::method.permutation_draws`, `.permutation_seed` | 20000, 109 |
| **rival AUROC 0.6478** | `c139.json::rule_prior.auroc` | 0.6478215767634855 |
| rival permutation p 0.0367 | `c139.json::rule_prior.permutation_p` | 0.03674816259187041 |
| rival bootstrap interval 0.4778 to 0.8024 | `c139.json::rule_prior.bootstrap.ci95` | [0.4778347524835128, 0.8024198705741962] |
| rival lower bound NOT above 0.5 | `c139.json::rule_prior.bootstrap.lower_bound_above_half` | false |
| rival LOPO range 0.6205 to 0.6852 | `c139.json::rule_prior.lopo_auroc_min`, `.lopo_auroc_max` | 0.6205205582798944, 0.6852131271218409 |
| **0 cells clear the corrected threshold** | `c139.json::summary.n_cells_clearing_bonferroni_graded_above_prior` | 0 |
| 0 in the other direction | `...n_cells_clearing_bonferroni_prior_above_graded` | 0 |
| **0 cells clear 0.05 uncorrected** | `c139.json::summary.n_cells_uncorrected_p_below_0_05` | 0 |
| 15 of 17 above the rival pointwise | `c139.json::summary.n_cells_with_graded_above_prior_pointwise` | 15 |
| strongest cell name | `c139.json::summary.strongest_cell` | promptconfirm::granite3.3:8b::P_rev_agree |
| strongest AUROC 0.7936 | `...strongest_cell_auroc` | 0.7935684647302904 |
| strongest difference +0.1457 | `...strongest_cell_diff` | 0.1457468879668049 |
| strongest SE 0.0986 | `cells["promptconfirm::granite3.3:8b::P_rev_agree"].se_diff` | 0.09864872294283605 |
| strongest z 1.4774 | `...z` | 1.4774330941036187 |
| strongest p 0.1396 | `...p_two_sided` | 0.1395595805234755 |
| strongest difference interval -0.0476 to 0.3391 | `...ci95_diff` | [-0.047604609001153764, 0.3390983849347636] |
| verdict B_NOT_SEPARABLE | `c139.json::summary.verdict` | "B_NOT_SEPARABLE" |
| rule strata 117/28/63/45 pairs | `c139.json::rule_strata.{1,2,3,4}.n_pairs` | 117, 28, 63, 45 |
| rule strata 3/3/4/2 positives | `...n_positives` | 3, 3, 4, 2 |
| rates 0.0256 / 0.1071 / 0.0635 / 0.0444 | `...pharmacist_precision` | 0.02564102564102564, 0.10714285714285714, 0.06349206349206349, 0.044444444444444446 |
| within-stratum AUROC 0.7339 / 0.5600 / 0.8771 / 0.8953 | `c139.json::secondary_within_rule_strata_descriptive.strata.{1,2,3,4}.auroc` | 0.7339181286549707, 0.56, 0.8771186440677966, 0.8953488372093024 |
| 11 cells with a rule-type cue either direction | `intervals_17.json::summary.cells_with_a_significant_rule_type_cue_in_either_direction` (length) | 11 |
| 0 confounded under the registered one-sided rule | `intervals_17.json::summary.n_confounded_cells` | 0 |
| 2 confounded direction-free | `...confounded_cells_direction_free_same_thresholds` (length) | 2 |
| 0 meet the full mirror of the bounded null | `...n_cells_meeting_the_full_mirror_of_the_bounded_null_on_g253` | 0 |
| detectability floor 0.7355 | `intervals_17.json::summary.registered_floor_g253` | 0.7355 |
| stratum-exact floor 0.7397 | `intervals_17.json::cells[*].g253.stratum_exact_hanley_mcneil_floor` | 0.7396795211041093 |
| 0 intervals begin above the floor | `...cells_whose_interval_begins_above_the_floor_on_any_stratum` (length) | 0 |
| 2 duplicate pairs, 15 distinct cells | `intervals_17.json::summary.duplicate_auroc_groups`, `.duplicate_note` | 2 groups |

Table 3's 17 rows of `auroc_graded`, `auc_diff_graded_minus_prior`, `se_diff` and
`p_two_sided` come one-to-one from `c139.json::cells[<key>]`. Every printed value
is that key rounded to 4 places.

### 1.3 Demonstration 3, data randomisation (clinical)

Artifact: `results/accord/labeller_interval_closure/permuted_label_control.json`
Pre-registration: `accord/labeller_interval_closure/PERMUTED_LABEL_PREREGISTRATION.md`

| Claim | Key | Value |
|---|---|---|
| seeds 42, 123, 2024 | `design.seeds` | [42, 123, 2024] |
| permutation seed 137, one draw, fixed across seeds | `design.permutation_seed`, `.n_permutation_draws`, `.permutation_fixed_across_seeds` | 137, 1, true |
| split seed 1337 | `design.split_seed` | 1337 |
| hyperparameters | `design.epochs/batch_size/lr/weight_decay/max_length/optimizer` | 3, 16, 2e-05, 0.01, 160, AdamW |
| 1183 train, 205 val, 299 gold, 291 leaked excluded | `gates.split_size.*` | 1183, 205, 299, 291 |
| join one-to-one | `gates.join_completeness.passed` | true |
| 1132 positives before and after | `gates.permutation.n_positive_before_permutation`, `.n_positive_after_permutation` | 1132, 1132 |
| marginal preserved | `gates.permutation.marginal_preserved` | true |
| **100 of 1183 rows changed, 0.0845** | `gates.permutation.n_rows_whose_label_changed`, `.fraction_of_rows_whose_label_changed` | 100, 0.08453085376162299 |
| 1132 of 1183 training rows positive | `gates.permutation.n_positive_before_permutation`, `.n_rows` | 1132, 1183 (see Section 0.6: the artifact's "94.7 percent" note is NOT this ratio and is not used) |
| **G299 ensemble AUROC 0.4640** | `g299.ensemble.auroc` | 0.46399535423925664 |
| G299 ensemble AP 0.0446 | `g299.ensemble.ap` | 0.04456352387707617 |
| G299 interval 0.3129 to 0.6177 | `g299.ensemble.bootstrap.ci95` | [0.3129145408163266, 0.6177424109048228] |
| per-seed G299 0.4980 / 0.4869 / 0.4233 | `g299.seed42.auroc`, `g299.seed123.auroc`, `g299.seed2024.auroc` | 0.4979674796747968, 0.4869337979094077, 0.42334494773519166 |
| G253 permuted 0.4810, AP 0.0570 | `g253.ensemble.auroc`, `.ap` | 0.4809820193637621, 0.05702244151821122 |
| G253 permuted interval 0.3238 to 0.6330 | `g253.ensemble.bootstrap.ci95` | [0.3238142109268143, 0.6329693767371589] |
| real-label 0.5360 | `comparison_vs_real_label_run.real_label_ensemble_auroc_g253` | 0.5359612724757953 |
| real-label interval 0.3454 to 0.7196 | `...real_label_bootstrap_ci95_g253` | [0.345434996612753, 0.7195712309820194] |
| paired difference 0.0561, interval -0.1108 to 0.2255, contains zero | `...paired_bootstrap_difference_same_253_pairs` | 0.05611879263949312, [-0.11081893313298274, 0.22552142564417804], true |
| Outcome A | `outcome` | "A" |
| the uninformativeness caveat | `what_this_does_not_establish`, `pool_label_marginal_note` | verbatim source of the manuscript's wording |

### 1.4 The ceiling check (clinical)

Artifact: `results/accord/c120_exact_join/c120.json`
Pre-registration: `accord/c120_exact_join/C120_PREREGISTRATION.md`
Register keys: `c120.g253.ceiling_mcc`, `c120.g253.fraction_of_ceiling`,
`c120.g299.ceiling_mcc`, `c120.g299.fraction_of_ceiling` in
`results/accord/VERIFIED_FACTS.json`.

| Claim | Key | Value |
|---|---|---|
| G253: 253 items, 12 positives, 223 flags | `g253.n`, `.n_positives`, `.n_flagged` | 253, 12, 223 |
| G253 flag rate 0.8814 | `g253.flag_rate` | 0.8814229249011858 |
| G253 cells 11/212/1/29 | `g253.tp/fp/fn/tn` | 11, 212, 1, 29 |
| G253 observed MCC 0.0243 | `g253.mcc` | 0.02432604884780849 |
| **G253 ceiling MCC 0.0818** | `g253.ceiling_mcc` | 0.08184465032907529 |
| G253 fraction of ceiling 0.2972 | `g253.fraction_of_ceiling` | 0.2972222222222222 |
| G253 hypergeometric p 0.5722 | `g253.p_hypergeom_one_sided` | 0.5721624076259317 |
| G253 random joins 0.5711 | `g253.random_join_fraction_ge_observed` | 0.5711 |
| G299: 299 items, 12 positives, 261 flags | `g299.n`, `.n_positives`, `.n_flagged` | 299, 12, 261 |
| G299 flag rate 0.8729 | `g299.flag_rate` | 0.8729096989966555 |
| G299 cells 11/250/1/37 | `g299.tp/fp/fn/tn` | 11, 250, 1, 37 |
| G299 observed MCC 0.0269, ceiling 0.0780, fraction 0.3443 | `g299.mcc`, `.ceiling_mcc`, `.fraction_of_ceiling` | 0.02686309383069346, 0.07802274386494407, 0.34429824561403505 |
| G299 p 0.5347, random joins 0.5442 | `g299.p_hypergeom_one_sided`, `.random_join_fraction_ge_observed` | 0.5347404622881329, 0.5442 |
| the ceiling construction is the same formula the gate uses | `accord/c120_exact_join/C120_PREREGISTRATION.md` lines 26 to 28 | "the same formula as marginal_feasibility_gate.py" |

### 1.5 Pharmacist annotator wording

The manuscript sentence "A pharmacist annotated 299 candidate pairs: 12
contradictions, 241 not contradictions, and 46 ambiguous" is the thesis sentence
at `accord/thesis/msu_ch4_results.tex:58` with the credentialing adjective
removed, per instruction. The base rate `0.0474` on G253 is stated at
`msu_ch4_results.tex:61`.

### 1.6 Compliance checks performed on the manuscript source

- No clinical note text, no `HADM_ID`, no `SUBJECT_ID`, no patient-level row.
  Every clinical number in the manuscript is a count, a rate, a correlation or
  an interval.
- Nothing under `~/mimic_local_only` was read.
- No hosted API provider or commercial model is named. Only local open-weight
  model tags appear.
- No en dashes and no em dashes in `main.tex`. Ranges in prose use "to". The
  only `--` in the project directory is in `refs.bib` page fields, which is
  standard BibTeX and matches the project's existing bibliography.
- No self-citation. `refs.bib` contains no entry by the author.
- The repository is not mentioned in the manuscript.
- The annotator's credentialing status is not mentioned.

---

## 2. Three reviewer objections I expect, and where the manuscript answers them

### Objection 1. "Your rival is fitted in sample, so of course nothing beats it.
### The non-separation is an artefact of an inflated comparator."

This is the strongest objection and the one a good reviewer will lead with.
Answered in two places and not buried:

- Methods, "The honest asymmetry" (Section 3.3): states that an in-sample rival
  is an upper reading of what the cheap variable can do and that this makes the
  comparison **harsher** on the stage, explicitly refusing the C139
  pre-registration's own description of the in-sample fit as "conservative in
  favour of the graded scores", which is the wrong direction.
- Discussion 5.1, second paragraph: states that the non-separation is "partly
  attributable to an inflated comparator", that some configurations might separate from a
  cross-fitted rival, and that Table 3 "is not evidence that the graded scores
  carry nothing" but evidence that "this study cannot distinguish them from a
  trivial metadata score, which is a weaker claim, and it is the only claim we
  make."

The defence of the design choice is also given: a project choosing its own
comparator will choose a weak one, and the fitted optimum has no free parameters
to tune. A cross-fitted rival is the obvious revision to offer if asked.

### Objection 2. "With 12 positives this study cannot show anything either way,
### so the negative results are uninformative."

Answered in Discussion 5.3 (Limitations), first paragraph, which concedes the
premise and then turns it: at 12 positives the permutation test against chance is
the only test with enough power to return a significant result, and the tests
with more realistic nulls do not. The paper's claim is about what licenses a
certification, not about the stage's true quality. Three independent supports
keep the argument alive at this sample size:

- The collapse finding does not depend on sample size at all. It is a count of
  verdict disagreements (zero) and an arithmetic property of a verdict function
  that reads its data only through `(N, K, n)`.
- The ceiling finding does not depend on the gold labels beyond their count. The
  bound of 0.0818 follows from three integers.
- Demonstration 1 has a 10,000 item test frame and 31 units, so the screening
  characteristics are not resting on 12 positives.

The manuscript also adds, under Table 1, that the screening proportions are
computed on 24 or 31 units and should be read with Wilson-interval width in mind,
and that this does not affect the exact disagreement count.

### Objection 3. "SNLI is not clinical text. Why is this in a biomedical
### informatics journal, and does the gate result transfer?"

Answered in Section 3.6 (materials) and Discussion 5.3. The design reason is
stated: the collapse finding needs a test frame large enough to measure student
quality precisely and a gold standard at scale, neither of which the clinical
corpus provides, and the binarised SNLI task carries the low prevalence that
makes the clinical problem hard. The transfer argument is stronger than external
validity: the collapse result does not depend on the data, only on the gate
reading its data through `(N, K, n)`. The clinical corpus then supplies the rival
and ceiling findings. The "method of methods" framing in the Introduction and the
Statement of Significance ties the scope to a biomedical problem space defined by
very low prevalence, one annotator, and label-correlated metadata.

### Objection 4, lower probability but fatal if unaddressed. "This is a negative-
### result paper and JBI only considers ML papers with substantial novelty
### beyond the state of the art."

Pre-empted by framing, not by argument. The phrase "method of methods" appears in
the Introduction ("A method of methods, not a new model") and in the Statement of
Significance, the paper explicitly disclaims a new classifier and any SoA
accuracy advance, and Discussion 5.2 says plainly that none of the three
components is new as a statistical idea and names what is new instead: the
composition, the pre-registerable specification, and the demonstrations. See
Section 3 below for the fallback article type.

---

## 3. Submission record

### 3.1 Article type

Submit as **Research Paper**.

Fallback if an editor pushes back on novelty: **Special Communication**, which
JBI defines (per the coordinator's reading of the guide for authors) as
"generalizable novel conclusions/recommendations/lessons-learned ... based on the
application of existing biomedical informatics methods". That description fits
this paper exactly: three existing methods, applied to a problem space, yielding
a generalizable recommendation. It is not invitation only. Do not volunteer the
fallback in the cover letter; hold it for a revise-or-reclassify response.

### 3.2 Prior publication disclosure

This work appeared in the author's MS thesis, which is deposited and public.
Elsevier does not treat a thesis as prior publication. The JBI submission
declaration states that work must not "have been published previously except in
the form of a preprint, abstract, published lecture, **academic thesis**, or
registered report" (guide for authors,
`https://www.elsevier.com/journals/journal-of-biomedical-informatics/1532-0464/guide-for-authors`,
which redirects to the ScienceDirect guide page; see Section 0.4 of this file:
the quoted sentence was supplied by the coordinator and I could not read the live
page, so **confirm the wording before relying on it**).

Disclose anyway. Suggested cover-letter sentence:

> Portions of this work were reported in the author's MS thesis, which is
> deposited and publicly accessible. Per the journal's submission declaration,
> prior appearance in an academic thesis does not constitute prior publication.
> The manuscript submitted here is a new and substantially reframed treatment:
> the thesis reports the three studies as chapters of an applied project, whereas
> this manuscript extracts and specifies the checking procedure as a method of
> methods and reports the three studies as demonstrations of it.

Also disclose in the cover letter that two readings carried in the thesis are
corrected here: the mechanism in Demonstration 2 is named as confounding by
rule-correlated content rather than leakage of a rule identifier (the leakage
description was withdrawn), and the C139 pre-registration's description of the
in-sample rival as "conservative in favour of the graded scores" is replaced with
the correct direction.

### 3.3 Graphical abstract

Required at submission. Do not over-invest. Suggested content, one landscape
panel split in two, both halves built from numbers already in the manuscript:

- **Left half, "Certified against chance."** The 17 clinical configurations as a
  column of dots against a vertical chance line at 0.5, with a large caption
  "16 of 17 reject chance" and the strongest value 0.7936 called out.
- **Right half, "Not certified against the rival."** The same 17 dots against a
  vertical line at the rival's 0.6478 with its shaded interval, caption
  "0 of 17 separate from a score built from four numbers", smallest p 0.1396
  called out.
- A single strip beneath both halves: "same data, same 253 pairs, 12 positives."

This is panel (a) and panel (b) of Figure 2 with heavier labelling, so
`make_figures.py` can be extended with a `figure_graphical_abstract` function
rather than a new pipeline. Keep text at 10 pt or larger and keep it readable in
greyscale.

### 3.4 Float budget

4 tables and 2 figures, 6 of the permitted 8 combined. Two slots left for
revision.

---

## 4. Figure generator

`make_figures.py` reads only
`results/accord/c136_gate_validation/c136.json` and
`results/accord/c139_rule_prior/c139.json`, resolves the repository root as two
directories above itself (overridable as `argv[1]`), and writes `fig1.pdf` and
`fig2.pdf` **into the paper directory root**, not a subfolder. It fails loudly if
either artifact is missing. It hard-codes no data value: every number plotted,
including the two threshold lines, the rival's AUROC and interval, the
disagreement count and both Youden J values printed in the Figure 1 title, is
read from the JSON at run time. Requires `matplotlib`. Figures are greyscale-safe
(marker shape and fill carry the distinctions, not colour) and are emitted as PDF
with `pdf.fonttype` 42.

The script is run by hand. `main.tex` never invokes it, and nothing in the
document requires shell escape.

Two values in `main.tex` figure captions duplicate values the script prints into
the figures themselves (19.5833 and 3.5833 in Figure 1; 0.6478, 0.4778 and 0.8024
in Figure 2). After the first successful build, confirm the caption text matches
what the script rendered.

---

## 5. Editorial Manager build constraints, and how this directory satisfies them

| Constraint | Status |
|---|---|
| bibtex, not biber, not biblatex | `\bibliographystyle{elsarticle-num}` + `\bibliography{refs}`. No biblatex, no `\printbibliography`, no `\usepackage[...]{natbib}` (the class loads natbib itself). |
| ship the generated `.bbl` | **NOT YET DONE.** The build was not run. See Section 0.1. |
| flat directory, no subfolders | Flat. `main.tex`, `refs.bib`, `make_figures.py`, `NOTES.md`, `elsarticle.cls`, `elsarticle-num.bst` and (after the build) `fig1.pdf`, `fig2.pdf`, `main.bbl`. No directory is created by anything here. |
| `\includegraphics` with no path | `\includegraphics[width=\textwidth]{fig1}` and `{fig2}`. No path, no extension. |
| filenames: one period, no special characters | `fig1.pdf`, `fig2.pdf`. The earlier drafts used `fig1_collapse.pdf` and then `fig1-collapse.pdf`; both were renamed. |
| no two files sharing a basename | Satisfied for figures. `main.tex` / `main.bbl` / `main.pdf` necessarily share `main`, which is the manuscript itself and is what EM expects. |
| no shell escape, no minted, no TikZ externalization, no pgfplots | None used. Only `amsmath` is loaded by the preamble; `graphicx` comes from the class. Figures are pre-built static PDFs. |
| TeX Live 2024, pdflatex | `amsmath` and `graphicx` are both in `collection-latex`, so the document builds under BasicTeX with the staged `elsarticle.cls`. `booktabs` was removed for exactly this reason: it lives in `collection-latexrecommended`, which BasicTeX does not install. All four tables now use plain `\hline`. |
| one style file | `elsarticle.cls` plus `elsarticle-num.bst`. No second class or style. |
| pdf image format | Both figures are PDF. |
| `review` option for the submitted version | `\documentclass[review]{elsarticle}`. |

### 5.1 Why the class option list is exactly `[review]`

`elsarticle.cls` v3.4 calls `\ProcessOptions` (line 113), not
`\ProcessOptions*`, so declared options execute in **declaration order**, not in
the order the user wrote them. `review` is declared at line 75 and runs
`\global\preprinttrue\gdef\@blstr{1.5}`. `3p` is declared at line 84 and runs
`\global\preprintfalse`. Passing `[review,3p]` therefore executes `review` first
and then silently cancels it, losing the double spacing and the reviewer line
numbers. An earlier draft of this manuscript had `[review,3p,number]` and would
have compiled without error into the wrong layout. `number` is already the class
default via `\ExecuteOptions{...,number,preprint,...}` at line 111, so it is
redundant and was dropped. The `times` option was not used, because it pulls in
`txfonts` from `collection-fontsrecommended`, which BasicTeX does not install.

### 5.2 Files to exclude from the submission bundle

`NOTES.md`, `make_figures.py`, `REFERENCE-elsarticle-template-num.tex` and
`REFERENCE-elsdoc.pdf`. The first two belong in the repository, not in EM; the
last two are the coordinator's reference copies and are to be removed at
packaging time.
