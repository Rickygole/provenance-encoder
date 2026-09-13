# C107 pre-registration: a permutation test for the TF-IDF baseline, and a
# gold-only nested-CV baseline on G_253

Committed before the first datum. The outcome is appended below after the run; the
rule above the outcome line is never edited.

## Why

Chapter 4's bag-of-words subsection (`accord/thesis/msu_ch4_results.tex`, the
"A bag-of-words baseline" subsection, table `tab:baselines`) reports TF-IDF plus
logistic regression at AUROC 0.5761 / AP 0.1571 on $G_{299}$, well above the
no-skill line, and then dismisses the whole comparison as noise one section later
("The bag-of-words baseline outranking the neural configurations is not a real
ordering either... sits inside that same noise") without ever running a test
against the baseline's own no-skill line. The same subsection also states the
baseline "was not run on $G_{253}$" (line ~937). This pre-registration commits a
test for the first claim and closes the second gap.

## What runs

### Task (a): reproduction, G_253 extension, and a label-permutation null

1. Reproduce the registered TF-IDF baseline exactly: import `build_splits` from
   `accord.control.data` (unmodified) and refit the identical pipeline
   `accord.control.baselines` fits at module level -- TF-IDF, `max_features=5000`,
   `ngram_range=(1,2)`, `min_df=2`, over `nursing_sentence + " [SEP] " +
   discharge_sentence`, feeding `LogisticRegression(max_iter=2000,
   random_state=42)` trained on the same 1,183-row weak-label pool, scored on the
   299-row gold set. `accord.control.baselines.main()` itself is not called (it
   writes into `~/mimic_local_only/accord_control`); only its already-committed
   output, `results/accord/control/baselines.json`, is read for the assertion.
2. Assert the reproduction matches the registered numbers to 4 decimals:
   `round(auroc, 4) == round(0.5760743321718931, 4)` and
   `round(ap, 4) == round(0.1570977148911732, 4)` on `gold_all` (n=299). Abort
   (raise, write nothing) if either assertion fails.
3. $G_{253}$ is `splits.gold["pharm_label"] != "ambiguous"` (12 positives, 241
   negatives; this is the same column `data.py::load_gold` already populates, not
   a new join). Score the SAME fitted classifier -- no refit -- on that subset,
   exactly as the registered $G_{134}$ row already does for a different mask.
   This is the "run on $G_{253}$" the thesis says never happened.
4. Label-permutation null, holding the fitted scores fixed: one shared
   `numpy.random.SeedSequence(42)`, spawned into two independent child streams (one
   per evaluation set, so $G_{299}$ and $G_{253}$ do not share a permutation
   tree -- the convention this project adopted after C101). For each set, draw
   10,000 permutations of the gold label vector *restricted to that set's own
   rows*, recompute AUROC and AP against the fixed scores each time, and report
   the one-sided p as `(null_statistic >= observed_statistic).mean()` over the
   10,000 draws, for AP and for AUROC, on $G_{299}$ and on $G_{253}$ (four p
   values total).

### Task (b): gold-only TF-IDF logistic regression, nested leave-one-positive-out CV, $G_{253}$

No weak-label training pool is used here; the classifier is fit only on gold rows,
in cross-validation, so the "gold-only" claim holds without a fitted-elsewhere
score leaking in.

- **Outer loop, one fold per positive.** The 12 positives and 241 negatives are
  each independently shuffled by a spawned child of the same `SeedSequence(42)`
  tree. The 241 shuffled negatives are cut into 12 contiguous chunks: 241 = 12
  times 20 plus 1, so chunk 0 gets 21 negatives and chunks 1 through 11 get 20
  each. Outer fold $i$: test = positive $i$ union negative chunk $i$; train =
  every other row. This is a partition, so pooled out-of-fold predictions cover
  all 253 rows exactly once, each scored by a model that never saw it in training.
- **Inner loop, model selection.** Within each outer-train set (11 positives plus
  220 or 221 negatives), `sklearn.model_selection.GridSearchCV` over a
  `Pipeline(TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=2),
  LogisticRegression(max_iter=2000, random_state=42))`, `cv=StratifiedKFold(5,
  shuffle=True, random_state=42)`, `scoring="average_precision"`, grid
  `C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]`. `GridSearchCV` refits the pipeline
  (vectorizer included) on each inner fold independently, so the vectorizer's
  vocabulary is never fit on a row later scored by that fold. The best `C` refits
  on the full outer-train set and scores the outer-test set.
- **Pooled evaluation.** One AUROC, one AP, and a one-sided Mann-Whitney U p
  (positives greater than negatives) over the 253 pooled out-of-fold scores.
  A second label-permutation null, same recipe as task (a) (10,000 draws, a third
  spawned child of the same seed tree), on the pooled out-of-fold scores.

## Decision rule

**Task (a), primary:** Outcome A, the linear baseline carries signal -- the
one-sided permutation p for AP on $G_{299}$ is below 0.05. Outcome B, otherwise.
Reported either way, alongside the same test for AUROC and for $G_{253}$, none of
which is the deciding number for A vs B.

**Task (b), secondary,** measured against the pre-registered
`mde_recomputed.gold_ambiguous_excluded.minimum_detectable_auc` (0.7355, read from
`results/accord/VERIFIED_FACTS.json`, not recomputed here): Outcome A if the
pooled out-of-fold AUROC is at or above 0.7355 AND the one-sided Mann-Whitney p is
below 0.05. Outcome B otherwise. This is reported beside task (a), not folded into
it, and is not a test of any RQ.

**Outcome C, invalid, either task:** the reproduction assertion in task (a) step 2
fails, or fewer than 253 gold rows are available for either task, or a fold in
task (b) cannot be constructed as specified (e.g. a class disappears from an
inner-training fold).

Task (a) adds four p values (AP and AUROC, on $G_{299}$ and $G_{253}$) to the
hypothesis family `accord/stats/test_register.py` enumerates for Chapter 4's
multiplicity subsection. This pre-registration does not edit that register or its
count; it records here, before the run, that the family the thesis reports is
short by however many of these four the chapter goes on to quote, and leaves the
correction to whoever next edits `test_register.py`.

## Kill condition

No change of vectorizer, classifier, grid, or fold construction after results are
seen. If a run is interrupted, it restarts from scratch, never spliced. The
reproduction assertion (task a, step 2) is a hard abort, not a warning: if it
fails, no permutation number is computed or reported from that run.

## Outcome

