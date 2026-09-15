# C130 pre-registration: does the ranking null reproduce with a reproducible labeller in place of the lost deployed run?

Committed before the first datum. The outcome is appended below after the run; the rule above the outcome line is
never edited.

## Why

The detector reported in the thesis (`accord/control/train.py`, three seeds, `results/accord/decontam/`) was trained
on `weak_label` from `~/mimic_local_only/accord_supp/accord_weak_supervision_labels_decontam.csv`: 1,425 of 1,679
pool rows positive (0.8487), produced by a deployed weak-labelling run whose script and exact prompt were never
saved. `accord/differential/run_multillm_replication.py` reuses the archived verbatim prompt (`prompts_existing.csv`,
`config_id == "llama_baseline"`) against local `llama3:8b` and is the only reproducible stand-in for that lost run.
This pre-registration retrains the same control configuration with the replication run's labels, on the same
subject-disjoint training rows, to ask whether the detector's chance-level ranking of the gold pairs (the thesis's
headline null) survives when the lost labeller is replaced by a labeller whose script and prompt are both on disk.

## A gap found while preparing this run, recorded here because it bears on what can be gated today

`run_multillm_replication.py` was executed for `llama3:8b` (`~/mimic_local_only/accord_multillm/run_log.txt`,
`results/accord/multillm/per_model_results.json`, register prefix `overcall.llama3_8b.*`), but by design (its own
docstring: "Writes ONLY aggregate results ... to `~/mimic_local_only/accord_multillm/`. No note text, and no per-row
raw model output text, is ever written to disk or printed") it never persisted the per-row prediction vector for
either the 1,679-row candidate pool or the 299-row gold set. `reg_df[f"pred_{model}"]` and `gold_join[f"pred_{model}"]`
exist only in that process's memory; only aggregate counts (`n_positive`, `n_negative`, the pool/gold confusion cells,
the language-regression coefficients) reached disk. A search of `~/mimic_local_only` and `results/accord/` for a
per-row `llama3:8b` pool-label file (by row count, by column name, by content) found none. So the premise that the
per-row replication labels are already sitting under `accord_multillm` does not hold: **the labelling inference has
already been run once, but its row-level output was discarded, not merely uncatalogued.**

This run therefore needs a companion artifact that does not exist yet:

- **Path (fixed now):** `~/mimic_local_only/accord_multillm/pool_replication_labels_llama3_8b.csv`
- **Required columns:** `HADM_ID, drug, rule, nursing_sentence, discharge_sentence, replication_label` where
  `replication_label` is one of `contradiction`, `not_contradiction`, `UNPARSEABLE` (the same three-way category
  `run_multillm_replication.py::parse_prediction` already produces; `UNPARSEABLE` must not be silently folded into
  either class, matching the fix C10 required).
- **Coverage:** one row per row of the 1,679-row pool
  (`accord_weak_supervision_labels_decontam.csv`), same join key.
- **Provenance:** must come from the same prompt, model, temperature, and seed
  `run_multillm_replication.py` already used for `llama3:8b` (`llama_baseline` verbatim prompt, temperature 0, seed
  42, `num_predict` 50) so that this run's own marginal-match gate (below) can confirm it is the same run, not a
  drifted rerun.

Producing this file means re-running `llama3:8b` over the 1,679-row pool through Ollama (`run_log.txt` shows the
original pass took 1,181s / about 20 minutes for the pool alone). That is exactly the kind of Ollama run this task
was told not to start now, for the same memory-contention reason training is deferred. **This pre-registration and
its code are complete and committed now; the label file's absence is not a defect in this run, it is the reason
`--dry-run` exists** -- it lets the gates below be checked, and found wanting on this one point, without spending any
compute. The coordinator must either regenerate the per-row file (a small, additive change to
`run_multillm_replication.py`, or a new companion script, neither written here since it was not asked for) before
launching the full run, or the full run will abort at the same gate with Outcome C.

## What is retrained

Exactly the `accord/control/train.py` configuration (BiomedBERT cross-encoder, `MAX_LENGTH=160`, `lr=2e-5`,
`batch_size=16`, `epochs=3`, `AdamW`, `weight_decay=0.01`), on `accord.control.data.build_splits()`'s train/val/gold
split, unmodified except for one substitution: the training pool's label column is replaced. `train.py` is not
edited; `accord/c130_reproducible_student/c130_run.py` gets the substitution into the training loop by supplying its
own `build_splits`-equivalent in place of `accord.control.train`'s imported name, so the training loop, optimizer,
scheduler-free schedule, seeding, and gold-scoring code inside `train.py` run unchanged.

- **Same rows.** `accord.control.data.build_splits()`'s `train` split: the 1,183-row, subject-disjoint (from the 266
  gold subjects), row-level-decontaminated training pool, unchanged. Verified now (2026-09-14, CPU, no model load):
  `build_splits()` returns `n_train=1183`, `n_val=205`, `n_leaked_rows_excluded=291`, matching the numbers this task
  states for the lost run's training pool.
- **Only the label column replaced.** `y` for those 1,183 rows comes from `replication_label == "contradiction"`
  (`UNPARSEABLE` rows are excluded from the training pool, not coerced to a class -- see the join gates below), in
  place of `weak_label == "contradiction"`.
- **Validation split (205 rows) is NOT relabelled.** The task's instruction is specific to "the SAME subject-disjoint
  1,183-row training pool"; `val_loss` in `train.py` is diagnostic only (no early stopping, no model selection on it),
  so leaving it on the original label keeps this a single, stated, minimal substitution rather than an unstated
  second one.
- **Gold set (299 rows) is untouched.** `pharm_pos` (pharmacist label) is still the evaluation ground truth; nothing
  about gold scoring changes.
- **Same three seeds, same hyperparameters** as `accord/control/run_all.sh`: 42, 123, 2024; 3 epochs each.
- **Wall time (estimated from this exact configuration's own prior run, not a new benchmark):**
  `~/mimic_local_only/accord_control/metrics_seed{42,123,2024}.json` record per-epoch times for this identical
  model/data-size/hyperparameter configuration on this machine's MPS device: seed 42 (170.2 + 100.5 + 144.2 = 415s
  training, first seed pays one-time model-download/compile cost), seed 123 (97.2 + 89.6 + 92.0 = 279s), seed 2024
  (94.7 + 88.9 + 84.8 = 268s). File-mtime deltas for that run put the full per-seed wall time (model load + 3 epochs +
  gold scoring + file write) at about 5 minutes for the first seed and about 4:45-5:00 for the other two. C130 trains
  the identical architecture on the identical row count (1,183 train / 205 val / 299 gold), so **estimated wall time
  is about 5 minutes per seed, about 15 minutes for the three seeds run serially**, plus the label-generation Ollama
  pass noted above if that file does not yet exist (about 20 minutes, from the same run's own pool timing).

## Join key (fixed now)

`HADM_ID, drug, rule, nursing_sentence, discharge_sentence` -- the same five-column key
`accord/control/data.py`, `accord/differential/run_multillm_replication.py`, and `accord/stats/reconcile_c28.py`
already use to join `accord_weak_supervision_labels_decontam.csv` against
`accord_decontamination_audit.csv` / `annotation_sample_300.csv`. Not `pair_id`: the 1,679-row pool carries no
`pair_id` column (that identifier exists only on the 300-row annotation sample and the 299-row gold set).

## Gates (checked by `c130_run.py --dry-run`, no model loaded, no Ollama call, CPU only)

1. **Split-size gate.** `accord.control.data.build_splits()` returns `n_train == 1183`. If not, abort (Outcome C):
   the underlying source files have changed since this document was written and the "same 1,183-row pool" premise no
   longer holds.
2. **Label-file-exists gate.** The replication label file (path above) exists and is readable. **If missing, this is
   reported and the run stops here** -- this is the gap described above, expected to be the actual result of today's
   dry run.
3. **Label-file-uniqueness gate.** The replication label file has no duplicate rows under the join key (each pool row
   the file claims to cover, it covers exactly once).
4. **Join-completeness gate ("every pool row has exactly one replication label").** Left-joining the 1,183-row
   training split onto the replication label file on the join key yields exactly 1,183 rows (no fan-out) and zero
   rows with a missing `replication_label` (no pool row left unlabelled).
5. **Marginal-match gate.** The replication label file's `contradiction` rate over its `llama3:8b`-parseable rows,
   restricted to the 1,679-row pool, rounds (4 dp) to the already-registered
   `overcall.llama3_8b.pool.positive_rate` = 0.9469922572960096 (`results/accord/VERIFIED_FACTS.json`, source
   `results/accord/multillm/per_model_results.json`). A mismatch means the file is not the same replication run this
   pre-registration cites (a drifted rerun, a different model, or a different prompt), and the run aborts (Outcome
   C) rather than silently using it.

None of these five gates trains a model. The dry run either passes all five and reports the sizes it would train on,
or reports exactly which gate stopped it and why, and writes only that gate summary to
`~/mimic_local_only/logs/c130_run_<timestamp>.log`; per instruction, it writes nothing to `results/` in `--dry-run`.

## What is scored, once training runs

- **G_299**: all 299 gold pairs, `pharm_pos` as truth (ambiguous pairs counted as negative, matching
  `accord/control/report.py`'s own `gold_all` convention).
- **G_253**: the 253 non-ambiguous gold pairs (`pharm_label != "ambiguous"`; 12 positive, 241 negative -- the same
  composition `accord/public_controls/sizectl_c108/power_curves.py` used for the registered power floor cited below,
  and the same 12/241 cited in `accord/c126_teacher_student/C126_PREREGISTRATION.md`'s deployed-labeller AUROC).
- **Ensemble** = the unweighted mean of the three seeds' sigmoid scores on the gold set (identical construction to
  `accord/control/report.py` and `accord/c126_teacher_student/c126_run.py`).
- **Percentile bootstrap 95% CI on AUROC**, B=10,000, seed 42, `accord/control/report.py::bootstrap_auroc_ci`
  (resamples with a single class present are skipped and do not count toward the 10,000), on both G_253 and G_299,
  for the ensemble and for each seed individually.
- **Average precision (AP)** on both strata, ensemble and per seed.
- **Teacher signal, where available.** "The teacher" is the `llama3:8b` replication run's own prediction, used as a
  0/1 score, exactly as `accord/c126_teacher_student/C126_PREREGISTRATION.md` used the deployed weak labeller's
  label as a 0/1 score against the pharmacist. On **G_299**, this is already derivable without any new run: the
  registered `llama3:8b` gold confusion cells (`results/accord/multillm/per_model_results.json`'s
  `per_model["llama3:8b"]["gold"]["confusion"]`, TP=12 FP=273 FN=0 TN=14) give recall=1.0, specificity=14/287=0.04878,
  and AUROC-of-a-binary-score = (recall + specificity) / 2 = 0.52439 (the identical construction C126 used for the
  deployed labeller). **On G_253, the teacher AUROC is not derivable from any existing artifact**: the registered
  gold confusion treats all 299 pairs with ambiguous folded into the negative side, and there is no per-pair-id
  breakdown on disk that would let the 46 ambiguous rows be excluded from that confusion matrix. `c130_run.py` will
  compute the G_253 teacher figure only if a per-row `llama3:8b`-replication prediction file for the 299 gold pairs
  is also present at
  `~/mimic_local_only/accord_multillm/gold_replication_labels_llama3_8b.csv` (same schema, gold's join key). If that
  file is absent (true today), the G_253 teacher entry in the artifact is written as `null` with a stated reason;
  this does not block the primary decision rule (below), which is about the student's own AUROC, not the
  teacher-student gap.
- **Student-minus-teacher paired bootstrap difference**, B=10,000, seed 43, on whichever of G_253/G_299 the teacher
  score is available for -- identical construction to `accord/c126_teacher_student/c126_run.py::paired_diff`
  (resample pairs jointly for student and teacher, same draw).

## Decision rule (fixed now)

The decision-relevant stratum is **G_253**, because the registered power floor below is calibrated to exactly that
stratum's composition (12 positive, 241 negative -- `results/accord/public_controls/sizectl_c108/power_curves.json`,
register keys `sizectl108.power_curves.n_pos_slots`=12, `n_neg_slots`=241). G_299 is reported alongside, descriptive
only, for continuity with `accord/control/report.py`'s own primary stratum.

- **"The ranking null reproduces without the lost run"**: the ensemble's G_253 AUROC 95% bootstrap CI includes 0.5
  AND its upper bound is below 0.7466 (`sizectl108.power_curves.primary_crossing_power_0_80`, the AUROC at which this
  gold set's simulated power curve crosses 80% -- the same ceiling `accord/control/report.py` and
  `accord/c112_decontam_defectfree/` cite for this stratum's detection floor).
- **"Does not reproduce"**: the CI excludes 0.5 on the high side (lower bound above 0.5).
- **Inconclusive**: the CI includes 0.5 but its upper bound is at or above 0.7466 -- the CI is too wide to
  distinguish "chance" from "a real effect this evaluation could not resolve," the same caveat
  `accord/control/report.py`'s own `INCONCLUSIVE_WIDE_CI` branch and `accord/c112_decontam_defectfree/`'s registered
  interpretation both already carry for this gold set's size.
- **Invalid (Outcome C)** if: the label join fails (any of gates 2-4 above), the replication pool marginal does not
  match its registered value (gate 5), or a seed fails to train (a non-zero exit, a NaN loss, or a metrics file not
  written for that seed). Per instruction, Outcome C on any of these grounds stops the run before any AUROC is
  computed; no partial decision is recorded.

## Kill condition

No stratum, metric, seed, or hyperparameter is changed after any number from this run is seen. A dry-run gate
failure reported today (expected: gate 2, the label file does not exist yet) is not a reason to loosen gate 2 to a
softer check; it is a reason to defer the run until the file exists, exactly as instructed.

## Compliance

Local only. `c130_run.py` never runs Ollama and never calls a hosted API. Any code path that loads
`accord.control.data` or reads `~/mimic_local_only` redirects stdout/stderr to a file under
`~/mimic_local_only/logs/`; that log is never read back into this task, per instruction. Only the aggregate JSON
under `results/accord/c130_reproducible_student/` (counts, AUROC, AP, CI bounds, seeds, booleans -- no `pair_id`, no
`HADM_ID`, no `SUBJECT_ID`, no note text) and a short numeric summary reach real stdout or `results/`. `--dry-run`
writes nothing to `results/` at all; its gate summary goes only to the local log, and a short pass/fail-per-gate line
(counts and booleans only, never a join key value) reaches real stdout so the coordinator can see the result without
opening the log. `register_c130.py` (prefix `c130`) is written but not run, per instruction; it is for a separate
agent to execute once the full run has produced `results/accord/c130_reproducible_student/c130.json`.

## Addendum: input generation

Committed before `persist_replication_labels.py` is run, per instruction: this section describes the
label-generation step, not its results, so it does not touch the decision rule above.

The gap the dry run found (gate 2, "A gap found while preparing this run" above) is closed by
`accord/c130_reproducible_student/persist_replication_labels.py`, which re-runs the identical `llama3:8b` inference
`run_multillm_replication.py` already ran once and persists the per-row output that run discarded. It does not
re-derive the prompt or the parsing logic by hand: `SYSTEM_PROMPT`, `USER_TEMPLATE`, `TEMPERATURE`, `SEED`, and
`MAX_TOKENS` are statically parsed out of `run_multillm_replication.py` (via `ast`, the module is never imported --
importing it would fire its own top-level Ollama run for all three models) and asserted equal to local copies before
anything else runs; `parse_prediction` and `call_ollama`'s function bodies are lifted by the same static parse and
executed verbatim in a private namespace, so the request construction and response parsing are the original code,
not a re-implementation of it. The one deliberate deviation is `OLLAMA_URL`, pinned to the literal loopback IP
`127.0.0.1` (the original used `localhost`) per this task's data-agreement requirement that Ollama is called only at
`127.0.0.1:11434`; this changes nothing the model receives.

Row source and order: the pool is `ws`, read directly from `accord_weak_supervision_labels_decontam.csv` in file
order, not `run_multillm_replication.py`'s own `reg_df` (that file's `ws` left-joined with regression covariates it
does not need here). This is a deliberate simplification, not a deviation, because that run's own archived output
(`run_log.txt`, `per_model_results.json`) already shows a 1.0000 covariate match rate and
`per_model["llama3:8b"]["pool"]["n"] == 1679`: no row was dropped, and a left join preserves row order, so `reg_df`
and `ws` are the same 1,679 (nursing, discharge) pairs in the same order. The gold set is
`accord.control.data.load_gold()`, the same `pair_id <-> HADM_ID` join `run_multillm_replication.py` performs inline,
asserted to the same 299 rows.

Outputs: `~/mimic_local_only/accord_multillm/pool_replication_labels_llama3_8b.csv` (1,679 rows) and
`.../gold_replication_labels_llama3_8b.csv` (299 rows), both `HADM_ID, drug, rule, nursing_sentence,
discharge_sentence, replication_label`, plus `results/accord/c130_reproducible_student/replication_relabel_aggregates.json`
(pool/gold positive and parse-failure counts, the marginal-match check against the already-registered
`overcall.llama3_8b.pool.positive_rate`, an Ollama-egress boolean, wall time). If the relabelled pool's positive rate
does not round to the registered value at 4 decimals, the aggregates file records `outcome: "C"` and the script exits
non-zero, so the pipeline chain (`persist_replication_labels.py && c130_run.py --dry-run && ...`) stops before
`c130_run.py`'s own gate 5 would independently reach the same conclusion, and before any training.

## Outcome

(appended after the full run; not yet run. The dry-run gate results from 2026-09-14, run before any training
started, are reported in this task's final summary rather than here, since `--dry-run` intentionally writes nothing
to this repository beyond this file and the code below.)
