# C109 pre-registration: does the C103 reversed-prompt AUROC generalise across wording?

Committed before the first datum. The outcome is appended below after the run; the
rule above the outcome line is never edited (CLAUDE.md convention set by C44).

## Background

Under C103, `llama3:8b` with a reversed prompt ("are these consistent?", a "no"
read as conflict) reached AUROC 0.7503 on the 253 non-ambiguous gold pairs (12
positives), and `granite3.3:8b` reversed reached 0.7365, both just above the
registered minimum detectable AUROC 0.7355
(`results/accord/mde_recomputed.json::strata.gold_ambiguous_excluded.minimum_detectable_auc`).
That result was secondary and descriptive (C103_PREREGISTRATION.md's Secondary
section), not part of C103's primary decision rule.

An independent audit (`accord/prompt_sensitivity/audit_reversed.py`, artifact
`results/accord/prompt_sensitivity/audit_reversed.json`) reproduced the
`llama3:8b`/P_reversed cell exactly and found it correctly oriented but fragile:
leave-one-positive-out ranges the AUROC from 0.7276 to 0.7801 (below the MDE in 2
of 12 drops), and a stratified bootstrap (5,000 draws, seed 42) gives a 95 percent
interval of about 0.64 to 0.86. The effect is concentrated in Rules 1 and 2
(AUROC 0.8525, n 145, 6 positives) -- the rows this project has already found to
carry an exemplar-selection defect -- and falls to 0.6042 (p 0.20) on the
defect-free Rules 3-4 stratum (n 134, 6 positives), whose MDE is 0.8058.

## Question

Is the C103 reversed-prompt result contradiction detection that generalises
across wording, or a cue concentrated in the rule-mined Rule 1/2 rows?

## What runs

- Models: `llama3:8b` and `granite3.3:8b` only, served locally by Ollama on
  `127.0.0.1:11434`, temperature 0, `think` recorded explicitly as `False` for
  every call, via `accord/differential/model_tags.py::ollama_model_tag(label,
  False)`: `llama3:8b` -> `_llama3_8b_nothink`, `granite3.3:8b` ->
  `_granite33_8b_nothink`. Same `/api/chat` call shape as `run_c103.py`: one
  system message plus one user message, `stream: false`, `logprobs: true`,
  `top_logprobs: 8`, `num_predict: 1`, `num_ctx: 4096`.
- Pairs: the 299 annotated gold pairs (`G_299`) from
  `~/mimic_local_only/from_drive_2026-08-08/outputs/pharmacist_labeled_clean.csv`,
  `label_clean` column; `nursing_sentence` and `discharge_sentence` substituted
  into each prompt. `G_253` (the 253 pairs with `label_clean != "ambiguous"`, 12
  positive) is the AUROC basis for the primary decision rule, the registered
  confound test, and the registered stratum report, matching C103's and C98's
  basis exactly.
- Prompts: frozen, committed verbatim in
  `accord/prompt_sensitivity/prompts_c109.py`, never edited after this file is
  committed. Four cells per model:
  - **P_rev_agree**: "Do these two notes agree on the medication instruction?
    Answer Yes or No."
  - **P_rev_follow**: "Could a clinician follow both of these medication
    instructions at the same time? Answer Yes or No."
  - **P_rev_compatible**: "Is the discharge summary compatible with the nursing
    note about the medication instruction? Answer Yes or No."
    (P_rev_agree, P_rev_follow, P_rev_compatible are the three NEW paraphrases;
    none was used in C103.)
  - **P_reversed_rerun**: the ORIGINAL C103 `P_reversed` prompt ("are these two
    sentences consistent with each other about medication instructions?"),
    imported byte-for-byte from `accord/prompt_sensitivity/prompts.py`
    (`PROMPTS["P_reversed"]`), never retyped, and re-run under this harness as a
    determinism check against the already-committed C103 cache.
  - All four keep the reversed convention: a "No" is the flag. Score =
    log p(No) - log p(Yes), the same sign convention `run_c103.py` uses for
    P_reversed (raw label/log-odds parsed over the Yes/No vocabulary, then
    flipped at the point of use in `run_c109.py`, never by rewriting the
    vocabulary sets themselves).
- Recorded for every call: the parsed hard label (first word of the actual
  reply, matched case-insensitively against Yes/No, punctuation-stripped;
  unmatched is a parse failure, recorded as its own outcome, never silently
  dropped) and the first-token log-odds (each side floored at -20 if absent from
  the top 8; degenerate = either side floored), mapped through the reversed
  flag exactly as C103 does.
- Per-pair numeric records (pair_id, flag label, flag log-odds, degenerate flag,
  parse-failure flag, thinking-non-empty flag, prompt-truncation flag) are
  cached locally under `~/mimic_local_only/accord_prompt_sensitivity_cache/*.jsonl`
  -- numbers and the project-internal pair_id only, never sentence text -- the
  same directory and convention C103 used. C109's prompt names
  (`P_rev_agree`, `P_rev_follow`, `P_rev_compatible`, `P_reversed_rerun`) are new
  filenames there; nothing C103 wrote is read for write, only read for the
  determinism check.

## Primary decision (llama3:8b only; the cell that motivated this pre-registration)

**Outcome A, wording-robust:** at least 2 of the 3 NEW paraphrases
(`P_rev_agree`, `P_rev_follow`, `P_rev_compatible`) reach, on `G_253`:
1. AUROC >= 0.7355 (the registered MDE,
   `results/accord/mde_recomputed.json::strata.gold_ambiguous_excluded.minimum_detectable_auc`),
   AND
2. a 20,000-draw label-permutation p < 0.05 / 3 (Bonferroni over the 3 new
   paraphrases; permutation test: shuffle the `G_253` labels, recompute AUROC on
   the fixed scores, p = (1 + count of permuted AUROC >= observed AUROC) /
   (1 + 20000), one-sided, matching `audit_reversed.py`'s convention).

Both conditions must hold for the SAME paraphrase for it to count toward the
"at least 2 of 3" threshold.

**Outcome B:** otherwise (fewer than 2 of the 3 new paraphrases clear both
conditions).

Reported as: for Outcome A, the C103 reversed-prompt signal is a wording-robust
property of the score, not an artifact of one specific phrasing -- SUBJECT to
the registered confound test below, which can downgrade an Outcome A to
"wording-robust but confounded with rule type" without changing the letter.
For Outcome B, the signal is not shown to survive rephrasing at this
evaluation's resolution; reported with the per-paraphrase AUROC and permutation
p, and nothing is concluded about a smaller effect.

**Outcome C, invalid:** any of, for any one of the 8 cells (2 models x 4
prompts, 299 pairs each):
- the determinism check fails: `P_reversed_rerun`'s hard labels do not match
  C103's committed `P_reversed` cache exactly (same pair_id, same flag_label,
  for every one of the 299 pairs with a scored label in both runs), or its
  flag log-odds differ from C103's by more than 0.05 on any pair;
- parse-failure rate above 5%;
- degenerate score rate (either side of the log-odds floored at -20) above 5%;
- any call flagged as prompt-truncated (`prompt_eval_count >= num_ctx` at
  `num_ctx = 4096`, the same heuristic `run_c103.py` uses);
- a model process killed twice during the run (a cell started for the 3rd time;
  `run_c109.py` tracks per-cell attempt counts in
  `results/accord/prompt_sensitivity/c109/attempts.json` and refuses a 3rd
  start rather than resume or restart silently).

A cell failing any of these is named invalid. If any llama3:8b cell among the 4
is invalid, the run is Outcome C, invalid, not A or B; the primary decision rule
is stated only over llama3:8b, so a granite3.3:8b-only invalidity does not by
itself force Outcome C for the primary decision (it is still reported and still
disqualifies granite3.3:8b's cells from the Secondary section).

## Registered confound test (reported whatever the primary outcome; interpretation fixed now)

Among NEGATIVES only (from `G_253`, i.e. `label_clean == "not_contradiction"`,
ambiguous dropped), the AUROC of each cell's flag log-odds score for predicting
Rule 1-2 membership versus Rule 3-4 membership (positive class = Rule 1 or 2),
rule labels joined on `pair_id` from
`~/mimic_local_only/derived/gold_standard_accord_results.csv`'s `rule` column.
Reported per model x prompt cell (8 cells), with a 20,000-draw label-permutation
p (same convention as the primary decision rule's permutation test, applied to
the rule-membership label instead of the contradiction label).

**Fixed now:** if this AUROC is >= 0.65 with permutation p < 0.05 for a given
cell, that cell's score is registered as carrying a rule-type cue. If this holds
for at least 2 of the 3 new llama3:8b paraphrases that satisfy the primary
decision rule's per-paraphrase criteria, an Outcome A for llama3:8b must be
reported as "wording-robust but confounded with rule type", not as
contradiction detection, regardless of how large or significant the primary
AUROC is. This sentence is the interpretation; it is not re-derived or
re-negotiated after the confound-test number is in hand.

## Registered stratum report

For every one of the 8 cells (2 models x 4 prompts), AUROC with a 20,000-draw
label-permutation p (same convention), computed separately on:
- Rules 1-2, ambiguous dropped (`G_253` restricted to `rule in {1, 2}`), and
- Rules 3-4, ambiguous dropped (`G_253` restricted to `rule in {3, 4}`).

**Fixed now:** the Rules 3-4 stratum has 6 positives (matching C103's audit,
`results/accord/prompt_sensitivity/audit_reversed.json::defect_free.rules34
_ambiguous_dropped` / the registered `defect_free_subset` MDE of 0.8058,
`results/accord/mde_recomputed.json::strata.defect_free_subset
.minimum_detectable_auc`). A Rules 3-4 AUROC below 0.8058 in this run CANNOT be
read as confirming a null effect at this sample size, and a Rules 3-4 AUROC at
or above chance but below 0.8058 CANNOT be read as confirming an effect either;
only a value at or above 0.8058 (with permutation p < 0.05) would clear this
run's own resolution on that stratum. This ceiling on what the stratum can
show is stated now, before the datum, so it cannot be adjusted after the number
is seen.

## Secondary, descriptive

- `granite3.3:8b` under the same 4 prompts, reported the same way as llama3:8b
  (positive rate, AUROC + permutation p on `G_253`, the two rule strata), never
  folded into the primary decision rule, which is stated only over llama3:8b.
- For each of the 4 llama3:8b cells only: leave-one-positive-out AUROC range
  (drop each `G_253` positive in turn, recompute AUROC on the remaining 252
  pairs) and a stratified bootstrap 95 percent interval (5,000 resamples, seed
  42, resampling positives and negatives separately with replacement), the same
  method `audit_reversed.py` used for the C103 `P_reversed` cell.

## Kill condition

- If a cell (one model x one prompt, 299 calls) is interrupted partway, it is
  restarted from the first pair. Never spliced from a partial run plus a
  resumed tail. `run_c109.py` tracks per-cell attempt counts in
  `results/accord/prompt_sensitivity/c109/attempts.json`; a cell started a 3rd
  time (killed twice already) is refused rather than started again, and named
  invalid (see Outcome C above).
- Nothing about the decision rule, the paraphrase wording, the confound-test
  threshold (0.65 / p<0.05), or the stratum-report ceiling (0.8058) is changed
  after any datum.
- No further paraphrases are added after results are in hand. The 3 new
  paraphrases plus the P_reversed determinism check are the complete prompt set
  for this pre-registration; a 4th paraphrase, or a swap of one of these three,
  is a new pre-registration (C11x), never an edit to this one.

## Compliance notes specific to this run

- Every process here that opens a file under `~/mimic_local_only` writes its
  full stdout/stderr to `~/mimic_local_only/logs/<run>.log`; nothing from that
  log is read into this session. Only the aggregate JSON under
  `results/accord/prompt_sensitivity/c109/` is read back.
- `lsof -nP -i -a -p <pid>` for the `ollama serve` process and the running
  `run_c109.py` process is captured to
  `results/accord/prompt_sensitivity/c109/egress_evidence_<run>.txt` at the
  start of the run; any connection other than loopback stops the run.
- `OLLAMA_HOST` is asserted to resolve to a loopback hostname before the first
  call, reimplemented locally in `run_c109.py` exactly as `run_c103.py` does.
- No `--diagnose` flag exists on `run_c109.py` or `analyze_c109.py`. No raw
  model reply text is ever stored, printed, or logged; only the exception type
  name is logged on a failed call, and only to the local mimic_local_only log
  file. Any unhandled exception in `analyze_c109.py` is caught at the top level
  and only its type name plus the last stack frame's function/line number is
  written to a small debug file under `results/`, never the exception's own
  message (which can quote a data value back).

## Artifacts

`results/accord/prompt_sensitivity/c109/cells/*.json` (8 files),
`results/accord/prompt_sensitivity/c109/analysis.json`,
`results/accord/prompt_sensitivity/c109/egress_evidence_<run>.txt`,
`results/accord/prompt_sensitivity/c109/attempts.json`. Register prefix
`promptconfirm.*`, script written at
`accord/prompt_sensitivity/register_promptconfirm.py`, not run by this agent
(the coordinator runs register scripts serially, per this task's edit-scope
rules).

## Outcome

Not yet run. This section is appended, and the rule above is left unchanged,
only after `run_c109.py` and `analyze_c109.py` have produced
`results/accord/prompt_sensitivity/c109/analysis.json` against the real 299
gold pairs.
