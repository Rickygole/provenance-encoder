# C103 pre-registration: does naming a conflict before asking drive the over-calling?

Committed before the first datum. The outcome is appended below after the run; the
rule above the outcome line is never edited (CLAUDE.md convention set by C44).

## Question

Is the clinical over-calling this project has measured produced by the prompt
naming a conflict type / rule id before asking, rather than by asking about a
medication conflict at all?

## What runs

- Models: `llama3:8b`, `granite3.3:8b`, `qwen3:8b`, all served locally by Ollama
  on `127.0.0.1:11434`, temperature 0, `think` recorded explicitly as `False`
  for every call (qwen3 is a thinking-capable model; llama3 and granite3.3 are
  not, but `think: false` is sent to all three so the payload is identical in
  shape across models). Filename tags via
  `accord/differential/model_tags.py::ollama_model_tag(label, False)`:
  `llama3:8b` -> `_llama3_8b_nothink`, `granite3.3:8b` -> `_granite33_8b_nothink`,
  `qwen3:8b` -> `_qwen3_8b_nothink`. These are NEW-scheme tags (think recorded
  explicitly), never the three frozen `LEGACY_TAGS` strings, because those are
  reserved for the pre-C94 runs made before this project tracked think at all.
- Pairs: the 299 annotated gold pairs (`G_299`) from
  `~/mimic_local_only/from_drive_2026-08-08/outputs/pharmacist_labeled_clean.csv`,
  `label_clean` column; `nursing_sentence` and `discharge_sentence` are the two
  fields substituted into each prompt. `G_253` (the 253 pairs with
  `label_clean != "ambiguous"`, 12 positive) is the secondary AUROC basis,
  matching C98's basis exactly.
- Prompts: frozen, committed verbatim in `accord/prompt_sensitivity/prompts.py`,
  never edited after this file is committed:
  - **P_asserting**: the recovered clinical baseline prompt. Copied
    byte-for-byte from `prompts_existing.csv`'s `config_id == "gpt4o_zeroshot"`
    row (the same row `accord/differential/run_frontier_compliant.py`'s
    `load_prompts()` loads), by a script
    (`accord/prompt_sensitivity/extract_p_asserting.py`) that read the row and
    wrote it into `prompts.py` without ever printing it to a console or a log.
    This is the prompt that names a conflict type / rule before asking.
  - **P_neutral**: asks only whether the two sentences give conflicting
    medication instructions. Names no conflict type and no rule id.
  - **P_reversed**: asks whether the two sentences are consistent. A "no" (not
    consistent) is the flag. Scored so higher log-odds and label 1 always mean
    "more conflict", by flipping the raw parse and the raw log-odds sign at the
    point of use (`run_c103.py`), never by rewriting the vocabulary sets
    themselves.
  - All three go over `/api/chat` as one system message plus one user message,
    `stream: false`, `logprobs: true`, `top_logprobs: 8`, `num_predict: 1`,
    exactly as `accord/differential/graded_score_probe.py` calls `/api/chat`.
- Recorded for every call: the parsed hard label (first word of the actual
  reply, matched against the prompt's answer vocabulary, case-insensitive,
  punctuation-stripped; unmatched is a parse failure, recorded as such and
  excluded from the positive-rate numerator and denominator alike is NOT done
  here -- a parse failure is recorded as its own outcome and contributes to
  the parse-failure rate, never silently dropped) and the first-token log-odds
  (log P(first token matches the positive vocabulary, best top-8 match) minus
  log P(matches the negative vocabulary, best top-8 match), each floored at
  -20 if absent from the top 8; degenerate = either side floored). For
  P_asserting this is Contradiction vs Not; for P_neutral and P_reversed it is
  Yes vs No, mapped through the `reversed` flag in `prompts.py` so that, after
  mapping, higher log-odds and label 1 always mean "flagged as a conflict" for
  all three prompts alike.
- Per-pair numeric records (pair_id, flag label, flag log-odds, degenerate
  flag, parse-failure flag, thinking-non-empty flag, prompt-truncation flag)
  are cached locally under
  `~/mimic_local_only/accord_prompt_sensitivity_cache/*.jsonl` -- numbers and
  the project-internal pair_id only, never sentence text -- so the
  cross-prompt McNemar pairing and the G_253 AUROC can be computed without
  holding the full run in one process's memory. Nothing from that cache is
  printed or copied into `results/`; only the aggregates below are.

## Primary measure

Positive (flagged) rate on `G_299`, per model x prompt.

## Decision rule

**Outcome A, prompt drives the over-calling:** for at least 2 of the 3 models,
BOTH of:
1. P_neutral's positive rate is below P_asserting's positive rate on `G_299`,
   by a margin whose exact two-sided McNemar p (sign test on the discordant
   pairs between P_asserting and P_neutral, `scipy.stats.binomtest` on the
   smaller discordant count against 0.5, two-sided) is below 0.05, AND
2. the neutral over-calling ratio, P_neutral's positive rate divided by the
   gold base rate 12/299 (0.04013...), is at or below the public-text maximum
   over-call ratio already registered at
   `results/accord/stats/overcall_decomposition.json::max_over_call_ratio`
   (3.5000000000000004, cell `llama3:8b::A_bare::p0.02`).

Reported as: naming a conflict type before asking is a measurable driver of
the over-calling, and removing it brings the model's neutral-prompt
over-calling back down to a rate no worse than the public-text ceiling this
project has already registered for prompts that never mention a clinical
conflict at all.

**Outcome B, anything else:** naming the conflict is not shown to be a
sufficient explanation on its own; reported as such, with the per-model
numbers.

**Outcome C, invalid:** any of, for any one cell (one model x one prompt):
parse-failure rate above 5%; any call with `thinking` non-empty (should be
impossible with `think: false`, but checked, not assumed); any call flagged
as prompt-truncated (see `run_c103.py`'s truncation heuristic: `prompt_eval
_count >= num_ctx` at a committed `num_ctx`); a model process killed twice
during the run. A cell failing any of these is named as invalid; the
2-of-3-models rule above is then read only over the remaining valid models,
and if fewer than 2 models have all three of their cells valid, the overall
run is Outcome C, invalid, not A or B.

## Secondary

- AUROC of the (mapped) log-odds on `G_253`, per model x prompt, with a
  one-sided Mann-Whitney p against the registered minimum detectable AUROC
  0.7355 (`results/accord/mde_recomputed.json::strata.gold_ambiguous_excluded
  .minimum_detectable_auc`), read the same way C98 reads it.
- P_reversed reported descriptively (positive rate, AUROC) beside the other
  two, not folded into the primary decision rule, which is stated only over
  P_asserting and P_neutral.
- The marginal feasibility floor for each cell's hard label, via
  `accord/stats/marginal_feasibility_gate.py`'s `best_table` /
  `attainable_floor_p` (imported, not reimplemented), on `G_299`.

## Kill condition

If a cell (one model x one prompt, 299 calls) is interrupted partway, it is
restarted from the first pair. Never spliced from a partial run plus a
resumed tail.

## Compliance notes specific to this run

- Every process here that opens a file under `~/mimic_local_only` writes its
  full stdout/stderr to `~/mimic_local_only/logs/<run>.log`; nothing from that
  log is read into this session. Only the aggregate JSON under
  `results/accord/prompt_sensitivity/` is read back.
- `lsof -nP -i -a -p <pid>` for the `ollama serve` process and the running
  `run_c103.py` process is captured to
  `results/accord/prompt_sensitivity/egress_evidence_<run>.txt` at the start
  of the run; any connection other than loopback stops the run.
- `OLLAMA_HOST` is asserted to resolve to a loopback hostname before the first
  call, the same check `run_frontier_compliant.py`'s `assert_loopback_host`
  performs (reimplemented locally in `run_c103.py`, not imported, since that
  function is a free function with no MIMIC-reading side effect but this file
  avoids importing that module's argparse-heavy `main` at all).

## Outcome

Recorded 2026-09-13 after the run; the rule above is unchanged.

**Outcome B, naming the conflict is not shown to be a sufficient explanation
on its own.** All 9 cells (3 models x 3 prompts, 299 pairs each) are valid:
0 parse failures, 0 call failures, 0 thinking leakage, 0 truncation in every
cell (degenerate log-odds rate at most 2.0%, from `llama3:8b/P_asserting`,
well under any threshold this pre-registration set).

Per-model McNemar (P_asserting vs P_neutral, exact two-sided):
- `llama3:8b`: positive rate rose from 0.8729 (asserting) to 0.9365
  (neutral), the WRONG direction for Outcome A (p=0.0054, but margin
  direction fails, so this model does not satisfy the criteria regardless
  of the ratio).
- `granite3.3:8b`: positive rate fell from 0.7492 to 0.3579 (p=5.8e-32,
  right direction), but the neutral over-calling ratio is 8.92, above the
  registered public-text ceiling of 3.5000000000000004.
- `qwen3:8b`: positive rate fell from 0.5686 to 0.1940 (p=8.1e-29, right
  direction), but the neutral over-calling ratio is 4.83, still above the
  3.5 ceiling.

0 of 3 models satisfy the full Outcome A criteria (margin direction AND
McNemar p<0.05 AND ratio at or below the public-text ceiling); the
pre-registered threshold is at least 2 of 3. Removing the conflict-naming
does significantly reduce over-calling for 2 of the 3 models (large McNemar
effects, right direction), but even the reduced, neutral-prompt over-calling
remains well above the ceiling this project has already registered for
prompts that never mention a clinical conflict at all -- so naming the
conflict is a real contributor for at least those two models, but it is not
shown to be the whole story: something else is still driving over-calling
once the conflict-naming is removed. For `llama3:8b` specifically, removing
the conflict-naming did not reduce over-calling at all; it INCREASED it, a
finding at least as important as the intended contrast.

Secondary (G_253, one-sided Mann-Whitney vs MDE 0.7355): only one of the
nine cells clears the registered minimum detectable AUROC --
`llama3:8b/P_reversed`, AUROC 0.7503, p=0.0017. Every other cell's AUROC is
below the MDE (range 0.578-0.737), so no other prompt/model cell is read as
discriminating at this evaluation's resolution; nothing is concluded about a
smaller effect for those cells. `P_reversed` is descriptive only, per the
pre-registration, and is not part of the primary decision rule.

Marginal feasibility floors (`accord/stats/marginal_feasibility_gate.py`,
G_299 hard label) range from 2.9e-6 (`granite3.3:8b/P_neutral`) to 0.448
(`llama3:8b/P_neutral`); every cell's floor is reported in
`analysis.json::per_model_prompt.*.marginal_feasibility_floor_p`.

Artifacts: `results/accord/prompt_sensitivity/cells/*.json` (9 files),
`results/accord/prompt_sensitivity/analysis.json`,
`results/accord/prompt_sensitivity/egress_evidence_c103_20260913T192835.txt`
(loopback only, confirmed). Register prefix `promptsens.*`, script written at
`accord/prompt_sensitivity/register_promptsens.py`, not run by this agent.
