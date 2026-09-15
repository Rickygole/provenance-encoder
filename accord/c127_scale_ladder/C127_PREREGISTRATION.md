# C127 pre-registration: does over-calling attenuate with scale within one model family?

Committed before the first datum of this run. The rule below is never edited after a result lands;
a change would be a new pre-registration, not a fix. See C128_PREREGISTRATION.md, committed
alongside this file and before any Ollama call this experiment makes, for the companion
prospective-feasibility check that reads the same per-pair caches.

## Why

C103 (`accord/prompt_sensitivity/`) showed qwen3:8b, llama3:8b and granite3.3:8b all over-call the
gold contradiction rate by a wide margin under `P_asserting`, the recovered clinical-framing
prompt. That result sits at one size per family. The scale ladder asks the question C103 could not:
holding the family, the prompt and every decoding setting fixed, does over-calling get better as a
single model family gets bigger?

## What is computed

**Prompt and settings (frozen, byte-identical to C103's `P_asserting` cell):** the `P_asserting`
system/user prompt from `accord/prompt_sensitivity/prompts.py`, `think=False`, `temperature=0`,
`num_predict=1`, `NUM_CTX=4096`, one `/api/chat` call per pair, first-token top-8 logprobs scored
exactly as `accord/prompt_sensitivity/run_c103.py::parse_and_score` does. Implemented by
`accord/c127_scale_ladder/run_c127.py`, which imports `PROMPTS["P_asserting"]` and mirrors
`run_c103.py`'s data loading, join gate (the same 299-row `pharmacist_labeled_clean.csv`), caching
layout, parsing and truncation check line for line, rather than re-deriving any of it.

**Models:** qwen3 at 0.6B, 1.7B, 4B, 8B, 14B, run serially (one model loaded at a time), on all 299
gold pairs.

**Reuse rule:** if a per-pair cache for a given size already exists at
`~/mimic_local_only/accord_prompt_sensitivity_cache/<tag>__P_asserting.jsonl` from an earlier run,
it is reused only after checking that the earlier run's recorded settings (model label, `think`,
`num_ctx`/`num_predict`/`temperature` where recorded, prompt identity, and that per-pair log-odds
rather than only a hard label were captured) are byte-identical to the settings above; otherwise
the cell is rerun from scratch. This check is performed and its outcome logged in `run_c127.py`'s
console output and `c127.json`, not assumed.

**Per size, on G_299** (299 pairs; the 46 ambiguous pairs count negative for every hard-label
computation, exactly as C103's cells already do implicitly by scoring `flag_label` against the
12/299 base rate):
- hard-label positive rate (`n_flagged / 299`) and over-calling ratio (`rate / (12/299)`);
- observed confusion counts (tp/tn/fp/fn) and MCC against the pharmacist hard label (ambiguous =
  negative), using `accord.stats.marginal_feasibility_gate.mcc`;
- the marginal ceiling for the same marginals (`n=299`, `n_pos=12`, `n_flag`) via
  `marginal_feasibility_gate.best_table` and `.mcc`, and the attainable floor p via
  `.attainable_floor_p` -- the best table and smallest one-sided hypergeometric p this labeller's
  marginal could have produced, reported beside (not instead of) the observed table;
- parse failures, degenerate log-odds, `thinking_nonempty` count, and truncation count (a call is
  truncated if `prompt_eval_count >= NUM_CTX`, matching `run_c103.py`).

**Per size, on G_253** (the 253 non-ambiguous pairs, as C103):
- graded AUROC of `flag_logodds` against the pharmacist contradiction label;
- one-sided Mann-Whitney U p (positives greater than negatives);
- a label-permutation p: B = 20000 permutations of the G_253 labels holding scores fixed, seed 42,
  one-sided as `p = mean(null_auroc >= observed_auroc)` (the convention of
  `accord/tfidf_permutation/tfidfperm_run.py::perm_null`).

## Pre-registered question and decision rule

**Question:** does over-calling attenuate with scale within the qwen3 family under `P_asserting`?

**Rule:** compute the Spearman rank correlation (`scipy.stats.spearmanr`) between log parameter
count (any monotonic transform of {0.6, 1.7, 4, 8, 14}; Spearman is invariant to it) and the
over-calling ratio, across the five sizes.

- **"Attenuates"** if `rho <= -0.9` AND the 14B over-calling ratio is below 3.5 (the public-text
  over-calling ceiling this thesis uses, `results/accord/stats/overcall_decomposition.json
  ::public.max_over_call_ratio`, registered as `overcalldecomp.public.max_over_call_ratio`).
- **"Does not attenuate"** if `rho > -0.9` OR the 14B ratio is at or above 3.5.

**Descriptive secondary** (reported per size, not part of the primary rule): whether the graded
AUROC on G_253 meets the registered decision rule from C103/the MDE recomputation --
`AUROC >= 0.7355` (`mde_recomputed.gold_ambiguous_excluded.minimum_detectable_auc`) AND one-sided
Mann-Whitney `p < 0.05`.

## Invalidity (Outcome C)

The run is invalid, and no attenuation verdict is reported, if **any** of the five sizes has:
- a parse-failure rate above 5 percent (`n_parse_failures / 299 > 0.05`), or
- any truncated call (`n_truncated > 0`).

If Outcome C fires, `c127.json` records which size(s) failed the gate and why, and the Spearman
rule above is not evaluated.

## Compliance

Local only; every call goes to `127.0.0.1:11434`. No note text, sentence text, pair_id, HADM_ID or
SUBJECT_ID is ever printed to stdout/stderr or written outside the local per-pair cache (as numeric
flags only, exactly as `run_c103.py`'s cache schema). `run_c127.py` redirects everything it prints
while the gold CSV is in scope to a log file under `~/mimic_local_only/logs/`; only a short
numeric-only summary line per (size, cell) reaches real stdout, matching `run_c103.py`. Egress
evidence (an `lsof` snapshot of the Ollama server and this process, confirming loopback-only
listening sockets and no outbound connection) is written before the first call and checked for a
non-loopback line, exactly as `run_c103.py::write_egress_evidence` does.

## Outcome

(appended after the run; the rule above is not edited)

**Outcome C** (run analyzed 2026-09-15; artifact `results/accord/c127_scale_ladder/c127.json`). gate failed for size(s) ['4b']: parse-failure rate above 0.05 or a truncated call; the Spearman rule is not evaluated. Per-size over-calling ratios: 0.6b=24.833, 1.7b=23.000, 4b=0.000, 8b=14.167, 14b=13.417. Per-size graded AUROC (G_253): 0.6b=0.4879, 1.7b=0.5719, 4b=0.5000, 8b=0.5776, 14b=0.6563.

