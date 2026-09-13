# C99 pre-registration: a within-family scale step, qwen3 8B to 14B, local

Committed before the first datum. The outcome is appended below after the run; the
rule above the outcome line is never edited.

## Why

`capability_ladder.py` reports an over-calling spread across the three installed 8B
models, `capability.spread_8b_frontier_zero_shot_g299`, of a factor this project has
not yet related to a within-family step at higher capability. `qwen3:14b` is the
largest model this machine's 16GB of unified memory can serve (see C94's note on the
measured hardware ceiling); it is not a frontier-scale model and this run does not
claim it is one.

## What runs

- Models: `qwen3:8b` and `qwen3:14b`, both Q4_K_M, served locally by Ollama. Nothing
  leaves the machine.
  - `qwen3:8b` digest: `500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41`
  - `qwen3:14b` digest: `bdbd181c33f2ed1b31c972991882db3cf4d192569092138a7d29e973cd9debe8`
- Templates: the frontier zero-shot and few-shot templates from the committed prompt
  table (`config_id=gpt4o_zeroshot`, `config_id=gpt4o_fewshot`), the same templates
  every other row in the capability ladder is read against.
- Settings: temperature 0, max_tokens 50, `think` false (recorded explicitly, not
  left unset), `num_ctx` 4096. All 300 annotated pairs sent; scored against the 299
  gold-labelled pairs (G_299), matching the thesis's basis.
- The prior `qwen3:8b` zero-shot artifact (think effectively on via a 1024-token
  override, 67 of 300 draws unparseable) is superseded under C94's own rule ("A run
  whose `failed_parses` is not zero is discarded and re-run, not reported") and is
  not scored here. This run is a fresh `qwen3:8b` pass with `think` recorded
  explicitly false, tagged `_qwen3_8b_nothink`, distinct from the frozen legacy
  `_qwen3` artifact, which stays on disk unscored and unreferenced by this run.

## The statistic and the ceiling

- **S** = the zero-shot over-calling ratio of `qwen3:8b` divided by that of
  `qwen3:14b`, both on G_299 under the frontier zero-shot template
  (`capability.within_family_step_qwen3_8b_to_14b`, written by
  `accord/differential/capability_ladder.py`).
- **D** = `capability.spread_8b_frontier_zero_shot_g299`, the max/min over-calling
  ratio across the three local 8B models under the frontier zero-shot template on
  G_299. Written by `capability_ladder.py` in the commit that lands `qwen3:8b`,
  BEFORE any `qwen3:14b` datum exists. That commit hash is recorded in the outcome
  below and D is read from the register, never retyped or recomputed a second way.

## Decision rule

- **Outcome A, a scale step separable from between-model noise:** `(S >= D or
  S <= 1/D)` AND an exact two-sided McNemar test on the discordant G_299 pairs
  between `qwen3:8b` and `qwen3:14b` zero-shot predictions has `p < 0.05`. Reported
  with the direction (8B over-calls more, or 14B over-calls more; both directions
  named here in advance, before either digest was pulled). `qwen3:14b` is explicitly
  NOT frontier scale and this outcome is not read as evidence about frontier-scale
  capability.
- **Outcome B, not separable:** anything else. Reported as: the within-family step at
  this scale is not distinguishable from the spread the three 8B models already show
  at fixed capability; nothing is concluded about a scale step in either direction.
- **Outcome C, invalid:** `failed_parses > 0` for either model, `n_thinking_nonempty
  > 0` for either model, `n_prompt_truncated > 0` for either model, `n_pairs != 300`
  for either run, a digest mismatch against the values recorded above, or a run
  killed twice by the memory manager. Reported as not run; no S or p is quoted.

## Kill condition

If a run is interrupted (including an out-of-memory kill), it is restarted from the
first pair, never spliced. A second kill of the same run is Outcome C.

## Outcome

Recorded after the run; the rule above is unchanged.

PLACEHOLDER_OUTCOME_FILLED_AFTER_RUN
