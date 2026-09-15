# C131 pre-registration: scale ladder v2, a post hoc repair of C127/C128's qwen3:4b decoding defect

Committed before the first C131 datum. The rule below is never edited after a result lands; a
change would be a new pre-registration, not a fix. See C127_PREREGISTRATION.md and
C128_PREREGISTRATION.md for the runs this repairs, and their Outcome sections for what went wrong.

**This is a post hoc repair.** C127 (`accord/c127_scale_ladder/`) was called Outcome C: under
`P_asserting`, `think=False`, `temperature=0`, `num_predict=1`, qwen3:4b emitted a chain-of-thought
preamble token ("Okay", "First") as its first content token on all 299 pairs, a 100 percent
parse-failure rate, so no attenuation verdict could be read. C128 (`accord/c128_prospective_feasibility/`)
was Outcome C for the same reason: it depends on every C127 cell being valid. Both defects were
observed BEFORE this file was written, and the remedy below exists because of that observation, not
before it. Everything that follows -- the decoding remedy, the decision rules, the reuse rule, and
the requirement that the remedy be validated on synthetic prompts before it ever touches a
credentialed pair -- is fixed here, before any C131 datum, exactly as C127/C128's own rules were
fixed before their first datum. Nothing below is negotiable once the first C131 Ollama call against
gold data is made.

## Why

C127's decoding defect is specific and mechanical: `num_predict=1` asks the server for exactly one
generated token, and qwen3:4b's first token under `P_asserting` is reliably a preamble word, not the
answer. That is a property of this model's own decoding under `think=false`, independently
reproduced with a synthetic, non-clinical probe (recorded in C128's own Outcome section). It is not
a defect in `run_c127.py`'s parsing, and it does not, by itself, mean qwen3:4b cannot answer the
question -- only that its answer is not always the very first token it emits. Widening the token
budget and reading further into the reply is a narrow, mechanical fix for a narrow, mechanical
defect; it changes nothing about the prompt, the model, the temperature, or the gold data.

## The remedy

**Unchanged from C127, for all five sizes:** the `P_asserting` system/user prompt text (byte-identical
to `accord/prompt_sensitivity/prompts.py::PROMPTS["P_asserting"]`), `temperature=0`, `think=False`,
`NUM_CTX=4096`, one `/api/chat` call per pair, qwen3 at 0.6B/1.7B/4B/8B/14B, on all 299 gold pairs,
run serially, one model loaded at a time.

**Changed:** `num_predict` is raised from 1 to 8 (`NUM_PREDICT=8`), fixed here before any datum. This
is the only decoding-setting change this file makes.

**Hard label:** `content.split()` on the reply's full content string (which, at `num_predict=8`, can
contain a preamble plus the answer, not just a single truncated token) is scanned in order; the first
word, after stripping trailing `.,:;!?` and lower-casing, that matches an entry of
`PROMPTS["P_asserting"]["pos_tokens"]` or `["neg_tokens"]`, **or the literal whole words
`"contradiction"` / `"not_contradiction"`,** sets the hard label (1 or 0 respectively). The addition
is narrow and was found while designing this remedy, before any datum: `pos_tokens`/`neg_tokens` are
C103's frozen, fragment-based vocabulary (`"contr"`, `"not"`, ...), built around `num_predict=1`'s
content being a single sub-word fragment, never a complete word. At `num_predict=8` the model can
finish the word instead: `"contradiction"` already has a whole-word entry in `ASSERTING_POS`
(`"Contradiction"`), but `P_asserting`'s own other literal answer word, `"not_contradiction"` (joined
without a space, per the system prompt's instruction to answer with exactly that word), has no
whole-word counterpart among `ASSERTING_NEG`'s fragments and would otherwise parse-fail every time
the model answers it in full. Both whole-word forms are added explicitly, by exact string, not by a
prefix rule (so a genuinely different word starting "not", e.g. "nothing", is never misread as the
negative answer). If no word in the full content matches either vocabulary, the pair is a parse
failure -- exactly C127's parse-failure definition, now checked against up to 8 generated tokens' worth of text instead
of 1.

**Graded score -- what the Ollama API actually returns, checked before this method was chosen:**
`run_c103.py::call_chat` (imported by `run_c127.py`) requests `"logprobs": true, "top_logprobs": 8`
and reads `r["logprobs"][0]["top_logprobs"]` -- i.e., Ollama's `/api/chat` response carries a
`logprobs` field that is a LIST, one entry per generated token, and EACH entry carries its own
`top_logprobs` (the top-K alternative tokens Ollama considered at that generation step) as well as
the token actually chosen at that step (`logprobs[i]["token"]`) and that token's own log-probability
(`logprobs[i]["logprob"]`). C103/C127 only ever read index 0 because `num_predict=1` only ever
generates one token. At `num_predict=8` the same field is a list of up to 8 such entries, one per
generated position, each individually carrying the same `top_logprobs` shape C103's
`parse_and_score` already scores position 0 with.

The graded score is therefore read at **the first position `i` (0-indexed) whose actually-generated
token (`logprobs[i]["token"]`, stripped) is itself a member of `pos_tokens ∪ neg_tokens`** -- the
first answer-vocabulary token position, scanned over `logprobs[0..min(7, len(logprobs)-1)]`. At that
position, `p` = the log-probability of whichever `pos_tokens` member appears in that position's
`top_logprobs` (else `None`), `n` = the same for `neg_tokens`, `flag_logodds = (p or -20.0) - (n or
-20.0)`, `degenerate = (p is None or n is None)` -- the identical formula `run_c103.py::parse_and_score`
already applies at position 0, just applied at whichever position the answer actually starts. If no
position in `logprobs[0..7]` has a chosen token in the answer vocabulary, the graded score for that
pair is `None` (not evaluable), independent of whether the hard label parsed (the hard label reads
the decoded text; the graded score reads token identity, and the two can disagree at the margin --
both are recorded per pair so that is visible, not hidden).

Every per-pair record additionally keeps, for the fallback below and for audit: `answer_token_position`
(the index above, or `null`), `n_generated_tokens` (`len(logprobs)`), and `first_token_is_answer`
(whether `logprobs[0]["token"]` itself is in the answer vocabulary -- i.e., whether this pair would
have parsed under C127's original `num_predict=1` method too).

**Fallback, stated before any datum, in case the addendum below finds the API does not expose
per-position `logprobs` on this Ollama build:** if the validation step (below) finds that
`logprobs[i]` for `i>0` is empty, missing `top_logprobs`, or otherwise unusable, `run_c131.py` still
performs the single `num_predict=8` call per pair (needed for the hard label regardless), but the
graded score for a pair is taken ONLY from `logprobs[0]` (the `first_token_is_answer` case -- the
existing C127/C103 num_predict=1-equivalent graded score, since position 0's distribution does not
depend on `num_predict`) and is `None` for every pair where `first_token_is_answer` is false. If, for
a given size, fewer than two label classes remain among the pairs with a non-`None` graded score on
G_253, that size's `auroc_253` is reported as `null` and `meets_registered_auroc_rule` as `false`,
i.e. "not evaluable," exactly as the analysis code's existing empty-class guard already does for any
other degenerate case. This fallback is written now and is available to `analyze_c131.py`
unconditionally; which branch is actually used for the clinical run is decided by, and recorded in,
the validation addendum before that run, not by silent runtime detection.

**Validated only on synthetic, non-clinical probe prompts, before any clinical call.** Before
`run_c131.py` is ever pointed at the gold CSV, `accord/c131_scale_ladder_v2/validate_remedy_synthetic.py`
sends a small, fixed set of synthetic, non-MIMIC probe prompts (using `P_asserting`'s own
system/user template with placeholder text that names no patient, note, or clinical fact) to at
least one qwen3 size at `num_predict=8`, and checks: (a) `r["logprobs"]` has more than one entry when
generation continues past one token; (b) each entry after index 0 carries its own `top_logprobs` in
the same shape as index 0; (c) the hard-label word-scan and the graded first-answer-token-position
method above are computable on that response without raising. Its result -- pass/fail per check, and
whether the API confirms per-position `logprobs` on this Ollama build -- is written to
`results/accord/c131_scale_ladder_v2/synthetic_probe_validation.json` and appended as an addendum to
this file, committed BEFORE `run_c131.py` is pointed at any gold pair. `run_c131.py` refuses to run
against the gold CSV if that validation artifact is missing.

**Reuse rule for the 8B and 14B cells only** (no other size is a reuse candidate): C127's own
`_qwen3_8b_nothink__P_asserting.jsonl` / `_qwen3_14b_nothink__P_asserting.jsonl` caches (num_predict=1)
recorded zero parse failures and zero truncated calls for both sizes across all 299 pairs -- i.e.,
the very first generated token was already an answer-vocabulary token on every pair, for both sizes.
Because decoding is greedy at `temperature=0`, the probability distribution Ollama samples from at
generation position 0 is a function of (model, prompt, `num_ctx`) only -- `num_predict` bounds how
many tokens are generated AFTER position 0, and cannot change what is sampled at position 0 itself.
Therefore, for any size whose existing C127 cell already answered at position 0 on every one of the
299 pairs, the remedy's own "first answer-vocabulary token position" is provably position 0 for
every pair too, and the remedy's hard label and graded log-odds at that position are the same
numbers C127's cell already recorded -- not merely expected to match, but unable to differ, because
nothing about num_predict=8 vs num_predict=1 can alter the distribution sampled at a position both
calls would generate identically. `run_c131.py::check_reuse_provable` checks this precondition
mechanically before trusting it (settings byte-identical to `run_c127.check_reuse`'s check --
prompt, `think`, `n=299`, `NUM_CTX` -- AND `n_parse_failures==0` AND `n_truncated==0` in the existing
cell) and refuses reuse, rerunning that size under the remedy instead, if either fails. If the check
fails for either 8B or 14B, **all five sizes are rerun** under the remedy (not just the failing one),
per this rule; a reused size's per-pair records are read directly from C127's original cache file
(`~/mimic_local_only/accord_prompt_sensitivity_cache/`), never copied or rewritten, so "byte-identical"
is enforced by referencing the same bytes rather than by comparing two copies.

## What is computed

Per size, on G_299 (299 pairs; ambiguous pairs count negative, as C127): hard-label positive rate,
over-calling ratio against `12/299`, observed confusion (tp/tn/fp/fn) and MCC against the pharmacist
label, the marginal ceiling MCC and attainable floor p for the same marginals
(`accord.stats.marginal_feasibility_gate`), parse failures, degenerate log-odds, `thinking_nonempty`,
truncation, and (new to C131) `n_no_answer_token_position` (pairs where no position in
`logprobs[0..7]` held an answer-vocabulary token) and mean `n_generated_tokens`.

Per size, on G_253 (the 253 non-ambiguous pairs, as C127): graded AUROC of `flag_logodds` against the
pharmacist contradiction label, one-sided Mann-Whitney U p, and a label-permutation p (B=20000, seed
42, one-sided, mirroring `accord/tfidf_permutation/tfidfperm_run.py::perm_null`'s convention) --
identical formulas to C127's, applied to whichever graded-score source (primary or fallback) the
validation addendum selects.

**The C128 prospective-feasibility check, applied to C131's own cells, unchanged in every rule and
seed:** the same 100/199 pilot/held-out split of the 299 gold pairs, drawn with
`numpy.random.default_rng(20260915)` exactly as `C128_PREREGISTRATION.md` specifies (never persisted
as a pair_id-to-split mapping, recomputed at analysis time only), the same `pilot_n_pos <= 1` ->
`not_evaluable` rule, the same `pilot_floor_p > 0.05` -> `ruled_out` / else `not_ruled_out` rule, the
same held-out hypergeometric test at the observed table, the same `demonstrated` /
`not_demonstrated` / `not_applicable (no ruled-out predictions)` rule for the check as a whole. This
is not a new check; it is C128's own rule, re-applied to C131's per-pair caches instead of C127's,
because C127's caches are the ones being repaired.

## Pre-registered question and decision rule (unchanged from C127)

**Question:** does over-calling attenuate with scale within the qwen3 family under `P_asserting`?

**Rule:** Spearman rank correlation (`scipy.stats.spearmanr`) between log parameter count and the
over-calling ratio across the five sizes.

- **"Attenuates"** if `rho <= -0.9` AND the 14B over-calling ratio is below 3.5
  (`overcalldecomp.public.max_over_call_ratio`, currently 3.5).
- **"Does not attenuate"** if `rho > -0.9` OR the 14B ratio is at or above 3.5.

**Descriptive secondary** (reported per size, not part of the primary rule): whether the graded
AUROC on G_253 meets `AUROC >= 0.7355` (`mde_recomputed.gold_ambiguous_excluded.minimum_detectable_auc`)
AND one-sided Mann-Whitney `p < 0.05` -- identical thresholds to C127.

## Invalidity (Outcome C), unchanged from C127

The run is invalid, and no attenuation verdict is reported, if **any** of the five sizes has a
parse-failure rate above 5 percent (`n_parse_failures / 299 > 0.05`) or any truncated call
(`n_truncated > 0`) -- computed identically to C127, just against the remedy's wider parse (up to 8
tokens of content) instead of one token. If Outcome C fires, `c131.json` records which size(s)
failed the gate and why, and the Spearman rule is not evaluated. Additionally, per the validation
requirement above, the run is invalid and refuses to start against gold data at all if
`results/accord/c131_scale_ladder_v2/synthetic_probe_validation.json` does not exist or reports that
the hard-label word-scan or the graded-score position scan raised on the synthetic probes.

## Compliance

Local only; every call goes to `127.0.0.1:11434`. No note text, sentence text, pair_id, HADM_ID or
SUBJECT_ID is ever printed to stdout/stderr or written outside the local per-pair cache (as numeric
flags only). `run_c131.py` redirects everything it prints while the gold CSV is in scope to a log
file under `~/mimic_local_only/logs/`; only a short numeric-only summary line per (size, cell)
reaches real stdout, matching `run_c103.py`/`run_c127.py`. Egress evidence (an `lsof` snapshot of the
Ollama server and this process, confirming loopback-only listening sockets and no outbound
connection) is written before the first clinical call and checked for a non-loopback line, exactly as
`run_c103.py::write_egress_evidence` / `run_c127.py::write_egress_evidence` do. The synthetic-probe
validation script sends no MIMIC content and is not subject to the credentialed-logging redirect,
but is still constrained to `127.0.0.1:11434` and its own egress check, since it is still an Ollama
call.

## Validation addendum

(appended here, from `results/accord/c131_scale_ladder_v2/synthetic_probe_validation.json`, before
`run_c131.py` is ever pointed at the gold CSV -- the rule above is not edited)

## Outcome

(appended after the run; the rule above is not edited)
