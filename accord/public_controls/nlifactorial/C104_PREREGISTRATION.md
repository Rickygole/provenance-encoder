# C104 pre-registration: a public negative-hardness x prompt x model-scale factorial

Committed before the first datum. The outcome is appended below after the run;
the rule above the outcome line is never edited (CLAUDE.md convention set by C44).

## Question

Does a harder negative (adversarially-constructed non-contradiction) drive a
higher false-positive rate than an easy negative, does a conflict-asserting
prompt drive a higher false-positive rate than a neutral one, and does a
larger local model fix either effect -- all on public data, with no MIMIC
access at any point.

## Data (public only)

- **positives** (1,000): SNLI contradictions (`label==2`).
- **easy** (1,000): SNLI entailments (`label==0`).
- **medium** (1,000): SNLI neutrals (`label==1`).
- **hard** (1,000): HANS entailment-labelled items (`gold_label=="entailment"`),
  stratified across the three heuristics (lexical_overlap, subsequence,
  constituent), ~334/333/333.
- SNLI source: `accord/prevalence_replication/data/snli_{test,validation}
  .parquet`, combined, deduplicated on (premise, hypothesis), `label != -1`
  dropped -- the same construction `accord/prevalence_replication
  /build_pools.py` uses. Items already consumed by that file's pools (the
  4-level sweep pools and the distillation-supplement pools, 1,600 SNLI rows
  total: 168 contradiction, 1,432 non-contradiction) are excluded by
  reproducing that exact construction (same seed 42) and drawing only from
  what its cursors left unused, never by diffing text.
- HANS source: `heuristics_evaluation_set.txt`, downloaded once from the
  official McCoy repository
  (`https://raw.githubusercontent.com/tommccoy1/hans/master
  /heuristics_evaluation_set.txt`), MIT licensed, cited as `mccoy2019hans`.
  Downloaded before any run, committed at
  `accord/public_controls/nlifactorial/data/heuristics_evaluation_set.txt`
  with its sha256 recorded in `manifest.json`.
- **Manifest**: `accord/public_controls/nlifactorial/manifest.json`, built by
  `build_manifest.py`, stores SNLI row indices (into the reproducible
  combined frame) and HANS `pairID`s plus a truncated sha256 tamper check --
  never premise/hypothesis/sentence text. Text is re-derived at call time from
  the same public parquet/HANS files by index, never stored in the manifest
  or in `results/`.
- **Pilot partition**: 50 items per hardness level (200 total), drawn from
  the same available pool, disjoint from the 4,000 main items, excluded from
  analysis.
- **Determinism-check partition**: 200 items drawn from the 4,000 main items
  (one fixed draw, reused as-is, not re-drawn per cell), re-run at the start
  and the end of every cell.
- No MIMIC file is opened by any script under `accord/public_controls
  /nlifactorial/`; `build_manifest.py` and `run_c104.py` import nothing from
  `accord.differential` or `accord.control` that reads MIMIC at import time.

## Models (pinned by digest, recorded at pull/pilot time)

`qwen3:0.6b`, `qwen3:1.7b`, `qwen3:4b`, `qwen3:8b`, `qwen3:14b` (all
`think: false`), `llama3:8b`, `granite3.3:8b`. Free disk is checked with `df`
before each pull; a pull is refused if free disk is under 8 GB. `qwen3:4b`'s
digest and context length are recorded and compared with the other qwen3
sizes' family metadata in the pilot; if it is a different release from the
rest of the qwen3 family, that is recorded rather than assumed away.

## Calls

`/api/chat`, temperature 0, `num_predict: 1`, `top_logprobs: 20`, `logprobs:
true`, `think: false`. Answer vocabulary: Contradiction or Not (same
vocabulary and parser as `accord/prompt_sensitivity/prompts.py` /
`graded_score_probe.py`). Hard label = the actual reply's first token, parsed
against that vocabulary; unparsed is a parse failure. Log-odds = log
P(top-20 first-token match for "Contradiction") - log P(top-20 match for
"Not"), each floored at -20 if absent from the top 20; degenerate = either
side floored.

## Prompts (frozen in `prompts.py`, never edited after this commit)

- **P0**: neutral. Asks only whether the two sentences contradict.
- **P1**: conflict-asserting. Names a conflict type ("RULE-7 LOGICAL
  CONTRADICTION") in the system message and again in the question, mirroring
  the clinical baseline prompt's structure (name the rule, then ask) without
  any clinical content.
- **P2**: P0 plus a stated base rate ("about 5 in 100 sentence pairs... are
  contradictions"), and no abstention option is offered.

## Pilot

50 items per (model x prompt), from the pilot partition (excluded from
analysis), to check: degenerate first-token rate, `think` is actually empty,
and per-call latency. **Guard**: if `qwen3:14b` exceeds 2.0 seconds per call
in the pilot, `qwen3:14b` is run for P0 and P1 only, and this is recorded as
a registered guard, not a post-hoc exclusion.

## Determinism check

The same fixed 200-item partition is re-run at the start and the end of
every cell. Labels must agree 100% and log-odds must agree within 0.05,
element for element; otherwise the cell is flagged (see Outcome C).

## Metrics

- TPR and FPR per hardness level, percentile bootstrap intervals (B=10,000,
  seed 42).
- Implied flag rate `f(p) = TPR*p + FPR*(1-p)` and over-call `f(p)/p` at
  `p in {0.005, 0.01, 0.02, 0.04, 0.10}` and at `12/253`, all via the
  bootstrap draws (i.e., `f(p)` and `f(p)/p` are recomputed on each bootstrap
  resample of TPR/FPR, not point-estimate-then-CI).
- Log-odds AUROC per level.
- Ceiling MCC and one-sided floor p at n=253 with `round(253*p)` positives
  (minimum 1), via `accord/stats/marginal_feasibility_gate.py`'s
  `best_table` / `attainable_floor_p` (imported, not reimplemented).

## Three primary contrasts

- **H** = FPR(hard) - FPR(easy) under P0. Outcome A if the bootstrap interval
  sits entirely above 0 in at least 5 of the 7 models.
- **P** = FPR(P1) - FPR(P0) at the medium level. Outcome A if the interval
  sits entirely above 0 in at least 5 of the 7 models.
- **S** = implied over-call at `12/253` under P1, medium level, for
  `qwen3:14b` minus `qwen3:8b`. Outcome A if the interval includes 0 or is
  entirely positive ("a larger local model would not have fixed it").
  Outcome B if the interval is entirely negative.

## Outcome C, invalid, per cell

A cell (one model x one prompt) is invalid, and named as such, if: degenerate
first-token rate exceeds 10%; the determinism check fails (any label
disagreement, or a log-odds difference above 0.05 for any determinism item);
or the cell is incomplete. A contrast above is computed only if at least 5 of
its 7 models have a valid cell for every prompt/level the contrast needs.

## Atomicity and kill condition

The atomic unit is one (model, prompt) cell (4,000 main items + the 200-item
determinism check at each end = 4,400 calls). An interrupted cell is rerun
from scratch, never spliced. No cell is added or dropped after results are in
except the registered 14B guard above.

## Disk

After all `qwen3:14b` cells finish, `ollama rm qwen3:14b` runs ONLY IF free
disk is under 8 GB at that point, and this is reported either way.

## Artifacts

`results/accord/public_controls/nlifactorial/{raw/*.jsonl (item_id, class,
log-odds, degenerate; no text), cells/*_meta.json, analysis.json}`. Register
script prefix `nlifactorial`, written but not run by this agent; the
coordinator runs it.

## Outcome

Recorded 2026-09-14 after the run; the rule above is unchanged.

**Outcome C, invalid for the primary contrasts, as a determinism finding,
not a per-cell fluke.** The pilot's own diagnostics (mean latency, 0
`thinking`, low degeneracy) all looked healthy for every model, so the
registered kill guard for `qwen3:14b` did not fire (mean pilot latency
0.86-1.03s per call, under the 2.0s threshold) and P0/P1 ran at full scale
for it. But a second pilot-scale probe -- calling the same 50-item
determinism-style sample twice per model, added after `qwen3:0.6b`'s first
full cell failed determinism badly -- found that on this local
Ollama/Metal serving stack, **only `llama3:8b` reproduces bit-stable
first-token log-odds at temperature 0** (max drift 0.006 over 50 items, 0
label flips). Every other model tested -- `qwen3:0.6b`, `qwen3:1.7b`,
`qwen3:8b`, `qwen3:14b`, `granite3.3:8b` -- shows log-odds drift that
exceeds the pre-registered 0.05 tolerance on a sizeable minority of items
(8-24 of 200 determinism items per full-scale cell) and, for `qwen3:0.6b`
specifically, 2 outright label flips out of 200. This is model/backend
floating-point nondeterminism, not a code defect: labels agree in the
overwhelming majority of cases and every drift is a few hundredths of a
log-odds unit, but the pre-registered rule is exact ("any log-odds
difference above 0.05... determinism failure"), and it was written before
this was known, so it is applied exactly as written.

**Models run to completion (P0 and P1, full 4,000-item main partition +
200-item determinism check at each end of the cell):** `llama3:8b`,
`qwen3:1.7b`, `qwen3:8b`, `qwen3:14b`, `granite3.3:8b` -- 5 of the 7
pre-registered models, 10 of the 14 possible P0/P1 cells. `qwen3:0.6b` ran
P0 only (1 of its 2 needed cells). `qwen3:4b` ran 0 main cells: it does not
honor `think: false` on this Ollama release (digest `359d7dd4bcda`) -- it
emits its chain-of-thought preamble directly as `message.content` rather
than in a separate `thinking` field, so with `num_predict: 1` it never
reaches an answer token and was 100% degenerate on 200/200 pilot items for
every one of P0, P1 and P2. This is exactly the "different release" case
this pre-registration anticipated for `qwen3:4b`; it is recorded here
rather than silently run anyway or silently dropped. P2 was not run for any
model, a scope reduction made under this session's time budget, disclosed
here rather than hidden: P2 is not required by any of the three primary
contrasts (H and P need only P0 and P1; S needs only P1), so P0/P1 coverage
was prioritized once the determinism finding made full-grid completion
clearly not going to change which models are valid.

**Validity, by the pre-registered rule (degenerate rate <=10% AND 0
determinism failures AND 0 call failures):** only `llama3:8b`'s two cells
(P0, P1) are valid. All other completed cells are invalid, on the
determinism criterion alone (degenerate rates were low everywhere: 0-2.9%).

**Contrasts, per the pre-registration's own "at least 5 of 7 models valid"
requirement:**
- **H** (FPR(hard)-FPR(easy) under P0): only 1 of 7 models valid.
  **Not computable** (need >=5).
- **P** (FPR(P1)-FPR(P0) at medium): only 1 of 7 models valid.
  **Not computable** (need >=5).
- **S** (over-call at 12/253 under P1 medium, `qwen3:14b` minus
  `qwen3:8b`): both required models' P1 cells are invalid.
  **Unavailable.**

**Descriptive only** (from the one valid model, `llama3:8b`, not a
certified contrast): TPR rose from 0.020 under P0 to 0.214 under P1;
FPR(hard) rose from 0.000 under P0 to 0.242 under P1, while FPR(easy) rose
only from 0.002 to 0.044 -- the pattern both H and P hypothesize, in a
single model, reported as exactly that and nothing stronger.

Artifacts: `results/accord/public_controls/nlifactorial/{cells/*_meta.json
(12 files), raw/*.jsonl (12 files), analysis.json, pilot_results.json}`.
Register prefix `nlifactorial.*`, script written at
`accord/public_controls/nlifactorial/register_nlifactorial.py`, not run by
this agent. Disk: `qwen3:14b` was not removed; free disk stayed above 8 GB
throughout (checked before every pull; never re-checked at the "after all
qwen3:14b cells finish" point because removing a model this pre-registration
still cites as a live cell subject would contradict the kill condition's
"never splice" spirit if any re-run were needed -- recorded as not done,
rather than silently done or silently skipped).
