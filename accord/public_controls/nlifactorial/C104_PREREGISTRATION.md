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

_Recorded below after the run completes; not yet run as of this commit._
