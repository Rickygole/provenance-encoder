# C128 pre-registration: a prospective feasibility check on a 100-pair pilot, before the C127 scale ladder runs

Committed before any C127 Ollama call and before any datum this experiment reads. The rule below is
never edited after a result lands. This is E2: it checks whether the marginal-feasibility screen
(`accord/stats/marginal_feasibility_gate.py`) could have been applied *prospectively*, on a small
pilot drawn before the full ladder ran, rather than read off the finished 299-pair result the way
every earlier use of that gate in this project has been.

## Why

The feasibility gate is a precondition computable from two marginals: how many pairs the gold
standard confirms and how many the labeller flags. It needs no per-pair correctness to compute, so
it can in principle be applied on a small pilot before the rest of the evaluation set is even
scored, and used to decide in advance whether a full run could show anything. This experiment
tests that claim against real held-out data instead of asserting it.

## Pilot definition (fixed here, before any datum)

A pilot of 100 of the 299 gold pairs, drawn with `numpy.random.default_rng(20260915)`:

```python
rng = numpy.random.default_rng(20260915)
pilot_idx = sorted(rng.choice(299, size=100, replace=False).tolist())
```

applied to the gold CSV's row order (`pharmacist_labeled_clean.csv`, the same 299-row frame
`run_c103.py`/`run_c127.py` load; row 0 is `PAIR_0001`, row 298 is `PAIR_0299`, matching file
order, never re-sorted by any other key). `pilot_idx` selects the pilot rows; the remaining 199
rows are the held-out set. **Pilot membership is never written to `results/` or printed to
stdout**: because it is a deterministic function of a committed seed and the gold file's row order,
it is recomputed by `accord/c128_prospective_feasibility/analyze_c128.py` every time it is needed
and is not otherwise persisted, so no pair_id-to-split mapping is ever stored outside this
derivation. Only aggregate pilot/held-out counts (positives, flags) are written to
`results/accord/c128_prospective_feasibility/c128.json`.

## What is computed

Reads the same per-pair `P_asserting` caches C127 produces (or reuses), one per qwen3 size, under
`~/mimic_local_only/accord_prompt_sensitivity_cache/`. No new Ollama calls: this experiment is an
analysis of C127's per-pair scores, split into the pilot and held-out rows above.

For each of the five qwen3 sizes:

**Pilot rows only (n=100):**
- `pilot_n_pos`: count of pharmacist `contradiction` labels among the 100 pilot pairs;
- `pilot_n_flag`: count of the labeller's `flag_label == 1` among the same 100 pairs (ambiguous
  counts negative, as in C127);
- attainable floor p (`accord.stats.marginal_feasibility_gate.attainable_floor_p(100, pilot_n_pos,
  pilot_n_flag)`) and the ceiling MCC (`mcc(*best_table(100, pilot_n_pos, pilot_n_flag))`), the
  best table these two pilot marginals permit.
- **Prediction:** `"ruled_out"` if `pilot_floor_p > 0.05` (the pilot marginal could not have
  supported rejecting chance even in its best case); `"not_ruled_out"` otherwise.
- **Not-evaluable rule, stated in advance:** if `pilot_n_pos` is 0 or 1, the pilot has too few
  positives for the floor to be informative (a floor computed from 0 or 1 confirmed pairs is
  driven entirely by `pilot_n_flag`, not by any discriminating signal), so this size's prediction
  is reported as `"not_evaluable"` and is excluded from the agreement count below, rather than
  being forced into `"ruled_out"` or `"not_ruled_out"`.

**Held-out rows only (n=199, the 299 minus the 100 pilot pairs):**
- `heldout_n_pos`, `heldout_n_flag`, and the observed true-positive overlap `heldout_tp` (pairs
  flagged by the labeller AND confirmed contradiction, among the 199 held-out pairs);
- the held-out one-sided hypergeometric p at the OBSERVED table (not the best-case floor):
  `scipy.stats.hypergeom.sf(heldout_tp - 1, 199, heldout_n_pos, heldout_n_flag)`.
- **Held-out outcome:** `"confirmed_uninformative"` if this p is `>= 0.05`; `"not_confirmed"`
  otherwise.

**Per-size agreement:** whether the pilot prediction and the held-out outcome agree, defined only
for `"ruled_out"` vs `"not_ruled_out"` predictions:
- if prediction is `"ruled_out"`: agreement is true iff held-out outcome is `"confirmed_uninformative"`;
- if prediction is `"not_ruled_out"`: this experiment makes no held-out claim (the rule can only
  rule out, never confirm informativeness), so agreement is reported as `"not_applicable"`, not as
  a pass or fail;
- if prediction is `"not_evaluable"`: agreement is `"not_evaluable"`.

## Decision rule for the check itself

The prospective check is **"demonstrated"** if every size whose pilot prediction is `"ruled_out"`
is also `"confirmed_uninformative"` on its held-out set (agreement true for all such sizes), and at
least one size actually predicted `"ruled_out"` (a check with zero ruled-out predictions has
nothing to demonstrate and is reported as **"not applicable (no ruled-out predictions)"** rather
than as a pass). It is **"not demonstrated"** if any `"ruled_out"` prediction disagrees with its
held-out outcome. Sizes with `"not_ruled_out"` or `"not_evaluable"` predictions never count against
demonstration, per the rule above: the check can only rule out, never confirm informativeness, and
a labeller this project cannot evaluate on 100 pairs is not evidence against the check either way.

## Invalidity (Outcome C)

Invalid, no check outcome is reported, if: the per-pair cache for any size is missing or its C127
run was itself invalid (Outcome C) under `C127_PREREGISTRATION.md`'s gate, or the pilot/held-out
split does not partition the 299 pairs exactly (100 + 199, no overlap, no omission) when
recomputed at analysis time.

## Compliance

Local only; reads the same numeric-only per-pair caches C127 writes, never note text, pair text, or
pair_id-to-split mappings. `analyze_c128.py` writes aggregate-only counts and p-values to
`results/accord/c128_prospective_feasibility/c128.json`; it opens `~/mimic_local_only` paths only
to read the per-pair cache and the gold label column, and any stdout/stderr produced while those
paths are open is redirected to a log file under `~/mimic_local_only/logs/` by the invoking
command.

## Outcome

(appended after the run; the rule above is not edited)

**Outcome C** (run analyzed 2026-09-15; artifact `results/accord/c128_prospective_feasibility/c128.json`).
C127's registered gate (5 percent parse-failure ceiling, no truncation) failed for qwen3:4b: under
identical settings to the other four sizes, qwen3:4b's `/api/chat` content channel emitted a
chain-of-thought preamble token (e.g. "First", "Okay") instead of the one-word answer on all 299
pairs (0/299 parsed; independently reproduced with a synthetic, non-clinical probe prompt outside
the credentialed cache, confirming this is the model's own behaviour under `think=false` and
`num_predict=1`, not a parsing defect in `run_c127.py`). Per this file's invalidity rule, the
prospective feasibility check is not evaluated for any size while any size's C127 cell is invalid;
no per-size prediction, held-out outcome, or agreement is computed or registered.
