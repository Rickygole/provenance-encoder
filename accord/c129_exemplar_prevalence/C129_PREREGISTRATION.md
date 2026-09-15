# C129 pre-registration: how often the original positional selector's design
# would show a non-trigger exemplar, measured on the rebuilt candidate pool

Committed before the first datum. Outcome appended below; the rule above the outcome line is never
edited.

## Why

`accord/phase3_mining.py`'s module docstring (lines 1-28) names "trigger-exemplar dissociation": the
original Rules 1 and 2 mining code (`ACCORD_Phase2_Contradiction_Mining`, cells 4-5) decided a
conflict over the SET of actions seen across every mention of a drug in a note, but then displayed the
FIRST mention of that drug by row position (`.iloc[0]`) to the pharmacist, LLM, and model -- a mention
that need not be one whose own action made the set-level conflict true. The rate at which this actually
happened in the annotated gold set cannot be recovered: the original schema recorded no mention-level
position field, so there is no artifact that says which mention fired (`phase3_mining.py` lines 14-20:
"nothing in the mentions table said *where* a mention was, so mining could not tell mentions apart").

The rebuild (`accord/phase3_mining.py::mine_rule1`, `mine_rule2`) fixes this by restricting the
exemplar to mentions whose own `action` field satisfies the rule branch that fired (the "trigger set"),
and it computes, per candidate, exactly which mentions qualify. This gives us something the original
never recorded: for every rebuilt candidate, we can compare the ORIGINAL selector's rule
(`n_sub.iloc[0]` / `d_sub.iloc[0]`, the same subsetting `mine_rule1`/`mine_rule2` themselves use) against
the trigger set, on the rebuilt pool.

**This is not the gold set's defect rate.** It measures a property of a selector *design* applied to a
*different* pool (the rebuild's own candidates, from the rebuild's own extraction), not the actual
historical selection that produced the 165 pooled Rule 1/2 gold exemplars. See "Row order" below for
exactly what the code lets us rule in and rule out about the relationship between the two.

## Inputs (found in code, not guessed)

`accord/phase3_mining.py::main` reads `OUT / "hf_medication_mentions.csv.gz"` where
`OUT = Path(__file__).resolve().parent / "local_only"`, i.e. `accord/local_only/`, which is a symlink
to `/Users/rickygole/mimic_local_only/accord/` (verified with `readlink`). This run reads the same file,
read-only, and writes nothing there. `accord/local_only/phase3_mining_summary.json` is the rebuild's own
prior recorded run (fields `rule1_candidates` ... `total_candidates`); it is the "rebuild's recorded
counts" gate below (`results/accord/source_summaries/mining_summary.json` is a different, older artifact:
its counts, 886/203/521/370/1980, match the hardcoded `reference_ruleN` constants inside
`phase3_mining.py` almost exactly and its git history shows it was committed as a source summary of the
ORIGINAL reference pipeline, not this rebuild -- it is not used as the regeneration gate for that
reason).

## What is computed

On the candidate pool produced by calling `phase3_mining.mine_rule1` and `mine_rule2` directly (imported,
not modified, not reimplemented for the conflict condition itself), for Rule 1, Rule 2, and pooled:

- **n candidates.**
- For each candidate, the **trigger set** on each side: nursing mentions of the drug whose own `action`
  equals the required nursing action for that candidate (`new_n_candidates` in `mine_rule1`/`mine_rule2`'s
  own code), and symmetrically for discharge.
- The **positional pick**: `n_sub.iloc[0]` / `d_sub.iloc[0]` on the identical `n_sub`/`d_sub` subframes
  `mine_rule1`/`mine_rule2` themselves build (same drug, same note side, same admission) -- this is
  exactly `old_n`/`old_d` inside those functions, not a separate reconstruction.
- Count and Wilson 95 percent interval for: positional pick outside the trigger set on the nursing side;
  on the discharge side; on either side. Pooled = Rule 1 and Rule 2 candidates concatenated.
- As a construction check: the same three rates for the rebuild's OWN selector (`pick_exemplar`'s output,
  `new_n`/`new_d`) -- must be exactly 0 on all three, because `pick_exemplar` only ever chooses among
  members of the trigger set already filtered by the calling code; a nonzero value here means this
  script's mirroring of `mine_rule1`/`mine_rule2` has a bug, not a real finding.
- The **share of candidates with a single mention per side** (nursing side, discharge side, and both
  sides), where `len(n_sub) == 1` or `len(d_sub) == 1` respectively: position cannot matter for that side
  because there is only one row `.iloc[0]` could ever return.

### Row order: exactly what it is, and what the code lets us rule in or out

`n_sub = nursing[nursing["drug"] == drug]` and `nursing = group[group["note_role"] == "nursing"]` where
`group` is one `groupby("HADM_ID")` group of `classified` -- pandas preserves the relative row order of
the input frame inside each group and inside each boolean-mask subset, so `n_sub`'s row order is exactly
the row order of `hf_medication_mentions.csv.gz` restricted to that admission, note role, and drug.
`accord/phase2_medication.py::extract_mentions_from_doc` builds that CSV by iterating `doc.sents` in
order (`for sent in doc.sents`, `sentence_idx = i` at line 128) and appending each sentence's mentions as
it goes, so within one note a drug's mentions appear in ascending `sentence_idx`, i.e., first-occurring
position in the note text first. That is what "the same row order the original used" means here: it is
the rebuild's own, code-verified order, and it is what `mine_rule1`/`mine_rule2` themselves use for their
`.iloc[0]` -- there is no separate reconstruction of order in this script.

Whether this order matches the ORIGINAL reference notebook's row order for the same drug and note is a
different question, and the code lets us resolve it only partway:
- **Ruled out as a joint risk:** there is no join, no separate mentions table, and no key mismatch here.
  `old_n`/`old_d` and the trigger set are computed from the identical `n_sub`/`d_sub` frame in the same
  function call -- unlike the gold-set join measured in C88/C90/C120, there is nothing here for two
  artifacts to disagree about.
- **Not ruled out, and not recoverable:** `accord/phase2_medication.py`'s own docstring (lines 1-24)
  states this mentions table is a fresh REBUILD extraction using spacy's `sentencizer` component instead
  of the reference's full `en_core_web_sm` pipeline for sentence boundaries, over the same notes but a
  new pass -- so even same-note sentence boundaries, and therefore mention order, can shift at edge
  cases relative to the reference. And per `phase3_mining.py` lines 14-20, the original pipeline recorded
  no position field at all, so there is no artifact from the reference run to diff this rebuild's order
  against, in either direction. This measurement is therefore a property of the rebuild's own pool, full
  stop -- it says how often the ORIGINAL SELECTOR'S DESIGN (first row by position) would land outside
  the trigger set if run on THIS pool, not how often it did so on the pool that actually produced the
  165 pooled Rule 1/2 gold exemplars.

## Decision framing

Descriptive. No threshold, no pass/fail claim about the gold set. The result answers: "on the rebuild's
own candidate pool, if the original's positional design were applied instead of the fix, how often would
it show something other than a trigger mention" -- separately by side, by rule, and pooled, with
intervals, and with the single-mention share reported alongside so the reader can see how much of the
denominator could not have been affected by the bug at all.

## Invalid (Outcome C)

If `phase3_mining.mine_rule1`/`mine_rule2`/`mine_rule3`/`mine_rule4` cannot be imported and run against
`accord/local_only/hf_medication_mentions.csv.gz`, or if the regenerated candidate counts
(`rule1_candidates`, `rule2_candidates`, `rule3_candidates`, `rule4_candidates`, `total_candidates`) do
not exactly match the values recorded in `accord/local_only/phase3_mining_summary.json`, nothing below the
gate is computed or reported, and the outcome is C.

## Compliance

Local only. `stdout`/`stderr` redirected to `~/mimic_local_only/logs/`, not read back into this
conversation. `accord/cloud_paths.assert_local_only` is called on every path this script reads or writes
under the credentialed data root. The output JSON (`results/accord/c129_exemplar_prevalence/c129.json`)
holds only counts, rates, and intervals -- no sentence text, no drug names tied to a row, no `HADM_ID`,
no `SUBJECT_ID`, no `pair_id`.

## Outcome

**Outcome A** (run 2026-09-14, artifact `results/accord/c129_exemplar_prevalence/c129.json`). Both
gates held: the regenerated pool (`rule1_candidates` 1061, `rule2_candidates` 246, `rule3_candidates`
659, `rule4_candidates` 330, `total_candidates` 2296) matched `accord/local_only/phase3_mining_summary.json`
exactly, and this script's mirror of `mine_rule1`/`mine_rule2` produced the same candidate counts as
calling those functions directly (1061/1061, 246/246).

On the rebuilt pool, if the original's positional design (`.iloc[0]`) were applied instead of the fix:

- **Rule 1** (n = 1061): outside the trigger set 11.8 percent nursing side (125/1061, 95 percent Wilson
  0.100-0.139), 32.6 percent discharge side (346/1061, 0.299-0.355), 39.5 percent either side (419/1061,
  0.366-0.425).
- **Rule 2** (n = 246): 15.4 percent nursing side (38/246, 0.115-0.205), 40.2 percent discharge side
  (99/246, 0.343-0.465), 49.6 percent either side (122/246, 0.434-0.558).
- **Pooled** (n = 1307): 12.5 percent nursing side (163/1307, 0.108-0.144), 34.0 percent discharge side
  (445/1307, 0.315-0.367), 41.4 percent either side (541/1307, 0.388-0.441).
- Construction check: the rebuild's own selector (`pick_exemplar`) landed outside the trigger set 0/1307
  times on every side, as required.
- Single-mention-per-side share (position cannot matter there): pooled 68.7 percent nursing side, 26.3
  percent discharge side, 19.3 percent both sides at once -- i.e., on the discharge side, in nearly 74
  percent of candidates a drug is mentioned more than once, which is exactly where the original design
  had more than one row to get wrong.

This is descriptive, about a selector design applied to the rebuild's own pool, not the gold set's
defect rate (see "Row order" above for what the code does and does not let us conclude about the
relationship between the two).
