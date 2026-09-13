# C106 pre-registration: a direct exemplar check on the displayed sentences

Committed before the first datum. The outcome is appended below after the run; the
rule above the outcome line is never edited.

## Why

`accord/thesis/msu_ch4_results.tex`, Section "The exemplar selection defect",
reports two irreconcilable readings of how often the exemplar-selection defect
fired on the 165 pooled Rule 1 and Rule 2 gold candidates: a reconstruction over
137 resolved rows finds 65 wrong (0.4745), and an exact key join over 23 rows
finds 0 wrong, and the chapter states plainly that "one of these two numbers is
still wrong and this thesis still does not know which." Both readings depend on a
join back to a mentions-table snapshot or a join key. This run instead reads the
sentences the pharmacist actually saw
(`~/mimic_local_only/from_drive_2026-08-08/outputs/annotation_sheet_for_pharmacist_clean.csv`,
columns include `pair_id`, `rule`, `drug`, `drug_class`, `nursing_sentence`,
`discharge_sentence`) and re-derives, from the sentence text itself, whether each
displayed pair actually exhibits the conflict its rule asserts, using the
project's own extractor rather than a reconstructed or joined label.

**This is not a ground-truth measurement.** The mention extractor
(`accord/phase2_medication.py::extract_mentions_from_doc`) and the action
classifier it calls (`accord/dictionaries.py::ACTION_PATTERNS`,
`resolve_action`) have never been validated against human annotation of action
classification -- only against each other, in the reconstruction's own
calibration check (0.8934 nursing-side, 0.9508 discharge-side agreement with the
`.iloc[0]` labels, `results/accord/exemplar/exemplar_correctness_recomputed.json`).
This run reports what the extractor, applied fresh to the displayed text, says
about the displayed text. It is a third instrument, not an arbiter, and its
numbers are reported as extractor-dependent, not as the true defect rate.

## What runs

Both rule-generating branches this check covers are read directly from
`accord/phase3_mining.py::mine_rule1` and `mine_rule2` (imported for their
constants and read for their logic; not modified):

- **Rule 1** candidates carry one drug in the `drug` column. A genuine conflict
  requires that drug's action to be `continuation` in one of the two displayed
  sentences and `cessation` in the other -- either direction, since the
  annotation sheet does not retain which of the two directions
  (`mine_rule1`'s `cond1` vs `cond2`) triggered the original candidate at the
  full-note level, and either direction is a real Rule 1 conflict on its own
  terms.
- **Rule 2** candidates carry `"drugA vs drugB"` in the `drug` column
  (`mine_rule2`'s `drug_label = f'{new_n["drug"]} vs {new_d["drug"]}'`): `drugA`
  is checked against `nursing_sentence`, `drugB` against `discharge_sentence`,
  same opposite-action, either-direction requirement.

Per row:

1. Normalize the row's drug name(s) exactly as `extract_mentions_from_doc` does
   internally (`ABBREV.get(drug.lower(), drug.lower())` then
   `VARIANTS.get(..., ...)`). The `drug` column is already the extractor's own
   normalized string (it is copied verbatim from the mentions table by
   `mine_rule1`/`mine_rule2`), so this step is expected to be a no-op; it is kept
   explicit rather than assumed.
2. Apply `phase2_medication.py::expand_text` (imported, not modified) to
   `nursing_sentence` and `discharge_sentence`, matching the abbreviation/variant
   expansion the reference pipeline runs before sentence segmentation.
3. Wrap each expanded sentence alone as a one-element sentence list
   `[(text, 0, len(text))]` and call
   `extract_mentions_from_doc(sentences, note_role)` (imported, not modified) on
   it. With one sentence, the function's own context window
   (`texts[max(0,i-1):min(len,i+2)]`) degenerates to that single sentence, which
   is exactly what "the displayed sentence" means here: no neighboring sentence
   was ever shown to the pharmacist.
4. A side is **decidable** if the extractor's regex finds the target drug
   anywhere in that displayed sentence at all (a mention record is returned for
   it); if not, that side is undecidable (drug absent from the text the extractor
   can search, most plausibly because the 200-character truncation cut it, or
   because the canonical drug string does not literally appear in the sentence
   even though the mention exists). A row is decidable iff both sides are.
5. Among decidable rows, a row is **correct** (no exemplar defect) iff the pair
   of actions found is exactly one `continuation` and one `cessation`, in either
   assignment to the two sides; otherwise it is **wrong** (defect fired): the
   displayed pair does not itself carry the opposite-action conflict its rule
   asserts, whether because the action found is `initiation`, `dose_modification`,
   `unclassified`, or the same polarity on both sides.
6. `n_at_cap` is the count of the 165 rows whose `discharge_sentence` has length
   exactly 200 (`results/accord/annotation_sheet_span_profile.json` already
   reports 200 as the whole sheet's cap and 37 discharge cells at it across all
   300 rows; this run recomputes the count within the 165-row Rule 1/2 subpopulation
   specifically). `defect_rate_excl_capped` is the same statistic as step 5,
   restricted to decidable rows whose discharge cell is not at the cap.
7. Wilson 95% interval (`accord.stats._overcall_common.wilson_ci`, imported, not
   modified; two-sided, z=1.96) on the decidable-row defect count, and again on
   the capped-excluded count.

**What this run cannot do, disclosed in advance.** The 23 key-joined rows and the
137 reconstructed rows are named only by aggregate counts in the retained
artifacts (`results/accord/exemplar/exemplar_correctness_recomputed.json`,
`~/mimic_local_only/accord_exemplar/audit_exemplar_correctness_rule_only.csv`);
neither carries `pair_id`, and no join script that would recover which specific
165 rows belong to which of those two subsets was found anywhere in this repo
(`git log --all -- '*exemplar*'` and a search of `accord/` for the two artifact
names turn up only consumers of the finished JSON, not a generator). Row-level
agreement -- did this instrument and the reconstruction call the *same* row wrong
-- is therefore not computable from what is committed, and is not attempted. Only
aggregate agreement is reported: whether this run's Wilson interval contains,
excludes, or sits nearer one of the two existing point estimates, 0 and 0.4745.

## Decision rule

- **Outcome A, settles the conflict:** the Wilson interval on the decidable-row
  defect rate excludes one of {0, 0.4745} and either includes, or is numerically
  nearer to, the other. Reported as: the direct measurement favors the
  [nearer] reading and is inconsistent with the [excluded] one, under this
  extractor.
- **Outcome B, does not settle it:** the interval includes both, or excludes
  both. Reported as: the direct measurement neither confirms nor rules out either
  existing reading.
- **Outcome C, invalid:** fewer than 100 decidable rows among the 165, or the
  required columns (`pair_id`, `rule`, `drug`, `drug_class`, `nursing_sentence`,
  `discharge_sentence`) are absent from the source file. Reported as not run; no
  rate is quoted.

Whatever the outcome, it is reported as a measurement made by an unvalidated
instrument (the extractor and action classifier), not as ground truth, and beside
-- not in place of -- the two existing readings.

## Kill condition

If the run is interrupted, it is restarted from scratch, never spliced. No change
to the normalization, the opposite-action decision rule, the decidability
criterion, or the cap threshold after the outcome is seen.

## Outcome

Recorded 2026-09-13 after the run; the rule above is unchanged.

**Outcome B, does not settle the conflict, and lands far outside both existing
readings.** Of the 165 Rule 1 (134) and Rule 2 (31) candidates, 146 are decidable
(19 undecidable: 13 nursing-side, 5 discharge-side, 1 both, the target drug's own
regex not found in that displayed sentence at all). Among the 146 decidable rows,
136 are called wrong under this instrument's opposite-action criterion: defect
rate 0.9315068493150684 (136/146), Wilson 95% interval
[0.8785152243518337, 0.962373545629886]. That interval excludes both existing
point estimates, 0 (the 23-row key join) and 0.4745 (the 137-row reconstruction),
so per the rule above this is Outcome B, not A: the direct measurement does not
land near either existing reading, it lands well above both, and settles neither.
15 of the 165 discharge cells sit at the 200-character cap; excluding them,
137 decidable rows remain and the rate is 129/137 = 0.9416058394160584, interval
[0.8890051616346314, 0.970116868292974] -- capping is not what is driving the
high rate, since removing the capped rows moves it up, not down.

**Why the number is this high, most likely: the isolated-sentence context window,
not the historical selection defect.** Of the 136 rows called wrong, 69 have
`unclassified` on at least one side (the extractor found the target drug but no
action-pattern keyword anywhere in that one sentence) and 67 have a real,
classified action on both sides that is simply not the opposite pair (both
`continuation`, both `initiation`, or a non-opposite mix such as
`continuation`+`initiation`). This pre-registration flagged this risk before
running: `extract_mentions_from_doc`'s own context window is normally three
sentences (the sentence plus one neighbor each side) drawn from an un-truncated
note; here, by construction, it is exactly the one sentence the pharmacist saw,
frequently truncated to 200 characters. A cessation cue ("hold", "discontinued")
one clause before or after the extracted sentence is invisible to this
instrument and was not invisible to the historical extraction that produced the
`nursing_action`/`discharge_action` columns audited by the other two readings.
The 69 `unclassified`-involving rows are the more clearly context-window-driven
half; the 67 both-classified-but-wrong rows are closer in spirit to the defect
the other two readings measure, though still short of it, since a same-polarity
or off-pair reading here does not distinguish "the pharmacist's displayed pair
genuinely does not conflict" from "this one-sentence window missed the cue that
would have made it conflict."

**Reading against the thesis.** This is a third number, not a tiebreaker, and it
is reported as such: `msu_ch4_results.tex`'s "range" of 0 to 0.4745
(Section "The exemplar selection defect") does not contain 0.9315, so a reader
who wanted a single quotable ceiling from this exercise should not take either
existing number as one; the true ceiling, under any instrument tried so far,
is unknown. This measurement is reported beside the other two, not in place of
them, per the pre-registration's own caveat: it is an unvalidated extractor
applied under a materially harder condition (one truncated sentence, no
neighbor) than either of the other two instruments faced, and the gap is very
plausibly instrument artifact rather than a true rate three times the
reconstruction's.

**What could not be done, as disclosed in advance.** Row-level agreement with the
23 key-joined or 137 reconstructed rows was not attempted: no artifact in this
repo retains which of the 165 `pair_id`s belong to either subset (only aggregate
counts survive in
`results/accord/exemplar/exemplar_correctness_recomputed.json` and
`~/mimic_local_only/accord_exemplar/audit_exemplar_correctness_rule_only.csv`,
neither of which carries `pair_id`), so only the aggregate comparison above is
reported.

Artifact: `results/accord/exemplar_direct/exemplar_direct.json`. Register prefix
`exemplardirect.*` (register script written, not run; see
`accord/exemplar_direct/register_exemplardirect.py`).

