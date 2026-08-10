# Revision log

**This file exists because `CORRECTIONS.md` C46 declared "the record closes here" and sixteen more
corrections landed after it, thirty-seven commits on that day alone, and not one new chapter was
written in the same period.** The intellectual fix — deciding to stop — was tried and failed within
hours. This is the structural fix.

## The rule

From 2026-08-10, new findings go **here**, not in `CORRECTIONS.md`.

An entry in this file **may** change:
- a number, a table cell, or a figure
- the content or length of a chapter
- a stated limitation

An entry in this file **may not** change:
- the thesis statement
- the contribution list in `CONTRIBUTION.md`
- the chapter structure

**Changing the frame requires an external signature.** All three previous reframes were
self-adjudicated, each on genuine evidence, and each cost about a week. The frame is frozen once
`CONTRIBUTION.md` is signed off; after that it moves only if someone other than the author says it
should.

## The scoreboard

The daily measure is **draft lines written**, not corrections found. A day that produced no chapter
prose failed, regardless of what it discovered. The marginal correction is now worth close to
nothing — C57 was a full correction cycle spent fixing something no chapter cites, by its own
admission — and the marginal page is worth a great deal.

## Entries

*(none yet)*

### R1 (2026-08-10) — the self-audit statistic is retracted; it is unreproducible and wrong in membership

**First entry, and it is a test of this file's own rule.** This finding would once have become
correction C63. It does not, because it changes a number and a contribution's evidence, not the
thesis statement. `CORRECTIONS.md` stays closed at C62.

`results/accord/self_audit_statistics.json` is **retracted** and moved to
`results/accord/retracted/`. Four independently verified defects:

1. **No generating script exists anywhere in the repository.** I produced it in a throwaway shell
   heredoc and never saved the code. The number cannot be recomputed, challenged or corrected by
   anyone, including me. **This is verbatim the defect C49 disqualified the survey for** — and it
   sits under the claim that error-detection procedures fail in the regime they audit. It is the
   thesis's own thesis, committed by its author, in its author's evidence.
2. **The denominator was advanced without recomputing the numerator.** The file records
   `total: 58`. I reported "20 of 62". The numerator was frozen at 58 corrections while the
   denominator moved to 62.
3. **The most consequential retraction is missing from the numerator.** `C56` — the entry titled
   *"The 'opposite directions' that three corrections were built on is an artifact of `abs()`"*,
   which retracts C41, C47 and C50 by name — is **not** in the set. Neither is C57. Verified
   directly against the JSON.
4. **The classification matches no stateable rule.** An independent hand-code against the explicit
   rule *"does this entry withdraw, refute or narrow a claim made in a prior numbered correction?"*
   gives **22 of 62 strict, 30 generous**, intersecting the published set on 12 of a 30-member
   union — **Jaccard 0.40**. The totals nearly agree only because four false positives cancel ten
   false negatives. That is worse than inflation: the number cannot be defended as "right anyway,"
   because its rightness is an accident of two opposed errors. A namespace collision contributes —
   `PHD_PLAN.md` uses `C1`–`C4` for *contributions* while `CORRECTIONS.md` uses them for
   *corrections*, and the two are mixed inside `CORRECTIONS.md` itself.

**Action: the rate claim is dropped entirely, not repaired.** By this project's own C49 rule,
repairing it would require a written instrument, persisted rows and a blind second coder — two to
four more weeks of auditing, which is the activity that has produced zero chapters in six weeks.
The claim it supported does not need a rate.

**What replaces it — a demonstration, which needs no denominator, no comparator and no coding
table.** Four documented instances where a correcting analysis committed the error it was
correcting (C15, C43, C47, C52 — C43 two days after C22 documented that exact failure mode), and
one unlogged `abs()` at `accord/selection_experiment/audit.py:47` that silently invalidated four
directional readings, three of them corrections of corrections (C56). Every instance traces to a
named entry and, for C56, to a line of code.

### R2 (2026-08-10) — the gate-vs-checklist comparison is retracted; its treatment arm has no positives

The "executable gate beats prose checklist" claim was to be the transferable artifact. It is
retracted. Verified against the files:

- **The gate never fired.** Every persisted result records the degeneracy check passing — 6 of 6
  `"ok": true` across the selection, threshold-robustness and PUMS artifacts, and
  `full_run.log:18` reads `Non-degeneracy check: PASS`. There is no recorded FAIL anywhere. A check
  that has only ever evaluated true negatives has a measured catch rate of **0/0**.
- **The "checklist" arm was executable too.** `accord/prevalence_replication/run_distillation.py:96`
  computes `degenerate_hard`, `:106` persists it, and `distillation_results.json:7` records `true`.
  The threshold-independent metric was computed as well, with a source comment naming the exact
  failure mode. C42 was written anyway. **The code computed the defect, wrote it to disk, and I read
  a different field.**

So the real contrast is not *executable versus prose*. It is *a halting check with nothing to catch*
against *a non-halting check that fired correctly and was ignored*. Read honestly, this evidence
argues executability was **not** the binding constraint — the opposite of what it was quoted for.
It also commits the error C59 recorded twenty entries earlier: *"100% agreement across 19 unanimous
negatives says almost nothing about whether the two coders would agree on a positive."*

**What replaces it as the prescriptive contribution: the sign-provenance rule**, which unlike the
gate has a demonstrated true positive. A metric stored as `abs()` or `min/max` destroys the sign and
licenses no directional claim. The C52 retrofit reproduced every prior value **bit-for-bit (max
discrepancy 0.0 across 9,177 fields)** while flipping **seven of fourteen** slopes and changing one
credibility call (C57), verified by two independently written implementations agreeing to
**0.00e+00**.

### R3 (2026-08-10) — DESIGN.md carries numbers C39 corrected

`DESIGN.md:352` still reads **"36,504 examples, 3,730 positives, 734 in the test set"**. C39
corrected these to **34,018 / 2,975 / 570**. C39 named only `THESIS_STATUS.md` and `PHD_PLAN.md`;
C46 claimed to have audited *"every numeric claim in the ten thesis chapters"* — `DESIGN.md` is not
a chapter, so it was never in scope. `WRITING_PLAN.md:127` budgets a week of Chapter 4 against
`DESIGN.md:234-442` calling it *"restructuring not discovery."* It is discovery. Corrected below.
