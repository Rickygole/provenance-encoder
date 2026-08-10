# C44 convergence sweep — pre-registered decision rule

**Written 2026-08-09, BEFORE the sweep was run.** Committed before any result exists. Do not edit
after the first number lands; if the rule turns out to be badly specified, record that as a
correction rather than revising this file.

## Why this exists

This project has retracted five mechanism or directional claims in two weeks (C14, C42/C43, C47,
C50's real-data leg, C56). In every case the claim was written *after* the numbers were seen and
was shaped, unconsciously, to fit them. The one time a kill condition was committed in advance
(C50, commit `966c87e`), it fired against the claim I wanted and I found out in a day instead of at
a viva.

So: the rule below is written while the sweep has not been started.

## What is being tested

C44 reports a **5.2 accuracy-point** AG News gap between a 99M-parameter from-scratch encoder
(0.8878) and pretrained BERT (0.9396), at 3 epochs, 3 seeds. That gap is the **positive control**
for Instrument II — it is the evidence the instrument can detect a real pretraining benefit, which
is what licenses reading the clinical null (C37: gap +0.0031, Welch p = 0.690) as a genuine null
rather than as an experiment too insensitive to see anything.

**The threat.** Every arm trained for exactly 3 epochs with no per-arm tuning, and the 100M arm
gained only +0.009 over the 10M arm. If the scratch arms simply had not finished learning, the
5.2-point gap measures *training budget* and the control collapses.

The sweep runs budgets **3 / 6 / 12 / 24 epochs × 4 arms × 3 seeds**, each with its own complete
OneCycle schedule, selection on a held-out validation split. Both scratch and BERT get every budget.

## The decision rule

Read the gap between `scratch_100M` and `pretrained_bert` at the **24-epoch budget**, and the
seed spread of each arm at that budget.

### Outcome A — the control holds

> **Gap ≥ 3.0 accuracy points at the 24-epoch budget, AND the arm ordering is seed-resolved**
> (the two arms' ±1 SD bands do not overlap).

Instrument II has a verified positive control. C44 stands. **Chapter 5 stands at ~320 lines** and
the thesis is nine chapters, ~3,310 lines. C37's clinical null is reportable as a null: the
instrument demonstrably detects a pretraining benefit where one exists, and does not detect one on
the clinical task.

### Outcome B — the control does not hold

> **Gap < 3.0 accuracy points at the 24-epoch budget, OR the ordering is not seed-resolved.**

The positive control fails. **C44 is withdrawn.** C37 reverts to *"an experiment too insensitive to
see anything"* and must be written that way — not as a null result about pretraining.

**Chapter 5 collapses to ~140 lines and merges into Chapter 6** as a second proxy-substitution
case, reported under the thesis frame as an episode whose verdict is **UNINTERPRETABLE** and whose
cost is one GPU session. The thesis becomes eight chapters, ~3,130 lines.

This is not a failure of the thesis. Under the retrofitted-control frame, an instrument whose
positive control does not hold is a *result* — it is the case where adding the control revealed
that the original reading was never interpretable. That is the same shape as C52/C56 and it belongs
in the same taxonomy.

### Boundary case

If the gap lands in **[2.5, 3.5]** points, or the SD bands touch without overlapping, the result is
**INCONCLUSIVE** and neither Outcome A nor B may be asserted. In that case: report the budget curve
as the finding, state that the control is unresolved at this compute scale, and write Chapter 5 at
the Outcome B length. **Do not** resolve a boundary case by adding seeds until it moves — that is
optional stopping and it is how C47 happened.

## What is NOT being decided here

The sweep does not test whether pretraining helps on the clinical task. It tests whether the AG
News comparison is sensitive enough to be a control. Those are different questions and the write-up
must not merge them.

## What is blocked on this

**Only:** Chapter 5's existence and length, one sentence in Chapter 1, and one row in
`fig-forest`. **Nothing** in Chapters 3, 4, 6, 7, 8 or 9 depends on the outcome. The lead
instrument is Instrument IV, whose evidence is complete and needs no GPU, specifically so that this
sweep is not on the critical path.
