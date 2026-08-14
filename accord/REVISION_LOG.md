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

### R4 (2026-08-10) — the round-two novelty pass closes the self-audit frame; the general proposition is occupied in five literatures

Round one reported that one clause "searched clean." That was an artifact of searching four
ML-adjacent indexes. Round two reached psychometrics, metascience, patient safety and software
engineering, and the proposition **"error-detection procedures are themselves instruments and fail
in the regime they audit"** is stated — and in several cases *measured* — in all of them:

| | |
|---|---|
| Ioannidis, *Why Science Is Not Necessarily Self-Correcting*, Perspect Psychol Sci 2012 (PMID 26168125) | the thesis sentence, in metascience, fourteen years old |
| Huang et al., *LLMs Cannot Self-Correct Reasoning Yet*, ICLR 2024 (arXiv:2310.01798) | the second clause, in ML, with two years of follow-on work |
| Shebl, Franklin & Barber, *Is failure mode and effect analysis reliable?*, J Patient Saf 2009 (PMID 19920447) | an error-detection method tested as an instrument and found unreliable, seventeen years ago |
| Yin et al., *How do fixes become bugs?*, ESEC/FSE 2011 | the rate at which corrections are themselves incorrect, at scale, many codebases |
| Nuijten et al. 2016 (PMID 26497820) + arXiv:1610.01010 + arXiv:2408.07948 | the statcheck line: an error detector's own false-positive and miss rates, including first-person validation |
| DeMillo, Lipton & Sayward 1978 | mutation testing — measuring your own detector's miss rate by planting known defects. Forty-eight years old |

**What remains genuinely unfound:** no single researcher releasing their own correction log and
computing a recurrence rate over it. But that is a **sampling-frame difference, not a conceptual
one**, and "nobody has done it on themselves" will not be accepted as a substitute for "nobody has
done it." **Honest rating: 3/10, and 3 only because of the released artifact, not the claim.**

**R2's retraction is independently confirmed and out-measured.** Nuijten & Wicherts, PsyArXiv 2023
(DOI 10.31234/osf.io/bxau9): preregistered, two implementing journals against matched controls,
7,000+ articles — journals with the automated tool *in the workflow* show a considerably steeper
decrease in reporting inconsistencies. That is "executable enforcement works where prose does not,"
controlled and preregistered, on an error-detection procedure. Also Sadowski et al., CACM 2018
(Google static analysis: the lesson is workflow blocking, not analysis quality); Urbach et al.,
NEJM 2014 (population-scale null for surgical checklists); and arXiv:2606.16974, which finds
documentation improvements across five conferences **predate** the formal checklist requirements.
**No novelty remains here, not even in the measurement.** One paragraph with citations, not a
chapter.

**A proposal considered and declined.** The pass suggested a *matched first-person / field-wide
prevalence comparison* — the 4/75 survey rate placed beside a rate computed on the author's own
record — as the one unoccupied claim. **Declined, because it requires exactly the statistic R1 just
retracted.** Rebuilding it to survey standard means a written instrument, persisted rows and a blind
second coder: two to four more weeks of auditing, which is the activity that has produced zero
chapters in six weeks. Recording the option here so the decision is visible rather than silently
dropped; it can be revisited only if an advisor says novelty is required.

**Coverage still open after two passes, stated so it is not mistaken for a clean result:** Semantic
Scholar returned HTTP 429 on every attempt in *both* sessions. FAccT and AIES were enumerated via
DBLP, which indexes **titles only** — a relevant paper whose title lacks the search terms is
invisible. Messick remains unverified. General web search was unavailable in both passes.

### R5 (2026-08-10) — C62 is a positive control, not a contribution

`arXiv:2607.03949` (TESSERA v2, July 2026), verified directly: **395 controlled training runs across
15 downstream tasks**, headline *"pretraining loss barely predicts downstream performance
(|Pearson r| < 0.2), so selecting models by loss wastes a large share of the compute."* Two further
2026 preprints agree — arXiv:2602.11618 (chemical language models; loss-landscape alternatives also
fail to predict) and arXiv:2604.22348 (medical claims model, task-dependent saturation).

**C62's 48-run sweep is a small replication of a 395-run result published a month earlier.** It
remains a valid positive control — it licenses reading C37's clinical null as a measured null rather
than an underpowered one — but it is **not a contribution and the chapter must not be written as
one.** `CONTRIBUTION.md` item 1 is amended accordingly.

### R6 (2026-08-10) — C54's checkpoint premise is false; 23 checkpoints exist on the Drive mount

C54 states: *"No ACCORD or Stage 0 checkpoint exists on disk at all."* **That is wrong.**
`My Drive/thesis/outputs/` contains **23 `.pt` files**, including `accord_stage1_best.pt`,
`accord_stage2_best.pt`, `accord_v2_best.pt`, `atcg_pretrained.pt`, the three
`accord_v2_decontam_seed{0,1,2}.pt`, and seven ablation checkpoints. Verified by listing the mount.

The audit that produced the claim searched the repository and the MIMIC quarantine and **did not
search the Drive mount** — the same mount this project has been syncing to all week. I relayed
"no checkpoint exists anywhere" as verified fact without checking the one location most likely to
hold model weights.

**What survives in C54, and it is the load-bearing part:** the argument that the 22,446,855
configuration was never trained rests on two things independent of the checkpoint inventory — no
code path anywhere unfreezes encoder layers (`accord/stage0/train.py` freezes the backbone
unconditionally), and 590,592 of the reconstructed total is a pooler that is off the Stage 0
forward path and receives zero gradient, verified by a real forward+backward. Both stand. The
36.88× live-parameter ratio stands and is correctly propagated.

**What must change:** every sentence in C54 and in any chapter asserting that no checkpoint exists
must be rewritten before printing, and the 23 checkpoints must be inventoried for parameter counts
— one of them may match a configuration the draft describes. Until that is done, **no claim of the
form "X was never trained" may rest on checkpoint absence.**

**Second occurrence of the same error class.** C36 recorded ~30 "untraceable" numbers; the figures
agent later found two of them were in the Drive quarantine it never searched. An artifact is not
absent because the search did not reach where it lives.

### R7 (2026-08-10) — the number audit: 15 mismatches, 9 of them abstract-level; the introduction inverts the central claim

`results/accord/number_provenance_audit.json` — 72 adjudicated claims, 1,791 numeric literals
extracted, 96 quantities independently recomputed by re-runnable scripts under `accord/stats/`.

| verdict | total | abstract-level |
|---|---|---|
| VERIFIED | 48 | 21 |
| MISMATCH | 15 | **9** |
| SUPERSEDED | 3 | 2 |
| UNTRACEABLE | 6 | 2 |

**The worst one, now fixed.** `ch1:240` read *"precision $0.047$ against a base rate of
$0.0424 \pm 0.0023$"* — in the paragraph headed "The headline is stated once, plainly." But
$0.0424$ **is ACCORD's decontaminated precision**, not a base rate, and the sentence also reverted
precision to the contaminated $0.047$ mid-clause. Because $0.047 > 0.0424$, the introduction read
as *the detector beats the pool's base rate* — the exact inverse of `ch7:201` (*"ACCORD's
decontaminated precision of 4.24% ± 0.23 is below it"*) and of the thesis's central negative
result. Corrected to state precision $0.0424 \pm 0.0023$ against the base rate $0.0474$
(12 of 253), with the direction named explicitly. **This is C53's failure class — a cell carrying
another quantity's value — in the most-read paragraph of the document.**

**Five more blocking items, all edits, none requiring an experiment:**

1. **"Above chance" was formally retracted at `ch7:484-491`** (CI [−0.0319, 0.2484] contains zero)
   and the retraction never reached `ch1:254`, `ch10:83`, `ch10:110` — where it carries the claim
   "the task is not the problem" — and `ch7:791` contradicts `ch7:484` **inside the same chapter**.
2. **Top-15 contradictions: "zero" in the framing chapters, "one" in the body.** `ch1:190` and
   `ch10:73` say zero; `ch7:293`, `ch7:930`, `ch8:321` say one. **One** is correct (ensemble;
   per-seed 1, 0, 0). Zero is the contaminated count.
3. **Denominator: 0.5145 and the GPT-4o comparison are attributed to n=299; the artifact is n=253**
   — `ch1:377`, `ch7:233`, `ch10:70`, `ch1:249`, `ch10:107`. `ch7`'s own caption concedes n=253 and
   contradicts the table it captions.
4. **C16-superseded in seven places:** α "moved by 1.8% and 0.6%" is a *contaminated single seed*.
   The 3-seed decontaminated values give **1.5% and 0.4%**. `ch6:189` additionally describes that
   single seed as "across three seeds".
5. **Two provenance notes are false and cost the draft real evidence.** `ch6:301-306` says
   `transfer_results_correct.json` "could not be located on disk" — **it exists**, and 0.0654 /
   0.0595 / 0.9603 reproduce exactly. `ch7:85` claims the MedNLI figures are "the only numbers in
   this thesis whose producing artifact could not be found", which `ch6:305` contradicts. Fixing
   these **strengthens** the negative result by restoring artifacts the draft had written off.

**Also logged:** the C58 bootstrap CI [−0.1040, +0.0966] has no artifact — it exists only in the
correction and the .tex, and an independent re-run gives [−0.1016, +0.0970]. Right in substance,
not reproducible to the printed digits because the seed was never recorded. **That is the C48
pattern and it is mine, from this morning.** Mixed `ddof` conventions inside one comparison
(`ch7:242/259` vs `ch7:284`), `±` never defined, and no multiple-comparison accounting across ~20
tests are logged in the JSON.

**Neither the survey nor C62 has any thesis exposure** — grep finds no chapter citing either. Both
are verified and unused, which is consistent with the finding that the audit has decoupled from the
document.

**Verdict: not submittable until items 1–5 and R6 are fixed. All are edits.**


### R8 (2026-08-11) — three source conflicts resolved before they reached the page

The chapter agents refused to state numbers they could not reconcile, which is what the fact-sheet
discipline is for. Each conflict is resolved below and the resolved value is now a key in
`VERIFIED_FACTS.json`.

**1. The exemplar defect rate. `FRAMING.md` was wrong.** It carried 41.4%. The number-provenance
audit rates the underlying exemplar-correctness figures **UNTRACEABLE**: reported 0.605 / 0.504 /
0.586 against recomputed 0.5556 / 0.4138 / 0.5255. The defensible statement comes from
`exemplar_correctness_recomputed.json`: of the rule 1 and 2 candidates whose exemplar could be
resolved, **65 of 137 were wrong, a rate of 0.4745**, with **28 unresolved**. `FRAMING.md` is
corrected. No chapter may state 41.4%.

**2. The sign-flip count. Three sources disagreed; the artifact settles it.** `CONTRIBUTION.md` said
seven of fourteen, the audit summary said six of eight, its itemised list implied seven of nine.
Recomputed directly from `sign_provenance_drift_recomputation.json`, counting every metric with both
a stored absolute slope and a recomputed signed slope: **7 sign flips of 14 comparable metrics**.
Two independent counts agree. Stored as `drift.n_sign_flips` and `drift.n_metrics_compared`.

**3. The survey reliability coefficients. Two studies, not one number.**
`results/accord/survey/second_coder_reliability.json` reports 0.783 / 0.737 / 1.0. That file is the
**10-paper study C49 disqualified for having no rows on disk**, and it describes the retired 31-paper
survey. The current figures come from the 19-paper blind recode with rows on disk, recomputed by
script: **κ 0.855 on the permissive reading, 0.730 on null-attribute handling, and undefined on the
strict decisive cell** where both coders returned the same value on all 19 papers. The old file is
superseded and must not be cited.

**Also added to the fact sheet**, closing gaps the agents flagged: the per-model over-calling ratios
(pool **15.82 to 23.60**, gold **15.42 to 23.75**, zero unparseable responses across all three
models), the anisotropy control at **0.9113** against matched clinical **0.9176**, and the
recomputed exemplar strata. The fact sheet now carries **over 600 verified entries**.

### R9 (2026-08-13) — the minimum detectable AUROC is unreproducible; the full Hanley and McNeil procedure supersedes it

The stored minimum detectable AUROC for the primary evaluation set, 0.7382, does not reproduce under
either procedure actually in this project: not the null-standard-error approximation written out in
the methods chapter (which gives 0.7397 at four-decimal precision, not 0.7382), and not the full Hanley and
McNeil (1982) sampling-variance procedure already implemented in the fig-power figure script (which
gives 0.7355). `accord/stats/recompute_mde.py` implements the full procedure for all three ambiguity
treatments in Table 4 and writes `results/accord/mde_recomputed.json`. The recomputed values,
0.7355 for the ambiguity-excluded primary set, 0.7350 for the ambiguity-retained set and 0.8058 for the
defect-free stratum, are adopted throughout the thesis, replacing every instance of the old value.
Text and figure now agree, because both read the same procedure: the fig-power marker was already
computed this way and the text had not caught up to it.
