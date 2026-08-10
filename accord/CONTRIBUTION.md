# The contribution — one page

**Status: DRAFT for advisor sign-off. Once signed, this is the frozen frame. Every chapter is
written against it. Changing it requires an external signature, not a self-adjudicated decision.**

Send this page. Not the 129-page draft, not `CORRECTIONS.md`. One question: **is this a pass — yes
or no? If no, what is missing?** Do not present alternative framings; that returns a menu, which is
what happened the last three times.

---

## The thesis

> Known evaluation-validity failures are diagnosed one instrument at a time, in isolation, usually
> by authors auditing other people's work. This thesis applies those diagnoses end to end to a
> single clinical NLP pipeline — including to its own prior published claims — and reports what the
> controls change, what they cost, and which conclusions do not survive them.

The claims below are **not** novel as mechanisms. Each is attributed to the work that owns it. What
is new is the measurement on this pipeline, and the fact that the controls were actually run.

---

## The five contributions

**1. A measured null, not an underpowered one, on clinical pretraining transfer.**
*Known:* pretraining loss is an unreliable proxy for downstream performance — Lourie, Hu & Cho,
EMNLP Findings 2025 (~39% of the time), and **TESSERA v2, arXiv:2607.03949, July 2026: 395
controlled runs, 15 tasks, |Pearson r| < 0.2.** The general finding is settled and this project's
48-run sweep is a small replication of it — **the sweep is a positive control, not a contribution,
and the chapter must not be written as one** (REVISION_LOG R5).
*New here:* the same instrument resolves a **4.53-point** pretraining benefit on AG News at 24
epochs with SD ≤ 0.003 and disjoint bands, and returns **+0.0031 (Welch p = 0.690)** on the clinical
task. The positive control was adjudicated by a rule committed **before the sweep was run**; eight
times the training budget bought the scratch arm 0.41 points, so the arms were converged and the
null is a measurement rather than an absence of power.
*Evidence:* `results/accord/scaling/convergence_results.json`, `accord/scaling/C44_PREREGISTRATION.md`,
CORRECTIONS.md C37/C44/C62.

**2. A reproducible anomaly in how pool-level fairness audits respond to upstream selection.**
*Known:* disaggregated fairness evaluation is not interpretable under selection — Pfohl et al.,
NeurIPS 2025; Fogliato et al., AISTATS 2020.
*New here:* on **one population with baseline disparity held fixed**, two selection channels produce
signed demographic-parity drifts of **opposite sign** (Regime A −0.00780, Regime B +0.01177, both
CIs excluding zero) — and a controlled simulator sweep over 7 baselines × 6 strengths × 40
replicates **inverts the mapping**. The anomaly is unexplained and is reported as unexplained.
*Evidence:* `results/accord/signed_gap_analysis.json`, `results/accord/c47_baseline_sweep*.json`,
CORRECTIONS.md C50/C56/C57.

**3. A prevalence-of-practice measurement with released rows.**
*Known:* fairness is measured stage-by-stage across pipelines including filters — Biswas & Rajan,
ESEC/FSE 2021.
*New here:* on a 75-paper corpus deliberately enriched for fairness-forward work, **4/75 strict
(5.3%) and 14/75 permissive (18.7%)** audit the selection step, with both readings reported side by
side and the 10-paper gap named as the honest ambiguity. **502/502 quotes machine-verified**, blind
second coder, κ = 0.855 on the permissive field — and κ **undefined** on the strict cell, stated as
a limitation rather than reported as 100%.
*Evidence:* `results/accord/survey/` (protocol, 75-row coding table, screening log, second-coder
table), CORRECTIONS.md C55/C59.

**4. Sign provenance: a metric stored as a magnitude licenses no directional claim.**
*Known:* nothing directly; adjacent to measurement-provenance and reproducibility practice.
*New here:* every disparity gap in this project was stored as `abs()` and four-fifths as `min/max`,
with no per-group values persisted. Retrofitting signed storage **reproduced every prior value
bit-for-bit — max discrepancy 0.0 across 9,177 fields — while flipping seven of fourteen drift
slopes and changing one credibility call.** Two independently written implementations agree to
**0.00e+00**. The rule is transferable and it has a demonstrated true positive: it invalidated four
of this project's own directional readings, three of which were themselves corrections of
corrections.
*Evidence:* `results/accord/sign_provenance_audit.json`,
`results/accord/sign_provenance_drift_recomputation.json`, REVISION_LOG R2, CORRECTIONS.md
C52/C56/C57.

**5. What the controls cost.**
*Known:* nobody reports this.
*New here:* the control that retracted the observational core of three corrections cost **14.2
minutes** of compute. The convergence control that verified contribution 1 cost 9.2 GPU-hours.
*Evidence:* `results/accord/control_cost_log.json`. **Honest limitation: person-hours were not
logged before 2026-08-09 and are not reconstructed. The cost claim is compute-only.**

---

## What this thesis does not claim

- No mechanism is claimed for the contribution-2 anomaly. Five mechanism claims were proposed and
  refuted in this project; the sixth is not being offered.
- The survey is not a prevalence estimate for clinical ML. The corpus is enriched, coverage is PMC
  open access plus arXiv and ACL, and 459 screened-in papers were never full-text retrieved.
- No subgroup harm is demonstrated anywhere in this work.
- The self-audit record (`CORRECTIONS.md`, 62 entries) is **Appendix A and one chapter, not the
  spine**. **No rate is claimed from it.** An attempt to compute one was retracted (REVISION_LOG R1):
  the generating script was never saved, the denominator advanced without the numerator, the most
  consequential retraction was missing from it, and an independent hand-code intersected it on only
  12 of a 30-member union. What the record supports is a **demonstration, not a statistic** — four
  named instances of a correcting analysis committing the error it was correcting, and one unlogged
  `abs()` that invalidated four directional readings.
- **No claim that executable gates beat prose checklists.** That comparison was retracted
  (REVISION_LOG R2): the gate has no recorded failure in 6 of 6 runs, so its catch rate is 0/0,
  while the "checklist" arm computed the defect, wrote it to disk, and was misread. The evidence
  argues executability was not the binding constraint.

---

## What remains before submission

| | cost |
|---|---|
| Delete: 7 proposed-figure blocks, the unbuilt architecture spec, duplicated ethics statement | ~1 day |
| Merge ch3+ch4+ch6+ch7 → one pipeline chapter, 2,890 → ~430 lines | ~5 days |
| Write chapters 5, 7, 4; then 6, 8, 9; introduction and title last | ~4 weeks |
| Second coder on a subsample **enriched for the 4 strict-yes papers** (C59) | ~1 day |

**Zero experiments remain.** 62 corrections, all resolved. The draft compiles at 129 pages with
0 undefined references and 0 undefined citations.

---

## The one question for the advisor

The claim novelty of items 1–3 is low by conference standards — each is a known failure mode applied
to a new pipeline. Items 4 and 5 are new but small. **Is execution-at-this-standard an acceptable
doctoral contribution, or is a novel mechanism required?**

If a novel mechanism is required, the honest answer is that this project does not have one, and the
plan changes.
