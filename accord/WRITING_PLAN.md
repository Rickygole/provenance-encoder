# Writing plan — final

**2026-08-09.** Experimental record closed (CORRECTIONS.md C46). This is the execution plan for
what remains. Two independent reviews converged on the same reframe; where they differed, the
adjudication is noted.

---

## 1. The thesis statement

The previous statement was falsified in one clause, overclaimed in another, and undone in a third.
*"The fairness audit"* — C38/C40 show the audit **is** sensitive under feature-driven selection.
*"Given a theoretical account"* — C31, elementary, Hernán 2004 is the framework. *"A protocol
proposed and validated"* — not built, n=1, and the telescoping identity proves nothing.

**The replacement:**

> **[Revised by C47 — the audit clause below understates the finding. The pool audit is not
> insensitive; its reading moves in a direction determined by the population's pre-existing
> disparity, inventing a gap where there is none and erasing one where there is. Rewrite this
> clause before Chapter 1.]**

> Five instruments routinely used to certify clinical NLP pipelines — the pool-level fairness
> audit, pretraining loss as a proxy for downstream utility, the LLM weak labeller, the
> architectural ablation, and the low-prevalence agreement metric — are each insensitive to the
> quantity they are credited with measuring **over a regime that standard reporting does not
> identify**. In every case the regime boundary, not the point estimate, is what the evidence
> supports. Locating that boundary requires running the instrument against a positive control,
> which here cost hours and either reversed or bounded the reported conclusion — and the field does
> not do it.

| instrument | insensitive where | positive control that located the boundary |
|---|---|---|
| pool fairness audit | **sign set by baseline disparity, not by selection (C47)** — MIMIC baseline 0.0084, gap *rises* to 0.0895; PUMS baseline 0.1181, gap *falls* to 0.0988. C29's 5.9e-16 invariance is the zero-baseline special case | planted-gap control DP 0.2797 (C30); absent from the original sweep (C24) |
| pretraining loss | the clinical task — 1.320 nats → 0.002–0.004 AUROC; gap +0.0031, p=0.690 (C37) | **AG News: 5.2 points at 99M params, 38 pooled sd, p<0.0001 (C44)** |
| LLM weak labeller | magnitude is task-specific — 15.8–23.6× clinical (C20), floor ≥7× (C23) | public-text sweep: max 3.5×, under-calls by 25% (C42). Mechanism *does* generalize (C43) |
| architectural ablation | when the semantic feature is unnecessary | **A5: an arbitrary ordinal recovers 114% of true role labels' gain** |
| low-prevalence metric | near the base rate — Q/P = 20.08 | **all four MCC intervals contain zero (C36)** — the ordering was over four noise systems |

The fifth instrument is what makes this a thesis rather than four negative results: the metrics
chapter already written is not a side derivation, it is instrument V, and C36 is its control.

**Title** (`main.tex:44`) stands, but the abstract must say **"over identifiable regimes."** The
unqualified title is now stronger than the evidence.

---

## 2. Chapter plan

**5,657 lines → ~4,050 across nine chapters plus appendices.**

| # | chapter | action | now | target |
|---|---|---|---|---|
| 1 | Introduction | rewrite from scratch, **last** | 379 | 340 |
| 2 | Background | retrim, add the new citations | 573 | 420 |
| 3 | **Instrument I — the selection audit** | **WRITE NEW** | 0 | 600 |
| 4 | **Instrument II — the pretraining proxy** | **WRITE NEW** | 0 | 380 |
| 5 | Instrument III — the weak labeller | keep, expand with C42/C43 | 410 | 430 |
| 6 | **The clinical pipeline as an instrument case** | **MERGE ch3+ch4+ch6+ch7** | 2,890 | 1,020 |
| 7 | Instrument V — the low-prevalence metric | keep, cut 40%; C36 replaces §The Systems | 562 | 330 |
| 8 | Discussion | cut 30% | 664 | 450 |
| 9 | Conclusion | retarget | 179 | 150 |
| A | `CORRECTIONS.md` verbatim | new | — | — |
| B | metric-exchange derivation | new | — | 60 |

Instrument IV lives split across Ch 4 (provenance, A5) and Ch 6 (ACCORD's falsifications) — it is
the one instrument that fired in **both** codebases, and Ch 8 should say so.

### The 46 TODOs

- **17 close on evidence in hand** — ch6:224/:657 by C19; ch7:366/:809/:818/:687 and ch8:243 by
  C16; ch8:457, ch9:202 by C36; ch5:272/:313/:365 by C42.
- **14 close by deletion** — they belong to tables being cut.
- **6 close by recomputation**, under a day. Critically **ch7:196 — the 278 vs 261 `ambiguous`
  discrepancy (C28) must resolve before either number appears.**
- **9 become stated limitations**, not TODOs.

**Rule: zero `[TODO]` markers survive.** Each is closed, deleted, or becomes a sentence beginning
*"This was not measured."* A `[TODO]` inside a list of practices prescribed to others (ch6:224,
ch6:526) is the worst artifact in the document.

---

## 3. The DAG material — §3.7 of Chapter 3, ~130 lines. Not a chapter.

Both reviews agreed. Its job is narrow: the worked instance of standing rule **R3** — *report the
specification curve, never one row* — not a finding about language and not about fairness. C25
already concluded it is not a defensible standalone result; a chapter re-inflates it.

Structure: the contested adjustment set → two DAGs drawn side by side with their estimands stated
**before any number** → the specification curve as the result (5 specs × 3 cohorts) → primary
analysis → pre-specified sensitivity at equal prominence → three diagnostics with what each does
and does not license → the envelope.

**The within-era result goes in the primary block, not sensitivity.** It is the only result that
discriminates "the pooled null is real" from "the pooled null is era mixing": CareVue only,
same specification, **OR 0.847 [0.724, 0.992], p = 0.040** against a pooled null at p = 0.13.
Immediately after it: MetaVision cannot replicate — 11 exposed of 2,271 — and this **narrows C25
explicitly, in the text, not silently.**

**The envelope must end with three sentences that are not optional:** no causal claim in either
direction; no claim of subgroup harm at all; `LANGUAGE`-missingness is an administrative indicator,
four-fifths **passes** on both actually-protected attributes (named minority 1.053, recorded
non-English 0.804), and 0.468 is reported as a raw selection rate, not a disparate-impact finding.
Without them, C24, C25 and Watkins et al. are each a reviewer's opening move.

---

## 4. Write order

**Cut before you correct, correct before you write, write the introduction last.**

Every hour propagating C16 through `ch7_evaluation.tex` is wasted if 60% of that chapter is about
to be deleted. **This reverses the current task order** — the C16 propagation must happen *after*
the merge, on 1,020 lines instead of 2,890.

| step | work | cost |
|---|---|---|
| **0** | Freeze the contribution list — one page, five instruments, each with its control and regime boundary. Every later cut is decided against it | ~4 h |
| **1** | **Delete.** Seven `\section{Proposed Figures}` blocks (20 figures are rendered); ch6:69-163 specified architecture; ch3:166-226 six partition attempts; the duplicated ethics statement; ch8:202-299 reconstructed confusion matrices | ~1 day, ~450 lines |
| **2** | **Merge ch3+ch4+ch6+ch7 → Chapter 6.** 2,890 → 1,020 | ~5 days |
| **3** | **Propagate C16 and the withdrawals** through what survives. Highest risk of error in the plan, which is why it runs on the small surface with a written checklist | ~3 days |
| **4** | **Chapter 3** — the selection audit. Largest new writing; the survey's wording governs Chapter 1 | ~2 weeks |
| **5** | **Chapter 4** — the pretraining proxy. Results unambiguous, control clean, `DESIGN.md:234-442` needs restructuring not discovery | ~1 week |
| **6** | Chapter 5 expansion — C42/C43 into an existing chapter of the right shape | ~4 days |
| **7** | Chapters 7, 8, 9 | ~1 week |
| **8** | **Chapter 1 and the abstract — last.** Writing it early guarantees writing it twice | ~4 days |
| **9** | Chapter 2 | ~3 days |

**During step 1, at no extra cost:** re-verify against PDFs the CARE-CPM signalling questions and
the Haghighathoseini chi-square sentence — C35 flags both as paraphrased by the fetch layer and
both are load-bearing in Chapter 3 — and obtain Colacci et al. 2025.

---

## 5. What to cut

**Architecture + evaluation: 1,636 → 610 lines.** Every novel component was falsified; C4's rule is
*describe the model that produced the results*, and that is short.

Survives from ch6: the three falsifications (120 lines — this is instrument IV firing), Stage 0 plus
the anisotropy control (50 — the cheapest positive control in the thesis, and it refutes its target
*positively*), experimental hygiene C5/C7/C8/C18 (40), a claimed-vs-actual table (10).
**Cut outright:** the PPC/GGHE/ATCG/CDE specification, which never ran.

Survives from ch7: no-rank-signal rebuilt on the three-way agreement (120), no operating point (40),
severity ranking **cut to 20** — ρ = −0.0001 measures an untrained module and 43 lines invites the
reader to mistake it for a finding about clinical severity — system comparison **rebuilt on C36 with
no ordering** (50), MedNLI (30), error taxonomy (50), exemplar contamination using E5's measured
65 of 137 (40), synthesis (40).

**Figures 20 → 15.** Rebuild `fig-intro-headline` — it encodes contaminated numbers and the
withdrawn "above chance" claim. `fig-threshold-surface` and `fig-spec-curve` become load-bearing in
Ch 3; `fig-scaling` and `fig-gap-trajectory` in Ch 4.

---

## 6. Risks

1. **The frame is an analogy with no fallback.** The system frame is dead (C1/C2/C4/C16/C19) and
   the fairness frame is dead (C24/C25/C28/C38). Mitigation: lead with **Chapter 4**, the AG News
   control, because it is the only case where the instrument demonstrably works elsewhere — that is
   what converts "things failed" into "failure is regime-dependent."
2. **BLOCKER — the survey has no per-paper coding table on disk at all (C49).** This is not
   "publish the table," it is "the table does not exist." The 31 papers are not named anywhere in
   the repository; the 25/4 cross-tab, the κ range and the 21 quotes have no rows behind them. The
   survey must be **re-coded with row-level output persisted** before any survey number enters a
   chapter, and the recode governs — if it does not reproduce 0/25 and 0/4, the new result stands.
   **Step 4 (Chapter 3, two weeks) cannot start until this closes.** Then adjudicate Banda, Wang
   and Haghighathoseini on the page and report strict and literal readings side by side.
3. **C44's scratch arms may be under-trained** — 3 epochs, 40k examples, no per-arm tuning, and the
   100M arm gaining only +0.009 over 10M is the tell. If the control is weak, C37 reverts to
   "an experiment too insensitive to see anything." **This is the one remaining experiment with a
   real argument for running it**, against the standing decision to close the record.
4. **Merging four chapters may destroy the traceability that is the actual credibility.** Mitigation:
   a per-number provenance table mapping every reported quantity to a committed script — cheap now,
   impossible later.
5. **C45 will be re-litigated.** Decide in advance not to manufacture a single estimate under
   pressure. This is the fifth time a claim moved when a covariate moved; the sixth would be at the
   viva.
