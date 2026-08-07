# From master's-scale work to a PhD dissertation — plan of record

**Written 2026-08-07.** Two tracks that must not be conflated: what happens in the next four
months while applications are live, and what the dissertation itself would be. Conflating a
thesis-grade observation with a paper-grade claim is what produced seven retractions this week;
the separation below is deliberate.

**Design rule, applied to every item.** Each contribution carries a **pre-registered kill
condition** written before the work starts. If a claim needs an adjustment set to survive, assume
it will be retracted. Every claim that died this week required a covariate; every claim that lived
requires arithmetic.

---

## The dissertation, stated once

> The instruments used to evaluate clinical NLP pipelines do not measure what they are believed to
> measure. This is demonstrated at three points — the fairness audit, the pretraining proxy metric,
> and the leakage control — and a stage-decomposition protocol is proposed and validated as the fix.

Four contributions. Two are demonstrations, one is the method, one is the theory.

| # | Contribution | State |
|---|---|---|
| **C1** | **Audit blindness** — pool audits cannot see upstream selection; 31-paper survey; null-attribute handling | ~70% |
| **C2** | **The protocol** — stage-decomposition audit, built and validated across ≥3 pipelines | not started |
| **C3** | **The theory** — prove the invariance, characterize its boundary | not started |
| **C4** | **Proxy-metric decoupling** — 1.32 nats of pretraining signal → 0.0004 AUROC; small-encoder competitiveness at 734 test positives | data exists, unwritten |

C1 and C4 are independent demonstrations of the same thesis. C2 is the fix. C3 is why it works.

---

# TRACK A — now to 2026-09-10 (34 days)

Objective: **one submittable artifact before applications.** Not a dissertation chapter.

### A1. PubMed + ACL Anthology novelty pass — **do first, 1 day**
The existing search covered arXiv and OpenReview only. C1's framing lives partly in
PubMed-indexed venues.
- **Kill condition:** a JAMIA/JBI/AMIA reporting-practice review already documents null-attribute
  handling across a comparable sample. If found, C1's cleanest delta is gone and Track A pivots to
  C4 as the primary paper.
- Everything downstream is conditional on this. It can only move the assessment down, and it costs
  a day.

### A2. PubMedBERT with 3 seeds, matched protocol — **2 days compute**
`results/task_runs.json` shows all three of your arms at 3 seeds each and **PubMedBERT at one seed
(`concat_s0`, 0.8223)**. The comparator in the headline table is single-seed; your own C7 retracts
a claim for exactly this.
- Run `concat_s1`, `concat_s2` under the identical matched protocol and per-segment token cap.
- **Kill condition:** if PubMedBERT's 3-seed mean exceeds your random-init arm by more than the
  pooled seed std (~0.008), the "97.4% at a tenth the parameters" claim weakens and C4 becomes a
  pure proxy-metric-decoupling paper without the efficiency angle. Still publishable, different
  abstract.
- **This is the single highest-value experiment available.** It unlocks the only positive result in
  the body of work.

### A3. Write two papers, submit both to ML4H 2026 — **3 weeks**

**Paper A → Proceedings (8pp, archival PMLR).** Audit blindness.
- Lead with the survey, not the MIMIC cohort. The survey survived a falsifier (C21) and blind
  second coding (κ 0.737–1.000, C27); the real-data leg did not (C22, C24, C25).
- Demote the synthetic sweep to **replication of Pfohl et al. §5**, which already ran the S←X,
  S←Y, S←{Y,A} conditions. Do not present it as evidence (C26).
- Answer two attacks in the body, not the limitations: Watkins et al. on four-fifths applied to a
  non-protected attribute, and Biswas & Rajan 2021 on stage-level fairness at n=37 pipelines.
- Report the decisive cell under **both** strict and literal readings with the per-paper coding
  table and quotes. A bare "0 of 25" would be the failure mode the paper is about.
- Disclose the breached recall gap (0.25 at n_pos=4) in the table, not just the appendix.

**Paper B → Findings (4pp, non-archival).** Proxy-metric decoupling.
- **Change from the earlier plan:** Paper B is C4, not the clinical negative result. C4 is a
  positive result with 734 test positives; the clinical negative has 12 and is underpowered by its
  own admission (MDE 0.739).
- Content: 1.320 ± 0.015 nats of pretraining-loss separation across 7 arms × 3 seeds producing
  0.002–0.004 downstream AUROC; three pretraining arms statistically indistinguishable with seed
  noise exceeding the effect; random-init highest mean.

**Do not merge A and B.** B's evaluation is well-powered; A's real-data leg is not. Merging imports
the weakness in the wrong direction.

### A4. Compliance — **half a day, non-negotiable before any submission**
- Move the two note-text CSVs and raw MIMIC tables off Drive to `mimic_local_only/`.
- Fix `ch3_data.tex:595`, which contradicts `:604` seven lines apart.
- Strip the HF-API Llama row from any submitted table (record-only, C20).

**Gate at 2026-09-10:** submitted, or explicitly deferred to CHIL (Feb 2027). Do not slip into
October — that window belongs to applications.

---

# TRACK B — 2026-09-11 to 2026-12-15

**Applications only.** Start no new experiment.

- Thesis propagation: C16 through Chapters 7–9 (every AUROC there is the contaminated run; 0.460 →
  0.5145), the ~30 untraceable numbers, the 15 mismatches, the two open `[TODO]`s in Chapter 6.
- Reorganize to seven chapters, ~3,150 lines. The audit-dissociation chapter does not exist yet.
- `CORRECTIONS.md` verbatim as Appendix A.
- ML4H rebuttal (~early Oct), decision (~late Oct). Cite status accurately.
- SoP: lead with the program, not the corrections. Reference the corrections record in one
  sentence and have it ready for interviews.

---

# TRACK C — the dissertation arc (18–30 months, inside a PhD program)

### C2 — the protocol, built and validated (6–9 months)
The prescription currently exists as a sentence. Make it an artifact.
- Specify the stage-decomposition audit: report subgroup composition at **every** stage boundary
  against the source population, plus the log-disparity decomposition per stage.
- Validate on **≥3 pipelines, at least one not your own** — this is the requirement that converts a
  critique into a method, and n=1 is the current fatal limitation.
- Include the degenerate-classifier check (C22) and a minimum-detectable-gap table as mandatory
  preconditions before any parity metric is reported.
- **Kill condition:** if the protocol recovers nothing beyond a raw composition table, it is a
  checklist, not a method, and must be described as one.
- **Needs a cluster** for the multi-pipeline validation. Not feasible on the M1.

### C3 — the theory (4–6 months)
- Prove: under S ⊥ Y | (A, X), a predictor on the selected pool satisfies equalized FPR regardless
  of how strongly S depends on A. Characterize the boundary where it fails.
- This converts the simulation from a restatement of its own setup into a numerical illustration of
  a theorem — the objection C24 and C26 both landed.
- Position against Bareinboim–Pearl recoverability and Pfohl et al. explicitly. **Claim the
  instantiation, not the structure.** Claiming new theory here is how C1 died.
- **Kill condition:** if the result is a direct corollary of published recoverability theory, report
  it as such and fold it into C2 as the protocol's justification.

### C4 — the positive result, fully developed (3–4 months)
Beyond A2's three seeds:
- Non-clinical replication: sweep an Ollama labeller's marginal over an imbalanced text set, show
  the student tracks the marginal not the conditional. ~3 nights local, no credentialed data.
- Scaling: does small-encoder competitiveness hold at 30M, 100M? **Needs a cluster.**
- **Kill condition:** if the gap to PubMedBERT widens monotonically with scale, the efficiency claim
  is a small-data artifact. Report it.

### C1 completion (3 months, after the ML4H cycle)
- Extend the survey to a **pre-registered systematic review** with a published search string and a
  second coder on ≥20%, replacing the purposive sample.
- The mortality-cohort selection experiment: 34,018 examples, 2,975 positives, selection on
  *outcome-predictive* features — deliberately violating the independence assumption the current
  simulation assumes by fiat. 12–16 hours, runs on the M1.

### What must be true for this to be a PhD
1. **Three independent contributions**, not one claim stated four ways.
2. **At least one thing that works.** C4 is it. Without it this is a critique portfolio.
3. **A method, validated beyond n=1.** C2.
4. **External validation** — one pipeline, one dataset, one attribute is the current ceiling.

---

## Sequencing summary

| when | what | gate |
|---|---|---|
| Aug 8 | PubMed/ACL pass | can kill C1's delta → pivot to C4 |
| Aug 9–10 | PubMedBERT 3 seeds | can weaken C4's efficiency claim |
| Aug 11–31 | draft both papers | |
| Sept 1–10 | internal review, **submit ML4H** | hard deadline |
| Sept–Dec | applications, thesis propagation | **no new experiments** |
| Year 1 | C2 protocol, C1 systematic review | needs cluster |
| Year 2 | C3 theory, C4 scaling | needs cluster |
| Year 3 | external validation, dissertation | |

## Honest statement of the gap

Today, against a PhD bar, this is **one chapter of four**, with a second chapter's data collected
and unwritten. The rigor apparatus — falsification design, correction record, reliability
checking — is already at PhD level and is the hardest part to teach. What is missing is scale,
a method, and something that works.

C2 and C3 both require compute this machine does not have. That is the honest reason this is a
plan for a PhD program rather than a plan for the next four months.
