# From master's-scale work to a PhD dissertation — full plan of record

**Written 2026-08-07.** Supersedes the earlier short plan. Three tracks that must never be
conflated: the 34-day sprint to a submittable artifact, the applications window, and the
dissertation arc itself. Conflating a thesis-grade observation with a paper-grade claim produced
seven retractions this week; the separation below is the correction.

---

## 0. Standing rules

These are not aspirations. They are the operating discipline this project paid for.

**R1 — Every claim carries a pre-registered kill condition, written before the work starts.**
Name what result would refute it. If you cannot name one, it is not a claim.

**R2 — If a claim needs an adjustment set to survive, assume it will be retracted.** Every claim
that died this week required a covariate; every claim that lived requires arithmetic. Before
writing any new claim: *does this need a covariate?* If yes, pre-register the specification curve
alongside it, or drop it.

**R3 — Report the specification curve, never one row of it.** A single odds ratio in a contested
setting is a claim about a DAG dressed as a claim about data.

**R4 — Run the falsifier first, not last.** Every collapse this week happened because the falsifier
ran after the claim was already load-bearing. The audit-practice survey was the exception, and it
is the only leg still standing.

**R5 — Be suspicious of any review that produces a NEW positive claim.** C14 was manufactured from
a null by adding a covariate, by the same document that caught C1–C13. Reviews should kill things.

**R6 — Nothing MIMIC-derived leaves the machine.** Enforced by `scripts/sync_results.sh`:
structural exclusion on ingest, content scan on egress, abort rather than filter.

---

## 1. The dissertation

> **The instruments used to evaluate clinical NLP pipelines do not measure what they are believed
> to measure.** This is demonstrated at three independent points — the fairness audit, the
> pretraining proxy metric, and the leakage control — and a stage-decomposition protocol is
> proposed, validated, and given a theoretical account.

Four contributions. Two demonstrations, one method, one theory. C1 and C4 are independent
instances of the same phenomenon on different instruments, which is what makes it a program rather
than a finding.

---

## 2. Evidence inventory — what actually exists today

Every number independently recomputed, traceable to a committed script.

### Strong (survived adversarial review)

| finding | numbers | why it holds |
|---|---|---|
| Audits are disjoint literatures | 31 papers; of 25 auditing a model on a filtered pool, **0** audit the filter; of 4 auditing selection against a population denominator, **0** audit a model | survived a falsifier designed to kill it (C21) and blind second coding, **κ 0.737–1.000** (C27) |
| Null-attribute handling | **13 of 31** report nothing; those that do mostly **drop** the records | cleanest novelty delta |
| Proxy-metric decoupling | 7 arms × 3 seeds; **1.320 ± 0.015 nats** of pretraining separation → **0.002–0.004** AUROC; between-arm spread 0.0051 vs within-arm seed std up to 0.0079 | well-powered, 734 test positives |
| Small-encoder competitiveness | **10.6M params, random init, AUROC 0.8003 ± 0.0031** vs PubMedBERT **109M params / 3.1B words → 0.8223** | 36,504 examples, 3,730 positives |
| Rule-transfer collapse | F1 **0.9603** in-domain → **0.0654** sepsis / **0.0595** AKI | gold-independent, implementation-independent |
| LLM over-call floor | 3 local 8B models **15.8×–23.6×**; GPT-4o **7.0×**; floor **≥7×** vs a 4.0% base rate | 1,679 pairs, 0% unparseable |
| Two models at chance | ACCORD decontaminated **0.5145** [0.321, 0.703]; subject-disjoint control **0.497 ± 0.035**; DeLong **p = 0.930** | the failure is not the implementation |
| Anisotropy control | 20 unrelated non-clinical sentences at cosine **0.9113** vs **0.9176** matched clinical | refutes its target positively |

### Compromised (do not build on)

| claim | why |
|---|---|
| Real-data audit dissociation | audited classifier is 87.3% positive — parity near-guaranteed (C22); four-fifths **passes** on both protected attributes (C24) |
| Selection effect | HF **0.894** p=0.13 under correct volume spec (C17); era confounded — `LANGUAGE` missing for **60.8%** of CareVue vs **0.5%** of MetaVision (C25) |
| Synthetic sweep | no positive control: or_select 1.00 → DP gap 0.00112, or_select 0.05 → 0.00129 (C24); pre-empted by Pfohl §5 (C26) |
| Model-agnostic miscalibration | GPT-4o 7.0× vs 8B 23.9×; few-shot moves same model 7.0→13.9× (C23) |

---

## 3. The four contributions

### C1 — Audit blindness

**Claim.** A per-group fairness audit computed on a post-selection pool is invariant to the
selection rate itself. No audit at the model stage can detect that a subgroup was removed
upstream — whether the removal is discriminatory, benign, or purely mechanical. The field's
convention of dropping null-attribute records deletes precisely the population most likely to have
been removed.

**Never phrase this as "the audit missed a fairness harm."** There is no demonstrated harm. Four-
fifths is defined on raw selection rates and deliberately does not ask about cause — that is what
makes the raw ratio the correct statistic and makes C17/C25 irrelevant to the claim.

**State:** ~70%. Survey done and reliability-checked. Real-data leg compromised. Simulation
demoted to replication.

**Needed:**
1. PubMed + ACL Anthology novelty pass — **1 day, blocking**
2. Convert the purposive survey to a **pre-registered systematic review**: published search string, PRISMA flow, second coder on ≥20% — 3 weeks
3. Operating-point sweep: audit metrics as a function of positive rate across ACCORD `contra_prob`, the C19 control, and TF-IDF+LR — **1 day, laptop**
4. Mortality-cohort selection experiment: 34,018 examples, 2,975 positives, selection on *outcome-predictive* features, deliberately violating the independence assumption — **12–16h, laptop**
5. Answer Watkins et al. on four-fifths applied to a non-protected attribute — writing

**Kill condition:** a JAMIA/JBI/AMIA reporting-practice review already documents null-attribute
handling across a comparable sample. → C1's delta is gone; C4 becomes the primary paper.

**Venue:** ML4H 2026 Proceedings (Sept 10) → CHIL 2027 → JAMIA Open

---

### C2 — The stage-decomposition protocol

**Claim.** Reporting subgroup composition at every stage boundary against the source population,
plus a per-stage log-disparity decomposition, recovers disparities a conventional model-stage audit
cannot see.

**State:** not started. Currently one prescriptive sentence.

**Needed:**
1. Specify the protocol formally: stage boundaries, denominator, decomposition, and the mandatory
   preconditions — the degenerate-classifier check (C22) and a minimum-detectable-gap table before
   any parity metric is reported
2. Validate on **≥3 pipelines, at least one not your own**. This is the requirement that converts a
   critique into a method; n=1 is the current fatal limitation
3. Cost it honestly: analyst time, data access, and whether it runs when the source population is
   itself selected — MIMIC-III is an ICU cohort, so it is
4. Release as an executable artifact, not a checklist

**Kill condition:** if it recovers nothing beyond a raw composition table, it is a checklist and
must be described as one.

**Compute:** needs a cluster for multi-pipeline validation. **Not feasible on the M1.**
**Timeline:** 6–9 months, Year 1.

---

### C3 — The invariance result

**Claim.** Under S ⊥ Y | (A, X), a predictor evaluated on the selected pool satisfies equalized FPR
regardless of how strongly S depends on A. Plus a characterization of where it breaks.

**State:** not started. The current simulation assumes the condition by fiat, which is why it has
no contrast.

**Needed:**
1. State it as a proposition and prove it
2. Characterize the boundary as a function of violation strength
3. Rebuild the simulation with the assumption **deliberately violated** — group associated with
   features and outcome, selection depending on documentation volume
4. Position against Bareinboim–Pearl recoverability and Pfohl et al. explicitly

**Claim the instantiation, not the structure.** Claiming new theory here is how C1's first version
died.

**Kill condition:** if it is a direct corollary of published recoverability theory, report it as
such and fold it into C2 as the protocol's justification.

**Compute:** trivial. **Timeline:** 4–6 months, Year 1–2.

---

### C4 — Proxy-metric decoupling

**Claim.** Pretraining-loss improvements do not predict downstream utility, and a small
from-scratch encoder is competitive with a large pretrained one on clinical multi-note tasks.

**State:** data exists, unwritten. This is the only positive result in the body of work.

**Needed:**
1. **PubMedBERT at 3 seeds under the matched protocol — 2 days, blocking.** `results/task_runs.json`
   shows all three of your arms at 3 seeds and PubMedBERT at **one** (`concat_s0`). C7 retracts a
   claim for exactly this
2. Non-clinical replication: sweep an Ollama labeller's marginal over an imbalanced text set, show
   the student tracks the marginal not the conditional — ~3 nights, laptop, no credentialed data
3. Scaling: does competitiveness hold at 30M, 100M? **Needs a cluster**

**Kill condition:** if PubMedBERT's 3-seed mean exceeds random-init by more than the pooled seed
std (~0.008), the efficiency claim weakens and this becomes a pure proxy-metric paper. Still
publishable, different abstract. If the gap widens monotonically with scale, the efficiency claim
is a small-data artifact — report it.

**Venue:** ML4H 2026 Findings (Sept 10) → a full version at CHIL or ACL Findings

---

## 4. TRACK A — the 34-day sprint

**Objective: one submittable artifact before applications.** Not a dissertation chapter.

| dates | task | hours | gate |
|---|---|---|---|
| **Aug 8** | PubMed + ACL Anthology novelty pass | 8 | **can kill C1 → pivot to C4 as primary** |
| **Aug 9–10** | PubMedBERT seeds 1, 2, matched protocol | 16 | **can weaken C4's efficiency claim** |
| Aug 11 | Operating-point sweep, 3 scorers | 8 | repairs C1's real-data leg or kills it |
| Aug 12–13 | Compliance: Drive cleanup, `ch3_data.tex` contradiction, strip HF-API row | 8 | **blocking for any submission** |
| Aug 14–24 | Draft Paper A (8pp) | 60 | |
| Aug 25–29 | Draft Paper B (4pp) | 24 | |
| Aug 30–Sept 5 | Internal review; answer Watkins and Biswas & Rajan in the body | 24 | |
| Sept 6–9 | Revise, format, data-availability statement | 16 | |
| **Sept 10** | **Submit both** | | **hard deadline** |

### Paper A → ML4H Proceedings (8pp, archival PMLR)
Lead with the survey — it is the only leg that strengthened under review. Demote the sweep to
replication of Pfohl §5. Report the decisive cell under **both** strict and literal readings with
the per-paper coding table and quotes; a bare "0 of 25" would be the failure mode the paper is
about. Disclose the breached recall gap (0.25 at n_pos=4) in the table, not the appendix.

### Paper B → ML4H Findings (4pp, non-archival)
**C4, not the clinical negative result.** C4 has 734 test positives and a positive finding; the
clinical negative has 12 and is underpowered by its own admission (MDE 0.739). For a PhD applicant,
leading with something that works matters.

**Do not merge A and B.** B is well-powered; A's real-data leg is not.

---

## 5. TRACK B — applications window, Sept 11 – Dec 15

**Start no new experiment.** This window is protected.

- **Thesis propagation** — C16 through Chapters 7–9 (every AUROC there is the contaminated run:
  0.460 → **0.5145**); the ~30 untraceable numbers, clustered in the MedNLI and architecture tables
  including a parameter count (22,446,855) that exists nowhere on disk; the 15 mismatches; the two
  open `[TODO]`s in Chapter 6 that sit inside a list of practices prescribed to others
- **Reorganize** to seven chapters, ~3,150 lines. The audit-dissociation chapter does not exist yet
  and is where nearly all new writing goes
- `CORRECTIONS.md` verbatim as Appendix A — it is the credibility instrument, do not paraphrase
- ML4H rebuttal ~early Oct, decision ~late Oct. Cite status accurately
- Fellowship applications per the funding map
- **SoP:** lead with the program, not the corrections. One sentence referencing the correction
  record; have it ready for interviews, not in the opening paragraph

---

## 6. TRACK C — the dissertation arc

| period | work | needs |
|---|---|---|
| **Year 1 H1** | C1 systematic review; C3 proposition and proof; mortality selection experiment | laptop |
| **Year 1 H2** | C2 protocol specified and built; first external pipeline validation | cluster |
| **Year 2 H1** | C2 validation across ≥3 pipelines; C4 scaling to 30M/100M | cluster |
| **Year 2 H2** | C3 boundary characterization; C4 non-clinical replication at scale | cluster |
| **Year 3** | External/prospective validation; dissertation writing; job market | |

### What must be true for this to be a PhD
1. **Three independent contributions**, not one claim stated four ways
2. **At least one thing that works** — C4 is it. Without it this is a critique portfolio
3. **A method validated beyond n=1** — C2
4. **External validation** — one pipeline, one dataset, one attribute is the current ceiling

---

## 7. Risk register

| risk | likelihood | impact | mitigation |
|---|---|---|---|
| PubMed pass kills C1's delta | medium | high | C4 becomes primary; it is independent |
| PubMedBERT 3 seeds weakens C4 | low–medium | medium | proxy-metric decoupling survives without the efficiency angle |
| ML4H rejects both | medium | medium | CHIL Feb 2027; non-archival Findings burns nothing |
| No cluster access in Year 1 | medium | **high** | C2 and C3 both blocked; C1 and C4 laptop work fills the gap |
| Reviewer recognizes the withdrawn NeurIPS submission | high | low | different paper, corrections disclosed openly |
| A JAMIA reporting review surfaces late | medium | high | run the PubMed pass **first** — this is R4 |
| Compliance item surfaces at review | low | **very high** | Track A Aug 12–13 is blocking |
| Eighth retraction | medium | medium | R2: if it needs a covariate, pre-register the curve |

---

## 8. Compute and infrastructure

| need | today | required |
|---|---|---|
| C1 experiments | M1 sufficient | — |
| C4 seeds and replication | M1 sufficient | — |
| C4 scaling to 30M/100M | **blocked** | ~4× A100-days |
| C2 multi-pipeline validation | **blocked** | cluster + access to third-party pipelines |
| C3 theory and simulation | M1 sufficient | — |

Roughly half the dissertation is laptop-feasible. That half should be finished before a program
starts, which is the strongest possible application material.

---

## 9. Decision gates

**Gate 1 — Aug 8, PubMed pass.** Clean → proceed with A as primary. Pre-empted → C4 becomes
primary, A drops to Findings, and C1 is reframed as a replication with a reliability contribution.

**Gate 2 — Aug 10, PubMedBERT seeds.** Random-init within seed std → efficiency claim holds.
PubMedBERT clearly ahead → C4 is proxy-metric decoupling only.

**Gate 3 — Sept 10.** Submitted, or explicitly deferred to CHIL. **Do not slip into October** —
that window belongs to applications.

**Gate 4 — late Oct, ML4H decision.** Accept → cite it. Reject → revise for CHIL; applications
proceed either way on "under review."

---

## 10. Honest statement of the gap

Today, against a PhD bar, this is **one chapter of four**, with a second chapter's data collected
and unwritten. The rigor apparatus — falsification design, correction record, reliability
checking — is already at PhD level and is the hardest part to teach. What is missing is scale, a
method, and something that works.

C2 and C3 both require compute this machine does not have. That is the honest reason this is a
plan for a PhD program rather than a plan for the next four months.

Two claims survived falsifiers designed to kill them: the ≥7× LLM over-call floor and the survey
cross-tab. **Structure everything so survival-under-falsification is visible** — the loudest signal
this body of work currently sends is destruction, and that is not the signal to send in December.

---

*Advisor matching, funding map, and the full C2/C3 experiment matrix are being compiled separately
and will be appended as §11–13.*
