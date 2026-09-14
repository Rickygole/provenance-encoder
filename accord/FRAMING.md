# The frame — diagnostic case study

**Chosen 2026-08-10. Re-scoped 2026-08-11 to the Morgan State MS format.** This supersedes the
five-instrument frame (dead: C60), the retroactive-control frame (dead: R4), and the self-audit
frame (dead: R4). Frozen once the advisor signs `CONTRIBUTION.md`; changes require an external
signature — see `REVISION_LOG.md`.

> **This is a Master of Science thesis at Morgan State University, not a doctoral dissertation.**
> That materially lowers the bar this frame has to clear, and it retires the question that consumed
> two days of novelty checking. **At MS level, execution and rigour ARE the contribution** — a
> novel mechanism is not required, and the fact that four framings died to literature searches is
> not a problem to solve but a section of the Discussion. The diagnostic frame is comfortably above
> bar rather than borderline. Format spec: `FORMAT.md`.

---

## 1. The thesis statement

> Detecting dangerous medication contradictions between nursing notes and discharge summaries is a
> real patient-safety problem, and the standard pipeline for it — rule-based candidate mining, LLM
> weak supervision, and a purpose-built neural detector — fails. This thesis establishes **where it
> fails and why**, by running at each stage the control that would distinguish a real finding from
> an artifact of the measurement. Every stage failed a control. The controls cost hours. None of
> them is novel, and none of them was run before this work.

Three properties make this defensible where the previous three framings were not:

- **The failure is the subject, not an embarrassment to route around.** ACCORD does not work. Under
  every previous frame that was an awkward fact; here it is the research question.
- **The methodology is the method, not the claim.** Positive controls, degeneracy preconditions and
  sign provenance are attributed to the work that owns them (Lipsitch 2010, Adebayo 2018, Hewitt &
  Liang 2019, Feinstein & Cicchetti 1990, Egami 2023, Lourie 2025). **The thesis does not claim
  them.** It claims what they revealed about this pipeline.
- **It cannot be killed by a literature search.** Every demolition of 2026-08-10 — TESSERA v2
  out-scaling C62 at 395 runs, Nuijten & Wicherts out-measuring the gate claim, five literatures
  owning the self-audit proposition — leaves this frame untouched, because it never claimed those
  results were new.

---

## 2. The research questions

**RQ1 — Candidate generation.** Can rule-based mining over paired clinical notes yield a candidate
pool with usable precision? *Answer: no. The pool base rate is 0.0474 on the 253 non-ambiguous pairs, and of the rule 1 and
2 candidates whose exemplar could be resolved, 65 of 137 were justified by the wrong sentence.*

**RQ2 — Weak supervision.** Can an LLM weak labeller substitute for expert annotation at this base
rate? *Answer: no, and the failure has a floor. 15.8–23.6× over-calling across three local 8B
models, ≥7× including GPT-4o.*

**RQ3 — Architecture.** Do the proposed architectural components contribute what they are credited
with? *Answer: no, and the controls say so positively rather than by absence of evidence.*

**RQ4 — Evaluation.** Is the measured failure a property of the task or of the instruments used to
measure it? *Answer: of the task, and this is the question the whole thesis turns on — it required
a positive control to answer, and the control was run.*

**RQ5 — Practice.** Does the field run the controls that would have caught these failures?
*Answer: rarely. 4 of 75 papers, strictly; 14 of 75 generously.*

RQ4 is load-bearing. Without it, every negative result reduces to "the experiment was too small."
With it, the nulls are measurements.

---

## 3. The chapter map — six chapters, per the MSU template

**Six chapters.** The official template mandates five (Introduction, Literature Review,
Methodology, Results, Summary & Conclusion); the local CS precedent splits Discussion out to make
six, and both colleague theses do this. **The nine-chapter plan below is retired** — the five
research questions become *sections* of Methodology and Results, not chapters. The argument is
unchanged; only the container is.

**Scale target: ~1,200–1,600 lines of chapter LaTeX, double-spaced.** The colleague comparator is
481 lines. The current draft is 5,810 single-spaced. **Deletion is now the dominant task by a wide
margin**, and the earlier ~3,300-line target was set against the wrong norm.

| MSU chapter | absorbs | RQ |
|---|---|---|
| **1. Introduction** | problem, four RQs, contributions, thesis structure | — |
| **2. Literature Review** | contradiction detection, clinical NLP evaluation, class imbalance, **and the control literature this thesis applies** (Lipsitch, Adebayo, Hewitt & Liang, Feinstein & Cicchetti, Egami, Lourie, TESSERA) | — |
| **3. Methodology** | MIMIC-III cohort, the 3-role→2-role retreat, four mining rules, gold standard, subject-disjoint splits, the detector, **and the control designed for each stage — stated as conditions before any result** | all |
| **4. Results** | candidate precision + exemplar defect (RQ1); over-calling and the ≥7× floor (RQ2); the three falsifications and the anisotropy control (RQ3); three configurations at chance + the AG News positive control (RQ4); the 75-paper survey (RQ5); the base-rate metrics | all |
| **5. Discussion** | what each control changed; the unexplained selection-channel anomaly reported *as* unexplained; sign provenance; limitations; threats to validity | — |
| **6. Summary & Conclusion** | findings restated, future work | — |
| References | APA 7, hanging indent | — |
| Appendix A | `CORRECTIONS.md` | — |
| Appendix B | `REVISION_LOG.md` | — |
| Appendix C | per-number provenance table | — |

### Retired nine-chapter map, kept for the section-level content it specifies

Every substantive section follows one shape, and the shape is the thesis:

> **the reading a competent paper would report → the control that was omitted, stated as a
> condition before any number → the control's result → what survived → what it cost**

| # | chapter | contains | target | status |
|---|---|---|---|---|
| 1 | **The problem and the failure** | Clinical motivation, the four RQs, the headline stated once and correctly, the diagnostic shape | 300 | rewrite **last** |
| 2 | **Background** | Contradiction detection, clinical NLP evaluation, class imbalance, and the control literature this thesis *applies* — Lipsitch, Adebayo, Hewitt & Liang, Feinstein & Cicchetti, Egami, Lourie, TESSERA | 380 | retrim + ~80 new |
| 3 | **The pipeline and the cohort** | MIMIC-III, ICD-9 extraction, the 3-role→2-role retreat, four mining rules, gold standard, subject-disjoint splits, hygiene. **Zero evaluative claims** | 430 | merge ch3+ch4 |
| 4 | **RQ1 — candidate generation** | Per-rule precision; the `.iloc[0]` exemplar defect (**65 wrong of 137 resolved, 0.4745**; 28 unresolved. The 41.4% figure previously carried here is untraceable, see REVISION_LOG R8); the rebuild; why the 4% precision is an upper bound on annotation validity | 380 | reorganize |
| 5 | **RQ2 — weak supervision** | 15.8–23.6× over-calling, `weak_confidence` a deterministic function of the label, GPT-4o at 7.0× establishing a ≥7× floor; **C42's public-text control: max 3.5×, so the magnitude is task-specific and the mechanism generalizes (C43)** | 380 | reorganize + C42/C43 |
| 6 | **RQ3 — architecture under control** | The three falsifications (authority-weight init artifact, rule-head leakage 0.9603→0.0654/0.0595, symmetric ACNE beating full); the anisotropy control (0.9113 vs 0.9176) refuting its target **positively**; the provenance encoder's 7-arm comparison as the same instrument in a second system | 430 | merge ch6 + `DESIGN.md` §6c–6d |
| 7 | **RQ4 — is the null real?** | Three configurations at chance (0.5145 [0.321,0.703]; subject-disjoint 0.497±0.035; DeLong p=0.930); TF-IDF at 0.576 beating both neural models; MDE 0.7382 at 12 positives; **the positive control: 4.53 points on AG News at 24 epochs, SD ≤0.003, pre-registered rule, versus +0.0031 (p=0.690) clinically** | 400 | **write new** |
| 8 | **Metrics that report the base rate** | Prevalence dependence of balanced accuracy / PABAK / MCC; C36's four intervals all containing zero, so the reported ordering was over noise; C22's 87.3%-positive classifier; **sign provenance: `abs()` storage licenses no directional claim — 0.0 across 9,177 fields while 7 of 14 slopes flip** | 340 | cut 40% + ~90 new |
| 9 | **What the controls changed, and what practice does** | The survey (4/75, 14/75, κ 0.855 permissive, undefined strict — stated); the cost table; the unexplained selection-channel anomaly reported *as* unexplained; limitations; future work | 400 | merge ch9+ch10 |
| A | `CORRECTIONS.md` verbatim | 62 entries | — | Appendix |
| B | `REVISION_LOG.md` | post-freeze revisions | — | Appendix |
| C | Per-number provenance table | every quantity → committed script | 100 | build during the merge |

---

## 4. What each chapter must NOT claim

This list exists because every previous frame died by over-claiming, and the failures are on record.

- **Ch 4:** the exemplar defect is a methods finding about candidate-generation pipelines. It is
  **not** evidence the 4% precision is wrong — it makes 4% an *upper bound on annotation validity*.
- **Ch 5:** the over-calling magnitude does **not** generalize. C42 measured 3.5% max on public text.
  The *mechanism* generalizes (C43); the number does not.
- **Ch 6:** the provenance encoder result is **not** a novel architecture finding. Hewitt & Liang
  (EMNLP 2019, Best Paper runner-up) own the control-task design *and* its conclusion. Cite them as
  the method applied. The hard block mask reproduces Zhao et al. (ACL 2024); it does not improve it.
- **Ch 7:** the evaluation cannot establish the task is unsolvable — MDE 0.7382 at 12 positives. It
  establishes that **this pipeline** does not solve it and that the instrument could have seen a
  4.53-point effect. C62 is a **positive control, not a contribution**: TESSERA v2 (arXiv:2607.03949)
  reports the general result at 395 runs.
- **Ch 8:** the metric critique is 36 years old (Feinstein & Cicchetti 1990). Apply it; do not claim
  it.
- **Ch 9:** the survey is **not** a prevalence estimate for clinical ML — enriched corpus, PMC-plus-
  arXiv coverage, 459 screened-in papers never retrieved. κ is **undefined** on the strict cell; do
  not report 100%. No subgroup harm is demonstrated anywhere in this work. **No mechanism** is
  offered for the selection-channel anomaly.
- **Everywhere:** no rate is claimed from the correction record (R1). No claim that executable gates
  beat prose checklists (R2, and Nuijten & Wicherts 2023 out-measures it).

---

## 5. The title

Current title asserts the claim C46 refuted and must go.

> **Why Cross-Note Medication Contradiction Detection Fails: A Failure Analysis of a
> Clinical NLP Pipeline**

It names the domain, the negative result, and the method, and it promises nothing the evidence does
not deliver.

---

## 6. The viva defence

Eight questions an examiner will actually ask, with the answer and where it lives.

**"Isn't this just a negative result?"**
It is a *diagnosed* negative result. Each stage's failure is separated from measurement artifact by
a control, and RQ4 shows the instrument could resolve a 4.53-point effect where one existed. The
contribution is the diagnosis, not the disappointment.

**"What is novel?"**
As mechanisms, little, and the thesis says so on page one. What is new is the measurement on this
pipeline: a measured null rather than an underpowered one; a survey with released rows showing 4 of
75 papers run the check that would have caught the upstream defect; and a reproducible,
unexplained anomaly in how a pool audit responds to selection channel. Ch 2 attributes every method
to its owner.

**"Your system doesn't work. Why is that a thesis?"**
Because the field's published pipelines share its structure, and this work shows the structure fails
for reasons that are measurable rather than incidental — and that the checks distinguishing those
reasons are almost never run (Ch 9).

**"Only twelve positives."**
Stated in the abstract, with MDE 0.7382 at 80% power. The thesis never claims the task is
unsolvable. Ch 7's positive control is what converts "we saw nothing" into "the instrument would
have seen a 4.53-point effect and saw +0.0031."

**"Your survey is n=75 on an enriched corpus."**
Conceded in the chapter. The enrichment runs *against* the finding — a corpus over-sampling
fairness-forward work still yields 4/75 strict. Both readings are reported side by side, κ is
undefined on the strict cell and stated as such, and all 75 rows with 502 verified quotes are
released.

**"You corrected yourself 62 times. Why should I trust the 63rd number?"**
Appendix A is the record, Appendix C is a per-number provenance table mapping every reported
quantity to a committed script, and the numbers were audited by a process that found 15 mismatches
including one in the introduction that inverted the central claim — before submission rather than
after. The honest answer to "did your own audit fail?" is yes, four documented times, and the
retraction of my own summary statistic (R1) is in the record.

**"Why should I believe the architecture components don't work?"**
Because the controls fired positively, not by absence of evidence: an arbitrary ordinal recovers
114% of true role labels' gain, misaligned blocks recover 47% where aligned recover 100%, and
symmetric ACNE beats the full model. Ch 6.

**"What would change your conclusions?"**
Re-annotation against corrected exemplars; a cohort with more than 12 positives; and the
selection-channel anomaly explained. Named in Ch 9 as future work, not hedges.

---

## 7. What is cut

~840 lines before merges: seven `\section{Proposed Figures}` blocks (20 figures are rendered); the
specified-but-never-built architecture (C4, C54); six partition attempts reduced to a statement of
the retreat; the duplicated ethics statement; reconstructed confusion matrices; severity ranking cut
to 20 lines (ρ = −0.0001 on an untrained module invites a misreading); the class-imbalance methods
survey; MedNLI reduced to one row.

**All 46 TODO markers die** — closed on evidence, deleted with their tables, or rewritten as *"This
was not measured."* Two of them (`ch6:224`, `ch6:526`) sit inside practices prescribed to others and
go first.

---

## 8. Order of work

**Blocking, before any prose (~2 days):** the six items in R7 — the `ch1:240` inversion (**fixed**),
the un-propagated "above chance" retraction, the top-15 zero/one split, the n=299 vs n=253
denominators, the C16-superseded α values in seven places, and re-verifying C54 against the Drive
mount after R6.

Then: **delete** (1 day) → **merge ch3+ch4 and ch6+ch7** (5 days) → **propagate corrections on the
merged surface** (3 days) → **Ch 7, then 4, 5, 6, 8, 9** (~4 weeks) → **Ch 2** (3 days) → **Ch 1,
abstract and title last** (4 days).

Write **Chapter 7 first**. It is the chapter RQ4 turns on, its evidence is complete and verified,
and it is the one an examiner will test hardest.

**The scoreboard is draft lines, not findings.** A day that produces no chapter prose failed,
whatever it discovered.
