# Provenance-Aware Clinical Encoder — Design v2

Supersedes the case-law design. v1 was killed by a proof, not an opinion; this document
fixes the specific defect and records why each choice exists.

## 1. The defect in v1

Softmax is shift-invariant: `softmax(s + c) = softmax(s)`.

If every token in a sequence carries the same provenance `r`, then the authority bias
`A[r_i, r_j] = A[r, r]` is a single constant added to every entry of the logit row. It
contributes **exactly zero** — not "a little." The same argument kills a document-level
time feature.

Every benchmark v1 named is single-document: ECtHR A/B, SCOTUS, EUR-LEX, LEDGAR,
UNFAIR-ToS. CaseHOLD is worse — it is *constructed* by masking the citation, which is the
only place the cited court's provenance would appear.

So in v1 the mechanism fired during pretraining and was mathematically switched off at
every point of measurement. What remained at test time was a vanilla transformer with 56
content channels per head instead of 64 — a 12.5% capacity cut paid for a dead mechanism.

**Root cause:** the case-law pivot was made to escape the MIMIC DUA, and it landed in a
domain whose benchmarks are single-document.

## 2. The fix

### 2.1 Evaluate where the mechanism can fire

The input unit must contain **≥2 distinct provenances in one sequence**. Clinical note
bundles are natively this: nursing note + discharge summary + physician note, different
authors, different institutional standing, different timestamps. `A[r_i, r_j]` then has
distinct row and column indices and is not a constant shift.

Going local (Ollama, no egress) removes the reason for the pivot. Clinical is both the
compliant substrate and the only one where the architecture is non-trivial.

### 2.2 Row-center `A` — for identifiability, NOT for inertness

**Correction to an earlier version of this document.** Row-centering does *not* fix
inertness. On a single-provenance sequence, `A_c[r,r]` is still one constant across the
entire matrix, and a constant added to every logit is still a no-op. The only cure for
inertness is a genuinely multi-provenance **input**. `tests/test_invariants.py::
test_single_provenance_bias_is_inert` asserts exactly this, with a non-trivial `A`.

What row-centering *does* buy is identifiability. `A` and `A + c·1ᵀ` are the same function
under softmax, so an uncentered spectrum is not a property of the learned model — two runs
converging to identical behaviour can report different spectra. Enforce:

```
A ← A − rowmean(A)          # every forward pass, not a regularizer
```

and report the spectrum of the centered matrix only. `test_row_shift_is_a_gauge_symmetry`
verifies the symmetry holds.

### 2.3 Drop the carved-channel implementation

`U ∈ R^{8×4}` + `V ∈ R^{8×4}` is 64 parameters — **identical** to an unconstrained
`A ∈ R^{8×8}` — while being strictly less expressive (rank ≤ 4 of a possible 8) and
costing 8 of 64 content channels per head. It is dominated on every axis.

The 58× penalty that motivated carving was an MPS kernel-fusion artifact. Pretraining does
not run on MPS. Use a plain additive `K×K` table gathered per (layer, head) — which is what
T5 relative-position bias does and is well supported everywhere.

Also: if authority is ordinal in the natural sense `A[i,j] = g(rank_i − rank_j)`, then `A`
is Toeplitz and generically **full rank**. A rank-4 cap cannot represent the hypothesis it
was supposed to test.

### 2.4 Temporal bias must be monotone

A 4-dim RoPE slice is a sum of two cosines — periodic, so it wraps and spuriously boosts
documents at particular log-age ratios. Precedential and clinical staleness are monotone.
Use an ALiBi-style monotone bias with a learned per-head slope over `Δt`, and report the
learned function plotted over the observed range.

## 3. The baselines that decide whether there is a paper

Run these **before** any architecture work. Any one can end the project cheaply.

| Arm | Description | If it ties A4 |
|---|---|---|
| **A1** | vanilla encoder, no provenance | — |
| **A2** | vanilla + provenance as literal prefix tokens `[NURSING] [2130-05-12]` | **no architecture paper** — 6 vocab entries did it |
| **A3** | vanilla + additive `K×K` logit bias, no channel carve | the carve was never the contribution |
| **A4** | full proposed model | — |

Plus two cheap pre-checks:
- **Linear probe:** predict note type from role-marker-stripped text with a frozen encoder
  + logistic regression. If accuracy > 95%, the provenance channel carries no bits the text
  doesn't already have, and a compute-matched study will find nothing.
- **Capacity-matched null:** 56 content + 8 semantically-free learned channels. Without it,
  every internal control is measured against a handicapped reference.

## 4. Positioning

There is **no established multi-provenance clinical benchmark.** That is a finding, not an
obstacle. The honest paper is therefore *a task plus an architecture*, and the architecture
is also evaluated on established single-document clinical benchmarks (n2c2 2018 ADE,
MedNLI) to show no regression — which answers the reviewer who demanded established-setting
evaluation.

Do not claim "competes with BERT/PubMedBERT." Claim what is measured: at scale X on corpus
Y, provenance conditioning changes task Z by W, with these controls.

## 5. Compliance by construction

| Stage | Data | Location | Egress |
|---|---|---|---|
| Pretrain | PubMed abstracts / PMC OA (CC0 + CC BY only) | cloud or local | open data, no restriction |
| Fine-tune + evaluate | MIMIC note bundles | **local only** | **none** |
| LLM baselines | MIMIC prompts | **Ollama on localhost** | **none** |

No MIMIC byte crosses a network. This replaces the GPT-4o API calls, which PhysioNet's
published guidance identifies as prohibited under the credentialed DUA.

Retain a per-document license field through the whole pipeline so the NOTICE file can be
regenerated and provenance proven.

## 6. Errors from ACCORD that this design must not repeat

Each is a concrete, checkable rule.

| ACCORD defect | Evidence | Rule |
|---|---|---|
| `authority_scale` initialized at `[0.5, 1.0]`, reported as "converged" to 0.508/1.004 | `accord_phase3_v2_ablations.py:76,84` | **Init every interpretable parameter at zero.** Never at a hypothesized value. |
| α applied to a branch immediately LayerNorm'd ⇒ near-zero gradient | `:105-108` | Verify the parameter has a non-vanishing gradient path before interpreting it. |
| Row-level random split, no grouping | `:58-60`; no `hadm_id`/`groupby` in file | **Group-disjoint splits by `SUBJECT_ID` and `HADM_ID`.** |
| Rule ID written into input and predicted from it ⇒ F1 = 0.953 | `:217` | No target may be derivable from an input feature in the same forward pass. |
| Role/temporal/disease inputs hardcoded constant for every example | `:380-388` | Assert non-zero variance on every categorical input before training. |
| Identical params across contaminated and decontaminated runs read as confirmation | paper §9 | Identical parameters across materially different data is evidence the parameter **did not train**. |

**Mandatory reporting for every parameter called "learned":**
`‖θ_final − θ_init‖ / ‖θ_init‖`, gradient-norm trace, and a trajectory-from-init plot.
One number would have caught the α artifact.

## 6b. MEASURED RESULTS — the premise check failed

Run 2026-08-02, locally, on the full MIMIC-III NOTEEVENTS table (2,083,180 rows).

### Corpus facts (`scripts/scan_notes.py`)

- 58,361 distinct admissions; **54,316 (93.1%) carry ≥2 distinct note categories**,
  49,279 (84.4%) carry ≥3. The mechanism *can* fire on this data — unlike every legal
  benchmark, where it is provably inert.
- 38,548 admissions have both nursing and discharge notes; 8,714 have all three roles.
  ACCORD used 253 pairs with 12 positives. Scale is not the binding constraint here.
- Mean characters: Nursing/other 799, Nursing 1,786, Physician 7,139, **Discharge 9,619**.
  A 12× spread. `α_D ≈ 2 α_N` is exactly what length alone would produce.

### Role probe (`scripts/role_probe.py`)

3-way role classification, 18,000 notes, TF-IDF + logistic regression, group-disjoint
split by `SUBJECT_ID`.

| Condition | Accuracy |
|---|---|
| document length alone (2 features) | 74.9% |
| raw text | 99.8% |
| explicit role markers scrubbed | 99.7% |
| + structural headers scrubbed | 99.7% |
| **scrubbed AND length-matched** | **97.5%** (macro-F1 0.9747) |

Length-matching drives the length-only baseline to 36.7% against a 33.3% chance floor, so
the length confound is genuinely eliminated — and role is still 97.5% recoverable.

### Diffuseness (`scripts/probe_robustness.py`)

Iteratively deleting the most role-discriminative n-grams and refitting:

| Top features removed | Accuracy |
|---|---|
| 0 | 97.5% |
| 1,000 | 95.2% |
| 20,000 | 91.7% |
| 40,000 of 43,037 | 66.0% |

Removing the top 1,000 features costs **2.3 points**. Halving the above-chance margin
requires deleting 93% of the vocabulary. The signal is not carried by markers; it is
distributed across the entire lexicon.

### Granularity and the information bound (`scripts/probe_granularity.py`)

The encoder conditions attention over a 512-token window, so whole-note recoverability is
the wrong measurement. Re-run on random within-note windows, with the Fano upper bound on
`H(role | text)` against a marginal `H(role) = log2(3) = 1.585` bits:

| Window (tokens) | Accuracy | `H(role \| text) ≤` | Fraction of marginal |
|---|---|---|---|
| 64 | 91.4% | 0.508 bits | 32.1% |
| 128 | 94.8% | 0.350 bits | 22.1% |
| 256 | 97.3% | 0.205 bits | 12.9% |
| **512 (model window)** | **98.9%** | **0.098 bits** | **6.2%** |
| whole note | 99.7% | 0.032 bits | 2.0% |

**At the granularity the model actually consumes, the role label can supply at most 0.098
bits.** This upper-bounds what *any* architecture conditioning on that label could gain.

What carries the signal (top discriminative n-grams) is substantially template and
register, not clinical content — `medquist36` (a transcription vendor tag), `job`,
`date sex`, `sig` for discharge; `pt`, `cont`, `sats`, `gtt` for nursing. The robustness
ablation already covers this objection: deleting the top 1,000 features costs 2.3 points,
so the conclusion holds after removing every template artifact.

### Local LLM baseline (`scripts/ollama_role_baseline.py`)

Llama 3 8B via Ollama on `127.0.0.1` — the compliant replacement for ACCORD's
GPT-4o calls, which PhysioNet's guidance identifies as prohibited for restricted
data. The script refuses any non-local host.

Zero-shot, temperature 0, 300 notes, scrub level 2, first 1500 chars:

| Model | Same input | Accuracy |
|---|---|---|
| TF-IDF + logistic regression | first 1500 chars | **96.4%** |
| Llama 3 8B zero-shot | first 1500 chars | **33.0%** (chance 33.3%) |

The model predicted `discharge` **zero times in 300** and `physician` 287 times —
degenerate collapse to one label. The first instinct was a harness bug: scrub
level 2 removes the admin header that opens a discharge summary, and truncation
shows only the head. The control refutes that — a linear model recovers role from
**the identical bytes** at 96.4%. The information is present; this
model-and-prompt does not use it.

Limits, stated because the temptation to over-read this is exactly the ACCORD
failure mode: one prompt, one model, zero-shot, no few-shot demonstrations, no
prompt search. The defensible claim is "this configuration is at chance," not
"LLMs cannot do this." Few-shot is the obvious next test and has not been run.

What it does establish: the redundancy result is **not** an artifact of giving the
probe more text than a realistic consumer would see.

### 6c. THE MECHANISM, identified (controls, 1000 steps, 10M, seq 256)

All arms below shuffle segment order, so role is decorrelated from bundle
position (bundles are chronological; segment index alone predicts role at 45.1%
against a 32.6% majority baseline).

| Arm | Bias keyed on | val_loss | vs A1 |
|---|---|---|---|
| A1 | nothing | 4.3465 | — |
| A2 | provenance as literal prefix tokens | ≈4.38 | ~0.00 |
| A4 | **true role labels** | 3.5325 | −0.814 |
| **A5** | **ordinal position in bundle (no provenance meaning)** | **3.4216** | **−0.925** |

Full control set, monotone in partition quality:

| Bias keyed on | distinct ids/seq | val_loss | % of A4's gain |
|---|---|---|---|
| nothing | — | 4.3465 | 0% |
| note-length bucket | 2.77 | 4.0174 | 40% |
| random role per segment | 3.98 | 3.8587 | 60% |
| true role labels | 3.70 | 3.5325 | 100% |
| **ordinal segment index** | ~5.8 | **3.4216** | **114%** |

**A5 recovers 114% of A4's gain.** Labelling segments by an arbitrary ordinal is
*better* than labelling them by who wrote the note.

One nuance that keeps the claim honest: random roles yield *more* distinct ids per
sequence than real roles (3.98 vs 3.70) yet perform 0.33 nats worse. So resolution
alone is not the whole story — **consistency across examples** matters too (nursing
is always id 1 under real labels, arbitrary under random). The precise claim is
that the bias needs a *consistent, high-resolution* segment partition. Semantic
content contributes nothing beyond that.

**Boilerplate diagnostic — negative, which strengthens the result.** At seq 256 each
note contributes ~36 tokens, so training sits in note openings. If the advantage
were template identification, it would concentrate there. Loss gap (A1 − A4) by
position within segment:

| tokens 0–8 | 8–20 | 20–40 |
|---|---|---|
| +0.8869 | +0.8201 | +0.7865 |

Flat. Not a truncation artifact.

Why: **91.2% of bundles contain ≥2 segments sharing a role.** True role labels are
a *lossy* partition — two nursing notes in one admission collapse to one id and
become indistinguishable to the bias. Segment index never collides.

So the mechanism is **document segmentation**: a shared-across-layers additive
logit bias inducing soft block-diagonal attention ("attend within your own
note"), which is worth a great deal under MLM on packed multi-document input.
Provenance is a degraded proxy for the partition, not the source of the gain.

Note also that A4 is unchanged by shuffling (3.5325 vs 3.5415), which rules out
the competing story that role was merely a stand-in for chronological position.

**Comparison class.** This device is not new. T5's relative position bias is a
learned additive logit table computed once and shared across every layer; this
is the same object with the bucketing function `f(i−j)` replaced by
`f(seg(i), seg(j))`. ALiBi is the same idea with a fixed monotone bias, and its
headline result is likewise faster early training. The correct baselines are
T5-RPE and ALiBi, not BERT.

**A control that must never be cited.** A *global* derangement of role labels is
vacuous: `A` is a free K×K table, so under relabeling σ the model simply learns
`A'[σ(r),σ(r')] = A[r,r']`. The loss surface is isomorphic and the result is
fixed at the uncorrupted value by construction. It was implemented, recognized as
powerless, and removed. Guarded in code.

### 6d. FINAL ARM COMPARISON — 7 arms × 3 seeds, all controls

10M params, seq 256, batch 8, 1000 steps, MLM on packed multi-document bundles,
segment order shuffled. Mean ± std over 3 seeds. "% " is the fraction of the
segment-index bias's gain over vanilla.

| Arm | val_loss | vs vanilla | % | uses segment info |
|---|---|---|---|---|
| vanilla (no bias) | 4.3554 ±0.0085 | — | 0% | no |
| T5 relative position bias | 4.1035 ±0.0029 | −0.2520 | 27% | no |
| misaligned block bias | 3.9109 ±0.0293 | −0.4445 | 47% | no |
| ALiBi distance bias | 3.7435 ±0.0023 | −0.6119 | 65% | no |
| true role labels | 3.5310 ±0.0076 | −0.8244 | 87% | yes |
| segment-index bias | 3.4078 ±0.0225 | −0.9476 | 100% | yes |
| **hard block mask (fixed)** | **3.2772 ±0.0124** | **−1.0782** | **114%** | yes |

Seed spreads are ≤0.03 against effects of 0.25–1.08, so every ordering is
resolved.

**What each control settled:**

1. **Provenance contributes nothing, and slightly hurts.** True role labels are
   0.123 nats *worse* than an arbitrary segment ordinal. The original thesis is
   dead, measured rather than argued.
2. **"Learned" is decoration.** A fixed block-diagonal mask beats the learned
   bias by 0.131 nats. The operative method is intra-document attention masking
   (Zhao et al., ACL 2024) — this work reproduces it, it does not improve on it.
3. **Document boundaries genuinely matter.** Misaligned blocks with the same
   block count recover only 47% versus 100% for aligned. Alignment is worth
   0.503 nats, so "any coarse blocking" is not the explanation.
4. **Locality explains most but not all.** ALiBi recovers 65% with no segment
   information at all, leaving a 0.336-nat segment-alignment residual.
5. **T5-RPE recovers only 27%**, so the baseline's lack of relative position is
   not the story either.

**Bottom line.** The finding is not a provenance architecture and not a novel
mechanism. It is a controlled decomposition of a known one: on packed
multi-document input, most of the benefit attributed to metadata conditioning is
locality, a further chunk is document-boundary alignment, and none of it is
metadata semantics — and the best-performing form is the simplest and already
published.

### 6e. DOWNSTREAM: does any of it survive on a real labeled task?

The task ACCORD never had. In-hospital mortality from the notes of the first 48
hours, input = a bundle of notes from several authors. Discharge summaries
excluded (they state the outcome), posthumous notes excluded, splits
group-disjoint by `SUBJECT_ID`.

**36,504 examples, 3,730 positives, 734 in the test set** — against ACCORD's 253
and 12, and 6.8× the 548 positives ACCORD's own power analysis said it needed.

| Model | non-emb params | pretraining | AUROC | AUPRC |
|---|---|---|---|---|
| PubMedBERT + concatenation | 109,482,240 | 3.1B words | 0.8223 | 0.3878 |
| ours, block-mask pretrained | 10,638,720 | ~6M tokens | 0.8014 | 0.3258 |
| ours, **random init** | 10,638,720 | **none** | 0.7998 | 0.3290 |
| ours, vanilla pretrained | 10,638,336 | ~6M tokens | 0.7937 | 0.3121 |

**Two findings, and the second undercuts the first.**

1. A from-scratch 10M encoder reaches **97.5% of PubMedBERT's AUROC** with 1/11
   the parameters and roughly 1/1000 the pretraining data. Competitive per
   parameter on the input shape it was built for.

2. **Pretraining contributes nothing.** Random initialization (0.7998 / 0.3290)
   matches the block-mask pretrained model (0.8014 / 0.3258) and beats it on
   AUPRC. And the 1.337-nat MLM advantage of block masking converts to only
   +0.008 AUROC downstream.

So the pretraining-loss results in §6c–6d, however cleanly controlled, **do not
transfer**. What produces the downstream number is 25,527 labeled examples, not
the architecture and not the pretraining objective. Any claim resting on MLM loss
as a proxy for clinical utility is unsupported by this evidence — which is
precisely the kind of proxy-metric failure ACCORD was originally written about.

Caveats: one seed per arm, 3 epochs, seq 256, val-set selection. The
PubMedBERT row is a single run kept as a reference point, not a matched
comparison — it uses a different tokenizer and 512 positions.

### Consequence — stated precisely

**The information-channel framing is dead.** "Provenance supplies information the text does
not" is false in clinical notes, with a measured bound of ≤0.098 bits at the model's window.

**This does NOT prove that conditioning cannot help**, and the earlier version of this
document overclaimed exactly that. Redundant features can still be useful inductive bias —
absolute position is largely recoverable from a bag of tokens, yet positional encodings are
load-bearing, and T5/ALiBi relative-position biases supply nothing the model could not
compute in principle. Conflating "redundant" with "useless" is the same inferential error
as ACCORD's, running in the opposite direction.

### Independent kill: the mechanism is already inside the baseline

Expand the baseline's own attention logit assuming role is additively encoded,
`h_i = c_i + u_{r_i}`:

```
s_ij = c_i^T M c_j  +  c_i^T M u_rj  +  u_ri^T M c_j  +  u_ri^T M u_rj
                                                         ^^^^^^^^^^^^ = A[r_i, r_j]
```

The authority matrix already exists inside the vanilla QK bilinear form, at zero extra
parameters and rank up to `min(K, d_head)`. A separate `A` is a **rank-restricted copy**
of something the baseline owns, purchased with head capacity. Worse, the discarded cross
terms `c_i^T M u_rj` and `u_ri^T M c_j` express *content-conditional* deference, which is
the clinically correct unit: a discharge summary's copy-pasted HPI carries near-zero
independent authority while its Discharge Medications section is the most authoritative
statement in the chart. A document-uniform `A` weights them identically.

**Authority attaches to the assertion, not the document.** That is the real finding, and it
is why the architecture is the wrong shape.

## 7. Open

- Which endpoint served the original GPT-4o evaluation (decides whether a PhysioNet
  self-report is owed).
- Repo inventory for `accord_neurips2026_supplemental` and
  `clinical-llm-evaluation-supplemental` — `git rm` + commit does not remove history.
- Whether a multi-provenance clinical task can be defined with enough positives to escape
  ACCORD's N=12 problem. This is the binding constraint on the whole thesis.
