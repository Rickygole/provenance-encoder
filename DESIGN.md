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
