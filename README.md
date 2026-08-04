# provenance-encoder

A from-scratch transformer encoder (no pretrained backbone) that treats document
provenance — who wrote a note, with what standing, when — as an architectural
primitive, plus the experiments that test whether that is worth doing.

**Status: complete, and the hypothesis is falsified.** The architecture is built,
trains, and was tested against every control that could explain its apparent
advantage. Provenance conditioning contributes nothing — an arbitrary segment
ordinal outperforms real authorship labels, and a fixed block mask outperforms
the learned bias. The falsification, and the controls that produced it, are the
deliverable. See `DESIGN.md` §6b–6d.

## Headline result (7 arms × 3 seeds)

| Arm | val_loss | % of segment-index gain |
|---|---|---|
| vanilla | 4.3554 ±0.0085 | 0% |
| T5 relative position bias | 4.1035 ±0.0029 | 27% |
| misaligned block bias | 3.9109 ±0.0293 | 47% |
| ALiBi distance bias | 3.7435 ±0.0023 | 65% |
| true role labels | 3.5310 ±0.0076 | 87% |
| segment-index bias | 3.4078 ±0.0225 | 100% |
| **hard block mask (fixed)** | **3.2772 ±0.0124** | **114%** |

Most of the apparent benefit of metadata conditioning is locality; a further part
is document-boundary alignment; none of it is metadata semantics. The best form
is the simplest and was already published (Zhao et al., ACL 2024).

## Downstream result (multi-note mortality, 36,504 examples / 3,730 positives)

| Model | non-emb params | pretraining | AUROC | AUPRC |
|---|---|---|---|---|
| PubMedBERT + concatenation | 109,482,240 | 3.1B words | 0.8223 | 0.3878 |
| ours, block-mask pretrained | 10,638,720 | ~6M tokens | 0.8014 | 0.3258 |
| ours, **random init** | 10,638,720 | **none** | 0.7998 | 0.3290 |
| ours, vanilla pretrained | 10,638,336 | ~6M tokens | 0.7937 | 0.3121 |

### Three seeds, pretraining seed varied

| Arm | AUROC | AUPRC |
|---|---|---|
| **random init (NO pretraining)** | **0.8003 ±0.0031** | 0.3242 ±0.0062 |
| block-mask pretrained | 0.7956 ±0.0079 | 0.3330 ±0.0063 |
| vanilla pretrained | 0.7952 ±0.0062 | 0.3142 ±0.0139 |

A from-scratch 10M encoder reaches ~97% of PubMedBERT's AUROC at 1/11 the
parameters and ~1/1000 the pretraining data. **But pretraining contributes
nothing**: seed noise (±0.0079) exceeds the between-arm spread (0.0051), the
un-pretrained model has the highest mean, and the single-seed ordering flips
between seeds.

The sharp version: a pretraining-loss gap of **1.320 ± 0.015 nats** predicts a
downstream AUROC difference of **0.0004**. A metric stable to three decimals that
forecasts nothing — a proxy-metric failure, measured. All numbers in `results/`.

## Why this exists

Grew out of ACCORD (NeurIPS 2026, withdrawn). Reviewers said the architecture was
unvalidated, splits were contaminated, and the headline "learned authority
weights" might be an artifact. Auditing the code confirmed the artifact: the
parameter was initialized at exactly the values reported as discovered. Every
design decision here is traceable to one of those failures, and several are
enforced as tests that fail the build.

## Findings so far

| Question | Answer | Where |
|---|---|---|
| Can the mechanism fire on this data? | Yes — 93.1% of admissions carry ≥2 note roles | `scripts/scan_notes.py` |
| Is provenance recoverable from text alone? | **98.9% at the model's 512-token window**; `H(role\|text) ≤ 0.098` of 1.585 bits | `scripts/probe_granularity.py` |
| Is that just markers or length? | No. Deleting the top 1,000 features costs 2.3 points; length-matching leaves 97.5% | `scripts/probe_robustness.py` |
| Is the bias inert on single-provenance input? | Yes, provably and exactly | `tests/test_invariants.py` |

## Layout

```
src/model/      config.py (arms), encoder.py (the encoder + bias)
src/data/       roles.py (frozen taxonomy), bundles.py (packing, MLM)
src/train/      instrument.py (assertions that fail runs)
scripts/        scan_notes, role_probe, probe_robustness, probe_granularity,
                check_matched_slice, build_bundles, train_tokenizer,
                pretrain, run_sweep.sh, analyze_sweep
tests/          test_invariants.py
local_only/     MIMIC-derived artifacts. gitignored. never leaves the machine.
```

## The four arms

One model class, arms derived from a single config flag so a config cannot
describe two interventions at once.

| Arm | Intervention |
|---|---|
| A1 | vanilla, no provenance |
| **A2** | **provenance as six literal vocabulary tokens; architecturally identical to A1** |
| A3 | row-centered full-rank K×K additive attention bias |
| A4 | A3 + role/authority embeddings + monotone time bias |

**A2 is the decisive baseline.** If A2 matches A4, a tokenizer change did the
work and there is no architectural contribution. Every arm reserves the same
token budget so A2 is not silently handicapped.

## Running it

```bash
python3 tests/test_invariants.py          # invariants, no data needed
python3 scripts/scan_notes.py             # corpus aggregates
python3 scripts/role_probe.py             # the premise check
python3 scripts/probe_granularity.py      # recoverability at model granularity
python3 scripts/build_bundles.py          # multi-provenance bundles
python3 scripts/train_tokenizer.py        # 32k WordPiece from scratch
STEPS=1000 bash scripts/run_sweep.sh      # 4 arms x 3 seeds
python3 scripts/analyze_sweep.py          # paired comparison
```

## Data handling

MIMIC-III is PhysioNet credentialed data. Everything touching it runs locally
and emits aggregates only — no note text is printed, logged, or sent anywhere.
LLM baselines run through Ollama on localhost rather than a hosted API, which
PhysioNet's published guidance prohibits for restricted data. `local_only/` is
gitignored, as are the MIMIC table names, and `/Users/rickygole` is itself a git
repository, which is why those rules are aggressive.

Pretraining corpora, if used, are open (PubMed / PMC OA, CC0 and CC BY only).
No MIMIC-derived artifact is cloud-synced.

## Invariants

`tests/test_invariants.py` encodes failures rather than features:

1. The authority bias is **exactly inert** on single-provenance input (softmax
   shift-invariance). This is why single-document benchmarks cannot measure it.
2. It is active on multi-provenance input.
3. `A → A + c·1ᵀ` is a gauge symmetry, so only the row-centered spectrum may be
   interpreted.
4. Authority is zero-initialized — never at a hypothesized value.
5. The temporal bias starts negligible and is monotone in |Δt|.
6. Every interpretable parameter is enumerable, so the trainer can prove it moved.
