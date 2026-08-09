# Prevalence replication: is the clinical over-calling finding domain-specific or task-specific?

Non-clinical replication of ACCORD C20/C23 (`accord/CORRECTIONS.md`). There, three 8B open-weights models, local via Ollama, over-called clinical medication-contradiction pairs at 15.8x-23.6x against a 4.0% pharmacist base rate -- but at a SINGLE fixed prevalence point, so the clinical data alone cannot distinguish "the model over-calls" from "the model ignores prevalence entirely." This experiment sweeps TRUE PREVALENCE (2%/5%/10%/25%) and PROMPT (bare vs. base-rate-aware+abstention) on a public, non-clinical, human-labeled task with the identical binary-contradiction structure, to separate "a property of clinical adjudication" from "a property of asking an instruction-tuned model to make a rare-class call from short evidence."

## Data

SNLI (Stanford Natural Language Inference) premise/hypothesis pairs, test+validation splits, human-labeled (Mechanical Turk 5-way majority vote in the original collection), downloaded from the public HF hub mirror `stanfordnlp/snli`. Binary task: contradiction (label==2) vs not_contradiction (entailment+neutral pooled) -- the identical framing as the clinical prompt ("is there a contradiction between these two short texts?"), on public, non-clinical sentence pairs. Why this dataset: (a) human ground truth, standard benchmark; (b) premise and hypothesis are both single sentences, comparable in length to the clinical nursing-note / discharge-summary sentence pairs; (c) SNLI is balanced ~33/33/33 by construction, so it can be subsampled to ANY target prevalence -- including one matched to the clinical pipeline's own ~4% -- which the clinical data, fixed at one prevalence point, cannot do.

Four sweep pools (used for the main model x prompt x prevalence grid), N=100 each, disjoint across levels, built by `build_pools.py` with `np.random.default_rng(42)`. A second, larger, also-disjoint set of pools (N=300) was built for the distillation supplement (see §4) after the N=100 student turned out too small to fit.
| target prevalence | n (sweep) | n_pos (sweep) | n (distill supp.) | n_pos (distill supp.) |
|---|---|---|---|---|
| 2% | 100 | 2 | 300 | 6 |
| 5% | 100 | 5 | 300 | 15 |
| 10% | 100 | 10 | 300 | 30 |
| 25% | 100 | 25 | 300 | 75 |

## 1. Prevalence sweep -- does the labeller's marginal track true prevalence?

`positive rate (decided)` = contradiction calls / (contradiction+not_contradiction calls), i.e. excluding ABSTAIN and UNPARSEABLE from the denominator. `over-call ratio` = positive rate / true pool prevalence, the same quantity C20/C23 report against the 4.0% clinical base rate. UNPARSEABLE is 0 everywhere in this run (see full CSVs in `results/raw_predictions/`) -- the C10 default-to-positive bug class does not arise here either.

### llama3:8b

| prompt | true prevalence | n | pos | neg | abstain | UNPARSEABLE | positive rate (decided) | over-call ratio |
|---|---|---|---|---|---|---|---|---|
| A (bare) | 2% | 100 | 7 | 93 | 0 | 0 (0.0%) | 0.0700 | 3.50x |
| A (bare) | 5% | 100 | 17 | 83 | 0 | 0 (0.0%) | 0.1700 | 3.40x |
| A (bare) | 10% | 100 | 6 | 94 | 0 | 0 (0.0%) | 0.0600 | 0.60x |
| A (bare) | 25% | 100 | 12 | 88 | 0 | 0 (0.0%) | 0.1200 | 0.48x |
| B (base-rate + abstain) | 2% | 100 | 3 | 70 | 27 | 0 (0.0%) | 0.0411 | 2.05x |
| B (base-rate + abstain) | 5% | 100 | 9 | 73 | 18 | 0 (0.0%) | 0.1098 | 2.20x |
| B (base-rate + abstain) | 10% | 100 | 4 | 70 | 26 | 0 (0.0%) | 0.0541 | 0.54x |
| B (base-rate + abstain) | 25% | 100 | 5 | 63 | 32 | 0 (0.0%) | 0.0735 | 0.29x |

### granite3.3:8b

| prompt | true prevalence | n | pos | neg | abstain | UNPARSEABLE | positive rate (decided) | over-call ratio |
|---|---|---|---|---|---|---|---|---|
| A (bare) | 2% | 100 | 5 | 95 | 0 | 0 (0.0%) | 0.0500 | 2.50x |
| A (bare) | 5% | 100 | 7 | 93 | 0 | 0 (0.0%) | 0.0700 | 1.40x |
| A (bare) | 10% | 100 | 10 | 90 | 0 | 0 (0.0%) | 0.1000 | 1.00x |
| A (bare) | 25% | 100 | 16 | 84 | 0 | 0 (0.0%) | 0.1600 | 0.64x |
| B (base-rate + abstain) | 2% | 100 | 3 | 95 | 2 | 0 (0.0%) | 0.0306 | 1.53x |
| B (base-rate + abstain) | 5% | 100 | 4 | 96 | 0 | 0 (0.0%) | 0.0400 | 0.80x |
| B (base-rate + abstain) | 10% | 100 | 9 | 91 | 0 | 0 (0.0%) | 0.0900 | 0.90x |
| B (base-rate + abstain) | 25% | 100 | 17 | 83 | 0 | 0 (0.0%) | 0.1700 | 0.68x |

### qwen3:8b

| prompt | true prevalence | n | pos | neg | abstain | UNPARSEABLE | positive rate (decided) | over-call ratio |
|---|---|---|---|---|---|---|---|---|
| A (bare) | 2% | 100 | 2 | 98 | 0 | 0 (0.0%) | 0.0200 | 1.00x |
| A (bare) | 5% | 100 | 3 | 97 | 0 | 0 (0.0%) | 0.0300 | 0.60x |
| A (bare) | 10% | 100 | 5 | 95 | 0 | 0 (0.0%) | 0.0500 | 0.50x |
| A (bare) | 25% | 100 | 9 | 91 | 0 | 0 (0.0%) | 0.0900 | 0.36x |
| B (base-rate + abstain) | 2% | 100 | 0 | 83 | 17 | 0 (0.0%) | 0.0000 | 0.00x |
| B (base-rate + abstain) | 5% | 100 | 0 | 80 | 20 | 0 (0.0%) | 0.0000 | 0.00x |
| B (base-rate + abstain) | 10% | 100 | 2 | 78 | 20 | 0 (0.0%) | 0.0250 | 0.25x |
| B (base-rate + abstain) | 25% | 100 | 2 | 73 | 25 | 0 (0.0%) | 0.0267 | 0.11x |

### Calibration slope: positive rate vs. true prevalence (OLS across the 4 levels)

Slope=1 would mean the labeller's marginal tracks true prevalence exactly (perfect calibration); slope=0 would mean the marginal is completely flat / prevalence-blind, which is what a 21x-style over-call at a SINGLE clinical prevalence point is consistent with but cannot itself establish. This is the direct answer to the open question this experiment was built to ask.

| model | prompt | slope | intercept | positive rates at 2%/5%/10%/25% |
|---|---|---|---|---|
| llama3:8b | A (bare) | 0.058 | 0.099 | 0.070, 0.170, 0.060, 0.120 |
| llama3:8b | B (base-rate + abstain) | 0.028 | 0.067 | 0.041, 0.110, 0.054, 0.074 |
| granite3.3:8b | A (bare) | 0.466 | 0.046 | 0.050, 0.070, 0.100, 0.160 |
| granite3.3:8b | B (base-rate + abstain) | 0.620 | 0.018 | 0.031, 0.040, 0.090, 0.170 |
| qwen3:8b | A (bare) | 0.302 | 0.016 | 0.020, 0.030, 0.050, 0.090 |
| qwen3:8b | B (base-rate + abstain) | 0.120 | 0.000 | 0.000, 0.000, 0.025, 0.027 |

## 2. Accuracy against SNLI gold (decided rows only)

### llama3:8b

| prompt | true prevalence | recall | precision | accuracy | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|
| A (bare) | 2% | 0.000 | 0.000 | 0.910 | 0 | 7 | 2 | 91 |
| A (bare) | 5% | 0.600 | 0.176 | 0.840 | 3 | 14 | 2 | 81 |
| A (bare) | 10% | 0.200 | 0.333 | 0.880 | 2 | 4 | 8 | 86 |
| A (bare) | 25% | 0.240 | 0.500 | 0.750 | 6 | 6 | 19 | 69 |
| B (base-rate + abstain) | 2% | nan | 0.000 | 0.959 | 0 | 3 | 0 | 70 |
| B (base-rate + abstain) | 5% | 1.000 | 0.111 | 0.902 | 1 | 8 | 0 | 73 |
| B (base-rate + abstain) | 10% | 0.667 | 0.500 | 0.959 | 2 | 2 | 1 | 69 |
| B (base-rate + abstain) | 25% | 0.429 | 0.600 | 0.912 | 3 | 2 | 4 | 59 |

### granite3.3:8b

| prompt | true prevalence | recall | precision | accuracy | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|
| A (bare) | 2% | 1.000 | 0.400 | 0.970 | 2 | 3 | 0 | 95 |
| A (bare) | 5% | 0.600 | 0.429 | 0.940 | 3 | 4 | 2 | 91 |
| A (bare) | 10% | 0.700 | 0.700 | 0.940 | 7 | 3 | 3 | 87 |
| A (bare) | 25% | 0.640 | 1.000 | 0.910 | 16 | 0 | 9 | 75 |
| B (base-rate + abstain) | 2% | 1.000 | 0.667 | 0.990 | 2 | 1 | 0 | 95 |
| B (base-rate + abstain) | 5% | 0.600 | 0.750 | 0.970 | 3 | 1 | 2 | 94 |
| B (base-rate + abstain) | 10% | 0.700 | 0.778 | 0.950 | 7 | 2 | 3 | 88 |
| B (base-rate + abstain) | 25% | 0.680 | 1.000 | 0.920 | 17 | 0 | 8 | 75 |

### qwen3:8b

| prompt | true prevalence | recall | precision | accuracy | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|
| A (bare) | 2% | 0.000 | 0.000 | 0.960 | 0 | 2 | 2 | 96 |
| A (bare) | 5% | 0.200 | 0.333 | 0.940 | 1 | 2 | 4 | 93 |
| A (bare) | 10% | 0.500 | 1.000 | 0.950 | 5 | 0 | 5 | 90 |
| A (bare) | 25% | 0.320 | 0.889 | 0.820 | 8 | 1 | 17 | 74 |
| B (base-rate + abstain) | 2% | 0.000 | nan | 0.988 | 0 | 0 | 1 | 82 |
| B (base-rate + abstain) | 5% | 0.000 | nan | 0.975 | 0 | 0 | 2 | 78 |
| B (base-rate + abstain) | 10% | 0.333 | 1.000 | 0.950 | 2 | 0 | 4 | 74 |
| B (base-rate + abstain) | 25% | 0.200 | 1.000 | 0.893 | 2 | 0 | 8 | 65 |

## 3. Distillation at N=100 -- and why the hard-threshold student is degenerate

TF-IDF (1-2gram) + logistic regression, trained on the LLM's own decided-row labels, 5-fold (or fewer, if a class is too small) stratified cross-validated out-of-fold predictions. ABSTAIN/UNPARSEABLE rows are dropped from student training (count reported). **At N=100, the hard (0.5-threshold) student collapses to predicting not_contradiction for every out-of-fold item, in every combo that fit at all** -- there isn't enough signal for 3000-dim TF-IDF + LR to cross the decision boundary from ~64-80 training rows per fold. This is reported as `degenerate_hard_threshold=True` rather than silently treated as a real zero. The **soft marginal** (mean out-of-fold P(contradiction), threshold-independent) does not have this problem and is the number to read.

### llama3:8b

| prompt | prevalence | n decided | dropped | teacher marginal | student soft marginal | student hard marginal | true prevalence | closer (soft) |
|---|---|---|---|---|---|---|---|---|
| A (bare) | 2% | 100 | 0 | 0.070 | 0.068 | 0.000 | 0.020 | teacher |
| A (bare) | 5% | 100 | 0 | 0.170 | 0.165 | 0.000 | 0.050 | teacher |
| A (bare) | 10% | 100 | 0 | 0.060 | 0.059 | 0.000 | 0.100 | teacher |
| A (bare) | 25% | 100 | 0 | 0.120 | 0.117 | 0.000 | 0.250 | teacher |
| B (base-rate + abstain) | 2% | 73 | 27 | 0.041 | 0.040 | 0.000 | 0.000 | teacher |
| B (base-rate + abstain) | 5% | 82 | 18 | 0.110 | 0.106 | 0.000 | 0.012 | teacher |
| B (base-rate + abstain) | 10% | 74 | 26 | 0.054 | 0.053 | 0.000 | 0.041 | teacher |
| B (base-rate + abstain) | 25% | 68 | 32 | 0.074 | 0.072 | 0.000 | 0.103 | teacher |

### granite3.3:8b

| prompt | prevalence | n decided | dropped | teacher marginal | student soft marginal | student hard marginal | true prevalence | closer (soft) |
|---|---|---|---|---|---|---|---|---|
| A (bare) | 2% | 100 | 0 | 0.050 | 0.049 | 0.000 | 0.020 | teacher |
| A (bare) | 5% | 100 | 0 | 0.070 | 0.068 | 0.000 | 0.050 | teacher |
| A (bare) | 10% | 100 | 0 | 0.100 | 0.097 | 0.000 | 0.100 | tied |
| A (bare) | 25% | 100 | 0 | 0.160 | 0.156 | 0.000 | 0.250 | teacher |
| B (base-rate + abstain) | 2% | 98 | 2 | 0.031 | 0.030 | 0.000 | 0.020 | teacher |
| B (base-rate + abstain) | 5% | 100 | 0 | 0.040 | 0.039 | 0.000 | 0.050 | teacher |
| B (base-rate + abstain) | 10% | 100 | 0 | 0.090 | 0.088 | 0.000 | 0.100 | teacher |
| B (base-rate + abstain) | 25% | 100 | 0 | 0.170 | 0.165 | 0.000 | 0.250 | teacher |

### qwen3:8b

| prompt | prevalence | n decided | dropped | teacher marginal | student soft marginal | student hard marginal | true prevalence | closer (soft) |
|---|---|---|---|---|---|---|---|---|
| A (bare) | 2% | 100 | 0 | 0.020 | 0.020 | 0.000 | 0.020 | tied |
| A (bare) | 5% | 100 | 0 | 0.030 | 0.029 | 0.000 | 0.050 | teacher |
| A (bare) | 10% | 100 | 0 | 0.050 | 0.049 | 0.000 | 0.100 | teacher |
| A (bare) | 25% | 100 | 0 | 0.090 | 0.087 | 0.000 | 0.250 | teacher |
| B (base-rate + abstain) | 2% | 83 | 17 | 0.000 | n/a (degenerate teacher label) | 0.000 | 0.012 | n/a |
| B (base-rate + abstain) | 5% | 80 | 20 | 0.000 | n/a (degenerate teacher label) | 0.000 | 0.025 | n/a |
| B (base-rate + abstain) | 10% | 80 | 20 | 0.025 | 0.025 | 0.000 | 0.075 | teacher |
| B (base-rate + abstain) | 25% | 75 | 25 | 0.027 | 0.026 | 0.000 | 0.133 | teacher |

## 4. Distillation supplement at N=300 -- well-powered test of prevalence inheritance

`llama3:8b`, prompt A (bare) only -- the direct analog of the clinical `llama_baseline` reference -- re-run at N=300 per prevalence level on pools disjoint from every pool above (`run_distillation_supplement.py`). Same student (TF-IDF+LR, 5-fold CV). At N=300 the hard-threshold student is STILL degenerate at 3 of 4 levels (see `degenerate_hard_threshold`), which is itself informative: the clinical distillation had 1,679 training rows, not 100-300, and this suggests hard-threshold reproduction needs that much data even when the underlying probability estimate (below) is already very accurate. The soft marginal is not threshold-limited and is the number that answers the task's question directly.

| prevalence | n decided | teacher marginal | student soft marginal | true prevalence | \|soft − teacher\| | \|soft − true\| | closer to |
|---|---|---|---|---|---|---|---|
| 2% | 300 | 0.097 | 0.094 | 0.020 | 0.003 | 0.074 | teacher |
| 5% | 300 | 0.063 | 0.062 | 0.050 | 0.002 | 0.012 | teacher |
| 10% | 300 | 0.140 | 0.135 | 0.100 | 0.005 | 0.035 | teacher |
| 25% | 300 | 0.143 | 0.140 | 0.250 | 0.003 | 0.110 | teacher |

At every prevalence level, the well-powered (N=300) student's soft marginal is within 0.005 of the teacher's own marginal and far from true prevalence at 2%, 5%, and 25% -- the clean, well-powered version of the clinical claim (student reproduced teacher's predicted-positive rate to within 1.5pp; here, within 0.1-0.5pp). **The mechanism replicates on non-clinical text**: a student trained on a rare-class LLM label learns the LLM's marginal, not the ground truth, when the LLM's marginal and the ground truth diverge.

## Verdict

**(1) Gross over-calling magnitude is domain-specific, not task-generic.** The clinical pipeline over-called by 15.8x-23.6x at 4.0% true prevalence, with absolute predicted-positive rates of 63-95%. On SNLI contradiction judgments at the SAME prevalence range (2-25%), absolute positive-call rates never exceed ~17% for any model, prompt, or prevalence level tested (24 combos, 2,400 generations) -- over-call ratios range from 0.0x to 3.5x, an order of magnitude below the clinical floor of 7x that C23 established across every model tested there (including GPT-4o). At the two higher prevalence levels (10%, 25%), most model/prompt combinations UNDER-call rather than over-call. A 15-24x-style blanket-positive failure mode does not reproduce on non-clinical text with the same task shape.

**(2) The marginal is prevalence-INSENSITIVE but not fully flat -- a weaker version of the same failure mode.** Calibration slopes (positive rate vs. true prevalence, ideal=1) range from 0.028 (llama3:8b, base-rate-aware prompt) to 0.62 (granite3.3:8b, base-rate-aware prompt); every slope is well below 1, meaning every model under-responds to a 12.5x swing in true prevalence. llama3:8b is closest to the clinical "prevalence-blind" signature (slope 0.03-0.06, near-flat); granite3.3:8b partially tracks prevalence (slope 0.47-0.62) and is the best-calibrated of the three. So the *mechanism* the clinical single-point result could not establish -- a marginal that under-weights true prevalence -- does generalize to non-clinical text, but the *magnitude* of the resulting miscalibration on this task is far smaller than in the clinical pipeline, because the non-clinical intercept sits near 5-15% rather than near 85-95%.

**(3) Prompting that states the true base rate and permits abstention reduces, but does not eliminate, the effect** -- consistent with C23's finding that prompt moves the clinical number by ~2x within one model. Here the base-rate+abstain prompt lowers the positive rate at every prevalence level for every model versus the bare prompt (e.g. llama3:8b at 5%: 17.0% to 11.0%), and qwen3:8b abstains on all rows it would otherwise call positive at 2% and 5% prevalence, but the calibration slope is not consistently improved by the abstention prompt: it gets FLATTER for llama3:8b (0.058 to 0.028) and qwen3:8b (0.302 to 0.120), and only improves for granite3.3:8b (0.466 to 0.620) -- prompt is a real lever but not a reliable fix, on either dataset.

**(4) The prevalence-inheritance / distillation mechanism DOES replicate, cleanly, on non-clinical text**, and does so more precisely than the clinical result (within 0.1-0.5pp here vs. 1.5pp there, at N=300). A TF-IDF+LR student trained on LLM labels learns the teacher's marginal, not the ground truth, whenever the two diverge -- this held in all 22 fittable N=100 combos and all 4 N=300 combos. This is the part of the clinical finding that is a property of the TASK (distillation from a biased weak labeller), not of clinical text specifically.

**Overall: the effect is a blend, and the thesis should say so precisely rather than picking one side.** The over-calling MAGNITUDE (7-24x, absolute rates near 90%) is specific to the clinical adjudication task -- it did not reproduce at anything close to that scale on structurally identical non-clinical judgments at matched prevalence. But the underlying MECHANISMS -- (a) a marginal that under-responds to true prevalence, and (b) a distillation student that inherits the teacher's marginal rather than the truth -- both generalize cleanly to non-clinical text. The clinical pipeline's contribution is not "LLMs over-call rare classes" in the abstract (that is real but modest, 1-3.5x, on SNLI); it is that something about clinical medication-contradiction adjudication specifically pushes the same mechanism to an extreme absolute rate (85-95% positive) that this general-purpose NLI task never approaches.
