"""
Combine sweep_results.json + distillation_results.json +
distillation_supplement_results.json into the final markdown report.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")

with open(os.path.join(RESULTS_DIR, "sweep_results.json")) as f:
    sweep = json.load(f)
with open(os.path.join(RESULTS_DIR, "distillation_results.json")) as f:
    distill = json.load(f)
with open(os.path.join(RESULTS_DIR, "distillation_supplement_results.json")) as f:
    distill_supp = json.load(f)

MODELS = sweep["models"]
LEVELS = sweep["prevalence_levels"]
PROMPT_LABEL = {"A_bare": "A (bare)", "B_baserate_abstain": "B (base-rate + abstain)"}

lines = []
lines.append("# Prevalence replication: is the clinical over-calling finding domain-specific or task-specific?\n")
lines.append(
    "Non-clinical replication of ACCORD C20/C23 (`accord/CORRECTIONS.md`). There, three "
    "8B open-weights models, local via Ollama, over-called clinical medication-contradiction "
    "pairs at 15.8x-23.6x against a 4.0% pharmacist base rate -- but at a SINGLE fixed "
    "prevalence point, so the clinical data alone cannot distinguish \"the model over-calls\" "
    "from \"the model ignores prevalence entirely.\" This experiment sweeps TRUE PREVALENCE "
    "(2%/5%/10%/25%) and PROMPT (bare vs. base-rate-aware+abstention) on a public, "
    "non-clinical, human-labeled task with the identical binary-contradiction structure, to "
    "separate \"a property of clinical adjudication\" from \"a property of asking an "
    "instruction-tuned model to make a rare-class call from short evidence.\"\n"
)

lines.append("## Data\n")
lines.append(
    "SNLI (Stanford Natural Language Inference) premise/hypothesis pairs, test+validation "
    "splits, human-labeled (Mechanical Turk 5-way majority vote in the original collection), "
    "downloaded from the public HF hub mirror `stanfordnlp/snli`. Binary task: contradiction "
    "(label==2) vs not_contradiction (entailment+neutral pooled) -- the identical framing as "
    "the clinical prompt (\"is there a contradiction between these two short texts?\"), on "
    "public, non-clinical sentence pairs. Why this dataset: (a) human ground truth, standard "
    "benchmark; (b) premise and hypothesis are both single sentences, comparable in length to "
    "the clinical nursing-note / discharge-summary sentence pairs; (c) SNLI is balanced "
    "~33/33/33 by construction, so it can be subsampled to ANY target prevalence -- including "
    "one matched to the clinical pipeline's own ~4% -- which the clinical data, fixed at one "
    "prevalence point, cannot do.\n"
)
lines.append(
    "Four sweep pools (used for the main model x prompt x prevalence grid), N=100 each, "
    "disjoint across levels, built by `build_pools.py` with `np.random.default_rng(42)`. A "
    "second, larger, also-disjoint set of pools (N=300) was built for the distillation "
    "supplement (see §4) after the N=100 student turned out too small to fit."
)
lines.append("| target prevalence | n (sweep) | n_pos (sweep) | n (distill supp.) | n_pos (distill supp.) |")
lines.append("|---|---|---|---|---|")
for lvl in LEVELS:
    lines.append(f"| {lvl:.0%} | 100 | {round(lvl*100)} | 300 | {round(lvl*300)} |")
lines.append("")

lines.append("## 1. Prevalence sweep -- does the labeller's marginal track true prevalence?\n")
lines.append(
    "`positive rate (decided)` = contradiction calls / (contradiction+not_contradiction "
    "calls), i.e. excluding ABSTAIN and UNPARSEABLE from the denominator. `over-call ratio` "
    "= positive rate / true pool prevalence, the same quantity C20/C23 report against the "
    "4.0% clinical base rate. UNPARSEABLE is 0 everywhere in this run (see full CSVs in "
    "`results/raw_predictions/`) -- the C10 default-to-positive bug class does not arise "
    "here either.\n"
)
for model in MODELS:
    lines.append(f"### {model}\n")
    lines.append("| prompt | true prevalence | n | pos | neg | abstain | UNPARSEABLE | positive rate (decided) | over-call ratio |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for prompt in ["A_bare", "B_baserate_abstain"]:
        for lvl in LEVELS:
            m = sweep["results"][model][prompt][str(lvl)]
            lines.append(
                f"| {PROMPT_LABEL[prompt]} | {lvl:.0%} | {m['n']} | {m['n_contradiction']} | "
                f"{m['n_not_contradiction']} | {m['n_abstain']} | {m['n_unparseable']} "
                f"({m['unparseable_rate']:.1%}) | {m['positive_rate_of_decided']:.4f} | "
                f"{m['over_call_ratio']:.2f}x |"
            )
    lines.append("")

lines.append("### Calibration slope: positive rate vs. true prevalence (OLS across the 4 levels)\n")
lines.append(
    "Slope=1 would mean the labeller's marginal tracks true prevalence exactly (perfect "
    "calibration); slope=0 would mean the marginal is completely flat / prevalence-blind, "
    "which is what a 21x-style over-call at a SINGLE clinical prevalence point is consistent "
    "with but cannot itself establish. This is the direct answer to the open question this "
    "experiment was built to ask.\n"
)
lines.append("| model | prompt | slope | intercept | positive rates at 2%/5%/10%/25% |")
lines.append("|---|---|---|---|---|")
for model in MODELS:
    for prompt in ["A_bare", "B_baserate_abstain"]:
        xs = np.array(LEVELS)
        ys = np.array([sweep["results"][model][prompt][str(l)]["positive_rate_of_decided"] for l in LEVELS])
        slope, intercept = np.polyfit(xs, ys, 1)
        lines.append(
            f"| {model} | {PROMPT_LABEL[prompt]} | {slope:.3f} | {intercept:.3f} | "
            f"{', '.join(f'{y:.3f}' for y in ys)} |"
        )
lines.append("")

lines.append("## 2. Accuracy against SNLI gold (decided rows only)\n")
for model in MODELS:
    lines.append(f"### {model}\n")
    lines.append("| prompt | true prevalence | recall | precision | accuracy | TP | FP | FN | TN |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for prompt in ["A_bare", "B_baserate_abstain"]:
        for lvl in LEVELS:
            m = sweep["results"][model][prompt][str(lvl)]
            c = m["confusion_decided_only"]
            def fmt(x):
                return "nan" if x != x else f"{x:.3f}"
            lines.append(
                f"| {PROMPT_LABEL[prompt]} | {lvl:.0%} | {fmt(m['recall_decided'])} | "
                f"{fmt(m['precision_decided'])} | {fmt(m['accuracy_decided'])} | "
                f"{c['TP']} | {c['FP']} | {c['FN']} | {c['TN']} |"
            )
    lines.append("")

lines.append("## 3. Distillation at N=100 -- and why the hard-threshold student is degenerate\n")
lines.append(
    "TF-IDF (1-2gram) + logistic regression, trained on the LLM's own decided-row labels, "
    "5-fold (or fewer, if a class is too small) stratified cross-validated out-of-fold "
    "predictions. ABSTAIN/UNPARSEABLE rows are dropped from student training (count "
    "reported). **At N=100, the hard (0.5-threshold) student collapses to predicting "
    "not_contradiction for every out-of-fold item, in every combo that fit at all** -- there "
    "isn't enough signal for 3000-dim TF-IDF + LR to cross the decision boundary from ~64-80 "
    "training rows per fold. This is reported as `degenerate_hard_threshold=True` rather than "
    "silently treated as a real zero. The **soft marginal** (mean out-of-fold P(contradiction), "
    "threshold-independent) does not have this problem and is the number to read.\n"
)
for model in MODELS:
    lines.append(f"### {model}\n")
    lines.append("| prompt | prevalence | n decided | dropped | teacher marginal | student soft marginal | student hard marginal | true prevalence | closer (soft) |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for prompt in ["A_bare", "B_baserate_abstain"]:
        for lvl in LEVELS:
            d = distill.get(model, {}).get(prompt, {}).get(str(lvl))
            if d is None:
                lines.append(f"| {PROMPT_LABEL[prompt]} | {lvl:.0%} | -- | -- | -- | -- | -- | -- | (no record) |")
                continue
            if not d["fit"] and "teacher_marginal" not in d:
                lines.append(f"| {PROMPT_LABEL[prompt]} | {lvl:.0%} | -- | -- | -- | -- | -- | -- | NOT FIT: {d['reason']} |")
                continue
            soft = d.get("student_marginal_oof_soft")
            soft_s = f"{soft:.3f}" if soft is not None else "n/a (degenerate teacher label)"
            closer_soft = d.get("closer_to_soft", d.get("closer_to", "n/a"))
            lines.append(
                f"| {PROMPT_LABEL[prompt]} | {lvl:.0%} | {d['n_decided']} | "
                f"{d['n_dropped_abstain_or_unparseable']} | {d['teacher_marginal']:.3f} | "
                f"{soft_s} | {d['student_marginal_oof']:.3f} | {d['true_prevalence_decided']:.3f} | "
                f"{closer_soft} |"
            )
    lines.append("")

lines.append("## 4. Distillation supplement at N=300 -- well-powered test of prevalence inheritance\n")
lines.append(
    "`llama3:8b`, prompt A (bare) only -- the direct analog of the clinical `llama_baseline` "
    "reference -- re-run at N=300 per prevalence level on pools disjoint from every pool "
    "above (`run_distillation_supplement.py`). Same student (TF-IDF+LR, 5-fold CV). At N=300 "
    "the hard-threshold student is STILL degenerate at 3 of 4 levels (see "
    "`degenerate_hard_threshold`), which is itself informative: the clinical distillation had "
    "1,679 training rows, not 100-300, and this suggests hard-threshold reproduction needs "
    "that much data even when the underlying probability estimate (below) is already very "
    "accurate. The soft marginal is not threshold-limited and is the number that answers the "
    "task's question directly.\n"
)
lines.append("| prevalence | n decided | teacher marginal | student soft marginal | true prevalence | \\|soft − teacher\\| | \\|soft − true\\| | closer to |")
lines.append("|---|---|---|---|---|---|---|---|")
for lvl in LEVELS:
    d = distill_supp[str(lvl)]
    if not d["fit"]:
        lines.append(f"| {lvl:.0%} | -- | -- | -- | -- | -- | -- | NOT FIT: {d['reason']} |")
        continue
    soft = d["student_marginal_oof_soft"]
    teacher = d["teacher_marginal"]
    true = d["true_prevalence_decided"]
    lines.append(
        f"| {lvl:.0%} | {d['n_decided']} | {teacher:.3f} | {soft:.3f} | {true:.3f} | "
        f"{abs(soft-teacher):.3f} | {abs(soft-true):.3f} | {d['closer_to_soft']} |"
    )
lines.append("")
lines.append(
    "At every prevalence level, the well-powered (N=300) student's soft marginal is within "
    "0.005 of the teacher's own marginal and far from true prevalence at 2%, 5%, and 25% -- "
    "the clean, well-powered version of the clinical claim (student reproduced teacher's "
    "predicted-positive rate to within 1.5pp; here, within 0.1-0.5pp). **The mechanism "
    "replicates on non-clinical text**: a student trained on a rare-class LLM label learns "
    "the LLM's marginal, not the ground truth, when the LLM's marginal and the ground truth "
    "diverge.\n"
)

lines.append("## Verdict\n")
lines.append(
    "**(1) Gross over-calling magnitude is domain-specific, not task-generic.** The clinical "
    "pipeline over-called by 15.8x-23.6x at 4.0% true prevalence, with absolute "
    "predicted-positive rates of 63-95%. On SNLI contradiction judgments at the SAME "
    "prevalence range (2-25%), absolute positive-call rates never exceed ~17% for any model, "
    "prompt, or prevalence level tested (24 combos, 2,400 generations) -- over-call ratios "
    "range from 0.0x to 3.5x, an order of magnitude below the clinical floor of 7x that C23 "
    "established across every model tested there (including GPT-4o). At the two higher "
    "prevalence levels (10%, 25%), most model/prompt combinations UNDER-call rather than "
    "over-call. A 15-24x-style blanket-positive failure mode does not reproduce on non-clinical "
    "text with the same task shape.\n"
)
lines.append(
    "**(2) The marginal is prevalence-INSENSITIVE but not fully flat -- a weaker version of "
    "the same failure mode.** Calibration slopes (positive rate vs. true prevalence, ideal=1) "
    "range from 0.028 (llama3:8b, base-rate-aware prompt) to 0.62 (granite3.3:8b, "
    "base-rate-aware prompt); every slope is well below 1, meaning every model under-responds "
    "to a 12.5x swing in true prevalence. llama3:8b is closest to the clinical "
    "\"prevalence-blind\" signature (slope 0.03-0.06, near-flat); granite3.3:8b partially "
    "tracks prevalence (slope 0.47-0.62) and is the best-calibrated of the three. So the "
    "*mechanism* the clinical single-point result could not establish -- a marginal that "
    "under-weights true prevalence -- does generalize to non-clinical text, but the "
    "*magnitude* of the resulting miscalibration on this task is far smaller than in the "
    "clinical pipeline, because the non-clinical intercept sits near 5-15% rather than "
    "near 85-95%.\n"
)
lines.append(
    "**(3) Prompting that states the true base rate and permits abstention reduces, but does "
    "not eliminate, the effect** -- consistent with C23's finding that prompt moves the "
    "clinical number by ~2x within one model. Here the base-rate+abstain prompt lowers the "
    "positive rate at every prevalence level for every model versus the bare prompt (e.g. "
    "llama3:8b at 5%: 17.0% to 11.0%), and qwen3:8b abstains on all rows it would otherwise "
    "call positive at 2% and 5% prevalence, but the calibration slope is not consistently "
    "improved by the abstention prompt: it gets FLATTER for llama3:8b (0.058 to 0.028) and "
    "qwen3:8b (0.302 to 0.120), and only improves for granite3.3:8b (0.466 to 0.620) -- "
    "prompt is a real lever but not a reliable fix, on either dataset.\n"
)
lines.append(
    "**(4) The prevalence-inheritance / distillation mechanism DOES replicate, cleanly, "
    "on non-clinical text**, and does so more precisely than the clinical result (within "
    "0.1-0.5pp here vs. 1.5pp there, at N=300). A TF-IDF+LR student trained on LLM labels "
    "learns the teacher's marginal, not the ground truth, whenever the two diverge -- this "
    "held in all 22 fittable N=100 combos and all 4 N=300 combos. This is the part of the "
    "clinical finding that is a property of the TASK (distillation from a biased weak "
    "labeller), not of clinical text specifically.\n"
)
lines.append(
    "**Overall: the effect is a blend, and the thesis should say so precisely rather than "
    "picking one side.** The over-calling MAGNITUDE (7-24x, absolute rates near 90%) is "
    "specific to the clinical adjudication task -- it did not reproduce at anything close to "
    "that scale on structurally identical non-clinical judgments at matched prevalence. But "
    "the underlying MECHANISMS -- (a) a marginal that under-responds to true prevalence, and "
    "(b) a distillation student that inherits the teacher's marginal rather than the truth -- "
    "both generalize cleanly to non-clinical text. The clinical pipeline's contribution is "
    "not \"LLMs over-call rare classes\" in the abstract (that is real but modest, 1-3.5x, on "
    "SNLI); it is that something about clinical medication-contradiction adjudication "
    "specifically pushes the same mechanism to an extreme absolute rate (85-95% positive) "
    "that this general-purpose NLI task never approaches.\n"
)

with open(os.path.join(RESULTS_DIR, "REPORT.md"), "w") as f:
    f.write("\n".join(lines))
print(f"Wrote {os.path.join(RESULTS_DIR, 'REPORT.md')}")
