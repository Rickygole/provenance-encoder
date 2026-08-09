"""
Combine sweep_results.json + distillation_results.json into the final
markdown report for the prevalence-replication experiment.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")

with open(os.path.join(RESULTS_DIR, "sweep_results.json")) as f:
    sweep = json.load(f)
with open(os.path.join(RESULTS_DIR, "distillation_results.json")) as f:
    distill = json.load(f)

MODELS = sweep["models"]
LEVELS = sweep["prevalence_levels"]
PROMPT_LABEL = {"A_bare": "A (bare)", "B_baserate_abstain": "B (base-rate + abstain)"}

lines = []
lines.append("# Prevalence replication: is the clinical over-calling finding domain-specific or task-specific?\n")
lines.append(
    "Non-clinical replication of ACCORD C20/C23 (`accord/CORRECTIONS.md`): three 8B "
    "open-weights models, local via Ollama, over-called clinical medication-contradiction "
    "pairs at 15.8x-23.6x against a 4.0% pharmacist base rate, at a single fixed prevalence "
    "point. This experiment sweeps TRUE PREVALENCE (2%/5%/10%/25%) and PROMPT (bare vs "
    "base-rate-aware+abstention) on a public, non-clinical, human-labeled task with the same "
    "binary-contradiction structure, to separate \"a property of clinical adjudication\" from "
    "\"a property of asking an instruction-tuned model to make a rare-class call from short "
    "evidence.\"\n"
)

lines.append("## Data\n")
lines.append(
    "SNLI (Stanford NLI) premise/hypothesis pairs, test+validation splits, "
    "human-labeled (MTurk), downloaded from the public HF hub mirror. Binary task: "
    "contradiction (label==2) vs not_contradiction (entailment+neutral pooled). Four pools, "
    "N=100 each, disjoint across prevalence levels, built by `build_pools.py` with "
    "`np.random.default_rng(42)`.\n"
)
lines.append("| target prevalence | n | n_pos | achieved prevalence |")
lines.append("|---|---|---|---|")
for lvl in LEVELS:
    lines.append(f"| {lvl:.0%} | 100 | {round(lvl*100)} | {lvl:.4f} (exact by construction) |")
lines.append("")

lines.append("## 1. Prevalence sweep -- does the labeller's marginal track true prevalence?\n")
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

lines.append("## 3. Distillation -- does the student reproduce the teacher's marginal, or the true prevalence?\n")
lines.append(
    "TF-IDF (1-2gram) + logistic regression, trained on the LLM's own decided-row labels, "
    "5-fold (or fewer, if a class is too small) stratified cross-validated out-of-fold "
    "predictions. ABSTAIN/UNPARSEABLE rows are dropped from student training (count "
    "reported). Where the teacher label is near-degenerate (all/almost-all one class), the "
    "student's fit is not meaningful and this is reported explicitly rather than run anyway.\n"
)
for model in MODELS:
    lines.append(f"### {model}\n")
    lines.append("| prompt | prevalence | n decided | dropped | teacher marginal | student OOF marginal | true prevalence | closer to |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for prompt in ["A_bare", "B_baserate_abstain"]:
        for lvl in LEVELS:
            key_model = model
            d = distill.get(key_model, {}).get(prompt, {}).get(str(lvl))
            if d is None:
                lines.append(f"| {PROMPT_LABEL[prompt]} | {lvl:.0%} | -- | -- | -- | -- | -- | (no distillation record) |")
                continue
            if not d["fit"] and "teacher_marginal" not in d:
                lines.append(f"| {PROMPT_LABEL[prompt]} | {lvl:.0%} | -- | -- | -- | -- | -- | NOT FIT: {d['reason']} |")
                continue
            fit_note = "" if d["fit"] else " (degenerate teacher label, not cross-validated)"
            lines.append(
                f"| {PROMPT_LABEL[prompt]} | {lvl:.0%} | {d['n_decided']} | "
                f"{d['n_dropped_abstain_or_unparseable']} | {d['teacher_marginal']:.3f} | "
                f"{d['student_marginal_oof']:.3f}{fit_note} | {d['true_prevalence_decided']:.3f} | "
                f"{d.get('closer_to', 'n/a')} |"
            )
    lines.append("")

with open(os.path.join(RESULTS_DIR, "REPORT.md"), "w") as f:
    f.write("\n".join(lines))
print(f"Wrote {os.path.join(RESULTS_DIR, 'REPORT.md')}")
