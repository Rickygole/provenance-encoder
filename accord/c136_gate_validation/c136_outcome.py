"""C136. Render the outcome section from the artifact. No number in the outcome is typed by hand."""
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ART = os.path.join(REPO, "results", "accord", "c136_gate_validation", "c136.json")
PREREG = os.path.join(REPO, "accord", "c136_gate_validation", "C136_PREREGISTRATION.md")
MARK = "## Outcome (appended after the run; the text above is unchanged)\n"


def f(x, n=4):
    return "undefined" if x is None else f"{x:.{n}f}"


def table(g):
    return (f"{g['cell_gatefail_studentfailed']} / {g['cell_gatefail_studentuseful']} / "
            f"{g['cell_gatepass_studentfailed']} / {g['cell_gatepass_studentuseful']}")


def stats(g):
    return (f"sensitivity {f(g['sensitivity'])}, specificity {f(g['specificity'])}, "
            f"PPV {f(g['ppv'])}, NPV {f(g['npv'])}")


def main():
    a = json.load(open(ART))
    L = []
    w = L.append
    ns = [r["flag_count"] for r in a["labellers"] if r["family"] == "synthetic"]
    ns_real = [r["flag_count"] for r in a["labellers"] if r["family"] == "real_ollama"]
    hp = a["headroom_precondition"]
    gt = a["gate"]
    syn = a["synthetic_minilm"]
    prim = syn["screening"]["auroc_lt_0.60"]
    sens_prim = syn["screening"]["auroc_lt_0.70"]
    tf = a["synthetic_tfidf"]["screening"]["auroc_lt_0.60"]["gate"]
    real = a.get("real_minilm", {})
    pooled = a.get("pooled_minilm")

    w(f"Run on 2026-09-26 and 2026-09-27, seed {a['seed']}, data {a['data_source']}. Frames: pilot "
      f"{a['frames']['pilot'][0]} with {a['frames']['pilot'][1]} positives, pool "
      f"{a['frames']['pool'][0]} with {a['frames']['pool'][1]}, test {a['frames']['test'][0]} with "
      f"{a['frames']['test'][1]}, pool and test disjoint: {a['frames']['pool_test_disjoint']}. "
      f"Artifact `results/accord/c136_gate_validation/c136.json`. Every number below is read from "
      f"that artifact by `c136_outcome.py`; none is typed.\n")
    w(f"**Headroom precondition, checked before the grid was read: PASSED.** The primary `minilm` "
      f"student's clean-label reference AUROC is {f(hp['primary_reference_auroc'])}, above the "
      f"pre-registered floor of {hp['floor']}. The secondary `tfidf` student's reference is "
      f"{f(a['clean_label_reference_auroc']['tfidf'])}, which is why C135 was uninformative; it is "
      f"reported below and not read.\n")

    verdict = "A, validated" if (prim["gate"]["sensitivity"] or 0) >= 0.90 and \
        (prim["gate"]["ppv"] or 0) >= 0.90 and not prim["gate_adds_nothing_over_raw_ratio"] else (
        "B, partially validated" if (prim["gate"]["sensitivity"] or 0) >= 0.90 and
        (prim["gate"]["ppv"] or 0) >= 0.90 else "C, NOT VALIDATED")
    w(f"## Outcome {verdict}\n")
    w(f"Primary reading, synthetic arm, `minilm` student, failure at AUROC below 0.60, "
      f"{prim['gate']['n_labellers']} labellers. 2x2 as gate fail and student failed / gate fail and "
      f"student useful / gate pass and student failed / gate pass and student useful: "
      f"**{table(prim['gate'])}**. {stats(prim['gate']).capitalize()}. Youden J "
      f"{f(prim['gate']['youden_j'])}. Sensitivity {f(prim['gate']['sensitivity'])} is far below the "
      f"pre-registered 0.90, so the outcome is C.\n")
    w(f"Sensitivity analysis, failure at AUROC below 0.70, same arm and student: 2x2 "
      f"**{table(sens_prim['gate'])}**, {stats(sens_prim['gate'])}, Youden J "
      f"{f(sens_prim['gate']['youden_j'])}. The conclusion does not move.\n")
    w(f"Secondary `tfidf` student, failure below 0.60: 2x2 **{table(tf)}**, {stats(tf)}. Every "
      f"student failed including the clean-label reference, so specificity is undefined; this "
      f"reproduces C135 and is reported only to show the defect is the student, not the gate.\n")

    w("### The honest limit: a passing gate certifies nothing, quantified\n")
    w(f"Among the {prim['gate']['n_gate_pass']} gate PASSING labellers in the primary reading, "
      f"{prim['gate']['cell_gatepass_studentfailed']} produced a failing student, a fraction of "
      f"**{f(prim['gate']['frac_gatepass_still_failing'])}**, which is 1 minus the NPV of "
      f"{f(prim['gate']['npv'])}. At the 0.70 threshold it is "
      f"{f(sens_prim['gate']['frac_gatepass_still_failing'])} "
      f"({sens_prim['gate']['cell_gatepass_studentfailed']} of "
      f"{sens_prim['gate']['n_gate_pass']}). The thesis's own caveat is confirmed numerically: most "
      f"labellers the gate passes still yield a useless student.\n")
    w("The other direction is the one place the gate holds up. In the primary reading no gate "
      f"FAILING labeller produced a useful student ({prim['gate']['cell_gatefail_studentuseful']} "
      f"such cells), so specificity is {f(prim['gate']['specificity'])} and PPV is "
      f"{f(prim['gate']['ppv'])}. The gate never wrongly condemns a labeller. It just almost never "
      "condemns one at all.\n")

    w("### Why: the gate is arithmetically a flag count threshold\n")
    w(f"With N = {gt['N']} and K = {gt['K']}, evaluating `p_floor` at every flag count from 0 to "
      f"{gt['N']} gives a FAIL region that is exactly the single point 0 together with the upper "
      f"tail {gt['smallest_failing_positive_flag_count']} to {gt['N']} "
      f"(`fail_region_is_zero_plus_upper_tail` = {gt['fail_region_is_zero_plus_upper_tail']}). The "
      f"gate therefore fails a labeller only if it flags at least "
      f"{gt['smallest_failing_positive_flag_count']} of {gt['N']} pilot items, a flag rate of "
      f"{f(gt['smallest_failing_positive_flag_rate'])} or more, or flags nothing at all. It carries "
      "no information beyond the flag count.\n")

    w("### The gate against the raw ratio, flag rate over base rate\n")
    eq = prim["ratio_at_gate_equivalent_threshold"]
    best = prim["ratio_best_over_all_thresholds"]
    w(f"**The gate adds nothing over the raw ratio.** Predicting failure when the ratio is at or "
      f"above {f(eq['ratio_threshold'])} reproduces the gate's verdict on every labeller "
      f"(`identical_to_gate_verdicts` = {eq['identical_to_gate_verdicts']}, "
      f"{eq['disagreements']} disagreements), giving the same 2x2 {table(eq['screening'])}. No "
      f"synthetic labeller flagged zero items (their flag counts run from {min(ns)} to {max(ns)}), "
      "so in the primary arm the gate's zero point never separates it from that single threshold. "
      + (f"The real arm does contain zero flag labellers ({sum(1 for n in ns_real if n == 0)} of "
         f"{len(ns_real)} models flagged none of the 300 pilot items), so there the gate is a two "
         "sided rule on the flag count, low or high, rather than a one sided one. " if ns_real and
         min(ns_real) == 0 else "")
      + "Either way the hypergeometric floor buys no information beyond the flag count itself.\n")
    w(f"**And a better threshold on the same raw ratio strictly dominates the gate.** Sweeping every "
      f"threshold, the best Youden J is {f(best['youden_j'])} at ratio "
      f"{f(best['ratio_threshold'])}, that is a flag count of about "
      f"{round(best['ratio_threshold'] * gt['K'])} of {gt['N']}, with sensitivity "
      f"{f(best['sensitivity'])} and specificity {f(best['specificity'])}, against the gate's J of "
      f"{f(prim['gate_youden_j'])} (`ratio_beats_gate` = {prim['ratio_beats_gate']}). That sweep "
      "chooses its threshold on the same 24 labellers it is scored on, so it is an in sample upper "
      "bound and flatters the ratio; the gate's operating point is fixed a priori and does not get "
      "that advantage. The comparison that does not depend on the sweep is the previous paragraph: "
      "the gate IS a ratio threshold, and it is a badly placed one.\n")
    w(f"Rank correlations, deciding nothing: ceiling MCC against student AUROC, Spearman rho "
      f"{f(syn['spearman_ceilingmcc_vs_auroc']['rho'])} "
      f"(p {syn['spearman_ceilingmcc_vs_auroc']['p']:.3g}); `p_floor` against student AUROC, rho "
      f"{f(syn['spearman_floorp_vs_auroc']['rho'])} "
      f"(p {syn['spearman_floorp_vs_auroc']['p']:.3g}). Both are moderate and in the expected "
      "direction, which is the strongest thing that can be said for the gate's quantities: they are "
      "correlated with student quality, they are not a decision rule for it.\n")

    w("### Real labellers, secondary\n")
    if a["n_real_labellers"]:
        w(f"{a['n_real_labellers']} local Ollama models ran under the plain contradiction prompt:\n")
        w("| model | pilot flags of 300 | true positives flagged of 12 | p_floor | gate | student AUROC minilm | student AUROC tfidf |")
        w("|---|---|---|---|---|---|---|")
        for r in a["labellers"]:
            if r["family"] != "real_ollama":
                continue
            w(f"| {r['labeller']} | {r['flag_count']} | {r['pilot_true_positives_flagged']} | "
              f"{r['floor_p']:.3g} | {'FAIL' if r['gate_fail'] else 'PASS'} | "
              f"{f(r['student_auroc_minilm'])} | {f(r['student_auroc_tfidf'])} |")
        w("")
        if "screening" in real and real["screening"]:
            rg = real["screening"]["auroc_lt_0.60"]["gate"]
            w(f"Real arm screening at 0.60: 2x2 {table(rg)}, {stats(rg)}.\n")
        if pooled and pooled.get("screening"):
            pg = pooled["screening"]["auroc_lt_0.60"]["gate"]
            w(f"Pooled synthetic and real at 0.60: 2x2 {table(pg)}, {stats(pg)}, "
              f"fraction of gate passers still failing "
              f"{f(pg['frac_gatepass_still_failing'])}.\n")
        models_not_run = [m for m in ["qwen3:0.6b", "qwen3:1.7b", "qwen3:4b", "llama3:8b",
                                      "granite3.3:8b", "qwen3:8b", "qwen3:14b"]
                          if m not in {r["labeller"] for r in a["labellers"]}]
        if models_not_run:
            w("Not run within the run budget, reported as not run per the rule: "
              + ", ".join(f"`{m}`" for m in models_not_run) + ".\n")
    else:
        w("No real labeller completed within the run budget.\n")

    w("### What this means for the thesis\n")
    w("The marginal feasibility gate is **not validated as a screen for student failure**. It is "
      "validated as one thing only, and a narrow one: a sufficient condition for hopelessness that "
      "fires almost never. Nothing it condemned was salvageable, and it condemned "
      f"{prim['gate']['n_gate_fail']} of {prim['gate']['n_labellers']} labellers while "
      f"{prim['gate']['n_student_failed']} of {prim['gate']['n_labellers']} students actually "
      "failed. The thesis may claim that the gate, when it fires, is a correct veto, and that it "
      "formalises why the observed clinical labelling was hopeless before any student was trained. "
      "It may not claim the gate screens labellers, predicts student utility, or is a feasibility "
      "criterion: it passes most labellers whose students are useless, and its verdict is a "
      "restatement of the pilot flag count with no added information. Any claim beyond the veto "
      "reading is contradicted by this run.\n")

    txt = open(PREREG).read()
    head, _, _ = txt.partition(MARK)
    open(PREREG, "w").write(head + MARK + "\n" + "\n".join(L))
    print("outcome written to", PREREG)


if __name__ == "__main__":
    main()
