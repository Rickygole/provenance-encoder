"""C138d: independent recount of the uncorrected test family.

Chapter 4 (sec:r-multiplicity) states the uncorrected family has 23 tests and
enumerates five blocks. This script re-enumerates every p value in the thesis
that is read against a threshold, applies the chapter's own membership rule
("a test belongs to this family by its inferential target, not by what a run
was commissioned for"), and reports the count.

Every p value is read from the artifact behind it, not typed, except where the
thesis asserts a comparison without printing the value; those are marked
value_printed_in_thesis = false and the assertion's location is recorded.

Aggregates only. No note text, no HADM_ID, no SUBJECT_ID. No model inference.
"""
from __future__ import annotations

import collections
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
RES = ROOT / "results/accord"
OUT_DIR = RES / "promised_analyses"
OUT = OUT_DIR / "c138d_uncorrected_family_recount.json"

ALPHA = 0.05


def load(rel):
    return json.loads((RES / rel).read_text())


def dig(obj, dotted):
    for part in dotted.split("."):
        obj = obj[part]
    return obj


def main():
    reg = load("stats/test_register.json")
    gate = load("stats/marginal_feasibility_gate.json")
    c109 = load("prompt_sensitivity/c109/analysis.json")
    step2 = load("step2_signal_check.json")

    reg_by_id = {t["id"]: t for t in reg["tests"]}
    fam_ids = reg["uncorrected_family"]["ids"]

    entries = []

    def add(**kw):
        kw.setdefault("value_printed_in_thesis", True)
        entries.append(collections.OrderedDict(sorted(kw.items())))

    # ---- Block A: the 23 the chapter already counts -----------------------
    for tid in fam_ids:
        t = reg_by_id[tid]
        add(id=tid, p=t["p"], test=t["test"], target=t["what"],
            source=t["source"], thesis_location=t.get("where"),
            read_against="alpha=0.05, uncorrected",
            preregistration=None, in_chapter_count=True, verdict="IN_FAMILY",
            block="A_counted_by_chapter")

    # ---- Block B1: the contaminated-run threshold sweep -------------------
    # msu_ch4_results.tex:924. Post hoc; the subsection says the sweep "was run
    # on the contaminated released run only" and reports two permutation p
    # values against the same maximisation on random scores, then draws the null
    # verdict "no threshold rescues the detector". Its inferential target is a
    # thresholded decision on the contaminated run's two score columns, which
    # the thesis itself distinguishes from a ranking (msu_ch4_results.tex:358
    # to 361 and the tab:feasgate note), so it is not the same question as
    # mw_contaminated.
    for name, val, what in (
        ("threshsweep_contra_prob_g253", 0.8195, "best MCC at any cutoff on the contaminated run's contradiction probability, G_253, against the same maximisation on random scores"),
        ("threshsweep_harm_score_g253", 0.7642, "best MCC at any cutoff on the contaminated run's harm score, G_253, against the same maximisation on random scores"),
    ):
        add(id=name, p=val, test="Permutation test (maximised MCC on random scores)",
            target=what,
            source="msu_ch4_results.tex:924 (prose; no results/ artifact holds this p value)",
            thesis_location="sec:r-ranking, msu_ch4_results.tex:923-926",
            read_against="alpha=0.05, uncorrected, via the null verdict 'no threshold rescues the detector'",
            preregistration=None, in_chapter_count=False, verdict="IN_FAMILY",
            block="B1_threshold_sweep_missed_by_chapter")

    # ---- Block B2: the four gate rows where the floor p IS the observed test
    # The tab:feasgate note says the floor p is "a precondition rather than a
    # test unless observed MCC equals ceiling MCC, where it is also the exact
    # test observed". The multiplicity subsection excludes all floor p values
    # wholesale. The two statements disagree about the same numbers.
    for rid, r in gate["rows"].items():
        if not r.get("attains_ceiling"):
            continue
        add(id=f"feasgate_observed_exact_{rid}",
            p=r["observed_p_one_sided_hypergeom"],
            test="Hypergeometric exact, one-sided",
            target=f"hard-label agreement of {r['model']} {r['strategy']} on {r['basis']} against chance; observed MCC equals ceiling MCC so the floor p is the observed exact test",
            source="results/accord/stats/marginal_feasibility_gate.json::rows." + rid + ".observed_p_one_sided_hypergeom",
            thesis_location="tab:feasgate, msu_ch4_results.tex:409-425, Rejectable column",
            read_against="alpha=0.05, the Rejectable column",
            preregistration=None, in_chapter_count=False, verdict="IN_FAMILY_ON_THE_TABLE_NOTE_READING",
            block="B2_gate_rows_attaining_ceiling")

    # ---- Block B3: the descriptive Granite 3.3 8B prompt-wording cells -----
    # Registered as descriptive only, so covered by no decision rule, and read
    # against the uncorrected 0.05 level at msu_ch4_results.tex:704-706.
    for wording, cell in c109["per_model_prompt"]["granite3.3:8b"].items():
        add(id=f"perm_c109_granite33_8b_{wording}", p=cell["permutation_p_253"],
            test="Permutation test (label shuffle)",
            target=f"Granite 3.3 8B AUROC on G_253 under wording {wording}, against its own label-permutation null",
            source=f"results/accord/prompt_sensitivity/c109/analysis.json::per_model_prompt.granite3.3:8b.{wording}.permutation_p_253",
            thesis_location="msu_ch4_results.tex:704-706",
            read_against="alpha=0.05, uncorrected; the chapter says these cells are 'uncorrected across models'",
            preregistration="C109, registered as descriptive only, so no decision rule covers them",
            in_chapter_count=False,
            verdict="IN_FAMILY_DESCRIPTIVE_CELLS_HAVE_NO_COVERING_RULE",
            value_printed_in_thesis=wording in ("P_rev_agree", "P_rev_compatible"),
            block="B3_descriptive_granite_wordings")

    # ---- Block C: read against a threshold, covered by a pre-registration,
    # and absent from the 47-test register, so the register is incomplete.
    for name, val, src, loc, prereg, what in (
        ("perm_c112_rules34_rescore_g108", 0.1910,
         "msu_ch4_results.tex:897 (prose)", "msu_ch4_results.tex:896-898", "C112 Rules 3 and 4 rescore",
         "label-permutation p of the decontaminated ensemble AUROC on G_108"),
        ("rankperm_c116_tfidf_minus_decontam", 0.17875,
         "msu_ch4_results.tex:1091 (prose)", "msu_ch4_results.tex:1090-1092", "C116 rank-based paired test",
         "rank-permuted paired difference, TF-IDF minus decontaminated ensemble, G_253"),
        ("rankperm_c116_nli_minus_decontam", 0.1452,
         "msu_ch4_results.tex:1092 (prose)", "msu_ch4_results.tex:1090-1092", "C116 rank-based paired test",
         "rank-permuted paired difference, NLI minus decontaminated ensemble, G_253"),
        ("rankperm_c116_tfidf_minus_nli", 0.96465,
         "msu_ch4_results.tex:1092 (prose)", "msu_ch4_results.tex:1090-1092", "C116 rank-based paired test",
         "rank-permuted paired difference, TF-IDF minus NLI, G_253"),
        ("perm_nli_ap_g253", 0.01335,
         "msu_ch4_results.tex:1078 (prose)", "msu_ch4_results.tex:1076-1080", "C107 bag-of-words robustness checks",
         "NLI average precision against its own label-permutation null, G_253"),
        ("perm_nli_ap_g299", 0.0120,
         "msu_ch4_results.tex:1078 (prose)", "msu_ch4_results.tex:1076-1080", "C107 bag-of-words robustness checks",
         "NLI average precision against its own label-permutation null, G_299"),
        ("perm_decontam_ap_g253", 0.4629,
         "msu_ch4_results.tex:1080 (prose)", "msu_ch4_results.tex:1079-1080", "C107 bag-of-words robustness checks",
         "decontaminated ensemble average precision against its own label-permutation null, G_253"),
        ("mw_nestedcv_goldonly_g253", 0.9666,
         "msu_ch4_results.tex:1096 (prose)", "msu_ch4_results.tex:1093-1097", "pre-registered nested cross-validation variant",
         "one-sided Mann-Whitney on the gold-only nested-CV out-of-fold scores, G_253"),
        ("hypergeom_c120_exactjoin_g299", 0.5347,
         "accord/c120_exact_join/C120_PREREGISTRATION.md outcome record", "msu_ch4_results.tex:330", "C120 exact-join agreement",
         "deployed weak labeller's hard-label agreement against chance, G_299"),
        ("hypergeom_c120_exactjoin_g253", 0.5722,
         "accord/c120_exact_join/C120_PREREGISTRATION.md outcome record", "msu_ch4_results.tex:333", "C120 exact-join agreement",
         "deployed weak labeller's hard-label agreement against chance, G_253"),
        ("perm_c109_llama3_8b_reversed_rerun", c109["per_model_prompt"]["llama3:8b"]["P_reversed_rerun"]["permutation_p_253"],
         "results/accord/prompt_sensitivity/c109/analysis.json::per_model_prompt.llama3:8b.P_reversed_rerun.permutation_p_253",
         "msu_ch4_results.tex:695-697", "C109 prompt-wording confirmation",
         "Llama 3 8B AUROC on G_253 under the identical rerun, against its label-permutation null"),
    ):
        add(id=name, p=val, test="see target", target=what, source=src,
            thesis_location=loc, read_against="alpha=0.05 or a registered rule threshold",
            preregistration=prereg, in_chapter_count=False,
            verdict="OUT_OF_FAMILY_PREREGISTERED_BUT_MISSING_FROM_THE_47_TEST_REGISTER",
            block="C_preregistered_but_unregistered")

    # ---- Block D: read against no threshold, or belonging to no family -----
    for name, val, src, loc, why in (
        ("superseded_paired_tfidf_vs_decontam", 0.00005, "msu_ch4_results.tex:1083 (prose)",
         "msu_ch4_results.tex:1082-1086", "superseded by the C116 rank-based paired test; printed but carries no verdict"),
        ("superseded_paired_nli_vs_decontam", 0.10035, "msu_ch4_results.tex:1084 (prose)",
         "msu_ch4_results.tex:1082-1086", "superseded by the C116 rank-based paired test; printed but carries no verdict"),
        ("montecarlo_c120_randomjoin_g299", 0.5442, "accord/c120_exact_join/C120_PREREGISTRATION.md outcome record",
         "msu_ch4_results.tex:330-331", "same inferential target as hypergeom_c120_exactjoin_g299; counts once by the chapter's own rule"),
        ("montecarlo_c120_randomjoin_g253", 0.5711, "accord/c120_exact_join/C120_PREREGISTRATION.md outcome record",
         "msu_ch4_results.tex:334", "same inferential target as hypergeom_c120_exactjoin_g253; counts once by the chapter's own rule"),
        ("mw_contaminated_step2_duplicate", dig(step2, "accord_contra_prob_distribution_by_group.mann_whitney_u_p_value_pos_vs_neg"),
         "results/accord/step2_signal_check.json::accord_contra_prob_distribution_by_group.mann_whitney_u_p_value_pos_vs_neg",
         "tab:decontam", "already counted as mw_contaminated in block A; listed so the contaminated run is not counted twice"),
    ):
        add(id=name, p=val, test="see why", target=why, source=src, thesis_location=loc,
            read_against="none, or the same target as a counted test",
            preregistration=None, in_chapter_count=False,
            verdict="OUT_OF_FAMILY", block="D_no_family_or_duplicate")

    # ---- counts -----------------------------------------------------------
    by_block = collections.Counter(e["block"] for e in entries)
    n_a = by_block["A_counted_by_chapter"]
    n_b1 = by_block["B1_threshold_sweep_missed_by_chapter"]
    n_b2 = by_block["B2_gate_rows_attaining_ceiling"]
    n_b3 = by_block["B3_descriptive_granite_wordings"]

    res = collections.OrderedDict()
    res["provenance"] = {
        "generated_by": "accord/promised_analyses/c138d_uncorrected_family_recount.py",
        "membership_rule_quoted": "msu_ch4_results.tex:1208, 'A test belongs to this family by its inferential target, not by what a run was commissioned for'",
        "alpha": ALPHA,
        "no_model_inference": True,
    }
    res["chapter_stated_count"] = reg["uncorrected_family"]["n_tests"]
    res["counts"] = collections.OrderedDict([
        ("block_A_counted_by_chapter", n_a),
        ("block_B1_threshold_sweep", n_b1),
        ("block_B2_gate_rows_attaining_ceiling", n_b2),
        ("block_B3_descriptive_granite_wordings", n_b3),
        ("minimum_correct_count_A_plus_B1", n_a + n_b1),
        ("count_honouring_the_tab_feasgate_note", n_a + n_b1 + n_b2),
        ("count_honouring_both_the_note_and_the_descriptive_only_registration", n_a + n_b1 + n_b2 + n_b3),
    ])
    res["bonferroni_thresholds"] = {
        str(n): ALPHA / n for n in (n_a, n_a + n_b1, n_a + n_b1 + n_b2, n_a + n_b1 + n_b2 + n_b3)
    }
    res["nominally_significant_among_the_additions"] = [
        e["id"] for e in entries
        if e["block"].startswith(("B1", "B2", "B3")) and isinstance(e["p"], (int, float)) and e["p"] < ALPHA
    ]
    res["tests_belonging_to_no_family"] = [
        {"id": e["id"], "p": e["p"], "why": e["target"]} for e in entries if e["block"].startswith("D")
    ]
    res["register_completeness"] = {
        "n_tests_in_47_test_register": reg["n_tests"],
        "n_thesis_tests_read_against_a_threshold_absent_from_it": by_block["C_preregistered_but_unregistered"] + n_b1 + n_b2 + n_b3,
        "note": "results/accord/stats/test_register.json enumerates 47 tests but omits every entry in blocks B and C, so its count cannot move when a chapter adds a test of those kinds.",
    }
    res["entries"] = entries

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "entries"}, indent=1))
    print()
    for e in entries:
        if e["block"].startswith(("B", "C", "D")):
            pv = e["p"]
            print(f"{e['block'][:2]}  {e['id']:<48} p={pv:<10} {e['verdict']}")


if __name__ == "__main__":
    main()
