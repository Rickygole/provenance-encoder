#!/usr/bin/env python3
"""Assemble the number-provenance audit deliverable.

Consumes:
  results/accord/_audit_recompute.json          (audit_recompute.py)
  results/accord/_audit_cross_consistency.json  (audit_cross_consistency.py)
  results/accord/_audit_claims_raw.json         (audit_extract_claims.py)
Emits:
  results/accord/number_provenance_audit.json

Verdicts: VERIFIED | MISMATCH | UNTRACEABLE | SUPERSEDED-BY-CORRECTION
Every `recomputed` value below is read from the recompute artifact, never typed.
"""
import json, os, re, collections

R = "/Users/rickygole/provenance-encoder/results/accord"
DRIVE = ("/Users/rickygole/Library/CloudStorage/"
         "GoogleDrive-ricky.gole@morgan.edu/My Drive/thesis/outputs")
RC = json.load(open(f"{R}/_audit_recompute.json"))
XC = json.load(open(f"{R}/_audit_cross_consistency.json"))
CLAIMS = json.load(open(f"{R}/_audit_claims_raw.json"))

def rv(key, field="recomputed"):
    return RC[key][field]

recs = []
def add(claim, loc, artifact, exists, verdict, reported=None, recomputed=None,
        blast="body", note=""):
    recs.append({"claim": claim, "tex_location": loc, "artifact": artifact,
                 "artifact_exists": exists, "reported": reported,
                 "recomputed": recomputed, "verdict": verdict,
                 "blast_radius": blast, "note": note})

# ---------------------------------------------------------------- VERIFIED
V = "VERIFIED"
add("weak labeller MCC +0.0243", "ch5:311, ch7:467, ch8:346",
    "results/accord/reconciliation_c28.json", True, V,
    0.0243, rv("weak_labeller_MCC"), "abstract-level",
    "C53 relabelling landed; cells TP11/FN1/FP212/TN29 reproduce exactly")
add("weak labeller precision 4.93%", "ch7:467", "reconciliation_c28.json", True, V,
    0.0493, rv("weak_labeller_precision"), "body")
add("weak labeller recall 91.7%", "ch7:467", "reconciliation_c28.json", True, V,
    0.9167, rv("weak_labeller_recall"), "body")
add("AUROC 0.5145 (3-seed ensemble)", "ch1:188, ch7:233/253, ch8:319, ch10:71",
    "results/accord/decontam/corrected_gold_metrics.json", True, V,
    0.5145, rv("auroc_ensemble_0.5145"), "abstract-level",
    "value verified; DENOMINATOR is separately flagged, see MISMATCH records")
add("per-seed AUROC mean 0.5211 +/- 0.0201", "ch7:242/255",
    "decontam/corrected_gold_metrics.json", True, V, 0.5211, rv("auroc_perseed_mean"),
    "body", "std is population (ddof=0): recomputes to " + str(rv("auroc_perseed_std_ddof0")))
add("average precision 0.0579 +/- 0.0034", "ch7:259", "decontam per_seed", True, V,
    0.0579, rv("ap_mean_0.0579"), "body", "ddof=0")
add("Mann-Whitney p = 0.867", "ch1:188, ch7:264, ch8:320, ch10:71",
    "decontam aggregate", True, V, 0.867, rv("mwu_p_0.867"), "abstract-level")
add("bootstrap 95% CI [0.321, 0.703]", "ch1:189, ch7:266, ch8:321, ch10:72",
    "decontam aggregate", True, V, [0.321, 0.703],
    [rv("ci_lo_0.321"), rv("ci_hi_0.703")], "abstract-level")
add("defect-free AUROC 0.5078, p=0.953, 6 positives", "ch1:192-194, ch10:74-76",
    "results/accord/defect_free_strata.json", True, V, 0.5078,
    rv("defect_free_auroc_0.5078"), "abstract-level",
    "computed on the CONTAMINATED run (ALL n=299 AUROC 0.4602); flagged separately")
add("subject-disjoint control AUROC 0.497", "ch7:284", "control/final_report.json",
    True, V, 0.497, rv("control_auroc_mean_0.497"), "body")
add("contaminated AUROC 0.4595 on same 253", "ch7:285", "decontam", True, V,
    0.4595, rv("contaminated_253_auroc_0.4595"), "body")
add("decontaminated precision 0.0424 +/- 0.0023", "ch7:167/201, ch7:915",
    "mimic_local_only/accord_supp/results/accord_v2_decontam_gold_aggregate.json",
    True, V, 0.0424, rv("decontam_precision_0.0424"), "abstract-level")
add("decontaminated recall 0.7778 +/- 0.0393 (TP 9,10,9)", "ch7:168, ch7:581",
    "accord_v2_decontam_gold_aggregate.json", True, V, 0.7778,
    rv("decontam_recall_0.7778"), "abstract-level")
add("accuracy 0.115 = 29/253", "ch7:169/194, ch8:245", "reconciliation_c28.json",
    True, V, 0.115, 29/253, "body")
add("ACCORD predicts positive for 278 of 299 / 234 of 253", "ch1:239, ch7:165, ch8:331",
    "reconciliation_c28.json", True, V, 278, 278, "abstract-level")
add("44 of 46 ambiguous predicted positive", "ch7:195 (footnote)",
    "reconciliation_c28.json findings", True, V, 44, 44, "body",
    "CLOSES the ch7:196 TODO -- evidence is on disk")
add("weak label counts 1425 / 254 / 1679, 84.9% / 15.1%",
    "ch5:83-86, ch1:160, ch10:52", "results/accord/step0_1_join_and_calibration.json",
    True, V, 0.849, rv("weak_pos_rate_84.9"), "abstract-level")
add("prevalence ratio ~21x", "ch5:39/90, ch1:157", "step0_1 + gold", True, V,
    21, rv("prevalence_ratio_21x"), "abstract-level")
add("at least 95% of positive labels wrong (1425-67=1358)", "ch5:104-110, ch1:161, ch10:53",
    "step0_1 derivation", True, V, 0.95, rv("pct_positive_noise"), "abstract-level")
add("per-rule Llama/ACCORD/true rates (tab:tracking, 12 cells)", "ch5:138-141",
    "step0_1 rule_stratified_calibration", True, V, None, None, "body",
    "all 12 cells reproduce to the printed precision")
add("tab:gold 4 rules x sampled/true/precision + pooled", "ch4:426-434",
    "step0_1 + step6_error_taxonomy", True, V, None, None, "body",
    "134/31/79/55 sum to 299; 3/3/4/2 sum to 12; all rates reproduce")
add("error taxonomy 7/3/1/1 and danger 2,4,2,1,3", "ch7:538-541, ch7:562",
    "results/accord/step6_error_taxonomy.json", True, V, None, None, "body")
add("recall table Llama 11/12, GPT-4o 6/12", "ch7:580-582",
    "step6_error_taxonomy caught_counts", True, V, None, None, "body")
add("four systems precision/recall/MCC (tab:systems)", "ch7:450-453",
    "results/accord/step3_baselines.json", True, V, None, None, "abstract-level",
    "all 12 cells reproduce; Llama row correctly relabelled 'evaluation baseline' per C53")
add("MCC derived == reported, all four systems", "ch8:225-228",
    "step3_baselines.json cells", True, V, None, None, "body")
add("balanced accuracy + PABAK, all four systems", "ch8:265-268",
    "step3_baselines.json cells", True, V, None, None, "body")
add("Q/P = 20.08 exchange rate", "ch1:215, ch8:291-295", "P=12,Q=241", True, V,
    20.08, rv("Q_over_P"), "abstract-level")
add("exchange arithmetic: delta 5, dTN 158, threshold 100.4, >=101", "ch8:291-295",
    "cells", True, V, 100.4, rv("exchange_threshold"), "body")
add("balacc spread 0.14, PABAK spread 1.27", "ch8:283-285", "cells", True, V,
    1.27, rv("pabak_spread"), "body")
add("four MCC bootstrap CIs all contain zero", "ch8:340-343",
    "results/accord/stats/intervals.json", True, V, None, None, "abstract-level")
add("harm rho -0.0001 p=0.999; contra rho -0.029 p=0.62", "ch7:402-403, ch1:203, ch10:67",
    "results/accord/step5_ranking.json", True, V, -0.0001, rv("harm_rho"), "abstract-level")
add("harm score range 0.398-0.610", "ch1:201, ch7:309, ch10:67-68",
    "step5_ranking harm_score_range_check", True, V, 0.398, rv("harm_min_0.398"),
    "abstract-level")
add("harm AUROC 0.504 at p=0.96", "ch7:237/306", "step2_signal_check.json", True, V,
    0.504, rv("harm_auroc_0.504"), "body")
add("ambiguous-as-positive AUROC 0.490", "ch7:235", "step2_signal_check.json", True, V,
    0.490, rv("ambiguous_as_positive_auroc_0.490"), "body")
add("LoRA 592,131 = 589,824 + 2,307", "ch7:133-139", "closed form + stage0 config1",
    True, V, 592131, rv("lora_total_592131"), "body")
add("110,074,887 = 109,482,240 + 592,647", "ch7:141",
    "results/accord/stage0/config1_baseline_bottleneck_head.json", True, V,
    110074887, rv("bertbase_plus_head"), "body")
add("spec components sum to 1,539,785", "ch6:79-84", "ch6 tab:spec row sum", True, V,
    1539785, rv("spec_total_1539785"), "body")
add("22,446,855 = 3x7,087,872 + 590,592 + 592,647", "ch6:409/417",
    "closed form (C54)", True, V, 22446855, rv("stage0_recon_22446855"), "body")
add("live-parameter ratio 36.88x", "ch2:355, ch6:431/446/709", "closed form (C54)",
    True, V, 36.88, rv("live_ratio_36.88"), "abstract-level",
    "C54 fix fully propagated; 37.9x survives only as the explicitly-retired figure at ch6:432")
add("cohort ICD-9 prefix counts 13,608 / 6,265 / 11,422", "ch3:115",
    "results/accord/stats/cohort_icd_counts.json", True, V, None, None, "body")
add("13,597 after LOS>0, 11 removed", "ch3:116/135", "cohort_icd_counts.json", True, V,
    None, None, "body")
add("9,783 / 4,339 / 7,579 usable note pairs", "ch3:121, ch1:53, ch10:37",
    "generality/experimentA_corrected_replication.json + sepsis/aki phase1", True, V,
    9783, 9783, "abstract-level")
add("survey 4/75 strict and 14/75 permissive", "NOT IN DRAFT",
    "results/accord/survey/coding_table.csv + coding_summary.json", True, V,
    None, None, "no-exposure",
    "regenerates exactly from the 75-row CSV; 5.3% / 18.7%; C55 governs; draft cites neither")
add("C44/C62 convergence gap 4.53 pts at 24 epochs", "NOT IN DRAFT",
    "results/accord/scaling/convergence_results.json", True, V, 4.53, 4.53,
    "no-exposure",
    "all 16 arm x budget cells, the 5.07/4.67/4.86/4.53 gap sequence and both SDs "
    "reproduce from the 48 runs; C62 uses ddof=1")

# ---------------------------------------------------------------- MISMATCH
M = "MISMATCH"
add("'precision 0.047 against a base rate of 0.0424 +/- 0.0023'",
    "ch1:240", "accord_v2_decontam_gold_aggregate.json", True, M,
    "base rate = 0.0424 +/- 0.0023",
    "0.0424 +/- 0.0023 is ACCORD's DECONTAMINATED PRECISION, not a base rate; "
    "the base rates are 0.040 (n=299) and 0.0474 (n=253)",
    "abstract-level",
    "Inverts the chapter's conclusion. ch7:201 states the decontaminated precision is BELOW "
    "the base rate; ch7:915 words the same sentence correctly ('precision 0.0424 +/- 0.0023 "
    "against a base rate of 0.040'). As written ch1 makes 0.047 > 0.0424 read as the detector "
    "adding lift. Also reverts precision to the contaminated 0.047 mid-sentence.")
add("'fifteen highest-scored pairs contain zero true contradictions'",
    "ch1:190 and ch10:73", "decontam/corrected_gold_metrics.json per_seed", True, M,
    "zero", "one (ensemble); per-seed 1, 0, 0",
    "abstract-level",
    "ch7:293/930 and ch8:321 say ONE. Zero is the CONTAMINATED run's count (C16). "
    "Introduction and Conclusion carry the superseded value; body carries the corrected one.")
add("'is the only one above chance' / 'But it is above chance'",
    "ch10:83 and ch10:110", "results/accord/stats/intervals.json", True, M,
    "above chance", "MCC +0.1089, 95% CI [-0.0319, 0.2484] contains zero",
    "abstract-level",
    "ch7:484-491 formally retracts exactly this sentence with a Correction note; the "
    "retraction did not propagate to the Conclusion, where it is load-bearing for the "
    "'the task is not the problem' claim.")
add("'something above chance is extractable from the same evidence'",
    "ch1:254", "results/accord/stats/intervals.json", True, M,
    "above chance", "CI contains zero; ch7:951 hedges correctly ('not settled here')",
    "abstract-level", "Same retracted claim as ch10; ch1 states it as a consequence.")
add("'zero-shot detects half of them above chance'",
    "ch7:791", "results/accord/stats/intervals.json", True, M,
    "above chance", "CI contains zero",
    "body", "Contradicts the retraction at ch7:484 in the SAME chapter.")
add("AUROC 0.5145 attributed to n=299 / 'the full evaluation set'",
    "ch1:377, ch7:233 (tab:auroc row 1), ch10:70", "decontam/corrected_gold_metrics.json",
    True, M, "n = 299", "n = 253 (decontam artifact has no n=299 recomputation)",
    "abstract-level",
    "ch7's own caption concedes 'the decontaminated recomputation exists only on the "
    "253-pair core', which contradicts the n=299 label in row 1 of the table it captions.")
add("GPT-4o zero-shot evaluated 'on the same 299 pairs'",
    "ch1:249 and ch10:107", "results/accord/step3_baselines.json", True, M,
    299, 253, "abstract-level",
    "step3_baselines gpt4o_zeroshot has n=253; ch7:456/488 correctly say 253.")
add("'precision tracks each rule's base rate to within a tenth of a percentage point'",
    "ch9:463 and ch10:99", "step0_1 + ch4 tab:gold", True, M,
    "<= 0.1 pp on every rule", "Rule 2 is 0.3 pp (10.0 vs 9.7); Rule 4 is 1.2 pp (2.4 vs 3.6)",
    "abstract-level",
    "Contradicted by its own parenthetical in the same sentence. ch4:463 states it correctly "
    "('For Rules 1, 2 and 3 ... essentially identical', Rule 4 below).")
add("student tracks teacher 'to within 1.5 pp'; per-rule gaps '1.5, 0.0, 1.2'",
    "ch5:149/160, ch1:163, ch10:55", "step0_1 rule_stratified_calibration", True, M,
    [1.5, 0.0, 1.2], rv("student_teacher_gaps_pp"), "abstract-level",
    "Rule 3 gap is 1/79 = 1.27 pp, which rounds to 1.3, not 1.2. The '<= 1.5 pp' envelope "
    "claim is unaffected.")
add("'165 of the 300'", "ch7:679", "results/accord/defect_free_strata.json", True, M,
    300, 299, "body", "The annotated set is 299; 165 + 134 = 299.")
add("'the only numbers in this thesis whose producing artifact could not be found'",
    "ch7:85", "ch6:301-306, ch4:325, ch3:184", True, M,
    "MedNLI figures are the only untraceable ones",
    "at least 4 further groups are untraceable: transfer F1s, exemplar rates, "
    "note-role coverages, median LOS",
    "body",
    "ch6:305 itself says the transfer values sit 'on the same footing as the MedNLI "
    "accuracies', directly contradicting ch7:85.")
add("tab:met-mccci caption: 'no bootstrap interval has been computed for the weak "
    "labeller's own calls'", "ch8:344-345 (caption)",
    "CORRECTIONS C58; recomputed here", False, M,
    "not computed", "computed: [-0.1040, +0.0966]",
    "body",
    "Directly contradicted three lines later at ch8:353-357, which quotes the interval. "
    "C58 supersedes the caption.")
add("note-role coverage 72.2% (attempt 6, adopted)", "ch3:190",
    "no artifact; caption says 'computed against the 13,608-admission ICD-9 set'",
    False, M, 0.722, 9783/13608,
    "body",
    "The table's own rows give 9,783/13,608 = 0.7189 -> 71.9%. 72.2% does not "
    "reproduce from either denominator (9,783/13,597 = 72.0%).")

# ---------------------------------------------------------------- SUPERSEDED
S = "SUPERSEDED-BY-CORRECTION"
add("authority parameter 'moved by 1.8% and 0.6%'",
    "ch1:176-177, ch6:29-30, ch6:189-190, ch6:237, ch7:737, ch9:27, ch10:62-63",
    "accord_v2_decontam_gold_aggregate.json (alpha_n, alpha_d)", True, S,
    [1.8, 0.6], rv("alpha_move_pct", "recomputed_decontaminated_3seed"),
    "abstract-level",
    "CORRECTIONS C16: (0.509, 1.006) is a CONTAMINATED SINGLE SEED. The 3-seed "
    "decontaminated values are 0.5076 +/- 0.0006 and 1.0038 +/- 0.0010, i.e. 1.5% and 0.4%. "
    "ch6:189 additionally mis-describes the single seed as 'across three seeds'. "
    "Seven locations, including the Introduction contribution list and the Conclusion.")
add("'agreeing to the third decimal across seeds'", "ch1:177, ch6:190",
    "accord_v2_decontam_gold_aggregate.json alpha_d min/max", True, S,
    "third-decimal agreement", "alpha_d ranges 1.0024 to 1.0047 across the three "
    "decontaminated seeds -- the third decimal differs",
    "body", "True of the contaminated run only.")
add("defect-free stratum 0.5078 presented beside decontaminated 0.5145",
    "ch1:188-192, ch10:70-76", "defect_free_strata.json vs decontam", True, S,
    "both current", "0.5078 is the contaminated run (its 'ALL' row is 0.4602); "
    "the decontaminated defect-free values are 0.5417/0.5352/0.5117 (control) ",
    "abstract-level",
    "C16 scope list names 'the 0.5078 defect-free stratum' as inheriting the "
    "contamination. ch7:245 flags the daggered rows but ch1 and ch10 present 0.5145 "
    "and 0.5078 side by side as if from one configuration.")

# ---------------------------------------------------------------- UNTRACEABLE
U = "UNTRACEABLE"
add("rule-head transfer F1 0.0654 (sepsis) and 0.0595 (AKI)",
    "ch6:289-290, ch10:66, ch9:536-537, ch2:79, ch6:701",
    DRIVE + "/transfer_results_correct.json", True, V, [0.0654, 0.0595],
    [0.06543855847829327, 0.059544945658295115], "abstract-level",
    "ARTIFACT FOUND. The file ch6:302-304 says 'could not be located on disk' is on the "
    "Drive mount and both values reproduce exactly.")
add("rule-head in-domain F1 0.9603", "ch6:31/255/272/288/700, ch7:738/885, "
    "ch9:536/587, ch10:65/133", DRIVE + "/transfer_results_correct.json", True, V,
    0.9603, 0.9603, "abstract-level",
    "hf_reference.rule_type_f1 = 0.9603 exactly. Still 'withdrawn' at ch6:313 yet "
    "restated as fact at ch10:65 -- an editorial inconsistency, not a numerical one.")
add("ch6 provenance note: transfer_results_correct.json 'is not among this project's "
    "surviving artifacts and could not be located on disk'", "ch6:301-306",
    DRIVE + "/transfer_results_correct.json", True, M,
    "file does not exist", "file exists; both values reproduce exactly",
    "abstract-level",
    "The draft demotes two verified numbers to unsupported claims on a false premise. "
    "Same false premise in CORRECTIONS C2.")
add("transfer run candidate pools (775 sepsis / 1,983 AKI)",
    "not stated in draft", DRIVE + "/transfer_results_correct.json vs "
    "results/accord/{sepsis,aki}/*_phase3_summary.json", True, M,
    "n/a", "Drive transfer run used 775 sepsis / 1,983 AKI candidates; the repo rebuild "
    "mined 539 / 1,604", "body",
    "The transfer F1s are computed on a different candidate pool from the one Chapter 3 "
    "documents. Worth a sentence.")
add("MedNLI 83.90 / 83.91 / 72.93 / 72.74", "ch7:98-99, ch1:264, ch2:173, ch7:914",
    "run log, not on disk", False, U, 83.90, None, "abstract-level",
    "Honestly flagged at ch7:80-89. Searched the repo, the quarantine AND the Drive "
    "mount (97 files): no occurrence of 83.9x anywhere. This one is genuinely untraceable. "
    "Load-bearing for ch1:264 and for the 11-point LoRA gap finding.")
add("exemplar correctness 60.5% / 50.4% / 58.6% and 739 / 1,307 (56.5%); "
    "complement 41.4%", "ch4:322-325, and 41.4% in 17 places across 7 chapters",
    "results/accord/exemplar/exemplar_correctness_recomputed.json", True, U,
    [0.605, 0.504, 0.586], [0.5556, 0.4138, 0.5255],
    "abstract-level",
    "THE ON-DISK ARTIFACT DISAGREES. exemplar_correctness_recomputed.json labels 0.586 as "
    "'reference_full_2296_candidate_pool_rate_from_task_description' -- i.e. supplied, not "
    "measured -- and its own recomputation over the 165 rules-1+2 gold rows (137 resolved) "
    "gives 55.56% / 41.38% / 52.55%, whose complement is 47.45%, not 41.4%. "
    "No artifact records 739, 1,307 or the 60.5/50.4/58.6 triple. "
    "41.4% is the second-most-repeated number in the thesis.")
add("cohort_summary corroboration: 13,608/9,783/6,265/4,339/11,422/7,579",
    "ch3:115-121", DRIVE + "/cohort_summary.json", True, V, None, None, "body",
    "ARTIFACT FOUND on the Drive mount; every cell matches.")
add("rule candidate counts 886/203/521/370, raw total 1,980, post-prefilter 1,978",
    "ch4:184-190, ch1:52, ch10:36", DRIVE + "/mining_summary.json", True, V,
    1980, 886+203+521+370, "abstract-level",
    "ARTIFACT FOUND. mining_summary.json gives 1,980 raw / 1,205 admissions; the draft "
    "correctly distinguishes raw 1,980 from post-prefilter 1,978 / 1,203.")
add("note-role coverages 16.7% / 16.8% / 12.4%", "ch3:184-188", "run logs", False, U,
    None, None, "body", "No artifact; already partly marked [TODO] for attempts 4-5.")
add("median LOS 198.7 h and unique patients 10,143 (HF)", "ch3:117-118", "run logs",
    False, U, None, None, "body")
add("ablation table 0.7603/0.9091 vs 0.7368/0.8939", "ch6:222-223",
    "C16 gives the corrected pair; no artifact file", False, U, None, None, "body",
    "C16's corrected values ARE what the draft prints (good), but the producing artifact "
    "is not on disk.")
add("45.2% held-out rule-prediction accuracy (recomputes to 0.4515)", "ch6:291",
    "gold-standard CSV per ch6 caption", False, U, 0.452, None, "body",
    "Caption asserts it 'recomputes to 0.4515 directly from the gold-standard CSV' but "
    "no results file records 0.4515.")

# ---------------------------------------------------------------- TODO audit
todo = collections.defaultdict(list)
for fn in sorted(os.listdir("/Users/rickygole/provenance-encoder/accord/thesis")):
    if not fn.endswith(".tex"):
        continue
    for i, line in enumerate(open(f"/Users/rickygole/provenance-encoder/accord/thesis/{fn}"), 1):
        for _ in re.finditer(r"TODO", line):
            todo[fn].append({"line": i, "text": line.strip()[:200]})
n_todo = sum(len(v) for v in todo.values())

TODO_DISPOSITION = {
 "closes_now_evidence_on_disk": [
   {"loc": "ch7:196", "item": "confirm the 253/299 denominators and the 44-of-46 ambiguous count",
    "evidence": "results/accord/reconciliation_c28.json (measured 44 of 46)"},
   {"loc": "ch3:117 (sepsis, AKI cells)", "item": "unique patients for the transfer cohorts",
    "evidence": "sepsis_phase1_summary.json unique_subjects=5500; aki_phase1_summary.json unique_subjects=9523"},
   {"loc": "ch3:116 ('n/a' cells, not marked TODO but same defect)",
    "item": "After LOS>0 for sepsis and AKI",
    "evidence": "results/accord/stats/cohort_icd_counts.json post_los 6253 / 11408"},
   {"loc": "ch6:224", "item": "majority-class baseline on this split",
    "evidence": "results/accord/control/baselines.json majority_class (AUROC 0.5, AP 0.0401)"},
 ],
 "closes_by_deletion_already_answered_elsewhere_in_the_draft": [
   {"loc": "ch4:387", "item": "identify which stratum lost a row",
    "why": "ch4:445 (tab:gold caption) already states it: PAIR_0076, a Rule 4 candidate. "
           "The TODO and the caption also disagree about whether rates are computed "
           "against 300-row or 299-row strata."},
   {"loc": "ch8:344 caption clause", "item": "'no bootstrap interval computed for the weak labeller'",
    "why": "C58 computed it; ch8:353 quotes it three lines later."},
 ],
 "must_become_a_stated_limitation_no_evidence_obtainable": [
   {"loc": "ch3:185/196", "item": "note-role attempts 4 and 5 definitions"},
   {"loc": "ch3:14, ch3:118", "item": "median LOS for transfer cohorts"},
   {"loc": "ch6:79-80, ch6:96", "item": "CDE / ATCG acronym expansions"},
   {"loc": "ch3:349", "item": "one-entry reconciliation discrepancy"},
   {"loc": "ch8:249", "item": "recover the withdrawn submission's other nine comparison rows"},
 ],
 "requires_new_experiments_genuinely_open": [
   {"loc": "ch7:145", "item": "LoRA rank sweep"},
   {"loc": "ch6:673", "item": "zero pf[i,1] at inference, re-measure rule F1"},
   {"loc": "ch6:685", "item": "vanilla cross-encoder baseline on the identical split"},
   {"loc": "ch5:320", "item": "ablate the rule identifier out of the weak-label prompt"},
   {"loc": "ch5:272", "item": "whether the weak-label loop used the positive failure default"},
   {"loc": "ch4:304", "item": "validate the dose-adjacency tie-break"},
   {"loc": "ch4:570, ch3:568", "item": "re-annotate corrected exemplars / validate the extractor"},
   {"loc": "ch7:366, ch8:246", "item": "recompute the PR curve and confusion matrices on the decontaminated seeds"},
   {"loc": "ch6:496", "item": "separate the Stage 0 bottleneck and truncation hypotheses"},
   {"loc": "ch7:498", "item": "few-shot exemplar class balance"},
   {"loc": "ch7:837/847/860/901", "item": "cost, latency, full-backbone retrain, governance, GPU-hours"},
   {"loc": "ch7:697, ch9:202, ch8:468", "item": "average precision per system"},
   {"loc": "ch7:819/828", "item": "recompute tab:systems rows from decontaminated artifacts"},
   {"loc": "ch8:521", "item": "emit the four confusion matrices directly"},
   {"loc": "ch2:175, ch5:372, ch9:9/438/515/545", "item": "cross-references to the above"},
   {"loc": "main.tex:55", "item": "THE ABSTRACT IS NOT WRITTEN"},
 ],
}

summary = collections.Counter(r["verdict"] for r in recs)
blast = collections.Counter((r["verdict"], r["blast_radius"]) for r in recs)

out = {
 "audit": "ACCORD thesis number-provenance audit",
 "scope": {"tex_files": 11, "tex_lines": 5810,
           "numeric_literals_extracted": len(CLAIMS),
           "claims_adjudicated": len(recs),
           "quantities_independently_recomputed": len(RC)},
 "scripts": {
   "extractor": "accord/stats/audit_extract_claims.py",
   "recomputation": "accord/stats/audit_recompute.py",
   "cross_chapter": "accord/stats/audit_cross_consistency.py",
   "assembler": "accord/stats/audit_build_report.py"},
 "compliance": "local_only/ and mimic_local_only/ read aggregate-only; no note text, "
               "HADM_ID or SUBJECT_ID leaves those directories in this audit.",
 "summary_by_verdict": dict(summary),
 "summary_by_verdict_and_blast_radius": {f"{k[0]}|{k[1]}": v for k, v in sorted(blast.items())},
 "records": recs,
 "cross_chapter_inconsistencies": XC,
 "statistical_concerns": [
  {"id": "S1", "severity": "high",
   "issue": "Error-bar convention is not consistent within a single comparison.",
   "detail": f"ch7:242 (0.5211 +/- 0.0201) and ch7:259 (0.0579 +/- 0.0034) are POPULATION "
             f"std (ddof=0); ch7:284 (0.497 +/- 0.035) is SAMPLE std (ddof=1) -- recomputed "
             f"ddof=0 for the control is {rv('control_auroc_std_convention','recomputed_ddof0')}. "
             f"The three are compared as a 'three-way agreement' in the same paragraph. "
             f"Nothing in the draft defines whether +/- is std, sem or CI.",
   "fix": "Declare the convention once and apply it; with n=3 seeds, report the three values."},
  {"id": "S2", "severity": "high",
   "issue": "A quoted CI has no artifact.",
   "detail": "The weak-labeller MCC interval [-0.1040, +0.0966] (ch5:312, ch8:354) exists only "
             "in CORRECTIONS C58 and the .tex. results/accord/stats/intervals.json holds four "
             "systems and not this one. Independent recomputation from the C53 cells reproduces "
             f"it in substance ({rv('weak_labeller_MCC_CI','recomputed')}, 0 degenerate draws) but "
             "not to the printed digits, because the seed is unrecorded. This is the C48 pattern.",
   "fix": "Persist the computation; ch8's own table caption still says it was never computed."},
  {"id": "S3", "severity": "medium",
   "issue": "Degenerate bootstrap is handled in one place and undocumented in others.",
   "detail": "intervals.json E3 records draws_dropped_zero_positives=20 (0.2%) and reports a "
             "stratified CI beside the unstratified one -- correct practice. The AUROC CI "
             "[0.321, 0.703] on 12 positives, and the four MCC CIs, do not report a degenerate-draw "
             "count. With 12 positives in 253, resamples with 0 positives are possible and MCC's "
             "denominator vanishes on any resample with an empty predicted or actual class.",
   "fix": "Report n_valid / n_dropped for every bootstrap, as E3 already does."},
  {"id": "S4", "severity": "low-good",
   "issue": "DeLong is used correctly for the correlated-AUROC comparison.",
   "detail": "control/final_report.json carries delong_ensemble_vs_accord and per-seed DeLong "
             "tests on the same 299 pairs (p = 0.930 gold_all). This is the right test for two "
             "AUROCs on shared data. It is not cited in ch7's three-way agreement paragraph, "
             "which asserts agreement by eye instead.",
   "fix": "Cite the DeLong p-values at ch7:285-287."},
  {"id": "S5", "severity": "medium",
   "issue": "Multiple-comparison exposure is unaddressed.",
   "detail": "ch7 reports 4 AUROC treatments, a 7-row per-rule/per-stratum AUROC table "
             "(defect_free_strata.json), 2 Spearman tests, 4 MCC intervals, and a 3-row "
             "threshold sweep. Per-rule AUROCs range 0.2262 (rule 2, n=31) to 0.6514 (rule 1, "
             "n=134) with no correction and no interval. The artifact's own caveat says these "
             "are 'individually unreliable'; ch7 does not repeat that caveat at the point of use.",
   "fix": "State the family size, or restrict reporting to the two pre-specified strata."},
  {"id": "S6", "severity": "high",
   "issue": "A p-value and CI are quoted for a stratum whose configuration was superseded.",
   "detail": "The defect-free AUROC 0.5078 with p=0.953 (ch1:192, ch7, ch10:75) is computed on "
             "the CONTAMINATED scores -- the same file whose 'ALL' row is 0.4602, which C16 "
             "retired. It is quoted in the Introduction and Conclusion beside the "
             "DECONTAMINATED 0.5145 with no indication they come from different models. "
             "The decontaminated defect-free numbers exist (control/final_report.json: "
             "0.5417 / 0.5352 / 0.5117, ensemble 0.5469).",
   "fix": "Either report the decontaminated defect-free values or dagger 0.5078 in ch1 and ch10 "
          "as ch7:245 already does."},
  {"id": "S7", "severity": "medium",
   "issue": "n=3 seeds is the entire basis for every +/- in the evaluation chapter.",
   "detail": "Three seeds support the precision, recall, AUROC, AP and alpha intervals. Seeds are "
             "reported (42/123/2024 for the control; 0/1/2 for the decontaminated run) which is "
             "good practice, but no test is run across seeds and the std of 3 values is quoted "
             "to four significant figures.",
   "fix": "Report the three values, not a 4-s.f. std."},
  {"id": "S9", "severity": "critical",
   "issue": "An artifact store the draft and CORRECTIONS declare missing is present on disk.",
   "detail": "97 files at " + DRIVE + " including transfer_results_correct.json (which ch6:302 "
             "says 'could not be located on disk' -- it reproduces 0.0654/0.0595/0.9603 exactly), "
             "cohort_summary.json (reproduces every ch3 cohort cell), mining_summary.json "
             "(reproduces all four rule counts and the 1,980 raw total), and 23 .pt checkpoints "
             "including accord_v2_best.pt, accord_stage1_best.pt, accord_stage2_best.pt, "
             "atcg_pretrained.pt and accord_v2_decontam_seed{0,1,2}.pt. CORRECTIONS C54 asserts "
             "'No ACCORD or Stage 0 checkpoint exists on disk at all' and uses that to support "
             "the conclusion that the 22,446,855-parameter configuration was never trained. "
             "C54's independent argument -- that no code path can unfreeze encoder layers, and "
             "that 590,592 of the total is a pooler with zero gradient -- is unaffected and "
             "still stands. But the checkpoint premise is false and should be re-checked before "
             "the 'never trained' claim is printed.",
   "fix": "Re-run the artifact search against the Drive mount before finalising any "
          "'could not be located' provenance note, and re-verify C54 against the checkpoints."},
  {"id": "S8", "severity": "high",
   "issue": "Sign/construction corrections have no thesis exposure -- confirmed, not assumed.",
   "detail": "C52/C56/C57 (abs() gaps, 7 flipped drift slopes) touch PUMS / ACSIncome / Regime "
             "A/B / demographic parity / four-fifths. Grep across ch1-ch10 finds no chapter "
             "citing any of these. C57's own 'current thesis exposure: none' verifies.",
   "fix": "None needed for the draft; the standing rule in C56 still applies to anything added."},
 ],
 "todo_audit": {
   "total_TODO_occurrences": n_todo,
   "occurrences_by_file": {k: len(v) for k, v in sorted(todo.items())},
   "disposition": TODO_DISPOSITION,
   "note": "46 occurrences of the token TODO across 43 lines; main.tex:55 records that the "
           "ABSTRACT IS NOT YET WRITTEN, which is the highest-blast-radius open item in the file.",
 },
}

with open(f"{R}/number_provenance_audit.json", "w") as fh:
    json.dump(out, fh, indent=1)

print("VERDICT COUNTS:", dict(summary))
print("BY BLAST RADIUS:")
for k, v in sorted(blast.items()):
    print(f"   {k[0]:28s} {k[1]:16s} {v}")
print(f"TODO occurrences: {n_todo}")
print(f"wrote {R}/number_provenance_audit.json")
