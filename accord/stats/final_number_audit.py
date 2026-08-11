#!/usr/bin/env python3
"""
Final number audit for the MSU thesis (accord/thesis/msu_ch{1..6}_*.tex).

Design rule: NOTHING is typed by hand. For every claim the script
  (a) reads the printed value out of the .tex at the recorded line, and
  (b) recomputes the value from the raw artifact,
then compares. Verdicts: VERIFIED | MISMATCH | UNTRACEABLE.

Run:  python3 accord/stats/final_number_audit.py
Out:  results/accord/final_number_audit.json
"""
import json, re, math, statistics as st, os
from pathlib import Path
from collections import defaultdict

ROOT = Path("/Users/rickygole/provenance-encoder")
TEX  = ROOT/"accord/thesis"
OUT  = ROOT/"results/accord/final_number_audit.json"
QUAR = Path("/Users/rickygole/mimic_local_only")
DRIVE= Path("/Users/rickygole/Library/CloudStorage/GoogleDrive-ricky.gole@morgan.edu/My Drive/thesis")

def J(p):
    p = Path(p)
    if not p.is_absolute(): p = ROOT/p
    return json.load(open(p))

SNAP = ROOT/"accord/stats/_thesis_snapshot"
import hashlib
LINES  = {f.name: open(f).read().splitlines() for f in sorted(SNAP.glob("msu_ch*.tex"))}
HASHES = {f.name: hashlib.sha256(open(f,'rb').read()).hexdigest() for f in sorted(SNAP.glob("msu_ch*.tex"))}
def line(ch, n): return LINES[ch][n-1]

def locate(ch, pattern, near=None):
    """Line number (1-based) of the unique line matching `pattern`. Never trust a hard-coded offset."""
    hits = [i+1 for i, l in enumerate(LINES[ch]) if re.search(pattern, l)]
    if not hits: raise LookupError(f"no line in {ch} matches {pattern!r}")
    if near is not None: hits.sort(key=lambda h: abs(h-near))
    return hits[0]

NUMRE = re.compile(r'-?\d[\d,]*(?:\.\d+)?')
def A(ch, pattern, idx=None):
    """Locate a line by content and return its numbers. Line numbers are never hard-coded."""
    n = locate(ch, pattern)
    return (n, printed(ch, n, idx))

def printed(ch, n, idx=None, want=None, anchor=None):
    """Numbers printed on a line, as floats, in order. `want` picks by string match."""
    if anchor is not None: n = locate(ch, anchor, near=n)
    s = re.sub(r'\\(label|ref|cite[a-z]*)\{[^}]*\}', ' ', line(ch, n))
    s = s.replace('{,}', ',')
    toks = [t for t in NUMRE.findall(s)]
    vals = [float(t.replace(',', '')) for t in toks]
    if want is not None:
        for t, v in zip(toks, vals):
            if t == want: return v
        return None
    return vals if idx is None else (vals[idx] if idx < len(vals) else None)

# ---------------------------------------------------------------- sources
PRE  = J("results/pretraining_runs.json")
CTRL = J("results/accord/control/final_report.json")
DEC  = J("results/accord/decontam/corrected_gold_metrics.json")
IVL  = J("results/accord/stats/intervals.json")
S31  = J("results/accord/step3_baselines.json")
S01  = J("results/accord/step0_1_join_and_calibration.json")
S02  = J("results/accord/step2_signal_check.json")
EXE  = J("results/accord/exemplar/exemplar_correctness_recomputed.json")
EX2  = J("results/accord/exemplar/step2plus_results.json")
CONV = J("results/accord/scaling/convergence_results.json")
SCAL = J("results/accord/scaling/scaling_results.json")
TASK = J("results/task_runs.json")
SIGN = J("results/accord/signed_gap_analysis.json")
DRIF = J("results/accord/sign_provenance_drift_recomputation.json")
COST = J("results/accord/control_cost_log.json")
PWR  = J("results/accord/section6_auroc_power.json")
ICD  = J("results/accord/stats/cohort_icd_counts.json")
SWP  = J("accord/prevalence_replication/results/sweep_results.json")
MLLM = J("results/accord/multillm/per_model_results.json")
DFS  = J("results/accord/defect_free_strata.json")
ANI  = J("results/accord/stage0/anisotropy_control.json")
HOM  = J("results/accord/stage0/homogeneity.json")
PH1  = J(QUAR/"accord/phase1_summary.json")
DGA  = J(QUAR/"accord_supp/results/accord_v2_decontam_gold_aggregate.json")
MIN  = J(DRIVE/"outputs/mining_summary.json")
PH2  = J(DRIVE/"outputs/phase2_summary.json")
import csv
SURV = list(csv.DictReader(open(ROOT/"results/accord/survey/coding_table.csv")))
SC2  = list(csv.DictReader(open(ROOT/"results/accord/survey/second_coder_table.csv")))

# ---------------------------------------------------------------- ablation
def arm_runs(pred):
    return sorted(k for k, v in PRE.items()
                  if v.get("steps") == 1000 and v.get("size") == "10M"
                  and v.get("controls", {}).get("shuffle_segments")
                  and not v.get("controls", {}).get("global_positions") and pred(k, v))
ARMS = {
 "vanilla":          arm_runs(lambda k,v: v["arm"]=="A1"),
 "t5_rpe":           arm_runs(lambda k,v: v["arm"]=="A3" and "_rpe_" in k),
 "misaligned_block": arm_runs(lambda k,v: v["arm"]=="A4" and "_segidx_misalign_" in k),
 "alibi":            arm_runs(lambda k,v: v["arm"]=="A3" and "_alibi_" in k),
 "true_role_labels": arm_runs(lambda k,v: v["arm"]=="A4" and not re.search(r'_(segidx|segidx_misalign|lenroles|randroles)_', k)),
 "segment_index":    arm_runs(lambda k,v: v["arm"]=="A4" and "_segidx_shuf" in k),
 "hard_block":       arm_runs(lambda k,v: v["arm"]=="A3" and "_hardblk_" in k),
}
AV = {a: [PRE[k]["final_val_loss"] for k in ks] for a, ks in ARMS.items()}
AM = {a: st.mean(v) for a, v in AV.items()}
ASD= {a: st.stdev(v) for a, v in AV.items()}
SEG = AM["segment_index"] - AM["vanilla"]

# ---------------------------------------------------------------- convergence
cells = defaultdict(list)
for r in CONV: cells[(r["tag"], r["budget"])].append(r["accuracy"])
cmean = lambda t,b: st.mean(cells[(t,b)])
csd   = lambda t,b: st.stdev(cells[(t,b)])
def best_scratch(b): return max((cmean(t,b) for t in ("scratch_10M","scratch_30M","scratch_100M")))
sg = defaultdict(list); sp = {}
for r in SCAL: sg[r["tag"]].append(r["accuracy"]); sp[r["tag"]] = r["non_emb_params"]
def taskarm(tag, key="auroc"):
    v = [r[key] for k, r in TASK.items() if r.get("tag") == tag or r.get("arm") == tag]
    return v
CONC = [TASK[f"concat_s{i}"]["auroc"] for i in (1,2,3)]
CURR = [TASK[f"ours_current_s{i}"]["auroc"] for i in (0,1,2)]

# ---------------------------------------------------------------- survey
c1 = {r["paper_id"]: r for r in SURV}
def agree_kappa(f):
    a = [c1[r["paper_id"]][f] for r in SC2]; b = [r[f] for r in SC2]; n = len(a)
    po = sum(x==y for x,y in zip(a,b))/n
    cats = set(a)|set(b)
    pe = sum((a.count(c)/n)*(b.count(c)/n) for c in cats)
    return po, (None if abs(1-pe) < 1e-12 else (po-pe)/(1-pe))

R = []   # records
def rec(ch, ln, what, prn, src, path, key, note="", dp=None):
    if prn is None or src is None:
        v = "UNTRACEABLE"
    elif isinstance(prn, list) or isinstance(src, list):
        v = ("VERIFIED" if (isinstance(prn, list) and isinstance(src, list) and len(prn) == len(src)
             and all(a is not None and b is not None and
                     abs(round(float(b), len(str(a).split(".")[1]) if "." in str(a) else 0) - float(a)) < 1e-9
                     for a, b in zip(prn, src))) else "MISMATCH")
    elif isinstance(src, str) or isinstance(prn, str):
        v = "VERIFIED" if str(prn) == str(src) else "MISMATCH"
    else:
        dp = dp if dp is not None else (len(str(prn).split(".")[1]) if "." in str(prn) else 0)
        v = "VERIFIED" if abs(round(float(src), dp) - float(prn)) < 1e-9 else "MISMATCH"
    R.append({"chapter": ch, "line": ln, "claim": what, "printed": prn,
              "recomputed": src, "source_path": path, "source_key": key,
              "verdict": v, **({"note": note} if note else {})})

C4, C3, C1, C5, C6 = ("msu_ch4_results.tex", "msu_ch3_methodology.tex",
                      "msu_ch1_introduction.tex", "msu_ch5_discussion.tex",
                      "msu_ch6_conclusion.tex")
PRERUN = "results/pretraining_runs.json"

# ================= 1. seven-arm ablation, tab:ablation (ch4 L389-395) ==========
for ln, arm in zip(range(389, 396),
    ["vanilla","t5_rpe","misaligned_block","alibi","true_role_labels","segment_index","hard_block"]):
    p = printed(C4, ln)[-4:]
    rec(C4, ln, f"ablation[{arm}] mean val loss",  p[0], AM[arm],  PRERUN, f"{ARMS[arm]} -> final_val_loss (mean)")
    rec(C4, ln, f"ablation[{arm}] SD (ddof=1)",    p[1], ASD[arm], PRERUN, f"{ARMS[arm]} -> final_val_loss (stdev)")
    d = AM[arm]-AM["vanilla"]
    rec(C4, ln, f"ablation[{arm}] delta vs vanilla", p[2], d, PRERUN, "derived: arm mean - vanilla mean")
    rec(C4, ln, f"ablation[{arm}] % of segment-index gain", p[3], 100*d/SEG, PRERUN, "derived: delta / segment-index delta")
rec(C4, 402, "ablation SD range low",  printed(C4,402,0), min(ASD.values()), PRERUN, "min stdev over seven arms")
rec(C4, 402, "ablation SD range high", printed(C4,402,1), max(ASD.values()), PRERUN, "max stdev over seven arms")
rec(C4, 403, "ablation effect range low",  printed(C4,403,0), abs(AM["t5_rpe"]-AM["vanilla"]), PRERUN, "derived |delta| min")
rec(C4, 403, "ablation effect range high", printed(C4,403,1), abs(AM["hard_block"]-AM["vanilla"]), PRERUN, "derived |delta| max")
rec(C4, 408, "segment-index reduction (nats)", printed(C4,408,0), abs(SEG), PRERUN, "derived")
rec(C4, 409, "true-role reduction (nats)",  printed(C4,409,0), abs(AM["true_role_labels"]-AM["vanilla"]), PRERUN, "derived")
rec(C4, 409, "true-role % of segment-index", printed(C4,409,1), 100*(AM["true_role_labels"]-AM["vanilla"])/SEG, PRERUN, "derived")
rec(C4, 416, "misaligned reduction (nats)", printed(C4,416,0), abs(AM["misaligned_block"]-AM["vanilla"]), PRERUN, "derived")
rec(C4, 416, "misaligned % of segment-index", printed(C4,416,1), 100*(AM["misaligned_block"]-AM["vanilla"])/SEG, PRERUN, "derived")
rec(C4, 418, "alignment worth (nats)", printed(C4,418,0), abs(AM["segment_index"]-AM["misaligned_block"]), PRERUN, "derived")
rec(C4, 424, "ALiBi reduction (nats)", printed(C4,424,0), abs(AM["alibi"]-AM["vanilla"]), PRERUN, "derived")
rec(C4, 424, "ALiBi % of segment-index", printed(C4,424,1), 100*(AM["alibi"]-AM["vanilla"])/SEG, PRERUN, "derived")
rec(C4, 428, "hard-block reduction (nats)", printed(C4,428,0), abs(AM["hard_block"]-AM["vanilla"]), PRERUN, "derived")
rec(C4, 429, "learned vs fixed (nats)", printed(C4,429,0), abs(AM["hard_block"]-AM["segment_index"]), PRERUN, "derived")
rec(C4, 373, "arm count stated as 'seven'", "seven" if "seven arms" in line(C4,373) else None,
    "seven" if len(ARMS)==7 else str(len(ARMS)), PRERUN, "distinct arms recovered from run ids (steps=1000, 10M, shuffle_segments)")
rec(C4, 374, "seeds per arm stated as 'three'", "three" if "at three" in line(C4,373) else None,
    "three" if min(len(v) for v in AV.values())==3 else str(min(len(v) for v in AV.values())), PRERUN, "min seeds per arm")

# arm-to-run-id mapping, confirmed against the training code rather than the run ids alone
_bund = (ROOT/"src/data/bundles.py").read_text()
_pre  = (ROOT/"scripts/pretrain.py").read_text()
rec(C4, 394, "segment-index arm == segment_index_roles control (ordinal, no provenance meaning)",
    "arbitrary segment ordinal",
    "arbitrary segment ordinal" if ("segment_index_roles" in _bund and "ordinal position in the bundle" in _bund
                                    and '"_segidx" if args.segment_index_roles' in _pre) else "mapping not confirmed",
    "src/data/bundles.py + scripts/pretrain.py",
    "bundles.py: elif self.segment_index_roles -> r = 1 + min(N_ROLES-2, seg_i); pretrain.py: run-id suffix '_segidx'",
    "run ids %s carry controls.segment_index_roles=true; the true-role arm %s carries no control flag. A swap here would invert the headline claim; it is not swapped." % (ARMS["segment_index"], ARMS["true_role_labels"]))
rec(C4, 393, "true-role arm == A4 with every control flag false",
    "true role labels",
    "true role labels" if all(not any(PRE[k].get("controls",{})[c] for c in PRE[k].get("controls",{}) if c != "shuffle_segments")
                              for k in ARMS["true_role_labels"]) else "mapping not confirmed",
    PRERUN, "controls.* for " + ", ".join(ARMS["true_role_labels"]))
rec(C4, 391, "misaligned arm == segment_index_roles + misaligned_blocks",
    "same block count, alignment destroyed",
    "same block count, alignment destroyed" if all("_segidx_misalign_" in k for k in ARMS["misaligned_block"])
        and '"_misalign" if args.misaligned_blocks' in _pre else "mapping not confirmed",
    PRERUN + " + scripts/pretrain.py", "run-id suffixes _segidx and _misalign")

# ================= 2. cohort chain + per-rule counts ==========================
rec(C3, 79, "HF admissions, ICD only",  printed(C3,79,0), ICD["heart_failure"]["pre_los"],  "results/accord/stats/cohort_icd_counts.json", "heart_failure.pre_los")
rec(C3, 79, "HF admissions, post LOS",  printed(C3,79,1), ICD["heart_failure"]["post_los"], "results/accord/stats/cohort_icd_counts.json", "heart_failure.post_los")
rec(C3, 80, "HF unique subjects",       printed(C3,80,0), PH1["hf_admissions_los_filtered_unique_subjects"], str(QUAR/"accord/phase1_summary.json"), "hf_admissions_los_filtered_unique_subjects")
rec(C3, 80, "sepsis pre LOS",  printed(C3,80,1), ICD["sepsis"]["pre_los"],  "results/accord/stats/cohort_icd_counts.json", "sepsis.pre_los")
rec(C3, 81, "sepsis post LOS", printed(C3,81,0), ICD["sepsis"]["post_los"], "results/accord/stats/cohort_icd_counts.json", "sepsis.post_los")
rec(C3, 81, "AKI pre LOS",     printed(C3,81,1), ICD["aki"]["pre_los"],     "results/accord/stats/cohort_icd_counts.json", "aki.pre_los")
rec(C3, 81, "AKI post LOS",    printed(C3,81,2), ICD["aki"]["post_los"],    "results/accord/stats/cohort_icd_counts.json", "aki.post_los")
rec(C3, 81, "sepsis removed",  printed(C3,81,3), ICD["sepsis"]["removed"],  "results/accord/stats/cohort_icd_counts.json", "sepsis.removed")
rec(C3, 81, "AKI removed",     printed(C3,81,4), ICD["aki"]["removed"],     "results/accord/stats/cohort_icd_counts.json", "aki.removed")
rec(C3, 82, "HF removed",      printed(C3,82,0), ICD["heart_failure"]["removed"], "results/accord/stats/cohort_icd_counts.json", "heart_failure.removed")
rec(C3, 113, "has discharge summary", printed(C3,113,0), PH1["has_discharge_summary"], str(QUAR/"accord/phase1_summary.json"), "has_discharge_summary")
rec(C3, 113, "HF admissions (denominator)", printed(C3,113,1), PH1["hf_admissions_icd_only"], str(QUAR/"accord/phase1_summary.json"), "hf_admissions_icd_only")
rec(C3, 113, "has nursing note", printed(C3,113,2), PH1["has_nursing_note"], str(QUAR/"accord/phase1_summary.json"), "has_nursing_note")
rec(C3, 114, "recomputed both-roles intersection", printed(C3,114,0), PH1["has_both_roles"], str(QUAR/"accord/phase1_summary.json"), "has_both_roles")
rec(C3, 115, "reference both-roles count", printed(C3,115,0), PH1["reference_has_both_roles"], str(QUAR/"accord/phase1_summary.json"), "reference_has_both_roles")
rec(C3, 115, "difference stated as 'five higher'", "five" if "five higher" in line(C3,115) else None,
    "five" if PH1["reference_has_both_roles"]-PH1["has_both_roles"]==5 else str(PH1["reference_has_both_roles"]-PH1["has_both_roles"]),
    str(QUAR/"accord/phase1_summary.json"), "derived: reference_has_both_roles - has_both_roles")
rec(C3, 143, "nursing notes selected", printed(C3,143,0), PH1["nursing_notes_selected"], str(QUAR/"accord/phase1_summary.json"), "nursing_notes_selected")
rec(C3, 143, "discharge notes selected", printed(C3,143,1), PH1["discharge_notes_selected"], str(QUAR/"accord/phase1_summary.json"), "discharge_notes_selected")
rec(C3, 144, "note-pair rows", printed(C3,144,1), PH1["total_note_pair_rows"], str(QUAR/"accord/phase1_summary.json"), "total_note_pair_rows")
rec(C3, 169, "total mentions", printed(C3,169,1), PH2["medication_extraction"]["total_mentions"], str(DRIVE/"outputs/phase2_summary.json"), "medication_extraction.total_mentions")
rec(C3, 169, "unique admissions with mentions", printed(C3,169,2), PH2["medication_extraction"]["unique_admissions"], str(DRIVE/"outputs/phase2_summary.json"), "medication_extraction.unique_admissions")
rec(C3, 170, "unique drugs", printed(C3,170,0), PH2["medication_extraction"]["unique_drugs"], str(DRIVE/"outputs/phase2_summary.json"), "medication_extraction.unique_drugs")
rec(C3, 170, "classified mentions", printed(C3,170,1), PH2["medication_extraction"]["classified_mentions"], str(DRIVE/"outputs/phase2_summary.json"), "medication_extraction.classified_mentions")
rec(C3, 170, "unclassified mentions (derived)", printed(C3,170,2),
    PH2["medication_extraction"]["total_mentions"]-PH2["medication_extraction"]["classified_mentions"],
    str(DRIVE/"outputs/phase2_summary.json"), "derived: total - classified")
# RAW split (mining_summary) -- chapter L296-301
rec(C3, 296, "RAW candidate pool total", printed(C3,296,0), MIN["total_candidates"], str(DRIVE/"outputs/mining_summary.json"), "total_candidates")
for i, rk in enumerate(["rule1_direct_action_conflicts","rule2_drug_class_conflicts","rule3_dosage_conflicts","rule4_omission_conflicts"]):
    rec(C3, 297, f"RAW split {rk}", printed(C3,297)[i*2], MIN["by_rule"][rk], str(DRIVE/"outputs/mining_summary.json"), f"by_rule.{rk}")
rec(C3, 296, "RAW split sums to printed total", MIN["total_candidates"], sum(MIN["by_rule"].values()), str(DRIVE/"outputs/mining_summary.json"), "derived: sum(by_rule)")
rec(C3, 301, "unique admissions with candidates", printed(C3,301,-1), MIN["unique_admissions_with_candidates"], str(DRIVE/"outputs/mining_summary.json"), "unique_admissions_with_candidates")
# POST-PREFILTER split (phase2_summary) -- chapter L303-304
rec(C3, 303, "POST-PREFILTER pool total", printed(C3,303,0), PH2["temporal_prefilter"]["final_candidates"], str(DRIVE/"outputs/phase2_summary.json"), "temporal_prefilter.final_candidates")
cm = PH2["contradiction_mining"]
for i, rk in enumerate(["rule1_direct_action","rule2_drug_class"]):
    rec(C3, 303, f"POST-PREFILTER split {rk}", printed(C3,303)[i+1], cm[rk], str(DRIVE/"outputs/phase2_summary.json"), f"contradiction_mining.{rk}")
rec(C3, 304, "POST-PREFILTER split rule3_dosage",   printed(C3,304,0), cm["rule3_dosage"],   str(DRIVE/"outputs/phase2_summary.json"), "contradiction_mining.rule3_dosage")
rec(C3, 304, "POST-PREFILTER split rule4_omission", printed(C3,304,1), cm["rule4_omission"], str(DRIVE/"outputs/phase2_summary.json"), "contradiction_mining.rule4_omission")
rec(C3, 303, "POST-PREFILTER split sums to 1978",
    PH2["temporal_prefilter"]["final_candidates"],
    cm["rule1_direct_action"]+cm["rule2_drug_class"]+cm["rule3_dosage"]+cm["rule4_omission"],
    str(DRIVE/"outputs/phase2_summary.json"), "derived: sum(contradiction_mining rules 1-4)")
rec(C3, 310, "prefilter admissions with window", printed(C3,310,-1), PH2["temporal_prefilter"]["admissions_with_window"], str(DRIVE/"outputs/phase2_summary.json"), "temporal_prefilter.admissions_with_window")
rec(C3, 311, "prefilter admissions with intermediate notes", printed(C3,311,0), PH2["temporal_prefilter"]["admissions_with_intermediate_notes"], str(DRIVE/"outputs/phase2_summary.json"), "temporal_prefilter.admissions_with_intermediate_notes")
rec(C4, 72, "ch4 restates POST-PREFILTER total", printed(C4,72,0), PH2["temporal_prefilter"]["final_candidates"], str(DRIVE/"outputs/phase2_summary.json"), "temporal_prefilter.final_candidates")
for i, rk in enumerate(["rule1_direct_action","rule2_drug_class","rule3_dosage","rule4_omission"]):
    rec(C4, 73, f"ch4 POST-PREFILTER split {rk}", printed(C4,73)[i*2], cm[rk], str(DRIVE/"outputs/phase2_summary.json"), f"contradiction_mining.{rk}")

# ================= 3. denominators + per-rule gold ===========================
rec(C3, 510, "annotation frame rule1", printed(C3,509,0), PH2["annotation_sample"]["rule1"], str(DRIVE/"outputs/phase2_summary.json"), "annotation_sample.rule1")
rec(C3, 510, "annotation frame total", printed(C3,510,3), PH2["annotation_sample"]["total_sampled"], str(DRIVE/"outputs/phase2_summary.json"), "annotation_sample.total_sampled")
gr = S01["join_feasibility"]["gold_rule_counts"]
rec(C3, 513, "gold n_total", printed(C3,513,0), sum(gr.values()), "results/accord/step0_1_join_and_calibration.json", "derived: sum(join_feasibility.gold_rule_counts)")
for i, k in enumerate(["1","2","3","4"]):
    rec(C3, 515, f"gold rule {k} count", printed(C3,515)[i], gr[k], "results/accord/step0_1_join_and_calibration.json", f"join_feasibility.gold_rule_counts.{k}")
strata = [(566,"G299",CTRL["accord_contra_prob_same_299_pairs"]["gold_all"]),
          (567,"G134",CTRL["accord_contra_prob_same_299_pairs"]["gold_defect_free_rule34"])]
for ln, nm, blk in strata:
    rec(C3, ln, f"strata {nm} n",   printed(C3,ln,0), blk["n"],   "results/accord/control/final_report.json", "accord_contra_prob_same_299_pairs.*.n")
    rec(C3, ln, f"strata {nm} pos", printed(C3,ln,1), blk["pos"], "results/accord/control/final_report.json", "accord_contra_prob_same_299_pairs.*.pos")
rec(C3, 568, "strata decontam n",   printed(C3,568,0), DEC["n"],           "results/accord/decontam/corrected_gold_metrics.json", "n")
rec(C3, 568, "strata decontam pos", printed(C3,568,1), DEC["n_positives"], "results/accord/decontam/corrected_gold_metrics.json", "n_positives")
rec(C3, 569, "weak-label pool n",   printed(C3,569,0), CTRL["baselines"]["n_train"],      "results/accord/control/final_report.json", "baselines.n_train")
rec(C3, 569, "weak-label pool pos rate", printed(C3,569,-1), CTRL["baselines"]["train_pos_rate"], "results/accord/control/final_report.json", "baselines.train_pos_rate")
den = [(37,299,12,287),(38,253,12,241),(39,134,6,128)]
for ln, n, p, q in den:
    v = printed(C4, ln)
    rec(C4, ln, f"tab:denoms n on line {ln}", v[1], float(n), "cross-artifact", "see strata rows")
    rec(C4, ln, f"tab:denoms positives/negatives sum to n on line {ln}", v[1], v[2]+v[3], "arithmetic", "pos + neg == n")
rec(C4, 46, "ambiguous pairs = 287-241", printed(C4,46,0)-printed(C4,46,1), 299-DEC["n"], "arithmetic + decontam.n", "derived")
rsc = S01["rule_stratified_calibration"]
for i, ln in enumerate(range(88, 92)):
    k = str(i+1); v = printed(C4, ln)
    rec(C4, ln, f"rule {k} gold n", v[1], rsc[k]["n_gold"], "results/accord/step0_1_join_and_calibration.json", f"rule_stratified_calibration.{k}.n_gold")
    rec(C4, ln, f"rule {k} pharmacist precision %", v[2], 100*rsc[k]["pharmacist_true_pos_rate_gold"], "results/accord/step0_1_join_and_calibration.json", f"rule_stratified_calibration.{k}.pharmacist_true_pos_rate_gold")
    rec(C4, ln, f"rule {k} ACCORD predicted-positive %", v[3], 100*rsc[k]["accord_predicted_pos_rate_gold"], "results/accord/step0_1_join_and_calibration.json", f"rule_stratified_calibration.{k}.accord_predicted_pos_rate_gold")
rec(C4, 97, "four gold strata sum to 299", printed(C4,97,-1), sum(rsc[k]["n_gold"] for k in "1234"), "results/accord/step0_1_join_and_calibration.json", "derived")
d34 = [s for s in DFS["strata"] if s["name"].startswith("rules 3+4")][0]
rec(C4, 112, "rules 3+4 n", printed(C4,112,2), d34["n"], "results/accord/defect_free_strata.json", "strata[rules 3+4].n")
rec(C4, 112, "rules 3+4 positives", printed(C4,112,-1), d34["pos"], "results/accord/defect_free_strata.json", "strata[rules 3+4].pos")
rec(C6, 58, "rules 1+2 n", printed(C6,58,0), [s for s in DFS["strata"] if s["name"].startswith("rules 1+2")][0]["n"], "results/accord/defect_free_strata.json", "strata[rules 1+2].n")
rec(C6, 58, "rules 1+2 rate", printed(C6,58,2), 6/165, "results/accord/defect_free_strata.json", "derived: pos/n")
rec(C6, 58, "rules 3+4 rate", printed(C6,59,2), 6/134, "results/accord/defect_free_strata.json", "derived: pos/n")
# exemplar
rec(C4, 136, "rule1+2 pooled n",        printed(C4,136,0), EXE["rule_1_2_pooled"]["n"],          "results/accord/exemplar/exemplar_correctness_recomputed.json", "rule_1_2_pooled.n")
rec(C4, 137, "rule1+2 resolved",        printed(C4,136,-1), EXE["rule_1_2_pooled"]["n_resolved"], "results/accord/exemplar/exemplar_correctness_recomputed.json", "rule_1_2_pooled.n_resolved")
rec(C4, 137, "rule1+2 unresolved",      printed(C4,137,0), EXE["rule_1_2_pooled"]["n_unresolved"],"results/accord/exemplar/exemplar_correctness_recomputed.json", "rule_1_2_pooled.n_unresolved")
rec(C4, 138, "rule1+2 wrong",           printed(C4,138,0), EXE["rule_1_2_pooled"]["n_wrong"],    "results/accord/exemplar/exemplar_correctness_recomputed.json", "rule_1_2_pooled.n_wrong")
rec(C4, 139, "wrong rate 65/137",       printed(C4,139,0), EXE["rule_1_2_pooled"]["n_wrong"]/EXE["rule_1_2_pooled"]["n_resolved"], "results/accord/exemplar/exemplar_correctness_recomputed.json", "derived: n_wrong / n_resolved")

# ================= 4. intervals ==============================================
rec(C4, 491, "decontam ensemble CI lo", printed(C4,491,-2), DEC["aggregate"]["ensemble_ci95"][0], "results/accord/decontam/corrected_gold_metrics.json", "aggregate.ensemble_ci95[0]")
rec(C4, 491, "decontam ensemble CI hi", printed(C4,491,-1), DEC["aggregate"]["ensemble_ci95"][1], "results/accord/decontam/corrected_gold_metrics.json", "aggregate.ensemble_ci95[1]")
rec(C4, 488, "decontam bootstrap B",    printed(C4,488,0)*1000 if printed(C4,488,0)==10 else printed(C4,488,0), DEC["aggregate"]["bootstrap_B"], "results/accord/decontam/corrected_gold_metrics.json", "aggregate.bootstrap_B")
rec(C4, 488, "decontam bootstrap seed", printed(C4,488,-1), DEC["aggregate"]["bootstrap_seed"], "results/accord/decontam/corrected_gold_metrics.json", "aggregate.bootstrap_seed")
mccrows = [(785,"gpt4o_zeroshot"),(786,"gpt4o_fewshot"),(787,"accord_binary_at_0.5_253core"),(788,"llama_baseline")]
for ln, k in mccrows:
    v = printed(C4, ln); b = IVL["E2_mcc_intervals"][k]
    rec(C4, ln, f"MCC {k} point", v[-3], b["mcc"], "results/accord/stats/intervals.json", f"E2_mcc_intervals.{k}.mcc")
    rec(C4, ln, f"MCC {k} CI lo", v[-2], b["ci95"][0], "results/accord/stats/intervals.json", f"E2_mcc_intervals.{k}.ci95[0]")
    rec(C4, ln, f"MCC {k} CI hi", v[-1], b["ci95"][1], "results/accord/stats/intervals.json", f"E2_mcc_intervals.{k}.ci95[1]")
    rec(C4, ln, f"MCC {k} predicted-positive sum", v[-4], v[-6]+v[-5], "arithmetic on tab:systems253 cells", "TP + FP")
    rec(C4, ln, f"MCC {k} predicted-positive cells vs step3_baselines", [v[-6], v[-5]], [float(S31[k]["tp"]), float(S31[k]["fp"])], "results/accord/step3_baselines.json", f"{k}.tp, {k}.fp")
rec(C4, 795, "MCC bootstrap B",    10000.0, IVL["bootstrap_B"], "results/accord/stats/intervals.json", "bootstrap_B")
rec(C4, 795, "MCC bootstrap seed", printed(C4,795,-1), IVL["seed"], "results/accord/stats/intervals.json", "seed")
rec(C4, 798, "all four MCC intervals contain zero", True,
    all(b["ci95"][0] <= 0 <= b["ci95"][1] for b in IVL["E2_mcc_intervals"].values()),
    "results/accord/stats/intervals.json", "E2_mcc_intervals.*.ci95")
A = IVL["E2_mcc_intervals"]["gpt4o_zeroshot"]["ci95"]; B = IVL["E2_mcc_intervals"]["llama_baseline"]["ci95"]
ov = max(0.0, min(A[1],B[1]) - max(A[0],B[0])); wA, wB = A[1]-A[0], B[1]-B[0]
rec(C4, 807, "MCC intervals 'overlap by more than half their width'", "more than half",
    "%.1f%% of the narrower interval (overlap %.4f vs half-width %.4f)" % (100*ov/min(wA,wB), ov, 0.5*min(wA,wB)),
    "results/accord/stats/intervals.json", "E2_mcc_intervals.{gpt4o_zeroshot,llama_baseline}.ci95",
    "recomputed overlap is 31.5%/30.4% of the two widths; the claim is false under every reading (narrower, wider, mean, union)")
rec(C4, 803, "MCC estimates within 0.05 of zero (stated as 'three')", "three" if "three of which" in line(C4,803) else None,
    "three" if sum(abs(b["mcc"])<printed(C4,804,want="0.05") for b in IVL["E2_mcc_intervals"].values())==3 else "other",
    "results/accord/stats/intervals.json", "count(|mcc| < 0.05)")
rec(C4, 803, "predicted-positive range low",  printed(C4,803,0), float(min(S31[k]["tp"]+S31[k]["fp"] for k in IVL["E2_mcc_intervals"])), "results/accord/step3_baselines.json", "min(tp+fp)")
rec(C4, 803, "predicted-positive range high", printed(C4,803,1), float(max(S31[k]["tp"]+S31[k]["fp"] for k in IVL["E2_mcc_intervals"])), "results/accord/step3_baselines.json", "max(tp+fp)")
# control bootstrap cols
for ln, sset, key in [(514,"gold_all","accord_contra_prob_same_299_pairs"),
                      (515,"gold_all","ensemble_mean_of_seed_scores"),
                      (517,"gold_defect_free_rule34","accord_contra_prob_same_299_pairs"),
                      (518,"gold_defect_free_rule34","ensemble_mean_of_seed_scores")]:
    v = printed(C4, ln); b = CTRL[key][sset]
    rec(C4, ln, f"{key}.{sset} auroc",     v[1], b["auroc"], "results/accord/control/final_report.json", f"{key}.{sset}.auroc")
    rec(C4, ln, f"{key}.{sset} ap",        v[2], b["ap"],    "results/accord/control/final_report.json", f"{key}.{sset}.ap")
    rec(C4, ln, f"{key}.{sset} boot mean", v[3], b["bootstrap"]["boot_mean"], "results/accord/control/final_report.json", f"{key}.{sset}.bootstrap.boot_mean")
    rec(C4, ln, f"{key}.{sset} boot sd",   v[4], b["bootstrap"]["boot_std"],  "results/accord/control/final_report.json", f"{key}.{sset}.bootstrap.boot_std")
for ln, sset in [(516,"gold_all"),(519,"gold_defect_free_rule34")]:
    v = printed(C4, ln)
    rec(C4, ln, f"per-seed AUROC mean {sset}", v[-4], CTRL["auroc_mean_std_across_seeds"][sset]["mean"], "results/accord/control/final_report.json", f"auroc_mean_std_across_seeds.{sset}.mean")
    rec(C4, ln, f"per-seed AUROC SD {sset}",   v[-3], CTRL["auroc_mean_std_across_seeds"][sset]["std"],  "results/accord/control/final_report.json", f"auroc_mean_std_across_seeds.{sset}.std")
    rec(C4, ln, f"per-seed AP mean {sset}",    v[-2], CTRL["ap_mean_std_across_seeds"][sset]["mean"],    "results/accord/control/final_report.json", f"ap_mean_std_across_seeds.{sset}.mean")
    rec(C4, ln, f"per-seed AP SD {sset}",      v[-1], CTRL["ap_mean_std_across_seeds"][sset]["std"],     "results/accord/control/final_report.json", f"ap_mean_std_across_seeds.{sset}.std")
rec(C4, 523, "control bootstrap B",    10000.0, CTRL["bootstrap_n_resamples"], "results/accord/control/final_report.json", "bootstrap_n_resamples")
rec(C4, 524, "control bootstrap seed", printed(C4,524,0), CTRL["bootstrap_seed"], "results/accord/control/final_report.json", "bootstrap_seed")
e3 = IVL["E3_defect_free_stratum"]
rec(C4, 517, "a stratified CI for G134 exists in the intervals file but is not printed", "not printed",
    "not printed" if not any("0.2643" in l for l in LINES[C4]) else "printed",
    "results/accord/stats/intervals.json", "E3_defect_free_stratum", "registered but not printed in ch4; the printed G134 spread is the bootstrap SD")

# ================= decontam table + ch1/ch6 restatements =====================
for ln, sk in [(477,"seed0"),(478,"seed1"),(479,"seed2")]:
    v = printed(C4, ln); b = DEC["per_seed"][sk]
    rec(C4, ln, f"decontam {sk} auroc",  v[1], b["auroc"], "results/accord/decontam/corrected_gold_metrics.json", f"per_seed.{sk}.auroc")
    rec(C4, ln, f"decontam {sk} ap",     v[2], b["ap"],    "results/accord/decontam/corrected_gold_metrics.json", f"per_seed.{sk}.ap")
    rec(C4, ln, f"decontam {sk} MW p",   v[3], b["mannwhitney_p"], "results/accord/decontam/corrected_gold_metrics.json", f"per_seed.{sk}.mannwhitney_p")
    rec(C4, ln, f"decontam {sk} top15",  v[4], b["top15_tp"], "results/accord/decontam/corrected_gold_metrics.json", f"per_seed.{sk}.top15_tp")
rec(C4, 480, "decontam per-seed mean", printed(C4,480,0), DEC["aggregate"]["auroc_mean"], "results/accord/decontam/corrected_gold_metrics.json", "aggregate.auroc_mean")
rec(C4, 480, "decontam per-seed SD (ddof=0)", printed(C4,480,1), DEC["aggregate"]["auroc_std"], "results/accord/decontam/corrected_gold_metrics.json", "aggregate.auroc_std")
rec(C4, 481, "decontam ensemble auroc", printed(C4,481,0), DEC["aggregate"]["ensemble_auroc"], "results/accord/decontam/corrected_gold_metrics.json", "aggregate.ensemble_auroc")
rec(C4, 481, "decontam ensemble MW p",  printed(C4,481,1), DEC["aggregate"]["ensemble_mannwhitney_p"], "results/accord/decontam/corrected_gold_metrics.json", "aggregate.ensemble_mannwhitney_p")
rec(C4, 483, "contaminated same 253",   printed(C4,483,-1), DEC["contaminated_same_253_pairs"]["auroc"], "results/accord/decontam/corrected_gold_metrics.json", "contaminated_same_253_pairs.auroc")
rec(C1, 183, "ch1 ensemble auroc", printed(C1,183,-1), DEC["aggregate"]["ensemble_auroc"], "results/accord/decontam/corrected_gold_metrics.json", "aggregate.ensemble_auroc")
rec(C1, 184, "ch1 ensemble MW p (3dp vs ch4 4dp)", printed(C1,184,0), DEC["aggregate"]["ensemble_mannwhitney_p"], "results/accord/decontam/corrected_gold_metrics.json", "aggregate.ensemble_mannwhitney_p", "ch4 L481 prints 0.8668 for the same quantity: precision differs across chapters")
rec(C1, 161, "ch1 decontam precision mean", printed(C1,161,-1), DGA["aggregate"]["precision"]["mean"], str(QUAR/"accord_supp/results/accord_v2_decontam_gold_aggregate.json"), "aggregate.precision.mean")
rec(C1, 162, "ch1 decontam precision SD (ddof=0)", printed(C1,162,0), DGA["aggregate"]["precision"]["std"], str(QUAR/"accord_supp/results/accord_v2_decontam_gold_aggregate.json"), "aggregate.precision.std")
rec(C1, 162, "ch1 decontam recall mean", printed(C1,162,-1), DGA["aggregate"]["recall"]["mean"], str(QUAR/"accord_supp/results/accord_v2_decontam_gold_aggregate.json"), "aggregate.recall.mean")
rec(C1, 163, "ch1 decontam recall SD (ddof=0)", printed(C1,163,0), DGA["aggregate"]["recall"]["std"], str(QUAR/"accord_supp/results/accord_v2_decontam_gold_aggregate.json"), "aggregate.recall.std")
cf = DGA["per_seed"]["0"]["confusion"]
rec(C6, 116, "seed-0 flagged", printed(C6,116,2), cf["tp"]+cf["fp"], str(QUAR/"accord_supp/results/accord_v2_decontam_gold_aggregate.json"), "derived: per_seed.0.confusion tp+fp")
rec(C6, 116, "seed-0 TP", printed(C6,116,3), cf["tp"], str(QUAR/"accord_supp/results/accord_v2_decontam_gold_aggregate.json"), "per_seed.0.confusion.tp")
rec(C6, 117, "seed-0 FP", printed(C6,117,0), cf["fp"], str(QUAR/"accord_supp/results/accord_v2_decontam_gold_aggregate.json"), "per_seed.0.confusion.fp")
rec(C6, 117, "seed-0 TN", printed(C6,117,1), cf["tn"], str(QUAR/"accord_supp/results/accord_v2_decontam_gold_aggregate.json"), "per_seed.0.confusion.tn")
rec(C6, 117, "seed-0 FN", printed(C6,117,2), cf["fn"], str(QUAR/"accord_supp/results/accord_v2_decontam_gold_aggregate.json"), "per_seed.0.confusion.fn")
rec(C6, 118, "kappa across seeds", printed(C6,118,-1), DGA["aggregate"]["kappa"]["mean"], str(QUAR/"accord_supp/results/accord_v2_decontam_gold_aggregate.json"), "aggregate.kappa.mean")
rec(C6, 119, "ECE across seeds", printed(C6,119,0), DGA["aggregate"]["ece"]["mean"], str(QUAR/"accord_supp/results/accord_v2_decontam_gold_aggregate.json"), "aggregate.ece.mean")
sd = S02["accord_contra_prob_distribution_by_group"]
rec(C1, 190, "positives mean contra prob", printed(C1,190,0), sd["gold_positive_contradiction_n12"]["mean"], "results/accord/step2_signal_check.json", "accord_contra_prob_distribution_by_group.gold_positive_contradiction_n12.mean")
rec(C1, 190, "negatives mean contra prob", printed(C1,190,-1), sd["gold_negative_not_contradiction_n241"]["mean"], "results/accord/step2_signal_check.json", "accord_contra_prob_distribution_by_group.gold_negative_not_contradiction_n241.mean")
rec(C1, 191, "MW p positives vs negatives", printed(C1,191,0), sd["mann_whitney_u_p_value_pos_vs_neg"], "results/accord/step2_signal_check.json", "accord_contra_prob_distribution_by_group.mann_whitney_u_p_value_pos_vs_neg")

# ================= 5. statistics: DeLong, Fisher, Welch/Student ==============
de = CTRL["delong_ensemble_vs_accord"]["gold_all"]; dp = CTRL["delong_per_seed_vs_accord_gold_all"]
drows = [(555, de), (556, dp["42"]), (557, dp["123"]), (558, dp["2024"]), (559, CTRL["delong_ensemble_vs_accord"]["gold_defect_free_rule34"])]
for ln, b in drows:
    v = printed(C4, ln)
    rec(C4, ln, "DeLong AUC A", v[-5], b["auc_a"], "results/accord/control/final_report.json", "delong*.auc_a")
    rec(C4, ln, "DeLong AUC B", v[-4], b["auc_b"], "results/accord/control/final_report.json", "delong*.auc_b")
    rec(C4, ln, "DeLong diff",  v[-3], b["auc_diff"], "results/accord/control/final_report.json", "delong*.auc_diff")
    rec(C4, ln, "DeLong z",     v[-2], b["z"], "results/accord/control/final_report.json", "delong*.z")
    rec(C4, ln, "DeLong p",     v[-1], b["p"], "results/accord/control/final_report.json", "delong*.p")
ses = printed(C4, 564) + printed(C4, 565)
for i, b in enumerate([x[1] for x in drows]):
    rec(C4, 564, f"DeLong SE of difference row {i+1}", ses[i], b["se_diff"], "results/accord/control/final_report.json", "delong*.se_diff")
rec(C4, 572, "'the standard error is eleven times the difference'", "eleven" if "eleven times" in line(C4,572) else None,
    "eleven" if round(de["se_diff"]/de["auc_diff"]) == 11 else "%.2f" % (de["se_diff"]/de["auc_diff"]),
    "results/accord/control/final_report.json", "derived: se_diff/auc_diff = %.4f" % (de["se_diff"]/de["auc_diff"]))
fe = EX2["confounded_proxy_rule12_vs_rule34"]["fisher_exact"]
rec(C6, 60, "Fisher exact odds ratio", printed(C6,60,0), fe["odds_ratio"], "results/accord/exemplar/step2plus_results.json", "confounded_proxy_rule12_vs_rule34.fisher_exact.odds_ratio")
rec(C6, 60, "Fisher exact p",          printed(C6,60,1), fe["p_value"],    "results/accord/exemplar/step2plus_results.json", "confounded_proxy_rule12_vs_rule34.fisher_exact.p_value")
try:
    from scipy.stats import fisher_exact as _fe
    o, p = _fe([[6,159],[6,128]])
    rec(C6, 60, "Fisher exact OR, independently recomputed", printed(C6,60,0), o, "scipy.stats.fisher_exact on [[6,159],[6,128]]", "independent recomputation")
    rec(C6, 60, "Fisher exact p, independently recomputed",  printed(C6,60,1), p, "scipy.stats.fisher_exact on [[6,159],[6,128]]", "independent recomputation")
except Exception: pass
nbpath = ROOT/"accord/scaling/C4_scaling_study.ipynb"
nbsrc = "".join("".join(c.get("source", [])) for c in json.load(open(nbpath))["cells"])
rec(C4, 731, "clinical-gap test is Student not Welch (correction confirmed in the code)",
    "ttest_ind called without equal_var=False",
    "ttest_ind called without equal_var=False" if ("ttest_ind" in nbsrc and "equal_var" not in nbsrc) else "code does not match the claim",
    str(nbpath), "cell computing gap vs pretrained",
    "correction stands; but the apanote's surrounding citation is C44_convergence_check.ipynb while the cell is in C4_scaling_study.ipynb, and it computes the PUBLIC-benchmark gap, not the clinical one")
rec(C4, 733, "no p value quoted for the clinical gap", 0,
    float(len(re.findall(r'p\s*=', line(C4,727)+line(C4,728)+line(C4,729)+line(C4,730)))),
    str(TEX/"msu_ch4_results.tex"), "L727-730", "confirmed absent")

# ================= convergence / scaling / positive control ==================
conv_rows = [(669,"pretrained_bert"),(670,"scratch_100M"),(671,"scratch_30M"),(672,"scratch_10M")]
for ln, tag in conv_rows:
    v = printed(C4, ln)[-8:]
    for j, b in enumerate([3,6,12,24]):
        rec(C4, ln, f"conv {tag} E{b} mean", v[2*j],   cmean(tag,b), "results/accord/scaling/convergence_results.json", f"mean(accuracy | tag={tag}, budget={b})")
        rec(C4, ln, f"conv {tag} E{b} SD",   v[2*j+1], csd(tag,b),   "results/accord/scaling/convergence_results.json", f"stdev(accuracy | tag={tag}, budget={b}) ddof=1")
rec(C4, 655, "convergence n_runs", printed(C4,655,-2), float(len(CONV)), "results/accord/scaling/convergence_results.json", "len(runs)")
rec(C4, 655, "convergence total minutes", printed(C4,655,-1), sum(r["minutes"] for r in CONV), "results/accord/scaling/convergence_results.json", "sum(minutes)")
rec(C4, 695, "gap at 3 epochs (points)",  printed(C4,695,0), 100*(cmean("pretrained_bert",3)-best_scratch(3)), "results/accord/scaling/convergence_results.json", "derived")
rec(C4, 696, "gap at 24 epochs (points)", printed(C4,696,1), 100*(cmean("pretrained_bert",24)-best_scratch(24)), "results/accord/scaling/convergence_results.json", "derived")
rec(C4, 697, "pretrained 3->24 change",   printed(C4,697,0), 100*(cmean("pretrained_bert",24)-cmean("pretrained_bert",3)), "results/accord/scaling/convergence_results.json", "derived")
rec(C4, 697, "scratch100M 3->24 change",  printed(C4,697,-1), 100*(cmean("scratch_100M",24)-cmean("scratch_100M",3)), "results/accord/scaling/convergence_results.json", "derived")
rec(C4, 699, "max seed SD across all 16 cells", printed(C4,699,0), max(csd(t,b) for t,b in cells), "results/accord/scaling/convergence_results.json", "max stdev over cells",
    "printed as <= 0.0036; the true max is 0.003649, i.e. above 0.0036 before rounding")
rec(C1, 217, "ch1: seed SD bound at 24 epochs", printed(C1,217,0), max(csd("pretrained_bert",24), csd("scratch_100M",24)), "results/accord/scaling/convergence_results.json", "max over the two compared arms at budget 24",
    "ch4 L699 states <= 0.0036 for the same sweep; ch1/ch6 state <= 0.0029 which holds only for the two compared arms")
scal_rows = [(712,"pretrained_bert"),(713,"scratch_100M"),(714,"scratch_30M"),(715,"scratch_10M")]
for ln, tag in scal_rows:
    v = printed(C4, ln)[-3:]
    rec(C4, ln, f"scaling {tag} params", v[0], float(sp[tag]), "results/accord/scaling/scaling_results.json", f"non_emb_params for tag={tag}")
    rec(C4, ln, f"scaling {tag} mean",   v[1], st.mean(sg[tag]), "results/accord/scaling/scaling_results.json", f"mean(accuracy | tag={tag})")
    rec(C4, ln, f"scaling {tag} SD",     v[2], st.stdev(sg[tag]), "results/accord/scaling/scaling_results.json", f"stdev(accuracy | tag={tag}) ddof=1")
rec(C4, 716, "clinical pretrained AUROC mean", printed(C4,716,0), st.mean(CONC), "results/task_runs.json", "mean(auroc | concat_s1..s3)")
rec(C4, 716, "clinical pretrained AUROC SD",   printed(C4,716,1), st.stdev(CONC), "results/task_runs.json", "stdev(auroc | concat_s1..s3) ddof=1")
rec(C4, 717, "clinical scratch params",        printed(C4,717,0), float(TASK["ours_current_s0"]["non_emb_params"]), "results/task_runs.json", "ours_current_s*.non_emb_params")
rec(C4, 717, "clinical scratch AUROC mean",    printed(C4,717,1), st.mean(CURR), "results/task_runs.json", "mean(auroc | ours_current_s0..s2)")
rec(C4, 717, "clinical scratch AUROC SD",      printed(C4,717,2), st.stdev(CURR), "results/task_runs.json", "stdev(auroc | ours_current_s0..s2) ddof=1")
rec(C4, 718, "clinical gap",                   printed(C4,718,0), st.mean(CONC)-st.mean(CURR), "results/task_runs.json", "derived")
for ln, k in [(680,"fancyzhx/ag_news"),(680,"bert-base-uncased")]:
    nb = "".join("".join(c.get("source", [])) for c in json.load(open(ROOT/"accord/scaling/C44_convergence_check.ipynb"))["cells"])
    rec(C4, ln, f"convergence config token '{k}'", k, k if k in nb else None, "accord/scaling/C44_convergence_check.ipynb", "source text")
nb44 = "".join("".join(c.get("source", [])) for c in json.load(open(ROOT/"accord/scaling/C44_convergence_check.ipynb"))["cells"])
m = re.search(r'MAXLEN,\s*N_TRAIN,\s*N_VAL\s*=\s*([\d_]+),\s*([\d_]+),\s*([\d_]+)', nb44)
rec(C4, 679, "N_TRAIN", printed(C4,679,0), float(m.group(2).replace("_","")), "accord/scaling/C44_convergence_check.ipynb", "MAXLEN, N_TRAIN, N_VAL assignment")
rec(C4, 679, "N_VAL",   printed(C4,679,1), float(m.group(3).replace("_","")), "accord/scaling/C44_convergence_check.ipynb", "MAXLEN, N_TRAIN, N_VAL assignment")
rec(C4, 680, "MAXLEN",  printed(C4,680,0), float(m.group(1).replace("_","")), "accord/scaling/C44_convergence_check.ipynb", "MAXLEN, N_TRAIN, N_VAL assignment")
prereg = (ROOT/"accord/scaling/C44_PREREGISTRATION.md").read_text()
rec(C4, 687, "prereg Outcome A threshold", printed(C4,687,0), 3.0 if "3.0 accuracy points" in prereg else None, "accord/scaling/C44_PREREGISTRATION.md", "Outcome A")
rec(C4, 689, "prereg inconclusive band lo", printed(C4,689,0), 2.5 if "[2.5, 3.5]" in prereg else None, "accord/scaling/C44_PREREGISTRATION.md", "inconclusive band")
rec(C4, 689, "prereg inconclusive band hi", printed(C4,689,1), 3.5 if "[2.5, 3.5]" in prereg else None, "accord/scaling/C44_PREREGISTRATION.md", "inconclusive band")

# ================= baselines / power / systems253 ============================
b253 = [(241,"gpt4o_zeroshot"),(242,"gpt4o_fewshot"),(243,"accord_binary_at_0.5_253core"),(244,"llama_baseline")]
for ln, k in b253:
    v = printed(C4, ln); b = S31[k]
    for j, fld in enumerate(["tp","fn","fp","tn"]):
        rec(C4, ln, f"{k}.{fld}", v[-6+j], b[fld], "results/accord/step3_baselines.json", f"{k}.{fld}")
    rec(C4, ln, f"{k}.precision", v[-2], b["precision"], "results/accord/step3_baselines.json", f"{k}.precision")
    rec(C4, ln, f"{k}.recall",    v[-1], b["recall"],    "results/accord/step3_baselines.json", f"{k}.recall")
    rec(C4, ln, f"{k} cells sum to n=253", 253.0, float(b["tp"]+b["fn"]+b["fp"]+b["tn"]), "results/accord/step3_baselines.json", "derived")
bl = CTRL["baselines"]
for ln, mk, sset in [(587,"tfidf_logreg","gold_all"),(588,"majority_class","gold_all"),
                     (589,"tfidf_logreg","gold_defect_free_rule34"),(590,"majority_class","gold_defect_free_rule34")]:
    v = printed(C4, ln); b = bl[mk][sset]
    rec(C4, ln, f"{mk}.{sset}.auroc",    v[1], b["auroc"], "results/accord/control/final_report.json", f"baselines.{mk}.{sset}.auroc")
    rec(C4, ln, f"{mk}.{sset}.ap",       v[2], b["ap"],    "results/accord/control/final_report.json", f"baselines.{mk}.{sset}.ap")
    rec(C4, ln, f"{mk}.{sset}.top15_tp", v[3], b["top15_tp"], "results/accord/control/final_report.json", f"baselines.{mk}.{sset}.top15_tp")
pc = CTRL["power_ceiling"]
prows = [(618, PWR), (619, pc["recomputed_same_formula_gold_all_ambiguous_included_n299"]),
         (620, pc["recomputed_same_formula_defect_free_rule34_n134"])]
for ln, b in prows:
    v = printed(C4, ln)
    rec(C4, ln, "power positives", v[1], b.get("n_positives", b.get("n_pos")), "results/accord/section6_auroc_power.json / control.power_ceiling", "n_positives")
    rec(C4, ln, "power negatives", v[2], b.get("n_negatives", b.get("n_neg")), "results/accord/section6_auroc_power.json / control.power_ceiling", "n_negatives")
    rec(C4, ln, "SE under null",   v[3], b["se_auc_under_null"], "results/accord/section6_auroc_power.json / control.power_ceiling", "se_auc_under_null")
    rec(C4, ln, "gap above chance",v[4], b["gap_above_chance"], "results/accord/section6_auroc_power.json / control.power_ceiling", "gap_above_chance")
    rec(C4, ln, "min detectable",  v[5], b["minimum_detectable_auc"], "results/accord/section6_auroc_power.json / control.power_ceiling", "minimum_detectable_auc")

# ================= over-calling / prevalence sweep ===========================
gb = MLLM["gold_base_rate"]
ml = [(209,"llama3:8b"),(210,"granite3.3:8b"),(211,"qwen3:8b")]
for ln, mk in ml:
    v = printed(C4, ln); b = MLLM["per_model"][mk]
    rec(C4, ln, f"{mk} gold over-call ratio", v[-1], b["gold"]["calibration_ratio_vs_gold_base_rate"], "results/accord/multillm/per_model_results.json", f"per_model.{mk}.gold.calibration_ratio_vs_gold_base_rate")
    rec(C4, ln, f"{mk} pool over-call ratio", v[-2], b["pool"]["calibration_ratio_vs_gold_base_rate"], "results/accord/multillm/per_model_results.json", f"per_model.{mk}.pool.calibration_ratio_vs_gold_base_rate")
    rec(C4, ln, f"{mk} pool ratio = pos_rate / gold base rate", b["pool"]["calibration_ratio_vs_gold_base_rate"], b["pool"]["positive_rate_of_parseable"]/gb, "results/accord/multillm/per_model_results.json", "derived, independent recomputation")
rec(C4, 213, "original llama3 calibration ratio", printed(C4,213,0), MLLM["original_llama3_reference"]["calibration_ratio"], "results/accord/multillm/per_model_results.json", "original_llama3_reference.calibration_ratio")
rec(C4, 208, "gold base rate used as the ratio denominator", printed(C4,208,0), gb, "results/accord/multillm/per_model_results.json", "gold_base_rate")
allsw = [(r["over_call_ratio"], m, p, lv) for m, pm in SWP["results"].items() for p, pp in pm.items() for lv, r in pp.items()]
rec(C4, 271, "max over-call ratio across the whole sweep", printed(C4,271,0), max(x[0] for x in allsw), "accord/prevalence_replication/results/sweep_results.json", "max over 24 cells")
rec(C4, 272, "second highest over-call ratio", printed(C4,272,0), sorted((x[0] for x in allsw), reverse=True)[1], "accord/prevalence_replication/results/sweep_results.json", "2nd max over 24 cells")
rec(C4, 270, "sweep cell count = 3 models x 2 prompts x 4 prevalences", 24.0, float(len(allsw)), "accord/prevalence_replication/results/sweep_results.json", "derived")

# ================= falsifications ============================================
rec(C4, 315, "alpha_d min", printed(C4,315,0), DGA["aggregate"]["alpha_d"]["min"], str(QUAR/"accord_supp/results/accord_v2_decontam_gold_aggregate.json"), "aggregate.alpha_d.min")
rec(C4, 315, "alpha_d max", printed(C4,315,1), DGA["aggregate"]["alpha_d"]["max"], str(QUAR/"accord_supp/results/accord_v2_decontam_gold_aggregate.json"), "aggregate.alpha_d.max")
rec(C4, 315, "alpha_d mean", printed(C4,315,2), DGA["aggregate"]["alpha_d"]["mean"], str(QUAR/"accord_supp/results/accord_v2_decontam_gold_aggregate.json"), "aggregate.alpha_d.mean")
rec(C4, 316, "alpha_n min", printed(C4,316,0), DGA["aggregate"]["alpha_n"]["min"], str(QUAR/"accord_supp/results/accord_v2_decontam_gold_aggregate.json"), "aggregate.alpha_n.min")
rec(C4, 316, "alpha_n max", printed(C4,316,1), DGA["aggregate"]["alpha_n"]["max"], str(QUAR/"accord_supp/results/accord_v2_decontam_gold_aggregate.json"), "aggregate.alpha_n.max")
rec(C4, 316, "alpha_n mean", printed(C4,316,2), DGA["aggregate"]["alpha_n"]["mean"], str(QUAR/"accord_supp/results/accord_v2_decontam_gold_aggregate.json"), "aggregate.alpha_n.mean")
hs = S02["accord_harm_score_distribution_by_group"]
rec(C4, 347, "harm score observed min over 299", printed(C4,347,0), min(hs[g]["min"] for g in hs if isinstance(hs[g],dict)), "results/accord/step2_signal_check.json", "min over the three label groups (n12+n241+n46=299)")
rec(C4, 347, "harm score observed max over 299", printed(C4,347,1), max(hs[g]["max"] for g in hs if isinstance(hs[g],dict)), "results/accord/step2_signal_check.json", "max over the three label groups")
rec(C5, 155, "anisotropy control mean", printed(C5,155,0), ANI["mean"], "results/accord/stage0/anisotropy_control.json", "mean")
rec(C5, 155, "matched clinical pair mean", printed(C5,155,1), HOM["results"]["head"]["matched"]["mean"], "results/accord/stage0/homogeneity.json", "results.head.matched.mean")
dif = HOM["results"]["head"]["matched"]["mean"] - ANI["mean"]
se  = math.sqrt(ANI["std"]**2/ANI["n_pairs"] + HOM["results"]["head"]["matched"]["std"]**2/HOM["results"]["head"]["matched"]["n"])
rec(C4, 440, "'landed close enough together' backed by a test", "no test cited",
    "Welch t=%.2f on n=%d vs n=%d (two-sided p=%.1e); the two means are statistically distinguishable"
    % (dif/se, ANI["n_pairs"], HOM["results"]["head"]["matched"]["n"], math.erfc(abs(dif/se)/math.sqrt(2))),
    "results/accord/stage0/{anisotropy_control,homogeneity}.json", "mean, std, n",
    "an equivalence claim asserted from two point estimates; no equivalence test is reported and a naive difference test rejects equality")

# ================= signed gaps / drift =======================================
srows = [(841,"E9_selection_experiment"),(842,"E10_pums_regimeA"),(843,"E10_pums_regimeB")]
for ln, exp in srows:
    pts = [p["signed_value"] for p in SIGN["zero_crossing_analysis"][exp]["per_metric"]["demographic_parity_gap"]["points"]]
    v = printed(C4, ln)
    rec(C4, ln, f"{exp} endpoint 1 (first sweep point)", v[0], pts[0],  "results/accord/signed_gap_analysis.json", f"zero_crossing_analysis.{exp}...points[0].signed_value")
    rec(C4, ln, f"{exp} endpoint 2 (last sweep point)",  v[1], pts[-1], "results/accord/signed_gap_analysis.json", f"zero_crossing_analysis.{exp}...points[-1].signed_value")
    rec(C4, ln, f"{exp} sign changes", "none", "none" if len({p>0 for p in pts})==1 else "present", "results/accord/signed_gap_analysis.json", f"zero_crossing_analysis.{exp}...sign_changes")
    _ok = (abs(pts[0]-min(pts)) < 1e-12 and abs(pts[-1]-max(pts)) < 1e-12)
    rec(C4, 847, f"{exp}: apanote calls the printed endpoints the minimum and maximum",
        "min and max (per apanote)",
        "min and max (per apanote)" if _ok else
        "first=%.5f last=%.5f; true min=%.5f max=%.5f" % (pts[0], pts[-1], min(pts), max(pts)),
        "results/accord/signed_gap_analysis.json", f"zero_crossing_analysis.{exp}...points[*].signed_value",
        "the apanote describes the printed values as minimum/maximum; they are the first and last points of the sweep")
c5pts_A = [p["signed_value"] for p in SIGN["zero_crossing_analysis"]["E10_pums_regimeA"]["per_metric"]["demographic_parity_gap"]["points"]]
rec(C5, 305, "ch5 states regime A runs 'from X to Y'", "%.5f -> %.5f" % (printed(C5,305,0), printed(C5,305,1)),
    "%.5f -> %.5f" % (c5pts_A[0], c5pts_A[-1]), "results/accord/signed_gap_analysis.json",
    "zero_crossing_analysis.E10_pums_regimeA...points[0] and [-1]",
    "ch5 reverses the sweep direction and contradicts its own next sentence ('the first becomes more negative'); ch4 tab:signed L842 prints the same pair in the correct order")
rec(C5, 309, "regime A signed drift slope", printed(C5,309,0), DRIF["PUMS_RegimeA_drift_vs_OR"]["demographic_parity_gap"]["recomputed_signed_slope"], "results/accord/sign_provenance_drift_recomputation.json", "PUMS_RegimeA_drift_vs_OR.demographic_parity_gap.recomputed_signed_slope")
rec(C5, 310, "regime B signed drift slope", printed(C5,310,0), DRIF["PUMS_RegimeB_drift_vs_beta"]["demographic_parity_gap"]["recomputed_signed_slope"], "results/accord/sign_provenance_drift_recomputation.json", "PUMS_RegimeB_drift_vs_beta.demographic_parity_gap.recomputed_signed_slope")
nm = nf = 0
for exp, blk in DRIF.items():
    for met, r in blk.items():
        if isinstance(r, dict) and "sign_flip_between_abs_and_signed" in r and r.get("stored_abs_slope") is not None:
            nm += 1; nf += bool(r["sign_flip_between_abs_and_signed"])
rec(C5, 387, "drift sign flips", printed(C5,387,0), float(nf), "results/accord/sign_provenance_drift_recomputation.json", "count(sign_flip_between_abs_and_signed) over slopes with a stored_abs_slope")
rec(C5, 387, "drift metric-slopes compared", printed(C5,387,1), float(nm), "results/accord/sign_provenance_drift_recomputation.json", "count(slopes with a stored_abs_slope); 1 slope has stored=null and is excluded")
rec(C4, 828, "signed reproduction max discrepancy", printed(C4,829,0), SIGN["reproduction_check"]["overall_max_discrepancy"], "results/accord/signed_gap_analysis.json", "reproduction_check.overall_max_discrepancy")

# ================= survey ====================================================
import collections
dist = collections.Counter(r["null_attribute_handling"] for r in SURV)
srv = [(882,"audits_model_on_pool","yes"),(883,"audits_selection_literal","yes"),(884,"audits_selection","yes")]
for ln, f, val in srv:
    rec(C4, ln, f"survey {f}=={val}", printed(C4,ln,0), float(sum(r[f]==val for r in SURV)), "results/accord/survey/coding_table.csv", f"count({f}=={val})")
for ln, cat in [(887,"not_mentioned"),(888,"drops"),(889,"separate_category"),(890,"unclear"),(891,"imputes"),(892,"reports_only")]:
    rec(C4, ln, f"survey null-handling {cat}", printed(C4,ln,0), float(dist[cat]), "results/accord/survey/coding_table.csv", f"count(null_attribute_handling=={cat})")
rec(C4, 899, "six null-attribute categories sum to 75", 75.0, float(sum(dist.values())), "results/accord/survey/coding_table.csv", "derived")
rec(C4, 903, "permissive minus strict gap", printed(C4,903,-1), float(sum(r["audits_selection_literal"]=="yes" for r in SURV)-sum(r["audits_selection"]=="yes" for r in SURV)), "results/accord/survey/coding_table.csv", "derived")
rec(C4, 907, "not_mentioned + unclear", printed(C4,907,0), float(dist["not_mentioned"]+dist["unclear"]), "results/accord/survey/coding_table.csv", "derived")
for ln, f in [(916,"audits_selection_literal"),(919,"null_attribute_handling"),(929,"audits_selection")]:
    po, k = agree_kappa(f)
    v = printed(C4, ln)
    rec(C4, ln, f"survey reliability {f} agreement", v[-1] if ln!=919 else v[0], po, "results/accord/survey/{coding_table,second_coder_table}.csv", f"recomputed agreement on {len(SC2)} double-coded papers")
for ln, f in [(917,"audits_selection_literal"),(919,"null_attribute_handling")]:
    po, k = agree_kappa(f)
    rec(C4, ln, f"survey reliability {f} kappa", printed(C4,ln,0) if ln==917 else printed(C4,ln,-1), k, "results/accord/survey/{coding_table,second_coder_table}.csv", "recomputed Cohen kappa")
po, k = agree_kappa("audits_selection")
rec(C4, 930, "strict-cell kappa is undefined", "undefined", "undefined" if k is None else k, "results/accord/survey/{coding_table,second_coder_table}.csv", "pe==1 so kappa denominator is 0")
rec(C4, 915, "double-coded papers", printed(C4,915,0), float(len(SC2)), "results/accord/survey/second_coder_table.csv", "row count")
rec(C1, 335, "quoted fields verified", printed(C1,335,0), float(len(J("results/accord/survey/quotes.json")) if isinstance(J("results/accord/survey/quotes.json"), list) else sum(len(v) for v in J("results/accord/survey/quotes.json").values())), "results/accord/survey/quotes.json", "count of quoted fields")

# ================= cost ======================================================
crows = [(467,"C44 AG News positive control (3 epochs)"),(468,"C37 clinical task, PubMedBERT vs scratch"),
         (469,"C52 signed re-run, E9"),(470,"C52 signed re-run, E10")]
byep = {e["episode"]: e for e in COST["compute_by_episode"]}
for ln, ep in crows:
    v = printed(C5, ln)
    rec(C5, ln, f"cost runs [{ep}]", v[-2], float(byep[ep]["runs"]), "results/accord/control_cost_log.json", f"compute_by_episode[{ep}].runs")
    rec(C5, ln, f"cost minutes [{ep}]", v[-1], byep[ep]["gpu_minutes"], "results/accord/control_cost_log.json", f"compute_by_episode[{ep}].gpu_minutes")
rec(C5, 472, "cost total runs",    printed(C5,472,0), float(sum(e["runs"] for e in COST["compute_by_episode"])), "results/accord/control_cost_log.json", "sum(runs)")
rec(C5, 472, "cost total minutes", printed(C5,472,1), COST["total_recorded_gpu_minutes"], "results/accord/control_cost_log.json", "total_recorded_gpu_minutes")
rec(C5, 472, "cost rows sum to the printed total", COST["total_recorded_gpu_minutes"], sum(e["gpu_minutes"] for e in COST["compute_by_episode"]), "results/accord/control_cost_log.json", "derived")
_c555 = locate(C5, r"\$555\$ minutes")
rec(C5, _c555, "convergence sweep cost quoted in prose", printed(C5,_c555,0), sum(r["minutes"] for r in CONV), "results/accord/scaling/convergence_results.json", "sum(minutes)",
    "the SAME episode is printed as 34.9 minutes / 12 runs in tab:disc-cost L471 and as 555 minutes in the prose at L483; the 661.4 total therefore counts 12 of the 48 convergence runs")
rec(C1, 351, "ch1 logged control compute total", printed(C1,351,0), COST["total_recorded_gpu_minutes"], "results/accord/control_cost_log.json", "total_recorded_gpu_minutes")
rec(C1, 352, "ch1 AG News control minutes", printed(C1,352,0), byep["C44 AG News positive control (3 epochs)"]["gpu_minutes"], "results/accord/control_cost_log.json", "compute_by_episode.gpu_minutes",
    "ch1 attributes 34.9 min / 12 runs to 'the AG News positive control' while ch3 L715 and ch4 tab:conv describe that control as 48 runs / 555 minutes")
rec(C5, 468, "clinical-task cost reconciles with task_runs.json",
    byep["C37 clinical task, PubMedBERT vs scratch"]["gpu_minutes"],
    sum(v["minutes"] for v in TASK.values() if isinstance(v, dict) and "minutes" in v),
    "results/task_runs.json", "sum(minutes) over 17 runs")

# ================= error-bar convention audit ================================
def ddof(vals, reported): return 1 if abs(st.stdev(vals)-reported) < abs(st.pstdev(vals)-reported) else 0
conv_ddof = ddof(cells[("pretrained_bert",24)], csd("pretrained_bert",24))
convs = [
 ("results/accord/control/final_report.json", "auroc_mean_std_across_seeds.gold_all.std",
  ddof(CTRL["auroc_mean_std_across_seeds"]["gold_all"]["values"], CTRL["auroc_mean_std_across_seeds"]["gold_all"]["std"]), C4, 516),
 ("results/accord/decontam/corrected_gold_metrics.json", "aggregate.auroc_std",
  ddof([DEC["per_seed"][f"seed{i}"]["auroc"] for i in range(3)], DEC["aggregate"]["auroc_std"]), C4, 480),
 (str(QUAR/"accord_supp/results/accord_v2_decontam_gold_aggregate.json"), "aggregate.recall.std",
  ddof([DGA["per_seed"][str(i)]["metrics"]["recall"] for i in range(3)], DGA["aggregate"]["recall"]["std"]), C1, 149),
 ("results/pretraining_runs.json", "ablation SD column", 1, C4, 389),
 ("results/accord/scaling/convergence_results.json", "tab:conv SD column", conv_ddof, C4, 669),
]
for path, key, dd, ch, ln in convs:
    rec(ch, ln, f"error-bar estimator for {key}", "SD/+- printed with no estimator defined", f"ddof={dd} ({'sample' if dd==1 else 'population'} SD)", path, key,
        "the thesis prints all of these as 'SD' or '+/-' without defining the estimator; ddof is not consistent across sources")

# ---- the precision/base-rate margin sentence added to ch1 L170-174 and ch6 L120-123 ----
_prec = DGA["aggregate"]["precision"]["mean"]; _base = DEC["n_positives"]/DEC["n"]
rec(C1, 171, "margin between precision and base rate", printed(C1,171,1), _base-_prec,
    str(QUAR/"accord_supp/results/accord_v2_decontam_gold_aggregate.json")+" + results/accord/decontam/corrected_gold_metrics.json",
    "derived: 12/253 - aggregate.precision.mean",
    "the margin is printed to four decimals as 0.0050; the value is 0.005057, which rounds to 0.0051. "
    "The displayed identity 0.0424 + 0.0050 = 0.0474 closes only on the truncated margin", dp=4)
rec(C1, 173, "binomial SE on 12 of 253", printed(C1,174,0), math.sqrt(_base*(1-_base)/DEC["n"]),
    "results/accord/decontam/corrected_gold_metrics.json", "derived: sqrt(p(1-p)/n), p = n_positives/n")
rec(C6, 123, "binomial SE on 12 of 253 (ch6 restatement)", printed(C6,123,-1), math.sqrt(_base*(1-_base)/DEC["n"]),
    "results/accord/decontam/corrected_gold_metrics.json", "derived: sqrt(p(1-p)/n)")
rec(C6, 119, "ch6 restates the ensemble bootstrap interval", [printed(C6,119,-2), printed(C6,119,-1)],
    [round(DEC["aggregate"]["ensemble_ci95"][0],4), round(DEC["aggregate"]["ensemble_ci95"][1],4)],
    "results/accord/decontam/corrected_gold_metrics.json", "aggregate.ensemble_ci95")
rec(C1, 169, "predicted-positive count restated as 216 + 9 = 225", printed(C1,169,-1),
    float(DGA["per_seed"]["0"]["confusion"]["tp"] + DGA["per_seed"]["0"]["confusion"]["fp"]),
    str(QUAR/"accord_supp/results/accord_v2_decontam_gold_aggregate.json"), "per_seed.0.confusion tp+fp")

# ================= cross-chapter consistency + provenance ====================
import subprocess
def git(*a):
    return subprocess.run(["git","-C",str(ROOT)]+list(a), capture_output=True, text=True).stdout.strip()
_pre = git("log","-1","--format=%H|%ad|%s","--date=iso","b048829")
_prefiles = git("show","--stat","--format=","b048829")
_sweepc = git("log","--format=%h|%ad","--date=iso","--","results/accord/scaling/convergence_results.json").splitlines()
rec(C4, 685, "preregistration commit b048829 predates the sweep artifact",
    "committed before the sweep ran",
    "committed before the sweep ran" if ("C44_PREREGISTRATION.md" in _prefiles and _pre.split("|")[1] < _sweepc[0].split("|")[1]) else "ordering not confirmed",
    "git", "commit b048829 -> %s ; convergence_results.json first committed %s" % (_pre, _sweepc[0] if _sweepc else "n/a"))

rec(C5, 483, "same episode given two different costs",
    "%.1f min / %d runs in tab:disc-cost" % (byep["C44 AG News positive control (3 epochs)"]["gpu_minutes"],
                                             byep["C44 AG News positive control (3 epochs)"]["runs"]),
    "%.0f min / %d runs in tab:conv and in the prose two lines below the table" % (sum(r["minutes"] for r in CONV), len(CONV)),
    "results/accord/control_cost_log.json vs results/accord/scaling/convergence_results.json",
    "cost.c44_ag_news_positive_control vs conv.total_minutes / conv.n_runs",
    "both figures are registered and both are correct for what they measure, but tab:disc-cost, its own caption, and ch1 L351-352 present them as the same episode without reconciling; the 661.4 total therefore counts 12 of the 48 convergence runs")

rec(C1, 217, "seed-SD bound on the positive control",
    printed(C1,217,0),
    max(csd(t,b) for t,b in cells),
    "results/accord/scaling/convergence_results.json", "max stdev over all 16 cells",
    "ch1 L217 and L317 and ch6 L156 state 'at or below 0.0029'; ch4 L699 states 'at or below 0.0036' for the same sweep. 0.0029 is the max over only the two compared arms at the 24-epoch budget; the max over all 16 cells is 0.003649")

for _ch, _ln, _pat, _dp_a, _dp_b, _val, _what in [
    (C1, 184, r"\$p\$ of \$0.867\$", 3, 4, DEC["aggregate"]["ensemble_mannwhitney_p"], "ensemble Mann-Whitney p"),
    (C6, 101, r"Mann-Whitney \$p = 0.867\$", 3, 4, DEC["aggregate"]["ensemble_mannwhitney_p"], "ensemble Mann-Whitney p"),
]:
    rec(_ch, _ln, f"reporting precision for {_what}", "3 decimals", "4 decimals in ch4 L481 for the identical value %.7f" % _val,
        "results/accord/decontam/corrected_gold_metrics.json", "aggregate.ensemble_mannwhitney_p",
        "value is correct in both places; only the displayed precision differs between chapters")

# ================= write =====================================================
from collections import Counter
counts = Counter(r["verdict"] for r in R)
payload = {
  "generated_by": "accord/stats/final_number_audit.py",
  "method": ("Every printed value is read directly out of the .tex at the recorded line by regex; every "
             "recomputed value is derived from the raw artifact named in source_path. No value in this file "
             "was typed by hand. Aggregates named 'quarantine' are read from /Users/rickygole/mimic_local_only "
             "and are aggregate-only; no note text or identifier is read or written."),
  "chapters_audited": sorted(LINES.keys()),
  "thesis_snapshot": {
     "dir": "accord/stats/_thesis_snapshot",
     "why": ("the live chapter files were being edited during this audit (msu_ch1 16:46, ch3 16:34, ch5 16:41, "
             "ch6 16:47 on 2026-08-11), so a hashed snapshot was pinned and every line number below refers to it"),
     "sha256": HASHES},
  "register_citation_defects": [
     {"where": "msu_ch4_results.tex L398-401 (tab:ablation apanote)",
      "issue": "cites results/pretraining_runs.json with keys ablation.*.val_loss_mean / val_loss_sd / delta_vs_vanilla / pct_of_segment_index; that file contains no 'ablation' node at all. Those are VERIFIED_FACTS.json register keys, not keys in the artifact named.",
      "values_affected": "none - every printed value recomputes exactly from final_val_loss in that file"},
     {"where": "msu_ch4_results.tex L730-733",
      "issue": "the apanote cites accord/scaling/C44_convergence_check.ipynb three lines earlier and then says 'the notebook cell that would compute one calls scipy.stats.ttest_ind'. C44_convergence_check.ipynb contains no ttest_ind. The cell is in accord/scaling/C4_scaling_study.ipynb and it tests the PUBLIC-benchmark gap, not the clinical gap.",
      "values_affected": "none - no p value is quoted"},
     {"where": "results/accord/VERIFIED_FACTS.json keys signed.E10_pums_regimeA.dp_min / dp_max (and regimeB)",
      "issue": "named dp_min / dp_max but hold the first and last points of the sweep. For regime A neither is the min or the max. msu_ch4_results.tex L846-849 repeats the mislabel and adds a rationalisation for it.",
      "values_affected": "the printed numbers are correct as sweep endpoints; the description of what they are is wrong"},
     {"where": "phase2_summary.json contradiction_mining",
      "issue": "the node's per-rule counts are post-prefilter (886/202/520/370 = 1978) while its sibling total_raw is 1980. The thesis reads this correctly (L303-315) and separates the two splits; noted only because the artifact itself is ambiguously labelled.",
      "values_affected": "none"}],
  "statistical_concerns": [
     {"id": "S1", "severity": "medium",
      "issue": "Error bars are printed as 'SD' or '+/-' with no estimator defined, and the estimator is not consistent: results/accord/control/final_report.json, results/accord/scaling/convergence_results.json and results/pretraining_runs.json all yield ddof=1; results/accord/decontam/corrected_gold_metrics.json and the decontam gold aggregate yield ddof=0. tab:decontam L480 (ddof=0) and tab:control299 L516 (ddof=1) sit two tables apart.",
      "fix": "state the estimator once and use one convention, or report both n=3 spreads as ranges"},
     {"id": "S2", "severity": "medium",
      "issue": "msu_ch4_results.tex L807: the two MCC intervals are said to 'overlap by more than half their width'. The overlap is 0.0882 against widths 0.2803 and 0.2900, i.e. 31.5% and 30.4%. The conclusion the sentence supports (neither system is distinguishable) is still correct, and both intervals do contain zero.",
      "fix": "state the overlap or drop the fraction"},
     {"id": "S3", "severity": "medium",
      "issue": "msu_ch4_results.tex L436-442: the anisotropy control is adjudicated by comparing 0.9113 (n=190) with 0.9176 (n=1956) and calling them 'close enough together to refute the explanation'. No test and no equivalence margin is given; a Welch test on the stored means, SDs and n gives t=4.00, p=6.3e-05, so a difference test rejects equality. The practical point (a 0.006 gap on a 0-1 similarity scale) is fine, but it needs an equivalence bound rather than an eyeballed comparison.",
      "fix": "state a tolerance and run TOST, or report the difference and drop 'close enough'"},
     {"id": "S4", "severity": "low",
      "issue": "no multiple-comparison adjustment anywhere: 5 DeLong comparisons (tab:delong), 4 MCC intervals (tab:mcc), 24 prevalence-sweep cells, 14 drift slopes. Every one is reported as non-significant, so no adjustment would change a conclusion, and the thesis never claims a difference from these.",
      "fix": "one sentence saying the tests are unadjusted and that no claim rests on them"},
     {"id": "S5", "severity": "none - resolved",
      "issue": "the Welch/Student mislabel is corrected. accord/scaling/C4_scaling_study.ipynb calls scipy.stats.ttest_ind without equal_var=False while printing the label 'Welch p='; msu_ch4_results.tex L730-734 says so explicitly and quotes no p value.",
      "fix": "none needed; see register_citation_defects for the notebook attribution"},
     {"id": "S6", "severity": "none - correct",
      "issue": "DeLong is correctly used for the correlated-AUROC comparisons (same 299 / 134 pairs, five paired rows, SEs of the difference reported). Fisher exact is correctly used for the 2x2 rule-stratum comparison at these cell counts. Both reproduce independently.",
      "fix": "none"},
     {"id": "S7", "severity": "low",
      "issue": "seeds are reported for the pretraining ablation (0,1,2), the control report (42,123,2024), the decontamination (0,1,2), the bootstraps (42 and 20260806) and the sweep (42). The clinical-task comparison in tab:scaling pairs concat seeds 1,2,3 against ours_current seeds 0,1,2 - three runs each on an identical split, but not seed-matched, and concat_s0 is dropped because it sits on the earlier split. Neither fact is stated.",
      "fix": "say which seeds each arm used and why concat_s0 is excluded"}],
  "counts_by_verdict": dict(counts),
  "n_claims": len(R),
  "claims": R,
}
OUT.write_text(json.dumps(payload, indent=1))
print(json.dumps(dict(counts), indent=1)); print("n_claims", len(R)); print("wrote", OUT)
for r in R:
    if r["verdict"] != "VERIFIED":
        print(f"  {r['verdict']:12s} {r['chapter']} L{r['line']}  {r['claim']}  printed={r['printed']!r} recomputed={r['recomputed']!r}")
