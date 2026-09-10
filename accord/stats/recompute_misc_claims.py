"""Recomputations for the remaining audited prose claims."""
import json, math, statistics as st
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]

# 1. MCC interval overlap claim (ch4 L807-808: "overlap by more than half their width")
iv = json.load(open(ROOT/"results/accord/stats/intervals.json"))["E2_mcc_intervals"]
A = iv["gpt4o_zeroshot"]["ci95"]; B = iv["llama_baseline"]["ci95"]
wA, wB = A[1]-A[0], B[1]-B[0]
ov = max(0.0, min(A[1], B[1]) - max(A[0], B[0]))
print("MCC overlap: A=%s (w=%.4f) B=%s (w=%.4f) overlap=%.4f" % (A, wA, B, wB, ov))
print("  overlap/wA=%.4f  overlap/wB=%.4f  overlap/mean_w=%.4f  overlap/union_w=%.4f"
      % (ov/wA, ov/wB, ov/((wA+wB)/2), ov/(max(A[1],B[1])-min(A[0],B[0]))))
print("  claim 'more than half their width' holds:", ov > 0.5*min(wA, wB))
print("  MCC values span: %.4f to %.4f" % (min(v["mcc"] for v in iv.values()), max(v["mcc"] for v in iv.values())))
print("  n MCC point estimates with |mcc| < 0.05:", sum(abs(v["mcc"]) < 0.05 for v in iv.values()))

# 2. anisotropy control vs matched clinical pairs: is 'close enough' backed by a test?
an = json.load(open(ROOT/"results/accord/stage0/anisotropy_control.json"))
ho = json.load(open(ROOT/"results/accord/stage0/homogeneity.json"))["results"]["head"]["matched"]
d = ho["mean"] - an["mean"]
se = math.sqrt(an["std"]**2/an["n_pairs"] + ho["std"]**2/ho["n"])
t = d/se
print("\nAnisotropy: control mean=%.6f (n=%d, sd=%.6f) vs matched mean=%.6f (n=%d, sd=%.6f)"
      % (an["mean"], an["n_pairs"], an["std"], ho["mean"], ho["n"], ho["std"]))
print("  difference=%.6f  Welch se=%.6f  t=%.3f  (two-sided p ~ %.2e)" % (d, se, t, 2*math.erfc(abs(t)/math.sqrt(2))/2*2))

# 3. clinical-task compute: does task_runs.json sum to the logged 612.3 minutes / 17 runs?
tr = json.load(open(ROOT/"results/task_runs.json"))
mins = [v["minutes"] for v in tr.values() if isinstance(v, dict) and "minutes" in v]
print("\ntask_runs.json: n_runs=%d  total_minutes=%.4f  (logged: 17 runs, 612.3)" % (len(mins), sum(mins)))

# 4. SD convention audit: ddof used by each artifact that the thesis prints a +/- from
def ddof_of(vals, reported):
    s1, s0 = st.stdev(vals), st.pstdev(vals)
    return ("ddof=1" if abs(s1-reported) < abs(s0-reported) else "ddof=0"), s1, s0
print("\n--- SD convention per source ---")
fr = json.load(open(ROOT/"results/accord/control/final_report.json"))
blk = fr["auroc_mean_std_across_seeds"]["gold_all"]
print("  control/final_report auroc gold_all:", ddof_of(blk["values"], blk["std"]), "reported", blk["std"])
dc = json.load(open(ROOT/"results/accord/decontam/corrected_gold_metrics.json"))
v = [dc["per_seed"][f"seed{i}"]["auroc"] for i in range(3)]
print("  decontam/corrected_gold_metrics auroc:", ddof_of(v, dc["aggregate"]["auroc_std"]), "reported", dc["aggregate"]["auroc_std"])
ag = json.load(open("/Users/rickygole/mimic_local_only/accord_supp/results/accord_v2_decontam_gold_aggregate.json"))
rec = [ag["per_seed"][str(i)]["metrics"]["recall"] for i in range(3)]
print("  decontam_gold_aggregate recall:", ddof_of(rec, ag["aggregate"]["recall"]["std"]), "reported", ag["aggregate"]["recall"]["std"])
pr = [ag["per_seed"][str(i)]["metrics"]["precision"] for i in range(3)]
print("  decontam_gold_aggregate precision:", ddof_of(pr, ag["aggregate"]["precision"]["std"]), "reported", ag["aggregate"]["precision"]["std"])
conv = json.load(open(ROOT/"results/accord/scaling/convergence_results.json"))
c = [r["accuracy"] for r in conv if r["tag"]=="pretrained_bert" and r["budget"]==24]
print("  convergence pretrained E24 (thesis prints 0.0029):", ddof_of(c, 0.0029))
ab = [json.load(open(ROOT/"results/pretraining_runs.json"))[k]["final_val_loss"]
      for k in ("A1_10M_st1000_s0_shuf","A1_10M_st1000_s1_shuf","A1_10M_st1000_s2_shuf")]
print("  ablation vanilla (thesis prints 0.0085):", ddof_of(ab, 0.0085))
