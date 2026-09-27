"""Audit pass 1: gold strata, per-rule table, marginals, systems253, mcc, harm band.
Reads: local_only/derived/gold_standard_accord_results.csv (299 rows, aggregates only)
Writes: results/accord/audit_c137/a1_gold_and_rules.json
"""
import json, math, pathlib
import numpy as np, pandas as pd
from scipy.stats import hypergeom, fisher_exact, spearmanr, mannwhitneyu
from sklearn.metrics import roc_auc_score, average_precision_score, cohen_kappa_score, matthews_corrcoef

ROOT = pathlib.Path("/Users/rickygole/provenance-encoder")
OUT = ROOT/"results/accord/audit_c137"; OUT.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(ROOT/"local_only/derived/gold_standard_accord_results.csv")
R = {}
R["n_rows"] = int(len(df))
R["pharm_label_counts"] = {str(k): int(v) for k, v in df.pharm_label.value_counts().items()}
R["accord_pred_counts"] = {str(k): int(v) for k, v in df.accord_pred.value_counts().items()}
R["rule_counts"] = {str(k): int(v) for k, v in df.rule.value_counts().sort_index().items()}

y299 = (df.pharm_label == "contradiction").astype(int).values
amb = (df.pharm_label == "ambiguous").values
R["n_pos_299"] = int(y299.sum()); R["n_amb"] = int(amb.sum())
R["n_neg_299"] = int(299 - y299.sum())
R["n_253"] = int((~amb).sum()); R["n_pos_253"] = int(y299[~amb].sum())
R["n_neg_253"] = int((~amb).sum() - y299[~amb].sum())

# per-rule table
per_rule = {}
for r in sorted(df.rule.unique()):
    s = df[df.rule == r]
    prec = float((s.pharm_label == "contradiction").mean())
    app = float((s.accord_pred == "contradiction").mean())
    per_rule[int(r)] = dict(n=int(len(s)), pharm_precision=prec, accord_pred_pos_rate=app,
                            n_pos=int((s.pharm_label == "contradiction").sum()),
                            n_accord_pos=int((s.accord_pred == "contradiction").sum()))
R["per_rule"] = per_rule
R["per_rule_n_sum"] = int(sum(v["n"] for v in per_rule.values()))
R["per_rule_pos_sum"] = int(sum(v["n_pos"] for v in per_rule.values()))
# reconstruction printed in the note: 134*0.9701 + 31*0.9677 + 79*0.9747 + 55*0.7455
R["note_reconstruction_printed_rounded"] = 134*0.9701 + 31*0.9677 + 79*0.9747 + 55*0.7455
R["note_reconstruction_exact_counts"] = int(sum(v["n_accord_pos"] for v in per_rule.values()))

# rule group pairs
g12 = df[df.rule.isin([1, 2])]; g34 = df[df.rule.isin([3, 4])]
R["g12"] = dict(n=int(len(g12)), pos=int((g12.pharm_label == "contradiction").sum()))
R["g34"] = dict(n=int(len(g34)), pos=int((g34.pharm_label == "contradiction").sum()))
R["g12_rate"] = R["g12"]["pos"]/R["g12"]["n"]; R["g34_rate"] = R["g34"]["pos"]/R["g34"]["n"]
tbl = [[R["g12"]["pos"], R["g12"]["n"]-R["g12"]["pos"]], [R["g34"]["pos"], R["g34"]["n"]-R["g34"]["pos"]]]
orr, pv = fisher_exact(tbl)
R["rule_group_fisher"] = dict(table=tbl, odds_ratio=float(orr), p_two_sided=float(pv))
# G145 / G108
g145 = df[(df.rule.isin([1, 2])) & (~amb)]
g108 = df[(df.rule.isin([3, 4])) & (~amb)]
R["G145"] = dict(n=int(len(g145)), pos=int((g145.pharm_label == "contradiction").sum()))
R["G108"] = dict(n=int(len(g108)), pos=int((g108.pharm_label == "contradiction").sum()))
R["G134"] = dict(n=int(len(g34)), pos=R["g34"]["pos"])
R["G145_plus_G108_n"] = R["G145"]["n"]+R["G108"]["n"]
R["G145_plus_G108_pos"] = R["G145"]["pos"]+R["G108"]["pos"]

# base rates / majority-class AP
R["base_rate_299"] = 12/299; R["base_rate_253"] = 12/253
R["base_rate_134"] = R["G134"]["pos"]/R["G134"]["n"]; R["base_rate_108"] = R["G108"]["pos"]/R["G108"]["n"]

# ACCORD released run confusion on G253 at tau=0.5 via accord_pred
sub = df[~amb]
ys = (sub.pharm_label == "contradiction").astype(int).values
pred_pred = (sub.accord_pred == "contradiction").astype(int).values
def conf(y, p):
    tp = int(((y == 1) & (p == 1)).sum()); fn = int(((y == 1) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum()); tn = int(((y == 0) & (p == 0)).sum())
    return dict(tp=tp, fn=fn, fp=fp, tn=tn, n=tp+fn+fp+tn,
                precision=tp/(tp+fp) if tp+fp else None, recall=tp/(tp+fn) if tp+fn else None,
                mcc=float(matthews_corrcoef(y, p)))
R["accord_pred_G253"] = conf(ys, pred_pred)
prob = (sub.accord_contra_prob.values >= 0.5).astype(int)
R["accord_prob_ge05_G253"] = conf(ys, prob)
R["accord_prob_ge05_G299"] = conf(y299, (df.accord_contra_prob.values >= 0.5).astype(int))

# contaminated released run AUROC on the sets
R["contaminated_auroc_G253"] = float(roc_auc_score(ys, sub.accord_contra_prob.values))
R["contaminated_auroc_G299"] = float(roc_auc_score(y299, df.accord_contra_prob.values))
m34 = df.rule.isin([3, 4]).values
R["contaminated_auroc_G134"] = float(roc_auc_score(y299[m34], df.accord_contra_prob.values[m34]))
R["contaminated_ap_G299"] = float(average_precision_score(y299, df.accord_contra_prob.values))
R["contaminated_ap_G134"] = float(average_precision_score(y299[m34], df.accord_contra_prob.values[m34]))
R["contaminated_mw_p_G253_two_sided"] = float(mannwhitneyu(sub.accord_contra_prob.values[ys==1], sub.accord_contra_prob.values[ys==0], alternative="two-sided").pvalue)

# harm score band and spearman
R["harm_min"] = float(df.accord_harm_score.min()); R["harm_max"] = float(df.accord_harm_score.max())
rho, p = spearmanr(df.accord_harm_score.values, df.danger_rating.values)
R["harm_vs_danger_299"] = dict(rho=float(rho), p=float(p))
rho, p = spearmanr(df.accord_contra_prob.values, df.danger_rating.values)
R["prob_vs_danger_299"] = dict(rho=float(rho), p=float(p))
pos = df[df.pharm_label == "contradiction"]
rho, p = spearmanr(pos.accord_contra_prob.values, pos.danger_rating.values)
R["prob_vs_danger_pos12"] = dict(rho=float(rho), p=float(p), n=int(len(pos)))
rho, p = spearmanr(pos.accord_harm_score.values, pos.danger_rating.values)
R["harm_vs_danger_pos12"] = dict(rho=float(rho), p=float(p), n=int(len(pos)))

# MCC ceilings at the deployed marginals
def ceil_mcc(n, npos, nflag):
    tp = min(npos, nflag); fp = nflag-tp; fn = npos-tp; tn = n-tp-fp-fn
    den = math.sqrt(float(tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return (tp*tn-fp*fn)/den if den else 0.0
R["ceil_mcc_261_299"] = ceil_mcc(299, 12, 261)
R["ceil_mcc_223_253"] = ceil_mcc(253, 12, 223)
R["floor_p_261_299"] = float(hypergeom.sf(min(12, 261)-1, 299, 12, 261))
R["floor_p_223_253"] = float(hypergeom.sf(min(12, 223)-1, 253, 12, 223))
# deployed labeller confusion as printed
R["deployed_G299_printed"] = dict(tp=11, fp=250, fn=1, tn=37, sum=11+250+1+37)
R["deployed_G299_mcc_recomputed"] = ceil_mcc(0, 0, 0)  # placeholder
def mcc_from(tp, tn, fp, fn):
    den = math.sqrt(float(tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)); return (tp*tn-fp*fn)/den if den else 0.0
R["deployed_G299_mcc_recomputed"] = mcc_from(11, 37, 250, 1)
R["deployed_G299_hyper_p"] = float(hypergeom.sf(11-1, 299, 12, 261))
R["deployed_G253_mcc_recomputed"] = mcc_from(11, 253-11-212-1, 212, 1)
# G253: flags 223, tp 11 -> fp 212, fn 1, tn 253-11-212-1 = 29
R["deployed_G253_cells"] = dict(tp=11, fp=212, fn=1, tn=253-11-212-1)
R["deployed_G253_hyper_p"] = float(hypergeom.sf(11-1, 253, 12, 223))
R["deployed_G253_mcc"] = mcc_from(11, 29, 212, 1)
# over-calling ratios
R["overcall_ratio_check"] = {}
for name, rate, n in [("llama_pool", 0.9470, 1679), ("llama_gold", 0.9532, 299),
                      ("granite_pool", 0.7219, 1679), ("granite_gold", 0.7425, 299),
                      ("qwen_pool", 0.6349, 1679), ("qwen_gold", 0.6187, 299),
                      ("deployed_pool", 0.8487, 1679)]:
    R["overcall_ratio_check"][name] = dict(printed_rate=rate, ratio_at_unrounded_base=rate*299/12,
                                           ratio_at_rounded_base=rate/0.0401)
# nested pool arithmetic
R["pool_chain"] = dict(c1978_minus_299=1978-299, then_minus_291_205=1978-299-291-205,
                       pos_1183_at_08521=1183*0.8521)
R["pool_1183_x_08521"] = 1183*0.8521
R["deployed_pool_count_1425_over_1679"] = 1425/1679
json.dump(R, open(OUT/"a1_gold_and_rules.json", "w"), indent=1, sort_keys=True)
print(json.dumps(R, indent=1, sort_keys=True))
