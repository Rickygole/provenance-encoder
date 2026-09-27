import json,pathlib,numpy as np,pandas as pd
from scipy.stats import mannwhitneyu, ranksums, norm
from sklearn.metrics import average_precision_score, roc_auc_score
ROOT=pathlib.Path(__file__).resolve().parents[2]; OUT=ROOT/"results/accord/audit_c137"
df=pd.read_csv(ROOT/"local_only/derived/gold_standard_accord_results.csv")
sub=df[df.pharm_label!="ambiguous"]; y=(sub.pharm_label=="contradiction").values
a=sub.accord_contra_prob.values[y]; b=sub.accord_contra_prob.values[~y]
R={}
R["mw_asymp_cc_true"]=float(mannwhitneyu(a,b,alternative="two-sided",use_continuity=True,method="asymptotic").pvalue)
R["mw_asymp_cc_false"]=float(mannwhitneyu(a,b,alternative="two-sided",use_continuity=False,method="asymptotic").pvalue)
R["mw_exact"]=float(mannwhitneyu(a,b,alternative="two-sided",method="exact").pvalue)
R["ranksums"]=float(ranksums(a,b).pvalue)
R["mw_auto"]=float(mannwhitneyu(a,b,alternative="two-sided").pvalue)
# 4dp rounding of each
R["rounded"]={k:round(v,4) for k,v in R.items() if isinstance(v,float)}
R["stored_in_register"]=0.6418
R["ap_contaminated_G253"]=float(average_precision_score(y.astype(int),sub.accord_contra_prob.values))
R["auroc_harm_G253"]=float(roc_auc_score(y.astype(int),sub.accord_harm_score.values))
R["auroc_harm_G299"]=float(roc_auc_score((df.pharm_label=="contradiction").astype(int),df.accord_harm_score.values))
R["ap_harm_G253"]=float(average_precision_score(y.astype(int),sub.accord_harm_score.values))
# ambiguous-as-positive
yp=(df.pharm_label!="not_contradiction").astype(int)
R["n_pos_amb_as_pos"]=int(yp.sum())
R["auroc_amb_as_pos"]=float(roc_auc_score(yp,df.accord_contra_prob.values))
R["ap_amb_as_pos"]=float(average_precision_score(yp,df.accord_contra_prob.values))
R["base_rate_58_299"]=58/299
json.dump(R,open(OUT/"a4_mw_variants.json","w"),indent=1,sort_keys=True);print(json.dumps(R,indent=1,sort_keys=True))
