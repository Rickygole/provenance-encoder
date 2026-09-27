import json, pathlib, numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
ROOT=pathlib.Path("/Users/rickygole/provenance-encoder"); OUT=ROOT/"results/accord/audit_c137"
df=pd.read_csv(ROOT/"local_only/derived/gold_standard_accord_results.csv")
R={}
R["prob_quantiles"]={str(q): float(np.quantile(df.accord_contra_prob,q)) for q in [0,.05,.25,.5,.75,.95,1]}
R["n_prob_ge_05_299"]=int((df.accord_contra_prob>=0.5).sum())
R["n_accord_pred_contra_299"]=int((df.accord_pred=="contradiction").sum())
# cross tab
R["crosstab_pred_vs_prob05"]={f"{a}|prob_ge05={b}":int(c) for (a,b),c in
   pd.crosstab(df.accord_pred, df.accord_contra_prob>=0.5).stack().items()}
# what threshold on prob reproduces accord_pred's 278 positives?
s=np.sort(df.accord_contra_prob.values)[::-1]
R["implied_threshold_for_278"]=float(s[277])
R["implied_threshold_for_234_on_G253"]=float(np.sort(df[df.pharm_label!="ambiguous"].accord_contra_prob.values)[::-1][233])
R["prob_of_accord_pred_groups"]={g: dict(min=float(x.min()),max=float(x.max()),mean=float(x.mean()),n=int(len(x)))
  for g,x in df.groupby("accord_pred").accord_contra_prob}
R["harm_of_accord_pred_groups"]={g: dict(min=float(x.min()),max=float(x.max()),n=int(len(x)))
  for g,x in df.groupby("accord_pred").accord_harm_score}
R["n_harm_ge_05_299"]=int((df.accord_harm_score>=0.5).sum())
R["n_harm_ge_05_253"]=int((df[df.pharm_label!="ambiguous"].accord_harm_score>=0.5).sum())
# rule_pred column
R["accord_rule_pred_counts"]={str(k):int(v) for k,v in df.accord_rule_pred.value_counts().items()}
# MW p variants for contaminated run on G253
sub=df[df.pharm_label!="ambiguous"]; y=(sub.pharm_label=="contradiction").values
a=sub.accord_contra_prob.values[y]; b=sub.accord_contra_prob.values[~y]
for alt in ["two-sided","greater","less"]:
    R[f"mw_contra_G253_{alt}"]=float(mannwhitneyu(a,b,alternative=alt).pvalue)
    R[f"mw_contra_G253_{alt}_asymp"]=float(mannwhitneyu(a,b,alternative=alt,method="asymptotic").pvalue)
    R[f"mw_contra_G253_{alt}_exact"]=float(mannwhitneyu(a,b,alternative=alt,method="exact").pvalue)
json.dump(R,open(OUT/"a2_tau_probe.json","w"),indent=1,sort_keys=True); print(json.dumps(R,indent=1,sort_keys=True))
