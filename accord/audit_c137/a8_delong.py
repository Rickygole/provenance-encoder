"""Audit pass 8: DeLong 1988 paired ROC test, implemented from the structural components,
recomputed from the raw per-pair control scores and the released ACCORD probability."""
import json,math,pathlib
import numpy as np, pandas as pd
from scipy.stats import norm
ROOT=pathlib.Path(__file__).resolve().parents[2]; OUT=ROOT/"results/accord/audit_c137"
def midrank(x):
    x=np.asarray(x,float); order=np.argsort(x,kind="mergesort"); s=x[order]; n=len(x); r=np.empty(n); i=0
    while i<n:
        j=i
        while j+1<n and s[j+1]==s[i]: j+=1
        r[i:j+1]=0.5*(i+j)+1.0; i=j+1
    out=np.empty(n); out[order]=r; return out
def delong(y,sa,sb):
    y=np.asarray(y); k=2; S=[np.asarray(sa,float),np.asarray(sb,float)]
    m=int((y==1).sum()); n=int((y==0).sum())
    auc=np.empty(k); V10=np.empty((k,m)); V01=np.empty((k,n))
    for r in range(k):
        p=S[r][y==1]; q=S[r][y==0]
        tx=midrank(p); ty=midrank(q); tz=midrank(np.r_[p,q])
        auc[r]=(tz[:m].sum()-m*(m+1)/2.0)/(m*n)
        V10[r]=(tz[:m]-tx)/n
        V01[r]=1.0-(tz[m:]-ty)/m
    S10=np.cov(V10,ddof=1); S01=np.cov(V01,ddof=1)
    C=S10/m+S01/n
    diff=auc[0]-auc[1]; var=C[0,0]+C[1,1]-2*C[0,1]; se=math.sqrt(var)
    z=diff/se; p=2*norm.sf(abs(z))
    return dict(auc_a=float(auc[0]),auc_b=float(auc[1]),diff=float(diff),se=float(se),z=float(z),p=float(p))
gold=pd.read_csv(ROOT/"local_only/derived/gold_standard_accord_results.csv").set_index("pair_id")
R={}
scores={}
for s in [42,123,2024]:
    c=pd.read_csv(ROOT/f"local_only/accord_control/scores_seed{s}.csv").set_index("pair_id")
    scores[s]=c
    assert (c.pharm_pos.reindex(gold.index).values==(gold.pharm_label=="contradiction").astype(int).values).all()
idx=gold.index
y=(gold.pharm_label=="contradiction").astype(int).values
accord=gold.accord_contra_prob.reindex(idx).values
ens=np.mean([scores[s].score.reindex(idx).values for s in (42,123,2024)],axis=0)
R["G299_ensemble_vs_accord"]=delong(y,ens,accord)
for s in (42,123,2024):
    R[f"G299_seed{s}_vs_accord"]=delong(y,scores[s].score.reindex(idx).values,accord)
m34=gold.rule.reindex(idx).isin([3,4]).values
R["G134_ensemble_vs_accord"]=delong(y[m34],ens[m34],accord[m34])
# per-seed AUROC mean/SD conventions
from sklearn.metrics import roc_auc_score, average_precision_score
for tag,mask in [("G299",np.ones(len(y),bool)),("G134",m34)]:
    a=[roc_auc_score(y[mask],scores[s].score.reindex(idx).values[mask]) for s in (42,123,2024)]
    ap=[average_precision_score(y[mask],scores[s].score.reindex(idx).values[mask]) for s in (42,123,2024)]
    R[f"{tag}_per_seed"]=dict(auroc=a,auroc_mean=float(np.mean(a)),auroc_sd_pop=float(np.std(a,ddof=0)),
        auroc_sd_samp=float(np.std(a,ddof=1)),ap=ap,ap_mean=float(np.mean(ap)),
        ap_sd_pop=float(np.std(ap,ddof=0)),ap_sd_samp=float(np.std(ap,ddof=1)),
        ens_auroc=float(roc_auc_score(y[mask],ens[mask])),ens_ap=float(average_precision_score(y[mask],ens[mask])))
    order=np.argsort(-ens[mask])[:15]; R[f"{tag}_per_seed"]["ens_top15_tp"]=int(y[mask][order].sum())
    R[f"{tag}_per_seed"]["seed_top15_tp"]=[int(y[mask][np.argsort(-scores[s].score.reindex(idx).values[mask])[:15]].sum()) for s in (42,123,2024)]
    R[f"{tag}_per_seed"]["accord_top15_tp"]=int(y[mask][np.argsort(-accord[mask])[:15]].sum())
json.dump(R,open(OUT/"a8_delong.json","w"),indent=1,sort_keys=True); print(json.dumps(R,indent=1,sort_keys=True))
