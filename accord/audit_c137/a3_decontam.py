"""Audit pass 3: recompute the decontaminated detector headline from raw per-pair scores."""
import json, math, pathlib
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score, average_precision_score, cohen_kappa_score, matthews_corrcoef
ROOT=pathlib.Path(__file__).resolve().parents[2]; OUT=ROOT/"results/accord/audit_c137"
P=ROOT/"local_only/from_drive_2026-09-13/outputs/gold_eval_decontam_predictions.csv"
df=pd.read_csv(P); y=df.y_true.values
R={"n":int(len(df)),"n_pos":int(y.sum()),"n_neg":int((y==0).sum())}
probs={s:df[f"prob_seed{s}"].values for s in (0,1,2)}
preds={s:df[f"pred_seed{s}"].values for s in (0,1,2)}
ens=np.mean([probs[s] for s in (0,1,2)],axis=0)
for s in (0,1,2):
    p=probs[s]
    order=np.argsort(-p)[:15]
    R[f"seed{s}"]=dict(auroc=float(roc_auc_score(y,p)), ap=float(average_precision_score(y,p)),
        mw_two=float(mannwhitneyu(p[y==1],p[y==0],alternative="two-sided").pvalue),
        mw_two_exact=float(mannwhitneyu(p[y==1],p[y==0],alternative="two-sided",method="exact").pvalue),
        top15_tp=int(y[order].sum()))
    pr=preds[s]
    tp=int(((y==1)&(pr==1)).sum()); fp=int(((y==0)&(pr==1)).sum())
    fn=int(((y==1)&(pr==0)).sum()); tn=int(((y==0)&(pr==0)).sum())
    R[f"seed{s}"].update(tp=tp,fp=fp,fn=fn,tn=tn,sum=tp+fp+fn+tn,
        precision=tp/(tp+fp), recall=tp/(tp+fn), kappa=float(cohen_kappa_score(y,pr)),
        mcc=float(matthews_corrcoef(y,pr)))
    # pred vs prob>=0.5 consistency
    R[f"seed{s}"]["pred_equals_prob_ge_05"]=bool((pr==(p>=0.5).astype(int)).all())
    # ECE 10 equal-width bins
    bins=np.clip((p*10).astype(int),0,9); ece=0.0
    for b in range(10):
        m=bins==b
        if m.sum(): ece+=m.sum()/len(p)*abs(p[m].mean()-y[m].mean())
    R[f"seed{s}"]["ece_10bin_equalwidth"]=float(ece)
aur=[R[f"seed{s}"]["auroc"] for s in (0,1,2)]
R["per_seed_auroc_mean"]=float(np.mean(aur))
R["per_seed_auroc_sd_pop"]=float(np.std(aur,ddof=0)); R["per_seed_auroc_sd_samp"]=float(np.std(aur,ddof=1))
R["per_seed_auroc_min"]=min(aur); R["per_seed_auroc_max"]=max(aur)
for fld in ["precision","recall","kappa","ece_10bin_equalwidth"]:
    v=[R[f"seed{s}"][fld] for s in (0,1,2)]
    R[f"mean_{fld}"]=float(np.mean(v)); R[f"sd_pop_{fld}"]=float(np.std(v,ddof=0)); R[f"sd_samp_{fld}"]=float(np.std(v,ddof=1))
R["ensemble_auroc"]=float(roc_auc_score(y,ens)); R["ensemble_ap"]=float(average_precision_score(y,ens))
R["ensemble_mw_two"]=float(mannwhitneyu(ens[y==1],ens[y==0],alternative="two-sided").pvalue)
R["ensemble_mw_two_exact"]=float(mannwhitneyu(ens[y==1],ens[y==0],alternative="two-sided",method="exact").pvalue)
# bootstrap intervals, unstratified row bootstrap, B=10000 seed 42
def boot_ci(score,y,B=10000,seed=42,strat=False):
    rng=np.random.default_rng(seed); n=len(y); out=[]
    ip=np.where(y==1)[0]; ineg=np.where(y==0)[0]
    for _ in range(B):
        if strat:
            idx=np.concatenate([rng.choice(ip,len(ip),True),rng.choice(ineg,len(ineg),True)])
        else:
            idx=rng.integers(0,n,n)
            if len(np.unique(y[idx]))<2: continue
        out.append(roc_auc_score(y[idx],score[idx]))
    out=np.array(out); return float(np.percentile(out,2.5)),float(np.percentile(out,97.5)),len(out),float(out.mean()),float(out.std(ddof=1))
lo,hi,k,bm,bs=boot_ci(ens,y); R["ensemble_boot_row_seed42"]=dict(lo=lo,hi=hi,kept=k,mean=bm,sd=bs)
lo,hi,k,bm,bs=boot_ci(ens,y,strat=True); R["ensemble_boot_strat_seed42"]=dict(lo=lo,hi=hi,kept=k,mean=bm,sd=bs)
# binomial SE on precision mentioned in note: sqrt(p(1-p)/n) at p=0.0424,n=?
R["binom_se_prec_0424_n253"]=math.sqrt(0.0424*(1-0.0424)/253)
R["binom_se_prec_0474_n253"]=math.sqrt(0.0474*(1-0.0474)/253)
R["binom_se_prec_mean_over_predpos"]=math.sqrt(0.0424*(1-0.0424)/225)
# coverage note: negatives per seed positive
R["neg_called_pos_per_seed"]=[R[f"seed{s}"]["fp"] for s in (0,1,2)]
json.dump(R,open(OUT/"a3_decontam.json","w"),indent=1,sort_keys=True); print(json.dumps(R,indent=1,sort_keys=True))
