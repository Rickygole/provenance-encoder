"""Audit pass 6: independent DeLong recompute and independent Hanley-McNeil MDE recompute."""
import json,math,pathlib
import numpy as np, pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, average_precision_score
ROOT=pathlib.Path(__file__).resolve().parents[2]; OUT=ROOT/"results/accord/audit_c137"
R={"mde":{},"delong":{}}
# ---- Hanley-McNeil MDE, written from the formula, not read from any file
def se0(np_,nn_):
    A=0.5;Q1=A/(2-A);Q2=2*A*A/(1+A)
    var=(A*(1-A)+(np_-1)*(Q1-A*A)+(nn_-1)*(Q2-A*A))/(np_*nn_)
    return math.sqrt(var)
def se_at(A,np_,nn_):
    Q1=A/(2-A);Q2=2*A*A/(1+A)
    return math.sqrt((A*(1-A)+(np_-1)*(Q1-A*A)+(nn_-1)*(Q2-A*A))/(np_*nn_))
def mde_alt(np_,nn_,alpha=0.05,power=0.8):
    za=norm.ppf(1-alpha/2); zb=norm.ppf(power); A=0.75
    for _ in range(500):
        A=0.5+za*se0(np_,nn_)+zb*se_at(A,np_,nn_)
    return A
def mde_null(np_,nn_,alpha=0.05,power=0.8):
    return 0.5+(norm.ppf(1-alpha/2)+norm.ppf(power))*se0(np_,nn_)
for name,(p,q) in {"G253":(12,241),"G299":(12,287),"G134":(6,128),"G108":(6,102),
                   "G145":(6,139),"pos7_neg246":(7,246),"pos6_neg247":(6,247),"pos5_neg248":(5,248),
                   "pos13_neg240":(13,240)}.items():
    R["mde"][name]=dict(n_pos=p,n_neg=q,se0=se0(p,q),se0_r4=round(se0(p,q),4),
        mde_alt=mde_alt(p,q),mde_alt_r4=round(mde_alt(p,q),4),
        mde_null=mde_null(p,q),mde_null_r4=round(mde_null(p,q),4),
        gap_alt_r4=round(mde_alt(p,q)-0.5,4))
# ---- DeLong, implemented from DeLong 1988 structural components
def midrank(x):
    x=np.asarray(x,float); order=np.argsort(x); s=x[order]; n=len(x); r=np.empty(n)
    i=0
    while i<n:
        j=i
        while j+1<n and s[j+1]==s[i]: j+=1
        r[i:j+1]=0.5*(i+j)+1; i=j+1
    out=np.empty(n); out[order]=r; return out
def delong_cov(y,scores):
    y=np.asarray(y); pos=[s[y==1] for s in scores]; neg=[s[y==0] for s in scores]
    m=len(pos[0]); n=len(neg[0]); k=len(scores)
    aucs=np.empty(k); V10=np.empty((k,m)); V01=np.empty((k,n))
    for r in range(k):
        tx=midrank(pos[r]); ty=midrank(neg[r]); tz=midrank(np.r_[pos[r],neg[r]])
        aucs[r]=(tz[:m].sum()-m*(m+1)/2)/(m*n)
        V10[r]=(tz[:m]-tx)/n
        V01[r]=1.0-(tz[m:]-ty)/m
    S10=np.cov(V10); S01=np.cov(V01)
    if k==1: S10=np.array([[float(S10)]]); S01=np.array([[float(S01)]])
    S=S10/m+S01/n
    return aucs,S
def delong_test(y,sa,sb):
    aucs,S=delong_cov(y,[sa,sb])
    diff=aucs[0]-aucs[1]
    var=S[0,0]+S[1,1]-2*S[0,1]
    se=math.sqrt(var); z=diff/se; p=2*norm.sf(abs(z))
    return float(aucs[0]),float(aucs[1]),float(diff),se,float(z),float(p)
gold=pd.read_csv(ROOT/"local_only/derived/gold_standard_accord_results.csv")
y299=(gold.pharm_label=="contradiction").astype(int).values
accord=gold.accord_contra_prob.values
ctl={}
for s in [42,123,2024]:
    c=pd.read_csv(ROOT/f"local_only/accord_control/scores_seed{s}.csv")
    ctl[s]=c
R["control_cols"]=list(ctl[42].columns); R["control_shape"]=list(ctl[42].shape)
json.dump(R,open(OUT/"a6_delong_mde.json","w"),indent=1,sort_keys=True)
print(json.dumps(R["mde"],indent=1,sort_keys=True)); print(R["control_cols"],R["control_shape"])
print(ctl[42].head(3).to_string())
