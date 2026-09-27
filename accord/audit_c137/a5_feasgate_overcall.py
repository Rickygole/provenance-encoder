"""Audit pass 5: every feasibility-gate cell and every over-calling ratio, recomputed."""
import json,math,pathlib,glob
from fractions import Fraction
from scipy.stats import hypergeom
ROOT=pathlib.Path("/Users/rickygole/provenance-encoder"); OUT=ROOT/"results/accord/audit_c137"
FACTS=json.load(open(ROOT/"results/accord/VERIFIED_FACTS.json"))
def val(k):
    v=FACTS[k]; return v.get("value") if isinstance(v,dict) else v
def mcc(tp,tn,fp,fn):
    den=math.sqrt(float(tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)); return (tp*tn-fp*fn)/den if den else 0.0
def kap(tp,tn,fp,fn):
    n=tp+tn+fp+fn; po=(tp+tn)/n; pe=((tp+fp)*(tp+fn)+(fn+tn)*(fp+tn))/(n*n); return (po-pe)/(1-pe)
def best(n,npos,nflag):
    tp=min(npos,nflag);fp=nflag-tp;fn=npos-tp;tn=n-tp-fp-fn;return tp,tn,fp,fn
def floorp(n,npos,nflag): return float(hypergeom.sf(min(npos,nflag)-1,n,npos,nflag))
R={"overcall":{},"feasgate_scored":{},"feasgate_marginal":{}}
# --- over-calling ratios from the register rates, exact identity check
base=Fraction(12,299)
for tag,n in [("llama3_8b",1679),("granite3_3_8b",1679),("qwen3_8b",1679)]:
    for setname,nn in [("pool",n),("gold",299)]:
        rate=val(f"overcall.{tag}.{setname}.positive_rate"); ratio=val(f"overcall.{tag}.{setname}.ratio")
        cnt=rate*nn
        R["overcall"][f"{tag}.{setname}"]=dict(rate=rate,rate_r4=round(rate,4),ratio=ratio,ratio_r4=round(ratio,4),
            n=nn,implied_count=cnt,count_is_integer=abs(cnt-round(cnt))<1e-6,count=int(round(cnt)),
            ratio_from_rate=rate*299/12, identity_holds=abs(ratio-rate*299/12)<1e-9,
            ratio_recomputed_from_count=round(cnt)/nn*299/12)
d=val("overcall.deployed_llama3.pool.positive_rate"); dr=val("overcall.original_llama3_calibration_ratio")
R["overcall"]["deployed.pool"]=dict(rate=d,rate_r4=round(d,4),ratio=dr,ratio_r4=round(dr,4),n=1679,
    implied_count=d*1679,count=int(round(d*1679)),identity_holds=abs(dr-d*299/12)<1e-9)
# --- feasgate scored rows from raw per-pair metrics artifacts
DIFF=ROOT/"results/accord/differential"
names={"granite33":"Granite 3.3 8B","llama3":"Llama 3 8B","qwen3_8b_nothink":"Qwen 3 8B","qwen3_14b_nothink":"Qwen 3 14B"}
for tag,disp in names.items():
    a=json.load(open(DIFF/f"frontier_ollama_metrics_{tag}.json"))
    for strat in ["zero_shot","few_shot"]:
        for basis in ["G_253","G_299"]:
            c=a["by_strategy"][strat][basis]
            n=c["n"];npos=c["n_positives"];nflag=c["n_flagged"]
            tp,tn,fp,fn=c["tp"],c["tn"],c["fp"],c["fn"]
            btp,btn,bfp,bfn=best(n,npos,nflag)
            R["feasgate_scored"][f"{disp}|{strat}|{basis}"]=dict(
                n=n,npos=npos,nflag=nflag,cells_sum=tp+tn+fp+fn,cells_sum_ok=(tp+tn+fp+fn==n),
                flag_rate=nflag/n, flag_rate_r4=round(nflag/n,4),
                tp_plus_fp=tp+fp, tp_plus_fp_equals_nflag=(tp+fp==nflag),
                tp_plus_fn_equals_npos=(tp+fn==npos),
                obs_mcc=mcc(tp,tn,fp,fn), obs_mcc_r4=round(mcc(tp,tn,fp,fn),4),
                stored_mcc=c["mcc"], mcc_matches_store=abs(mcc(tp,tn,fp,fn)-c["mcc"])<1e-9,
                ceil_mcc=mcc(btp,btn,bfp,bfn), ceil_mcc_r4=round(mcc(btp,btn,bfp,bfn),4),
                ceil_kappa=kap(btp,btn,bfp,bfn), ceil_kappa_r4=round(kap(btp,btn,bfp,bfn),4),
                floor_p=floorp(n,npos,nflag), floor_p_r4=round(floorp(n,npos,nflag),4),
                obs_hyper_p=float(hypergeom.sf(tp-1,n,npos,nflag)),
                obs_hyper_p_r4=round(float(hypergeom.sf(tp-1,n,npos,nflag)),4),
                stored_hyper_p=c["hypergeom_p_one_sided"],
                on_ceiling=abs(mcc(tp,tn,fp,fn)-mcc(btp,btn,bfp,bfn))<1e-9,
                rejectable=floorp(n,npos,nflag)<0.05,
                obs_rejects=float(hypergeom.sf(tp-1,n,npos,nflag))<0.05,
                over_calling_ratio=c.get("over_calling_ratio"))
# --- the 5 marginal-only rows
for rid,n,npos,nflag,obsmcc in [("deployed_g299",299,12,261,val("c120.g299.mcc")),
        ("llama_baseline_g253",253,12,242,val("mccint.E2_mcc_intervals.llama_baseline.mcc")),
        ("replication_llama3_g299",299,12,int(round(val("overcall.llama3_8b.gold.positive_rate")*299)),None),
        ("replication_granite_g299",299,12,int(round(val("overcall.granite3_3_8b.gold.positive_rate")*299)),None),
        ("replication_qwen_g299",299,12,int(round(val("overcall.qwen3_8b.gold.positive_rate")*299)),None)]:
    btp,btn,bfp,bfn=best(n,npos,nflag)
    R["feasgate_marginal"][rid]=dict(n=n,npos=npos,nflag=nflag,flag_rate=nflag/n,flag_rate_r4=round(nflag/n,4),
        obs_mcc=obsmcc, ceil_mcc_r4=round(mcc(btp,btn,bfp,bfn),4), ceil_kappa_r4=round(kap(btp,btn,bfp,bfn),4),
        floor_p=floorp(n,npos,nflag), floor_p_r4=round(floorp(n,npos,nflag),4),
        rejectable=floorp(n,npos,nflag)<0.05)
json.dump(R,open(OUT/"a5_feasgate_overcall.json","w"),indent=1,sort_keys=True)
print(json.dumps(R,indent=1,sort_keys=True))
