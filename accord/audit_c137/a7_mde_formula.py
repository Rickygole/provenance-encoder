"""Audit pass 7: the minimum detectable AUROC formula.

Three candidate thresholds at 80 percent power and two-sided alpha 0.05:
  ROUTE A (as implemented in accord/stats/recompute_mde.py):
      smallest a with (a - 0.5) / SE_HM(a) >= z_{1-alpha/2} + z_{power}
      i.e. the ALTERNATIVE standard error used for both terms.
  ROUTE B (the standard Hanley and McNeil / Obuchowski power form):
      smallest a with (a - 0.5) >= z_{1-alpha/2} * SE_HM(0.5) + z_{power} * SE_HM(a)
  ROUTE C (null standard error for both terms):
      a = 0.5 + (z_{1-alpha/2} + z_{power}) * SE_HM(0.5)
Route A is anti-conservative because SE_HM(a) < SE_HM(0.5) for a > 0.5.
"""
import json,math,pathlib
from scipy.stats import norm
ROOT=pathlib.Path(__file__).resolve().parents[2]; OUT=ROOT/"results/accord/audit_c137"
FACTS=json.load(open(ROOT/"results/accord/VERIFIED_FACTS.json"))
def V(k):
    v=FACTS.get(k); return None if v is None else (v.get("value") if isinstance(v,dict) else v)
ZA=norm.ppf(0.975); ZB=norm.ppf(0.80)
def se_hm(a,m,n):
    q1=a/(2-a); q2=2*a*a/(1+a)
    return math.sqrt((a*(1-a)+(m-1)*(q1-a*a)+(n-1)*(q2-a*a))/(m*n))
def routeA(m,n):
    lo,hi=0.5,0.9999
    for _ in range(200):
        a=(lo+hi)/2
        if (a-0.5)/max(se_hm(a,m,n),1e-12) < ZA+ZB: lo=a
        else: hi=a
    return (lo+hi)/2
def routeB(m,n):
    lo,hi=0.5,0.9999
    for _ in range(200):
        a=(lo+hi)/2
        if (a-0.5) < ZA*se_hm(0.5,m,n)+ZB*se_hm(a,m,n): lo=a
        else: hi=a
    return (lo+hi)/2
def routeC(m,n): return 0.5+(ZA+ZB)*se_hm(0.5,m,n)
STRATA={"G253":(12,241),"G299":(12,287),"G134":(6,128),"G108":(6,102),"G145":(6,139),
        "tip_k5_pos7":(7,246),"tip_k6_pos6":(6,247),"tip_k7_pos5":(5,248),"relabel_pos13":(13,240)}
REGKEY={"G253":"mde_recomputed.gold_ambiguous_excluded","G299":"mde_recomputed.gold_ambiguous_retained",
        "G134":"mde_recomputed.defect_free_subset","G108":"mde_recomputed.rules34_ambiguous_dropped",
        "G145":"mde_recomputed.rules12_ambiguous_dropped","tip_k5_pos7":"c132.task_c.pos7_neg246",
        "tip_k6_pos6":"c132.task_c.pos6_neg247","tip_k7_pos5":"c132.task_c.pos5_neg248",
        "relabel_pos13":"c133.floors_neg_to_pos.k1"}
R={"z_alpha_two_sided":ZA,"z_power":ZB,"strata":{}}
for k,(m,n) in STRATA.items():
    a,b,c=routeA(m,n),routeB(m,n),routeC(m,n)
    R["strata"][k]=dict(n_pos=m,n_neg=n,se_null=se_hm(0.5,m,n),
        routeA_as_implemented=round(a,4),routeB_standard=round(b,4),routeC_null_only=round(c,4),
        registered=V(REGKEY[k]+".minimum_detectable_auc"),
        registered_null_variance=V(REGKEY[k]+".minimum_detectable_auc_null_variance"),
        registered_matches_routeA=bool(V(REGKEY[k]+".minimum_detectable_auc")==round(a,4)),
        routeB_minus_routeA=round(b-a,4))
# consequence: the seventh-flip claim
R["seventh_flip_claim"]=dict(
    auroc_after_7_flips=V("c133.task_a.decontam_ensemble.steps.k7.auroc"),
    n_pos_after_7_flips=V("c133.task_a.decontam_ensemble.steps.k7.n_pos"),
    floor_routeA=R["strata"]["tip_k7_pos5"]["routeA_as_implemented"],
    floor_routeB=R["strata"]["tip_k7_pos5"]["routeB_standard"],
    floor_routeC=R["strata"]["tip_k7_pos5"]["routeC_null_only"])
c=R["seventh_flip_claim"]
c["clears_under_routeA"]=bool(c["auroc_after_7_flips"]>c["floor_routeA"])
c["clears_under_routeB"]=bool(c["auroc_after_7_flips"]>c["floor_routeB"])
c["clears_under_routeC"]=bool(c["auroc_after_7_flips"]>c["floor_routeC"])
# consequence: the 13-positive relabel claim
R["relabel_13pos_claim"]=dict(upper_end=V("c133.task_b.decontam_ensemble.steps.k1.ci95_hi"),
    floor_routeA=R["strata"]["relabel_pos13"]["routeA_as_implemented"],
    floor_routeB=R["strata"]["relabel_pos13"]["routeB_standard"])
json.dump(R,open(OUT/"a7_mde_formula.json","w"),indent=1,sort_keys=True); print(json.dumps(R,indent=1,sort_keys=True))
