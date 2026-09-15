"""Three tests bearing on whether discharge-summary length is a confounder or a
mediator of LANGUAGE-missingness in the candidate-yield regression.

The choice moves the replication between 1/3 and 2/3 cohorts and is not settled
by the yield regression itself. These tests do not prove a DAG; they check
whether the mediator story has a mechanism and whether era explains the effect.

Aggregates only. No note text. MIMIC-III credentialed, local only.
"""
import json, pathlib, numpy as np, pandas as pd, statsmodels.formula.api as smf, warnings
warnings.filterwarnings("ignore")

Q = "/Users/rickygole/mimic_local_only/from_drive_2026-08-08/data"
OUT = pathlib.Path("/Users/rickygole/mimic_local_only/accord_dag"); OUT.mkdir(exist_ok=True)
res = {}

# ---- assemble -------------------------------------------------------------
n = pd.read_csv("/Users/rickygole/mimic_local_only/accord/hf_note_pairs.csv.gz")
n.columns = [c.lower() for c in n.columns]
n["chars"] = n["text"].fillna("").str.len()
w = n.pivot_table(index="hadm_id", columns="final_role", values="chars",
                  aggfunc="sum").fillna(0).reset_index()
w.columns = [str(c).lower() for c in w.columns]

cand = pd.read_csv("/Users/rickygole/mimic_local_only/accord/contradiction_candidates.csv.gz")
cand.columns = [c.lower() for c in cand.columns]
w["has_cand"] = w.hadm_id.isin(set(cand.hadm_id)).astype(int)

adm = pd.read_csv(f"{Q}/ADMISSIONS.csv",
                  usecols=["HADM_ID","LANGUAGE","ADMISSION_TYPE","ADMITTIME","DISCHTIME","INSURANCE"])
adm.columns = [c.lower() for c in adm.columns]
d = w.merge(adm.drop_duplicates("hadm_id"), on="hadm_id", how="left")
d["lang"] = d.language.map(lambda x: "MISSING" if pd.isna(x) or str(x).strip()=="" else
                           ("ENGLISH" if str(x).strip().upper()=="ENGL" else "OTHER"))
d["los"] = (pd.to_datetime(d.dischtime)-pd.to_datetime(d.admittime)).dt.total_seconds()/3600

# era proxy from NOTEEVENTS category (C25)
era = {}
hs = set(d.hadm_id)
for ch in pd.read_csv("/Users/rickygole/mimic_local_only/raw_mimiciii/NOTEEVENTS.csv.gz",
                      usecols=["HADM_ID","CATEGORY"], chunksize=500_000, low_memory=False):
    ch = ch[ch.CATEGORY.isin(["Nursing","Nursing/other"]) & ch.HADM_ID.isin(hs)]
    for h, c in zip(ch.HADM_ID, ch.CATEGORY): era.setdefault(h, set()).add(c)
lab = lambda s: "CAREVUE" if s=={"Nursing/other"} else ("METAVISION" if s=={"Nursing"} else "BOTH")
d = d.merge(pd.DataFrame([(h, lab(s)) for h, s in era.items()],
                         columns=["hadm_id","era"]), on="hadm_id", how="left")
d = d.dropna(subset=["los","era"])
d["ln_n"] = np.log1p(d.nursing); d["ln_d"] = np.log1p(d.discharge)
d["missing"] = (d.lang=="MISSING").astype(int)
print(f"n = {len(d):,}")

# ---- TEST 1: does missingness predict discharge length after era+type? -----
print("\n=== TEST 1: is there an A -> D path at all? ===")
t1 = {}
for name, f in [("unadjusted", "ln_d ~ missing"),
                ("+ era", "ln_d ~ missing + C(era)"),
                ("+ era + admission type", "ln_d ~ missing + C(era) + C(admission_type)"),
                ("+ era + type + LOS", "ln_d ~ missing + C(era) + C(admission_type) + los")]:
    m = smf.ols(f, data=d).fit()
    b, ci, p = m.params["missing"], m.conf_int().loc["missing"], m.pvalues["missing"]
    t1[name] = dict(coef=round(b,4), ci=[round(ci[0],4), round(ci[1],4)], p=float(p))
    print(f"  {name:<26} beta={b:+.4f}  [{ci[0]:+.4f}, {ci[1]:+.4f}]  p={p:.3g}")
res["test1_A_to_D_path"] = t1

# ---- TEST 2: within-era analysis -------------------------------------------
print("\n=== TEST 2: does the yield effect survive within era? ===")
t2 = {}
K = "missing"
for era_name in ["CAREVUE","METAVISION"]:
    sub = d[d.era==era_name]
    n_miss = int(sub.missing.sum())
    row = {"n": len(sub), "n_missing": n_miss}
    if n_miss < 30 or len(sub)-n_miss < 30:
        row["skipped"] = "too few in one cell"; t2[era_name] = row
        print(f"  {era_name:<12} n={len(sub):>5}  missing={n_miss:>5}  -> skipped"); continue
    for name, f in [("raw","has_cand ~ missing + C(admission_type) + los"),
                    ("+ volume","has_cand ~ missing + C(admission_type) + los + ln_n + ln_d")]:
        m = smf.logit(f, data=sub).fit(disp=0)
        b, ci = m.params[K], m.conf_int().loc[K]
        row[name] = dict(OR=round(np.exp(b),4),
                         ci=[round(np.exp(ci[0]),4), round(np.exp(ci[1]),4)],
                         p=float(m.pvalues[K]))
        print(f"  {era_name:<12} {name:<10} OR={np.exp(b):.3f} "
              f"[{np.exp(ci[0]):.3f}, {np.exp(ci[1]):.3f}]  p={m.pvalues[K]:.3g}")
    t2[era_name] = row
res["test2_within_era"] = t2

# ---- TEST 3: negative control outcome --------------------------------------
print("\n=== TEST 3: negative control — does missingness predict LOS? ===")
t3 = {}
for name, f in [("unadjusted","los ~ missing"),
                ("+ era + type","los ~ missing + C(era) + C(admission_type)")]:
    m = smf.ols(f, data=d).fit()
    b, ci, p = m.params["missing"], m.conf_int().loc["missing"], m.pvalues["missing"]
    t3[name] = dict(coef=round(b,3), ci=[round(ci[0],3), round(ci[1],3)], p=float(p))
    print(f"  {name:<16} beta={b:+.2f} h  [{ci[0]:+.2f}, {ci[1]:+.2f}]  p={p:.3g}")
res["test3_negative_control_los"] = t3

res["era_by_missing"] = pd.crosstab(d.era, d.lang, normalize="index").round(4).to_dict()
(OUT/"dag_tests.json").write_text(json.dumps(res, indent=2))
print(f"\nwrote {OUT/'dag_tests.json'}")
