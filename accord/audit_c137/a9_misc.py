"""Audit pass 9: authority-parameter spreads, SD conventions, prose arithmetic, figure data."""
import json,math,pathlib,glob
import numpy as np
ROOT=pathlib.Path("/Users/rickygole/provenance-encoder"); OUT=ROOT/"results/accord/audit_c137"
F=json.load(open(ROOT/"results/accord/VERIFIED_FACTS.json"))
def V(k):
    v=F.get(k); return None if v is None else (v.get("value") if isinstance(v,dict) else v)
R={}
for nm,key in [("discharge","alpha_discharge"),("nursing","alpha_nursing")]:
    vals=[V(f"decontam_gold.per_seed.{i}.authority_scale.{key}") for i in (0,1,2)]
    R[f"authority_{nm}"]=dict(vals=vals,mean=float(np.mean(vals)),mean_r4=round(float(np.mean(vals)),4),
        min_r4=round(min(vals),4),max_r4=round(max(vals),4),
        sd_pop=float(np.std(vals,ddof=0)),sd_pop_r4=round(float(np.std(vals,ddof=0)),4),
        sd_samp=float(np.std(vals,ddof=1)),sd_samp_r4=round(float(np.std(vals,ddof=1)),4),
        drift_pct=(float(np.mean(vals))-(1.0 if nm=="discharge" else 0.5))/(1.0 if nm=="discharge" else 0.5)*100)
# prose arithmetic that must recompute from operands
R["arith"]={
 "expected_top15_G253": 15*12/253, "expected_top15_G299": 15*12/299,
 "ratio_qwen8_over_qwen14": 14.1667/13.4167, "ratio_exact_170_over_161": (170/12)/(161/12),
 "spread_3x8B_max_over_min": 21.75/14.1667, "spread_exact": (261/12)/(170/12),
 "delong_se_over_diff_row1": 0.1025/0.0090, "delong_se_over_diff_row2": 0.1052/0.0012,
 "delong_se_over_diff_row3_seed123": 0.0979/0.0427, "delong_se_over_diff_row4_seed2024": 0.0967/0.0679,
 "delong_se_over_diff_row5_G134": 0.0496/0.0391,
 "decontam_minus_contaminated": 0.5145228215767634-0.4595435684647302,
 "g253_minus_g108_gap": 0.6094771241830065-0.5145228215767634,
 "prob_all23_correct": 0.5255**23, "one_minus_0p4745_pow23": (1-0.4745)**23,
 "amb_rescale_factor": 12/58,
 "pool_1183_times_0p8521": 1183*0.8521,
 "pool_1008_over_1183": 1008/1183,
 "pool_1008_over_1183_r4": round(1008/1183,4),
 "train_pos_rate_registered": V("control.baselines.train_pos_rate"),
 "train_pos_rate_times_1183": V("control.baselines.train_pos_rate")*1183,
 "pool_chain_1978_299": 1978-299, "pool_chain_minus_291_205": 1978-299-291-205,
 "pool_1388_minus_205": V("control.baselines.n_pool_after_subject_exclusion")-V("control.baselines.n_validation"),
 "rules_sum_1978": 886+202+520+370, "extraction_total_raw": V("extraction.contradiction_mining.total_raw"),
 "annotation_sample_sum": 134+31+79+V("extraction.annotation_sample.rule4"),
 "public_cells_7_15_2": 7+15+2,
 "nli_g299_ci_width": 0.7375145180023228-0.4091173054587689,
 "decontam_ci_width": 0.7033717778609418-0.3209574856693268,
 "c130_ci_width": 0.7195712309820194-0.345434996612753,
 "mcc_accord_ci_width": 0.0753-(-0.1626), "mcc_llama_ci_width": 0.0563-(-0.2337),
 "exemplardirect_146_minus_137": 146-137, "exemplardirect_69_plus_67": 69+67,
 "exemplardirect_136_plus_10": 136+10, "exemplardirect_146_plus_19": 146+19,
 "recon_137_plus_28": 137+28, "recon_72_plus_65": 72+65, "recon_65_over_137": 65/137,
 "c129_1061_plus_246": 1061+246,
 "mcnemar_30_plus_21": 30+21,
 "deployed_11_250_1_37": 11+250+1+37, "deployed_g253_11_212_1_29": 11+212+1+29,
 "n_tests_family": 5+5+1+8+4,
 "n_tests_family_plus_threshold_sweep": 5+5+1+8+4+2,
 "harm_band_width": 0.609565019607544-0.3977051973342895,
}
# figure data files
figdir=ROOT/"accord/thesis/figures"
R["figure_data_files"]=sorted(p.name for p in figdir.glob("*.json"))[:40]
json.dump(R,open(OUT/"a9_misc.json","w"),indent=1,sort_keys=True); print(json.dumps(R,indent=1,sort_keys=True))
