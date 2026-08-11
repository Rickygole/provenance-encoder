import json
from pathlib import Path
ROOT = Path("/Users/rickygole/provenance-encoder")
d = json.load(open(ROOT/"results/accord/sign_provenance_drift_recomputation.json"))
n_metrics = n_flips = n_credcall = 0
maxdisc = 0.0
for exp, blk in d.items():
    if not isinstance(blk, dict): continue
    for met, r in blk.items():
        if not isinstance(r, dict) or "sign_flip_between_abs_and_signed" not in r: continue
        n_metrics += 1
        n_flips += bool(r["sign_flip_between_abs_and_signed"])
        n_credcall += bool(r.get("credibility_call_differs"))
        sa, ra = r.get("stored_abs_slope"), r.get("recomputed_abs_slope")
        if sa is not None and ra is not None: maxdisc = max(maxdisc, abs(sa-ra))
        print(f"{exp:32s} {met:22s} flip={r['sign_flip_between_abs_and_signed']} credcall_differs={r.get('credibility_call_differs')}")
print("\nn_metrics_compared =", n_metrics)
print("n_sign_flips       =", n_flips)
print("n_credibility_call_differs =", n_credcall)
print("max |stored_abs - recomputed_abs| =", maxdisc)

print("\n--- restricted to metric-slopes that HAVE a stored_abs_slope (the comparable set) ---")
nm = nf = 0
for exp, blk in d.items():
    if not isinstance(blk, dict): continue
    for met, r in blk.items():
        if not isinstance(r, dict) or "sign_flip_between_abs_and_signed" not in r: continue
        if r.get("stored_abs_slope") is None:
            print(f"  EXCLUDED (no stored value): {exp}.{met} flip={r['sign_flip_between_abs_and_signed']}")
            continue
        nm += 1; nf += bool(r["sign_flip_between_abs_and_signed"])
print("  n_metrics_compared =", nm, " n_sign_flips =", nf, " share = %.4f" % (nf/nm))
r = d["PUMS_RegimeB_drift_vs_beta"]["fpr_gap"]
print("\n  credibility-call-differing slope:", json.dumps(r, indent=1))
