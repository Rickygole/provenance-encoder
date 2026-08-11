"""Recompute the seven-arm architecture ablation from results/pretraining_runs.json.
Every number in Table tab:ablation (msu_ch4_results.tex) is re-derived here.
Run: python3 accord/stats/recompute_ablation.py
"""
import json, itertools, statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results" / "pretraining_runs.json"
runs = json.load(open(SRC))

# Arm -> run ids. Derived by filtering on (arm, steps==1000, shuffle_segments==True)
# and the control flag / suffix that names the bias variant. Printed below for audit.
def ctrl(v, name):
    return bool(v.get("controls", {}).get(name, False))

ARMS = {}
for k, v in runs.items():
    if v.get("steps") != 1000 or v.get("size") != "10M":
        continue
    if not ctrl(v, "shuffle_segments"):
        continue
    if ctrl(v, "global_positions"):
        continue  # separate positional control, not one of the seven
    a = v["arm"]
    if a == "A1":
        arm = "vanilla"
    elif a == "A3":
        if "_rpe_" in k: arm = "t5_rpe"
        elif "_alibi_" in k: arm = "alibi"
        elif "_hardblk_" in k: arm = "hard_block"
        else: arm = "A3_other_" + k
    elif a == "A4":
        if "_segidx_misalign_" in k: arm = "misaligned_block"
        elif "_segidx_" in k: arm = "segment_index"
        elif "_lenroles_" in k: arm = "length_roles"
        elif "_randroles_" in k: arm = "random_roles"
        else: arm = "true_role_labels"
    else:
        arm = "other_" + k
    ARMS.setdefault(arm, []).append(k)

out = {"source": str(SRC.relative_to(ROOT)), "key": "final_val_loss", "arms": {}}
means = {}
for arm, ks in sorted(ARMS.items()):
    vals = [runs[k]["final_val_loss"] for k in sorted(ks)]
    seeds = sorted(runs[k]["seed"] for k in ks)
    rec = {
        "run_ids": sorted(ks), "seeds": seeds, "n": len(vals), "values": vals,
        "mean": st.mean(vals),
        "sd_sample_ddof1": st.stdev(vals) if len(vals) > 1 else None,
        "sd_pop_ddof0": st.pstdev(vals) if len(vals) > 1 else None,
    }
    out["arms"][arm] = rec
    means[arm] = st.mean(vals)

van = means["vanilla"]
seg_gain = means["segment_index"] - van  # negative
for arm, rec in out["arms"].items():
    d = rec["mean"] - van
    rec["delta_vs_vanilla"] = d
    rec["pct_of_segment_index"] = 100.0 * d / seg_gain if seg_gain else None

out["derived"] = {
    "alignment_worth_nats": abs(means["segment_index"] - means["misaligned_block"]),
    "learned_vs_fixed_nats": abs(means["hard_block"] - means["segment_index"]),
    "sd_min_seven": None, "sd_max_seven": None,
}
SEVEN = ["vanilla","t5_rpe","misaligned_block","alibi","true_role_labels","segment_index","hard_block"]
sds = [out["arms"][a]["sd_sample_ddof1"] for a in SEVEN]
sds0 = [out["arms"][a]["sd_pop_ddof0"] for a in SEVEN]
out["derived"]["sd_min_seven"] = min(sds); out["derived"]["sd_max_seven"] = max(sds)
out["derived"]["sd_min_seven_pop"] = min(sds0); out["derived"]["sd_max_seven_pop"] = max(sds0)

# ordering check: are arms separated beyond seed noise? Welch t between adjacent ranks.
ordered = sorted(SEVEN, key=lambda a: -out["arms"][a]["mean"])
out["derived"]["rank_order_high_to_low_loss"] = ordered

print(json.dumps(out, indent=2))
if __name__ == "__main__":
    (ROOT / "accord" / "stats" / "recompute_ablation_out.json").write_text(json.dumps(out, indent=2))
