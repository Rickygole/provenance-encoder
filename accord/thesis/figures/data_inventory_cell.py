# =============================================================================
# WHAT DATA IS AVAILABLE, AND WHICH FIGURE USES IT
# Reads every stored result file and prints what is in it. Nothing is typed in.
# =============================================================================
import json, csv, pathlib, statistics as st

R = pathlib.Path('/content/repo/results')          # Colab
if not R.exists(): R = pathlib.Path('results')     # local fallback

FIG_USES = {
 "accord/step0_1_join_and_calibration.json": "3 per-rule ranking",
 "accord/step3_baselines.json":              "2 headline, 12 exchange rate",
 "accord/decontam/corrected_gold_metrics.json": "2 headline, 8 strata",
 "accord/control/final_report.json":         "8 strata, 6 power",
 "accord/stats/intervals.json":              "2 headline, 7 forest",
 "accord/scaling/convergence_results.json":  "9 scaling",
 "accord/scaling/scaling_results.json":      "9 scaling",
 "pretraining_runs.json":                    "5 falsification",
 "accord/multillm/per_model_results.json":   "4 prevalence response",
 "accord/prevalence_replication/results/sweep_results.json": "4 prevalence response",
 "accord/selection/threshold_robustness_results.json": "10 threshold surface",
 "accord/signed_gap_analysis.json":          "10 threshold surface",
 "accord/section6_auroc_power.json":         "6 power",
 "accord/exemplar/exemplar_correctness_recomputed.json": "13 exemplar",
 "accord/survey/coding_table.csv":           "not plotted; Appendix D",
}

def flat(o, pre="", out=None, depth=0):
    if out is None: out = {}
    if depth > 4: return out
    if isinstance(o, dict):
        for k, v in o.items(): flat(v, f"{pre}{k}.", out, depth+1)
    elif isinstance(o, list):
        if o and all(isinstance(x, (int, float)) for x in o):
            out[pre.rstrip('.')] = f"[{len(o)} numbers] min {min(o):.4g} max {max(o):.4g}"
        else:
            out[pre.rstrip('.')] = f"[list of {len(o)}]"
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        out[pre.rstrip('.')] = o
    elif isinstance(o, (str, bool)):
        out[pre.rstrip('.')] = str(o)[:60]
    return out

files = sorted(list(R.rglob("*.json")) + list(R.rglob("*.csv")))
print(f"{len(files)} result files mounted at {R}\n")

SHOW_ALL = False        # set True to print every key of every file, which is long

for f in files:
    rel = str(f.relative_to(R))
    used = FIG_USES.get(rel)
    if not SHOW_ALL and not used:
        continue
    print("=" * 92)
    print(f"{rel}")
    print(f"   used by figure: {used}" if used else "   not used by any figure")
    print("=" * 92)
    try:
        if f.suffix == ".csv":
            rows = list(csv.DictReader(open(f)))
            print(f"   {len(rows)} rows, columns: {', '.join(list(rows[0])[:8])}")
            if rows:
                for k in list(rows[0])[:6]:
                    vals = [r[k] for r in rows]
                    uniq = sorted(set(vals))
                    print(f"     {k:32s} {len(uniq)} distinct" +
                          (f"  {uniq[:5]}" if len(uniq) <= 8 else ""))
        else:
            d = json.load(open(f))
            if isinstance(d, list) and d and isinstance(d[0], dict):
                # A file that is a list of run records: summarise the runs, not the file.
                keys = list(d[0])
                print(f"   {len(d)} records, fields: {', '.join(keys[:9])}")
                grp = {}
                for r in d:
                    label = r.get('tag') or r.get('arm') or r.get('model') or 'all'
                    grp.setdefault(label, []).append(r)
                for label, rs in sorted(grp.items()):
                    nums = [k for k in keys if isinstance(rs[0].get(k), (int, float))
                            and not isinstance(rs[0].get(k), bool)]
                    parts = []
                    for k in nums[:4]:
                        v = [r[k] for r in rs if isinstance(r.get(k), (int, float))]
                        if v:
                            parts.append(f"{k}={st.mean(v):.4g}" + (f" sd {st.stdev(v):.3g}" if len(v) > 1 else ""))
                    print(f"     {label:22s} n={len(rs):<3} {'  '.join(parts)}")
                print()
                continue
            kv = flat(d)
            print(f"   {len(kv)} values")
            for k, v in list(kv.items())[:28]:
                vs = f"{v:.6g}" if isinstance(v, float) else str(v)
                print(f"     {k[:58]:58s} {vs}")
            if len(kv) > 28:
                print(f"     ... and {len(kv)-28} more")
    except Exception as e:
        print(f"   could not read: {e}")
    print()

print("=" * 92)
print("Set SHOW_ALL = True above to dump every file, including the ones no figure uses.")

# ---------------------------------------------------------------------------
# Values that are present in the data but must not be plotted.
# ---------------------------------------------------------------------------
LANDMINES = {
 "reference_full_2296_candidate_pool_rate_from_task_description":
   "0.586 was supplied by a task description, never measured, over a 2,296 pool "
   "that appears nowhere in this work. Struck from the thesis. Use n_wrong divided "
   "by n_resolved, which is 65 of 137.",
 "dp_min":  "named for a minimum but holds the sweep's FIRST point; for PUMS regime A "
            "that is not the minimum. Read the points, not the key name.",
 "dp_max":  "named for a maximum but holds the sweep's LAST point. Same caution.",
}
print()
print("=" * 92)
print("KEYS PRESENT IN THE DATA THAT MUST NOT BE PLOTTED")
print("=" * 92)
found = 0
for f in files:
    if f.suffix != ".json":
        continue
    try:
        raw = f.read_text()
    except Exception:
        continue
    for key, why in LANDMINES.items():
        if key in raw:
            print(f"  {f.relative_to(R)}")
            print(f"    key : {key}")
            print(f"    why : {why}")
            print()
            found += 1
if not found:
    print("  none present")
