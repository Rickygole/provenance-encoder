"""Aggregate the 4 training-config JSON results + homogeneity.json into a
single summary table. Aggregates only -- no note text ever touched here."""
import json
import glob

OUT_DIR = "/Users/rickygole/mimic_local_only/accord_stage0"

CONFIGS = [
    ("config1_baseline_bottleneck_head.json", "1: baseline (bottleneck + head)"),
    ("config2_h1_direct_head.json", "2: H1 ablation (direct head + head window)"),
    ("config3_h2_bottleneck_tail.json", "3: H2 ablation (bottleneck + tail window)"),
    ("config4_both_direct_tail.json", "4: both fixed (direct head + tail window)"),
]

print(f"{'config':<45}{'trainable':>12}{'final_train_loss':>18}{'final_dev_loss':>16}{'final_dev_acc':>16}")
for fname, label in CONFIGS:
    path = f"{OUT_DIR}/{fname}"
    try:
        with open(path) as f:
            r = json.load(f)
    except FileNotFoundError:
        print(f"{label:<45}  MISSING ({path})")
        continue
    print(f"{label:<45}{r['trainable_params']:>12,}{r['final_train_loss']:>18.4f}"
          f"{r['final_dev_loss']:>16.4f}{r['final_dev_acc']:>16.4f}")

print()
try:
    with open(f"{OUT_DIR}/homogeneity.json") as f:
        h = json.load(f)
    for window, res in h["results"].items():
        m, mm = res["matched"], res["mismatched"]
        print(f"[homogeneity:{window}] matched mean={m['mean']:.4f} std={m['std']:.4f} "
              f"| mismatched mean={mm['mean']:.4f} std={mm['std']:.4f} "
              f"| AUROC(cosine separates matched)={res['auroc_cosine_separates_matched']:.4f}")
except FileNotFoundError:
    print("homogeneity.json MISSING")
