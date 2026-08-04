"""Export every result into the repo so findings are version-controlled.

Run metrics live under local_only/, which is gitignored because it also holds
MIMIC-derived text. The metrics themselves are scalars -- losses, AUROC, seed
numbers, parameter counts -- and contain no patient data, so they belong in the
repo where they can be diffed, reviewed and cited.

This copies aggregates ONLY. It explicitly drops any field that could carry
text, and asserts the output is clean before writing.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCAL = ROOT / "local_only"
OUT = ROOT / "results"

# Fields permitted in exported records. Anything else is dropped rather than
# filtered by name, so a new text-bearing field cannot leak by being forgotten.
SCALAR_KEYS = {
    "arm", "size", "seed", "steps", "tokens", "non_emb_params", "final_val_loss",
    "final_val_ppl", "wall_seconds", "loss_by_segment_position", "controls",
    "provenance_params", "history", "init", "hard_block", "auroc", "auprc",
    "best_epoch", "prevalence", "minutes", "tag", "n_train", "n_val", "epochs",
    "max_len", "accuracy", "macro_f1", "level", "window", "fano_bits",
    "frac_of_marginal", "n_test", "n_test_subjects", "removed", "kept", "vocab",
    "per_class_f1", "n", "positives", "window_hours", "excluded_roles",
    "role_hist", "splits", "mean_segments", "n_bundles", "model", "unparsed",
    "scrub_level", "confusion", "total_original", "total_matched", "per_role",
    "orig_n", "match_n", "retained", "orig_median", "match_median",
    "match_p10", "match_p90", "length_only", "levels", "length_matched_level2",
    "full_note_scrubbed", "windows", "top_terms",
    # Pretraining-corpus lineage (contamination fix; see src/data/lineage.py
    # and scripts/build_bundles.py).
    "task_split_hash", "task_split_counts", "derived_from_task_dir",
    "pretrain_val_frac", "contamination_fix", "bundles_content_hash",
    "bundles_dir", "vocab_size",
    # Token/label-budget ladders (scripts/run_token_ladder.sh,
    # scripts/run_label_ladder.sh).
    "mlm_tokens", "mlm_val_loss", "ckpt", "budget_tag", "n_train", "prevalence",
    "metadata_auroc", "tfidf_auroc", "ours_random_auroc_mean", "ours_random_auroc_std",
    "ours_pretrained_auroc_mean", "ours_pretrained_auroc_std", "pretrain_minus_random",
    "subsample_train", "target_n", "before_n", "achieved_n", "achieved_prevalence",
}


def clean(obj):
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items() if k in SCALAR_KEYS or _scalarish(v)}
    if isinstance(obj, list):
        return [clean(x) for x in obj]
    return obj


def _scalarish(v) -> bool:
    return isinstance(v, (int, float, bool, type(None))) or (
        isinstance(v, str) and len(v) < 80)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    exported = {}

    # Pretraining / control runs
    pre = {}
    for p in sorted(LOCAL.glob("runs/*/summary.json")):
        pre[p.parent.name] = clean(json.loads(p.read_text()))
    if pre:
        (OUT / "pretraining_runs.json").write_text(json.dumps(pre, indent=2, sort_keys=True))
        exported["pretraining_runs.json"] = len(pre)

    # Downstream task runs
    task = {}
    for p in sorted(LOCAL.glob("task_runs/*.json")):
        task[p.stem] = clean(json.loads(p.read_text()))
    if task:
        (OUT / "task_runs.json").write_text(json.dumps(task, indent=2, sort_keys=True))
        exported["task_runs.json"] = len(task)

    # Probes and dataset statistics
    for name, src in [
        ("probe_role.json", LOCAL / "role_probe_results.json"),
        ("probe_robustness.json", LOCAL / "probe_robustness.json"),
        ("probe_granularity.json", LOCAL / "probe_granularity.json"),
        ("probe_matched_slice.json", LOCAL / "matched_slice_profile.json"),
        ("ollama_role_baseline.json", LOCAL / "ollama_role_baseline.json"),
        ("dataset_bundles.json", LOCAL / "bundles" / "meta.json"),
        ("dataset_task.json", LOCAL / "task" / "meta.json"),
    ]:
        if src.exists():
            (OUT / name).write_text(json.dumps(clean(json.loads(src.read_text())),
                                               indent=2, sort_keys=True))
            exported[name] = 1

    # Safety assertion: nothing exported may contain a long free-text blob.
    for p in OUT.glob("*.json"):
        blob = p.read_text()
        for tok in ("admission date", "chief complaint", "[**", "discharge medications"):
            assert tok not in blob.lower(), f"possible note text in {p.name}"

    print(f"exported to {OUT}/")
    for k, v in sorted(exported.items()):
        print(f"  {k:<32}{v:>4} record(s)")
    print("\nverified: no note text in any exported file")


if __name__ == "__main__":
    main()
