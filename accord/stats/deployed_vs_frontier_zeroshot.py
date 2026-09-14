"""Is the deployed llama3:8b weak labeller the same labelling as the local
llama3:8b frontier zero-shot run (whose prompt text equals C103's P_asserting)?
Committee round after C113, paper-reviewer finding 8: both flag 261 of 299 with
TP 11. Compares the two label vectors pair by pair and writes counts only.

Local only. Reuses reconcile_c28.build_join (the exact gold <-> weak-label join).
No pair_id, identifier or text is printed or written.

    python3 accord/stats/deployed_vs_frontier_zeroshot.py
"""
import contextlib
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "accord" / "stats"))
LOG = pathlib.Path.home() / "mimic_local_only" / "logs" / f"deployed_vs_fzs_{time.strftime('%Y%m%dT%H%M%S')}.log"
OUT = ROOT / "results" / "accord" / "stats" / "deployed_vs_frontier_zeroshot.json"
PRED = ROOT / "results" / "accord" / "differential" / "frontier_ollama_llama3_zero_shot.json"


def main():
    with LOG.open("w") as lf, contextlib.redirect_stdout(lf), contextlib.redirect_stderr(lf):
        from reconcile_c28 import build_join
        j, _ = build_join()
        preds = {p["pair_id"]: int(p["pred"]) for p in json.loads(PRED.read_text())["predictions"]}
        j = j[j["pair_id"].isin(preds)].copy()
        j["fzs"] = j["pair_id"].map(preds)
        n = int(len(j))
        both_pos = int(((j.weak_pos == 1) & (j.fzs == 1)).sum())
        both_neg = int(((j.weak_pos == 0) & (j.fzs == 0)).sum())
        out = {
            "n_joined": n,
            "n_deployed_positive": int(j.weak_pos.sum()),
            "n_frontier_zero_shot_positive": int(j.fzs.sum()),
            "n_agree": both_pos + both_neg,
            "n_deployed_pos_fzs_neg": int(((j.weak_pos == 1) & (j.fzs == 0)).sum()),
            "n_deployed_neg_fzs_pos": int(((j.weak_pos == 0) & (j.fzs == 1)).sum()),
        }
        out["identical"] = out["n_agree"] == n
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
