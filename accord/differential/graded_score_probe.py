"""C98: a graded local score on G_253. See C98_PREREGISTRATION.md.

Local only: every request goes to an Ollama server on 127.0.0.1. Reads
credentialed gold data from mimic_local_only and prints and writes aggregates only.
No note text, pair text or identifier is printed or written.
"""
import json
import pathlib
import sys
import urllib.request

import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from accord.cloud_paths import assert_local_only  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
GOLD = pathlib.Path.home() / "mimic_local_only/from_drive_2026-08-08/outputs/pharmacist_labeled_clean.csv"
OUT = ROOT / "results/accord/differential/graded_score_granite33_g253.json"
HOST = "http://127.0.0.1:11434/api/chat"
MODEL = "granite3.3:8b"
POS = {"contr", "Contr", "con", "Con", "CONTR", "Contradiction"}
NEG = {"not", "Not", "NOT", "N", "No"}
SYSTEM = "Answer with one word: Contradiction or Not."
USER = ("Nursing note: {n}\nDischarge summary: {d}\n"
        "Do these give conflicting medication instructions?")

assert_local_only(GOLD)
assert HOST.startswith("http://127.0.0.1"), "local only"


def score(n, d):
    body = {"model": MODEL, "stream": False, "logprobs": True, "top_logprobs": 8,
            "options": {"temperature": 0, "num_predict": 1},
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": USER.format(n=n, d=d)}]}
    req = urllib.request.Request(HOST, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        r = json.loads(urllib.request.urlopen(req, timeout=300).read())
    except Exception as exc:  # type name only: a body can quote the request
        raise SystemExit(f"request failed: {type(exc).__name__}")
    lp = r.get("logprobs") or []
    top = (lp[0].get("top_logprobs") if lp else []) or []
    p = n_ = None
    for t in top:
        tok = t.get("token", "").strip()
        if p is None and tok in POS:
            p = t["logprob"]
        if n_ is None and tok in NEG:
            n_ = t["logprob"]
    return ((p if p is not None else -20.0) - (n_ if n_ is not None else -20.0),
            p is None or n_ is None)


def main():
    g = pd.read_csv(GOLD)
    g = g[g["label_clean"] != "ambiguous"].reset_index(drop=True)
    y = (g["label_clean"] == "contradiction").astype(int).tolist()
    scores, deg = [], 0
    for i, row in g.iterrows():
        s, d = score(str(row["nursing_sentence"]), str(row["discharge_sentence"]))
        scores.append(s); deg += d
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(g)} scored", flush=True)
    pos = [s for s, t in zip(scores, y) if t]
    neg = [s for s, t in zip(scores, y) if not t]
    auc = float(roc_auc_score(y, scores))
    _, p = mannwhitneyu(pos, neg, alternative="greater")
    facts = json.loads((ROOT / "results/accord/VERIFIED_FACTS.json").read_text())
    mde = facts["mde_recomputed.gold_ambiguous_excluded.minimum_detectable_auc"]["value"]
    frac_deg = deg / len(g)
    if len(scores) != 253 or frac_deg > 0.25:
        outcome = "C"
    elif auc >= mde and p < 0.05:
        outcome = "A"
    else:
        outcome = "B"
    out = {"preregistration": "accord/differential/C98_PREREGISTRATION.md",
           "model": MODEL, "basis": "G_253", "n": len(g), "n_positives": sum(y),
           "auroc": auc, "mannwhitney_p_one_sided": float(p),
           "n_degenerate": deg, "fraction_degenerate": frac_deg,
           "n_distinct_scores": len(set(scores)),
           "mde_auroc": mde, "outcome": outcome}
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
