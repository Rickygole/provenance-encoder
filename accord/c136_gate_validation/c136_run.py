"""C136. The marginal feasibility gate as a screen for student failure, on a design with headroom.

Rule fixed in C136_PREREGISTRATION.md before this script produced a number. Public data only (SNLI).
No hosted API beyond a local Ollama daemon. Writes results/accord/c136_gate_validation/c136.json.

  python3 accord/c136_gate_validation/c136_run.py --arm synthetic
  python3 accord/c136_gate_validation/c136_run.py --arm real
  python3 accord/c136_gate_validation/c136_run.py --arm combine
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "superseded_first_attempt"))
# The gate, the frames, the labeller grid, the Ollama caller and the screening arithmetic are
# imported from C136's own superseded first attempt, not re-implemented, so they cannot differ from
# the versions those rules were fixed with.
from run_superseded import (  # noqa: E402
    SEED, N_PILOT, K_PILOT, N_POOL, K_POOL, N_TEST, K_TEST, P_FLOOR_ALPHA,
    FAIL_THRESHOLDS, SYNTHETIC_GRID, OLLAMA_MODELS, PROMPT,
    gate_verdict, gate_fail_region, load_snli_frames, synthetic_labels,
    ollama_flags, screen, sweep_ratio,
)

OUTDIR = os.path.join(REPO, "results", "accord", "c136_gate_validation")
MINILM = "sentence-transformers/all-MiniLM-L6-v2"
REFERENCE_FLOOR = 0.75  # pre-registered headroom precondition on the primary student


# ---------------------------------------------------------------- students

class TfidfStudent:
    name = "tfidf"

    def __init__(self, frames):
        pool_x = np.array([f"{p} [SEP] {h}" for p, h in zip(*frames["pool_pair"])], dtype=object)
        test_x = np.array([f"{p} [SEP] {h}" for p, h in zip(*frames["test_pair"])], dtype=object)
        self.vec = TfidfVectorizer(lowercase=True, sublinear_tf=True, min_df=2, ngram_range=(1, 2))
        self.Xp = self.vec.fit_transform(pool_x)
        self.Xt = self.vec.transform(test_x)
        self.y_test = frames["test_y"]

    def fit_score(self, y):
        y = np.asarray(y, int)
        if y.min() == y.max():
            return 0.5, True
        clf = LogisticRegression(C=1.0, solver="liblinear", class_weight="balanced", max_iter=2000)
        clf.fit(self.Xp, y)
        return float(roc_auc_score(self.y_test, clf.decision_function(self.Xt))), False


class MiniLMStudent:
    name = "minilm"

    def __init__(self, frames):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(MINILM)
        self.mdl = AutoModel.from_pretrained(MINILM).eval()
        self.dev = "mps" if torch.backends.mps.is_available() else "cpu"
        self.mdl = self.mdl.to(self.dev)
        self.Xp = self._feat(*frames["pool_pair"])
        self.Xt = self._feat(*frames["test_pair"])
        self.y_test = frames["test_y"]

    def _enc(self, texts, bs=256):
        out = []
        for i in range(0, len(texts), bs):
            b = self.tok(list(texts[i:i + bs]), padding=True, truncation=True,
                         max_length=96, return_tensors="pt").to(self.dev)
            with self.torch.no_grad():
                h = self.mdl(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).float()
            e = (h * m).sum(1) / m.sum(1)
            out.append(self.torch.nn.functional.normalize(e, dim=1).cpu().numpy())
        return np.vstack(out)

    def _feat(self, prem, hyp):
        u, v = self._enc(prem), self._enc(hyp)
        return np.hstack([u, v, np.abs(u - v), u * v])

    def fit_score(self, y):
        y = np.asarray(y, int)
        if y.min() == y.max():
            return 0.5, True
        clf = LogisticRegression(C=4.0, max_iter=3000, class_weight="balanced")
        clf.fit(self.Xp, y)
        return float(roc_auc_score(self.y_test, clf.decision_function(self.Xt))), False


# ---------------------------------------------------------------- frames

def build_frames():
    """C135's SNLI draw, re-exposed as premise / hypothesis pairs (same rows, same order)."""
    f = load_snli_frames()
    out = {"_source": f["_source"], "_disjoint": f["_disjoint"]}
    for k in ("pilot", "pool", "test"):
        text, y = f[k]
        pairs = [t.split(" [SEP] ", 1) for t in text]
        out[f"{k}_pair"] = (np.array([a for a, _ in pairs], dtype=object),
                            np.array([b for _, b in pairs], dtype=object))
        out[f"{k}_y"] = y
        out[f"{k}_text"] = text
    return out


# ---------------------------------------------------------------- analysis

def analyse(rows, tag, student):
    key = f"student_auroc_{student}"
    rows = [r for r in rows if key in r]
    v = [r["gate_fail"] for r in rows]
    a = [r[key] for r in rows]
    ratios = [r["ratio_flagrate_over_baserate"] for r in rows]
    out = {"tag": tag, "student": student, "n_labellers": len(rows), "screening": {}}
    if not rows:
        return out
    fail_region = gate_fail_region(N_PILOT, K_PILOT)
    tail = [n for n in fail_region if n > 0]
    tail_ok = bool(tail) and tail == list(range(min(tail), N_PILOT + 1))
    for thr in FAIL_THRESHOLDS:
        s = screen(v, a, thr)
        eq = None
        if tail_ok:
            t = min(tail) / K_PILOT
            pred = [r >= t for r in ratios]
            eq = {"ratio_threshold": t,
                  "identical_to_gate_verdicts": bool(pred == v),
                  "disagreements": int(sum(int(p != q) for p, q in zip(pred, v))),
                  "screening": screen(pred, a, thr)}
        _, best = sweep_ratio(ratios, a, thr)
        out["screening"][f"auroc_lt_{thr:.2f}"] = {
            "gate": s,
            "ratio_at_gate_equivalent_threshold": eq,
            "ratio_best_over_all_thresholds": best,
            "gate_youden_j": s["youden_j"],
            "ratio_beats_gate": bool(best is not None and s["youden_j"] is not None
                                     and best["youden_j"] > s["youden_j"] + 1e-12),
            "gate_adds_nothing_over_raw_ratio": bool(eq is not None
                                                     and eq["identical_to_gate_verdicts"]),
        }
    if len(rows) > 2:
        rc = spearmanr([r["ceiling_mcc"] for r in rows], a)
        rp = spearmanr([r["floor_p"] for r in rows], a)
        out["spearman_ceilingmcc_vs_auroc"] = {"rho": float(rc.statistic), "p": float(rc.pvalue)}
        out["spearman_floorp_vs_auroc"] = {"rho": float(rp.statistic), "p": float(rp.pvalue)}
    return out


# ---------------------------------------------------------------- arms

def make_students(frames):
    return [MiniLMStudent(frames), TfidfStudent(frames)]


def run_synthetic(frames, students):
    pilot_y, pool_y = frames["pilot_y"], frames["pool_y"]
    assert int(pilot_y.sum()) == K_PILOT
    ref = {s.name: s.fit_score(pool_y)[0] for s in students}
    print("  clean-label reference AUROC:", {k: round(x, 4) for k, x in ref.items()}, flush=True)
    rng = np.random.default_rng(SEED)
    rows = []
    for tpr, fpr in SYNTHETIC_GRID:
        lp = synthetic_labels(pilot_y, tpr, fpr, rng)
        lt = synthetic_labels(pool_y, tpr, fpr, rng)
        g = gate_verdict(N_PILOT, K_PILOT, int(lp.sum()))
        r = {"labeller": f"synth_tpr{tpr:.2f}_fpr{fpr:.2f}", "family": "synthetic",
             "tpr": tpr, "fpr": fpr,
             "pilot_true_positives_flagged": int(((lp == 1) & (pilot_y == 1)).sum()),
             "pool_flag_count": int(lt.sum()), "pool_flag_rate": float(lt.mean()),
             "ratio_flagrate_over_baserate": g["flag_count"] / K_PILOT, **g}
        for s in students:
            au, dg = s.fit_score(lt)
            r[f"student_auroc_{s.name}"] = au
            r[f"student_degenerate_{s.name}"] = dg
        rows.append(r)
        print(f"  {r['labeller']}  n={g['flag_count']:3d}  p_floor={g['floor_p']:.3g}  "
              f"gate_fail={str(g['gate_fail']):5s}  minilm={r['student_auroc_minilm']:.4f}  "
              f"tfidf={r['student_auroc_tfidf']:.4f}", flush=True)
    return rows, ref


def run_real(frames, students, models):
    pilot_t, pool_t = frames["pilot_text"], frames["pool_text"]
    pilot_y = frames["pilot_y"]
    rows = []
    for m in models:
        t0 = time.time()
        try:
            fp_, up_p = ollama_flags(m, pilot_t, True)
            fl_, up_l = ollama_flags(m, pool_t, True)
        except Exception as e:
            print(f"  {m}: NOT RUN ({type(e).__name__})", flush=True)
            continue
        g = gate_verdict(N_PILOT, K_PILOT, int(fp_.sum()))
        r = {"labeller": m, "family": "real_ollama",
             "pilot_true_positives_flagged": int(((fp_ == 1) & (pilot_y == 1)).sum()),
             "pilot_unparsed": up_p, "pool_unparsed": up_l,
             "pool_flag_count": int(fl_.sum()), "pool_flag_rate": float(fl_.mean()),
             "ratio_flagrate_over_baserate": g["flag_count"] / K_PILOT, **g}
        for s in students:
            au, dg = s.fit_score(fl_)
            r[f"student_auroc_{s.name}"] = au
            r[f"student_degenerate_{s.name}"] = dg
        r["seconds"] = round(time.time() - t0, 1)
        rows.append(r)
        print(f"  {m}  n={g['flag_count']:3d}  p_floor={g['floor_p']:.3g}  "
              f"gate_fail={str(g['gate_fail']):5s}  minilm={r['student_auroc_minilm']:.4f}  "
              f"tfidf={r['student_auroc_tfidf']:.4f}  ({r['seconds']}s)", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["synthetic", "real", "combine"], required=True)
    ap.add_argument("--models", default=",".join(OLLAMA_MODELS))
    a = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)

    if a.arm == "combine":
        syn = json.load(open(os.path.join(OUTDIR, "c136_synthetic.json")))
        rows = list(syn["labellers"])
        rp = os.path.join(OUTDIR, "c136_real.json")
        real = json.load(open(rp))["labellers"] if os.path.exists(rp) else []
        fr = gate_fail_region(N_PILOT, K_PILOT)
        tail = [n for n in fr if n > 0]
        ref = syn["clean_label_reference_auroc"]
        art = {
            "check": "C136 marginal feasibility gate validation",
            "prereg": "accord/c136_gate_validation/C136_PREREGISTRATION.md",
            "seed": SEED,
            "data_source": syn["data_source"],
            "frames": syn["frames"],
            "clean_label_reference_auroc": ref,
            "primary_student": "minilm",
            "headroom_precondition": {"floor": REFERENCE_FLOOR,
                                      "primary_reference_auroc": ref["minilm"],
                                      "passed": bool(ref["minilm"] >= REFERENCE_FLOOR)},
            "gate": {"alpha": P_FLOOR_ALPHA,
                     "N": N_PILOT, "K": K_PILOT,
                     "fail_region_flag_counts": fr,
                     "fail_region_is_zero_plus_upper_tail": bool(
                         0 in fr and tail and tail == list(range(min(tail), N_PILOT + 1))),
                     "smallest_failing_positive_flag_count": min(tail) if tail else None,
                     "smallest_failing_positive_flag_rate": (min(tail) / N_PILOT) if tail else None},
            "labellers": rows + real,
            "n_real_labellers": len(real),
        }
        for st in ("minilm", "tfidf"):
            art[f"synthetic_{st}"] = analyse(rows, "synthetic", st)
            art[f"real_{st}"] = (analyse(real, "real_ollama", st) if len(real) > 2 else
                                 {"tag": "real_ollama", "student": st, "n_labellers": len(real),
                                  "note": "too few real labellers for screening statistics"})
            art[f"pooled_{st}"] = analyse(rows + real, "pooled", st) if real else None
        art["primary_reading"] = art["synthetic_minilm"]["screening"]["auroc_lt_0.60"]
        p = os.path.join(OUTDIR, "c136.json")
        json.dump(art, open(p, "w"), indent=2, sort_keys=True)
        print("wrote", p)
        return

    frames = build_frames()
    meta = {"data_source": frames["_source"], "seed": SEED,
            "frames": {"pilot": [N_PILOT, int(frames["pilot_y"].sum())],
                       "pool": [N_POOL, int(frames["pool_y"].sum())],
                       "test": [N_TEST, int(frames["test_y"].sum())],
                       "pool_test_disjoint": frames["_disjoint"]}}
    print(json.dumps(meta), flush=True)
    students = make_students(frames)
    if a.arm == "synthetic":
        rows, ref = run_synthetic(frames, students)
        meta["clean_label_reference_auroc"] = ref
        p = os.path.join(OUTDIR, "c136_synthetic.json")
    else:
        rows = run_real(frames, students, [m for m in a.models.split(",") if m])
        p = os.path.join(OUTDIR, "c136_real.json")
    json.dump({**meta, "labellers": rows}, open(p, "w"), indent=2, sort_keys=True)
    print("wrote", p)


if __name__ == "__main__":
    sys.exit(main())
