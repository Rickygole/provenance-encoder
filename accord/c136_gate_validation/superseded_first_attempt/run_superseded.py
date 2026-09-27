"""C135. Validate the marginal feasibility gate as a screen for student failure.

Rule fixed in C135_PREREGISTRATION.md before this script produced a number.
Public data only (SNLI). No hosted API. Writes results/accord/c135_gate_validation/c135.json.

Usage:
  python3 accord/c135_gate_validation/c135_run.py --arm synthetic
  python3 accord/c135_gate_validation/c135_run.py --arm real --models qwen3:0.6b,qwen3:1.7b
  python3 accord/c135_gate_validation/c135_run.py --arm combine
"""
import argparse
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.stats import hypergeom, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

SEED = 42
N_PILOT, K_PILOT = 300, 12
N_POOL, K_POOL = 5000, 200
N_TEST, K_TEST = 10000, 400
PREVALENCE = 0.04
P_FLOOR_ALPHA = 0.05
FAIL_THRESHOLDS = (0.60, 0.70)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTDIR = os.path.join(REPO, "results", "accord", "c135_gate_validation")

SYNTHETIC_GRID = (
    [(0.90, f) for f in (0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.87, 0.93, 0.97)]
    + [(0.50, f) for f in (0.01, 0.05, 0.20, 0.50, 0.87)]
    + [(0.10, f) for f in (0.01, 0.05, 0.20, 0.50)]
    + [(0.04, f) for f in (0.03, 0.30, 0.90)]
)

OLLAMA_MODELS = ["qwen3:0.6b", "qwen3:1.7b", "qwen3:4b", "llama3:8b",
                 "granite3.3:8b", "qwen3:8b", "qwen3:14b"]

PROMPT = ("Premise: {p}\nHypothesis: {h}\n\n"
          "Does the hypothesis contradict the premise? Answer YES or NO only.")


# ---------------------------------------------------------------- the gate

def gate_ceiling_mcc(N, K, n):
    """Maximum MCC attainable with marginals (N, K, n) fixed."""
    tp = min(n, K)
    fp = n - tp
    fn = K - tp
    tn = N - n - fn
    num = tp * tn - fp * fn
    den2 = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if den2 <= 0:
        return 0.0
    return float(num / np.sqrt(float(den2)))


def gate_floor_p(N, K, n):
    """Smallest attainable one sided hypergeometric p value for marginals (N, K, n)."""
    m = min(n, K)
    return float(hypergeom.sf(m - 1, N, K, n))


def gate_verdict(N, K, n):
    p = gate_floor_p(N, K, n)
    return {"flag_count": int(n), "flag_rate": n / N,
            "ceiling_mcc": gate_ceiling_mcc(N, K, n), "floor_p": p,
            "gate_fail": bool(p >= P_FLOOR_ALPHA)}


def gate_fail_region(N, K):
    """Every flag count the gate fails, checked rather than asserted."""
    fails = [n for n in range(N + 1) if gate_floor_p(N, K, n) >= P_FLOOR_ALPHA]
    return fails


# ---------------------------------------------------------------- data

def load_snli_frames():
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/snli")
    rng = np.random.default_rng(SEED)
    out = {}

    def prep(split):
        d = ds[split]
        lab = np.asarray(d["label"])
        keep = lab >= 0
        prem = np.asarray(d["premise"], dtype=object)[keep]
        hyp = np.asarray(d["hypothesis"], dtype=object)[keep]
        y = (lab[keep] == 2).astype(int)
        text = np.array([f"{a} [SEP] {b}" for a, b in zip(prem, hyp)], dtype=object)
        return text, y

    val_text, val_y = prep("validation")
    tr_text, tr_y = prep("train")

    # pilot from validation
    pi = rng.permutation(np.flatnonzero(val_y == 1))[:K_PILOT]
    ni = rng.permutation(np.flatnonzero(val_y == 0))[:N_PILOT - K_PILOT]
    idx = rng.permutation(np.concatenate([pi, ni]))
    out["pilot"] = (val_text[idx], val_y[idx])

    # pool and test from train, disjoint
    tp = rng.permutation(np.flatnonzero(tr_y == 1))
    tn = rng.permutation(np.flatnonzero(tr_y == 0))
    pool_i = np.concatenate([tp[:K_POOL], tn[:N_POOL - K_POOL]])
    test_i = np.concatenate([tp[K_POOL:K_POOL + K_TEST],
                             tn[N_POOL - K_POOL:N_POOL - K_POOL + N_TEST - K_TEST]])
    pool_i = rng.permutation(pool_i)
    test_i = rng.permutation(test_i)
    out["pool"] = (tr_text[pool_i], tr_y[pool_i])
    out["test"] = (tr_text[test_i], tr_y[test_i])
    out["_disjoint"] = bool(len(set(pool_i.tolist()) & set(test_i.tolist())) == 0)
    out["_source"] = "SNLI (stanfordnlp/snli), binary contradiction vs not"
    return out


def load_gaussian_frames():
    """Fallback only. Two class Gaussian features, no text."""
    rng = np.random.default_rng(SEED)
    dim, sep = 50, 0.55

    def draw(n, k):
        y = np.concatenate([np.ones(k, int), np.zeros(n - k, int)])
        x = rng.normal(size=(n, dim))
        x[y == 1] += sep / np.sqrt(dim)
        p = rng.permutation(n)
        return x[p], y[p]

    out = {}
    out["pilot"] = draw(N_PILOT, K_PILOT)
    out["pool"] = draw(N_POOL, K_POOL)
    out["test"] = draw(N_TEST, K_TEST)
    out["_disjoint"] = True
    out["_source"] = "FALLBACK synthetic Gaussian features, dim 50, separability 0.55"
    return out


# ---------------------------------------------------------------- students

class Student:
    def __init__(self, frames, is_text):
        self.is_text = is_text
        pool_x, _ = frames["pool"]
        test_x, _ = frames["test"]
        if is_text:
            self.vec = TfidfVectorizer(lowercase=True, sublinear_tf=True,
                                       min_df=2, ngram_range=(1, 2))
            self.Xp = self.vec.fit_transform(pool_x)
            self.Xt = self.vec.transform(test_x)
        else:
            self.Xp, self.Xt = pool_x, test_x
        self.y_test = frames["test"][1]

    def auroc(self, pool_labels):
        y = np.asarray(pool_labels, int)
        if y.min() == y.max():
            return 0.5, True
        clf = LogisticRegression(C=1.0, solver="liblinear",
                                 class_weight="balanced", max_iter=2000)
        clf.fit(self.Xp, y)
        s = clf.decision_function(self.Xt)
        return float(roc_auc_score(self.y_test, s)), False


# ---------------------------------------------------------------- labellers

def synthetic_labels(y_true, tpr, fpr, rng):
    u = rng.random(len(y_true))
    thr = np.where(y_true == 1, tpr, fpr)
    return (u < thr).astype(int)


def ollama_flags(model, texts, is_text, workers=8):
    if not is_text:
        raise RuntimeError("real arm needs text")

    def one(t):
        p, h = t.split(" [SEP] ", 1)
        body = json.dumps({"model": model, "prompt": PROMPT.format(p=p, h=h),
                           "stream": False, "think": False,
                           "options": {"num_predict": 8, "temperature": 0, "seed": SEED}}).encode()
        req = urllib.request.Request("http://localhost:11434/api/generate", data=body,
                                     headers={"Content-Type": "application/json"})
        for _ in range(3):
            try:
                return json.load(urllib.request.urlopen(req, timeout=180))["response"]
            except Exception:
                time.sleep(2)
        return ""

    with ThreadPoolExecutor(workers) as ex:
        reps = list(ex.map(one, list(texts)))
    flags, unparsed = [], 0
    for r in reps:
        s = r.strip().upper()
        if s.startswith("YES"):
            flags.append(1)
        else:
            flags.append(0)
            if not s.startswith("NO"):
                unparsed += 1
    return np.asarray(flags, int), unparsed


# ---------------------------------------------------------------- screening

def screen(verdicts, aurocs, thr):
    """gate FAIL is the positive test; student failed (auroc < thr) is the condition."""
    tp = fp = fn = tn = 0
    for v, a in zip(verdicts, aurocs):
        failed = a < thr
        if v and failed:
            tp += 1
        elif v and not failed:
            fp += 1
        elif (not v) and failed:
            fn += 1
        else:
            tn += 1
    d = lambda a, b: (a / b) if b else None
    return {"threshold": thr,
            "n_labellers": len(verdicts),
            "cell_gatefail_studentfailed": tp,
            "cell_gatefail_studentuseful": fp,
            "cell_gatepass_studentfailed": fn,
            "cell_gatepass_studentuseful": tn,
            "n_student_failed": tp + fn, "n_student_useful": fp + tn,
            "n_gate_fail": tp + fp, "n_gate_pass": fn + tn,
            "sensitivity": d(tp, tp + fn), "specificity": d(tn, tn + fp),
            "ppv": d(tp, tp + fp), "npv": d(tn, tn + fn),
            "frac_gatepass_still_failing": d(fn, fn + tn),
            "youden_j": (d(tp, tp + fn) + d(tn, tn + fp) - 1)
                        if (tp + fn) and (tn + fp) else None}


def sweep_ratio(ratios, aurocs, thr):
    """Every threshold on the raw ratio n/K. FAIL predicted when ratio >= t."""
    cands = sorted(set([0.0] + [r for r in ratios] + [r + 1e-9 for r in ratios]))
    rows = []
    for t in cands:
        v = [r >= t for r in ratios]
        s = screen(v, aurocs, thr)
        s["ratio_threshold"] = t
        rows.append(s)
    best = max((r for r in rows if r["youden_j"] is not None),
               key=lambda r: r["youden_j"], default=None)
    return rows, best


# ---------------------------------------------------------------- run

def run_synthetic(frames, is_text):
    stu = Student(frames, is_text)
    pilot_y = frames["pilot"][1]
    pool_y = frames["pool"][1]
    assert int(pilot_y.sum()) == K_PILOT, int(pilot_y.sum())
    rng = np.random.default_rng(SEED)
    oracle, _ = stu.auroc(pool_y)
    allneg, _ = stu.auroc(np.zeros_like(pool_y))
    print(f"  reference: clean-label (oracle) student AUROC = {oracle:.4f}", flush=True)
    rows = []
    for tpr, fpr in SYNTHETIC_GRID:
        lp = synthetic_labels(pilot_y, tpr, fpr, rng)
        lt = synthetic_labels(pool_y, tpr, fpr, rng)
        g = gate_verdict(N_PILOT, K_PILOT, int(lp.sum()))
        auroc, degen = stu.auroc(lt)
        rows.append({"labeller": f"synth_tpr{tpr:.2f}_fpr{fpr:.2f}",
                     "family": "synthetic", "tpr": tpr, "fpr": fpr,
                     "pool_flag_count": int(lt.sum()),
                     "pool_flag_rate": float(lt.mean()),
                     "ratio_flagrate_over_baserate": g["flag_count"] / K_PILOT,
                     "student_auroc": auroc, "student_degenerate": degen, **g})
        rows[-1]["_oracle_reference_auroc"] = oracle
        print(f"  {rows[-1]['labeller']}  n={g['flag_count']:3d}  "
              f"p_floor={g['floor_p']:.3g}  gate_fail={g['gate_fail']}  "
              f"auroc={auroc:.4f}", flush=True)
    return rows


def run_real(frames, is_text, models):
    stu = Student(frames, is_text)
    pilot_x, pilot_y = frames["pilot"]
    pool_x, _ = frames["pool"]
    rows = []
    for m in models:
        t0 = time.time()
        try:
            fp_, up_p = ollama_flags(m, pilot_x, is_text)
            fl_, up_l = ollama_flags(m, pool_x, is_text)
        except Exception as e:
            print(f"  {m}: NOT RUN ({type(e).__name__})", flush=True)
            continue
        g = gate_verdict(N_PILOT, K_PILOT, int(fp_.sum()))
        auroc, degen = stu.auroc(fl_)
        pt = int(((fp_ == 1) & (pilot_y == 1)).sum())
        rows.append({"labeller": m, "family": "real_ollama",
                     "pilot_true_positives_flagged": pt,
                     "pilot_unparsed": up_p, "pool_unparsed": up_l,
                     "pool_flag_count": int(fl_.sum()),
                     "pool_flag_rate": float(fl_.mean()),
                     "ratio_flagrate_over_baserate": g["flag_count"] / K_PILOT,
                     "student_auroc": auroc, "student_degenerate": degen,
                     "seconds": round(time.time() - t0, 1), **g})
        print(f"  {m}  n={g['flag_count']:3d}  p_floor={g['floor_p']:.3g}  "
              f"gate_fail={g['gate_fail']}  auroc={auroc:.4f}  "
              f"({rows[-1]['seconds']}s)", flush=True)
    return rows


def analyse(rows, tag):
    v = [r["gate_fail"] for r in rows]
    a = [r["student_auroc"] for r in rows]
    ratios = [r["ratio_flagrate_over_baserate"] for r in rows]
    out = {"tag": tag, "n_labellers": len(rows), "screening": {}}
    for thr in FAIL_THRESHOLDS:
        key = f"auroc_lt_{thr:.2f}"
        s = screen(v, a, thr)
        fail_region = gate_fail_region(N_PILOT, K_PILOT)
        # ratio threshold reproducing the gate's fail region, if the region is an upper tail
        contiguous_tail = [n for n in fail_region if n > 0]
        tail_ok = bool(contiguous_tail) and contiguous_tail == list(
            range(min(contiguous_tail), N_PILOT + 1))
        eq = None
        if tail_ok:
            t = min(contiguous_tail) / K_PILOT
            pred = [r >= t for r in ratios]
            eq = {"ratio_threshold": t,
                  "identical_to_gate_verdicts": bool(pred == v),
                  "disagreements": int(sum(int(p != q) for p, q in zip(pred, v))),
                  "screening": screen(pred, a, thr)}
        sw, best = sweep_ratio(ratios, a, thr)
        out["screening"][key] = {
            "gate": s,
            "ratio_at_gate_equivalent_threshold": eq,
            "ratio_best_over_all_thresholds": best,
            "gate_youden_j": s["youden_j"],
            "ratio_beats_gate": (best is not None and s["youden_j"] is not None
                                 and best["youden_j"] > s["youden_j"] + 1e-12),
        }
    ok = [(r["ceiling_mcc"], r["floor_p"], r["student_auroc"]) for r in rows]
    if len(ok) > 2:
        c = [x[0] for x in ok]; p = [x[1] for x in ok]; y = [x[2] for x in ok]
        rc = spearmanr(c, y); rp = spearmanr(p, y)
        out["spearman_ceilingmcc_vs_auroc"] = {"rho": float(rc.statistic), "p": float(rc.pvalue)}
        out["spearman_floorp_vs_auroc"] = {"rho": float(rp.statistic), "p": float(rp.pvalue)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["synthetic", "real", "combine"], required=True)
    ap.add_argument("--models", default=",".join(OLLAMA_MODELS))
    a = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)

    if a.arm == "combine":
        base = json.load(open(os.path.join(OUTDIR, "c135_synthetic.json")))
        rows = list(base["labellers"])
        rp = os.path.join(OUTDIR, "c135_real.json")
        real_rows = json.load(open(rp))["labellers"] if os.path.exists(rp) else []
        art = {
            "check": "C135 marginal feasibility gate validation",
            "seed": SEED, "data_source": base["data_source"],
            "frames": base["frames"],
            "gate": {"alpha": P_FLOOR_ALPHA,
                     "fail_region_flag_counts": gate_fail_region(N_PILOT, K_PILOT),
                     "fail_region_is_zero_plus_upper_tail": None},
            "labellers": rows + real_rows,
            "primary_synthetic": analyse(rows, "synthetic"),
            "real": analyse(real_rows, "real_ollama") if len(real_rows) > 2 else
                    {"tag": "real_ollama", "n_labellers": len(real_rows),
                     "note": "too few real labellers for screening statistics"},
            "pooled": analyse(rows + real_rows, "pooled") if real_rows else None,
            "real_labeller_rows": real_rows,
        }
        fr = art["gate"]["fail_region_flag_counts"]
        tail = [n for n in fr if n > 0]
        art["gate"]["fail_region_is_zero_plus_upper_tail"] = bool(
            0 in fr and tail and tail == list(range(min(tail), N_PILOT + 1)))
        art["gate"]["smallest_failing_positive_flag_count"] = min(tail) if tail else None
        p = os.path.join(OUTDIR, "c135.json")
        json.dump(art, open(p, "w"), indent=2, sort_keys=True)
        print("wrote", p)
        return

    try:
        frames = load_snli_frames()
        is_text = True
    except Exception as e:
        print(f"SNLI unavailable ({type(e).__name__}: {e}); using Gaussian fallback", flush=True)
        frames = load_gaussian_frames()
        is_text = False
    meta = {"data_source": frames["_source"], "is_text": is_text,
            "frames": {"pilot": [N_PILOT, int(frames["pilot"][1].sum())],
                       "pool": [N_POOL, int(frames["pool"][1].sum())],
                       "test": [N_TEST, int(frames["test"][1].sum())],
                       "pool_test_disjoint": frames["_disjoint"]}}
    print(json.dumps(meta), flush=True)
    if a.arm == "synthetic":
        rows = run_synthetic(frames, is_text)
        p = os.path.join(OUTDIR, "c135_synthetic.json")
    else:
        rows = run_real(frames, is_text, [m for m in a.models.split(",") if m])
        p = os.path.join(OUTDIR, "c135_real.json")
    if a.arm == "synthetic":
        meta["clean_label_oracle_student_auroc"] = rows[0]["_oracle_reference_auroc"]
        for r in rows:
            r.pop("_oracle_reference_auroc", None)
    json.dump({**meta, "seed": SEED, "labellers": rows}, open(p, "w"), indent=2, sort_keys=True)
    print("wrote", p)


if __name__ == "__main__":
    sys.exit(main())
