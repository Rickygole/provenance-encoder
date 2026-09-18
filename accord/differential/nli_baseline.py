"""C100: an off-the-shelf NLI baseline, offline, sandboxed, CPU-only. See
C100_PREREGISTRATION.md.

THE OFFLINE ENV VARS ARE SET AS THE VERY FIRST LINES OF THIS MODULE, before any
other import. transformers and huggingface_hub read these at import time, and
setting them after `import transformers` has already run is too late.
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")

import collections  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import pathlib  # noqa: E402
import socket  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from scipy.stats import mannwhitneyu  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from accord.cloud_paths import assert_local_only  # noqa: E402
from accord.control.delong import delong_roc_test  # noqa: E402

MODEL_DIR = pathlib.Path.home() / "models_public" / "roberta-large-mnli"
MANIFEST = ROOT / "results" / "accord" / "differential" / "nli_model_manifest.json"
MIMIC_LOCAL_ONLY_DIR = pathlib.Path(os.environ.get("MIMIC_LOCAL_ONLY_DIR", pathlib.Path.home() / "mimic_local_only"))
GOLD = (MIMIC_LOCAL_ONLY_DIR / "from_drive_2026-08-08"
        / "outputs" / "pharmacist_labeled_clean.csv")
DETECTOR = (MIMIC_LOCAL_ONLY_DIR / "derived"
           / "gold_standard_accord_results.csv")
PER_PAIR_OUT = (MIMIC_LOCAL_ONLY_DIR / "derived"
               / "nli_scores_roberta_large_mnli.csv")
OUT = ROOT / "results" / "accord" / "differential" / "nli_roberta_large_mnli.json"
FACTS = ROOT / "results" / "accord" / "VERIFIED_FACTS.json"
PREFIX = "nlibaseline"

MAX_TOKENS = 512
B = 10000
SEED = 42
TOP_K = 15
MDE_KEY = ("mde_recomputed.gold_ambiguous_excluded"
          ".minimum_detectable_auc")
ALPHA = 0.05

# The two register keys the detector's recomputed AUROC must match, to 4
# decimal places, before the paired DeLong test runs against it. If either
# artifact has drifted since these were registered, the join below would
# silently compare the NLI baseline against a stale detector score.
DETECTOR_AUROC_KEYS = {
    "G_253": "signal.contra_prob_vs_pharm_label__ambiguous_excluded_n253.auroc",
    "G_299": "signal.contra_prob_vs_pharm_label__ambiguous_as_negative_n299.auroc",
}


class NetworkGuardTripped(RuntimeError):
    pass


def install_socket_guard():
    """Patch socket.socket.connect, socket.create_connection and
    socket.getaddrinfo to raise. Installed AFTER the model is loaded (which is
    already offline and local-only) and BEFORE the gold CSV is opened, so
    nothing between here and process exit can reach the network even by
    accident."""

    def _blocked(*a, **k):
        raise NetworkGuardTripped(
            "network access attempted after the socket guard was installed; "
            "this run must be fully offline")

    socket.socket.connect = _blocked
    socket.create_connection = _blocked
    socket.getaddrinfo = _blocked


def verify_manifest_digests():
    manifest = json.loads(MANIFEST.read_text())
    for name, expected in manifest["files_sha256"].items():
        p = MODEL_DIR / name
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        got = h.hexdigest()
        if got != expected:
            sys.exit(f"SHA-256 mismatch for {name}: manifest says {expected}, "
                     f"disk has {got}. Refusing to run against a model whose "
                     f"files do not match what was downloaded and recorded.")
    return manifest


def load_model():
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR), local_files_only=True, torch_dtype=torch.float32)
    model.eval()
    id2label = {int(k): v.upper() for k, v in model.config.id2label.items()}
    contra_idx = None
    for idx, lab in id2label.items():
        if lab.startswith("CONTRADICT"):
            contra_idx = idx
    if contra_idx is None:
        sys.exit(f"no CONTRADICTION label in model.config.id2label: {id2label}")
    return tok, model, contra_idx, id2label


def score_pair(tok, model, contra_idx, premise, hypothesis):
    """Contradiction probability for one (premise, hypothesis) direction.
    truncation=False: the encoding is never silently cut, and every encoding
    is asserted to fit MAX_TOKENS before ever reaching the model."""
    enc = tok(premise, hypothesis, truncation=False, return_tensors="pt")
    n_tokens = enc["input_ids"].shape[1]
    if n_tokens > MAX_TOKENS:
        return None, n_tokens
    with torch.no_grad():
        logits = model(**enc).logits[0]
    probs = F.softmax(logits, dim=-1)
    return float(probs[contra_idx]), n_tokens


def stratified_bootstrap_auc_ci(y, scores, b=B, seed=SEED):
    y = np.asarray(y)
    scores = np.asarray(scores)
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    boot = np.empty(b)
    for i in range(b):
        pi = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        ni = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([pi, ni])
        boot[i] = roc_auc_score(y[idx], scores[idx])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    if not MANIFEST.exists():
        sys.exit(f"{MANIFEST} missing; run fetch_nli_model.py first")
    manifest = verify_manifest_digests()
    tok, model, contra_idx, id2label = load_model()
    print(f"model loaded: {MODEL_DIR.name}  commit_sha={manifest['commit_sha']}  "
          f"id2label={id2label}  contradiction_index={contra_idx}")

    # The socket guard goes on AFTER the model is loaded and BEFORE the gold
    # CSV is opened; everything from here on that could conceivably reach the
    # network raises instead.
    install_socket_guard()

    assert_local_only(GOLD)
    assert_local_only(DETECTOR)
    assert_local_only(PER_PAIR_OUT)

    gold = pd.read_csv(GOLD)
    n_total = len(gold)
    scores, dir1, dir2, n_tok_seen, n_over_limit = [], [], [], [], 0
    for i, row in gold.iterrows():
        nursing, discharge = str(row["nursing_sentence"]), str(row["discharge_sentence"])
        p1, t1 = score_pair(tok, model, contra_idx, nursing, discharge)
        p2, t2 = score_pair(tok, model, contra_idx, discharge, nursing)
        n_tok_seen.extend([t1, t2])
        if p1 is None or p2 is None:
            n_over_limit += 1
            scores.append(None)
        else:
            scores.append((p1 + p2) / 2.0)
        dir1.append(p1)
        dir2.append(p2)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n_total} scored", flush=True)

    # Caught already per-pair above (p is None whenever an encoding exceeds
    # MAX_TOKENS); max_tokens_seen is the hard number Outcome C's "any
    # encoding over 512" gate is read against.
    max_tokens_seen = max(n_tok_seen)

    per_pair = pd.DataFrame({
        "pair_id": gold["pair_id"], "label_clean": gold["label_clean"],
        "contra_prob_direction1": dir1, "contra_prob_direction2": dir2,
        "contra_prob_mean": scores,
    })
    PER_PAIR_OUT.parent.mkdir(parents=True, exist_ok=True)
    per_pair.to_csv(PER_PAIR_OUT, index=False)

    d253 = per_pair[per_pair.label_clean != "ambiguous"].copy()
    d299 = per_pair.copy()
    y253 = (d253.label_clean == "contradiction").astype(int).to_numpy()
    y299 = (d299.label_clean == "contradiction").astype(int).to_numpy()
    s253 = d253.contra_prob_mean.to_numpy()
    s299 = d299.contra_prob_mean.to_numpy()

    n_over_limit_total = n_over_limit
    n_pairs_ok = int(per_pair.contra_prob_mean.notna().sum())

    facts = json.loads(FACTS.read_text())
    mde_auc = facts[MDE_KEY]["value"]

    auroc_253 = float(roc_auc_score(y253, s253))
    pos253 = s253[y253 == 1]
    neg253 = s253[y253 == 0]
    _, mw_p = mannwhitneyu(pos253, neg253, alternative="greater")
    mw_p = float(mw_p)

    invalid = (max_tokens_seen > MAX_TOKENS or n_over_limit_total > 0
              or per_pair.contra_prob_mean.isna().any()
              or len(d253) != 253 or len(d299) != 299)
    if invalid:
        outcome = "C"
    elif auroc_253 >= mde_auc and mw_p < ALPHA:
        outcome = "A"
    else:
        outcome = "B"

    # Secondary, G_299 (ambiguous counted negative).
    auroc_299 = float(roc_auc_score(y299, s299))
    ci_lo, ci_hi = stratified_bootstrap_auc_ci(y299, s299)
    ap_299 = float(average_precision_score(y299, s299))
    top_k_idx = np.argsort(-s299)[:TOP_K]
    top_k_tp = int(y299[top_k_idx].sum())

    # Per-direction AUROCs, descriptive only.
    auroc_dir1_299 = float(roc_auc_score(y299, d299.contra_prob_direction1.to_numpy()))
    auroc_dir2_299 = float(roc_auc_score(y299, d299.contra_prob_direction2.to_numpy()))

    # Paired DeLong vs the detector. Recompute the detector's own AUROC first
    # and abort if it has drifted from the register.
    det = pd.read_csv(DETECTOR)
    det_y253 = (det[det.pharm_label != "ambiguous"].pharm_label
               == "contradiction").astype(int)
    det_s253 = det[det.pharm_label != "ambiguous"].accord_contra_prob
    det_auroc_253 = round(float(roc_auc_score(det_y253, det_s253)), 4)
    det_y299 = (det.pharm_label == "contradiction").astype(int)
    det_auroc_299 = round(float(roc_auc_score(det_y299, det.accord_contra_prob)), 4)
    registered_253 = round(facts[DETECTOR_AUROC_KEYS["G_253"]]["value"], 4)
    registered_299 = round(facts[DETECTOR_AUROC_KEYS["G_299"]]["value"], 4)
    if det_auroc_253 != registered_253 or det_auroc_299 != registered_299:
        sys.exit(f"detector AUROC drifted from the register: recomputed "
                 f"G_253={det_auroc_253} (registered {registered_253}), "
                 f"G_299={det_auroc_299} (registered {registered_299}); "
                 f"aborting rather than DeLong-testing against a stale score")

    joined = per_pair.merge(det[["pair_id", "pharm_label", "accord_contra_prob"]],
                            on="pair_id", how="inner")
    j253 = joined[joined.label_clean != "ambiguous"]
    j299 = joined
    y_j253 = (j253.label_clean == "contradiction").astype(int).to_numpy()
    y_j299 = (j299.label_clean == "contradiction").astype(int).to_numpy()
    delong_253 = delong_roc_test(y_j253, j253.contra_prob_mean.to_numpy(),
                                 j253.accord_contra_prob.to_numpy())
    delong_299 = delong_roc_test(y_j299, j299.contra_prob_mean.to_numpy(),
                                 j299.accord_contra_prob.to_numpy())

    out = collections.OrderedDict([
        ("preregistration", "accord/differential/C100_PREREGISTRATION.md"),
        ("model", "FacebookAI/roberta-large-mnli"),
        ("commit_sha", manifest["commit_sha"]),
        ("id2label", id2label),
        ("n_total", n_total),
        ("n_pairs_scored", n_pairs_ok),
        ("n_over_token_limit", n_over_limit_total),
        ("max_tokens_seen", max_tokens_seen),
        ("primary_g253", collections.OrderedDict([
            ("n", len(d253)), ("n_positives", int(y253.sum())),
            ("auroc", auroc_253), ("mannwhitney_p_one_sided", mw_p),
            ("mde_auroc", mde_auc),
        ])),
        ("secondary_g299", collections.OrderedDict([
            ("n", len(d299)), ("n_positives", int(y299.sum())),
            ("auroc", auroc_299),
            ("auroc_ci_lo", ci_lo), ("auroc_ci_hi", ci_hi),
            ("auroc_bootstrap_B", B), ("auroc_bootstrap_seed", SEED),
            ("average_precision", ap_299),
            ("top_k", TOP_K), ("top_k_true_positives", top_k_tp),
        ])),
        ("per_direction_g299_descriptive_only", collections.OrderedDict([
            ("auroc_direction1_nursing_premise", auroc_dir1_299),
            ("auroc_direction2_discharge_premise", auroc_dir2_299),
        ])),
        ("delong_vs_detector", collections.OrderedDict([
            ("detector_auroc_g253_recomputed", det_auroc_253),
            ("detector_auroc_g299_recomputed", det_auroc_299),
            ("g253", delong_253), ("g299", delong_299),
        ])),
        ("outcome", outcome),
        ("per_pair_scores_path",
         "~/mimic_local_only/derived/nli_scores_roberta_large_mnli.csv "
         "(outside version control)"),
    ])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"  G_253 AUROC {auroc_253:.4f} (MDE {mde_auc}) p={mw_p:.4f}  outcome={outcome}")
    print(f"  G_299 AUROC {auroc_299:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]  AP={ap_299:.4f}  "
          f"top-{TOP_K} TP={top_k_tp}")
    print(f"  DeLong vs detector: G_253 p={delong_253['p']:.4f}  "
          f"G_299 p={delong_299['p']:.4f}")
    print("  no note text was written to this repository or to the console.")


if __name__ == "__main__":
    main()
